"""Tests for cascaded PID controller."""

import pytest
from usv_control.cascaded_pid import (
    PIDParams,
    PIDController,
    CascadedPID,
    default_cascaded_pid,
)


class TestPIDController:
    """Test suite for single PID loop."""

    def test_proportional_only(self) -> None:
        """Test P-only control: output should be kp * error."""
        params = PIDParams(kp=2.0, ki=0.0, kd=0.0, output_min=-10.0, output_max=10.0)
        pid = PIDController(params)

        output = pid.update(setpoint=5.0, measurement=3.0, dt=0.1)
        # P-only: output = 2.0 * (5 - 3) = 4.0
        assert abs(output - 4.0) < 0.01

    def test_integral_accumulation(self) -> None:
        """Test that integral term accumulates over time."""
        params = PIDParams(kp=0.0, ki=1.0, kd=0.0, output_min=-10.0, output_max=10.0)
        pid = PIDController(params)

        # Constant error of 1.0 for 5 steps of 0.1s
        total = 0.0
        for _ in range(5):
            total += pid.update(setpoint=1.0, measurement=0.0, dt=0.1)
        # After 5 steps: integral = 5 * 0.1 = 0.5, output = 0.5
        assert abs(total - 0.5) < 0.02

    def test_integral_anti_windup(self) -> None:
        """Test that integral is clamped at integral_max."""
        params = PIDParams(
            kp=0.0, ki=1.0, kd=0.0,
            output_min=-10.0, output_max=10.0,
            integral_max=0.3,
        )
        pid = PIDController(params)

        # Run many steps with constant error
        for _ in range(100):
            output = pid.update(setpoint=10.0, measurement=0.0, dt=0.1)

        # Integral should be clamped at 0.3
        assert abs(pid.state.integral - 0.3) < 0.01

    def test_integral_separation(self) -> None:
        """Test that integral is zeroed when error exceeds threshold."""
        params = PIDParams(kp=1.0, ki=1.0, kd=0.0, output_min=-10.0, output_max=10.0)
        pid = PIDController(params)

        # Small error — integral should accumulate
        pid.update(setpoint=1.0, measurement=0.5, dt=0.1, integral_sep_threshold=2.0)
        assert abs(pid.state.integral) > 0.001

        # Large error — integral should be zeroed
        pid.update(setpoint=100.0, measurement=0.0, dt=0.1, integral_sep_threshold=2.0)
        assert abs(pid.state.integral) < 0.001

    def test_output_clamping(self) -> None:
        """Test that output is clamped to [output_min, output_max]."""
        params = PIDParams(kp=100.0, ki=0.0, kd=0.0, output_min=-1.0, output_max=1.0)
        pid = PIDController(params)

        output = pid.update(setpoint=100.0, measurement=0.0, dt=0.1)
        assert output == 1.0

    def test_derivative_first(self) -> None:
        """Test derivative-first: setpoint jump should NOT cause derivative kick."""
        params = PIDParams(kp=0.0, ki=0.0, kd=10.0, output_min=-10.0, output_max=10.0)
        pid = PIDController(params)

        # First call: measurement hasn't changed, derivative = 0
        output1 = pid.update(setpoint=100.0, measurement=0.0, dt=0.1)
        # Derivative on measurement: measurement was 0, still 0 → d_out = 0
        assert abs(output1) < 0.01

        # Second call: measurement changed, derivative reacts
        output2 = pid.update(setpoint=100.0, measurement=10.0, dt=0.1)
        # d(measurement)/dt = (10 - 0)/0.1 = 100, d_out = -10 * 100 = -1000 → clamped
        assert output2 < -5.0

    def test_reset(self) -> None:
        """Test that reset clears internal state."""
        params = PIDParams(kp=1.0, ki=1.0, kd=1.0)
        pid = PIDController(params)

        pid.update(setpoint=5.0, measurement=0.0, dt=0.1)
        assert pid.state.integral != 0.0

        pid.reset()
        assert pid.state.integral == 0.0
        assert pid.state.first_run


class TestCascadedPID:
    """Test suite for cascaded PID (heading + speed)."""

    def test_default_factory(self) -> None:
        """Test that default_cascaded_pid returns a valid controller."""
        ctrl = default_cascaded_pid()
        assert ctrl is not None
        assert ctrl.heading_outer is not None
        assert ctrl.heading_inner is not None
        assert ctrl.speed is not None

    def test_basic_update(self) -> None:
        """Test a basic cascaded PID update cycle."""
        ctrl = default_cascaded_pid()

        throttle, diff_torque = ctrl.update(
            desired_heading=0.5,
            desired_speed=0.8,
            current_heading=0.0,
            current_speed=0.0,
            current_angular_vel=0.0,
            dt=0.05,
        )

        # Both outputs should be within valid range
        assert -1.0 <= throttle <= 1.0
        assert -1.0 <= diff_torque <= 1.0
        # With error, should produce non-zero output
        assert throttle > 0.0

    def test_zero_error(self) -> None:
        """Test that zero error produces near-zero output."""
        ctrl = default_cascaded_pid()

        throttle, diff_torque = ctrl.update(
            desired_heading=0.0,
            desired_speed=0.0,
            current_heading=0.0,
            current_speed=0.0,
            current_angular_vel=0.0,
            dt=0.05,
        )

        assert abs(throttle) < 0.01
        assert abs(diff_torque) < 0.01

    def test_reset(self) -> None:
        """Test that reset clears all internal PID states."""
        ctrl = default_cascaded_pid()

        ctrl.update(0.5, 0.8, 0.0, 0.0, 0.0, 0.05)
        ctrl.reset()

        # After reset, all integrals should be zero
        assert ctrl.heading_outer.state.integral == 0.0
        assert ctrl.heading_inner.state.integral == 0.0
        assert ctrl.speed.state.integral == 0.0
