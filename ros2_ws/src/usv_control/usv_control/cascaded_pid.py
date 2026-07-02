"""Cascaded PID controller for USV heading and speed regulation.

Features (tuned for water surface dynamics):
  - Anti-windup: Prevents integrator saturation during prolonged errors
    (e.g., fighting current with limited thruster authority).
  - Integral separation: Disables integral term when error is large
    to prevent windup during large maneuvers.
  - Derivative-first: Computes derivative on measurement, not error,
    to avoid "derivative kick" on setpoint changes.

Architecture:
  Outer loop (heading) → Inner loop (angular velocity) → Thruster PWM
  Outer loop (speed)   → Inner loop (throttle)          → Thruster PWM
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PIDParams:
    """PID gain parameters with output limits."""

    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    output_min: float = -1.0
    output_max: float = 1.0
    integral_max: float = 0.5  # Anti-windup: clamp integrator


@dataclass
class PIDState:
    """Internal PID state (updated each iteration)."""

    integral: float = 0.0
    prev_error: float = 0.0
    prev_measurement: float = 0.0
    first_run: bool = True


class PIDController:
    """Single PID loop with anti-windup, integral separation, derivative-first.

    Use one instance for heading control, another for speed control.
    """

    def __init__(self, params: PIDParams) -> None:
        """Initialize PID controller.

        Args:
            params: Gain and limit configuration.
        """
        self.params: PIDParams = params
        self.state: PIDState = PIDState()

    def reset(self) -> None:
        """Reset internal state (useful when switching control modes)."""
        self.state = PIDState()

    def update(
        self,
        setpoint: float,
        measurement: float,
        dt: float,
        integral_sep_threshold: Optional[float] = None,
    ) -> float:
        """Compute control output for one time step.

        Args:
            setpoint: Desired value (e.g., target heading radians).
            measurement: Current measured value.
            dt: Time step in seconds.
            integral_sep_threshold: If |error| > this, zero the integral.
                                    Defaults to 0.3 * (output_max - output_min).

        Returns:
            Control output in [output_min, output_max].
        """
        error = setpoint - measurement

        if self.state.first_run:
            self.state.prev_measurement = measurement
            self.state.prev_error = error
            self.state.first_run = False

        # Proportional
        p_out = self.params.kp * error

        # Integral with anti-windup (clamping) and separation
        if integral_sep_threshold is None:
            integral_sep_threshold = 0.3 * (
                self.params.output_max - self.params.output_min
            )

        if self.params.ki > 1e-9 and abs(error) < integral_sep_threshold:
            self.state.integral += error * dt
            # Clamp integral
            self.state.integral = max(
                -self.params.integral_max,
                min(self.params.integral_max, self.state.integral),
            )
        else:
            self.state.integral = 0.0

        i_out = self.params.ki * self.state.integral

        # Derivative-first (on measurement, not error)
        # d(measurement)/dt avoids derivative kick on setpoint jumps
        if dt > 1e-9:
            d_measurement = (measurement - self.state.prev_measurement) / dt
        else:
            d_measurement = 0.0
        d_out = -self.params.kd * d_measurement

        # Sum and clamp
        output = p_out + i_out + d_out
        output = max(
            self.params.output_min,
            min(self.params.output_max, output),
        )

        # Update state
        self.state.prev_error = error
        self.state.prev_measurement = measurement

        return output


class CascadedPID:
    """Cascaded PID controller for USV motion.

    Outer loop: heading PID → desired angular velocity
                speed PID  → desired throttle
    Inner loop: angular velocity PID → differential torque
                (throttle mapped directly to differential drive)

    This cascaded structure provides smoother response than a single
    PID because the inner velocity loop linearizes the nonlinear
    water dynamics before the outer position loop acts.
    """

    def __init__(
        self,
        heading_outer: PIDParams,
        heading_inner: PIDParams,
        speed_params: PIDParams,
    ) -> None:
        """Initialize cascaded PID.

        Args:
            heading_outer: Outer heading loop gains.
            heading_inner: Inner angular velocity loop gains.
            speed_params: Speed control loop gains.
        """
        self.heading_outer: PIDController = PIDController(heading_outer)
        self.heading_inner: PIDController = PIDController(heading_inner)
        self.speed: PIDController = PIDController(speed_params)

    def update(
        self,
        desired_heading: float,
        desired_speed: float,
        current_heading: float,
        current_speed: float,
        current_angular_vel: float,
        dt: float,
    ) -> tuple[float, float]:
        """Compute throttle and differential torque.

        Args:
            desired_heading: Target heading angle (radians).
            desired_speed: Target forward speed (m/s).
            current_heading: Current measured heading (radians).
            current_speed: Current measured speed (m/s).
            current_angular_vel: Current angular velocity (rad/s).
            dt: Time step (seconds).

        Returns:
            (throttle, diff_torque) both in [-1.0, 1.0].
            throttle > 0 = forward, diff_torque > 0 = turn left.
        """
        # Speed control (single loop — water drag is self-damping)
        throttle = self.speed.update(desired_speed, current_speed, dt)

        # Heading control (cascaded)
        # Outer: heading error → desired angular velocity
        desired_ang_vel = self.heading_outer.update(
            desired_heading, current_heading, dt
        )
        # Inner: angular velocity → differential torque
        diff_torque = self.heading_inner.update(
            desired_ang_vel, current_angular_vel, dt
        )

        return throttle, diff_torque

    def reset(self) -> None:
        """Reset all internal PID states."""
        self.heading_outer.reset()
        self.heading_inner.reset()
        self.speed.reset()


def default_cascaded_pid() -> CascadedPID:
    """Factory for a default-tuned cascaded PID for a 50-80cm USV.

    Tuned empirically for a storage-box-sized vessel with:
      - Mass ~8 kg
      - Dual brushless thrusters (T200-class)
      - Cruise speed ~0.8 m/s, max ~1.5 m/s

    Returns:
        CascadedPID with reasonable default gains.
    """
    heading_outer = PIDParams(
        kp=2.5, ki=0.05, kd=0.3,
        output_min=-1.5, output_max=1.5,  # angular velocity rad/s
        integral_max=0.3,
    )
    heading_inner = PIDParams(
        kp=3.0, ki=0.1, kd=0.1,
        output_min=-1.0, output_max=1.0,  # differential torque
        integral_max=0.2,
    )
    speed_params = PIDParams(
        kp=4.0, ki=0.2, kd=0.5,
        output_min=-1.0, output_max=1.0,  # throttle
        integral_max=0.4,
    )
    return CascadedPID(heading_outer, heading_inner, speed_params)
