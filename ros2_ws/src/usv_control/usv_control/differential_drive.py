"""Differential drive kinematics for USV dual-propeller propulsion.

Converts (throttle, diff_torque) from the cascaded PID into individual
left/right thruster commands for the Gazebo differential drive plugin.

Also handles dead-zone compensation — small PID outputs that would fail
to overcome static water friction are boosted to the minimum effective thrust.
"""

from typing import Tuple


class DifferentialDrive:
    """Differential drive mixer and dead-zone compensator.

    Maps (throttle, diff_torque) → (left_thrust, right_thrust),
    with optional dead-zone threshold to overcome static friction
    of water on small hulls.
    """

    def __init__(
        self,
        wheel_separation: float = 0.40,
        max_thrust: float = 1.0,
        dead_zone: float = 0.05,
    ) -> None:
        """Initialize differential drive.

        Args:
            wheel_separation: Distance between left and right propellers (m).
            max_thrust: Maximum normalized thrust command.
            dead_zone: Minimum command magnitude before thrust activates.
                       Prevents humming at near-zero commands.
        """
        self.wheel_separation: float = wheel_separation
        self.max_thrust: float = max_thrust
        self.dead_zone: float = dead_zone

    def mix(self, throttle: float, diff_torque: float) -> Tuple[float, float]:
        """Mix throttle and differential torque into left/right commands.

        Args:
            throttle: Forward throttle command [-1.0, 1.0].
            diff_torque: Differential torque [-1.0, 1.0], positive = turn left.

        Returns:
            (left_thrust, right_thrust) each in [-1.0, 1.0].
        """
        # Mix: left gets extra thrust for positive diff_torque (turn right)
        left = throttle + diff_torque * 0.5
        right = throttle - diff_torque * 0.5

        # Clamp to [-max_thrust, max_thrust]
        left = max(-self.max_thrust, min(self.max_thrust, left))
        right = max(-self.max_thrust, min(self.max_thrust, right))

        # Apply dead zone
        left = self._apply_dead_zone(left)
        right = self._apply_dead_zone(right)

        return left, right

    def _apply_dead_zone(self, value: float) -> float:
        """Apply dead zone to a single thruster command.

        Values below dead_zone are zeroed to avoid thruster humming.
        Values just above are boosted to the minimum effective thrust.

        Args:
            value: Raw normalized command.

        Returns:
            Dead-zone-compensated command.
        """
        if abs(value) < self.dead_zone:
            return 0.0
        # Preserve sign, boost to at least dead_zone magnitude
        sign = 1.0 if value > 0 else -1.0
        boosted = max(abs(value), self.dead_zone) * sign
        return max(-self.max_thrust, min(self.max_thrust, boosted))

    def twist_to_thrust(
        self, linear_x: float, angular_z: float
    ) -> Tuple[float, float]:
        """Convert Twist velocities directly to thruster commands.

        Uses kinematic model for differential drive:
          v_left  = (v - ω * d/2) / r
          v_right = (v + ω * d/2) / r

        where d = wheel_separation, r = effective propeller radius.

        Args:
            linear_x: Desired forward velocity (m/s).
            angular_z: Desired angular velocity (rad/s, CCW positive).

        Returns:
            (left_thrust, right_thrust) normalized [-1.0, 1.0].
        """
        half_sep = self.wheel_separation / 2.0
        left_vel = (linear_x - angular_z * half_sep)
        right_vel = (linear_x + angular_z * half_sep)

        # Normalize to [-1, 1] assuming max speed ~1.5 m/s
        max_speed = 1.5
        left = left_vel / max_speed
        right = right_vel / max_speed

        # Clamp
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))

        return self._apply_dead_zone(left), self._apply_dead_zone(right)
