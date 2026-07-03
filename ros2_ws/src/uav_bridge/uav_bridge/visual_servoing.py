"""Visual servoing: UAV-guided precision approach for USVs.

When a USV needs to approach a target (debris, docking station, another
vessel) and GPS is unreliable due to multipath or drift, the UAV provides
closed-loop visual guidance.

The UAV tracks both the USV and the target in its overhead camera frame,
computes the pixel-space error vector, and sends corrective steering
commands to the USV. This forms an image-based visual servoing (IBVS)
loop running at camera framerate (~30 Hz).

Control law:
  error_px = target_px - usv_px
  correction_angle = K_p * error_px_x + K_d * d(error_px_x)/dt
  correction_throttle = K_p_dist * |error_px|

This is particularly effective because the UAV's overhead view provides
a near-perfect Euclidean projection — unlike on-board cameras that suffer
from perspective distortion and wave-induced jitter.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class ServoingParams:
    """Visual servoing controller gains.

    Attributes:
        kp_angle: Proportional gain for lateral correction.
        kd_angle: Derivative gain for lateral correction damping.
        kp_distance: Proportional gain for forward speed.
        max_correction_angle: Maximum steering correction per cycle (rad).
        dead_zone_px: Pixel error below which no correction is issued.
    """

    kp_angle: float = 0.005         # rad per pixel error
    kd_angle: float = 0.001         # rad per pixel/s
    kp_distance: float = 0.01       # m/s per pixel error
    max_correction_angle: float = 0.3   # ~17 degrees
    dead_zone_px: float = 5.0       # pixels


@dataclass
class TrackedObject:
    """State of a tracked object in the UAV camera frame."""

    px: float              # X pixel coordinate
    py: float              # Y pixel coordinate
    width_px: float        # Bounding box width
    height_px: float       # Bounding box height
    confidence: float      # Detection confidence


class VisualServoingController:
    """Image-based visual servoing using UAV overhead camera.

    Provides closed-loop guidance for a USV approaching a target.
    The UAV runs this controller and sends (heading_correction, throttle)
    commands to the USV over the star-topology radio link.
    """

    def __init__(self, params: Optional[ServoingParams] = None) -> None:
        """Initialize visual servoing controller.

        Args:
            params: Controller gains. Uses defaults if None.
        """
        self.params: ServoingParams = params or ServoingParams()

        # State
        self._prev_error_x: float = 0.0
        self._prev_error_y: float = 0.0
        self._dt: float = 0.033  # ~30 Hz camera

    def compute_command(
        self,
        usv: TrackedObject,
        target: TrackedObject,
        dt: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Compute steering and throttle commands for the USV.

        Args:
            usv: Tracked USV position in UAV camera frame.
            target: Tracked target position in UAV camera frame.
            dt: Time since last update (seconds). Uses default if None.

        Returns:
            (steering_correction_rad, throttle_normalized).
            Positive steering = turn left, positive throttle = forward.
        """
        if dt is not None:
            self._dt = dt

        p = self.params

        # Pixel-space error
        error_x = target.px - usv.px
        error_y = target.py - usv.py
        error_dist = np.sqrt(error_x**2 + error_y**2)

        # Dead zone
        if error_dist < p.dead_zone_px:
            self._prev_error_x = 0.0
            self._prev_error_y = 0.0
            return (0.0, 0.0)

        # Lateral correction (PD on horizontal error)
        d_error_x = (error_x - self._prev_error_x) / max(self._dt, 1e-6)
        steering = p.kp_angle * error_x + p.kd_angle * d_error_x
        steering = max(-p.max_correction_angle, min(p.max_correction_angle, steering))

        # Forward speed (proportional to distance error, mainly vertical)
        throttle = p.kp_distance * error_dist
        throttle = max(-1.0, min(1.0, throttle))

        # If target is behind the USV, reverse slowly
        if error_y > 0:
            throttle = -0.2

        # Update state
        self._prev_error_x = error_x
        self._prev_error_y = error_y

        return (steering, throttle)

    def estimate_arrival(
        self,
        usv: TrackedObject,
        target: TrackedObject,
        threshold_px: float = 10.0,
    ) -> bool:
        """Check if the USV has arrived at the target.

        Args:
            usv: USV position.
            target: Target position.
            threshold_px: Arrival threshold in pixels.

        Returns:
            True if the USV is within threshold of the target.
        """
        error = np.sqrt(
            (target.px - usv.px) ** 2 + (target.py - usv.py) ** 2
        )
        return error < threshold_px

    def reset(self) -> None:
        """Reset internal state."""
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
