"""UAV overhead perception: birds-eye object detection for water surface.

The UAV provides a "God's-eye view" that eliminates:
  - Line-of-sight occlusion (small boats hidden behind larger vessels)
  - Perspective projection distortion (IPM errors from wave jitter)
  - Multipath RF interference (elevated antenna = clean LOS to all USVs)

From 20-30 m altitude, the downward-facing camera sees the entire
operation area. YOLO detections are in near-perfect BEV projection
(simple affine warp), with obstacle world coordinates computed
directly from the UAV's GNSS + gimbal attitude.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class OverheadDetection:
    """A single detection from UAV overhead view."""

    class_id: int
    class_name: str
    world_x: float          # Longitude-projected or local X (m)
    world_y: float          # Latitude-projected or local Y (m)
    width_m: float          # Object width on water surface (m)
    height_m: float         # Object length on water surface (m)
    confidence: float       # Detection confidence [0, 1]


class OverheadPerception:
    """UAV-based overhead perception pipeline.

    Provides:
      - Downward YOLO inference for water surface objects
      - Direct pixel-to-world projection (simplified by near-nadir view)
      - Global obstacle map assembly from UAV perspective
    """

    def __init__(
        self,
        uav_altitude: float = 25.0,
        camera_fov_h: float = 1.2,     # radians (~69°)
        camera_fov_v: float = 0.9,     # radians (~52°)
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> None:
        """Initialize overhead perception.

        Args:
            uav_altitude: UAV flight altitude above water surface (m).
            camera_fov_h: Horizontal field of view (radians).
            camera_fov_v: Vertical field of view (radians).
            image_width: Camera image width (pixels).
            image_height: Camera image height (pixels).
        """
        self.uav_altitude: float = uav_altitude
        self.camera_fov_h: float = camera_fov_h
        self.camera_fov_v: float = camera_fov_v
        self.image_width: int = image_width
        self.image_height: int = image_height

        # Ground sample distance (m/pixel) at this altitude
        self._gsd_x: float = (
            2.0 * uav_altitude * np.tan(camera_fov_h / 2.0) / image_width
        )
        self._gsd_y: float = (
            2.0 * uav_altitude * np.tan(camera_fov_v / 2.0) / image_height
        )

    def pixel_to_world(
        self,
        px: float,
        py: float,
        uav_x: float,
        uav_y: float,
        uav_yaw: float = 0.0,
    ) -> Tuple[float, float]:
        """Convert UAV image pixel to world coordinates.

        Assumes near-nadir (straight-down) view. The pixel offset from
        image center, scaled by ground sample distance, gives the world
        offset from the UAV's nadir point.

        Args:
            px: Horizontal pixel coordinate.
            py: Vertical pixel coordinate.
            uav_x: UAV world X (nadir point).
            uav_y: UAV world Y (nadir point).
            uav_yaw: UAV yaw angle (radians, 0 = North).

        Returns:
            (world_x, world_y) of the detected object.
        """
        # Offset from image center
        dx_img = px - self.image_width / 2.0
        dy_img = py - self.image_height / 2.0

        # Convert to meters on the ground
        dx_world = dx_img * self._gsd_x
        dy_world = dy_img * self._gsd_y

        # Rotate by UAV yaw
        cos_yaw = np.cos(uav_yaw)
        sin_yaw = np.sin(uav_yaw)

        world_x = uav_x + dx_world * cos_yaw - dy_world * sin_yaw
        world_y = uav_y + dx_world * sin_yaw + dy_world * cos_yaw

        return world_x, world_y

    def compute_ground_footprint(self) -> Tuple[float, float]:
        """Compute the ground area covered by the UAV camera.

        Returns:
            (width_m, height_m) of the camera footprint on the water.
        """
        width_m = 2.0 * self.uav_altitude * np.tan(self.camera_fov_h / 2.0)
        height_m = 2.0 * self.uav_altitude * np.tan(self.camera_fov_v / 2.0)
        return width_m, height_m

    def is_in_fov(self, world_x: float, world_y: float, uav_x: float, uav_y: float) -> bool:
        """Check if a world point is within the UAV's current field of view.

        Args:
            world_x, world_y: Point to check.
            uav_x, uav_y: UAV nadir position.

        Returns:
            True if the point is visible.
        """
        half_w, half_h = self.compute_ground_footprint()
        half_w /= 2.0
        half_h /= 2.0

        dx = abs(world_x - uav_x)
        dy = abs(world_y - uav_y)

        return dx <= half_w and dy <= half_h
