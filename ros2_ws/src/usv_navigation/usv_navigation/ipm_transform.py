"""Inverse Perspective Mapping (IPM) for water surface.

Converts front-facing camera perspective view to bird's-eye view (BEV)
for building 2D occupancy grid maps used by path planners.

Includes real-time IMU pitch/roll compensation to counteract wave-induced
camera jitter that would otherwise cause obstacle coordinate drift.
"""

from typing import Optional, Tuple

import numpy as np


class IPMTransformer:
    """Inverse perspective mapping with IMU-based dynamic de-jitter.

    Transforms pixel-space detections to world-frame coordinates on the
    water surface plane (z = 0), accounting for camera pitch/roll from waves.

    Attributes:
        camera_matrix: 3x3 intrinsic camera matrix.
        camera_height: Height of camera above water surface (m).
        pitch_offset: Mounting pitch offset (radians).
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        camera_height: float = 0.33,
        pitch_offset: float = -0.3,
    ) -> None:
        """Initialize IPM transformer.

        Args:
            camera_matrix: 3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
            camera_height: Camera mounting height above waterline (m).
            pitch_offset: Fixed camera pitch angle from horizontal (radians).
                          Negative = looking down.
        """
        self.camera_matrix: np.ndarray = camera_matrix
        self.camera_height: float = camera_height
        self.pitch_offset: float = pitch_offset

        # Cached homography (recomputed on IMU update)
        self._homography: Optional[np.ndarray] = None
        self._last_pitch: float = 0.0
        self._last_roll: float = 0.0

    def update_imu(self, pitch: float, roll: float) -> None:
        """Update cached homography with latest IMU attitude.

        Call this at IMU rate (~100 Hz) to keep the homography current.
        The homography is deliberately recomputed only when pitch or roll
        changes by more than 0.5 degrees to avoid unnecessary math.

        Args:
            pitch: Current camera pitch angle (radians, positive = nose up).
            roll: Current camera roll angle (radians, positive = roll right).
        """
        delta = abs(pitch - self._last_pitch) + abs(roll - self._last_roll)
        if delta < 0.0087:  # ~0.5 degrees
            return

        self._last_pitch = pitch
        self._last_roll = roll
        self._homography = self._compute_homography(pitch, roll)

    def _compute_homography(self, pitch: float, roll: float) -> np.ndarray:
        """Compute 3x3 homography from image plane to ground plane.

        Uses the camera extrinsic rotation (from IMU) and height to map
        each image pixel (u, v) to a world ground point (X, Y, Z=0).

        Args:
            pitch: Current effective pitch (mount_offset + IMU pitch).
            roll: Current effective roll (IMU roll).

        Returns:
            3x3 homography matrix: pixel → ground-plane coordinates.
        """
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Rotation: camera → world (small-angle approximation for pitch/roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        # Homography from image to ground (z=0 plane)
        # Derived from pinhole camera model with known height.
        # H = K * [r1, r2, t] where r1,r2 are columns of R and t = [0, 0, -h]^T
        r1 = np.array([cr, sr * sp, -sr * cp])
        r2 = np.array([0, cp, sp])
        t = np.array([0, 0, -self.camera_height])

        H = np.column_stack([r1, r2, t])
        H = self.camera_matrix @ H

        return H

    def pixel_to_world(
        self, u: float, v: float, pitch: float = 0.0, roll: float = 0.0
    ) -> Tuple[float, float]:
        """Project a single pixel to world ground coordinates.

        Args:
            u: Horizontal pixel coordinate.
            v: Vertical pixel coordinate.
            pitch: Current IMU pitch (radians), if not pre-updated.
            roll: Current IMU roll (radians), if not pre-updated.

        Returns:
            (X, Y) world coordinates on the water surface in camera frame.
        """
        if self._homography is None or pitch != self._last_pitch or roll != self._last_roll:
            self.update_imu(pitch, roll)

        assert self._homography is not None, "Homography must be computed before projection"

        pixel = np.array([u, v, 1.0])
        ground = self._homography @ pixel
        ground /= ground[2]  # normalize homogeneous coordinate

        return float(ground[0]), float(ground[1])

    def bbox_to_world(
        self,
        bbox: Tuple[float, float, float, float],
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """Project a bounding box to world coordinates.

        Projects the bottom-center point of the bounding box, as this
        corresponds to the contact point with the water surface.

        Args:
            bbox: (center_x, center_y, width, height) in pixel coordinates.
            pitch: Current IMU pitch.
            roll: Current IMU roll.

        Returns:
            (world_x, world_y, world_width_m, world_height_m).
        """
        cx, cy, w, h = bbox
        # Use bottom-center for ground projection
        wx, wy = self.pixel_to_world(cx, cy + h / 2, pitch, roll)

        # Approximate world size from pixel size using homography scale
        scale = self._pixel_scale_at(wx, wy)
        ww = w * scale
        wh = h * scale

        return wx, wy, ww, wh

    def _pixel_scale_at(self, world_x: float, world_y: float) -> float:
        """Estimate meters-per-pixel at a given world point.

        Uses similar-triangles approximation based on camera height
        and focal length.

        Args:
            world_x: X coordinate in world frame.
            world_y: Y coordinate (distance ahead) in world frame.

        Returns:
            Scale factor in meters per pixel.
        """
        fx = self.camera_matrix[0, 0]
        distance = np.sqrt(world_x**2 + world_y**2 + self.camera_height**2)
        return distance / fx
