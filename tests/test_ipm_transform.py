"""Tests for inverse perspective mapping (IPM) transform."""

import numpy as np
import pytest
from usv_navigation.ipm_transform import IPMTransformer


class TestIPMTransformer:
    """Test suite for IPM with IMU-based de-jitter."""

    @pytest.fixture
    def camera_matrix(self) -> np.ndarray:
        """Standard pinhole camera matrix (640×480, ~90° HFOV)."""
        return np.array([
            [400.0, 0.0, 320.0],
            [0.0, 400.0, 240.0],
            [0.0, 0.0, 1.0],
        ])

    def test_initialization(self, camera_matrix: np.ndarray) -> None:
        """Test that IPMTransformer initializes with correct parameters."""
        ipm = IPMTransformer(camera_matrix, camera_height=0.33, pitch_offset=-0.3)
        assert ipm.camera_height == 0.33
        assert ipm.pitch_offset == -0.3
        assert ipm._homography is None

    def test_homography_computation(self, camera_matrix: np.ndarray) -> None:
        """Test that a valid homography is computed from IMU data."""
        ipm = IPMTransformer(camera_matrix, camera_height=0.33, pitch_offset=-0.3)
        ipm.update_imu(pitch=-0.1, roll=0.02)

        assert ipm._homography is not None
        assert ipm._homography.shape == (3, 3)

    def test_pixel_to_world_center(self, camera_matrix: np.ndarray) -> None:
        """Test projection of image center to world coordinates."""
        ipm = IPMTransformer(camera_matrix, camera_height=0.33, pitch_offset=-0.3)

        # Image center, camera level
        wx, wy = ipm.pixel_to_world(320.0, 240.0, pitch=-0.3, roll=0.0)
        # Center pixel should project roughly straight ahead
        assert wy > 0  # ahead of the camera

    def test_pixel_to_world_bottom(self, camera_matrix: np.ndarray) -> None:
        """Test that pixels near the bottom of the image map closer."""
        ipm = IPMTransformer(camera_matrix, camera_height=0.33, pitch_offset=-0.3)

        wx_far, wy_far = ipm.pixel_to_world(320.0, 100.0, pitch=-0.3, roll=0.0)
        wx_near, wy_near = ipm.pixel_to_world(320.0, 400.0, pitch=-0.3, roll=0.0)

        # Bottom pixels should be closer
        assert wy_near < wy_far

    def test_imu_update_threshold(self, camera_matrix: np.ndarray) -> None:
        """Test that homography is not recomputed for tiny IMU changes."""
        ipm = IPMTransformer(camera_matrix, camera_height=0.33)
        ipm.update_imu(pitch=0.0, roll=0.0)
        h1 = ipm._homography.copy()

        # Tiny change (0.001 rad ≈ 0.06°)
        ipm.update_imu(pitch=0.001, roll=0.0)
        h2 = ipm._homography

        # Should be the same (below threshold)
        assert h1 is h2  # same object reference

    def test_imu_update_above_threshold(self, camera_matrix: np.ndarray) -> None:
        """Test that homography is recomputed for significant IMU changes."""
        ipm = IPMTransformer(camera_matrix, camera_height=0.33)
        ipm.update_imu(pitch=0.0, roll=0.0)
        h1 = ipm._homography.copy()

        # Significant change (0.05 rad ≈ 3°)
        ipm.update_imu(pitch=0.05, roll=0.0)
        h2 = ipm._homography

        # Should be recomputed
        assert not np.allclose(h1, h2)

    def test_bbox_to_world(self, camera_matrix: np.ndarray) -> None:
        """Test bounding box projection to world coordinates."""
        ipm = IPMTransformer(camera_matrix, camera_height=0.33, pitch_offset=-0.3)

        bbox = (320.0, 300.0, 80.0, 60.0)  # centered, near bottom
        wx, wy, ww, wh = ipm.bbox_to_world(bbox, pitch=-0.3, roll=0.0)

        # Should return reasonable world coordinates
        assert wy > 0
        assert ww > 0
        assert wh > 0

    def test_pixel_scale_positive(self, camera_matrix: np.ndarray) -> None:
        """Test that pixel-to-meter scale is positive."""
        ipm = IPMTransformer(camera_matrix, camera_height=0.33)
        scale = ipm._pixel_scale_at(1.0, 5.0)
        assert scale > 0
