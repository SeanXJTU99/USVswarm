"""Camera-IMU extrinsic calibration for USV perception stabilization.

Estimates the rigid 6-DOF transform between the camera and IMU frames.
Critical for IPM accuracy — without this calibration, wave-induced camera
pitch/roll causes meter-level obstacle coordinate drift on the costmap.

Uses hand-eye calibration: matches camera visual odometry with IMU
preintegration over synchronized segments.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


class CameraIMUCalibrator:
    """Hand-eye calibration between camera and IMU.

    Collects synchronized camera-IMU motion segments over time,
    then solves AX = XB for the unknown extrinsic transform T_cam_imu.
    """

    def __init__(self, min_segments: int = 10) -> None:
        """Initialize calibrator.

        Args:
            min_segments: Minimum motion segments before solving.
        """
        self.min_segments: int = min_segments

        # Buffered motion segments
        self._cam_motions: list[np.ndarray] = []   # list of 4x4
        self._imu_motions: list[np.ndarray] = []   # list of 4x4

    def add_motion_segment(
        self,
        cam_delta: np.ndarray,
        imu_delta: np.ndarray,
    ) -> None:
        """Add a synchronized camera-IMU motion segment.

        Args:
            cam_delta: Camera frame delta transform (4x4), from visual odometry.
            imu_delta: IMU delta transform (4x4), from IMU preintegration.
        """
        self._cam_motions.append(cam_delta)
        self._imu_motions.append(imu_delta)

    def solve(self) -> Optional[np.ndarray]:
        """Solve AX = XB for the camera-IMU extrinsic transform.

        Uses the Tsai-Lenz method:
          1. Solve rotation: R_cam * R_x = R_x * R_imu
          2. Solve translation: (I - R_cam) * t_x = t_cam - R_x * t_imu

        Returns:
            4x4 transformation matrix T_cam_imu, or None if insufficient data.
        """
        if len(self._cam_motions) < self.min_segments:
            return None

        # Extract rotations and translations from each segment
        R_cam_list, t_cam_list = [], []
        R_imu_list, t_imu_list = [], []

        for cam, imu in zip(self._cam_motions, self._imu_motions):
            R_cam_list.append(cam[:3, :3])
            t_cam_list.append(cam[:3, 3])
            R_imu_list.append(imu[:3, :3])
            t_imu_list.append(imu[:3, 3])

        # Solve rotation (axis-angle method)
        R_x = self._solve_rotation(R_cam_list, R_imu_list)
        if R_x is None:
            return None

        # Solve translation
        t_x = self._solve_translation(R_cam_list, t_cam_list, R_imu_list, t_imu_list, R_x)
        if t_x is None:
            return None

        # Build 4x4 transform
        T = np.eye(4)
        T[:3, :3] = R_x
        T[:3, 3] = t_x
        return T

    def _solve_rotation(
        self,
        R_cam: list[np.ndarray],
        R_imu: list[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Solve R_cam * R_x = R_x * R_imu for R_x.

        Each pair gives a constraint: log(R_cam) = R_x * log(R_imu) * R_x^T.
        Stacked into a linear system in the Lie algebra.

        Args:
            R_cam: Camera rotation matrices.
            R_imu: IMU rotation matrices.

        Returns:
            3x3 rotation matrix R_x, or None on failure.
        """
        n = len(R_cam)
        M = np.zeros((3 * n, 3))

        for i in range(n):
            # Axis-angle of camera rotation
            r_cam = R.from_matrix(R_cam[i])
            angle_cam = r_cam.magnitude()
            if angle_cam < 1e-6:
                continue
            axis_cam = r_cam.as_rotvec() / angle_cam

            r_imu = R.from_matrix(R_imu[i])
            angle_imu = r_imu.magnitude()
            if angle_imu < 1e-6:
                continue
            axis_imu = r_imu.as_rotvec() / angle_imu

            # Skew-symmetric cross product matrix
            cross = np.array([
                [0, -axis_imu[2], axis_imu[1]],
                [axis_imu[2], 0, -axis_imu[0]],
                [-axis_imu[1], axis_imu[0], 0],
            ])

            M[3*i:3*(i+1), :] = cross + cross @ cross / (1 + np.dot(axis_cam, axis_imu))

        try:
            U, _, Vt = np.linalg.svd(M)
            x = Vt[-1, :]
            x /= np.linalg.norm(x)
        except np.linalg.LinAlgError:
            return None

        # Rodrigues: axis-angle → rotation matrix
        angle = np.linalg.norm(x)
        if angle < 1e-9:
            return np.eye(3)
        axis = x / angle
        return R.from_rotvec(axis * angle).as_matrix()

    def _solve_translation(
        self,
        R_cam: list[np.ndarray],
        t_cam: list[np.ndarray],
        R_imu: list[np.ndarray],
        t_imu: list[np.ndarray],
        R_x: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Solve (I - R_cam) * t_x = t_cam - R_x * t_imu for t_x.

        Args:
            R_cam: Camera rotation matrices.
            t_cam: Camera translation vectors.
            R_imu: IMU rotation matrices.
            t_imu: IMU translation vectors.
            R_x: Previously solved rotation.

        Returns:
            3-element translation vector t_x, or None on failure.
        """
        n = len(R_cam)
        A = np.zeros((3 * n, 3))
        b = np.zeros(3 * n)

        for i in range(n):
            A[3*i:3*(i+1), :] = np.eye(3) - R_cam[i]
            b[3*i:3*(i+1)] = t_cam[i] - R_x @ t_imu[i]

        try:
            t_x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None

        return t_x
