"""Line-of-Sight (LOS) guidance law for USV path following.

Implements a variable look-ahead distance LOS algorithm.
Given a planned path (waypoints) and current vessel pose, computes the
desired heading angle and speed to smoothly converge onto the path.

The look-ahead distance adapts to speed:
  - Fast → look further ahead (stable, less correction)
  - Slow → look closer (agile, precise tracking near waypoints)
"""

from typing import List, Optional, Tuple

import numpy as np


class LOSGuidance:
    """Line-of-Sight guidance controller.

    The guidance layer sits between the path planner (A*) and the
    low-level controller (cascaded PID). It converts waypoints into
    real-time heading and speed references.

    Attributes:
        lookahead_base: Base look-ahead distance (m).
        lookahead_speed_gain: Additional look-ahead per m/s of speed.
        acceptance_radius: Waypoint arrival radius (m).
    """

    def __init__(
        self,
        lookahead_base: float = 2.0,
        lookahead_speed_gain: float = 1.5,
        acceptance_radius: float = 1.0,
    ) -> None:
        """Initialize LOS guidance.

        Args:
            lookahead_base: Minimum look-ahead distance in meters.
            lookahead_speed_gain: Multiplier for speed-dependent look-ahead extension.
            acceptance_radius: Circle of acceptance around waypoints (m).
        """
        self.lookahead_base: float = lookahead_base
        self.lookahead_speed_gain: float = lookahead_speed_gain
        self.acceptance_radius: float = acceptance_radius

        self._waypoints: List[Tuple[float, float]] = []
        self._current_wp_idx: int = 0

    def set_waypoints(self, waypoints: List[Tuple[float, float]]) -> None:
        """Load a new set of waypoints for the guidance law to track.

        Args:
            waypoints: List of (x, y) waypoints in world frame.
        """
        self._waypoints = waypoints
        self._current_wp_idx = 0

    def compute_heading(
        self,
        pos_x: float,
        pos_y: float,
        current_speed: float,
        current_heading: float,
    ) -> Tuple[float, float, bool]:
        """Compute the desired heading angle to follow the current path segment.

        Args:
            pos_x: Current USV X position (m, world frame).
            pos_y: Current USV Y position (m, world frame).
            current_speed: Current forward speed (m/s).
            current_heading: Current heading angle (radians, world frame).

        Returns:
            (desired_heading_rad, desired_speed_mps, waypoint_reached).
        """
        if not self._waypoints or self._current_wp_idx >= len(self._waypoints):
            return current_heading, 0.0, True

        target = self._waypoints[self._current_wp_idx]
        dist_to_wp = np.hypot(target[0] - pos_x, target[1] - pos_y)

        # Check waypoint arrival
        if dist_to_wp < self.acceptance_radius:
            self._current_wp_idx += 1
            if self._current_wp_idx >= len(self._waypoints):
                return current_heading, 0.0, True
            target = self._waypoints[self._current_wp_idx]
            dist_to_wp = np.hypot(target[0] - pos_x, target[1] - pos_y)

        # Variable look-ahead distance (speed-adaptive)
        lookahead = self.lookahead_base + self.lookahead_speed_gain * current_speed

        # Line-of-sight angle to target
        alpha = np.arctan2(target[1] - pos_y, target[0] - pos_x)

        # Cross-track error compensation
        # For path-following (not just point-tracking), compute the projection
        # onto the current path segment and steer toward a point lookahead meters
        # ahead along the segment.
        if self._current_wp_idx > 0:
            prev_wp = self._waypoints[self._current_wp_idx - 1]
            alpha = self._los_on_segment(
                prev_wp, target, pos_x, pos_y, lookahead
            )

        return alpha, self._desired_speed(dist_to_wp), False

    def _los_on_segment(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        px: float,
        py: float,
        lookahead: float,
    ) -> float:
        """Compute LOS angle constrained to a path segment.

        Projects the vessel position onto the segment line, finds the
        look-ahead point at distance `lookahead` ahead of the projection,
        and returns the angle to that virtual target point.

        Args:
            p0: Segment start point (x, y).
            p1: Segment end point (x, y).
            px: Current vessel X position.
            py: Current vessel Y position.
            lookahead: Look-ahead distance (m).

        Returns:
            Desired heading angle (radians).
        """
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len_sq = dx**2 + dy**2

        if seg_len_sq < 1e-6:
            return np.arctan2(p1[1] - py, p1[0] - px)

        # Project vessel position onto the segment
        t = ((px - p0[0]) * dx + (py - p0[1]) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))  # clamp to segment

        proj_x = p0[0] + t * dx
        proj_y = p0[1] + t * dy

        # Look-ahead point: project + lookahead along the segment direction
        seg_length = np.sqrt(seg_len_sq)
        ahead_x = proj_x + (dx / seg_length) * lookahead
        ahead_y = proj_y + (dy / seg_length) * lookahead

        return np.arctan2(ahead_y - py, ahead_x - px)

    def _desired_speed(self, dist_to_wp: float) -> float:
        """Compute desired speed based on distance to next waypoint.

        Slows down when approaching waypoints to avoid overshoot.

        Args:
            dist_to_wp: Distance to the current waypoint (m).

        Returns:
            Desired speed (m/s).
        """
        if dist_to_wp < self.acceptance_radius * 3:
            return 0.3  # slow approach
        return 0.8  # cruise speed (m/s)

    @property
    def is_finished(self) -> bool:
        """Whether all waypoints have been visited."""
        return self._current_wp_idx >= len(self._waypoints)
