"""Leader-Follower formation control using L-α (distance-angle) strategy.

In a leader-follower formation:
  - The leader follows a global path (from A* or teleoperation).
  - Followers maintain fixed relative distance (L) and bearing angle (α)
    from the leader.

The L-α control law:
  1. Compute the follower's desired position based on leader's pose + (L, α)
  2. Use PID / LOS to steer the follower to that desired position
  3. If the leader turns, followers automatically adjust to maintain formation

This is a centralized-adjacent approach: followers need the leader's pose,
but once received, each follower plans locally.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class FormationShape:
    """Defines the desired formation geometry.

    Each follower is specified by (L, α):
      L: desired distance behind/ahead of the leader (m).
      α: desired bearing angle relative to leader's heading (radians).
         0 = directly ahead, π = directly behind, π/2 = right, -π/2 = left.
    """

    distance: float
    angle: float


class LeaderFollowerController:
    """L-α leader-follower formation controller.

    Given the leader's pose and a desired formation offset (L, α),
    computes the target position for this follower and generates
    heading/speed commands to reach it.
    """

    def __init__(
        self,
        formation: FormationShape,
        k_distance: float = 0.8,
        k_angle: float = 1.2,
    ) -> None:
        """Initialize the leader-follower controller.

        Args:
            formation: Desired formation offset (L, α).
            k_distance: Proportional gain for distance error correction.
            k_angle: Proportional gain for angle error correction.
        """
        self.formation: FormationShape = formation
        self.k_distance: float = k_distance
        self.k_angle: float = k_angle

    def compute_target(
        self,
        leader_x: float,
        leader_y: float,
        leader_heading: float,
    ) -> Tuple[float, float]:
        """Compute the follower's target (x, y) based on leader pose.

        The target is offset from the leader by distance L at angle α
        relative to the leader's heading.

        Args:
            leader_x, leader_y: Leader's world position.
            leader_heading: Leader's heading angle (radians).

        Returns:
            (target_x, target_y) world coordinates.
        """
        # Offset direction = leader heading + formation bearing angle
        angle = leader_heading + self.formation.angle
        target_x = leader_x + self.formation.distance * np.cos(angle)
        target_y = leader_y + self.formation.distance * np.sin(angle)
        return target_x, target_y

    def compute_control(
        self,
        own_x: float,
        own_y: float,
        own_heading: float,
        leader_x: float,
        leader_y: float,
        leader_heading: float,
    ) -> Tuple[float, float]:
        """Compute heading and speed commands for formation keeping.

        Args:
            own_x, own_y: Follower's current position.
            own_heading: Follower's current heading (radians).
            leader_x, leader_y: Leader's current position.
            leader_heading: Leader's current heading (radians).

        Returns:
            (desired_heading_rad, desired_speed_mps).
        """
        target_x, target_y = self.compute_target(
            leader_x, leader_y, leader_heading
        )

        # Distance error
        dx = target_x - own_x
        dy = target_y - own_y
        dist_error = np.sqrt(dx**2 + dy**2)

        # Angle to target
        angle_to_target = np.arctan2(dy, dx)

        # Heading error
        heading_error = angle_to_target - own_heading
        # Normalize to [-π, π]
        heading_error = np.arctan2(
            np.sin(heading_error), np.cos(heading_error)
        )

        # Control law: speed proportional to distance, heading proportional to error
        desired_speed = min(self.k_distance * dist_error, 1.5)
        desired_speed = max(0.1, desired_speed)

        desired_heading = own_heading + self.k_angle * heading_error

        return desired_heading, desired_speed

    def update_formation(self, distance: float, angle: float) -> None:
        """Update the formation offset dynamically (e.g., formation switch).

        Args:
            distance: New desired distance (m).
            angle: New desired bearing angle (radians).
        """
        self.formation.distance = distance
        self.formation.angle = angle
