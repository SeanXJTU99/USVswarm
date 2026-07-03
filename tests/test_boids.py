"""Tests for Boids flocking controller."""

import math
import pytest
from usv_swarm.boids_controller import BoidsController, BoidsParams, NeighborState


class TestBoidsController:
    """Test suite for Reynolds Boids flocking algorithm."""

    def test_no_neighbors(self) -> None:
        """Test that controller maintains current state with no neighbors."""
        ctrl = BoidsController()
        heading, speed = ctrl.compute_force(0.0, 0.0, 0.0, 0.5, [])
        assert heading == 0.0
        assert speed == 0.5

    def test_cohesion_pulls_toward_center(self) -> None:
        """Test that cohesion steers toward the average neighbor position."""
        params = BoidsParams(
            cohesion_weight=10.0,
            alignment_weight=0.0,
            separation_weight=0.0,
            perception_radius=50.0,
        )
        ctrl = BoidsController(params)

        # Two neighbors to the right
        neighbors = [
            NeighborState(x=5.0, y=0.0, heading=0.0, speed=0.5),
            NeighborState(x=7.0, y=0.0, heading=0.0, speed=0.5),
        ]

        heading, speed = ctrl.compute_force(0.0, 0.0, 0.0, 0.5, neighbors)
        # Should steer right (toward neighbors at x=6 average)
        assert abs(heading - 0.0) < 0.1  # directly right

    def test_separation_pushes_away(self) -> None:
        """Test that separation steers away from nearby neighbors."""
        params = BoidsParams(
            cohesion_weight=0.0,
            alignment_weight=0.0,
            separation_weight=10.0,
            separation_radius=3.0,
            perception_radius=50.0,
        )
        ctrl = BoidsController(params)

        # Neighbor directly ahead, very close
        neighbors = [
            NeighborState(x=1.0, y=0.0, heading=0.0, speed=0.5),
        ]

        heading, speed = ctrl.compute_force(0.0, 0.0, 0.0, 0.5, neighbors)
        # Should steer away (left or right, roughly π from neighbor)
        assert abs(abs(heading) - math.pi) < 0.2 or abs(heading) > 1.5

    def test_alignment_matches_heading(self) -> None:
        """Test that alignment steers toward the average neighbor heading."""
        params = BoidsParams(
            cohesion_weight=0.0,
            alignment_weight=10.0,
            separation_weight=0.0,
            perception_radius=50.0,
        )
        ctrl = BoidsController(params)

        # Two neighbors heading north (π/2)
        neighbors = [
            NeighborState(x=5.0, y=0.0, heading=math.pi / 2, speed=0.5),
            NeighborState(x=5.0, y=1.0, heading=math.pi / 2, speed=0.5),
        ]

        heading, speed = ctrl.compute_force(0.0, 0.0, 0.0, 0.5, neighbors)
        # Should favor the north direction
        assert heading > 0.5  # roughly northward

    def test_perception_radius_filters_distant_neighbors(self) -> None:
        """Test that neighbors beyond perception_radius are ignored."""
        params = BoidsParams(
            cohesion_weight=10.0,
            alignment_weight=0.0,
            separation_weight=0.0,
            perception_radius=5.0,
        )
        ctrl = BoidsController(params)

        # Far neighbor
        neighbors = [
            NeighborState(x=20.0, y=0.0, heading=0.0, speed=0.5),
        ]

        heading, speed = ctrl.compute_force(0.0, 0.0, 0.0, 0.5, neighbors)
        # No visible neighbors → maintain current state
        assert heading == 0.0
        assert speed == 0.5

    def test_max_speed_clamping(self) -> None:
        """Test that output speed never exceeds max_speed."""
        params = BoidsParams(
            cohesion_weight=10.0,
            alignment_weight=10.0,
            separation_weight=10.0,
            max_speed=1.5,
            perception_radius=50.0,
        )
        ctrl = BoidsController(params)

        neighbors = [
            NeighborState(x=20.0, y=0.0, heading=0.0, speed=2.0),
        ]

        _, speed = ctrl.compute_force(0.0, 0.0, 0.0, 0.5, neighbors)
        assert speed <= 1.5

    def test_speed_floor(self) -> None:
        """Test that speed never drops below the minimum creep speed."""
        ctrl = BoidsController()
        neighbors = [
            NeighborState(x=0.1, y=0.0, heading=0.0, speed=0.0),
        ]

        _, speed = ctrl.compute_force(0.0, 0.0, 0.0, 0.5, neighbors)
        assert speed >= 0.2
