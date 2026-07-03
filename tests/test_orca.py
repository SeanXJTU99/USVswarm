"""Tests for ORCA collision avoidance algorithm."""

import pytest
from usv_swarm.orca_avoidance import ORCAAvoidance, ORCAParams, AgentState


class TestORCAAvoidance:
    """Test suite for ORCA reciprocal collision avoidance."""

    def _make_agent(
        self, x: float, y: float, vx: float = 0.0, vy: float = 0.0, radius: float = 0.5
    ) -> AgentState:
        """Helper to create an AgentState."""
        return AgentState(position=(x, y), velocity=(vx, vy), radius=radius)

    def test_no_neighbors(self) -> None:
        """Test that preferred velocity is returned when no neighbors exist."""
        orca = ORCAAvoidance()
        own = self._make_agent(0.0, 0.0, 0.5, 0.0)
        preferred = (0.5, 0.0)

        safe_vx, safe_vy = orca.compute_safe_velocity(own, preferred, [])
        assert abs(safe_vx - 0.5) < 0.01
        assert abs(safe_vy - 0.0) < 0.01

    def test_stationary_obstacle_avoidance(self) -> None:
        """Test that ORCA avoids a stationary obstacle ahead."""
        params = ORCAParams(time_horizon=5.0, radius=0.5, max_speed=1.5, neighbor_dist=10.0)
        orca = ORCAAvoidance(params)

        own = self._make_agent(0.0, 0.0, 1.0, 0.0)  # Moving right at 1 m/s
        other = self._make_agent(2.0, 0.0, 0.0, 0.0)  # Stationary obstacle ahead

        preferred = (1.0, 0.0)
        safe_vx, safe_vy = orca.compute_safe_velocity(own, preferred, [other])

        # Should deviate from straight-ahead to avoid collision
        # Either slowing down or steering sideways
        assert not (abs(safe_vx - 1.0) < 0.01 and abs(safe_vy) < 0.01)

    def test_head_on_collision(self) -> None:
        """Test reciprocal avoidance for head-on collision."""
        params = ORCAParams(time_horizon=5.0, radius=0.5, max_speed=1.5, neighbor_dist=10.0)
        orca = ORCAAvoidance(params)

        own = self._make_agent(0.0, 0.0, 1.0, 0.0)      # Moving right
        other = self._make_agent(4.0, 0.0, -1.0, 0.0)    # Moving left (toward us)

        preferred = (1.0, 0.0)
        safe_vx, safe_vy = orca.compute_safe_velocity(own, preferred, [other])

        # ORCA should suggest a lateral deviation (both agents share responsibility)
        assert abs(safe_vy) > 0.01  # Should have some lateral component

    def test_no_collision_risk(self) -> None:
        """Test that no avoidance is applied when there's no collision risk."""
        params = ORCAParams(time_horizon=1.0, radius=0.5, max_speed=1.5, neighbor_dist=10.0)
        orca = ORCAAvoidance(params)

        own = self._make_agent(0.0, 0.0, 1.0, 0.0)      # Moving right
        other = self._make_agent(0.0, 10.0, 0.0, 0.0)    # Far away, not on collision course

        preferred = (1.0, 0.0)
        safe_vx, safe_vy = orca.compute_safe_velocity(own, preferred, [other])

        # Should keep preferred velocity (no collision risk)
        assert abs(safe_vx - 1.0) < 0.1

    def test_collision_imminent_detection(self) -> None:
        """Test the collision prediction helper."""
        orca = ORCAAvoidance()

        # Two agents on collision course
        own = self._make_agent(0.0, 0.0, 1.0, 0.0)
        other = self._make_agent(2.0, 0.0, -1.0, 0.0)

        assert orca.is_collision_imminent(own, other, time_horizon=3.0)

    def test_collision_not_imminent(self) -> None:
        """Test that parallel courses don't trigger collision."""
        orca = ORCAAvoidance()

        own = self._make_agent(0.0, 0.0, 1.0, 0.0)
        other = self._make_agent(0.0, 5.0, 1.0, 0.0)  # Parallel, 5m apart

        assert not orca.is_collision_imminent(own, other, time_horizon=5.0)

    def test_velocity_in_bounds(self) -> None:
        """Test that output velocity never exceeds max_speed."""
        params = ORCAParams(max_speed=1.5)
        orca = ORCAAvoidance(params)

        own = self._make_agent(0.0, 0.0, 0.5, 0.0)
        preferred = (10.0, 0.0)  # Way above max

        safe_vx, safe_vy = orca.compute_safe_velocity(own, preferred, [])
        speed = (safe_vx**2 + safe_vy**2) ** 0.5
        assert speed <= 1.5 + 0.01
