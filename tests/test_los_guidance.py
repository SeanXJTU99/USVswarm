"""Tests for LOS guidance law."""

import math
import pytest
from usv_control.los_guidance import LOSGuidance


class TestLOSGuidance:
    """Test suite for Line-of-Sight guidance controller."""

    def test_initialization(self) -> None:
        """Test that LOS guidance initializes with correct defaults."""
        los = LOSGuidance(lookahead_base=2.0, lookahead_speed_gain=1.5, acceptance_radius=1.0)
        assert los.lookahead_base == 2.0
        assert los.lookahead_speed_gain == 1.5
        assert los.acceptance_radius == 1.0
        assert los.is_finished

    def test_set_waypoints(self) -> None:
        """Test that waypoints are correctly loaded."""
        los = LOSGuidance()
        waypoints = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        los.set_waypoints(waypoints)
        assert not los.is_finished

    def test_heading_to_single_waypoint(self) -> None:
        """Test heading calculation for a single waypoint directly ahead."""
        los = LOSGuidance(lookahead_base=2.0, lookahead_speed_gain=0.0)
        los.set_waypoints([(10.0, 0.0)])

        # USV at origin, heading east, stationary
        heading, speed, done = los.compute_heading(0.0, 0.0, 0.0, 0.0)
        # Should point toward (10, 0)
        assert abs(heading - 0.0) < 0.01
        assert not done

    def test_heading_90_degrees(self) -> None:
        """Test heading when waypoint is perpendicular."""
        los = LOSGuidance(lookahead_base=2.0, lookahead_speed_gain=0.0)
        los.set_waypoints([(0.0, 10.0)])

        heading, speed, done = los.compute_heading(0.0, 0.0, 0.0, 0.0)
        assert abs(heading - math.pi / 2) < 0.01

    def test_waypoint_arrival(self) -> None:
        """Test that the controller detects waypoint arrival."""
        los = LOSGuidance(acceptance_radius=1.0)
        los.set_waypoints([(10.0, 0.0), (20.0, 0.0)])

        # USV very close to first waypoint
        heading, speed, done = los.compute_heading(9.5, 0.0, 0.5, 0.0)
        # Should have advanced to second waypoint
        assert not done
        # Should now point toward (20, 0)
        assert abs(heading - 0.0) < 0.1

    def test_last_waypoint_arrival(self) -> None:
        """Test that controller reports done after final waypoint."""
        los = LOSGuidance(acceptance_radius=1.0)
        los.set_waypoints([(10.0, 0.0)])

        # USV at the waypoint
        heading, speed, done = los.compute_heading(10.0, 0.0, 0.5, 0.0)
        assert done
        assert speed == 0.0

    def test_speed_adaptive_lookahead(self) -> None:
        """Test that look-ahead distance increases with speed."""
        los = LOSGuidance(lookahead_base=2.0, lookahead_speed_gain=1.5)
        los.set_waypoints([(20.0, 0.0)])

        # At high speed, the heading should be similar (path-following, not chasing)
        _, speed_slow, _ = los.compute_heading(0.0, 0.0, 0.2, 0.0)
        _, speed_fast, _ = los.compute_heading(0.0, 0.0, 1.0, 0.0)
        # Fast speed gives higher desired speed (closer to cruise)
        assert speed_fast > speed_slow

    def test_empty_waypoints(self) -> None:
        """Test behavior with no waypoints."""
        los = LOSGuidance()
        heading, speed, done = los.compute_heading(0.0, 0.0, 0.5, 0.0)
        assert done
        assert speed == 0.0
