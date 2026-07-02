"""ORCA (Optimal Reciprocal Collision Avoidance) for multi-USV.

ORCA is the standard local collision avoidance algorithm for decentralized
multi-agent systems. Each agent computes a velocity obstacle (VO) for every
other agent, then solves a linear program to find the closest admissible
velocity to its preferred velocity.

Key property: each agent assumes the OTHER agent will also take half the
responsibility for avoidance. This reciprocity prevents the oscillations
and deadlocks that plague purely reactive approaches.

For USV swarms, ORCA is applied in the horizontal plane (2D velocity space)
since all vessels operate at the same water surface level.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ORCAParams:
    """ORCA algorithm parameters.

    Attributes:
        time_horizon: Look-ahead time for collision prediction (s).
        radius: Collision radius of each USV (m) — hull size + safety margin.
        max_speed: Maximum vessel speed (m/s).
        neighbor_dist: Maximum distance to consider another USV a neighbor (m).
    """

    time_horizon: float = 5.0
    radius: float = 0.5
    max_speed: float = 1.5
    neighbor_dist: float = 10.0


@dataclass
class AgentState:
    """State of a single agent for ORCA computation."""

    position: Tuple[float, float]
    velocity: Tuple[float, float]
    radius: float


class ORCAAvoidance:
    """Optimal Reciprocal Collision Avoidance for USV swarms.

    Solves the 2-agent ORCA half-plane constraint for each neighbor,
    then finds the closest admissible velocity to the preferred velocity
    by solving the intersection of all half-plane constraints.

    This is a lightweight implementation suitable for embedded boards
    (Jetson Nano, Raspberry Pi). Uses linear programming over a sampled
    set of candidate velocities rather than full LP solving.
    """

    # Candidate velocities to sample (polar grid)
    _SPEED_SAMPLES: int = 8
    _ANGLE_SAMPLES: int = 16

    def __init__(self, params: Optional[ORCAParams] = None) -> None:
        """Initialize ORCA avoidance.

        Args:
            params: Algorithm parameters. Uses defaults if None.
        """
        self.params: ORCAParams = params or ORCAParams()

    def compute_safe_velocity(
        self,
        own_state: AgentState,
        preferred_velocity: Tuple[float, float],
        neighbors: List[AgentState],
    ) -> Tuple[float, float]:
        """Compute an ORCA-safe velocity closest to the preferred velocity.

        Args:
            own_state: Own position, velocity, and radius.
            preferred_velocity: Desired velocity (from Boids / path planner).
            neighbors: States of all neighboring USVs.

        Returns:
            (vx, vy) safe velocity vector.
        """
        # Build half-plane constraints from each neighbor
        constraints: List[Tuple[np.ndarray, float]] = []

        for neighbor in neighbors:
            constraint = self._orca_constraint(own_state, neighbor)
            if constraint is not None:
                constraints.append(constraint)

        if not constraints:
            return preferred_velocity

        # Find best admissible velocity by sampling
        best_vel = (0.0, 0.0)
        best_cost = float("inf")

        pv = np.array(preferred_velocity)

        for speed in np.linspace(0, self.params.max_speed, self._SPEED_SAMPLES):
            for angle in np.linspace(0, 2 * np.pi, self._ANGLE_SAMPLES):
                vx = speed * np.cos(angle)
                vy = speed * np.sin(angle)
                v = np.array([vx, vy])

                # Check all constraints
                admissible = True
                for normal, limit in constraints:
                    if np.dot(normal, v) > limit:
                        admissible = False
                        break

                if admissible:
                    cost = np.linalg.norm(v - pv)
                    if cost < best_cost:
                        best_cost = cost
                        best_vel = (vx, vy)

        return best_vel

    def _orca_constraint(
        self,
        own: AgentState,
        other: AgentState,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Compute the ORCA half-plane constraint for one neighbor.

        The constraint is of the form: normal · v ≤ limit,
        meaning admissible velocities are those on the correct side
        of a line in velocity space.

        Args:
            own: Own agent state.
            other: Neighbor agent state.

        Returns:
            (normal, limit) defining the half-plane, or None if no
            constraint is needed (no imminent collision).
        """
        tau = self.params.time_horizon
        combined_radius = own.radius + other.radius

        # Relative position and velocity
        p_rel = np.array([other.position[0] - own.position[0],
                          other.position[1] - own.position[1]])
        v_rel = np.array([own.velocity[0] - other.velocity[0],
                          own.velocity[1] - other.velocity[1]])

        dist = np.linalg.norm(p_rel)
        if dist < 0.01:
            # Agents are overlapping — emergency separation
            direction = p_rel / 0.01
            return (direction, -0.5 * self.params.max_speed)

        # Time to closest approach
        t_closest = -np.dot(p_rel, v_rel) / max(np.dot(v_rel, v_rel), 1e-6)
        t_closest = max(0.0, min(t_closest, tau))

        # Predicted positions at closest approach
        p_closest = p_rel + v_rel * t_closest
        dist_closest = np.linalg.norm(p_closest)

        if dist_closest > combined_radius:
            return None  # No collision risk

        # ORCA half-plane
        # The avoidance responsibility is split: each agent takes half.
        # u is the required change in relative velocity.
        if dist_closest < 0.01:
            w = p_rel / dist  # unit vector from own to other
        else:
            w = p_closest / dist_closest

        u = w * (combined_radius - dist_closest) / tau - v_rel

        # Each agent takes half the responsibility
        avoidance = 0.5 * u

        # Constraint normal points in direction of required change
        n = avoidance / max(np.linalg.norm(avoidance), 1e-6)
        limit = np.dot(n, np.add(own.velocity, avoidance))

        return (n, limit)

    def is_collision_imminent(
        self,
        own_state: AgentState,
        other_state: AgentState,
        time_horizon: Optional[float] = None,
    ) -> bool:
        """Check if a collision with another agent is imminent.

        Args:
            own_state: Own state.
            other_state: Other agent state.
            time_horizon: Look-ahead time (defaults to params.time_horizon).

        Returns:
            True if a collision is predicted within the time horizon.
        """
        tau = time_horizon or self.params.time_horizon
        combined_radius = own_state.radius + other_state.radius

        p_rel = np.array([other_state.position[0] - own_state.position[0],
                          other_state.position[1] - own_state.position[1]])
        v_rel = np.array([own_state.velocity[0] - other_state.velocity[0],
                          own_state.velocity[1] - other_state.velocity[1]])

        dist = np.linalg.norm(p_rel)
        if dist < combined_radius:
            return True  # Already colliding

        t_closest = -np.dot(p_rel, v_rel) / max(np.dot(v_rel, v_rel), 1e-6)
        if t_closest < 0 or t_closest > tau:
            return False

        p_closest = p_rel + v_rel * t_closest
        return np.linalg.norm(p_closest) < combined_radius
