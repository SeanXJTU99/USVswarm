"""Reynolds Boids model for decentralized USV swarm formation.

Implements the three classic Reynolds rules as virtual force vectors:
  1. Cohesion  — steer toward the average position of neighbors
  2. Alignment — match the average heading and speed of neighbors
  3. Separation — steer away from neighbors that are too close

These three forces are weighted and summed to produce a resultant
force vector, which is converted to (desired_heading, desired_speed)
and fed into the cascaded PID controller.

The Boids model is fully decentralized — each USV runs its own
instance using only locally observable neighbor states (no central
planner). This is what makes it a "swarm" rather than a "fleet."
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class BoidsParams:
    """Tunable parameters for the Boids flocking behavior.

    Attributes:
        cohesion_weight: How strongly to steer toward the swarm center.
        alignment_weight: How strongly to match neighbors' velocity.
        separation_weight: How strongly to avoid nearby neighbors.
        perception_radius: Maximum distance to consider a neighbor (m).
        separation_radius: Distance below which separation force activates (m).
        max_force: Maximum resultant force magnitude.
        max_speed: Maximum forward speed (m/s).
    """

    cohesion_weight: float = 1.0
    alignment_weight: float = 0.8
    separation_weight: float = 1.5
    perception_radius: float = 15.0
    separation_radius: float = 3.0
    max_force: float = 2.0
    max_speed: float = 1.5


@dataclass
class NeighborState:
    """State snapshot of a single neighbor USV."""

    x: float
    y: float
    heading: float
    speed: float


class BoidsController:
    """Decentralized Boids flocking controller.

    Each USV runs one instance. It reads the states of all neighbors
    within perception_radius and outputs a desired heading and speed
    that balances the three Reynolds forces.
    """

    def __init__(self, params: Optional[BoidsParams] = None) -> None:
        """Initialize Boids controller.

        Args:
            params: Flocking parameters. Uses defaults if None.
        """
        self.params: BoidsParams = params or BoidsParams()

    def compute_force(
        self,
        own_x: float,
        own_y: float,
        own_heading: float,
        own_speed: float,
        neighbors: List[NeighborState],
    ) -> Tuple[float, float]:
        """Compute resultant virtual force from all three Boids rules.

        Args:
            own_x: Own X position (m, world frame).
            own_y: Own Y position (m, world frame).
            own_heading: Own heading angle (radians).
            own_speed: Own forward speed (m/s).
            neighbors: States of all detectable neighbor USVs.

        Returns:
            (desired_heading_rad, desired_speed_mps).
        """
        if not neighbors:
            # No neighbors — maintain current state
            return own_heading, own_speed

        p = self.params

        # Filter to perception radius
        visible: List[NeighborState] = []
        for n in neighbors:
            dist = np.hypot(n.x - own_x, n.y - own_y)
            if dist < p.perception_radius:
                visible.append(n)

        if not visible:
            return own_heading, own_speed

        # Compute three force vectors
        cohesion = self._cohesion(own_x, own_y, visible)
        alignment = self._alignment(own_heading, own_speed, visible)
        separation = self._separation(own_x, own_y, visible)

        # Weighted sum
        fx = (
            p.cohesion_weight * cohesion[0]
            + p.alignment_weight * alignment[0]
            + p.separation_weight * separation[0]
        )
        fy = (
            p.cohesion_weight * cohesion[1]
            + p.alignment_weight * alignment[1]
            + p.separation_weight * separation[1]
        )

        # Clamp resultant force magnitude
        force_mag = np.sqrt(fx**2 + fy**2)
        if force_mag > p.max_force:
            fx *= p.max_force / force_mag
            fy *= p.max_force / force_mag

        # Convert force vector → heading and speed
        desired_heading = np.arctan2(fy, fx)
        # Speed scales with force magnitude (weaker force = slower)
        desired_speed = min(force_mag / p.max_force * p.max_speed, p.max_speed)
        desired_speed = max(0.2, desired_speed)  # minimum creep speed

        return desired_heading, desired_speed

    def _cohesion(
        self,
        own_x: float,
        own_y: float,
        neighbors: List[NeighborState],
    ) -> Tuple[float, float]:
        """Compute cohesion force: steer toward the average neighbor position.

        Args:
            own_x, own_y: Own position.
            neighbors: Visible neighbors.

        Returns:
            (fx, fy) cohesion force vector.
        """
        if not neighbors:
            return (0.0, 0.0)

        avg_x = sum(n.x for n in neighbors) / len(neighbors)
        avg_y = sum(n.y for n in neighbors) / len(neighbors)

        # Direction from self to swarm center
        dx = avg_x - own_x
        dy = avg_y - own_y
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 0.01:
            return (0.0, 0.0)

        # Normalize to unit vector
        return (dx / dist, dy / dist)

    def _alignment(
        self,
        own_heading: float,
        own_speed: float,
        neighbors: List[NeighborState],
    ) -> Tuple[float, float]:
        """Compute alignment force: match average neighbor velocity vector.

        Args:
            own_heading: Own heading.
            own_speed: Own speed.
            neighbors: Visible neighbors.

        Returns:
            (fx, fy) alignment force vector.
        """
        if not neighbors:
            vx = own_speed * np.cos(own_heading)
            vy = own_speed * np.sin(own_heading)
            return (vx, vy)

        # Average velocity vector of neighbors
        avg_vx = sum(n.speed * np.cos(n.heading) for n in neighbors) / len(neighbors)
        avg_vy = sum(n.speed * np.sin(n.heading) for n in neighbors) / len(neighbors)

        mag = np.sqrt(avg_vx**2 + avg_vy**2)
        if mag < 0.01:
            return (0.0, 0.0)

        return (avg_vx / mag, avg_vy / mag)

    def _separation(
        self,
        own_x: float,
        own_y: float,
        neighbors: List[NeighborState],
    ) -> Tuple[float, float]:
        """Compute separation force: steer away from nearby neighbors.

        The force is inversely proportional to distance — closer
        neighbors generate stronger repulsion.

        Args:
            own_x, own_y: Own position.
            neighbors: Visible neighbors.

        Returns:
            (fx, fy) separation force vector.
        """
        fx, fy = 0.0, 0.0

        for n in neighbors:
            dx = own_x - n.x
            dy = own_y - n.y
            dist = np.sqrt(dx**2 + dy**2)

            if dist < 0.01:
                # Overlapping — apply a strong random direction
                angle = np.random.uniform(0, 2 * np.pi)
                fx += np.cos(angle)
                fy += np.sin(angle)
            elif dist < self.params.separation_radius:
                # Inverse distance weighting: stronger when closer
                weight = (self.params.separation_radius - dist) / self.params.separation_radius
                fx += (dx / dist) * weight
                fy += (dy / dist) * weight

        mag = np.sqrt(fx**2 + fy**2)
        if mag < 0.01:
            return (0.0, 0.0)

        return (fx / mag, fy / mag)
