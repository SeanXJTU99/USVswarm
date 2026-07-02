"""Consensus algorithm for decentralized swarm state agreement.

In a fully decentralized swarm, there is no leader. Decisions are made
by consensus — all agents iteratively exchange state with neighbors and
converge to a common value (e.g., average heading, average speed, or
formation centroid).

This module implements a discrete-time average consensus protocol based
on graph Laplacian dynamics:

  x_i(k+1) = x_i(k) + ε * Σ (x_j(k) - x_i(k))   for all neighbors j

where ε is the consensus step size (coupling gain). The system converges
to the average of initial states if the communication graph is connected.

Applications in USV swarm:
  - Agree on common heading for coordinated sweeping
  - Agree on common speed for formation flight
  - Distributed estimation of swarm centroid
"""

from typing import List, Optional, Tuple

import numpy as np


class ConsensusProtocol:
    """Distributed average consensus protocol.

    Each USV runs one instance. On each iteration, it receives the
    states of its communication neighbors and updates its local
    estimate toward the swarm average.
    """

    def __init__(
        self,
        step_size: float = 0.3,
        convergence_threshold: float = 0.01,
    ) -> None:
        """Initialize consensus protocol.

        Args:
            step_size: Coupling gain ε (0 < ε < 1/max_degree).
                       Higher = faster convergence but risk of instability.
            convergence_threshold: Stop iterating when |change| < this.
        """
        self.step_size: float = step_size
        self.convergence_threshold: float = convergence_threshold

        # Local state estimates
        self._heading_estimate: float = 0.0
        self._speed_estimate: float = 0.0
        self._centroid_x: float = 0.0
        self._centroid_y: float = 0.0

        self._converged: bool = False
        self._prev_heading: float = 0.0

    def update_heading(
        self,
        own_heading: float,
        neighbor_headings: List[float],
    ) -> float:
        """Perform one consensus iteration on heading.

        Args:
            own_heading: Current own heading (radians).
            neighbor_headings: Headings of all communication neighbors.

        Returns:
            Updated consensus heading estimate.
        """
        if not neighbor_headings:
            return own_heading

        # Handle circular topology of angles
        # Convert to unit vectors, average, convert back
        sum_sin = np.sin(own_heading)
        sum_cos = np.cos(own_heading)

        for h in neighbor_headings:
            sum_sin += np.sin(h)
            sum_cos += np.cos(h)

        n = 1 + len(neighbor_headings)
        avg_heading = np.arctan2(sum_sin / n, sum_cos / n)

        # Consensus update
        self._heading_estimate = own_heading + self.step_size * (
            avg_heading - own_heading
        )

        # Check convergence
        change = abs(self._heading_estimate - self._prev_heading)
        self._converged = change < self.convergence_threshold
        self._prev_heading = self._heading_estimate

        return self._heading_estimate

    def update_speed(
        self,
        own_speed: float,
        neighbor_speeds: List[float],
    ) -> float:
        """Perform one consensus iteration on speed.

        Args:
            own_speed: Current own speed (m/s).
            neighbor_speeds: Speeds of all communication neighbors.

        Returns:
            Updated consensus speed estimate.
        """
        if not neighbor_speeds:
            return own_speed

        avg_speed = (own_speed + sum(neighbor_speeds)) / (1 + len(neighbor_speeds))
        self._speed_estimate = own_speed + self.step_size * (avg_speed - own_speed)
        return self._speed_estimate

    def update_centroid(
        self,
        own_x: float,
        own_y: float,
        neighbor_positions: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Perform one consensus iteration on swarm centroid estimate.

        Args:
            own_x, own_y: Own position.
            neighbor_positions: Positions of all communication neighbors.

        Returns:
            (centroid_x, centroid_y) updated estimate.
        """
        if not neighbor_positions:
            return own_x, own_y

        sum_x = own_x + sum(p[0] for p in neighbor_positions)
        sum_y = own_y + sum(p[1] for p in neighbor_positions)
        n = 1 + len(neighbor_positions)

        avg_x = sum_x / n
        avg_y = sum_y / n

        self._centroid_x = own_x + self.step_size * (avg_x - own_x)
        self._centroid_y = own_y + self.step_size * (avg_y - own_y)

        return self._centroid_x, self._centroid_y

    @property
    def converged(self) -> bool:
        """Whether the consensus has converged."""
        return self._converged

    def reset(self) -> None:
        """Reset consensus state (use when swarm topology changes)."""
        self._converged = False
        self._prev_heading = 0.0
