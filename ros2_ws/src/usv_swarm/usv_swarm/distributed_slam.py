"""Distributed collaborative SLAM (C-SLAM) for multi-USV swarm.

Each USV runs a local SLAM instance (Cartographer). This module handles
the exchange and fusion of local map data between swarm members so that
every USV benefits from the observations of all others.

Key mechanism:
  1. Each USV publishes its local submap as an occupancy grid.
  2. Neighboring USVs receive and fuse foreign submaps into their
     own map, after transforming to the shared global frame.
  3. Loop closures detected by any USV are broadcast to all peers,
     enabling distributed pose-graph optimization.

Fusion uses Dempster-Shafer or simple log-odds update on the
occupancy grid cells.

Reference: "C-SLAM" survey, 2021-2022 multi-robot SLAM literature.
"""

from typing import Dict, Optional, Tuple

import numpy as np


class DistributedSLAM:
    """Distributed SLAM map fusion node.

    Manages the integration of peer submaps into the local occupancy grid.
    Uses log-odds representation for probabilistically sound fusion.
    """

    # Occupancy grid cell values and their log-odds equivalents
    # Free = 0, Unknown = 0.5, Occupied = 1.0 in probability
    PROB_FREE: float = 0.3
    PROB_UNKNOWN: float = 0.5
    PROB_OCCUPIED: float = 0.7

    def __init__(
        self,
        grid_width: int = 400,
        grid_height: int = 400,
        resolution: float = 0.05,
    ) -> None:
        """Initialize distributed SLAM fusion.

        Args:
            grid_width: Local grid width in cells.
            grid_height: Local grid height in cells.
            resolution: Grid resolution (m/cell).
        """
        self.resolution: float = resolution
        self.grid_width: int = grid_width
        self.grid_height: int = grid_height

        # Origin of local grid in world frame
        self._origin_x: float = -grid_width * resolution / 2.0
        self._origin_y: float = -grid_height * resolution / 2.0

        # Log-odds grid (initial: unknown = 0)
        self._log_odds: np.ndarray = np.zeros((grid_height, grid_width), dtype=np.float32)

        # Track which peers contributed to each cell (for debugging)
        self._source_count: np.ndarray = np.zeros(
            (grid_height, grid_width), dtype=np.int32
        )

    def probability_to_log_odds(self, prob: float) -> float:
        """Convert probability [0, 1] to log-odds.

        Args:
            prob: Occupancy probability.

        Returns:
            Log-odds value.
        """
        prob = max(0.001, min(0.999, prob))
        return np.log(prob / (1.0 - prob))

    def log_odds_to_probability(self, lo: float) -> float:
        """Convert log-odds to probability [0, 1].

        Args:
            lo: Log-odds value.

        Returns:
            Occupancy probability.
        """
        return 1.0 / (1.0 + np.exp(-lo))

    def fuse_submap(
        self,
        submap_data: np.ndarray,
        submap_origin_x: float,
        submap_origin_y: float,
        submap_resolution: float,
        peer_id: str,
    ) -> None:
        """Fuse a peer's submap into the local grid.

        Uses Bayesian log-odds fusion:
          L_fused = L_local + L_peer - L_prior

        This correctly handles conflicting observations from
        multiple USVs (e.g., one sees free, another sees occupied).

        Args:
            submap_data: 2D occupancy grid from peer (values 0-100 or 0-255).
            submap_origin_x: Origin X of peer's submap in world frame.
            submap_origin_y: Origin Y of peer's submap in world frame.
            submap_resolution: Peer's grid resolution (m/cell).
            peer_id: Identifier of the contributing USV.
        """
        peer_h, peer_w = submap_data.shape

        for py in range(peer_h):
            for px in range(peer_w):
                peer_val = submap_data[py, px]

                # Normalize to probability
                if peer_val < 0:
                    continue  # unknown in peer map
                prob_peer = peer_val / 255.0 if peer_val <= 255 else peer_val / 100.0

                if abs(prob_peer - self.PROB_UNKNOWN) < 0.05:
                    continue  # skip unknown cells

                # World coordinates of this cell
                wx = submap_origin_x + px * submap_resolution
                wy = submap_origin_y + py * submap_resolution

                # Local grid indices
                lx = int((wx - self._origin_x) / self.resolution)
                ly = int((wy - self._origin_y) / self.resolution)

                if not (0 <= lx < self.grid_width and 0 <= ly < self.grid_height):
                    continue  # outside local grid bounds

                # Log-odds fusion
                lo_peer = self.probability_to_log_odds(prob_peer)
                lo_prior = self.probability_to_log_odds(self.PROB_UNKNOWN)

                self._log_odds[ly, lx] += lo_peer - lo_prior
                self._source_count[ly, lx] += 1

    def get_fused_grid(self) -> np.ndarray:
        """Get the fused occupancy grid as 0-100 integer values.

        Returns:
            2D numpy array with nav_msgs/OccupancyGrid compatible values
            (0 = free, 100 = occupied, -1 = unknown).
        """
        prob = self.log_odds_to_probability(self._log_odds)
        grid = np.full_like(prob, -1, dtype=np.int8)

        # Only assign values where we have observations
        mask = self._source_count > 0
        grid[mask] = (prob[mask] * 100).astype(np.int8)

        return grid

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices.

        Args:
            wx: World X (m).
            wy: World Y (m).

        Returns:
            (col, row) grid indices.
        """
        col = int((wx - self._origin_x) / self.resolution)
        row = int((wy - self._origin_y) / self.resolution)
        return col, row
