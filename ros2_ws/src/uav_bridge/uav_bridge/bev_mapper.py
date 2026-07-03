"""BEV (Bird's-Eye View) mapper — UAV overhead orthomosaic builder.

Constructs a global top-down map of the operation area by stitching
sequential UAV camera frames into a geo-referenced orthomosaic.

Unlike the USV's IPM (which projects from a shallow oblique angle),
the UAV's near-nadir view produces geometrically accurate BEV images
with minimal distortion. This simplifies obstacle localization and
enables direct pixel-to-GPS mapping.

The mapper:
  1. Receives UAV camera frames + GPS/IMU telemetry
  2. Warps each frame to a common world grid using the UAV's pose
  3. Accumulates occupancy evidence in a running grid
  4. Publishes the global BEV map to all USVs
"""

from typing import Optional, Tuple

import numpy as np


class BEVMapper:
    """Accumulates UAV camera frames into a global bird's-eye view map.

    Maintains a running world-frame occupancy grid. Each incoming
    camera frame is projected to the ground plane and fused into
    the grid using log-odds (Bayesian) updates.
    """

    # Occupancy probability constants
    PROB_OCCUPIED: float = 0.7
    PROB_FREE: float = 0.3
    PROB_UNKNOWN: float = 0.5

    def __init__(
        self,
        world_width_m: float = 100.0,
        world_height_m: float = 100.0,
        resolution: float = 0.1,  # m/pixel (10 cm for UAV)
        origin_x: float = -50.0,
        origin_y: float = -50.0,
    ) -> None:
        """Initialize BEV mapper.

        Args:
            world_width_m: Map width in meters.
            world_height_m: Map height in meters.
            resolution: Grid resolution (m/cell).
            origin_x: World X of the grid origin (bottom-left).
            origin_y: World Y of the grid origin (bottom-left).
        """
        self.resolution: float = resolution
        self.origin_x: float = origin_x
        self.origin_y: float = origin_y

        self.grid_width: int = int(world_width_m / resolution)
        self.grid_height: int = int(world_height_m / resolution)

        # Log-odds occupancy grid
        self._log_odds: np.ndarray = np.zeros(
            (self.grid_height, self.grid_width), dtype=np.float32
        )
        # Hit count (for confidence estimation)
        self._hit_count: np.ndarray = np.zeros(
            (self.grid_height, self.grid_width), dtype=np.int32
        )

        # Free-space mask from segmentation (if available)
        self._free_space_mask: Optional[np.ndarray] = None

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell indices.

        Args:
            wx: World X (m).
            wy: World Y (m).

        Returns:
            (col, row) in grid coordinates.
        """
        col = int((wx - self.origin_x) / self.resolution)
        row = int((wy - self.origin_y) / self.resolution)
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid cell to world coordinates (cell center).

        Args:
            col: Grid column index.
            row: Grid row index.

        Returns:
            (wx, wy) world coordinates of cell center.
        """
        wx = self.origin_x + (col + 0.5) * self.resolution
        wy = self.origin_y + (row + 0.5) * self.resolution
        return wx, wy

    def mark_obstacle(self, wx: float, wy: float, radius_m: float = 0.3) -> None:
        """Mark a grid region as occupied (obstacle detected).

        Args:
            wx, wy: World coordinates of obstacle center.
            radius_m: Obstacle radius for inflation (m).
        """
        cx, cy = self.world_to_grid(wx, wy)
        radius_cells = int(radius_m / self.resolution) + 1

        r_min = max(0, cy - radius_cells)
        r_max = min(self.grid_height, cy + radius_cells + 1)
        c_min = max(0, cx - radius_cells)
        c_max = min(self.grid_width, cx + radius_cells + 1)

        # Log-odds update: add occupied evidence
        lo_occupied = np.log(self.PROB_OCCUPIED / (1.0 - self.PROB_OCCUPIED))
        lo_prior = np.log(self.PROB_UNKNOWN / (1.0 - self.PROB_UNKNOWN))

        region = self._log_odds[r_min:r_max, c_min:c_max]
        region += lo_occupied - lo_prior
        self._hit_count[r_min:r_max, c_min:c_max] += 1

    def mark_free_space(self, wx: float, wy: float) -> None:
        """Mark a single cell as free (observed water).

        Args:
            wx, wy: World coordinates.
        """
        col, row = self.world_to_grid(wx, wy)
        if 0 <= col < self.grid_width and 0 <= row < self.grid_height:
            lo_free = np.log(self.PROB_FREE / (1.0 - self.PROB_FREE))
            lo_prior = np.log(self.PROB_UNKNOWN / (1.0 - self.PROB_UNKNOWN))
            self._log_odds[row, col] += lo_free - lo_prior
            self._hit_count[row, col] += 1

    def fuse_detection_frame(
        self,
        obstacles: list[Tuple[float, float, float]],
        freespace_polygon: Optional[np.ndarray] = None,
    ) -> None:
        """Fuse a full UAV detection frame into the BEV map.

        Args:
            obstacles: List of (wx, wy, radius_m) for each detected obstacle.
            freespace_polygon: Optional polygon of visible free space
                               (for clearing out stale obstacles).
        """
        for wx, wy, radius in obstacles:
            self.mark_obstacle(wx, wy, radius)

    def get_occupancy_grid(self) -> np.ndarray:
        """Get the fused occupancy grid as 0-100 values.

        Returns:
            2D int8 array: 0=free, 100=occupied, -1=unknown.
        """
        prob = 1.0 / (1.0 + np.exp(-self._log_odds))
        grid = np.full_like(prob, -1, dtype=np.int8)

        mask = self._hit_count > 0
        grid[mask] = (prob[mask] * 100).astype(np.int8)

        return grid

    def get_probability_grid(self) -> np.ndarray:
        """Get the fused occupancy grid as probabilities [0, 1].

        Returns:
            2D float32 array of occupancy probabilities.
        """
        return 1.0 / (1.0 + np.exp(-self._log_odds))

    def clear(self) -> None:
        """Reset the BEV map."""
        self._log_odds.fill(0.0)
        self._hit_count.fill(0)
