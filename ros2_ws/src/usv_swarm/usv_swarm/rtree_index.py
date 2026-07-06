#!/usr/bin/env python3
"""R-tree spatial index for efficient neighbor queries in USV swarms.

Replaces O(N²) brute-force neighbor iteration in ORCA/DWA with O(log N)
spatial queries. Uses scipy.spatial.cKDTree (C implementation, no extra
dependencies on embedded boards) to index agent and obstacle positions
in the 2D (x, y) plane.

For 3 USVs, the difference is negligible. For 10+, the speedup is
significant: 100 agents × 8 speed samples × 16 angle samples = 12,800
ORCA constraint computations per cycle. With R-tree, only neighbors
within 10m are considered (typically 3-5 instead of all 99).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SpatialQueryResult:
    """Result of a spatial neighbor query.

    Attributes:
        indices: Indices of neighbors in the original point array.
        positions: (x, y) positions of matched neighbors.
        distances: Euclidean distances to each neighbor.
    """
    indices: np.ndarray
    positions: np.ndarray
    distances: np.ndarray


class USVSpatialIndex:
    """cKDTree-based spatial index for USV positions and obstacles.

    Usage::

        index = USVSpatialIndex()
        index.update_agents([(0, 0), (5, 3), (2, 8)])
        neighbors = index.query_neighbors(origin=(0, 0), radius=10.0)
        # neighbors.indices -> [0, 1, 2] within 10m

    Rebuild cost: O(N log N) per update_agents call.
    Query cost: O(log N) per query_neighbors call (vs O(N) brute force).
    """

    def __init__(self, leaf_size: int = 16):
        """
        Args:
            leaf_size: cKDTree leaf size. 16 is the sweet spot for 2D
                       point queries balancing tree depth vs leaf scanning.
        """
        self._tree: Optional[cKDTree] = None
        self._points: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self.leaf_size: int = leaf_size
        self._dirty: bool = True

    def update_agents(self, positions: List[Tuple[float, float]]) -> None:
        """Rebuild index from current agent positions.

        Called once per control cycle after receiving updated GPS positions
        from all agents via the ROS 2 DDS mesh network.

        Args:
            positions: List of (x, y) agent positions in world frame.
        """
        if not positions:
            self._points = np.empty((0, 2), dtype=np.float64)
            self._tree = None
            self._dirty = False
            return

        self._points = np.array(positions, dtype=np.float64)
        self._tree = cKDTree(self._points, leafsize=self.leaf_size)
        self._dirty = False

    def update_obstacles(self, positions: List[Tuple[float, float]]) -> None:
        """Build a separate index for static/dynamic obstacles.

        Obstacle positions come from YOLO detection → IPM → world frame.
        Kept separate from agent positions for collision role clarity.
        """
        if not positions:
            self._obstacle_points = np.empty((0, 2), dtype=np.float64)
            self._obstacle_tree = None
            return

        self._obstacle_points = np.array(positions, dtype=np.float64)
        self._obstacle_tree = cKDTree(
            self._obstacle_points, leafsize=self.leaf_size
        )

    def query_neighbors(
        self,
        origin: Tuple[float, float],
        radius: float,
        exclude_self: bool = True,
        self_position: Optional[Tuple[float, float]] = None,
    ) -> SpatialQueryResult:
        """Query all indexed points within radius of origin.

        Args:
            origin: Query center (x, y) in world frame.
            radius: Search radius in meters.
            exclude_self: If True, exclude the querying agent itself
                         (matched by exact position equality).
            self_position: The querying agent's exact position for
                           self-exclusion. Uses origin if None.

        Returns:
            SpatialQueryResult with neighbor indices, positions, and distances.
        """
        if self._dirty or self._tree is None or len(self._points) == 0:
            return SpatialQueryResult(
                indices=np.array([], dtype=int),
                positions=np.empty((0, 2)),
                distances=np.array([], dtype=float),
            )

        raw_indices = self._tree.query_ball_point(origin, radius)

        if exclude_self:
            ref = np.array(self_position or origin)
            indices = []
            positions_list = []
            distances_list = []
            for idx in raw_indices:
                pt = self._points[idx]
                dist = float(np.linalg.norm(pt - ref))
                if dist < 0.001:
                    continue  # Self
                indices.append(idx)
                positions_list.append(pt)
                distances_list.append(dist)

            if not indices:
                return SpatialQueryResult(
                    indices=np.array([], dtype=int),
                    positions=np.empty((0, 2)),
                    distances=np.array([], dtype=float),
                )
            return SpatialQueryResult(
                indices=np.array(indices, dtype=int),
                positions=np.array(positions_list),
                distances=np.array(distances_list, dtype=float),
            )

        positions = self._points[raw_indices]
        distances = np.linalg.norm(positions - np.array(origin), axis=1)
        return SpatialQueryResult(
            indices=np.array(raw_indices, dtype=int),
            positions=positions,
            distances=distances,
        )

    def nearest_n(
        self,
        origin: Tuple[float, float],
        n: int = 5,
        exclude_self: bool = True,
    ) -> SpatialQueryResult:
        """Query the N nearest indexed points to origin.

        Args:
            origin: Query center (x, y).
            n: Maximum number of neighbors to return.
            exclude_self: Exclude the querying agent.

        Returns:
            SpatialQueryResult sorted by ascending distance.
        """
        if self._dirty or self._tree is None or len(self._points) == 0:
            return SpatialQueryResult(
                indices=np.array([], dtype=int),
                positions=np.empty((0, 2)),
                distances=np.array([], dtype=float),
            )

        k = min(n + (1 if exclude_self else 0), len(self._points))
        distances, indices = self._tree.query(origin, k=k)

        # Handle scalar return for k=1
        if k == 1:
            distances = np.array([distances])
            indices = np.array([indices])

        if exclude_self:
            mask = distances > 0.001
            indices = indices[mask][:n]
            distances = distances[mask][:n]

        positions = self._points[indices]
        return SpatialQueryResult(
            indices=indices,
            positions=positions,
            distances=distances,
        )

    @property
    def size(self) -> int:
        return len(self._points)
