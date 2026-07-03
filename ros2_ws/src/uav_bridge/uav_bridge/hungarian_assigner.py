"""Hungarian algorithm for optimal task assignment in UAV-USV swarm.

The UAV (airborne compute center) runs global task assignment:
  - N tasks (e.g., debris locations, sampling points)
  - M USVs (workers available for assignment)
  - Cost = travel distance from each USV to each task

The Hungarian algorithm (Kuhn-Munkres) solves this assignment problem
in O(n³) time, producing the globally optimal one-to-one mapping
that minimizes total fleet travel distance.

Also known as the Munkres assignment algorithm.
"""

from typing import List, Optional, Tuple

import numpy as np


class HungarianAssigner:
    """Optimal task-to-USV assignment using the Hungarian algorithm.

    Given an M×N cost matrix (M USVs, N tasks), finds the assignment
    that minimizes total cost. Handles rectangular matrices (M ≠ N)
    by padding with dummy rows/columns.
    """

    def __init__(self) -> None:
        """Initialize the Hungarian assigner."""
        pass

    def assign(
        self,
        cost_matrix: np.ndarray,
        max_cost: Optional[float] = None,
    ) -> List[Tuple[int, int, float]]:
        """Solve the optimal assignment.

        Args:
            cost_matrix: M×N array where cost_matrix[i, j] = cost for
                         USV i to perform task j.
            max_cost: If set, assignments with cost > max_cost are
                      filtered out (task left unassigned).

        Returns:
            List of (usv_idx, task_idx, cost) for each assignment.
            Unassigned USVs and tasks are omitted.
        """
        m, n = cost_matrix.shape
        size = max(m, n)

        # Pad to square matrix
        padded = np.full((size, size), 1e9, dtype=np.float64)
        padded[:m, :n] = cost_matrix.astype(np.float64)

        # Hungarian algorithm implementation
        result = self._hungarian(padded)

        # Filter valid assignments (ignore padded cells)
        assignments: List[Tuple[int, int, float]] = []
        for i, j in enumerate(result):
            if i < m and j < n:
                cost = cost_matrix[i, j]
                if max_cost is None or cost <= max_cost:
                    assignments.append((i, j, cost))

        return assignments

    def _hungarian(self, cost: np.ndarray) -> List[int]:
        """Core Hungarian (Munkres) algorithm.

        Args:
            cost: Square cost matrix (N×N).

        Returns:
            List where result[row] = assigned column index.
        """
        n = cost.shape[0]
        u = np.zeros(n + 1)
        v = np.zeros(n + 1)
        p = np.zeros(n + 1, dtype=int)
        way = np.zeros(n + 1, dtype=int)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = np.full(n + 1, np.inf)
            used = np.zeros(n + 1, dtype=bool)

            while True:
                used[j0] = True
                i0 = p[j0]
                delta = np.inf
                j1 = 0

                for j in range(1, n + 1):
                    if not used[j]:
                        cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j

                for j in range(n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta

                j0 = j1
                if p[j0] == 0:
                    break

            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        # Build assignment: column assigned to each row
        assignment = [0] * n
        for j in range(1, n + 1):
            if p[j] != 0:
                assignment[p[j] - 1] = j - 1

        return assignment

    def compute_cost_matrix(
        self,
        usv_positions: List[Tuple[float, float]],
        task_positions: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Build a distance-based cost matrix from USV and task positions.

        Args:
            usv_positions: List of (x, y) for each USV.
            task_positions: List of (x, y) for each task.

        Returns:
            M×N cost matrix with Euclidean distances.
        """
        m = len(usv_positions)
        n = len(task_positions)
        cost = np.zeros((m, n))

        for i, (ux, uy) in enumerate(usv_positions):
            for j, (tx, ty) in enumerate(task_positions):
                cost[i, j] = np.sqrt((tx - ux) ** 2 + (ty - uy) ** 2)

        return cost
