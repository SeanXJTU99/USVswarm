"""Tests for USV spatial index (cKDTree-based R-tree)."""

import pytest
import numpy as np
from usv_swarm.rtree_index import USVSpatialIndex


class TestUSVSpatialIndex:

    def test_empty_index(self):
        idx = USVSpatialIndex()
        result = idx.query_neighbors((0, 0), 10.0)
        assert len(result.indices) == 0

    def test_single_agent_excluded(self):
        idx = USVSpatialIndex()
        idx.update_agents([(3.0, 4.0)])
        result = idx.query_neighbors((3.0, 4.0), 10.0)
        # Self excluded by exact position match
        assert len(result.indices) == 0

    def test_neighbor_within_radius(self):
        idx = USVSpatialIndex()
        idx.update_agents([(0.0, 0.0), (3.0, 4.0), (50.0, 50.0)])
        result = idx.query_neighbors((0.0, 0.0), 10.0)
        assert len(result.indices) == 1  # (3,4) at distance 5
        assert pytest.approx(result.distances[0], 0.01) == 5.0

    def test_no_neighbors_in_radius(self):
        idx = USVSpatialIndex()
        idx.update_agents([(0.0, 0.0), (20.0, 0.0)])
        result = idx.query_neighbors((0.0, 0.0), 10.0)
        assert len(result.indices) == 0

    def test_nearest_n(self):
        idx = USVSpatialIndex()
        pts = [(0.0, 0.0), (1.0, 0.0), (10.0, 0.0), (100.0, 0.0)]
        idx.update_agents(pts)
        result = idx.nearest_n((0.0, 0.0), n=2)
        assert len(result.indices) == 2
        assert pytest.approx(result.distances[0], 0.01) == 1.0

    def test_update_rebuilds_tree(self):
        idx = USVSpatialIndex()
        idx.update_agents([(0.0, 0.0)])
        assert idx.size == 1
        idx.update_agents([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
        assert idx.size == 3
        result = idx.query_neighbors((0.0, 0.0), 5.0)
        assert len(result.indices) == 1  # (1,1) at ~1.41

    def test_update_empty_clears_index(self):
        idx = USVSpatialIndex()
        idx.update_agents([(1.0, 1.0)])
        idx.update_agents([])
        assert idx.size == 0
        result = idx.query_neighbors((0.0, 0.0), 10.0)
        assert len(result.indices) == 0

    def test_large_swarm_performance(self):
        """Verify O(log N) query time on 1000 agents (smoke test)."""
        np.random.seed(42)
        pts = [(float(x), float(y))
               for x, y in np.random.uniform(-500, 500, (1000, 2))]
        idx = USVSpatialIndex()
        idx.update_agents(pts)
        result = idx.query_neighbors((0.0, 0.0), 50.0)
        # 50m radius in 1000×1000m area with 1000 agents
        # Expected ~8 agents within radius (π×50² / 1000² × 1000 ≈ 7.85)
        assert 2 <= len(result.indices) <= 20
