"""Tests for :mod:`luxar.utils.spatial_hash`.

Coverage:

- :class:`SpatialHashGrid` (online, CPU) — regression suite carried
  over from the original location at ``gsplats/seeds/tests/test_utils.py``;
  exercised here against the new module home.
- :class:`BatchedSpatialHashGrid` — radius / k-NN parity vs. brute force
  and ``scipy.spatial.cKDTree``; CPU↔GPU parity (skipped if no GPU);
  conservative auto-fallback to NumPy on simulated OOM; non-OOM
  ``RuntimeError``s propagate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from luxar.utils.spatial_hash import (
    BatchedSpatialHashGrid,
    SpatialHashGrid,
    _is_oom_error,
)

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _gpu_available() -> bool:
    return torch.cuda.is_available() or (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )


def _make_random_points(N: int, D: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(N, D).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Online grid (regression — same fixtures as the old seeds tests)
# ─────────────────────────────────────────────────────────────────────


class TestOnlineSpatialHashGrid:
    """Regression suite for the online :class:`SpatialHashGrid`."""

    def test_empty_grid(self):
        grid = SpatialHashGrid(cell_size=2.0, ndim=3)
        assert len(grid) == 0
        assert not grid.has_neighbor_within(np.array([0.0, 0.0, 0.0]), 1.0)

    def test_insert_and_len(self):
        grid = SpatialHashGrid(cell_size=2.0, ndim=2)
        idx0 = grid.insert(np.array([1.0, 1.0]))
        idx1 = grid.insert(np.array([5.0, 5.0]))
        assert (idx0, idx1) == (0, 1)
        assert len(grid) == 2

    def test_neighbor_detection_basic(self):
        grid = SpatialHashGrid(cell_size=2.0, ndim=2)
        grid.insert(np.array([0.0, 0.0]))
        # within distance
        assert grid.has_neighbor_within(np.array([1.0, 0.0]), 1.5)
        # beyond distance
        assert not grid.has_neighbor_within(np.array([3.0, 0.0]), 1.5)

    def test_negative_coords_handled(self):
        """Cells must be computed via ``math.floor``, not ``int()``."""
        grid = SpatialHashGrid(cell_size=2.0, ndim=2)
        grid.insert(np.array([-0.5, -0.5]))
        # +0.5 and -0.5 must land in different cells; without floor they'd collide
        assert grid.has_neighbor_within(np.array([-0.6, -0.6]), 0.5)
        assert not grid.has_neighbor_within(np.array([0.6, 0.6]), 0.5)

    def test_consistency_with_bruteforce(self):
        rng = np.random.RandomState(123)
        pts = rng.randn(200, 3).astype(np.float32) * 0.4
        grid = SpatialHashGrid(cell_size=0.3, ndim=3)
        for p in pts:
            grid.insert(p)
        queries = rng.randn(30, 3).astype(np.float32) * 0.4
        for q in queries:
            ref = bool(np.any(np.linalg.norm(pts - q, axis=1) < 0.2))
            ours = grid.has_neighbor_within(q, 0.2)
            assert ref == ours

    def test_points_property_returns_copy(self):
        grid = SpatialHashGrid(cell_size=2.0, ndim=2)
        grid.insert(np.array([1.0, 2.0]))
        p1 = grid.points
        p1[0, 0] = 999.0
        # Mutation of the returned array must not affect the grid's state.
        assert grid.points[0, 0] != 999.0

    # [Python-R2/D-C1] Worst-case-density correctness. The class docstring
    # promises has_neighbor_within() runs in O(1) amortized against the
    # 3^ndim neighbor cells; with uniform random inputs the buckets stay
    # small and the test never exercises the dense-bucket path. Pack many
    # points into a single cell, then verify both:
    #   - true-positive: a known point in the cluster is found by an
    #     identical probe with distance ≥ 0, regardless of cluster size
    #   - true-negative: a query well outside the 3-cell neighbour radius
    #     does NOT report a neighbor even with the dense bucket present
    @pytest.mark.parametrize("n_in_cell", [100, 1000, 10_000])
    def test_dense_cluster_correctness(self, n_in_cell):
        cell_size = 1.0
        grid = SpatialHashGrid(cell_size=cell_size, ndim=3)
        rng = np.random.RandomState(7)
        # All points within a sub-cell volume — they live in the same
        # bucket and the neighbor-scan must walk the whole list.
        cluster = rng.uniform(0.05, 0.95, size=(n_in_cell, 3)).astype(np.float32)
        for p in cluster:
            grid.insert(p)

        # True-positive: probe identical to a known cluster point. Any
        # mutation that broke the bucket lookup (wrong cell key, dropped
        # neighbor offset, missed self-cell scan) would fail to find it.
        target = cluster[n_in_cell // 2]
        assert grid.has_neighbor_within(target, 0.01) is True

        # True-negative: probe more than 3 cells away — the 3^ndim
        # neighbor expansion can't reach the cluster, regardless of how
        # dense it is.
        probe_far = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        assert grid.has_neighbor_within(probe_far, 0.5) is False


# ─────────────────────────────────────────────────────────────────────
# Batched grid: NumPy backend correctness
# ─────────────────────────────────────────────────────────────────────


class TestBatchedNumpy:
    @pytest.mark.parametrize("ndim", [2, 3])
    def test_radius_matches_brute_force(self, ndim):
        pts = _make_random_points(N=200, D=ndim)
        queries = _make_random_points(N=20, D=ndim, seed=7)
        grid = BatchedSpatialHashGrid.from_points(pts, cell_size=0.4, device="cpu")
        radius = 0.3
        hits = grid.query_radius(queries, radius=radius)

        for qi, q in enumerate(queries):
            d = np.linalg.norm(pts - q, axis=1)
            ref = set(np.where(d < radius)[0].tolist())
            ours = set(hits[qi].tolist())
            assert ref == ours, f"mismatch on query {qi}: ref={ref} vs ours={ours}"

    @pytest.mark.parametrize("k", [1, 5, 12])
    def test_knn_matches_cKDTree(self, k):
        pts = _make_random_points(N=500, D=3)
        queries = _make_random_points(N=50, D=3, seed=8)
        tree = cKDTree(pts)
        ref_d, _ = tree.query(queries, k=k)
        if k == 1:
            ref_d = ref_d[:, None]
        grid = BatchedSpatialHashGrid.from_points(pts, cell_size=0.5, device="cpu")
        my_d, my_i = grid.query_knn(queries, k=k)
        # Distances must match exactly (up to fp tolerance).
        np.testing.assert_allclose(my_d, ref_d.astype(np.float32), atol=1e-5)
        # Indices may differ on ties but the *distances* indexed are identical.
        diff = pts[my_i] - queries[:, None, :]
        recomputed = np.linalg.norm(diff, axis=-1)
        np.testing.assert_allclose(recomputed, my_d, atol=1e-5)

    def test_knn_sorted_ascending(self):
        pts = _make_random_points(N=100, D=3)
        queries = _make_random_points(N=10, D=3, seed=9)
        grid = BatchedSpatialHashGrid.from_points(pts, cell_size=0.5, device="cpu")
        d, _ = grid.query_knn(queries, k=8)
        for row in d:
            assert np.all(np.diff(row) >= 0), "row must be sorted ascending"

    def test_knn_with_fewer_than_k_points(self):
        """When N < k, trailing columns must be -1 / +inf."""
        pts = _make_random_points(N=3, D=2)
        queries = _make_random_points(N=2, D=2, seed=10)
        grid = BatchedSpatialHashGrid.from_points(pts, cell_size=0.5, device="cpu")
        d, idx = grid.query_knn(queries, k=5)
        # First 3 columns are valid; last 2 must be -1 / +inf.
        assert np.all(idx[:, :3] >= 0)
        assert np.all(idx[:, 3:] == -1)
        assert np.all(np.isinf(d[:, 3:]))

    def test_radius_above_cell_size_raises(self):
        pts = _make_random_points(N=10, D=2)
        grid = BatchedSpatialHashGrid.from_points(pts, cell_size=0.5, device="cpu")
        with pytest.raises(ValueError, match="exceeds cell_size"):
            grid.query_radius(pts[:1], radius=0.6)

    def test_negative_radius_raises(self):
        pts = _make_random_points(N=10, D=2)
        grid = BatchedSpatialHashGrid.from_points(pts, cell_size=0.5, device="cpu")
        with pytest.raises(ValueError, match="radius must be"):
            grid.query_radius(pts[:1], radius=-1.0)

    def test_invalid_k_raises(self):
        pts = _make_random_points(N=10, D=2)
        grid = BatchedSpatialHashGrid.from_points(pts, cell_size=0.5, device="cpu")
        with pytest.raises(ValueError, match="k must be >= 1"):
            grid.query_knn(pts[:1], k=0)

    def test_empty_points(self):
        pts = np.empty((0, 3), dtype=np.float32)
        queries = _make_random_points(N=5, D=3)
        grid = BatchedSpatialHashGrid.from_points(pts, cell_size=0.5, device="cpu")
        d, idx = grid.query_knn(queries, k=4)
        assert d.shape == (5, 4) and np.all(np.isinf(d))
        assert np.all(idx == -1)
        hits = grid.query_radius(queries, radius=0.3)
        assert all(h.size == 0 for h in hits)

    def test_empty_queries(self):
        pts = _make_random_points(N=10, D=3)
        queries = np.empty((0, 3), dtype=np.float32)
        grid = BatchedSpatialHashGrid.from_points(pts, cell_size=0.5, device="cpu")
        d, idx = grid.query_knn(queries, k=4)
        assert d.shape == (0, 4)
        assert idx.shape == (0, 4)

    def test_zero_cell_size_raises(self):
        pts = _make_random_points(N=10, D=2)
        with pytest.raises(ValueError, match="cell_size must be"):
            BatchedSpatialHashGrid.from_points(pts, cell_size=0.0, device="cpu")

    def test_wrong_query_shape_raises(self):
        pts = _make_random_points(N=10, D=3)
        grid = BatchedSpatialHashGrid.from_points(pts, cell_size=0.5, device="cpu")
        with pytest.raises(ValueError, match="query must have shape"):
            grid.query_knn(np.zeros((5, 2), dtype=np.float32), k=2)


# ─────────────────────────────────────────────────────────────────────
# Batched grid: GPU parity
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _gpu_available(), reason="No GPU available")
class TestBatchedGPU:
    def test_backend_matches_device(self):
        pts = _make_random_points(N=100, D=3)
        grid = BatchedSpatialHashGrid.from_points(pts, cell_size=0.5, device="auto")
        assert grid.backend == "torch"
        assert grid.device.type in {"cuda", "mps"}

    def test_knn_cpu_gpu_parity(self):
        pts = _make_random_points(N=500, D=3)
        queries = _make_random_points(N=50, D=3, seed=100)
        cpu = BatchedSpatialHashGrid.from_points(pts, cell_size=0.5, device="cpu")
        gpu = BatchedSpatialHashGrid.from_points(pts, cell_size=0.5, device="auto")
        d_cpu, _ = cpu.query_knn(queries, k=8)
        d_gpu, _ = gpu.query_knn(queries, k=8)
        np.testing.assert_allclose(d_cpu, d_gpu, atol=1e-5)

    def test_knn_gpu_matches_cKDTree(self):
        pts = _make_random_points(N=500, D=3)
        queries = _make_random_points(N=50, D=3, seed=101)
        tree = cKDTree(pts)
        ref_d, _ = tree.query(queries, k=8)
        gpu = BatchedSpatialHashGrid.from_points(pts, cell_size=0.5, device="auto")
        d_gpu, _ = gpu.query_knn(queries, k=8)
        np.testing.assert_allclose(d_gpu, ref_d.astype(np.float32), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
# Fallback policy
# ─────────────────────────────────────────────────────────────────────


class TestFallbackPolicy:
    def test_oom_helper_recognises_runtime_error(self):
        assert _is_oom_error(RuntimeError("CUDA out of memory: ..."))
        assert not _is_oom_error(RuntimeError("shape mismatch"))
        assert not _is_oom_error(ValueError("not a runtime error"))

    def test_oom_falls_back_to_numpy(self, monkeypatch, capsys):
        """Simulating an OOM during torch build must fall back, not raise."""
        if not _gpu_available():
            pytest.skip("Need GPU for this test")
        pts = _make_random_points(N=50, D=3)

        def boom(*args, **kwargs):
            raise RuntimeError("CUDA out of memory: simulated")

        monkeypatch.setattr(
            BatchedSpatialHashGrid,
            "_build_torch_state",
            staticmethod(boom),
        )

        grid = BatchedSpatialHashGrid.from_points(
            pts, cell_size=0.5, device="auto", fallback_to_cpu=True
        )
        assert grid.backend == "numpy"
        # An Arbol log line should have been written.
        captured = capsys.readouterr().out
        assert "GPU build failed" in captured

    def test_non_oom_runtime_error_propagates(self, monkeypatch):
        """Generic RuntimeError must NOT be silently caught."""
        if not _gpu_available():
            pytest.skip("Need GPU for this test")
        pts = _make_random_points(N=50, D=3)

        def boom(*args, **kwargs):
            raise RuntimeError("shape mismatch (this is a real bug)")

        monkeypatch.setattr(
            BatchedSpatialHashGrid,
            "_build_torch_state",
            staticmethod(boom),
        )
        with pytest.raises(RuntimeError, match="shape mismatch"):
            BatchedSpatialHashGrid.from_points(
                pts, cell_size=0.5, device="auto", fallback_to_cpu=True
            )

    def test_fallback_disabled_propagates_oom(self, monkeypatch):
        """``fallback_to_cpu=False`` must surface the OOM."""
        if not _gpu_available():
            pytest.skip("Need GPU for this test")
        pts = _make_random_points(N=50, D=3)

        def boom(*args, **kwargs):
            raise RuntimeError("CUDA out of memory: simulated")

        monkeypatch.setattr(
            BatchedSpatialHashGrid,
            "_build_torch_state",
            staticmethod(boom),
        )
        with pytest.raises(RuntimeError, match="out of memory"):
            BatchedSpatialHashGrid.from_points(
                pts, cell_size=0.5, device="auto", fallback_to_cpu=False
            )
