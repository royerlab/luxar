"""Tests for GSplats spatial ordering — the third geometry of the
Points/Lines/GSplats ordering triad.

Mirrors ``test_ordering_points.py`` / ``test_ordering_lines.py``:
``sort_splats_spatial`` must return a valid permutation and group by barrier
axes first, and ``compute_chunk_bounds_gsplats`` must expand the ellipsoidal
(coverage_sigma·σ) extent on SPATIAL axes only — a splat at time=0 must not
bleed into time=1's chunk bounds (the property that keeps single-timepoint
queries fetching only their own chunks).
"""

import numpy as np

from luxar.io.ordering import (
    _BARRIER_BOUND_EPS,
    compute_chunk_bounds_gsplats,
    sort_splats_spatial,
)


class TestSortSplatsSpatial:
    def test_sort_indices_are_a_permutation(self) -> None:
        rng = np.random.default_rng(0)
        centers = rng.random((50, 3)).astype(np.float32)
        idx, meta = sort_splats_spatial(centers, method="morton")
        # Ordering must be a pure permutation — no splat dropped or duplicated.
        np.testing.assert_array_equal(np.sort(idx), np.arange(50))
        assert meta["ordering"] in ("morton", "hilbert")

    def test_hilbert_and_morton_both_permute(self) -> None:
        rng = np.random.default_rng(1)
        centers = rng.random((32, 3)).astype(np.float32)
        for method in ("morton", "hilbert"):
            idx, _ = sort_splats_spatial(centers, method=method)
            np.testing.assert_array_equal(np.sort(idx), np.arange(32))

    def test_barrier_axis_groups_before_spatial(self) -> None:
        # Axis 0 is a categorical/time barrier: after ordering, splats must be
        # grouped by their axis-0 value (non-decreasing) so a chunk never
        # straddles two categories — mirrors sort_points_compound.
        rng = np.random.default_rng(2)
        spatial = rng.random((9, 2)).astype(np.float32)
        times = np.array([2, 0, 1, 2, 0, 1, 2, 0, 1], dtype=np.float32)
        centers = np.column_stack([times, spatial]).astype(np.float32)
        idx, _ = sort_splats_spatial(centers, slice_dims=[0])
        ordered_times = centers[idx, 0]
        assert np.all(np.diff(ordered_times) >= 0), ordered_times


class TestComputeChunkBoundsGSplats:
    def test_empty_returns_zero_chunks(self) -> None:
        bounds = compute_chunk_bounds_gsplats(
            np.zeros((0, 3), np.float32), np.zeros((0, 6), np.float32), chunk_size=8
        )
        assert bounds.shape == (0, 3, 2)

    def test_spatial_axes_get_sigma_expansion(self) -> None:
        # One chunk, one splat at the origin with a diagonal Cholesky whose
        # per-axis sigma is [2,3,4]; extent = sigma * coverage_sigma.
        centers = np.zeros((1, 3), dtype=np.float32)
        chol = np.array([[2, 0, 3, 0, 0, 4]], dtype=np.float32)  # diag L=[2,3,4]
        bounds = compute_chunk_bounds_gsplats(
            centers, chol, chunk_size=1, coverage_sigma=3.0
        )
        assert bounds.shape == (1, 3, 2)
        widths = bounds[0, :, 1] - bounds[0, :, 0]
        # width = 2 * sigma * coverage_sigma = 2 * [2,3,4] * 3 = [12, 18, 24]
        np.testing.assert_allclose(widths, [12.0, 18.0, 24.0], rtol=1e-5)

    def test_barrier_axis_is_not_sigma_expanded(self) -> None:
        # Two splats at time=0 and time=1 (axis 0), each with a LARGE axis-0
        # Cholesky value. Without a barrier the time axis would balloon by
        # sigma*coverage; declared as a barrier it must stay tight (only the
        # 0..1 center range plus a float epsilon) — the anti-bleed invariant.
        centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        chol = np.tile(np.array([5, 0, 1, 0, 0, 1], dtype=np.float32), (2, 1))
        expanded = compute_chunk_bounds_gsplats(
            centers, chol, chunk_size=2, coverage_sigma=3.0, slice_dims=None
        )
        barriered = compute_chunk_bounds_gsplats(
            centers, chol, chunk_size=2, coverage_sigma=3.0, slice_dims=[0]
        )
        # Without barrier the time-axis width is ballooned (≈ 1 + 2*5*3 = 31).
        assert (expanded[0, 0, 1] - expanded[0, 0, 0]) > 25.0
        # With barrier it is exactly the center range (1.0) + 2 epsilons.
        barrier_width = barriered[0, 0, 1] - barriered[0, 0, 0]
        np.testing.assert_allclose(
            barrier_width, 1.0 + 2 * _BARRIER_BOUND_EPS, atol=1e-4
        )
