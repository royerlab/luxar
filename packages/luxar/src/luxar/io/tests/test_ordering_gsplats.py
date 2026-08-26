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

    def test_uniform_cholesky_spans_multiple_chunks(self) -> None:
        # Regression for issue #729: the uniform-Cholesky convenience keeps a
        # single packed row (shape (1, k)) shared by all splats. When the splat
        # count exceeds one chunk, positional slicing yields an empty (0, k)
        # array for chunk 1+ and used to crash with a broadcast error. The
        # shared row must be applied to every chunk instead.
        # N does NOT divide evenly by chunk_size, so the partial final chunk
        # (201 splats) is exercised alongside the full ones.
        n_splats = 3001
        chunk_size = 700  # → 5 chunks (4×700 + 1×201)
        coverage_sigma = 3.0
        rng = np.random.default_rng(729)
        centers = rng.random((n_splats, 3)).astype(np.float32)
        # Single shared diagonal Cholesky. Packed lower-triangular row order:
        # covariance[d,d] = sum(L[d*(d+1)//2 + i]^2, i=0..d), so the per-axis
        # sigmas here are exactly [1, 2, 3].
        chol = np.array([[1, 0, 2, 0, 0, 3]], dtype=np.float32)
        sigmas = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        bounds = compute_chunk_bounds_gsplats(
            centers, chol, chunk_size=chunk_size, coverage_sigma=coverage_sigma
        )
        num_chunks = (n_splats + chunk_size - 1) // chunk_size
        assert bounds.shape == (num_chunks, 3, 2)
        # The uniform per-axis extent is a constant coverage_sigma·sigma_d, so
        # each chunk's width must be EXACTLY the raw center span plus one extent
        # on each side. Pins the diagonal-covariance math, not just "expanded".
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_splats)
            for d in range(3):
                width = bounds[chunk_idx, d, 1] - bounds[chunk_idx, d, 0]
                raw_span = centers[start:end, d].max() - centers[start:end, d].min()
                expected = raw_span + 2.0 * coverage_sigma * sigmas[d]
                assert np.isclose(width, expected, rtol=1e-5, atol=1e-4), (
                    chunk_idx,
                    d,
                    width,
                    expected,
                )

    def test_sigma_extent_survives_the_float32_store_at_large_coordinates(
        self,
    ) -> None:
        """A small σ must still widen the bound where it is under half an ULP.

        ``chunk_bounds`` is float32 while the σ-extent is a small ABSOLUTE
        quantity. Past ``|x| ~ 2**23`` the ULP exceeds 1, so accumulating
        ``center ± coverage_sigma·σ`` in float32 (and storing it round-to-
        nearest) lands back on the unpadded center: the stored bound is TIGHTER
        than the ellipsoid the renderer draws and the splat is dropped from
        queries at its own edge. Pins both the per-splat and the uniform
        (shared ``(1, k)`` row) Cholesky paths.
        """
        # ULP is 2.0 at 2e7, so a 0.3 extent rounds away without the fix.
        base = 2.0e7
        sigma, coverage = 0.1, 3.0
        extent = sigma * coverage
        centers = np.array([[base, base, base], [base, base, base]], dtype=np.float32)
        row = np.array([sigma, 0, sigma, 0, 0, sigma], dtype=np.float32)

        # (1, k) exercises the shared-row broadcast path, (N, k) the per-splat one.
        for chol in (row.reshape(1, 6), np.tile(row, (2, 1))):
            bounds = compute_chunk_bounds_gsplats(
                centers, chol, chunk_size=2, coverage_sigma=coverage
            )
            assert bounds.dtype == np.float32
            assert bounds.shape == (1, 3, 2)
            for dim in range(3):
                lo = np.float64(bounds[0, dim, 0])
                hi = np.float64(bounds[0, dim, 1])
                assert lo <= np.float64(base) - extent, (chol.shape, dim, lo)
                assert hi >= np.float64(base) + extent, (chol.shape, dim, hi)

    def test_barrier_epsilon_survives_the_float32_store_at_large_coordinates(
        self,
    ) -> None:
        """The barrier pad is absolute too, so it vanishes the same way.

        A categorical axis whose values are large (a millisecond timestamp, an
        acquisition index offset into an experiment) gets ``± _BARRIER_BOUND_EPS``
        added in float32; at ``|x| = 2e7`` that is ~1e-3 against a half-ULP of
        1.0, so the stored bound collapses onto the exact category value and a
        query landing a float ULP off it misses the chunk.
        """
        base = 2.0e7
        centers = np.array([[base, 0.0, 0.0]], dtype=np.float32)
        chol = np.array([[1.0, 0, 1.0, 0, 0, 1.0]], dtype=np.float32)

        bounds = compute_chunk_bounds_gsplats(
            centers, chol, chunk_size=1, coverage_sigma=3.0, slice_dims=[0]
        )

        assert np.float64(bounds[0, 0, 0]) < np.float64(base)
        assert np.float64(bounds[0, 0, 1]) > np.float64(base)

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


def test_barrier_bound_eps_matches_viewer_gsplats_step_fraction() -> None:
    """``_BARRIER_BOUND_EPS`` and the viewer's gsplats continuous-dim epsilon are
    ONE decision expressed twice, and nothing but this test links them.

    The viewer's ``gsplatsContinuousDimTolerance`` is documented as the
    reader-side mirror of this pad: at a unit step its step-scaled term is
    ``GSPLATS_CONTINUOUS_EPS_STEP_FRACTION × 1``, which must equal
    ``_BARRIER_BOUND_EPS``. Same style as the discrete rule's Python-side pin
    (``io/tests/_compiler/test_finalize.py``
    ``test_validate_discrete_ranges_tolerance_matches_viewer_quarter_step``),
    but parsing the TypeScript declaration instead of restating the number —
    a prose comment on each side is not a link.
    """
    from luxar.conftest import read_ts_number_const, viewer_source

    computer = viewer_source("src/data/loaders/spatial-query/tolerance-computer.ts")
    source = computer.read_text(encoding="utf-8")

    step_fraction = read_ts_number_const(source, "GSPLATS_CONTINUOUS_EPS_STEP_FRACTION")
    assert step_fraction == _BARRIER_BOUND_EPS, (
        f"GSPLATS_CONTINUOUS_EPS_STEP_FRACTION ({step_fraction}) must equal "
        f"_BARRIER_BOUND_EPS ({_BARRIER_BOUND_EPS}): the viewer applies the "
        "step-scaled term to the CONTINUOUS dims the writer leaves unpadded, "
        "and it is calibrated to the pad the writer puts on barrier dims. "
        f"Change BOTH — {computer} and "
        "luxar/io/_ordering/bounds.py — or the mirror silently desyncs."
    )

    # The step-scaled term must also stay strictly below the half-cell membership
    # gates, for the same reason the discrete quarter-cell reach does: pad + reach
    # must never sum to a full step at a unit-ish step. (There is no cap on the
    # reader's second, ABSOLUTE term — a review of #1183 removed the quarter-cell
    # ceiling that used to be here, because the band it caps is what the renderer
    # genuinely shows on a micro-step axis. See `gsplatsContinuousDimTolerance`.)
    assert 0.0 < step_fraction < 0.5, (
        f"GSPLATS_CONTINUOUS_EPS_STEP_FRACTION ({step_fraction}) scales with the "
        "step; at >= 0.5 a query on one cell reaches the neighbouring cell, "
        "which is the over-fetch the quarter-cell rules exist to prevent."
    )
