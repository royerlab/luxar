"""Tests for spatial ordering module."""

import numpy as np
import pytest

from luxar.io.ordering import (
    compute_auto_resolution,
    compute_chunk_bounds_gsplats,
    morton_encode_nd,
    normalize_coords_to_grid,
    sort_splats_spatial,
)


class TestMortonEncoding:
    """Test Morton code encoding."""

    def test_morton_encode_2d(self) -> None:
        """Test 2D Morton encoding."""
        # Simple 2D coordinates
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint32)
        morton = morton_encode_nd(coords, bits_per_dim=2)

        # Morton codes for (x, y):
        # (0,0) -> 0b00 = 0
        # (1,0) -> 0b01 = 1
        # (0,1) -> 0b10 = 2
        # (1,1) -> 0b11 = 3
        assert morton[0] == 0
        assert morton[1] == 1
        assert morton[2] == 2
        assert morton[3] == 3

    def test_morton_encode_3d(self) -> None:
        """Test 3D Morton encoding."""
        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.uint32)
        morton = morton_encode_nd(coords, bits_per_dim=2)

        # (0,0,0) -> 0b000 = 0
        # (1,0,0) -> 0b001 = 1
        assert morton[0] == 0
        assert morton[1] == 1


class TestCoordNormalization:
    """Test coordinate normalization to grid."""

    def test_normalize_basic(self) -> None:
        """Test basic normalization."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        min_coords = np.array([0.0, 0.0])
        max_coords = np.array([1.0, 1.0])

        grid = normalize_coords_to_grid(coords, min_coords, max_coords, resolution=2)

        # Should map to [0, 1]
        assert np.array_equal(grid, [[0, 0], [1, 1]])

    def test_normalize_degenerate_dimension(self) -> None:
        """Test normalization with zero-range dimension."""
        coords = np.array([[0.5, 0.0], [0.5, 1.0]], dtype=np.float32)
        min_coords = np.array([0.5, 0.0])  # X has zero range
        max_coords = np.array([0.5, 1.0])

        grid = normalize_coords_to_grid(coords, min_coords, max_coords, resolution=10)

        # X should map to middle (0) since range is zero
        # Y should map correctly
        assert grid[0, 0] == 0  # Degenerate X
        assert grid[1, 0] == 0  # Degenerate X
        assert grid[0, 1] == 0  # Y = 0
        assert grid[1, 1] == 9  # Y = 1

    def test_normalize_clamping(self) -> None:
        """Test that values outside bounds get clamped."""
        coords = np.array([[1.5, -0.5]], dtype=np.float32)  # Outside [0, 1]
        min_coords = np.array([0.0, 0.0])
        max_coords = np.array([1.0, 1.0])

        grid = normalize_coords_to_grid(coords, min_coords, max_coords, resolution=10)

        # Should clamp to [0, 9]
        assert grid[0, 0] == 9  # Clamped to max
        assert grid[0, 1] == 0  # Clamped to min


class TestAutoResolution:
    """Test automatic resolution computation."""

    def test_auto_resolution_small(self) -> None:
        """Test auto resolution for small spread."""
        coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        resolution = compute_auto_resolution(coords)

        # Spread = 1, target = 10, round to power of 2
        # Implementation rounds up: 10 -> 16 -> 256
        assert resolution == 256  # Actual behavior

    def test_auto_resolution_large(self) -> None:
        """Test auto resolution caps at max."""
        coords = np.array([[0, 0, 0], [10000, 10000, 10000]], dtype=np.float32)
        resolution = compute_auto_resolution(coords, max_resolution=2**16)

        assert resolution == 2**16  # Capped


class TestMortonSorting:
    """Test Morton-based spatial sorting."""

    def test_sort_morton_2d(self) -> None:
        """Test Morton sorting in 2D."""
        # Create scattered points
        centers = np.array(
            [
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )

        indices, metadata = sort_splats_spatial(centers, method="morton")

        # Check that (0,0) comes first
        assert indices[0] == 1  # Point at (0, 0)

        # Check metadata
        assert metadata["ordering"] == "morton"
        assert "ordering_min" in metadata
        assert "ordering_max" in metadata
        assert "ordering_bits_per_dim" in metadata

    def test_sort_morton_3d(self) -> None:
        """Test Morton sorting in 3D."""
        centers = np.random.rand(100, 3).astype(np.float32) * 10

        indices, metadata = sort_splats_spatial(
            centers, method="morton", resolution=256
        )

        # Check indices are valid
        assert len(indices) == 100
        assert set(indices) == set(range(100))

        # Check metadata
        assert metadata["ordering"] == "morton"


class TestHilbertSorting:
    """Test Hilbert curve sorting."""

    def test_sort_hilbert_2d(self) -> None:
        """Test Hilbert sorting in 2D."""
        centers = np.array(
            [
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )

        try:
            indices, metadata = sort_splats_spatial(centers, method="hilbert")

            # Check indices are valid
            assert len(indices) == 4
            assert set(indices) == {0, 1, 2, 3}

            # Check metadata
            assert metadata["ordering"] == "hilbert"
            assert "ordering_min" in metadata
            assert "ordering_max" in metadata

        except ImportError:
            pytest.skip("hilbertcurve package not installed")

    def test_sort_hilbert_3d(self) -> None:
        """Test Hilbert sorting in 3D."""
        centers = np.random.rand(100, 3).astype(np.float32) * 10

        try:
            indices, metadata = sort_splats_spatial(centers, method="hilbert")

            # Check indices are valid
            assert len(indices) == 100
            assert set(indices) == set(range(100))

        except ImportError:
            pytest.skip("hilbertcurve package not installed")


class TestSpatialSorting:
    """Test unified spatial sorting interface."""

    def test_sort_spatially_morton(self) -> None:
        """Test sort_splats_spatially with Morton."""
        centers = np.random.rand(50, 3).astype(np.float32)

        indices, metadata = sort_splats_spatial(centers, method="morton")

        assert len(indices) == 50
        assert metadata["ordering"] == "morton"

    def test_sort_spatially_hilbert(self) -> None:
        """Test sort_splats_spatially with Hilbert."""
        centers = np.random.rand(50, 3).astype(np.float32)

        try:
            indices, metadata = sort_splats_spatial(centers, method="hilbert")

            assert len(indices) == 50
            assert metadata["ordering"] == "hilbert"

        except ImportError:
            pytest.skip("hilbertcurve package not installed")

    def test_sort_spatially_invalid_method(self) -> None:
        """Test error on invalid method."""
        centers = np.random.rand(50, 3).astype(np.float32)

        with pytest.raises(ValueError, match="Unknown method"):
            sort_splats_spatial(centers, method="invalid")


class TestChunkBounds:
    """Test chunk bounding box computation."""

    def test_chunk_bounds_basic(self) -> None:
        """Test basic chunk bounds computation."""
        # Simple 2D splats
        centers = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=np.float32,
        )

        # Cholesky factors for identity covariance
        # 2D: [L00, L10, L11] where L = [[1, 0], [0, 1]]
        # So: [1.0, 0.0, 1.0]
        cholesky = np.array(
            [
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        bounds = compute_chunk_bounds_gsplats(
            centers, cholesky, chunk_size=2, coverage_sigma=3.0
        )

        # Should have 2 chunks
        assert bounds.shape == (2, 2, 2)

        # Check first chunk bounds include extent
        # Centers: [0, 0] and [1, 1]
        # Extent per dimension: sqrt(1) * 3 = 3.0
        # So min should be around -3, max around 4
        assert bounds[0, 0, 0] < 0  # Min X includes negative extent
        assert bounds[0, 0, 1] > 1  # Max X includes extent from (1, 1)

    def test_chunk_bounds_3d(self) -> None:
        """Test chunk bounds in 3D."""
        centers = np.random.rand(100, 3).astype(np.float32) * 10

        # 3D Cholesky: [L00, L10, L11, L20, L21, L22]
        # Identity covariance: L = diag(1, 1, 1) -> [1, 0, 1, 0, 0, 1]
        cholesky = np.tile([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], (100, 1)).astype(np.float32)

        bounds = compute_chunk_bounds_gsplats(centers, cholesky, chunk_size=10)

        # Should have 10 chunks
        assert bounds.shape == (10, 3, 2)

        # Check bounds are valid (min < max)
        assert np.all(bounds[:, :, 0] < bounds[:, :, 1])

    def test_chunk_bounds_anisotropic(self) -> None:
        """Test chunk bounds with anisotropic covariance."""
        # Single splat with elongated covariance
        centers = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)

        # Anisotropic: large in X, small in Y, Z
        # L = [[10, 0, 0], [0, 1, 0], [0, 0, 1]]
        # Packed: [10, 0, 1, 0, 0, 1]
        cholesky = np.array([[10.0, 0.0, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        bounds = compute_chunk_bounds_gsplats(
            centers, cholesky, chunk_size=1, coverage_sigma=3.0
        )

        # Extent in X: sqrt(100) * 3 = 30
        # Extent in Y, Z: sqrt(1) * 3 = 3
        x_extent = (bounds[0, 0, 1] - bounds[0, 0, 0]) / 2
        y_extent = (bounds[0, 1, 1] - bounds[0, 1, 0]) / 2
        z_extent = (bounds[0, 2, 1] - bounds[0, 2, 0]) / 2

        # X should be much larger than Y, Z
        assert x_extent > 25  # ~30
        assert y_extent < 5  # ~3
        assert z_extent < 5  # ~3

    # Regression test for production bug surfaced by Python round 7:
    # compute_chunk_bounds_gsplats crashed with ZeroDivisionError when
    # n_splats=0 (chunk_size resolved to 0 in _compute_chunk_size).
    def test_chunk_bounds_empty_input(self) -> None:
        """Zero-splat input returns an empty chunk-bounds array — no division
        by zero, no exception."""
        centers_3d = np.zeros((0, 3), dtype=np.float32)
        cholesky_3d = np.zeros((0, 6), dtype=np.float32)
        bounds_3d = compute_chunk_bounds_gsplats(
            centers_3d, cholesky_3d, chunk_size=1024
        )
        assert bounds_3d.shape == (0, 3, 2)
        assert bounds_3d.dtype == np.float32

        # 2D variant for symmetry
        centers_2d = np.zeros((0, 2), dtype=np.float32)
        cholesky_2d = np.zeros((0, 3), dtype=np.float32)
        bounds_2d = compute_chunk_bounds_gsplats(
            centers_2d, cholesky_2d, chunk_size=1024
        )
        assert bounds_2d.shape == (0, 2, 2)

    def test_chunk_bounds_empty_input_chunk_size_zero(self) -> None:
        """Even with the historically-buggy chunk_size=0 (caused
        ZeroDivisionError), the empty path returns a clean empty result."""
        centers = np.zeros((0, 3), dtype=np.float32)
        cholesky = np.zeros((0, 6), dtype=np.float32)
        bounds = compute_chunk_bounds_gsplats(centers, cholesky, chunk_size=0)
        assert bounds.shape == (0, 3, 2)


def _packed_cholesky(n: int, ndim: int, sigma: float, rng) -> np.ndarray:
    """Packed lower-triangular Cholesky factors for n isotropic splats of scale
    `sigma` in `ndim` dims (diagonal = sigma, off-diagonal = 0)."""
    k = ndim * (ndim + 1) // 2
    chol = np.zeros((n, k), dtype=np.float32)
    diag_idx = [d * (d + 1) // 2 + d for d in range(ndim)]
    chol[:, diag_idx] = sigma
    return chol


class TestBarrierAwareSorting:
    """`sort_splats_spatial(slice_dims=...)` — the barrier-aware (compound) path
    that keeps a chunk from straddling a categorical/time axis."""

    def test_slice_dims_none_matches_legacy_pure_spatial(self) -> None:
        """slice_dims=None reproduces the historical pure-spatial ordering
        byte-for-byte (3D data must be unaffected — regression guard)."""
        rng = np.random.default_rng(0)
        centers = (rng.random((500, 3)) * 100).astype(np.float32)
        legacy, _ = sort_splats_spatial(centers, method="hilbert")
        via_none, meta = sort_splats_spatial(centers, method="hilbert", slice_dims=None)
        via_empty, _ = sort_splats_spatial(centers, method="hilbert", slice_dims=[])
        np.testing.assert_array_equal(legacy, via_none)
        np.testing.assert_array_equal(legacy, via_empty)
        assert meta["slice_dims"] == []
        assert meta["ordering_dims"] == [0, 1, 2]

    def test_barrier_groups_are_contiguous(self) -> None:
        """With a time barrier, all splats of one timepoint are consecutive in
        the sorted order (the property that makes chunks single-timepoint)."""
        rng = np.random.default_rng(1)
        n = 900
        centers = np.empty((n, 4), dtype=np.float32)
        centers[:, :3] = rng.random((n, 3)) * 100
        centers[:, 3] = rng.integers(0, 6, size=n)  # 6 timepoints, index 3
        order, meta = sort_splats_spatial(centers, method="hilbert", slice_dims=[3])
        assert meta["slice_dims"] == [3]
        assert meta["ordering_dims"] == [0, 1, 2]
        t_sorted = centers[order, 3]
        # Non-decreasing time => each timepoint is a contiguous run.
        assert np.all(np.diff(t_sorted) >= 0)
        # Number of transitions == number of distinct timepoints - 1.
        assert np.count_nonzero(np.diff(t_sorted) != 0) == len(np.unique(centers[:, 3])) - 1

    def test_chunks_are_single_timepoint(self) -> None:
        """The bug's killer invariant: with a barrier, every chunk's time-extent
        is ~0 except the <=1 boundary chunk per transition."""
        rng = np.random.default_rng(2)
        n, chunk_size = 2000, 128
        centers = np.empty((n, 4), dtype=np.float32)
        centers[:, :3] = rng.random((n, 3)) * 100
        centers[:, 3] = rng.integers(0, 5, size=n)
        order, _ = sort_splats_spatial(centers, method="hilbert", slice_dims=[3])
        sorted_centers = centers[order]
        chol = _packed_cholesky(n, 4, sigma=2.0, rng=rng)
        bounds = compute_chunk_bounds_gsplats(
            sorted_centers, chol, chunk_size, coverage_sigma=3.0, slice_dims=[3]
        )
        t_extent = bounds[:, 3, 1] - bounds[:, 3, 0]
        # Single-timepoint chunks have extent ~1.0 (tight ±0.5), NOT the 3σ*2=6
        # they'd get without slice_dims. Boundary chunks (<= n_transitions) may
        # span 2 timepoints (extent ~2.0). The vast majority are single.
        n_boundary = np.count_nonzero(t_extent > 1.5)
        assert n_boundary <= len(np.unique(centers[:, 3]))  # <= transitions+1
        assert np.all(t_extent <= 2.0 + 1e-4)  # never the 6.0 σ-expansion

    def test_two_barrier_axes(self) -> None:
        """Two categorical axes (time + channel) both lexsort-first."""
        rng = np.random.default_rng(3)
        n = 800
        centers = np.empty((n, 5), dtype=np.float32)
        centers[:, :3] = rng.random((n, 3)) * 50
        centers[:, 3] = rng.integers(0, 3, size=n)  # time
        centers[:, 4] = rng.integers(0, 2, size=n)  # channel
        order, meta = sort_splats_spatial(centers, method="hilbert", slice_dims=[3, 4])
        assert meta["slice_dims"] == [3, 4]
        assert meta["ordering_dims"] == [0, 1, 2]
        # (time, channel) pairs are non-decreasing lexicographically.
        pairs = centers[order][:, [3, 4]]
        keys = pairs[:, 0] * 10 + pairs[:, 1]
        assert np.all(np.diff(keys) >= 0)


class TestBarrierChunkBounds:
    """`compute_chunk_bounds_gsplats(slice_dims=...)` — barrier axes get tight
    ±0.5 bounds, spatial axes keep the ellipsoidal σ extent."""

    def test_barrier_axis_no_sigma_expansion(self) -> None:
        rng = np.random.default_rng(4)
        n = 256
        centers = np.zeros((n, 4), dtype=np.float32)
        centers[:, :3] = rng.random((n, 3)) * 10
        centers[:, 3] = 2.0  # all at timepoint 2
        chol = _packed_cholesky(n, 4, sigma=5.0, rng=rng)
        bounds = compute_chunk_bounds_gsplats(
            centers, chol, chunk_size=n, coverage_sigma=3.0, slice_dims=[3]
        )
        # Barrier dim: tight ±0.5 around 2.0, NOT 2 ± 3*5.
        assert bounds[0, 3, 0] == pytest.approx(1.5)
        assert bounds[0, 3, 1] == pytest.approx(2.5)
        # Spatial dims keep the σ extent (much wider than the value spread).
        assert bounds[0, 0, 1] - bounds[0, 0, 0] > 10.0

    def test_different_timepoints_dont_overlap_in_barrier(self) -> None:
        """Two chunks at different timepoints do not overlap in the barrier dim
        even with huge σ (mirror points test_discrete_chunks_dont_overlap)."""
        rng = np.random.default_rng(5)
        centers = np.zeros((256, 4), dtype=np.float32)
        centers[:128, 3] = 0.0
        centers[128:, 3] = 1.0
        centers[:, :3] = rng.random((256, 3)) * 5
        chol = _packed_cholesky(256, 4, sigma=100.0, rng=rng)
        bounds = compute_chunk_bounds_gsplats(
            centers, chol, chunk_size=128, coverage_sigma=3.0, slice_dims=[3]
        )
        # No INTERIOR overlap in the barrier dim: chunk 0 (t=0, [-0.5,0.5]) max
        # <= chunk 1 (t=1, [0.5,1.5]) min. The half-cell edges touch at 0.5 (as
        # with Points' ±0.5 padding); a categorical query at an integer value
        # with tolerance < 0.5 still isolates one timepoint. Contrast the σ
        # expansion (100·3) that would make them overlap massively without
        # slice_dims.
        assert bounds[0, 3, 1] <= bounds[1, 3, 0] + 1e-6
        assert bounds[0, 3, 1] == pytest.approx(0.5)
        assert bounds[1, 3, 0] == pytest.approx(0.5)

    def test_query_at_timepoint_selects_only_its_chunks(self) -> None:
        """An AABB query at one timepoint intersects only that timepoint's
        chunks (mirror points test_query_at_discrete_value_finds_correct_chunks).

        Categorical navigation queries the exact integer value (tolerance < 0.5),
        so the ±0.5 half-cell padding isolates one timepoint cleanly."""
        rng = np.random.default_rng(6)
        n, chunk_size = 1500, 128
        centers = np.empty((n, 4), dtype=np.float32)
        centers[:, :3] = rng.random((n, 3)) * 100
        centers[:, 3] = rng.integers(0, 4, size=n)  # 4 timepoints
        order, _ = sort_splats_spatial(centers, method="hilbert", slice_dims=[3])
        sc = centers[order]
        chol = _packed_cholesky(n, 4, sigma=3.0, rng=rng)
        bounds = compute_chunk_bounds_gsplats(
            sc, chol, chunk_size, coverage_sigma=3.0, slice_dims=[3]
        )
        total = bounds.shape[0]
        # Point query at t=2 (categorical nav lands on the exact value).
        tq = 2.0
        hit = np.count_nonzero((bounds[:, 3, 0] <= tq) & (bounds[:, 3, 1] >= tq))
        # ~1/4 of chunks (one of four timepoints), NOT all of them.
        assert 1 <= hit <= total // 4 + 1
        # Sanity: without slice_dims the σ-expanded bounds would hit far more.
        wide = compute_chunk_bounds_gsplats(sc, chol, chunk_size, coverage_sigma=3.0)
        wide_hit = np.count_nonzero((wide[:, 3, 0] <= tq) & (wide[:, 3, 1] >= tq))
        assert wide_hit > hit


class TestDetectBarrierDims:
    """`detect_barrier_dims` — conservative auto-detection fallback."""

    def test_integer_time_axis_detected(self) -> None:
        from luxar.io.ordering import detect_barrier_dims

        rng = np.random.default_rng(7)
        centers = np.empty((5000, 4), dtype=np.float32)
        centers[:, :3] = rng.random((5000, 3)) * 1000  # continuous spatial
        centers[:, 3] = rng.integers(0, 51, size=5000)  # 51 timepoints
        assert detect_barrier_dims(centers) == [3]

    def test_continuous_spatial_not_detected(self) -> None:
        from luxar.io.ordering import detect_barrier_dims

        rng = np.random.default_rng(8)
        centers = (rng.random((5000, 3)) * 1000).astype(np.float32)
        assert detect_barrier_dims(centers) == []

    def test_wide_integer_axis_not_detected(self) -> None:
        """An integer-valued but high-cardinality axis (e.g. a fine spatial grid)
        exceeds the cardinality cap and is not misread as categorical."""
        from luxar.io.ordering import detect_barrier_dims

        rng = np.random.default_rng(9)
        centers = np.empty((5000, 2), dtype=np.float32)
        centers[:, 0] = rng.integers(0, 4000, size=5000)  # 4000 > 1024 cap
        centers[:, 1] = rng.integers(0, 3, size=5000)  # 3 categories
        assert detect_barrier_dims(centers, max_cardinality=1024) == [1]

    def test_empty_input(self) -> None:
        from luxar.io.ordering import detect_barrier_dims

        assert detect_barrier_dims(np.zeros((0, 4), dtype=np.float32)) == []
