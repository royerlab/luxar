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

