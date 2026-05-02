"""
Tests for seed detection functions in gsplats.
"""

import importlib.util

import numpy as np
import pytest
import torch

from luxar.gsplats.seeds import (
    combine_seeds,
    dedupe_farthest_first,
    local_maxima,
)

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

# Skip all tests if scipy is not available
pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")


class TestLocalMaxima:
    """Test local_maxima function."""

    def test_local_maxima_2d_simple(self) -> None:
        """Test local maxima detection on simple 2D image."""
        # Create simple 2D image with known peaks
        img = np.array(
            [
                [1, 2, 1],
                [2, 5, 2],  # Peak at (1, 1)
                [1, 2, 1],
            ],
            dtype=float,
        )

        coords = local_maxima(img, radius=1, thresh=4.0, top_k=None)

        # Should find one peak at (1, 1)
        expected = np.array([[1, 1]])
        np.testing.assert_array_equal(coords, expected)

    def test_local_maxima_multiple_peaks(self) -> None:
        """Test detection of multiple peaks."""
        # Create image with two separated peaks
        img = np.zeros((7, 7))
        img[1, 1] = 10  # Peak 1
        img[5, 5] = 8  # Peak 2

        coords = local_maxima(img, radius=1, thresh=5.0, top_k=None)

        # Should find both peaks
        assert len(coords) == 2
        peak_locations = set((r, c) for r, c in coords)
        assert (1, 1) in peak_locations
        assert (5, 5) in peak_locations

    def test_local_maxima_threshold_filtering(self) -> None:
        """Test that threshold properly filters peaks."""
        img = np.array([[1, 2, 1], [2, 5, 2], [1, 2, 1]], dtype=float)

        # High threshold should find no peaks
        coords_high = local_maxima(img, radius=1, thresh=10.0, top_k=None)
        assert len(coords_high) == 0

        # Low threshold should find peak
        coords_low = local_maxima(img, radius=1, thresh=1.0, top_k=None)
        assert len(coords_low) > 0

    def test_local_maxima_top_k_limiting(self) -> None:
        """Test top_k parameter limits number of peaks."""
        # Create image with multiple peaks of different strengths
        img = np.zeros((9, 9))
        img[1, 1] = 10  # Strongest
        img[3, 3] = 8  # Second
        img[5, 5] = 6  # Third
        img[7, 7] = 4  # Weakest

        # Should return top 2 strongest peaks
        coords = local_maxima(img, radius=1, thresh=1.0, top_k=2)
        assert len(coords) == 2

        # Check that we got the strongest peaks
        values = img[tuple(coords.T)]
        assert 10 in values  # Strongest peak
        assert 8 in values  # Second strongest

    def test_local_maxima_radius_effect(self) -> None:
        """Test that radius affects peak detection."""
        # Create image where peaks are close together
        img = np.zeros((5, 5))
        img[1, 1] = 10
        img[1, 3] = 9  # Close to first peak

        # Small radius should find both
        coords_small = local_maxima(img, radius=1, thresh=1.0, top_k=None)
        assert len(coords_small) == 2

        # Large radius should suppress one
        coords_large = local_maxima(img, radius=2, thresh=1.0, top_k=None)
        assert len(coords_large) == 1

    def test_local_maxima_3d(self) -> None:
        """Test local maxima detection in 3D."""
        img = np.zeros((5, 5, 5))
        img[2, 2, 2] = 10  # Central peak

        coords = local_maxima(img, radius=1, thresh=5.0, top_k=None)

        expected = np.array([[2, 2, 2]])
        np.testing.assert_array_equal(coords, expected)

    def test_local_maxima_minimum_radius(self) -> None:
        """Test that radius < 1 is corrected to 1."""
        img = np.array([[1, 2, 1], [2, 5, 2], [1, 2, 1]], dtype=float)

        coords_neg = local_maxima(img, radius=-1, thresh=1.0, top_k=None)
        coords_zero = local_maxima(img, radius=0, thresh=1.0, top_k=None)
        coords_one = local_maxima(img, radius=1, thresh=1.0, top_k=None)

        # All should give same result (radius=1)
        np.testing.assert_array_equal(coords_neg, coords_one)
        np.testing.assert_array_equal(coords_zero, coords_one)

    def test_local_maxima_empty_result(self) -> None:
        """Test handling of no peaks found."""
        # Uniform image should have no peaks above threshold
        img = np.ones((3, 3)) * 5.0

        coords = local_maxima(img, radius=1, thresh=10.0, top_k=None)
        assert coords.size == 0
        assert coords.shape == (0, 2)  # Empty but correct shape


class TestDedupe:
    """Test dedupe_farthest_first function."""

    def test_dedupe_farthest_first_basic(self) -> None:
        """Test basic deduplication functionality."""
        # Points that are close together
        coords = np.array(
            [
                [0, 0],
                [0.5, 0.5],  # Close to first
                [5, 5],
            ],
            dtype=float,
        )  # Far from others

        deduped, _ = dedupe_farthest_first(coords, min_distance=1.0)

        # Should keep first and last points
        assert len(deduped) == 2
        # Should be approximately the original points (order may vary)
        # Check if [0, 0] is in the result
        assert np.any([np.allclose(row, [0, 0], atol=0.1) for row in deduped])
        # Check if [5, 5] is in the result
        assert np.any([np.allclose(row, [5, 5], atol=0.1) for row in deduped])

    def test_dedupe_farthest_first_no_duplicates(self) -> None:
        """Test deduplication when points are already well separated."""
        coords = np.array([[0, 0], [5, 5], [10, 10]], dtype=float)

        deduped, _ = dedupe_farthest_first(coords, min_distance=2.0)

        # Should keep all points
        assert len(deduped) == 3

    def test_dedupe_farthest_first_all_duplicates(self) -> None:
        """Test when all points are too close together."""
        coords = np.array([[0, 0], [0.1, 0.1], [0.2, 0.2]], dtype=float)

        deduped, _ = dedupe_farthest_first(coords, min_distance=1.0)

        # Should keep only one point
        assert len(deduped) == 1

    def test_dedupe_farthest_first_1d(self) -> None:
        """Test deduplication in 1D."""
        coords = np.array(
            [
                [0],
                [0.5],  # Close
                [5],
            ],
            dtype=float,
        )

        deduped, _ = dedupe_farthest_first(coords, min_distance=1.0)

        # Should keep first and last
        assert len(deduped) == 2

    def test_dedupe_farthest_first_3d(self) -> None:
        """Test deduplication in 3D."""
        coords = np.array(
            [
                [0, 0, 0],
                [0.5, 0.5, 0.5],  # Close in 3D
                [5, 5, 5],
            ],
            dtype=float,
        )

        min_dist = 1.0
        deduped, _ = dedupe_farthest_first(coords, min_distance=min_dist)

        # Should remove middle point
        assert len(deduped) == 2

        # Verify minimum distance constraint
        for i in range(len(deduped)):
            for j in range(i + 1, len(deduped)):
                dist = np.linalg.norm(deduped[i] - deduped[j])
                assert dist >= min_dist - 1e-6

    def test_dedupe_farthest_first_empty_input(self) -> None:
        """Test deduplication with empty input."""
        coords = np.zeros((0, 2))

        deduped, _ = dedupe_farthest_first(coords, min_distance=1.0)

        assert len(deduped) == 0
        assert deduped.shape == (0, 2)

    def test_dedupe_farthest_first_single_point(self) -> None:
        """Test deduplication with single point."""
        coords = np.array([[1, 2]], dtype=float)

        deduped, _ = dedupe_farthest_first(coords, min_distance=1.0)

        np.testing.assert_array_equal(deduped, coords)

    def test_dedupe_farthest_first_returns_float(self) -> None:
        """Test that deduplication returns float coordinates."""
        coords = np.array([[0, 0], [5, 5]], dtype=int)

        deduped, _ = dedupe_farthest_first(coords, min_distance=1.0)

        assert deduped.dtype == float



class TestCombineSeeds:
    """Test combine_seeds function."""

    def test_combine_two_arrays(self) -> None:
        """Test basic combination of two seed arrays."""
        cand1 = np.array([[0, 0], [5, 5]], dtype=float)
        cand2 = np.array([[10, 10], [15, 15]], dtype=float)

        combined = combine_seeds(cand1, cand2)

        assert len(combined) == 4
        assert combined.shape[1] == 2

    def test_combine_with_deduplication(self) -> None:
        """Test combination with spatial deduplication."""
        cand1 = np.array([[0, 0], [5, 5]], dtype=float)
        cand2 = np.array(
            [[0.5, 0.5], [10, 10]], dtype=float
        )  # First is close to cand1[0]

        combined = combine_seeds(cand1, cand2, min_distance=2.0)

        # Should deduplicate the close candidates
        assert len(combined) == 3  # [0,0], [5,5], [10,10]

    def test_combine_empty_arrays(self) -> None:
        """Test combination with empty arrays."""
        cand1 = np.array([]).reshape(0, 2)
        cand2 = np.array([[5, 5]], dtype=float)

        combined = combine_seeds(cand1, cand2)

        assert len(combined) == 1
        np.testing.assert_array_equal(combined, cand2)

    def test_combine_all_empty(self) -> None:
        """Test combination when all arrays are empty."""
        cand1 = np.array([]).reshape(0, 2)
        cand2 = np.array([]).reshape(0, 2)

        combined = combine_seeds(cand1, cand2)

        assert combined.shape == (0, 2)

    def test_combine_priority_order(self) -> None:
        """Test that first array has priority in deduplication."""
        cand1 = np.array([[0, 0]], dtype=float)
        cand2 = np.array([[0.5, 0.5]], dtype=float)  # Close to cand1

        combined = combine_seeds(cand1, cand2, min_distance=1.0)

        # Should keep cand1's point (has priority)
        assert len(combined) == 1
        np.testing.assert_array_almost_equal(combined[0], [0, 0])

    def test_combine_multiple_arrays(self) -> None:
        """Test combining more than two arrays."""
        cand1 = np.array([[0, 0]], dtype=float)
        cand2 = np.array([[5, 5]], dtype=float)
        cand3 = np.array([[10, 10]], dtype=float)

        combined = combine_seeds(cand1, cand2, cand3)

        assert len(combined) == 3

    def test_combine_1d(self) -> None:
        """Test combination with 1D candidates."""
        cand1 = np.array([[0.0], [5.0]], dtype=float)
        cand2 = np.array([[10.0], [15.0]], dtype=float)

        combined = combine_seeds(cand1, cand2, min_distance=2.0)

        assert combined.shape[1] == 1
        assert len(combined) == 4

    def test_combine_invalid_method(self) -> None:
        """Test that invalid method raises error."""
        cand1 = np.array([[0, 0]], dtype=float)
        cand2 = np.array([[5, 5]], dtype=float)

        with pytest.raises(ValueError, match="Unknown combination method"):
            combine_seeds(cand1, cand2, method="invalid")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDedupeGPU:
    """Test GPU-accelerated deduplication."""

    def test_dedupe_gpu_vs_cpu_consistency(self) -> None:
        """GPU and CPU deduplication should produce consistent results."""
        # Create test coordinates
        np.random.seed(42)
        coords = np.random.rand(100, 3) * 50

        # CPU version
        cpu_deduped, cpu_indices = dedupe_farthest_first(
            coords, min_distance=3.0, device="cpu"
        )

        # GPU version
        gpu_deduped, gpu_indices = dedupe_farthest_first(
            coords, min_distance=3.0, device="cuda"
        )

        # Should produce same number of seeds
        assert len(cpu_deduped) == len(gpu_deduped)

        # Coordinates should match (order should be same due to farthest-first)
        assert np.allclose(cpu_deduped, gpu_deduped, atol=1e-5)

        # Indices should match
        assert np.array_equal(cpu_indices, gpu_indices)

    def test_dedupe_gpu_with_intensities(self) -> None:
        """GPU deduplication should work with intensity weighting."""
        np.random.seed(42)
        coords = np.random.rand(50, 2) * 20
        intensities = np.random.rand(50)

        # CPU version
        cpu_deduped, cpu_indices = dedupe_farthest_first(
            coords, min_distance=2.0, intensities=intensities, device="cpu"
        )

        # GPU version
        gpu_deduped, gpu_indices = dedupe_farthest_first(
            coords, min_distance=2.0, intensities=intensities, device="cuda"
        )

        # Should produce consistent results
        assert len(cpu_deduped) == len(gpu_deduped)
        assert np.allclose(cpu_deduped, gpu_deduped, atol=1e-5)
        assert np.array_equal(cpu_indices, gpu_indices)

    def test_dedupe_gpu_min_distance_constraint(self) -> None:
        """GPU deduplication should enforce minimum distance."""
        np.random.seed(42)
        coords = np.random.rand(100, 3) * 30

        min_dist = 5.0
        deduped, _ = dedupe_farthest_first(coords, min_distance=min_dist, device="cuda")

        # Verify minimum distance constraint
        for i in range(len(deduped)):
            for j in range(i + 1, len(deduped)):
                dist = np.linalg.norm(deduped[i] - deduped[j])
                assert dist >= min_dist - 1e-5, (
                    f"Points {i} and {j} too close: {dist} < {min_dist}"
                )

    def test_dedupe_gpu_empty_input(self) -> None:
        """GPU deduplication should handle empty input."""
        coords = np.zeros((0, 3))

        deduped, indices = dedupe_farthest_first(
            coords, min_distance=2.0, device="cuda"
        )

        assert len(deduped) == 0
        assert deduped.shape == (0, 3)
        assert len(indices) == 0

    def test_dedupe_gpu_auto_device(self) -> None:
        """Test auto device selection."""
        coords = np.random.rand(50, 2) * 20

        # Auto should use CUDA when available
        deduped, _ = dedupe_farthest_first(coords, min_distance=2.0, device="auto")

        assert len(deduped) > 0  # Should produce valid results

    def test_dedupe_gpu_2d_3d_4d(self) -> None:
        """GPU deduplication should work for different dimensions."""
        np.random.seed(42)

        for ndim in [2, 3, 4]:
            coords = np.random.rand(50, ndim) * 20

            deduped, indices = dedupe_farthest_first(
                coords, min_distance=2.0, device="cuda"
            )

            assert deduped.shape[1] == ndim
            assert len(deduped) > 0
            assert len(indices) == len(deduped)


class TestSpatialHashGrid:
    """Test SpatialHashGrid for proximity queries."""

    def test_empty_grid(self) -> None:
        """Empty grid should have no neighbors."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        grid = SpatialHashGrid(cell_size=2.0, ndim=3)
        assert len(grid) == 0
        assert grid.points.shape == (0, 3)

        point = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert not grid.has_neighbor_within(point, distance=1.0)

    def test_insert_and_len(self) -> None:
        """Inserting points should increase length."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        grid = SpatialHashGrid(cell_size=2.0, ndim=2)
        grid.insert(np.array([0.0, 0.0], dtype=np.float32))
        assert len(grid) == 1
        grid.insert(np.array([5.0, 5.0], dtype=np.float32))
        assert len(grid) == 2

    def test_points_property(self) -> None:
        """points property should return all inserted points."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        grid = SpatialHashGrid(cell_size=2.0, ndim=2)
        p1 = np.array([1.0, 2.0], dtype=np.float32)
        p2 = np.array([5.0, 6.0], dtype=np.float32)
        grid.insert(p1)
        grid.insert(p2)

        pts = grid.points
        assert pts.shape == (2, 2)
        np.testing.assert_array_almost_equal(pts[0], p1)
        np.testing.assert_array_almost_equal(pts[1], p2)

    def test_points_returns_copy(self) -> None:
        """points property should return a copy, not a view."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        grid = SpatialHashGrid(cell_size=2.0, ndim=2)
        grid.insert(np.array([1.0, 2.0], dtype=np.float32))

        pts = grid.points
        pts[0, 0] = 999.0
        # Internal state should be unchanged
        assert grid.points[0, 0] != 999.0

    def test_neighbor_within_distance(self) -> None:
        """Should detect a point within the query distance."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        grid = SpatialHashGrid(cell_size=2.0, ndim=2)
        grid.insert(np.array([0.0, 0.0], dtype=np.float32))

        # Point at distance 1.0 (within 2.0)
        close = np.array([0.7, 0.7], dtype=np.float32)
        assert grid.has_neighbor_within(close, distance=2.0)

    def test_no_neighbor_beyond_distance(self) -> None:
        """Should not detect a point beyond the query distance."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        grid = SpatialHashGrid(cell_size=5.0, ndim=2)
        grid.insert(np.array([0.0, 0.0], dtype=np.float32))

        # Point at distance ~7.07 (beyond 5.0)
        far = np.array([5.0, 5.0], dtype=np.float32)
        assert not grid.has_neighbor_within(far, distance=5.0)

    def test_boundary_distance_exact(self) -> None:
        """Point at exactly the boundary should not be detected (strict <)."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        grid = SpatialHashGrid(cell_size=5.0, ndim=2)
        grid.insert(np.array([0.0, 0.0], dtype=np.float32))

        # Point at exactly distance=5.0 along one axis
        boundary = np.array([5.0, 0.0], dtype=np.float32)
        assert not grid.has_neighbor_within(boundary, distance=5.0)

    def test_1d(self) -> None:
        """Should work in 1D."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        grid = SpatialHashGrid(cell_size=3.0, ndim=1)
        grid.insert(np.array([0.0], dtype=np.float32))
        grid.insert(np.array([10.0], dtype=np.float32))

        assert grid.has_neighbor_within(np.array([1.0], dtype=np.float32), 3.0)
        assert not grid.has_neighbor_within(np.array([5.0], dtype=np.float32), 3.0)

    def test_3d(self) -> None:
        """Should work in 3D."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        grid = SpatialHashGrid(cell_size=2.0, ndim=3)
        grid.insert(np.array([0.0, 0.0, 0.0], dtype=np.float32))

        # Distance = sqrt(3) ≈ 1.73, within 2.0
        close = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        assert grid.has_neighbor_within(close, distance=2.0)

        # Distance = sqrt(12) ≈ 3.46, beyond 2.0
        far = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        assert not grid.has_neighbor_within(far, distance=2.0)

    def test_4d(self) -> None:
        """Should work in 4D."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        grid = SpatialHashGrid(cell_size=3.0, ndim=4)
        grid.insert(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        # Distance = sqrt(4) = 2.0, within 3.0
        close = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        assert grid.has_neighbor_within(close, distance=3.0)

    def test_many_points_min_distance(self) -> None:
        """Poisson-disk-style usage: all accepted points maintain min_distance."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        rng = np.random.default_rng(42)
        min_dist = 3.0
        grid = SpatialHashGrid(cell_size=min_dist, ndim=3)

        candidates = rng.uniform(0, 50, size=(5000, 3)).astype(np.float32)

        for c in candidates:
            if not grid.has_neighbor_within(c, min_dist):
                grid.insert(c)

        pts = grid.points
        assert len(pts) > 10  # Sanity: should accept some

        # Verify all pairwise distances >= min_dist
        from scipy.spatial.distance import pdist

        dists = pdist(pts)
        assert np.all(dists >= min_dist - 1e-5), (
            f"Min pairwise distance {dists.min():.4f} < {min_dist}"
        )

    def test_consistency_with_bruteforce(self) -> None:
        """Grid queries should match brute-force distance checks."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        rng = np.random.default_rng(123)
        min_dist = 2.0
        ndim = 3
        stored = rng.uniform(0, 20, size=(100, ndim)).astype(np.float32)

        grid = SpatialHashGrid(cell_size=min_dist, ndim=ndim)
        for p in stored:
            grid.insert(p)

        # Test 200 random query points
        queries = rng.uniform(0, 20, size=(200, ndim)).astype(np.float32)
        for q in queries:
            grid_result = grid.has_neighbor_within(q, min_dist)
            # Brute force
            diffs = stored - q
            brute_min = np.min(np.sum(diffs**2, axis=1))
            brute_result = brute_min < min_dist**2
            assert grid_result == brute_result, (
                f"Mismatch at {q}: grid={grid_result}, brute={brute_result}, "
                f"min_dist_sq={brute_min:.4f}"
            )

    def test_array_growth(self) -> None:
        """Internal array should grow beyond initial allocation."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        grid = SpatialHashGrid(cell_size=0.1, ndim=2)
        # Insert more than 64 points (initial allocation)
        for i in range(200):
            grid.insert(np.array([float(i), 0.0], dtype=np.float32))

        assert len(grid) == 200
        pts = grid.points
        assert pts.shape == (200, 2)
        # Verify first and last are correct
        assert pts[0, 0] == 0.0
        assert pts[199, 0] == 199.0

    def test_negative_coordinates(self) -> None:
        """Negative coordinates should be hashed correctly (floor, not trunc)."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        grid = SpatialHashGrid(cell_size=2.0, ndim=2)
        # Insert at (-1, -1)
        grid.insert(np.array([-1.0, -1.0], dtype=np.float32))

        # Query at (-0.5, -0.5): distance = sqrt(0.5) ≈ 0.71, within 2.0
        close = np.array([-0.5, -0.5], dtype=np.float32)
        assert grid.has_neighbor_within(close, distance=2.0)

        # Query at (0.5, 0.5): distance = sqrt(4.5) ≈ 2.12, beyond 2.0
        far = np.array([0.5, 0.5], dtype=np.float32)
        assert not grid.has_neighbor_within(far, distance=2.0)

    def test_negative_coordinates_consistency_with_bruteforce(self) -> None:
        """Grid queries with negative coordinates should match brute-force."""
        from luxar.gsplats.seeds.utils import SpatialHashGrid

        rng = np.random.default_rng(99)
        min_dist = 2.0
        ndim = 3
        # Coordinates spanning negative and positive range
        stored = rng.uniform(-20, 20, size=(80, ndim)).astype(np.float32)

        grid = SpatialHashGrid(cell_size=min_dist, ndim=ndim)
        for p in stored:
            grid.insert(p)

        queries = rng.uniform(-20, 20, size=(150, ndim)).astype(np.float32)
        for q in queries:
            grid_result = grid.has_neighbor_within(q, min_dist)
            diffs = stored - q
            brute_min = np.min(np.sum(diffs**2, axis=1))
            brute_result = brute_min < min_dist**2
            assert grid_result == brute_result, (
                f"Mismatch at {q}: grid={grid_result}, brute={brute_result}, "
                f"min_dist_sq={brute_min:.4f}"
            )
