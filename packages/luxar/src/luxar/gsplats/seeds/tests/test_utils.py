"""
Tests for seed detection functions in gsplats.
"""

import importlib.util

import numpy as np
import pytest

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

        deduped = dedupe_farthest_first(coords, min_distance=1.0)

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

        deduped = dedupe_farthest_first(coords, min_distance=2.0)

        # Should keep all points
        assert len(deduped) == 3

    def test_dedupe_farthest_first_all_duplicates(self) -> None:
        """Test when all points are too close together."""
        coords = np.array([[0, 0], [0.1, 0.1], [0.2, 0.2]], dtype=float)

        deduped = dedupe_farthest_first(coords, min_distance=1.0)

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

        deduped = dedupe_farthest_first(coords, min_distance=1.0)

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
        deduped = dedupe_farthest_first(coords, min_distance=min_dist)

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

        deduped = dedupe_farthest_first(coords, min_distance=1.0)

        assert len(deduped) == 0
        assert deduped.shape == (0, 2)

    def test_dedupe_farthest_first_single_point(self) -> None:
        """Test deduplication with single point."""
        coords = np.array([[1, 2]], dtype=float)

        deduped = dedupe_farthest_first(coords, min_distance=1.0)

        np.testing.assert_array_equal(deduped, coords)

    def test_dedupe_farthest_first_returns_float(self) -> None:
        """Test that deduplication returns float coordinates."""
        coords = np.array([[0, 0], [5, 5]], dtype=int)

        deduped = dedupe_farthest_first(coords, min_distance=1.0)

        assert deduped.dtype == float


if __name__ == "__main__":
    pytest.main([__file__])


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
