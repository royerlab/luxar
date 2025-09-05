"""
Tests for candidate detection functions in gsplats.
"""

import importlib.util

import numpy as np
import pytest

from luxar.gsplats.candidates import (
    _dedupe,
    _dog_response,
    _local_maxima,
    find_candidates_overcomplete_nd,
)

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

# Skip all tests if scipy is not available
pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")


class TestLocalMaxima:
    """Test _local_maxima function."""

    def test_local_maxima_2d_simple(self):
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

        coords = _local_maxima(img, radius=1, thresh=4.0, top_k=None)

        # Should find one peak at (1, 1)
        expected = np.array([[1, 1]])
        np.testing.assert_array_equal(coords, expected)

    def test_local_maxima_multiple_peaks(self):
        """Test detection of multiple peaks."""
        # Create image with two separated peaks
        img = np.zeros((7, 7))
        img[1, 1] = 10  # Peak 1
        img[5, 5] = 8  # Peak 2

        coords = _local_maxima(img, radius=1, thresh=5.0, top_k=None)

        # Should find both peaks
        assert len(coords) == 2
        peak_locations = set((r, c) for r, c in coords)
        assert (1, 1) in peak_locations
        assert (5, 5) in peak_locations

    def test_local_maxima_threshold_filtering(self):
        """Test that threshold properly filters peaks."""
        img = np.array([[1, 2, 1], [2, 5, 2], [1, 2, 1]], dtype=float)

        # High threshold should find no peaks
        coords_high = _local_maxima(img, radius=1, thresh=10.0, top_k=None)
        assert len(coords_high) == 0

        # Low threshold should find peak
        coords_low = _local_maxima(img, radius=1, thresh=1.0, top_k=None)
        assert len(coords_low) > 0

    def test_local_maxima_top_k_limiting(self):
        """Test top_k parameter limits number of peaks."""
        # Create image with multiple peaks of different strengths
        img = np.zeros((9, 9))
        img[1, 1] = 10  # Strongest
        img[3, 3] = 8  # Second
        img[5, 5] = 6  # Third
        img[7, 7] = 4  # Weakest

        # Should return top 2 strongest peaks
        coords = _local_maxima(img, radius=1, thresh=1.0, top_k=2)
        assert len(coords) == 2

        # Check that we got the strongest peaks
        values = img[tuple(coords.T)]
        assert 10 in values  # Strongest peak
        assert 8 in values  # Second strongest

    def test_local_maxima_radius_effect(self):
        """Test that radius affects peak detection."""
        # Create image where peaks are close together
        img = np.zeros((5, 5))
        img[1, 1] = 10
        img[1, 3] = 9  # Close to first peak

        # Small radius should find both
        coords_small = _local_maxima(img, radius=1, thresh=1.0, top_k=None)
        assert len(coords_small) == 2

        # Large radius should suppress one
        coords_large = _local_maxima(img, radius=2, thresh=1.0, top_k=None)
        assert len(coords_large) == 1

    def test_local_maxima_3d(self):
        """Test local maxima detection in 3D."""
        img = np.zeros((5, 5, 5))
        img[2, 2, 2] = 10  # Central peak

        coords = _local_maxima(img, radius=1, thresh=5.0, top_k=None)

        expected = np.array([[2, 2, 2]])
        np.testing.assert_array_equal(coords, expected)

    def test_local_maxima_minimum_radius(self):
        """Test that radius < 1 is corrected to 1."""
        img = np.array([[1, 2, 1], [2, 5, 2], [1, 2, 1]], dtype=float)

        coords_neg = _local_maxima(img, radius=-1, thresh=1.0, top_k=None)
        coords_zero = _local_maxima(img, radius=0, thresh=1.0, top_k=None)
        coords_one = _local_maxima(img, radius=1, thresh=1.0, top_k=None)

        # All should give same result (radius=1)
        np.testing.assert_array_equal(coords_neg, coords_one)
        np.testing.assert_array_equal(coords_zero, coords_one)

    def test_local_maxima_empty_result(self):
        """Test handling of no peaks found."""
        # Uniform image should have no peaks above threshold
        img = np.ones((3, 3)) * 5.0

        coords = _local_maxima(img, radius=1, thresh=10.0, top_k=None)
        assert coords.size == 0
        assert coords.shape == (0, 2)  # Empty but correct shape


class TestDogResponse:
    """Test _dog_response function."""

    def test_dog_response_basic(self):
        """Test basic DoG response computation."""
        # Create simple blob-like image
        x, y = np.meshgrid(np.linspace(-3, 3, 21), np.linspace(-3, 3, 21))
        img = np.exp(-(x**2 + y**2))  # Gaussian blob

        dog = _dog_response(img, sigma=1.0, k=1.6)

        # DoG should have both positive and negative regions
        assert np.any(dog > 0)
        assert np.any(dog < 0)

        # Should have same shape as input
        assert dog.shape == img.shape

    def test_dog_response_different_k_values(self):
        """Test DoG with different k values."""
        img = np.ones((10, 10))  # Constant image

        dog1 = _dog_response(img, sigma=1.0, k=1.2)
        dog2 = _dog_response(img, sigma=1.0, k=2.0)

        # For constant image, DoG should be zero regardless of k
        np.testing.assert_allclose(dog1, 0.0, atol=1e-10)
        np.testing.assert_allclose(dog2, 0.0, atol=1e-10)

    def test_dog_response_various_sigmas(self):
        """Test DoG with various sigma values."""
        x, y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
        img = np.exp(-(x**2 + y**2))  # Gaussian blob

        for sigma in [0.5, 1.0, 2.0]:
            dog = _dog_response(img, sigma=sigma)

            # Should always have same shape
            assert dog.shape == img.shape

            # Should have meaningful range (not all zeros)
            assert np.std(dog) > 1e-6

    def test_dog_response_3d(self):
        """Test DoG response in 3D."""
        # Simple 3D Gaussian blob
        x, y, z = np.meshgrid(
            np.linspace(-2, 2, 11), np.linspace(-2, 2, 11), np.linspace(-2, 2, 11)
        )
        img = np.exp(-(x**2 + y**2 + z**2))

        dog = _dog_response(img, sigma=1.0)

        assert dog.shape == img.shape
        assert np.any(dog > 0)
        assert np.any(dog < 0)

    def test_dog_response_default_k(self):
        """Test that default k=1.6 is used."""
        img = np.random.randn(10, 10)

        dog_default = _dog_response(img, sigma=1.0)  # Uses k=1.6
        dog_explicit = _dog_response(img, sigma=1.0, k=1.6)

        np.testing.assert_array_equal(dog_default, dog_explicit)


class TestDedupe:
    """Test _dedupe function."""

    def test_dedupe_basic(self):
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

        deduped = _dedupe(coords, min_dist=1.0)

        # Should keep first and last points
        assert len(deduped) == 2
        # Should be approximately the original points (order may vary)
        # Check if [0, 0] is in the result
        assert np.any([np.allclose(row, [0, 0], atol=0.1) for row in deduped])
        # Check if [5, 5] is in the result
        assert np.any([np.allclose(row, [5, 5], atol=0.1) for row in deduped])

    def test_dedupe_no_duplicates(self):
        """Test deduplication when points are already well separated."""
        coords = np.array([[0, 0], [5, 5], [10, 10]], dtype=float)

        deduped = _dedupe(coords, min_dist=2.0)

        # Should keep all points
        assert len(deduped) == 3

    def test_dedupe_all_duplicates(self):
        """Test when all points are too close together."""
        coords = np.array([[0, 0], [0.1, 0.1], [0.2, 0.2]], dtype=float)

        deduped = _dedupe(coords, min_dist=1.0)

        # Should keep only one point
        assert len(deduped) == 1

    def test_dedupe_1d(self):
        """Test deduplication in 1D."""
        coords = np.array(
            [
                [0],
                [0.5],  # Close
                [5],
            ],
            dtype=float,
        )

        deduped = _dedupe(coords, min_dist=1.0)

        # Should keep first and last
        assert len(deduped) == 2

    def test_dedupe_3d(self):
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
        deduped = _dedupe(coords, min_dist=min_dist)

        # Should remove middle point
        assert len(deduped) == 2

        # Verify minimum distance constraint
        for i in range(len(deduped)):
            for j in range(i + 1, len(deduped)):
                dist = np.linalg.norm(deduped[i] - deduped[j])
                assert dist >= min_dist - 1e-6

    def test_dedupe_empty_input(self):
        """Test deduplication with empty input."""
        coords = np.zeros((0, 2))

        deduped = _dedupe(coords, min_dist=1.0)

        assert len(deduped) == 0
        assert deduped.shape == (0, 2)

    def test_dedupe_single_point(self):
        """Test deduplication with single point."""
        coords = np.array([[1, 2]], dtype=float)

        deduped = _dedupe(coords, min_dist=1.0)

        np.testing.assert_array_equal(deduped, coords)

    def test_dedupe_returns_float(self):
        """Test that deduplication returns float coordinates."""
        coords = np.array([[0, 0], [5, 5]], dtype=int)

        deduped = _dedupe(coords, min_dist=1.0)

        assert deduped.dtype == float


class TestFindCandidatesOvercompleteNd:
    """Test find_candidates_overcomplete_nd main function."""

    def test_find_candidates_2d_basic(self):
        """Test basic candidate finding in 2D."""
        # Create simple 2D image with known structures
        x, y = np.meshgrid(np.linspace(-5, 5, 31), np.linspace(-5, 5, 31))
        V = np.exp(-(x**2 + y**2) / 4) + 0.5 * np.exp(
            -((x - 3) ** 2 + (y - 3) ** 2) / 2
        )

        candidates = find_candidates_overcomplete_nd(V)

        # Should find some candidates
        assert len(candidates) > 0
        assert candidates.shape[1] == 2  # 2D coordinates

        # Coordinates should be within image bounds
        assert np.all(candidates >= 0)
        assert np.all(candidates < np.array(V.shape))

    def test_find_candidates_3d_basic(self):
        """Test basic candidate finding in 3D."""
        # Simple 3D blob
        x, y, z = np.meshgrid(
            np.linspace(-2, 2, 15), np.linspace(-2, 2, 15), np.linspace(-2, 2, 15)
        )
        V = np.exp(-(x**2 + y**2 + z**2))

        candidates = find_candidates_overcomplete_nd(V)

        assert len(candidates) > 0
        assert candidates.shape[1] == 3  # 3D coordinates

    def test_find_candidates_different_scales(self):
        """Test that different scales affect candidate detection."""
        V = np.random.randn(20, 20) + 2  # Noisy image

        candidates_fine = find_candidates_overcomplete_nd(
            V, scales=(0.5, 1.0), peaks_per_scale=50
        )

        candidates_coarse = find_candidates_overcomplete_nd(
            V, scales=(2.0, 3.0), peaks_per_scale=50
        )

        # Different scales should potentially give different candidates
        # At minimum, both should find some candidates
        assert len(candidates_fine) > 0
        assert len(candidates_coarse) > 0

    def test_find_candidates_parameter_effects(self):
        """Test effects of various parameters."""
        V = np.random.randn(25, 25) + 1

        # Test percentile threshold effect
        candidates_strict = find_candidates_overcomplete_nd(V, percentile_thresh=90.0)
        candidates_loose = find_candidates_overcomplete_nd(V, percentile_thresh=50.0)

        # Looser threshold should generally find more candidates
        assert len(candidates_loose) >= len(candidates_strict)

        # Test min_dist effect
        candidates_close = find_candidates_overcomplete_nd(V, min_dist=0.5)
        candidates_far = find_candidates_overcomplete_nd(V, min_dist=5.0)

        # Larger min_dist should reduce number of candidates
        assert len(candidates_far) <= len(candidates_close)

    def test_find_candidates_without_grid(self):
        """Test candidate finding without intensity grid."""
        V = np.random.randn(15, 15)

        candidates_with_grid = find_candidates_overcomplete_nd(
            V, add_intensity_grid=True
        )
        candidates_no_grid = find_candidates_overcomplete_nd(
            V, add_intensity_grid=False
        )

        # Both should work and find candidates
        assert len(candidates_with_grid) > 0
        assert len(candidates_no_grid) > 0

    def test_find_candidates_custom_grid_step(self):
        """Test custom grid step parameter."""
        V = np.ones((20, 20)) + 0.1 * np.random.randn(20, 20)

        candidates = find_candidates_overcomplete_nd(
            V, add_intensity_grid=True, grid_step=[3, 4]
        )

        assert len(candidates) > 0
        assert candidates.shape[1] == 2

    def test_find_candidates_1d(self):
        """Test candidate finding in 1D."""
        x = np.linspace(-3, 3, 101)
        V = np.exp(-(x**2)) + 0.3 * np.exp(-((x - 1) ** 2))  # Two Gaussians

        candidates = find_candidates_overcomplete_nd(V)

        assert len(candidates) > 0
        assert candidates.shape[1] == 1  # 1D coordinates

    def test_find_candidates_uniform_image(self):
        """Test candidate finding on uniform image."""
        V = np.ones((10, 10)) * 5.0  # Completely uniform

        # Should handle uniform image gracefully
        candidates = find_candidates_overcomplete_nd(V)

        # May find few or no candidates, but shouldn't crash
        assert candidates.shape[1] == 2

    def test_find_candidates_empty_result(self):
        """Test handling when no candidates are found."""
        # Very noisy image with high threshold
        V = 0.01 * np.random.randn(10, 10)

        candidates = find_candidates_overcomplete_nd(
            V, percentile_thresh=99.9, peaks_per_scale=1
        )

        # Should return empty array with correct shape
        assert candidates.shape[1] == 2
        # Length may be 0


class TestInputValidation:
    """Test input validation for all functions."""

    def test_find_candidates_input_validation(self):
        """Test input validation in find_candidates_overcomplete_nd."""
        V = np.random.randn(10, 10)

        # Test empty input
        with pytest.raises(ValueError, match="cannot be empty"):
            find_candidates_overcomplete_nd(np.array([]))

        # Test scalar input
        with pytest.raises(ValueError, match="at least 1 dimension"):
            find_candidates_overcomplete_nd(5.0)

        # Test invalid scales
        with pytest.raises(ValueError, match="non-empty sequence"):
            find_candidates_overcomplete_nd(V, scales=[])

        with pytest.raises(ValueError, match="positive"):
            find_candidates_overcomplete_nd(V, scales=[0.5, -1.0, 2.0])

        # Test invalid parameters
        with pytest.raises(ValueError, match="peaks_per_scale must be positive"):
            find_candidates_overcomplete_nd(V, peaks_per_scale=0)

        with pytest.raises(ValueError, match="between 0 and 100"):
            find_candidates_overcomplete_nd(V, percentile_thresh=150.0)

        with pytest.raises(ValueError, match="min_dist must be positive"):
            find_candidates_overcomplete_nd(V, min_dist=-1.0)

        # Test invalid grid_step
        with pytest.raises(ValueError, match="must have length"):
            find_candidates_overcomplete_nd(V, grid_step=[2, 3, 4])  # Wrong dimensions

        with pytest.raises(ValueError, match="positive"):
            find_candidates_overcomplete_nd(V, grid_step=[2, -1])

    def test_local_maxima_edge_cases(self):
        """Test edge cases for _local_maxima."""
        img = np.ones((3, 3))

        # Should handle case where no peaks exceed threshold
        coords = _local_maxima(img, radius=1, thresh=10.0, top_k=5)
        assert coords.size == 0

        # Should handle top_k larger than available peaks
        img[1, 1] = 2  # Add one peak
        coords = _local_maxima(img, radius=1, thresh=1.5, top_k=100)
        assert len(coords) <= 100  # Should not crash

    def test_dedupe_edge_cases(self):
        """Test edge cases for _dedupe."""
        # Test very small min_dist
        coords = np.array([[0, 0], [10, 10]], dtype=float)
        deduped = _dedupe(coords, min_dist=1e-10)
        assert len(deduped) == 2  # Should keep both

        # Test very large min_dist
        deduped = _dedupe(coords, min_dist=1000.0)
        assert len(deduped) == 1  # Should keep only one


if __name__ == "__main__":
    pytest.main([__file__])
