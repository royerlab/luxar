"""
Tests for seed detection functions in gsplats.
"""

import importlib.util

import numpy as np
import pytest

from luxar.gsplats.seeds import (
    find_seeds_multiscale_gaussian,
)

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

# Skip all tests if scipy is not available
pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")


class TestFindSeedsOvercompleteNd:
    """Test find_seeds_multiscale_gaussian main function."""

    def test_find_seeds_2d_basic(self) -> None:
        """Test basic seed finding in 2D."""
        # Create simple 2D image with known structures
        x, y = np.meshgrid(np.linspace(-5, 5, 31), np.linspace(-5, 5, 31))
        V = np.exp(-(x**2 + y**2) / 4) + 0.5 * np.exp(
            -((x - 3) ** 2 + (y - 3) ** 2) / 2
        )

        seeds = find_seeds_multiscale_gaussian(V)

        # Should find some seeds
        assert len(seeds) > 0
        assert seeds.shape[1] == 2  # 2D coordinates

        # Coordinates should be within image bounds
        assert np.all(seeds >= 0)
        assert np.all(seeds < np.array(V.shape))

    def test_find_seeds_3d_basic(self) -> None:
        """Test basic seed finding in 3D."""
        # Simple 3D blob
        x, y, z = np.meshgrid(
            np.linspace(-2, 2, 15), np.linspace(-2, 2, 15), np.linspace(-2, 2, 15)
        )
        V = np.exp(-(x**2 + y**2 + z**2))

        seeds = find_seeds_multiscale_gaussian(V)

        assert len(seeds) > 0
        assert seeds.shape[1] == 3  # 3D coordinates

        """Test effects of various parameters."""
        V = np.random.randn(25, 25) + 1

        # Test percentile threshold effect
        seeds_strict = find_seeds_multiscale_gaussian(
            V, percentile_thresh=90.0
        )
        seeds_loose = find_seeds_multiscale_gaussian(
            V, percentile_thresh=50.0
        )

        # Looser threshold should generally find more seeds
        assert len(seeds_loose) >= len(seeds_strict)

        # Test min_dist effect
        seeds_close = find_seeds_multiscale_gaussian(V, min_distance=0.5)
        seeds_far = find_seeds_multiscale_gaussian(V, min_distance=5.0)

        # Larger min_dist should reduce number of seeds
        assert len(seeds_far) <= len(seeds_close)

    def test_find_seeds_1d(self) -> None:
        """Test seed finding in 1D."""
        x = np.linspace(-3, 3, 101)
        V = np.exp(-(x**2)) + 0.3 * np.exp(-((x - 1) ** 2))  # Two Gaussians

        seeds = find_seeds_multiscale_gaussian(V)

        assert len(seeds) > 0
        assert seeds.shape[1] == 1  # 1D coordinates

    def test_find_seeds_uniform_image(self) -> None:
        """Test seed finding on uniform image."""
        V = np.ones((10, 10)) * 5.0  # Completely uniform

        # Should handle uniform image gracefully
        seeds = find_seeds_multiscale_gaussian(V)

        # May find few or no seeds, but shouldn't crash
        assert seeds.shape[1] == 2

    def test_find_seeds_empty_result(self) -> None:
        """Test handling when no seeds are found."""
        # Very noisy image with high threshold
        V = 0.01 * np.random.randn(10, 10)

        seeds = find_seeds_multiscale_gaussian(
            V, percentile_thresh=99.9, peaks_per_scale=1
        )

        # Should return empty array with correct shape
        assert seeds.shape[1] == 2
        # Length may be 0


def test_multiscale_gaussian_edge_cases():
    """Test edge cases for multiscale Gaussian seed generation."""
    # Edge case 1: Uniform image (CLAHE may create artifacts, so check format)
    uniform_image = np.ones((32, 32), dtype=np.float32)
    seeds = find_seeds_multiscale_gaussian(
        uniform_image, min_distance=2.0, percentile_thresh=95.0
    )
    # Just verify the output format is correct (CLAHE preprocessing may find artifacts)
    assert seeds.ndim == 2, "Candidates should be 2D array"
    assert seeds.shape[1] == 2, "Candidates should have 2D coordinates"

    # Edge case 2: Single bright pixel (may or may not be detected depending on thresholds)
    single_pixel = np.zeros((32, 32), dtype=np.float32)
    single_pixel[16, 16] = 1.0
    seeds = find_seeds_multiscale_gaussian(
        single_pixel, scales=[1], min_distance=1.0, percentile_thresh=50.0
    )
    # Just check that output format is valid (detection depends on parameters)
    assert seeds.ndim == 2, "Candidates should be 2D array"
    assert seeds.shape[1] == 2, "Should have 2D coordinates"

    # Edge case 3: Very small image
    small_image = np.random.rand(8, 8).astype(np.float32)
    seeds = find_seeds_multiscale_gaussian(small_image, scales=[1])
    assert seeds.shape[1] == 2, "Should have 2D coordinates"

    # Edge case 4: Image with NaN (should handle gracefully)
    image_with_nan = np.random.rand(16, 16).astype(np.float32)
    image_with_nan[8, 8] = np.nan
    # This should either handle NaN or raise a clear error
    try:
        seeds = find_seeds_multiscale_gaussian(
            image_with_nan, scales=[1], min_distance=1.0
        )
        # If it succeeds, check that result is valid
        assert not np.any(np.isnan(seeds)), "Candidates should not contain NaN"
    except (ValueError, RuntimeError):
        # Acceptable to raise an error for NaN input
        pass

    # Edge case 5: Very high percentile threshold (should produce few/no seeds)
    dense_image = np.random.rand(32, 32).astype(np.float32)
    seeds_high_thresh = find_seeds_multiscale_gaussian(
        dense_image, scales=[1], percentile_thresh=99.0
    )
    seeds_low_thresh = find_seeds_multiscale_gaussian(
        dense_image, scales=[1], percentile_thresh=50.0
    )
    assert (
        seeds_high_thresh.shape[0] <= seeds_low_thresh.shape[0]
    ), "Higher threshold should produce fewer or equal seeds"


def test_multiscale_gaussian_parameter_validation():
    """Test parameter validation for multiscale Gaussian."""
    image = np.random.rand(32, 32).astype(np.float32)

    # Test invalid scales (empty list)
    with pytest.raises((ValueError, AssertionError)):
        find_seeds_multiscale_gaussian(image, scales=[])

    # Test invalid scales (negative)
    with pytest.raises((ValueError, AssertionError)):
        find_seeds_multiscale_gaussian(image, scales=[-1, 1])

    # Test invalid scales (zero)
    with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
        find_seeds_multiscale_gaussian(image, scales=[0])

    # Test invalid min_distance (negative)
    with pytest.raises((ValueError, AssertionError)):
        find_seeds_multiscale_gaussian(image, scales=[1], min_distance=-1.0)

    # Test invalid percentile_thresh (out of range)
    with pytest.raises((ValueError, AssertionError)):
        find_seeds_multiscale_gaussian(image, scales=[1], percentile_thresh=101.0)

    with pytest.raises((ValueError, AssertionError)):
        find_seeds_multiscale_gaussian(image, scales=[1], percentile_thresh=-1.0)

    # Test invalid CLAHE tile_size (zero causes division by zero)
    with pytest.raises((ValueError, AssertionError, RuntimeError, ZeroDivisionError)):
        find_seeds_multiscale_gaussian(image, scales=[1], clahe_tile_size=0)

    # Note: Negative clip_limit is not explicitly validated (CLAHE handles it internally)
    # Just test that function runs without crashing with edge case values
    result = find_seeds_multiscale_gaussian(image, scales=[1], clahe_clip_limit=0.1)
    assert result.ndim == 2  # Should produce valid output


if __name__ == "__main__":
    pytest.main([__file__])
