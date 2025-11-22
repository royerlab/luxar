"""
Tests for seed detection functions in gsplats.
"""

import importlib.util

import numpy as np
import pytest

from luxar.gsplats.seeds import (
    dedupe_farthest_first,
    find_seeds_multiscale_decomposition,
    find_seeds_multiscale_gaussian,
    local_maxima,
)

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

# Skip all tests if scipy is not available
pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")


class TestInputValidation:
    """Test input validation for all functions."""

    def test_find_seeds_input_validation(self) -> None:
        """Test input validation in find_seeds_multiscale_gaussian."""
        V = np.random.randn(10, 10)

        # Test empty input
        with pytest.raises(ValueError, match="cannot be empty"):
            find_seeds_multiscale_gaussian(np.array([]))

        # Test scalar input
        with pytest.raises(ValueError, match="at least 1 dimension"):
            find_seeds_multiscale_gaussian(5.0)

        # Test invalid scales
        with pytest.raises(ValueError, match="non-empty sequence"):
            find_seeds_multiscale_gaussian(V, scales=[])

        with pytest.raises(ValueError, match="positive"):
            find_seeds_multiscale_gaussian(V, scales=[0.5, -1.0, 2.0])

        # Test invalid parameters
        with pytest.raises(ValueError, match="peaks_per_scale must be positive"):
            find_seeds_multiscale_gaussian(V, peaks_per_scale=0)

        with pytest.raises(ValueError, match="between 0 and 100"):
            find_seeds_multiscale_gaussian(V, percentile_thresh=150.0)

        with pytest.raises(ValueError, match="min_distance must be positive"):
            find_seeds_multiscale_gaussian(V, min_distance=-1.0)

    def testlocal_maxima_edge_cases(self) -> None:
        """Test edge cases for local_maxima."""
        img = np.ones((3, 3))

        # Should handle case where no peaks exceed threshold
        coords = local_maxima(img, radius=1, thresh=10.0, top_k=5)
        assert coords.size == 0

        # Should handle top_k larger than available peaks
        img[1, 1] = 2  # Add one peak
        coords = local_maxima(img, radius=1, thresh=1.5, top_k=100)
        assert len(coords) <= 100  # Should not crash

    def testdedupe_farthest_first_edge_cases(self) -> None:
        """Test edge cases for dedupe_farthest_first."""
        # Test very small min_dist
        coords = np.array([[0, 0], [10, 10]], dtype=float)
        deduped = dedupe_farthest_first(coords, min_distance=1e-10)
        assert len(deduped) == 2  # Should keep both

        # Test very large min_dist
        deduped = dedupe_farthest_first(coords, min_distance=1000.0)
        assert len(deduped) == 1  # Should keep only one


class TestDecompositionSeeds:
    """Test find_seeds_multiscale_decomposition function."""

    def test_basic_functionality_2d(self) -> None:
        """Test basic seed generation on 2D synthetic image."""
        # Create synthetic image with known features at multiple scales
        x = np.linspace(-5, 5, 64)
        y = np.linspace(-5, 5, 64)
        X, Y = np.meshgrid(x, y)

        # Create image with two Gaussian blobs at different scales
        img = (
            np.exp(-((X + 2) ** 2 + (Y + 2) ** 2) / 0.5)  # Small blob
            + 1.5 * np.exp(-((X - 2) ** 2 + (Y - 2) ** 2) / 2.0)  # Larger blob
        )

        # Generate seeds
        seeds = find_seeds_multiscale_decomposition(
            img,
            scales=[1, 2, 4, 8],
            ignore_finest_k=1,
            min_distance=2.0,
            threshold_rel=0.1,
            decompose_kwargs={"n_iters": 100, "verbose": False},
            verbose=False,
        )

        # Should find seeds
        assert len(seeds) > 0
        assert seeds.shape[1] == 2  # 2D coordinates

        # Coordinates should be within image bounds
        assert np.all(seeds[:, 0] >= 0)
        assert np.all(seeds[:, 0] <= img.shape[0])
        assert np.all(seeds[:, 1] >= 0)
        assert np.all(seeds[:, 1] <= img.shape[1])

    def test_ignore_finest_k_parameter(self) -> None:
        """Test that ignore_finest_k properly filters scales."""
        # Simple 2D image
        img = np.random.rand(32, 32) * 0.1
        img[10:15, 10:15] = 1.0  # Add a bright region

        # Test with different ignore_finest_k values
        seeds_k0 = find_seeds_multiscale_decomposition(
            img,
            scales=[1, 2, 4],
            ignore_finest_k=0,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        seeds_k1 = find_seeds_multiscale_decomposition(
            img,
            scales=[1, 2, 4],
            ignore_finest_k=1,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        # With k=0, may find more seeds (including noise from finest scale)
        # With k=1, should find fewer seeds (ignoring finest scale)
        # Note: This is a stochastic test, so we just check shapes are correct
        assert seeds_k0.shape[1] == 2
        assert seeds_k1.shape[1] == 2

    def test_min_distance_deduplication(self) -> None:
        """Test that min_distance properly deduplicates seeds."""
        # Create image with tight cluster of features
        img = np.zeros((64, 64))
        # Add several close peaks
        for i, j in [(30, 30), (31, 30), (30, 31), (32, 32)]:
            img[i, j] = 0.8

        seeds_small_dist = find_seeds_multiscale_decomposition(
            img,
            scales=[1, 2],
            ignore_finest_k=0,
            min_distance=1.0,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        seeds_large_dist = find_seeds_multiscale_decomposition(
            img,
            scales=[1, 2],
            ignore_finest_k=0,
            min_distance=10.0,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        # Larger min_distance should result in fewer or equal seeds
        assert len(seeds_large_dist) <= len(seeds_small_dist)

    def test_threshold_rel_filtering(self) -> None:
        """Test that threshold_rel properly filters weak peaks."""
        # Create image with peaks of different intensities
        img = np.zeros((64, 64))
        img[20, 20] = 1.0  # Strong peak
        img[40, 40] = 0.2  # Weak peak

        # High threshold should find fewer seeds
        seeds_high_thresh = find_seeds_multiscale_decomposition(
            img,
            scales=[1, 2],
            ignore_finest_k=0,
            threshold_rel=0.5,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        # Low threshold should find more seeds
        seeds_low_thresh = find_seeds_multiscale_decomposition(
            img,
            scales=[1, 2],
            ignore_finest_k=0,
            threshold_rel=0.05,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        # Should find at least as many with lower threshold
        assert len(seeds_low_thresh) >= len(seeds_high_thresh)

    def test_peaks_per_scale_limiting(self) -> None:
        """Test that peaks_per_scale limits seeds per scale."""
        # Create busy image with many features
        np.random.seed(42)
        img = np.random.rand(64, 64)
        img = img + 0.5  # Shift up to create many potential peaks

        seeds_unlimited = find_seeds_multiscale_decomposition(
            img,
            scales=[1, 2, 4],
            ignore_finest_k=1,
            peaks_per_scale=None,  # Unlimited
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        seeds_limited = find_seeds_multiscale_decomposition(
            img,
            scales=[1, 2, 4],
            ignore_finest_k=1,
            peaks_per_scale=5,  # Limit to 5 per scale
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        # Limited should find fewer or equal seeds
        assert len(seeds_limited) <= len(seeds_unlimited)

    def test_1d_image(self) -> None:
        """Test seed generation on 1D signal."""
        x = np.linspace(-5, 5, 128)
        signal = np.exp(-(x**2)) + 0.5 * np.exp(-((x - 2) ** 2) / 0.5)

        seeds = find_seeds_multiscale_decomposition(
            signal,
            scales=[1, 2, 4],
            ignore_finest_k=1,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        # Should find seeds
        assert len(seeds) > 0
        assert seeds.shape[1] == 1  # 1D coordinates

        # Coordinates should be within bounds
        assert np.all(seeds[:, 0] >= 0)
        assert np.all(seeds[:, 0] <= len(signal))

    def test_3d_volume(self) -> None:
        """Test seed generation on 3D volume."""
        # Small 3D volume for speed
        x = np.linspace(-2, 2, 16)
        y = np.linspace(-2, 2, 16)
        z = np.linspace(-2, 2, 16)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Create volume with one blob
        volume = np.exp(-(X**2 + Y**2 + Z**2) / 2.0)

        seeds = find_seeds_multiscale_decomposition(
            volume,
            scales=[1, 2, 4],
            ignore_finest_k=1,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        # Should find at least one candidate
        assert len(seeds) >= 1
        assert seeds.shape[1] == 3  # 3D coordinates

    def test_empty_result_handling(self) -> None:
        """Test handling when no seeds are found."""
        # Nearly uniform image
        img = np.ones((32, 32)) * 0.5 + 0.01 * np.random.randn(32, 32)

        seeds = find_seeds_multiscale_decomposition(
            img,
            scales=[1, 2],
            ignore_finest_k=0,
            threshold_rel=0.9,  # Very high threshold
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=False,
        )

        # Should return empty array with correct shape
        assert seeds.shape[1] == 2
        # May have 0 or very few seeds

    def test_input_validation(self) -> None:
        """Test input validation for find_seeds_multiscale_decomposition."""
        valid_img = np.random.rand(32, 32)

        # Test empty input
        with pytest.raises(ValueError, match="cannot be empty"):
            find_seeds_multiscale_decomposition(np.array([]))

        # Test scalar input
        with pytest.raises(ValueError, match="at least 1 dimension"):
            find_seeds_multiscale_decomposition(5.0)

        # Test invalid scales
        with pytest.raises(ValueError, match="non-empty list"):
            find_seeds_multiscale_decomposition(valid_img, scales=[])

        with pytest.raises(ValueError, match="positive"):
            find_seeds_multiscale_decomposition(valid_img, scales=[1, -2, 4])

        # Test invalid ignore_finest_k
        with pytest.raises(ValueError, match="non-negative"):
            find_seeds_multiscale_decomposition(valid_img, ignore_finest_k=-1)

        # Test ignore_finest_k >= len(scales) triggers warning
        with pytest.warns(UserWarning, match="ignore_finest_k"):
            seeds = find_seeds_multiscale_decomposition(
                valid_img,
                scales=[1, 2],
                ignore_finest_k=5,  # Too large
                decompose_kwargs={"n_iters": 10, "verbose": False},
                verbose=False,
            )
            # Should still work (with adjusted k)
            assert seeds.shape[1] == 2

        # Test invalid min_distance
        with pytest.raises(ValueError, match="positive"):
            find_seeds_multiscale_decomposition(valid_img, min_distance=-1.0)

        # Test invalid threshold_rel
        with pytest.raises(ValueError, match="between 0 and 1"):
            find_seeds_multiscale_decomposition(valid_img, threshold_rel=1.5)

        with pytest.raises(ValueError, match="between 0 and 1"):
            find_seeds_multiscale_decomposition(valid_img, threshold_rel=-0.1)

    def test_verbose_mode(self) -> None:
        """Test that verbose mode runs without errors."""
        img = np.random.rand(32, 32)
        img[15:18, 15:18] = 1.0

        # Should not raise any errors with verbose=True
        seeds = find_seeds_multiscale_decomposition(
            img,
            scales=[1, 2, 4],
            ignore_finest_k=1,
            decompose_kwargs={"n_iters": 50, "verbose": False},
            verbose=True,  # Enable verbose output
        )

        assert seeds.shape[1] == 2

    def test_decompose_kwargs_passthrough(self) -> None:
        """Test that decompose_kwargs are properly passed through."""
        img = np.random.rand(32, 32)
        img[15:18, 15:18] = 1.0

        # Test with custom decompose parameters
        seeds = find_seeds_multiscale_decomposition(
            img,
            scales=[1, 2, 4],
            ignore_finest_k=1,
            decompose_kwargs={
                "n_iters": 10,  # Very few iterations
                "lr": 0.1,  # Custom learning rate
                "loss_type": "mse",  # Different loss type
                "verbose": False,
            },
            verbose=False,
        )

        # Should complete without errors
        assert seeds.shape[1] == 2


if __name__ == "__main__":
    pytest.main([__file__])
