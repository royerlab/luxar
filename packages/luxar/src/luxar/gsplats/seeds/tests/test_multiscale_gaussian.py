"""
Tests for seed_from_gaussian - multiscale Gaussian blob detection.
"""

import importlib.util

import numpy as np
import pytest

from luxar.gsplats.fit_result import GSplatData
from luxar.gsplats.seeds import seed_from_gaussian

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

# Skip all tests if scipy is not available
pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")


def validate_gsplatdata(result: GSplatData, expected_ndim: int) -> None:
    """Validate GSplatData output format."""
    assert isinstance(result, GSplatData), "Result should be GSplatData"
    assert result.centers.ndim == 2, "Centers should be 2D array"
    assert result.centers.shape[1] == expected_ndim, f"Should have {expected_ndim}D coordinates"
    assert len(result.amplitudes) == len(result.centers), "Amplitudes should match centers count"
    assert len(result.sharpnesses) == len(result.centers), "Sharpnesses should match centers count"

    # Check cholesky factors shape
    tril_size = expected_ndim * (expected_ndim + 1) // 2
    assert result.cholesky_factors.shape == (len(result.centers), tril_size), (
        f"Cholesky factors should be (N, {tril_size})"
    )


class TestSeedFromGaussianBasic:
    """Test seed_from_gaussian basic functionality."""

    def test_find_seeds_2d_basic(self) -> None:
        """Test basic seed finding in 2D."""
        x, y = np.meshgrid(np.linspace(-5, 5, 31), np.linspace(-5, 5, 31))
        V = np.exp(-(x**2 + y**2) / 4) + 0.5 * np.exp(
            -((x - 3) ** 2 + (y - 3) ** 2) / 2
        )

        result = seed_from_gaussian(V)

        # Validate GSplatData format
        validate_gsplatdata(result, 2)
        assert len(result.centers) > 0, "Should find some seeds"

        # Coordinates should be within image bounds
        assert np.all(result.centers >= 0)
        assert np.all(result.centers < np.array(V.shape))

    def test_find_seeds_3d_basic(self) -> None:
        """Test basic seed finding in 3D."""
        x, y, z = np.meshgrid(
            np.linspace(-2, 2, 15), np.linspace(-2, 2, 15), np.linspace(-2, 2, 15)
        )
        V = np.exp(-(x**2 + y**2 + z**2))

        result = seed_from_gaussian(V)

        validate_gsplatdata(result, 3)
        assert len(result.centers) > 0

    def test_find_seeds_1d(self) -> None:
        """Test seed finding in 1D."""
        x = np.linspace(-3, 3, 101)
        V = np.exp(-(x**2)) + 0.3 * np.exp(-((x - 1) ** 2))

        result = seed_from_gaussian(V)

        validate_gsplatdata(result, 1)
        assert len(result.centers) > 0

    def test_parameter_effects(self) -> None:
        """Test effects of various parameters."""
        V = np.random.randn(25, 25) + 1

        # Test percentile threshold effect
        result_strict = seed_from_gaussian(V, percentile_thresh=90.0)
        result_loose = seed_from_gaussian(V, percentile_thresh=50.0)

        assert len(result_loose.centers) >= len(result_strict.centers)

        # Test min_distance effect
        result_close = seed_from_gaussian(V, min_distance=0.5)
        result_far = seed_from_gaussian(V, min_distance=5.0)

        assert len(result_far.centers) <= len(result_close.centers)


class TestSeedFromGaussianScaleInfo:
    """Test that scale information is preserved in Cholesky factors."""

    def test_scale_info_preserved(self) -> None:
        """Test that seeds from different scales have different sigmas."""
        x, y = np.meshgrid(np.linspace(-10, 10, 64), np.linspace(-10, 10, 64))

        # Large blob (should be detected at large scale)
        V_large = np.exp(-(x**2 + y**2) / 32)
        result_large = seed_from_gaussian(V_large, scales=[8.0, 16.0])

        # Small blob (should be detected at small scale)
        V_small = np.exp(-(x**2 + y**2) / 2)
        result_small = seed_from_gaussian(V_small, scales=[1.0, 2.0])

        if len(result_large.centers) > 0 and len(result_small.centers) > 0:
            # Extract sigma from Cholesky factors (diagonal elements)
            # For isotropic, L00 = sigma
            sigma_large = result_large.cholesky_factors[0, 0]  # First diagonal element
            sigma_small = result_small.cholesky_factors[0, 0]

            # Large blob should have larger sigma
            assert sigma_large > sigma_small, (
                f"Large blob sigma ({sigma_large}) should be > small blob sigma ({sigma_small})"
            )


class TestSeedFromGaussianEdgeCases:
    """Test edge cases."""

    def test_uniform_image(self) -> None:
        """Test seed finding on uniform image."""
        V = np.ones((10, 10)) * 5.0

        result = seed_from_gaussian(V)
        validate_gsplatdata(result, 2)

    def test_empty_result(self) -> None:
        """Test handling when no seeds are found."""
        V = 0.01 * np.random.randn(10, 10)

        result = seed_from_gaussian(V, percentile_thresh=99.9, peaks_per_scale=1)
        validate_gsplatdata(result, 2)
        # May have 0 seeds

    def test_single_bright_pixel(self) -> None:
        """Test with single bright pixel."""
        V = np.zeros((32, 32), dtype=np.float32)
        V[16, 16] = 1.0

        result = seed_from_gaussian(V, scales=[1], min_distance=1.0, percentile_thresh=50.0)
        validate_gsplatdata(result, 2)

    def test_very_small_image(self) -> None:
        """Test with very small image."""
        V = np.random.rand(8, 8).astype(np.float32)

        result = seed_from_gaussian(V, scales=[1])
        validate_gsplatdata(result, 2)


class TestSeedFromGaussianValidation:
    """Test parameter validation."""

    def test_empty_input_raises(self) -> None:
        """Test empty input raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            seed_from_gaussian(np.array([]))

    def test_scalar_input_raises(self) -> None:
        """Test scalar input raises error."""
        with pytest.raises(ValueError, match="at least 1 dimension"):
            seed_from_gaussian(5.0)

    def test_empty_scales_raises(self) -> None:
        """Test empty scales raises error."""
        V = np.random.rand(32, 32)
        with pytest.raises(ValueError, match="non-empty sequence"):
            seed_from_gaussian(V, scales=[])

    def test_negative_scales_raises(self) -> None:
        """Test negative scales raises error."""
        V = np.random.rand(32, 32)
        with pytest.raises(ValueError, match="positive"):
            seed_from_gaussian(V, scales=[-1, 1])

    def test_invalid_percentile_raises(self) -> None:
        """Test invalid percentile raises error."""
        V = np.random.rand(32, 32)
        with pytest.raises(ValueError, match="between 0 and 100"):
            seed_from_gaussian(V, percentile_thresh=101.0)

    def test_invalid_min_distance_raises(self) -> None:
        """Test invalid min_distance raises error."""
        V = np.random.rand(32, 32)
        with pytest.raises(ValueError, match="min_distance must be positive"):
            seed_from_gaussian(V, min_distance=-1.0)


if __name__ == "__main__":
    pytest.main([__file__])
