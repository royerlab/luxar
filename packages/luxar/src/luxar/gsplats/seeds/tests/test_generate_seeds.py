"""
Tests for the generate_seeds() unified entry point function.

All seeding methods now return GSplatData with scale-informed Gaussian shapes.
"""

import importlib.util
import warnings

import numpy as np
import pytest

from luxar.gsplats.fit_result import GSplatData
from luxar.gsplats.seeds import generate_seeds

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")


@pytest.fixture
def simple_2d_image():
    """Create a simple 2D image with Gaussian blobs."""
    x, y = np.meshgrid(np.linspace(-5, 5, 51), np.linspace(-5, 5, 51))
    V = np.exp(-(x**2 + y**2) / 4) + 0.5 * np.exp(-((x - 3) ** 2 + (y - 3) ** 2) / 2)
    return V


@pytest.fixture
def simple_3d_image():
    """Create a simple 3D volume with a Gaussian blob."""
    x, y, z = np.meshgrid(
        np.linspace(-2, 2, 21), np.linspace(-2, 2, 21), np.linspace(-2, 2, 21)
    )
    V = np.exp(-(x**2 + y**2 + z**2) / 2)
    return V


@pytest.fixture
def uniform_image():
    """Create a uniform image."""
    return np.ones((32, 32), dtype=float) * 5.0


@pytest.fixture
def noisy_image():
    """Create a noisy image."""
    return np.random.randn(32, 32) + 1.0


def validate_gsplatdata(
    result: GSplatData, expected_ndim: int, image_shape: tuple
) -> None:
    """Validate GSplatData output format."""
    assert isinstance(result, GSplatData), "Result should be GSplatData"
    assert result.centers.ndim == 2, "Centers should be 2D array (N, ndim)"
    assert result.centers.shape[1] == expected_ndim, (
        f"Should have {expected_ndim}D coordinates"
    )

    N = len(result.centers)
    assert len(result.amplitudes) == N, "Amplitudes should match centers count"
    assert len(result.sharpnesses) == N, "Sharpnesses should match centers count"

    tril_size = expected_ndim * (expected_ndim + 1) // 2
    assert result.cholesky_factors.shape == (N, tril_size), (
        f"Cholesky factors should be (N, {tril_size})"
    )

    # Check coordinates within bounds
    if N > 0:
        assert np.all(result.centers >= 0), (
            "Centers should have non-negative coordinates"
        )
        assert np.all(result.centers < np.array(image_shape)), (
            "Centers should be within bounds"
        )


class TestMethodSelection:
    """Test different method selection options."""

    def test_method_gaussian(self, simple_2d_image) -> None:
        """Test method='gaussian'."""
        result = generate_seeds(simple_2d_image, method="gaussian")
        validate_gsplatdata(result, 2, simple_2d_image.shape)
        assert len(result.centers) > 0

    def test_method_decomposition(self, simple_2d_image) -> None:
        """Test method='decomposition' (default)."""
        result = generate_seeds(simple_2d_image, method="decomposition")
        validate_gsplatdata(result, 2, simple_2d_image.shape)
        assert len(result.centers) > 0

    def test_method_both(self, simple_2d_image) -> None:
        """Test method='both'."""
        result = generate_seeds(simple_2d_image, method="both")
        validate_gsplatdata(result, 2, simple_2d_image.shape)
        assert len(result.centers) > 0

    def test_method_gaussian_then_decomposition(self, simple_2d_image) -> None:
        """Test method='gaussian,decomposition'."""
        result = generate_seeds(simple_2d_image, method="gaussian,decomposition")
        validate_gsplatdata(result, 2, simple_2d_image.shape)
        assert len(result.centers) > 0

    def test_method_case_insensitive(self, simple_2d_image) -> None:
        """Test that method string is case-insensitive."""
        result_lower = generate_seeds(simple_2d_image, method="gaussian")
        result_upper = generate_seeds(simple_2d_image, method="GAUSSIAN")

        # Both should be valid GSplatData with same number of seeds
        validate_gsplatdata(result_lower, 2, simple_2d_image.shape)
        validate_gsplatdata(result_upper, 2, simple_2d_image.shape)
        assert len(result_lower.centers) == len(result_upper.centers)


class TestParameterRouting:
    """Test parameter routing to methods."""

    def test_percentile_thresh_affects_gaussian(self, simple_2d_image) -> None:
        """Test percentile_thresh parameter."""
        result_strict = generate_seeds(
            simple_2d_image, method="gaussian", percentile_thresh=90.0
        )
        result_loose = generate_seeds(
            simple_2d_image, method="gaussian", percentile_thresh=50.0
        )

        assert len(result_loose.centers) >= len(result_strict.centers)

    def test_ignore_finest_k_affects_decomposition(self, simple_2d_image) -> None:
        """Test ignore_finest_k parameter."""
        result_k0 = generate_seeds(
            simple_2d_image, method="decomposition", ignore_finest_k=0
        )
        result_k2 = generate_seeds(
            simple_2d_image, method="decomposition", ignore_finest_k=2
        )

        validate_gsplatdata(result_k0, 2, simple_2d_image.shape)
        validate_gsplatdata(result_k2, 2, simple_2d_image.shape)

    def test_min_distance_common_param(self, simple_2d_image) -> None:
        """Test min_distance parameter."""
        for method in ["gaussian", "decomposition", "both"]:
            result_close = generate_seeds(
                simple_2d_image, method=method, min_distance=1.0
            )
            result_far = generate_seeds(
                simple_2d_image, method=method, min_distance=8.0
            )

            validate_gsplatdata(result_close, 2, simple_2d_image.shape)
            validate_gsplatdata(result_far, 2, simple_2d_image.shape)

    def test_unused_param_warning(self, simple_2d_image) -> None:
        """Test that unused parameters trigger warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generate_seeds(simple_2d_image, method="gaussian", nonexistent_param=123)
            assert len(w) == 1
            assert "not used by any selected method" in str(w[0].message)


class TestErrorCases:
    """Test error handling."""

    def test_invalid_method_raises(self, simple_2d_image) -> None:
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid method"):
            generate_seeds(simple_2d_image, method="invalid_method")

    def test_invalid_combined_method_raises(self, simple_2d_image) -> None:
        """Test invalid combined method raises error."""
        with pytest.raises(ValueError, match="Invalid method"):
            generate_seeds(simple_2d_image, method="gaussian,invalid")

    def test_too_many_methods_raises(self, simple_2d_image) -> None:
        """Test too many methods raises error."""
        with pytest.raises(ValueError, match="Expected single method or two"):
            generate_seeds(simple_2d_image, method="gaussian,decomposition,both")

    def test_empty_image_raises(self) -> None:
        """Test empty image raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            generate_seeds(np.array([]), method="gaussian")

    def test_scalar_raises(self) -> None:
        """Test scalar raises error."""
        with pytest.raises(ValueError, match="must have at least 1 dimension"):
            generate_seeds(np.array(5.0), method="gaussian")


class TestOutputFormat:
    """Test output format consistency."""

    def test_returns_gsplatdata(self, simple_2d_image) -> None:
        """Test that all methods return GSplatData."""
        for method in ["gaussian", "decomposition", "both"]:
            result = generate_seeds(simple_2d_image, method=method)
            assert isinstance(result, GSplatData)

    def test_correct_shape_2d(self, simple_2d_image) -> None:
        """Test correct shape for 2D images."""
        for method in ["gaussian", "decomposition", "both"]:
            result = generate_seeds(simple_2d_image, method=method)
            validate_gsplatdata(result, 2, simple_2d_image.shape)

    def test_correct_shape_3d(self, simple_3d_image) -> None:
        """Test correct shape for 3D images."""
        for method in ["gaussian", "decomposition", "both"]:
            result = generate_seeds(simple_3d_image, method=method)
            validate_gsplatdata(result, 3, simple_3d_image.shape)

    def test_cholesky_factors_positive_diagonal(self, simple_2d_image) -> None:
        """Test that Cholesky diagonal elements are positive (valid sigma)."""
        for method in ["gaussian", "decomposition"]:
            result = generate_seeds(simple_2d_image, method=method)
            if len(result.centers) > 0:
                # First diagonal element (L00 = sigma for isotropic)
                assert np.all(result.cholesky_factors[:, 0] > 0), (
                    f"Method {method}: Diagonal Cholesky elements should be positive"
                )


class TestIntegration:
    """Test integration across methods."""

    def test_combined_produces_more_seeds(self, simple_2d_image) -> None:
        """Test that combining methods produces more seeds."""
        result_g = generate_seeds(simple_2d_image, method="gaussian", min_distance=2.0)
        result_d = generate_seeds(
            simple_2d_image, method="decomposition", min_distance=2.0
        )
        result_both = generate_seeds(simple_2d_image, method="both", min_distance=2.0)

        max_individual = max(len(result_g.centers), len(result_d.centers))
        assert len(result_both.centers) >= max_individual

    def test_min_distance_deduplicates_combined(self, simple_2d_image) -> None:
        """Test min_distance deduplicates combined results."""
        result_small = generate_seeds(simple_2d_image, method="both", min_distance=1.0)
        result_large = generate_seeds(simple_2d_image, method="both", min_distance=5.0)

        assert len(result_large.centers) <= len(result_small.centers)

    def test_1d_images(self) -> None:
        """Test 1D images."""
        x = np.linspace(-3, 3, 101)
        V = np.exp(-(x**2)) + 0.3 * np.exp(-((x - 1) ** 2))

        for method in ["gaussian", "decomposition", "both"]:
            result = generate_seeds(V, method=method)
            validate_gsplatdata(result, 1, V.shape)


class TestSpecialCases:
    """Test special and edge cases."""

    def test_uniform_image(self, uniform_image) -> None:
        """Test uniform image handling."""
        for method in ["gaussian", "decomposition", "both"]:
            result = generate_seeds(uniform_image, method=method)
            validate_gsplatdata(result, 2, uniform_image.shape)

    def test_noisy_image(self, noisy_image) -> None:
        """Test noisy image handling."""
        for method in ["gaussian", "decomposition", "both"]:
            result = generate_seeds(noisy_image, method=method)
            validate_gsplatdata(result, 2, noisy_image.shape)

    def test_small_image(self) -> None:
        """Test small image."""
        small_img = np.random.rand(8, 8)
        for method in ["gaussian", "decomposition", "both"]:
            result = generate_seeds(small_img, method=method, scales=[1, 2])
            validate_gsplatdata(result, 2, small_img.shape)

    def test_negative_values(self) -> None:
        """Test image with negative values."""
        x, y = np.meshgrid(np.linspace(-5, 5, 51), np.linspace(-5, 5, 51))
        V = np.exp(-(x**2 + y**2) / 4) - 0.5

        for method in ["gaussian", "decomposition", "both"]:
            result = generate_seeds(V, method=method)
            validate_gsplatdata(result, 2, V.shape)


class TestReproducibility:
    """Test reproducibility."""

    def test_reproducible_results(self, simple_2d_image) -> None:
        """Test results are reproducible."""
        result1 = generate_seeds(simple_2d_image, method="gaussian")
        result2 = generate_seeds(simple_2d_image, method="gaussian")

        np.testing.assert_array_equal(result1.centers, result2.centers)
        np.testing.assert_array_equal(result1.amplitudes, result2.amplitudes)
        np.testing.assert_array_equal(
            result1.cholesky_factors, result2.cholesky_factors
        )

    def test_default_params_work(self, simple_2d_image) -> None:
        """Test default parameters work."""
        result = generate_seeds(simple_2d_image)
        validate_gsplatdata(result, 2, simple_2d_image.shape)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
