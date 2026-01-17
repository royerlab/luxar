"""
Tests for the seed_from_grid() uniform grid seeding function.
"""

import importlib.util

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.seeds.grid import seed_from_grid

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")


@pytest.fixture
def simple_2d_image():
    """Create a simple 2D image with varying intensity."""
    x, y = np.meshgrid(np.linspace(0, 1, 51), np.linspace(0, 1, 51))
    return (x + y) / 2  # Gradient image


@pytest.fixture
def simple_3d_image():
    """Create a simple 3D volume."""
    x, y, z = np.meshgrid(
        np.linspace(0, 1, 21), np.linspace(0, 1, 21), np.linspace(0, 1, 21)
    )
    return (x + y + z) / 3


@pytest.fixture
def uniform_image():
    """Create a uniform intensity image."""
    return np.ones((50, 50), dtype=float) * 5.0


def validate_gsplatdata(result: GSplatData, expected_ndim: int) -> None:
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


class TestBasicFunctionality:
    """Test basic grid seeding functionality."""

    def test_basic_2d(self, simple_2d_image) -> None:
        """Test basic 2D grid generation."""
        result = seed_from_grid(simple_2d_image)
        validate_gsplatdata(result, 2)
        assert len(result.centers) > 0, "Should generate seeds"

    def test_basic_3d(self, simple_3d_image) -> None:
        """Test basic 3D grid generation."""
        result = seed_from_grid(simple_3d_image)
        validate_gsplatdata(result, 3)
        assert len(result.centers) > 0, "Should generate seeds"

    def test_1d_image(self) -> None:
        """Test 1D image."""
        V = np.linspace(0, 1, 101)
        result = seed_from_grid(V)
        validate_gsplatdata(result, 1)
        assert len(result.centers) > 0

    def test_returns_gsplatdata(self, simple_2d_image) -> None:
        """Test that result is GSplatData."""
        result = seed_from_grid(simple_2d_image)
        assert isinstance(result, GSplatData)

    def test_coordinates_within_bounds(self, simple_2d_image) -> None:
        """Test that all coordinates are within image bounds."""
        result = seed_from_grid(simple_2d_image)
        if len(result.centers) > 0:
            assert np.all(result.centers >= 0), "Coordinates should be non-negative"
            assert np.all(result.centers < np.array(simple_2d_image.shape)), (
                "Coordinates should be within bounds"
            )


class TestSpacing:
    """Test spacing parameter."""

    def test_custom_spacing_scalar(self, simple_2d_image) -> None:
        """Test custom scalar spacing."""
        result_small = seed_from_grid(simple_2d_image, spacing=5.0)
        result_large = seed_from_grid(simple_2d_image, spacing=15.0)

        # Smaller spacing should produce more seeds
        assert len(result_small.centers) > len(result_large.centers)

    def test_custom_spacing_per_dimension(self) -> None:
        """Test per-dimension spacing."""
        V = np.ones((100, 50))
        result = seed_from_grid(V, spacing=[10.0, 5.0])
        validate_gsplatdata(result, 2)
        assert len(result.centers) > 0

    def test_spacing_mismatch_raises(self) -> None:
        """Test that mismatched spacing dimensions raises error."""
        V = np.ones((50, 50))
        with pytest.raises(ValueError, match="elements but V has"):
            seed_from_grid(V, spacing=[5.0, 5.0, 5.0])

    def test_negative_spacing_raises(self) -> None:
        """Test that negative spacing raises error."""
        V = np.ones((50, 50))
        with pytest.raises(ValueError, match="positive"):
            seed_from_grid(V, spacing=-5.0)

    def test_zero_spacing_raises(self) -> None:
        """Test that zero spacing raises error."""
        V = np.ones((50, 50))
        with pytest.raises(ValueError, match="positive"):
            seed_from_grid(V, spacing=0.0)

    def test_auto_spacing(self, simple_2d_image) -> None:
        """Test that auto spacing works."""
        result = seed_from_grid(simple_2d_image, spacing=None)
        validate_gsplatdata(result, 2)
        # Auto spacing should produce a reasonable number of seeds
        assert 10 < len(result.centers) < 5000


class TestAnisotropicSpacing:
    """Test aspect-ratio-aware spacing (respects anisotropy)."""

    def test_anisotropic_spacing_thin_volume(self) -> None:
        """Test that thin volumes get appropriate spacing in thin dimension."""
        # Create thin volume: 100×100×10 (Z is thin)
        V = np.random.rand(100, 100, 10)

        result = seed_from_grid(V, spacing=None)
        validate_gsplatdata(result, 3)

        # Check that seeds span the full Z range
        z_coords = result.centers[:, 2]
        assert z_coords.min() < 2.0, "Seeds should be near Z=0"
        assert z_coords.max() > 8.0, "Seeds should be near Z=10"

        # Should have multiple Z levels (not all in one plane)
        unique_z_levels = len(np.unique(np.round(z_coords)))
        assert unique_z_levels >= 5, f"Should have >=5 Z levels, got {unique_z_levels}"

    def test_anisotropic_spacing_very_thin(self) -> None:
        """Test extreme anisotropy: 1000×1000×10."""
        V = np.random.rand(1000, 1000, 10)

        result = seed_from_grid(V, spacing=None)
        validate_gsplatdata(result, 3)

        # Z spacing should be much smaller than XY spacing
        # We can infer this from seed distribution
        z_coords = result.centers[:, 2]
        x_coords = result.centers[:, 0]

        # Should have seeds across Z range
        z_span = z_coords.max() - z_coords.min()
        assert z_span > 7.0, "Seeds should span most of Z dimension"

        # Should have seeds across X range
        x_span = x_coords.max() - x_coords.min()
        assert x_span > 900.0, "Seeds should span most of X dimension"

    def test_isotropic_image_unchanged(self) -> None:
        """Test that isotropic images still work correctly."""
        # Square/cube images should behave similarly to before
        V = np.random.rand(50, 50)

        result = seed_from_grid(V, spacing=None)
        validate_gsplatdata(result, 2)

        # Should produce reasonable seed count
        assert 10 < len(result.centers) < 1000

    def test_anisotropic_2d(self) -> None:
        """Test anisotropic spacing in 2D (wide image)."""
        # 200×20 image (wide and short)
        V = np.random.rand(200, 20)

        result = seed_from_grid(V, spacing=None)
        validate_gsplatdata(result, 2)

        # Seeds should span both dimensions
        y_coords = result.centers[:, 0]
        x_coords = result.centers[:, 1]

        assert y_coords.max() - y_coords.min() > 180, "Seeds should span height"
        assert x_coords.max() - x_coords.min() > 15, "Seeds should span width"


class TestJitter:
    """Test jitter parameter."""

    def test_no_jitter_reproducible(self, simple_2d_image) -> None:
        """Test that no jitter produces reproducible results."""
        result1 = seed_from_grid(simple_2d_image, spacing=10.0, jitter=0.0)
        result2 = seed_from_grid(simple_2d_image, spacing=10.0, jitter=0.0)
        np.testing.assert_array_equal(result1.centers, result2.centers)

    def test_jitter_reproducible_with_seed(self, simple_2d_image) -> None:
        """Test that jitter is reproducible (uses fixed seed internally)."""
        result1 = seed_from_grid(simple_2d_image, spacing=10.0, jitter=0.25)
        result2 = seed_from_grid(simple_2d_image, spacing=10.0, jitter=0.25)
        np.testing.assert_array_equal(result1.centers, result2.centers)

    def test_jitter_affects_positions(self, simple_2d_image) -> None:
        """Test that jitter changes positions."""
        result_no_jitter = seed_from_grid(simple_2d_image, spacing=10.0, jitter=0.0)
        result_jitter = seed_from_grid(simple_2d_image, spacing=10.0, jitter=0.25)

        # Positions should be different
        assert not np.allclose(result_no_jitter.centers, result_jitter.centers)

    def test_jitter_out_of_range_raises(self, simple_2d_image) -> None:
        """Test that invalid jitter raises error."""
        with pytest.raises(ValueError, match="jitter"):
            seed_from_grid(simple_2d_image, jitter=-0.1)
        with pytest.raises(ValueError, match="jitter"):
            seed_from_grid(simple_2d_image, jitter=0.6)

    def test_jitter_stays_within_bounds(self, simple_2d_image) -> None:
        """Test that jittered coordinates stay within bounds."""
        result = seed_from_grid(simple_2d_image, spacing=5.0, jitter=0.5)
        if len(result.centers) > 0:
            assert np.all(result.centers >= 0)
            assert np.all(result.centers < np.array(simple_2d_image.shape))


class TestSigma:
    """Test sigma parameter.

    Grid seeding uses σ = spacing/2 by default so that splats cover the image
    with ~60% overlap at midpoints between grid points. This ensures good
    initial coverage while letting the optimizer refine shapes as needed.
    """

    def test_uses_specified_sigma(self, simple_2d_image) -> None:
        """Test that custom sigma is used when specified."""
        result = seed_from_grid(simple_2d_image, sigma=3.0)
        validate_gsplatdata(result, 2)

        if len(result.centers) > 0:
            # Custom sigma should be used
            # For isotropic 2D, diagonal elements should be the sigma value
            # Packed format: [L00, L10, L11] -> L00 and L11 are diagonal
            assert np.allclose(result.cholesky_factors[:, 0], 3.0)

    def test_negative_sigma_raises(self, simple_2d_image) -> None:
        """Test that negative sigma raises error."""
        with pytest.raises(ValueError, match="positive"):
            seed_from_grid(simple_2d_image, sigma=-1.0)

    def test_auto_sigma_uses_half_spacing(self, simple_2d_image) -> None:
        """Test that auto sigma uses spacing/2 for coverage."""
        spacing = 10.0
        result = seed_from_grid(simple_2d_image, spacing=spacing, sigma=None)

        if len(result.centers) > 0:
            # Auto sigma = spacing / 2 for coverage
            expected_sigma = spacing / 2.0
            assert np.allclose(result.cholesky_factors[:, 0], expected_sigma)


class TestIntensityFiltering:
    """Test intensity threshold parameters."""

    def test_exclude_below_absolute(self) -> None:
        """Test absolute threshold filtering."""
        # Create image with clear high/low regions
        V = np.zeros((50, 50))
        V[:25, :] = 1.0  # High region
        V[25:, :] = 0.1  # Low region

        result_all = seed_from_grid(V, spacing=10.0, exclude_below=None)
        result_filtered = seed_from_grid(V, spacing=10.0, exclude_below=0.5)

        # Filtered should have fewer seeds
        assert len(result_filtered.centers) < len(result_all.centers)

        # Filtered seeds should be in high region
        if len(result_filtered.centers) > 0:
            assert np.all(result_filtered.centers[:, 0] < 25)

    def test_exclude_below_percentile(self, simple_2d_image) -> None:
        """Test percentile threshold filtering."""
        result_all = seed_from_grid(simple_2d_image, spacing=10.0)
        result_filtered = seed_from_grid(
            simple_2d_image, spacing=10.0, exclude_below_percentile=50.0
        )

        # Filtered should have fewer seeds
        assert len(result_filtered.centers) < len(result_all.centers)

    def test_mutual_exclusivity_of_thresholds(self, simple_2d_image) -> None:
        """Test that both thresholds cannot be set."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            seed_from_grid(
                simple_2d_image, exclude_below=0.5, exclude_below_percentile=50.0
            )


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_image_raises(self) -> None:
        """Test empty image raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            seed_from_grid(np.array([]))

    def test_scalar_raises(self) -> None:
        """Test scalar raises error."""
        with pytest.raises(ValueError, match="at least 1 dimension"):
            seed_from_grid(np.array(5.0))

    def test_small_image(self) -> None:
        """Test very small image."""
        V = np.ones((5, 5))
        result = seed_from_grid(V, spacing=10.0)
        validate_gsplatdata(result, 2)
        # May have few or no seeds due to spacing

    def test_large_spacing_no_seeds(self) -> None:
        """Test that large spacing may produce no seeds."""
        V = np.ones((10, 10))
        result = seed_from_grid(V, spacing=100.0)
        validate_gsplatdata(result, 2)
        # Should handle gracefully (may have 0 or few seeds)

    def test_all_filtered_out(self) -> None:
        """Test when all seeds are filtered out."""
        V = np.ones((50, 50)) * 0.1
        result = seed_from_grid(V, spacing=10.0, exclude_below=1.0)
        validate_gsplatdata(result, 2)
        assert len(result.centers) == 0


class TestOutputFormat:
    """Test output format consistency."""

    def test_dtype_float32(self, simple_2d_image) -> None:
        """Test that outputs are float32."""
        result = seed_from_grid(simple_2d_image)
        assert result.centers.dtype == np.float32
        assert result.amplitudes.dtype == np.float32
        assert result.cholesky_factors.dtype == np.float32
        assert result.sharpnesses.dtype == np.float32

    def test_sharpness_value(self, simple_2d_image) -> None:
        """Test that sharpness is 2.0 (standard Gaussian)."""
        result = seed_from_grid(simple_2d_image)
        if len(result.sharpnesses) > 0:
            assert np.all(result.sharpnesses == 2.0)

    def test_positive_amplitudes(self, simple_2d_image) -> None:
        """Test that amplitudes match sampled values."""
        result = seed_from_grid(simple_2d_image)
        if len(result.amplitudes) > 0:
            # For gradient image, amplitudes should be positive
            assert np.all(result.amplitudes >= 0)

    def test_isotropic_cholesky(self, simple_2d_image) -> None:
        """Test that Cholesky factors are isotropic (diagonal)."""
        spacing = 10.0
        result = seed_from_grid(simple_2d_image, spacing=spacing)
        if len(result.centers) > 0:
            # 2D packed format: [L00, L10, L11]
            # Off-diagonal L10 should be 0
            assert np.allclose(result.cholesky_factors[:, 1], 0.0)
            # Diagonal elements should be equal (isotropic)
            assert np.allclose(
                result.cholesky_factors[:, 0], result.cholesky_factors[:, 2]
            )
            # Sigma should be spacing / 2 for coverage
            expected_sigma = spacing / 2.0
            assert np.allclose(result.cholesky_factors[:, 0], expected_sigma)


class TestReproducibility:
    """Test reproducibility."""

    def test_deterministic_output(self, simple_2d_image) -> None:
        """Test that outputs are deterministic."""
        result1 = seed_from_grid(simple_2d_image, spacing=10.0)
        result2 = seed_from_grid(simple_2d_image, spacing=10.0)

        np.testing.assert_array_equal(result1.centers, result2.centers)
        np.testing.assert_array_equal(result1.amplitudes, result2.amplitudes)
        np.testing.assert_array_equal(
            result1.cholesky_factors, result2.cholesky_factors
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
