"""
Tests for the generate_seeds() unified entry point function.

All seeding methods now return GSplatData with scale-informed Gaussian shapes.
"""

import importlib
import importlib.util
import sys
import warnings

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
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
def image_with_edges():
    """Create a 2D image with clear edges."""
    V = np.zeros((50, 50), dtype=float)
    V[15:35, 15:35] = 1.0  # Square
    return V


@pytest.fixture
def uniform_image():
    """Create a uniform image."""
    return np.ones((32, 32), dtype=float) * 5.0


@pytest.fixture
def noisy_image():
    """Create a noisy image."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((32, 32)) + 1.0


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

    def test_method_decomposition(self, simple_2d_image) -> None:
        """Test method='decomposition'."""
        result = generate_seeds(simple_2d_image, method="decomposition")
        validate_gsplatdata(result, 2, simple_2d_image.shape)
        assert len(result.centers) > 0

    def test_method_grid(self, simple_2d_image) -> None:
        """Test method='grid'."""
        result = generate_seeds(simple_2d_image, method="grid")
        validate_gsplatdata(result, 2, simple_2d_image.shape)
        assert len(result.centers) > 0

    def test_method_edges(self, image_with_edges) -> None:
        """Test method='edges'."""
        result = generate_seeds(image_with_edges, method="edges")
        validate_gsplatdata(result, 2, image_with_edges.shape)
        assert len(result.centers) > 0

    def test_method_auto(self, simple_2d_image) -> None:
        """Test method='auto' (default) combines all methods."""
        result = generate_seeds(simple_2d_image, method="auto")
        validate_gsplatdata(result, 2, simple_2d_image.shape)
        assert len(result.centers) > 0

    def test_default_is_auto(self, simple_2d_image) -> None:
        """Test that default method is 'auto'."""
        result = generate_seeds(simple_2d_image)
        validate_gsplatdata(result, 2, simple_2d_image.shape)
        assert len(result.centers) > 0

    def test_method_combined(self, simple_2d_image) -> None:
        """Test method='decomposition,grid'."""
        result = generate_seeds(simple_2d_image, method="decomposition,grid")
        validate_gsplatdata(result, 2, simple_2d_image.shape)
        assert len(result.centers) > 0

    def test_method_case_insensitive(self, simple_2d_image) -> None:
        """Test that method string is case-insensitive."""
        # Use grid method since it's deterministic (decomposition has some variability)
        result_lower = generate_seeds(simple_2d_image, method="grid")
        result_upper = generate_seeds(simple_2d_image, method="GRID")

        # Both should be valid GSplatData with same number of seeds
        validate_gsplatdata(result_lower, 2, simple_2d_image.shape)
        validate_gsplatdata(result_upper, 2, simple_2d_image.shape)
        assert len(result_lower.centers) == len(result_upper.centers)


class TestParameterRouting:
    """Test parameter routing to methods."""

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

    def test_spacing_affects_grid(self, simple_2d_image) -> None:
        """Test spacing parameter for grid method."""
        result_small = generate_seeds(simple_2d_image, method="grid", spacing=5.0)
        result_large = generate_seeds(simple_2d_image, method="grid", spacing=15.0)

        # Smaller spacing should produce more seeds
        assert len(result_small.centers) > len(result_large.centers)

    def test_edge_threshold_affects_edges(self, image_with_edges) -> None:
        """Test edge_threshold_rel parameter for edges method."""
        result_low = generate_seeds(
            image_with_edges, method="edges", edge_threshold_rel=0.05
        )
        result_high = generate_seeds(
            image_with_edges, method="edges", edge_threshold_rel=0.5
        )

        # Lower threshold should find more or equal seeds
        assert len(result_low.centers) >= len(result_high.centers)

    def test_min_distance_common_param(self, simple_2d_image) -> None:
        """Test min_distance parameter."""
        for method in ["decomposition", "grid", "auto"]:
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
            generate_seeds(simple_2d_image, method="grid", nonexistent_param=123)
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
            generate_seeds(simple_2d_image, method="decomposition,invalid")

    def test_three_methods_comma_separated(self, image_with_edges) -> None:
        """Test three methods can be combined via comma-separation."""
        result = generate_seeds(image_with_edges, method="decomposition,grid,edges")
        validate_gsplatdata(result, 2, image_with_edges.shape)
        assert len(result.centers) > 0

    def test_empty_image_raises(self) -> None:
        """Test empty image raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            generate_seeds(np.array([]), method="decomposition")

    def test_scalar_raises(self) -> None:
        """Test scalar raises error."""
        with pytest.raises(ValueError, match="must have at least 1 dimension"):
            generate_seeds(np.array(5.0), method="decomposition")


class TestOutputFormat:
    """Test output format consistency."""

    def test_returns_gsplatdata(self, simple_2d_image) -> None:
        """Test that all methods return GSplatData."""
        for method in ["decomposition", "grid", "auto"]:
            result = generate_seeds(simple_2d_image, method=method)
            assert isinstance(result, GSplatData)

    def test_correct_shape_2d(self, simple_2d_image) -> None:
        """Test correct shape for 2D images."""
        for method in ["decomposition", "grid", "auto"]:
            result = generate_seeds(simple_2d_image, method=method)
            validate_gsplatdata(result, 2, simple_2d_image.shape)

    def test_correct_shape_3d(self, simple_3d_image) -> None:
        """Test correct shape for 3D images."""
        for method in ["decomposition", "grid", "auto"]:
            result = generate_seeds(simple_3d_image, method=method)
            validate_gsplatdata(result, 3, simple_3d_image.shape)

    def test_cholesky_factors_positive_diagonal(self, simple_2d_image) -> None:
        """Test that Cholesky diagonal elements are positive (valid sigma)."""
        for method in ["decomposition", "grid"]:
            result = generate_seeds(simple_2d_image, method=method)
            if len(result.centers) > 0:
                # First diagonal element (L00 = sigma for isotropic)
                assert np.all(result.cholesky_factors[:, 0] > 0), (
                    f"Method {method}: Diagonal Cholesky elements should be positive"
                )


class TestIntegration:
    """Test integration across methods."""

    def test_auto_produces_seeds(self, simple_2d_image) -> None:
        """Test that auto mode produces seeds."""
        result = generate_seeds(simple_2d_image, method="auto", min_distance=2.0)
        validate_gsplatdata(result, 2, simple_2d_image.shape)
        assert len(result.centers) > 0

    def test_combined_produces_more_seeds(self, simple_2d_image) -> None:
        """Test that combining methods can produce more seeds."""
        result_d = generate_seeds(
            simple_2d_image, method="decomposition", min_distance=2.0
        )
        result_g = generate_seeds(simple_2d_image, method="grid", min_distance=2.0)
        result_combined = generate_seeds(
            simple_2d_image, method="decomposition,grid", min_distance=2.0
        )

        max_individual = max(len(result_d.centers), len(result_g.centers))
        # Combined should have roughly as many as the larger individual
        # (may be slightly less due to deduplication variability)
        assert len(result_combined.centers) >= max_individual * 0.9

    def test_min_distance_deduplicates_combined(self, simple_2d_image) -> None:
        """Test min_distance deduplicates combined results."""
        result_small = generate_seeds(simple_2d_image, method="auto", min_distance=1.0)
        result_large = generate_seeds(simple_2d_image, method="auto", min_distance=5.0)

        assert len(result_large.centers) <= len(result_small.centers)

    def test_1d_images(self) -> None:
        """Test 1D images."""
        x = np.linspace(-3, 3, 101)
        V = np.exp(-(x**2)) + 0.3 * np.exp(-((x - 1) ** 2))

        for method in ["decomposition", "grid", "auto"]:
            result = generate_seeds(V, method=method)
            validate_gsplatdata(result, 1, V.shape)


class TestAutoMode:
    """Test auto mode specifically."""

    def test_auto_budget_target_seeds(self, simple_2d_image) -> None:
        """Test target_seeds parameter in auto mode."""
        result_50 = generate_seeds(
            simple_2d_image, method="auto", target_seeds=50, min_distance=1.0
        )
        result_500 = generate_seeds(
            simple_2d_image, method="auto", target_seeds=500, min_distance=1.0
        )

        # More target seeds should generally produce more output
        # (though not exactly due to deduplication)
        validate_gsplatdata(result_50, 2, simple_2d_image.shape)
        validate_gsplatdata(result_500, 2, simple_2d_image.shape)

    def test_auto_with_edges(self, image_with_edges) -> None:
        """Test auto mode includes edge seeds when edges exist."""
        result = generate_seeds(image_with_edges, method="auto")
        validate_gsplatdata(result, 2, image_with_edges.shape)
        assert len(result.centers) > 0


class TestSpecialCases:
    """Test special and edge cases."""

    def test_uniform_image(self, uniform_image) -> None:
        """Test uniform image handling."""
        for method in ["decomposition", "grid", "auto"]:
            result = generate_seeds(uniform_image, method=method)
            validate_gsplatdata(result, 2, uniform_image.shape)

    def test_noisy_image(self, noisy_image) -> None:
        """Test noisy image handling."""
        for method in ["decomposition", "grid", "auto"]:
            result = generate_seeds(noisy_image, method=method)
            validate_gsplatdata(result, 2, noisy_image.shape)

    def test_small_image(self) -> None:
        """Test small image."""
        small_img = np.random.rand(8, 8)
        for method in ["decomposition", "grid", "auto"]:
            result = generate_seeds(small_img, method=method, scales=[1, 2])
            validate_gsplatdata(result, 2, small_img.shape)

    def test_negative_values(self) -> None:
        """Test image with negative values."""
        x, y = np.meshgrid(np.linspace(-5, 5, 51), np.linspace(-5, 5, 51))
        V = np.exp(-(x**2 + y**2) / 4) - 0.5

        for method in ["decomposition", "grid", "auto"]:
            result = generate_seeds(V, method=method)
            validate_gsplatdata(result, 2, V.shape)


class TestReproducibility:
    """Test reproducibility."""

    def test_reproducible_results(self, simple_2d_image) -> None:
        """Test results are reproducible for deterministic methods."""
        # Use grid method since it's fully deterministic
        # (decomposition has some variability due to iterative optimization)
        result1 = generate_seeds(simple_2d_image, method="grid")
        result2 = generate_seeds(simple_2d_image, method="grid")

        np.testing.assert_array_equal(result1.centers, result2.centers)
        np.testing.assert_array_equal(result1.amplitudes, result2.amplitudes)
        np.testing.assert_array_equal(
            result1.cholesky_factors, result2.cholesky_factors
        )

    def test_default_params_work(self, simple_2d_image) -> None:
        """Test default parameters work."""
        result = generate_seeds(simple_2d_image)
        validate_gsplatdata(result, 2, simple_2d_image.shape)


class TestEdgesImportNotSwallowed:
    """The `edges` seeder is a real module imported eagerly like the other
    three; the old lazy 'edges.py may not exist yet' guard that turned a
    genuine ImportError into a warning plus silently-missing seeds is gone."""

    def test_edges_imported_at_module_level(self) -> None:
        """generate.py binds the real seed_from_edges at import time."""
        from luxar.gsplats.seeds import generate as gen
        from luxar.gsplats.seeds.edges import seed_from_edges

        assert gen.seed_from_edges is seed_from_edges

    def test_edges_import_error_propagates(self, monkeypatch) -> None:
        """A genuine ImportError from inside edges.py must propagate, not be
        swallowed into a warning + silently-missing seeds."""
        from luxar.gsplats.seeds import generate as gen

        # A None entry in sys.modules makes importing edges raise ImportError,
        # standing in for a broken transitive import / missing optional dep.
        monkeypatch.setitem(sys.modules, "luxar.gsplats.seeds.edges", None)
        try:
            with pytest.raises(ImportError):
                importlib.reload(gen)
        finally:
            monkeypatch.undo()
            importlib.reload(gen)
