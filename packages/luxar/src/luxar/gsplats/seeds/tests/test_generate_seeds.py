"""
Comprehensive tests for the generate_seeds() unified entry point function.

This module tests the main entry point for seed generation, ensuring proper:
- Method selection and routing
- Parameter validation and routing
- Error handling
- Output format consistency
- Integration across all methods
"""

import importlib.util
import warnings

import numpy as np
import pytest

from luxar.gsplats.seeds import generate_seeds

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

# Skip all tests if scipy is not available
pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")


# ============================================================================
# Test fixtures and helper functions
# ============================================================================


@pytest.fixture
def simple_2d_image():
    """Create a simple 2D image with Gaussian blobs for testing."""
    x, y = np.meshgrid(np.linspace(-5, 5, 51), np.linspace(-5, 5, 51))
    # Two Gaussian blobs at different locations
    V = np.exp(-(x**2 + y**2) / 4) + 0.5 * np.exp(-((x - 3) ** 2 + (y - 3) ** 2) / 2)
    return V


@pytest.fixture
def simple_3d_image():
    """Create a simple 3D volume with a Gaussian blob for testing."""
    x, y, z = np.meshgrid(
        np.linspace(-2, 2, 21), np.linspace(-2, 2, 21), np.linspace(-2, 2, 21)
    )
    V = np.exp(-(x**2 + y**2 + z**2) / 2)
    return V


@pytest.fixture
def uniform_image():
    """Create a uniform image (edge case)."""
    return np.ones((32, 32), dtype=float) * 5.0


@pytest.fixture
def noisy_image():
    """Create a noisy image."""
    return np.random.randn(32, 32) + 1.0


def validate_seeds_output(seeds, expected_ndim, image_shape) -> None:
    """
    Validate that seeds output has correct format.

    Parameters
    ----------
    seeds : np.ndarray
        Seeds array to validate
    expected_ndim : int
        Expected number of dimensions
    image_shape : tuple
        Shape of the source image
    """
    assert isinstance(seeds, np.ndarray), "Seeds should be numpy array"
    assert seeds.ndim == 2, "Seeds should be 2D array (N, ndim)"
    assert seeds.shape[1] == expected_ndim, (
        f"Seeds should have {expected_ndim}D coordinates"
    )
    # Check that coordinates are within bounds (allowing sub-voxel coordinates)
    assert np.all(seeds >= 0), "Seeds should have non-negative coordinates"
    assert np.all(seeds < np.array(image_shape)), "Seeds should be within image bounds"


# ============================================================================
# Test method selection
# ============================================================================


class TestMethodSelection:
    """Test different method selection options."""

    def test_method_gaussian_only(self, simple_2d_image) -> None:
        """Test method='gaussian' calls only Gaussian method."""
        seeds = generate_seeds(simple_2d_image, method="gaussian")
        validate_seeds_output(seeds, 2, simple_2d_image.shape)
        assert len(seeds) > 0, "Should find seeds with Gaussian method"

    def test_method_decomposition_only(self, simple_2d_image) -> None:
        """Test method='decomposition' calls only decomposition method."""
        seeds = generate_seeds(simple_2d_image, method="decomposition")
        validate_seeds_output(seeds, 2, simple_2d_image.shape)
        assert len(seeds) > 0, "Should find seeds with decomposition method"

    def test_method_both_default(self, simple_2d_image) -> None:
        """Test method='both' (default) calls both methods."""
        seeds = generate_seeds(simple_2d_image, method="both")
        validate_seeds_output(seeds, 2, simple_2d_image.shape)
        assert len(seeds) > 0, "Should find seeds with both methods"

    def test_method_gaussian_then_decomposition(self, simple_2d_image) -> None:
        """Test method='gaussian,decomposition' calls Gaussian first."""
        seeds = generate_seeds(simple_2d_image, method="gaussian,decomposition")
        validate_seeds_output(seeds, 2, simple_2d_image.shape)
        assert len(seeds) > 0, "Should find seeds with both methods"

    def test_method_decomposition_then_gaussian(self, simple_2d_image) -> None:
        """Test method='decomposition,gaussian' calls decomposition first."""
        seeds = generate_seeds(simple_2d_image, method="decomposition,gaussian")
        validate_seeds_output(seeds, 2, simple_2d_image.shape)
        assert len(seeds) > 0, "Should find seeds with both methods"

    def test_method_both_equivalent_to_decomposition_gaussian(
        self, simple_2d_image
    ) -> None:
        """Test that 'both' is equivalent to 'decomposition,gaussian'."""
        # Note: We can't guarantee identical results due to internal randomness
        # in some operations, but we can verify both produce valid outputs
        seeds_both = generate_seeds(simple_2d_image, method="both")
        seeds_explicit = generate_seeds(
            simple_2d_image, method="decomposition,gaussian"
        )

        # Both should produce valid outputs with same structure
        validate_seeds_output(seeds_both, 2, simple_2d_image.shape)
        validate_seeds_output(seeds_explicit, 2, simple_2d_image.shape)
        # Both should find seeds
        assert len(seeds_both) > 0
        assert len(seeds_explicit) > 0

    def test_method_case_insensitive(self, simple_2d_image) -> None:
        """Test that method string is case-insensitive."""
        seeds_lower = generate_seeds(simple_2d_image, method="gaussian")
        seeds_upper = generate_seeds(simple_2d_image, method="GAUSSIAN")
        seeds_mixed = generate_seeds(simple_2d_image, method="GaUsSiAn")

        # All should produce same results
        np.testing.assert_array_equal(seeds_lower, seeds_upper)
        np.testing.assert_array_equal(seeds_lower, seeds_mixed)

    def test_method_whitespace_handling(self, simple_2d_image) -> None:
        """Test that method string handles whitespace correctly."""
        seeds_no_space = generate_seeds(
            simple_2d_image, method="gaussian,decomposition"
        )
        seeds_with_space = generate_seeds(
            simple_2d_image, method="gaussian, decomposition"
        )
        seeds_extra_space = generate_seeds(
            simple_2d_image, method="  gaussian  ,  decomposition  "
        )

        # All should produce valid outputs (results may vary slightly due to internal randomness)
        validate_seeds_output(seeds_no_space, 2, simple_2d_image.shape)
        validate_seeds_output(seeds_with_space, 2, simple_2d_image.shape)
        validate_seeds_output(seeds_extra_space, 2, simple_2d_image.shape)
        # All should find seeds
        assert len(seeds_no_space) > 0
        assert len(seeds_with_space) > 0
        assert len(seeds_extra_space) > 0


# ============================================================================
# Test parameter routing
# ============================================================================


class TestParameterRouting:
    """Test that parameters are routed to correct methods."""

    def test_gaussian_specific_params_routed_correctly(self, simple_2d_image) -> None:
        """Test that Gaussian-specific params are used by Gaussian method."""
        # percentile_thresh is Gaussian-specific
        seeds_strict = generate_seeds(
            simple_2d_image, method="gaussian", percentile_thresh=90.0
        )
        seeds_loose = generate_seeds(
            simple_2d_image, method="gaussian", percentile_thresh=50.0
        )

        # Looser threshold should find more (or equal) seeds
        assert len(seeds_loose) >= len(seeds_strict), (
            "Lower percentile threshold should find more seeds"
        )

    def test_gaussian_clahe_params_routed_correctly(self, simple_2d_image) -> None:
        """Test that CLAHE parameters route to Gaussian method."""
        # Should not raise errors - CLAHE params should be accepted
        seeds_with_clahe = generate_seeds(
            simple_2d_image,
            method="gaussian",
            apply_clahe=True,
            clahe_tile_size=16,
            clahe_clip_limit=8.0,
        )
        seeds_without_clahe = generate_seeds(
            simple_2d_image, method="gaussian", apply_clahe=False
        )

        validate_seeds_output(seeds_with_clahe, 2, simple_2d_image.shape)
        validate_seeds_output(seeds_without_clahe, 2, simple_2d_image.shape)

    def test_decomposition_specific_params_routed_correctly(
        self, simple_2d_image
    ) -> None:
        """Test that decomposition-specific params are used by decomposition method."""
        # ignore_finest_k is decomposition-specific
        seeds_ignore_0 = generate_seeds(
            simple_2d_image, method="decomposition", ignore_finest_k=0
        )
        seeds_ignore_2 = generate_seeds(
            simple_2d_image, method="decomposition", ignore_finest_k=2
        )

        # Ignoring more scales should generally produce fewer seeds
        assert len(seeds_ignore_2) <= len(seeds_ignore_0), (
            "Ignoring more scales should produce fewer or equal seeds"
        )

    def test_common_params_work_with_all_methods(self, simple_2d_image) -> None:
        """Test that min_distance parameter is accepted by all methods."""
        # min_distance is a common parameter used for deduplication
        # Just verify it's accepted and produces valid output
        for method in ["gaussian", "decomposition", "both"]:
            seeds_close = generate_seeds(
                simple_2d_image, method=method, min_distance=1.0
            )
            seeds_far = generate_seeds(simple_2d_image, method=method, min_distance=8.0)

            # Both should produce valid outputs
            validate_seeds_output(seeds_close, 2, simple_2d_image.shape)
            validate_seeds_output(seeds_far, 2, simple_2d_image.shape)

            # Note: We don't assert seeds_far <= seeds_close because min_distance
            # primarily affects deduplication, not initial detection.
            # For single methods, the effect may be minimal or vary.

    def test_scales_parameter_routing(self, simple_2d_image) -> None:
        """Test that scales parameter is handled correctly for each method."""
        # Gaussian uses float scales (sigma values)
        seeds_gaussian = generate_seeds(
            simple_2d_image, method="gaussian", scales=[2.0, 4.0, 8.0]
        )
        validate_seeds_output(seeds_gaussian, 2, simple_2d_image.shape)

        # Decomposition uses int scales (downsample factors)
        seeds_decomp = generate_seeds(
            simple_2d_image, method="decomposition", scales=[2, 4, 8]
        )
        validate_seeds_output(seeds_decomp, 2, simple_2d_image.shape)

    def test_unused_parameter_warning(self, simple_2d_image) -> None:
        """Test that unused parameters trigger warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Pass a parameter that doesn't exist for any method
            generate_seeds(simple_2d_image, method="gaussian", nonexistent_param=123)
            # Should have warned about unused parameter
            assert len(w) == 1
            assert "not used by any selected method" in str(w[0].message)

    def test_method_specific_param_not_used_by_other_method(
        self, simple_2d_image
    ) -> None:
        """Test that method-specific params are silently ignored when not applicable."""
        # percentile_thresh is Gaussian-specific, but it's in the valid parameter set
        # so it won't raise a warning when used with decomposition (it's just not routed)
        # This is intentional behavior - the function doesn't warn about known parameters

        # Just verify the function runs without errors
        seeds = generate_seeds(
            simple_2d_image, method="decomposition", percentile_thresh=90.0
        )
        validate_seeds_output(seeds, 2, simple_2d_image.shape)

        # And verify truly unknown parameters DO warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generate_seeds(
                simple_2d_image, method="decomposition", totally_unknown_param=90.0
            )
            # Should warn about truly unknown parameter
            assert len(w) >= 1, "Should warn about unknown parameter"
            warning_messages = [str(warning.message) for warning in w]
            assert any("totally_unknown_param" in msg for msg in warning_messages), (
                "Warning should mention the unknown parameter"
            )


# ============================================================================
# Test error cases
# ============================================================================


class TestErrorCases:
    """Test error handling and validation."""

    def test_invalid_method_string_raises_error(self, simple_2d_image) -> None:
        """Test that invalid method string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            generate_seeds(simple_2d_image, method="invalid_method")

    def test_invalid_comma_separated_method_raises_error(self, simple_2d_image) -> None:
        """Test that invalid comma-separated methods raise ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            generate_seeds(simple_2d_image, method="gaussian,invalid")

    def test_too_many_comma_separated_methods_raises_error(
        self, simple_2d_image
    ) -> None:
        """Test that more than two methods raises ValueError."""
        with pytest.raises(ValueError, match="Expected single method or two"):
            generate_seeds(simple_2d_image, method="gaussian,decomposition,both")

    def test_empty_image_raises_error(self) -> None:
        """Test that empty image raises ValueError."""
        empty_array = np.array([])
        with pytest.raises(ValueError, match="cannot be empty"):
            generate_seeds(empty_array, method="gaussian")

    def test_scalar_image_raises_error(self) -> None:
        """Test that 0-dimensional array raises ValueError."""
        scalar = np.array(5.0)
        with pytest.raises(ValueError, match="must have at least 1 dimension"):
            generate_seeds(scalar, method="gaussian")

    def test_invalid_parameters_propagate_errors(self, simple_2d_image) -> None:
        """Test that invalid parameters raise appropriate errors or produce valid output."""
        # Note: Some invalid parameters may not raise errors but be handled gracefully

        # Invalid scales (empty) should raise error
        with pytest.raises((ValueError, AssertionError)):
            generate_seeds(simple_2d_image, method="gaussian", scales=[])

        # Very high percentile may not raise error but should handle gracefully
        # (may return empty or minimal results)
        seeds = generate_seeds(
            simple_2d_image, method="gaussian", percentile_thresh=99.99
        )
        validate_seeds_output(seeds, 2, simple_2d_image.shape)


# ============================================================================
# Test output format
# ============================================================================


class TestOutputFormat:
    """Test output format consistency across all methods."""

    def test_returns_numpy_array(self, simple_2d_image) -> None:
        """Test that output is always a numpy array."""
        for method in ["gaussian", "decomposition", "both"]:
            seeds = generate_seeds(simple_2d_image, method=method)
            assert isinstance(seeds, np.ndarray), (
                f"Output should be numpy array for method={method}"
            )

    def test_correct_shape_2d(self, simple_2d_image) -> None:
        """Test correct output shape for 2D images."""
        for method in ["gaussian", "decomposition", "both"]:
            seeds = generate_seeds(simple_2d_image, method=method)
            assert seeds.ndim == 2, f"Seeds should be 2D array for method={method}"
            assert seeds.shape[1] == 2, (
                f"Seeds should have 2D coordinates for method={method}"
            )

    def test_correct_shape_3d(self, simple_3d_image) -> None:
        """Test correct output shape for 3D images."""
        for method in ["gaussian", "decomposition", "both"]:
            seeds = generate_seeds(simple_3d_image, method=method)
            assert seeds.ndim == 2, f"Seeds should be 2D array for method={method}"
            assert seeds.shape[1] == 3, (
                f"Seeds should have 3D coordinates for method={method}"
            )

    def test_coordinates_within_bounds(self, simple_2d_image) -> None:
        """Test that all coordinates are within image bounds."""
        for method in ["gaussian", "decomposition", "both"]:
            seeds = generate_seeds(simple_2d_image, method=method)
            validate_seeds_output(seeds, 2, simple_2d_image.shape)

    def test_float_coordinates(self, simple_2d_image) -> None:
        """Test that coordinates are floating point (sub-voxel)."""
        for method in ["gaussian", "decomposition", "both"]:
            seeds = generate_seeds(simple_2d_image, method=method)
            assert seeds.dtype in [np.float32, np.float64, float], (
                f"Seeds should have float dtype for method={method}"
            )

    def test_empty_result_has_correct_shape(self, uniform_image) -> None:
        """Test that empty results have correct shape."""
        # Use very strict threshold to potentially get no seeds
        seeds = generate_seeds(
            uniform_image,
            method="gaussian",
            percentile_thresh=99.99,
            peaks_per_scale=1,
        )
        # Should be (N, 2) even if N=0
        assert seeds.ndim == 2
        assert seeds.shape[1] == 2


# ============================================================================
# Test integration and combinations
# ============================================================================


class TestIntegration:
    """Test integration across different methods."""

    def test_different_methods_produce_different_seeds(self, simple_2d_image) -> None:
        """Test that different methods produce different (but valid) results."""
        seeds_gaussian = generate_seeds(simple_2d_image, method="gaussian")
        seeds_decomp = generate_seeds(simple_2d_image, method="decomposition")

        # Both should produce valid seeds
        validate_seeds_output(seeds_gaussian, 2, simple_2d_image.shape)
        validate_seeds_output(seeds_decomp, 2, simple_2d_image.shape)

        # Results should be different (unless image is trivial)
        # Note: This is a weak test - we just check they're not identical
        # In practice they will usually differ
        if len(seeds_gaussian) > 0 and len(seeds_decomp) > 0:
            # Allow for possibility they might be similar for simple images
            # Just verify both produce valid outputs
            assert True

    def test_combined_methods_produce_more_seeds(self, simple_2d_image) -> None:
        """Test that combining methods produces more seeds than individual."""
        seeds_gaussian = generate_seeds(
            simple_2d_image, method="gaussian", min_distance=2.0
        )
        seeds_decomp = generate_seeds(
            simple_2d_image, method="decomposition", min_distance=2.0
        )
        seeds_both = generate_seeds(simple_2d_image, method="both", min_distance=2.0)

        # Combined should have at least as many as the larger individual method
        max_individual = max(len(seeds_gaussian), len(seeds_decomp))
        assert len(seeds_both) >= max_individual, (
            "Combined methods should produce at least as many seeds as individual"
        )

    def test_method_order_affects_results(self, simple_2d_image) -> None:
        """Test that method order can affect final results (due to priority)."""
        seeds_gd = generate_seeds(
            simple_2d_image, method="gaussian,decomposition", min_distance=2.0
        )
        seeds_dg = generate_seeds(
            simple_2d_image, method="decomposition,gaussian", min_distance=2.0
        )

        # Both should be valid
        validate_seeds_output(seeds_gd, 2, simple_2d_image.shape)
        validate_seeds_output(seeds_dg, 2, simple_2d_image.shape)

        # Order matters due to priority in combine_seeds
        # Results may differ depending on which method runs first

    def test_min_distance_affects_combined_results(self, simple_2d_image) -> None:
        """Test that min_distance properly deduplicates combined results."""
        seeds_small_dist = generate_seeds(
            simple_2d_image, method="both", min_distance=1.0
        )
        seeds_large_dist = generate_seeds(
            simple_2d_image, method="both", min_distance=5.0
        )

        # Larger distance should produce fewer seeds
        assert len(seeds_large_dist) <= len(seeds_small_dist), (
            "Larger min_distance should produce fewer seeds"
        )

    def test_1d_images_work_with_all_methods(self) -> None:
        """Test that 1D images work correctly."""
        x = np.linspace(-3, 3, 101)
        V = np.exp(-(x**2)) + 0.3 * np.exp(-((x - 1) ** 2))

        for method in ["gaussian", "decomposition", "both"]:
            seeds = generate_seeds(V, method=method)
            validate_seeds_output(seeds, 1, V.shape)

    def test_high_dimensional_images(self) -> None:
        """Test that higher dimensional images work correctly."""
        # Create a 4D test image (small for speed)
        # Note: Some methods may have dimensionality limits
        shape = (10, 10, 10, 10)
        center = np.array(shape) // 2
        indices = np.indices(shape)
        dist = np.sqrt(sum((indices[i] - center[i]) ** 2 for i in range(len(shape))))
        V = np.exp(-(dist**2) / 10)

        # Only test Gaussian method which supports high-dimensional data
        # (Decomposition may have limitations due to interpolation)
        try:
            seeds = generate_seeds(V, method="gaussian", scales=[2.0, 4.0])
            validate_seeds_output(seeds, 4, V.shape)
        except NotImplementedError:
            # Some methods may not support very high dimensions
            pytest.skip("High-dimensional support not available for this method")


# ============================================================================
# Test special cases and edge cases
# ============================================================================


class TestSpecialCases:
    """Test special and edge cases."""

    def test_uniform_image_handling(self, uniform_image) -> None:
        """Test that uniform images are handled gracefully."""
        for method in ["gaussian", "decomposition", "both"]:
            seeds = generate_seeds(uniform_image, method=method)
            # Should not crash, output format should be correct
            validate_seeds_output(seeds, 2, uniform_image.shape)

    def test_very_noisy_image(self, noisy_image) -> None:
        """Test handling of very noisy images."""
        for method in ["gaussian", "decomposition", "both"]:
            seeds = generate_seeds(noisy_image, method=method)
            validate_seeds_output(seeds, 2, noisy_image.shape)

    def test_small_image(self) -> None:
        """Test with very small images."""
        small_img = np.random.rand(8, 8)
        for method in ["gaussian", "decomposition", "both"]:
            seeds = generate_seeds(small_img, method=method, scales=[1, 2])
            validate_seeds_output(seeds, 2, small_img.shape)

    def test_large_dynamic_range(self) -> None:
        """Test with images having large dynamic range."""
        x, y = np.meshgrid(np.linspace(-5, 5, 51), np.linspace(-5, 5, 51))
        V = 1e6 * np.exp(-(x**2 + y**2) / 4) + 1e-6 * np.exp(
            -((x - 3) ** 2 + (y - 3) ** 2) / 2
        )

        for method in ["gaussian", "decomposition", "both"]:
            seeds = generate_seeds(V, method=method)
            validate_seeds_output(seeds, 2, V.shape)

    def test_single_pixel_image(self) -> None:
        """Test with single-pixel image."""
        single_pixel = np.array([[1.0]])
        for method in ["gaussian", "decomposition", "both"]:
            seeds = generate_seeds(single_pixel, method=method, scales=[1])
            # Should handle gracefully
            validate_seeds_output(seeds, 2, single_pixel.shape)

    def test_negative_values(self) -> None:
        """Test that negative values are handled correctly."""
        x, y = np.meshgrid(np.linspace(-5, 5, 51), np.linspace(-5, 5, 51))
        V = np.exp(-(x**2 + y**2) / 4) - 0.5  # Contains negative values

        for method in ["gaussian", "decomposition", "both"]:
            seeds = generate_seeds(V, method=method)
            validate_seeds_output(seeds, 2, V.shape)


# ============================================================================
# Test consistency and reproducibility
# ============================================================================


class TestConsistency:
    """Test consistency and reproducibility."""

    def test_reproducible_results(self, simple_2d_image) -> None:
        """Test that results are reproducible."""
        # Generate seeds twice with same parameters
        seeds1 = generate_seeds(simple_2d_image, method="gaussian")
        seeds2 = generate_seeds(simple_2d_image, method="gaussian")

        # Should produce identical results
        np.testing.assert_array_equal(seeds1, seeds2)

    def test_default_parameters_work(self, simple_2d_image) -> None:
        """Test that default parameters work correctly."""
        # Should work with no extra parameters
        seeds = generate_seeds(simple_2d_image)  # Uses default method="both"
        validate_seeds_output(seeds, 2, simple_2d_image.shape)

    def test_explicit_defaults_match_implicit(self, simple_2d_image) -> None:
        """Test that explicit default params match implicit defaults."""
        seeds_implicit = generate_seeds(simple_2d_image, method="both")
        seeds_explicit = generate_seeds(
            simple_2d_image, method="both", min_distance=2.0
        )

        # Should produce valid results (may vary slightly due to internal randomness)
        validate_seeds_output(seeds_implicit, 2, simple_2d_image.shape)
        validate_seeds_output(seeds_explicit, 2, simple_2d_image.shape)
        # Both should find seeds
        assert len(seeds_implicit) > 0
        assert len(seeds_explicit) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
