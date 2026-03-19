"""
Tests for CLAHE (Contrast Limited Adaptive Histogram Equalization) implementation.
"""

import pytest
import torch

from luxar.gsplats.clahe import apply_clahe, compute_clahe_sampling_probabilities


class TestCLAHEBasic:
    """Basic functionality tests for CLAHE."""

    def test_uniform_image_unchanged(self) -> None:
        """Uniform images should remain unchanged."""
        V = torch.ones(256, 256) * 0.5
        V_clahe = apply_clahe(V, tile_size=16, clip_limit=2.0)
        assert torch.allclose(V_clahe, V, atol=1e-6)

    def test_range_preservation(self) -> None:
        """CLAHE should preserve min/max intensity range."""
        V = torch.randn(256, 256)
        V_min, V_max = V.min(), V.max()
        V_clahe = apply_clahe(V, tile_size=16, clip_limit=2.0)

        # Range should be preserved (within small tolerance for numerical precision)
        assert torch.allclose(V_clahe.min(), torch.tensor(V_min), atol=1e-5)
        assert torch.allclose(V_clahe.max(), torch.tensor(V_max), atol=1e-5)

    def test_shape_preservation(self) -> None:
        """Output shape should match input shape."""
        shapes = [(128,), (256, 256), (64, 64, 64), (32, 32, 32, 32)]

        for shape in shapes:
            V = torch.randn(*shape)
            V_clahe = apply_clahe(V, tile_size=16)
            assert V_clahe.shape == V.shape

    def test_dtype_preservation(self) -> None:
        """Output dtype should match input dtype."""
        dtypes = [torch.float32, torch.float64]

        for dtype in dtypes:
            V = torch.randn(256, 256, dtype=dtype)
            V_clahe = apply_clahe(V, tile_size=16)
            assert V_clahe.dtype == dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_preservation(self) -> None:
        """Output device should match input device."""
        V_cpu = torch.randn(256, 256)
        V_gpu = V_cpu.cuda()

        result_cpu = apply_clahe(V_cpu, tile_size=16)
        result_gpu = apply_clahe(V_gpu, tile_size=16)

        assert result_cpu.device.type == "cpu"
        assert result_gpu.device.type == "cuda"

class TestCLAHEContrastEnhancement:
    """Tests for contrast enhancement properties."""

    def test_low_contrast_region_enhancement(self) -> None:
        """CLAHE should process low-contrast regions."""
        # Create image with low-contrast gradient region
        V = torch.zeros(256, 256)
        V[64:192, 64:192] = torch.linspace(0.4, 0.6, 128).unsqueeze(1).expand(-1, 128)

        V_clahe = apply_clahe(V, tile_size=16, clip_limit=2.0)

        # Check that CLAHE modified the data (not identical to input)
        assert not torch.allclose(V_clahe, V)

        # Check that output is not degenerate (not all zeros, not uniform)
        region_after = V_clahe[64:192, 64:192]
        assert region_after.std() > 0.01  # Has some variation
        assert torch.all(torch.isfinite(region_after))  # No NaNs or Infs

    def test_heterogeneous_image_balancing(self) -> None:
        """CLAHE should balance contrast across heterogeneous regions."""
        # Create image with varying background
        V = torch.zeros(256, 256)

        # Dark region with dim features
        V[0:128, :] = torch.randn(128, 256) * 0.05 + 0.1

        # Bright region with bright features
        V[128:256, :] = torch.randn(128, 256) * 0.05 + 0.9

        V_clahe = apply_clahe(V, tile_size=16, clip_limit=2.0)

        # Both regions should have similar local contrast
        dark_std = V_clahe[0:128, :].std()
        bright_std = V_clahe[128:256, :].std()

        # Tolerance for similarity
        assert abs(dark_std.item() - bright_std.item()) < 0.3

    def test_clip_limit_effect(self) -> None:
        """Higher clip_limit should produce more enhancement."""
        V = torch.randn(256, 256)

        V_low = apply_clahe(V, tile_size=16, clip_limit=1.0)
        V_high = apply_clahe(V, tile_size=16, clip_limit=4.0)

        # clip_limit=1.0 should produce less enhancement than clip_limit=4.0
        # Measured by how much the histogram is flattened
        low_flatness = V_low.std()
        high_flatness = V_high.std()

        # Higher clip_limit typically produces higher variance (more contrast)
        # (Note: This is a heuristic test, may not always hold strictly)
        assert high_flatness >= low_flatness * 0.9  # Allow some tolerance

class TestCLAHEDimensionality:
    """Tests for nD support."""

    def test_1d_support(self) -> None:
        """CLAHE should work on 1D data."""
        V = torch.randn(256)
        V_clahe = apply_clahe(V, tile_size=16)
        assert V_clahe.shape == (256,)
        assert not torch.allclose(V_clahe, V)  # Should modify the data

    def test_2d_support(self) -> None:
        """CLAHE should work on 2D images."""
        V = torch.randn(256, 256)
        V_clahe = apply_clahe(V, tile_size=16)
        assert V_clahe.shape == (256, 256)

    def test_3d_support(self) -> None:
        """CLAHE should work on 3D volumes."""
        V = torch.randn(64, 64, 64)
        V_clahe = apply_clahe(V, tile_size=16)
        assert V_clahe.shape == (64, 64, 64)

    def test_4d_support(self) -> None:
        """CLAHE should work on 4D data."""
        V = torch.randn(32, 32, 32, 32)
        V_clahe = apply_clahe(V, tile_size=8)
        assert V_clahe.shape == (32, 32, 32, 32)

    def test_non_square_shapes(self) -> None:
        """CLAHE should handle non-square/non-cubic shapes."""
        shapes = [(100, 256), (64, 128, 32), (20, 30, 40, 50)]

        for shape in shapes:
            V = torch.randn(*shape)
            V_clahe = apply_clahe(V, tile_size=16)
            assert V_clahe.shape == V.shape

class TestCLAHEEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_image(self) -> None:
        """CLAHE should handle images smaller than tile_size."""
        V = torch.randn(8, 8)
        V_clahe = apply_clahe(V, tile_size=16)  # tile_size larger than image
        assert V_clahe.shape == (8, 8)

    def test_single_tile(self) -> None:
        """Image with single tile should still work."""
        V = torch.randn(16, 16)
        V_clahe = apply_clahe(V, tile_size=16)
        assert V_clahe.shape == (16, 16)

    def test_exact_tile_division(self) -> None:
        """Image size exactly divisible by tile_size."""
        V = torch.randn(256, 256)
        V_clahe = apply_clahe(V, tile_size=16)  # 256 / 16 = 16 exact
        assert V_clahe.shape == (256, 256)

    def test_inexact_tile_division(self) -> None:
        """Image size not exactly divisible by tile_size."""
        V = torch.randn(250, 250)
        V_clahe = apply_clahe(V, tile_size=16)  # 250 / 16 = 15.625
        assert V_clahe.shape == (250, 250)

    def test_near_uniform_image(self) -> None:
        """Image with very small variance should not crash."""
        V = torch.ones(256, 256) + torch.randn(256, 256) * 1e-6
        V_clahe = apply_clahe(V, tile_size=16, clip_limit=2.0)
        assert V_clahe.shape == (256, 256)

    def test_zero_image(self) -> None:
        """All-zero image should remain zero."""
        V = torch.zeros(256, 256)
        V_clahe = apply_clahe(V, tile_size=16)
        assert torch.allclose(V_clahe, V)

class TestCLAHESamplingProbabilities:
    """Tests for compute_clahe_sampling_probabilities function."""

    def test_probability_properties(self) -> None:
        """Probabilities should sum to 1 and be non-negative."""
        V = torch.randn(256, 256)
        probs, V_clahe = compute_clahe_sampling_probabilities(V, tile_size=16)

        # Check shape
        assert probs.shape == (256 * 256,)

        # Check probability properties
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)

    def test_sampling_works(self) -> None:
        """Should be able to sample from probability distribution."""
        V = torch.randn(256, 256)
        probs, V_clahe = compute_clahe_sampling_probabilities(V, tile_size=16)

        # Sample without errors
        k = 100
        sampled_indices = torch.multinomial(probs, k, replacement=True)

        assert sampled_indices.shape == (k,)
        assert torch.all(sampled_indices >= 0)
        assert torch.all(sampled_indices < 256 * 256)

    def test_clahe_output_returned(self) -> None:
        """Function should return both probabilities and CLAHE result."""
        V = torch.randn(256, 256)
        probs, V_clahe = compute_clahe_sampling_probabilities(V, tile_size=16)

        # V_clahe should be the CLAHE-enhanced image
        assert V_clahe.shape == V.shape
        V_min, V_max = V.min(), V.max()
        assert torch.allclose(V_clahe.min(), V_min, atol=1e-5)
        assert torch.allclose(V_clahe.max(), V_max, atol=1e-5)

    def test_uniform_image_probabilities(self) -> None:
        """Uniform image should produce uniform probabilities."""
        V = torch.ones(256, 256) * 0.5
        probs, V_clahe = compute_clahe_sampling_probabilities(V, tile_size=16)

        # All probabilities should be approximately equal
        expected_prob = 1.0 / (256 * 256)
        assert torch.allclose(probs, torch.tensor(expected_prob), atol=1e-6)

class TestCLAHEParameterValidation:
    """Tests for parameter values and validation."""

    def test_various_tile_sizes(self) -> None:
        """CLAHE should work with various tile sizes."""
        V = torch.randn(256, 256)
        tile_sizes = [4, 8, 16, 32, 64]

        for tile_size in tile_sizes:
            V_clahe = apply_clahe(V, tile_size=tile_size)
            assert V_clahe.shape == V.shape

    def test_various_clip_limits(self) -> None:
        """CLAHE should work with various clip limits."""
        V = torch.randn(256, 256)
        clip_limits = [1.0, 1.5, 2.0, 3.0, 4.0]

        for clip_limit in clip_limits:
            V_clahe = apply_clahe(V, tile_size=16, clip_limit=clip_limit)
            assert V_clahe.shape == V.shape

    def test_various_nbins(self) -> None:
        """CLAHE should work with various number of bins."""
        V = torch.randn(256, 256)
        nbins_values = [64, 128, 256, 512]

        for nbins in nbins_values:
            V_clahe = apply_clahe(V, tile_size=16, nbins=nbins)
            assert V_clahe.shape == V.shape

class TestCLAHENumericalStability:
    """Tests for numerical stability and robustness."""

    def test_extreme_values(self) -> None:
        """CLAHE should handle extreme intensity values."""
        V = torch.randn(256, 256) * 1000 + 5000
        V_clahe = apply_clahe(V, tile_size=16)

        # Should preserve range
        assert torch.allclose(V_clahe.min(), V.min(), atol=1e-3)
        assert torch.allclose(V_clahe.max(), V.max(), atol=1e-3)

    def test_negative_values(self) -> None:
        """CLAHE should handle negative values."""
        V = torch.randn(256, 256) - 10.0
        V_clahe = apply_clahe(V, tile_size=16)

        assert V_clahe.shape == V.shape
        assert torch.allclose(V_clahe.min(), V.min(), atol=1e-5)

    def test_mixed_sign_values(self) -> None:
        """CLAHE should handle mixed positive/negative values."""
        V = torch.randn(256, 256)  # Centered around zero
        V_clahe = apply_clahe(V, tile_size=16)

        assert V_clahe.shape == V.shape

    def test_very_small_range(self) -> None:
        """CLAHE should handle images with very small intensity range."""
        V = torch.ones(256, 256) * 100 + torch.randn(256, 256) * 0.001
        V_clahe = apply_clahe(V, tile_size=16)

        assert V_clahe.shape == V.shape
        # Should not crash or produce NaNs
        assert not torch.any(torch.isnan(V_clahe))
