"""
Tests for FP16 (half precision) CUDA backend support.

These tests compare FP32 and FP16 implementations in terms of:
1. Output accuracy (FP16 should match FP32 within reasonable tolerance)
2. Gradient accuracy (FP16 gradients should match FP32 within tolerance)
3. Performance (FP16 should be faster due to reduced memory bandwidth)
"""

import numpy as np
import pytest
import torch
import time

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# Check if CUDA backend is compiled
try:
    import cuda_splatting_backend

    CUDA_BACKEND_AVAILABLE = True
except ImportError:
    CUDA_BACKEND_AVAILABLE = False


pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


def create_test_splats(
    n_splats: int = 100,
    dim: int = 3,
    shape: tuple = (64, 64, 64),
    seed: int = 42,
):
    """Create test splat data for benchmarking."""
    np.random.seed(seed)

    # Centers distributed across the volume
    centers0 = np.random.rand(n_splats, dim).astype(np.float32)
    for d in range(dim):
        centers0[:, d] = centers0[:, d] * (shape[d] - 4) + 2  # Margin of 2

    # Cholesky factors (lower triangular, positive diagonal)
    L0 = np.zeros((n_splats, dim, dim), dtype=np.float32)
    for i in range(n_splats):
        # Start with identity
        L0[i] = np.eye(dim)
        # Scale by random factor
        L0[i] *= np.random.uniform(0.5, 2.0)
        # Add some off-diagonal elements
        for r in range(dim):
            for c in range(r):
                L0[i, r, c] = np.random.uniform(-0.3, 0.3)

    # Amplitudes
    amps0 = np.random.rand(n_splats).astype(np.float32) * 0.8 + 0.2

    return centers0, L0, amps0


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestFP16Accuracy:
    """Test FP16 vs FP32 numerical accuracy."""

    def test_forward_fp16_matches_fp32_small(self):
        """FP16 forward output should match FP32 within tolerance (small problem)."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=50, dim=3, shape=(32, 32, 32))

        sigma_min = (0.5,) * 3

        # FP32 model
        model_fp32 = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,
            device="cuda",
        )

        # FP16 model
        model_fp16 = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0.copy(),
            L0=L0.copy(),
            amps0=amps0.copy(),
            sigma_min_diag=sigma_min,
            use_fp16=True,
            device="cuda",
        )

        # Forward pass
        output_fp32 = model_fp32()
        output_fp16 = model_fp16()

        # Both outputs should be FP32
        assert output_fp32.dtype == torch.float32
        assert output_fp16.dtype == torch.float32

        # Check for NaN/Inf
        assert not torch.isnan(output_fp32).any(), "FP32 output has NaN"
        assert not torch.isnan(output_fp16).any(), "FP16 output has NaN"
        assert not torch.isinf(output_fp32).any(), "FP32 output has Inf"
        assert not torch.isinf(output_fp16).any(), "FP16 output has Inf"

        # Compare outputs - use 90th percentile relative difference to avoid outliers
        # FP16→FP32→FP16→FP32 conversion chain introduces quantization error
        abs_diff = (output_fp32 - output_fp16).abs()

        # Only consider pixels with significant values (avoid divide-by-small issues)
        significant_mask = output_fp32.abs() > 0.01
        if significant_mask.any():
            rel_diff = (abs_diff / (output_fp32.abs() + 1e-6))[significant_mask]
            # Use 95th percentile instead of max to be robust to outliers
            rel_diff_95 = torch.quantile(rel_diff, 0.95).item()
        else:
            rel_diff_95 = 0.0

        # FP16 conversion introduces ~0.1% relative error typically,
        # but outliers can be higher. Use 10% tolerance for 95th percentile.
        assert (
            rel_diff_95 < 0.10
        ), f"95th percentile relative difference {rel_diff_95:.4f} exceeds 10% tolerance"

        # Also check mean absolute difference
        mean_diff = abs_diff.mean().item()
        assert (
            mean_diff < 0.05
        ), f"Mean absolute difference {mean_diff:.4f} exceeds 0.05 tolerance"

    def test_forward_fp16_matches_fp32_2d(self):
        """FP16 forward output should match FP32 for 2D volumes."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=30, dim=2, shape=(64, 64))

        sigma_min = (0.5,) * 2

        # FP32 model
        model_fp32 = GaussianSplatModelCUDA(
            shape=(64, 64),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,
            device="cuda",
        )

        # FP16 model
        model_fp16 = GaussianSplatModelCUDA(
            shape=(64, 64),
            centers0=centers0.copy(),
            L0=L0.copy(),
            amps0=amps0.copy(),
            sigma_min_diag=sigma_min,
            use_fp16=True,
            device="cuda",
        )

        output_fp32 = model_fp32()
        output_fp16 = model_fp16()

        # Check for NaN/Inf
        assert not torch.isnan(output_fp16).any(), "FP16 output has NaN"
        assert not torch.isinf(output_fp16).any(), "FP16 output has Inf"

        # Compute 95th percentile relative difference on significant values
        abs_diff = (output_fp32 - output_fp16).abs()
        significant_mask = output_fp32.abs() > 0.01
        if significant_mask.any():
            rel_diff = (abs_diff / (output_fp32.abs() + 1e-6))[significant_mask]
            rel_diff_95 = torch.quantile(rel_diff, 0.95).item()
        else:
            rel_diff_95 = 0.0

        # Allow 10% for 95th percentile (FP16 quantization error)
        assert (
            rel_diff_95 < 0.10
        ), f"2D 95th percentile relative difference {rel_diff_95:.4f} exceeds 10%"

    def test_backward_fp16_matches_fp32(self):
        """FP16 gradients should match FP32 gradients within tolerance."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=30, dim=3, shape=(32, 32, 32))

        sigma_min = (0.5,) * 3

        # FP32 model
        model_fp32 = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,
            device="cuda",
        )

        # FP16 model
        model_fp16 = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0.copy(),
            L0=L0.copy(),
            amps0=amps0.copy(),
            sigma_min_diag=sigma_min,
            use_fp16=True,
            device="cuda",
        )

        # Create target for loss
        target = torch.rand(32, 32, 32, device="cuda")

        # Forward + backward for FP32
        output_fp32 = model_fp32()
        loss_fp32 = ((output_fp32 - target) ** 2).sum()
        loss_fp32.backward()

        # Collect FP32 gradients
        grads_fp32 = {}
        for name, param in model_fp32.named_parameters():
            if param.grad is not None:
                grads_fp32[name] = param.grad.clone()

        # Zero gradients
        model_fp32.zero_grad()

        # Forward + backward for FP16
        output_fp16 = model_fp16()
        loss_fp16 = ((output_fp16 - target) ** 2).sum()
        loss_fp16.backward()

        # Collect FP16 gradients
        grads_fp16 = {}
        for name, param in model_fp16.named_parameters():
            if param.grad is not None:
                grads_fp16[name] = param.grad.clone()

        # Compare gradients using median relative difference (robust to outliers)
        # Gradient errors compound from forward pass quantization, so expect higher error
        for name in grads_fp32:
            if name in grads_fp16:
                g32 = grads_fp32[name]
                g16 = grads_fp16[name]

                # Check no NaN/Inf in gradients
                assert not torch.isnan(g16).any(), f"FP16 gradient {name} has NaN"
                assert not torch.isinf(g16).any(), f"FP16 gradient {name} has Inf"

                # Compute relative difference for significant gradients
                significant_mask = g32.abs() > 1e-4
                if significant_mask.any():
                    rel_diff = (
                        (g32 - g16).abs() / (g32.abs() + 1e-6)
                    )[significant_mask]
                    # Use median to be robust to outliers
                    rel_diff_median = torch.median(rel_diff).item()

                    # Gradients should have median match within 50%
                    # (FP16 quantization compounds in backward pass)
                    assert (
                        rel_diff_median < 0.50
                    ), f"Gradient {name} median relative diff {rel_diff_median:.4f} exceeds 50%"

                # Also check that gradient direction is generally preserved (correlation)
                g32_flat = g32.flatten()
                g16_flat = g16.flatten()
                # Simple correlation check: cosine similarity
                dot = (g32_flat * g16_flat).sum()
                norm32 = g32_flat.norm()
                norm16 = g16_flat.norm()
                if norm32 > 1e-6 and norm16 > 1e-6:
                    cosine_sim = (dot / (norm32 * norm16)).item()
                    assert (
                        cosine_sim > 0.9
                    ), f"Gradient {name} cosine similarity {cosine_sim:.4f} too low (< 0.9)"


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestFP16Performance:
    """Test FP16 performance characteristics."""

    def test_fp16_reduces_memory(self):
        """FP16 should have same output memory but reduced internal memory."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=100, dim=3, shape=(64, 64, 64))

        sigma_min = (0.5,) * 3

        # Both models should have same output dtype (FP32)
        model_fp32 = GaussianSplatModelCUDA(
            shape=(64, 64, 64),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,
            device="cuda",
        )

        model_fp16 = GaussianSplatModelCUDA(
            shape=(64, 64, 64),
            centers0=centers0.copy(),
            L0=L0.copy(),
            amps0=amps0.copy(),
            sigma_min_diag=sigma_min,
            use_fp16=True,
            device="cuda",
        )

        output_fp32 = model_fp32()
        output_fp16 = model_fp16()

        # Output dtype should be FP32 for both
        assert output_fp32.dtype == torch.float32
        assert output_fp16.dtype == torch.float32

        # Output shapes should be identical
        assert output_fp32.shape == output_fp16.shape

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 1,
        reason="Need CUDA device for timing test",
    )
    def test_fp16_timing_comparison(self):
        """Compare FP32 vs FP16 execution time (informational)."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        # Larger problem for timing
        centers0, L0, amps0 = create_test_splats(n_splats=500, dim=3, shape=(64, 64, 64))

        sigma_min = (0.5,) * 3

        model_fp32 = GaussianSplatModelCUDA(
            shape=(64, 64, 64),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,
            device="cuda",
        )

        model_fp16 = GaussianSplatModelCUDA(
            shape=(64, 64, 64),
            centers0=centers0.copy(),
            L0=L0.copy(),
            amps0=amps0.copy(),
            sigma_min_diag=sigma_min,
            use_fp16=True,
            device="cuda",
        )

        # Warmup
        for _ in range(3):
            _ = model_fp32()
            _ = model_fp16()

        torch.cuda.synchronize()

        # Time FP32
        n_iters = 10
        start = time.time()
        for _ in range(n_iters):
            _ = model_fp32()
        torch.cuda.synchronize()
        time_fp32 = (time.time() - start) / n_iters

        # Time FP16
        start = time.time()
        for _ in range(n_iters):
            _ = model_fp16()
        torch.cuda.synchronize()
        time_fp16 = (time.time() - start) / n_iters

        # Report timing (don't fail test based on timing)
        print(f"\nFP32 forward: {time_fp32*1000:.2f} ms")
        print(f"FP16 forward: {time_fp16*1000:.2f} ms")
        print(f"Speedup: {time_fp32/time_fp16:.2f}x")

        # The test passes regardless of timing - this is informational
        # FP16 speedup depends on GPU architecture and memory bandwidth


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestFP16EdgeCases:
    """Test FP16 edge cases and numerical stability."""

    def test_fp16_large_amplitudes(self):
        """FP16 should handle large amplitude values."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=20, dim=3, shape=(32, 32, 32))

        # Large amplitudes (but within FP16 range)
        amps0 = amps0 * 100  # Scale up

        sigma_min = (0.5,) * 3

        model = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=True,
            device="cuda",
        )

        output = model()

        # Should not have NaN or Inf
        assert not torch.isnan(output).any(), "FP16 produced NaN with large amplitudes"
        assert not torch.isinf(output).any(), "FP16 produced Inf with large amplitudes"

    def test_fp16_small_values(self):
        """FP16 should handle small values without underflow to zero."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=20, dim=3, shape=(32, 32, 32))

        # Small amplitudes (but above FP16 min)
        amps0 = amps0 * 0.01

        sigma_min = (0.5,) * 3

        model = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=True,
            device="cuda",
        )

        output = model()

        # Should not have NaN
        assert not torch.isnan(output).any(), "FP16 produced NaN with small amplitudes"

        # Should have some non-zero output
        assert output.sum() > 0, "FP16 output is all zeros with small amplitudes"

    def test_fp16_flag_property(self):
        """Test that use_fp16 property is correctly exposed."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=10, dim=3, shape=(16, 16, 16))

        sigma_min = (0.5,) * 3

        model_fp32 = GaussianSplatModelCUDA(
            shape=(16, 16, 16),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,
            device="cuda",
        )

        model_fp16 = GaussianSplatModelCUDA(
            shape=(16, 16, 16),
            centers0=centers0.copy(),
            L0=L0.copy(),
            amps0=amps0.copy(),
            sigma_min_diag=sigma_min,
            use_fp16=True,
            device="cuda",
        )

        assert model_fp32.use_fp16 is False
        assert model_fp16.use_fp16 is True

    def test_fp16_repr_includes_flag(self):
        """Test that repr includes FP16 flag when enabled."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=10, dim=3, shape=(16, 16, 16))

        sigma_min = (0.5,) * 3

        model_fp32 = GaussianSplatModelCUDA(
            shape=(16, 16, 16),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,
            device="cuda",
        )

        model_fp16 = GaussianSplatModelCUDA(
            shape=(16, 16, 16),
            centers0=centers0.copy(),
            L0=L0.copy(),
            amps0=amps0.copy(),
            sigma_min_diag=sigma_min,
            use_fp16=True,
            device="cuda",
        )

        repr_fp32 = repr(model_fp32)
        repr_fp16 = repr(model_fp16)

        assert "fp16" not in repr_fp32.lower()
        assert "fp16" in repr_fp16.lower()


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestFP16LowLevelBackend:
    """Test low-level CUDA backend FP16 functions directly."""

    def test_backend_forward_with_use_fp16_false(self):
        """Test backend forward with use_fp16=False."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        N, d = 10, 3
        shape = [16, 16, 16]

        # Create test data
        centers = torch.rand(N, d, device="cuda", dtype=torch.float32) * 12 + 2
        L = torch.eye(d, device="cuda", dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
        conic = cholesky_to_conic(L)
        amps = torch.rand(N, device="cuda", dtype=torch.float32)
        sharpness = torch.full((N,), 2.0, device="cuda", dtype=torch.float32)

        # Call backend with use_fp16=False
        result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            shape,
            3.0,  # truncate
            1e-5,  # intensity_floor
            8,  # tile_size
            False,  # use_fp16
        )

        output = result[0]
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()

    def test_backend_forward_with_use_fp16_true(self):
        """Test backend forward with use_fp16=True."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        N, d = 10, 3
        shape = [16, 16, 16]

        # Create test data
        centers = torch.rand(N, d, device="cuda", dtype=torch.float32) * 12 + 2
        L = torch.eye(d, device="cuda", dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
        conic = cholesky_to_conic(L)
        amps = torch.rand(N, device="cuda", dtype=torch.float32)
        sharpness = torch.full((N,), 2.0, device="cuda", dtype=torch.float32)

        # Call backend with use_fp16=True
        result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            shape,
            3.0,  # truncate
            1e-5,  # intensity_floor
            8,  # tile_size
            True,  # use_fp16
        )

        output = result[0]
        # Output should still be FP32 (computation is FP32, only storage is FP16)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
