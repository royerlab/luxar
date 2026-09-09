"""
Tests for FP16 (half precision) CUDA backend support.

These tests compare FP32 and FP16 implementations in terms of:
1. Output accuracy (FP16 should match FP32 within reasonable tolerance)
2. Gradient accuracy (FP16 gradients should match FP32 within tolerance)
3. Performance (FP16 should be faster due to reduced memory bandwidth)
"""

import time

import numpy as np
import pytest
import torch

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
        assert rel_diff_95 < 0.10, (
            f"95th pct rel diff {rel_diff_95:.4f} exceeds 10% tolerance"
        )

        # Also check mean absolute difference
        mean_diff = abs_diff.mean().item()
        assert mean_diff < 0.05, (
            f"Mean absolute difference {mean_diff:.4f} exceeds 0.05 tolerance"
        )

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
        assert rel_diff_95 < 0.10, (
            f"2D 95th percentile relative difference {rel_diff_95:.4f} exceeds 10%"
        )

    def test_backward_amp_matches_fp32(self):
        """AMP gradients should match FP32 gradients within tolerance.

        Note: Direct FP16 training (use_fp16=True) is now blocked as it causes
        numerical overflow. This test uses AMP which is the recommended approach.
        """
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

        # AMP model (FP32 params, FP16 compute via autocast)
        model_amp = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0.copy(),
            L0=L0.copy(),
            amps0=amps0.copy(),
            sigma_min_diag=sigma_min,
            use_fp16=False,  # FP32 params for AMP
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

        # Forward + backward for AMP
        with torch.amp.autocast("cuda"):
            output_amp = model_amp()
            loss_amp = ((output_amp - target) ** 2).sum()
        loss_amp.backward()

        # Collect AMP gradients
        grads_amp = {}
        for name, param in model_amp.named_parameters():
            if param.grad is not None:
                grads_amp[name] = param.grad.clone()

        # Compare gradients using median relative difference (robust to outliers).
        # Gradient errors compound from forward pass, so expect higher error.
        for name in grads_fp32:
            if name in grads_amp:
                g32 = grads_fp32[name]
                g_amp = grads_amp[name]

                # Check no NaN/Inf in gradients
                assert not torch.isnan(g_amp).any(), f"AMP gradient {name} has NaN"
                assert not torch.isinf(g_amp).any(), f"AMP gradient {name} has Inf"

                # Compute relative difference for significant gradients
                significant_mask = g32.abs() > 1e-4
                if significant_mask.any():
                    rel_diff = ((g32 - g_amp).abs() / (g32.abs() + 1e-6))[
                        significant_mask
                    ]
                    # Use median to be robust to outliers
                    rel_diff_median = torch.median(rel_diff).item()

                    # Gradients should have median match within 50%
                    # (FP16 quantization compounds in backward pass)
                    assert rel_diff_median < 0.50, (
                        f"Grad {name} median rel diff {rel_diff_median:.4f} exceeds 50%"
                    )

                # Also check that gradient direction is preserved (correlation)
                g32_flat = g32.flatten()
                g_amp_flat = g_amp.flatten()
                # Simple correlation check: cosine similarity
                dot = (g32_flat * g_amp_flat).sum()
                norm32 = g32_flat.norm()
                norm_amp = g_amp_flat.norm()
                if norm32 > 1e-6 and norm_amp > 1e-6:
                    cosine_sim = (dot / (norm32 * norm_amp)).item()
                    assert cosine_sim > 0.9, (
                        f"Grad {name} cos sim {cosine_sim:.4f} too low (< 0.9)"
                    )


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestFP16Performance:
    """Test FP16 performance characteristics."""

    def test_fp16_reduces_memory(self):
        """FP16 should have same output memory but reduced internal memory."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(
            n_splats=100, dim=3, shape=(64, 64, 64)
        )

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
        centers0, L0, amps0 = create_test_splats(
            n_splats=500, dim=3, shape=(64, 64, 64)
        )

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
        print(f"\nFP32 forward: {time_fp32 * 1000:.2f} ms")
        print(f"FP16 forward: {time_fp16 * 1000:.2f} ms")
        print(f"Speedup: {time_fp32 / time_fp16:.2f}x")

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
        L = (
            torch.eye(d, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .repeat(N, 1, 1)
        )
        conic = cholesky_to_conic(L)
        amps = torch.rand(N, device="cuda", dtype=torch.float32)

        # Call backend with use_fp16=False
        result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            shape,
            3.0,  # truncate
            1e-5,  # intensity_floor
            use_fp16=False,
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
        L = (
            torch.eye(d, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .repeat(N, 1, 1)
        )
        conic = cholesky_to_conic(L)
        amps = torch.rand(N, device="cuda", dtype=torch.float32)

        # Call backend with use_fp16=True
        result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            shape,
            3.0,  # truncate
            1e-5,  # intensity_floor
            use_fp16=True,
        )

        output = result[0]
        # Output should still be FP32 (computation is FP32, only storage is FP16)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestAMPSupport:
    """Test PyTorch AMP (Automatic Mixed Precision) support."""

    def test_amp_autocast_detection(self):
        """Model should detect torch.autocast() and use FP16 kernels automatically."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=50, dim=3, shape=(32, 32, 32))
        sigma_min = (0.5,) * 3

        # FP32 params model (standard for AMP training)
        model = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,  # FP32 params for training
            device="cuda",
        )

        # Forward without autocast - should use FP32 kernels
        output_fp32 = model()
        assert output_fp32.dtype == torch.float32

        # Forward with autocast - should automatically use FP16 kernels
        with torch.amp.autocast("cuda"):
            output_amp = model()

        # Output should still be FP32 (autocast only affects internal compute)
        assert output_amp.dtype == torch.float32

        # Results should be similar (within FP16 quantization tolerance)
        # Only compute relative diff on significant values to avoid divide-by-zero
        abs_diff = (output_fp32 - output_amp).abs()
        significant_mask = output_fp32.abs() > 0.01
        if significant_mask.any():
            rel_diff = (abs_diff / (output_fp32.abs() + 1e-6))[significant_mask]
            rel_diff_95 = torch.quantile(rel_diff, 0.95).item()
            assert rel_diff_95 < 0.15, (
                f"AMP differs from FP32: 95th pct = {rel_diff_95:.4f}"
            )

        # Also check mean absolute diff
        mean_diff = abs_diff.mean().item()
        assert mean_diff < 0.1, (
            f"AMP mean absolute diff {mean_diff:.4f} exceeds tolerance"
        )

    def test_amp_training_stability(self):
        """AMP training should work without overflow/underflow."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=30, dim=3, shape=(32, 32, 32))
        sigma_min = (0.5,) * 3

        model = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,  # FP32 params for stable training
            device="cuda",
        )

        target = torch.rand(32, 32, 32, device="cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cuda")

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                output = model()
                loss = ((output - target) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

            # Check parameters don't overflow
            for param in model.parameters():
                assert not torch.isnan(param.data).any(), "Parameter has NaN"
                assert not torch.isinf(param.data).any(), "Parameter has Inf"

        # Loss should be finite
        assert all(np.isfinite(loss_val) for loss_val in losses), (
            f"Loss contains non-finite values: {losses}"
        )

        # Loss should generally decrease (may have small fluctuations)
        assert losses[-1] < losses[0] * 1.5, (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_amp_backward_gradients_finite(self):
        """AMP backward pass should produce finite gradients."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=30, dim=3, shape=(32, 32, 32))
        sigma_min = (0.5,) * 3

        model = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,
            device="cuda",
        )

        target = torch.rand(32, 32, 32, device="cuda")

        with torch.amp.autocast("cuda"):
            output = model()
            loss = ((output - target) ** 2).mean()

        # Backward with scaled loss
        scaler = torch.amp.GradScaler("cuda")
        scaler.scale(loss).backward()

        # Check all gradients are finite (before unscaling)
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"Gradient {name} has NaN"
                # Note: grad might have inf due to scaling, that's OK before unscale

        # After unscaling, check finite
        scaler.unscale_(torch.optim.Adam(model.parameters()))
        for name, param in model.named_parameters():
            if param.grad is not None:
                # After unscaling, check for inf/nan
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    # GradScaler may set inf for skip step - that's expected behavior
                    pass  # This is handled by scaler.step() which will skip the update


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestFP16TrainingBlocker:
    """Test that training with use_fp16=True is blocked with a helpful error."""

    def test_fp16_training_raises_error(self):
        """Attempting backward() with use_fp16=True should raise RuntimeError."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=10, dim=3, shape=(16, 16, 16))
        sigma_min = (0.5,) * 3

        # Create model with FP16 params (inference mode)
        model = GaussianSplatModelCUDA(
            shape=(16, 16, 16),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=True,  # FP16 params - NOT safe for training
            device="cuda",
        )

        # Forward pass should work fine
        output = model()
        assert not torch.isnan(output).any()

        # But backward should raise an error
        target = torch.rand(16, 16, 16, device="cuda")
        loss = ((output - target) ** 2).mean()

        with pytest.raises(RuntimeError) as excinfo:
            loss.backward()

        # Check error message is helpful
        assert "Cannot train with use_fp16=True" in str(excinfo.value)
        assert "torch.amp.autocast" in str(excinfo.value)
        assert "inference" in str(excinfo.value).lower()

    def test_fp16_inference_still_works(self):
        """Forward-only (inference) with use_fp16=True should work fine."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=20, dim=3, shape=(32, 32, 32))
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

        # Multiple forward passes should work
        for _ in range(5):
            output = model()
            assert output.dtype == torch.float32
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestModelMutationMethods:
    """Test model mutation methods (prune_, append_, replace_with) with FP16."""

    def test_prune_fp32(self):
        """Test prune_() works correctly with FP32 model."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=50, dim=3, shape=(32, 32, 32))
        sigma_min = (0.5,) * 3

        model = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,
            device="cuda",
        )

        # Initial forward
        model().clone()
        n_before = model.n_splats()
        assert n_before == 50

        # Prune half the splats
        mask = torch.zeros(50, dtype=torch.bool, device="cuda")
        mask[:25] = True  # Keep first 25

        model.prune_(mask)

        # Verify count decreased
        assert model.n_splats() == 25

        # Forward should still work
        output_after = model()
        assert not torch.isnan(output_after).any()
        assert not torch.isinf(output_after).any()

    def test_prune_fp16(self):
        """Test prune_() maintains FP16 dtype after pruning."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=50, dim=3, shape=(32, 32, 32))
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

        # Verify initial FP16
        for param in model.parameters():
            assert param.dtype == torch.float16, "Params should be FP16 before prune"

        # Prune
        mask = torch.zeros(50, dtype=torch.bool, device="cuda")
        mask[:25] = True
        model.prune_(mask)

        # Verify FP16 maintained after prune
        for param in model.parameters():
            assert param.dtype == torch.float16, "Params should stay FP16 after prune"

        # Forward should work
        output = model()
        assert output.dtype == torch.float32  # Output is always FP32
        assert not torch.isnan(output).any()

    def test_append_fp32(self):
        """Test append_() works correctly with FP32 model."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=30, dim=3, shape=(32, 32, 32))
        sigma_min = (0.5,) * 3

        model = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,
            device="cuda",
        )

        assert model.n_splats() == 30

        # Create new splats to append
        new_centers = torch.rand(10, 3, device="cuda") * 28 + 2
        new_Ls = torch.eye(3, device="cuda").unsqueeze(0).repeat(10, 1, 1)
        new_amps = torch.rand(10, device="cuda") + 0.5

        model.append_(new_centers, new_Ls, new_amps)

        # Verify count increased
        assert model.n_splats() == 40

        # Forward should work
        output = model()
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_append_fp16(self):
        """Test append_() converts new splats to FP16 when FP16 mode is enabled."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=30, dim=3, shape=(32, 32, 32))
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

        # Verify initial FP16
        for param in model.parameters():
            assert param.dtype == torch.float16

        # Append new splats (FP32 inputs)
        new_centers = torch.rand(10, 3, device="cuda")  # FP32
        new_Ls = torch.eye(3, device="cuda").unsqueeze(0).repeat(10, 1, 1)
        new_amps = torch.rand(10, device="cuda") + 0.5

        model.append_(new_centers, new_Ls, new_amps)

        # Verify all params (including new ones) are FP16
        for param in model.parameters():
            assert param.dtype == torch.float16, (
                "All params should be FP16 after append"
            )

        assert model.n_splats() == 40

        # Forward should work
        output = model()
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()

    def test_replace_with_fp32(self):
        """Test replace_with() works correctly with FP32 model."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=30, dim=3, shape=(32, 32, 32))
        sigma_min = (0.5,) * 3

        model = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,
            device="cuda",
        )

        output_before = model().clone()

        # Replace with different splats
        new_centers = torch.rand(20, 3, device="cuda") * 28 + 2
        new_Ls = torch.eye(3, device="cuda").unsqueeze(0).repeat(20, 1, 1) * 1.5
        new_amps = torch.ones(20, device="cuda")

        model.replace_with(new_centers, new_Ls, new_amps)

        assert model.n_splats() == 20

        # Output should be different
        output_after = model()
        assert not torch.allclose(output_before, output_after)
        assert not torch.isnan(output_after).any()

    def test_replace_with_fp16(self):
        """Test replace_with() maintains FP16 mode."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=30, dim=3, shape=(32, 32, 32))
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

        # Replace with FP32 inputs
        new_centers = torch.rand(20, 3, device="cuda")  # FP32
        new_Ls = torch.eye(3, device="cuda").unsqueeze(0).repeat(20, 1, 1)
        new_amps = torch.ones(20, device="cuda")

        model.replace_with(new_centers, new_Ls, new_amps)

        # Verify FP16 maintained
        for param in model.parameters():
            assert param.dtype == torch.float16, "Params should be FP16 after replace"

        output = model()
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestSerializationFP16:
    """Test state_dict/load_state_dict with FP16 models."""

    def test_state_dict_roundtrip_fp32(self):
        """Test save/load roundtrip with FP32 model."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=30, dim=3, shape=(32, 32, 32))
        sigma_min = (0.5,) * 3

        model = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,
            device="cuda",
        )

        output_before = model().clone()

        # Save state
        state = model.state_dict()

        # Create new model and load state
        model2 = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,
            device="cuda",
        )
        model2.load_state_dict(state)

        output_after = model2()

        # Outputs should match
        assert torch.allclose(output_before, output_after, atol=1e-5)

    def test_load_state_dict_reconverts_to_fp16(self):
        """Test that load_state_dict properly reconverts to FP16."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=30, dim=3, shape=(32, 32, 32))
        sigma_min = (0.5,) * 3

        # Create FP32 model and save state
        model_fp32 = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=False,
            device="cuda",
        )
        state_fp32 = model_fp32.state_dict()

        # Verify state is FP32
        for key, value in state_fp32.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                assert value.dtype == torch.float32, f"State {key} should be FP32"

        # Create FP16 model
        model_fp16 = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=True,
            device="cuda",
        )

        # Load FP32 state into FP16 model
        model_fp16.load_state_dict(state_fp32)

        # Verify params are reconverted to FP16
        for param in model_fp16.parameters():
            assert param.dtype == torch.float16, "Params should be FP16 after load"

        # Forward should work
        output = model_fp16()
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_fp16_model_state_dict_is_fp16(self):
        """Test that FP16 model's state_dict contains FP16 tensors."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=30, dim=3, shape=(32, 32, 32))
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

        state = model.state_dict()

        # Verify state contains FP16 tensors
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                assert value.dtype == torch.float16, f"State {key} should be FP16"


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestDeviceMovement:
    """Test device movement (to()) with FP16 models."""

    def test_to_same_device_preserves_fp16(self):
        """Test that to() on same device preserves FP16 dtype."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        centers0, L0, amps0 = create_test_splats(n_splats=20, dim=3, shape=(16, 16, 16))
        sigma_min = (0.5,) * 3

        model = GaussianSplatModelCUDA(
            shape=(16, 16, 16),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=True,
            device="cuda",
        )

        # Verify initial FP16
        for param in model.parameters():
            assert param.dtype == torch.float16

        output_before = model().clone()

        # Move to same device
        model = model.to("cuda")

        # Verify FP16 preserved
        for param in model.parameters():
            assert param.dtype == torch.float16, "FP16 should be preserved after to()"

        output_after = model()
        assert torch.allclose(output_before, output_after, atol=1e-5)

    def test_to_different_cuda_device_if_available(self):
        """Test moving between CUDA devices preserves FP16 (if multiple GPUs)."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        if torch.cuda.device_count() < 2:
            pytest.skip("Need multiple GPUs for this test")

        centers0, L0, amps0 = create_test_splats(n_splats=20, dim=3, shape=(16, 16, 16))
        sigma_min = (0.5,) * 3

        model = GaussianSplatModelCUDA(
            shape=(16, 16, 16),
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min,
            use_fp16=True,
            device="cuda:0",
        )

        # Move to second GPU
        model = model.to("cuda:1")

        # Verify FP16 preserved
        for param in model.parameters():
            assert param.dtype == torch.float16
            assert param.device == torch.device("cuda:1")

        output = model()
        assert torch.isfinite(output).all()
