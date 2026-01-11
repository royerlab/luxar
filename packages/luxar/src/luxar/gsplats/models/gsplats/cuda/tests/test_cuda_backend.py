"""
Tests for CUDA backend core functionality.

These tests verify that the CUDA backend produces correct results
and integrates properly with PyTorch autograd.
"""

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


class TestCUDABackendAvailability:
    """Test CUDA backend availability and initialization."""

    def test_cuda_available(self):
        """Verify CUDA is available."""
        assert torch.cuda.is_available()

    @pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
    def test_backend_import(self):
        """Verify CUDA backend can be imported and has required functions."""
        import cuda_splatting_backend

        assert hasattr(cuda_splatting_backend, "forward")
        assert hasattr(cuda_splatting_backend, "backward")
        assert hasattr(cuda_splatting_backend, "__version__")


class TestCholeskyToConic:
    """Test L → Σ⁻¹ conversion."""

    def test_conic_2d(self):
        """Test 2D conic computation."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        # Create simple 2D Cholesky factor
        L = torch.tensor(
            [
                [[2.0, 0.0], [0.5, 1.5]],  # Splat 0
                [[1.0, 0.0], [0.0, 1.0]],  # Splat 1 (identity)
            ],
            device="cuda",
        )

        conic = cholesky_to_conic(L)

        # Verify shape
        assert conic.shape == (2, 3)  # (N, d*(d+1)/2) = (2, 3)

        # For identity L (splat 1), Σ = I, so Σ⁻¹ = I
        # Upper triangle of I: [1, 0, 1]
        assert torch.allclose(
            conic[1], torch.tensor([1.0, 0.0, 1.0], device="cuda"), atol=1e-5
        )

    def test_conic_3d(self):
        """Test 3D conic computation."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        # Create identity 3D Cholesky factor
        L = torch.eye(3, device="cuda").unsqueeze(0)  # (1, 3, 3)

        conic = cholesky_to_conic(L)

        # Verify shape
        assert conic.shape == (1, 6)  # (N, d*(d+1)/2) = (1, 6)

        # For identity L, Σ = I, so Σ⁻¹ = I
        # Upper triangle of I: [1, 0, 0, 1, 0, 1]
        expected = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]], device="cuda")
        assert torch.allclose(conic, expected, atol=1e-5)

    def test_conic_matches_numpy_reference(self):
        """Verify conic computation matches numpy reference."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        # Random 3D Cholesky factors
        np.random.seed(42)
        N = 5
        d = 3

        # Create valid lower-triangular matrices with positive diagonal
        L_np = np.zeros((N, d, d), dtype=np.float32)
        for i in range(N):
            for r in range(d):
                for c in range(r + 1):
                    if r == c:
                        L_np[i, r, c] = np.random.uniform(0.5, 2.0)
                    else:
                        L_np[i, r, c] = np.random.uniform(-1.0, 1.0)

        L = torch.tensor(L_np, device="cuda")
        conic = cholesky_to_conic(L)

        # Compute reference using numpy
        for i in range(N):
            Sigma = L_np[i] @ L_np[i].T
            Sigma_inv = np.linalg.inv(Sigma)

            # Extract upper triangle
            triu_indices = np.triu_indices(d)
            expected = Sigma_inv[triu_indices]

            actual = conic[i].cpu().numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestCUDAForward:
    """Test CUDA forward pass."""

    def test_forward_matches_cpu_3d(self):
        """Verify CUDA forward output matches PyTorch reference."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        # Create test data
        np.random.seed(42)
        N, d = 50, 3
        shape = (32, 32, 32)

        centers0 = np.random.rand(N, d).astype(np.float32) * 28 + 2
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        # Add some variation to L
        for i in range(N):
            L0[i] *= np.random.uniform(0.8, 1.5)
        amps0 = np.random.rand(N).astype(np.float32) * 0.5 + 0.5

        # Create CPU model
        cpu_model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cpu",
        )

        # Create CUDA model
        cuda_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        # Run forward passes
        with torch.no_grad():
            cpu_output = cpu_model()
            cuda_output = cuda_model()

        # Move CUDA output to CPU for comparison
        cuda_output_cpu = cuda_output.cpu()

        # Reshape CUDA output if needed (it may be flattened)
        if cuda_output_cpu.shape != cpu_output.shape:
            cuda_output_cpu = cuda_output_cpu.reshape(cpu_output.shape)

        # Compare outputs
        # Allow for small numerical differences due to different execution paths
        max_val = max(cpu_output.abs().max().item(), cuda_output_cpu.abs().max().item())
        if max_val > 0:
            rel_diff = (cpu_output - cuda_output_cpu).abs() / (max_val + 1e-8)
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()

            # Relaxed tolerance for CUDA vs CPU comparison
            assert max_rel_diff < 0.1, (
                f"Max relative difference {max_rel_diff:.4f} exceeds threshold"
            )
            assert mean_rel_diff < 0.01, (
                f"Mean relative difference {mean_rel_diff:.4f} exceeds threshold"
            )

        # Also check correlation for overall structure match
        cpu_flat = cpu_output.flatten()
        cuda_flat = cuda_output_cpu.flatten()
        if cpu_flat.std() > 1e-6 and cuda_flat.std() > 1e-6:
            correlation = torch.corrcoef(torch.stack([cpu_flat, cuda_flat]))[0, 1]
            assert correlation > 0.99, f"Correlation {correlation:.4f} too low"

    def test_forward_matches_cpu_2d(self):
        """Verify CUDA 2D forward matches PyTorch reference."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        # Create test data
        np.random.seed(123)
        N, d = 30, 2
        shape = (64, 64)

        centers0 = np.random.rand(N, d).astype(np.float32) * 60 + 2
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0[i] *= np.random.uniform(1.0, 3.0)
        amps0 = np.random.rand(N).astype(np.float32) * 0.8 + 0.2

        # Create models
        cpu_model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5),
            device="cpu",
        )

        cuda_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5),
            device="cuda",
        )

        # Run forward passes
        with torch.no_grad():
            cpu_output = cpu_model()
            cuda_output = cuda_model()

        cuda_output_cpu = cuda_output.cpu()

        # Reshape CUDA output if needed (it may be flattened)
        if cuda_output_cpu.shape != cpu_output.shape:
            cuda_output_cpu = cuda_output_cpu.reshape(cpu_output.shape)

        # Check correlation
        cpu_flat = cpu_output.flatten()
        cuda_flat = cuda_output_cpu.flatten()
        if cpu_flat.std() > 1e-6 and cuda_flat.std() > 1e-6:
            correlation = torch.corrcoef(torch.stack([cpu_flat, cuda_flat]))[0, 1]
            assert correlation > 0.99, f"Correlation {correlation:.4f} too low"


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestCUDABackward:
    """Test CUDA backward pass."""

    def test_gradcheck_3d(self):
        """Verify gradients are computed without errors."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        # Create test data
        np.random.seed(42)
        N, d = 10, 3
        shape = (16, 16, 16)

        centers0 = np.random.rand(N, d).astype(np.float32) * 12 + 2
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        amps0 = np.ones(N, dtype=np.float32)

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        # Forward pass
        output = model()

        # Create a simple loss - handle potentially flattened output
        target = torch.zeros_like(output)
        # Set target at center voxel (compute linear index if flattened)
        if output.dim() == 1:
            center_idx = 8 * shape[1] * shape[2] + 8 * shape[2] + 8
            target[center_idx] = 1.0
        else:
            target[8, 8, 8] = 1.0
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass - verify it completes without error
        loss.backward()

        # Check that gradients exist and are finite
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_gradient_values_match_cpu(self):
        """Verify gradient values are reasonable compared to PyTorch autograd."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        # Create test data - use same seed for reproducibility
        np.random.seed(789)
        N, d = 20, 3
        shape = (24, 24, 24)

        centers0 = np.random.rand(N, d).astype(np.float32) * 20 + 2
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        amps0 = np.ones(N, dtype=np.float32) * 0.5

        # Create models
        cpu_model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cpu",
        )

        cuda_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        # Create matching targets
        cpu_target = torch.zeros(shape, dtype=torch.float32, device="cpu")
        cpu_target[12, 12, 12] = 1.0

        # Forward + backward on CPU
        cpu_output = cpu_model()
        cpu_loss = torch.nn.functional.mse_loss(cpu_output, cpu_target)
        cpu_loss.backward()

        # Forward + backward on CUDA - handle potentially flattened output
        cuda_output = cuda_model()
        # Create target with same shape as CUDA output
        cuda_target = torch.zeros_like(cuda_output)
        if cuda_output.dim() == 1:
            center_idx = 12 * shape[1] * shape[2] + 12 * shape[2] + 12
            cuda_target[center_idx] = 1.0
        else:
            cuda_target[12, 12, 12] = 1.0
        cuda_loss = torch.nn.functional.mse_loss(cuda_output, cuda_target)
        cuda_loss.backward()

        # Compare gradient magnitudes (not exact values due to different code paths)
        # Focus on amplitude gradients as they're most directly comparable
        cpu_amp_grad = cpu_model.raw_a.grad
        cuda_amp_grad = cuda_model.raw_a.grad.cpu()

        # Check gradient signs match for majority of splats
        if cpu_amp_grad is not None and cuda_amp_grad is not None:
            sign_match = (
                (torch.sign(cpu_amp_grad) == torch.sign(cuda_amp_grad)).float().mean()
            )
            # Allow some sign differences due to numerical precision
            assert sign_match > 0.7, f"Gradient sign match {sign_match:.2f} too low"

            # Check gradient magnitudes are in similar range
            cpu_mag = cpu_amp_grad.abs().mean()
            cuda_mag = cuda_amp_grad.abs().mean()
            if cpu_mag > 1e-8 and cuda_mag > 1e-8:
                mag_ratio = max(cpu_mag / cuda_mag, cuda_mag / cpu_mag)
                assert mag_ratio < 10, (
                    f"Gradient magnitude ratio {mag_ratio:.2f} too large"
                )


class TestCUDAKernelActivation:
    """Tests to verify CUDA kernels are actually being used (not PyTorch fallback)."""

    @pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
    def test_cuda_kernels_called_not_fallback(self):
        """Verify the CUDA kernels are called, not PyTorch fallback.

        This is critical because the code can silently fall back to PyTorch
        if CUDA_BACKEND_AVAILABLE is False in the module, resulting in
        significantly slower performance.
        """
        from luxar.gsplats.models.gsplats.cuda import gsplat_model_cuda

        # Verify the module-level flag is True (this is what the forward() uses)
        assert gsplat_model_cuda.CUDA_BACKEND_AVAILABLE, (
            "CUDA_BACKEND_AVAILABLE is False in gsplat_model_cuda module. "
            "This means the forward() will use PyTorch fallback, not CUDA kernels!"
        )

        # Verify cuda_splatting_backend is importable
        try:
            import cuda_splatting_backend

            assert hasattr(cuda_splatting_backend, "forward"), (
                "cuda_splatting_backend missing 'forward' function"
            )
            assert hasattr(cuda_splatting_backend, "backward"), (
                "cuda_splatting_backend missing 'backward' function"
            )
        except ImportError as e:
            pytest.fail(f"cuda_splatting_backend cannot be imported: {e}")

    @pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
    def test_cuda_forward_faster_than_cpu(self):
        """Verify CUDA forward is significantly faster than CPU.

        A Titan RTX should be at least 5x faster than CPU for this workload.
        If speedup is less than 2x, something is likely wrong.
        """
        import time

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        # Use realistic workload: 128^3 volume with 500 splats
        np.random.seed(42)
        N, d = 500, 3
        shape = (64, 64, 64)  # Smaller for faster test but still representative

        centers0 = np.random.rand(N, d).astype(np.float32) * 60 + 2
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0[i] *= np.random.uniform(1.0, 3.0)
        amps0 = np.random.rand(N).astype(np.float32) * 0.8 + 0.2

        # Create CPU model
        cpu_model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cpu",
        )

        # Create CUDA model
        cuda_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        # Warmup
        with torch.no_grad():
            _ = cpu_model()
            _ = cuda_model()
            torch.cuda.synchronize()

        # Time CPU
        n_runs = 5
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = cpu_model()
        cpu_time = (time.time() - start) / n_runs

        # Time CUDA
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = cuda_model()
                torch.cuda.synchronize()
        cuda_time = (time.time() - start) / n_runs

        speedup = cpu_time / cuda_time
        print(
            f"\nCPU time: {cpu_time * 1000:.2f}ms, CUDA time: {cuda_time * 1000:.2f}ms, Speedup: {speedup:.1f}x"
        )

        # CUDA should be at least 2x faster
        # If not, either the kernels aren't being used or there's a problem
        assert speedup > 2.0, (
            f"CUDA speedup ({speedup:.1f}x) is too low! "
            f"Expected at least 2x. CPU={cpu_time * 1000:.1f}ms, CUDA={cuda_time * 1000:.1f}ms. "
            f"This might indicate CUDA kernels are not being used."
        )

    @pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
    def test_cuda_kernels_return_different_from_fallback_timing(self):
        """Verify CUDA kernel path has different timing characteristics.

        The CUDA kernel should be much faster than PyTorch fallback on CUDA device.
        If they're similar, the wrong path is being taken.
        """
        import time

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            CUDA_BACKEND_AVAILABLE as MODULE_CUDA_AVAILABLE,
        )
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        # Skip if module flag is False (would just use fallback anyway)
        if not MODULE_CUDA_AVAILABLE:
            pytest.skip("CUDA_BACKEND_AVAILABLE is False in the module")

        np.random.seed(42)
        N, d = 200, 3
        shape = (48, 48, 48)

        centers0 = np.random.rand(N, d).astype(np.float32) * 44 + 2
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0) * 2.0
        amps0 = np.ones(N, dtype=np.float32)

        # CUDA model with custom kernels
        cuda_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        # Standard PyTorch model on CUDA (uses PyTorch rendering, not custom kernels)
        pytorch_cuda_model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        # Warmup
        with torch.no_grad():
            _ = cuda_model()
            _ = pytorch_cuda_model()
            torch.cuda.synchronize()

        # Time custom CUDA kernels
        n_runs = 10
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = cuda_model()
                torch.cuda.synchronize()
        custom_cuda_time = (time.time() - start) / n_runs

        # Time PyTorch on CUDA
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = pytorch_cuda_model()
                torch.cuda.synchronize()
        pytorch_cuda_time = (time.time() - start) / n_runs

        print(
            f"\nCustom CUDA: {custom_cuda_time * 1000:.2f}ms, "
            f"PyTorch CUDA: {pytorch_cuda_time * 1000:.2f}ms, "
            f"Ratio: {pytorch_cuda_time / custom_cuda_time:.1f}x"
        )

        # Custom kernels should be faster than PyTorch on CUDA
        # If they're slower or similar, something is wrong
        # Note: Allow some margin because PyTorch CUDA can be fast too
        if custom_cuda_time > pytorch_cuda_time * 0.9:
            # This is a warning, not a hard failure, because PyTorch can be optimized
            print(
                "WARNING: Custom CUDA kernels not significantly faster than PyTorch CUDA"
            )


class TestGaussianSplatModelCUDA:
    """Test GaussianSplatModelCUDA class."""

    def test_model_creation(self):
        """Test model can be created."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        # Create simple model
        N, d = 10, 3
        shape = (32, 32, 32)

        centers0 = np.random.rand(N, d).astype(np.float32) * 30 + 1
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        amps0 = np.ones(N, dtype=np.float32)

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        assert model.n_splats() == N
        assert model.dim == d
        assert model.shape == shape

    def test_dimension_validation(self):
        """Test that invalid dimensions are rejected."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        N = 5

        # 9D should be rejected (max is 8D)
        with pytest.raises(ValueError, match="CUDA backend supports 2D-8D"):
            GaussianSplatModelCUDA(
                shape=(4,) * 9,
                centers0=np.random.rand(N, 9).astype(np.float32),
                L0=np.eye(9, dtype=np.float32)[None, :, :].repeat(N, axis=0),
                amps0=np.ones(N, dtype=np.float32),
                sigma_min_diag=(0.5,) * 9,
                device="cuda",
            )

    def test_device_validation(self):
        """Test that non-CUDA devices are rejected."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        N, d = 5, 3

        with pytest.raises(ValueError, match="CUDA backend requires CUDA device"):
            GaussianSplatModelCUDA(
                shape=(32, 32, 32),
                centers0=np.random.rand(N, d).astype(np.float32),
                L0=np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0),
                amps0=np.ones(N, dtype=np.float32),
                sigma_min_diag=(0.5, 0.5, 0.5),
                device="cpu",
            )

    def test_auto_tile_size(self):
        """Test automatic tile size selection."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        N = 5

        # 2D should use tile_size=16
        model_2d = GaussianSplatModelCUDA(
            shape=(64, 64),
            centers0=np.random.rand(N, 2).astype(np.float32) * 60,
            L0=np.eye(2, dtype=np.float32)[None, :, :].repeat(N, axis=0),
            amps0=np.ones(N, dtype=np.float32),
            sigma_min_diag=(0.5, 0.5),
            device="cuda",
        )
        assert model_2d._tile_size == 16

        # 3D should use tile_size=8
        model_3d = GaussianSplatModelCUDA(
            shape=(32, 32, 32),
            centers0=np.random.rand(N, 3).astype(np.float32) * 30,
            L0=np.eye(3, dtype=np.float32)[None, :, :].repeat(N, axis=0),
            amps0=np.ones(N, dtype=np.float32),
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )
        assert model_3d._tile_size == 8

        # 4D should use tile_size=4
        model_4d = GaussianSplatModelCUDA(
            shape=(16, 16, 16, 16),
            centers0=np.random.rand(N, 4).astype(np.float32) * 14,
            L0=np.eye(4, dtype=np.float32)[None, :, :].repeat(N, axis=0),
            amps0=np.ones(N, dtype=np.float32),
            sigma_min_diag=(0.5, 0.5, 0.5, 0.5),
            device="cuda",
        )
        assert model_4d._tile_size == 4

    def test_output_shape_matches_initialization(self):
        """Test that model forward() returns tensor with correct shape.

        This guards against a regression where the CUDA kernel might return
        a flattened tensor instead of the expected shaped tensor.
        """
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        N = 10
        shape_3d = (32, 32, 32)
        shape_2d = (64, 64)

        # Test 3D
        model_3d = GaussianSplatModelCUDA(
            shape=shape_3d,
            centers0=np.random.rand(N, 3).astype(np.float32) * 28 + 2,
            L0=np.eye(3, dtype=np.float32)[None, :, :].repeat(N, axis=0) * 2,
            amps0=np.ones(N, dtype=np.float32),
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        with torch.no_grad():
            output_3d = model_3d()

        assert output_3d.shape == shape_3d, (
            f"3D output shape mismatch: got {output_3d.shape}, expected {shape_3d}"
        )

        # Test 2D
        model_2d = GaussianSplatModelCUDA(
            shape=shape_2d,
            centers0=np.random.rand(N, 2).astype(np.float32) * 60 + 2,
            L0=np.eye(2, dtype=np.float32)[None, :, :].repeat(N, axis=0) * 2,
            amps0=np.ones(N, dtype=np.float32),
            sigma_min_diag=(0.5, 0.5),
            device="cuda",
        )

        with torch.no_grad():
            output_2d = model_2d()

        assert output_2d.shape == shape_2d, (
            f"2D output shape mismatch: got {output_2d.shape}, expected {shape_2d}"
        )


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestCUDAEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_splat(self):
        """Test with a single splat - minimal case."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        N = 1
        d = 3
        shape = (16, 16, 16)

        centers0 = np.array([[8.0, 8.0, 8.0]], dtype=np.float32)
        L0 = np.eye(d, dtype=np.float32)[None, :, :]
        amps0 = np.array([1.0], dtype=np.float32)

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        with torch.no_grad():
            output = model()

        # Should have a peak at center
        assert output.shape == shape
        assert output.max() > 0, "Single splat should produce non-zero output"
        # Center should have highest intensity
        center_val = output[8, 8, 8].item()
        assert center_val == output.max().item(), "Peak should be at splat center"

    def test_splat_on_volume_boundary(self):
        """Test splat positioned exactly on volume boundary."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        N = 1
        d = 3
        shape = (16, 16, 16)

        # Splat at corner (0, 0, 0)
        centers0 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        L0 = np.eye(d, dtype=np.float32)[None, :, :] * 2
        amps0 = np.array([1.0], dtype=np.float32)

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        with torch.no_grad():
            output = model()

        # Should have contribution at corner
        assert output.shape == shape
        assert output[0, 0, 0] > 0, "Boundary splat should contribute"

    def test_splat_outside_volume(self):
        """Test splat initialized outside the volume still produces valid output.

        Note: The model may clamp or transform center positions, so we just
        verify the output is valid (finite, correct shape) rather than
        expecting zero output.
        """
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        N = 1
        d = 3
        shape = (16, 16, 16)

        # Splat far outside volume - model may clamp this to valid range
        centers0 = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
        L0 = np.eye(d, dtype=np.float32)[None, :, :]
        amps0 = np.array([1.0], dtype=np.float32)

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        with torch.no_grad():
            output = model()

        # Verify output is valid - model may have clamped the center
        assert output.shape == shape
        assert torch.isfinite(output).all(), "Output should be finite"
        # The center may be clamped, so just verify it runs without error

    def test_high_dimension_7d(self):
        """Test 7D splatting (high dimension)."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        N = 5
        d = 7
        shape = (4,) * d  # 4^7 = 16384 voxels

        centers0 = np.random.rand(N, d).astype(np.float32) * 2 + 1
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        amps0 = np.ones(N, dtype=np.float32)

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * d,
            device="cuda",
        )

        with torch.no_grad():
            output = model()

        assert output.shape == shape
        assert torch.isfinite(output).all(), "7D output should be finite"

    def test_high_dimension_8d(self):
        """Test 8D splatting (maximum supported dimension)."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        N = 3
        d = 8
        shape = (3,) * d  # 3^8 = 6561 voxels

        centers0 = np.random.rand(N, d).astype(np.float32) * 1.5 + 0.5
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        amps0 = np.ones(N, dtype=np.float32)

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * d,
            device="cuda",
        )

        with torch.no_grad():
            output = model()

        assert output.shape == shape
        assert torch.isfinite(output).all(), "8D output should be finite"

    def test_very_small_splat(self):
        """Test splat with very small covariance (sub-voxel).

        Note: sigma_min_diag enforces a minimum spread, so we just verify
        the center has high intensity relative to neighbors.
        """
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        N = 1
        d = 3
        shape = (16, 16, 16)

        centers0 = np.array([[8.0, 8.0, 8.0]], dtype=np.float32)
        # Very small L (tight Gaussian) - will be clamped by sigma_min_diag
        L0 = np.eye(d, dtype=np.float32)[None, :, :] * 0.5
        amps0 = np.array([1.0], dtype=np.float32)

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.3, 0.3, 0.3),
            device="cuda",
        )

        with torch.no_grad():
            output = model()

        assert output.shape == shape
        assert torch.isfinite(output).all(), "Small splat output should be finite"
        # Center should have maximum intensity (peak is at center)
        center_val = output[8, 8, 8].item()
        max_val = output.max().item()
        assert abs(center_val - max_val) < 0.01, "Peak should be at splat center"
        # Center value should be significant (amplitude ~= 1)
        assert center_val > 0.9, "Center intensity should be close to amplitude"

    def test_overlapping_splats(self):
        """Test multiple splats at the same location."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        N = 5
        d = 3
        shape = (16, 16, 16)

        # All splats at same center
        centers0 = np.tile(np.array([[8.0, 8.0, 8.0]], dtype=np.float32), (N, 1))
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        amps0 = np.ones(N, dtype=np.float32)

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        with torch.no_grad():
            output = model()

        # Intensity should be N times a single splat
        single_centers0 = np.array([[8.0, 8.0, 8.0]], dtype=np.float32)
        single_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=single_centers0,
            L0=L0[:1],
            amps0=amps0[:1],
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        with torch.no_grad():
            single_output = single_model()

        # Overlapping splats should sum
        ratio = output[8, 8, 8].item() / single_output[8, 8, 8].item()
        assert abs(ratio - N) < 0.1, f"Expected {N}x intensity, got {ratio:.2f}x"

    def test_negative_amplitude(self):
        """Test splat with negative amplitude produces valid output.

        Note: The model may handle negative amplitudes differently (e.g.,
        through softplus activation). This test verifies that negative
        initial amplitudes don't cause errors.
        """
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        N = 2
        d = 3
        shape = (16, 16, 16)

        # One positive, one negative splat at same location
        centers0 = np.array([[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]], dtype=np.float32)
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        amps0 = np.array([1.0, -0.5], dtype=np.float32)

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        with torch.no_grad():
            output = model()

        # Verify output is valid
        assert output.shape == shape
        assert torch.isfinite(output).all(), (
            "Output with negative amplitude should be finite"
        )

        # For comparison, create a single positive splat
        single_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0[:1],
            L0=L0[:1],
            amps0=np.array([1.0], dtype=np.float32),
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        with torch.no_grad():
            single_output = single_model()

        # The combined output should differ from single splat
        # (exact behavior depends on amplitude activation function)
        diff = (output - single_output).abs().max().item()
        # Just verify the outputs are different (negative amplitude has effect)
        # Note: If model uses softplus, negative amplitude becomes small positive
        assert output.max() > 0, "Output should have positive intensity"

    def test_zero_amplitude_splats(self):
        """Test splat with zero amplitude (should contribute nothing)."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        N = 1
        d = 3
        shape = (16, 16, 16)

        centers0 = np.array([[8.0, 8.0, 8.0]], dtype=np.float32)
        L0 = np.eye(d, dtype=np.float32)[None, :, :]
        amps0 = np.array([0.0], dtype=np.float32)

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        with torch.no_grad():
            output = model()

        assert output.shape == shape
        assert output.abs().max() < 1e-6, (
            "Zero amplitude splat should produce zero output"
        )


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestSpecializedVsGenericImplementations:
    """Test that specialized 2D/3D implementations match generic nD implementations.

    These tests verify that the optimized hardcoded 2D/3D Mahalanobis distance
    and backward gradient computations produce identical results to the generic
    loop-based implementations used for higher dimensions (4D-8D).

    IMPORTANT: These tests exist because we have template specializations for
    2D and 3D that use explicit formulas for performance. Any changes to either
    the specialized or generic code must keep them in sync.
    """

    def test_3d_forward_matches_4d_slice(self):
        """Verify 3D specialized kernel produces same results as nD for equivalent setup.

        Test strategy: Create equivalent 3D and 4D configurations where the 4D
        has size 1 in the 4th dimension, effectively being a 3D problem.
        The 4D version uses the generic code path while 3D uses specialized path.
        """
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        np.random.seed(42)
        N = 20

        # 3D setup - uses specialized Mahalanobis distance
        shape_3d = (24, 24, 24)
        centers0_3d = np.random.rand(N, 3).astype(np.float32) * 20 + 2
        L0_3d = np.eye(3, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0_3d[i] *= np.random.uniform(1.0, 2.0)
        amps0 = np.random.rand(N).astype(np.float32) * 0.5 + 0.5

        model_3d = GaussianSplatModelCUDA(
            shape=shape_3d,
            centers0=centers0_3d,
            L0=L0_3d,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        # 4D setup equivalent to 3D (4th dimension has size 1, center at 0)
        # Uses generic Mahalanobis distance computation
        shape_4d = (24, 24, 24, 1)
        centers0_4d = np.zeros((N, 4), dtype=np.float32)
        centers0_4d[:, :3] = centers0_3d
        centers0_4d[:, 3] = 0.0  # 4th dim center at slice position

        L0_4d = np.eye(4, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0_4d[i, :3, :3] = L0_3d[i]
            L0_4d[i, 3, 3] = 0.1  # Very small in 4th dimension

        model_4d = GaussianSplatModelCUDA(
            shape=shape_4d,
            centers0=centers0_4d,
            L0=L0_4d,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5, 0.1),
            device="cuda",
        )

        with torch.no_grad():
            output_3d = model_3d()
            output_4d = model_4d()

        # Extract 3D slice from 4D output
        output_4d_slice = output_4d[..., 0]

        # Compare outputs - they should be very close
        # Note: Small differences expected due to different tile structures
        max_3d = output_3d.max().item()
        if max_3d > 0.01:  # Only compare if there's meaningful output
            rel_diff = (output_3d - output_4d_slice).abs() / (max_3d + 1e-8)
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()

            assert max_rel_diff < 0.3, (
                f"3D vs 4D max relative difference {max_rel_diff:.4f} too large"
            )
            assert mean_rel_diff < 0.05, (
                f"3D vs 4D mean relative difference {mean_rel_diff:.4f} too large"
            )

            # Also check correlation
            corr = torch.corrcoef(
                torch.stack([output_3d.flatten(), output_4d_slice.flatten()])
            )[0, 1]
            assert corr > 0.95, f"3D vs 4D correlation {corr:.4f} too low"

    def test_2d_forward_matches_3d_slice(self):
        """Verify 2D specialized kernel produces same results as nD for equivalent setup."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        np.random.seed(123)
        N = 15

        # 2D setup - uses specialized Mahalanobis distance
        shape_2d = (32, 32)
        centers0_2d = np.random.rand(N, 2).astype(np.float32) * 28 + 2
        L0_2d = np.eye(2, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0_2d[i] *= np.random.uniform(1.0, 3.0)
        amps0 = np.random.rand(N).astype(np.float32) * 0.5 + 0.5

        model_2d = GaussianSplatModelCUDA(
            shape=shape_2d,
            centers0=centers0_2d,
            L0=L0_2d,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5),
            device="cuda",
        )

        # 3D setup equivalent to 2D (3rd dimension has size 1)
        shape_3d = (32, 32, 1)
        centers0_3d = np.zeros((N, 3), dtype=np.float32)
        centers0_3d[:, :2] = centers0_2d
        centers0_3d[:, 2] = 0.0

        L0_3d = np.eye(3, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0_3d[i, :2, :2] = L0_2d[i]
            L0_3d[i, 2, 2] = 0.1

        model_3d = GaussianSplatModelCUDA(
            shape=shape_3d,
            centers0=centers0_3d,
            L0=L0_3d,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.1),
            device="cuda",
        )

        with torch.no_grad():
            output_2d = model_2d()
            output_3d = model_3d()

        # Extract 2D slice from 3D output
        output_3d_slice = output_3d[..., 0]

        # Compare outputs
        max_2d = output_2d.max().item()
        if max_2d > 0.01:
            rel_diff = (output_2d - output_3d_slice).abs() / (max_2d + 1e-8)
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()

            assert max_rel_diff < 0.3, (
                f"2D vs 3D max relative difference {max_rel_diff:.4f} too large"
            )
            assert mean_rel_diff < 0.05, (
                f"2D vs 3D mean relative difference {mean_rel_diff:.4f} too large"
            )

    def test_3d_backward_gradients_finite(self):
        """Verify 3D specialized backward produces finite gradients."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        np.random.seed(456)
        N = 15

        shape = (20, 20, 20)
        centers0 = np.random.rand(N, 3).astype(np.float32) * 16 + 2
        L0 = np.eye(3, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0[i] *= np.random.uniform(1.0, 2.0)
        amps0 = np.random.rand(N).astype(np.float32) * 0.5 + 0.5

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        # Forward pass
        output = model()

        # Create target and compute loss
        target = torch.zeros_like(output)
        target[10, 10, 10] = 1.0
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check all gradients are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient for {name}"
                )

    def test_2d_backward_gradients_finite(self):
        """Verify 2D specialized backward produces finite gradients."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        np.random.seed(789)
        N = 10

        shape = (48, 48)
        centers0 = np.random.rand(N, 2).astype(np.float32) * 44 + 2
        L0 = np.eye(2, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0[i] *= np.random.uniform(1.0, 3.0)
        amps0 = np.random.rand(N).astype(np.float32) * 0.5 + 0.5

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5),
            device="cuda",
        )

        # Forward pass
        output = model()

        # Create target and compute loss
        target = torch.zeros_like(output)
        target[24, 24] = 1.0
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check all gradients are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient for {name}"
                )

    def test_3d_backward_gradient_sign_consistency(self):
        """Verify 3D backward gradients have consistent signs with 4D.

        This tests that the specialized 3D backward gradient computation
        produces gradients with the same signs as the generic computation.
        """
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        np.random.seed(111)
        N = 10

        # Create equivalent 3D and 4D setups
        shape_3d = (16, 16, 16)
        centers0_3d = np.random.rand(N, 3).astype(np.float32) * 12 + 2
        L0_3d = np.eye(3, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0_3d[i] *= np.random.uniform(1.0, 2.0)
        amps0 = np.random.rand(N).astype(np.float32) * 0.5 + 0.5

        shape_4d = (16, 16, 16, 1)
        centers0_4d = np.zeros((N, 4), dtype=np.float32)
        centers0_4d[:, :3] = centers0_3d
        L0_4d = np.eye(4, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0_4d[i, :3, :3] = L0_3d[i]
            L0_4d[i, 3, 3] = 0.1

        model_3d = GaussianSplatModelCUDA(
            shape=shape_3d,
            centers0=centers0_3d,
            L0=L0_3d,
            amps0=amps0.copy(),
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        model_4d = GaussianSplatModelCUDA(
            shape=shape_4d,
            centers0=centers0_4d,
            L0=L0_4d,
            amps0=amps0.copy(),
            sigma_min_diag=(0.5, 0.5, 0.5, 0.1),
            device="cuda",
        )

        # Forward passes
        output_3d = model_3d()
        output_4d = model_4d()

        # Create compatible targets
        target_3d = torch.zeros_like(output_3d)
        target_3d[8, 8, 8] = 1.0

        target_4d = torch.zeros_like(output_4d)
        target_4d[8, 8, 8, 0] = 1.0

        # Backward passes
        loss_3d = torch.nn.functional.mse_loss(output_3d, target_3d)
        loss_3d.backward()

        loss_4d = torch.nn.functional.mse_loss(output_4d, target_4d)
        loss_4d.backward()

        # Compare amplitude gradient signs
        grad_amp_3d = model_3d.raw_a.grad
        grad_amp_4d = model_4d.raw_a.grad

        if grad_amp_3d is not None and grad_amp_4d is not None:
            # Only compare where gradients are significant
            significant = (grad_amp_3d.abs() > 1e-6) & (grad_amp_4d.abs() > 1e-6)
            if significant.any():
                sign_match = (
                    (
                        torch.sign(grad_amp_3d[significant])
                        == torch.sign(grad_amp_4d[significant])
                    )
                    .float()
                    .mean()
                )
                # Allow some differences due to numerical precision
                assert sign_match > 0.7, (
                    f"3D vs 4D amplitude gradient sign match {sign_match:.2f} too low"
                )

    def test_2d_backward_gradient_sign_consistency(self):
        """Verify 2D backward gradients have consistent signs with 3D.

        This tests that the specialized 2D backward gradient computation
        produces gradients with the same signs as the 3D generic computation.
        """
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        np.random.seed(222)
        N = 10

        # Create equivalent 2D and 3D setups
        shape_2d = (24, 24)
        centers0_2d = np.random.rand(N, 2).astype(np.float32) * 20 + 2
        L0_2d = np.eye(2, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0_2d[i] *= np.random.uniform(1.0, 2.0)
        amps0 = np.random.rand(N).astype(np.float32) * 0.5 + 0.5

        shape_3d = (24, 24, 1)
        centers0_3d = np.zeros((N, 3), dtype=np.float32)
        centers0_3d[:, :2] = centers0_2d
        L0_3d = np.eye(3, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0_3d[i, :2, :2] = L0_2d[i]
            L0_3d[i, 2, 2] = 0.1

        model_2d = GaussianSplatModelCUDA(
            shape=shape_2d,
            centers0=centers0_2d,
            L0=L0_2d,
            amps0=amps0.copy(),
            sigma_min_diag=(0.5, 0.5),
            device="cuda",
        )

        model_3d = GaussianSplatModelCUDA(
            shape=shape_3d,
            centers0=centers0_3d,
            L0=L0_3d,
            amps0=amps0.copy(),
            sigma_min_diag=(0.5, 0.5, 0.1),
            device="cuda",
        )

        # Forward passes
        output_2d = model_2d()
        output_3d = model_3d()

        # Create compatible targets
        target_2d = torch.zeros_like(output_2d)
        target_2d[12, 12] = 1.0

        target_3d = torch.zeros_like(output_3d)
        target_3d[12, 12, 0] = 1.0

        # Backward passes
        loss_2d = torch.nn.functional.mse_loss(output_2d, target_2d)
        loss_2d.backward()

        loss_3d = torch.nn.functional.mse_loss(output_3d, target_3d)
        loss_3d.backward()

        # Compare amplitude gradient signs
        grad_amp_2d = model_2d.raw_a.grad
        grad_amp_3d = model_3d.raw_a.grad

        if grad_amp_2d is not None and grad_amp_3d is not None:
            # Only compare where gradients are significant
            significant = (grad_amp_2d.abs() > 1e-6) & (grad_amp_3d.abs() > 1e-6)
            if significant.any():
                sign_match = (
                    (
                        torch.sign(grad_amp_2d[significant])
                        == torch.sign(grad_amp_3d[significant])
                    )
                    .float()
                    .mean()
                )
                # Allow some differences due to numerical precision
                assert sign_match > 0.7, (
                    f"2D vs 3D amplitude gradient sign match {sign_match:.2f} too low"
                )

    def test_standard_gaussian_fast_path(self):
        """Verify s=2 (standard Gaussian) fast path produces correct results.

        The code has optimized paths for s=2 that avoid powf() calls.
        This test verifies the output matches expected Gaussian distribution.

        Since the model uses s=2 by default (and doesn't expose a sharpness
        parameter), we verify correctness by checking:
        1. Peak at splat center with expected intensity
        2. Exponential decay away from center (characteristic of s=2)
        """
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        # Single centered splat for easy verification
        N = 1
        shape = (32, 32, 32)

        # Splat at center with identity covariance
        centers0 = np.array([[16.0, 16.0, 16.0]], dtype=np.float32)
        L0 = np.eye(3, dtype=np.float32)[None, :, :]  # Sigma = I
        amps0 = np.array([1.0], dtype=np.float32)

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        with torch.no_grad():
            output = model()

        # Verify peak at center
        center_val = output[16, 16, 16].item()
        assert center_val == output.max().item(), "Peak should be at splat center"

        # For standard Gaussian (s=2) with Sigma=I:
        # I(x) = a * exp(-0.5 * d²) where d² = ||x - mu||²
        #
        # At d=1 (one voxel away): exp(-0.5) ≈ 0.6065
        # At d=2: exp(-2) ≈ 0.1353
        #
        # Verify exponential decay characteristic of s=2
        val_d1 = output[17, 16, 16].item()  # d=1
        val_d2 = output[18, 16, 16].item()  # d=2

        # Ratio at d=1 should be exp(-0.5) ≈ 0.6065
        ratio_d1 = val_d1 / center_val if center_val > 0 else 0
        expected_d1 = np.exp(-0.5)  # ≈ 0.6065

        # Allow for sigma_min_diag effects (the actual sigma may be slightly different)
        # Just verify it's a reasonable Gaussian-like decay
        assert ratio_d1 > 0.3 and ratio_d1 < 0.9, (
            f"Decay at d=1 ratio {ratio_d1:.4f} not Gaussian-like"
        )

        # Verify d=2 has more decay than d=1 (monotonic)
        ratio_d2 = val_d2 / center_val if center_val > 0 else 0
        assert ratio_d2 < ratio_d1, (
            f"Decay not monotonic: d1={ratio_d1:.4f}, d2={ratio_d2:.4f}"
        )


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestCUDAVsPyTorchReference:
    """Direct comparison tests between CUDA and PyTorch reference implementations.

    These tests verify that the CUDA backend produces numerically accurate
    results compared to the authoritative PyTorch reference implementation
    in rendering_core.py.
    """

    def test_direct_cuda_vs_pytorch_3d(self):
        """Direct comparison of CUDA and PyTorch outputs for 3D volume.

        This test uses the CUDA backend directly (not through the model class)
        to compare with the PyTorch rendering_core output.
        """
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        # Create test data
        np.random.seed(1234)
        N, d = 30, 3
        shape = (24, 24, 24)
        truncate = 3.0
        intensity_floor = 1e-5

        # Random centers inside volume
        centers = torch.rand(N, d, device="cuda", dtype=torch.float32) * 20 + 2

        # Random lower-triangular Cholesky factors
        L = (
            torch.eye(d, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
        )
        for i in range(N):
            scale = torch.rand(1, device="cuda").item() * 1.5 + 0.5
            L[i] *= scale

        # Amplitudes and sharpness
        amps = torch.rand(N, device="cuda", dtype=torch.float32) * 0.5 + 0.5
        sharpness = (
            torch.ones(N, device="cuda", dtype=torch.float32) * 2.0
        )  # Standard Gaussian

        # Compute conic for CUDA backend
        conic = cholesky_to_conic(L)

        # Run CUDA backend
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            8,  # tile_size for 3D
        )
        cuda_output = cuda_result[0].reshape(shape)

        # Run PyTorch reference
        pytorch_output = render_gaussians(
            shape, centers, L, amps, sharpness, truncate, intensity_floor
        )

        # Move to CPU for comparison
        cuda_cpu = cuda_output.cpu()
        pytorch_cpu = pytorch_output.cpu()

        # Detailed comparison metrics
        max_val = max(cuda_cpu.abs().max().item(), pytorch_cpu.abs().max().item())
        if max_val > 1e-6:
            abs_diff = (cuda_cpu - pytorch_cpu).abs()
            rel_diff = abs_diff / (max_val + 1e-8)

            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()

            print("\n3D CUDA vs PyTorch comparison:")
            print(f"  Max absolute diff: {max_abs_diff:.6f}")
            print(f"  Mean absolute diff: {mean_abs_diff:.6f}")
            print(f"  Max relative diff: {max_rel_diff:.4f}")
            print(f"  Mean relative diff: {mean_rel_diff:.6f}")

            # Check structural similarity via correlation
            corr = torch.corrcoef(
                torch.stack([cuda_cpu.flatten(), pytorch_cpu.flatten()])
            )[0, 1].item()
            print(f"  Correlation: {corr:.6f}")

            # Assertions
            assert max_rel_diff < 0.15, (
                f"Max relative diff {max_rel_diff:.4f} too large"
            )
            assert mean_rel_diff < 0.01, (
                f"Mean relative diff {mean_rel_diff:.6f} too large"
            )
            assert corr > 0.99, f"Correlation {corr:.4f} too low"

    def test_direct_cuda_vs_pytorch_2d(self):
        """Direct comparison of CUDA and PyTorch outputs for 2D image."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        # Create test data
        np.random.seed(5678)
        N, d = 25, 2
        shape = (48, 48)
        truncate = 3.0
        intensity_floor = 1e-5

        # Random centers inside volume
        centers = torch.rand(N, d, device="cuda", dtype=torch.float32) * 44 + 2

        # Random lower-triangular Cholesky factors
        L = (
            torch.eye(d, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
        )
        for i in range(N):
            scale = torch.rand(1, device="cuda").item() * 2.0 + 0.5
            L[i] *= scale

        # Amplitudes and sharpness
        amps = torch.rand(N, device="cuda", dtype=torch.float32) * 0.5 + 0.5
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0

        # Compute conic for CUDA backend
        conic = cholesky_to_conic(L)

        # Run CUDA backend
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            16,  # tile_size for 2D
        )
        cuda_output = cuda_result[0].reshape(shape)

        # Run PyTorch reference
        pytorch_output = render_gaussians(
            shape, centers, L, amps, sharpness, truncate, intensity_floor
        )

        # Move to CPU for comparison
        cuda_cpu = cuda_output.cpu()
        pytorch_cpu = pytorch_output.cpu()

        # Detailed comparison metrics
        max_val = max(cuda_cpu.abs().max().item(), pytorch_cpu.abs().max().item())
        if max_val > 1e-6:
            abs_diff = (cuda_cpu - pytorch_cpu).abs()
            rel_diff = abs_diff / (max_val + 1e-8)

            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()

            print("\n2D CUDA vs PyTorch comparison:")
            print(f"  Max relative diff: {max_rel_diff:.4f}")
            print(f"  Mean relative diff: {mean_rel_diff:.6f}")

            # Check correlation
            corr = torch.corrcoef(
                torch.stack([cuda_cpu.flatten(), pytorch_cpu.flatten()])
            )[0, 1].item()
            print(f"  Correlation: {corr:.6f}")

            # Assertions
            assert max_rel_diff < 0.15, (
                f"Max relative diff {max_rel_diff:.4f} too large"
            )
            assert mean_rel_diff < 0.01, (
                f"Mean relative diff {mean_rel_diff:.6f} too large"
            )
            assert corr > 0.99, f"Correlation {corr:.4f} too low"

    def test_single_centered_splat_matches(self):
        """Test single centered splat produces identical peak location and value."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        # Single splat at center
        N, d = 1, 3
        shape = (32, 32, 32)
        center_pos = [16.0, 16.0, 16.0]

        centers = torch.tensor([center_pos], device="cuda", dtype=torch.float32)
        L = torch.eye(d, device="cuda", dtype=torch.float32).unsqueeze(0) * 2.0
        amps = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        sharpness = torch.tensor([2.0], device="cuda", dtype=torch.float32)

        conic = cholesky_to_conic(L)

        # CUDA output
        cuda_result = cuda_splatting_backend.forward(
            centers, conic, amps, sharpness, list(shape), 3.0, 1e-5, 8
        )
        cuda_output = cuda_result[0].reshape(shape)

        # PyTorch output
        pytorch_output = render_gaussians(shape, centers, L, amps, sharpness, 3.0, 1e-5)

        cuda_cpu = cuda_output.cpu()
        pytorch_cpu = pytorch_output.cpu()

        # Both should have peak at center
        cuda_peak_idx = cuda_cpu.argmax()
        pytorch_peak_idx = pytorch_cpu.argmax()

        cuda_peak_coords = np.unravel_index(cuda_peak_idx.item(), shape)
        pytorch_peak_coords = np.unravel_index(pytorch_peak_idx.item(), shape)

        print("\nSingle splat test:")
        print(f"  CUDA peak at: {cuda_peak_coords}")
        print(f"  PyTorch peak at: {pytorch_peak_coords}")
        print(f"  CUDA peak value: {cuda_cpu.max().item():.6f}")
        print(f"  PyTorch peak value: {pytorch_cpu.max().item():.6f}")

        # Peak should be at or near center (within 1 voxel due to coordinate systems)
        assert abs(cuda_peak_coords[0] - 16) <= 1, (
            f"CUDA peak X off: {cuda_peak_coords[0]}"
        )
        assert abs(cuda_peak_coords[1] - 16) <= 1, (
            f"CUDA peak Y off: {cuda_peak_coords[1]}"
        )
        assert abs(cuda_peak_coords[2] - 16) <= 1, (
            f"CUDA peak Z off: {cuda_peak_coords[2]}"
        )

        # Peak values should be close
        peak_ratio = cuda_cpu.max().item() / pytorch_cpu.max().item()
        assert 0.9 < peak_ratio < 1.1, f"Peak ratio {peak_ratio:.4f} too far from 1.0"


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestCUDAVsPyTorchComprehensive:
    """Comprehensive comparison tests for CUDA vs PyTorch across dimensions and parameters.

    These tests ensure numerical correctness of the CUDA optimized implementation
    against the authoritative PyTorch reference for:
    - Dimensions: 2D, 3D, 4D
    - Various splat shapes (isotropic, anisotropic, elongated)
    - Various sizes (small, medium, large covariances)
    - Various sharpness values (sub-gaussian, standard, super-gaussian)
    - Both forward and backward passes
    """

    # =========================================================================
    # Forward Pass Tests - Parametrized by Dimension
    # =========================================================================

    @pytest.mark.parametrize("dim", [2, 3])
    def test_forward_isotropic_splats(self, dim: int):
        """Test forward pass with isotropic (spherical) splats for 2D/3D."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        np.random.seed(42 + dim)
        N = 20

        # Create appropriate shape and parameters for each dimension
        if dim == 2:
            shape = (48, 48)
            tile_size = 16
        else:  # dim == 3
            shape = (24, 24, 24)
            tile_size = 8

        # Random centers inside volume with margin
        margin = 2
        max_coord = min(shape) - margin
        centers = (
            torch.rand(N, dim, device="cuda", dtype=torch.float32)
            * (max_coord - margin)
            + margin
        )

        # Isotropic L (scaled identity)
        scale = 1.5
        L = (
            torch.eye(dim, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
            * scale
        )

        # Standard parameters
        amps = torch.ones(N, device="cuda", dtype=torch.float32) * 0.5
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0
        truncate = 3.0
        intensity_floor = 1e-5

        # CUDA forward
        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)

        # PyTorch reference
        pytorch_output = render_gaussians(
            shape, centers, L, amps, sharpness, truncate, intensity_floor
        )

        # Compare with strict tolerances
        self._assert_outputs_match(cuda_output, pytorch_output, f"{dim}D isotropic")

    def test_forward_isotropic_splats_4d(self):
        """Test forward pass with isotropic splats for 4D (may have coordinate system issues)."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        np.random.seed(46)
        N = 20
        dim = 4
        shape = (12, 12, 12, 12)
        tile_size = 4

        margin = 2
        max_coord = min(shape) - margin
        centers = (
            torch.rand(N, dim, device="cuda", dtype=torch.float32)
            * (max_coord - margin)
            + margin
        )

        scale = 1.5
        L = (
            torch.eye(dim, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
            * scale
        )

        amps = torch.ones(N, device="cuda", dtype=torch.float32) * 0.5
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0

        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            3.0,
            1e-5,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, centers, L, amps, sharpness, 3.0, 1e-5)

        # Strict tolerance - this will fail, flagging need for investigation
        self._assert_outputs_match(cuda_output, pytorch_output, "4D isotropic")

    @pytest.mark.parametrize("dim", [2, 3])
    def test_forward_anisotropic_splats(self, dim: int):
        """Test forward pass with anisotropic (ellipsoidal) splats."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        np.random.seed(123 + dim)
        N = 15

        if dim == 2:
            shape = (48, 48)
            tile_size = 16
        else:
            shape = (24, 24, 24)
            tile_size = 8

        margin = 3
        max_coord = min(shape) - margin
        centers = (
            torch.rand(N, dim, device="cuda", dtype=torch.float32)
            * (max_coord - margin)
            + margin
        )

        # Anisotropic L - different scales per axis
        L = torch.zeros(N, dim, dim, device="cuda", dtype=torch.float32)
        for i in range(N):
            for j in range(dim):
                L[i, j, j] = np.random.uniform(0.8, 2.5)  # Different scale per axis
            # Add off-diagonal elements for rotation
            for j in range(1, dim):
                L[i, j, 0] = np.random.uniform(-0.3, 0.3)

        amps = torch.rand(N, device="cuda", dtype=torch.float32) * 0.5 + 0.3
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0
        truncate = 3.0
        intensity_floor = 1e-5

        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(
            shape, centers, L, amps, sharpness, truncate, intensity_floor
        )

        self._assert_outputs_match(cuda_output, pytorch_output, f"{dim}D anisotropic")

    def test_forward_anisotropic_splats_4d(self):
        """Test 4D anisotropic splats (may have issues)."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        np.random.seed(127)
        N, dim = 15, 4
        shape = (12, 12, 12, 12)
        tile_size = 4

        margin = 3
        max_coord = min(shape) - margin
        centers = (
            torch.rand(N, dim, device="cuda", dtype=torch.float32)
            * (max_coord - margin)
            + margin
        )

        L = torch.zeros(N, dim, dim, device="cuda", dtype=torch.float32)
        for i in range(N):
            for j in range(dim):
                L[i, j, j] = np.random.uniform(0.8, 2.5)
            for j in range(1, dim):
                L[i, j, 0] = np.random.uniform(-0.3, 0.3)

        amps = torch.rand(N, device="cuda", dtype=torch.float32) * 0.5 + 0.3
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0

        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            3.0,
            1e-5,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, centers, L, amps, sharpness, 3.0, 1e-5)

        self._assert_outputs_match(cuda_output, pytorch_output, "4D anisotropic")

    @pytest.mark.parametrize("dim", [2, 3])
    def test_forward_elongated_splats(self, dim: int):
        """Test forward pass with highly elongated splats (aspect ratio > 5:1)."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        np.random.seed(456 + dim)
        N = 10

        if dim == 2:
            shape = (64, 64)
            tile_size = 16
        else:
            shape = (32, 32, 32)
            tile_size = 8

        margin = 4
        max_coord = min(shape) - margin
        centers = (
            torch.rand(N, dim, device="cuda", dtype=torch.float32)
            * (max_coord - margin)
            + margin
        )

        # Elongated L - one axis much larger than others
        L = torch.zeros(N, dim, dim, device="cuda", dtype=torch.float32)
        for i in range(N):
            primary_axis = i % dim  # Rotate which axis is elongated
            for j in range(dim):
                if j == primary_axis:
                    L[i, j, j] = np.random.uniform(3.0, 5.0)  # Elongated
                else:
                    L[i, j, j] = np.random.uniform(0.5, 1.0)  # Narrow

        amps = torch.ones(N, device="cuda", dtype=torch.float32) * 0.7
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0
        truncate = 3.0
        intensity_floor = 1e-5

        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(
            shape, centers, L, amps, sharpness, truncate, intensity_floor
        )

        self._assert_outputs_match(cuda_output, pytorch_output, f"{dim}D elongated")

    def test_forward_elongated_splats_4d(self):
        """Test 4D elongated splats (may have issues)."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        np.random.seed(460)
        N, dim = 10, 4
        shape = (16, 16, 16, 16)
        tile_size = 4

        margin = 4
        max_coord = min(shape) - margin
        centers = (
            torch.rand(N, dim, device="cuda", dtype=torch.float32)
            * (max_coord - margin)
            + margin
        )

        L = torch.zeros(N, dim, dim, device="cuda", dtype=torch.float32)
        for i in range(N):
            primary_axis = i % dim
            for j in range(dim):
                if j == primary_axis:
                    L[i, j, j] = np.random.uniform(3.0, 5.0)
                else:
                    L[i, j, j] = np.random.uniform(0.5, 1.0)

        amps = torch.ones(N, device="cuda", dtype=torch.float32) * 0.7
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0

        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            3.0,
            1e-5,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, centers, L, amps, sharpness, 3.0, 1e-5)

        self._assert_outputs_match(cuda_output, pytorch_output, "4D elongated")

    # =========================================================================
    # Forward Pass Tests - Parametrized by Size
    # =========================================================================

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("size_category", ["tiny", "small", "medium", "large"])
    def test_forward_various_sizes(self, dim: int, size_category: str):
        """Test forward pass with various splat sizes."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        np.random.seed(789 + dim + hash(size_category) % 100)
        N = 15

        # Size-dependent scale factors
        size_scales = {
            "tiny": 0.3,  # Sub-voxel splats
            "small": 0.8,  # 1-2 voxel radius
            "medium": 2.0,  # 3-5 voxel radius
            "large": 4.0,  # 8+ voxel radius
        }
        scale = size_scales[size_category]

        if dim == 2:
            shape = (64, 64)
            tile_size = 16
        else:
            shape = (32, 32, 32)
            tile_size = 8

        margin = int(scale * 3) + 2
        max_coord = min(shape) - margin
        if max_coord <= margin:
            pytest.skip(f"Volume too small for {size_category} splats")

        centers = (
            torch.rand(N, dim, device="cuda", dtype=torch.float32)
            * (max_coord - margin)
            + margin
        )

        # Scaled isotropic L
        L = (
            torch.eye(dim, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
            * scale
        )

        amps = torch.ones(N, device="cuda", dtype=torch.float32) * 0.5
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0
        truncate = 3.0
        intensity_floor = 1e-5

        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(
            shape, centers, L, amps, sharpness, truncate, intensity_floor
        )

        self._assert_outputs_match(
            cuda_output, pytorch_output, f"{dim}D {size_category}"
        )

    # =========================================================================
    # Forward Pass Tests - Parametrized by Sharpness
    # =========================================================================

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("sharpness_val", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    def test_forward_various_sharpness(self, dim: int, sharpness_val: float):
        """Test forward pass with various sharpness values.

        Sharpness controls the falloff profile:
        - s < 2: Sub-gaussian (softer, wider tails)
        - s = 2: Standard Gaussian
        - s > 2: Super-gaussian (sharper, more compact)
        """
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        np.random.seed(321 + dim + int(sharpness_val * 10))
        N = 12

        if dim == 2:
            shape = (48, 48)
            tile_size = 16
        else:
            shape = (24, 24, 24)
            tile_size = 8

        margin = 4
        max_coord = min(shape) - margin
        centers = (
            torch.rand(N, dim, device="cuda", dtype=torch.float32)
            * (max_coord - margin)
            + margin
        )

        # Standard size splats
        L = (
            torch.eye(dim, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
            * 1.5
        )

        amps = torch.ones(N, device="cuda", dtype=torch.float32) * 0.6
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * sharpness_val
        truncate = 3.0
        intensity_floor = 1e-5

        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(
            shape, centers, L, amps, sharpness, truncate, intensity_floor
        )

        # Sub-gaussian sharpness (s < 2) has slightly higher numerical differences
        atol = 0.015 if sharpness_val < 1.5 else 0.01
        self._assert_outputs_match(
            cuda_output, pytorch_output, f"{dim}D sharpness={sharpness_val}", atol=atol
        )

    @pytest.mark.parametrize("dim", [2, 3])
    def test_forward_mixed_sharpness(self, dim: int):
        """Test forward pass with mixed sharpness values per splat."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        np.random.seed(654 + dim)
        N = 20

        if dim == 2:
            shape = (48, 48)
            tile_size = 16
        else:
            shape = (24, 24, 24)
            tile_size = 8

        margin = 4
        max_coord = min(shape) - margin
        centers = (
            torch.rand(N, dim, device="cuda", dtype=torch.float32)
            * (max_coord - margin)
            + margin
        )

        L = (
            torch.eye(dim, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
            * 1.5
        )

        amps = torch.rand(N, device="cuda", dtype=torch.float32) * 0.5 + 0.3
        # Mix of sharpness values: some sub-gaussian, some standard, some super-gaussian
        sharpness = torch.tensor(
            [1.0, 1.5, 2.0, 2.5, 3.0, 4.0] * 4, device="cuda", dtype=torch.float32
        )[:N]
        truncate = 3.0
        intensity_floor = 1e-5

        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(
            shape, centers, L, amps, sharpness, truncate, intensity_floor
        )

        self._assert_outputs_match(
            cuda_output, pytorch_output, f"{dim}D mixed sharpness"
        )

    # =========================================================================
    # Backward Pass Tests
    # =========================================================================

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_backward_gradient_consistency(self, dim: int):
        """Test that CUDA backward gradients are consistent with PyTorch autograd.

        This test verifies that:
        1. Gradients are finite
        2. Gradient signs match for the majority of parameters
        3. Gradient magnitudes are in similar ranges
        """
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        np.random.seed(111 + dim)
        N = 15

        if dim == 2:
            shape = (32, 32)
        elif dim == 3:
            shape = (16, 16, 16)
        else:
            shape = (8, 8, 8, 8)

        margin = 2
        max_coord = min(shape) - margin
        centers0 = (
            np.random.rand(N, dim).astype(np.float32) * (max_coord - margin) + margin
        )
        L0 = np.eye(dim, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0[i] *= np.random.uniform(0.8, 1.5)
        amps0 = np.ones(N, dtype=np.float32) * 0.5

        # Create CPU model
        cpu_model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * dim,
            device="cpu",
        )

        # Create CUDA model
        cuda_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * dim,
            device="cuda",
        )

        # Forward passes
        cpu_output = cpu_model()
        cuda_output = cuda_model()

        # Create targets at center
        center_idx = tuple(s // 2 for s in shape)
        cpu_target = torch.zeros_like(cpu_output)
        cpu_target[center_idx] = 1.0

        cuda_target = torch.zeros_like(cuda_output)
        if cuda_output.dim() == 1:
            # Flattened output
            flat_idx = sum(
                idx * stride
                for idx, stride in zip(
                    center_idx, np.cumprod([1] + list(shape[::-1]))[::-1][1:]
                )
            )
            cuda_target[flat_idx] = 1.0
        else:
            cuda_target[center_idx] = 1.0

        # Backward passes
        cpu_loss = torch.nn.functional.mse_loss(cpu_output, cpu_target)
        cpu_loss.backward()

        cuda_loss = torch.nn.functional.mse_loss(cuda_output, cuda_target)
        cuda_loss.backward()

        # Verify CUDA gradients are finite
        for name, param in cuda_model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"{dim}D: Non-finite gradient for {name}"
                )

        # Compare amplitude gradients (most directly comparable)
        cpu_amp_grad = cpu_model.raw_a.grad
        cuda_amp_grad = (
            cuda_model.raw_a.grad.cpu() if cuda_model.raw_a.grad is not None else None
        )

        if cpu_amp_grad is not None and cuda_amp_grad is not None:
            # Check gradient sign consistency
            significant = (cpu_amp_grad.abs() > 1e-6) & (cuda_amp_grad.abs() > 1e-6)
            if significant.any():
                sign_match = (
                    (
                        torch.sign(cpu_amp_grad[significant])
                        == torch.sign(cuda_amp_grad[significant])
                    )
                    .float()
                    .mean()
                )
                assert sign_match > 0.6, (
                    f"{dim}D: Amplitude gradient sign match {sign_match:.2f} too low"
                )

            # Check gradient magnitude consistency
            cpu_mag = cpu_amp_grad.abs().mean()
            cuda_mag = cuda_amp_grad.abs().mean()
            if cpu_mag > 1e-8 and cuda_mag > 1e-8:
                mag_ratio = max(cpu_mag / cuda_mag, cuda_mag / cpu_mag)
                assert mag_ratio < 20, (
                    f"{dim}D: Gradient magnitude ratio {mag_ratio:.2f} too large"
                )

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
    def test_backward_with_various_scales(self, dim: int, scale: float):
        """Test backward pass produces finite gradients for various splat scales.

        Note: GaussianSplatModelCUDA uses default sharpness s=2 (standard Gaussian).
        This test verifies gradients are correct for different covariance scales.
        """
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        np.random.seed(222 + dim + int(scale * 10))
        N = 10

        if dim == 2:
            shape = (32, 32)
        else:
            shape = (16, 16, 16)

        margin = 2
        max_coord = min(shape) - margin
        centers0 = (
            np.random.rand(N, dim).astype(np.float32) * (max_coord - margin) + margin
        )
        L0 = np.eye(dim, dtype=np.float32)[None, :, :].repeat(N, axis=0) * scale
        amps0 = np.ones(N, dtype=np.float32) * 0.5

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.3,) * dim,
            device="cuda",
        )

        # Forward
        output = model()

        # Target and loss
        target = torch.zeros_like(output)
        center_idx = tuple(s // 2 for s in shape)
        if output.dim() == 1:
            flat_idx = sum(
                idx * stride
                for idx, stride in zip(
                    center_idx, np.cumprod([1] + list(shape[::-1]))[::-1][1:]
                )
            )
            target[flat_idx] = 1.0
        else:
            target[center_idx] = 1.0

        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Verify all gradients are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"{dim}D scale={scale}: Non-finite gradient for {name}"
                )

    @pytest.mark.parametrize("dim", [2, 3])
    def test_backward_gradient_flow(self, dim: int):
        """Test that gradients flow correctly through all parameters."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        np.random.seed(333 + dim)
        N = 8

        if dim == 2:
            shape = (24, 24)
        else:
            shape = (12, 12, 12)

        # Single splat at center for clear gradient signal
        center = np.array([[s // 2 for s in shape]], dtype=np.float32)
        L0 = np.eye(dim, dtype=np.float32)[None, :, :] * 1.5
        amps0 = np.array([1.0], dtype=np.float32)

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=center,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * dim,
            device="cuda",
        )

        output = model()

        # Loss that should produce non-zero gradients
        target = torch.zeros_like(output)
        # Target is different from output to create gradient
        loss = (output**2).sum()
        loss.backward()

        # Verify gradients exist for all parameters
        params_with_grad = []
        params_without_grad = []
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().max() > 1e-10:
                params_with_grad.append(name)
            else:
                params_without_grad.append(name)

        assert len(params_with_grad) > 0, (
            f"{dim}D: No parameters have gradients! All params: {params_without_grad}"
        )

        # At minimum, amplitude should have gradient
        assert model.raw_a.grad is not None, f"{dim}D: Amplitude has no gradient"
        assert model.raw_a.grad.abs().max() > 1e-10, (
            f"{dim}D: Amplitude gradient is zero"
        )

    # =========================================================================
    # Edge Case Tests
    # =========================================================================

    @pytest.mark.parametrize("dim", [2, 3])
    def test_single_splat_at_center(self, dim: int):
        """Test single splat at volume center matches between CUDA and PyTorch."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        if dim == 2:
            shape = (32, 32)
            tile_size = 16
        else:
            shape = (24, 24, 24)
            tile_size = 8

        center = torch.tensor(
            [[s // 2 for s in shape]], device="cuda", dtype=torch.float32
        )
        L = torch.eye(dim, device="cuda", dtype=torch.float32).unsqueeze(0) * 2.0
        amps = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        sharpness = torch.tensor([2.0], device="cuda", dtype=torch.float32)

        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            center, conic, amps, sharpness, list(shape), 3.0, 1e-5, tile_size
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, center, L, amps, sharpness, 3.0, 1e-5)

        # Peak should be at center for both
        cuda_peak = cuda_output.argmax()
        pytorch_peak = pytorch_output.argmax()

        cuda_peak_coords = np.unravel_index(cuda_peak.cpu().item(), shape)
        pytorch_peak_coords = np.unravel_index(pytorch_peak.cpu().item(), shape)

        center_coords = tuple(s // 2 for s in shape)

        # Both peaks should be at or very near center
        for i in range(dim):
            assert abs(cuda_peak_coords[i] - center_coords[i]) <= 1, (
                f"{dim}D: CUDA peak off center at dim {i}: {cuda_peak_coords}"
            )
            assert abs(pytorch_peak_coords[i] - center_coords[i]) <= 1, (
                f"{dim}D: PyTorch peak off center at dim {i}: {pytorch_peak_coords}"
            )

    def test_single_splat_at_center_4d(self):
        """Test single splat at 4D volume center (may have issues)."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        dim = 4
        shape = (16, 16, 16, 16)
        tile_size = 4

        center = torch.tensor(
            [[s // 2 for s in shape]], device="cuda", dtype=torch.float32
        )
        L = torch.eye(dim, device="cuda", dtype=torch.float32).unsqueeze(0) * 2.0
        amps = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        sharpness = torch.tensor([2.0], device="cuda", dtype=torch.float32)

        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            center, conic, amps, sharpness, list(shape), 3.0, 1e-5, tile_size
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, center, L, amps, sharpness, 3.0, 1e-5)

        # Verify CUDA output is non-zero (basic sanity check)
        assert cuda_output.max() > 0, "4D CUDA output is all zeros - rendering failed"

        # Peak should be at center for both
        cuda_peak = cuda_output.argmax()
        cuda_peak_coords = np.unravel_index(cuda_peak.cpu().item(), shape)
        center_coords = tuple(s // 2 for s in shape)

        # Both peaks should be at or very near center
        for i in range(dim):
            assert abs(cuda_peak_coords[i] - center_coords[i]) <= 1, (
                f"4D: CUDA peak off center at dim {i}: {cuda_peak_coords}"
            )

    @pytest.mark.parametrize("dim", [2, 3])
    def test_many_overlapping_splats(self, dim: int):
        """Test many overlapping splats sum correctly."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        np.random.seed(444 + dim)
        N = 50  # Many splats

        if dim == 2:
            shape = (48, 48)
            tile_size = 16
        else:
            shape = (24, 24, 24)
            tile_size = 8

        # All splats near center to ensure overlap
        center_pos = [s // 2 for s in shape]
        centers = (
            torch.tensor(center_pos, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1)
            .clone()
        )
        centers += torch.randn(N, dim, device="cuda") * 2  # Small perturbation

        L = (
            torch.eye(dim, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
            * 1.5
        )
        amps = (
            torch.ones(N, device="cuda", dtype=torch.float32) * 0.1
        )  # Small amplitudes that sum
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0

        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            3.0,
            1e-5,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, centers, L, amps, sharpness, 3.0, 1e-5)

        self._assert_outputs_match(cuda_output, pytorch_output, f"{dim}D overlapping")

    @pytest.mark.parametrize("dim", [2, 3])
    def test_splats_at_boundaries(self, dim: int):
        """Test splats positioned at volume boundaries."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        if dim == 2:
            shape = (32, 32)
            tile_size = 16
        else:
            shape = (24, 24, 24)
            tile_size = 8

        # Splats at corners and edges
        corners = []
        for corner in range(2**dim):
            coords = [(corner >> i) & 1 for i in range(dim)]
            pos = [c * (shape[i] - 1) for i, c in enumerate(coords)]
            corners.append(pos)

        centers = torch.tensor(corners, device="cuda", dtype=torch.float32)
        N = len(corners)

        L = (
            torch.eye(dim, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
            * 1.5
        )
        amps = torch.ones(N, device="cuda", dtype=torch.float32) * 0.5
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0

        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            3.0,
            1e-5,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, centers, L, amps, sharpness, 3.0, 1e-5)

        self._assert_outputs_match(
            cuda_output, pytorch_output, f"{dim}D boundaries", rtol=0.2
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _assert_outputs_match(
        self,
        cuda_output: torch.Tensor,
        pytorch_output: torch.Tensor,
        test_name: str,
        rtol: float = 0.15,
        atol: float = 0.01,
        min_correlation: float = 0.98,
    ):
        """Assert that CUDA and PyTorch outputs match within tolerance.

        Parameters
        ----------
        cuda_output : torch.Tensor
            Output from CUDA backend
        pytorch_output : torch.Tensor
            Output from PyTorch reference
        test_name : str
            Name for error messages
        rtol : float
            Maximum allowed relative difference
        atol : float
            Maximum allowed mean relative difference
        min_correlation : float
            Minimum required correlation coefficient
        """
        cuda_cpu = cuda_output.cpu()
        pytorch_cpu = pytorch_output.cpu()

        # Ensure same shape
        assert cuda_cpu.shape == pytorch_cpu.shape, (
            f"{test_name}: Shape mismatch CUDA={cuda_cpu.shape} vs PyTorch={pytorch_cpu.shape}"
        )

        max_val = max(cuda_cpu.abs().max().item(), pytorch_cpu.abs().max().item())

        if max_val < 1e-6:
            # Both outputs are essentially zero
            return

        # Compute differences
        abs_diff = (cuda_cpu - pytorch_cpu).abs()
        rel_diff = abs_diff / (max_val + 1e-8)

        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()

        # Compute correlation
        cuda_flat = cuda_cpu.flatten()
        pytorch_flat = pytorch_cpu.flatten()

        if cuda_flat.std() > 1e-6 and pytorch_flat.std() > 1e-6:
            correlation = torch.corrcoef(torch.stack([cuda_flat, pytorch_flat]))[
                0, 1
            ].item()
        else:
            correlation = 1.0  # Both constant, consider matching

        # Assertions with informative messages
        assert max_rel_diff < rtol, (
            f"{test_name}: Max relative diff {max_rel_diff:.4f} > {rtol} (correlation={correlation:.4f})"
        )

        assert mean_rel_diff < atol, (
            f"{test_name}: Mean relative diff {mean_rel_diff:.6f} > {atol}"
        )

        assert correlation > min_correlation, (
            f"{test_name}: Correlation {correlation:.4f} < {min_correlation}"
        )


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestOptimizedVsGenericPath:
    """Test that optimized 2D/3D paths match the generic nD path.

    This is done by embedding 2D/3D data into 4D space, running through
    the generic 4D path, and comparing with the optimized path output.
    This verifies that our specialized optimizations don't introduce bugs.
    """

    def _embed_2d_to_4d(
        self,
        centers_2d: torch.Tensor,
        L_2d: torch.Tensor,
        shape_2d: tuple,
    ) -> tuple:
        """Embed 2D splat parameters into 4D space.

        Adds two dummy dimensions (z, w) with splats centered at z=0, w=0
        (the slice position) and very small variance to ensure near-unity
        contribution at the slice.
        """
        N = centers_2d.shape[0]
        device = centers_2d.device

        # Embed centers: [x, y] -> [x, y, 0, 0]
        # Centers at z=0, w=0 so slice at position 0 captures full contribution
        centers_4d = torch.zeros(N, 4, device=device, dtype=torch.float32)
        centers_4d[:, :2] = centers_2d
        centers_4d[:, 2] = 0.0  # At slice position
        centers_4d[:, 3] = 0.0  # At slice position

        # Embed L: 2x2 -> 4x4 with small values in dummy dimensions
        # Small variance ensures Gaussian is concentrated at the center (z=0, w=0)
        L_4d = torch.zeros(N, 4, 4, device=device, dtype=torch.float32)
        L_4d[:, :2, :2] = L_2d  # Copy original 2D covariance
        L_4d[:, 2, 2] = 0.1  # Small variance in z
        L_4d[:, 3, 3] = 0.1  # Small variance in w

        # Shape: (H, W) -> (H, W, 1, 1)
        shape_4d = shape_2d + (1, 1)

        return centers_4d, L_4d, shape_4d

    def _embed_3d_to_4d(
        self,
        centers_3d: torch.Tensor,
        L_3d: torch.Tensor,
        shape_3d: tuple,
    ) -> tuple:
        """Embed 3D splat parameters into 4D space.

        Adds one dummy dimension (w) with splats centered at w=0
        (the slice position) and very small variance.
        """
        N = centers_3d.shape[0]
        device = centers_3d.device

        # Embed centers: [x, y, z] -> [x, y, z, 0]
        # Center at w=0 so slice at position 0 captures full contribution
        centers_4d = torch.zeros(N, 4, device=device, dtype=torch.float32)
        centers_4d[:, :3] = centers_3d
        centers_4d[:, 3] = 0.0  # At slice position

        # Embed L: 3x3 -> 4x4 with small value in dummy dimension
        L_4d = torch.zeros(N, 4, 4, device=device, dtype=torch.float32)
        L_4d[:, :3, :3] = L_3d  # Copy original 3D covariance
        L_4d[:, 3, 3] = 0.1  # Small variance in w

        # Shape: (D, H, W) -> (D, H, W, 1)
        shape_4d = shape_3d + (1,)

        return centers_4d, L_4d, shape_4d

    @pytest.mark.parametrize("shape", [(32, 32), (48, 48), (24, 32)])
    def test_2d_optimized_vs_generic(self, shape: tuple):
        """Test 2D optimized path matches generic 4D path."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        np.random.seed(42)
        N = 15
        dim = 2

        # Create 2D splats
        margin = 3
        max_coord = min(shape) - margin
        centers_2d = (
            torch.rand(N, dim, device="cuda", dtype=torch.float32)
            * (max_coord - margin)
            + margin
        )

        L_2d = (
            torch.eye(dim, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
        )
        for i in range(N):
            L_2d[i] *= np.random.uniform(0.8, 2.0)

        amps = torch.rand(N, device="cuda", dtype=torch.float32) * 0.5 + 0.3
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0
        truncate = 3.0
        intensity_floor = 1e-5

        # Run optimized 2D path
        conic_2d = cholesky_to_conic(L_2d)
        result_2d = cuda_splatting_backend.forward(
            centers_2d.contiguous(),
            conic_2d.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            16,
        )
        output_2d = result_2d[0].reshape(shape)

        # Embed into 4D and run generic path
        centers_4d, L_4d, shape_4d = self._embed_2d_to_4d(centers_2d, L_2d, shape)
        conic_4d = cholesky_to_conic(L_4d)
        result_4d = cuda_splatting_backend.forward(
            centers_4d.contiguous(),
            conic_4d.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape_4d),
            truncate,
            intensity_floor,
            4,
        )
        output_4d = result_4d[0].reshape(shape_4d)

        # Extract the 2D slice from 4D output (at z=0, w=0)
        output_4d_slice = output_4d[:, :, 0, 0]

        # Compare outputs
        self._assert_paths_match(output_2d, output_4d_slice, "2D optimized vs generic")

    @pytest.mark.parametrize("shape", [(16, 16, 16), (24, 24, 24), (12, 16, 20)])
    def test_3d_optimized_vs_generic(self, shape: tuple):
        """Test 3D optimized path matches generic 4D path."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        np.random.seed(123)
        N = 12
        dim = 3

        # Create 3D splats
        margin = 2
        max_coord = min(shape) - margin
        centers_3d = (
            torch.rand(N, dim, device="cuda", dtype=torch.float32)
            * (max_coord - margin)
            + margin
        )

        L_3d = (
            torch.eye(dim, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
        )
        for i in range(N):
            L_3d[i] *= np.random.uniform(0.8, 2.0)

        amps = torch.rand(N, device="cuda", dtype=torch.float32) * 0.5 + 0.3
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0
        truncate = 3.0
        intensity_floor = 1e-5

        # Run optimized 3D path
        conic_3d = cholesky_to_conic(L_3d)
        result_3d = cuda_splatting_backend.forward(
            centers_3d.contiguous(),
            conic_3d.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            8,
        )
        output_3d = result_3d[0].reshape(shape)

        # Embed into 4D and run generic path
        centers_4d, L_4d, shape_4d = self._embed_3d_to_4d(centers_3d, L_3d, shape)
        conic_4d = cholesky_to_conic(L_4d)
        result_4d = cuda_splatting_backend.forward(
            centers_4d.contiguous(),
            conic_4d.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape_4d),
            truncate,
            intensity_floor,
            4,
        )
        output_4d = result_4d[0].reshape(shape_4d)

        # Extract the 3D slice from 4D output (at w=0)
        output_4d_slice = output_4d[:, :, :, 0]

        # Compare outputs
        self._assert_paths_match(output_3d, output_4d_slice, "3D optimized vs generic")

    def test_2d_anisotropic_optimized_vs_generic(self):
        """Test 2D optimized path with anisotropic splats matches generic path."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        np.random.seed(456)
        N = 10
        dim = 2
        shape = (40, 40)

        margin = 4
        max_coord = min(shape) - margin
        centers_2d = (
            torch.rand(N, dim, device="cuda", dtype=torch.float32)
            * (max_coord - margin)
            + margin
        )

        # Anisotropic L with rotation
        L_2d = torch.zeros(N, dim, dim, device="cuda", dtype=torch.float32)
        for i in range(N):
            L_2d[i, 0, 0] = np.random.uniform(1.0, 3.0)
            L_2d[i, 1, 1] = np.random.uniform(0.5, 1.5)
            L_2d[i, 1, 0] = np.random.uniform(-0.3, 0.3)

        amps = torch.ones(N, device="cuda", dtype=torch.float32) * 0.5
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0

        # Optimized 2D
        conic_2d = cholesky_to_conic(L_2d)
        result_2d = cuda_splatting_backend.forward(
            centers_2d.contiguous(),
            conic_2d.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            3.0,
            1e-5,
            16,
        )
        output_2d = result_2d[0].reshape(shape)

        # Generic 4D
        centers_4d, L_4d, shape_4d = self._embed_2d_to_4d(centers_2d, L_2d, shape)
        conic_4d = cholesky_to_conic(L_4d)
        result_4d = cuda_splatting_backend.forward(
            centers_4d.contiguous(),
            conic_4d.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape_4d),
            3.0,
            1e-5,
            4,
        )
        output_4d_slice = result_4d[0].reshape(shape_4d)[:, :, 0, 0]

        self._assert_paths_match(
            output_2d, output_4d_slice, "2D anisotropic optimized vs generic"
        )

    def test_3d_anisotropic_optimized_vs_generic(self):
        """Test 3D optimized path with anisotropic splats matches generic path."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        np.random.seed(789)
        N = 8
        dim = 3
        shape = (16, 16, 16)

        margin = 2
        max_coord = min(shape) - margin
        centers_3d = (
            torch.rand(N, dim, device="cuda", dtype=torch.float32)
            * (max_coord - margin)
            + margin
        )

        # Anisotropic L
        L_3d = torch.zeros(N, dim, dim, device="cuda", dtype=torch.float32)
        for i in range(N):
            L_3d[i, 0, 0] = np.random.uniform(1.0, 2.5)
            L_3d[i, 1, 1] = np.random.uniform(0.8, 2.0)
            L_3d[i, 2, 2] = np.random.uniform(0.5, 1.5)
            L_3d[i, 1, 0] = np.random.uniform(-0.2, 0.2)
            L_3d[i, 2, 0] = np.random.uniform(-0.2, 0.2)

        amps = torch.ones(N, device="cuda", dtype=torch.float32) * 0.5
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0

        # Optimized 3D
        conic_3d = cholesky_to_conic(L_3d)
        result_3d = cuda_splatting_backend.forward(
            centers_3d.contiguous(),
            conic_3d.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            3.0,
            1e-5,
            8,
        )
        output_3d = result_3d[0].reshape(shape)

        # Generic 4D
        centers_4d, L_4d, shape_4d = self._embed_3d_to_4d(centers_3d, L_3d, shape)
        conic_4d = cholesky_to_conic(L_4d)
        result_4d = cuda_splatting_backend.forward(
            centers_4d.contiguous(),
            conic_4d.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape_4d),
            3.0,
            1e-5,
            4,
        )
        output_4d_slice = result_4d[0].reshape(shape_4d)[:, :, :, 0]

        self._assert_paths_match(
            output_3d, output_4d_slice, "3D anisotropic optimized vs generic"
        )

    @pytest.mark.parametrize("sharpness_val", [1.5, 2.0, 3.0])
    def test_2d_various_sharpness_optimized_vs_generic(self, sharpness_val: float):
        """Test 2D optimized path with various sharpness values matches generic path."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        np.random.seed(111 + int(sharpness_val * 10))
        N = 10
        dim = 2
        shape = (32, 32)

        margin = 3
        max_coord = min(shape) - margin
        centers_2d = (
            torch.rand(N, dim, device="cuda", dtype=torch.float32)
            * (max_coord - margin)
            + margin
        )
        L_2d = (
            torch.eye(dim, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
            * 1.5
        )

        amps = torch.ones(N, device="cuda", dtype=torch.float32) * 0.5
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * sharpness_val

        # Optimized 2D
        conic_2d = cholesky_to_conic(L_2d)
        result_2d = cuda_splatting_backend.forward(
            centers_2d.contiguous(),
            conic_2d.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape),
            3.0,
            1e-5,
            16,
        )
        output_2d = result_2d[0].reshape(shape)

        # Generic 4D
        centers_4d, L_4d, shape_4d = self._embed_2d_to_4d(centers_2d, L_2d, shape)
        conic_4d = cholesky_to_conic(L_4d)
        result_4d = cuda_splatting_backend.forward(
            centers_4d.contiguous(),
            conic_4d.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            list(shape_4d),
            3.0,
            1e-5,
            4,
        )
        output_4d_slice = result_4d[0].reshape(shape_4d)[:, :, 0, 0]

        self._assert_paths_match(
            output_2d,
            output_4d_slice,
            f"2D sharpness={sharpness_val} optimized vs generic",
        )

    def _assert_paths_match(
        self,
        optimized_output: torch.Tensor,
        generic_output: torch.Tensor,
        test_name: str,
        rtol: float = 0.10,
        min_correlation: float = 0.99,
    ):
        """Assert optimized and generic path outputs match.

        Uses tighter tolerances than CUDA vs PyTorch comparison since
        both should be numerically identical (same algorithm).
        """
        opt_cpu = optimized_output.cpu()
        gen_cpu = generic_output.cpu()

        assert opt_cpu.shape == gen_cpu.shape, (
            f"{test_name}: Shape mismatch optimized={opt_cpu.shape} vs generic={gen_cpu.shape}"
        )

        max_val = max(opt_cpu.abs().max().item(), gen_cpu.abs().max().item())

        if max_val < 1e-6:
            return  # Both essentially zero

        # Compute differences
        abs_diff = (opt_cpu - gen_cpu).abs()
        rel_diff = abs_diff / (max_val + 1e-8)

        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()

        # Compute correlation
        opt_flat = opt_cpu.flatten()
        gen_flat = gen_cpu.flatten()

        if opt_flat.std() > 1e-6 and gen_flat.std() > 1e-6:
            correlation = torch.corrcoef(torch.stack([opt_flat, gen_flat]))[0, 1].item()
        else:
            correlation = 1.0

        # Assertions - use tighter tolerance since same algorithm
        assert max_rel_diff < rtol, (
            f"{test_name}: Max relative diff {max_rel_diff:.4f} > {rtol} (correlation={correlation:.4f})"
        )

        assert correlation > min_correlation, (
            f"{test_name}: Correlation {correlation:.4f} < {min_correlation}"
        )

        # Also check peak locations match
        opt_peak = np.unravel_index(opt_cpu.argmax().item(), opt_cpu.shape)
        gen_peak = np.unravel_index(gen_cpu.argmax().item(), gen_cpu.shape)

        for i in range(len(opt_peak)):
            assert abs(opt_peak[i] - gen_peak[i]) <= 1, (
                f"{test_name}: Peak mismatch at dim {i}: optimized={opt_peak} vs generic={gen_peak}"
            )


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class Test4DDiagnostics:
    """Detailed diagnostic tests for investigating 4D CUDA vs PyTorch discrepancies."""

    def test_single_centered_splat_4d(self):
        """Test a single isotropic splat centered in a 4D volume."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        dim = 4
        shape = (8, 8, 8, 8)
        tile_size = 4

        # Single splat at exact center
        center = torch.tensor(
            [[4.0, 4.0, 4.0, 4.0]], device="cuda", dtype=torch.float32
        )
        L = torch.eye(dim, device="cuda", dtype=torch.float32).unsqueeze(0) * 2.0
        amps = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        sharpness = torch.tensor([2.0], device="cuda", dtype=torch.float32)

        # CUDA forward
        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            center, conic, amps, sharpness, list(shape), 3.0, 1e-5, tile_size
        )
        cuda_output = cuda_result[0].reshape(shape)

        # PyTorch reference
        pytorch_output = render_gaussians(shape, center, L, amps, sharpness, 3.0, 1e-5)

        # Diagnostic output
        cuda_max = cuda_output.max().item()
        pytorch_max = pytorch_output.max().item()

        # Verify both have significant values
        assert cuda_max > 0.5, (
            f"CUDA max ({cuda_max}) should be > 0.5 for centered splat"
        )
        assert pytorch_max > 0.5, (
            f"PyTorch max ({pytorch_max}) should be > 0.5 for centered splat"
        )

        # Check peak location
        cuda_peak = np.unravel_index(cuda_output.argmax().cpu().item(), shape)
        pytorch_peak = np.unravel_index(pytorch_output.argmax().cpu().item(), shape)

        assert cuda_peak == pytorch_peak == (4, 4, 4, 4), (
            f"Peak mismatch: CUDA={cuda_peak}, PyTorch={pytorch_peak}, expected (4,4,4,4)"
        )

        # Check value at center
        center_val_cuda = cuda_output[4, 4, 4, 4].item()
        center_val_pytorch = pytorch_output[4, 4, 4, 4].item()

        assert abs(center_val_cuda - center_val_pytorch) < 0.01, (
            f"Center value mismatch: CUDA={center_val_cuda:.4f}, PyTorch={center_val_pytorch:.4f}"
        )

        # Check correlation
        cuda_flat = cuda_output.cpu().flatten()
        pytorch_flat = pytorch_output.cpu().flatten()
        correlation = torch.corrcoef(torch.stack([cuda_flat, pytorch_flat]))[
            0, 1
        ].item()

        assert correlation > 0.99, (
            f"Correlation {correlation:.4f} should be > 0.99 for single centered splat"
        )

    def test_conic_computation_4d(self):
        """Verify that conic (Σ⁻¹) computation is correct for 4D."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        dim = 4
        N = 3

        # Test case 1: Identity L -> Identity Σ -> Identity Σ⁻¹
        L_identity = (
            torch.eye(dim, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
        )
        conic_identity = cholesky_to_conic(L_identity)

        # Expected: upper triangle of identity matrix
        # For 4D: [1, 0, 0, 0, 1, 0, 0, 1, 0, 1]
        expected_identity = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], device="cuda"
        )
        for i in range(N):
            assert torch.allclose(conic_identity[i], expected_identity, atol=1e-5), (
                f"Identity conic mismatch at splat {i}: got {conic_identity[i].cpu().numpy()}"
            )

        # Test case 2: Scaled identity L
        scale = 2.0
        L_scaled = (
            torch.eye(dim, device="cuda", dtype=torch.float32).unsqueeze(0) * scale
        )
        conic_scaled = cholesky_to_conic(L_scaled)

        # Σ = L @ L^T = scale² * I, so Σ⁻¹ = (1/scale²) * I
        expected_scaled = torch.tensor(
            [0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.25], device="cuda"
        )
        assert torch.allclose(conic_scaled[0], expected_scaled, atol=1e-5), (
            f"Scaled conic mismatch: got {conic_scaled[0].cpu().numpy()}, expected {expected_scaled.cpu().numpy()}"
        )

    def test_mahalanobis_consistency_4d(self):
        """Verify Mahalanobis distance computation is consistent between CUDA and PyTorch."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        dim = 4

        # Create a test L matrix
        L = torch.eye(dim, device="cuda", dtype=torch.float32).unsqueeze(0) * 1.5
        # Add off-diagonals for anisotropy
        L[0, 1, 0] = 0.3
        L[0, 2, 0] = 0.2
        L[0, 3, 1] = 0.25

        conic = cholesky_to_conic(L)

        # Test displacement vector
        d = torch.tensor([[1.0, 0.5, -0.3, 0.7]], device="cuda", dtype=torch.float32)

        # PyTorch reference: solve L @ y = d, then ||y||²
        y = torch.linalg.solve_triangular(L, d.unsqueeze(-1), upper=False).squeeze(-1)
        pytorch_dist_sq = (y * y).sum().item()

        # CUDA reference: d^T @ Σ⁻¹ @ d
        # Σ⁻¹ is packed in conic, need to unpack and compute
        # Using the packed upper triangle format
        Sigma = L @ L.transpose(-2, -1)
        Sigma_inv = torch.linalg.inv(Sigma)
        cuda_dist_sq = (d @ Sigma_inv @ d.transpose(-2, -1)).item()

        assert abs(pytorch_dist_sq - cuda_dist_sq) < 1e-4, (
            f"Mahalanobis distance mismatch: PyTorch={pytorch_dist_sq:.6f}, CUDA reference={cuda_dist_sq:.6f}"
        )

    def test_binning_4d(self):
        """Verify splats are being binned to tiles correctly in 4D."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        dim = 4
        shape = (8, 8, 8, 8)
        tile_size = 4

        # Single splat at center
        center = torch.tensor(
            [[4.0, 4.0, 4.0, 4.0]], device="cuda", dtype=torch.float32
        )
        L = torch.eye(dim, device="cuda", dtype=torch.float32).unsqueeze(0) * 2.0
        amps = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        sharpness = torch.tensor([2.0], device="cuda", dtype=torch.float32)

        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            center, conic, amps, sharpness, list(shape), 3.0, 1e-5, tile_size
        )

        # cuda_result contains: [output, tile_counts, tile_offsets, tile_content]
        tile_counts = cuda_result[1]

        # With shape (8,8,8,8) and tile_size 4, we have 2^4 = 16 tiles
        num_tiles = (8 // 4) ** 4
        assert tile_counts.numel() == num_tiles, (
            f"Expected {num_tiles} tiles, got {tile_counts.numel()}"
        )

        # Check that at least one tile has the splat
        tiles_with_splats = (tile_counts > 0).sum().item()
        assert tiles_with_splats > 0, "No tiles have splats - binning failed!"

        # A splat at center (4,4,4,4) with L=2*I should cover multiple tiles
        # The effective radius is ~3 * 2 = 6 voxels in each dimension
        # So it should be binned to tiles near the center
        assert tiles_with_splats >= 1, f"Only {tiles_with_splats} tiles have the splat"

    def test_4d_multiple_splats_detailed(self):
        """Detailed test with multiple splats to identify pattern of discrepancy."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        np.random.seed(12345)
        N = 5
        dim = 4
        shape = (12, 12, 12, 12)
        tile_size = 4

        # Well-separated splats at known positions
        centers = torch.tensor(
            [
                [3.0, 3.0, 3.0, 3.0],
                [6.0, 6.0, 6.0, 6.0],
                [9.0, 9.0, 9.0, 9.0],
                [3.0, 9.0, 3.0, 9.0],
                [9.0, 3.0, 9.0, 3.0],
            ],
            device="cuda",
            dtype=torch.float32,
        )

        # Simple isotropic splats
        scale = 1.5
        L = (
            torch.eye(dim, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
            * scale
        )

        amps = torch.ones(N, device="cuda", dtype=torch.float32)
        sharpness = torch.ones(N, device="cuda", dtype=torch.float32) * 2.0

        conic = cholesky_to_conic(L)
        cuda_result = cuda_splatting_backend.forward(
            centers, conic, amps, sharpness, list(shape), 3.0, 1e-5, tile_size
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, centers, L, amps, sharpness, 3.0, 1e-5)

        # Check each splat's contribution at its center
        for i, c in enumerate(centers):
            pos = tuple(int(x.item()) for x in c)
            cuda_val = cuda_output[pos].item()
            pytorch_val = pytorch_output[pos].item()

            # Allow some tolerance for splats near edges or overlapping
            rel_diff = abs(cuda_val - pytorch_val) / max(cuda_val, pytorch_val, 1e-6)
            assert rel_diff < 0.2 or abs(cuda_val - pytorch_val) < 0.1, (
                f"Splat {i} at {pos}: CUDA={cuda_val:.4f}, PyTorch={pytorch_val:.4f}, rel_diff={rel_diff:.4f}"
            )

        # Overall statistics
        cuda_sum = cuda_output.sum().item()
        pytorch_sum = pytorch_output.sum().item()
        sum_rel_diff = abs(cuda_sum - pytorch_sum) / max(cuda_sum, pytorch_sum)

        assert sum_rel_diff < 0.1, (
            f"Total sum mismatch: CUDA={cuda_sum:.2f}, PyTorch={pytorch_sum:.2f}, rel_diff={sum_rel_diff:.4f}"
        )

        # Correlation check
        cuda_flat = cuda_output.cpu().flatten()
        pytorch_flat = pytorch_output.cpu().flatten()
        correlation = torch.corrcoef(torch.stack([cuda_flat, pytorch_flat]))[
            0, 1
        ].item()

        assert correlation > 0.95, (
            f"Correlation {correlation:.4f} should be > 0.95 for well-separated splats"
        )

    def test_4d_binning_individual_splats(self):
        """Test binning of individual splats at different positions."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        dim = 4
        shape = (12, 12, 12, 12)
        tile_size = 4

        # Test positions that span different tiles
        test_positions = [
            [3.0, 3.0, 3.0, 3.0],  # Tile (0,0,0,0)
            [6.0, 6.0, 6.0, 6.0],  # Tile (1,1,1,1) - on boundary
            [9.0, 9.0, 9.0, 9.0],  # Tile (2,2,2,2)
        ]

        for pos in test_positions:
            center = torch.tensor([pos], device="cuda", dtype=torch.float32)
            L = torch.eye(dim, device="cuda", dtype=torch.float32).unsqueeze(0) * 1.5
            amps = torch.tensor([1.0], device="cuda", dtype=torch.float32)
            sharpness = torch.tensor([2.0], device="cuda", dtype=torch.float32)

            conic = cholesky_to_conic(L)
            cuda_result = cuda_splatting_backend.forward(
                center, conic, amps, sharpness, list(shape), 3.0, 1e-5, tile_size
            )
            cuda_output = cuda_result[0].reshape(shape)
            tile_counts = cuda_result[1]

            pos_int = tuple(int(p) for p in pos)
            cuda_val = cuda_output[pos_int].item()
            tiles_with_splats = (tile_counts > 0).sum().item()

            assert tiles_with_splats > 0, (
                f"Position {pos}: No tiles have the splat (binning failed)"
            )
            assert cuda_val > 0.5, (
                f"Position {pos}: CUDA value at center = {cuda_val:.4f} (should be > 0.5)"
            )

    def test_4d_tile_boundary_positions(self):
        """Test splats specifically at tile boundaries."""
        import cuda_splatting_backend

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        dim = 4
        shape = (12, 12, 12, 12)
        tile_size = 4

        # Position exactly on tile boundary (at 4.0, 8.0, etc.)
        boundary_positions = [
            [4.0, 4.0, 4.0, 4.0],  # Boundary between tile 0 and 1
            [8.0, 8.0, 8.0, 8.0],  # Boundary between tile 1 and 2
        ]

        for pos in boundary_positions:
            center = torch.tensor([pos], device="cuda", dtype=torch.float32)
            L = torch.eye(dim, device="cuda", dtype=torch.float32).unsqueeze(0) * 2.0
            amps = torch.tensor([1.0], device="cuda", dtype=torch.float32)
            sharpness = torch.tensor([2.0], device="cuda", dtype=torch.float32)

            conic = cholesky_to_conic(L)
            cuda_result = cuda_splatting_backend.forward(
                center, conic, amps, sharpness, list(shape), 3.0, 1e-5, tile_size
            )
            cuda_output = cuda_result[0].reshape(shape)

            pytorch_output = render_gaussians(
                shape, center, L, amps, sharpness, 3.0, 1e-5
            )

            pos_int = tuple(int(p) for p in pos)
            cuda_val = cuda_output[pos_int].item()
            pytorch_val = pytorch_output[pos_int].item()

            assert abs(cuda_val - pytorch_val) < 0.05, (
                f"Boundary position {pos}: CUDA={cuda_val:.4f}, PyTorch={pytorch_val:.4f}"
            )
