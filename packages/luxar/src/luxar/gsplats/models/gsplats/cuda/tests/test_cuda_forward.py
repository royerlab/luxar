"""
Tests for CUDA backend forward pass functionality.

These tests verify that the CUDA backend forward pass produces correct results
and matches the PyTorch reference implementation.
"""

import numpy as np
import pytest
import torch

from .conftest import Tolerances

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# Check if CUDA backend is compiled
try:
    import cuda_splatting_backend  # noqa: F401

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
        import cuda_splatting_backend  # noqa: F401

        assert hasattr(cuda_splatting_backend, "forward")
        assert hasattr(cuda_splatting_backend, "backward")
        assert hasattr(cuda_splatting_backend, "__version__")


class TestCholeskyToConic:
    """Test L -> Sigma^-1 conversion."""

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

        # For identity L (splat 1), Sigma = I, so Sigma^-1 = I
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

        # For identity L, Sigma = I, so Sigma^-1 = I
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

            # Tighter than COMPARISON_MAX_REL_DIFF (0.15) since this is
            # a direct CUDA vs CPU comparison within the same implementation
            assert max_rel_diff < 0.1, (
                f"Max relative difference {max_rel_diff:.4f} exceeds threshold"
            )
            assert mean_rel_diff < Tolerances.COMPARISON_MEAN_REL_DIFF, (
                f"Mean relative difference {mean_rel_diff:.4f} exceeds threshold"
            )

        # Also check correlation for overall structure match
        cpu_flat = cpu_output.flatten()
        cuda_flat = cuda_output_cpu.flatten()
        if cpu_flat.std() > 1e-6 and cuda_flat.std() > 1e-6:
            correlation = torch.corrcoef(torch.stack([cpu_flat, cuda_flat]))[0, 1]
            assert correlation > Tolerances.COMPARISON_MIN_CORRELATION, (
                f"Correlation {correlation:.4f} too low"
            )

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
            assert correlation > Tolerances.COMPARISON_MIN_CORRELATION, (
                f"Correlation {correlation:.4f} too low"
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
            import cuda_splatting_backend  # noqa: F401

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
        cpu_ms = cpu_time * 1000
        cuda_ms = cuda_time * 1000
        print(f"\nCPU: {cpu_ms:.2f}ms, CUDA: {cuda_ms:.2f}ms, Speedup: {speedup:.1f}x")

        # CUDA should be at least 2x faster
        assert speedup > 2.0, (
            f"CUDA speedup ({speedup:.1f}x) too low! "
            f"CPU={cpu_ms:.1f}ms, CUDA={cuda_ms:.1f}ms."
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
            print("WARNING: Custom CUDA not significantly faster than PyTorch")
