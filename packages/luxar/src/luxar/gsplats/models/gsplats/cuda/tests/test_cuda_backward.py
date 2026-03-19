"""
Tests for CUDA backend backward pass functionality.

These tests verify that the CUDA backend backward pass produces correct gradients
and integrates properly with PyTorch autograd.
"""

import numpy as np
import pytest
import torch

from .conftest import Tolerances, compute_L_row_norms

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# Check if CUDA backend is compiled
try:
    import cuda_splatting_backend  # noqa: F401

    CUDA_BACKEND_AVAILABLE = True
except ImportError:
    CUDA_BACKEND_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


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
            assert sign_match > Tolerances.BACKWARD_SIGN_MATCH, (
                f"Gradient sign match {sign_match:.2f} too low"
            )

            # Check gradient magnitudes are in similar range
            cpu_mag = cpu_amp_grad.abs().mean()
            cuda_mag = cuda_amp_grad.abs().mean()
            if cpu_mag > 1e-8 and cuda_mag > 1e-8:
                mag_ratio = max(cpu_mag / cuda_mag, cuda_mag / cpu_mag)
                assert mag_ratio < Tolerances.BACKWARD_MAG_RATIO, (
                    f"Gradient magnitude ratio {mag_ratio:.2f} too large"
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
        import cuda_splatting_backend  # noqa: F401

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
        truncate = 3.0
        intensity_floor = 1e-5

        # Run optimized 2D path
        conic_2d = cholesky_to_conic(L_2d)
        L_row_norms_2d = compute_L_row_norms(L_2d)
        result_2d = cuda_splatting_backend.forward(
            centers_2d.contiguous(),
            conic_2d.contiguous(),
            amps.contiguous(),
            L_row_norms_2d.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            16,
        )
        output_2d = result_2d[0].reshape(shape)

        # Embed into 4D and run generic path
        centers_4d, L_4d, shape_4d = self._embed_2d_to_4d(centers_2d, L_2d, shape)
        conic_4d = cholesky_to_conic(L_4d)
        L_row_norms_4d = compute_L_row_norms(L_4d)
        result_4d = cuda_splatting_backend.forward(
            centers_4d.contiguous(),
            conic_4d.contiguous(),
            amps.contiguous(),
            L_row_norms_4d.contiguous(),
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
        import cuda_splatting_backend  # noqa: F401

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
        truncate = 3.0
        intensity_floor = 1e-5

        # Run optimized 3D path
        conic_3d = cholesky_to_conic(L_3d)
        L_row_norms_3d = compute_L_row_norms(L_3d)
        result_3d = cuda_splatting_backend.forward(
            centers_3d.contiguous(),
            conic_3d.contiguous(),
            amps.contiguous(),
            L_row_norms_3d.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            8,
        )
        output_3d = result_3d[0].reshape(shape)

        # Embed into 4D and run generic path
        centers_4d, L_4d, shape_4d = self._embed_3d_to_4d(centers_3d, L_3d, shape)
        conic_4d = cholesky_to_conic(L_4d)
        L_row_norms_4d = compute_L_row_norms(L_4d)
        result_4d = cuda_splatting_backend.forward(
            centers_4d.contiguous(),
            conic_4d.contiguous(),
            amps.contiguous(),
            L_row_norms_4d.contiguous(),
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
        import cuda_splatting_backend  # noqa: F401

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

        # Optimized 2D
        conic_2d = cholesky_to_conic(L_2d)
        L_row_norms_2d = compute_L_row_norms(L_2d)
        result_2d = cuda_splatting_backend.forward(
            centers_2d.contiguous(),
            conic_2d.contiguous(),
            amps.contiguous(),
            L_row_norms_2d.contiguous(),
            list(shape),
            3.0,
            1e-5,
            16,
        )
        output_2d = result_2d[0].reshape(shape)

        # Generic 4D
        centers_4d, L_4d, shape_4d = self._embed_2d_to_4d(centers_2d, L_2d, shape)
        conic_4d = cholesky_to_conic(L_4d)
        L_row_norms_4d = compute_L_row_norms(L_4d)
        result_4d = cuda_splatting_backend.forward(
            centers_4d.contiguous(),
            conic_4d.contiguous(),
            amps.contiguous(),
            L_row_norms_4d.contiguous(),
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
        import cuda_splatting_backend  # noqa: F401

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

        # Optimized 3D
        conic_3d = cholesky_to_conic(L_3d)
        L_row_norms_3d = compute_L_row_norms(L_3d)
        result_3d = cuda_splatting_backend.forward(
            centers_3d.contiguous(),
            conic_3d.contiguous(),
            amps.contiguous(),
            L_row_norms_3d.contiguous(),
            list(shape),
            3.0,
            1e-5,
            8,
        )
        output_3d = result_3d[0].reshape(shape)

        # Generic 4D
        centers_4d, L_4d, shape_4d = self._embed_3d_to_4d(centers_3d, L_3d, shape)
        conic_4d = cholesky_to_conic(L_4d)
        L_row_norms_4d = compute_L_row_norms(L_4d)
        result_4d = cuda_splatting_backend.forward(
            centers_4d.contiguous(),
            conic_4d.contiguous(),
            amps.contiguous(),
            L_row_norms_4d.contiguous(),
            list(shape_4d),
            3.0,
            1e-5,
            4,
        )
        output_4d_slice = result_4d[0].reshape(shape_4d)[:, :, :, 0]

        self._assert_paths_match(
            output_3d, output_4d_slice, "3D anisotropic optimized vs generic"
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
        rel_diff.mean().item()

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
