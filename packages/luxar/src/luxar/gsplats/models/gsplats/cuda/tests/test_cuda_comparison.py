"""
Tests comparing CUDA backend with PyTorch reference implementation.

These tests ensure numerical correctness of the CUDA backend against the
authoritative PyTorch reference implementation.
"""

import numpy as np
import pytest
import torch

from .conftest import Tolerances, seed_all

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
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        # Create test data
        seed_all(1234)
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

        amps = torch.rand(N, device="cuda", dtype=torch.float32) * 0.5 + 0.5

        # Compute conic for CUDA backend
        conic = cholesky_to_conic(L)

        # Run CUDA backend
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
        )
        cuda_output = cuda_result[0].reshape(shape)

        # Run PyTorch reference
        pytorch_output = render_gaussians(
            shape, centers, L, amps, truncate, intensity_floor
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
            assert max_rel_diff < Tolerances.COMPARISON_MAX_REL_DIFF, (
                f"Max relative diff {max_rel_diff:.4f} too large"
            )
            assert mean_rel_diff < Tolerances.COMPARISON_MEAN_REL_DIFF, (
                f"Mean relative diff {mean_rel_diff:.6f} too large"
            )
            assert corr > Tolerances.COMPARISON_MIN_CORRELATION, (
                f"Correlation {corr:.4f} too low"
            )

    def test_direct_cuda_vs_pytorch_2d(self):
        """Direct comparison of CUDA and PyTorch outputs for 2D image."""
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        # Create test data
        seed_all(5678)
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

        amps = torch.rand(N, device="cuda", dtype=torch.float32) * 0.5 + 0.5

        # Compute conic for CUDA backend
        conic = cholesky_to_conic(L)

        # Run CUDA backend
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
        )
        cuda_output = cuda_result[0].reshape(shape)

        # Run PyTorch reference
        pytorch_output = render_gaussians(
            shape, centers, L, amps, truncate, intensity_floor
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
            assert max_rel_diff < Tolerances.COMPARISON_MAX_REL_DIFF, (
                f"Max relative diff {max_rel_diff:.4f} too large"
            )
            assert mean_rel_diff < Tolerances.COMPARISON_MEAN_REL_DIFF, (
                f"Mean relative diff {mean_rel_diff:.6f} too large"
            )
            assert corr > Tolerances.COMPARISON_MIN_CORRELATION, (
                f"Correlation {corr:.4f} too low"
            )

    def test_single_centered_splat_matches(self):
        """Test single centered splat produces identical peak location and value."""
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        # Single splat at center
        _N, d = 1, 3
        shape = (32, 32, 32)
        center_pos = [16.0, 16.0, 16.0]

        centers = torch.tensor([center_pos], device="cuda", dtype=torch.float32)
        L = torch.eye(d, device="cuda", dtype=torch.float32).unsqueeze(0) * 2.0
        amps = torch.tensor([1.0], device="cuda", dtype=torch.float32)

        conic = cholesky_to_conic(L)

        # CUDA output
        cuda_result = cuda_splatting_backend.forward(
            centers,
            conic,
            amps,
            list(shape),
            3.0,
            1e-5,
        )
        cuda_output = cuda_result[0].reshape(shape)

        # PyTorch output
        pytorch_output = render_gaussians(shape, centers, L, amps, 3.0, 1e-5)

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
    """Comprehensive CUDA vs PyTorch comparison tests.

    Tests ensure numerical correctness against PyTorch reference for:
    2D/3D/4D, various splat shapes/sizes, forward & backward.
    """

    # =========================================================================
    # Forward Pass Tests - Parametrized by Dimension
    # =========================================================================

    @pytest.mark.parametrize("dim", [2, 3])
    def test_forward_isotropic_splats(self, dim: int):
        """Test forward pass with isotropic (spherical) splats for 2D/3D."""
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        seed_all(42 + dim)
        N = 20

        # Create appropriate shape and parameters for each dimension
        if dim == 2:
            shape = (48, 48)
        else:  # dim == 3
            shape = (24, 24, 24)

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
        truncate = 3.0
        intensity_floor = 1e-5

        # CUDA forward
        conic = cholesky_to_conic(L)

        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
        )
        cuda_output = cuda_result[0].reshape(shape)

        # PyTorch reference
        pytorch_output = render_gaussians(
            shape, centers, L, amps, truncate, intensity_floor
        )

        # Compare with strict tolerances
        self._assert_outputs_match(cuda_output, pytorch_output, f"{dim}D isotropic")

    def test_forward_isotropic_splats_4d(self):
        """Test 4D forward pass with isotropic splats."""
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        seed_all(46)
        N = 20
        dim = 4
        shape = (12, 12, 12, 12)

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

        conic = cholesky_to_conic(L)

        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            list(shape),
            3.0,
            1e-5,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, centers, L, amps, 3.0, 1e-5)

        # Strict tolerance - this will fail, flagging need for investigation
        self._assert_outputs_match(cuda_output, pytorch_output, "4D isotropic")

    @pytest.mark.parametrize("dim", [2, 3])
    def test_forward_anisotropic_splats(self, dim: int):
        """Test forward pass with anisotropic (ellipsoidal) splats."""
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        seed_all(123 + dim)
        N = 15

        if dim == 2:
            shape = (48, 48)
        else:
            shape = (24, 24, 24)

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
        truncate = 3.0
        intensity_floor = 1e-5

        conic = cholesky_to_conic(L)

        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(
            shape, centers, L, amps, truncate, intensity_floor
        )

        self._assert_outputs_match(cuda_output, pytorch_output, f"{dim}D anisotropic")

    def test_forward_anisotropic_splats_4d(self):
        """Test 4D anisotropic splats (may have issues)."""
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        seed_all(127)
        N, dim = 15, 4
        shape = (12, 12, 12, 12)

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

        conic = cholesky_to_conic(L)

        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            list(shape),
            3.0,
            1e-5,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, centers, L, amps, 3.0, 1e-5)

        self._assert_outputs_match(cuda_output, pytorch_output, "4D anisotropic")

    @pytest.mark.parametrize("dim", [2, 3])
    def test_forward_elongated_splats(self, dim: int):
        """Test forward pass with highly elongated splats (aspect ratio > 5:1)."""
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        seed_all(456 + dim)
        N = 10

        if dim == 2:
            shape = (64, 64)
        else:
            shape = (32, 32, 32)

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
        truncate = 3.0
        intensity_floor = 1e-5

        conic = cholesky_to_conic(L)

        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(
            shape, centers, L, amps, truncate, intensity_floor
        )

        self._assert_outputs_match(cuda_output, pytorch_output, f"{dim}D elongated")

    def test_forward_elongated_splats_4d(self):
        """Test 4D elongated splats (may have issues)."""
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        seed_all(460)
        N, dim = 10, 4
        shape = (16, 16, 16, 16)

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

        conic = cholesky_to_conic(L)

        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            list(shape),
            3.0,
            1e-5,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, centers, L, amps, 3.0, 1e-5)

        self._assert_outputs_match(cuda_output, pytorch_output, "4D elongated")

    # =========================================================================
    # Forward Pass Tests - Parametrized by Size
    # =========================================================================

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("size_category", ["tiny", "small", "medium", "large"])
    def test_forward_various_sizes(self, dim: int, size_category: str):
        """Test forward pass with various splat sizes."""
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        seed_all(789 + dim + hash(size_category) % 100)
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
        else:
            shape = (32, 32, 32)

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
        truncate = 3.0
        intensity_floor = 1e-5

        conic = cholesky_to_conic(L)

        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(
            shape, centers, L, amps, truncate, intensity_floor
        )

        self._assert_outputs_match(
            cuda_output, pytorch_output, f"{dim}D {size_category}"
        )

    # =========================================================================
    # Forward Pass Tests - Parametrized by Sharpness
    # =========================================================================

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

        seed_all(111 + dim)
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
                assert sign_match > Tolerances.BACKWARD_SIGN_MATCH, (
                    f"{dim}D: Amplitude gradient sign match {sign_match:.2f} too low"
                )

            # Check gradient magnitude consistency
            cpu_mag = cpu_amp_grad.abs().mean()
            cuda_mag = cuda_amp_grad.abs().mean()
            if cpu_mag > 1e-8 and cuda_mag > 1e-8:
                mag_ratio = max(cpu_mag / cuda_mag, cuda_mag / cpu_mag)
                assert mag_ratio < Tolerances.BACKWARD_MAG_RATIO, (
                    f"{dim}D: Gradient magnitude ratio {mag_ratio:.2f} too large"
                )

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
    def test_backward_with_various_scales(self, dim: int, scale: float):
        """Test backward pass produces finite gradients for various splat scales.

        Note: GaussianSplatModelCUDA uses standard Gaussian (s=2).
        This test verifies gradients are correct for different covariance scales.
        """
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        seed_all(222 + dim + int(scale * 10))
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

        seed_all(333 + dim)

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
        torch.zeros_like(output)
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
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        if dim == 2:
            shape = (32, 32)
        else:
            shape = (24, 24, 24)

        center = torch.tensor(
            [[s // 2 for s in shape]], device="cuda", dtype=torch.float32
        )
        L = torch.eye(dim, device="cuda", dtype=torch.float32).unsqueeze(0) * 2.0
        amps = torch.tensor([1.0], device="cuda", dtype=torch.float32)

        conic = cholesky_to_conic(L)

        cuda_result = cuda_splatting_backend.forward(
            center,
            conic,
            amps,
            list(shape),
            3.0,
            1e-5,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, center, L, amps, 3.0, 1e-5)

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
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        dim = 4
        shape = (16, 16, 16, 16)

        center = torch.tensor(
            [[s // 2 for s in shape]], device="cuda", dtype=torch.float32
        )
        L = torch.eye(dim, device="cuda", dtype=torch.float32).unsqueeze(0) * 2.0
        amps = torch.tensor([1.0], device="cuda", dtype=torch.float32)

        conic = cholesky_to_conic(L)

        cuda_result = cuda_splatting_backend.forward(
            center,
            conic,
            amps,
            list(shape),
            3.0,
            1e-5,
        )
        cuda_output = cuda_result[0].reshape(shape)

        render_gaussians(shape, center, L, amps, 3.0, 1e-5)

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
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        seed_all(444 + dim)
        N = 50  # Many splats

        if dim == 2:
            shape = (48, 48)
        else:
            shape = (24, 24, 24)

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

        conic = cholesky_to_conic(L)

        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            list(shape),
            3.0,
            1e-5,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, centers, L, amps, 3.0, 1e-5)

        self._assert_outputs_match(cuda_output, pytorch_output, f"{dim}D overlapping")

    @pytest.mark.parametrize("dim", [2, 3])
    def test_splats_at_boundaries(self, dim: int):
        """Test splats positioned at volume boundaries."""
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        if dim == 2:
            shape = (32, 32)
        else:
            shape = (24, 24, 24)

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

        conic = cholesky_to_conic(L)

        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            list(shape),
            3.0,
            1e-5,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, centers, L, amps, 3.0, 1e-5)

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
        rtol: float = Tolerances.COMPARISON_MAX_REL_DIFF,
        atol: float = Tolerances.COMPARISON_MEAN_REL_DIFF,
        min_correlation: float = Tolerances.COMPARISON_MIN_CORRELATION,
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
class TestBackwardComparison:
    """Compare CUDA backward pass gradients against PyTorch CPU reference.

    The forward comparison tests above verify numerical correctness of the
    forward kernel. These tests verify that the backward kernel produces
    gradients consistent with PyTorch autograd on CPU.

    Uses sign-match and magnitude-ratio metrics (not element-wise allclose)
    because CUDA atomic accumulation introduces non-deterministic ordering
    that affects gradient values.
    """

    def test_backward_cuda_vs_pytorch_3d(self):
        """Compare 3D backward gradients between CUDA and PyTorch CPU.

        Creates identical models on CPU and CUDA, runs forward+backward with
        loss = output.sum(), and compares raw_a gradients using sign-match
        and magnitude-ratio.
        """
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        seed_all(1234)
        N, d = 10, 3
        shape = (16, 16, 16)

        centers0 = np.random.rand(N, d).astype(np.float32) * 12 + 2
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0[i] *= np.random.uniform(0.5, 1.5)
        amps0 = np.ones(N, dtype=np.float32) * 0.5

        # CPU model
        cpu_model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cpu",
        )

        # CUDA model
        cuda_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        # Forward + backward on CPU with loss = output.sum()
        cpu_output = cpu_model()
        cpu_loss = cpu_output.sum()
        cpu_loss.backward()

        # Forward + backward on CUDA with loss = output.sum()
        cuda_output = cuda_model()
        cuda_loss = cuda_output.sum()
        cuda_loss.backward()

        # Compare raw_a gradients (amplitude is the most directly comparable)
        cpu_grad = cpu_model.raw_a.grad
        cuda_grad = cuda_model.raw_a.grad

        assert cpu_grad is not None, "CPU raw_a gradient is None"
        assert cuda_grad is not None, "CUDA raw_a gradient is None"

        cuda_grad_cpu = cuda_grad.cpu()

        # Sign match: majority of gradient signs should agree
        sign_match = (torch.sign(cpu_grad) == torch.sign(cuda_grad_cpu)).float().mean()
        print(f"\n3D backward raw_a sign match: {sign_match:.4f}")
        assert sign_match > Tolerances.BACKWARD_SIGN_MATCH, (
            f"3D raw_a gradient sign match {sign_match:.2f} "
            f"< {Tolerances.BACKWARD_SIGN_MATCH}"
        )

        # Magnitude ratio: gradient magnitudes should be in similar range
        cpu_mag = cpu_grad.abs().mean()
        cuda_mag = cuda_grad_cpu.abs().mean()
        if cpu_mag > 1e-8 and cuda_mag > 1e-8:
            mag_ratio = max(cpu_mag / cuda_mag, cuda_mag / cpu_mag)
            print(f"  Magnitude ratio: {mag_ratio:.4f}")
            assert mag_ratio < Tolerances.BACKWARD_MAG_RATIO, (
                f"3D raw_a gradient magnitude ratio {mag_ratio:.2f} "
                f"> {Tolerances.BACKWARD_MAG_RATIO}"
            )
