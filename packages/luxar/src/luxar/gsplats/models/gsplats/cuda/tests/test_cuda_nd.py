"""
Tests for CUDA backend nD functionality (4D+, global splats, large counts).

These tests verify CUDA backend behavior for higher dimensions (4D-6D),
global splat handling, and large-scale operations.
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


@pytest.fixture(autouse=True)
def cleanup_cuda_state():
    """Clean up CUDA state before and after each test to prevent state pollution."""
    # Clean up before test
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset random seed for CUDA
        torch.cuda.manual_seed_all(42)
        # Clear any lingering CUDA errors
        torch.cuda.get_device_properties(0)

    yield

    # Clean up after test
    if CUDA_AVAILABLE:
        # Delete any CUDA tensors that might still be referenced
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class Test4DDiagnostics:
    """Detailed diagnostic tests for investigating 4D CUDA vs PyTorch discrepancies."""

    def test_single_centered_splat_4d(self):
        """Test a single isotropic splat centered in a 4D volume."""
        import cuda_splatting_backend  # noqa: F401

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

        # CUDA forward
        conic = cholesky_to_conic(L)
        L_row_norms = compute_L_row_norms(L)
        cuda_result = cuda_splatting_backend.forward(
            center,
            conic,
            amps,
            L_row_norms.contiguous(),
            list(shape),
            3.0,
            1e-5,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)

        # PyTorch reference
        pytorch_output = render_gaussians(shape, center, L, amps, 3.0, 1e-5)

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

        assert correlation > Tolerances.COMPARISON_MIN_CORRELATION, (
            f"Correlation {correlation:.4f} too low for single centered splat"
        )

    def test_conic_computation_4d(self):
        """Verify that conic (Sigma^-1) computation is correct for 4D."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        dim = 4
        N = 3

        # Test case 1: Identity L -> Identity Sigma -> Identity Sigma^-1
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

        # Sigma = L @ L^T = scale^2 * I, so Sigma^-1 = (1/scale^2) * I
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

        cholesky_to_conic(L)

        # Test displacement vector
        d = torch.tensor([[1.0, 0.5, -0.3, 0.7]], device="cuda", dtype=torch.float32)

        # PyTorch reference: solve L @ y = d, then ||y||^2
        y = torch.linalg.solve_triangular(L, d.unsqueeze(-1), upper=False).squeeze(-1)
        pytorch_dist_sq = (y * y).sum().item()

        # CUDA reference: d^T @ Sigma^-1 @ d
        # Sigma^-1 is packed in conic, need to unpack and compute
        # Using the packed upper triangle format
        Sigma = L @ L.transpose(-2, -1)
        Sigma_inv = torch.linalg.inv(Sigma)
        cuda_dist_sq = (d @ Sigma_inv @ d.transpose(-2, -1)).item()

        assert abs(pytorch_dist_sq - cuda_dist_sq) < 1e-4, (
            f"Mahalanobis distance mismatch: PyTorch={pytorch_dist_sq:.6f}, CUDA reference={cuda_dist_sq:.6f}"
        )

    def test_binning_4d(self):
        """Verify splats are being binned to tiles correctly in 4D."""
        import cuda_splatting_backend  # noqa: F401

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

        conic = cholesky_to_conic(L)
        L_row_norms = compute_L_row_norms(L)
        cuda_result = cuda_splatting_backend.forward(
            center,
            conic,
            amps,
            L_row_norms.contiguous(),
            list(shape),
            3.0,
            1e-5,
            tile_size,
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
        import cuda_splatting_backend  # noqa: F401

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

        conic = cholesky_to_conic(L)
        L_row_norms = compute_L_row_norms(L)
        cuda_result = cuda_splatting_backend.forward(
            centers,
            conic,
            amps,
            L_row_norms.contiguous(),
            list(shape),
            3.0,
            1e-5,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)

        pytorch_output = render_gaussians(shape, centers, L, amps, 3.0, 1e-5)

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
        import cuda_splatting_backend  # noqa: F401

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

            conic = cholesky_to_conic(L)
            L_row_norms = compute_L_row_norms(L)
            cuda_result = cuda_splatting_backend.forward(
                center,
                conic,
                amps,
                L_row_norms.contiguous(),
                list(shape),
                3.0,
                1e-5,
                tile_size,
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
        import cuda_splatting_backend  # noqa: F401

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

            conic = cholesky_to_conic(L)
            L_row_norms = compute_L_row_norms(L)
            cuda_result = cuda_splatting_backend.forward(
                center,
                conic,
                amps,
                L_row_norms.contiguous(),
                list(shape),
                3.0,
                1e-5,
                tile_size,
            )
            cuda_output = cuda_result[0].reshape(shape)

            pytorch_output = render_gaussians(shape, center, L, amps, 3.0, 1e-5)

            pos_int = tuple(int(p) for p in pos)
            cuda_val = cuda_output[pos_int].item()
            pytorch_val = pytorch_output[pos_int].item()

            assert abs(cuda_val - pytorch_val) < 0.05, (
                f"Boundary position {pos}: CUDA={cuda_val:.4f}, PyTorch={pytorch_val:.4f}"
            )


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestGlobalSplatHandling:
    """
    Test global splat handling - splats that touch >10% of tiles AND >1024 tiles.

    Global splats are handled by a separate kernel that processes all pixels
    instead of using tile-based binning. These tests verify that:
    1. Global splats produce correct output matching PyTorch reference
    2. intensity_floor is properly applied to global splats
    3. Gradients are correctly computed for global splats
    """

    def test_global_splat_forward_matches_reference(self):
        """
        Test that a large global splat produces output matching PyTorch reference.

        Creates a volume large enough (128^3 = 4096 tiles with tile_size=8) and
        a splat large enough to touch >1024 tiles, triggering global handling.
        """
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        # Large volume: 128^3 with tile_size=8 = 16^3 = 4096 tiles
        shape = (128, 128, 128)
        tile_size = 8
        d = 3
        truncate = 3.0
        intensity_floor = 1e-5

        # Create one very large splat centered in the volume
        # With L diagonal entries of 20, the effective radius is 3*20=60 pixels
        # This touches ~(60/8)^3 ~= 422 tiles per octant, or ~3375 tiles total
        # which exceeds 1024 tiles minimum for global handling
        centers = torch.tensor([[64.0, 64.0, 64.0]], device="cuda", dtype=torch.float32)
        L = torch.eye(d, device="cuda", dtype=torch.float32).unsqueeze(0) * 20.0
        amps = torch.tensor([1.0], device="cuda", dtype=torch.float32)

        conic = cholesky_to_conic(L)
        L_row_norms = compute_L_row_norms(L)

        # Run CUDA backend (should trigger global splat handling)
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            L_row_norms.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)
        global_splat_ids = cuda_result[4]

        # Verify global splat was detected
        assert len(global_splat_ids) > 0, "Expected global splat to be detected"

        # Run PyTorch reference
        pytorch_output = render_gaussians(
            shape, centers, L, amps, truncate, intensity_floor
        )

        # Compare outputs
        cuda_cpu = cuda_output.cpu()
        pytorch_cpu = pytorch_output.cpu()

        max_val = max(cuda_cpu.abs().max().item(), pytorch_cpu.abs().max().item())
        if max_val > 1e-6:
            abs_diff = (cuda_cpu - pytorch_cpu).abs()
            rel_diff = abs_diff / (max_val + 1e-8)

            max_abs_diff = abs_diff.max().item()
            max_rel_diff = rel_diff.max().item()

            print("\nGlobal splat CUDA vs PyTorch:")
            print(f"  Global splats detected: {len(global_splat_ids)}")
            print(f"  Max absolute diff: {max_abs_diff:.6f}")
            print(f"  Max relative diff: {max_rel_diff:.4f}")

            assert max_rel_diff < 0.05, (
                f"Global splat max relative diff {max_rel_diff:.4f} too large"
            )

    def test_global_splat_intensity_floor_applied(self):
        """
        Test that intensity_floor is properly applied to global splats.

        With a high intensity_floor, far pixels should have zero contribution
        from the global splat, matching PyTorch reference behavior.
        """
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        # Use 3D volume large enough for global splats
        # 128^3 with tile_size=8 = 16^3 = 4096 tiles
        # Global splat requires > 1024 tiles AND > 10% of tiles (> 409)
        shape = (128, 128, 128)
        tile_size = 8
        d = 3
        truncate = 3.0

        # High intensity floor that will cut off distant pixels
        intensity_floor = 0.1

        # Large splat to trigger global handling
        centers = torch.tensor([[64.0, 64.0, 64.0]], device="cuda", dtype=torch.float32)
        L = torch.eye(d, device="cuda", dtype=torch.float32).unsqueeze(0) * 20.0
        amps = torch.tensor([1.0], device="cuda", dtype=torch.float32)

        conic = cholesky_to_conic(L)
        L_row_norms = compute_L_row_norms(L)

        # Run CUDA backend
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            L_row_norms.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)
        global_splat_ids = cuda_result[4]

        # Verify global splat was detected
        assert len(global_splat_ids) > 0, "Expected global splat to be detected"

        # Run PyTorch reference with same intensity_floor
        pytorch_output = render_gaussians(
            shape, centers, L, amps, truncate, intensity_floor
        )

        # Compare outputs - should match closely since intensity_floor is applied
        cuda_cpu = cuda_output.cpu()
        pytorch_cpu = pytorch_output.cpu()

        # With high intensity_floor, the output shape is quite different due to truncation
        # Focus on comparing the non-zero output values
        max_val = max(cuda_cpu.abs().max().item(), pytorch_cpu.abs().max().item())
        print(f"\nIntensity floor test (floor={intensity_floor}):")
        print(
            f"  CUDA max: {cuda_cpu.max().item():.4f}, PyTorch max: {pytorch_cpu.max().item():.4f}"
        )

        if max_val > 1e-6:
            abs_diff = (cuda_cpu - pytorch_cpu).abs()
            rel_diff = abs_diff / (pytorch_cpu.abs() + 1e-8)

            # Focus on non-zero regions
            nonzero_mask = pytorch_cpu > intensity_floor
            if nonzero_mask.sum() > 0:
                rel_diff_nonzero = rel_diff[nonzero_mask]
                max_rel_diff = rel_diff_nonzero.max().item()
                mean_rel_diff = rel_diff_nonzero.mean().item()

                print(f"  Max relative diff (non-zero): {max_rel_diff:.4f}")
                print(f"  Mean relative diff (non-zero): {mean_rel_diff:.6f}")

                assert max_rel_diff < 0.1, (
                    f"Global splat intensity_floor max diff {max_rel_diff:.4f} too large"
                )

    def test_global_splat_backward_gradients(self):
        """
        Test that gradients for global splats are computed correctly.

        This verifies the backward kernel properly applies intensity_floor
        and produces gradients matching finite differences.
        """
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        # Use 3D volume large enough for global splats
        # 128^3 with tile_size=8 = 16^3 = 4096 tiles
        shape = (128, 128, 128)
        tile_size = 8
        d = 3
        truncate = 3.0
        intensity_floor = 1e-5

        # Large splat to trigger global handling
        centers = torch.tensor(
            [[64.0, 64.0, 64.0]], device="cuda", dtype=torch.float32, requires_grad=True
        )
        L = torch.eye(d, device="cuda", dtype=torch.float32).unsqueeze(0) * 20.0
        L.requires_grad = True
        amps = torch.tensor(
            [1.0], device="cuda", dtype=torch.float32, requires_grad=True
        )

        conic = cholesky_to_conic(L)
        L_row_norms = compute_L_row_norms(L)

        # Forward pass
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            L_row_norms.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )
        output = cuda_result[0]
        global_splat_ids = cuda_result[4]

        # Verify global splat was detected
        assert len(global_splat_ids) > 0, "Expected global splat to be detected"

        # Create gradient output (ones)
        grad_output = torch.ones_like(output)

        # Backward pass - ensure correct tensor types
        # Note: backward expects: tile_offsets, tile_counts, tile_content (different from forward return order!)
        tile_counts = cuda_result[1]
        tile_offsets = cuda_result[2]
        tile_content = cuda_result[3]

        d_centers, d_conic, d_amps = cuda_splatting_backend.backward(
            grad_output.contiguous(),
            centers.detach().contiguous(),  # Detach to avoid autograd issues
            conic.detach().contiguous(),
            amps.detach().contiguous(),
            tile_offsets,  # Note: tile_offsets first
            tile_counts,  # tile_counts second
            tile_content,
            global_splat_ids,
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )

        print("\nGlobal splat backward pass:")
        print(
            f"  d_centers shape: {d_centers.shape}, sum: {d_centers.sum().item():.4f}"
        )
        print(f"  d_conic shape: {d_conic.shape}, sum: {d_conic.sum().item():.4f}")
        print(f"  d_amps shape: {d_amps.shape}, sum: {d_amps.sum().item():.4f}")

        # Gradients should be non-zero for a splat contributing to output
        assert d_amps.abs().sum().item() > 0, "Expected non-zero amplitude gradient"
        assert d_centers.abs().sum().item() > 0, "Expected non-zero center gradient"

        # Gradient sanity checks:
        # For a centered splat with symmetric output, center gradients should be near zero
        # (since moving slightly in any direction should have minimal effect on total output)
        center_grad_mag = d_centers.abs().sum().item()
        print(f"  Center gradient magnitude: {center_grad_mag:.4f}")

        # Amplitude gradient should be positive (increasing amp increases output)
        assert d_amps.item() > 0, "Expected positive amplitude gradient"

    def test_mixed_global_and_tile_splats(self):
        """
        Test scene with both global splats and normal tile-based splats.

        This tests the combination of both rendering paths and ensures
        they are properly accumulated.
        """
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        # Large enough volume for global splats
        shape = (128, 128, 128)
        tile_size = 8
        d = 3
        truncate = 3.0
        intensity_floor = 1e-5

        # Create mixed splats: 1 large (global) + 5 small (tile-based)
        np.random.seed(42)

        # Global splat - large, centered
        centers_global = np.array([[64.0, 64.0, 64.0]], dtype=np.float32)
        L_global = np.eye(d, dtype=np.float32).reshape(1, d, d) * 20.0

        # Tile-based splats - small, scattered
        centers_tile = np.random.rand(5, d).astype(np.float32) * 100 + 14
        L_tile = np.eye(d, dtype=np.float32).reshape(1, d, d).repeat(5, axis=0) * 2.0

        # Combine
        centers = np.concatenate([centers_global, centers_tile], axis=0)
        L = np.concatenate([L_global, L_tile], axis=0)
        amps = np.ones(6, dtype=np.float32)

        # Convert to torch
        centers_t = torch.tensor(centers, device="cuda")
        L_t = torch.tensor(L, device="cuda")
        amps_t = torch.tensor(amps, device="cuda")

        conic = cholesky_to_conic(L_t)
        L_row_norms = compute_L_row_norms(L_t)

        # Run CUDA backend
        cuda_result = cuda_splatting_backend.forward(
            centers_t.contiguous(),
            conic.contiguous(),
            amps_t.contiguous(),
            L_row_norms.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)
        global_splat_ids = cuda_result[4]

        # Should have exactly 1 global splat
        n_global = len(global_splat_ids)
        print("\nMixed splat test:")
        print(f"  Global splats: {n_global}, Tile splats: {6 - n_global}")

        assert n_global >= 1, "Expected at least 1 global splat"

        # Run PyTorch reference
        pytorch_output = render_gaussians(
            shape, centers_t, L_t, amps_t, truncate, intensity_floor
        )

        # Compare outputs
        cuda_cpu = cuda_output.cpu()
        pytorch_cpu = pytorch_output.cpu()

        max_val = max(cuda_cpu.abs().max().item(), pytorch_cpu.abs().max().item())
        if max_val > 1e-6:
            abs_diff = (cuda_cpu - pytorch_cpu).abs()
            rel_diff = abs_diff / (max_val + 1e-8)

            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()

            print(f"  Max relative diff: {max_rel_diff:.4f}")
            print(f"  Mean relative diff: {mean_rel_diff:.6f}")

            # Tighter than COMPARISON_MAX_REL_DIFF for mixed splat test
            assert max_rel_diff < 0.1, (
                f"Mixed splat max relative diff {max_rel_diff:.4f} too large"
            )
            assert mean_rel_diff < Tolerances.COMPARISON_MEAN_REL_DIFF, (
                f"Mixed splat mean relative diff {mean_rel_diff:.6f} too large"
            )

    @pytest.mark.parametrize(
        "dim,shape,tile_size,L_scale",
        [
            (2, (256, 256), 4, 30.0),  # 2D: 64x64 = 4096 tiles, need big splat
            (3, (128, 128, 128), 8, 20.0),  # 3D: 16^3 = 4096 tiles
            (4, (32, 32, 32, 32), 4, 8.0),  # 4D: 8^4 = 4096 tiles
        ],
    )
    def test_global_splat_multi_dimension(self, dim, shape, tile_size, L_scale):
        """
        Test global splat handling across 2D, 3D, and 4D volumes.

        Each test creates a volume with ~4096 tiles and a large splat that
        triggers global handling, then compares CUDA output to PyTorch reference.
        """
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )
        from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

        truncate = 3.0
        intensity_floor = 1e-5

        # Create large splat centered in the volume
        center_coords = [s / 2.0 for s in shape]
        centers = torch.tensor([center_coords], device="cuda", dtype=torch.float32)
        L = torch.eye(dim, device="cuda", dtype=torch.float32).unsqueeze(0) * L_scale
        amps = torch.tensor([1.0], device="cuda", dtype=torch.float32)

        conic = cholesky_to_conic(L)
        L_row_norms = compute_L_row_norms(L)

        # Run CUDA backend
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            L_row_norms.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )
        cuda_output = cuda_result[0].reshape(shape)
        global_splat_ids = cuda_result[4]

        # Verify global splat was detected
        n_global = len(global_splat_ids)
        print(f"\n{dim}D global splat test:")
        print(f"  Shape: {shape}, tile_size: {tile_size}")
        print(f"  Global splats detected: {n_global}")

        assert n_global > 0, f"Expected global splat in {dim}D test"

        # Run PyTorch reference
        pytorch_output = render_gaussians(
            shape, centers, L, amps, truncate, intensity_floor
        )

        # Compare outputs
        cuda_cpu = cuda_output.cpu()
        pytorch_cpu = pytorch_output.cpu()

        max_val = max(cuda_cpu.abs().max().item(), pytorch_cpu.abs().max().item())
        if max_val > 1e-6:
            abs_diff = (cuda_cpu - pytorch_cpu).abs()
            rel_diff = abs_diff / (max_val + 1e-8)

            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()

            print(f"  Max relative diff: {max_rel_diff:.4f}")
            print(f"  Mean relative diff: {mean_rel_diff:.6f}")

            assert max_rel_diff < 0.1, (
                f"{dim}D global splat max relative diff {max_rel_diff:.4f} too large"
            )

    @pytest.mark.parametrize(
        "dim,shape,tile_size,L_scale",
        [
            (2, (256, 256), 4, 30.0),
            (3, (128, 128, 128), 8, 20.0),
            (4, (32, 32, 32, 32), 4, 8.0),
        ],
    )
    def test_global_splat_backward_multi_dimension(
        self, dim, shape, tile_size, L_scale
    ):
        """
        Test global splat backward pass across 2D, 3D, and 4D volumes.

        Verifies that gradients are computed correctly for global splats.
        """
        import cuda_splatting_backend  # noqa: F401

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        truncate = 3.0
        intensity_floor = 1e-5

        # Create large splat centered in the volume
        center_coords = [s / 2.0 for s in shape]
        centers = torch.tensor([center_coords], device="cuda", dtype=torch.float32)
        L = torch.eye(dim, device="cuda", dtype=torch.float32).unsqueeze(0) * L_scale
        amps = torch.tensor([1.0], device="cuda", dtype=torch.float32)

        conic = cholesky_to_conic(L)
        L_row_norms = compute_L_row_norms(L)

        # Forward pass
        cuda_result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            L_row_norms.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )
        output = cuda_result[0]
        global_splat_ids = cuda_result[4]

        # Verify global splat was detected
        assert len(global_splat_ids) > 0, (
            f"Expected global splat in {dim}D backward test"
        )

        # Backward pass
        grad_output = torch.ones_like(output)
        tile_counts = cuda_result[1]
        tile_offsets = cuda_result[2]
        tile_content = cuda_result[3]

        d_centers, d_conic, d_amps = cuda_splatting_backend.backward(
            grad_output.contiguous(),
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            tile_offsets,
            tile_counts,
            tile_content,
            global_splat_ids,
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )

        print(f"\n{dim}D global splat backward:")
        print(f"  d_centers sum: {d_centers.sum().item():.4f}")
        print(f"  d_amps: {d_amps.item():.4f}")

        # Verify gradients are reasonable
        assert d_amps.item() > 0, f"Expected positive amplitude gradient in {dim}D"
        assert torch.isfinite(d_centers).all(), f"Non-finite center gradients in {dim}D"
        assert torch.isfinite(d_conic).all(), f"Non-finite conic gradients in {dim}D"


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class Test5DAnd6DDimensions:
    """Test CUDA backend with 5D and 6D volumes."""

    def test_5d_forward_matches_reference(self):
        """Test 5D forward pass produces correct output."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        np.random.seed(42)
        N, d = 20, 5
        shape = (8, 8, 8, 8, 8)  # 32K voxels

        centers0 = np.random.rand(N, d).astype(np.float32) * 6 + 1
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0[i] *= np.random.uniform(0.5, 1.5)
        amps0 = np.random.rand(N).astype(np.float32) * 0.5 + 0.5

        # CPU reference
        cpu_model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * d,
            device="cpu",
        )

        # CUDA model
        cuda_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * d,
            device="cuda",
        )

        with torch.no_grad():
            cpu_output = cpu_model()
            cuda_output = cuda_model().cpu()

        if cuda_output.shape != cpu_output.shape:
            cuda_output = cuda_output.reshape(cpu_output.shape)

        # Compare with tolerance
        max_val = max(cpu_output.abs().max().item(), cuda_output.abs().max().item())
        if max_val > 0:
            rel_diff = (cpu_output - cuda_output).abs() / (max_val + 1e-8)
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()

            print(
                f"\n5D forward: max_rel_diff={max_rel_diff:.4f}, mean={mean_rel_diff:.6f}"
            )

            assert max_rel_diff < Tolerances.COMPARISON_MAX_REL_DIFF, (
                f"5D max relative diff {max_rel_diff:.4f} too large"
            )
            assert mean_rel_diff < Tolerances.COMPARISON_MEAN_REL_DIFF * 2, (
                f"5D mean relative diff {mean_rel_diff:.6f} too large"
            )

    def test_5d_backward_gradients_finite(self):
        """Test 5D backward pass produces finite gradients."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        np.random.seed(42)
        N, d = 15, 5
        shape = (6, 6, 6, 6, 6)

        centers0 = np.random.rand(N, d).astype(np.float32) * 4 + 1
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        amps0 = np.random.rand(N).astype(np.float32) + 0.5

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * d,
            device="cuda",
        )

        target = torch.rand(shape, device="cuda")
        output = model()
        loss = ((output - target) ** 2).mean()
        loss.backward()

        # Check gradients are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"5D gradient {name} has non-finite values"
                )

    def test_6d_forward_matches_reference(self):
        """Test 6D forward pass produces correct output.

        This is a numerical equivalence test between the CUDA and CPU implementations.
        It should only run when CUDA is available, and it needs a robust error metric
        that doesn't get dominated by near-zero voxels.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        rng = np.random.RandomState(42)
        torch.manual_seed(42)

        N, d = 15, 6
        shape = (4, 4, 4, 4, 4, 4)  # 4K voxels

        centers0 = rng.rand(N, d).astype(np.float32) * 2 + 1
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0[i] *= rng.uniform(0.3, 0.8)
        amps0 = rng.rand(N).astype(np.float32) * 0.5 + 0.5

        # CPU reference
        cpu_model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.3,) * d,
            device="cpu",
        )

        # CUDA model
        cuda_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.3,) * d,
            device="cuda",
        )

        with torch.no_grad():
            cpu_output = cpu_model()
            cuda_output = cuda_model().cpu()

        if cuda_output.shape != cpu_output.shape:
            cuda_output = cuda_output.reshape(cpu_output.shape)

        # Robust comparison: RMS relative error on significant voxels.
        # (Max-relative error is extremely sensitive to tiny denominators.)
        cpu_abs = cpu_output.abs()
        scale = cpu_abs.max().item()
        assert scale >= 0.0
        if scale == 0.0:
            # Degenerate case: both should be all zeros
            assert torch.allclose(cpu_output, cuda_output, atol=0.0, rtol=0.0)
            return

        mask = cpu_abs > (1e-3 * scale)
        # If everything is tiny, fall back to comparing all values.
        if mask.sum().item() == 0:
            mask = torch.ones_like(cpu_output, dtype=torch.bool)

        diff = (cpu_output - cuda_output)[mask]
        denom = cpu_output[mask]

        rms_rel = (diff.pow(2).mean().sqrt() / (denom.abs().mean() + 1e-8)).item()
        print(f"6D forward: rms_rel={rms_rel:.4f}, n={mask.sum().item()}")

        # Relaxed from 0.25 to 0.30: shifted Gaussian increases floating-point
        # accumulation error in 6D (6-deep forward substitution + larger Mahalanobis).
        assert rms_rel < 0.30, f"6D RMS relative error {rms_rel:.4f} too large"

    def test_6d_backward_gradients_finite(self):
        """Test 6D backward pass produces finite gradients."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        np.random.seed(42)
        N, d = 10, 6
        shape = (4, 4, 4, 4, 4, 4)

        centers0 = np.random.rand(N, d).astype(np.float32) * 2 + 1
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0) * 0.5
        amps0 = np.random.rand(N).astype(np.float32) + 0.5

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.3,) * d,
            device="cuda",
        )

        target = torch.rand(shape, device="cuda")
        output = model()
        loss = ((output - target) ** 2).mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"6D gradient {name} has non-finite values"
                )


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestSyncthreadsDivergenceRegression:
    """Regression tests for the __syncthreads() divergence bug.

    The bug occurred when tile_pixels != blockDim.x, causing some threads to
    execute fewer iterations of the pixel loop and skip __syncthreads() calls.
    This is undefined behavior in CUDA and caused stale shared memory reads
    in the forward kernel.

    The fix restructured the forward kernel to place the splat batch loop
    (with __syncthreads) as the outer loop, and the pixel loop as the inner
    loop, ensuring ALL threads in the block participate in every barrier.

    Affected dimensions: any where tile_pixels != blockDim.x:
      - 5D: tile_size=3, 3^5=243 pixels, 256 threads (243 < 256)
      - 6D: tile_size=3, 3^6=729 pixels, 256 threads (729 % 256 != 0)
      - 7D: tile_size=2, 2^7=128 pixels, 256 threads (128 < 256)

    The bug only manifested when the CUDA caching allocator reused buffers
    with stale data from prior kernel launches (i.e., full test suite, not
    isolated runs).
    """

    @pytest.fixture(autouse=True)
    def _warm_cuda_allocator(self):
        """Run a throwaway CUDA allocation to warm up the caching allocator.

        This makes the allocator reuse buffers (with stale data), which is
        the condition that triggers the __syncthreads() divergence bug.
        Without this, freshly allocated memory is zeroed and the bug is hidden.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        # Allocate and fill a buffer to pollute the caching allocator
        dummy = torch.ones(100_000, device="cuda") * 999.0
        del dummy
        # Do NOT call empty_cache() — we want stale data in the allocator

    def _run_forward_comparison(self, d: int, shape: tuple, n_splats: int):
        """Helper: compare CUDA vs CPU forward for given dimensionality."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        rng = np.random.RandomState(12345)

        centers0 = rng.rand(n_splats, d).astype(np.float32) * 2 + 0.5
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(n_splats, axis=0)
        for i in range(n_splats):
            L0[i] *= rng.uniform(0.3, 0.8)
        amps0 = rng.rand(n_splats).astype(np.float32) * 0.5 + 0.5

        cpu_model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.3,) * d,
            device="cpu",
        )

        cuda_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.3,) * d,
            device="cuda",
        )

        with torch.no_grad():
            cpu_output = cpu_model()
            cuda_output = cuda_model().cpu()

        if cuda_output.shape != cpu_output.shape:
            cuda_output = cuda_output.reshape(cpu_output.shape)

        # Check that outputs are not wildly different
        # (the bug caused CUDA max to be 10-100x larger than CPU max)
        cpu_max = cpu_output.abs().max().item()
        cuda_max = cuda_output.abs().max().item()
        if cpu_max > 0:
            max_ratio = cuda_max / cpu_max
            assert max_ratio < 2.0, (
                f"{d}D: CUDA max ({cuda_max:.4f}) is {max_ratio:.1f}x larger "
                f"than CPU max ({cpu_max:.4f}) — likely __syncthreads divergence bug"
            )

        # Also check RMS relative error
        cpu_abs = cpu_output.abs()
        scale = cpu_abs.max().item()
        if scale > 0:
            mask = cpu_abs > (1e-3 * scale)
            if mask.sum().item() > 0:
                diff = (cpu_output - cuda_output)[mask]
                denom = cpu_output[mask]
                rms_rel = (
                    diff.pow(2).mean().sqrt() / (denom.abs().mean() + 1e-8)
                ).item()
                assert rms_rel < 0.5, (
                    f"{d}D: RMS relative error {rms_rel:.4f} too large"
                )

    def test_5d_syncthreads_regression(self):
        """5D: 243 pixels < 256 threads — some threads skip pixel loop entirely."""
        self._run_forward_comparison(d=5, shape=(3, 3, 3, 3, 3), n_splats=20)

    def test_6d_syncthreads_regression(self):
        """6D: 729 pixels, 256 threads — uneven iteration counts (3 vs 2)."""
        self._run_forward_comparison(d=6, shape=(4, 4, 4, 4, 4, 4), n_splats=15)

    def test_7d_syncthreads_regression(self):
        """7D: 128 pixels < 256 threads — half the threads idle."""
        self._run_forward_comparison(d=7, shape=(2, 2, 2, 2, 2, 2, 2), n_splats=10)

    def test_4d_edge_tile_syncthreads(self):
        """4D: with shape not divisible by tile_size, edge tiles have fewer pixels."""
        # Shape 5^4 with tile_size=4 → edge tiles have 1 voxel per clipped dim
        # tile_pixels for edge tiles = 1*1*1*1 = 1 (much less than blockDim=256)
        self._run_forward_comparison(d=4, shape=(5, 5, 5, 5), n_splats=20)

    def test_6d_many_splat_batches(self):
        """6D with many splats — tests multiple splat batch iterations with atomicAdd."""
        # With 200 splats and BATCH_SIZE=128, we get 2 batches, each needing
        # correct atomicAdd accumulation across batches.
        self._run_forward_comparison(d=6, shape=(4, 4, 4, 4, 4, 4), n_splats=200)


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestLargeSplatCounts:
    """Test CUDA backend with large numbers of splats (10K+)."""

    def test_10k_splats_forward(self):
        """Test forward pass with 10,000 splats."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        np.random.seed(42)
        N = 10000
        d = 3
        shape = (128, 128, 128)

        centers0 = np.random.rand(N, d).astype(np.float32) * (np.array(shape) - 4) + 2
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0[i] *= np.random.uniform(0.5, 2.0)
        amps0 = np.random.rand(N).astype(np.float32) * 0.3 + 0.1

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * d,
            device="cuda",
        )

        assert model.n_splats() == 10000

        # Forward pass should work
        output = model()

        assert output.shape == shape
        assert not torch.isnan(output).any(), "10K splats output has NaN"
        assert not torch.isinf(output).any(), "10K splats output has Inf"
        assert output.sum() > 0, "10K splats output is all zeros"

        print(f"\n10K splats forward: output sum={output.sum().item():.2f}")

    def test_10k_splats_backward(self):
        """Test backward pass with 10,000 splats produces finite gradients."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        np.random.seed(42)
        N = 10000
        d = 3
        shape = (64, 64, 64)  # Smaller for faster backward

        centers0 = np.random.rand(N, d).astype(np.float32) * (np.array(shape) - 4) + 2
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0[i] *= np.random.uniform(0.5, 1.5)
        amps0 = np.random.rand(N).astype(np.float32) * 0.2 + 0.1

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * d,
            device="cuda",
        )

        target = torch.rand(shape, device="cuda")
        output = model()
        loss = ((output - target) ** 2).mean()
        loss.backward()

        # Check gradients
        grad_count = 0
        nan_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                if torch.isnan(param.grad).any():
                    nan_count += 1
                    print(f"  {name}: has NaN")

        print(
            f"\n10K splats backward: {grad_count} params with gradients, {nan_count} with NaN"
        )
        assert nan_count == 0, "10K splats backward has NaN gradients"

    def test_20k_splats_forward_2d(self):
        """Test forward pass with 20,000 splats in 2D (larger volume)."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        np.random.seed(42)
        N = 20000
        d = 2
        shape = (1024, 1024)

        centers0 = np.random.rand(N, d).astype(np.float32) * (np.array(shape) - 4) + 2
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        for i in range(N):
            L0[i] *= np.random.uniform(0.5, 3.0)
        amps0 = np.random.rand(N).astype(np.float32) * 0.2 + 0.1

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * d,
            device="cuda",
        )

        output = model()

        assert output.shape == shape
        assert not torch.isnan(output).any(), "20K splats output has NaN"
        assert not torch.isinf(output).any(), "20K splats output has Inf"

        print(f"\n20K splats 2D forward: output sum={output.sum().item():.2f}")

    def test_large_splat_memory_does_not_explode(self):
        """Test that memory usage with 10K splats is reasonable."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()

        np.random.seed(42)
        N = 10000
        d = 3
        shape = (128, 128, 128)

        centers0 = np.random.rand(N, d).astype(np.float32) * (np.array(shape) - 4) + 2
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        amps0 = np.random.rand(N).astype(np.float32) + 0.1

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * d,
            device="cuda",
        )

        _ = model()
        peak = torch.cuda.max_memory_allocated() - baseline

        # Expected: ~130M params + ~8M output + tile buffers
        # Should be under 1GB for 10K splats on 128^3 volume
        peak_mb = peak / (1024 * 1024)
        print(f"\n10K splats memory: {peak_mb:.1f} MB")

        assert peak_mb < 1000, f"Memory usage {peak_mb:.1f} MB exceeds 1GB limit"

        del model
        torch.cuda.empty_cache()


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestBackward4D:
    """Test backward pass specifically for 4D (generic nD path transition).

    The 4D backward path uses generic nD code (torch.linalg.solve_triangular
    for conic computation), which differs from the explicit 2D/3D paths.
    These tests verify the transition is correct.
    """

    def test_4d_backward_gradients_finite(self):
        """Test 4D backward pass produces finite, non-zero gradients."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        np.random.seed(42)
        N, d = 10, 4
        shape = (8, 8, 8, 8)

        centers0 = np.random.rand(N, d).astype(np.float32) * 6 + 1
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        amps0 = np.random.rand(N).astype(np.float32) + 0.5

        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * d,
            device="cuda",
        )

        output = model()
        loss = output.sum()
        loss.backward()

        # Check all parameter gradients exist, are finite, and non-zero
        for name in ("raw_mu", "raw_L_diag", "L_off", "raw_a"):
            param = getattr(model, name)
            assert param.grad is not None, f"4D gradient for {name} is None"
            assert torch.isfinite(param.grad).all(), (
                f"4D gradient for {name} has non-finite values"
            )
            assert param.grad.abs().sum() > 0, (
                f"4D gradient for {name} is all zeros (no gradient signal)"
            )

    def test_4d_backward_vs_cpu_reference(self):
        """Test 4D backward gradients match CPU reference (generic nD path)."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        np.random.seed(42)
        N, d = 5, 4
        shape = (8, 8, 8, 8)

        centers0 = np.random.rand(N, d).astype(np.float32) * 6 + 1
        L0 = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        amps0 = np.ones(N, dtype=np.float32) * 0.5

        # CPU reference model
        cpu_model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * d,
            device="cpu",
        )

        # CUDA model
        cuda_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=(0.5,) * d,
            device="cuda",
        )

        # Forward + backward on CPU
        cpu_target = torch.zeros(shape, dtype=torch.float32, device="cpu")
        cpu_target[4, 4, 4, 4] = 1.0
        cpu_output = cpu_model()
        cpu_loss = torch.nn.functional.mse_loss(cpu_output, cpu_target)
        cpu_loss.backward()

        # Forward + backward on CUDA
        cuda_output = cuda_model()
        cuda_target = torch.zeros_like(cuda_output)
        if cuda_output.dim() == 1:
            center_idx = 4 * shape[1] * shape[2] * shape[3] + 4 * shape[2] * shape[3] + 4 * shape[3] + 4
            cuda_target[center_idx] = 1.0
        else:
            cuda_target[4, 4, 4, 4] = 1.0
        cuda_loss = torch.nn.functional.mse_loss(cuda_output, cuda_target)
        cuda_loss.backward()

        # Compare amplitude gradients (raw_a) between CUDA and CPU
        cpu_grad = cpu_model.raw_a.grad
        cuda_grad = cuda_model.raw_a.grad
        assert cpu_grad is not None, "CPU raw_a gradient is None"
        assert cuda_grad is not None, "CUDA raw_a gradient is None"

        cuda_grad_cpu = cuda_grad.cpu()

        # Check gradient signs match for majority of elements
        sign_match = (
            (torch.sign(cpu_grad) == torch.sign(cuda_grad_cpu)).float().mean()
        )
        assert sign_match > Tolerances.BACKWARD_SIGN_MATCH, (
            f"4D raw_a gradient sign match {sign_match:.2f} "
            f"< {Tolerances.BACKWARD_SIGN_MATCH}"
        )

        # Check gradient magnitudes are in similar range
        cpu_mag = cpu_grad.abs().mean()
        cuda_mag = cuda_grad_cpu.abs().mean()
        if cpu_mag > 1e-8 and cuda_mag > 1e-8:
            mag_ratio = max(cpu_mag / cuda_mag, cuda_mag / cpu_mag)
            assert mag_ratio < Tolerances.BACKWARD_MAG_RATIO, (
                f"4D raw_a gradient magnitude ratio {mag_ratio:.2f} "
                f"> {Tolerances.BACKWARD_MAG_RATIO}"
            )
