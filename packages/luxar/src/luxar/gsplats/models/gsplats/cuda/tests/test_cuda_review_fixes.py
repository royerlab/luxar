"""
Tests for CUDA kernel review fixes.

These tests verify the correctness of fixes identified during the CUDA kernel review:
1. Buffer overflow guard in backward kernel's grad_output cache
2. Tighter early rejection via amplitude-based truncation in effective_truncate_sq
"""

import numpy as np
import pytest
import torch

from .conftest import Tolerances, compute_L_row_norms

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# Check if CUDA backend is compiled
try:
    import cuda_splatting_backend

    CUDA_BACKEND_AVAILABLE = True
except ImportError:
    CUDA_BACKEND_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
    pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled"),
]


def _make_splat_data(
    N: int,
    d: int,
    shape: tuple[int, ...],
    centers: np.ndarray | None = None,
    L_scale: float = 1.5,
    amps: np.ndarray | None = None,
):
    """Helper to create splat data tensors on CUDA."""
    from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import cholesky_to_conic

    device = torch.device("cuda:0")

    if centers is None:
        centers_t = (
            torch.rand(N, d, device=device, dtype=torch.float32) * (min(shape) - 4) + 2
        )
    else:
        centers_t = torch.tensor(centers, device=device, dtype=torch.float32)

    L = (
        torch.eye(d, device=device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(N, -1, -1)
        .clone()
        * L_scale
    )

    if amps is None:
        amps_t = torch.rand(N, device=device, dtype=torch.float32) * 0.5 + 0.5
    else:
        amps_t = torch.tensor(amps, device=device, dtype=torch.float32)

    conic = cholesky_to_conic(L)
    L_row_norms = compute_L_row_norms(L)

    return {
        "centers": centers_t,
        "L": L,
        "conic": conic,
        "amps": amps_t,
        "L_row_norms": L_row_norms,
        "shape": shape,
    }


def _run_cuda_forward(data, tile_size, truncate=3.0, intensity_floor=1e-5):
    """Run CUDA forward pass and return reshaped output + state."""
    result = cuda_splatting_backend.forward(
        data["centers"].contiguous(),
        data["conic"].contiguous(),
        data["amps"].contiguous(),
        data["L_row_norms"].contiguous(),
        list(data["shape"]),
        truncate,
        intensity_floor,
        tile_size,
    )
    output = result[0].reshape(data["shape"])
    return output, result


def _run_pytorch_reference(data, truncate=3.0, intensity_floor=1e-5):
    """Run PyTorch reference implementation."""
    from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

    return render_gaussians(
        data["shape"],
        data["centers"],
        data["L"],
        data["amps"],
        truncate,
        intensity_floor,
    )


class TestAmplitudeBasedTruncation:
    """Tests for Fix 2: amplitude-based tightening in effective_truncate_sq."""

    def test_low_amplitude_splats_match_pytorch(self):
        """Splats with low amplitude (near intensity_floor) should still match PyTorch."""
        np.random.seed(42)
        N, d = 20, 3
        shape = (24, 24, 24)
        intensity_floor = 1e-5

        # Mix of normal and low-amplitude splats
        amps = np.array(
            [1.0] * 10  # Normal amplitude
            + [intensity_floor * 2] * 5  # Just above floor (very tight truncation)
            + [intensity_floor * 100] * 5,  # Moderate amplitude
            dtype=np.float32,
        )

        data = _make_splat_data(N, d, shape, amps=amps)
        cuda_out, _ = _run_cuda_forward(
            data, tile_size=8, intensity_floor=intensity_floor
        )
        pytorch_out = _run_pytorch_reference(data, intensity_floor=intensity_floor)

        cuda_cpu = cuda_out.cpu()
        pytorch_cpu = pytorch_out.cpu()

        max_val = max(cuda_cpu.abs().max().item(), pytorch_cpu.abs().max().item())
        if max_val > 1e-6:
            rel_diff = (cuda_cpu - pytorch_cpu).abs() / (max_val + 1e-8)
            assert rel_diff.max().item() < Tolerances.COMPARISON_MAX_REL_DIFF, (
                f"Max relative diff {rel_diff.max().item():.4f} exceeds tolerance"
            )
            assert rel_diff.mean().item() < Tolerances.COMPARISON_MEAN_REL_DIFF, (
                f"Mean relative diff {rel_diff.mean().item():.6f} exceeds tolerance"
            )

    def test_amplitude_near_floor_correctness(self):
        """Splats with amplitude near/below intensity_floor produce correct output.

        Near the intensity floor, CUDA fast-math (__expf) and PyTorch may disagree
        on whether a pixel crosses the threshold. Instead of checking exact match,
        we verify:
        1. All output values are bounded by amplitude (physical maximum)
        2. Below-floor splats produce negligible output
        3. The total energy is bounded appropriately
        """
        N_above = 5
        N_below = 5
        d = 3
        shape = (16, 16, 16)
        intensity_floor = 1e-3

        centers_above = np.array(
            [[4, 4, 4], [8, 8, 8], [12, 12, 12], [4, 8, 12], [8, 4, 8]],
            dtype=np.float32,
        )
        centers_below = np.array(
            [[5, 5, 5], [9, 9, 9], [11, 11, 11], [5, 9, 11], [9, 5, 9]],
            dtype=np.float32,
        )
        centers = np.concatenate([centers_above, centers_below], axis=0)

        amps_above = np.full(N_above, intensity_floor * 1.01, dtype=np.float32)
        amps_below = np.full(N_below, intensity_floor * 0.99, dtype=np.float32)
        amps = np.concatenate([amps_above, amps_below])

        data = _make_splat_data(N_above + N_below, d, shape, centers=centers, amps=amps)
        cuda_out, _ = _run_cuda_forward(
            data, tile_size=8, intensity_floor=intensity_floor
        )

        cuda_cpu = cuda_out.cpu()

        # Physical constraint: no pixel can exceed max amplitude
        max_amp = float(max(amps))
        assert cuda_cpu.max().item() <= max_amp * 1.01, (
            f"Output {cuda_cpu.max().item():.6f} exceeds max amplitude {max_amp:.6f}"
        )

        # All non-zero values should be near or above the floor
        nonzero_mask = cuda_cpu > 0
        if nonzero_mask.any():
            min_nonzero = cuda_cpu[nonzero_mask].min().item()
            # Allow small tolerance for fast-math differences
            assert min_nonzero >= intensity_floor * 0.9, (
                f"Non-zero output {min_nonzero:.6f} significantly below floor {intensity_floor}"
            )

        # Total output should be small (all splats have tiny amplitudes)
        assert cuda_cpu.sum().item() < max_amp * 100, (
            "Total output unexpectedly large for near-floor splats"
        )

    def test_low_amplitude_2d(self):
        """2D version: low-amplitude splats match PyTorch reference."""
        np.random.seed(123)
        N, d = 15, 2
        shape = (48, 48)
        intensity_floor = 1e-5

        amps = np.array(
            [1.0] * 5 + [intensity_floor * 5] * 5 + [intensity_floor * 50] * 5,
            dtype=np.float32,
        )

        data = _make_splat_data(N, d, shape, amps=amps)
        cuda_out, _ = _run_cuda_forward(
            data, tile_size=16, intensity_floor=intensity_floor
        )
        pytorch_out = _run_pytorch_reference(data, intensity_floor=intensity_floor)

        cuda_cpu = cuda_out.cpu()
        pytorch_cpu = pytorch_out.cpu()

        max_val = max(cuda_cpu.abs().max().item(), pytorch_cpu.abs().max().item())
        if max_val > 1e-6:
            rel_diff = (cuda_cpu - pytorch_cpu).abs() / (max_val + 1e-8)
            assert rel_diff.max().item() < Tolerances.COMPARISON_MAX_REL_DIFF


class TestGradCacheBufferOverflowGuard:
    """Tests for Fix 1: buffer overflow guard in backward kernel's grad_output cache.

    The backward kernel has a fixed-size shared memory buffer (MAX_TILE_PIXELS)
    for caching grad_output. When a non-default tile_size is used such that
    tile_pixels > MAX_TILE_PIXELS, the cache must be disabled to avoid OOB writes.

    For 3D: MAX_TILE_PIXELS=512 (8^3), so tile_size=16 gives 16^3=4096 > 512.
    For 2D: MAX_TILE_PIXELS=256 (16^2), so tile_size=32 gives 32^2=1024 > 256.
    """

    def test_3d_oversized_tile_backward_no_crash(self):
        """3D backward with tile_size=16 (4096 pixels > 512 MAX_TILE_PIXELS) should not crash.

        Before Fix 1, this would write past the s_grad_output[512] buffer.
        After Fix 1, the kernel falls back to global memory reads for grad_output.
        """
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        np.random.seed(55)
        N, d = 10, 3
        shape = (32, 32, 32)
        tile_size = 16  # 16^3 = 4096 >> 512 MAX_TILE_PIXELS
        truncate = 3.0
        intensity_floor = 1e-5
        device = torch.device("cuda:0")

        centers = torch.rand(N, d, device=device, dtype=torch.float32) * 28 + 2
        L = (
            torch.eye(d, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(N, -1, -1)
            .clone()
            * 2.0
        )
        amps = torch.rand(N, device=device, dtype=torch.float32) * 0.5 + 0.5

        conic = cholesky_to_conic(L)
        L_row_norms = compute_L_row_norms(L)

        # Forward with oversized tile
        result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            L_row_norms.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )
        output = result[0]

        # Backward with oversized tile - this would crash without Fix 1
        grad_output = torch.ones_like(output)
        d_centers, d_conic, d_amps = cuda_splatting_backend.backward(
            grad_output.contiguous(),
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            shape_tensor_cached=result[1],
        )

        # Should produce finite gradients (not crash or produce garbage)
        assert torch.isfinite(d_centers).all(), "d_centers has non-finite values"
        assert torch.isfinite(d_conic).all(), "d_conic has non-finite values"
        assert torch.isfinite(d_amps).all(), "d_amps has non-finite values"

        # Amplitude gradients should be positive (uniform upstream gradient)
        active = d_amps.abs() > 1e-10
        if active.any():
            assert (d_amps[active] > 0).all()

    def test_3d_oversized_tile_forward_matches_default(self):
        """3D forward with tile_size=16 should produce the same result as tile_size=8.

        The tile_size only affects spatial binning, not the final output.
        """
        np.random.seed(66)
        N, d = 8, 3
        shape = (24, 24, 24)
        truncate = 3.0
        intensity_floor = 1e-5

        data = _make_splat_data(N, d, shape, L_scale=2.0)

        # Forward with default tile_size=8
        out_default, _ = _run_cuda_forward(
            data, tile_size=8, truncate=truncate, intensity_floor=intensity_floor
        )
        # Forward with oversized tile_size=16
        out_oversized, _ = _run_cuda_forward(
            data, tile_size=16, truncate=truncate, intensity_floor=intensity_floor
        )

        # Results should be very close (both FP32, same algorithm, just different tiling)
        diff = (out_default.cpu() - out_oversized.cpu()).abs()
        assert diff.max().item() < 1e-5, (
            f"tile_size=8 vs 16 differ by {diff.max().item():.6e}"
        )
