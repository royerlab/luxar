"""
CUDA kernel safety and numerical-invariant regression tests.

These tests guard correctness invariants of the CUDA splatting kernels:

- The backward kernel's grad_output cache must not overflow its buffer, and
  the backward pass must always yield finite, physically sensible gradients.
- Forward and backward must run on the CUDA device that owns their tensors.
- Amplitude-based early rejection in ``effective_truncate_sq`` must keep
  low-amplitude (near-``intensity_floor``) splats consistent with the PyTorch
  reference, and kernel output must stay bounded by per-splat amplitude.
"""

import numpy as np
import pytest
import torch

from .conftest import Tolerances

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

    return {
        "centers": centers_t,
        "L": L,
        "conic": conic,
        "amps": amps_t,
        "shape": shape,
    }


def _run_cuda_forward(data, truncate=3.0, intensity_floor=1e-5):
    """Run CUDA forward pass and return reshaped output + state."""
    result = cuda_splatting_backend.forward(
        data["centers"].contiguous(),
        data["conic"].contiguous(),
        data["amps"].contiguous(),
        list(data["shape"]),
        truncate,
        intensity_floor,
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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need multiple GPUs")
def test_forward_backward_on_non_default_cuda_device():
    """Kernels should match the default GPU and restore the caller's device.

    An illegal access here poisons the CUDA context and causes later tests in
    the worker to fail for unrelated reasons.
    """
    from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import cholesky_to_conic

    shape = (16, 16, 16)

    def run(device_index):
        device = torch.device(f"cuda:{device_index}")
        centers = torch.tensor([[8.0, 8.0, 8.0]], device=device)
        cholesky = torch.eye(3, device=device).unsqueeze(0)
        conic = cholesky_to_conic(cholesky)
        amps = torch.ones(1, device=device)

        with torch.cuda.device(0):
            output, shape_tensor = cuda_splatting_backend.forward(
                centers.contiguous(),
                conic.contiguous(),
                amps.contiguous(),
                list(shape),
                3.0,
                1e-5,
            )
            gradients = cuda_splatting_backend.backward(
                torch.ones_like(output),
                centers.contiguous(),
                conic.contiguous(),
                amps.contiguous(),
                list(shape),
                3.0,
                1e-5,
                shape_tensor_cached=shape_tensor,
            )
            assert torch.cuda.current_device() == 0

        assert output.device == device
        assert shape_tensor.device == device
        assert all(gradient.device == device for gradient in gradients)
        return output.cpu(), tuple(gradient.cpu() for gradient in gradients)

    default_output, default_gradients = run(0)
    other_output, other_gradients = run(1)

    torch.testing.assert_close(other_output, default_output)
    for other_gradient, default_gradient in zip(
        other_gradients, default_gradients, strict=True
    ):
        torch.testing.assert_close(other_gradient, default_gradient)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need multiple GPUs")
@pytest.mark.parametrize("mismatched_name", ["conic", "amps"])
def test_forward_rejects_mixed_cuda_devices(mismatched_name):
    """Forward should reject mixed-device inputs before launching a kernel."""
    centers = torch.tensor([[8.0, 8.0, 8.0]], device="cuda:1")
    conic = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]], device="cuda:1")
    amps = torch.ones(1, device="cuda:1")
    tensors = {"centers": centers, "conic": conic, "amps": amps}
    tensors[mismatched_name] = tensors[mismatched_name].to("cuda:0")

    with pytest.raises(RuntimeError, match="same CUDA device"):
        cuda_splatting_backend.forward(
            tensors["centers"],
            tensors["conic"],
            tensors["amps"],
            [16, 16, 16],
            3.0,
            1e-5,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need multiple GPUs")
def test_backward_rejects_mixed_cuda_devices():
    """Backward should reject a gradient from a different CUDA device."""
    centers = torch.tensor([[8.0, 8.0, 8.0]], device="cuda:1")
    conic = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]], device="cuda:1")
    amps = torch.ones(1, device="cuda:1")
    grad_output = torch.ones(16**3, device="cuda:0")

    with pytest.raises(RuntimeError, match="same CUDA device"):
        cuda_splatting_backend.backward(
            grad_output,
            centers,
            conic,
            amps,
            [16, 16, 16],
            3.0,
            1e-5,
        )


class TestAmplitudeBasedTruncation:
    """Invariant: amplitude-based truncation in effective_truncate_sq keeps
    low-amplitude splats consistent with the PyTorch reference and output
    bounded by amplitude."""

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
        cuda_out, _ = _run_cuda_forward(data, intensity_floor=intensity_floor)
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
        cuda_out, _ = _run_cuda_forward(data, intensity_floor=intensity_floor)

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
        cuda_out, _ = _run_cuda_forward(data, intensity_floor=intensity_floor)
        pytorch_out = _run_pytorch_reference(data, intensity_floor=intensity_floor)

        cuda_cpu = cuda_out.cpu()
        pytorch_cpu = pytorch_out.cpu()

        max_val = max(cuda_cpu.abs().max().item(), pytorch_cpu.abs().max().item())
        if max_val > 1e-6:
            rel_diff = (cuda_cpu - pytorch_cpu).abs() / (max_val + 1e-8)
            assert rel_diff.max().item() < Tolerances.COMPARISON_MAX_REL_DIFF


class TestBackwardKernelGradients:
    """Tests for backward kernel gradient correctness."""

    def test_3d_backward_produces_finite_gradients(self):
        """3D backward pass should produce finite, sensible gradients."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            cholesky_to_conic,
        )

        np.random.seed(55)
        N, d = 10, 3
        shape = (32, 32, 32)
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

        # Forward
        result = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
        )
        output = result[0]

        # Backward
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
