"""
CUDA-accelerated Gaussian splatting model for NVIDIA GPUs.

This module provides a high-performance replacement for GaussianSplatModel
using custom CUDA compute kernels for the forward and backward passes.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.utils.device import resolve_torch_device
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

# Import CUDA extension when available
try:
    import cuda_splatting_backend  # type: ignore[import-not-found]

    CUDA_BACKEND_AVAILABLE = True
except ImportError:
    CUDA_BACKEND_AVAILABLE = False


def cholesky_to_conic(L: torch.Tensor) -> torch.Tensor:
    """
    Convert Cholesky factors to conic (inverse covariance) representation.

    This function computes Σ⁻¹ = (L @ L^T)⁻¹ = L⁻ᵀ @ L⁻¹ via forward substitution,
    avoiding explicit matrix inversion for numerical stability.

    Args:
        L: (N, d, d) lower-triangular Cholesky factors

    Returns:
        conic: (N, d*(d+1)//2) packed upper-triangular elements of Σ⁻¹
               For 3D: [c_00, c_01, c_02, c_11, c_12, c_22]

    Note:
        The output is in row-major upper triangle order, matching the CUDA kernel
        expectations.
    """
    N, d, _ = L.shape
    device = L.device

    if d == 2:
        # 2D explicit implementation
        L00 = L[:, 0, 0]
        L10 = L[:, 1, 0]
        L11 = L[:, 1, 1]

        # K = L⁻¹ via forward substitution
        K00 = 1.0 / (L00 + 1e-9)
        K11 = 1.0 / (L11 + 1e-9)
        K10 = -L10 * K00 * K11

        # C = K^T @ K
        c_00 = K00 * K00 + K10 * K10
        c_01 = K10 * K11
        c_11 = K11 * K11

        return torch.stack([c_00, c_01, c_11], dim=1)

    elif d == 3:
        # 3D explicit implementation (matching Metal backend)
        L00 = L[:, 0, 0]
        L10 = L[:, 1, 0]
        L11 = L[:, 1, 1]
        L20 = L[:, 2, 0]
        L21 = L[:, 2, 1]
        L22 = L[:, 2, 2]

        # K = L⁻¹ via forward substitution
        K00 = 1.0 / (L00 + 1e-9)
        K11 = 1.0 / (L11 + 1e-9)
        K22 = 1.0 / (L22 + 1e-9)
        K10 = -L10 * K00 * K11
        K21 = -L21 * K11 * K22
        K20 = -(L20 * K00 + L21 * K10) * K22

        # C = K^T @ K
        c_00 = K00 * K00 + K10 * K10 + K20 * K20
        c_01 = K10 * K11 + K20 * K21
        c_02 = K20 * K22
        c_11 = K11 * K11 + K21 * K21
        c_12 = K21 * K22
        c_22 = K22 * K22

        return torch.stack([c_00, c_01, c_02, c_11, c_12, c_22], dim=1)

    else:
        # Generic nD: use triangular solve for numerical stability
        # Compute L⁻¹ via forward substitution (exploits triangular structure)
        # Σ⁻¹ = L⁻ᵀ @ L⁻¹ = (L⁻¹)ᵀ @ L⁻¹
        eye = torch.eye(d, device=device, dtype=L.dtype).unsqueeze(0).expand(N, -1, -1)
        L_inv = torch.linalg.solve_triangular(L, eye, upper=False)
        Sigma_inv = L_inv.transpose(-2, -1) @ L_inv

        # Extract upper triangle in row-major order
        indices = torch.triu_indices(d, d, device=device)
        result: torch.Tensor = Sigma_inv[:, indices[0], indices[1]]
        return result


try:
    _cholesky_to_conic_compiled = torch.compile(cholesky_to_conic)
except Exception:
    import warnings

    warnings.warn(
        "torch.compile unavailable (missing Triton?). "
        "cholesky_to_conic will run ~5x slower in eager mode. "
        "Install Triton: pip install triton",
        RuntimeWarning,
        stacklevel=1,
    )
    _cholesky_to_conic_compiled = cholesky_to_conic


class CUDASplatFunction(torch.autograd.Function):
    """Custom autograd function for CUDA-accelerated splatting."""

    @staticmethod
    def forward(
        ctx: Any,
        centers: torch.Tensor,  # (N, d)
        Ls: torch.Tensor,  # (N, d, d)
        amps: torch.Tensor,  # (N,)
        shape: Tuple[int, ...],
        truncate: float,
        intensity_floor: float,
        use_fp16: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass: render Gaussians to volume.

        The L → Conic conversion happens in PyTorch for autograd graph consistency.
        CUDA handles the pixel-parallel rendering.

        Args:
            use_fp16: If True, use FP16 precision for CUDA kernels.
                      When False, automatically detects torch.autocast() context
                      and uses FP16 kernels if autocast is enabled (AMP support).
                      Output is always FP32 regardless.
        """
        d = len(shape)

        # Determine if we should use FP16 kernels:
        # 1. Explicit use_fp16=True (inference mode with FP16 params)
        # 2. Inside torch.autocast() context (AMP training mode)
        use_fp16_kernel = use_fp16 or torch.is_autocast_enabled()

        # Compute conic (Σ⁻¹) from Cholesky factors (preserves dtype)
        # The clone creates an independent autograd leaf for the L→conic chain.
        # torch.compile fuses the forward+backward into optimized kernels (~0.3ms).
        Ls_for_conic = Ls.detach().clone().requires_grad_(True)
        conic = _cholesky_to_conic_compiled(Ls_for_conic)

        # Convert to FP16 for kernel if needed (AMP mode converts FP32 params to FP16)
        if use_fp16_kernel and centers.dtype != torch.float16:
            centers_kernel = centers.half().contiguous()
            conic_kernel = conic.half().contiguous()
            amps_kernel = amps.half().contiguous()
        else:
            centers_kernel = centers.contiguous()
            conic_kernel = conic.contiguous()
            amps_kernel = amps.contiguous()

        if CUDA_BACKEND_AVAILABLE:
            # Dispatch to CUDA kernels (use FP16 kernel if autocast or explicit)
            result = cuda_splatting_backend.forward(
                centers_kernel,
                conic_kernel,
                amps_kernel,
                list(shape),
                truncate,
                intensity_floor,
                use_fp16_kernel,
            )
            output = result[0]
            shape_tensor_cached = result[1]
        else:
            # Fallback to PyTorch rendering
            from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

            output = render_gaussians(
                shape, centers, Ls, amps, truncate, intensity_floor
            )
            shape_tensor_cached = None

        # Save for backward
        ctx.save_for_backward(centers, Ls, Ls_for_conic, conic, amps)
        ctx.centers_kernel = centers_kernel
        ctx.conic_kernel = conic_kernel
        ctx.amps_kernel = amps_kernel
        ctx.shape = shape
        ctx.truncate = truncate
        ctx.intensity_floor = intensity_floor
        ctx.shape_tensor_cached = shape_tensor_cached
        ctx.d = d
        ctx.use_fp16 = use_fp16_kernel
        ctx.explicit_fp16 = use_fp16

        output_tensor: torch.Tensor = output
        return output_tensor

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        """
        Backward pass: compute gradients.

        CUDA computes: d_centers, d_conic, d_amps
        PyTorch handles: d_conic → d_Ls (chain rule)
        """
        # Block training with FP16 params (use_fp16=True) - causes numerical overflow
        if ctx.explicit_fp16:
            raise RuntimeError(
                "Cannot train with use_fp16=True - FP16 params overflow.\n"
                "\n"
                "For training, use PyTorch AMP:\n"
                "  model = GaussianSplatModelCUDA(..., use_fp16=False)  # FP32 params\n"
                "  scaler = torch.amp.GradScaler('cuda')\n"
                "  with torch.amp.autocast('cuda'):\n"
                "      output = model()  # Auto-uses FP16 kernels\n"
                "      loss = criterion(output, target)\n"
                "  scaler.scale(loss).backward()\n"
                "  scaler.step(optimizer)\n"
                "\n"
                "use_fp16=True is only for inference with pre-trained models."
            )

        centers, Ls, Ls_for_conic, conic, amps = ctx.saved_tensors
        shape = ctx.shape
        truncate = ctx.truncate
        intensity_floor = ctx.intensity_floor
        use_fp16 = ctx.use_fp16

        if CUDA_BACKEND_AVAILABLE:
            # Build optional kwargs for cached device tensors
            cached_kwargs = {}
            if ctx.shape_tensor_cached is not None:
                cached_kwargs["shape_tensor_cached"] = ctx.shape_tensor_cached

            d_centers, d_conic, d_amps = cuda_splatting_backend.backward(
                grad_output.contiguous(),
                ctx.centers_kernel,
                ctx.conic_kernel,
                ctx.amps_kernel,
                list(shape),
                truncate,
                intensity_floor,
                use_fp16,
                **cached_kwargs,
            )

            # Chain rule: d_conic → d_Ls via PyTorch autograd.
            # torch.compile fuses both forward and backward into optimized kernels,
            # giving ~0.3ms (vs ~1.6ms without compile). Autograd is correct by
            # construction — compile just fuses the kernel launches.
            with torch.enable_grad():
                conic_recomputed = _cholesky_to_conic_compiled(Ls_for_conic)
            (d_Ls,) = torch.autograd.grad(
                outputs=conic_recomputed,
                inputs=Ls_for_conic,
                grad_outputs=d_conic,
            )
        else:
            # Fallback: recompute forward with gradient tracking and use autograd
            # This is slower than CUDA backward but ensures correctness when CUDA
            # backend is not available.
            from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

            # Recompute with gradient tracking
            with torch.enable_grad():
                centers_grad = centers.detach().clone().requires_grad_(True)
                Ls_grad = Ls.detach().clone().requires_grad_(True)
                amps_grad = amps.detach().clone().requires_grad_(True)

                output = render_gaussians(
                    shape,
                    centers_grad,
                    Ls_grad,
                    amps_grad,
                    truncate,
                    intensity_floor,
                )

            # Compute gradients via PyTorch autograd
            grads = torch.autograd.grad(
                outputs=output,
                inputs=[centers_grad, Ls_grad, amps_grad],
                grad_outputs=grad_output,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            d_centers = grads[0] if grads[0] is not None else torch.zeros_like(centers)
            d_Ls = grads[1] if grads[1] is not None else torch.zeros_like(Ls)
            d_amps = grads[2] if grads[2] is not None else torch.zeros_like(amps)

        # Return grads for: centers, Ls, amps + non-diff params
        # Order: centers, Ls, amps, shape, truncate, intensity_floor, use_fp16
        return d_centers, d_Ls, d_amps, None, None, None, None


class GaussianSplatModelCUDA(torch.nn.Module):
    """
    CUDA-accelerated Gaussian splat model for NVIDIA GPUs.

    This class wraps a standard GaussianSplatModel and overrides the forward()
    method to use CUDA compute kernels. All parameter management is delegated
    to the base model for consistency.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Dimensions of the target volume to reconstruct. Supports 2D-8D.
    centers0 : np.ndarray, shape (N, d)
        Initial center positions in voxel coordinates.
    L0 : np.ndarray, shape (N, d, d)
        Initial lower-triangular Cholesky factors.
    amps0 : np.ndarray, shape (N,)
        Initial amplitude values.
    sigma_min_diag : Sequence[float]
        Minimum diagonal values for Cholesky factor.
    sigma_max_diag : Sequence[float], optional
        Maximum diagonal values for Cholesky factor.
    amp_max : float, optional
        Maximum amplitude value. Prevents amplitude explosion during optimization.
    truncate : float, default=DEFAULT_TRUNCATION_RADIUS
        Truncation radius in standard deviations.
    intensity_floor : float, default=1e-5
        Minimum intensity threshold for early culling.
    use_fp16 : bool, default=False
        If True, store parameters in FP16 for inference bandwidth optimization.
        For training, leave this False and use PyTorch AMP instead (see examples).

        The model automatically detects torch.autocast() context and uses FP16
        kernels when AMP is enabled, regardless of this flag.
    device : torch.device, optional
        Must be a CUDA device.

    Examples
    --------
    >>> # Standard FP32 training
    >>> model = GaussianSplatModelCUDA(
    ...     shape=(128, 128, 128),
    ...     centers0=centers,
    ...     L0=L,
    ...     amps0=amps,
    ...     sigma_min_diag=(0.5, 0.5, 0.5),
    ...     device='cuda',
    ... )
    >>> output = model()

    >>> # Mixed-precision training with AMP (RECOMMENDED for training)
    >>> model = GaussianSplatModelCUDA(..., use_fp16=False)  # FP32 params
    >>> scaler = torch.amp.GradScaler('cuda')
    >>> with torch.amp.autocast('cuda'):
    ...     output = model()  # Auto-uses FP16 kernels
    ...     loss = criterion(output, target)
    >>> scaler.scale(loss).backward()
    >>> scaler.step(optimizer)
    >>> scaler.update()

    >>> # FP16 inference mode (for pre-trained models)
    >>> model_fp16 = GaussianSplatModelCUDA(..., use_fp16=True)  # FP16 params
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        centers0: np.ndarray,
        L0: np.ndarray,
        amps0: np.ndarray,
        sigma_min_diag: Sequence[float],
        sigma_max_diag: Optional[Sequence[float]] = None,
        amp_max: Optional[float] = None,
        max_eccentricity: Optional[float] = None,
        truncate: float = DEFAULT_TRUNCATION_RADIUS,
        intensity_floor: float = 1e-5,
        use_fp16: bool = False,
        voxel_size: Optional[np.ndarray] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()

        # Dimension validation
        d = len(shape)
        if d < 2 or d > 8:
            raise ValueError(
                f"CUDA backend supports 2D-8D (got {d}D). "
                f"For higher dimensions, use GaussianSplatModel."
            )

        # Device validation: this backend is CUDA-only, so opt out of Metal in
        # auto-selection and surface a clear error for any other device.
        resolved_device = resolve_torch_device(device, use_metal=False)

        if resolved_device.type != "cuda":
            raise ValueError(
                f"CUDA backend requires CUDA device (got {resolved_device}). "
                f"For CPU or MPS, use GaussianSplatModel or GaussianSplatModelMetal."
            )

        # Create base model for parameter management
        self._base = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
            sigma_max_diag=sigma_max_diag,
            amp_max=amp_max,
            max_eccentricity=max_eccentricity,
            truncate=truncate,
            voxel_size=voxel_size,
            device=resolved_device,
        )

        self._device = resolved_device
        self._shape = shape
        self._truncate = truncate
        self._intensity_floor = intensity_floor
        self._use_fp16 = use_fp16

        # Convert base params to FP16 if enabled (no conversion overhead)
        if use_fp16:
            self._convert_base_to_fp16()

    def _convert_base_to_fp16(self) -> None:
        """Convert all base model parameters to FP16 for bandwidth optimization."""
        for param in self._base.parameters():
            param.data = param.data.half()

        # Also convert buffer tensors if any
        base = self._base
        if hasattr(base, "sigma_min_diag") and base.sigma_min_diag is not None:
            base.sigma_min_diag = base.sigma_min_diag.half()
        if hasattr(base, "sigma_max_diag") and base.sigma_max_diag is not None:
            base.sigma_max_diag = base.sigma_max_diag.half()

    def forward(self) -> torch.Tensor:
        """
        Render Gaussians to volume using CUDA acceleration.

        Returns
        -------
        torch.Tensor
            Rendered volume with shape matching initialization.
            Always FP32 regardless of use_fp16 setting.
        """
        centers, Ls, amps = self.current_params()

        # NOTE: The C++ backend supports an optional output_buffer parameter for
        # pre-allocated buffer reuse (see bindings.cpp and cuda_splatting.cu). We
        # intentionally do NOT use it here. Benchmarking (2026-04-08) showed that
        # PyTorch's autograd requires .clone() on the output when the buffer is
        # reused (autograd detects in-place modification of a custom Function's
        # output on the next forward). The clone overhead negates the allocation
        # savings — the CUDA caching allocator already efficiently reuses freed
        # memory blocks, making torch::zeros essentially free. Net result was a
        # 5-17% regression across all configs. The C++ output_buffer + output_to_zero
        # infrastructure remains for potential future use in inference-only mode
        # (no autograd) or if PyTorch adds a way to opt out of this check.

        # `Function.apply` is an untyped classmethod up to torch 2.13 (so
        # `no-untyped-call` fires) and an `Any` alias from 2.14 (so the ignore
        # goes unused). Both torches are in range, hence both codes.
        output: torch.Tensor = CUDASplatFunction.apply(  # type: ignore[no-untyped-call, unused-ignore]
            centers,
            Ls,
            amps,
            self._shape,
            self._truncate,
            self._intensity_floor,
            self._use_fp16,
        )

        # Ensure output has correct shape (CUDA kernel may return flattened tensor)
        if output.shape != self._shape:
            output = output.view(self._shape)

        return output

    # Delegate all other methods to base model
    def current_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get current parameter values.

        When use_fp16=True, ensures all outputs are FP16. The base model's
        current_params() may return FP32 due to type promotion with hardcoded
        constants, so we convert here.
        """
        centers, Ls, amps = self._base.current_params()
        if self._use_fp16:
            # Ensure all outputs are FP16 for kernel input
            # Note: .half() on already-FP16 tensors is fast (~5x faster than FP32->FP16)
            centers = centers.half()
            Ls = Ls.half()
            amps = amps.half()
        return centers, Ls, amps

    def prune_(self, mask: torch.Tensor) -> None:
        """Remove splats according to boolean mask."""
        self._base.prune_(mask)
        # Re-convert to FP16 if enabled (pruning may reset dtypes)
        if self._use_fp16:
            self._convert_base_to_fp16()

    def append_(
        self,
        centers: torch.Tensor,
        Ls: torch.Tensor,
        amps: torch.Tensor,
    ) -> None:
        """Add new splats to the model."""
        self._base.append_(centers, Ls, amps)
        # Re-convert to FP16 if enabled (new params from append are FP32)
        if self._use_fp16:
            self._convert_base_to_fp16()

    def replace_with(
        self,
        centers: torch.Tensor,
        Ls: torch.Tensor,
        amps: torch.Tensor,
    ) -> None:
        """Replace all splats with new values."""
        self._base.replace_with(centers, Ls, amps)
        # Re-convert to FP16 if enabled (new params from replace_with are FP32)
        if self._use_fp16:
            self._convert_base_to_fp16()

    def n_splats(self) -> int:
        """Return number of splats."""
        return self._base.n_splats()

    def parameters(self, recurse: bool = True) -> Any:
        """Return iterator over model parameters."""
        return self._base.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Any:  # type: ignore[override]
        """Return iterator over (name, parameter) pairs."""
        return self._base.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """Return state dict for serialization."""
        return self._base.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Any, *args: Any, **kwargs: Any) -> Any:
        """Load state dict from serialization."""
        result = self._base.load_state_dict(state_dict, *args, **kwargs)
        # Re-convert to FP16 if enabled (loaded params are FP32)
        if self._use_fp16:
            self._convert_base_to_fp16()
        return result

    def to(self, device: Any) -> GaussianSplatModelCUDA:  # type: ignore[override]
        """Move model to device."""
        self._base = self._base.to(device)
        return super().to(device)

    # Expose base model attributes needed by optimizer and utilities
    @property
    def shape(self) -> tuple[int, ...]:
        return self._base.shape

    @property
    def dim(self) -> int:
        return self._base.dim

    @property
    def truncate(self) -> float:
        return self._base.truncate

    @property
    def raw_mu(self) -> torch.nn.Parameter:
        return self._base.raw_mu

    @property
    def raw_L_diag(self) -> torch.nn.Parameter:
        return self._base.raw_L_diag

    @property
    def L_off(self) -> torch.nn.Parameter:
        return self._base.L_off

    @property
    def raw_a(self) -> torch.nn.Parameter:
        return self._base.raw_a

    @property
    def sigma_min_diag(self) -> torch.Tensor:
        return self._base.sigma_min_diag

    @property
    def sigma_max_diag(self) -> torch.Tensor | None:
        return self._base.sigma_max_diag

    @property
    def amp_max(self) -> float | None:
        return self._base.amp_max

    @property
    def voxel_size(self) -> torch.Tensor | None:
        return self._base.voxel_size

    @property
    def use_fp16(self) -> bool:
        """Whether FP16 precision is enabled for CUDA kernels."""
        return self._use_fp16

    def __repr__(self) -> str:
        fp16_str = ", fp16=True" if self._use_fp16 else ""
        return (
            f"GaussianSplatModelCUDA("
            f"n_splats={self.n_splats()}, "
            f"shape={self._shape}, "
            f"device={next(self.parameters()).device}"
            f"{fp16_str})"
        )
