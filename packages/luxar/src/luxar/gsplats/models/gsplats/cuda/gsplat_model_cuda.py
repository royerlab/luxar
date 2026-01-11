"""
CUDA-accelerated Gaussian splatting model for NVIDIA GPUs.

This module provides a high-performance replacement for GaussianSplatModel
using custom CUDA compute kernels for the forward and backward passes.

See SPECIFICATIONS.md for implementation details and algorithm descriptions.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

# Import CUDA extension when available
try:
    import cuda_splatting_backend

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
        expectations. For coordinate convention details, see SPECIFICATIONS.md.
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
        # Generic nD: use PyTorch linalg
        # Note: This is slower but works for any dimension
        Sigma = L @ L.transpose(-2, -1)
        Sigma_inv = torch.linalg.inv(Sigma)

        # Extract upper triangle in row-major order
        indices = torch.triu_indices(d, d, device=device)
        return Sigma_inv[:, indices[0], indices[1]]


class CUDASplatFunction(torch.autograd.Function):
    """Custom autograd function for CUDA-accelerated splatting."""

    @staticmethod
    def forward(
        ctx,
        centers: torch.Tensor,  # (N, d)
        Ls: torch.Tensor,  # (N, d, d)
        amps: torch.Tensor,  # (N,)
        sharpness: torch.Tensor,  # (N,)
        shape: Tuple[int, ...],
        truncate: float,
        intensity_floor: float,
        tile_size: int,
    ) -> torch.Tensor:
        """
        Forward pass: render Gaussians to volume.

        The L → Conic conversion happens in PyTorch for autograd graph consistency.
        CUDA handles the pixel-parallel rendering.
        """
        d = len(shape)
        device = centers.device

        # Compute conic (Σ⁻¹) from Cholesky factors
        Ls_for_conic = Ls.detach().clone().requires_grad_(True)
        conic = cholesky_to_conic(Ls_for_conic)

        if CUDA_BACKEND_AVAILABLE:
            # Dispatch to CUDA kernels
            result = cuda_splatting_backend.forward(
                centers.contiguous(),
                conic.contiguous(),
                amps.contiguous(),
                sharpness.contiguous(),
                list(shape),
                truncate,
                intensity_floor,
                tile_size,
            )
            output = result[0]

            # Save tile data for backward pass
            tile_counts = result[1] if len(result) > 1 else None
            tile_offsets = result[2] if len(result) > 2 else None
            tile_content = result[3] if len(result) > 3 else None
            global_splat_ids = result[4] if len(result) > 4 else None
        else:
            # Fallback to PyTorch rendering
            from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

            output = render_gaussians(
                shape, centers, Ls, amps, sharpness, truncate, intensity_floor
            )
            tile_counts = None
            tile_offsets = None
            tile_content = None
            global_splat_ids = None

        # Save for backward
        ctx.save_for_backward(centers, Ls, Ls_for_conic, conic, amps, sharpness)
        ctx.shape = shape
        ctx.truncate = truncate
        ctx.intensity_floor = intensity_floor
        ctx.tile_size = tile_size
        ctx.tile_counts = tile_counts
        ctx.tile_offsets = tile_offsets
        ctx.tile_content = tile_content
        ctx.global_splat_ids = global_splat_ids
        ctx.d = d

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass: compute gradients.

        CUDA computes: d_centers, d_conic, d_amps, d_sharpness
        PyTorch handles: d_conic → d_Ls (chain rule)
        """
        centers, Ls, Ls_for_conic, conic, amps, sharpness = ctx.saved_tensors
        shape = ctx.shape
        truncate = ctx.truncate
        intensity_floor = ctx.intensity_floor
        tile_size = ctx.tile_size
        d = ctx.d

        device = centers.device

        if CUDA_BACKEND_AVAILABLE and ctx.tile_counts is not None:
            # Use CUDA backward kernels
            d_centers, d_conic, d_amps, d_sharpness = cuda_splatting_backend.backward(
                grad_output.contiguous(),
                centers,
                conic,
                amps,
                sharpness,
                ctx.tile_offsets,
                ctx.tile_counts,
                ctx.tile_content,
                ctx.global_splat_ids,
                list(shape),
                truncate,
                intensity_floor,
                tile_size,
            )

            # Chain rule: d_conic → d_Ls via PyTorch autograd
            with torch.enable_grad():
                conic_recomputed = cholesky_to_conic(Ls_for_conic)

            (d_Ls,) = torch.autograd.grad(
                outputs=conic_recomputed,
                inputs=Ls_for_conic,
                grad_outputs=d_conic,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
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
                sharpness_grad = sharpness.detach().clone().requires_grad_(True)

                output = render_gaussians(
                    shape,
                    centers_grad,
                    Ls_grad,
                    amps_grad,
                    sharpness_grad,
                    truncate,
                    intensity_floor,
                )

            # Compute gradients via PyTorch autograd
            grads = torch.autograd.grad(
                outputs=output,
                inputs=[centers_grad, Ls_grad, amps_grad, sharpness_grad],
                grad_outputs=grad_output,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            d_centers = grads[0] if grads[0] is not None else torch.zeros_like(centers)
            d_Ls = grads[1] if grads[1] is not None else torch.zeros_like(Ls)
            d_amps = grads[2] if grads[2] is not None else torch.zeros_like(amps)
            d_sharpness = (
                grads[3] if grads[3] is not None else torch.zeros_like(sharpness)
            )

        # Return gradients for: centers, Ls, amps, sharpness, shape, truncate, intensity_floor, tile_size
        return d_centers, d_Ls, d_amps, d_sharpness, None, None, None, None


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
    truncate : float, default=3.0
        Truncation radius in standard deviations.
    intensity_floor : float, default=1e-5
        Minimum intensity threshold for early culling.
    tile_size : int, optional
        Tile size for spatial binning. Auto-selected if None.
    device : torch.device, optional
        Must be a CUDA device.

    Examples
    --------
    >>> model = GaussianSplatModelCUDA(
    ...     shape=(128, 128, 128),
    ...     centers0=centers,
    ...     L0=L,
    ...     amps0=amps,
    ...     sigma_min_diag=(0.5, 0.5, 0.5),
    ...     device='cuda',
    ... )
    >>> output = model()  # Uses CUDA kernels
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
        truncate: float = 3.0,
        intensity_floor: float = 1e-5,
        tile_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # Dimension validation
        d = len(shape)
        if d < 2 or d > 8:
            raise ValueError(
                f"CUDA backend supports 2D-8D (got {d}D). "
                f"For higher dimensions, use GaussianSplatModel."
            )

        # Device validation
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device) if isinstance(device, str) else device

        if device.type != "cuda":
            raise ValueError(
                f"CUDA backend requires CUDA device (got {device}). "
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
            truncate=truncate,
            device=device,
        )

        # Auto-select tile size based on dimension
        if tile_size is None:
            tile_size = self._auto_tile_size(d)

        self._shape = shape
        self._truncate = truncate
        self._intensity_floor = intensity_floor
        self._tile_size = tile_size

    def _auto_tile_size(self, d: int) -> int:
        """Select optimal tile size based on dimension."""
        # Target ~256-512 voxels per tile for good occupancy
        # tile_size^d ≈ 256-512
        tile_sizes = {
            2: 16,  # 16² = 256
            3: 8,  # 8³ = 512
            4: 4,  # 4⁴ = 256
            5: 3,  # 3⁵ = 243
            6: 3,  # 3⁶ = 729
            7: 2,  # 2⁷ = 128
            8: 2,  # 2⁸ = 256
        }
        return tile_sizes.get(d, 4)

    def forward(self) -> torch.Tensor:
        """
        Render Gaussians to volume using CUDA acceleration.

        Returns
        -------
        torch.Tensor
            Rendered volume with shape matching initialization.
        """
        centers, Ls, amps, sharpness = self._base.current_params()

        output = CUDASplatFunction.apply(
            centers,
            Ls,
            amps,
            sharpness,
            self._shape,
            self._truncate,
            self._intensity_floor,
            self._tile_size,
        )

        # Ensure output has correct shape (CUDA kernel may return flattened tensor)
        if output.shape != self._shape:
            output = output.view(self._shape)

        return output

    # Delegate all other methods to base model
    def current_params(self):
        """Get current parameter values."""
        return self._base.current_params()

    def prune_(self, mask: torch.Tensor):
        """Remove splats according to boolean mask."""
        self._base.prune_(mask)

    def append_(
        self,
        centers: torch.Tensor,
        Ls: torch.Tensor,
        amps: torch.Tensor,
        sharpness: torch.Tensor,
    ):
        """Add new splats to the model."""
        self._base.append_(centers, Ls, amps, sharpness)

    def replace_with(
        self,
        centers: torch.Tensor,
        Ls: torch.Tensor,
        amps: torch.Tensor,
        sharpness: torch.Tensor,
    ):
        """Replace all splats with new values."""
        self._base.replace_with(centers, Ls, amps, sharpness)

    def n_splats(self) -> int:
        """Return number of splats."""
        return self._base.n_splats()

    def parameters(self, recurse: bool = True):
        """Return iterator over model parameters."""
        return self._base.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Return iterator over (name, parameter) pairs."""
        return self._base.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        """Return state dict for serialization."""
        return self._base.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load state dict from serialization."""
        return self._base.load_state_dict(state_dict, *args, **kwargs)

    def to(self, device):
        """Move model to device."""
        self._base = self._base.to(device)
        return super().to(device)

    # Expose base model attributes needed by optimizer and utilities
    @property
    def shape(self):
        return self._base.shape

    @property
    def dim(self):
        return self._base.dim

    @property
    def truncate(self):
        return self._base.truncate

    @property
    def raw_mu(self):
        return self._base.raw_mu

    @property
    def raw_L_diag(self):
        return self._base.raw_L_diag

    @property
    def L_off(self):
        return self._base.L_off

    @property
    def raw_a(self):
        return self._base.raw_a

    @property
    def sharpness_offsets_raw(self):
        return self._base.sharpness_offsets_raw

    @property
    def sigma_min_diag(self):
        return self._base.sigma_min_diag

    @property
    def sigma_max_diag(self):
        return self._base.sigma_max_diag

    @property
    def amp_max(self):
        return self._base.amp_max

    def __repr__(self):
        return (
            f"GaussianSplatModelCUDA("
            f"n_splats={self.n_splats()}, "
            f"shape={self._shape}, "
            f"device={next(self.parameters()).device})"
        )
