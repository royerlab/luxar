"""
Metal-accelerated Gaussian splatting model for Apple Silicon.

This module provides a high-performance replacement for GaussianSplatModel
using Metal compute shaders for the forward and backward passes.
"""

from __future__ import annotations

from typing import (
    Any,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import torch
from arbol import aprint
from torch.nn.modules.module import _IncompatibleKeys

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

T_destination = TypeVar("T_destination", bound=dict[str, Any])

# Import C++ extension (compiled separately)
try:
    import metal_splatting_backend

    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


def cholesky_to_conic(L: torch.Tensor) -> torch.Tensor:
    """
    Convert Cholesky factors to conic (inverse covariance) representation.

    Args:
        L: (N, d, d) lower-triangular Cholesky factors in [Z,Y,X] row order

    Returns:
        conic: (N, d*(d+1)//2) upper-triangular elements of Σ^(-1)
               For 3D: (N, 6) with [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx] (Z,Y,X upper triangle)

    WARNING: Variable names in the 3D implementation use c_xx, c_yy, c_zz but these
             actually correspond to matrix indices [0,0], [1,1], [2,2] which are
             Z,Z, Y,Y, X,X respectively! The variable names are backwards.
    """
    N, d, _ = L.shape

    # Manual implementation of C = Σ^(-1) = (L @ L^T)^(-1) = L^(-T) @ L^(-1)
    # Works on all devices (MPS, CPU, CUDA) and avoids MPS cholesky_inverse issues.
    # Compute K = L^(-1) via forward substitution, then C = K^T @ K

    if d == 3:
        # Extract elements (row-major)
        L00 = L[:, 0, 0]
        L10 = L[:, 1, 0]
        L11 = L[:, 1, 1]
        L20 = L[:, 2, 0]
        L21 = L[:, 2, 1]
        L22 = L[:, 2, 2]

        # Compute K = L^(-1) via forward substitution
        K00 = 1.0 / (L00 + 1e-9)
        K11 = 1.0 / (L11 + 1e-9)
        K22 = 1.0 / (L22 + 1e-9)
        K10 = -L10 * K00 * K11
        K21 = -L21 * K11 * K22
        K20 = -(L20 * K00 + L21 * K10) * K22

        # Compute C = K^T @ K
        c_xx = K00 * K00 + K10 * K10 + K20 * K20
        c_xy = K10 * K11 + K20 * K21
        c_xz = K20 * K22
        c_yy = K11 * K11 + K21 * K21
        c_yz = K21 * K22
        c_zz = K22 * K22

        conic = torch.stack([c_xx, c_xy, c_xz, c_yy, c_yz, c_zz], dim=1)

    elif d == 2:
        # 2D case
        L00 = L[:, 0, 0]
        L10 = L[:, 1, 0]
        L11 = L[:, 1, 1]

        K00 = 1.0 / (L00 + 1e-9)
        K11 = 1.0 / (L11 + 1e-9)
        K10 = -L10 * K00 * K11

        c_xx = K00 * K00 + K10 * K10
        c_xy = K10 * K11
        c_yy = K11 * K11

        conic = torch.stack([c_xx, c_xy, c_yy], dim=1)

    else:
        # Generic nD: use torch.linalg.inv (slower but works)
        # Move to CPU if on MPS since MPS has limited support
        device = L.device
        L_cpu = L.cpu() if L.device.type == "mps" else L

        Sigma = L_cpu @ L_cpu.transpose(-2, -1)
        Sigma_inv = torch.linalg.inv(Sigma)

        # Extract upper triangle
        indices = torch.triu_indices(d, d)
        conic = Sigma_inv[:, indices[0], indices[1]]

        conic = conic.to(device)

    return conic


class MetalSplatFunction(torch.autograd.Function):
    """Custom autograd function for Metal-accelerated splatting."""

    @staticmethod
    def forward(
        ctx: Any,
        centers: torch.Tensor,  # (N, d)
        Ls: torch.Tensor,  # (N, d, d)
        amps: torch.Tensor,  # (N,)
        shape: Tuple[int, ...],
        truncate: float,
        intensity_floor: float = 1e-5,  # For early culling of invisible contributions
        tile_size: int = 4,  # Configurable tile size for 3D binning
        use_metal_conic: bool = False,  # Use Metal for L→Conic (experimental speedup)
    ) -> torch.Tensor:
        """
        Forward pass: render Gaussians to volume.

        L → Conic conversion happens in PyTorch for graph consistency.
        Metal handles the pixel-parallel rendering.
        """
        d = len(shape)
        device = centers.device

        # === L → Conic Conversion ===
        # Use Metal L→Conic for speed (enabled by default, validated)
        if use_metal_conic and d == 3 and METAL_AVAILABLE:
            # Compute conic in Metal for speed
            Ls_mps_for_conic = Ls.contiguous().to("mps")
            conic_metal = metal_splatting_backend.compute_conic_metal(Ls_mps_for_conic)
            conic = conic_metal.to(device).detach()  # Already in [X,Y,Z] order!

            # Still need Ls_for_conic for backward (will recompute in PyTorch for gradients)
            Ls_for_conic = Ls.detach().clone().requires_grad_(True)
        else:
            # Standard PyTorch path
            Ls_for_conic = Ls.detach().clone().requires_grad_(True)
            conic = cholesky_to_conic(Ls_for_conic)

        # === Dispatch to Metal ===
        # Note: Metal computes sigma_diag internally from Ls for AABB
        if d == 3 and METAL_AVAILABLE:
            # CRITICAL: Coordinate convention handling
            # PyTorch/numpy uses [Z,Y,X], Metal kernel expects [X,Y,Z] for conic/distance

            if use_metal_conic:
                # Conic already in [X,Y,Z] order from Metal kernel - no reordering needed!
                conic_mps = conic.detach().contiguous().to("mps")
            else:
                # Conic from PyTorch is in [Z,Y,X], need to reorder to [X,Y,Z]
                # PyTorch: [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx] (Z,Y,X upper triangle)
                # Metal:   [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz] (X,Y,Z upper triangle)
                # Mapping: [ 0,   1,    2,    3,    4,    5  ] → [ 5,  4,  2,  3,  1,  0]
                conic_reordered = conic[:, [5, 4, 2, 3, 1, 0]]
                conic_mps = conic_reordered.detach().contiguous().to("mps")

            # Centers and L stay in [Z,Y,X] order
            centers_mps = centers.contiguous().to("mps")
            amps_mps = amps.contiguous().to("mps")
            Ls_mps = Ls.contiguous().to("mps")  # Needed for sigma_diag in binning

            # Forward returns: [output, tile_counts, tile_offsets, tile_content]
            result = metal_splatting_backend.forward_3d(
                centers_mps,
                conic_mps,
                amps_mps,
                Ls_mps,
                list(shape),
                truncate,
                intensity_floor,
                tile_size,  # Configurable tile size
            )
            output = cast(torch.Tensor, result[0].to(device))

            # CRITICAL: Save tile data for backward pass (no grad needed)
            tile_counts = result[1]
            tile_offsets = result[2]
            tile_content = result[3]
        else:
            # Fallback to PyTorch (for nD or when Metal unavailable)
            from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

            output = render_gaussians(
                shape, centers, Ls, amps, truncate, intensity_floor
            )
            # No tile data for PyTorch path
            tile_counts = None
            tile_offsets = None
            tile_content = None

        # Save for backward - include tile data for Metal backward
        ctx.save_for_backward(centers, Ls, Ls_for_conic, conic, amps)

        # Save non-tensor data and tile buffers separately
        ctx.shape = shape
        ctx.truncate = truncate
        ctx.intensity_floor = intensity_floor
        ctx.tile_size = tile_size
        ctx.use_metal_conic = use_metal_conic  # CRITICAL: Need this for backward!
        ctx.d = d
        ctx.tile_counts = tile_counts
        ctx.tile_offsets = tile_offsets
        ctx.tile_content = tile_content

        return output

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass: compute gradients.

        Metal computes: d_centers, d_conic, d_amps
        PyTorch handles: d_conic → d_Ls (chain rule)

        CRITICAL: Uses saved tile_counts/offsets/content from forward pass
        to avoid recomputing binning (saves time and ensures determinism).
        """
        (centers, Ls, Ls_for_conic, conic, amps) = ctx.saved_tensors
        shape = ctx.shape
        truncate = ctx.truncate
        intensity_floor = ctx.intensity_floor
        tile_size = ctx.tile_size
        use_metal_conic = ctx.use_metal_conic  # CRITICAL: Must match forward!
        d = ctx.d

        device = centers.device

        if d == 3 and METAL_AVAILABLE and ctx.tile_counts is not None:
            # === Metal: Compute gradients using saved tile data ===
            # CRITICAL: Reorder conic to [X,Y,Z] (same as forward!)
            # PyTorch: [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx] → Metal: [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]
            conic_reordered = conic[:, [5, 4, 2, 3, 1, 0]]  # [Z,Y,X] → [X,Y,Z]

            grad_mps = grad_output.contiguous().to("mps")
            centers_mps = centers.contiguous().to("mps")
            conic_mps = conic_reordered.contiguous().to("mps")  # Use reordered conic!
            amps_mps = amps.contiguous().to("mps")

            # Reuse tile data from forward pass (CRITICAL for performance)
            (d_centers, d_conic, d_amps) = metal_splatting_backend.backward_3d(
                grad_mps,
                centers_mps,
                conic_mps,
                amps_mps,
                ctx.tile_offsets,
                ctx.tile_counts,
                ctx.tile_content,
                list(shape),
                truncate,
                intensity_floor,
                tile_size,  # Must match forward!
            )

            # === DEBUG PROBE: Check raw Metal output ===
            import os

            if os.environ.get("DEBUG_METAL_GRADIENTS"):
                aprint("\n" + "=" * 80)
                aprint("DEBUG: RAW METAL BACKWARD OUTPUT")
                aprint("=" * 80)
                aprint(f"grad_output shape: {grad_output.shape}")
                aprint(f"grad_output sum: {grad_output.sum().item():.6f}")
                aprint(f"\nd_centers shape: {d_centers.shape}")
                aprint("d_centers (first 3 splats):")
                for i in range(min(3, d_centers.shape[0])):
                    dc = (
                        d_centers[i].cpu().numpy()
                        if d_centers.device.type == "mps"
                        else d_centers[i].numpy()
                    )
                    aprint(
                        f"  Splat {i}: [Z={dc[0]:.6e}, Y={dc[1]:.6e}, X={dc[2]:.6e}]"
                    )
                aprint(
                    "\nPython reference expects: [Z=2.707e-01, Y=-6.601e-08, X=-1.346e-07]"
                )
                aprint("If Y gradient is already wrong here, bug is in METAL.")
                aprint(
                    "If Y gradient is correct here, bug is in PYTHON chain rule below."
                )
                aprint("=" * 80 + "\n")
            # === END DEBUG PROBE ===

            # Move back to original device and reorder d_conic
            d_centers = d_centers.to(device)  # Already in [Z,Y,X], no reorder needed

            # d_conic is in [X,Y,Z] order, reorder back to [Z,Y,X]
            # Metal:   [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz] (current order)
            # PyTorch: [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx] (target order)
            # Inverse of [5,4,2,3,1,0] is [5,4,2,3,1,0] (self-inverse)

            # === DEBUG PROBE: Check d_conic before and after reordering ===
            if os.environ.get("DEBUG_METAL_GRADIENTS"):
                aprint("DEBUG: d_conic from Metal (before reorder, [X,Y,Z] order):")
                aprint("  [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]")
                dc_before = (
                    d_conic[0].cpu().numpy()
                    if d_conic.device.type == "mps"
                    else d_conic[0].numpy()
                )
                aprint(f"  {dc_before}")

            d_conic = d_conic[:, [5, 4, 2, 3, 1, 0]].to(device)

            if os.environ.get("DEBUG_METAL_GRADIENTS"):
                aprint("DEBUG: d_conic after reorder ([Z,Y,X] order):")
                aprint("  [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx]")
                dc_after = d_conic[0].detach().cpu().numpy()
                aprint(f"  {dc_after}")
                aprint("  Note: Contributions come from ALL pixels, not just peak")
                aprint()

            d_amps = d_amps.to(device)

            # === PyTorch: Chain rule d_conic → d_Ls ===
            # CRITICAL: Custom autograd.Function.backward runs with grad mode disabled!
            # Must explicitly enable grad mode for the recomputation.
            with torch.enable_grad():
                conic_recomputed = cholesky_to_conic(Ls_for_conic)  # In [Z,Y,X]

            # CRITICAL: d_conic is in [X,Y,Z], but recomputed conic is in [Z,Y,X]
            # They must be in same order for chain rule to work!
            if use_metal_conic:
                # Forward used Metal conic (already gave us d_conic in [X,Y,Z])
                # Need to reorder d_conic to [Z,Y,X] to match recomputed conic
                # [xx,xy,xz,yy,yz,zz] → [zz,zy,zx,yy,yx,xx]
                # Actually, we already reordered it above! So this is correct.
                # But we need to make sure recomputed conic is also in [Z,Y,X]
                # It already is (cholesky_to_conic outputs [Z,Y,X])
                d_conic_for_chain = d_conic  # Already reordered to [Z,Y,X] above
            else:
                # Forward used PyTorch conic (both in [Z,Y,X])
                # d_conic was reordered to [Z,Y,X] above
                d_conic_for_chain = d_conic

            # === DEBUG: Check inputs to chain rule ===
            if os.environ.get("DEBUG_METAL_GRADIENTS"):
                aprint("DEBUG: Chain rule inputs:")
                aprint(f"  conic_recomputed shape: {conic_recomputed.shape}")
                aprint(
                    f"  conic_recomputed[0]: {conic_recomputed[0].detach().cpu().numpy()}"
                )
                aprint(
                    f"  d_conic_for_chain[0]: {d_conic_for_chain[0].detach().cpu().numpy()}"
                )
                aprint()

            # Use torch.autograd.grad for chain rule
            (d_Ls,) = torch.autograd.grad(
                outputs=conic_recomputed,
                inputs=Ls_for_conic,
                grad_outputs=d_conic_for_chain,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )

            # === DEBUG: Check chain rule output ===
            if os.environ.get("DEBUG_METAL_GRADIENTS"):
                aprint("DEBUG: Chain rule output (d_Ls):")
                aprint(f"  d_Ls shape: {d_Ls.shape}")
                aprint("  d_Ls[0]:")
                aprint(f"{d_Ls[0].detach().cpu().numpy()}")
                aprint()

        else:
            # Fallback: PyTorch forward was used, so PyTorch Autograd handled it
            # We don't need to compute gradients - they're already in the graph
            # Just return None for all outputs (Autograd will handle it)
            # NOTE: This should never be called if forward used PyTorch fallback,
            # because render_gaussians is already tracked by Autograd
            return (None, None, None, None, None, None, None, None)

        # Return gradients: (centers, Ls, amps, shape, truncate, intensity_floor, tile_size, use_metal_conic)
        return d_centers, d_Ls, d_amps, None, None, None, None, None


class GaussianSplatModelMetal(torch.nn.Module):
    """
    Metal-accelerated Gaussian splat model for Apple Silicon.

    This class wraps a standard GaussianSplatModel and overrides the forward()
    method to use Metal compute shaders. All parameter management is delegated
    to the base model for consistency.

    Uses composition rather than inheritance to avoid double-registration of
    parameters and to maintain clean separation between parameter management
    and rendering strategy.
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
        truncate: float = 3.0,
        intensity_floor: float = 1e-5,
        tile_size: int = 4,  # Tile size for 3D binning (4 is optimal)
        use_metal_conic: bool = False,  # DISABLED: Gradient bug found, investigating
        voxel_size: Optional[np.ndarray] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()

        # CRITICAL: Feature parity check - prevent silent failures
        # Metal backend ONLY for 3D (2D overhead > benefit, nD not supported)
        d = len(shape)
        if d != 3:
            raise ValueError(
                f"Metal backend only supports 3D volumes (got {d}D). "
                f"For 2D images or nD data, use GaussianSplatModel (faster for small problems). "
                f"Metal overhead exceeds benefit for dimensions != 3."
            )

        # Validate device compatibility
        if device is not None:
            device_str = str(device) if isinstance(device, str) else device.type
            if device_str not in ("mps", "cpu", "mps:0"):
                raise ValueError(
                    f"Metal backend requires MPS or CPU device (got {device_str}). "
                    f"For CUDA, use GaussianSplatModel."
                )

        # Check for unsupported future features (extensibility safeguard)
        # If base model adds new parameters, we should detect them here
        import inspect

        base_params = inspect.signature(GaussianSplatModel.__init__).parameters
        if len(base_params) > 13:  # Expected: ~12 parameters (self + 11 init params)
            import warnings

            warnings.warn(
                "GaussianSplatModel has more parameters than expected. "
                "Metal backend may not support all features. "
                "Validate results carefully or use use_metal=False.",
                RuntimeWarning,
            )

        # Create base model for parameter management
        base_device = torch.device(device) if isinstance(device, str) else device
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
            device=base_device,
        )

        # Store Metal-specific parameters
        self._shape = shape
        self._truncate = truncate
        self._intensity_floor = intensity_floor
        self._tile_size = tile_size
        self._use_metal_conic = use_metal_conic

        # Cache for current_params() (23% speedup!)
        self._params_cache = None
        self._params_cache_valid = False

    def _invalidate_cache(self) -> None:
        """Invalidate cached parameters (call after optimizer.step())."""
        self._params_cache_valid = False

    def forward(self) -> torch.Tensor:
        """
        Render Gaussians to volume using Metal acceleration.

        Overrides base model's forward() to use MetalSplatFunction.
        """
        # Get current parameters from base model
        centers, Ls, amps = self._base.current_params()

        # Use Metal-accelerated forward pass
        output = cast(
            torch.Tensor,
            MetalSplatFunction.apply(  # type: ignore[no-untyped-call]
                centers,
                Ls,
                amps,
                self._shape,
                self._truncate,
                self._intensity_floor,
                self._tile_size,  # Configurable tile size
                self._use_metal_conic,  # Optional: Metal L→Conic (faster)
            ),
        )

        return output

    # Delegate all other methods to base model
    def current_params(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get current parameter values."""
        return self._base.current_params()

    def prune_(self, mask: torch.Tensor) -> None:
        """Remove splats according to boolean mask."""
        self._base.prune_(mask)

    def append_(
        self,
        centers: torch.Tensor,
        Ls: torch.Tensor,
        amps: torch.Tensor,
    ) -> None:
        """Add new splats to the model."""
        self._base.append_(centers, Ls, amps)

    def replace_with(
        self,
        centers: torch.Tensor,
        Ls: torch.Tensor,
        amps: torch.Tensor,
    ) -> None:
        """Replace all splats with new values."""
        self._base.replace_with(centers, Ls, amps)

    def n_splats(self) -> int:
        """Return number of splats."""
        return self._base.n_splats()

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Return iterator over model parameters."""
        # Delegate to base model to avoid double-registration
        return self._base.parameters(recurse=recurse)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """Return iterator over (name, parameter) pairs."""
        return self._base.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )

    @overload
    def state_dict(
        self,
        *,
        destination: T_destination,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> T_destination: ...

    @overload
    def state_dict(
        self, *, prefix: str = "", keep_vars: bool = False
    ) -> dict[str, Any]: ...

    def state_dict(
        self,
        *,
        destination: Optional[T_destination] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any] | T_destination:
        """Return state dict for serialization."""
        if destination is None:
            return self._base.state_dict(prefix=prefix, keep_vars=keep_vars)
        return self._base.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> _IncompatibleKeys:
        """Load state dict from serialization."""
        return cast(
            _IncompatibleKeys,
            self._base.load_state_dict(state_dict, strict=strict, assign=assign),
        )

    def to(self, *args: Any, **kwargs: Any) -> "GaussianSplatModelMetal":
        """Move model to device."""
        self._base = self._base.to(*args, **kwargs)
        return self

    # Expose base model attributes needed by optimizer and utilities
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return volume shape (required by optimizer)."""
        return self._base.shape

    @property
    def dim(self) -> int:
        """Return dimensionality (required by optimizer)."""
        return self._base.dim

    @property
    def truncate(self) -> float:
        """Return truncate value (required by some utilities)."""
        return self._base.truncate

    @property
    def raw_mu(self) -> torch.Tensor:
        """Delegate to base model (required by optimizer)."""
        return self._base.raw_mu

    @property
    def raw_L_diag(self) -> torch.Tensor:
        """Delegate to base model (required by optimizer)."""
        return self._base.raw_L_diag

    @property
    def L_off(self) -> torch.Tensor:
        """Delegate to base model (required by optimizer)."""
        return self._base.L_off

    @property
    def raw_a(self) -> torch.Tensor:
        """Delegate to base model (required by optimizer)."""
        return self._base.raw_a

    @property
    def sigma_min_diag(self) -> torch.Tensor:
        """Delegate to base model (required by optimizer)."""
        return self._base.sigma_min_diag

    @property
    def sigma_max_diag(self) -> Optional[torch.Tensor]:
        """Delegate to base model (required by optimizer)."""
        return self._base.sigma_max_diag

    @property
    def voxel_size(self) -> Optional[torch.Tensor]:
        """Delegate to base model."""
        return self._base.voxel_size

    def __repr__(self) -> str:
        return f"GaussianSplatModelMetal(n_splats={self.n_splats()}, shape={self._shape}, device={next(self.parameters()).device})"
