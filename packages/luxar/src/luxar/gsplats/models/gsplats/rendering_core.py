# rendering_core.py
"""
Core rendering engine for Gaussian splatting.

This module contains the main rendering implementation including:
- Memory-optimized chunk size calculation
- Grid caching for performance
- Specialized 2D/3D fast paths with explicit forward substitution
- Generic nD renderer with AABB truncation and sharpness support
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


def _calculate_optimal_chunk_size(
    K: int, d: int, device: torch.device, dtype: torch.dtype
) -> int:
    """
    Calculate optimal P chunk size based on available memory and tensor dimensions.

    This function determines the largest chunk size that can safely fit in memory,
    considering the actual memory footprint of the (K, d, P_chunk) tensors used
    in the renderer (delta, y, and intermediate results).

    Parameters
    ----------
    K : int
        Number of splats in the current group
    d : int
        Number of dimensions
    device : torch.device
        Target device for computation
    dtype : torch.dtype
        Data type for tensors

    Returns
    -------
    int
        Optimal chunk size for P dimension, bounded to reasonable range
    """
    # Get available memory with safety margin
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            free_mem, _ = torch.cuda.mem_get_info(device)
            available_mem = int(free_mem * 0.6)  # 60% safety margin for CUDA
        except Exception:
            # Fallback if memory info unavailable
            available_mem = 2 * (1024**3)  # 2GB conservative estimate
    else:
        # Conservative estimates for CPU/MPS
        if device.type == "mps":
            available_mem = 4 * (1024**3)  # 4GB for Apple Silicon unified memory
        else:
            available_mem = 8 * (1024**3)  # 8GB for CPU

    # Calculate memory footprint per P element
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    # Account for: delta (K,d,P), y (K,d,P), expo (K,P), vals (K,P), plus overhead
    memory_per_p = K * (2 * d + 2) * bytes_per_element * 1.5  # 1.5x overhead factor

    if memory_per_p <= 0:
        return 131072  # Fallback default

    # Calculate maximum P_chunk that fits in available memory
    max_p_chunk = max(1, int(available_mem / memory_per_p))

    # Clamp to reasonable range for performance
    min_chunk = 1024  # Minimum for kernel efficiency
    max_chunk = 1024**2  # Maximum to avoid very large kernel launches

    # Use default if calculation seems unreasonable
    if max_p_chunk < min_chunk:
        return 131072  # Fallback to current default

    return min(max_chunk, max_p_chunk)


# ===== Fast-path helpers for 2D/3D ==========================================

# Simple process-wide cache for base grids and linear offsets.
# Keyed by (device, dtype, strides_tuple, box_shape_tuple).
_GRID_CACHE: Dict[
    Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], Tuple[torch.Tensor, torch.Tensor]
] = {}


def clear_grid_cache() -> None:
    """Clear the cached base grids and linear offsets.

    Call this to free GPU/CPU memory when changing volume shapes
    or after completing a fitting session.
    """
    _GRID_CACHE.clear()


def _cached_base_and_offsets(
    box_shape: Sequence[int],
    strides: torch.Tensor,  # (d,), long
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      base: (d, P) float tensor with coordinates [0..h_i-1] mesh, flattened.
      lin_offsets: (P,) long tensor of row-major flat offsets for this box_shape.

    Note: Uses integer coordinates [0, 1, 2, ...] for backwards compatibility.
    The CUDA backend uses pixel-centered coordinates (i + 0.5) which is
    more physically accurate but produces small numerical differences.
    """
    key = (
        device.type,
        str(dtype),
        tuple(int(s) for s in strides.tolist()),
        tuple(int(s) for s in box_shape),
    )
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]

    ranges = [torch.arange(int(s), device=device, dtype=dtype) for s in box_shape]
    grids = torch.meshgrid(*ranges, indexing="ij")  # list of d arrays
    base = torch.stack([g.reshape(-1) for g in grids], dim=0)  # (d, P)

    # Compute row-major offsets once for this box shape.
    base_l = torch.stack([g.reshape(-1).to(torch.long) for g in grids], dim=0)  # (d, P)
    lin_offsets = (base_l.T * strides).sum(dim=1)  # (P,)

    _GRID_CACHE[key] = (base, lin_offsets)
    return _GRID_CACHE[key]


@torch.jit.ignore  # type: ignore[misc]  # jit-able but optional; ignore keeps it simple if torch.compile() is used outside
def _group_by_box_gpu(
    lo: torch.Tensor, hi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-friendly grouping by AABB size.
    Returns (uniq_sizes, inv), where uniq_sizes is (G, d) and inv is (N,).
    """
    sizes = (hi - lo).to(torch.int32)  # (N, d)

    # MPS doesn't support torch.unique with dim argument, fallback to CPU
    if sizes.device.type == "mps":
        sizes_cpu = sizes.cpu()
        uniq, inv = torch.unique(sizes_cpu, dim=0, return_inverse=True)
        uniq, inv = uniq.to(sizes.device), inv.to(sizes.device)
    else:
        uniq, inv = torch.unique(sizes, dim=0, return_inverse=True)

    return uniq, inv


# ---- Explicit forward-substitution for 2D/3D (no linalg kernels) -----------


def _fwd_norm2_2d(L: torch.Tensor, d0: torch.Tensor, d1: torch.Tensor) -> torch.Tensor:
    """
    Solve L y = [d0, d1]^T for each splat (batched) and return ||y||^2.
    L: (K, 2, 2), d0/d1: (K, P)
    Returns: (K, P)
    """
    l11 = L[:, 0, 0].unsqueeze(1)  # (K,1)
    l21 = L[:, 1, 0].unsqueeze(1)
    l22 = L[:, 1, 1].unsqueeze(1)

    y0 = d0 / torch.clamp(l11, min=1e-6)
    y1 = (d1 - l21 * y0) / torch.clamp(l22, min=1e-6)
    return y0.mul(y0).add_(y1.mul(y1))


def _fwd_norm2_3d(
    L: torch.Tensor, d0: torch.Tensor, d1: torch.Tensor, d2: torch.Tensor
) -> torch.Tensor:
    """
    Solve L y = [d0, d1, d2]^T for each splat (batched) and return ||y||^2.
    L: (K, 3, 3), d0/d1/d2: (K, P)
    Returns: (K, P)
    """
    l11 = L[:, 0, 0].unsqueeze(1)  # (K,1)
    l21 = L[:, 1, 0].unsqueeze(1)
    l22 = L[:, 1, 1].unsqueeze(1)
    l31 = L[:, 2, 0].unsqueeze(1)
    l32 = L[:, 2, 1].unsqueeze(1)
    l33 = L[:, 2, 2].unsqueeze(1)

    y0 = d0 / torch.clamp(l11, min=1e-6)
    y1 = (d1 - l21 * y0) / torch.clamp(l22, min=1e-6)
    y2 = (d2 - l31 * y0 - l32 * y1) / torch.clamp(l33, min=1e-6)
    return y0.mul(y0).add_(y1.mul(y1)).add_(y2.mul(y2))


# ---- Helper functions for nD rendering --------------------------------------


def _linear_strides(shape: Sequence[int], device: torch.device | str) -> torch.Tensor:
    """Row-major linear strides for an nD tensor with given shape."""
    d = len(shape)
    s = [1]
    for i in range(d - 1, 0, -1):
        s.insert(0, s[0] * shape[i])
    return torch.tensor(s, device=device, dtype=torch.long)  # (d,)


def _group_by_box(
    lo: torch.Tensor, hi: torch.Tensor
) -> Dict[Tuple[int, ...], torch.Tensor]:
    """
    Group splats by their AABB shape so we can reuse a single base grid per group.
    Returns: dict { box_shape_tuple : idx_tensor }  (CPU tuple keys, GPU indices)

    Optimized to minimize GPU-CPU synchronization by using torch.unique on GPU.
    """
    sizes = (hi - lo).to(torch.int32)  # (N, d)

    # Use torch.unique on GPU to find unique sizes and group indices
    # MPS doesn't support torch.unique with dim argument, fallback to CPU
    if sizes.device.type == "mps":
        sizes_cpu = sizes.cpu()
        uniq, inv = torch.unique(sizes_cpu, dim=0, return_inverse=True)
        uniq, inv = uniq.to(sizes.device), inv.to(sizes.device)
    else:
        uniq, inv = torch.unique(sizes, dim=0, return_inverse=True)

    groups: Dict[Tuple[int, ...], torch.Tensor] = {}
    # Only transfer the small unique array to CPU for dict keys
    uniq_cpu = uniq.cpu().tolist()

    for g, key in enumerate(map(tuple, uniq_cpu)):
        # Find indices for this group on GPU
        idx = torch.nonzero(inv == g, as_tuple=False).squeeze(1)
        groups[key] = idx

    return groups


# ---- Specialized renderers --------------------------------------------------


def _compute_aabb_with_intensity_floor(
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    sharpness: torch.Tensor,
    shape: Sequence[int],
    truncate: float,
    intensity_floor: float | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute AABB bounds with optional intensity-aware shrinking.

    This helper deduplicates the AABB computation logic used across 2D, 3D, and nD renderers.

    Parameters
    ----------
    centers : torch.Tensor, shape (N, d)
        Splat center positions.
    Ls : torch.Tensor, shape (N, d, d)
        Lower-triangular Cholesky factors.
    amps : torch.Tensor, shape (N,)
        Splat amplitudes.
    sharpness : torch.Tensor, shape (N,)
        Per-splat sharpness values.
    shape : Sequence[int]
        Output volume shape.
    truncate : float
        Truncation radius in standard deviations.
    intensity_floor : float | None
        Minimum intensity threshold for amplitude-aware culling.
    device : torch.device
        Device for tensor operations.

    Returns
    -------
    lo : torch.Tensor, shape (N, d)
        Lower bounds of AABBs.
    hi : torch.Tensor, shape (N, d)
        Upper bounds of AABBs.
    valid : torch.Tensor, shape (N,)
        Boolean mask indicating valid (non-empty) AABBs.
    """
    # Compute sigma diagonal: Σ_ii = row-wise sum(L^2)
    sigma_diag = torch.sum(Ls * Ls, dim=2)  # (N, d)

    # Sharpness-adjusted truncation for generalized Gaussian exp(-0.5 * r^s)
    effective_truncate = truncate ** (2.0 / sharpness)  # (N,)
    radii = torch.clamp(
        (effective_truncate[:, None] * torch.sqrt(torch.clamp(sigma_diag, 1e-8)))
        .ceil()
        .to(torch.long),
        min=1,
    )

    # Initial AABB bounds
    lo = torch.clamp((centers - radii).floor().to(torch.long), min=0)
    hi = torch.minimum(
        (centers + radii).ceil().to(torch.long) + 1,
        torch.tensor(shape, device=device, dtype=torch.long),
    )

    # Optional amplitude-aware shrinking
    if intensity_floor is not None and intensity_floor > 0:
        eps = torch.tensor(intensity_floor, device=device, dtype=torch.float32)
        a = torch.clamp(amps, min=1e-12)
        log_ratio = torch.clamp(2.0 * torch.log(torch.clamp(a / eps, min=1.0)), min=0.0)
        tmax = torch.pow(
            log_ratio, 1.0 / sharpness
        )  # (N,) - sharpness-adjusted threshold

        shrink = torch.clamp(
            (tmax[:, None] * torch.sqrt(torch.clamp(sigma_diag, 1e-8)))
            .ceil()
            .to(torch.long),
            min=1,
        )
        radii = torch.minimum(radii, shrink)
        lo = torch.clamp((centers - radii).floor().to(torch.long), min=0)
        hi = torch.minimum(
            (centers + radii).ceil().to(torch.long) + 1,
            torch.tensor(shape, device=device, dtype=torch.long),
        )

    # Validate non-empty boxes
    valid = (hi > lo).all(1)
    return lo, hi, valid


def _render_gaussians_2d(
    shape: Sequence[int],
    centers: torch.Tensor,  # (N,2)
    Ls: torch.Tensor,  # (N,2,2)
    amps: torch.Tensor,  # (N,)
    sharpness: torch.Tensor,  # (N,)
    truncate: float,
    intensity_floor: float,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Specialized 2D renderer with explicit forward substitution."""
    device = centers.device
    out = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    out_flat = out.view(-1)
    strides = _linear_strides(shape, device)  # (2,)

    # Compute AABB bounds using helper
    lo, hi, valid = _compute_aabb_with_intensity_floor(
        centers, Ls, amps, sharpness, shape, truncate, intensity_floor, device
    )

    # Filter invalid boxes
    if not torch.all(valid):
        centers, Ls, amps, sharpness = (
            centers[valid],
            Ls[valid],
            amps[valid],
            sharpness[valid],
        )
        lo, hi = lo[valid], hi[valid]
        if centers.numel() == 0:
            return out

    # Group on GPU
    uniq, inv = _group_by_box_gpu(lo, hi)

    for g in range(uniq.shape[0]):
        box_shape = uniq[g].tolist()  # [h0, h1]
        idx = torch.nonzero(inv == g, as_tuple=False).squeeze(1)

        mu = centers[idx]  # (K,2)
        L = Ls[idx]  # (K,2,2)
        a = amps[idx]  # (K,)
        s = sharpness[idx]  # (K,)
        lo_sel = lo[idx]  # (K,2)

        base, lin_offsets = _cached_base_and_offsets(
            box_shape, strides, device, dtype=torch.float32
        )  # (2,P), (P,)
        base_idx = (lo_sel.to(torch.long) * strides).sum(dim=1)  # (K,)

        P = base.shape[1]
        # Calculate optimal chunk size for memory management (2D)
        P_chunk = chunk_size or _calculate_optimal_chunk_size(
            K=len(idx), d=2, device=device, dtype=torch.float32
        )
        for p0 in range(0, P, P_chunk):
            p1 = min(P, p0 + P_chunk)
            # Δ = base + lo - μ
            d0 = base[0, p0:p1][None, :] + lo_sel[:, 0:1] - mu[:, 0:1]  # (K,Pc)
            d1 = base[1, p0:p1][None, :] + lo_sel[:, 1:2] - mu[:, 1:2]  # (K,Pc)

            # ||y||^2 via explicit forward-substitution
            expo = _fwd_norm2_2d(L, d0, d1)  # (K,Pc)

            # Apply sharpness: exp(-0.5 * ||y||^s) where s = sharpness
            # ||y||^s = (||y||^2)^(s/2) = expo^(s/2)
            # Clamp expo to avoid log(0) in gradients of pow(expo, s/2)
            expo_safe = torch.clamp(expo, min=1e-10)
            vals = torch.exp(-0.5 * torch.pow(expo_safe, s[:, None] / 2.0)) * a[:, None]

            idx_flat = (base_idx[:, None] + lin_offsets[p0:p1][None, :]).reshape(-1)
            out_flat.index_add_(0, idx_flat, vals.reshape(-1))

    return out


def _render_gaussians_3d(
    shape: Sequence[int],
    centers: torch.Tensor,  # (N,3)
    Ls: torch.Tensor,  # (N,3,3)
    amps: torch.Tensor,  # (N,)
    sharpness: torch.Tensor,  # (N,)
    truncate: float,
    intensity_floor: float,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Specialized 3D renderer with explicit forward substitution."""
    device = centers.device
    out = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    out_flat = out.view(-1)
    strides = _linear_strides(shape, device)  # (3,)

    # Compute AABB bounds using helper
    lo, hi, valid = _compute_aabb_with_intensity_floor(
        centers, Ls, amps, sharpness, shape, truncate, intensity_floor, device
    )

    # Filter invalid boxes
    if not torch.all(valid):
        centers, Ls, amps, sharpness = (
            centers[valid],
            Ls[valid],
            amps[valid],
            sharpness[valid],
        )
        lo, hi = lo[valid], hi[valid]
        if centers.numel() == 0:
            return out

    # Group on GPU
    uniq, inv = _group_by_box_gpu(lo, hi)

    for g in range(uniq.shape[0]):
        box_shape = uniq[g].tolist()  # [h0, h1, h2]
        idx = torch.nonzero(inv == g, as_tuple=False).squeeze(1)

        mu = centers[idx]  # (K,3)
        L = Ls[idx]  # (K,3,3)
        a = amps[idx]  # (K,)
        s = sharpness[idx]  # (K,)
        lo_sel = lo[idx]  # (K,3)

        base, lin_offsets = _cached_base_and_offsets(
            box_shape, strides, device, dtype=torch.float32
        )  # (3,P),(P,)
        base_idx = (lo_sel.to(torch.long) * strides).sum(dim=1)  # (K,)

        P = base.shape[1]
        # Calculate optimal chunk size for memory management (3D)
        P_chunk = chunk_size or _calculate_optimal_chunk_size(
            K=len(idx), d=3, device=device, dtype=torch.float32
        )
        for p0 in range(0, P, P_chunk):
            p1 = min(P, p0 + P_chunk)
            # Δ = base + lo - μ
            d0 = base[0, p0:p1][None, :] + lo_sel[:, 0:1] - mu[:, 0:1]  # (K,Pc)
            d1 = base[1, p0:p1][None, :] + lo_sel[:, 1:2] - mu[:, 1:2]
            d2 = base[2, p0:p1][None, :] + lo_sel[:, 2:3] - mu[:, 2:3]

            expo = _fwd_norm2_3d(L, d0, d1, d2)  # (K,Pc)

            # Apply sharpness: exp(-0.5 * ||y||^s) where s = sharpness
            # ||y||^s = (||y||^2)^(s/2) = expo^(s/2)
            # Clamp expo to avoid log(0) in gradients of pow(expo, s/2)
            expo_safe = torch.clamp(expo, min=1e-10)
            vals = torch.exp(-0.5 * torch.pow(expo_safe, s[:, None] / 2.0)) * a[:, None]

            idx_flat = (base_idx[:, None] + lin_offsets[p0:p1][None, :]).reshape(-1)
            out_flat.index_add_(0, idx_flat, vals.reshape(-1))

    return out


# ===== Main Rendering Function ==============================================


def render_gaussians(
    shape: Sequence[int],
    centers: torch.Tensor,  # (N, d) voxel coords
    Ls: torch.Tensor,  # (N, d, d) lower-tri
    amps: torch.Tensor,  # (N,)
    sharpness: torch.Tensor,  # (N) sharpness values (s = 2 * exp(s'))
    truncate: float = 3.0,
    intensity_floor: float = 1e-5,  # for amplitude-aware culling
    chunk_size: Optional[int] = None,  # P-dimension chunk size for memory control
) -> torch.Tensor:
    """
    Fast vectorized renderer with 2D/3D fast-paths and per-splat sharpness.
    Falls back to the generic nD implementation for d != 2 and d != 3.

    Renders Gaussians with generalized falloff: exp(-0.5 * ||y||^s) where s is sharpness.
    s = 2 is standard Gaussian, s > 2 is sharper, s < 2 is softer.

    Parameters
    ----------
    shape : Sequence[int]
        Output shape of the rendered image/volume.
    centers : torch.Tensor, shape (N, d)
        Center positions in voxel coordinates.
    Ls : torch.Tensor, shape (N, d, d)
        Lower-triangular Cholesky factors.
    amps : torch.Tensor, shape (N,)
        Splat amplitudes.
    sharpness : torch.Tensor, shape (N,)
        Per-splat sharpness values.
    truncate : float, default=3.0
        Truncation radius in standard deviations.
    intensity_floor : float, default=1e-5
        Minimum intensity threshold for amplitude-aware culling.
    chunk_size : int, optional
        Chunk size for memory management.

    Returns
    -------
    torch.Tensor
        Rendered image/volume.
    """
    d = len(shape)
    if d == 2:
        return _render_gaussians_2d(
            shape, centers, Ls, amps, sharpness, truncate, intensity_floor, chunk_size
        )
    if d == 3:
        return _render_gaussians_3d(
            shape, centers, Ls, amps, sharpness, truncate, intensity_floor, chunk_size
        )

    # --- Generic nD implementation ---
    device = centers.device
    out = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    out_flat = out.view(-1)
    strides = _linear_strides(shape, device)  # (d,)

    # Compute AABB bounds using helper
    lo, hi, valid = _compute_aabb_with_intensity_floor(
        centers, Ls, amps, sharpness, shape, truncate, intensity_floor, device
    )

    # Filter invalid boxes
    if not torch.all(valid):
        centers = centers[valid]
        Ls = Ls[valid]
        amps = amps[valid]
        sharpness = sharpness[valid]
        lo = lo[valid]
        hi = hi[valid]

    if centers.numel() == 0:
        return out

    # --- Group by box size for reuse of base grid ---
    groups = _group_by_box(lo, hi)  # { (h1,..,hd) : idx }
    for box_shape, idx in groups.items():
        # Splat subset
        mu = centers[idx]  # (K, d)
        L = Ls[idx]  # (K, d, d)
        a = amps[idx]  # (K,)
        s = sharpness[idx]  # (K,)

        # Build base grid (one per group, on device)
        # coords_i = [0, 1, ..., h_i-1]  -> broadcast to P points
        ranges = [
            torch.arange(s, device=device, dtype=torch.float32) for s in box_shape
        ]
        grids = torch.meshgrid(*ranges, indexing="ij")
        P = int(np.prod(box_shape))  # More efficient than loop
        # per-axis base coords flattened (P,)
        base = torch.stack([g.reshape(-1) for g in grids], dim=0)  # (d, P)

        # Precompute base indices and linear offsets for the group
        lo_f = lo[idx].to(torch.float32)  # (K, d)
        strides_f = strides.to(torch.float32)
        lin_offsets = (base.T @ strides_f).to(torch.long)  # (P,)
        base_idx = (lo[idx].to(torch.long) * strides).sum(dim=1)  # (K,)

        # *** MEMORY OPTIMIZATION: Process P dimension in chunks to prevent OOM ***
        # Calculate optimal chunk size based on available memory and tensor dimensions
        P_chunk = chunk_size or _calculate_optimal_chunk_size(
            K=len(idx), d=d, device=device, dtype=torch.float32
        )
        for p0 in range(0, P, P_chunk):
            p1 = min(P, p0 + P_chunk)

            # Process chunk: base coordinates and indices for this chunk
            base_chunk = base[:, p0:p1]  # (d, Pc)
            lin_offsets_chunk = lin_offsets[p0:p1]  # (Pc,)

            # Per-splat Δ = base + lo - μ   (broadcast to (K, d, Pc))
            delta = (
                base_chunk[None, :, :] + lo_f[:, :, None] - mu[:, :, None]
            )  # (K, d, Pc)

            # Solve L y = Δ  (batched lower-tri solve with Pc RHS per splat)
            # PyTorch Version Compatibility: See models/utils/lt_solver.py for full details
            # - Modern: torch.linalg.solve_triangular (PyTorch >= 1.9)
            # - Legacy: torch.triangular_solve (PyTorch 1.12-1.13, removed in 2.0+)
            # - MPS Note: Both functions have identical 10× CPU overhead on Apple Silicon
            try:
                y = torch.linalg.solve_triangular(L, delta, upper=False)
            except AttributeError:
                # Legacy PyTorch 1.12-1.13 fallback (deprecated API, removed in 2.0+)
                # AttributeError: torch.linalg has no attribute 'solve_triangular'
                # Note: triangular_solve returns (solution, cloned_matrix) tuple
                y, _ = torch.triangular_solve(delta, L, upper=False)

            # Exponent and values: exp(-0.5 * ||y||^s) * a where s is sharpness
            # ||y||^s = (||y||^2)^(s/2) = expo^(s/2)
            expo = torch.sum(y * y, dim=1)  # (K, Pc)
            # Clamp expo to avoid log(0) in gradients of pow(expo, s/2)
            expo_safe = torch.clamp(expo, min=1e-10)
            vals = (
                torch.exp(-0.5 * torch.pow(expo_safe, s[:, None] / 2.0)) * a[:, None]
            )  # (K, Pc)

            # Absolute flat indices (K, Pc) -> (K*Pc,)
            idx_flat = (base_idx[:, None] + lin_offsets_chunk[None, :]).reshape(-1)
            vals_flat = vals.reshape(-1)

            # Accumulate into output
            out_flat.index_add_(0, idx_flat, vals_flat)

    return out
