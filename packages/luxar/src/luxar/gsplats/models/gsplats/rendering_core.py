# rendering_core.py
"""
Core rendering engine for Gaussian splatting.

This module contains the main rendering implementation including:
- Memory-optimized chunk size calculation
- Grid caching for performance
- Specialized 2D/3D fast paths with explicit forward substitution
- Generic nD renderer with AABB truncation
"""

from __future__ import annotations

import math
import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import numpy as np
import torch

from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS


def calculate_optimal_chunk_size(
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

# Process-wide cache for base grids and linear offsets.
#
# Cache sizing must adapt across laptops, workstations, and HPC nodes.  The
# default policy derives a conservative byte budget from the current device, and
# users can override it with environment variables for large repeated renders.
#
# Supported overrides:
#   LUXAR_GSPLAT_GRID_CACHE_MAX_BYTES=0          # disable cache
#   LUXAR_GSPLAT_GRID_CACHE_MAX_GB=8             # total budget per device
#   LUXAR_GSPLAT_GRID_CACHE_MAX_ENTRY_BYTES=...
#   LUXAR_GSPLAT_GRID_CACHE_MAX_ENTRY_GB=6       # max one cached support grid
_GRID_CACHE_CUDA_FRACTION = 0.10
_GRID_CACHE_CPU_FRACTION = 0.05
_GRID_CACHE_MAX_ENTRY_FRACTION = 0.75
_GRID_CACHE_DEFAULT_CUDA_MAX_BYTES = 2 * (1024**3)
_GRID_CACHE_DEFAULT_CPU_MAX_BYTES = 4 * (1024**3)
_GRID_CACHE_DEFAULT_MPS_BYTES = 512 * (1024**2)
_GRID_CACHE_FALLBACK_BYTES = 512 * (1024**2)

DeviceCacheKey = Tuple[str, Optional[int]]
GridCacheKey = Tuple[str, Tuple[int, ...], Tuple[int, ...]]


@dataclass
class _GridCacheEntry:
    base: torch.Tensor
    lin_offsets: torch.Tensor
    nbytes: int


_GRID_CACHE: Dict[DeviceCacheKey, OrderedDict[GridCacheKey, _GridCacheEntry]] = {}
_GRID_CACHE_BYTES: Dict[DeviceCacheKey, int] = {}


def clear_grid_cache() -> None:
    """Clear the cached base grids and linear offsets.

    Call this to free GPU/CPU memory when changing volume shapes
    or after completing a fitting session.
    """
    _GRID_CACHE.clear()
    _GRID_CACHE_BYTES.clear()


def _parse_nonnegative_int_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        warnings.warn(
            f"Ignoring invalid {name}={raw!r}; expected a non-negative integer.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if value < 0:
        warnings.warn(
            f"Ignoring invalid {name}={raw!r}; expected a non-negative integer.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return value


def _parse_nonnegative_gb_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        warnings.warn(
            f"Ignoring invalid {name}={raw!r}; expected a non-negative number.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if value < 0:
        warnings.warn(
            f"Ignoring invalid {name}={raw!r}; expected a non-negative number.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return int(value * (1024**3))


def _read_byte_override(bytes_var: str, gb_var: str) -> Optional[int]:
    bytes_value = _parse_nonnegative_int_env(bytes_var)
    if bytes_value is not None:
        return bytes_value
    return _parse_nonnegative_gb_env(gb_var)


def _device_cache_key(device: torch.device) -> DeviceCacheKey:
    index = cast(Optional[int], getattr(device, "index", None))
    if device.type == "cuda" and index is None and torch.cuda.is_available():
        index = torch.cuda.current_device()
    return (device.type, index)


def _available_cpu_memory_bytes() -> Optional[int]:
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def _default_grid_cache_budget_bytes(device: torch.device) -> int:
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            free_mem, _ = torch.cuda.mem_get_info(device)
            return min(
                int(free_mem * _GRID_CACHE_CUDA_FRACTION),
                _GRID_CACHE_DEFAULT_CUDA_MAX_BYTES,
            )
        except Exception:
            return _GRID_CACHE_FALLBACK_BYTES

    if device.type == "mps":
        return _GRID_CACHE_DEFAULT_MPS_BYTES

    available = _available_cpu_memory_bytes()
    if available is not None:
        return min(
            int(available * _GRID_CACHE_CPU_FRACTION),
            _GRID_CACHE_DEFAULT_CPU_MAX_BYTES,
        )

    return _GRID_CACHE_FALLBACK_BYTES


def _grid_cache_budget_bytes(device: torch.device) -> int:
    override = _read_byte_override(
        "LUXAR_GSPLAT_GRID_CACHE_MAX_BYTES", "LUXAR_GSPLAT_GRID_CACHE_MAX_GB"
    )
    if override is not None:
        return override
    return _default_grid_cache_budget_bytes(device)


def _grid_cache_max_entry_bytes(device: torch.device, budget_bytes: int) -> int:
    override = _read_byte_override(
        "LUXAR_GSPLAT_GRID_CACHE_MAX_ENTRY_BYTES",
        "LUXAR_GSPLAT_GRID_CACHE_MAX_ENTRY_GB",
    )
    if override is not None:
        return min(override, budget_bytes)
    return int(budget_bytes * _GRID_CACHE_MAX_ENTRY_FRACTION)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _grid_entry_nbytes(base: torch.Tensor, lin_offsets: torch.Tensor) -> int:
    return _tensor_nbytes(base) + _tensor_nbytes(lin_offsets)


def _evict_until_within_budget(
    device_key: DeviceCacheKey, incoming_bytes: int, budget_bytes: int
) -> None:
    cache = _GRID_CACHE.get(device_key)
    if not cache:
        return

    current_bytes = _GRID_CACHE_BYTES.get(device_key, 0)
    while current_bytes + incoming_bytes > budget_bytes and cache:
        _, entry = cache.popitem(last=False)
        current_bytes = max(0, current_bytes - entry.nbytes)

    if cache:
        _GRID_CACHE_BYTES[device_key] = current_bytes
    else:
        _GRID_CACHE.pop(device_key, None)
        _GRID_CACHE_BYTES.pop(device_key, None)


def get_grid_cache_stats() -> Dict[str, Any]:
    """Return cache occupancy and configured budgets for debugging/tests."""
    devices: Dict[str, Dict[str, int]] = {}
    for device_key, cache in _GRID_CACHE.items():
        device = torch.device(
            device_key[0]
            if device_key[1] is None
            else f"{device_key[0]}:{device_key[1]}"
        )
        budget = _grid_cache_budget_bytes(device)
        devices[_format_device_cache_key(device_key)] = {
            "entries": len(cache),
            "bytes": _GRID_CACHE_BYTES.get(device_key, 0),
            "budget_bytes": budget,
            "max_entry_bytes": _grid_cache_max_entry_bytes(device, budget),
        }
    return {"devices": devices}


def _format_device_cache_key(device_key: DeviceCacheKey) -> str:
    device_type, index = device_key
    return device_type if index is None else f"{device_type}:{index}"


def cached_base_and_offsets(
    box_shape: Sequence[int],
    strides: torch.Tensor,  # (d,), long
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      base: (d, P) float tensor with integer voxel coordinates [0..h_i-1], flattened.
      lin_offsets: (P,) long tensor of row-major flat offsets for this box_shape.

    Note: Luxar currently samples Gaussian splats at integer voxel coordinates.
    Keep this convention in sync with the CUDA backend in kernels_core.cuh.
    """
    device_key = _device_cache_key(device)
    key: GridCacheKey = (
        str(dtype),
        tuple(int(s) for s in strides.tolist()),
        tuple(int(s) for s in box_shape),
    )
    device_cache = _GRID_CACHE.get(device_key)
    if device_cache is not None and key in device_cache:
        entry = device_cache.pop(key)
        device_cache[key] = entry
        return entry.base, entry.lin_offsets

    ranges = [torch.arange(int(s), device=device, dtype=dtype) for s in box_shape]
    grids = torch.meshgrid(*ranges, indexing="ij")  # list of d arrays
    base = torch.stack([g.reshape(-1) for g in grids], dim=0)  # (d, P)

    # Compute row-major offsets without materializing a full (d, P) int64 tensor.
    # For large boxes (e.g. 767³), base_l would be 10.8 GB — instead we accumulate
    # per-dimension contributions into a single (P,) int64 output.
    lin_offsets = torch.zeros(base.shape[1], dtype=torch.long, device=device)
    for g, s in zip(grids, strides):
        lin_offsets.add_(g.reshape(-1).to(torch.long) * s.item())
    del grids, ranges  # Free meshgrid tensors immediately

    budget_bytes = _grid_cache_budget_bytes(device)
    entry_bytes = _grid_entry_nbytes(base, lin_offsets)
    max_entry_bytes = _grid_cache_max_entry_bytes(device, budget_bytes)

    # A budget of 0 disables caching. Oversized entries are still returned for
    # the current render, but are not kept as long-lived CPU/GPU tensors.
    if budget_bytes <= 0 or entry_bytes > budget_bytes or entry_bytes > max_entry_bytes:
        return base, lin_offsets

    _evict_until_within_budget(device_key, entry_bytes, budget_bytes)

    device_cache = _GRID_CACHE.setdefault(device_key, OrderedDict())
    device_cache[key] = _GridCacheEntry(base, lin_offsets, entry_bytes)
    _GRID_CACHE_BYTES[device_key] = _GRID_CACHE_BYTES.get(device_key, 0) + entry_bytes
    return base, lin_offsets


@torch.jit.ignore  # type: ignore  # jit-able but optional; ignore keeps it simple if torch.compile() is used outside
def group_by_box_gpu(
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


def fwd_norm2_2d(L: torch.Tensor, d0: torch.Tensor, d1: torch.Tensor) -> torch.Tensor:
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


def fwd_norm2_3d(
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


def linear_strides(shape: Sequence[int], device: torch.device | str) -> torch.Tensor:
    """Row-major linear strides for an nD tensor with given shape."""
    d = len(shape)
    s = [1]
    for i in range(d - 1, 0, -1):
        s.insert(0, s[0] * shape[i])
    return torch.tensor(s, device=device, dtype=torch.long)  # (d,)


def group_by_box(
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


def compute_aabb_with_intensity_floor(
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
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
    # Compute sigma diagonal: Sigma_ii = row-wise sum(L^2)
    sigma_diag = torch.sum(Ls * Ls, dim=2)  # (N, d)

    # Standard Gaussian truncation: truncate is in units of standard deviations
    radii = torch.clamp(
        (truncate * torch.sqrt(torch.clamp(sigma_diag, 1e-8))).ceil().to(torch.long),
        min=1,
    )

    # Initial AABB bounds
    lo = torch.clamp((centers - radii).floor().to(torch.long), min=0)
    hi = torch.minimum(
        (centers + radii).ceil().to(torch.long) + 1,
        torch.tensor(shape, device=device, dtype=torch.long),
    )

    # Optional amplitude-aware shrinking (shifted Gaussian)
    if intensity_floor is not None and intensity_floor > 0:
        eps = torch.tensor(intensity_floor, device=device, dtype=torch.float32)
        a = torch.clamp(amps, min=1e-12)

        # Shifted Gaussian: a·scale·(exp(-0.5·D²) - C) = floor
        # → exp(-0.5·D²) = floor/(a·scale) + C
        # → D = sqrt(-2·ln(floor/(a·scale) + C))
        shift_C = math.exp(-0.5 * truncate * truncate)
        inv_one_minus_C = 1.0 / (1.0 - shift_C)
        threshold = eps / (a * inv_one_minus_C) + shift_C
        # Guard: where threshold >= 1, splat is invisible
        threshold = torch.clamp(threshold, max=1.0 - 1e-7)
        tmax = torch.sqrt(torch.clamp(-2.0 * torch.log(threshold), min=0.0))  # (N,)

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
    truncate: float,
    intensity_floor: Optional[float],
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Specialized 2D renderer with explicit forward substitution."""
    device = centers.device
    # Shifted Gaussian constants for C⁰ continuous truncation
    shift_C = math.exp(-0.5 * truncate * truncate)
    scale = 1.0 / (1.0 - shift_C)
    out = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    out_flat = out.view(-1)
    strides = linear_strides(shape, device)  # (2,)

    # Compute AABB bounds using helper
    lo, hi, valid = compute_aabb_with_intensity_floor(
        centers, Ls, amps, shape, truncate, intensity_floor, device
    )

    # Filter invalid boxes
    if not torch.all(valid):
        centers, Ls, amps = (
            centers[valid],
            Ls[valid],
            amps[valid],
        )
        lo, hi = lo[valid], hi[valid]
        if centers.numel() == 0:
            return out

    # Group on GPU
    uniq, inv = group_by_box_gpu(lo, hi)

    for g in range(uniq.shape[0]):
        box_shape = uniq[g].tolist()  # [h0, h1]
        idx = torch.nonzero(inv == g, as_tuple=False).squeeze(1)

        mu = centers[idx]  # (K,2)
        L = Ls[idx]  # (K,2,2)
        a = amps[idx]  # (K,)
        lo_sel = lo[idx]  # (K,2)

        base, lin_offsets = cached_base_and_offsets(
            box_shape, strides, device, dtype=torch.float32
        )  # (2,P), (P,)
        base_idx = (lo_sel.to(torch.long) * strides).sum(dim=1)  # (K,)

        P = base.shape[1]
        # Calculate optimal chunk size for memory management (2D)
        P_chunk = chunk_size or calculate_optimal_chunk_size(
            K=len(idx), d=2, device=device, dtype=torch.float32
        )
        for p0 in range(0, P, P_chunk):
            p1 = min(P, p0 + P_chunk)
            d0 = base[0, p0:p1][None, :] + lo_sel[:, 0:1] - mu[:, 0:1]  # (K,Pc)
            d1 = base[1, p0:p1][None, :] + lo_sel[:, 1:2] - mu[:, 1:2]  # (K,Pc)

            # ||y||^2 via explicit forward-substitution
            dist_sq = fwd_norm2_2d(L, d0, d1)  # (K,Pc)

            # Shifted Gaussian: a·scale·max(0, exp(-0.5·D²) - C)
            vals = (
                a[:, None]
                * scale
                * torch.clamp(torch.exp(-0.5 * dist_sq) - shift_C, min=0.0)
            )

            idx_flat = (base_idx[:, None] + lin_offsets[p0:p1][None, :]).reshape(-1)
            out_flat.index_add_(0, idx_flat, vals.reshape(-1))

    return out


def _render_gaussians_3d(
    shape: Sequence[int],
    centers: torch.Tensor,  # (N,3)
    Ls: torch.Tensor,  # (N,3,3)
    amps: torch.Tensor,  # (N,)
    truncate: float,
    intensity_floor: Optional[float],
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Specialized 3D renderer with explicit forward substitution."""
    device = centers.device
    # Shifted Gaussian constants for C⁰ continuous truncation
    shift_C = math.exp(-0.5 * truncate * truncate)
    scale = 1.0 / (1.0 - shift_C)
    out = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    out_flat = out.view(-1)
    strides = linear_strides(shape, device)  # (3,)

    # Compute AABB bounds using helper
    lo, hi, valid = compute_aabb_with_intensity_floor(
        centers, Ls, amps, shape, truncate, intensity_floor, device
    )

    # Filter invalid boxes
    if not torch.all(valid):
        centers, Ls, amps = (
            centers[valid],
            Ls[valid],
            amps[valid],
        )
        lo, hi = lo[valid], hi[valid]
        if centers.numel() == 0:
            return out

    # Group on GPU
    uniq, inv = group_by_box_gpu(lo, hi)

    for g in range(uniq.shape[0]):
        box_shape = uniq[g].tolist()  # [h0, h1, h2]
        idx = torch.nonzero(inv == g, as_tuple=False).squeeze(1)

        mu = centers[idx]  # (K,3)
        L = Ls[idx]  # (K,3,3)
        a = amps[idx]  # (K,)
        lo_sel = lo[idx]  # (K,3)

        base, lin_offsets = cached_base_and_offsets(
            box_shape, strides, device, dtype=torch.float32
        )  # (3,P),(P,)
        base_idx = (lo_sel.to(torch.long) * strides).sum(dim=1)  # (K,)

        P = base.shape[1]
        # Calculate optimal chunk size for memory management (3D)
        P_chunk = chunk_size or calculate_optimal_chunk_size(
            K=len(idx), d=3, device=device, dtype=torch.float32
        )
        for p0 in range(0, P, P_chunk):
            p1 = min(P, p0 + P_chunk)
            d0 = base[0, p0:p1][None, :] + lo_sel[:, 0:1] - mu[:, 0:1]  # (K,Pc)
            d1 = base[1, p0:p1][None, :] + lo_sel[:, 1:2] - mu[:, 1:2]
            d2 = base[2, p0:p1][None, :] + lo_sel[:, 2:3] - mu[:, 2:3]

            dist_sq = fwd_norm2_3d(L, d0, d1, d2)  # (K,Pc)

            # Shifted Gaussian: a·scale·max(0, exp(-0.5·D²) - C)
            vals = (
                a[:, None]
                * scale
                * torch.clamp(torch.exp(-0.5 * dist_sq) - shift_C, min=0.0)
            )

            idx_flat = (base_idx[:, None] + lin_offsets[p0:p1][None, :]).reshape(-1)
            out_flat.index_add_(0, idx_flat, vals.reshape(-1))

    return out


# ===== Main Rendering Function ==============================================


def render_gaussians(
    shape: Sequence[int],
    centers: torch.Tensor,  # (N, d) voxel coords
    Ls: torch.Tensor,  # (N, d, d) lower-tri
    amps: torch.Tensor,  # (N,)
    truncate: float = DEFAULT_TRUNCATION_RADIUS,
    intensity_floor: Optional[
        float
    ] = 1e-5,  # for amplitude-aware culling; None disables
    chunk_size: Optional[int] = None,  # P-dimension chunk size for memory control
) -> torch.Tensor:
    """
    Fast vectorized renderer with 2D/3D fast-paths.
    Falls back to the generic nD implementation for d != 2 and d != 3.

    Renders shifted Gaussians: a * scale * max(0, exp(-0.5 * ||y||^2) - C)
    where y = L^{-1}(x - mu), C = exp(-0.5 * T^2), scale = 1/(1-C).
    The shift ensures C^0 continuity at the truncation boundary.

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
    truncate : float, default=DEFAULT_TRUNCATION_RADIUS
        Truncation radius in standard deviations.
    intensity_floor : float or None, default=1e-5
        Minimum intensity threshold for amplitude-aware culling.
        Pass ``None`` (or a non-positive value) to disable culling entirely.
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
            shape, centers, Ls, amps, truncate, intensity_floor, chunk_size
        )
    if d == 3:
        return _render_gaussians_3d(
            shape, centers, Ls, amps, truncate, intensity_floor, chunk_size
        )

    # --- Generic nD implementation ---
    device = centers.device
    # Shifted Gaussian constants for C⁰ continuous truncation
    shift_C = math.exp(-0.5 * truncate * truncate)
    scale = 1.0 / (1.0 - shift_C)
    out = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    out_flat = out.view(-1)
    strides = linear_strides(shape, device)  # (d,)

    # Compute AABB bounds using helper
    lo, hi, valid = compute_aabb_with_intensity_floor(
        centers, Ls, amps, shape, truncate, intensity_floor, device
    )

    # Filter invalid boxes
    if not torch.all(valid):
        centers = centers[valid]
        Ls = Ls[valid]
        amps = amps[valid]
        lo = lo[valid]
        hi = hi[valid]

    if centers.numel() == 0:
        return out

    # --- Group by box size for reuse of base grid ---
    groups = group_by_box(lo, hi)  # { (h1,..,hd) : idx }
    for box_shape, idx in groups.items():
        # Splat subset
        mu = centers[idx]  # (K, d)
        L = Ls[idx]  # (K, d, d)
        a = amps[idx]  # (K,)

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
        P_chunk = chunk_size or calculate_optimal_chunk_size(
            K=len(idx), d=d, device=device, dtype=torch.float32
        )
        for p0 in range(0, P, P_chunk):
            p1 = min(P, p0 + P_chunk)

            # Process chunk: base coordinates and indices for this chunk
            base_chunk = base[:, p0:p1]  # (d, Pc)
            lin_offsets_chunk = lin_offsets[p0:p1]  # (Pc,)

            # Per-splat delta = base + lo - mu   (broadcast to (K, d, Pc))
            delta = (
                base_chunk[None, :, :] + lo_f[:, :, None] - mu[:, :, None]
            )  # (K, d, Pc)

            # Solve L y = delta  (batched lower-tri solve with Pc RHS per splat)
            # MPS note: torch.linalg.solve_triangular has ~10x CPU overhead on Apple Silicon
            y = torch.linalg.solve_triangular(L, delta, upper=False)

            # Shifted Gaussian: a·scale·max(0, exp(-0.5·D²) - C)
            dist_sq = torch.sum(y * y, dim=1)  # (K, Pc)
            vals = (
                a[:, None]
                * scale
                * torch.clamp(torch.exp(-0.5 * dist_sq) - shift_C, min=0.0)
            )  # (K, Pc)

            # Absolute flat indices (K, Pc) -> (K*Pc,)
            idx_flat = (base_idx[:, None] + lin_offsets_chunk[None, :]).reshape(-1)
            vals_flat = vals.reshape(-1)

            # Accumulate into output
            out_flat.index_add_(0, idx_flat, vals_flat)

    return out
