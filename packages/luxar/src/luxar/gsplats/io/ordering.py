"""Spatial ordering algorithms for Gaussian splats.

This module provides Morton and Hilbert ordering for GSplats to improve
compression and spatial query performance.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def morton_encode_nd(coords: np.ndarray, bits_per_dim: int = 16) -> np.ndarray:  # type: ignore[return]
    """Encode nD integer coordinates to Morton codes via bit interleaving.

    Args:
        coords: Integer coordinates, shape (N, d)
        bits_per_dim: Bits to use per dimension (default 16)

    Returns:
        Morton codes, shape (N,), dtype uint64
    """
    n_points, n_dims = coords.shape
    morton = np.zeros(n_points, dtype=np.uint64)

    for bit in range(bits_per_dim):
        for dim in range(n_dims):
            coord_bit = (coords[:, dim] >> bit) & 1
            morton |= coord_bit.astype(np.uint64) << (bit * n_dims + dim)

    return morton


def normalize_coords_to_grid(
    coords: np.ndarray, min_coords: np.ndarray, max_coords: np.ndarray, resolution: int
) -> np.ndarray:  # type: ignore[return]
    """Normalize float coordinates to integer grid [0, resolution-1].

    Args:
        coords: Float coordinates, shape (N, d)
        min_coords: Minimum bounds, shape (d,)
        max_coords: Maximum bounds, shape (d,)
        resolution: Grid resolution (e.g., 2^16 = 65536)

    Returns:
        Integer coordinates, shape (N, d), dtype uint32
    """
    # Normalize to [0, 1]
    ranges = max_coords - min_coords
    # Handle degenerate dimensions (zero range)
    ranges = np.where(ranges > 0, ranges, 1.0)
    normalized = (coords - min_coords) / ranges

    # Clamp to [0, 1] (handle floating point errors)
    normalized = np.clip(normalized, 0.0, 1.0)

    # Scale to [0, resolution-1]
    grid_coords = (normalized * (resolution - 1)).astype(np.uint32)

    return grid_coords


def compute_auto_resolution(coords: np.ndarray, max_resolution: int = 2**16) -> int:
    """Compute appropriate resolution based on data spread.

    Args:
        coords: Float coordinates, shape (N, d)
        max_resolution: Maximum resolution (default 65536)

    Returns:
        Resolution as power of 2, capped at max_resolution
    """
    spread = coords.max(axis=0) - coords.min(axis=0)
    max_spread = spread.max()

    # Target ~10 grid cells per unit of spread
    target_resolution = int(max_spread * 10)

    # Clamp to [256, max_resolution]
    resolution = min(max_resolution, max(256, target_resolution))

    # Round to nearest power of 2
    resolution = 2 ** int(np.log2(resolution))

    return resolution


def sort_splats_morton(
    centers: np.ndarray,
    resolution: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Sort splats using Morton (Z-order) curve.

    Args:
        centers: Splat centers, shape (N, d), float32
        resolution: Grid resolution (auto-computed if None)

    Returns:
        sort_indices: Indices to sort splats, shape (N,)
        metadata: Dict with ordering info (min, max, bits_per_dim, resolution)
    """
    n_splats, ndim = centers.shape

    # Auto-compute resolution if not provided
    if resolution is None:
        resolution = compute_auto_resolution(centers)

    # Compute bits per dimension (max 64 bits total)
    bits_per_dim = min(21, 64 // ndim)  # 21 bits for 3D, 16 for 4D, etc.

    # Get bounds
    min_coords = centers.min(axis=0)
    max_coords = centers.max(axis=0)

    # Normalize to integer grid
    grid_coords = normalize_coords_to_grid(centers, min_coords, max_coords, resolution)

    # Compute Morton codes
    morton_codes = morton_encode_nd(grid_coords, bits_per_dim=bits_per_dim)

    # Sort by Morton code
    sort_indices = np.argsort(morton_codes)

    # Metadata
    metadata = {
        "ordering": "morton",
        "morton_min": min_coords.tolist(),
        "morton_max": max_coords.tolist(),
        "morton_bits_per_dim": bits_per_dim,
        "morton_resolution": resolution,
    }

    return sort_indices, metadata


def sort_splats_hilbert(
    centers: np.ndarray,
    resolution: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Sort splats using Hilbert curve.

    Args:
        centers: Splat centers, shape (N, d), float32
        resolution: Grid resolution (auto-computed if None, must be power of 2)

    Returns:
        sort_indices: Indices to sort splats, shape (N,)
        metadata: Dict with ordering info (min, max, bits_per_dim, resolution)
    """
    try:
        from hilbertcurve.hilbertcurve import (
            HilbertCurve,  # type: ignore[import-untyped]
        )
    except ImportError:
        raise ImportError(
            "hilbertcurve package is required for Hilbert ordering. "
            "Install with: pip install hilbertcurve"
        )

    n_splats, ndim = centers.shape

    # Auto-compute resolution if not provided
    if resolution is None:
        resolution = compute_auto_resolution(centers)

    # Resolution must be power of 2 for Hilbert
    # Compute bits per dimension
    bits_per_dim = min(21, 64 // ndim)
    hilbert_resolution = 2**bits_per_dim

    # Get bounds
    min_coords = centers.min(axis=0)
    max_coords = centers.max(axis=0)

    # Normalize to integer grid [0, hilbert_resolution-1]
    grid_coords = normalize_coords_to_grid(
        centers, min_coords, max_coords, hilbert_resolution
    )

    # Create Hilbert curve
    hilbert = HilbertCurve(bits_per_dim, ndim)

    # Compute Hilbert indices
    hilbert_indices = np.array(
        [hilbert.distance_from_point(grid_coords[i]) for i in range(n_splats)],
        dtype=np.uint64,
    )

    # Sort by Hilbert index
    sort_indices = np.argsort(hilbert_indices)

    # Metadata (same format as Morton for consistency)
    metadata = {
        "ordering": "hilbert",
        "morton_min": min_coords.tolist(),  # Keep name for compatibility
        "morton_max": max_coords.tolist(),
        "morton_bits_per_dim": bits_per_dim,
        "hilbert_resolution": hilbert_resolution,
    }

    return sort_indices, metadata


def sort_splats_spatially(
    centers: np.ndarray,
    method: Literal["morton", "hilbert"] = "hilbert",
    resolution: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Sort splats using spatial ordering (Morton or Hilbert curve).

    Args:
        centers: Splat centers, shape (N, d), float32
        method: Ordering method ("morton" or "hilbert")
        resolution: Grid resolution (auto if None, max 2^16)

    Returns:
        sort_indices: Indices to reorder splats
        metadata: Dict with ordering information for storage

    Example:
        >>> centers = np.random.rand(10000, 3).astype(np.float32)
        >>> indices, meta = sort_splats_spatially(centers, method="hilbert")
        >>> sorted_centers = centers[indices]
    """
    if method == "morton":
        return sort_splats_morton(centers, resolution)
    elif method == "hilbert":
        return sort_splats_hilbert(centers, resolution)
    else:
        raise ValueError(f"Unknown ordering method: {method}")


def compute_chunk_bounds(
    centers: np.ndarray,
    cholesky_factors: np.ndarray,
    chunk_size: int,
    coverage_sigma: float = 3.0,
) -> np.ndarray:  # type: ignore[return]
    """Compute bounding boxes for chunks of spatially-ordered splats.

    The bounds include the ellipsoidal extent of each splat (computed from
    Cholesky factors) to ensure splats are not missed during spatial queries.

    Args:
        centers: Splat centers (already sorted), shape (N, d)
        cholesky_factors: Packed Cholesky factors (already sorted), shape (N, k)
        chunk_size: Number of splats per chunk
        coverage_sigma: Coverage radius in standard deviations (default 3.0 = 99.7%)

    Returns:
        chunk_bounds: Bounding boxes, shape (num_chunks, d, 2)
            [..., d, 0] = minimum bound in dimension d
            [..., d, 1] = maximum bound in dimension d
    """
    n_splats, ndim = centers.shape
    num_chunks = (n_splats + chunk_size - 1) // chunk_size

    chunk_bounds = np.zeros((num_chunks, ndim, 2), dtype=np.float32)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_splats)

        chunk_centers = centers[start_idx:end_idx]
        chunk_cholesky = cholesky_factors[start_idx:end_idx]

        # Compute ellipsoidal extent for each splat in this chunk
        # Cholesky factors are packed: [L00, L10, L11, L20, L21, L22, ...]
        # Covariance diagonal: covariance[d,d] = sum(L[start + i]^2 for i in 0..d)
        extents = np.zeros((end_idx - start_idx, ndim), dtype=np.float32)

        for d in range(ndim):
            # For dimension d, start index in packed array
            start_chol_idx = d * (d + 1) // 2
            # Sum squares of L[start_chol_idx : start_chol_idx + d + 1]
            for i in range(d + 1):
                extents[:, d] += chunk_cholesky[:, start_chol_idx + i] ** 2

            # Extent = sqrt(covariance) * coverage_sigma
            extents[:, d] = np.sqrt(extents[:, d]) * coverage_sigma

        # Compute bounds including extent
        mins = (chunk_centers - extents).min(axis=0)
        maxs = (chunk_centers + extents).max(axis=0)

        chunk_bounds[chunk_idx, :, 0] = mins
        chunk_bounds[chunk_idx, :, 1] = maxs

    return chunk_bounds
