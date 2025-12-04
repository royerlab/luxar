"""Spatial ordering algorithms for Points and GSplats.

This module provides Morton and Hilbert ordering for both Points (with compound
ordering for discrete dimensions) and GSplats (simple spatial ordering).
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from luxar.core import Dimension


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


def hilbert_encode_nd(coords: np.ndarray, bits_per_dim: int = 16) -> np.ndarray:
    """Encode nD integer coordinates to Hilbert curve indices.

    Args:
        coords: Integer coordinates, shape (N, d)
        bits_per_dim: Bits to use per dimension (default 16)

    Returns:
        Hilbert indices, shape (N,), dtype uint64
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

    n_points, n_dims = coords.shape

    # Create Hilbert curve (resolution = 2^bits_per_dim)
    hilbert = HilbertCurve(bits_per_dim, n_dims)

    # Compute Hilbert indices
    hilbert_indices = np.array(
        [hilbert.distance_from_point(coords[i]) for i in range(n_points)],
        dtype=np.uint64,
    )

    return hilbert_indices


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


def sort_points_compound(
    positions: np.ndarray,
    dimensions: list[Dimension],
    method: Literal["morton", "hilbert"] = "hilbert",
) -> tuple[np.ndarray, dict]:
    """Sort Points using compound ordering (discrete dims → spatial curve).

    This implements the compound ordering strategy from luxar.io spec:
    - Primary sort: Discrete dimensions (lexicographic)
    - Secondary sort: Morton/Hilbert code of spatial dimensions

    Args:
        positions: Point positions, shape (N, d), all dimensions
        dimensions: Dimension objects defining discrete/spatial properties
        method: Spatial curve method ("morton" or "hilbert"), default "hilbert"

    Returns:
        sort_indices: Indices to reorder points
        metadata: Dict with ordering metadata
    """
    n_points, n_dims = positions.shape

    # Identify dimension categories (per spec Section "Compound Ordering")
    slice_dims = [i for i, d in enumerate(dimensions) if d.discrete and not d.display]
    ordering_dims = [i for i, d in enumerate(dimensions) if not d.discrete or d.display]

    # Compute bits per ordering dimension
    if ordering_dims:
        bits_per_dim = min(21, 64 // len(ordering_dims))
    else:
        bits_per_dim = 21  # Fallback

    # Extract ordering dimension coordinates
    if ordering_dims:
        ordering_coords = positions[:, ordering_dims]
        ordering_min = ordering_coords.min(axis=0)
        ordering_max = ordering_coords.max(axis=0)

        # Normalize to grid (using 2^bits_per_dim resolution)
        grid_coords = normalize_coords_to_grid(
            ordering_coords, ordering_min, ordering_max, 2**bits_per_dim
        )

        # Compute spatial curve codes
        if method == "morton":
            spatial_codes = morton_encode_nd(grid_coords, bits_per_dim)
        elif method == "hilbert":
            spatial_codes = hilbert_encode_nd(grid_coords, bits_per_dim)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        # No ordering dimensions - all discrete
        spatial_codes = np.zeros(n_points, dtype=np.uint64)
        ordering_min = np.array([])
        ordering_max = np.array([])

    # Create compound sort key
    if slice_dims:
        # Extract discrete dimension values
        slice_values = positions[:, slice_dims]

        # Create sort keys: (discrete_tuple, spatial_code)
        # Use lexicographic sort on discrete dims, then spatial code
        sort_indices = np.lexsort(
            [spatial_codes]
            + [slice_values[:, i] for i in range(len(slice_dims) - 1, -1, -1)]
        )
    else:
        # Pure spatial ordering (no discrete dimensions)
        sort_indices = np.argsort(spatial_codes)

    # Build metadata
    metadata = {
        "ordering": method,
        "slice_dims": slice_dims,
        "ordering_dims": ordering_dims,
        "ordering_min": ordering_min.tolist() if len(ordering_min) > 0 else [],
        "ordering_max": ordering_max.tolist() if len(ordering_max) > 0 else [],
        "ordering_bits_per_dim": bits_per_dim,
    }

    return sort_indices, metadata


def sort_splats_spatial(
    centers: np.ndarray,
    method: Literal["morton", "hilbert"] = "hilbert",
    resolution: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """Sort GSplats using simple spatial ordering (no compound ordering).

    GSplats don't have discrete dimensions, so this is pure spatial ordering.

    Args:
        centers: Splat centers, shape (N, d), float32
        method: Spatial curve method ("morton" or "hilbert")
        resolution: Grid resolution (auto-computed if None)

    Returns:
        sort_indices: Indices to reorder splats
        metadata: Dict with ordering metadata
    """
    n_splats, ndim = centers.shape

    # Auto-compute resolution if not provided
    if resolution is None:
        resolution = compute_auto_resolution(centers)

    # Compute bits per dimension
    bits_per_dim = min(21, 64 // ndim)

    # Get bounds
    min_coords = centers.min(axis=0)
    max_coords = centers.max(axis=0)

    # Normalize to integer grid
    grid_coords = normalize_coords_to_grid(
        centers, min_coords, max_coords, 2**bits_per_dim
    )

    # Compute spatial curve codes
    if method == "morton":
        spatial_codes = morton_encode_nd(grid_coords, bits_per_dim)
    elif method == "hilbert":
        spatial_codes = hilbert_encode_nd(grid_coords, bits_per_dim)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Sort by spatial code
    sort_indices = np.argsort(spatial_codes)

    # Metadata
    metadata = {
        "ordering": method,
        "ordering_min": min_coords.tolist(),
        "ordering_max": max_coords.tolist(),
        "ordering_bits_per_dim": bits_per_dim,
    }

    return sort_indices, metadata


def compute_chunk_bounds_points(
    positions: np.ndarray,
    radii: Optional[np.ndarray],
    chunk_size: int,
) -> np.ndarray:  # type: ignore[return]
    """Compute chunk bounding boxes for Points (includes radius extent).

    Args:
        positions: Point positions (already sorted), shape (N, d)
        radii: Point radii (already sorted), shape (N,), or None
        chunk_size: Number of points per chunk

    Returns:
        chunk_bounds: Bounding boxes, shape (num_chunks, d, 2)
    """
    n_points, ndim = positions.shape
    num_chunks = (n_points + chunk_size - 1) // chunk_size

    chunk_bounds = np.zeros((num_chunks, ndim, 2), dtype=np.float32)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_points)

        chunk_positions = positions[start_idx:end_idx]

        if radii is not None:
            chunk_radii = radii[start_idx:end_idx]
            # Bounds include radius extent
            mins = (chunk_positions - chunk_radii[:, np.newaxis]).min(axis=0)
            maxs = (chunk_positions + chunk_radii[:, np.newaxis]).max(axis=0)
        else:
            # No radii provided - add small safety margin to prevent missing points
            # at chunk boundaries when default radius is applied during rendering
            # Safety margin: 1% of coordinate range or 0.01, whichever is larger
            coord_range = chunk_positions.max(axis=0) - chunk_positions.min(axis=0)
            safety_margin = np.maximum(coord_range * 0.01, 0.01)

            mins = chunk_positions.min(axis=0) - safety_margin
            maxs = chunk_positions.max(axis=0) + safety_margin

        chunk_bounds[chunk_idx, :, 0] = mins
        chunk_bounds[chunk_idx, :, 1] = maxs

    return chunk_bounds


def compute_chunk_bounds_gsplats(
    centers: np.ndarray,
    cholesky_factors: np.ndarray,
    chunk_size: int,
    coverage_sigma: float = 3.0,
) -> np.ndarray:  # type: ignore[return]
    """Compute chunk bounding boxes for GSplats (includes ellipsoidal extent).

    Args:
        centers: Splat centers (already sorted), shape (N, d)
        cholesky_factors: Packed Cholesky factors (already sorted), shape (N, k)
        chunk_size: Number of splats per chunk
        coverage_sigma: Coverage radius in standard deviations (default 3.0)

    Returns:
        chunk_bounds: Bounding boxes, shape (num_chunks, d, 2)
    """
    n_splats, ndim = centers.shape
    num_chunks = (n_splats + chunk_size - 1) // chunk_size

    chunk_bounds = np.zeros((num_chunks, ndim, 2), dtype=np.float32)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_splats)

        chunk_centers = centers[start_idx:end_idx]
        chunk_cholesky = cholesky_factors[start_idx:end_idx]

        # Compute ellipsoidal extent (per spec: extent[d] = sqrt(covariance[d,d]) * 3σ)
        extents = np.zeros((end_idx - start_idx, ndim), dtype=np.float32)

        for d in range(ndim):
            # Covariance diagonal from Cholesky factors
            # For dimension d: start_idx = d*(d+1)//2
            start_chol_idx = d * (d + 1) // 2
            # covariance[d,d] = sum(L[start_chol_idx + i]^2 for i in 0..d)
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
