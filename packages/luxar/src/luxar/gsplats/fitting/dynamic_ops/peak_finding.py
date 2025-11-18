# peak_finding.py
"""Residual peak finding for dynamic seeding operations."""

from __future__ import annotations

import itertools
from typing import List, Tuple

import torch


def _find_residual_peaks(
    residual: torch.Tensor,
    k_max_residuals: int,
    nms_radius_vox: float,
    enable_tiled: bool = False,
    num_tiles_per_dim: int = 8,
    k_per_tile: int = 1,
) -> List[Tuple[int, ...]]:
    """
    Find k strongest residual peaks with spatial exclusion (non-maximum suppression).

    Supports two modes:
    - Global mode (enable_tiled=False): Find top k peaks globally
    - Tiled mode (enable_tiled=True): Divide into num_tiles_per_dim tiles per dimension,
      find k_per_tile peaks in each tile for fair spatial coverage

    Args:
        residual: Residual image (target - prediction)
        k_max_residuals: Number of peaks to find (global mode only)
        nms_radius_vox: Minimum distance between peaks
        enable_tiled: If True, use tile-based seeding for spatial fairness
        num_tiles_per_dim: Number of tiles per dimension (e.g., 8 → 8×8=64 tiles for 2D, 8³=512 for 3D)
        k_per_tile: Number of peaks to find per tile (tiled mode only)

    Returns:
        List of peak coordinates as tuples
    """
    if enable_tiled:
        return _find_residual_peaks_tiled(
            residual, num_tiles_per_dim, k_per_tile, nms_radius_vox
        )
    else:
        return _find_residual_peaks_global(residual, k_max_residuals, nms_radius_vox)


def _find_residual_peaks_global(
    residual: torch.Tensor, k_max_residuals: int, nms_radius_vox: float
) -> List[Tuple[int, ...]]:
    """
    Find k strongest residual peaks globally with spatial exclusion (original method).

    This is the original global peak finding - all regions compete for top k spots.
    Bright/high-residual regions tend to dominate.

    Args:
        residual: Residual image (target - prediction)
        k_max_residuals: Number of peaks to find globally
        nms_radius_vox: Minimum distance between peaks

    Returns:
        List of peak coordinates as tuples (sorted by residual magnitude, descending)
    """
    residual_abs = torch.abs(residual)
    d = residual.ndim

    # Create kernel for non-maximum suppression
    kernel_size = int(2 * nms_radius_vox + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    if d == 2:
        max_pooled = torch.nn.functional.max_pool2d(
            residual_abs[None, None],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )[0, 0]
    elif d == 3:
        max_pooled = torch.nn.functional.max_pool3d(
            residual_abs[None, None],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )[0, 0]
    else:
        # Fallback for other dimensions
        max_pooled = residual_abs

    # Find local maxima
    is_peak = (residual_abs >= max_pooled) & (residual_abs > 0)
    peak_indices = torch.nonzero(is_peak, as_tuple=False)

    if len(peak_indices) == 0:
        return []

    # Get values and sort by magnitude
    peak_values = residual_abs[tuple(peak_indices.T)]
    sorted_indices = torch.argsort(peak_values, descending=True)

    # Take top k
    top_k = min(k_max_residuals, len(sorted_indices))
    selected_peaks = peak_indices[sorted_indices[:top_k]]

    return [tuple(peak.tolist()) for peak in selected_peaks]


def _find_residual_peaks_tiled(
    residual: torch.Tensor,
    num_tiles_per_dim: int | None,
    k_per_tile: int,
    nms_radius_vox: float,
) -> List[Tuple[int, ...]]:
    """
    Find residual peaks using tile-based approach for spatial fairness.

    Divides each dimension into num_tiles_per_dim equal tiles and finds k_per_tile
    peaks within each tile independently. This ensures all regions get attention
    regardless of brightness or residual magnitude.

    Args:
        residual: Residual image (target - prediction)
        num_tiles_per_dim: Number of tiles per dimension. If None, auto-selected:
            2D: 16 (16×16 = 256 tiles)
            3D: 6 (6×6×6 = 216 tiles)
            4D: 4 (4⁴ = 256 tiles)
            5D+: 2 (2^d tiles)
        k_per_tile: Number of peaks to find per tile
        nms_radius_vox: Minimum distance between peaks (applied within each tile)

    Returns:
        List of peak coordinates as tuples (in global image coordinates)

    Notes:
        - Each tile is processed independently (no inter-tile competition)
        - Bright regions cannot dominate dim regions
        - Tiles with residuals below convergence threshold contribute 0 peaks
        - Total peaks ≈ k_per_tile × num_active_tiles
        - Coordinates are returned in global image space
    """
    residual_abs = torch.abs(residual)
    shape = residual.shape
    d = residual.ndim

    # Auto-select num_tiles_per_dim based on dimensionality
    if num_tiles_per_dim is None:
        if d == 2:
            num_tiles_per_dim = 16  # 16×16 = 256 tiles
        elif d == 3:
            num_tiles_per_dim = 6  # 6×6×6 = 216 tiles
        elif d == 4:
            num_tiles_per_dim = 4  # 4⁴ = 256 tiles
        else:
            num_tiles_per_dim = 2  # 2^d tiles

    all_peaks = []

    # Calculate tile size for each dimension
    # Divide each dimension into num_tiles_per_dim equal parts
    tile_sizes = [max(1, dim_size // num_tiles_per_dim) for dim_size in shape]

    # Generate tile grid coordinates (start positions)
    tile_starts = []
    for i, dim_size in enumerate(shape):
        # Create num_tiles_per_dim evenly-spaced starting positions
        starts = [
            (dim_size * tile_idx) // num_tiles_per_dim
            for tile_idx in range(num_tiles_per_dim)
        ]
        tile_starts.append(starts)

    # Process each tile
    for tile_coords in itertools.product(*tile_starts):
        # Calculate tile boundaries
        slices = []
        tile_origin = []
        for i, start in enumerate(tile_coords):
            # Calculate end position for this tile
            # Extend to next tile start, or image boundary if last tile
            end = min(start + tile_sizes[i], shape[i])

            slices.append(slice(start, end))
            tile_origin.append(start)

        # Extract tile
        tile = residual_abs[tuple(slices)]

        # Skip very small tiles
        if tile.numel() < 4:
            continue

        # Find peaks within this tile using NMS
        tile_peaks = _find_peaks_in_tile(tile, k_per_tile, nms_radius_vox)

        # Convert local tile coordinates to global image coordinates
        for local_coords in tile_peaks:
            global_coords = tuple(local_coords[i] + tile_origin[i] for i in range(d))
            all_peaks.append(global_coords)

    return all_peaks


def _find_peaks_in_tile(
    tile: torch.Tensor, k_max: int, nms_radius_vox: float
) -> List[Tuple[int, ...]]:
    """
    Find up to k_max peaks within a single tile using NMS.

    Args:
        tile: Tile region (absolute residual values)
        k_max: Maximum number of peaks to find
        nms_radius_vox: NMS radius for spatial exclusion

    Returns:
        List of peak coordinates (in local tile coordinates)
    """
    d = tile.ndim

    # Create kernel for non-maximum suppression
    kernel_size = int(2 * nms_radius_vox + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Ensure kernel doesn't exceed tile size
    min_tile_dim = min(tile.shape)
    kernel_size = min(kernel_size, min_tile_dim)

    # Only enforce minimum if tile is large enough
    if min_tile_dim >= 3 and kernel_size < 3:
        kernel_size = 3  # Minimum 3×3(×3) kernel when tile allows it
    elif kernel_size % 2 == 0:
        kernel_size = max(1, kernel_size - 1)  # Ensure odd kernel

    # Apply max pooling for NMS
    if d == 2:
        max_pooled = torch.nn.functional.max_pool2d(
            tile[None, None],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )[0, 0]
    elif d == 3:
        max_pooled = torch.nn.functional.max_pool3d(
            tile[None, None],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )[0, 0]
    else:
        # Fallback: no NMS for unsupported dimensions
        max_pooled = tile

    # Find local maxima
    is_peak = (tile >= max_pooled) & (tile > 0)
    peak_indices = torch.nonzero(is_peak, as_tuple=False)

    if len(peak_indices) == 0:
        return []

    # Get values and sort by magnitude
    peak_values = tile[tuple(peak_indices.T)]
    sorted_indices = torch.argsort(peak_values, descending=True)

    # Take top k
    top_k = min(k_max, len(sorted_indices))
    selected_peaks = peak_indices[sorted_indices[:top_k]]

    return [tuple(peak.tolist()) for peak in selected_peaks]
