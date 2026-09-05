"""Grid quantization helpers for spatial ordering.

Space-filling curves operate on integer grid coordinates; these helpers
normalize float positions into an integer grid and pick an appropriate grid
resolution from the data spread.
"""

from __future__ import annotations

import numpy as np


def normalize_coords_to_grid(
    coords: np.ndarray, min_coords: np.ndarray, max_coords: np.ndarray, resolution: int
) -> np.ndarray:
    """Normalize float coordinates to integer grid [0, resolution-1].

    Args:
        coords: Float coordinates, shape (N, d)
        min_coords: Minimum bounds, shape (d,)
        max_coords: Maximum bounds, shape (d,)
        resolution: Grid resolution (e.g., 2^16 = 65536)

    Returns:
        Integer coordinates, shape (N, d), dtype uint32
    """
    working_dtype = np.promote_types(coords.dtype, np.float32)
    coords = coords.astype(working_dtype, copy=False)
    min_coords = min_coords.astype(working_dtype, copy=False)
    max_coords = max_coords.astype(working_dtype, copy=False)

    # Normalize to [0, 1]
    ranges = max_coords - min_coords
    # Handle degenerate dimensions (zero range)
    ranges = np.where(ranges > 0, ranges, 1.0)
    normalized = (coords - min_coords) / ranges

    # Clamp to [0, 1] (handle floating point errors)
    normalized = np.clip(normalized, 0.0, 1.0)

    # Scale to [0, resolution-1]
    grid_coords = (normalized * (resolution - 1)).astype(np.uint32)

    return np.asarray(grid_coords)


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
    resolution = int(2 ** int(np.log2(resolution)))

    return resolution
