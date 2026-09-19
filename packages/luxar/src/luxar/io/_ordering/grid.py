"""Grid quantization helpers for spatial ordering.

Space-filling curves operate on integer grid coordinates; these helpers
normalize float positions into an integer grid and pick an appropriate grid
resolution from the data spread.
"""

from __future__ import annotations

import numpy as np

from ...utils.spatial_ordering import (
    normalize_coords_to_grid as normalize_coords_to_grid,
)


def compute_auto_resolution(coords: np.ndarray, max_resolution: int = 2**16) -> int:
    """Compute appropriate resolution based on data spread.

    Args:
        coords: Float coordinates, shape (N, d)
        max_resolution: Maximum resolution (default 65536)

    Returns:
        Resolution as power of 2, capped at max_resolution
    """
    coords = coords.astype(np.promote_types(coords.dtype, np.float32), copy=False)
    spread = coords.max(axis=0) - coords.min(axis=0)
    max_spread = spread.max()

    # Target ~10 grid cells per unit of spread
    target_resolution = int(max_spread * 10)

    # Clamp to [256, max_resolution]
    resolution = min(max_resolution, max(256, target_resolution))

    # Round to nearest power of 2
    resolution = int(2 ** int(np.log2(resolution)))

    return resolution
