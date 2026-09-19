"""Layer-neutral coordinate quantization and Morton encoding primitives."""

from __future__ import annotations

import numpy as np


def normalize_coords_to_grid(
    coords: np.ndarray,
    min_coords: np.ndarray,
    max_coords: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Normalize floating coordinates to an integer grid."""
    working_dtype = np.promote_types(coords.dtype, np.float32)
    values = coords.astype(working_dtype, copy=False)
    minimum = min_coords.astype(working_dtype, copy=False)
    maximum = max_coords.astype(working_dtype, copy=False)
    ranges = np.where(maximum - minimum > 0, maximum - minimum, 1.0)
    normalized = np.clip((values - minimum) / ranges, 0.0, 1.0)
    return np.asarray((normalized * (resolution - 1)).astype(np.uint32))


def morton_encode_nd(coords: np.ndarray, bits_per_dim: int = 16) -> np.ndarray:
    """Encode nD integer coordinates into one uint64 Morton code per row."""
    morton = np.zeros(coords.shape[0], dtype=np.uint64)
    for bit in range(bits_per_dim):
        for dimension in range(coords.shape[1]):
            coord_bit = (coords[:, dimension] >> bit) & 1
            morton |= coord_bit.astype(np.uint64) << (bit * coords.shape[1] + dimension)
    return morton
