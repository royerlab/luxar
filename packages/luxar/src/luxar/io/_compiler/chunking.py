"""Chunk-shape heuristic shared by every compiler dataset serializer.

Private support module for :class:`luxar.io.compiler.LuxarZarrCompiler`. Lives one
level under the orchestrator because it has many in-package consumers (every
``datasets/`` serializer, the gsplat array writer, and the spatial-ordering glue).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...typing_utils.constants import TARGET_CHUNK_BYTES


def calculate_intelligent_chunks(
    shape: Tuple[int, ...],
    target_chunk_bytes: int = TARGET_CHUNK_BYTES,
    spatial_index_data: Optional[Dict[str, Any]] = None,
    *,
    dtype: np.dtype,
) -> Tuple[int, ...]:
    """Calculate optimal chunk shape for a dataset.

    When spatial ordering data is available, uses chunk_size from ordering.

    The byte-target heuristic depends on dtype itemsize: a uint8 colors
    array of shape (N, 3) yields different optimal chunks than a float32
    positions array of the same shape, so ``dtype`` is required and
    keyword-only — every call site passes the actual array dtype.

    Args:
        shape: Shape of the dataset.
        target_chunk_bytes: Target chunk payload size in bytes.
        spatial_index_data: Optional ordering data (with chunk_size).
        dtype: NumPy dtype of the array being chunked.

    Returns:
        Optimized chunk shape.
    """
    element_size = max(1, int(dtype.itemsize))
    target_elements = max(1, int(target_chunk_bytes) // element_size)

    if len(shape) == 1:
        # 1D array - use spatial index chunk_size if available for alignment
        if spatial_index_data and "chunk_size" in spatial_index_data:
            return (min(shape[0], spatial_index_data["chunk_size"]),)
        return (min(shape[0], target_elements),)

    if len(shape) == 2:
        # 2D array (e.g., positions) - chunk along first dimension
        n_points, n_dims = shape

        # If ordering data is available, use its chunk_size
        if spatial_index_data and "chunk_size" in spatial_index_data:
            chunk_points = spatial_index_data["chunk_size"]
            return (chunk_points, n_dims)

        # Fallback to standard byte-based chunking
        chunk_points = min(n_points, max(1, target_elements // n_dims))
        return (chunk_points, n_dims)

    # For higher dimensions, use reasonable byte-based defaults
    return tuple(min(s, target_elements) for s in shape)
