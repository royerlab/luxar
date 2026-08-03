"""Chunk-shape heuristic shared by every compiler dataset serializer.

Private support module for :class:`luxar.io.compiler.LuxarZarrCompiler`. Lives one
level under the orchestrator because it has many in-package consumers (every
``datasets/`` serializer, the gsplat array writer, and the spatial-ordering glue).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...typing_utils.constants import TARGET_CHUNK_BYTES


def _atom_aligned_rows(ideal_rows: int, atom: int, n_rows: int) -> int:
    """Largest multiple of ``atom`` not exceeding ``ideal_rows`` (at least one
    atom), clamped to ``n_rows`` and floored at 1 row so an empty array can
    never yield an invalid 0-row zarr chunk. Keeps a per-array zarr chunk on
    the spatial-index query grid (viewer row ranges are multiples of ``atom``)
    while letting each array size its chunk to its own dtype byte budget."""
    atom = max(1, int(atom))
    multiple = max(1, int(ideal_rows) // atom)
    return max(1, min(int(n_rows), multiple * atom))


def calculate_intelligent_chunks(
    shape: Tuple[int, ...],
    target_chunk_bytes: int = TARGET_CHUNK_BYTES,
    spatial_index_data: Optional[Dict[str, Any]] = None,
    *,
    dtype: np.dtype,
    per_array_bytes: bool = False,
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
        per_array_bytes: Opt-in (default ``False``). When ``True`` and a
            spatial ``chunk_size`` atom is present, size the first-axis chunk
            to this array's OWN dtype byte budget, rounded DOWN to a multiple
            of the atom (never below one atom), so large points scenes issue
            far fewer requests. When ``False`` the result is byte-for-byte
            identical to the historical atom-sized behavior (gsplats/lines
            keep the atom).

    Returns:
        Optimized chunk shape.
    """
    element_size = max(1, int(dtype.itemsize))
    target_elements = max(1, int(target_chunk_bytes) // element_size)

    atom = None
    if spatial_index_data and "chunk_size" in spatial_index_data:
        atom = int(spatial_index_data["chunk_size"])

    if len(shape) == 1:
        # 1D array - use spatial index chunk_size if available for alignment
        if atom is not None:
            if per_array_bytes:
                return (_atom_aligned_rows(target_elements, atom, shape[0]),)
            return (min(shape[0], atom),)
        return (min(shape[0], target_elements),)

    if len(shape) == 2:
        # 2D array (e.g., positions) - chunk along first dimension
        n_points, n_dims = shape
        ideal = max(1, target_elements // n_dims)

        # If ordering data is available, use its chunk_size
        if atom is not None:
            if per_array_bytes:
                return (_atom_aligned_rows(ideal, atom, n_points), n_dims)
            return (atom, n_dims)

        # Fallback to standard byte-based chunking
        return (min(n_points, ideal), n_dims)

    # For higher dimensions, use reasonable byte-based defaults
    return tuple(min(s, target_elements) for s in shape)
