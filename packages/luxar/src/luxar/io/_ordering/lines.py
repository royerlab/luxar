"""Lines spatial ordering (dual: vertices in D-space + segments in 2D-space).

Line vertices are ordered like Points (in D-space); segments are ordered in
(2xD)-space by concatenating both endpoints, so spatially-coherent segments land
in the same chunk. Chunk bounds include the line-width extent on spatial axes.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from luxar.core import Dimension

from .bounds import _BARRIER_BOUND_EPS
from .curves.hilbert import hilbert_encode_nd
from .curves.morton import morton_encode_128bit, morton_encode_nd
from .grid import normalize_coords_to_grid
from .points import sort_points_compound


def convert_to_indexed(
    n_vertices: int, line_type: str, indices: Optional[np.ndarray]
) -> np.ndarray:
    """Convert any line type to indexed segment pairs.

    All line types are internally converted to the unified indexed representation
    for efficient spatial ordering and storage.

    Args:
        n_vertices: Number of vertices in the lines
        line_type: One of "segments", "polyline", "loop", "indexed"
        indices: For "indexed" type, the user-provided index array

    Returns:
        segments: (S, 2) uint32 array of vertex index pairs

    Raises:
        ValueError: If line_type is invalid or indices missing for "indexed" type
    """
    if line_type == "indexed":
        if indices is None:
            raise ValueError("Indexed line type requires indices array")
        return indices.reshape(-1, 2).astype(np.uint32)
    elif line_type == "segments":
        # Each pair of consecutive vertices forms a segment
        return np.arange(n_vertices, dtype=np.uint32).reshape(-1, 2)
    elif line_type == "polyline":
        # Connect consecutive vertices: (0,1), (1,2), (2,3), ...
        return np.column_stack(
            [
                np.arange(n_vertices - 1, dtype=np.uint32),
                np.arange(1, n_vertices, dtype=np.uint32),
            ]
        )
    elif line_type == "loop":
        # Like polyline but connect last to first
        return np.column_stack(
            [
                np.arange(n_vertices, dtype=np.uint32),
                np.roll(np.arange(n_vertices, dtype=np.uint32), -1),
            ]
        )
    else:
        raise ValueError(
            f"Invalid line_type '{line_type}'. Must be one of: segments, polyline, loop, indexed"
        )


def sort_segments_compound(
    segment_coords_2d: np.ndarray,
    dimensions: list[Dimension],
    method: Literal["morton", "hilbert"] = "hilbert",
) -> tuple[np.ndarray, dict]:
    """Sort segments using compound ordering in (2×D)-space.

    Segments are represented as (2×D)-dimensional points by concatenating both
    endpoint coordinates. This captures the full geometric nature of segments
    (position, orientation, length) for spatial coherence.

    The compound ordering strategy is:
    - Primary sort: Discrete dimensions from both endpoints (lexicographic)
    - Secondary sort: Morton/Hilbert code of spatial dimensions from both endpoints

    Args:
        segment_coords_2d: Segment coordinates in (2×D)-space, shape (S, 2*d)
            First d columns are start point, second d columns are end point.
        dimensions: Original Dimension objects (may have more dims than data)
        method: Spatial curve method ("morton" or "hilbert"), default "hilbert"

    Returns:
        sort_indices: Indices to reorder segments
        metadata: Dict with ordering metadata in (2×D)-space
    """
    n_segments, n_dims_2d = segment_coords_2d.shape
    # Use actual data dimensionality (segment_coords_2d is 2×D, so D = n_dims_2d // 2)
    n_dims_original = n_dims_2d // 2
    # Use only the first n_dims_original dimensions from the dimensions list
    dimensions = dimensions[:n_dims_original]

    # Build (2×D) dimension classification
    # Discrete dims from both endpoints: if dim 3,4 are discrete in D-space,
    # then dims 3,4,d+3,d+4 are discrete in (2×D)-space
    slice_dims_2d = []
    ordering_dims_2d = []

    for i, d in enumerate(dimensions):
        if d.discrete and not d.display:
            slice_dims_2d.append(i)  # Start point discrete dim
            slice_dims_2d.append(n_dims_original + i)  # End point discrete dim
        else:
            ordering_dims_2d.append(i)  # Start point spatial dim
            ordering_dims_2d.append(n_dims_original + i)  # End point spatial dim

    # Compute bits per ordering dimension (auto-select 128-bit if needed)
    if ordering_dims_2d:
        bits_per_dim = min(21, 64 // len(ordering_dims_2d))
        use_128bit = bits_per_dim < 10
        if use_128bit:
            bits_per_dim = min(21, 128 // len(ordering_dims_2d))
    else:
        bits_per_dim = 21
        use_128bit = False

    # Extract ordering dimension coordinates
    spatial_codes: tuple[np.ndarray, np.ndarray] | np.ndarray
    if ordering_dims_2d:
        ordering_coords = segment_coords_2d[:, ordering_dims_2d]
        ordering_min = ordering_coords.min(axis=0)
        ordering_max = ordering_coords.max(axis=0)

        # Normalize to grid
        grid_coords = normalize_coords_to_grid(
            ordering_coords, ordering_min, ordering_max, 2**bits_per_dim
        )

        # Compute spatial curve codes
        if method == "morton":
            if use_128bit:
                high, low = morton_encode_128bit(grid_coords, bits_per_dim)
                spatial_codes = (high, low)  # Tuple for lexsort
            else:
                spatial_codes = morton_encode_nd(grid_coords, bits_per_dim)
        elif method == "hilbert":
            # Hilbert doesn't have 128-bit implementation - fall back to morton for high dims
            if use_128bit:
                high, low = morton_encode_128bit(grid_coords, bits_per_dim)
                spatial_codes = (high, low)
            else:
                spatial_codes = hilbert_encode_nd(grid_coords, bits_per_dim)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        # No ordering dimensions - all discrete
        spatial_codes = np.zeros(n_segments, dtype=np.uint64)
        ordering_min = np.array([])
        ordering_max = np.array([])

    # Create compound sort key
    if slice_dims_2d:
        # Extract discrete dimension values
        slice_values = segment_coords_2d[:, slice_dims_2d]

        # Create sort keys: (discrete_tuple, spatial_code)
        if isinstance(spatial_codes, tuple):
            # 128-bit: lexsort by (low, high, discrete_dims...)
            high, low = spatial_codes
            sort_indices = np.lexsort(
                [low, high]
                + [slice_values[:, i] for i in range(len(slice_dims_2d) - 1, -1, -1)]
            )
        else:
            sort_indices = np.lexsort(
                [spatial_codes]
                + [slice_values[:, i] for i in range(len(slice_dims_2d) - 1, -1, -1)]
            )
    else:
        # Pure spatial ordering (no discrete dimensions)
        if isinstance(spatial_codes, tuple):
            high, low = spatial_codes
            sort_indices = np.lexsort([low, high])
        else:
            sort_indices = np.argsort(spatial_codes)

    # Build metadata
    metadata = {
        "ordering": method,
        "slice_dims": slice_dims_2d,
        "ordering_dims": ordering_dims_2d,
        "ordering_min": ordering_min.tolist() if len(ordering_min) > 0 else [],
        "ordering_max": ordering_max.tolist() if len(ordering_max) > 0 else [],
        "ordering_bits_per_dim": bits_per_dim,
    }

    return sort_indices, metadata


def order_lines_spatial(
    vertices: np.ndarray,
    segments: np.ndarray,
    dimensions: list[Dimension],
    method: Literal["morton", "hilbert"] = "hilbert",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply dual spatial ordering to lines using Morton or Hilbert curves.

    Lines have dual ordering:
    1. Vertices are ordered in D-space (like Points)
    2. Segments are ordered in (2×D)-space (concatenating both endpoints)

    This is the main entry point for Lines spatial ordering.

    Args:
        vertices: (V, D) float32 vertex positions
        segments: (S, 2) uint32 index pairs (from convert_to_indexed)
        dimensions: List of Dimension objects (may have more dims than data)
        method: Ordering method ("morton" or "hilbert")

    Returns:
        sorted_vertices: (V, D) float32 reordered vertex positions
        sorted_segments: (S, 2) uint32 reordered and remapped segment indices
        vertex_sort_indices: Indices to recover original vertex order
        segment_sort_indices: Indices to recover original segment order
        metadata: Dict with "vertex_ordering" and "segment_ordering" sub-dicts
    """
    _V, D = vertices.shape  # V unused but kept for reference

    # Use only the first D dimensions from the dimensions list
    # This handles cases where scene has more dimensions than data
    dimensions = dimensions[:D]

    # 1. Order vertices in D-space (with compound ordering for discrete dims)
    vertex_sort_indices, vertex_metadata = sort_points_compound(
        vertices, dimensions, method=method
    )
    sorted_vertices = vertices[vertex_sort_indices]

    # 2. Create inverse mapping for index remapping
    # After sorting, vertex at original index i is now at inverse_map[i]
    inverse_map = np.argsort(vertex_sort_indices).astype(np.uint32)
    remapped_segments = inverse_map[segments]

    # 3. Build (2×D) segment coordinates for ordering
    # Each segment becomes a 2D point: [x1, y1, z1, x2, y2, z2] for 3D
    segment_coords_2d = np.concatenate(
        [
            sorted_vertices[remapped_segments[:, 0]],  # Start points
            sorted_vertices[remapped_segments[:, 1]],  # End points
        ],
        axis=1,
    )  # Shape: (S, 2×D)

    # 4. Order segments in (2×D)-space (with compound ordering)
    segment_sort_indices, segment_metadata = sort_segments_compound(
        segment_coords_2d, dimensions, method=method
    )
    sorted_segments = remapped_segments[segment_sort_indices]

    return (
        sorted_vertices,
        sorted_segments,
        vertex_sort_indices,
        segment_sort_indices,
        {"vertex_ordering": vertex_metadata, "segment_ordering": segment_metadata},
    )


def compute_vertex_chunk_bounds(
    vertices: np.ndarray,
    chunk_size: int,
    slice_dims: Optional[list[int]] = None,
) -> np.ndarray:
    """Compute chunk bounding boxes for vertices (no radius/width expansion).

    Args:
        vertices: Vertex positions (already sorted), shape (V, D)
        chunk_size: Number of vertices per chunk
        slice_dims: Indices of discrete (non-spatial) dimensions (padded by a
            float-boundary epsilon only — the reader's per-dimension tolerance
            owns the query reach)

    Returns:
        chunk_bounds: Bounding boxes, shape (num_chunks, D, 2)
    """
    n_vertices, n_dims = vertices.shape
    num_chunks = (n_vertices + chunk_size - 1) // chunk_size
    discrete_dims = set(slice_dims) if slice_dims else set()

    chunk_bounds = np.zeros((num_chunks, n_dims, 2), dtype=np.float32)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_vertices)
        chunk_verts = vertices[start_idx:end_idx]

        for d in range(n_dims):
            if d in discrete_dims:
                # Discrete: tight bounds padded by a float-boundary epsilon only
                # (the reader's per-dimension tolerance owns the query reach).
                chunk_bounds[chunk_idx, d, 0] = (
                    chunk_verts[:, d].min() - _BARRIER_BOUND_EPS
                )
                chunk_bounds[chunk_idx, d, 1] = (
                    chunk_verts[:, d].max() + _BARRIER_BOUND_EPS
                )
            else:
                # Spatial: exact bounds (no size expansion for vertices)
                chunk_bounds[chunk_idx, d, 0] = chunk_verts[:, d].min()
                chunk_bounds[chunk_idx, d, 1] = chunk_verts[:, d].max()

    return chunk_bounds


def compute_segment_chunk_bounds(
    vertices: np.ndarray,
    segments: np.ndarray,
    widths: np.ndarray,
    chunk_size: int,
    slice_dims: Optional[list[int]] = None,
) -> np.ndarray:
    """Compute chunk bounding boxes for segments (includes line width).

    Segment bounds are in D-dimensional space (not 2×D) for view frustum
    intersection tests. Each segment's bounds include the line width extent.

    IMPORTANT: widths must be a full (V,) array. Broadcast widths should be
    expanded with np.full(V, width_value) before calling this function.

    Args:
        vertices: Vertex positions (already sorted), shape (V, D)
        segments: Segment index pairs (already sorted), shape (S, 2)
        widths: Vertex widths (already sorted), shape (V,)
        chunk_size: Number of segments per chunk
        slice_dims: Indices of discrete (non-spatial) dimensions (padded by a
            float-boundary epsilon only — the reader's per-dimension tolerance
            owns the query reach)

    Returns:
        chunk_bounds: Bounding boxes, shape (num_chunks, D, 2)
    """
    _V, D = vertices.shape
    S = segments.shape[0]
    num_chunks = (S + chunk_size - 1) // chunk_size
    discrete_dims = set(slice_dims) if slice_dims else set()

    chunk_bounds = np.zeros((num_chunks, D, 2), dtype=np.float32)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, S)
        chunk_segs = segments[start_idx:end_idx]

        # Get vertex positions and widths for this chunk's segments
        p1 = vertices[chunk_segs[:, 0]]  # Start vertex positions
        p2 = vertices[chunk_segs[:, 1]]  # End vertex positions
        w1 = widths[chunk_segs[:, 0]]  # Start vertex widths
        w2 = widths[chunk_segs[:, 1]]  # End vertex widths
        max_w = np.maximum(w1, w2)  # Conservative bound per segment

        for d in range(D):
            if d in discrete_dims:
                # Discrete: no width expansion, only a float-boundary epsilon.
                # The reader's per-dimension tolerance owns the query reach
                # (see _BARRIER_BOUND_EPS).
                chunk_bounds[chunk_idx, d, 0] = (
                    min(p1[:, d].min(), p2[:, d].min()) - _BARRIER_BOUND_EPS
                )
                chunk_bounds[chunk_idx, d, 1] = (
                    max(p1[:, d].max(), p2[:, d].max()) + _BARRIER_BOUND_EPS
                )
            else:
                # Spatial: include width extent
                chunk_bounds[chunk_idx, d, 0] = min(
                    (p1[:, d] - max_w).min(), (p2[:, d] - max_w).min()
                )
                chunk_bounds[chunk_idx, d, 1] = max(
                    (p1[:, d] + max_w).max(), (p2[:, d] + max_w).max()
                )

    return chunk_bounds
