"""Lines dual spatial-ordering glue: build vertex+segment ordering + write to zarr."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
import zarr
from arbol import aprint
from numpy.typing import NDArray

from luxar._zarr_compat import create_array

from ....encoding.compression import resolve_compressor
from ....typing_utils.aliases import (
    PositionArray,
    ScalarArray,
)
from ..context import DatasetCtx, OrderingCtx

if TYPE_CHECKING:
    from ....encoding.compression import CompressorLike


def build_lines_ordering(
    vertices: PositionArray,
    segments: NDArray[np.uint32],
    widths: Union[ScalarArray, float],
    n_vertices: int,
    n_dims: int,
    n_segments: int,
    ctx: OrderingCtx,
    store: zarr.Group,
    *,
    dataset_ctx: Optional[DatasetCtx] = None,
) -> Optional[Dict[str, Any]]:
    """Build dual spatial ordering for Lines (vertices + segments).

    Args:
        vertices: Vertex positions
        segments: Segment index pairs
        widths: Vertex widths (array or scalar)
        n_vertices: Number of vertices
        n_dims: Number of dimensions
        n_segments: Number of segments
        ctx: Spatial-ordering configuration (enable flag + method).
        store: Root zarr group (read for ``scene_dimensions``).
        dataset_ctx: Encoder configuration, used to ask how far the store will
            move a vertex (see
            :meth:`~luxar.encoding.encoder.ArrayEncoder.coordinate_round_trip_slack`)
            and enlarge a width (see
            :meth:`~luxar.encoding.encoder.ArrayEncoder.positive_scalar_round_trip_slack`)
            so spatial segment bounds contain the DECODED footprint, not just
            the authored one. ``None`` (a direct caller) ⇒ authored bounds.

    Returns:
        Dict with sorted arrays, sort indices, chunk bounds, and ordering metadata.
        Or None if spatial ordering is disabled.
    """
    if not ctx.enable_spatial_index or n_vertices == 0:
        return None

    # Get scene dimensions from attrs
    if "scene_dimensions" not in store.attrs:
        aprint("  ⚠️ No scene dimensions - skipping spatial ordering")
        return None

    from ....core.dimensions import Dimensions

    scene_dims_dict = store.attrs["scene_dimensions"]
    dimensions = Dimensions.from_dict(scene_dims_dict)

    aprint(f"  🔍 Applying dual {ctx.ordering_method} ordering...")

    # Import ordering functions
    from ...ordering import (
        compute_segment_chunk_bounds,
        compute_vertex_chunk_bounds,
        order_lines_spatial,
    )

    # Apply dual spatial ordering
    (
        sorted_vertices,
        sorted_segments,
        vertex_sort_indices,
        segment_sort_indices,
        ordering_metadata,
    ) = order_lines_spatial(
        vertices,
        segments,
        dimensions.dimensions,
        method=ctx.ordering_method,
    )

    # Compute chunk sizes (from TARGET_CHUNK_BYTES)
    from ....typing_utils import TARGET_CHUNK_BYTES

    # Vertex chunk size
    bytes_per_vertex = n_dims * 4 + 8  # Position + width + overhead
    vertex_chunk_size = max(1024, TARGET_CHUNK_BYTES // bytes_per_vertex)
    vertex_chunk_size = min(vertex_chunk_size, n_vertices)

    # Segment chunk size
    bytes_per_segment = 8 + 8  # 2 uint32 indices + overhead
    segment_chunk_size = max(1024, TARGET_CHUNK_BYTES // bytes_per_segment)
    segment_chunk_size = min(segment_chunk_size, n_segments)

    # Add chunk_size to metadata
    ordering_metadata["vertex_ordering"]["chunk_size"] = vertex_chunk_size
    ordering_metadata["segment_ordering"]["chunk_size"] = segment_chunk_size

    # How far can the encoder move a vertex from what it is handed? Both bound
    # sets are in D-space over this same array — the very one the writer later
    # hands the encoder — so they share one slack vector (issue #1655).
    #
    # `allow_lut=False` MUST match what `write_lines_arrays` passes when it
    # encodes `vertices` (geometry_writers/lines.py): the lines spatial-index
    # loader reads that array as raw chunked zarr, so a LUT is blocked there.
    # A LUT-eligible vertices array is therefore quantized like any other, and
    # asking with the default would report "exact" and leave both bound sets
    # unpadded around coordinates the store moved.
    coord_slack = (
        dataset_ctx.encoder.coordinate_round_trip_slack(
            sorted_vertices, dataset_ctx.encoding_mode, allow_lut=False
        )
        if dataset_ctx is not None
        else None
    )

    # Compute vertex chunk bounds
    vertex_chunk_bounds = compute_vertex_chunk_bounds(
        sorted_vertices,
        vertex_chunk_size,
        slice_dims=ordering_metadata["vertex_ordering"]["slice_dims"],
        coord_slack=coord_slack,
    )

    # Compute segment chunk bounds
    # Need to expand widths if scalar or broadcasted
    # These branches deliberately use float32: the bounds then start from the
    # same value the viewer decodes, even though the writer stores authored widths.
    if isinstance(widths, (int, float)):
        widths_expanded = np.full(n_vertices, float(widths), dtype=np.float32)
    elif isinstance(widths, np.ndarray) and widths.shape[0] == 1:
        # Broadcasted
        widths_expanded = np.full(n_vertices, widths[0], dtype=np.float32)
    else:
        widths_expanded = widths[vertex_sort_indices]  # Apply same reordering

    scalar_slack = (
        dataset_ctx.encoder.positive_scalar_round_trip_slack(
            widths_expanded,
            dataset_ctx.encoding_mode,
            # Must match write_positive_scalar's default used below.
            positive_scalar_encoding="linear",
        )
        if dataset_ctx is not None
        else None
    )

    segment_chunk_bounds = compute_segment_chunk_bounds(
        sorted_vertices,
        sorted_segments,
        widths_expanded,
        segment_chunk_size,
        slice_dims=ordering_metadata["vertex_ordering"][
            "slice_dims"
        ],  # Use D-space dims
        coord_slack=coord_slack,
        scalar_slack=scalar_slack,
    )

    aprint(
        f"  ✓ Dual ordering complete: {len(vertex_chunk_bounds)} vertex chunks, "
        f"{len(segment_chunk_bounds)} segment chunks"
    )

    return {
        "sorted_vertices": sorted_vertices,
        "sorted_segments": sorted_segments,
        "vertex_sort_indices": vertex_sort_indices,
        "segment_sort_indices": segment_sort_indices,
        "vertex_chunk_bounds": vertex_chunk_bounds,
        "segment_chunk_bounds": segment_chunk_bounds,
        "ordering": ctx.ordering_method,
        **ordering_metadata,
    }


def write_lines_ordering_to_zarr(
    group: zarr.Group,
    ordering_data: Dict[str, Any],
    compressor: "CompressorLike",
) -> None:
    """Write Lines spatial ordering metadata and dual chunk bounds to Zarr.

    Args:
        group: Parent Zarr group
        ordering_data: Ordering data with chunk bounds and metadata
        compressor: Scene default compressor for the chunk-bounds arrays.
    """
    aprint("  📝 Writing Lines spatial ordering metadata...")

    # Write vertex_chunk_bounds
    vertex_chunk_bounds = ordering_data["vertex_chunk_bounds"]
    if len(vertex_chunk_bounds) > 0:
        n_dims = vertex_chunk_bounds.shape[1]
        create_array(
            group,
            "vertex_chunk_bounds",
            data=vertex_chunk_bounds,
            shape=vertex_chunk_bounds.shape,
            dtype=np.float32,
            chunks=(vertex_chunk_bounds.shape[0], n_dims, 2),
            compressor=resolve_compressor(compressor, np.float32),
        )

    # Write segment_chunk_bounds
    segment_chunk_bounds = ordering_data["segment_chunk_bounds"]
    if len(segment_chunk_bounds) > 0:
        n_dims = segment_chunk_bounds.shape[1]
        create_array(
            group,
            "segment_chunk_bounds",
            data=segment_chunk_bounds,
            shape=segment_chunk_bounds.shape,
            dtype=np.float32,
            chunks=(segment_chunk_bounds.shape[0], n_dims, 2),
            compressor=resolve_compressor(compressor, np.float32),
        )

    aprint(
        f"  ✓ Dual spatial ordering written: {len(vertex_chunk_bounds)} vertex chunks, "
        f"{len(segment_chunk_bounds)} segment chunks"
    )
