"""Points spatial-ordering glue: build Morton/Hilbert ordering + write to zarr."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import zarr
from arbol import aprint
from numpy.typing import NDArray

from ....typing_utils.protocols import CompressorProtocol
from ..context import OrderingCtx


def build_points_ordering(
    positions: NDArray[np.float32],
    n_points: int,
    n_dims: int,
    radii: Optional[Union[NDArray[np.float32], float]],
    ctx: OrderingCtx,
    store: zarr.Group,
) -> Optional[Dict[str, Any]]:
    """Apply spatial ordering using Morton/Hilbert curves.

    Args:
        positions: Point positions
        n_points: Number of points
        n_dims: Number of dimensions
        radii: Optional radii array
        ctx: Spatial-ordering configuration (enable flag + method).
        store: Root zarr group (read for ``scene_dimensions``).

    Returns:
        Dict with:
        - sorted_positions: Reordered positions
        - sort_order: Indices to apply to other arrays
        - chunk_bounds: (num_chunks, n_dims, 2) array
        - ordering_metadata: Dict from sort_points_compound
        Or None if ordering disabled/not applicable
    """
    if not ctx.enable_spatial_index or n_points == 0:
        return None

    # Get scene dimensions from attrs
    if "scene_dimensions" not in store.attrs:
        aprint("  ⚠️ No scene dimensions - skipping spatial ordering")
        return None

    from ....core.dimensions import Dimensions

    scene_dims_dict = store.attrs["scene_dimensions"]
    dimensions = Dimensions.from_dict(scene_dims_dict)

    aprint(f"  🔍 Applying {ctx.ordering_method} ordering...")

    # Apply compound ordering
    from ...ordering import compute_chunk_bounds_points, sort_points_compound

    sort_indices, ordering_metadata = sort_points_compound(
        positions,
        dimensions.dimensions,  # List of Dimension objects
        method=ctx.ordering_method,
    )

    # Reorder positions
    sorted_positions = positions[sort_indices]

    # Compute chunk size (from TARGET_CHUNK_BYTES)
    from ....typing_utils import TARGET_CHUNK_BYTES

    bytes_per_point = n_dims * 4 + 16  # Conservative estimate
    chunk_size = max(1024, TARGET_CHUNK_BYTES // bytes_per_point)
    chunk_size = min(chunk_size, n_points)

    # Compute chunk bounds
    # Handle scalar radii vs array radii vs broadcasted radii
    if radii is not None:
        if isinstance(radii, np.ndarray):
            # Check if radii are broadcasted (shape (1,) or (1, k))
            if radii.shape[0] == 1:
                # Broadcasted radii - keep scalar to avoid large allocations
                sorted_radii = float(radii.flat[0])
            else:
                # Regular array radii - apply reordering
                sorted_radii = radii[sort_indices]
        else:
            # Scalar radii - no reordering needed
            sorted_radii = float(radii)
    else:
        sorted_radii = None
    chunk_bounds = compute_chunk_bounds_points(
        sorted_positions,
        sorted_radii,
        chunk_size,
        slice_dims=ordering_metadata["slice_dims"],
    )

    aprint(
        f"  ✓ Ordering complete: {len(ordering_metadata['slice_dims'])} discrete dims, "
        f"{len(ordering_metadata['ordering_dims'])} spatial dims"
    )

    return {
        "sorted_positions": sorted_positions,
        "sort_order": sort_indices,
        "chunk_bounds": chunk_bounds,
        "chunk_size": chunk_size,
        **ordering_metadata,  # ordering, slice_dims, ordering_dims, etc.
    }


def write_points_ordering_to_zarr(
    group: zarr.Group,
    ordering_data: Dict[str, Any],
    compressor: CompressorProtocol,
) -> None:
    """Write spatial ordering metadata and chunk bounds to Zarr.

    Args:
        group: Parent Zarr group
        ordering_data: Ordering data with chunk_bounds and metadata
        compressor: Scene default compressor for the chunk_bounds array.
    """
    aprint("  📝 Writing spatial ordering metadata...")

    # Write ordering metadata directly to group attrs (simple, clean)
    ordering_metadata = {
        "ordering": ordering_data["ordering"],
        "slice_dims": ordering_data["slice_dims"],
        "ordering_dims": ordering_data["ordering_dims"],
        "ordering_min": ordering_data["ordering_min"],
        "ordering_max": ordering_data["ordering_max"],
        "ordering_bits_per_dim": ordering_data["ordering_bits_per_dim"],
        "chunk_size": ordering_data["chunk_size"],
    }
    group.attrs.update(ordering_metadata)

    # Write chunk_bounds array directly to group
    chunk_bounds = ordering_data["chunk_bounds"]
    if len(chunk_bounds) > 0:
        group.create_dataset(
            "chunk_bounds",
            data=chunk_bounds,
            shape=chunk_bounds.shape,
            dtype=np.float32,
            chunks=(chunk_bounds.shape[0], chunk_bounds.shape[1], 2),
            compressor=compressor,
        )

    aprint(
        f"  ✓ Spatial ordering written: {ordering_data['ordering']} with {len(chunk_bounds)} chunks"
    )
