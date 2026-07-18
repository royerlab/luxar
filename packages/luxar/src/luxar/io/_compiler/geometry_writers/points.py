"""Points write pipeline (body of ``LuxarZarrCompiler.write_points``).

Extracted from the orchestrator so the Points concern lives in one place. The
orchestrator method is now a thin delegate that builds a
:class:`~luxar.io._compiler.context.GeometryWriteCtx`, runs this, then writes
the returned metadata into its cache. All zarr writes are unchanged (this is a
behavior-preserving move).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from arbol import aprint
from numpy.typing import NDArray

from ....typing_utils.aliases import NodePath, PointsMetadata
from ....typing_utils.constants import SHARPNESS_MAX
from ..bounds import compute_position_bounds
from ..context import GeometryWriteCtx
from ..dataset_writers.colors import write_colors
from ..dataset_writers.positions import write_positions
from ..dataset_writers.scalars import (
    write_bounded_scalar,
    write_radii,
    write_scalars,
)
from ..labels.image_labels import write_image_labels_csr
from ..labels.text_labels import write_labels_csr
from ..node_common import (
    apply_default_render_attrs,
    prepare_transform_attrs,
    validate_render_attrs,
)
from ..spatial_ordering.points import (
    build_points_ordering,
    write_points_ordering_to_zarr,
)


def write_points(
    ctx: GeometryWriteCtx,
    path: NodePath,
    positions: NDArray[np.float32],
    colors: Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]] = None,
    radii: Optional[Union[NDArray[np.float32], float]] = None,
    sharpness: Optional[Union[NDArray[np.float32], float]] = None,
    scalars: Optional[Union[NDArray[np.float32], float]] = None,
    labels: Optional["Sequence[str]"] = None,
    image_labels: Optional[Any] = None,
    **attrs: Any,
) -> PointsMetadata:
    """Write points data progressively to Zarr (see ``write_points`` docstring).

    Returns the node metadata; the caller records it in the metadata cache.
    """
    # Import validation functions locally to avoid circular imports
    from ....validation.base import (
        validate_colors_for_writing,
        validate_positions_for_writing,
        validate_radii_for_writing,
        validate_sharpness_for_writing,
    )

    # 0. Fail fast on invalid render attrs BEFORE creating the group, so a
    # bad value cannot leave a partial node on disk.
    validate_render_attrs(attrs)

    # 1. Setup: Create group and validate positions
    path = path.lstrip("/")
    group = ctx.store.require_group(path)
    n_points, n_dims = validate_positions_for_writing(positions)

    aprint(f"📝 Writing {n_points:,} points ({n_dims}D) to {path}")

    # 2. Log scalar inputs (no expansion - passed to encoder)
    if radii is not None and isinstance(radii, (int, float)):
        aprint(f"  → Uniform radius {radii:.3f} for all points")
    if sharpness is not None and isinstance(sharpness, (int, float)):
        aprint(f"  → Uniform sharpness {sharpness:.1f} for all points")
    if colors is not None and isinstance(colors, (list, tuple)):
        aprint(f"  → Uniform color RGB{list(colors)} for all points")

    # 3. Apply spatial ordering if enabled (reorders arrays only)
    # Note: Spatial ordering uses radii to compute chunk_bounds.
    # Scalar/broadcasted radii are handled without expanding to full arrays.
    radii_for_ordering = radii

    ordering_data = build_points_ordering(
        positions, n_points, n_dims, radii_for_ordering, ctx.ordering_ctx, ctx.store
    )

    # Apply spatial reordering to arrays only (skip scalars and broadcasted arrays)
    if ordering_data is not None:
        positions = ordering_data["sorted_positions"]
        # Apply sort order only to non-broadcasted array attributes
        # Broadcasted arrays (shape[0] == 1) should NOT be reordered
        if colors is not None and isinstance(colors, np.ndarray):
            if colors.shape[0] > 1:  # Not broadcasted
                colors = colors[ordering_data["sort_order"]]
            # else: broadcasted, skip reordering
        if radii is not None and isinstance(radii, np.ndarray):
            if radii.shape[0] > 1:  # Not broadcasted
                radii = radii[ordering_data["sort_order"]]
            # else: broadcasted, skip reordering
        if sharpness is not None and isinstance(sharpness, np.ndarray):
            if sharpness.shape[0] > 1:  # Not broadcasted
                sharpness = sharpness[ordering_data["sort_order"]]
            # else: broadcasted, skip reordering
        if scalars is not None and isinstance(scalars, np.ndarray):
            if scalars.shape[0] > 1:  # Not broadcasted
                scalars = scalars[ordering_data["sort_order"]]
            # else: broadcasted, skip reordering

    # 3. Write positions dataset
    write_positions(group, positions, ordering_data, ctx.dataset_ctx)

    # 4. Initialize metadata
    metadata: PointsMetadata = {
        "n_points": n_points,
        "ndim": n_dims,
        "path": path,
        "has_colors": False,
        "has_radii": False,
        "has_sharpness": False,
    }

    # 5. Write optional datasets
    if colors is not None:
        # Validate arrays only (scalars validated by encoder)
        if isinstance(colors, np.ndarray):
            validate_colors_for_writing(colors, n_points)
        write_colors(group, colors, ordering_data, n_points, ctx.dataset_ctx)
        metadata["has_colors"] = True

    if radii is not None:
        # Validate arrays only (scalars validated by encoder)
        if isinstance(radii, np.ndarray):
            validate_radii_for_writing(radii, n_points)
        max_radius = write_radii(group, radii, ordering_data, n_points, ctx.dataset_ctx)
        metadata["max_radius"] = max_radius
        metadata["has_radii"] = True
        group.attrs["max_radius"] = max_radius

    if sharpness is not None:
        # Validate arrays only (scalars validated by encoder)
        if isinstance(sharpness, np.ndarray):
            validate_sharpness_for_writing(sharpness, n_points)
        # Canonical BOUNDED_SCALAR helper (shared with Lines
        # "sharpnesses"). Sharpness is a normalized [0, 1] knob, so it
        # carries no `max_sharpness` — the bounds tuple is fixed at
        # (0.0, SHARPNESS_MAX) and the decoder reads it from the encoding
        # metadata (mirrors the radii-without-max pattern).
        write_bounded_scalar(
            group,
            sharpness,
            "sharpnesses",
            (0.0, SHARPNESS_MAX),
            ordering_data,
            n_points,
            ctx.dataset_ctx,
            "sharpness",
        )
        metadata["has_sharpness"] = True

    if scalars is not None:
        write_scalars(group, scalars, ordering_data, n_points, ctx.dataset_ctx)
        metadata["has_scalars"] = True
        group.attrs["has_scalars"] = True

    # 5b. Write colormap LUT if colormap is a custom array
    ctx.write_colormap_lut(group, attrs)

    # 6. Process transform + nd_transform attrs (shared with write_lines)
    prepare_transform_attrs(attrs, ctx.store)

    # 7. Set default rendering attributes if not provided
    apply_default_render_attrs(attrs)

    # 8. Store attributes
    group.attrs.update(attrs)
    group.attrs["type"] = "points"
    group.attrs["n_points"] = n_points

    # 9. Compute and store position bounds (nD bounding box)
    # This is computed from the final positions (potentially reordered)
    position_bounds = compute_position_bounds(positions)
    group.attrs["position_bounds"] = position_bounds
    metadata["position_bounds"] = position_bounds

    # Update scene-level bounds (union of all node bounds). Skipped
    # when ``write_points_multi_lod`` is the caller — the parent
    # multi-LOD writer aggregates the global bounds once instead of
    # accumulating each subgroup's contribution separately.
    if not attrs.pop("_skip_scene_bounds", False):
        ctx.update_scene_bounds(position_bounds)

    # 10. Write spatial ordering metadata if built
    if ordering_data is not None:
        write_points_ordering_to_zarr(group, ordering_data, ctx.compressor)
        metadata["has_spatial_index"] = True

    # 11. Write labels if provided (CSR-style: label_offsets + label_bytes)
    if labels is not None:
        sort_order = ordering_data["sort_order"] if ordering_data is not None else None
        write_labels_csr(group, labels, n_points, ctx.compressor, sort_order)
        metadata["has_labels"] = True

    # 12. Write image labels if provided (CSR-style, no compression on blobs)
    if image_labels is not None:
        sort_order = ordering_data["sort_order"] if ordering_data is not None else None
        write_image_labels_csr(
            group, image_labels, n_points, ctx.compressor, sort_order
        )
        metadata["has_image_labels"] = True

    # 13. Finish (the caller records metadata in its cache)
    aprint(f"✅ Points written to {path}")

    return metadata
