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
from ....validation.writing import (
    POINTS_RESERVED_ATTRS,
    validate_points_channels,
    validate_render_attrs,
)
from ..bounds import compute_position_bounds
from ..context import GeometryWriteCtx
from ..dataset_writers.colors import write_colors
from ..dataset_writers.positions import write_positions
from ..dataset_writers.scalars import (
    write_bounded_scalar,
    write_radii,
    write_scalars,
)
from ..labels.image_labels import (
    write_image_labels_csr,
)
from ..labels.text_labels import write_string_channels_csr
from ..node_common import (
    apply_default_render_attrs,
    prepare_transform_attrs,
    record_forwarded_sort_order,
    validate_node_path,
    warn_if_over_element_cap,
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
    keys: Optional["Sequence[str]"] = None,
    **attrs: Any,
) -> PointsMetadata:
    """Write points data progressively to Zarr (see ``write_points`` docstring).

    Returns the node metadata; the caller records it in the metadata cache.

    Private forwarding flag ``_return_sort_order`` (opt-in): when truthy, the
    returned metadata carries a ``"sort_order"`` key holding this node's spatial
    permutation (``None`` when no spatial reordering was applied). Only
    ``write_points_multi_lod`` sets it — it needs each level's permutation to
    build the ladder's union label CSR, and the permutation is not persisted on
    disk. Opt-in so the flat path never parks a big index array in the
    compiler's metadata cache.

    ``_skip_element_cap_warning`` is private plumbing for additive ladders: the
    parent warns on the concatenated total, so per-level warnings are redundant.
    """
    # Private forwarding flag: the multi-LOD writer needs this node's spatial
    # permutation to build the ladder's union label CSR (the permutation is not
    # persisted on disk). Popped FIRST so it never reaches the attr validator or
    # .zattrs.
    return_sort_order = attrs.pop("_return_sort_order", False)
    skip_element_cap_warning = bool(attrs.pop("_skip_element_cap_warning", False))

    # Import validation functions locally to avoid circular imports
    from ....validation.base import (
        validate_colors_for_writing,
        validate_positions_for_writing,
        validate_radii_for_writing,
        validate_sharpness_for_writing,
    )

    # 0. Fail-fast pre-write gate: everything here runs BEFORE the zarr group
    # is created and before any array lands on disk, so an invalid input
    # cannot leave a partial node behind. NOTE this gate is best-effort, not
    # transactional: validators that need the store (custom colormap LUT
    # resolution) still run post-write and can leak a partial node on failure
    # (F7 residual — transactional/temp-dir writes are a separate project).
    # image_labels' LENGTH/index and its per-item TYPE dispatch — including the
    # (H,W[,3|4]) ndarray-shape check, which check_image_label_type validates
    # eagerly since ndim/shape[2] need no PIL round-trip — now run here too
    # (step 0f, below, via validate_image_labels_for_writing /
    # check_image_label_type); only normalize_image_label's actual blob
    # normalization still runs post-write — reading a str/Path file and the PIL
    # encode itself (including the Pillow-not-installed ImportError, which the
    # adders' except (ValueError, TypeError) funnel does not catch either).
    #
    # 0a. Pure attr validators + reserved writer-stamp collisions.
    validate_render_attrs(attrs, reserved_attrs=POINTS_RESERVED_ATTRS)
    # 0b. Node path: every segment must be a valid node name — an empty path
    # would resolve require_group("") to the scene ROOT and clobber it.
    path = validate_node_path(path)
    # 0c. Positions shape/finiteness.
    n_points, n_dims = validate_positions_for_writing(positions)
    # 0d-0f. Per-point channel sweep (colors, radii, sharpness, scalars, labels,
    # keys, then image_labels), shared verbatim with the pre-split gate the
    # partition / LOD wrappers run against the source count — see
    # validate_points_channels.
    validate_points_channels(
        n_points,
        colors=colors,
        radii=radii,
        sharpness=sharpness,
        scalars=scalars,
        labels=labels,
        image_labels=image_labels,
        keys=keys,
    )
    # 0g. Transform / nd_transform normalization is pure attr processing
    # (reads only the scene dimensions), so run it in the gate too — a bad
    # transform must not leave a partial node behind.
    prepare_transform_attrs(attrs, ctx.store)

    # 1. Setup: Create group
    group = ctx.store.require_group(path)

    aprint(f"📝 Writing {n_points:,} points ({n_dims}D) to {path}")

    # 2. Log scalar inputs (no expansion - passed to encoder)
    if radii is not None and isinstance(radii, (int, float)):
        aprint(f"  → Uniform radius {radii:.3f} for all points")
    if sharpness is not None and isinstance(sharpness, (int, float)):
        aprint(f"  → Uniform sharpness {sharpness:.1f} for all points")
    if colors is not None and isinstance(colors, (list, tuple)):
        aprint(f"  → Uniform color RGB(A){list(colors)} for all points")

    # 3. Apply spatial ordering if enabled (reorders arrays only)
    # Note: Spatial ordering uses radii to compute chunk_bounds.
    # Scalar/broadcasted radii are handled without expanding to full arrays.
    radii_for_ordering = radii

    ordering_data = build_points_ordering(
        positions,
        n_points,
        n_dims,
        radii_for_ordering,
        ctx.ordering_ctx,
        ctx.store,
        dataset_ctx=ctx.dataset_ctx,
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
        # Belt and braces — the fail-fast gate (step 0d) already validated
        if isinstance(colors, np.ndarray):
            validate_colors_for_writing(colors, n_points, channels=(3, 4))
        write_colors(
            group,
            colors,
            ordering_data,
            n_points,
            ctx.dataset_ctx,
            per_array_bytes=True,
        )
        metadata["has_colors"] = True

    if radii is not None:
        # Belt and braces — the fail-fast gate (step 0d) already validated
        if isinstance(radii, np.ndarray):
            validate_radii_for_writing(radii, n_points)
        max_radius = write_radii(group, radii, ordering_data, n_points, ctx.dataset_ctx)
        metadata["max_radius"] = max_radius
        metadata["has_radii"] = True
        group.attrs["max_radius"] = max_radius

    if sharpness is not None:
        # Belt and braces — the fail-fast gate (step 0d) already validated
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
            per_array_bytes=True,
        )
        metadata["has_sharpness"] = True

    if scalars is not None:
        write_scalars(
            group,
            scalars,
            ordering_data,
            n_points,
            ctx.dataset_ctx,
            per_array_bytes=True,
        )
        metadata["has_scalars"] = True

    # 5b. Write colormap LUT if colormap is a custom array
    ctx.write_colormap_lut(group, attrs)

    # 6. Transform + nd_transform attrs were already normalized in the
    # fail-fast gate (step 0f) — prepare_transform_attrs is NOT idempotent
    # (it transposes the matrix), so it must run exactly once.

    # 7. Set default rendering attributes if not provided
    apply_default_render_attrs(attrs)

    # POPPED BEFORE the attrs land, not after. `_skip_scene_bounds` is private
    # plumbing between the ladder writers and this one — "the parent aggregates
    # the bbox, do not do it per level" — and popping it below the
    # `group.attrs.update(attrs)` wrote it to disk on every sub-LOD of every
    # ladder. Harmless to a reader that ignores unknown keys, but it is an
    # internal flag in the on-disk format, and it round-trips: a tool that reads a
    # level's attrs and re-writes them hands it back as a caller attr.
    skip_scene_bounds = bool(attrs.pop("_skip_scene_bounds", False))
    # 8. Store attributes
    group.attrs.update(attrs)
    group.attrs["type"] = "points"
    group.attrs["n_points"] = n_points
    warn_if_over_element_cap(
        "points", n_points, group.name, enabled=not skip_element_cap_warning
    )
    group.attrs["ndim"] = n_dims
    # Presence flags mirror the Lines writer (has_colors/has_sharpness) so every
    # geometry type stamps the same attrs the viewer can rely on.
    group.attrs["has_colors"] = metadata["has_colors"]
    group.attrs["has_radii"] = metadata["has_radii"]
    group.attrs["has_sharpness"] = metadata["has_sharpness"]
    # Below attrs.update like the other flags: a user-supplied attrs dict
    # must never clobber the writer's presence truth.
    group.attrs["has_scalars"] = metadata.get("has_scalars", False)

    # 9. Compute and store position bounds (nD bounding box)
    # This is computed from the final positions (potentially reordered)
    position_bounds = compute_position_bounds(positions)
    group.attrs["position_bounds"] = position_bounds
    metadata["position_bounds"] = position_bounds

    # Update scene-level bounds (union of all node bounds). Skipped
    # when ``write_points_multi_lod`` is the caller — the parent
    # multi-LOD writer aggregates the global bounds once instead of
    # accumulating each subgroup's contribution separately.
    if not skip_scene_bounds:
        ctx.update_scene_bounds(position_bounds)

    # 10. Write spatial ordering metadata if built
    if ordering_data is not None:
        write_points_ordering_to_zarr(group, ordering_data, ctx.compressor)
        metadata["has_spatial_index"] = True
        # Mirrors the Lines writer: the returned metadata is what backs
        # ``Points.ordering``, so omitting it would make that property report
        # "none" for a node that is in fact spatially ordered on disk.
        metadata["ordering"] = ordering_data["ordering"]
    else:
        metadata["ordering"] = "none"

    # 10b. Forward the spatial permutation to a multi-LOD parent on request
    # (``None`` means "no spatial reordering — identity").
    record_forwarded_sort_order(
        metadata,
        return_sort_order,
        ordering_data["sort_order"] if ordering_data is not None else None,
    )

    sort_order = ordering_data["sort_order"] if ordering_data is not None else None
    write_string_channels_csr(
        group,
        labels=labels,
        keys=keys,
        n_elements=n_points,
        compressor=ctx.compressor,
        sort_order=sort_order,
        metadata=metadata,
    )

    # 12. Write image labels if provided (CSR-style, no compression on blobs)
    if image_labels is not None:
        write_image_labels_csr(
            group, image_labels, n_points, ctx.compressor, sort_order
        )
        metadata["has_image_labels"] = True

    # 13. Finish (the caller records metadata in its cache)
    aprint(f"✅ Points written to {path}")

    return metadata
