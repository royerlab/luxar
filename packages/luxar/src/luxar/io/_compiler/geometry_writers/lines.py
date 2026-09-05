"""Lines write pipeline (body of ``LuxarZarrCompiler.write_lines``).

Extracted from the orchestrator. Lines are dual-indexed (vertices in D-space,
segments in 2xD-space); the raw ``vertices``/``segments`` arrays are encoded
directly through the shared :class:`~luxar.io._compiler.context.GeometryWriteCtx`
(same ctx as Points — its ``dataset_ctx`` carries the encoder/mode/compressor).
Behavior-preserving.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from arbol import aprint
from numpy.typing import NDArray

from ....encoding import SemanticType
from ....typing_utils.aliases import NodePath
from ....typing_utils.constants import SHARPNESS_MAX
from ....validation.writing import (
    LINES_RESERVED_ATTRS,
    validate_line_indices,
    validate_lines_channels,
    validate_render_attrs,
)
from ...ordering import convert_to_indexed
from ..bounds import compute_position_bounds
from ..chunking import calculate_intelligent_chunks
from ..context import GeometryWriteCtx
from ..dataset_writers.colors import write_colors
from ..dataset_writers.scalars import (
    write_bounded_scalar,
    write_positive_scalar,
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
from ..spatial_ordering.lines import build_lines_ordering, write_lines_ordering_to_zarr

_AUTHORING_LINT_MIN_VERTICES = 16
_AUTHORING_LINT_SHARED_THRESHOLD = 0.9


def _exploded_chain_fraction(vertices: NDArray[np.float32]) -> Optional[float]:
    """Return the forward-chain adjacency fraction when it strongly signals intent.

    Immediate ``(a, b), (b, a)`` reversals are excluded: they are common in
    directed/symmetric graph edge lists and are not evidence of a polyline.
    """
    if len(vertices) < _AUTHORING_LINT_MIN_VERTICES:
        return None
    vertices_array = np.asarray(vertices)
    shared = np.all(vertices_array[1:-1:2] == vertices_array[2::2], axis=1)
    reversals = np.all(vertices_array[0:-2:2] == vertices_array[3::2], axis=1)
    forward_shared = shared & ~reversals
    if forward_shared.size == 0:
        return None
    shared_fraction = float(np.mean(forward_shared))
    if shared_fraction <= _AUTHORING_LINT_SHARED_THRESHOLD:
        return None
    return shared_fraction


def _line_authoring_warning_key(ctx: GeometryWriteCtx, path: str) -> str:
    """Collapse partition leaves to their logical parent warning key."""
    parent_path, separator, _leaf = path.rpartition("/")
    if separator and parent_path in ctx.store:
        parent = ctx.store[parent_path]
        if getattr(parent, "attrs", {}).get("kind") == "partition":
            return parent_path
    return path


def write_lines(
    ctx: GeometryWriteCtx,
    path: NodePath,
    vertices: NDArray[np.float32],
    widths: Union[NDArray[np.float32], float],
    colors: Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]] = None,
    sharpness: Optional[Union[NDArray[np.float32], float]] = None,
    scalars: Optional[Union[NDArray[np.float32], float]] = None,
    indices: Optional[NDArray[np.uint32]] = None,
    line_type: str = "polyline",
    labels: Optional["Sequence[str]"] = None,
    image_labels: Optional[Any] = None,
    keys: Optional["Sequence[str]"] = None,
    **attrs: Any,
) -> dict[str, Any]:
    """Write lines data to Zarr with dual spatial indexing (see ``write_lines``).

    Returns the node metadata; the caller records it in the metadata cache.

    Private forwarding flag ``_return_sort_order`` (opt-in): when truthy, the
    returned metadata carries a ``"sort_order"`` key holding this node's
    per-VERTEX spatial permutation (``ordering_data["vertex_sort_indices"]``, or
    ``None`` when no spatial reordering was applied). Only
    ``write_lines_multi_lod`` sets it — it needs each level's permutation to
    build the ladder's union label CSR, and the permutation is not persisted on
    disk. The key name matches the Points writer's so both multi-LOD writers
    read one key. Opt-in so the flat path never parks a big index array in the
    compiler's metadata cache.

    ``_skip_element_cap_warning`` is private plumbing for additive ladders: the
    parent warns on the concatenated total, so per-level warnings are redundant.
    """
    # Private forwarding flag: the multi-LOD writer needs this node's per-vertex
    # spatial permutation to build the ladder's union label CSR (the permutation
    # is not persisted on disk). Popped FIRST so it never reaches the attr
    # validator or .zattrs.
    return_sort_order = attrs.pop("_return_sort_order", False)
    skip_element_cap_warning = bool(attrs.pop("_skip_element_cap_warning", False))

    from ....validation.base import (
        validate_colors_for_writing,
        validate_positions_for_writing,
    )

    # 0. Fail-fast pre-write gate: everything here runs BEFORE the zarr group
    # is created and before any array lands on disk, so an invalid input
    # cannot leave a partial node behind. NOTE this gate is best-effort, not
    # transactional: validators that need the store (custom colormap LUT
    # resolution) still run post-write and can leak a partial node on failure
    # (F7 residual — transactional/temp-dir writes are a separate project).
    # image_labels' LENGTH/index and its per-item TYPE dispatch — including the
    # (H,W[,3|4]) ndarray-shape check, which check_image_label_type validates
    # eagerly since ndim/shape[2] need no PIL round-trip — now run here
    # too (step 0h, below, via validate_image_labels_for_writing /
    # check_image_label_type); only normalize_image_label's actual blob
    # normalization still runs post-write — reading a str/Path file and the PIL
    # encode itself (including the Pillow-not-installed ImportError, which
    # the adder's except (ValueError, TypeError) funnel does not catch
    # either).
    #
    # 0a. Pure attr validators + reserved writer-stamp collisions.
    validate_render_attrs(attrs, reserved_attrs=LINES_RESERVED_ATTRS)
    # 0b. Node path: every segment must be a valid node name — an empty path
    # would resolve require_group("") to the scene ROOT and clobber it.
    path = validate_node_path(path)
    # 0c. Vertices shape/finiteness.
    n_vertices, n_dims = validate_positions_for_writing(vertices)

    # 0d. Validate line type
    valid_line_types = ("segments", "polyline", "loop", "indexed")
    if line_type not in valid_line_types:
        raise ValueError(
            f"Invalid line_type '{line_type}'. Must be one of {valid_line_types}"
        )

    # Validate type-specific requirements
    if line_type == "segments" and n_vertices % 2 != 0:
        raise ValueError(f"Segments require even number of vertices, got {n_vertices}")
    if line_type == "polyline" and n_vertices < 2:
        raise ValueError(f"Polyline requires at least 2 vertices, got {n_vertices}")
    if line_type == "loop" and n_vertices < 3:
        raise ValueError(f"Loop requires at least 3 vertices, got {n_vertices}")
    if line_type == "indexed":
        if indices is None:
            raise ValueError("Indexed line type requires indices array")
        # Layout / parity / dtype / bounds, shared verbatim with the split
        # paths' pre-split gate — see validate_line_indices.
        indices = validate_line_indices(indices, n_vertices)

    # 0d-bis. Prepare the warn-only authoring lint, but print it after the node
    # header so multi-node output identifies the affected path. Requiring >90%
    # forward adjacency and excluding immediate reversals avoids treating common
    # DFS/BFS, wireframe, and symmetric graph-edge orderings as polylines.
    authoring_warning_fraction = (
        _exploded_chain_fraction(vertices) if line_type == "segments" else None
    )

    # 0e-0h. Per-vertex channel sweep (widths first, then colors, sharpness,
    # scalars, labels, image_labels, keys), shared verbatim with the pre-split gate
    # the partition / LOD wrappers run against the source count — see
    # validate_lines_channels.
    validate_lines_channels(
        n_vertices,
        widths=widths,
        colors=colors,
        sharpness=sharpness,
        scalars=scalars,
        labels=labels,
        image_labels=image_labels,
        keys=keys,
    )
    # 0i. Transform / nd_transform normalization is pure attr processing
    # (reads only the scene dimensions), so run it in the gate too — a bad
    # transform must not leave a partial node behind.
    prepare_transform_attrs(attrs, ctx.store)

    # 1. Setup: Create group
    group = ctx.store.require_group(path)

    aprint(f"📝 Writing {n_vertices:,} line vertices ({n_dims}D) to {path}")

    if authoring_warning_fraction is not None:
        warning_key = _line_authoring_warning_key(ctx, path)
        if ctx.claim_authoring_warning("lines", warning_key):
            aprint(
                f"  ⚠️ Node '{warning_key}': {authoring_warning_fraction:.0%} "
                "of consecutive segments share a forward endpoint coordinate "
                "— this looks like continuous polylines exploded into "
                "independent segments. Authored this way, interior joints do "
                "not share vertex indices, so thick lines render as bead "
                "chains. Use line_type='polyline' for one chain or "
                "line_type='indexed' with shared vertex indices for multiple "
                "chains."
            )

    # Scalars are now passed directly to encoder - no expansion needed
    # Just log what we're receiving
    if isinstance(widths, (int, float)):
        aprint(f"  → Uniform width {widths:.3f} for all vertices")
    if sharpness is not None and isinstance(sharpness, (int, float)):
        aprint(f"  → Uniform sharpness {sharpness:.1f} for all vertices")
    if colors is not None and isinstance(colors, (list, tuple)):
        aprint(f"  → Uniform color RGB(A){list(colors)} for all vertices")

    # Convert line type to indexed representation (unified internal format)
    segments = convert_to_indexed(n_vertices, line_type, indices)
    n_segments = segments.shape[0]

    aprint(f"  → Converted {line_type} to {n_segments:,} indexed segments")

    # Get max_width before spatial ordering
    if isinstance(widths, (int, float)):
        max_width = float(widths)
    else:
        max_width = float(np.max(widths))

    # Apply dual spatial ordering if enabled
    ordering_data = build_lines_ordering(
        vertices,
        segments,
        widths,
        n_vertices,
        n_dims,
        n_segments,
        ctx.ordering_ctx,
        ctx.store,
        dataset_ctx=ctx.dataset_ctx,
    )

    # Apply reordering if spatial ordering was applied
    if ordering_data is not None:
        vertices = ordering_data["sorted_vertices"]
        segments = ordering_data["sorted_segments"]
        vertex_sort_order = ordering_data["vertex_sort_indices"]

        # Reorder per-vertex arrays (skip scalars and broadcasted)
        if isinstance(widths, np.ndarray) and widths.shape[0] > 1:
            widths = widths[vertex_sort_order]
        if (
            colors is not None
            and isinstance(colors, np.ndarray)
            and colors.shape[0] > 1
        ):
            colors = colors[vertex_sort_order]
        if (
            sharpness is not None
            and isinstance(sharpness, np.ndarray)
            and sharpness.shape[0] > 1
        ):
            sharpness = sharpness[vertex_sort_order]
        if (
            scalars is not None
            and isinstance(scalars, np.ndarray)
            and scalars.shape[0] > 1
        ):
            scalars = scalars[vertex_sort_order]

    # Write vertices using ArrayEncoder (COORDINATE)
    chunks_2d = calculate_intelligent_chunks(
        (n_vertices, n_dims),
        spatial_index_data=ordering_data.get("vertex_ordering")
        if ordering_data
        else None,
        dtype=vertices.dtype,
        per_array_bytes=True,
    )
    ctx.dataset_ctx.encoder.encode(
        data=vertices,
        zarr_group=group,
        name="vertices",
        semantic_type=SemanticType.COORDINATE,
        mode=ctx.dataset_ctx.encoding_mode,
        chunks=chunks_2d,
        compressor=ctx.dataset_ctx.compressor,
        # The lines spatial-index loader reads vertices/segments as raw
        # chunked zarr and does not resolve array_ref, so dedup of these
        # structural arrays would silently drop geometry for a byte-
        # identical sibling (e.g. two identical components in a partition).
        # LUT is blocked for the same raw-read reason: grid-snapped
        # vertices (few unique coordinate values) would store as
        # lut_uint8/16 indices and decode as garbage geometry.
        deduplicate=False,
        allow_lut=False,
    )

    # Write segments array (always, not just for indexed type). Its atom is the
    # SEGMENT ordering grid, not the vertex one — the grid the viewer resolves
    # matched segment partitions to row ranges against (chunk-index-loader.ts) —
    # but the sizing rule is the same as every other array's: this array's own
    # dtype byte budget, rounded down to a whole multiple of that atom. One atom
    # alone is half the byte target (a 4,096-segment atom of uint32 pairs is
    # 32 KB against 64 KB), which cost the segments array twice the requests it
    # needs.
    chunks_segments = calculate_intelligent_chunks(
        segments.shape,
        spatial_index_data=ordering_data["segment_ordering"] if ordering_data else None,
        dtype=segments.dtype,
        per_array_bytes=True,
    )
    ctx.dataset_ctx.encoder.encode(
        data=segments,
        zarr_group=group,
        name="segments",
        semantic_type=SemanticType.INDEX,
        mode=ctx.dataset_ctx.encoding_mode,
        chunks=chunks_segments,
        compressor=ctx.dataset_ctx.compressor,
        deduplicate=False,  # see vertices note above
    )
    aprint(f"  ✓ Wrote segments ({n_segments:,} pairs)")

    # Write widths via the canonical POSITIVE_SCALAR helper (shared
    # with Points "radii" and GSplats "amplitudes"). The helper
    # picks the same default precision for every geometry.
    write_positive_scalar(
        group,
        widths,
        "widths",
        ordering_data.get("vertex_ordering") if ordering_data else None,
        n_vertices,
        ctx.dataset_ctx,
        "width",
        per_array_bytes=True,
        # The chunk-bound slack assumes widths cannot reuse another array's encoding.
        deduplicate=False,
    )

    # Initialize metadata
    metadata: dict[str, Any] = {
        "n_vertices": n_vertices,
        "n_segments": n_segments,
        "ndim": n_dims,
        "original_line_type": line_type,  # Store original user-specified type
        "has_colors": False,
        "has_sharpness": False,
        "max_width": max_width,
    }

    # Write optional datasets
    if colors is not None:
        if isinstance(colors, np.ndarray):
            validate_colors_for_writing(colors, n_vertices, channels=(3, 4))
        # Use the canonical COLOR helper (shared with Points / GSplats /
        # Mesh) so the default-precision and color_mode-detection logic is
        # symmetric across all four geometry types.
        write_colors(
            group,
            colors,
            ordering_data.get("vertex_ordering") if ordering_data else None,
            n_vertices,
            ctx.dataset_ctx,
            per_array_bytes=True,
        )
        metadata["has_colors"] = True

    if sharpness is not None:
        from ....validation.base import validate_sharpness_for_writing

        if isinstance(sharpness, np.ndarray):
            validate_sharpness_for_writing(sharpness, n_vertices)
        # Canonical BOUNDED_SCALAR helper (shared with Points
        # "sharpnesses"). Same bounds tuple as Points so the
        # encoder's Uint8-quantization step produces matching disk
        # layouts.
        write_bounded_scalar(
            group,
            sharpness,
            "sharpnesses",
            (0.0, SHARPNESS_MAX),
            ordering_data.get("vertex_ordering") if ordering_data else None,
            n_vertices,
            ctx.dataset_ctx,
            "sharpness",
            per_array_bytes=True,
        )
        metadata["has_sharpness"] = True

    if scalars is not None:
        # Lines' ordering_data is NESTED (vertex_ordering / segment_ordering),
        # unlike Points' flat dict, so the vertex sub-dict must be unwrapped the
        # way every sibling array above does it. Passing the outer dict meant the
        # chunk calculator found no `chunk_size` and fell back to a pure byte
        # budget, leaving `scalars` the one per-vertex array NOT on the atom grid
        # (e.g. 16384 rows against a 3276 atom — 16384 % 3276 == 4).
        write_scalars(
            group,
            scalars,
            ordering_data.get("vertex_ordering") if ordering_data else None,
            n_vertices,
            ctx.dataset_ctx,
            per_array_bytes=True,
        )
        metadata["has_scalars"] = True

    # Write colormap LUT if colormap is a custom array
    ctx.write_colormap_lut(group, attrs)

    # Write spatial ordering data (chunk bounds and metadata)
    if ordering_data is not None:
        write_lines_ordering_to_zarr(group, ordering_data, ctx.compressor)
        metadata["has_spatial_index"] = True
        metadata["ordering"] = ordering_data["ordering"]
        metadata["vertex_ordering"] = ordering_data["vertex_ordering"]
        metadata["segment_ordering"] = ordering_data["segment_ordering"]
    else:
        metadata["ordering"] = "none"

    # Forward the per-vertex spatial permutation to a multi-LOD parent on
    # request (``None`` means "no spatial reordering — identity").
    record_forwarded_sort_order(
        metadata,
        return_sort_order,
        ordering_data["vertex_sort_indices"] if ordering_data is not None else None,
    )

    # Transform + nd_transform attrs were already normalized in the fail-fast
    # gate (step 0h) — prepare_transform_attrs is NOT idempotent (it
    # transposes the matrix), so it must run exactly once.

    # Set default rendering attributes if not provided
    # (must match write_points/write_gsplats)
    apply_default_render_attrs(attrs)

    # POPPED BEFORE the attrs land, not after. `_skip_scene_bounds` is private
    # plumbing between the ladder writers and this one — "the parent aggregates
    # the bbox, do not do it per level" — and popping it below the
    # `group.attrs.update(attrs)` wrote it to disk on every sub-LOD of every
    # ladder. Harmless to a reader that ignores unknown keys, but it is an
    # internal flag in the on-disk format, and it round-trips: a tool that reads a
    # level's attrs and re-writes them hands it back as a caller attr.
    skip_scene_bounds = bool(attrs.pop("_skip_scene_bounds", False))
    # Set attributes (all core metadata per spec Section 6.6)
    group.attrs.update(attrs)
    group.attrs["type"] = "lines"
    group.attrs["n_vertices"] = n_vertices
    group.attrs["n_segments"] = n_segments
    warn_if_over_element_cap(
        "lines", n_segments, group.name, enabled=not skip_element_cap_warning
    )
    group.attrs["ndim"] = n_dims
    group.attrs["original_line_type"] = line_type
    group.attrs["has_colors"] = metadata["has_colors"]
    group.attrs["has_sharpness"] = metadata["has_sharpness"]
    # Below attrs.update like the other flags: a user-supplied attrs dict
    # must never clobber the writer's presence truth.
    group.attrs["has_scalars"] = metadata.get("has_scalars", False)
    group.attrs["max_width"] = max_width

    # Add ordering metadata to attrs if present
    if ordering_data is not None:
        group.attrs["ordering"] = ordering_data["ordering"]
        group.attrs["vertex_ordering"] = ordering_data["vertex_ordering"]
        group.attrs["segment_ordering"] = ordering_data["segment_ordering"]
    else:
        group.attrs["ordering"] = "none"

    # Compute and store position bounds (nD bounding box) for dynamic clipping
    position_bounds = compute_position_bounds(vertices)
    group.attrs["position_bounds"] = position_bounds
    metadata["position_bounds"] = position_bounds

    # Update scene-level bounds (union of all node bounds). Skipped
    # when ``write_lines_multi_lod`` is the caller — the parent
    # writer aggregates global bounds once.
    if not skip_scene_bounds:
        ctx.update_scene_bounds(position_bounds)

    sort_order = (
        ordering_data["vertex_sort_indices"] if ordering_data is not None else None
    )
    write_string_channels_csr(
        group,
        labels=labels,
        keys=keys,
        n_elements=n_vertices,
        compressor=ctx.compressor,
        sort_order=sort_order,
        metadata=metadata,
    )

    # Write image labels if provided (CSR-style, no compression on blobs)
    if image_labels is not None:
        write_image_labels_csr(
            group, image_labels, n_vertices, ctx.compressor, sort_order
        )
        metadata["has_image_labels"] = True

    aprint(f"✅ Lines written to {path}")

    return metadata
