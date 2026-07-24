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
from ..labels.image_labels import write_image_labels_csr
from ..labels.text_labels import write_labels_csr
from ..node_common import (
    LINES_RESERVED_ATTRS,
    apply_default_render_attrs,
    prepare_transform_attrs,
    validate_broadcast_color,
    validate_node_path,
    validate_render_attrs,
    validate_scalars_preflight,
)
from ..spatial_ordering.lines import build_lines_ordering, write_lines_ordering_to_zarr


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
    **attrs: Any,
) -> dict[str, Any]:
    """Write lines data to Zarr with dual spatial indexing (see ``write_lines``).

    Returns the node metadata; the caller records it in the metadata cache.
    """
    from ....validation.base import (
        validate_colors_for_writing,
        validate_labels_for_writing,
        validate_positions_for_writing,
        validate_sharpness_for_writing,
        validate_widths_for_writing,
    )

    # 0. Fail-fast pre-write gate: everything here runs BEFORE the zarr group
    # is created and before any array lands on disk, so an invalid input
    # cannot leave a partial node behind. NOTE this gate is best-effort, not
    # transactional: validators that need the store (image_labels, custom
    # colormap LUT resolution) still run post-write and can leak a partial
    # node on failure (F7 residual — transactional/temp-dir writes are a
    # separate project).
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
        if len(indices) < 2:
            raise ValueError("Indexed requires at least 2 indices")
        if len(indices) % 2 != 0:
            raise ValueError("Indices must have even length (pairs)")
        # Bounds check BOTH ends before convert_to_indexed casts to uint32:
        # a negative index would silently wrap to ~4 billion and blow up
        # with a raw IndexError deep inside the spatial ordering.
        if np.min(indices) < 0:
            raise ValueError(f"Index {np.min(indices)} < 0 (indices must be >= 0)")
        if np.max(indices) >= n_vertices:
            raise ValueError(f"Index {np.max(indices)} >= n_vertices {n_vertices}")

    # 0e. Shared validator (the Lines sibling of validate_radii_for_writing)
    validate_widths_for_writing(widths, n_vertices)

    # 0f. Pre-flight length sweep over ALL provided per-vertex arrays. The
    # spatial-ordering fancy-indexing below silently TRUNCATES a too-long
    # array and raises a raw IndexError on a too-short one, so lengths must
    # be checked before build_lines_ordering runs. The per-dataset validators
    # further down remain in place (belt and braces).
    if colors is not None:
        if isinstance(colors, np.ndarray):
            # channels=(3, 4): lines accept RGBA since volumetric phase 4
            # (the alpha column is per-vertex opacity) — mirrors points.
            validate_colors_for_writing(colors, n_vertices, channels=(3, 4))
        elif isinstance(colors, (list, tuple)):
            validate_broadcast_color(colors, "colors")
    if sharpness is not None:
        # Validates arrays AND broadcast scalars (same [0, 1] bounds).
        validate_sharpness_for_writing(sharpness, n_vertices)
    if scalars is not None:
        validate_scalars_preflight(scalars, n_vertices)
    # 0g. Labels: sequence-of-str type + length check (the CSR serializer
    # would otherwise AttributeError on a non-str entry AFTER the arrays
    # were written).
    if labels is not None:
        validate_labels_for_writing(labels, n_vertices)
    # 0h. Transform / nd_transform normalization is pure attr processing
    # (reads only the scene dimensions), so run it in the gate too — a bad
    # transform must not leave a partial node behind.
    prepare_transform_attrs(attrs, ctx.store)

    # 1. Setup: Create group
    group = ctx.store.require_group(path)

    aprint(f"📝 Writing {n_vertices:,} line vertices ({n_dims}D) to {path}")

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

    # Write segments array (always, not just for indexed type)
    segment_chunk_size = (
        ordering_data["segment_ordering"]["chunk_size"] if ordering_data else 2048
    )
    ctx.dataset_ctx.encoder.encode(
        data=segments,
        zarr_group=group,
        name="segments",
        semantic_type=SemanticType.INDEX,
        mode=ctx.dataset_ctx.encoding_mode,
        chunks=(segment_chunk_size, 2),
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
        # Use the canonical COLOR helper (shared with Points / GSplats)
        # so the default-precision and color_mode-detection logic is
        # symmetric across all three geometry types.
        write_colors(
            group,
            colors,
            ordering_data.get("vertex_ordering") if ordering_data else None,
            n_vertices,
            ctx.dataset_ctx,
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
        )
        metadata["has_sharpness"] = True

    if scalars is not None:
        write_scalars(group, scalars, ordering_data, n_vertices, ctx.dataset_ctx)
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

    # Transform + nd_transform attrs were already normalized in the fail-fast
    # gate (step 0h) — prepare_transform_attrs is NOT idempotent (it
    # transposes the matrix), so it must run exactly once.

    # Set default rendering attributes if not provided
    # (must match write_points/write_gsplats)
    apply_default_render_attrs(attrs)

    # Set attributes (all core metadata per spec Section 6.6)
    group.attrs.update(attrs)
    group.attrs["type"] = "lines"
    group.attrs["n_vertices"] = n_vertices
    group.attrs["n_segments"] = n_segments
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
    if not attrs.pop("_skip_scene_bounds", False):
        ctx.update_scene_bounds(position_bounds)

    # Write labels if provided (CSR-style: label_offsets + label_bytes)
    # For lines, labels are per-vertex (n_vertices)
    if labels is not None:
        sort_order = (
            ordering_data["vertex_sort_indices"] if ordering_data is not None else None
        )
        write_labels_csr(group, labels, n_vertices, ctx.compressor, sort_order)
        metadata["has_labels"] = True

    # Write image labels if provided (CSR-style, no compression on blobs)
    if image_labels is not None:
        sort_order = (
            ordering_data["vertex_sort_indices"] if ordering_data is not None else None
        )
        write_image_labels_csr(
            group, image_labels, n_vertices, ctx.compressor, sort_order
        )
        metadata["has_image_labels"] = True

    aprint(f"✅ Lines written to {path}")

    return metadata
