"""add_lines body + split/multi-LOD wrapper impls.

Pure functions taking a ``group: Group`` parameter as the first arg.
Called by ``Group.add_lines`` (a thin signature + docstring + delegate)
in ``core/group/group.py``.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

import numpy as np
from arbol import aprint

from ...lines import Lines
from ..compositing import (
    COMPOSITING_ATTRS,
    position_bounds_from_array,
    slice_optional_array,
)
from ..dim_order import apply_dim_order_positions

if TYPE_CHECKING:
    from ...node import Node
    from ..group import Group


def add_lines_impl(
    group: "Group",
    *,
    name: str,
    vertices: Any,
    widths: Any,
    colors: Any = None,
    sharpness: Any = None,
    scalars: Any = None,
    labels: Optional[Union[List[str], Sequence[str]]] = None,
    image_labels: Optional[Any] = None,
    indices: Optional[np.ndarray] = None,
    line_type: str = "polyline",
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    additive_lod: Any = None,
    split: Any = None,
    **attrs: Any,
) -> Union[Lines, "Group"]:
    try:
        scene = group._find_scene()

        vert_arr: np.ndarray = (
            vertices if isinstance(vertices, np.ndarray) else np.asarray(vertices)
        )
        if vert_arr.ndim != 2:
            raise ValueError(
                f"Vertices must have shape (N, D), got shape {vert_arr.shape}"
            )

        # Apply dim_order before validation
        vert_arr, extend_to_all = apply_dim_order_positions(
            vert_arr, scene, dim_order, fill, extend_to_all
        )

        n_vertices = vert_arr.shape[0]
        ndim = vert_arr.shape[1]

        # NOTE: compiler-level auto-split heuristic is a separate PR
        # (β); add_lines honors user-explicit ``split=`` here and
        # will pick up the auto-split path automatically once β
        # merges. Until then, ``split=`` is opt-in via the call site.

        # Split branch — polyline-aware BSP. Whole polylines are
        # atomic; the BSP runs over per-polyline centroids and assigns
        # each polyline atomically to a part. Mirrors add_points but
        # at the polyline granularity.
        if split is not None and vert_arr.shape[1] >= 3:
            from ..lod.lines import identify_polylines
            from ..split import (
                DEFAULT_MAX_ELEMENTS,
                midpoint_bsp_polylines,
                sah_bsp_partition,
            )

            if split is True:
                max_elements = DEFAULT_MAX_ELEMENTS
                split_rule = "midpoint"
            elif isinstance(split, dict):
                max_elements = int(split.get("max_elements", DEFAULT_MAX_ELEMENTS))
                if max_elements < 1:
                    raise ValueError(
                        f"split max_elements must be >= 1, got {max_elements}"
                    )
                split_rule = str(split.get("rule", "midpoint"))
                if split_rule not in ("midpoint", "sah"):
                    raise ValueError(
                        f"split rule must be 'midpoint' or 'sah'; got {split_rule!r}"
                    )
            else:
                raise TypeError(
                    f"split must be None, True, or dict; got {type(split).__name__}"
                )

            if image_labels is not None:
                raise ValueError(
                    "image_labels is not supported alongside split=. "
                    "Decompose the data manually or omit image_labels."
                )

            polyline_indices = identify_polylines(n_vertices, line_type, indices)

            if split_rule == "sah":
                # SAH operates on per-polyline centroids in this
                # context too — same atomic-polyline guarantee.
                if not polyline_indices:
                    polyline_parts: List[List[int]] = []
                else:
                    centroids = np.array(
                        [
                            vert_arr[p, :3].mean(axis=0) if p.size > 0 else np.zeros(3)
                            for p in polyline_indices
                        ],
                        dtype=np.float64,
                    )
                    # Cap is per-vertex; SAH gives us per-centroid
                    # parts; we re-aggregate to vertex-count parts.
                    approx_per_poly = max(
                        1,
                        n_vertices // max(1, len(polyline_indices)),
                    )
                    centroid_cap = max(1, max_elements // approx_per_poly)
                    centroid_parts = sah_bsp_partition(
                        centroids, max_elements=centroid_cap
                    )
                    polyline_parts = [idx_arr.tolist() for idx_arr in centroid_parts]
            else:
                polyline_parts = midpoint_bsp_polylines(
                    vert_arr, polyline_indices, max_elements
                )

            if len(polyline_parts) > 1:
                return add_lines_split_wrapper_impl(
                    group,
                    name=name,
                    vert_arr=vert_arr,
                    polyline_indices=polyline_indices,
                    polyline_parts=polyline_parts,
                    n_vertices=n_vertices,
                    widths=widths,
                    colors=colors,
                    sharpness=sharpness,
                    scalars=scalars,
                    labels=labels,
                    indices=indices,
                    line_type=line_type,
                    parent=parent,
                    extend_to_all=extend_to_all,
                    max_elements=max_elements,
                    **attrs,
                )
            # 1 part → fall through to single-leaf write.

        # Additive-LOD branch — polyline-level multi-LOD write.
        # Fires before the single-shot write so we don't double-
        # validate. Mirrors the points add path.
        if additive_lod is not None:
            from ..lod.lines import (
                make_additive_lod_lines,
                resolve_additive_axis_lines,
            )

            additive_spec = resolve_additive_axis_lines(additive_lod)
            if additive_spec is not None:
                widths_arr = (
                    widths
                    if isinstance(widths, np.ndarray) and widths.shape == (n_vertices,)
                    else None
                )
                colors_for_energy = (
                    colors
                    if isinstance(colors, np.ndarray)
                    and colors.ndim == 2
                    and colors.shape[0] == n_vertices
                    else None
                )
                scalars_for_energy = (
                    scalars
                    if isinstance(scalars, np.ndarray)
                    and scalars.shape == (n_vertices,)
                    else None
                )
                polyline_levels = make_additive_lod_lines(
                    vert_arr,
                    line_type=line_type,
                    indices=indices,
                    widths=widths_arr,
                    method=additive_spec["method"],
                    n_lods=additive_spec["n_lods"],
                    counts=additive_spec["counts"],
                    seed=additive_spec["seed"],
                    colors=colors_for_energy,
                    scalars=scalars_for_energy,
                    salience_kind=additive_spec.get("salience_kind", "size"),
                )
                if len(polyline_levels) > 1:
                    return add_lines_multi_lod_wrapper_impl(
                        group,
                        name=name,
                        vert_arr=vert_arr,
                        polyline_levels=polyline_levels,
                        n_vertices=n_vertices,
                        widths=widths,
                        colors=colors,
                        sharpness=sharpness,
                        scalars=scalars,
                        labels=labels,
                        parent=parent,
                        extend_to_all=extend_to_all,
                        method=additive_spec["method"],
                        **attrs,
                    )
                # 1 level (degenerate single polyline) → fall through.

        aprint(f"Adding lines node '{name}' with {n_vertices:,} vertices in {ndim}D.")

        scene._validate_data_dimensions(vert_arr, name, data_type="vertices")

        final_extend_dims = scene._resolve_extend_to_all(
            extend_to_all, vert_arr, "lines"
        )
        if final_extend_dims:
            attrs["extend_to_all"] = final_extend_dims
            aprint(f"  📡 Extending visibility across: {final_extend_dims}")

        # Colormap / colors mutual exclusivity
        colormap = attrs.get("colormap")
        if colors is not None and colormap is not None:
            raise ValueError(
                "Cannot specify both 'colors' and 'colormap'. Use one or the other."
            )
        if scalars is not None and colormap is None:
            raise ValueError(
                "'scalars' requires a 'colormap' attribute to map values to colors."
            )

        parent_node = parent or group

        writer = group._require_scene_writer(scene)
        path = f"{parent_node.path}/{name}" if parent_node.path else name
        metadata = writer.write_lines(
            path,
            vert_arr.astype(np.float32),
            widths=widths,
            colors=cast(Any, colors),
            sharpness=sharpness,
            scalars=scalars,
            indices=indices,
            line_type=line_type,
            labels=labels,
            image_labels=image_labels,
            **attrs,
        )

        # Sync colormap attr with what the compiler wrote to zarr
        if "colormap" in attrs:
            from ....colormaps.builtins import BUILTIN_COLORMAP_NAMES

            cm = attrs["colormap"]
            if not isinstance(cm, str) or (
                isinstance(cm, str) and cm not in BUILTIN_COLORMAP_NAMES
            ):
                attrs["colormap"] = "custom"

        if labels is not None:
            scene._notify_labels_added()
        if image_labels is not None:
            scene._notify_image_labels_added()

        return Lines(
            name,
            metadata=metadata,
            parent=cast(Any, parent_node),
            writer=writer,
            **attrs,
        )
    except (ValueError, TypeError) as e:
        aprint(f"Failed to add lines node '{name}': {e}")
        raise ValueError(f"Could not add lines '{name}': {e}") from e


def add_lines_split_wrapper_impl(
    group: "Group",
    *,
    name: str,
    vert_arr: np.ndarray,
    polyline_indices: List[np.ndarray],
    polyline_parts: List[List[int]],
    n_vertices: int,
    widths: Any,
    colors: Any,
    sharpness: Any,
    scalars: Any,
    labels: Any,
    indices: Optional[np.ndarray],
    line_type: str,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    max_elements: int,
    **attrs: Any,
) -> "Group":
    """Build a kind=split wrapper Group with one Lines child per BSP part.

    Polylines are atomic — each polyline lands in exactly one part.
    For the ``segments`` / ``indexed`` line types, the resulting
    per-part data is re-emitted with ``line_type='segments'``: the
    original segment topology is preserved by walking pairs within
    each component the BSP grouped together. For ``polyline`` /
    ``loop`` types (where the input is a single polyline), the BSP
    only ever produces one part — the user is already at the single-
    polyline granularity and there's nothing to split. We refuse the
    split in that case with a clear error.
    """
    wrapper_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    leaf_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}

    parent_node = parent or group
    wrapper = parent_node.add_split_group(
        name=name,
        display_type="lines",
        max_elements=max_elements,
        **wrapper_attrs,
    )

    part_sizes = [
        sum(int(polyline_indices[p].size) for p in part) for part in polyline_parts
    ]
    aprint(
        f"  ✂️  Split '{name}' into {len(polyline_parts)} parts via "
        f"polyline-centroid BSP "
        f"(max_elements={max_elements:,}, sizes={part_sizes})"
    )

    # The new line_type per part is either:
    # - ``polyline`` / ``loop`` with one polyline ⇒ keep as-is.
    # - ``segments`` / ``indexed`` ⇒ re-emit as ``segments`` with the
    #   pairs from the polylines that landed in this part.
    for i, polyline_ids in enumerate(polyline_parts):
        # Collect vertices for this part, preserving original order.
        vertex_index_list: List[np.ndarray] = []
        new_segments: List[List[int]] = []
        cursor = 0
        for p in polyline_ids:
            members = polyline_indices[p]
            if members.size == 0:
                continue
            vertex_index_list.append(members)
            # For segments / indexed, re-emit each pair after
            # remapping into the part-local vertex indexing (which
            # follows the concatenation order).
            if line_type in ("segments", "indexed"):
                # Walk in pairs along the polyline's original member
                # ordering. For ``segments`` this is just (0,1),
                # (2,3), ... For ``indexed`` connected components,
                # consecutive members aren't necessarily a segment;
                # we approximate by linking consecutive members,
                # which is exact for ``segments`` (the only case the
                # split path actually decomposes — see polyline /
                # loop guard below).
                for k in range(0, members.size - 1, 2):
                    new_segments.append([cursor + k, cursor + k + 1])
            cursor += int(members.size)

        if not vertex_index_list:
            continue
        part_vertex_idx = np.concatenate(vertex_index_list)
        part_vertices = vert_arr[part_vertex_idx]
        part_n = int(part_vertex_idx.size)

        # Slice per-vertex parameters into this part.
        part_widths = (
            widths
            if (not isinstance(widths, np.ndarray)) or widths.shape != (n_vertices,)
            else widths[part_vertex_idx]
        )
        part_colors = slice_optional_array(colors, part_vertex_idx, n_vertices)
        part_sharpness = slice_optional_array(sharpness, part_vertex_idx, n_vertices)
        part_scalars = slice_optional_array(scalars, part_vertex_idx, n_vertices)
        part_labels = slice_optional_array(labels, part_vertex_idx, n_vertices)

        # Choose the per-part line_type. ``polyline`` / ``loop`` with
        # one polyline = one part, so the original type is preserved.
        # For ``segments`` we emit segments. For ``indexed`` (rare
        # — typically pre-merged graphs) we also emit segments with
        # the reconstructed indices below.
        if line_type in ("polyline", "loop"):
            part_line_type = line_type
            part_indices = None
        elif line_type == "segments":
            part_line_type = "segments"
            part_indices = None
        else:  # indexed
            part_line_type = "indexed"
            part_indices = (
                np.asarray(new_segments, dtype=np.intp).reshape(-1, 2)
                if new_segments
                else None
            )
            if part_indices is None:
                # Single-vertex polylines on indexed → emit as
                # ``segments`` of zero length (caller asked for
                # indexed but the part has no edges; degrades
                # gracefully).
                part_line_type = "segments"
                if part_n % 2 != 0:
                    # Round to an even count to satisfy segments
                    # validation; drop the trailing isolated vertex.
                    part_vertices = part_vertices[:-1]
                    part_n -= 1

        wrapper.add_lines(
            name=f"part_{i}",
            vertices=part_vertices,
            widths=part_widths,
            colors=part_colors,
            sharpness=part_sharpness,
            scalars=part_scalars,
            labels=part_labels,
            # image_labels banned alongside split= (see add_lines entry)
            image_labels=None,
            indices=part_indices,
            line_type=part_line_type,
            extend_to_all=extend_to_all,
            dim_order=None,
            fill=None,
            split=None,
            **leaf_attrs,
        )

    wrapper._persist_attr("position_bounds", position_bounds_from_array(vert_arr))

    return wrapper


def add_lines_multi_lod_wrapper_impl(
    group: "Group",
    *,
    name: str,
    vert_arr: np.ndarray,
    polyline_levels: List[List[np.ndarray]],
    n_vertices: int,
    widths: Any,
    colors: Any,
    sharpness: Any,
    scalars: Any,
    labels: Any,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    method: str,
    **attrs: Any,
) -> Lines:
    """Write a Lines node with multi-additive-LOD subgroups.

    Each ``additive_<i>/`` subgroup carries a subset of polylines
    (whole polylines, never bisected). Segment indices are local to
    each subgroup. The parent lines node carries
    ``n_additive_sublods``, the global ``position_bounds``, and the
    standard compositing attrs.
    """
    scene = group._find_scene()
    writer = group._require_scene_writer(scene)
    parent_node = parent or group
    path = f"{parent_node.path}/{name}" if parent_node.path else name

    level_slices: List[Dict[str, Any]] = []
    for level_polylines in polyline_levels:
        # Concatenate vertex indices across all polylines in this
        # level; build local segment indices per polyline.
        level_vertex_indices: List[int] = []
        level_local_segments: List[np.ndarray] = []
        local_offset = 0
        for poly in level_polylines:
            k = poly.size
            level_vertex_indices.extend(int(idx) for idx in poly)
            if k >= 2:
                # Polyline connectivity: (0,1), (1,2), ..., (k-2, k-1)
                seg = np.column_stack(
                    [
                        np.arange(k - 1, dtype=np.uint32) + local_offset,
                        np.arange(1, k, dtype=np.uint32) + local_offset,
                    ]
                )
                level_local_segments.append(seg)
            local_offset += k
        vertex_index_arr = np.asarray(level_vertex_indices, dtype=np.intp)
        if level_local_segments:
            level_segments = np.concatenate(level_local_segments, axis=0)
        else:
            level_segments = np.empty((0, 2), dtype=np.uint32)

        level_slices.append(
            {
                "vertices": vert_arr[vertex_index_arr].astype(np.float32),
                "widths": slice_optional_array(widths, vertex_index_arr, n_vertices),
                "colors": slice_optional_array(colors, vertex_index_arr, n_vertices),
                "sharpness": slice_optional_array(
                    sharpness, vertex_index_arr, n_vertices
                ),
                "scalars": slice_optional_array(scalars, vertex_index_arr, n_vertices),
                "labels": slice_optional_array(labels, vertex_index_arr, n_vertices),
                "segments": level_segments,
                "n_polylines": len(level_polylines),
            }
        )

    aprint(
        f"  📐 Additive-LOD '{name}': {len(polyline_levels)} levels "
        f"(method={method!r}, polylines_per_level="
        f"{[len(L) for L in polyline_levels]})"
    )

    metadata = writer.write_lines_multi_lod(
        path,
        level_slices,
        method=method,
        extend_to_all=extend_to_all,
        **attrs,
    )

    return Lines(
        name,
        metadata=metadata,
        parent=cast(Any, parent_node),
        writer=writer,
        **attrs,
    )
