"""add_lines body + partition/multi-LOD wrapper impls.

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
    sync_custom_colormap_attr,
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
    substitutive_lod: Any = None,
    partition: Any = None,
    **attrs: Any,
) -> Union[Lines, "Group"]:
    if additive_lod is not None and substitutive_lod is not None:
        raise ValueError(
            "additive_lod and substitutive_lod are mutually exclusive coarsening "
            "strategies for Lines (append vs replace on the same LOD axis); "
            "pass only one."
        )
    if substitutive_lod is not None and partition is not None:
        raise ValueError(
            "partition= and substitutive_lod= cannot be combined yet "
            "(partition-of-substitutive is not implemented). Use one or the other."
        )
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

        # Substitutive-LOD branch — coarse levels are synthesised gsplats (each
        # segment lifted to isotropic "bead" Gaussians, then reduced by the
        # gsplat substitutive pipeline) under a kind=lod Group whose finest
        # child is the original Lines node. Fires before partition; mutually
        # exclusive with additive_lod/partition (checked above). ``vert_arr`` is
        # already dim_order-transformed, so children use dim_order=None/fill=None.
        if substitutive_lod is not None and n_vertices > 0:
            from ..lod.lines import resolve_substitutive_axis_lines

            substitutive_spec = resolve_substitutive_axis_lines(substitutive_lod)
            if substitutive_spec is not None:
                return add_lines_substitutive_lod_wrapper_impl(
                    group,
                    name=name,
                    vert_arr=vert_arr,
                    widths=widths,
                    colors=colors,
                    sharpness=sharpness,
                    scalars=scalars,
                    labels=labels,
                    image_labels=image_labels,
                    indices=indices,
                    line_type=line_type,
                    parent=parent,
                    extend_to_all=extend_to_all,
                    spec=substitutive_spec,
                    **attrs,
                )

        # NOTE: add_lines has no compiler-level auto-partition heuristic — only a
        # user-explicit ``partition=`` is honored (Points additionally auto-
        # partitions large clouds; Lines do not).

        # ``partition=False`` is an explicit no-partition bypass (the same
        # sentinel resolve_auto_partition normalises for Points), used by the
        # substitutive finest-child + degenerate fallback so a nested add_lines
        # can't re-partition. Normalise it to None here (lines does not yet wire
        # the auto-partition heuristic, so there is nothing else to resolve).
        if partition is False:
            partition = None

        # Partition branch — polyline-aware BSP. Whole polylines are
        # atomic; the BSP runs over per-polyline centroids and assigns
        # each polyline atomically to a part. Mirrors add_points but
        # at the polyline granularity.
        if partition is not None and vert_arr.shape[1] >= 3:
            from ..lod.lines import identify_polylines
            from ..partition import (
                DEFAULT_MAX_ELEMENTS,
                median_bsp_polylines,
                midpoint_bsp_polylines,
                sah_bsp_partition,
                warn_if_oversized_single_part,
            )

            if partition is True:
                max_elements = DEFAULT_MAX_ELEMENTS
                partition_rule = "median"
            elif isinstance(partition, dict):
                max_elements = int(partition.get("max_elements", DEFAULT_MAX_ELEMENTS))
                if max_elements < 1:
                    raise ValueError(
                        f"partition max_elements must be >= 1, got {max_elements}"
                    )
                partition_rule = str(partition.get("rule", "median"))
                if partition_rule not in ("median", "midpoint", "sah"):
                    raise ValueError(
                        f"partition rule must be 'median', 'midpoint', or 'sah'; "
                        f"got {partition_rule!r}"
                    )
            else:
                raise TypeError(
                    f"partition must be None, True, or dict; "
                    f"got {type(partition).__name__}"
                )

            if image_labels is not None:
                raise ValueError(
                    "image_labels is not supported alongside partition=. "
                    "Decompose the data manually or omit image_labels."
                )

            polyline_indices = identify_polylines(n_vertices, line_type, indices)

            if partition_rule == "sah":
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
            elif partition_rule == "midpoint":
                polyline_parts = midpoint_bsp_polylines(
                    vert_arr, polyline_indices, max_elements
                )
            else:
                polyline_parts = median_bsp_polylines(
                    vert_arr, polyline_indices, max_elements
                )

            warn_if_oversized_single_part(
                len(polyline_parts),
                sum(int(polyline_indices[p].size) for p in polyline_parts[0])
                if polyline_parts
                else 0,
                max_elements,
                name,
            )
            if len(polyline_parts) > 1:
                return add_lines_partition_wrapper_impl(
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
                    additive_lod=additive_lod,
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
        sync_custom_colormap_attr(attrs)

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


def add_lines_partition_wrapper_impl(
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
    additive_lod: Any = None,
    **attrs: Any,
) -> "Group":
    """Build a kind=partition wrapper Group with one Lines child per BSP part.

    Polylines / connected components are atomic — each lands in exactly
    one part. ``segments`` parts are re-emitted as ``segments`` (their
    consecutive member pairs are the segments). ``indexed`` parts are
    re-emitted as ``indexed`` with the original edges remapped to
    part-local vertex indices, so the exact graph topology is preserved
    (no edge is dropped or fabricated, and odd-sized components no longer
    crash). For ``polyline`` / ``loop`` types (where the input is a single
    polyline), the BSP only ever produces one part — the user is already at
    the single-polyline granularity and there's nothing to partition. We
    refuse the partition in that case with a clear error.
    """
    wrapper_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    leaf_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}

    parent_node = parent or group
    wrapper = parent_node.add_partition_group(
        name=name,
        display_type="lines",
        max_elements=max_elements,
        **wrapper_attrs,
    )

    part_sizes = [
        sum(int(polyline_indices[p].size) for p in part) for part in polyline_parts
    ]
    aprint(
        f"  ✂️  Partitioned '{name}' into {len(polyline_parts)} parts via "
        f"polyline-centroid BSP "
        f"(max_elements={max_elements:,}, sizes={part_sizes})"
    )

    # For ``indexed`` inputs, bucket the ORIGINAL edges by part up front.
    # Each connected component (hence each edge) lands wholly in one part,
    # so the exact graph topology is preserved by remapping real edges into
    # part-local indices below — never by re-pairing sorted union-find
    # members (which fabricated/dropped edges and crashed on odd-sized
    # components).
    part_edges: List[List[tuple[int, int]]] = [[] for _ in polyline_parts]
    if line_type == "indexed" and indices is not None and len(indices) > 0:
        vertex_to_part = np.full(n_vertices, -1, dtype=np.int64)
        for part_i, polyline_ids in enumerate(polyline_parts):
            for p in polyline_ids:
                vertex_to_part[polyline_indices[p]] = part_i
        # ``indices`` is the flat (2E,) edge list; view it as (E, 2).
        for edge in np.asarray(indices, dtype=np.int64).reshape(-1, 2):
            a, b = int(edge[0]), int(edge[1])
            pa = int(vertex_to_part[a])
            if pa >= 0 and int(vertex_to_part[b]) == pa:
                part_edges[pa].append((a, b))

    # The new line_type per part is either:
    # - ``polyline`` / ``loop`` with one polyline ⇒ keep as-is.
    # - ``segments`` ⇒ re-emit as ``segments`` (consecutive member pairs).
    # - ``indexed`` ⇒ re-emit as ``indexed`` with the part's real edges
    #   remapped to part-local vertex indices.
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
            # ``segments`` members are stored pairwise, so consecutive
            # pairs (0,1),(2,3),... ARE the segments. (``indexed`` edges
            # are remapped from the original edge list below instead.)
            if line_type == "segments":
                for k in range(0, members.size - 1, 2):
                    new_segments.append([cursor + k, cursor + k + 1])
            cursor += int(members.size)

        if not vertex_index_list:
            continue
        part_vertex_idx = np.concatenate(vertex_index_list)
        part_vertices = vert_arr[part_vertex_idx]
        part_n = int(part_vertex_idx.size)

        # Remap this part's original edges to part-local vertex indices.
        if line_type == "indexed":
            local_of = {int(g): loc for loc, g in enumerate(part_vertex_idx)}
            new_segments = [[local_of[a], local_of[b]] for (a, b) in part_edges[i]]

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
        # ``segments`` emits segments. ``indexed`` emits the part's
        # remapped real edges (or degrades to empty segments if the part
        # happens to contain only isolated, edgeless vertices).
        if line_type in ("polyline", "loop"):
            part_line_type = line_type
            part_indices = None
        elif line_type == "segments":
            part_line_type = "segments"
            part_indices = None
        else:  # indexed
            part_line_type = "indexed"
            # add_lines expects a flat (2M,) index list for indexed lines.
            part_indices = (
                np.asarray(new_segments, dtype=np.intp).reshape(-1)
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
            # image_labels banned alongside partition= (see add_lines entry)
            image_labels=None,
            indices=part_indices,
            line_type=part_line_type,
            extend_to_all=extend_to_all,
            dim_order=None,
            fill=None,
            partition=None,
            # Inner LOD ladder per spatial part — each part decides its own
            # ladder independently (partition-of-additive-LOD composition,
            # mirroring add_points). Dropped silently before this fix.
            additive_lod=additive_lod,
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

    # Mirror the writer's custom-colormap resolution (ndarray/matplotlib name
    # -> 'custom') so the returned node matches what zarr stores.
    sync_custom_colormap_attr(attrs)

    return Lines(
        name,
        metadata=metadata,
        parent=cast(Any, parent_node),
        writer=writer,
        **attrs,
    )


def add_lines_substitutive_lod_wrapper_impl(
    group: "Group",
    *,
    name: str,
    vert_arr: np.ndarray,
    widths: Any,
    colors: Any,
    sharpness: Any,
    scalars: Any,
    labels: Any,
    image_labels: Any,
    indices: Optional[np.ndarray],
    line_type: str,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    spec: Dict[str, Any],
    **attrs: Any,
) -> Union["Group", Lines]:
    """Write a Lines node whose coarse LOD levels are synthesised gsplats.

    Each segment is lifted to a string of isotropic "bead" Gaussians (see
    :func:`luxar.gsplats.lift.lift_lines_to_gsplats` — beads are view-independent
    and sum to a smooth tube, unlike one elongated anisotropic Gaussian), the
    gsplat substitutive pipeline synthesises fewer-but-larger representative
    levels (per-bin mass-preserving amplitudes + a ``max_aspect`` anisotropy cap
    keep every level's brightness and hue view-coherent — see
    :func:`luxar.gsplats.lift.coarse_substitutive_levels`), and those become the
    coarse children of a ``kind=lod`` Group whose **finest** child is the
    original Lines node. Mirrors
    :func:`add_points_substitutive_lod_wrapper_impl`.
    """
    from ....gsplats.lift import coarse_substitutive_levels, lift_lines_to_gsplats
    from ..lod.group import coverage_fractions

    # Scalar+colormap lines: pass scalars+colormap THROUGH to the lift, which
    # interpolates the scalar per bead then maps it through the LUT (matching the
    # line shader's interpolate-then-LUT order — important for non-linear
    # colormaps). The finest Lines child keeps scalars+colormap natively.
    scalars_for_lift = None
    colors_for_lift = colors
    if scalars is not None and colors is None:
        if attrs.get("colormap") is None:
            raise ValueError(
                "substitutive_lod on Lines with scalars requires a colormap "
                "(scalars map to colour via a colormap LUT). Pass colormap=..., "
                "or provide explicit per-vertex colors."
            )
        scalars_for_lift = scalars

    lifted = lift_lines_to_gsplats(
        vert_arr,
        widths,
        line_type=line_type,
        indices=indices,
        colors=colors_for_lift,
        scalars=scalars_for_lift,
        colormap=attrs.get("colormap") if scalars_for_lift is not None else None,
        opacity=1.0,
        truncation_radius=float(spec["truncation_radius"]),
    )

    # Resolve which dims coarsening may merge over (default Auto = displayed dims,
    # grouping by non-displayed dims so coarse beads never blend across a
    # categorical/sliced axis). Bead columns == vertex/scene dim columns.
    from ..lod.group import resolve_coarsen_dims

    coarsen_dims = resolve_coarsen_dims(
        group._find_scene(), int(lifted.ndim), spec.get("coarsen_dims")
    )

    # Synthesise coarse gsplat levels (level 0 dropped, render-light conserved).
    coarse = coarse_substitutive_levels(
        lifted,
        compression_factor=int(spec["compression_factor"]),
        levels=int(spec["levels"]),
        method=str(spec["method"]),
        device=spec.get("device", "auto"),
        seed=spec.get("seed"),
        coarsen_dims=coarsen_dims,
        max_aspect=spec.get("max_aspect", 3.0),
    )

    # Degenerate -> flat Lines node. Covers BOTH no coarse levels AND an
    # edge-less / all-zero-width set (the lift yields 0 beads, so every coarse
    # level is empty: coarse[-1] is the coarsest). vert_arr is already
    # dim_order-transformed; partition=False so auto-partition can't re-fire.
    if not coarse or int(coarse[-1].n_splats) == 0:
        aprint(
            f"  ⚠ substitutive_lod '{name}': no liftable segments / too small to "
            "synthesise coarse levels; writing a flat Lines node."
        )
        return add_lines_impl(
            group,
            name=name,
            vertices=vert_arr,
            widths=widths,
            colors=colors,
            sharpness=sharpness,
            scalars=scalars,
            labels=labels,
            image_labels=image_labels,
            indices=indices,
            line_type=line_type,
            parent=parent,
            extend_to_all=extend_to_all,
            partition=False,
            **attrs,
        )

    coarse_first = list(reversed(coarse))  # coarsest first
    # Finest "count" is the full lifted BEAD count, not n_vertices: the coarse
    # gsplat children are bead reductions (N_beads/K^l), so measuring the finest
    # in the same bead currency keeps the ladder counts strictly ascending and
    # gives the real Lines node the highest switch threshold (a vertex count is
    # a different, much smaller scale and would collapse the top thresholds).
    counts = [int(c.n_splats) for c in coarse_first] + [int(lifted.n_splats)]
    explicit = spec.get("coverage_fractions")
    if explicit is not None:
        if len(explicit) != len(counts):
            raise ValueError(
                f"coverage_fractions has {len(explicit)} entries but the LOD ladder "
                f"has {len(counts)} levels ({len(coarse_first)} gsplat + 1 lines)"
            )
        coverage_vals = list(explicit)
    else:
        # Viewport-relative coverage fractions ``sqrt(N_i/N_finest)`` (count ratios;
        # the viewer anchors the finest at fills-screen). No per-level radius or
        # world-extent needed.
        coverage_vals = coverage_fractions(counts)

    lod_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    child_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}
    child_attrs.pop("coverage_fraction", None)
    lod_attrs.setdefault("display_type", "lines")  # finest child is lines
    # Coarse gsplat children carry baked per-splat colours, so they must NOT also
    # receive `colormap` (gsplats reject colors+colormap); it stays on the finest
    # Lines child only.
    gsplat_child_attrs = {k: v for k, v in child_attrs.items() if k != "colormap"}

    parent_node = parent or group
    aprint(
        f"  📐 Substitutive-LOD '{name}': {len(coarse_first)} gsplat levels + lines "
        f"(counts coarsest→finest={counts}, K={spec['compression_factor']})"
    )
    lod_group_node = parent_node.add_lod_group(name, **lod_attrs)

    # Coarse gsplat children (coarsest first). vert_arr already dim_order-applied.
    for idx, lvl_data in enumerate(coarse_first):
        lod_group_node.add_gsplats_from_data(
            name=f"child_{idx}",
            result=lvl_data,
            extend_to_all=extend_to_all,
            dim_order=None,
            fill=None,
            lod_group=None,
            additive_lod=None,
            coverage_fraction=coverage_vals[idx],
            **gsplat_child_attrs,
        )

    # Finest child: the original Lines node (carries all vertices + image_labels).
    lod_group_node.add_lines(
        name=f"child_{len(coarse_first)}",
        vertices=vert_arr,
        widths=widths,
        colors=colors,
        sharpness=sharpness,
        scalars=scalars,
        labels=labels,
        image_labels=image_labels,
        indices=indices,
        line_type=line_type,
        extend_to_all=extend_to_all,
        dim_order=None,
        fill=None,
        additive_lod=None,
        substitutive_lod=None,
        partition=False,
        coverage_fraction=coverage_vals[-1],
        **child_attrs,
    )

    return lod_group_node
