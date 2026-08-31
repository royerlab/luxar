"""add_lines body + partition/multi-LOD wrapper impls.

Pure functions taking a ``group: Group`` parameter as the first arg.
Called by ``Group.add_lines`` (a thin signature + docstring + delegate)
in ``core/group/group.py``.
"""

from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
from arbol import aprint

from ...lines import Lines
from ..compositing import (
    ABSENT_WHEN_NONE_RENDER_ATTRS,
    COMPOSITING_ATTRS,
    funnel_add_error,
    is_broadcast_color,
    position_bounds_from_array,
    preflight_extend_to_all,
    reject_mesh_only_appearance,
    slice_optional_array,
    strip_absent_attr_kwargs,
    sync_custom_colormap_attr,
    unnest_add_error,
    validate_line_indices_before_split,
    validate_lines_channels_before_split,
)
from ..dim_order import apply_dim_order_positions
from ..partition import is_requested, reject_mismatched_partition_parent

if TYPE_CHECKING:
    from ...node import Node
    from ..group import Group
    from ..partition import BSPNode


def _sah_polyline_centroids(
    vert_arr: np.ndarray,
    polyline_indices: List[np.ndarray],
) -> np.ndarray:
    """Per-polyline centroids for the SAH split in every position column.

    An empty polyline (unreachable via ``identify_polylines``; kept for
    defensive symmetry with ``median_bsp_polylines``) contributes a zero row
    of the SAME width as the real rows, so 2D input never builds a ragged
    array (#901).
    """
    return np.array(
        [
            vert_arr[p].mean(axis=0) if p.size > 0 else np.zeros(vert_arr.shape[1])
            for p in polyline_indices
        ],
        dtype=np.float64,
    )


def _build_line_partition_tree(
    vert_arr: np.ndarray,
    polyline_indices: List[np.ndarray],
    n_vertices: int,
    max_elements: int,
    rule: str,
    split_axes: Sequence[int],
) -> Tuple[Optional["BSPNode"], List[List[int]]]:
    """Build the requested atomic-polyline BSP and flatten its leaves."""
    from ..partition import bsp_leaf_parts, spatial_bsp_polyline_tree, spatial_bsp_tree

    if rule != "sah":
        tree = spatial_bsp_polyline_tree(
            vert_arr,
            polyline_indices,
            max_elements,
            rule=rule,
            split_axes=split_axes,
        )
    elif not polyline_indices:
        tree = None
    else:
        centroids = _sah_polyline_centroids(vert_arr, polyline_indices)
        approximate_vertices_per_polyline = max(
            1, n_vertices // max(1, len(polyline_indices))
        )
        centroid_cap = max(1, max_elements // approximate_vertices_per_polyline)
        tree = spatial_bsp_tree(
            centroids,
            max_elements=centroid_cap,
            rule="sah",
            split_axes=split_axes,
        )

    parts = [] if tree is None else [part.tolist() for part in bsp_leaf_parts(tree)]
    return tree, parts


def _verified_indexed_additive(
    *,
    additive_lod: Any,
    line_type: str,
    indices: Optional[np.ndarray],
    n_vertices: int,
) -> None:
    """Fail before partition writes if an indexed additive ladder is unsafe.

    Kept outside :func:`add_lines_impl` to avoid adding boolean sub-conditions to
    an already 24-branch function under the C901 ratchet.

    The multi-LOD writer carries no edge list — it rebuilds one by chaining each
    connected component in ascending vertex order — so every component's
    undirected edge multiset, including duplicate multiplicity, must equal its
    consecutive vertex pairs.

    Raises:
        ValueError: an explicit ladder was requested and the topology would be
            corrupted by it.
    """
    if additive_lod is None or additive_lod is False:
        return
    if line_type != "indexed" or indices is None:
        return

    from ..lod.lines import indexed_components_are_chains

    if not indexed_components_are_chains(
        n_vertices, np.asarray(indices, dtype=np.intp).reshape(-1, 2)
    ):
        raise ValueError(
            "line_type='indexed' cannot take an additive ladder unless every "
            "connected component's undirected edge multiset, including duplicate "
            "multiplicity, equals its consecutive vertex pairs — the streaming "
            "writer rebuilds edges by chaining each component in ascending vertex "
            "order. Pass additive_lod=False to write this node without a ladder."
        )


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
    keys: Optional[Union[List[str], Sequence[str]]] = None,
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
    # additive_lod and substitutive_lod COMPOSE: substitutive chooses WHICH level
    # renders at the current zoom, additive describes HOW each level streams in.
    # Passing substitutive_lod alone ladders every level by default; pass
    # additive_lod=False to opt out. See lod/group.py's "Composed axes" section.
    if is_requested(substitutive_lod) and is_requested(partition):
        raise ValueError(
            "partition= and substitutive_lod= cannot be combined yet "
            "(Points supports a global overview LOD above partitioned fine detail; "
            "Lines does not implement that topology yet). Use one or the other."
        )
    try:
        # "An explicit None means absent" (#1574), applied ONCE here rather than
        # at each consumer — this module's own two ``sync_custom_colormap_attr``
        # call sites included — so no structural branch can see the raw None.
        # Only render attrs are in the set: the structural keys whose None also
        # means absent (``colors``/``labels``/``image_labels``/``partition``) are
        # named params of this function and can never reach ``**attrs``.
        strip_absent_attr_kwargs(attrs, ABSENT_WHEN_NONE_RENDER_ATTRS)

        # Fail-fast pre-write gate: reject invalid names (empty/'/'/dot-
        # prefixed — an empty name resolves to the zarr ROOT group and would
        # clobber the scene root) and duplicate siblings BEFORE any zarr
        # write. Node.__init__ re-checks both post-write (belt and braces).
        from ....validation.base import validate_node_name

        validate_node_name(name)
        (parent or group)._ensure_no_duplicate_child(name)
        reject_mismatched_partition_parent(parent or group, "lines", name)
        reject_mesh_only_appearance("lines", name, attrs)

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

        # Colormap / colors mutual exclusivity — validated BEFORE the
        # substitutive/partition/additive branches so every path rejects
        # invalid combinations (the LOD wrappers used to return early and
        # skip these checks entirely).
        if colors is not None and attrs.get("colormap") is not None:
            raise ValueError(
                "Cannot specify both 'colors' and 'colormap'. Use one or the other."
            )
        if scalars is not None and attrs.get("colormap") is None:
            raise ValueError(
                "'scalars' requires a 'colormap' attribute to map values to colors."
            )

        # Scene-dimension COUNT check — same placement and same reasons as in
        # ``add_points_impl``: above the substitutive/partition/additive branches
        # (so a mismatch is refused against the caller's own array and name,
        # before any wrapper group exists), after ``apply_dim_order_positions``
        # (which decides the final column count), and below the colours gate (so
        # a bad colours/colormap combination keeps precedence). Only the count
        # half is hoisted — the per-dimension range ``UserWarning`` stays in the
        # flat write below, so its count is unchanged (once per dimension per
        # written leaf; none under an additive ladder) rather than gaining one
        # more firing here for the source array. Do not move it back below the
        # branches.
        scene._validate_dimension_count(vert_arr, name, data_type="vertices")

        # Node-attrs gate — same validator the flat writer runs as its own first
        # step (write_lines' step 0a), hoisted here for the same reason as the
        # dimension-count check just above (#1446 is the model for placement;
        # Mesh and GSplats followed with the identical entry gate in #1534 —
        # ``add_mesh_impl`` / ``add_gsplats_impl``, likewise run once above
        # every structural branch and likewise on the live ``attrs`` rather
        # than a copy), and covering all three split paths below, not just
        # substitutive: substitutive_lod= forwards the non-compositing
        # remainder of `**attrs` to a synthesised gsplats `child_0`;
        # partition= forwards it to each `part_i`; additive_lod= goes straight
        # to the multi-LOD writer, which calls this same validator with NO
        # reserved-attrs set at all. A key
        # the flat writer would reject — gsplat-RESERVED-but-lines-unknown
        # (`amplitude_range=`; LINES_RESERVED_ATTRS doesn't carry it but
        # GSPLATS_RESERVED_ATTRS does), a plain typo (`blending=`), or (on the
        # additive path only) genuinely lines-RESERVED (`ordering=`,
        # `max_width=`, misreported as *unknown* rather than *reserved*) —
        # used to be refused only from inside the first child, by which point
        # the wrapper group itself (childless: zero coarse levels / parts) was
        # already on disk; a `position_bounds=` collision under additive_lod=
        # didn't even raise — it silently clobbered the writer's own stamp and
        # broke `finalize()` later, far from the actual cause. Below the
        # colours/dimension-count gates above (same precedence those already
        # keep: a bad colours/colormap combination or a dimension mismatch
        # outranks an attrs typo) and above every wrapper branch, so nothing is
        # written before it runs. This also deliberately outranks the #1437
        # channel gate at the top of the partition/substitutive/multi-LOD
        # wrappers below, each branch's own kwarg-spec checks (a malformed
        # `partition=`/`substitutive_lod=` spec, the `indices` topology
        # check), and `extend_to_all` resolution — including on the flat
        # fall-through below, where a call that trips both now reports the
        # attrs fault first (it used to report the extend_to_all fault; see
        # step 10 in README.md). The flat writer validates node attrs before
        # it ever looks at channels, so this hoist keeps that same order on
        # every path. The validator only inspects `attrs` and raises on the
        # first problem it finds — it is read-only, so running it here on the
        # live dict (not a copy) is safe. Calling the SAME validator here
        # means the split path refuses byte-identically to the flat path
        # below (which still runs it once more, inside write_lines —
        # idempotent).
        from ....io._compiler.node_common import (
            LINES_RESERVED_ATTRS,
            validate_render_attrs,
        )

        validate_render_attrs(attrs, reserved_attrs=LINES_RESERVED_ATTRS)

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
                # Topology first (see validate_line_indices_before_split): the
                # finest child is written LAST, so a malformed edge list was
                # refused only after the coarse levels were already on disk.
                validate_line_indices_before_split(indices, n_vertices, line_type)
                preflight_extend_to_all(scene, extend_to_all, vert_arr, "lines")
                return add_lines_substitutive_lod_wrapper_impl(
                    group,
                    name=name,
                    vert_arr=vert_arr,
                    widths=widths,
                    colors=colors,
                    sharpness=sharpness,
                    scalars=scalars,
                    labels=labels,
                    keys=keys,
                    image_labels=image_labels,
                    indices=indices,
                    line_type=line_type,
                    parent=parent,
                    extend_to_all=extend_to_all,
                    spec=substitutive_spec,
                    additive_lod=additive_lod,
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
        # A dataset with <2 spatial dims can't be split; drop the request
        # with a warning rather than in silence.
        if partition is not None:
            from ..partition import warn_if_partition_needs_more_dims

            if not warn_if_partition_needs_more_dims(vert_arr.shape[1], name):
                partition = None

        if partition is not None:
            from ..lod.lines import identify_polylines
            from ..partition import (
                resolve_partition_spec,
                warn_if_oversized_single_part,
            )

            max_elements, partition_rule = resolve_partition_spec(partition)

            if image_labels is not None:
                raise ValueError(
                    "image_labels is not supported alongside partition=. "
                    "Decompose the data manually or omit image_labels."
                )

            # Topology first, and BEFORE identify_polylines: it checks only
            # dtype and bounds and then reshapes to pairs, so a malformed edge
            # list is silently reinterpreted there (or dies on a raw reshape).
            validate_line_indices_before_split(indices, n_vertices, line_type)
            _verified_indexed_additive(
                additive_lod=additive_lod,
                line_type=line_type,
                indices=indices,
                n_vertices=n_vertices,
            )

            polyline_indices = identify_polylines(n_vertices, line_type, indices)

            tree, polyline_parts = _build_line_partition_tree(
                vert_arr,
                polyline_indices,
                n_vertices,
                max_elements,
                partition_rule,
                scene.dimensions.displayed,
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
                assert tree is not None
                preflight_extend_to_all(scene, extend_to_all, vert_arr, "lines")
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
                    keys=keys,
                    indices=indices,
                    line_type=line_type,
                    parent=parent,
                    extend_to_all=extend_to_all,
                    max_elements=max_elements,
                    bsp_tree=tree.to_serializable(),
                    additive_lod=additive_lod,
                    **attrs,
                )
            # 1 part → fall through to single-leaf write.

        # Additive-LOD branch — polyline-level multi-LOD write.
        # Fires before the single-shot write so we don't double-
        # validate. Mirrors the points add path.
        if (
            additive_lod is not None
            and additive_lod is not False
            and image_labels is not None
        ):
            from ..lod.lines import resolve_additive_axis_lines

            # The multi-LOD writer has no image_labels channel, so laddering
            # would silently drop them. Refuse the ladder, not the labels: fall
            # through to the single-leaf write below (which forwards
            # image_labels). This branch only fires for an EXPLICIT
            # ``additive_lod`` (the guard excludes ``None``/``False``), so warn
            # — the caller asked for something that cannot be honoured. Mirrors
            # the substitutive path's suppress_reason guard in
            # compose_additive_under_substitutive, which warns on the same
            # explicit-request condition. Still validate (and discard) the spec
            # so a malformed additive_lod= fails fast here exactly as it would
            # without image_labels.
            resolve_additive_axis_lines(additive_lod)
            warnings.warn(
                f"'{name}': the requested streaming ladder cannot be honoured "
                "(image_labels is set); writing a flat node.",
                UserWarning,
                stacklevel=2,
            )
        elif additive_lod is not None:
            from ..lod.lines import (
                make_additive_lod_lines,
                resolve_additive_axis_lines,
            )
            from ..lod.reveal import resolve_reveal_spatial_dims

            additive_spec = resolve_additive_axis_lines(additive_lod)
            if additive_spec is not None:
                validate_line_indices_before_split(indices, n_vertices, line_type)
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
                    reveal_centre=additive_spec.get("reveal_centre"),
                    spatial_dims=resolve_reveal_spatial_dims(
                        additive_spec, scene, vert_arr.shape[1]
                    ),
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
                        keys=keys,
                        parent=parent,
                        extend_to_all=extend_to_all,
                        method=additive_spec["method"],
                        counts=additive_spec["counts"],
                        widths_for_energy=widths_arr,
                        colors_for_energy=colors_for_energy,
                        scalars_for_energy=scalars_for_energy,
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
            keys=keys,
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
        # Un-nest BEFORE printing too, or arbol still echoes an internal
        # child name (`part_0`) that the raised exception no longer names
        # (#1491) — see funnel_add_error. A cross-geometry inner prefix (a
        # substitutive_lod= gsplats child) is deliberately left alone by
        # funnel_add_error, so it stays visible here too.
        inner = unnest_add_error("lines", name, e)
        aprint(f"Failed to add lines node '{name}': {inner}")
        raise ValueError(funnel_add_error("lines", name, e)) from e


def _collect_partition_vertex_indices(
    polyline_indices: List[np.ndarray],
    polyline_parts: List[List[int]],
) -> List[np.ndarray]:
    """Concatenate each part's atomic polylines in deterministic order."""
    part_vertex_indices: List[np.ndarray] = []
    for polyline_ids in polyline_parts:
        members = [
            polyline_indices[p] for p in polyline_ids if polyline_indices[p].size > 0
        ]
        part_vertex_indices.append(
            np.concatenate(members).astype(np.intp, copy=False)
            if members
            else np.empty(0, dtype=np.intp)
        )
    return part_vertex_indices


def _bucket_indexed_edge_indices(
    indices: np.ndarray,
    part_vertex_indices: List[np.ndarray],
    n_vertices: int,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Group indexed edges by part and build a global-to-local vertex map.

    Returns the global edge array, one stable edge-index view per part, and an
    ``(n_vertices,)`` map from global vertex index to its part-local index.
    Stability preserves the original authored edge order within each part. The
    persistent grouping storage is one permutation rather than millions of
    Python edge tuples; callers gather and remap one part at a time.
    """
    n_parts = len(part_vertex_indices)
    edges = np.asarray(indices, dtype=np.intp).reshape(-1, 2)
    vertex_map = np.full(n_vertices, -1, dtype=np.intp)

    for part_i, part_vertices in enumerate(part_vertex_indices):
        vertex_map[part_vertices] = part_i

    if edges.size == 0:
        edge_buckets = [np.empty(0, dtype=np.intp) for _ in range(n_parts)]
    else:
        edge_parts = vertex_map[edges[:, 0]]
        valid = (edge_parts >= 0) & (vertex_map[edges[:, 1]] == edge_parts)
        all_valid = bool(np.all(valid))
        valid_edge_indices: Optional[np.ndarray] = None
        if all_valid:
            valid_edge_parts = edge_parts
        else:
            valid_edge_indices = np.flatnonzero(valid).astype(np.intp, copy=False)
            valid_edge_parts = edge_parts[valid]

        part_order = np.argsort(valid_edge_parts, kind="stable").astype(
            np.intp, copy=False
        )
        if all_valid:
            edge_order = part_order
        else:
            assert valid_edge_indices is not None
            edge_order = valid_edge_indices[part_order]
        counts = np.bincount(valid_edge_parts, minlength=n_parts)
        offsets = np.concatenate(
            (
                np.array([0], dtype=np.intp),
                np.cumsum(counts, dtype=np.intp),
            )
        )
        edge_buckets = [
            edge_order[int(offsets[i]) : int(offsets[i + 1])] for i in range(n_parts)
        ]

    # The part-id map is no longer needed. Reuse its buffer for local vertex
    # indices instead of allocating a second n_vertices-sized array. Callers
    # consume buckets and part_vertex_indices in this same part order.
    vertex_to_local = vertex_map
    for part_vertices in part_vertex_indices:
        vertex_to_local[part_vertices] = np.arange(part_vertices.size, dtype=np.intp)

    return edges, edge_buckets, vertex_to_local


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
    keys: Any = None,
    indices: Optional[np.ndarray],
    line_type: str,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    max_elements: int,
    bsp_tree: Dict[str, Any],
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
    crash). A part containing only isolated vertices (no edges) has
    nothing drawable — indexed rendering never references it — and is
    skipped. For ``polyline`` / ``loop`` types (where the input is a single
    polyline), the BSP only ever produces one part — the user is already at
    the single-polyline granularity and there's nothing to partition. We
    refuse the partition in that case with a clear error.
    """
    # Entering a wrapper IS "a split is about to happen": from here on every
    # per-VERTEX channel is sliced per part, and `slice_optional_array` passes a
    # wrong-length value through whole. Scoped to the split paths so the
    # plain-leaf CHANNEL-gate order is untouched. The one exception is the
    # node-attrs gate (#1529, add_lines_impl above): it runs at the adder
    # entry, before this wrapper is even chosen, so it outranks this channel
    # gate too — a call that trips both reports the attrs fault.
    validate_lines_channels_before_split(
        n_vertices,
        widths=widths,
        colors=colors,
        sharpness=sharpness,
        scalars=scalars,
        labels=labels,
        keys=keys,
    )

    # A uniform RGB(A) list/tuple is the one leaf parameter whose OWN length can
    # collide with the vertex count, so classify it up front instead of letting
    # the length test gather it (see compositing.is_broadcast_color).
    uniform_color = is_broadcast_color(colors)

    wrapper_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    leaf_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}

    # An indexed graph with no edges has nothing drawable in ANY part (every
    # vertex is isolated), so every leaf below would be skipped. Refuse it
    # up front with the same error the single-leaf writer raises, instead of
    # silently writing an empty partition group.
    if line_type == "indexed" and (indices is None or np.asarray(indices).size == 0):
        raise ValueError("Indexed requires at least 2 indices")

    parent_node = parent or group
    wrapper = parent_node.add_partition_group(
        name=name,
        display_type="lines",
        max_elements=max_elements,
        **wrapper_attrs,
    )

    part_vertex_indices = _collect_partition_vertex_indices(
        polyline_indices, polyline_parts
    )
    part_sizes = [int(part_vertices.size) for part_vertices in part_vertex_indices]
    aprint(
        f"  ✂️  Partitioned '{name}' into {len(polyline_parts)} parts via "
        f"polyline-centroid BSP "
        f"(max_elements={max_elements:,}, sizes={part_sizes})"
    )

    # For ``indexed`` inputs, bucket the ORIGINAL edges by part up front.
    # Each connected component (hence each edge) lands wholly in one part.
    # The grouping is a stable NumPy permutation, not one Python tuple per
    # edge; remapping uses one reusable global-to-local array, not per-part
    # dictionaries. Exact authored topology and edge order are preserved.
    indexed_edges: Optional[np.ndarray] = None
    part_edge_indices: List[np.ndarray] = []
    vertex_to_local: Optional[np.ndarray] = None
    if line_type == "indexed" and indices is not None:
        indexed_edges, part_edge_indices, vertex_to_local = (
            _bucket_indexed_edge_indices(
                indices,
                part_vertex_indices,
                n_vertices,
            )
        )

    # The new line_type per part is either:
    # - ``polyline`` / ``loop`` with one polyline ⇒ keep as-is.
    # - ``segments`` ⇒ re-emit as ``segments`` (consecutive member pairs).
    # - ``indexed`` ⇒ re-emit as ``indexed`` with the part's real edges
    #   remapped to part-local vertex indices.
    written_parts = []
    for i, part_vertex_idx in enumerate(part_vertex_indices):
        if part_vertex_idx.size == 0:
            continue
        part_vertices = vert_arr[part_vertex_idx]

        # Remap this part's original edges to part-local vertex indices. The
        # gather is bounded by one part; the global edge order and vertex map
        # remain shared across all parts.
        part_indices: Optional[np.ndarray] = None
        if line_type == "indexed":
            assert indexed_edges is not None
            assert vertex_to_local is not None
            edge_ids = part_edge_indices[i]
            if edge_ids.size > 0:
                part_indices = vertex_to_local[indexed_edges[edge_ids]].reshape(-1)

        # Slice per-vertex parameters into this part.
        part_widths = (
            widths
            if (not isinstance(widths, np.ndarray)) or widths.shape != (n_vertices,)
            else widths[part_vertex_idx]
        )
        part_colors = (
            colors
            if uniform_color
            else slice_optional_array(colors, part_vertex_idx, n_vertices)
        )
        part_sharpness = slice_optional_array(sharpness, part_vertex_idx, n_vertices)
        part_scalars = slice_optional_array(scalars, part_vertex_idx, n_vertices)
        part_labels = slice_optional_array(labels, part_vertex_idx, n_vertices)

        # Choose the per-part line_type. ``polyline`` / ``loop`` with
        # one polyline = one part, so the original type is preserved.
        # ``segments`` emits segments. ``indexed`` emits the part's
        # remapped real edges (a part with no edges at all — only
        # isolated vertices — has nothing drawable and is skipped).
        if line_type in ("polyline", "loop"):
            part_line_type = line_type
            part_indices = None
        elif line_type == "segments":
            part_line_type = "segments"
            part_indices = None
        else:  # indexed
            part_line_type = "indexed"
            if part_indices is None:
                # The part holds only isolated vertices (single-vertex
                # components) — an indexed graph draws nothing for a vertex
                # no segment references, so there is no leaf to write.
                # Degrading to ``segments`` here used to FABRICATE visible
                # edges between distinct isolated vertices, desync the
                # per-vertex attributes on odd-sized parts, and crash on
                # one-vertex parts (trimmed to an empty write).
                continue

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
        written_parts.append(i)

    from ..partition import persist_pruned_bsp_tree

    persist_pruned_bsp_tree(wrapper, bsp_tree, written_parts)

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
    keys: Any = None,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    method: str,
    counts: Any = None,
    widths_for_energy: Optional[np.ndarray] = None,
    colors_for_energy: Optional[np.ndarray] = None,
    scalars_for_energy: Optional[np.ndarray] = None,
    **attrs: Any,
) -> Lines:
    """Write a Lines node with multi-additive-LOD subgroups.

    Each ``additive_<i>/`` subgroup carries a subset of polylines
    (whole polylines, never bisected). Segment indices are local to
    each subgroup. The parent lines node carries
    ``n_additive_sublods``, the global ``position_bounds``, and the
    standard compositing attrs.

    ``labels`` (per-vertex) are written by the writer as ONE union CSR on the
    parent (the subgroups carry none — the loader concatenates levels into one
    buffer), so the scene is notified here exactly as on the flat path.

    ``counts`` and the ``*_for_energy`` arrays only feed the ladder's quality
    stamps — see the Points twin for what the viewer does with them.
    """
    # See add_lines_partition_wrapper_impl: every per-vertex channel is about to
    # be sliced per level, and a per-level length check cannot catch a
    # wrong-length value whose length happens to match some level's vertex count.
    validate_lines_channels_before_split(
        n_vertices,
        widths=widths,
        colors=colors,
        sharpness=sharpness,
        scalars=scalars,
        labels=labels,
        keys=keys,
    )
    uniform_color = is_broadcast_color(colors)

    scene = group._find_scene()
    writer = group._require_scene_writer(scene)
    parent_node = parent or group
    path = f"{parent_node.path}/{name}" if parent_node.path else name

    # Ladder quality stamps, in the Lines energy currency (mean luminance x
    # tube volume per polyline). Energy is per-POLYLINE here, so the levels are
    # summed over their member polylines; the flattened polyline list is
    # rebuilt in level order, which is exactly the order the builder sliced.
    from ..lod.group import (
        additive_level_stats,
        breakpoints_kind_of,
        resolve_ladder_extend_to_all,
    )
    from ..lod.lines import compute_lines_energy

    flat_polylines = [poly for level in polyline_levels for poly in level]
    poly_energy = compute_lines_energy(
        vert_arr,
        flat_polylines,
        widths_for_energy,
        colors_for_energy,
        scalars_for_energy,
    )
    level_energies: List[float] = []
    cursor = 0
    for level in polyline_levels:
        level_energies.append(float(poly_energy[cursor : cursor + len(level)].sum()))
        cursor += len(level)
    per_level_stats, _reference_energy, parent_level_stats = additive_level_stats(
        level_energies,
        [int(sum(int(poly.size) for poly in level)) for level in polyline_levels],
        method=method,
        breakpoints_kind=breakpoints_kind_of(counts),
        energy_kind="lines-tube-volume",
    )
    caller_level_stats = attrs.get("level_stats")
    if caller_level_stats is None:
        attrs["level_stats"] = parent_level_stats
    elif "reference_energy" in parent_level_stats:
        # The ladder stamped energy_fraction_cum on every sub-LOD, so the parent
        # must carry the paired reference_energy (both-or-neither); a caller dict
        # that omits it would silently break the pairing. The caller dict rides
        # through the writer's JSON-safety guard: a non-finite caller value
        # (NaN/±Inf) is dropped — the computed reference_energy then shows
        # through — instead of reaching .zattrs as a bare NaN token the viewer's
        # strict JSON.parse rejects. Finite caller keys still win.
        from ....io._compiler.gsplat_tree import json_safe_value

        _, safe_caller = json_safe_value(caller_level_stats)
        attrs["level_stats"] = {
            "reference_energy": parent_level_stats["reference_energy"],
            **(safe_caller or {}),
        }

    level_slices: List[Dict[str, Any]] = []
    for level_i, level_polylines in enumerate(polyline_levels):
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
                "colors": colors
                if uniform_color
                else slice_optional_array(colors, vertex_index_arr, n_vertices),
                "sharpness": slice_optional_array(
                    sharpness, vertex_index_arr, n_vertices
                ),
                "scalars": slice_optional_array(scalars, vertex_index_arr, n_vertices),
                "labels": slice_optional_array(labels, vertex_index_arr, n_vertices),
                "keys": slice_optional_array(keys, vertex_index_arr, n_vertices),
                "segments": level_segments,
                "n_polylines": len(level_polylines),
                "lod_stats": per_level_stats[level_i],
            }
        )

    aprint(
        f"  📐 Additive-LOD '{name}': {len(polyline_levels)} levels "
        f"(method={method!r}, polylines_per_level="
        f"{[len(L) for L in polyline_levels]})"
    )

    # Nothing else resolves the ``"all"`` sentinel on this branch (unlike the
    # partition wrapper, we do not recurse through the leaf adder) and the
    # writer stamps the value VERBATIM onto the parent group AND every
    # ``additive_<i>/`` sub-LOD — see ``resolve_ladder_extend_to_all`` for why
    # ``None`` stays unresolved. Must stay an explicit kwarg: in ``attrs`` it
    # would collide.
    final_extend_dims = resolve_ladder_extend_to_all(
        scene, extend_to_all, vert_arr, "lines"
    )

    metadata = writer.write_lines_multi_lod(
        path,
        level_slices,
        extend_to_all=final_extend_dims,
        **attrs,
    )

    # Ladder labels live in one CSR on the parent node, so the scene needs the
    # same hover-overlay injection the flat path gets.
    if labels is not None:
        scene._notify_labels_added()

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
    keys: Any = None,
    image_labels: Any,
    indices: Optional[np.ndarray],
    line_type: str,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    spec: Dict[str, Any],
    additive_lod: Any = None,
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
    # Before the lift: the coarse levels cost a full gsplat reduce, and the
    # finest child (written LAST, after every coarse level is already on disk) is
    # where a wrong-length channel would otherwise be caught — stranding a
    # partial kind=lod group. Fail here instead, before anything is written.
    # The node-attrs gate (#1529, add_lines_impl above) already ran at the
    # adder entry, before this wrapper was even chosen, so it outranks this
    # channel gate too — a call that trips both reports the attrs fault.
    validate_lines_channels_before_split(
        int(vert_arr.shape[0]),
        widths=widths,
        colors=colors,
        sharpness=sharpness,
        scalars=scalars,
        labels=labels,
        keys=keys,
        image_labels=image_labels,
    )

    from ....gsplats.lift import coarse_substitutive_levels, lift_lines_to_gsplats
    from ..lod.group import (
        compose_additive_under_substitutive,
        gsplat_additive_lod_from,
        level_additive_lod,
        resident_slice_count,
        resolve_lod_ladder,
    )
    from ..lod.lines import indexed_components_are_chains, resolve_additive_axis_lines

    # Resolve the coarse and finest policies independently. The coarse children
    # are gsplat clouds and can always stream; the suppression reasons below are
    # properties of the original Lines child only.
    # Keep the existing element-domain stream counts for composed coarse
    # children. Retuning those counts for gsplat bytes-per-element is a separate
    # cross-geometry policy change, not part of suppression scoping.
    # A default ladder's first chunk is a whole-node download budget, so the
    # slice count is what keeps it from arriving divided on an nD node (#2374).
    slices = (
        resident_slice_count(group._find_scene(), vert_arr)
        if additive_lod is None
        else 1
    )
    coarse_additive = compose_additive_under_substitutive(
        additive_lod,
        resolve=resolve_additive_axis_lines,
        elements=int(vert_arr.shape[0]),
        name=name,
        slices=slices,
    )
    finest_suppress_reason = (
        # The multi-LOD writer has no image_labels channel, so laddering would
        # silently drop them. Refuse the ladder, not the labels.
        "image_labels is set"
        if image_labels is not None
        # One polyline cannot be split without breaking segment topology, so a
        # ladder here is a no-op the builder would only warn about.
        else f"line_type={line_type!r} is a single polyline"
        if line_type in ("polyline", "loop")
        # Indexed lines carry an explicit edge list the additive multi-LOD writer
        # discards — it re-derives one by chaining each connected component in
        # ascending vertex order (see lod/lines.py::_indexed_connected_components),
        # which is faithful exactly when every component's undirected edge
        # multiset, including duplicate multiplicity, equals its consecutive
        # vertex pairs. Test that contract rather than refusing every indexed node.
        else (
            f"line_type={line_type!r} has an explicit edge multiset that does not "
            "equal its consecutive vertex pairs, so the ladder would rewrite edges"
        )
        if additive_lod is not False
        and line_type == "indexed"
        and not (
            indices is not None
            and indexed_components_are_chains(
                int(vert_arr.shape[0]),
                np.asarray(indices, dtype=np.intp).reshape(-1, 2),
            )
        )
        else None
    )
    from ..lod.reveal import is_reveal_additive_method

    reveal_note = (
        " Coarse levels use self_energy ordering, so reveal_centre is not applied."
        if coarse_additive is not None
        and is_reveal_additive_method(str(coarse_additive.get("method")))
        else ""
    )
    finest_additive = compose_additive_under_substitutive(
        additive_lod,
        resolve=resolve_additive_axis_lines,
        elements=int(vert_arr.shape[0]),
        name=name,
        slices=slices,
        suppress_reason=finest_suppress_reason,
        suppression_outcome=(
            "the finest level will load all-at-once; coarse levels keep their "
            "ladder where one applies."
            f"{reveal_note}"
        ),
    )
    # Same reason as the channel check above: the finest child is written LAST, so
    # a reveal_centre that does not match the DERIVED shell axes would otherwise
    # raise once every coarse level is already on disk. The scorer ranks whole
    # polylines, so the array whose extent decides the shell axes is the per-
    # polyline bbox CENTRES, not the vertices.
    #
    # Guarded on `wants_reveal_centre_preflight` rather than on
    # `finest_additive is not None`, because deriving those representatives is
    # NOT free: identify_polylines + polyline_bbox_centres loop in Python over
    # every polyline (~2.5 s for a 400k-vertex `segments` node), and the composed
    # ladder defaults to ON — so the unguarded form paid that on every
    # `add_lines(substitutive_lod=…)` call, reveal or not.
    from ..lod.reveal import preflight_reveal_centre, wants_reveal_centre_preflight

    if wants_reveal_centre_preflight(finest_additive):
        from ..lod.lines import identify_polylines, polyline_bbox_centres

        preflight_reveal_centre(
            finest_additive,
            group._find_scene(),
            polyline_bbox_centres(
                vert_arr,
                identify_polylines(int(vert_arr.shape[0]), line_type, indices),
            ),
            "vertices",
        )
    compression_factor = int(spec["compression_factor"])

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
            keys=keys,
            image_labels=image_labels,
            indices=indices,
            line_type=line_type,
            parent=parent,
            extend_to_all=extend_to_all,
            partition=False,
            # Forward the ladder: too small to coarsen is not too small to
            # stream, and dropping it here silently lost the ladder.
            additive_lod=finest_additive,
            **attrs,
        )

    coarse_first = list(reversed(coarse))  # coarsest first
    # Finest "count" is the full lifted BEAD count, not n_vertices: the coarse
    # gsplat children are bead reductions (N_beads/K^l), so measuring the finest
    # in the same bead currency keeps the ladder counts strictly ascending and
    # gives the real Lines node the highest switch threshold (a vertex count is
    # a different, much smaller scale and would collapse the top thresholds).
    counts = [int(c.n_splats) for c in coarse_first] + [int(lifted.n_splats)]
    parent_node = parent or group
    # Thresholds AND the selector naming their units, from the one shared rule
    # (``lod.group.resolve_lod_ladder``): an explicit ``coverage_fractions=[...]``
    # is used verbatim under the legacy units it was authored in, otherwise the
    # screen-area halving ladder is derived — re-anchored at fills-screen when the
    # insertion point is partition-bound. ``add_lines`` rejects ``partition=``
    # together with ``substitutive_lod=``, but a caller CAN hand-build a
    # ``kind=partition`` wrapper and call this once per part — the shape
    # ``demo_biodiversity_planetary_scale`` uses for its Points layers (its
    # ``add_lines`` calls take the plain scene-level ``partition=`` path instead,
    # with no ladder) — which is how that anchor is reached here.
    coverage_vals, lod_selector = resolve_lod_ladder(
        spec.get("coverage_fractions"),
        counts,
        parent_node,
        name=name,
        length_error=lambda n_explicit, n_levels: (
            f"coverage_fractions has {n_explicit} entries but the LOD ladder "
            f"has {n_levels} levels ({len(coarse_first)} gsplat + 1 lines)"
        ),
    )

    lod_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    child_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}
    child_attrs.pop("coverage_fraction", None)
    lod_attrs.setdefault("display_type", "lines")  # finest child is lines
    # Coarse gsplat children carry baked per-splat colours, so they must NOT also
    # receive `colormap` (gsplats reject colors+colormap); it stays on the finest
    # Lines child only.
    gsplat_child_attrs = {k: v for k, v in child_attrs.items() if k != "colormap"}

    aprint(
        f"  📐 Substitutive-LOD '{name}': {len(coarse_first)} gsplat levels + lines "
        f"(counts coarsest→finest={counts}, K={spec['compression_factor']})"
    )
    lod_group_node = parent_node.add_lod_group(name, selector=lod_selector, **lod_attrs)

    # Coarse gsplat children (coarsest first). vert_arr already dim_order-applied.
    for idx, lvl_data in enumerate(coarse_first):
        # Every level gets its own ladder (mirrors gsplats/lod/pyramid.py), with
        # the sibling-aware first chunk on all but the coarsest.
        lvl_spec = level_additive_lod(
            coarse_additive,
            level_n=int(lvl_data.n_splats),
            compression_factor=compression_factor,
            is_coarsest=(idx == 0),
            slices=slices,
        )
        lod_group_node.add_gsplats_from_data(
            name=f"child_{idx}",
            result=lvl_data,
            extend_to_all=extend_to_all,
            dim_order=None,
            fill=None,
            lod_group=None,
            additive_lod=gsplat_additive_lod_from(lvl_spec, int(lvl_data.n_splats)),
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
        keys=keys,
        image_labels=image_labels,
        indices=indices,
        line_type=line_type,
        extend_to_all=extend_to_all,
        dim_order=None,
        fill=None,
        additive_lod=level_additive_lod(
            finest_additive,
            level_n=int(vert_arr.shape[0]),
            compression_factor=compression_factor,
            is_coarsest=False,
            slices=slices,
        ),
        substitutive_lod=None,
        partition=False,
        coverage_fraction=coverage_vals[-1],
        **child_attrs,
    )

    return lod_group_node
