"""add_gsplats body + partition wrapper impl.

Pure functions taking a ``group: Group`` parameter as the first arg.
Called by ``Group.add_gsplats`` (a thin signature + docstring + delegate)
in ``core/group/group.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union, cast

import numpy as np
from arbol import aprint

from ...gsplats import GSplats
from ..auto_partition import resolve_auto_partition
from ..compositing import (
    ABSENT_WHEN_NONE_RENDER_ATTRS,
    COMPOSITING_ATTRS,
    funnel_add_error,
    is_broadcast_color,
    mirror_written_colormap,
    position_bounds_from_array,
    preflight_extend_to_all,
    reject_layer_order_inside_specialized_group,
    reject_lines_only_join,
    reject_mesh_only_appearance,
    slice_optional_array,
    strip_absent_attr_kwargs,
    sync_custom_colormap_attr,
    unnest_add_error,
    validate_gsplats_channels_before_split,
)
from ..dim_order import apply_dim_order_cholesky, apply_dim_order_positions
from ..partition import reject_mismatched_partition_parent

if TYPE_CHECKING:
    from ...node import Node
    from ..group import Group


def _normalize_scene_label_channel(
    label_ids: Any,
    label_vocabulary: Optional[Dict[int, str]],
    n_splats: int,
) -> tuple[Optional[np.ndarray], Optional[Dict[int, str]]]:
    from ....gsplats.gsplat_data import validate_label_channel

    normalized_ids = None if label_ids is None else np.asarray(label_ids)
    normalized_vocabulary = validate_label_channel(
        normalized_ids, label_vocabulary, n_splats
    )
    return normalized_ids, normalized_vocabulary


def add_gsplats_impl(
    group: "Group",
    *,
    name: str,
    centers: Any,
    amplitudes: Any,
    cholesky_factors: Any,
    colors: Any = None,
    label_ids: Any = None,
    label_vocabulary: Optional[Dict[int, str]] = None,
    labels: Optional[Union[List[str], Sequence[str]]] = None,
    image_labels: Optional[Any] = None,
    keys: Optional[Union[List[str], Sequence[str]]] = None,
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    fill_sigma: Optional[Dict[str, float]] = None,
    partition: Any = None,
    **attrs: Any,
) -> Union[GSplats, "Group"]:
    try:
        # "An explicit None means absent" (#1574), applied ONCE here rather than
        # at each consumer, so neither the flat write nor the partition branch
        # (nor this module's one ``sync_custom_colormap_attr`` call) can see the
        # raw None. The partition branch matters as much as the flat write:
        # it forwards this same dict on to every ``part_i``, so a strip scoped
        # to the flat path leaves the raw None in each part's write — measured,
        # all four parts ship the LUT-less ``'custom'``.
        # This is also what makes the leaf agree with the pipeline door above it:
        # ``add_gsplats_from_data`` strips the same way before delegating here,
        # so a ``colormap=None`` no longer means one thing through the data door
        # and another through this one. Only render attrs are in the set — the
        # structural keys whose None also means absent
        # (``colors``/``labels``/``image_labels``/``partition``) are named params
        # of this function and can never reach ``**attrs``.
        strip_absent_attr_kwargs(attrs, ABSENT_WHEN_NONE_RENDER_ATTRS)

        # Fail-fast pre-write gate: reject invalid names (empty/'/'/dot-
        # prefixed — an empty name resolves to the zarr ROOT group and would
        # clobber the scene root) and duplicate siblings BEFORE any zarr
        # write. Node.__init__ re-checks both post-write (belt and braces).
        from ....validation.base import validate_node_name

        validate_node_name(name)
        (parent or group)._ensure_no_duplicate_child(name)
        reject_mismatched_partition_parent(parent or group, "gsplats", name)
        reject_layer_order_inside_specialized_group(
            "gsplats", name, attrs, parent or group
        )
        reject_lines_only_join("gsplats", name, attrs)
        reject_mesh_only_appearance("gsplats", name, attrs)

        scene = group._find_scene()

        ctr_arr: np.ndarray = (
            centers if isinstance(centers, np.ndarray) else np.asarray(centers)
        )
        if ctr_arr.ndim != 2:
            raise ValueError(
                f"Centers must have shape (N, D), got shape {ctr_arr.shape}"
            )

        d_data = ctr_arr.shape[1]

        chol_arr: np.ndarray = (
            cholesky_factors
            if isinstance(cholesky_factors, np.ndarray)
            else np.asarray(cholesky_factors)
        )

        # Apply dim_order: transform both centers and cholesky_factors
        if dim_order is not None:
            ctr_arr, extend_to_all = apply_dim_order_positions(
                ctr_arr, scene, dim_order, fill, extend_to_all
            )
            chol_arr = apply_dim_order_cholesky(
                chol_arr, d_data, scene, dim_order, fill_sigma
            )

        n_splats = ctr_arr.shape[0]
        ndim = ctr_arr.shape[1]
        label_ids, label_vocabulary = _normalize_scene_label_channel(
            label_ids, label_vocabulary, n_splats
        )

        # Colormap / colors mutual exclusivity — validated BEFORE the partition
        # branch, as the Points and Lines adders do. Checked after the branch it
        # was refused only from inside ``part_0``, leaving the store holding a
        # childless ``kind=partition`` group where the plain-leaf path writes
        # nothing at all. This also moves it above ``resolve_auto_partition`` and
        # ``_validate_data_dimensions``, so it takes precedence over a dimension
        # mismatch on the FLAT path too — which is exactly the precedence the two
        # siblings already have, and the point of the move is that all three
        # adders answer this the same way.
        if colors is not None and attrs.get("colormap") is not None:
            raise ValueError(
                "Cannot specify both 'colors' and 'colormap'. Use one or the other."
            )

        # Scene-dimension COUNT check — above the partition branch (so a mismatch
        # is refused against the caller's own array and name instead of from
        # inside ``part_0``, which left a childless ``kind=partition`` group on
        # disk), after the dim_order transform (which decides the final column
        # count), and below the colours gate so that refusal keeps precedence as
        # documented above. Only the count half is hoisted — the per-dimension
        # range ``UserWarning`` stays in the flat write below, so its count is
        # unchanged (once per dimension per written leaf) rather than gaining one
        # more firing here for the source array. Do not move it back below the
        # branch.
        scene._validate_dimension_count(ctr_arr, name, data_type="centers")

        # Node-attrs gate (#1534) — the GSplats peer of the Points/Lines hoist
        # (#1529). ``partition=`` is this adder's only split path (this module
        # has no substitutive_lod=/additive_lod=/lod_group= door — those live
        # on ``add_gsplats_from_data``, a different adder), and it forwards the
        # non-compositing remainder of ``**attrs`` to each synthesised
        # ``part_i`` — so a bad attr used to be refused only from inside the
        # first part, by which point the wrapper's childless ``kind=partition``
        # group was already on disk. Below the colours and dimension-count
        # gates above (same precedence those already keep) and above the
        # partition branch, so nothing is written before it runs. The flat
        # writer below still validates the same dict once more inside
        # ``write_gsplats`` — the validator is read-only, so running it here on
        # the live ``attrs`` (not a copy) is safe and idempotent.
        from ....io._compiler.node_common import (
            GSPLATS_RESERVED_ATTRS,
            validate_render_attrs,
        )

        validate_render_attrs(attrs, reserved_attrs=GSPLATS_RESERVED_ATTRS)

        # Apply compiler-level auto-partition heuristic (opt-in; default
        # off). User-explicit ``partition=`` always wins.
        partition = resolve_auto_partition(scene, n_splats, partition)

        # Partition branch — decompose into N children if the user opted in
        # AND the BSP produces more than one part.
        # A dataset with <2 spatial dims can't be split; drop the request
        # with a warning rather than in silence.
        if partition is not None:
            from ..partition import warn_if_partition_needs_more_dims

            if not warn_if_partition_needs_more_dims(ndim, name):
                partition = None

        if partition is not None and n_splats > 0:
            from ..partition import (
                bsp_leaf_parts,
                resolve_partition_spec,
                spatial_bsp_tree,
                warn_if_oversized_single_part,
            )

            max_elements, partition_rule = resolve_partition_spec(partition)

            if image_labels is not None:
                raise ValueError(
                    "image_labels is not supported alongside partition=. "
                    "Decompose the data manually or omit image_labels."
                )

            tree = spatial_bsp_tree(
                ctr_arr,
                max_elements,
                rule=partition_rule,
                split_axes=scene.dimensions.displayed,
            )
            parts = bsp_leaf_parts(tree)
            warn_if_oversized_single_part(
                len(parts), int(parts[0].size) if parts else 0, max_elements, name
            )
            if len(parts) > 1:
                preflight_extend_to_all(scene, extend_to_all, ctr_arr, "splats")
                return add_gsplats_partition_wrapper_impl(
                    group,
                    name=name,
                    ctr_arr=ctr_arr,
                    chol_arr=chol_arr,
                    amplitudes=amplitudes,
                    parts=parts,
                    n_splats=n_splats,
                    colors=colors,
                    label_ids=label_ids,
                    label_vocabulary=label_vocabulary,
                    labels=labels,
                    keys=keys,
                    parent=parent,
                    extend_to_all=extend_to_all,
                    max_elements=max_elements,
                    bsp_tree=tree.to_serializable(),
                    **attrs,
                )
            # 1 part → fall through to single-leaf write.

        aprint(f"Adding gsplats node '{name}' with {n_splats:,} splats in {ndim}D.")

        scene._validate_data_dimensions(ctr_arr, name, data_type="centers")

        final_extend_dims = scene._resolve_extend_to_all(
            extend_to_all, ctr_arr, "splats"
        )
        if final_extend_dims:
            attrs["extend_to_all"] = final_extend_dims
            aprint(f"  📡 Extending visibility across: {final_extend_dims}")

        parent_node = parent or group

        writer = group._require_scene_writer(scene)
        path = f"{parent_node.path}/{name}" if parent_node.path else name
        metadata = writer.write_gsplats(
            path,
            ctr_arr.astype(np.float32),
            amplitudes=amplitudes,
            cholesky_factors=chol_arr,
            colors=cast(Any, colors),
            label_ids=label_ids,
            label_vocabulary=label_vocabulary,
            labels=labels,
            keys=keys,
            image_labels=image_labels,
            **attrs,
        )

        # Sync colormap attr with what the compiler wrote to zarr
        sync_custom_colormap_attr(attrs)

        # The compiler sets a default "gray" colormap for gsplats without
        # colors/colormap — unless an ancestor authored a palette, which the
        # gray would shadow (#1600). Mirror whatever it actually wrote onto the
        # Node attrs, so the in-memory node matches the zarr state and its own
        # attr write-back cannot put on disk what the writer declined.
        mirror_written_colormap(attrs, writer, path)

        if labels is not None:
            scene._notify_labels_added()
        if image_labels is not None:
            scene._notify_image_labels_added()

        return GSplats(
            name,
            metadata=metadata,
            parent=cast(Any, parent_node),
            writer=writer,
            **attrs,
        )
    except (ValueError, TypeError) as e:
        # Un-nest BEFORE printing too, or arbol still echoes an internal
        # child name (`part_0`) that the raised exception no longer names
        # (#1491) — see funnel_add_error.
        inner = unnest_add_error("gsplats", name, e)
        aprint(f"Failed to add gsplats node '{name}': {inner}")
        raise ValueError(funnel_add_error("gsplats", name, e)) from e


def add_gsplats_partition_wrapper_impl(
    group: "Group",
    *,
    name: str,
    ctr_arr: np.ndarray,
    chol_arr: np.ndarray,
    amplitudes: Any,
    parts: List[np.ndarray],
    n_splats: int,
    colors: Any,
    label_ids: Any,
    label_vocabulary: Optional[Dict[int, str]],
    labels: Any,
    keys: Any = None,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    max_elements: int,
    bsp_tree: Dict[str, Any],
    **attrs: Any,
) -> "Group":
    """Build a kind=partition wrapper Group with one GSplats child per BSP part."""
    # Entering a wrapper IS "a split is about to happen": from here on every
    # per-splat channel is sliced per part, and `slice_optional_array` passes a
    # wrong-length value through whole — which would give every part the SAME
    # labels/colors/amplitudes (part 1's tooltips would be part 0's). GSplats
    # have no additive/substitutive wrapper here (gsplat LOD goes through
    # GSplatData, already length-consistent), so this is the only split path.
    #
    # Two per-splat parameters have a broadcast form whose OWN length can collide
    # with the splat count, so classify both up front rather than letting
    # `slice_optional_array`'s length test gather them: a uniform RGB(A)
    # list/tuple (3 or 4 splats), and a UNIFORM 1-D Cholesky of shape (k,) —
    # k = D(D+1)/2, so 6 for 3-D data, which a 6-splat node matches exactly.
    # Parts are disjoint here, so the gathered slice never has a legal length and
    # the symptom is a REFUSED legal input (see compositing.is_broadcast_color),
    # not a silent mis-write.
    #
    # The Cholesky verdict comes BACK from the gate rather than being recomputed:
    # `validate_gsplat_inputs` already decided it, and a second copy of the rule
    # here is the drift this whole gate exists to prevent.
    uniform_cholesky = validate_gsplats_channels_before_split(
        ctr_arr,
        amplitudes,
        chol_arr,
        colors=colors,
        labels=labels,
        keys=keys,
    )
    uniform_color = is_broadcast_color(colors)

    wrapper_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    leaf_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}

    parent_node = parent or group
    wrapper = parent_node.add_partition_group(
        name=name,
        display_type="gsplats",
        max_elements=max_elements,
        **wrapper_attrs,
    )

    aprint(
        f"  ✂️  Partitioned '{name}' into {len(parts)} parts via BSP "
        f"(max_elements={max_elements:,}, "
        f"sizes={[int(p.size) for p in parts]})"
    )

    for i, indices in enumerate(parts):
        wrapper.add_gsplats(
            name=f"part_{i}",
            centers=ctr_arr[indices],
            amplitudes=slice_optional_array(amplitudes, indices, n_splats),
            cholesky_factors=chol_arr
            if uniform_cholesky
            else slice_optional_array(chol_arr, indices, n_splats),
            colors=colors
            if uniform_color
            else slice_optional_array(colors, indices, n_splats),
            label_ids=slice_optional_array(label_ids, indices, n_splats),
            label_vocabulary=label_vocabulary,
            labels=slice_optional_array(labels, indices, n_splats),
            keys=slice_optional_array(keys, indices, n_splats),
            image_labels=None,
            extend_to_all=extend_to_all,
            # dim_order / fill / fill_sigma already applied to ctr_arr +
            # chol_arr upstream — do not re-apply in the per-part call.
            dim_order=None,
            fill=None,
            fill_sigma=None,
            # ``partition=False`` bypasses compiler auto-partition (see
            # add_points_partition_wrapper_impl for rationale).
            partition=False,
            **leaf_attrs,
        )
    from ..partition import persist_pruned_bsp_tree

    # Unlike lines, every resolved gsplats part is written in order.
    persist_pruned_bsp_tree(wrapper, bsp_tree, range(len(parts)))

    wrapper._persist_attr("position_bounds", position_bounds_from_array(ctr_arr))

    return wrapper
