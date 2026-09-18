"""add_points body + partition/multi-LOD wrapper impls.

Pure functions taking a ``group: Group`` parameter as the first arg.
Called by ``Group.add_points`` (a thin signature + docstring + delegate)
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
    Union,
    cast,
)

import numpy as np
from arbol import aprint

from ....typing_utils.constants import (
    DEFAULT_POINT_RADIUS,
)
from ....typing_utils.json_safe import json_safe_value
from ....validation.writing import (
    POINTS_RESERVED_ATTRS,
    validate_render_attrs,
)
from ...points import Points
from ..auto_partition import resolve_auto_partition
from ..compositing import (
    ABSENT_WHEN_NONE_RENDER_ATTRS,
    COMPOSITING_ATTRS,
    funnel_add_error,
    is_broadcast_color,
    position_bounds_from_array,
    preflight_extend_to_all,
    reject_layer_order_inside_specialized_group,
    reject_lines_only_join,
    reject_mesh_only_appearance,
    slice_optional_array,
    strip_absent_attr_kwargs,
    sync_custom_colormap_attr,
    unnest_add_error,
    validate_points_channels_before_split,
)
from ..dim_order import apply_dim_order_positions
from ..partition import is_requested, reject_mismatched_partition_parent

if TYPE_CHECKING:
    from ...node import Node
    from ..group import Group


# DEFAULT_POINT_RADIUS — the radius applied when `radii` is not supplied — is
# imported above from `luxar.typing_utils.constants`, which owns it: the spatial
# index expands a no-radii chunk's bounds by the same number, so the two agree by
# construction. It stays importable from this module for existing callers.


def add_points_impl(
    group: "Group",
    *,
    name: str,
    positions: Any,
    colors: Any = None,
    radii: Any = None,
    sharpness: Any = None,
    scalars: Any = None,
    labels: Optional[Union[List[str], Sequence[str]]] = None,
    image_labels: Optional[Any] = None,
    keys: Optional[Union[List[str], Sequence[str]]] = None,
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    partition: Any = None,
    additive_lod: Any = None,
    substitutive_lod: Any = None,
    **attrs: Any,
) -> Union[Points, "Group"]:
    # additive_lod and substitutive_lod COMPOSE: substitutive chooses WHICH level
    # renders at the current zoom, additive describes HOW each level streams in.
    # Passing substitutive_lod alone ladders every level by default; pass
    # additive_lod=False to opt out. See lod/group.py's "Composed axes" section.
    try:
        # "An explicit None means absent" (#1574), applied ONCE here rather than
        # at each consumer: the colours gate below, the entry attrs gate, the
        # three structural branches and the flat write all read this same dict,
        # and ``sync_custom_colormap_attr`` alone has two call sites in this
        # module. Above every one of them, so no route can see the raw None.
        # Only render attrs are in the set — the structural keys whose None also
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
        reject_mismatched_partition_parent(parent or group, "points", name)
        reject_layer_order_inside_specialized_group(
            "points", name, attrs, parent or group
        )
        reject_lines_only_join("points", name, attrs)
        reject_mesh_only_appearance("points", name, attrs)

        scene = group._find_scene()

        pos_arr: np.ndarray = (
            positions if isinstance(positions, np.ndarray) else np.asarray(positions)
        )
        if pos_arr.ndim != 2:
            raise ValueError(
                f"Positions must have shape (N, D), got shape {pos_arr.shape}"
            )

        # Apply dim_order before validation
        pos_arr, extend_to_all = apply_dim_order_positions(
            pos_arr, scene, dim_order, fill, extend_to_all
        )

        n_points = pos_arr.shape[0]
        ndim = pos_arr.shape[1]

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

        # Scene-dimension COUNT check — deliberately here, above the
        # substitutive/partition/additive branches and below the colours gate.
        # Below the branches (where the full ``_validate_data_dimensions`` still
        # sits) the split paths never saw it: an additive ladder happily wrote
        # ``additive_<i>`` nodes whose column count contradicted the scene, and a
        # partition/substitutive wrapper refused only from inside ``part_0`` /
        # ``child_0``, stranding a childless wrapper group the flat path would
        # never have created. It must run AFTER ``apply_dim_order_positions``
        # (that is what fixes the final column count) and BEFORE any wrapper
        # group is written — do not move it back down. Only the count half is
        # hoisted: the per-dimension range ``UserWarning`` stays in the flat
        # write below, so its count is exactly what it was — once per dimension
        # per WRITTEN LEAF on the partition/substitutive paths, and none at all
        # under an additive ladder, whose writer never validates (a pre-existing
        # gap, pinned by a control test) — instead of gaining one more firing
        # here for the source array. Below the colours gate so a bad colours/colormap
        # combination keeps precedence over a dimension mismatch (all three
        # adders answer this the same way).
        scene._validate_dimension_count(pos_arr, name, data_type="positions")

        # Node-attrs gate — same validator the flat writer runs as its own first
        # step (write_points' step 0a), hoisted here for the same reason as the
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
        # the flat writer would reject — gsplat-RESERVED-but-points-unknown
        # (`amplitude_range=`; POINTS_RESERVED_ATTRS doesn't carry it but
        # GSPLATS_RESERVED_ATTRS does), a plain typo (`blending=`), or (on the
        # additive path only) genuinely points-RESERVED (`ordering=`,
        # `max_radius=`, misreported as *unknown* rather than *reserved*) —
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
        # `partition=`/`substitutive_lod=` spec), and `extend_to_all`
        # resolution — including on the flat fall-through below, where a call
        # that trips both now reports the attrs fault first (it used to report
        # the extend_to_all fault; see step 10 in README.md). The flat writer
        # validates node attrs before it ever looks at channels, so this hoist
        # keeps that same order on every path. The validator only inspects
        # `attrs` and raises on the first problem it finds — it is read-only,
        # so running it here on the live dict (not a copy) is safe. Calling
        # the SAME validator here means the split path refuses
        # byte-identically to the flat path below (which still runs it once
        # more, inside write_points — idempotent).
        validate_render_attrs(attrs, reserved_attrs=POINTS_RESERVED_ATTRS)

        # Substitutive-LOD branch — coarse levels are synthesised gsplats (each
        # point lifted to an isotropic Gaussian, then reduced by the gsplat
        # substitutive pipeline) under a kind=lod Group whose finest child is the
        # original Points node. Fires BEFORE (auto-)partition so substitutive
        # takes precedence over the opt-in auto-partition heuristic. An explicit
        # partition= composes as an overview: the coarse levels stay
        # global and the finest child becomes a spatial partition. Auto-partition
        # remains lower precedence, so enabling the compiler heuristic does not
        # silently change an ordinary substitutive ladder's topology.
        # ``pos_arr`` is already dim_order-transformed, so children are written
        # with dim_order=None/fill=None to avoid double application.
        if substitutive_lod is not None and n_points > 0:
            from ..lod.points import resolve_substitutive_axis_points

            substitutive_spec = resolve_substitutive_axis_points(substitutive_lod)
            if substitutive_spec is not None:
                fine_partition = None
                if is_requested(partition):
                    fine_partition = _resolve_points_partition(
                        pos_arr,
                        partition,
                        name,
                        image_labels,
                        scene.dimensions.displayed,
                    )
                preflight_extend_to_all(scene, extend_to_all, pos_arr, "points")
                return add_points_substitutive_lod_wrapper_impl(
                    group,
                    name=name,
                    pos_arr=pos_arr,
                    n_points=n_points,
                    colors=colors,
                    radii=radii,
                    sharpness=sharpness,
                    scalars=scalars,
                    labels=labels,
                    keys=keys,
                    image_labels=image_labels,
                    parent=parent,
                    extend_to_all=extend_to_all,
                    spec=substitutive_spec,
                    additive_lod=additive_lod,
                    fine_partition=fine_partition,
                    **attrs,
                )

        # Apply compiler-level auto-partition heuristic (opt-in; default
        # off) before evaluating the partition branch. User-explicit
        # partition= always wins — resolve_auto_partition passes it through
        # unchanged.
        partition = resolve_auto_partition(scene, n_points, partition)

        # Partition branch — decompose into N children if the user opted in
        # AND the BSP produces more than one part. Single-part outcomes
        # fall through to the regular single-leaf write below.
        # A dataset with <2 spatial dims can't be split; drop the request
        # with a warning rather than in silence.
        if partition is not None:
            partition_plan = _resolve_points_partition(
                pos_arr,
                partition,
                name,
                image_labels,
                scene.dimensions.displayed,
            )
            if partition_plan is not None:
                max_elements, parts, bsp_tree = partition_plan
                preflight_extend_to_all(scene, extend_to_all, pos_arr, "points")
                return add_points_partition_wrapper_impl(
                    group,
                    name=name,
                    pos_arr=pos_arr,
                    parts=parts,
                    n_points=n_points,
                    colors=colors,
                    radii=radii,
                    sharpness=sharpness,
                    scalars=scalars,
                    labels=labels,
                    keys=keys,
                    parent=parent,
                    extend_to_all=extend_to_all,
                    max_elements=max_elements,
                    bsp_tree=bsp_tree,
                    additive_lod=additive_lod,
                    **attrs,
                )
            # 1 part → fall through to additive-LOD / single-leaf write.

        # Additive-LOD branch — multi-level progressive writes via
        # add_points_multi_lod_wrapper_impl. Fires after the
        # 1-part-partition fall-through so a user can pass both
        # ``partition=`` and ``additive_lod=`` and get the inner LOD
        # ladder when partition doesn't fire.
        if (
            additive_lod is not None
            and additive_lod is not False
            and image_labels is not None
        ):
            from ..lod.points import resolve_additive_axis_points

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
            resolve_additive_axis_points(additive_lod)
            warnings.warn(
                f"'{name}': the requested streaming ladder cannot be honoured "
                "(image_labels is set); writing a flat node.",
                UserWarning,
                stacklevel=2,
            )
        elif additive_lod is not None:
            from ..lod.points import (
                make_additive_lod_points,
                resolve_additive_axis_points,
            )
            from ..lod.reveal import resolve_reveal_spatial_dims

            additive_spec = resolve_additive_axis_points(additive_lod)
            if additive_spec is not None:
                # Per-element radii needed for salience; broadcast
                # scalars to per-element array if applicable.
                if isinstance(radii, np.ndarray) and radii.shape == (n_points,):
                    radii_arr = radii
                else:
                    radii_arr = None
                # Per-element colors / scalars used by the energy:
                # breakpoints and salience_kind='energy' branches —
                # both need a luminance-and-size scoring of the
                # input.
                colors_for_energy = (
                    colors
                    if isinstance(colors, np.ndarray)
                    and colors.ndim == 2
                    and colors.shape[0] == n_points
                    else None
                )
                scalars_for_energy = (
                    scalars
                    if isinstance(scalars, np.ndarray) and scalars.shape == (n_points,)
                    else None
                )
                levels = make_additive_lod_points(
                    pos_arr,
                    radii=radii_arr,
                    method=additive_spec["method"],
                    n_lods=additive_spec["n_lods"],
                    counts=additive_spec["counts"],
                    seed=additive_spec["seed"],
                    colors=colors_for_energy,
                    scalars=scalars_for_energy,
                    salience_kind=additive_spec.get("salience_kind", "size"),
                    reveal_center=additive_spec.get("reveal_center"),
                    spatial_dims=resolve_reveal_spatial_dims(
                        additive_spec, scene, ndim
                    ),
                )
                if len(levels) > 1:
                    return add_points_multi_lod_wrapper_impl(
                        group,
                        name=name,
                        pos_arr=pos_arr,
                        levels=levels,
                        n_points=n_points,
                        colors=colors,
                        radii=radii,
                        sharpness=sharpness,
                        scalars=scalars,
                        labels=labels,
                        keys=keys,
                        parent=parent,
                        extend_to_all=extend_to_all,
                        method=additive_spec["method"],
                        counts=additive_spec["counts"],
                        radii_for_energy=radii_arr,
                        colors_for_energy=colors_for_energy,
                        scalars_for_energy=scalars_for_energy,
                        **attrs,
                    )
                # 1 level (degenerate) → fall through to single-leaf.

        aprint(f"Adding points node '{name}' with {n_points:,} points in {ndim}D.")

        scene._validate_data_dimensions(pos_arr, name, data_type="positions")

        final_extend_dims = scene._resolve_extend_to_all(
            extend_to_all, pos_arr, "points"
        )
        if final_extend_dims:
            attrs["extend_to_all"] = final_extend_dims
            aprint(f"  📡 Extending visibility across: {final_extend_dims}")

        parent_node = parent or group

        if radii is None:
            radii = DEFAULT_POINT_RADIUS
            aprint(f"  📐 Using default radius: {DEFAULT_POINT_RADIUS}")

        writer = group._require_scene_writer(scene)
        path = f"{parent_node.path}/{name}" if parent_node.path else name
        metadata = writer.write_points(
            path,
            pos_arr.astype(np.float32),
            colors=cast(Any, colors),
            radii=radii,
            sharpness=sharpness,
            scalars=scalars,
            labels=labels,
            keys=keys,
            image_labels=image_labels,
            **attrs,
        )

        # Sync colormap attr with what the compiler wrote to zarr:
        # - Array colormaps are resolved and stored as "custom"
        # - Non-built-in string names (matplotlib/colorcet) are also
        #   resolved to LUT and stored as "custom"
        sync_custom_colormap_attr(attrs)

        # Notify scene that labels exist (for hover overlay auto-injection)
        if labels is not None:
            scene._notify_labels_added()
        if image_labels is not None:
            scene._notify_image_labels_added()

        return Points(
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
        inner = unnest_add_error("points", name, e)
        aprint(f"Failed to add points node '{name}': {inner}")
        raise ValueError(funnel_add_error("points", name, e)) from e


def _resolve_points_partition(
    pos_arr: np.ndarray,
    partition: Any,
    name: str,
    image_labels: Any,
    split_axes: Sequence[int],
) -> Optional[tuple[int, List[np.ndarray], Dict[str, Any]]]:
    """Resolve and execute a points partition, returning only a real split."""
    from ..partition import (
        bsp_leaf_parts,
        resolve_partition_spec,
        spatial_bsp_tree,
        warn_if_oversized_single_part,
        warn_if_partition_needs_more_dims,
    )

    if pos_arr.shape[0] == 0:
        return None

    if not warn_if_partition_needs_more_dims(pos_arr.shape[1], name):
        return None

    max_elements, partition_rule = resolve_partition_spec(partition)
    if image_labels is not None:
        raise ValueError(
            "image_labels is not supported alongside partition=. "
            "Decompose the data manually or omit image_labels."
        )

    tree = spatial_bsp_tree(
        pos_arr, max_elements, rule=partition_rule, split_axes=split_axes
    )
    parts = bsp_leaf_parts(tree)
    warn_if_oversized_single_part(
        len(parts), int(parts[0].size) if parts else 0, max_elements, name
    )
    return (max_elements, parts, tree.to_serializable()) if len(parts) > 1 else None


def add_points_partition_wrapper_impl(
    group: "Group",
    *,
    name: str,
    pos_arr: np.ndarray,
    parts: List[np.ndarray],
    n_points: int,
    colors: Any,
    radii: Any,
    sharpness: Any,
    scalars: Any,
    labels: Any,
    keys: Any = None,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    max_elements: int,
    bsp_tree: Dict[str, Any],
    additive_lod: Any = None,
    additive_lod_slices: int = 1,
    wrapper_coverage_fraction: Optional[float] = None,
    **attrs: Any,
) -> "Group":
    """Build a kind=partition wrapper Group with one Points child per BSP part."""
    # Entering a wrapper IS "a split is about to happen": from here on every
    # per-point channel is sliced per part, and `slice_optional_array` passes a
    # wrong-length value through whole. Scoped to the split paths so the
    # plain-leaf CHANNEL-gate order (and therefore which error a multi-fault
    # call reports, among channel-vs-channel faults) is untouched. The one
    # exception is the node-attrs gate (#1529, add_points_impl above): it runs
    # at the adder entry, before this wrapper is even chosen, so it outranks
    # this channel gate too — a call that trips both reports the attrs fault.
    validate_points_channels_before_split(
        n_points,
        colors=colors,
        radii=radii,
        sharpness=sharpness,
        scalars=scalars,
        labels=labels,
        keys=keys,
    )

    # A uniform RGB(A) list/tuple is the one leaf parameter whose OWN length can
    # collide with the point count (a 3-point node with an RGB triple), so
    # classify it up front instead of letting the length test gather it.
    uniform_color = is_broadcast_color(colors)

    wrapper_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    if wrapper_coverage_fraction is not None:
        wrapper_attrs["coverage_fraction"] = wrapper_coverage_fraction
    leaf_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}

    parent_node = parent or group
    wrapper = parent_node.add_partition_group(
        name=name,
        display_type="points",
        max_elements=max_elements,
        **wrapper_attrs,
    )

    aprint(
        f"  ✂️  Partitioned '{name}' into {len(parts)} parts via BSP "
        f"(max_elements={max_elements:,}, "
        f"sizes={[int(p.size) for p in parts]})"
    )

    for i, indices in enumerate(parts):
        part_additive_lod = additive_lod
        if additive_lod_slices > 1:
            from ..lod.group import level_additive_lod

            part_additive_lod = level_additive_lod(
                additive_lod,
                level_n=int(indices.size),
                compression_factor=1,
                is_coarsest=True,
                slices=additive_lod_slices,
            )
        wrapper.add_points(
            name=f"part_{i}",
            positions=pos_arr[indices],
            colors=colors
            if uniform_color
            else slice_optional_array(colors, indices, n_points),
            radii=slice_optional_array(radii, indices, n_points),
            sharpness=slice_optional_array(sharpness, indices, n_points),
            scalars=slice_optional_array(scalars, indices, n_points),
            labels=slice_optional_array(labels, indices, n_points),
            keys=slice_optional_array(keys, indices, n_points),
            # image_labels banned alongside partition= (see add_points entry)
            image_labels=None,
            extend_to_all=extend_to_all,
            # dim_order / fill already applied to pos_arr upstream — do
            # not re-apply in the per-part recursion.
            dim_order=None,
            fill=None,
            # ``partition=False`` (not ``None``) → explicit no-partition that
            # ALSO bypasses the compiler-level auto-partition heuristic.
            # ``None`` would re-trigger auto-partition on each part when
            # the compiler's threshold is smaller than the user's
            # cap, blowing up the leaf count.
            partition=False,
            # Inner LOD ladder per spatial part — each part decides
            # its own ladder independently. Allows the Partition-of-
            # AdditiveLOD composition from the plan.
            additive_lod=part_additive_lod,
            **leaf_attrs,
        )
    from ..partition import persist_pruned_bsp_tree

    # Unlike lines, every resolved points part is written in order.
    persist_pruned_bsp_tree(wrapper, bsp_tree, range(len(parts)))

    # Persist the wrapper's position_bounds (per-axis min/max of the
    # full input) so picking / scene-bounds-cache treat the layer as
    # one logical entity. Computed directly from ``pos_arr`` — same
    # result as unioning per-child bboxes, simpler than round-tripping
    # through the children's on-disk attrs.
    wrapper._persist_attr("position_bounds", position_bounds_from_array(pos_arr))

    return wrapper


def add_points_multi_lod_wrapper_impl(
    group: "Group",
    *,
    name: str,
    pos_arr: np.ndarray,
    levels: List[np.ndarray],
    n_points: int,
    colors: Any,
    radii: Any,
    sharpness: Any,
    scalars: Any,
    labels: Any,
    keys: Any = None,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    method: str,
    counts: Any = None,
    radii_for_energy: Optional[np.ndarray] = None,
    colors_for_energy: Optional[np.ndarray] = None,
    scalars_for_energy: Optional[np.ndarray] = None,
    **attrs: Any,
) -> Points:
    """Write a Points node with multi-additive-LOD subgroups.

    Produces ``<path>/additive_<i>/`` subgroups (one per LOD level),
    each carrying the points assigned to that level. The parent
    points node carries ``n_additive_sublods=N``, a global
    ``position_bounds``, and the standard compositing attrs.

    The returned :class:`Points` node is the parent (the user's
    logical "one node"). The viewer's progressive loader walks the
    subgroups; the user never sees the decomposition.

    ``labels`` are written by the writer as ONE union CSR on the parent (the
    subgroups carry none — the loader concatenates levels into one buffer), so
    the scene is notified here exactly as on the flat path.

    ``counts`` and the three ``*_for_energy`` arrays are only used to stamp the
    ladder's quality metadata (``lod_stats.energy_fraction_cum`` per level plus
    ``level_stats.reference_energy`` on the parent), which lets the viewer
    release a LOD swap on committed energy instead of raw element count.
    """
    # See add_points_partition_wrapper_impl: every per-point channel is about to
    # be sliced per level, and a per-level length check cannot catch a
    # wrong-length value whose length happens to match some level's.
    validate_points_channels_before_split(
        n_points,
        colors=colors,
        radii=radii,
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

    # Ladder quality stamps. The energy is a pure per-element function of the
    # same inputs the ordering used, so summing it per level reproduces the
    # ladder's cumulative curve exactly — no need to thread it out of the
    # builder (whose List[level] return shape many tests depend on).
    from ..lod.group import (
        additive_level_stats,
        breakpoints_kind_of,
        resolve_ladder_extend_to_all,
    )
    from ..lod.points import compute_points_energy

    energy = compute_points_energy(
        n_points, radii_for_energy, colors_for_energy, scalars_for_energy
    )
    per_level_stats, _reference_energy, parent_level_stats = additive_level_stats(
        [float(energy[idx].sum()) for idx in levels],
        [int(idx.size) for idx in levels],
        method=method,
        breakpoints_kind=breakpoints_kind_of(counts),
        energy_kind="points-luminance-volume",
    )

    # Build per-level slice tuples for the writer.
    level_slices: List[Dict[str, Any]] = []
    for level_i, level_indices in enumerate(levels):
        level_slices.append(
            {
                "positions": pos_arr[level_indices].astype(np.float32),
                "colors": colors
                if uniform_color
                else slice_optional_array(colors, level_indices, n_points),
                "radii": slice_optional_array(radii, level_indices, n_points),
                "sharpness": slice_optional_array(sharpness, level_indices, n_points),
                "scalars": slice_optional_array(scalars, level_indices, n_points),
                "labels": slice_optional_array(labels, level_indices, n_points),
                "keys": slice_optional_array(keys, level_indices, n_points),
                "lod_stats": per_level_stats[level_i],
            }
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
        _, safe_caller = json_safe_value(caller_level_stats)
        attrs["level_stats"] = {
            "reference_energy": parent_level_stats["reference_energy"],
            **(safe_caller or {}),
        }

    aprint(
        f"  📐 Additive-LOD '{name}': {len(levels)} levels "
        f"(method={method!r}, sizes={[int(L.size) for L in levels]})"
    )

    # Nothing else resolves the ``"all"`` sentinel on this branch (unlike the
    # partition wrapper, we do not recurse through the leaf adder) and the
    # writer stamps the value VERBATIM onto the parent group AND every
    # ``additive_<i>/`` sub-LOD — see ``resolve_ladder_extend_to_all`` for why
    # ``None`` stays unresolved. Must stay an explicit kwarg: in ``attrs`` it
    # would collide.
    final_extend_dims = resolve_ladder_extend_to_all(
        scene, extend_to_all, pos_arr, "points"
    )

    metadata = writer.write_points_multi_lod(
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

    return Points(
        name,
        metadata=metadata,
        parent=cast(Any, parent_node),
        writer=writer,
        **attrs,
    )


def _stratified_point_order(
    scene: Any,
    positions: np.ndarray,
    compression: int,
    seed: int,
    extend_to_all: Optional[Union[List[str], str]],
) -> np.ndarray:
    """Order displayed-space points evenly, round-robin across hidden slices."""
    from ..lod.group import resolve_same_type_axes
    from ..lod.spatial_uniform import stratified_grid_order

    displayed, hidden, _ = resolve_same_type_axes(
        scene, positions, extend_to_all, "points"
    )
    rng = np.random.default_rng(seed)

    def local_order(rows: np.ndarray) -> np.ndarray:
        shuffled = rows[rng.permutation(rows.size)]
        sample = positions[shuffled][:, displayed[:3]]
        if sample.shape[1] < 3:
            sample = np.pad(sample, ((0, 0), (0, 3 - sample.shape[1])))
        target = max(1, int(rows.size) // compression)
        depth = min(31, max(1, int(np.ceil(np.log2(target) / 3.0)) + 2))
        order, _ = stratified_grid_order(sample, depth)
        return shuffled[order]

    all_rows = np.arange(positions.shape[0], dtype=np.intp)
    if all_rows.size == 0 or not hidden:
        return local_order(all_rows)

    hidden_values = (
        positions[:, hidden[0]] if len(hidden) == 1 else positions[:, hidden]
    )
    _, slice_ids = np.unique(hidden_values, axis=0, return_inverse=True)
    within_slice_rank = np.empty(positions.shape[0], dtype=np.intp)
    grouped_rows = np.argsort(slice_ids, kind="stable").astype(np.intp, copy=False)
    boundaries = np.flatnonzero(np.diff(slice_ids[grouped_rows])) + 1
    for rows in np.split(grouped_rows, boundaries):
        ordered = local_order(rows)
        within_slice_rank[ordered] = np.arange(ordered.size, dtype=np.intp)
    return np.lexsort((slice_ids, within_slice_rank)).astype(np.intp, copy=False)


def _same_type_point_coarse_levels(
    *,
    method: str,
    scene: Any,
    name: str,
    pos_arr: np.ndarray,
    n_points: int,
    radii: Any,
    energy: np.ndarray,
    source_colors: Any,
    displayed: List[int],
    discrete_hidden: List[int],
    compression_factor: int,
    levels: int,
    seed: int,
    extend_to_all: Any,
) -> List[Any]:
    """Construct finest-to-coarsest same-type point level descriptors."""
    from ....gsplats.lift import coarse_point_levels, same_type_merge_level
    from ..lod.group import same_type_color_classes

    if method != "merge":
        aprint(f"  📐 Substitutive-LOD '{name}': ordering {n_points:,} points")
        order = _stratified_point_order(
            scene, pos_arr, compression_factor, seed, extend_to_all
        )
        return coarse_point_levels(
            order, energy, compression_factor=compression_factor, levels=levels
        )
    radius_values = np.broadcast_to(
        np.asarray(DEFAULT_POINT_RADIUS if radii is None else radii, dtype=np.float64),
        (n_points,),
    )
    color_classes = same_type_color_classes(source_colors)
    barrier_parts = [pos_arr[:, discrete_hidden]] if discrete_hidden else []
    if color_classes is not None:
        barrier_parts.append(color_classes[:, None])
    barriers = np.column_stack(barrier_parts) if barrier_parts else None
    coarse: List[Any] = []
    previous_count = n_points
    for level in range(1, levels + 1):
        count = max(1, n_points // (compression_factor**level))
        if count >= previous_count:
            continue
        coarse.append(
            same_type_merge_level(
                pos_arr,
                radius_values,
                energy,
                n_target=count,
                spatial_dims=displayed[:3],
                barrier_keys=barriers,
            )
        )
        previous_count = count
    return coarse


def _same_type_point_level_payload(
    *,
    method: str,
    level: Any,
    reduction_level: int,
    pos_arr: np.ndarray,
    n_points: int,
    radii: Any,
    colors: Any,
    source_colors: Any,
    scalars: Any,
    sharpness: Any,
    labels: Any,
    keys: Any,
    energy: np.ndarray,
    child_attrs: Dict[str, Any],
    compensation: Any,
    auto_compensates: bool,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, float]:
    """Materialize one point child payload for either same-type method."""
    from ..lod.group import subsampled_same_type_level_channels
    from ..lod.points import compute_points_energy

    if method != "merge":
        indices, measured_gain = level
        gain = (
            measured_gain
            if compensation == "auto" and auto_compensates
            else 1.0
            if compensation == "auto"
            else float(compensation) ** reduction_level
        )
        level_colors, level_scalars, level_attrs = subsampled_same_type_level_channels(
            colors=colors,
            colors_for_energy=source_colors,
            scalars=scalars,
            indices=indices,
            n_elements=n_points,
            gain=gain,
            child_attrs=child_attrs,
        )
        return (
            pos_arr[indices],
            slice_optional_array(radii, indices, n_points),
            level_colors,
            level_scalars,
            slice_optional_array(sharpness, indices, n_points),
            slice_optional_array(labels, indices, n_points),
            slice_optional_array(keys, indices, n_points),
            level_attrs,
            gain,
        )
    level_positions, level_radii, assignments, merge_weights = level
    totals = np.bincount(
        assignments, weights=merge_weights, minlength=len(level_positions)
    )
    color_source = (
        np.ones((n_points, 3), dtype=np.float32)
        if source_colors is None
        else source_colors
    )
    level_colors = np.column_stack(
        [
            np.bincount(
                assignments,
                weights=merge_weights * color_source[:, channel],
                minlength=len(level_positions),
            )
            / totals
            for channel in range(color_source.shape[1])
        ]
    ).astype(np.float32)
    target = np.bincount(assignments, weights=energy, minlength=len(level_positions))
    represented = compute_points_energy(
        len(level_positions), level_radii, level_colors, None
    )
    level_colors[:, :3] *= np.divide(
        target, represented, out=np.zeros_like(target), where=represented > 0.0
    )[:, None].astype(np.float32)
    level_attrs = dict(child_attrs)
    level_attrs.pop("colormap", None)
    return (
        level_positions,
        level_radii,
        level_colors,
        None,
        None,
        None,
        None,
        level_attrs,
        1.0,
    )


def _report_point_merge_channels(
    method: str, *, sharpness: Any, labels: Any, keys: Any, scalars: Any, colors: Any
) -> None:
    """Report point channels that merge drops or bakes into representative color."""
    if method != "merge":
        return
    for channel, value in (
        ("sharpness", sharpness),
        ("labels", labels),
        ("keys", keys),
    ):
        if value is not None:
            aprint(
                f"  ↳ merge drops per-point {channel}; representatives have no source identity"
            )
    if scalars is not None and colors is None:
        aprint("  ↳ merge bakes per-point scalars through the authored colormap")


def _warn_continuous_same_type_axes(geometry: str, axes: List[int]) -> None:
    """Warn when continuous hidden axes cannot define independent merge slices."""
    if not axes:
        return
    warnings.warn(
        f"substitutive_lod with coarse={geometry!r} found continuous hidden "
        f"dimensions {tuple(axes)}; they are not treated as independent slices, "
        "so per-slice preservation is not applied. Hidden slice dimensions "
        "should be discrete (categorical / timepoint / channel).",
        RuntimeWarning,
        stacklevel=3,
    )


def _add_points_subsampled_lod_wrapper_impl(
    group: "Group",
    *,
    name: str,
    pos_arr: np.ndarray,
    n_points: int,
    colors: Any,
    radii: Any,
    sharpness: Any,
    scalars: Any,
    labels: Any,
    keys: Any,
    image_labels: Any,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    spec: Dict[str, Any],
    additive_lod: Any,
    fine_partition: Optional[tuple[int, List[np.ndarray], Dict[str, Any]]],
    attrs: Dict[str, Any],
) -> Union["Group", Points]:
    """Write spatially stratified Points levels above the original Points node."""
    from ....gsplats.lift import lift_points_to_gsplats
    from ....gsplats.lod.quality import mixture_quality
    from ....utils.lod_breakpoints import hidden_coordinate_count
    from ..lod.group import (
        assert_same_type_slice_capacity,
        compose_additive_under_substitutive,
        effective_blending_mode,
        level_additive_lod,
        level_attrs_with_quality,
        normalized_same_type_colors,
        resolve_lod_ladder,
        resolve_same_type_axes,
    )
    from ..lod.points import compute_points_energy, resolve_additive_axis_points

    colors_for_energy = colors
    if scalars is not None and colors is None:
        colormap = attrs.get("colormap")
        if colormap is None:
            raise ValueError(
                "substitutive_lod on Points with scalars requires a colormap "
                "(scalars map to colour via a colormap LUT). Pass colormap=..., "
                "or provide explicit per-point colors."
            )
        from ....colormaps import scalars_to_colors

        scalar_array = np.broadcast_to(
            np.asarray(scalars, dtype=np.float64).reshape(-1), (n_points,)
        )
        colors_for_energy = scalars_to_colors(scalar_array, colormap)

    scene = group._find_scene()
    compression_factor = int(spec["compression_factor"])
    displayed, discrete_hidden, continuous_hidden = resolve_same_type_axes(
        scene, pos_arr, extend_to_all, "points"
    )
    resident_slices = hidden_coordinate_count(pos_arr, discrete_hidden)
    _warn_continuous_same_type_axes("points", continuous_hidden)
    assert_same_type_slice_capacity(
        same_type="points",
        element_name="points",
        elements=n_points,
        resident_slices=resident_slices,
        compression_factor=compression_factor,
        levels=int(spec["levels"]),
    )
    radii_for_energy = None
    if radii is not None:
        radii_for_energy = np.broadcast_to(
            np.asarray(radii, dtype=np.float64).reshape(-1), (n_points,)
        )
    normalized_colors_for_energy = normalized_same_type_colors(
        colors_for_energy,
        n_points,
        integer_storage=colors is not None and not is_broadcast_color(colors),
    )
    energy = compute_points_energy(
        n_points,
        radii_for_energy,
        normalized_colors_for_energy,
        None,
    )
    method = spec.get("method", "subsample")
    _report_point_merge_channels(
        method,
        sharpness=sharpness,
        labels=labels,
        keys=keys,
        scalars=scalars,
        colors=colors,
    )
    coarse = _same_type_point_coarse_levels(
        method=method,
        scene=scene,
        name=name,
        pos_arr=pos_arr,
        n_points=n_points,
        radii=radii,
        energy=energy,
        source_colors=normalized_colors_for_energy,
        displayed=displayed,
        discrete_hidden=discrete_hidden,
        compression_factor=compression_factor,
        levels=int(spec["levels"]),
        seed=int(spec.get("seed") or 0),
        extend_to_all=extend_to_all,
    )
    if not coarse:
        aprint(
            f"  ⚠ substitutive_lod '{name}': input too small to synthesize coarse "
            "levels; writing the flat Points node."
        )
        return add_points_impl(
            group,
            name=name,
            positions=pos_arr,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
            scalars=scalars,
            labels=labels,
            keys=keys,
            image_labels=image_labels,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=None,
            fill=None,
            partition=False,
            additive_lod=additive_lod,
            substitutive_lod=None,
            **attrs,
        )

    slices = resident_slices if additive_lod is None else 1
    resolved_additive = compose_additive_under_substitutive(
        additive_lod,
        resolve=resolve_additive_axis_points,
        elements=n_points,
        name=name,
        slices=slices,
    )
    parent_node = parent or group
    coarse_first = list(reversed(coarse))
    counts = [int(level[0].shape[0]) for level in coarse_first] + [n_points]
    coverage_vals, lod_selector = resolve_lod_ladder(
        spec.get("coverage_fractions"),
        counts,
        parent_node,
        name=name,
        partition_bound=fine_partition is not None,
        length_error=lambda n_explicit, n_levels: (
            f"coverage_fractions has {n_explicit} entries but the LOD ladder "
            f"has {n_levels} levels ({len(coarse_first)} points + 1 finest)"
        ),
    )
    lod_attrs = {key: value for key, value in attrs.items() if key in COMPOSITING_ATTRS}
    child_attrs = {
        key: value for key, value in attrs.items() if key not in COMPOSITING_ATTRS
    }
    child_attrs.pop("coverage_fraction", None)
    lod_attrs.setdefault("display_type", "points")
    aprint(
        f"  📐 Substitutive-LOD '{name}': {len(coarse_first)} point levels + finest "
        f"(counts coarsest→finest={counts}, K={compression_factor})"
    )
    lod_group_node = parent_node.add_lod_group(name, selector=lod_selector, **lod_attrs)

    blending_mode = effective_blending_mode(parent_node, attrs, "points")
    auto_compensates = blending_mode in {"additive", "luminous"}
    compensation = spec.get("brightness_compensation", "auto")
    quality_stamps = bool(spec["quality_stamps"])
    quality_reference = (
        lift_points_to_gsplats(
            pos_arr,
            DEFAULT_POINT_RADIUS if radii is None else radii,
            colors=None,
        )
        if quality_stamps
        else None
    )
    for child_index, level in enumerate(coarse_first):
        reduction_level = len(coarse_first) - child_index
        (
            level_positions,
            level_radii,
            level_colors,
            level_scalars,
            level_sharpness,
            level_labels,
            level_keys,
            level_attrs,
            gain,
        ) = _same_type_point_level_payload(
            method=method,
            level=level,
            reduction_level=reduction_level,
            pos_arr=pos_arr,
            n_points=n_points,
            radii=radii,
            colors=colors,
            source_colors=normalized_colors_for_energy,
            scalars=scalars,
            sharpness=sharpness,
            labels=labels,
            keys=keys,
            energy=energy,
            child_attrs=child_attrs,
            compensation=compensation,
            auto_compensates=auto_compensates,
        )
        if quality_reference is not None:
            quality_level = lift_points_to_gsplats(
                level_positions,
                DEFAULT_POINT_RADIUS if level_radii is None else level_radii,
                colors=None,
            )
            level_attrs = level_attrs_with_quality(
                level_attrs,
                mixture_quality(quality_level, quality_reference, device="cpu").quality,
            )
        with warnings.catch_warnings():
            if gain > 1.0:
                warnings.filterwarnings(
                    "ignore",
                    message=r".*HDR colors with maximum value .* detected.*",
                    category=UserWarning,
                )
            lod_group_node.add_points(
                f"child_{child_index}",
                level_positions,
                colors=level_colors,
                radii=level_radii,
                sharpness=level_sharpness,
                scalars=level_scalars,
                labels=level_labels,
                keys=level_keys,
                image_labels=None,
                extend_to_all=extend_to_all,
                partition=False,
                additive_lod=level_additive_lod(
                    resolved_additive,
                    level_n=int(level_positions.shape[0]),
                    compression_factor=compression_factor,
                    is_coarsest=(child_index == 0),
                    slices=slices,
                ),
                substitutive_lod=None,
                coverage_fraction=coverage_vals[child_index],
                **level_attrs,
            )

    finest_index = len(coarse_first)
    if fine_partition is not None:
        max_elements, parts, bsp_tree = fine_partition
        add_points_partition_wrapper_impl(
            group,
            name=f"child_{finest_index}",
            pos_arr=pos_arr,
            parts=parts,
            n_points=n_points,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
            scalars=scalars,
            labels=labels,
            keys=keys,
            parent=lod_group_node,
            extend_to_all=extend_to_all,
            max_elements=max_elements,
            bsp_tree=bsp_tree,
            additive_lod=resolved_additive,
            additive_lod_slices=slices,
            wrapper_coverage_fraction=coverage_vals[-1],
            **(
                level_attrs_with_quality(child_attrs, 1.0)
                if quality_stamps
                else child_attrs
            ),
        )
    else:
        lod_group_node.add_points(
            f"child_{finest_index}",
            pos_arr,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
            scalars=scalars,
            labels=labels,
            keys=keys,
            image_labels=image_labels,
            extend_to_all=extend_to_all,
            partition=False,
            additive_lod=level_additive_lod(
                resolved_additive,
                level_n=n_points,
                compression_factor=compression_factor,
                is_coarsest=False,
                slices=slices,
            ),
            substitutive_lod=None,
            coverage_fraction=coverage_vals[-1],
            **(
                level_attrs_with_quality(child_attrs, 1.0)
                if quality_stamps
                else child_attrs
            ),
        )
    return lod_group_node


def add_points_substitutive_lod_wrapper_impl(
    group: "Group",
    *,
    name: str,
    pos_arr: np.ndarray,
    n_points: int,
    colors: Any,
    radii: Any,
    sharpness: Any,
    scalars: Any,
    labels: Any,
    keys: Any = None,
    image_labels: Any,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    spec: Dict[str, Any],
    additive_lod: Any = None,
    fine_partition: Optional[tuple[int, List[np.ndarray], Dict[str, Any]]] = None,
    **attrs: Any,
) -> Union["Group", Points]:
    """Write a Points node whose coarse LOD levels are synthesised gsplats.

    Each point is lifted to an isotropic Gaussian (see
    :func:`luxar.gsplats.lift.lift_points_to_gsplats`), the gsplat substitutive
    pipeline (:func:`luxar.gsplats.make_substitutive_lod`) synthesises
    fewer-but-larger representative levels, and those become the coarse children
    of a ``kind=lod`` Group whose **finest** child is the original Points node,
    or ``fine_partition`` for the overview topology.
    Coarse-level amplitudes are per-bin mass-preserving and rescaled to conserve
    render-light (``sum a·σ³``) so brightness is stable across the LOD seam (no
    zoom-out dimming); a ``max_aspect`` anisotropy cap (default 3) keeps merged
    splats near-isotropic so their brightness stays view-independent (see
    :func:`luxar.gsplats.lift.coarse_substitutive_levels`).

    Compositing attrs (opacity, gamma, ...) land on the ``kind=lod`` Group;
    ``opacity`` is therefore applied once at composite time to both the points
    child and the gsplat children (the lift uses ``opacity=1``).
    """
    # Before the lift: the coarse levels cost a full gsplat reduce, and the
    # finest child (written LAST, after every coarse level is already on disk) is
    # where a wrong-length channel would otherwise be caught — stranding a
    # partial kind=lod group. Fail here instead, before anything is written.
    # The node-attrs gate (#1529, add_points_impl above) already ran at the
    # adder entry, before this wrapper was even chosen, so it outranks this
    # channel gate too — a call that trips both reports the attrs fault.
    validate_points_channels_before_split(
        n_points,
        colors=colors,
        radii=radii,
        sharpness=sharpness,
        scalars=scalars,
        labels=labels,
        keys=keys,
        image_labels=image_labels,
    )

    if spec.get("coarse", "gsplats") == "points":
        return _add_points_subsampled_lod_wrapper_impl(
            group,
            name=name,
            pos_arr=pos_arr,
            n_points=n_points,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
            scalars=scalars,
            labels=labels,
            keys=keys,
            image_labels=image_labels,
            parent=parent,
            extend_to_all=extend_to_all,
            spec=spec,
            additive_lod=additive_lod,
            fine_partition=fine_partition,
            attrs=attrs,
        )

    from ....gsplats.lift import coarse_substitutive_levels, lift_points_to_gsplats
    from ..lod.group import (
        compose_additive_under_substitutive,
        gsplat_additive_lod_from,
        level_additive_lod,
        level_attrs_with_quality,
        resident_slice_count,
        resolve_lod_ladder,
    )
    from ..lod.points import resolve_additive_axis_points
    from ..lod.reveal import is_reveal_additive_method, preflight_reveal_center

    # The coarse gsplat children never carry image_labels, so only the original
    # Points child must refuse its additive ladder when image_labels is present.
    # Keep the existing element-domain stream counts for composed coarse
    # children. Retuning those counts for gsplat bytes-per-element is a separate
    # cross-geometry policy change, not part of suppression scoping.
    # A default ladder's first chunk is a whole-node download budget, so the
    # slice count is what keeps it from arriving divided on an nD node (#2374).
    slices = (
        resident_slice_count(group._find_scene(), pos_arr)
        if additive_lod is None
        else 1
    )
    coarse_additive = compose_additive_under_substitutive(
        additive_lod,
        resolve=resolve_additive_axis_points,
        elements=n_points,
        name=name,
        slices=slices,
    )
    reveal_note = (
        " Coarse levels use self_energy ordering, so reveal_center is not applied."
        if coarse_additive is not None
        and is_reveal_additive_method(str(coarse_additive.get("method")))
        else ""
    )
    finest_additive = compose_additive_under_substitutive(
        additive_lod,
        resolve=resolve_additive_axis_points,
        elements=n_points,
        name=name,
        slices=slices,
        # The multi-LOD writer has no image_labels channel, so laddering would
        # silently drop them. Refuse the ladder, not the labels.
        suppress_reason="image_labels is set" if image_labels is not None else None,
        suppression_outcome=(
            "the finest level will load all-at-once; coarse levels keep their "
            "ladder where one applies."
            f"{reveal_note}"
        ),
    )
    # Same reason as the channel check above: the finest child is written LAST, so
    # a reveal_center that does not match the DERIVED shell axes would otherwise
    # raise once every coarse level is already on disk.
    preflight_reveal_center(finest_additive, group._find_scene(), pos_arr, "positions")
    compression_factor = int(spec["compression_factor"])

    # Scalar+colormap points have no per-splat scalar channel on gsplats, so bake
    # scalars -> RGB (via the same LUT normalisation the viewer uses) and lift
    # with those colours. The FINEST child keeps scalars+colormap (native,
    # interactively re-colourable); only the coarse gsplat levels carry baked
    # colours — so a live colormap change re-colours the finest level but not the
    # coarse ones (documented caveat). Scalars without a colormap is still an
    # error (scalars require a colormap to map to colour).
    colors_for_lift = colors
    if scalars is not None and colors is None:
        colormap = attrs.get("colormap")
        if colormap is None:
            raise ValueError(
                "substitutive_lod on Points with scalars requires a colormap "
                "(scalars map to colour via a colormap LUT). Pass colormap=..., "
                "or provide explicit per-point colors."
            )
        from ....colormaps import scalars_to_colors

        # Broadcast a scalar / size-1 `scalars` to per-point length so the baked
        # colours array matches n_points (a uniform scalar is a documented input).
        s = np.broadcast_to(
            np.asarray(scalars, dtype=np.float64).reshape(-1), (n_points,)
        )
        colors_for_lift = scalars_to_colors(s, colormap)

    radii_for_lift = DEFAULT_POINT_RADIUS if radii is None else radii
    lifted = lift_points_to_gsplats(
        pos_arr,
        radii_for_lift,
        colors=colors_for_lift,
        opacity=1.0,
        truncation_radius=float(spec["truncation_radius"]),
    )

    # Resolve which dims coarsening may merge over (default Auto = displayed dims,
    # grouping by non-displayed dims so coarse splats never blend across a
    # categorical/sliced axis). pos_arr columns == lifted gsplat columns.
    from ..lod.group import resolve_coarsen_dims

    coarsen_dims = resolve_coarsen_dims(
        group._find_scene(), int(lifted.ndim), spec.get("coarsen_dims")
    )

    # Synthesise coarse gsplat levels (level 0 dropped, render-light conserved so
    # the LOD seam does not brighten/dim). Returns finest -> coarsest.
    coarse = coarse_substitutive_levels(
        lifted,
        compression_factor=int(spec["compression_factor"]),
        levels=int(spec["levels"]),
        method=str(spec["method"]),
        device=spec.get("device", "auto"),
        seed=spec.get("seed"),
        coarsen_dims=coarsen_dims,
        max_aspect=spec.get("max_aspect", 3.0),
        quality_stamps=bool(spec["quality_stamps"]),
    )

    # Degenerate input -> finest Points shape rather than a one-child LOD group.
    # Covers BOTH no coarse levels AND an all-zero-radius cloud (the lift yields
    # 0 splats, so every coarse level is empty: coarse[-1] is the coarsest, and
    # writing a 0-element coarsest gsplat child would crash downstream). Mirrors
    # the lines.py degenerate guard. ``pos_arr`` is already dim_order-transformed,
    # so dim_order/fill are None and partition is disabled.
    if not coarse or int(coarse[-1].n_splats) == 0:
        fallback = (
            "partitioned Points branch"
            if fine_partition is not None
            else "flat Points node"
        )
        aprint(
            f"  ⚠ substitutive_lod '{name}': input too small to synthesise coarse "
            f"levels; writing the {fallback}."
        )
        if fine_partition is not None:
            max_elements, parts, bsp_tree = fine_partition
            # Each part sizes its own ladder; sibling-aware whole-cloud sizing
            # would let the first chunk swallow an entire part.
            return add_points_partition_wrapper_impl(
                group,
                name=name,
                pos_arr=pos_arr,
                parts=parts,
                n_points=n_points,
                colors=colors,
                radii=radii,
                sharpness=sharpness,
                scalars=scalars,
                labels=labels,
                keys=keys,
                parent=parent,
                extend_to_all=extend_to_all,
                max_elements=max_elements,
                bsp_tree=bsp_tree,
                additive_lod=finest_additive,
                additive_lod_slices=slices,
                **attrs,
            )
        return add_points_impl(
            group,
            name=name,
            positions=pos_arr,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
            scalars=scalars,
            labels=labels,
            keys=keys,
            image_labels=image_labels,
            parent=parent,
            extend_to_all=extend_to_all,
            partition=False,
            # Forward the ladder: a cloud too small to coarsen is not
            # necessarily too small to stream, and dropping it here was how the
            # ladder silently vanished on the degenerate path.
            additive_lod=finest_additive,
            **attrs,
        )

    # Children coarsest -> finest: [coarsest gsplat .. finest gsplat, points].
    coarse_first = list(reversed(coarse))  # coarsest first
    # Finest "count" is the lifted BEAD count: the coarse gsplat children are bead
    # reductions of ``lifted`` and the finest radius is measured on ``lifted`` too,
    # so the count/extent pair stays in one bead currency (matches lines.py). For
    # Points the finest node rejects non-positive radii, so no points are dropped
    # by the lift and ``lifted.n_splats == n_points`` — but keying off ``lifted``
    # keeps the two geometries' wrappers structurally identical and robust.
    counts = [int(c.n_splats) for c in coarse_first] + [int(lifted.n_splats)]
    parent_node = parent or group
    # Thresholds AND the selector naming their units, from the one shared rule
    # (``lod.group.resolve_lod_ladder``): an explicit ``coverage_fractions=[...]``
    # is used verbatim under the legacy units it was authored in, otherwise the
    # screen-area halving ladder is derived — re-anchored at fills-screen when the
    # insertion point is partition-bound or the finest child is the partitioned
    # branch of an overview topology.
    coverage_vals, lod_selector = resolve_lod_ladder(
        spec.get("coverage_fractions"),
        counts,
        parent_node,
        name=name,
        partition_bound=fine_partition is not None,
        length_error=lambda n_explicit, n_levels: (
            f"coverage_fractions has {n_explicit} entries but the LOD ladder "
            f"has {n_levels} levels ({len(coarse_first)} gsplat + 1 finest)"
        ),
    )

    # Compositing attrs ride on the kind=lod Group; everything else (colormap,
    # truncation_radius, ...) rides onto each child.
    lod_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    child_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}
    child_attrs.pop("coverage_fraction", None)
    lod_attrs.setdefault("display_type", "points")  # finest child is points
    # The coarse gsplat children carry baked per-splat colours (from the lift),
    # so they must NOT also receive `colormap` (gsplats reject colors+colormap).
    # `colormap` (when scalars were baked) stays on the finest Points child only.
    gsplat_child_attrs = {k: v for k, v in child_attrs.items() if k != "colormap"}

    aprint(
        f"  📐 Substitutive-LOD '{name}': {len(coarse_first)} gsplat levels + finest "
        f"(counts coarsest→finest={counts}, K={spec['compression_factor']})"
    )
    lod_group_node = parent_node.add_lod_group(name, selector=lod_selector, **lod_attrs)

    # Coarse gsplat children (coarsest first). pos_arr is already
    # dim_order-transformed, so children use dim_order=None/fill=None.
    for idx, lvl_data in enumerate(coarse_first):
        # Every level gets its own ladder (mirrors gsplats/lod/pyramid.py), with
        # the sibling-aware first chunk on all but the coarsest. Levels smaller
        # than one chunk resolve to a single level and stay flat leaves.
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
            **(
                level_attrs_with_quality(
                    gsplat_child_attrs, float(lvl_data.stats["quality"])
                )
                if spec["quality_stamps"]
                else gsplat_child_attrs
            ),
        )

    if fine_partition is not None:
        max_elements, parts, bsp_tree = fine_partition
        # Each part sizes its own ladder; sibling-aware whole-cloud sizing
        # would let the first chunk swallow an entire part.
        add_points_partition_wrapper_impl(
            group,
            name=f"child_{len(coarse_first)}",
            pos_arr=pos_arr,
            parts=parts,
            n_points=n_points,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
            scalars=scalars,
            labels=labels,
            keys=keys,
            parent=lod_group_node,
            extend_to_all=extend_to_all,
            max_elements=max_elements,
            bsp_tree=bsp_tree,
            additive_lod=finest_additive,
            additive_lod_slices=slices,
            wrapper_coverage_fraction=coverage_vals[-1],
            **(
                level_attrs_with_quality(child_attrs, 1.0)
                if spec["quality_stamps"]
                else child_attrs
            ),
        )
        return lod_group_node

    # Finest child: the original Points node (carries all N points + image_labels;
    # partition=False so the auto-partition heuristic cannot split it underneath).
    lod_group_node.add_points(
        f"child_{len(coarse_first)}",
        pos_arr,
        colors=colors,
        radii=radii,
        sharpness=sharpness,
        scalars=scalars,
        labels=labels,
        keys=keys,
        image_labels=image_labels,
        extend_to_all=extend_to_all,
        dim_order=None,
        fill=None,
        partition=False,
        additive_lod=level_additive_lod(
            finest_additive,
            level_n=n_points,
            compression_factor=compression_factor,
            is_coarsest=False,
            slices=slices,
        ),
        substitutive_lod=None,
        coverage_fraction=coverage_vals[-1],
        **(
            level_attrs_with_quality(child_attrs, 1.0)
            if spec["quality_stamps"]
            else child_attrs
        ),
    )

    return lod_group_node
