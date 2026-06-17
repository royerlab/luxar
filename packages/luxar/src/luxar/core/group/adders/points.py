"""add_points body + partition/multi-LOD wrapper impls.

Pure functions taking a ``group: Group`` parameter as the first arg.
Called by ``Group.add_points`` (a thin signature + docstring + delegate)
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
    Tuple,
    Union,
    cast,
)

import numpy as np
from arbol import aprint

from ...points import Points
from ..auto_partition import resolve_auto_partition
from ..compositing import (
    COMPOSITING_ATTRS,
    position_bounds_from_array,
    slice_optional_array,
)
from ..dim_order import apply_dim_order_positions

if TYPE_CHECKING:
    from ...node import Node
    from ..group import Group


# Default radius used when radii are not provided
DEFAULT_POINT_RADIUS = 0.5


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
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    grid_shape: Optional[Tuple[int, ...]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    partition: Any = None,
    additive_lod: Any = None,
    substitutive_lod: Any = None,
    **attrs: Any,
) -> Union[Points, "Group"]:
    if additive_lod is not None and substitutive_lod is not None:
        raise ValueError(
            "additive_lod and substitutive_lod are mutually exclusive coarsening "
            "strategies for Points (append vs replace on the same LOD axis); "
            "pass only one."
        )
    if substitutive_lod is not None and partition is not None:
        raise ValueError(
            "partition= and substitutive_lod= cannot be combined yet "
            "(partition-of-substitutive — a kind=partition of per-part gsplat "
            "LOD ladders — is not implemented). Use one or the other."
        )
    try:
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

        # Substitutive-LOD branch — coarse levels are synthesised gsplats (each
        # point lifted to an isotropic Gaussian, then reduced by the gsplat
        # substitutive pipeline) under a kind=lod Group whose finest child is the
        # original Points node. Fires BEFORE (auto-)partition so substitutive
        # takes precedence over the opt-in auto-partition heuristic; explicit
        # partition= is rejected up front (mutually exclusive, checked above).
        # ``pos_arr`` is already dim_order-transformed, so children are written
        # with dim_order=None/fill=None to avoid double application.
        if substitutive_lod is not None and n_points > 0:
            from ..lod.points import resolve_substitutive_axis_points

            substitutive_spec = resolve_substitutive_axis_points(substitutive_lod)
            if substitutive_spec is not None:
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
                    image_labels=image_labels,
                    parent=parent,
                    extend_to_all=extend_to_all,
                    grid_shape=grid_shape,
                    spec=substitutive_spec,
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
        if partition is not None and pos_arr.shape[1] >= 3:
            from ..partition import (
                DEFAULT_MAX_ELEMENTS,
                median_bsp_partition,
                midpoint_bsp_partition,
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

            if partition_rule == "sah":
                parts = sah_bsp_partition(pos_arr, max_elements)
            elif partition_rule == "midpoint":
                parts = midpoint_bsp_partition(pos_arr, max_elements)
            else:
                parts = median_bsp_partition(pos_arr, max_elements)
            warn_if_oversized_single_part(
                len(parts), int(parts[0].size) if parts else 0, max_elements, name
            )
            if len(parts) > 1:
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
                    parent=parent,
                    extend_to_all=extend_to_all,
                    grid_shape=grid_shape,
                    max_elements=max_elements,
                    additive_lod=additive_lod,
                    **attrs,
                )
            # 1 part → fall through to additive-LOD / single-leaf write.

        # Additive-LOD branch — multi-level progressive writes via
        # add_points_multi_lod_wrapper_impl. Fires after the
        # 1-part-partition fall-through so a user can pass both
        # ``partition=`` and ``additive_lod=`` and get the inner LOD
        # ladder when partition doesn't fire.
        if additive_lod is not None:
            from ..lod.points import (
                make_additive_lod_points,
                resolve_additive_axis_points,
            )

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
                        parent=parent,
                        extend_to_all=extend_to_all,
                        grid_shape=grid_shape,
                        method=additive_spec["method"],
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
            image_labels=image_labels,
            grid_shape=grid_shape,
            **attrs,
        )

        # Sync colormap attr with what the compiler wrote to zarr:
        # - Array colormaps are resolved and stored as "custom"
        # - Non-built-in string names (matplotlib/colorcet) are also
        #   resolved to LUT and stored as "custom"
        if "colormap" in attrs:
            from ....colormaps.builtins import BUILTIN_COLORMAP_NAMES

            cm = attrs["colormap"]
            if not isinstance(cm, str) or (
                isinstance(cm, str) and cm not in BUILTIN_COLORMAP_NAMES
            ):
                attrs["colormap"] = "custom"

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
        aprint(f"Failed to add points node '{name}': {e}")
        raise ValueError(f"Could not add points '{name}': {e}") from e


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
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    grid_shape: Optional[Tuple[int, ...]],
    max_elements: int,
    additive_lod: Any = None,
    **attrs: Any,
) -> "Group":
    """Build a kind=partition wrapper Group with one Points child per BSP part."""
    wrapper_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
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
        wrapper.add_points(
            name=f"part_{i}",
            positions=pos_arr[indices],
            colors=slice_optional_array(colors, indices, n_points),
            radii=slice_optional_array(radii, indices, n_points),
            sharpness=slice_optional_array(sharpness, indices, n_points),
            scalars=slice_optional_array(scalars, indices, n_points),
            labels=slice_optional_array(labels, indices, n_points),
            # image_labels banned alongside partition= (see add_points entry)
            image_labels=None,
            extend_to_all=extend_to_all,
            grid_shape=grid_shape,
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
            additive_lod=additive_lod,
            **leaf_attrs,
        )

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
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    grid_shape: Optional[Tuple[int, ...]],
    method: str,
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
    """
    scene = group._find_scene()
    writer = group._require_scene_writer(scene)
    parent_node = parent or group
    path = f"{parent_node.path}/{name}" if parent_node.path else name

    # Build per-level slice tuples for the writer.
    level_slices: List[Dict[str, Any]] = []
    for level_indices in levels:
        level_slices.append(
            {
                "positions": pos_arr[level_indices].astype(np.float32),
                "colors": slice_optional_array(colors, level_indices, n_points),
                "radii": slice_optional_array(radii, level_indices, n_points),
                "sharpness": slice_optional_array(sharpness, level_indices, n_points),
                "scalars": slice_optional_array(scalars, level_indices, n_points),
                "labels": slice_optional_array(labels, level_indices, n_points),
            }
        )

    aprint(
        f"  📐 Additive-LOD '{name}': {len(levels)} levels "
        f"(method={method!r}, sizes={[int(L.size) for L in levels]})"
    )

    metadata = writer.write_points_multi_lod(
        path,
        level_slices,
        method=method,
        grid_shape=grid_shape,
        extend_to_all=extend_to_all,
        **attrs,
    )

    return Points(
        name,
        metadata=metadata,
        parent=cast(Any, parent_node),
        writer=writer,
        **attrs,
    )


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
    image_labels: Any,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    grid_shape: Optional[Tuple[int, ...]],
    spec: Dict[str, Any],
    **attrs: Any,
) -> Union["Group", Points]:
    """Write a Points node whose coarse LOD levels are synthesised gsplats.

    Each point is lifted to an isotropic Gaussian (see
    :func:`luxar.gsplats.lift.lift_points_to_gsplats`), the gsplat substitutive
    pipeline (:func:`luxar.gsplats.make_substitutive_lod`) synthesises
    fewer-but-larger representative levels, and those become the coarse children
    of a ``kind=lod`` Group whose **finest** child is the original Points node.
    Coarse-level amplitudes are rescaled to conserve render-light (``sum a·σ³``)
    so brightness is stable across the LOD seam (no zoom-out dimming).

    Compositing attrs (opacity, gamma, ...) land on the ``kind=lod`` Group;
    ``opacity`` is therefore applied once at composite time to both the points
    child and the gsplat children (the lift uses ``opacity=1``).
    """
    from ....gsplats.lift import coarse_substitutive_levels, lift_points_to_gsplats
    from ..lod.group import lod_thresholds

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
        pos_arr, radii_for_lift, colors=colors_for_lift, opacity=1.0,
        truncation_radius=float(spec["truncation_radius"]),
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
    )

    # Degenerate input -> flat Points node rather than a one-child LOD group.
    # Covers BOTH no coarse levels AND an all-zero-radius cloud (the lift yields
    # 0 splats, so every coarse level is empty: coarse[-1] is the coarsest, and
    # writing a 0-element coarsest gsplat child would crash downstream). Mirrors
    # the lines.py degenerate guard. ``pos_arr`` is already dim_order-transformed,
    # so dim_order/fill are None and partition is disabled.
    if not coarse or int(coarse[-1].n_splats) == 0:
        aprint(
            f"  ⚠ substitutive_lod '{name}': input too small to synthesise coarse "
            "levels; writing a flat Points node."
        )
        return add_points_impl(
            group, name=name, positions=pos_arr, colors=colors, radii=radii,
            sharpness=sharpness, scalars=scalars, labels=labels,
            image_labels=image_labels, parent=parent, extend_to_all=extend_to_all,
            grid_shape=grid_shape, partition=False, **attrs,
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
    explicit = spec.get("min_pixel_sizes")
    if explicit is not None:
        if len(explicit) != len(counts):
            raise ValueError(
                f"min_pixel_sizes has {len(explicit)} entries but the LOD ladder "
                f"has {len(counts)} levels ({len(coarse_first)} gsplat + 1 points)"
            )
        min_pixel_sizes = list(explicit)
    else:
        # Per-level element radius (world units), coarsest→finest. The finest
        # level is the points rendered as ``lifted`` gsplats, so use its
        # principal_radii for consistency with the coarse gsplat levels.
        pct = float(spec["extent_percentile"])
        aniso = bool(spec["extent_anisotropy"])
        extents = [
            float(np.percentile(c.principal_radii(aniso), pct)) for c in coarse_first
        ] + [float(np.percentile(lifted.principal_radii(aniso), pct))]
        # Node extent W = world bbox diagonal. Computed from the FINEST level's
        # positions here, whereas the gsplat path (lod_dispatch) takes the union
        # over all substitutive levels' centers. These are EQUAL by construction
        # — a coarse level's representatives are convex/weighted-mean combinations
        # of the finest centers, so they always lie inside the finest bbox; the
        # union therefore equals the finest bbox exactly (verified empirically:
        # 0.0 difference). Keep this finest-only form — it is cheaper and the
        # equality means the three geometries stay self-calibrated together. Do
        # not "fix" the apparent asymmetry; it produces identical thresholds.
        lo, hi = pos_arr.min(axis=0), pos_arr.max(axis=0)
        node_extent = float(np.linalg.norm(hi - lo)) if n_points else None
        min_pixel_sizes = lod_thresholds(
            str(spec["lod_method"]),  # type: ignore[arg-type]
            element_counts=counts,
            element_extents=extents,
            node_extent=node_extent,
            base_pixel_size=spec.get("base_pixel_size"),
        )

    # Compositing attrs ride on the kind=lod Group; everything else (colormap,
    # truncation_radius, ...) rides onto each child.
    lod_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    child_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}
    child_attrs.pop("min_pixel_size", None)
    lod_attrs.setdefault("display_type", "points")  # finest child is points
    # The coarse gsplat children carry baked per-splat colours (from the lift),
    # so they must NOT also receive `colormap` (gsplats reject colors+colormap).
    # `colormap` (when scalars were baked) stays on the finest Points child only.
    gsplat_child_attrs = {k: v for k, v in child_attrs.items() if k != "colormap"}

    parent_node = parent or group
    aprint(
        f"  📐 Substitutive-LOD '{name}': {len(coarse_first)} gsplat levels + points "
        f"(counts coarsest→finest={counts}, K={spec['compression_factor']})"
    )
    lod_group_node = parent_node.add_lod_group(
        name, base_pixel_size=spec.get("base_pixel_size"), **lod_attrs
    )

    # Coarse gsplat children (coarsest first). pos_arr is already
    # dim_order-transformed, so children use dim_order=None/fill=None.
    for idx, lvl_data in enumerate(coarse_first):
        lod_group_node.add_gsplats_from_data(
            name=f"child_{idx}",
            result=lvl_data,
            extend_to_all=extend_to_all,
            dim_order=None,
            fill=None,
            lod_group=None,
            additive_lod=None,
            min_pixel_size=min_pixel_sizes[idx],
            **gsplat_child_attrs,
        )

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
        image_labels=image_labels,
        extend_to_all=extend_to_all,
        grid_shape=grid_shape,
        dim_order=None,
        fill=None,
        partition=False,
        additive_lod=None,
        substitutive_lod=None,
        min_pixel_size=min_pixel_sizes[-1],
        **child_attrs,
    )

    return lod_group_node
