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
    **attrs: Any,
) -> Union[Points, "Group"]:
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
