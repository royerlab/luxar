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
    COMPOSITING_ATTRS,
    position_bounds_from_array,
    slice_optional_array,
    sync_custom_colormap_attr,
)
from ..dim_order import apply_dim_order_cholesky, apply_dim_order_positions

if TYPE_CHECKING:
    from ...node import Node
    from ..group import Group


def add_gsplats_impl(
    group: "Group",
    *,
    name: str,
    centers: Any,
    amplitudes: Any,
    cholesky_factors: Any,
    colors: Any = None,
    labels: Optional[Union[List[str], Sequence[str]]] = None,
    image_labels: Optional[Any] = None,
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    fill_sigma: Optional[Dict[str, float]] = None,
    partition: Any = None,
    **attrs: Any,
) -> Union[GSplats, "Group"]:
    try:
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

        # Apply compiler-level auto-partition heuristic (opt-in; default
        # off). User-explicit ``partition=`` always wins.
        partition = resolve_auto_partition(scene, n_splats, partition)

        # Partition branch — decompose into N children if the user opted in
        # AND the BSP produces more than one part.
        if partition is not None and ctr_arr.shape[1] >= 3:
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
                parts = sah_bsp_partition(ctr_arr, max_elements)
            elif partition_rule == "midpoint":
                parts = midpoint_bsp_partition(ctr_arr, max_elements)
            else:
                parts = median_bsp_partition(ctr_arr, max_elements)
            warn_if_oversized_single_part(
                len(parts), int(parts[0].size) if parts else 0, max_elements, name
            )
            if len(parts) > 1:
                return add_gsplats_partition_wrapper_impl(
                    group,
                    name=name,
                    ctr_arr=ctr_arr,
                    chol_arr=chol_arr,
                    amplitudes=amplitudes,
                    parts=parts,
                    n_splats=n_splats,
                    colors=colors,
                    labels=labels,
                    parent=parent,
                    extend_to_all=extend_to_all,
                    max_elements=max_elements,
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

        # Colormap / colors mutual exclusivity
        colormap = attrs.get("colormap")
        if colors is not None and colormap is not None:
            raise ValueError(
                "Cannot specify both 'colors' and 'colormap'. Use one or the other."
            )

        parent_node = parent or group

        writer = group._require_scene_writer(scene)
        path = f"{parent_node.path}/{name}" if parent_node.path else name
        metadata = writer.write_gsplats(
            path,
            ctr_arr.astype(np.float32),
            amplitudes=amplitudes,
            cholesky_factors=chol_arr,
            colors=cast(Any, colors),
            labels=labels,
            image_labels=image_labels,
            **attrs,
        )

        # Sync colormap attr with what the compiler wrote to zarr
        sync_custom_colormap_attr(attrs)

        # The compiler sets default "gray" colormap for gsplats without
        # colors/colormap. Propagate that to the Node attrs so the
        # in-memory node matches the zarr state.
        if not metadata.get("has_colors") and "colormap" not in attrs:
            attrs["colormap"] = "gray"

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
        aprint(f"Failed to add gsplats node '{name}': {e}")
        raise ValueError(f"Could not add gsplats '{name}': {e}") from e


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
    labels: Any,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    max_elements: int,
    **attrs: Any,
) -> "Group":
    """Build a kind=partition wrapper Group with one GSplats child per BSP part."""
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
            cholesky_factors=slice_optional_array(chol_arr, indices, n_splats),
            colors=slice_optional_array(colors, indices, n_splats),
            labels=slice_optional_array(labels, indices, n_splats),
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

    wrapper._persist_attr("position_bounds", position_bounds_from_array(ctr_arr))

    return wrapper
