"""LOD-axis dispatch for `add_gsplats_from_data`.

Two free functions:

* :func:`add_gsplats_as_lod_group_impl` — build a kind=lod Group with
  one gsplats child per substitutive level.
* :func:`add_gsplats_multi_lod_impl` — write a multi-additive-LOD
  GSplats node (per-LOD subgroups under one parent gsplats node).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import numpy as np
from arbol import aprint

from ...gsplats import GSplats
from ..compositing import COMPOSITING_ATTRS
from ..dim_order import apply_dim_order_cholesky, apply_dim_order_positions

if TYPE_CHECKING:
    from ....gsplats.gsplat_data import GSplatData
    from ...node import Node
    from ..group import Group


def add_gsplats_as_lod_group_impl(
    group: "Group",
    *,
    name: str,
    result: "GSplatData",
    explicit_min_pixel_sizes: Optional[List[float]],
    base_pixel_size: Optional[float] = None,
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    fill_sigma: Optional[Dict[str, float]] = None,
    **attrs: Any,
) -> "Group":
    """Build a kind=lod ``Group`` with one gsplats child per substitutive level.

    Children are written in coarsest→finest order and named
    ``child_<i>``. Compositing attrs (opacity, gamma, intensity,
    offset, blending_mode, transform, layer, visible, nd_transform)
    land on the kind=lod ``Group`` itself; per-leaf gsplats attrs
    (truncation_radius, extend_to_all, colormap) ride into each child.
    """
    from ..lod.group import derive_min_pixel_sizes

    # Substitutive convention: index 0 = finest, n-1 = coarsest. The
    # LOD group needs coarsest first.
    n_sub = result.n_substitutive
    order = list(range(n_sub - 1, -1, -1))

    # Per-level total splat count (sum across each level's additive
    # ladder) — used both for logging and for auto-deriving
    # min_pixel_sizes when the user didn't supply them.
    splat_counts: List[int] = [
        sum(sub.n_splats for sub in result.substitutive_levels[s].additive_sublods)
        for s in order
    ]

    if explicit_min_pixel_sizes is not None:
        if len(explicit_min_pixel_sizes) != n_sub:
            raise ValueError(
                f"min_pixel_sizes has {len(explicit_min_pixel_sizes)} "
                f"entries but the lod_group has {n_sub} substitutive levels"
            )
        min_pixel_sizes = list(explicit_min_pixel_sizes)
    else:
        min_pixel_sizes = derive_min_pixel_sizes(
            splat_counts, base_pixel_size=base_pixel_size
        )

    # Separate compositing attrs (go on the kind=lod Group) from
    # per-leaf gsplats attrs (go on each child). Anything not in the
    # compositing set falls through to the child level.
    #
    # ``colormap`` is intentionally NOT compositing here: the writer
    # auto-defaults a missing colormap to "gray" per leaf, which
    # under nearest-ancestor-wins would shadow a parent's setting.
    # Keep it on each child so the user's intent survives.
    lod_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    child_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}
    # Defense in depth: ``add_gsplats_from_data`` rejects min_pixel_size
    # at entry, but if this method is invoked through a different path
    # (e.g. internal recursion) we still need to strip it — the loop
    # below passes a derived value as an explicit kwarg, and a duplicate
    # in ``**child_attrs`` would raise TypeError.
    child_attrs.pop("min_pixel_size", None)

    parent_node = parent or group
    # All children in this branch are gsplats leaves, so the user-facing
    # ``display_type`` is unambiguously "gsplats". Set it here so the
    # on-disk attrs are self-describing.
    lod_attrs.setdefault("display_type", "gsplats")
    # Persist the base_pixel_size override on the wrapper (when set)
    # so downstream consumers can identify a non-default ladder
    # without re-deriving from min_pixel_sizes.
    lod_group_node = parent_node.add_lod_group(
        name,
        base_pixel_size=base_pixel_size,
        **lod_attrs,
    )

    aprint(
        f"Adding multi-resolution gsplats node '{name}' as kind=lod "
        f"Group with {n_sub} substitutive levels: "
        f"{[f'{n:,}' for n in splat_counts]} splats"
    )

    # Add one gsplats child per substitutive level, coarsest first.
    for child_idx, s in enumerate(order):
        child_name = f"child_{child_idx}"
        level_view = result.at_substitutive(s)
        # Recursive dispatch — but explicitly None on both LOD axes so
        # the resolvers no-op and we never re-enter the kind=lod branch.
        lod_group_node.add_gsplats_from_data(
            name=child_name,
            result=level_view,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            lod_group=None,
            additive_lod=None,
            min_pixel_size=min_pixel_sizes[child_idx],
            **child_attrs,
        )

    return lod_group_node


def add_gsplats_multi_lod_impl(
    group: "Group",
    *,
    name: str,
    result: "GSplatData",
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    fill_sigma: Optional[Dict[str, float]] = None,
    **attrs: Any,
) -> GSplats:
    """Write multi-LOD GSplatData with per-LOD subgroups."""
    try:
        scene = group._find_scene()
        d_data = result.ndim

        # Build per-LOD tuples, applying dim_order to each LOD
        lod_tuples: List[
            tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]
        ] = []
        lod_stats_list: List[Dict[str, Any]] = []

        for lod in result.additive_sublods:
            ctr_arr = lod.centers.copy()
            chol_arr = lod.cholesky_factors.copy()

            if dim_order is not None:
                ctr_arr, extend_to_all = apply_dim_order_positions(
                    ctr_arr, scene, dim_order, fill, extend_to_all
                )
                chol_arr = apply_dim_order_cholesky(
                    chol_arr, d_data, scene, dim_order, fill_sigma
                )

            scene._validate_data_dimensions(ctr_arr, name, data_type="centers")

            lod_tuples.append(
                (
                    ctr_arr.astype(np.float32),
                    lod.amplitudes,
                    chol_arr,
                    lod.colors,
                )
            )
            lod_stats_list.append(dict(lod.stats))

        n_splats = result.n_splats
        ndim = lod_tuples[0][0].shape[1]
        aprint(
            f"Adding multi-LOD gsplats node '{name}' with "
            f"{n_splats:,} splats in {ndim}D ({result.n_additive_sublods} LODs)."
        )

        final_extend_dims = scene._resolve_extend_to_all(
            extend_to_all, lod_tuples[0][0], "splats"
        )
        if final_extend_dims:
            attrs["extend_to_all"] = final_extend_dims
            aprint(f"  📡 Extending visibility across: {final_extend_dims}")

        colormap = attrs.get("colormap")
        if any(t[3] is not None for t in lod_tuples) and colormap is not None:
            raise ValueError(
                "Cannot specify both 'colors' and 'colormap'. Use one or the other."
            )

        parent_node = parent or group
        writer = group._require_scene_writer(scene)
        path = f"{parent_node.path}/{name}" if parent_node.path else name

        metadata = writer.write_gsplats_multi_lod(  # type: ignore[attr-defined]
            path,
            lods=lod_tuples,
            lod_stats=lod_stats_list,
            **attrs,
        )

        if "colormap" in attrs:
            from ....colormaps.builtins import BUILTIN_COLORMAP_NAMES

            cm = attrs["colormap"]
            if not isinstance(cm, str) or (
                isinstance(cm, str) and cm not in BUILTIN_COLORMAP_NAMES
            ):
                attrs["colormap"] = "custom"

        if not metadata.get("has_colors") and "colormap" not in attrs:
            attrs["colormap"] = "gray"

        return GSplats(
            name,
            metadata=metadata,
            parent=cast(Any, parent_node),
            writer=writer,
            **attrs,
        )
    except (ValueError, TypeError) as e:
        aprint(f"Failed to add multi-LOD gsplats node '{name}': {e}")
        raise ValueError(f"Could not add gsplats '{name}': {e}") from e
