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
    explicit_coverage_fractions: Optional[List[float]],
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
    from ..lod.group import coverage_fractions

    # Substitutive convention: index 0 = finest, n-1 = coarsest. The
    # LOD group needs coarsest first.
    n_sub = result.n_substitutive
    order = list(range(n_sub - 1, -1, -1))

    # Per-level total splat count (sum across each level's additive
    # ladder) — used both for logging and for auto-deriving
    # coverage_fractions when the user didn't supply them.
    splat_counts: List[int] = [
        sum(sub.n_splats for sub in result.substitutive_levels[s].additive_sublods)
        for s in order
    ]

    if explicit_coverage_fractions is not None:
        if len(explicit_coverage_fractions) != n_sub:
            raise ValueError(
                f"coverage_fractions has {len(explicit_coverage_fractions)} "
                f"entries but the lod_group has {n_sub} substitutive levels"
            )
        coverage_vals = list(explicit_coverage_fractions)
    else:
        # Auto-derive (coarsest-first) as viewport-relative coverage fractions
        # ``sqrt(N_i/N_finest)`` — count ratios only, so no per-level radius or
        # world-extent is needed; the viewer anchors the finest at fills-screen.
        coverage_vals = coverage_fractions(splat_counts)

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
    # Defense in depth: ``add_gsplats_from_data`` rejects coverage_fraction
    # at entry, but if this method is invoked through a different path
    # (e.g. internal recursion) we still need to strip it — the loop
    # below passes a derived value as an explicit kwarg, and a duplicate
    # in ``**child_attrs`` would raise TypeError.
    child_attrs.pop("coverage_fraction", None)

    parent_node = parent or group
    # All children in this branch are gsplats leaves, so the user-facing
    # ``display_type`` is unambiguously "gsplats". Set it here so the
    # on-disk attrs are self-describing.
    lod_attrs.setdefault("display_type", "gsplats")
    # default_level (the viewer's INITIAL level, before the coverage selector
    # runs) is left at add_lod_group's default of 0 = the COARSEST child
    # (child_0). This is a progressive-load hint: the scene appears instantly at
    # low detail, then refines. It is intentionally decoupled from the data-model
    # default_substitutive (the finest level the accessors return) — defaulting
    # the viewer to the finest would eager-load every lod group at full
    # resolution and render "backwards" (matches gsplat_tree.write_gsplat_node).
    lod_group_node = parent_node.add_lod_group(name, **lod_attrs)

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
            coverage_fraction=coverage_vals[child_idx],
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
    """Write multi-additive-LOD GSplatData as a single leaf via the shared walker.

    Builds a :class:`~luxar.gsplats.tree.GSplatLeaf` (additive ladder) — applying
    ``dim_order`` per sub-LOD as a tree transform — and hands it to
    :meth:`LuxarZarrCompiler.write_gsplat_leaf_subtree`, the single authoring path
    shared with the standalone ``.gsplats.zarr`` writer. There is no parallel
    additive-ladder writer.
    """
    try:
        from ....gsplats.gsplat_data import AdditiveSubLOD
        from ....gsplats.tree import GSplatLeaf

        scene = group._find_scene()
        d_data = result.ndim

        # Build per-sub-LOD AdditiveSubLODs, applying dim_order as a tree
        # transform on each (centers + Cholesky reshaped/padded into scene dims).
        sublods: List[AdditiveSubLOD] = []
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

            sublods.append(
                AdditiveSubLOD(
                    centers=ctr_arr.astype(np.float32),
                    amplitudes=lod.amplitudes,
                    cholesky_factors=chol_arr,
                    colors=lod.colors,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )
            )

        leaf = GSplatLeaf(additive_sublods=sublods)

        n_splats = result.n_splats
        ndim = sublods[0].centers.shape[1]
        aprint(
            f"Adding multi-LOD gsplats node '{name}' with "
            f"{n_splats:,} splats in {ndim}D ({result.n_additive_sublods} LODs)."
        )

        final_extend_dims = scene._resolve_extend_to_all(
            extend_to_all, sublods[0].centers, "splats"
        )
        if final_extend_dims:
            attrs["extend_to_all"] = final_extend_dims
            aprint(f"  📡 Extending visibility across: {final_extend_dims}")

        colormap = attrs.get("colormap")
        if any(s.colors is not None for s in sublods) and colormap is not None:
            raise ValueError(
                "Cannot specify both 'colors' and 'colormap'. Use one or the other."
            )

        parent_node = parent or group
        writer = group._require_scene_writer(scene)
        path = f"{parent_node.path}/{name}" if parent_node.path else name

        metadata = writer.write_gsplat_leaf_subtree(  # type: ignore[attr-defined]
            path,
            leaf,
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
