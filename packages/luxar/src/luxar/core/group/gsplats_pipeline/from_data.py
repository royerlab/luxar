"""`add_gsplats_from_data` body — high-level dispatch over GSplatData.

Resolves the two LOD axes (substitutive via ``lod_group=`` and additive
via ``additive_lod=``), then routes to the appropriate write path:

* multi-substitutive → kind=lod Group via :func:`add_gsplats_as_lod_group_impl`
* single-substitutive + multi-additive → multi-LOD subgroups via
  :func:`add_gsplats_multi_lod_impl`
* single-substitutive + single-additive → flat single-leaf gsplats node
  via ``Group.add_gsplats``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .lod_dispatch import (
    add_gsplats_as_lod_group_impl,
    add_gsplats_multi_lod_impl,
)

if TYPE_CHECKING:
    from ...gsplats import GSplats
    from ...node import Node
    from ..group import Group


def add_gsplats_from_data_impl(
    group: "Group",
    *,
    name: str,
    result: Any,  # GSplatData (runtime-typed; importing it here pulls heavy deps)
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    fill_sigma: Optional[Dict[str, float]] = None,
    lod_group: Any = None,
    additive_lod: Any = None,
    **attrs: Any,
) -> Union["GSplats", "Group"]:
    from luxar.gsplats.gsplat_data import GSplatData

    from ..lod.gsplats import (
        resolve_additive_axis_gsplats,
        resolve_substitutive_axis_gsplats,
    )

    if not isinstance(result, GSplatData):
        raise TypeError(f"Expected GSplatData, got {type(result).__name__}")

    # Propagate truncation_radius through attrs (unless caller overrode it)
    if "truncation_radius" not in attrs:
        attrs["truncation_radius"] = result.truncation_radius

    # Resolve the two LOD axes. Substitutive first (it can produce a
    # multi-level result), then additive (uniform across levels).
    result, explicit_min_pixel_sizes, base_pixel_size = resolve_substitutive_axis_gsplats(
        result, lod_group
    )
    result = resolve_additive_axis_gsplats(result, additive_lod)

    # Multi-substitutive → kind=lod Group with one gsplats child per level
    if result.n_substitutive > 1:
        if "min_pixel_size" in attrs:
            raise ValueError(
                "min_pixel_size must not be passed when the resolved "
                "result is multi-substitutive: thresholds are derived "
                "per-child (or set via lod_group=dict(min_pixel_sizes="
                "[...]))."
            )
        return add_gsplats_as_lod_group_impl(
            group,
            name=name,
            result=result,
            explicit_min_pixel_sizes=explicit_min_pixel_sizes,
            base_pixel_size=base_pixel_size,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            **attrs,
        )

    # Single-substitutive: flat or multi-additive path
    if result.n_additive_sublods <= 1:
        return group.add_gsplats(
            name=name,
            centers=result.centers,
            amplitudes=result.amplitudes,
            cholesky_factors=result.cholesky_factors,
            colors=result.colors,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            **attrs,
        )

    return add_gsplats_multi_lod_impl(
        group,
        name=name,
        result=result,
        parent=parent,
        extend_to_all=extend_to_all,
        dim_order=dim_order,
        fill=fill,
        fill_sigma=fill_sigma,
        **attrs,
    )
