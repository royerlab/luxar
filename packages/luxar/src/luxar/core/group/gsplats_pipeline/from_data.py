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


def _reject_before_wrapper(
    group: "Group",
    *,
    name: str,
    result: Any,  # GSplatData
    dim_order: Optional[List[str]],
    fill: Optional[Dict[str, float]],
    fill_sigma: Optional[Dict[str, float]],
    colormap: Any,
) -> None:
    """Refuse what is judgeable up front BEFORE the ``kind=lod`` group exists (#1446).

    :func:`add_gsplats_as_lod_group_impl` calls ``add_lod_group`` first and writes
    children afterwards, so without this the refusal came from inside ``child_0``
    — blaming an internal child the caller never wrote and leaving ``name`` in the
    store as a childless ``kind=lod`` group, where the same data with
    ``lod_group=False`` writes nothing at all. The other two dispatch targets need
    no such gate: the flat path is ``Group.add_gsplats``, which checks inside its
    own funnel, and :func:`add_gsplats_multi_lod_impl` validates every sub-LOD
    while building them, before its single write.

    The colours/colormap exclusion goes FIRST, exactly as it does at the top of
    the three leaf adders: it is a kwarg fault, and kwarg faults outrank input
    faults everywhere else in this codebase. Checking the width first would make
    ``lod_group=`` disagree with its own flat path about which fault a call that
    trips both is told about — the one invariant this whole change is for. It also
    closes the same stranding class for a colours+colormap call whose width is
    perfectly fine, which used to reach ``child_0`` with the wrapper on disk.

    WHAT else is checkable up front depends on ``dim_order``. Without one, the
    incoming column count must already equal the scene's, which is exactly
    ``validate_dimension_count`` (rank guard included). With one, the
    post-transform width is ``scene_ndim`` by construction (``apply_dim_order``
    allocates it that way), so the count check can never fire downstream — what
    fires there instead is one of the SPEC refusals: ``dim_order``'s
    length-vs-columns, duplicate names, a name absent from the scene, a bad
    ``fill`` key (all in ``validate_dim_order_spec``) or a bad ``fill_sigma`` key
    (``validate_fill_sigma_keys``). All of them are judged from the spec plus the
    column count alone, so all of them are checkable here. Everything left
    downstream genuinely needs the transformed per-level arrays.

    Called from INSIDE the multi-substitutive branch, below the
    ``coverage_fraction`` refusal, so that structural-kwarg fault outranks
    everything here — the same shape as the ordering inside the gate itself. The
    cost is that an invalid call still pays for the substitutive resolve above;
    correct-precedence-first is the deliberate trade.

    ``TypeError`` is caught alongside ``ValueError`` because the spec validators
    can raise it on a non-sequence ``dim_order`` (``len()`` of an int), and the
    leaf adders' funnel converts both to a ``ValueError`` — a split path that
    leaked the raw ``TypeError`` would diverge in exception TYPE, which is
    precisely what the parity tests compare. The ``Could not add gsplats
    '<name>': `` prefix is applied by hand because this module has no try/except
    funnel of its own; it is byte-identical to what ``add_gsplats_impl``
    produces, which the #1446 tests pin against a direct ``add_gsplats`` call.
    """
    from ...scene.dim_order import validate_dim_order_spec
    from ...scene.validation import validate_array_rank
    from ..dim_order import validate_fill_sigma_keys

    centers = result.centers
    try:
        # ``result.colors`` is what the flat path forwards as ``colors=`` (the
        # finest level's), so this asks the same question of the same values.
        if result.colors is not None and colormap is not None:
            raise ValueError(
                "Cannot specify both 'colors' and 'colormap'. Use one or the other."
            )
        if dim_order is not None:
            validate_array_rank(centers, "centers")
            # Same order as the flat path, which applies dim_order to the centers
            # (spec + fill) before it embeds the Cholesky factors (fill_sigma).
            scene = group._find_scene()
            validate_dim_order_spec(scene, dim_order, centers.shape[1], fill)
            validate_fill_sigma_keys(scene, dim_order, fill_sigma)
        else:
            group._find_scene()._validate_dimension_count(
                centers, name, data_type="centers"
            )
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not add gsplats '{name}': {e}") from e


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

    # Resolve coarsen_dims (which dims substitutive coarsening may merge over)
    # on the COMPUTE path only — scene + data ndim are available here, but the
    # stored-pyramid path rejects compute kwargs. Default Auto = displayed dims,
    # grouping by non-displayed dims (same semantics as Points/Lines).
    if isinstance(lod_group, dict):
        computing = result.n_substitutive <= 1 or bool(lod_group.get("recompute"))
        if computing:
            from ..lod.group import _validate_coarsen_dims_spec, resolve_coarsen_dims

            raw = _validate_coarsen_dims_spec(lod_group.get("coarsen_dims"))
            resolved = resolve_coarsen_dims(group._find_scene(), int(result.ndim), raw)
            lod_group = {**lod_group, "coarsen_dims": resolved}

    # Resolve the two LOD axes. Substitutive first (it can produce a
    # multi-level result), then additive (uniform across levels).
    result, explicit_coverage_fractions = resolve_substitutive_axis_gsplats(
        result, lod_group
    )
    result = resolve_additive_axis_gsplats(result, additive_lod)

    # Multi-substitutive → kind=lod Group with one gsplats child per level
    if result.n_substitutive > 1:
        if "coverage_fraction" in attrs:
            raise ValueError(
                "coverage_fraction must not be passed when the resolved "
                "result is multi-substitutive: thresholds are derived "
                "per-child (or set via lod_group=dict(coverage_fractions="
                "[...]))."
            )
        _reject_before_wrapper(
            group,
            name=name,
            result=result,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            colormap=attrs.get("colormap"),
        )
        return add_gsplats_as_lod_group_impl(
            group,
            name=name,
            result=result,
            explicit_coverage_fractions=explicit_coverage_fractions,
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
