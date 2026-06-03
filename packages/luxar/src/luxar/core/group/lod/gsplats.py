"""luxar.core.group.lod.gsplats – GSplats-specific LOD axis resolvers.

Interprets the ``lod_group=`` (substitutive) and ``additive_lod=`` (additive)
convenience kwargs against a :class:`~luxar.gsplats.gsplat_data.GSplatData`
input, used by ``Scene.add_gsplats_from_data(...)``. These two resolvers are
the GSplats counterparts to ``lod.points.resolve_additive_axis_points`` /
``lod.lines.resolve_additive_axis_lines`` — peers, one per leaf geometry.

The geometry-agnostic machinery they build on — ``derive_min_pixel_sizes``,
``validate_lod_group``, ``resolve_display_type``, ``BASE_PIXEL_SIZE`` and the
monotonicity guard — lives in the type-neutral :mod:`luxar.core.group.lod.group`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from .group import _assert_strict_ascending

if TYPE_CHECKING:
    from ....gsplats.gsplat_data import GSplatData


#: Sentinel-typed alias for the value vocabulary of ``lod_group=`` and
#: ``additive_lod=``. Order: ``None`` (pass-through), ``True`` (require
#: stored), ``False`` (collapse to finest), ``dict[str, Any]`` (compute,
#: optionally with ``recompute=True``).
LODAxisSpec = Union[None, bool, dict]


# ────────────────────────────────────────────────────────────────────────
# Resolvers for the ``lod_group=`` / ``additive_lod=`` convenience kwargs
# ────────────────────────────────────────────────────────────────────────


def resolve_substitutive_axis_gsplats(
    data: "GSplatData",
    spec: LODAxisSpec,
) -> Tuple["GSplatData", Optional[List[float]], Optional[float]]:
    """Apply ``lod_group=`` semantics to ``data``.

    Returns ``(resolved_data, explicit_min_pixel_sizes_or_None,
    base_pixel_size_or_None)``.

    - ``explicit_min_pixel_sizes`` is non-None only when the user passed
      ``dict(min_pixel_sizes=[...])`` — otherwise downstream code
      auto-derives from per-level splat counts via
      :func:`luxar.core.group.lod.group.derive_min_pixel_sizes`.
    - ``base_pixel_size`` is non-None only when the user passed
      ``dict(base_pixel_size=...)``. It overrides the global
      :data:`luxar.core.group.lod.group.BASE_PIXEL_SIZE` default for the
      auto-derivation path.

    Semantics:

    +---------------------------+-----------------------------------------+
    | Spec                      | Behaviour                               |
    +===========================+=========================================+
    | ``None`` (default)        | Auto-lower: if ``n_substitutive > 1``,  |
    |                           | keep the full pyramid so it becomes a   |
    |                           | ``kind=lod`` Group (no work discarded). |
    |                           | Pass ``lod_group=False`` to collapse to |
    |                           | the finest level instead.               |
    +---------------------------+-----------------------------------------+
    | ``True``                  | Require ``n_substitutive > 1`` already; |
    |                           | raise otherwise. Use stored.            |
    +---------------------------+-----------------------------------------+
    | ``False``                 | Collapse to finest substitutive level   |
    |                           | (index 0), discard the rest.            |
    +---------------------------+-----------------------------------------+
    | ``dict(...)``             | Use stored if present; else compute via |
    |                           | :func:`make_substitutive_lod` with the  |
    |                           | dict as kwargs.                         |
    +---------------------------+-----------------------------------------+
    | ``dict(..., recompute=    | Force recompute, ignoring stored.       |
    | True)``                   |                                         |
    +---------------------------+-----------------------------------------+
    """
    explicit_min_pixel_sizes: Optional[List[float]] = None
    base_pixel_size: Optional[float] = None

    if spec is None:
        # Auto-lower: a multi-substitutive pyramid is expensive to build, so
        # the default no longer silently drops it. Returning the full data
        # routes it to the kind=lod Group builder (same as ``lod_group=True``)
        # downstream. Use ``lod_group=False`` to collapse to the finest level.
        return data, None, None

    if spec is True:
        if data.n_substitutive <= 1:
            raise ValueError(
                "lod_group=True requires the input GSplatData to already "
                "carry substitutive levels (n_substitutive > 1); got "
                f"n_substitutive={data.n_substitutive}. Use "
                "lod_group=dict(...) to compute them on the fly."
            )
        return data, None, None

    if spec is False:
        if data.n_substitutive > 1:
            # Convention: finest substitutive level is index 0.
            return data.at_substitutive(0), None, None
        return data, None, None

    if isinstance(spec, dict):
        kwargs = dict(spec)  # copy — don't mutate caller's dict
        recompute = bool(kwargs.pop("recompute", False))
        explicit_min_pixel_sizes = kwargs.pop("min_pixel_sizes", None)
        if explicit_min_pixel_sizes is not None:
            explicit_min_pixel_sizes = [float(v) for v in explicit_min_pixel_sizes]
            # Fail fast on bad explicit lists at the resolver instead of
            # deferring to a later validate_lod_group() the user may never
            # call. Same invariant the auto-derivation enforces.
            _assert_strict_ascending(
                explicit_min_pixel_sizes, "lod_group=dict(min_pixel_sizes=...)"
            )
        bps_raw = kwargs.pop("base_pixel_size", None)
        if bps_raw is not None:
            base_pixel_size = float(bps_raw)
            if base_pixel_size <= 0:
                raise ValueError(
                    "lod_group=dict(base_pixel_size=...) must be positive; "
                    f"got {base_pixel_size}"
                )

        if data.n_substitutive > 1 and not recompute:
            # Stored pyramid takes precedence — but silently accepting
            # compute-affecting kwargs would mask user intent. Reject any
            # leftover keys (only ``min_pixel_sizes`` / ``base_pixel_size``
            # / ``recompute`` are honored on the stored path).
            if kwargs:
                raise ValueError(
                    "lod_group=dict(...) carries compute kwargs "
                    f"({sorted(kwargs)}) but data already has "
                    f"n_substitutive={data.n_substitutive}. Pass "
                    "recompute=True to override the stored pyramid, or "
                    "drop the compute kwargs to reuse it."
                )
            return data, explicit_min_pixel_sizes, base_pixel_size

        # Compute. Align with ``make_substitutive_lod``'s canonical default
        # (3 levels) so ``lod_group=dict()`` yields the same pyramid as a
        # bare ``make_substitutive_lod(data)`` call.
        kwargs.setdefault("levels", 3)
        from ....gsplats.lod.substitutive import make_substitutive_lod

        new_data = make_substitutive_lod(data, **kwargs)
        return new_data, explicit_min_pixel_sizes, base_pixel_size

    raise TypeError(f"lod_group must be None, bool, or dict; got {type(spec).__name__}")


def resolve_additive_axis_gsplats(
    data: "GSplatData",
    spec: LODAxisSpec,
) -> "GSplatData":
    """Apply ``additive_lod=`` semantics uniformly across substitutive levels.

    Semantics mirror :func:`resolve_substitutive_axis_gsplats` but operate per
    substitutive level (each level's additive ladder is treated
    independently).

    +---------------------------+-----------------------------------------+
    | Spec                      | Behaviour                               |
    +===========================+=========================================+
    | ``None`` (default)        | Pass-through; use whatever ladder each  |
    |                           | substitutive level carries.             |
    +---------------------------+-----------------------------------------+
    | ``True``                  | Require every substitutive level to     |
    |                           | already carry ``> 1`` additive sub-LODs.|
    +---------------------------+-----------------------------------------+
    | ``False``                 | Flatten each substitutive level's       |
    |                           | additive ladder into a single sub-LOD.  |
    +---------------------------+-----------------------------------------+
    | ``dict(...)``             | Compute ladder on each level missing    |
    |                           | one; pass stored through otherwise.     |
    +---------------------------+-----------------------------------------+
    | ``dict(..., recompute=    | Force recompute on every level.         |
    | True)``                   |                                         |
    +---------------------------+-----------------------------------------+
    """
    from ....gsplats.gsplat_data import GSplatData, SubstitutiveLevel

    if spec is None:
        return data

    if spec is True:
        for i, lvl in enumerate(data.substitutive_levels):
            if lvl.n_additive_lods <= 1:
                raise ValueError(
                    "additive_lod=True requires every substitutive level "
                    "to already carry an additive ladder "
                    "(n_additive_lods > 1); substitutive level "
                    f"{i} has n_additive_lods={lvl.n_additive_lods}. Use "
                    "additive_lod=dict(...) to compute on the fly."
                )
        return data

    if spec is False:
        # Flatten each substitutive level into a single AdditiveSubLOD.
        new_levels: list[SubstitutiveLevel] = []
        for lvl in data.substitutive_levels:
            single = (
                GSplatData(additive_sublods=list(lvl.additive_sublods))
                .flattened()
                .additive_sublods[0]
            )
            new_levels.append(
                SubstitutiveLevel(
                    additive_sublods=[single],
                    compression_factor=lvl.compression_factor,
                    parent_method=lvl.parent_method,
                    level_index=lvl.level_index,
                    stats=dict(lvl.stats),
                )
            )
        return GSplatData.from_substitutive_levels(
            new_levels,
            stats=dict(data.stats),
            default_substitutive=data.default_substitutive,
        )

    if isinstance(spec, dict):
        kwargs = dict(spec)
        recompute = bool(kwargs.pop("recompute", False))
        # Default to 4-level ladder for a bare ``dict()`` so the convenience
        # API yields a useful result without parameters.
        kwargs.setdefault("n_lods", 4)
        from ....gsplats.lod.additive import make_additive_lod

        result = data
        for s in range(result.n_substitutive):
            needs_compute = (
                recompute or result.substitutive_levels[s].n_additive_lods <= 1
            )
            if needs_compute:
                result = make_additive_lod(result, substitutive_level=s, **kwargs)
        return result

    raise TypeError(
        f"additive_lod must be None, bool, or dict; got {type(spec).__name__}"
    )
