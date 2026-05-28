"""luxar.lod_group – LODGroup scene-graph node + convenience-API helpers.

An ``LODGroup`` selects one of N alternative children at runtime based on a
view-driven metric (currently: the projected bbox diagonal in pixels). It is
geometry-agnostic — children can be points, lines, gsplats, or nested groups.

Each child carries a ``min_pixel_size`` attribute on its own ``.zattrs``: the
viewer picks the finest child whose threshold is satisfied by the current
view. Coarsest level conventionally has ``min_pixel_size = 0`` (always
applicable). Per-child thresholds must form a strictly monotonic increasing
sequence.

The standalone builder requires the user to supply ``min_pixel_size`` on
each child. The convenience-API path (``Scene.add_gsplats_from_data(
... lod_group=...)``) auto-derives thresholds using a √(N_finer / N_coarsest)
heuristic anchored at ``BASE_PIXEL_SIZE``.

This module also hosts the resolution helpers used by the convenience API
to interpret ``lod_group=`` / ``additive_lod=`` kwargs against a
``GSplatData`` input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from arbol import aprint

from .group import Group

if TYPE_CHECKING:
    from ..gsplats.gsplat_data import GSplatData
    from .node import Node
    from ..io.writer import ZarrWriterProtocol


#: Sentinel-typed alias for the value vocabulary of ``lod_group=`` and
#: ``additive_lod=``. Order: ``None`` (pass-through), ``True`` (require
#: stored), ``False`` (collapse to finest), ``dict[str, Any]`` (compute,
#: optionally with ``recompute=True``).
LODAxisSpec = Union[None, bool, dict]


#: Base value for auto-derived ``min_pixel_size`` thresholds. Child *i* gets
#: ``BASE_PIXEL_SIZE * sqrt(n_splats[i] / n_splats[0])`` (coarsest splats =
#: child 0). Splats can resolve roughly √N effective screen pixels of detail,
#: so the threshold scales linearly with that. 10 px is a reasonable
#: detail-floor below which a finer level isn't worth the cost.
BASE_PIXEL_SIZE: float = 10.0


def _assert_strict_ascending(thresholds: List[float], source: str) -> None:
    """Validate ``thresholds`` is strictly monotonic increasing (coarsest→finest).

    Shared between :func:`resolve_substitutive_axis` (explicit
    ``min_pixel_sizes=`` path) and :func:`derive_min_pixel_sizes` so both
    paths apply the same invariant — and so explicit lists fail at the
    resolver instead of deferring to a later :meth:`LODGroup.validate` call
    that the user may never make.
    """
    prev = float("-inf")
    for i, v in enumerate(thresholds):
        if v <= prev:
            raise ValueError(
                f"{source}: min_pixel_sizes must be strictly increasing in "
                f"coarsest→finest order; entry {i}={v} is not greater than "
                f"previous {prev}"
            )
        prev = v


class LODGroup(Group):
    """A scene-graph node that picks one of N children at runtime.

    Children are alternative representations of the same content, typically
    at different fidelities. Inherits all data-adding methods from
    :class:`Group` (``add_points``, ``add_lines``, ``add_gsplats``,
    ``add_gsplats_from_data``, ``add_group``); each child must additionally
    carry a ``min_pixel_size`` attribute on its ``.zattrs``.

    Example::

        lod = scene.add_lod_group("multires")
        lod.add_gsplats_from_data("coarse", coarse_data, min_pixel_size=0)
        lod.add_gsplats_from_data("medium", medium_data, min_pixel_size=100)
        lod.add_gsplats_from_data("fine", fine_data, min_pixel_size=500)

    Args:
        name: Name of the lod_group node.
        parent: Parent node in the scene hierarchy.
        writer: Writer interface for progressive writing.
        selector: Selector mode. Currently only ``"pixel_size"`` is supported
            (reserved for future modes like distance- or coverage-based).
        default_level: Initial active level index for the manual-override UI.
        **attrs: Additional node attributes.
    """

    def __init__(
        self,
        name: str,
        parent: Optional[Node] = None,
        writer: Optional[ZarrWriterProtocol] = None,
        *,
        selector: str = "pixel_size",
        default_level: int = 0,
        **attrs: Any,
    ) -> None:
        if selector != "pixel_size":
            raise ValueError(
                f"selector must be 'pixel_size' (other modes reserved for "
                f"future use), got {selector!r}"
            )
        if default_level < 0:
            raise ValueError(f"default_level must be >= 0, got {default_level}")

        attrs["type"] = "lod_group"
        attrs["selector"] = selector
        attrs["default_level"] = int(default_level)
        super().__init__(name, parent=parent, writer=writer, **attrs)

    @property
    def selector(self) -> str:
        """Selector mode (currently always ``"pixel_size"``)."""
        return str(self.attrs.get("selector", "pixel_size"))

    @property
    def default_level(self) -> int:
        """Initial active level index for the manual-override UI."""
        return int(self.attrs.get("default_level", 0))

    def child_min_pixel_sizes(self) -> list[float]:
        """Return the ``min_pixel_size`` of each child, in insertion order.

        Used by validation and by the convenience API to derive thresholds
        for newly-appended children.
        """
        return [float(c.attrs.get("min_pixel_size", 0.0)) for c in self.children]

    def validate(self) -> None:
        """Check that children form a valid lod_group.

        Raises ``ValueError`` if:

        - the group has zero children;
        - ``default_level`` is out of range
          (``not 0 <= default_level < len(children)``);
        - any child is missing ``min_pixel_size`` in its attrs;
        - the per-child ``min_pixel_size`` values are not strictly monotonic
          increasing in insertion order.

        Call this manually before finalizing if you want eager validation;
        otherwise the viewer falls back to silently ignoring malformed
        children at load time.
        """
        if not self.children:
            raise ValueError(f"LODGroup '{self.path or self.name}' has no children")
        n_children = len(self.children)
        default_level = self.default_level
        if not 0 <= default_level < n_children:
            raise ValueError(
                f"LODGroup '{self.path or self.name}' has "
                f"default_level={default_level}, must be in [0, {n_children})"
            )
        prev = float("-inf")
        for i, child in enumerate(self.children):
            if "min_pixel_size" not in child.attrs:
                raise ValueError(
                    f"LODGroup child {i} ({child.name!r}) is missing "
                    "'min_pixel_size' in its attrs"
                )
            value = float(child.attrs["min_pixel_size"])
            if value <= prev:
                raise ValueError(
                    f"LODGroup child {i} ({child.name!r}) has "
                    f"min_pixel_size={value}, must be strictly greater than "
                    f"previous child's {prev}"
                )
            prev = value


def resolve_substitutive_axis(
    data: "GSplatData",
    spec: LODAxisSpec,
) -> Tuple["GSplatData", Optional[List[float]]]:
    """Apply ``lod_group=`` semantics to ``data``.

    Returns ``(resolved_data, explicit_min_pixel_sizes_or_None)``. The
    second element is non-None only when the user passed
    ``dict(min_pixel_sizes=[...])`` — otherwise downstream code
    auto-derives from per-level splat counts via
    :func:`derive_min_pixel_sizes`.

    Semantics:

    +---------------------------+-----------------------------------------+
    | Spec                      | Behaviour                               |
    +===========================+=========================================+
    | ``None`` (default)        | Pass-through; if ``n_substitutive > 1``,|
    |                           | drop to the default substitutive level. |
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

    if spec is None:
        if data.n_substitutive > 1:
            return data.at_substitutive(data.default_substitutive), None
        return data, None

    if spec is True:
        if data.n_substitutive <= 1:
            raise ValueError(
                "lod_group=True requires the input GSplatData to already "
                "carry substitutive levels (n_substitutive > 1); got "
                f"n_substitutive={data.n_substitutive}. Use "
                "lod_group=dict(...) to compute them on the fly."
            )
        return data, None

    if spec is False:
        if data.n_substitutive > 1:
            # Convention: finest substitutive level is index 0.
            return data.at_substitutive(0), None
        return data, None

    if isinstance(spec, dict):
        kwargs = dict(spec)  # copy — don't mutate caller's dict
        recompute = bool(kwargs.pop("recompute", False))
        explicit_min_pixel_sizes = kwargs.pop("min_pixel_sizes", None)
        if explicit_min_pixel_sizes is not None:
            explicit_min_pixel_sizes = [float(v) for v in explicit_min_pixel_sizes]
            # Fail fast on bad explicit lists at the resolver instead of
            # deferring to a later LODGroup.validate() the user may never
            # call. Same invariant the auto-derivation enforces.
            _assert_strict_ascending(
                explicit_min_pixel_sizes, "lod_group=dict(min_pixel_sizes=...)"
            )

        if data.n_substitutive > 1 and not recompute:
            # Stored pyramid takes precedence — but silently accepting
            # compute-affecting kwargs would mask user intent. Reject any
            # leftover keys (only ``min_pixel_sizes`` / ``recompute`` are
            # honored on the stored path).
            if kwargs:
                raise ValueError(
                    "lod_group=dict(...) carries compute kwargs "
                    f"({sorted(kwargs)}) but data already has "
                    f"n_substitutive={data.n_substitutive}. Pass "
                    "recompute=True to override the stored pyramid, or "
                    "drop the compute kwargs to reuse it."
                )
            return data, explicit_min_pixel_sizes

        # Compute. Align with ``make_substitutive_lod``'s canonical default
        # (3 levels) so ``lod_group=dict()`` yields the same pyramid as a
        # bare ``make_substitutive_lod(data)`` call.
        kwargs.setdefault("levels", 3)
        from ..gsplats.lod.substitutive import make_substitutive_lod

        new_data = make_substitutive_lod(data, **kwargs)
        return new_data, explicit_min_pixel_sizes

    raise TypeError(
        f"lod_group must be None, bool, or dict; got {type(spec).__name__}"
    )


def resolve_additive_axis(
    data: "GSplatData",
    spec: LODAxisSpec,
) -> "GSplatData":
    """Apply ``additive_lod=`` semantics uniformly across substitutive levels.

    Semantics mirror :func:`resolve_substitutive_axis` but operate per
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
    from ..gsplats.gsplat_data import GSplatData, SubstitutiveLevel

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
            single = GSplatData(
                additive_sublods=list(lvl.additive_sublods)
            ).flattened().additive_sublods[0]
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
        from ..gsplats.lod.additive import make_additive_lod

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


def derive_min_pixel_sizes(splat_counts: list[int]) -> list[float]:
    """Auto-derive monotonic ``min_pixel_size`` thresholds from splat counts.

    Coarsest child (index 0) gets ``0.0``; each subsequent child *i* gets
    ``BASE_PIXEL_SIZE * sqrt(n_splats[i] / n_splats[0])``. Splats can
    resolve ~√N effective screen pixels of detail, so the threshold scales
    linearly with that. ``BASE_PIXEL_SIZE`` (~10 px) is the detail floor
    below which a finer level isn't worth the cost.

    Args:
        splat_counts: One entry per child, in coarsest→finest order. Must
            be non-empty and the first entry must be > 0.

    Returns:
        List of thresholds, same length as ``splat_counts``. Strictly
        monotonic increasing for monotonic non-decreasing input.
    """
    if not splat_counts:
        raise ValueError("splat_counts must be non-empty")
    n0 = splat_counts[0]
    if n0 <= 0:
        raise ValueError(f"coarsest child must have at least 1 splat, got {n0}")
    thresholds: list[float] = [0.0]
    for n in splat_counts[1:]:
        thresholds.append(BASE_PIXEL_SIZE * (n / n0) ** 0.5)
    # Defensive: guarantee strict monotonicity even when later children have
    # the same n_splats as the coarsest (degenerate, but the auto-derivation
    # should not produce non-monotonic thresholds).
    for i in range(1, len(thresholds)):
        if thresholds[i] <= thresholds[i - 1]:
            thresholds[i] = thresholds[i - 1] + 1.0
    # Belt-and-braces: same invariant the explicit-list path is checked
    # against in ``resolve_substitutive_axis``. Free now that the loop
    # above runs.
    _assert_strict_ascending(thresholds, "derive_min_pixel_sizes")
    return thresholds
