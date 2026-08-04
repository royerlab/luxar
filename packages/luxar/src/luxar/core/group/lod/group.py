"""luxar.core.group.lod.group – Geometry-agnostic helpers for the LOD-kind Group.

The LOD-kind ``Group`` selects one of N alternative children at runtime based
on a view-driven metric (currently: the projected bbox diagonal in pixels). It
is geometry-agnostic — children can be ``points``, ``lines``, ``gsplats``, or
themselves a specialized group (``kind=lod`` / ``kind=partition``).

Each child carries its own ``coverage_fraction`` attribute (strictly monotonic
increasing in coarsest→finest order; coarsest = 0.0, finest = 1.0). This is a
**dimensionless, viewport-relative** threshold: the viewer multiplies it by the
current viewport diagonal (in pixels) and picks the finest child whose resulting
pixel threshold is satisfied by the group's on-screen size. So the finest level
(coverage 1.0) activates when the object fills the screen, and coarser levels step
in geometrically as it shrinks — on any monitor.

The standalone builder ``add_lod_group()`` lets users assemble these by hand; the
convenience paths (e.g. ``Scene.add_gsplats_from_data(..., lod_group=...)``)
auto-derive them via ``coverage_fractions``: ``coverage_i = sqrt(N_i / N_finest)``
where ``N_i`` is level *i*'s splat count. Because it is a **ratio** of counts, it
is immune to non-displayed-dimension multiplicity (e.g. a stacked time axis inflates
every level's count equally and cancels), and it makes no absolute-resolution claim
(a coarse/blocky level is simply mapped onto a smaller apparent size, not a true
detail estimate).

Everything here is type-agnostic and shared across all leaf geometries
(Points, Lines, GSplats) and the Partition kind. The geometry-specific
``lod_group=`` / ``additive_lod=`` axis resolvers live next to their data
types (e.g. ``lod.gsplats`` for ``GSplatData``).

This module hosts:

* ``coverage_fractions`` (``sqrt(N_i / N_finest)``, coarsest 0.0) and its shared
  ``_apply_monotonicity_guard``.
* The free-function validator ``validate_lod_group``, callable on any
  ``Group`` whose ``attrs["kind"] == "lod"``.
* The shared ``resolve_display_type`` helper used by both LOD and Partition
  kinds, which walks down through nested specialized groups to determine what
  geometry type the user sees this layer as, and ``compute_lod_display_type``
  which derives an LOD group's ``display_type`` from its finest child.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from arbol import aprint

from ....validation.types import validate_truncation_radius

if TYPE_CHECKING:
    from ...node import Node


# ────────────────────────────────────────────────────────────────────────
# Display-type resolution (shared with the Partition kind)
# ────────────────────────────────────────────────────────────────────────


def resolve_display_type(node: "Node") -> str:
    """Return the geometry type this node would appear as to the user.

    For plain leaves and plain groups, this is the node's own ``type`` attr
    (``"points"`` / ``"lines"`` / ``"gsplats"`` / ``"group"``). For specialized
    groups (``kind == "lod"`` or ``kind == "partition"``), this is the
    ``display_type`` attr the writer recorded on them — which is itself
    derived transitively when one specialized group wraps another.

    Used by:
        * ``validate_partition_group`` — to compare homogeneity across children
          even when some children are themselves specialized groups.
        * ``compute_lod_display_type`` — to walk a finest-child chain
          through nested LOD/Partition groups down to a real geometry leaf.
        * The compiler at finalize, to compute ``display_type`` for a
          freshly-assembled specialized group.
    """
    kind = node.attrs.get("kind")
    if kind in ("lod", "partition"):
        display = node.attrs.get("display_type")
        if isinstance(display, str):
            return display
    return str(node.attrs.get("type", "group"))


def compute_lod_display_type(children: List["Node"]) -> str:
    """Derive an LOD group's ``display_type`` from its finest child.

    Convention: children are stored in coarsest→finest order, so the
    finest is the last entry. If that child is itself a kind=lod / kind=partition
    group, recurse through its own ``display_type``.
    """
    if not children:
        raise ValueError(
            "compute_lod_display_type: cannot derive display_type from an "
            "empty children list"
        )
    return resolve_display_type(children[-1])


# ────────────────────────────────────────────────────────────────────────
# coverage_fraction monotonicity invariant
# ────────────────────────────────────────────────────────────────────────


def _assert_strict_ascending(thresholds: List[float], source: str) -> None:
    """Validate ``thresholds`` is strictly monotonic increasing (coarsest→finest).

    Shared between the geometry-specific ``lod_group=`` resolvers (explicit
    ``coverage_fractions=`` path) and :func:`coverage_fractions` so both
    paths apply the same invariant — and so explicit lists fail at the
    resolver instead of deferring to a later :func:`validate_lod_group` call
    that the user may never make.
    """
    prev = float("-inf")
    for i, v in enumerate(thresholds):
        if v <= prev:
            raise ValueError(
                f"{source}: coverage_fractions must be strictly increasing in "
                f"coarsest→finest order; entry {i}={v} is not greater than "
                f"previous {prev}"
            )
        prev = v


def coverage_fractions(element_counts: list[int]) -> list[float]:
    """Auto-derive monotonic viewport-relative ``coverage_fraction`` thresholds.

    Coarsest child (index 0) gets ``0.0`` (always-eligible floor); each subsequent
    child *i* gets ``sqrt(N_i / N_finest)`` where ``N_finest`` is the finest (last)
    level's splat count — so the finest child is ``1.0`` and activates when the
    object fills the screen (the viewer multiplies each fraction by the current
    viewport diagonal in pixels). Coarser levels land at geometric fractions below.

    Because the fraction is a **ratio** of splat counts:

    * any non-displayed-dimension multiplicity (e.g. a stacked time axis that
      inflates every level's count by the same factor) **cancels**;
    * no absolute on-screen size or physical element radius is assumed — a
      coarse/blocky level is simply mapped onto a smaller apparent size, not a
      claim about true resolvable detail.

    ``sqrt`` is the screen-area law (a level with 4× the splats is ~2× denser
    linearly), giving ~2×-per-level spacing for the usual K=4 substitutive ladder.

    Args:
        element_counts: One entry per child, in coarsest→finest order. Must be
            non-empty and the finest (last) entry must be > 0.

    Returns:
        List of coverage fractions in ``[0, 1]``, same length as ``element_counts``,
        strictly monotonic increasing with the finest anchored at ``1.0``; the
        monotonicity guard resolves any equal/non-monotone entries by nudging the
        earlier (coarser) entries *downward*, so the ``[0, 1]`` contract holds
        for degenerate ladders too (the derived values always round-trip through
        the explicit-``coverage_fractions=`` validators).
    """
    if not element_counts:
        raise ValueError("element_counts must be non-empty")
    n_finest = element_counts[-1]
    if n_finest <= 0:
        raise ValueError(
            "finest LOD level is empty (0 elements), so per-level coverage "
            "fractions cannot be derived (they normalise by the finest count). "
            "This usually means a reduction culled every representative — e.g. the "
            f"input splats all have non-positive amplitude. Got "
            f"element_counts={list(element_counts)}."
        )
    fractions: list[float] = [0.0]
    for n in element_counts[1:]:
        fractions.append((n / n_finest) ** 0.5)
    return _apply_monotonicity_guard(fractions, "coverage_fractions")


def _apply_monotonicity_guard(thresholds: list[float], source: str) -> list[float]:
    """Enforce strict ascending thresholds WITHIN ``[0, 1]``, then assert.

    Defensive: guarantee strict monotonicity even when later children have the
    same (or fewer) elements than a previous level — WITHOUT ever breaching the
    viewport-relative contract (all values in ``[0, 1]``, coarsest ``0.0``,
    finest ``1.0``). An upward bump would push equal-tail ladders above 1.0
    (e.g. counts ``[10, 1000, 1000]``), producing values the explicit-input
    validators (``coverage_fractions=[...]``) reject if a user feeds the
    derived list back in. So duplicates resolve by nudging the *earlier*
    (coarser) entries DOWNWARD instead.

    Strategy (total — never raises on derived input):

    1. Cap the finest (last) entry at the ``1.0`` anchor (only a non-monotone
       count ladder can derive above it; a monotone one lands exactly on 1.0).
    2. Backward pass: any earlier entry not strictly below its successor is
       nudged down to ``successor / 1.1`` — a *relative* nudge, so near-equal
       levels separate proportionally to their scale, always toward 0 and
       never above 1.0.
    3. The downward nudge bottoms out at ``0`` for a zero-valued successor (a
       zero-count intermediate level derives to ``0``), collapsing degenerate
       entries into a zero prefix after index 0; a final pass lifts that
       prefix onto the same geometric ramp strictly between the ``0.0``
       coarsest floor and the first positive threshold.

    The trailing ``_assert_strict_ascending`` is the same invariant the
    explicit-``coverage_fractions`` path is checked against.
    """
    n = len(thresholds)
    if n >= 2:
        thresholds[-1] = min(thresholds[-1], 1.0)
        for i in range(n - 2, 0, -1):
            if thresholds[i] >= thresholds[i + 1]:
                thresholds[i] = thresholds[i + 1] / _MONOTONIC_NUDGE
        # Lift any zero prefix (index 0 stays the 0.0 coarsest floor).
        first_pos = next((i for i in range(1, n) if thresholds[i] > 0.0), None)
        if first_pos is not None:
            for i in range(1, first_pos):
                thresholds[i] = thresholds[first_pos] / _MONOTONIC_NUDGE ** (
                    first_pos - i
                )
    _assert_strict_ascending(thresholds, source)
    return thresholds


#: Relative separation factor used by :func:`_apply_monotonicity_guard`:
#: a colliding coarser entry is nudged DOWN to ``successor / 1.1``, keeping the
#: separation proportional to the threshold's scale while never leaving [0, 1].
_MONOTONIC_NUDGE: float = 1.1


# ────────────────────────────────────────────────────────────────────────
# Validator
# ────────────────────────────────────────────────────────────────────────


def validate_lod_group(group: "Node") -> None:
    """Check that a kind=lod ``Group`` is well-formed.

    Raises ``ValueError`` if:

    - the group has zero children;
    - ``default_level`` is out of range (``not 0 <= default_level <
      len(children)``);
    - any child is missing ``coverage_fraction`` in its attrs;
    - the per-child ``coverage_fraction`` values are not strictly monotonic
      increasing in insertion order.

    Call this manually before finalizing if you want eager validation;
    otherwise the viewer falls back to silently ignoring malformed
    children at load time.
    """
    if not group.children:
        raise ValueError(f"LOD group '{group.path or group.name}' has no children")
    n_children = len(group.children)
    default_level = int(group.attrs.get("default_level", 0))
    if not 0 <= default_level < n_children:
        raise ValueError(
            f"LOD group '{group.path or group.name}' has "
            f"default_level={default_level}, must be in [0, {n_children})"
        )
    prev = float("-inf")
    for i, child in enumerate(group.children):
        if "coverage_fraction" not in child.attrs:
            raise ValueError(
                f"LOD-group child {i} ({child.name!r}) is missing "
                "'coverage_fraction' in its attrs"
            )
        value = float(child.attrs["coverage_fraction"])
        if value <= prev:
            raise ValueError(
                f"LOD-group child {i} ({child.name!r}) has "
                f"coverage_fraction={value}, must be strictly greater than "
                f"previous child's {prev}"
            )
        prev = value


# ─────────────────────────────────────────────────────────────────────
# Shared substitutive-LOD axis resolver (Points + Lines)
# ─────────────────────────────────────────────────────────────────────
#
# Both geometries coarsen by lifting elements to gsplats and running the gsplat
# substitutive pipeline, so the ``substitutive_lod=`` kwarg vocabulary is
# identical. Keep ONE implementation here so the per-geometry resolvers
# (``resolve_substitutive_axis_points`` / ``resolve_substitutive_axis_lines``)
# can never drift.

#: Defaults for ``substitutive_lod=True`` / ``substitutive_lod=dict()`` — mirror
#: the gsplat substitutive defaults (compression_factor=4, levels=3, auto method).
DEFAULT_SUBSTITUTIVE_K: int = 4
DEFAULT_SUBSTITUTIVE_LEVELS: int = 3
DEFAULT_SUBSTITUTIVE_METHOD: str = "auto"
#: Accepted substitutive reduction methods (passed to make_substitutive_lod).
SUBSTITUTIVE_METHODS = frozenset(
    {"auto", "kmeans", "kmeans_lloyd", "greedy", "greedy_lloyd"}
)


def _validate_coarsen_dims_spec(value: Any) -> Any:
    """Shape/type-validate the raw ``coarsen_dims`` spec value (no scene yet).

    Accepts ``None``, the sentinels ``"display"`` / ``"all"``, or a non-empty
    list/tuple of dim names (str) and/or column indices (int). Names and the
    ``"display"`` default are resolved against the scene later by
    :func:`resolve_coarsen_dims`.
    """
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("display", "displayed"):
            return "display"
        if v in ("all", "*"):
            return "all"
        raise ValueError(
            f"coarsen_dims string must be 'display' or 'all'; got {value!r}"
        )
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("coarsen_dims must be non-empty")
        out: list[Any] = []
        for x in value:
            if isinstance(x, bool):
                raise ValueError("coarsen_dims entries must be int or str, not bool")
            if isinstance(x, int):
                out.append(int(x))
            elif isinstance(x, str):
                out.append(x)
            else:
                raise ValueError(
                    "coarsen_dims entries must be int (column index) or str "
                    f"(dimension name); got {type(x).__name__}"
                )
        return out
    raise TypeError(
        "coarsen_dims must be None, 'display'/'all', or a list of dim "
        f"names/indices; got {type(value).__name__}"
    )


def resolve_coarsen_dims(scene: Any, n_cols: int, raw: Any) -> Optional[tuple]:
    """Resolve a raw ``coarsen_dims`` spec into concrete center-column indices.

    ``scene`` supplies the dimension metadata; ``n_cols`` is the lifted gsplat
    dimensionality (== the position columns being coarsened). Returns a sorted
    tuple of allowed-coarsen column indices, or ``None`` meaning "coarsen over
    all dims" (no barrier — the historical behavior).

    Default (``raw is None``) is **Auto**: coarsen over the scene's *displayed*
    dims and group by the *non-displayed* dims. This only applies when the
    positions are aligned with the scene (``n_cols == scene.ndim``); otherwise
    (dim_order / extend_to_all reshaped the columns) Auto safely falls back to
    all-dims. ``"display"`` is the explicit form of Auto and errors if unaligned;
    ``"all"`` forces all-dims; a list resolves names via ``Dimensions.get_index``
    and ints as direct column indices.
    """
    dims = getattr(scene, "_dimensions", None) if scene is not None else None
    aligned = dims is not None and int(dims.ndim) == int(n_cols)

    def _finalize(idxs: Any) -> Optional[tuple]:
        norm = sorted({int(i) for i in idxs})
        if not norm:
            raise ValueError("coarsen_dims must be non-empty")
        for i in norm:
            if i < 0 or i >= n_cols:
                raise ValueError(
                    f"coarsen_dims index {i} out of range for {n_cols} dims"
                )
        return None if len(norm) == n_cols else tuple(norm)

    if raw == "all":
        return None
    if raw is None or raw == "display":
        if not aligned:
            if raw == "display":
                raise ValueError(
                    "coarsen_dims='display' requires positions aligned with the "
                    f"scene dims (got {n_cols} columns, scene ndim "
                    f"{getattr(dims, 'ndim', '?')}). Pass explicit indices."
                )
            return None  # Auto, unaligned -> safe all-dims fallback
        assert dims is not None  # implied by `aligned`
        displayed = [d for d in dims.displayed if 0 <= d < n_cols]
        non_displayed = [d for d in range(n_cols) if d not in set(displayed)]
        if not non_displayed:
            return None  # nothing to group by -> coarsen everything (no-op barrier)
        return _finalize(displayed)
    # Explicit list of names / indices.
    idxs: list[int] = []
    for x in raw:
        if isinstance(x, str):
            if not aligned:
                # Names resolve to scene-dim indices, which only equal center
                # columns when positions span the scene 1:1. Refuse rather than
                # silently mapping a name to the wrong column.
                raise ValueError(
                    "coarsen_dims by name requires positions aligned with the "
                    f"scene dims (got {n_cols} columns, scene ndim "
                    f"{getattr(dims, 'ndim', '?')}). Pass explicit column indices."
                )
            idxs.append(int(dims.get_index(x)))  # type: ignore[union-attr]
        else:
            idxs.append(int(x))
    return _finalize(idxs)


def resolve_substitutive_axis(spec: Any, geometry: str) -> Optional[Dict[str, Any]]:
    """Normalize the ``substitutive_lod=`` kwarg into a spec dict (or ``None``).

    Geometry-agnostic — ``geometry`` ("Points"/"Lines") only flavours the error
    message. Vocabulary:

    * ``None`` / ``False`` → no-op (caller writes a flat / additive node).
    * ``True`` / ``dict()`` → defaults (K=4, levels=3, method="auto").
    * ``dict(...)`` → keys ``compression_factor`` (alias ``K``), ``levels``
      (alias ``n_lods``), ``method`` (reduction algorithm), ``truncation_radius``,
      ``device``, ``seed``, ``coverage_fractions`` (explicit per-level
      viewport-relative thresholds, strict-ascending in [0, 1]), ``coarsen_dims``,
      ``max_aspect`` (per-splat anisotropy cap on the coarse levels, default 3.0;
      ``None`` disables — see :func:`luxar.gsplats.lift._cap_aspect`).
      Unrecognized keys raise. LOD switch thresholds are otherwise auto-derived as
      ``coverage_fractions`` (``sqrt(N_i/N_finest)``) — no method selector or
      per-dataset anchor knob.
    """
    if spec is None or spec is False:
        return None
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        raise TypeError(
            f"substitutive_lod must be None, bool, or dict; got {type(spec).__name__}"
        )
    kwargs = dict(spec)

    compression_factor = int(
        kwargs.pop("compression_factor", kwargs.pop("K", DEFAULT_SUBSTITUTIVE_K))
    )
    if compression_factor < 2:
        raise ValueError(f"compression_factor must be >= 2, got {compression_factor}")

    levels = int(
        kwargs.pop("levels", kwargs.pop("n_lods", DEFAULT_SUBSTITUTIVE_LEVELS))
    )
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")

    method = str(kwargs.pop("method", DEFAULT_SUBSTITUTIVE_METHOD)).replace("-", "_")
    if method not in SUBSTITUTIVE_METHODS:
        raise ValueError(
            f"method must be one of {sorted(SUBSTITUTIVE_METHODS)}; got {method!r}"
        )

    if "truncation_radius" in kwargs:
        truncation_radius = float(kwargs.pop("truncation_radius"))
    else:
        # NOT the codebase-wide DEFAULT_TRUNCATION_RADIUS (2.75): this value
        # feeds lift_points_to_gsplats / lift_lines_to_gsplats, whose T is a
        # profile-matching parameter calibrated at 3.0 — see
        # LIFT_TRUNCATION_RADIUS in luxar.gsplats.lift for the derivation.
        # Lazy import, mirroring the adders (keeps luxar.gsplats out of the
        # core import graph).
        from ....gsplats.lift import LIFT_TRUNCATION_RADIUS

        truncation_radius = LIFT_TRUNCATION_RADIUS
    validate_truncation_radius(truncation_radius)

    device = kwargs.pop("device", "auto")
    seed = kwargs.pop("seed", None)
    if seed is not None:
        seed = int(seed)

    explicit_coverage = kwargs.pop("coverage_fractions", None)
    if explicit_coverage is not None:
        explicit_coverage = [float(m) for m in explicit_coverage]
        if not explicit_coverage:
            raise ValueError(
                "substitutive_lod=dict(coverage_fractions=...) must be non-empty "
                "(one strictly-ascending value in [0, 1] per LOD level)"
            )
        _assert_strict_ascending(
            explicit_coverage, "substitutive_lod=dict(coverage_fractions=...)"
        )
        if explicit_coverage[0] < 0.0 or explicit_coverage[-1] > 1.0:
            raise ValueError(
                "substitutive_lod=dict(coverage_fractions=...): values must lie in "
                f"[0, 1] (coarsest→finest); got {explicit_coverage}"
            )

    # Dims coarsening may cluster over; complement = hard grouping barriers.
    # Shape/type only here (scene dims aren't known yet); names + the "display"
    # default are resolved in the scene-aware adder via resolve_coarsen_dims().
    coarsen_dims = _validate_coarsen_dims_spec(kwargs.pop("coarsen_dims", None))

    # Per-splat anisotropy cap on the coarse levels (None disables). The merge
    # elongates lifted isotropic beads level over level; the cap bounds the
    # view-dependent ray-integral flare at max_aspect (mass-preserving).
    max_aspect = kwargs.pop("max_aspect", 3.0)
    if max_aspect is not None:
        max_aspect = float(max_aspect)
        if max_aspect < 1.0:
            raise ValueError(
                f"max_aspect must be >= 1 (or None to disable), got {max_aspect}"
            )

    if kwargs:
        raise ValueError(
            f"substitutive_lod for {geometry}: unrecognized keys {sorted(kwargs)}. "
            "Valid keys: compression_factor (K), levels (n_lods), method, "
            "truncation_radius, device, seed, coverage_fractions, coarsen_dims, "
            "max_aspect."
        )

    return {
        "compression_factor": compression_factor,
        "levels": levels,
        "method": method,
        "truncation_radius": truncation_radius,
        "device": device,
        "seed": seed,
        "coverage_fractions": explicit_coverage,
        "coarsen_dims": coarsen_dims,
        "max_aspect": max_aspect,
    }


# ────────────────────────────────────────────────────────────────────────
# Additive-ladder quality stamps (Points + Lines)
# ────────────────────────────────────────────────────────────────────────
#
# The viewer's never-downgrade display gate can release a coarse→fine LOD swap
# as soon as the committed prefix carries enough of the level's energy, instead
# of waiting for the raw element count to pass the coarser sibling. That needs
# two numbers on disk, and it needs BOTH or it silently falls back to the count
# rule (``lod-display-gate.ts`` foldProgress poisons the whole subtree aggregate
# to null if either is missing on any visible leaf):
#
#   * ``lod_stats.energy_fraction_cum`` on each ``additive_<i>/`` subgroup — the
#     cumulative fraction of the leaf's energy carried by that prefix.
#   * ``level_stats.reference_energy`` on the leaf itself — the leaf's total
#     energy, used as a relative weight when several leaves fold together.
#
# GSplats have stamped these since the Q·e work (``gsplats/lod/additive.py``);
# these helpers give Points and Lines the same stamps in their own energy
# currency. Key names and the clamping/guard behaviour mirror the gsplat side
# exactly so the viewer needs no per-geometry branch.


def breakpoints_kind_of(counts: Any) -> str:
    """Name the breakpoint vocabulary that produced a ladder, for the stamps.

    Mirrors the ``kind`` string the gsplat ladder records, so a reader can tell
    a bandwidth-derived geometric ladder from an equal-count or energy split
    without re-deriving it.
    """
    if isinstance(counts, str):
        if counts.startswith("stream:"):
            return "stream"
        if counts.startswith("energy:"):
            return "energy-fractions"
        return counts
    if counts is None:
        return "equal-count"
    return "explicit-counts"


def additive_level_stats(
    level_energies: List[float],
    level_counts: List[int],
    *,
    method: str,
    breakpoints_kind: str,
    energy_kind: str,
) -> tuple[List[Dict[str, Any]], Optional[float], Dict[str, Any]]:
    """Build the per-sub-LOD and per-leaf stamps for an additive ladder.

    Args:
        level_energies: Per-level (not cumulative) energy sums, level order.
        level_counts: Per-level element counts, same order and length.
        method: The ordering method that produced the ladder.
        breakpoints_kind: From :func:`breakpoints_kind_of`.
        energy_kind: Provenance of the energy quantity, e.g.
            ``"points-luminance-volume"``. The viewer ignores it; it documents
            that this currency is not comparable with the gsplat one.

    Returns:
        ``(per_level_lod_stats, reference_energy, parent_level_stats)``.
        ``reference_energy`` is ``None`` — and no ``energy_fraction_cum`` is
        stamped — when the total energy is not positive and finite (all-black
        colors, zero radii). That is deliberate: an absent stamp makes the
        viewer fall back to its count rule, whereas a fabricated 0.0 would make
        it release swaps on data that carries no energy at all.
    """
    if len(level_energies) != len(level_counts):
        raise ValueError(
            f"level_energies has {len(level_energies)} entries but level_counts "
            f"has {len(level_counts)}; internal error"
        )

    total = float(sum(level_energies))
    usable = total > 0.0 and total == total and total != float("inf")

    per_level: List[Dict[str, Any]] = []
    cum_energy = 0.0
    cum_n = 0
    for i, (energy, count) in enumerate(zip(level_energies, level_counts)):
        cum_energy += float(energy)
        cum_n += int(count)
        stats: Dict[str, Any] = {
            "lod_method": method,
            "lod_level": i,
            "lod_breakpoints_kind": breakpoints_kind,
            "lod_n_elements": int(count),
            "lod_cumulative_n": cum_n,
        }
        if usable:
            frac = cum_energy / total
            if frac == frac:  # not NaN
                stats["energy_fraction_cum"] = min(1.0, max(0.0, frac))
        per_level.append(stats)

    parent: Dict[str, Any] = {
        "energy_kind": energy_kind,
        "lod_method": method,
        "lod_n_lods": len(level_counts),
        "lod_breakpoints_kind": breakpoints_kind,
    }
    if usable:
        parent["reference_energy"] = total

    return per_level, (total if usable else None), parent


# ────────────────────────────────────────────────────────────────────────
# Composed axes: an additive ladder INSIDE a substitutive level
# ────────────────────────────────────────────────────────────────────────
#
# The two coarsening axes answer different questions and compose cleanly:
# ``substitutive_lod`` chooses WHICH level renders at the current zoom, and
# ``additive_lod`` describes HOW each of those levels streams in. GSplats have
# always composed them (``gsplats/lod/pyramid.py`` ladders every substitutive
# level); Points and Lines used to reject the combination, which left the finest
# level of a substitutive ladder as the one node in the system that could not
# paint progressively — it committed all-or-nothing, however large it was.
#
# The helpers below are the shared plumbing for that composition. They take the
# geometry's own resolver as a callable, so there is no geometry branching here
# and the Points and Lines call sites cannot drift apart.

#: Approximate on-disk bytes per Points element / Lines vertex (AUTO-encoded
#: positions + colors + radius/width). The element-geometry counterpart of the
#: ~45 B/splat figure the gsplat streaming ladder is sized against.
DEFAULT_LADDER_BYTES_PER_ELEMENT: float = 16.0

#: Download-time budget for a composed ladder's first chunk. 200 ms is short
#: enough to read as "immediate" and long enough to carry a useful first paint.
DEFAULT_LADDER_TARGET_MS: float = 200.0


def default_composed_additive_lod() -> Dict[str, Any]:
    """The ladder a substitutive Points/Lines level gets when none is requested.

    A bandwidth-derived ``stream:`` ladder, NOT an equal-count one: an
    equal-count split into 4 still ends with an N/4-sized commit, which on a
    multi-million-element level is seconds of frozen main thread — exactly the
    pathology the composition exists to remove. ``stream:`` makes first paint
    cost one small chunk and doubles from there.

    ``method="random"`` because a random prefix of a cloud looks like the whole
    cloud at lower density at every k, which is the best possible partial paint.
    An energy ordering would front-load ``energy_fraction_cum`` (so the viewer's
    committed-energy gate releases sooner), but on the common constant-radius
    cloud it degenerates to pure luminance order — for a scalar-coloured UMAP
    that means the whole high-scalar region paints first, a spatially biased and
    visibly wrong first frame. Callers who want the earlier release opt in with
    ``additive_lod=dict(method="salience", salience_kind="energy", ...)``.
    """
    from ....utils.lod_breakpoints import DEFAULT_BANDWIDTH_MBPS, streaming_chunk_splats

    chunk = streaming_chunk_splats(
        DEFAULT_LADDER_TARGET_MS,
        DEFAULT_BANDWIDTH_MBPS,
        DEFAULT_LADDER_BYTES_PER_ELEMENT,
    )
    return {"method": "random", "counts": f"stream:{chunk}", "seed": 0}


def compose_additive_under_substitutive(
    additive_lod: Any,
    *,
    resolve: Callable[[Any], Optional[Dict[str, Any]]],
    name: str,
    suppress_reason: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve the additive spec to use for the levels of a substitutive ladder.

    Vocabulary (``None`` behaves differently here than on a plain leaf, which is
    the whole point — a substitutive level is by construction both the largest
    node in the scene and the last one loaded, so it should stream by default):

    =================  =============================  ==========================
    ``additive_lod=``  plain leaf (unchanged)         under ``substitutive_lod``
    =================  =============================  ==========================
    ``None``           no ladder                      **default stream ladder**
    ``False``          no ladder                      no ladder (the opt-out)
    ``True`` / ``{}``  the resolver's defaults        the resolver's defaults
    ``dict(...)``      the caller's ladder            the caller's ladder
    =================  =============================  ==========================

    Args:
        additive_lod: The user's kwarg value, verbatim.
        resolve: The geometry's ``resolve_additive_axis_*`` function. The result
            is normalized through it, so the returned dict is idempotent under
            re-resolution — which is what makes it safe to hand straight to the
            PUBLIC ``add_points`` / ``add_lines`` for the finest child.
        name: Node name, for messages.
        suppress_reason: When set, no ladder is built and the reason is
            reported. Used for the cases where laddering would lose data or be a
            no-op rather than a win. A *default* ladder (``additive_lod`` left as
            ``None``) is skipped quietly; an *explicitly* requested one
            (``True`` / a ``dict``) raises a ``UserWarning``, since the caller
            asked for something that cannot be honoured.

    Returns:
        A normalized spec dict, or ``None`` for "write flat levels".
    """
    if additive_lod is False:
        return None
    if suppress_reason is not None:
        if additive_lod is not None:
            warnings.warn(
                f"'{name}': the requested streaming ladder cannot be honoured "
                f"({suppress_reason}); levels will load all-at-once.",
                UserWarning,
                stacklevel=2,
            )
        else:
            aprint(
                f"  ℹ️  '{name}': streaming ladder skipped ({suppress_reason}); "
                "levels will load all-at-once."
            )
        return None
    spec = additive_lod if additive_lod is not None else default_composed_additive_lod()
    return resolve(spec)


def level_additive_lod(
    spec: Optional[Dict[str, Any]],
    *,
    level_n: int,
    compression_factor: int,
    is_coarsest: bool,
) -> Optional[Dict[str, Any]]:
    """Specialize a composed ladder spec for one level of the group.

    Applies the sibling-aware rule from ``gsplats/lod/pyramid.py``: every level
    that HAS a coarser sibling raises its first chunk to ``ceil(n / (2·K))``, so
    an upgrade's committed prefix passes that sibling within a chunk or two
    instead of only at the end of the ladder. The coarsest level is left alone —
    it is the eager default level, and its small first chunk is the
    fast-first-paint path.

    Levels smaller than their first chunk collapse to a single level upstream and
    the caller falls through to a flat leaf, so tiny coarse levels need no
    special-casing here.
    """
    if spec is None or level_n <= 0:
        return None
    out = dict(spec)
    if not is_coarsest:
        from ....utils.lod_breakpoints import sibling_aware_stream_breakpoints

        counts = out.get("counts")
        if isinstance(counts, str):
            out["counts"] = sibling_aware_stream_breakpoints(
                counts, level_n, compression_factor
            )
    return out


def gsplat_additive_lod_from(
    spec: Optional[Dict[str, Any]], level_n: int
) -> Optional[Dict[str, Any]]:
    """Translate an element-geometry ladder spec into the GSplats vocabulary.

    The only place the two additive vocabularies meet. Two deliberate choices:

    * ``method`` is NOT carried over. The Points/Lines methods name orderings in
      the element domain (``spatial-uniform`` over point positions); the coarse
      children of a substitutive ladder are merged Gaussian beads, where the
      bead-domain orderings apply.
    * ``self_energy``, not ``auto``. ``auto`` routes levels of <= 5000 splats to
      the submodular ``greedy``, whose sparse-Gram build is a pure-Python
      per-pair loop scaling with OVERLAP DENSITY — and coarse levels of a lifted
      cloud are maximally overlapping merged blobs, the worst case for it.
      ``self_energy`` is O(N log N), never builds a Gram, and is still
      energy-front-loaded.
    """
    if spec is None or level_n <= 0:
        return None
    from ....gsplats.lod.additive import clamp_counts_breakpoints

    counts = spec.get("counts")
    if counts is None:
        counts = "equal-count"
    elif isinstance(counts, str) and counts.startswith("energy:"):
        # The element-domain ``energy:<frac,...>`` spec has no counterpart in the
        # GSplat resolver's string vocabulary (only ``equal-count`` / ``stream:<c>``).
        # Translate it to the float-list form the resolver already understands as
        # cumulative energy fractions (``_resolve_breakpoints`` → energy-fractions),
        # resolved against the coarse child's own self-energy cumulative.
        fracs = [float(s) for s in counts[len("energy:") :].split(",") if s.strip()]
        # Mirror the element-domain parser (points.py::_energy_breakpoints_to_counts):
        # sort, drop f<=0, clamp f>=1 → 1.0, dedup — yielding a strictly-increasing
        # list in (0, 1]. The GSplat float resolver is strict and would otherwise
        # raise AFTER the wrapper kind=lod group was written, leaving a childless
        # partial group; a spec accepted on a plain Points/Lines leaf must never
        # abort the coarse GSplat child of the composed build.
        counts = sorted({min(f, 1.0) for f in fracs if f > 0.0})
        # All fractions non-positive (e.g. "energy:0", "energy:-1,0") degenerate
        # to a single full level on a plain leaf; match that here rather than
        # letting the strict resolver raise on an empty list.
        if not counts:
            counts = [1.0]
    breakpoints = clamp_counts_breakpoints(counts, level_n)
    out: Dict[str, Any] = {"method": "self_energy", "breakpoints": breakpoints}
    if spec.get("n_lods") is not None:
        out["n_lods"] = spec["n_lods"]
    return out
