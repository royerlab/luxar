"""luxar.core.group.lod.group – Geometry-agnostic helpers for the LOD-kind Group.

The LOD-kind ``Group`` selects one of N alternative children at runtime based
on a view-driven metric (currently: the projected bbox diagonal in pixels). It
is geometry-agnostic — children can be ``points``, ``lines``, ``gsplats``,
``mesh``, or themselves a specialized group (``kind=lod`` / ``kind=partition``).

Each child carries its own ``coverage_fraction`` attribute (strictly monotonic
increasing in coarsest→finest order; coarsest = 0.0, finest = 1.0 for a
WHOLE-OBJECT ladder and ``MAX_COVERAGE_FRACTION`` = 4.0 for a partition-bound
one). This is a **dimensionless, viewport-relative** threshold: the viewer
multiplies it by a fixed fraction (a quarter) of the current viewport diagonal (in
pixels) and picks the finest child whose resulting pixel threshold is satisfied by
the group's on-screen size. So coverage 1.0 activates once the object's projected
bbox diagonal reaches about a quarter of the viewport diagonal — i.e. at any normal
full-frame view — and coarser levels step in geometrically as it shrinks below
that, on any monitor. 4.0 means "once this node alone fills the viewport", which is
the right anchor for one tile of a spatial partition (see
``partitioned_coverage_fractions``).

The standalone builder ``add_lod_group()`` lets users assemble these by hand; the
convenience paths (e.g. ``Scene.add_gsplats_from_data(..., lod_group=...)``,
``add_points`` / ``add_lines`` / ``add_mesh`` ``substitutive_lod=``) auto-derive
them via
``derive_coverage_fractions``, which picks between ``coverage_fractions`` and
``partitioned_coverage_fractions`` from the insertion point's ancestry. Both share
one shape, ``coverage_i ∝ sqrt(N_i / N_finest)`` where ``N_i`` is level *i*'s
element count. Because it is a **ratio** of counts, it
is immune to non-displayed-dimension multiplicity (e.g. a stacked time axis inflates
every level's count equally and cancels), and it makes no absolute-resolution claim
(a coarse/blocky level is simply mapped onto a smaller apparent size, not a true
detail estimate).

Almost everything here is type-agnostic and shared across all leaf geometries
(Points, Lines, GSplats, Mesh) and the Partition kind; the one exception is the
mesh coarsening-method constants (``MESH_SUBSTITUTIVE_METHODS`` /
``DEFAULT_MESH_SUBSTITUTIVE_METHOD``), which live here beside the mixture-reducer
set they are deliberately disjoint from. The geometry-specific
``lod_group=`` / ``additive_lod=`` axis resolvers live next to their data
types (e.g. ``lod.gsplats`` for ``GSplatData``).

This module hosts:

* ``coverage_fractions`` (``sqrt(N_i / N_finest)``, coarsest 0.0) and its shared
  ``_apply_monotonicity_guard``, its partition-bound sibling
  ``partitioned_coverage_fractions``, and the ``derive_coverage_fractions`` /
  ``is_partition_bound`` pair the scene adders use to choose between them from
  the ladder's insertion point.
* The free-function validator ``validate_lod_group``, callable on any
  ``Group`` whose ``attrs["kind"] == "lod"``.
* The shared ``resolve_display_type`` helper used by both LOD and Partition
  kinds, which walks down through nested specialized groups to determine what
  geometry type the user sees this layer as, and ``compute_lod_display_type``
  which derives an LOD group's ``display_type`` from its finest child.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Final, List, Literal, Optional

from arbol import aprint

from ....typing_utils.geometry_capabilities import require_lod_display_type
from ....validation.types import validate_truncation_radius

if TYPE_CHECKING:
    from ...node import Node


#: Upper bound on any ``coverage_fraction`` — authored OR derived.
#:
#: This is ``1 / FILL_FACTOR`` (the viewer's LOD anchor, ``0.25`` — see
#: ``scene/lod-group-registry.ts``): the viewer compares each threshold against
#: ``coverage metric = projected bbox diagonal / (FILL_FACTOR × viewport
#: diagonal)``, so an object whose projected diagonal exactly equals the viewport
#: diagonal — a *screen-filling* object — produces a metric of ``1 / FILL_FACTOR``
#: = 4.0. The bound therefore means "a level may be required to fill the screen,
#: at most", which is precisely what the old bound of ``1.0`` meant back when
#: ``FILL_FACTOR`` was 1.0. Keeping the bound at 1.0 after the anchor moved would
#: silently shrink what an author can express.
#:
#: ``1.0`` is the finest anchor of a WHOLE-OBJECT ladder (reached at ~a quarter of
#: the viewport diagonal, i.e. any normal full-frame view) — see
#: :func:`coverage_fractions`, whose *derived* output contract stays ``[0, 1]``.
#: Values above ``1.0`` hold a level until *later* than that, which is what a
#: spatially tiled layer needs (each tile projects to only a fraction of the
#: viewport). They arrive three ways:
#:
#: 1. an explicit ``coverage_fractions=[...]`` list on ``substitutive_lod=`` /
#:    ``lod_group=`` (checked by the resolvers in this module);
#: 2. automatically, via :func:`partitioned_coverage_fractions` — what every
#:    partition-bound producer derives (the ``adaptive`` / ``overview`` gsplat
#:    recipes, the two gsplat writers' topology-aware fallback, and the scene
#:    adders through :func:`derive_coverage_fractions`);
#: 3. a hand-authored per-child ``coverage_fraction=`` on ``add_lod_group`` (the
#:    shape ``examples/partition_of_lod_example.py`` builds), for which
#:    :func:`validate_lod_group` is the only guard — it bypasses the resolvers
#:    entirely.
#:
#: So this bound is not an author-only escape hatch: it is also the finest value a
#: derived ladder can take.
MAX_COVERAGE_FRACTION: Final = 4.0


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

    Rejects a resolved display type that is a geometry type with **no LOD
    support**. Note what this does *not* do: it is not a homogeneity check and
    not an allowlist. A ``kind=lod`` group is intentionally heterogeneous-
    tolerant (a coarse points level under a fine gsplats level is a supported
    composition, and ``TestDisplayTypeResolution`` pins that), and
    ``display_type`` also legitimately carries non-geometry marker strings on
    nested groups — so both of those keep passing. Only a *known* geometry type
    whose capability row says ``lod=False`` is refused.

    Without this the failure is silent and deferred: ``resolve_display_type``
    falls through to ``attrs.get("type", "group")`` for a plain leaf, so a mesh
    child would be accepted here and produce a ``kind=lod`` group with
    ``display_type="mesh"`` that no viewer path can load — a broken store written
    without complaint. The partition sibling has always had its guard
    (``add_partition_group_impl``); the LOD path only ever *derived* the value.
    """
    if not children:
        raise ValueError(
            "compute_lod_display_type: cannot derive display_type from an "
            "empty children list"
        )
    display_type = resolve_display_type(children[-1])
    require_lod_display_type(display_type, f"finest child {children[-1].name!r}")
    return display_type


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

    Non-finite entries are rejected up front: ``NaN <= prev`` is false, so a NaN
    would pass this check AND become ``prev``, which then makes every later
    comparison false too — the whole tail of the ladder would go unvalidated.
    """
    prev = float("-inf")
    for i, v in enumerate(thresholds):
        if not math.isfinite(v):
            raise ValueError(
                f"{source}: coverage_fractions entries must be finite numbers; "
                f"entry {i}={v} is not"
            )
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
    level's splat count — so the finest child is ``1.0`` and activates once the
    object's projected bbox diagonal reaches about a quarter of the viewport
    diagonal, i.e. at any normal full-frame view (the viewer multiplies each
    fraction by a quarter of the current viewport diagonal in pixels). Coarser
    levels land at geometric fractions below and step in as the object shrinks.

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


def partitioned_coverage_fractions(element_counts: list[int]) -> list[float]:
    """:func:`coverage_fractions` re-anchored at **fills-screen** for a ladder
    that is bound to a spatial partition.

    **The rule.** The viewer's quarter-viewport anchor (``FILL_FACTOR`` 0.25) is
    calibrated for a lod group whose levels are alternative renderings of the
    WHOLE object, seen at a normal full-frame view — that is the #1361 fix. Two
    topologies opt out of it, for two DIFFERENT reasons — one geometric, one a
    product contract. Do not conflate them:

    * **A per-part ladder** (one ``kind=lod`` group per BSP tile — the
      ``adaptive`` recipe). This one is GEOMETRY. The switching group's bbox is a
      single TILE, so its projected diagonal is intrinsically a fraction of the
      whole object's and the metric reads systematically low; a whole-object
      anchor would make every tile select its finest level while the object is
      merely full-frame.
    * **An overview cap above a partition** (the ``overview`` recipe's
      ``[coarse_leaf, fine_partition]`` pair). This one is a CONTRACT, not
      geometry: both children cover the whole dataset, so this group's bbox IS
      the whole object and its metric reads exactly like a ``levels`` group's.
      It is pinned at the fills-screen anchor because the recipe promises
      "instant coarse overview level + fine tiles on zoom" — the coarse cap is
      what you see at the opening framing and the fine branch is the zoom-in
      branch. The consequence is deliberate and worth stating plainly: **#1361's
      blur is RETAINED by design for ``overview``.** Showing its fine branch at
      the opening framing would mean loading the entire dataset on frame 1, which
      is precisely the cost this recipe exists to avoid for huge N. Reach for
      ``levels`` instead when you want full detail immediately.

    **A one-part partition is neither of those.** ``to_spatial_partition`` wraps
    even a single BSP leaf in a ``kind=partition``, and that part's bbox IS the
    whole object's — so the geometric argument above does not hold and callers
    must use plain :func:`coverage_fractions` for it, or the finest level would be
    held back until the object overfills the screen (the very #1361 blur). Both
    tree writers' topology fallbacks and ``build_adaptive`` special-case it.

    In both cases the ladder keeps the pre-#1361 anchor by scaling the derived
    fractions by :data:`MAX_COVERAGE_FRACTION` (``1 / FILL_FACTOR``), so the
    finest lands on ``4.0`` — which in coverage-metric space means exactly what
    ``1.0`` meant before the anchor moved. This is the same ×4 rescale the
    hand-tuned ``demo_biodiversity_planetary_scale`` ladder uses, so all three
    are one rule rather than three special cases.

    Scaling is order-preserving and by a power of two (hence exact in binary
    floating point), so strict ascent survives, ``0.0`` stays ``0.0``, and the
    output lands in ``[0, MAX_COVERAGE_FRACTION]`` — the explicit-list bound —
    with the finest exactly on it.

    **The assumption.** The rule rests on the partition being a real TILING, i.e.
    >= 2 parts, so that a part genuinely projects to a fraction of the whole. A
    ONE-PART ``kind=partition`` breaks it: the "tile" IS the whole object, and this
    anchor is then a factor of 4 too coarse. The two entries below are the ones
    that have been audited — the tree-building producers that CAN see the sibling
    count (they exclude the shape), and the scene-adder path (which cannot). Read
    the list as illustrative, not exhaustive:

    * ``luxar gsplat lod --recipe adaptive`` (``gsplats/lod/recipes.py::
      build_adaptive``) holds the whole ``partition.children`` list before building
      any ladder, so it selects :func:`coverage_fractions` when there is a single
      part. Both tree writers do the same from their own recursion flag —
      ``io/_compiler/gsplat_tree.py::write_gsplat_node`` and
      ``core/group/gsplats_pipeline/from_io.py::graft_gsplat_node`` only mark
      children as partition-bound when ``len(node.children) > 1``. That matters
      because ``GSplatData.to_spatial_partition`` wraps even a single BSP leaf in a
      ``GSplatPartition``, so a dataset smaller than ``--max-elements`` (default
      ``DEFAULT_MAX_ELEMENTS`` = 1,000,000) reaches these paths as a one-part
      partition routinely — it is the common case, not an edge one.
    * The SCENE-ADDER path (:func:`derive_coverage_fractions`) genuinely cannot
      check it: part 0's ladder is derived before part 1 has been added, so the
      sibling count does not exist yet. This is a documented caveat, not a coded
      guard — the only place the ×4 anchor can be applied to a lone part. The
      compiler's finalize walk, which DOES see the final sibling count, warns
      about it after the fact (``io/_compiler/finalize/lod_backfill.py::
      warn_one_part_partition_anchors``); it does not rewrite anything, because an
      explicit ``coverage_fractions=`` list is indistinguishable from a derived
      one on disk.

    Two further producers reach this anchor for a lone part and are known,
    pre-existing, and out of scope here: ``GSplatData.partition_from_regions``
    (``gsplats/_data/composition.py``) returns the bare ``build_part_lod(...)``
    node — a 4.0-anchored ``kind=lod`` with no partition wrapper at all — when
    exactly one region is non-empty, and the tiled batch merge
    (``gsplats/batch/merge_orchestrator.py::_finalize_part_node``) hands the same
    ``build_part_lod`` node to the streaming writer for a single-tile run, which
    still emits a one-part ``kind=partition`` around it.

    Args:
        element_counts: One entry per child, in coarsest→finest order (same
            contract as :func:`coverage_fractions`).

    Returns:
        The derived ladder scaled by :data:`MAX_COVERAGE_FRACTION`: coarsest
        ``0.0``, finest ``MAX_COVERAGE_FRACTION``, strictly ascending.
    """
    return [f * MAX_COVERAGE_FRACTION for f in coverage_fractions(element_counts)]


def is_partition_bound(node: "Node") -> bool:
    """Is ``node`` inside a ``kind=partition`` group (itself, or any ancestor)?

    ``node`` is the **insertion point** — the node a scene adder is about to
    attach a ``kind=lod`` group *to* (i.e. the future lod group's parent). So the
    partition wrapper itself counts, and so does any partition further up: a
    plain ``add_group`` sitting between the partition and the ladder still leaves
    the ladder inside ONE tile, which is the only thing that matters here (a
    tile's projected bbox diagonal is intrinsically a fraction of the whole
    object's, so a whole-object anchor reads systematically low — see
    :func:`partitioned_coverage_fractions`).

    Caveat on that intermediate ``add_group``: this function accepts it, but
    :func:`~luxar.core.group.partition.validate_partition_group` would REJECT the
    same tree as non-homogeneous, because ``resolve_display_type`` returns
    ``"group"`` for a plain Group child. Nothing raises today (that validator is
    only invoked from its own tests, not by the compiler) and the viewer resolves
    such a tree fine, but the two modules disagree — whoever wires the validator
    into the write path must reconcile them (teach it to see through a plain group,
    or stop blessing the shape here).

    This is the scene-graph mirror of the ``under_partition`` /
    ``_under_partition`` recursion flag the two gsplat writers thread down their
    trees (``io/_compiler/gsplat_tree.py::write_gsplat_node`` and
    ``core/group/gsplats_pipeline/from_io.py::graft_gsplat_node``). Those walk a
    detached ``GSplatNode`` tree top-down and so can carry the flag; a scene adder
    is handed only its insertion point, so it walks ``parent`` links up instead.
    The two meet in ``graft_gsplat_node``, which SEEDS its recursion flag from this
    walk — once, at the entry call, where the insertion point is still a node the
    graft did not create. Deeper down it must keep using the threaded flag: this
    walk would then only rediscover the graft's own freshly-written
    ``kind=partition`` wrapper and would overrule that recursion's one-part
    exclusion (a single part is not a tiling — see
    :func:`partitioned_coverage_fractions`).
    """
    current: Optional["Node"] = node
    while current is not None:
        if current.attrs.get("kind") == "partition":
            return True
        current = current.parent
    return False


def derive_coverage_fractions(
    element_counts: list[int], insertion_point: "Node", *, name: str
) -> list[float]:
    """Pick the right anchor for an auto-derived ladder, from where it is going.

    The single chokepoint all FOUR scene adders (``add_points`` / ``add_lines`` /
    ``add_mesh`` ``substitutive_lod=``, ``add_gsplats_from_data`` ``lod_group=``)
    use when the caller did NOT pass an explicit ``coverage_fractions=`` list:

    * insertion point inside a ``kind=partition`` (see
      :func:`is_partition_bound`) → :func:`partitioned_coverage_fractions`, the
      fills-screen anchor, because this ladder switches on ONE TILE's projected
      size, which is intrinsically a fraction of the whole object's;
    * otherwise → :func:`coverage_fractions`, the whole-object anchor.

    The adders reject ``partition=`` together with ``substitutive_lod=``, so the
    partition-bound shape only ever arrives hand-built: a caller creates the
    ``kind=partition`` wrapper itself and calls the adder once per part (what
    ``demos/demo_biodiversity_planetary_scale.py`` does). That is worth one log
    line rather than a silent anchor switch, so the partition-bound case reports
    which anchor it chose.

    Args:
        element_counts: One entry per level, coarsest→finest (same contract as
            :func:`coverage_fractions`).
        insertion_point: The node the ``kind=lod`` group is being added to.
        name: The lod group's name, for the log line.

    Returns:
        The derived ladder, on whichever anchor the insertion point implies.
    """
    if not is_partition_bound(insertion_point):
        return coverage_fractions(element_counts)
    fractions = partitioned_coverage_fractions(element_counts)
    aprint(
        f"  🧩 Substitutive-LOD '{name}': kind=partition ancestor detected — "
        "anchoring this per-tile ladder at fills-screen (finest "
        f"coverage_fraction={fractions[-1]:.1f}, not 1.0), since a tile's projected "
        "size is only a fraction of the whole object's."
    )
    return fractions


def _apply_monotonicity_guard(thresholds: list[float], source: str) -> list[float]:
    """Enforce strict ascending thresholds WITHIN ``[0, 1]``, then assert.

    Defensive: guarantee strict monotonicity even when later children have the
    same (or fewer) elements than a previous level — WITHOUT ever breaching the
    viewport-relative contract (all values in ``[0, 1]``, coarsest ``0.0``,
    finest ``1.0``). An upward bump would push equal-tail ladders above 1.0
    (e.g. counts ``[10, 1000, 1000]``), silently moving the finest level's
    switch point *later* than the anchor the derivation promises — the explicit
    ``coverage_fractions=[...]`` escape hatch (bounded by
    ``MAX_COVERAGE_FRACTION``) is where an author opts into that deliberately,
    not something a count ladder should trigger by accident. So duplicates
    resolve by nudging the *earlier* (coarser) entries DOWNWARD instead, and the
    derived output stays inside ``[0, 1]`` — a strict subset of what the
    explicit-input validators accept, so a derived list always round-trips.

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
      increasing in insertion order;
    - any ``coverage_fraction`` is not finite (NaN / ±inf), or falls outside
      ``[0, MAX_COVERAGE_FRACTION]`` (the same bound the
      explicit-``coverage_fractions=`` resolvers enforce — this is the check for
      a hand-built ``add_lod_group`` ladder, which does not go through those
      resolvers).

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
        # Non-finite values must be rejected FIRST: every comparison below is
        # false for NaN, so a NaN would slip past both the range check and the
        # monotonicity one — and then become ``prev``, disabling the monotonicity
        # check for the whole rest of the ladder. ±inf would pass monotonicity too.
        if not math.isfinite(value):
            raise ValueError(
                f"LOD-group child {i} ({child.name!r}) has "
                f"coverage_fraction={value}, which is not a finite number"
            )
        if value < 0.0 or value > MAX_COVERAGE_FRACTION:
            raise ValueError(
                f"LOD-group child {i} ({child.name!r}) has "
                f"coverage_fraction={value}, must lie in "
                f"[0, {MAX_COVERAGE_FRACTION:g}]. The upper bound is "
                "1/FILL_FACTOR — the coverage metric a screen-filling object "
                "produces; 1.0 is a WHOLE-OBJECT ladder's finest anchor (~a "
                "quarter of the viewport diagonal), and higher values hold a "
                "level until the object is larger still (what a partition-bound "
                "ladder derives, and what an explicit list may ask for)."
            )
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
#:
#: Every one of these is a GAUSSIAN-MIXTURE reducer: it merges elements into
#: fewer, larger representative Gaussians. Points and Lines admit them only
#: because both LIFT to gsplats before coarsening — the set is a property of the
#: reduction, not of the geometry that asked for it.
SUBSTITUTIVE_METHODS = frozenset(
    {"auto", "kmeans", "kmeans_lloyd", "greedy", "greedy_lloyd"}
)

#: Mesh coarsening methods. Disjoint from the mixture set above except for
#: ``auto``, and that is the whole point of keeping the two apart.
#:
#: Mesh is the first geometry that does NOT lift to gsplats: a surface is
#: coarsened by DECIMATION (merge vertices, reindex faces, drop the triangles
#: that collapsed), which has no mixture to reduce and no ``kmeans`` to run.
#: Accepting ``method="kmeans"`` on a mesh would be accepting a word that names
#: nothing the code can do, so it is refused with the reason rather than
#: silently mapped onto something else.
#:
#: ``qem`` — Garland-Heckbert edge collapse — is the tier this set is shaped to
#: admit next (issue #1348). It is not listed until it exists: a method name
#: that validates and then raises is worse than one that never validated.
MESH_SUBSTITUTIVE_METHODS = frozenset({"auto", "cluster"})

#: Default mesh coarsening method. ``auto`` resolves to ``cluster`` today — the
#: only implemented tier — and becomes a real size-derived choice when ``qem``
#: lands (#1348). Kept as the default anyway so that upgrade is not a
#: behaviour change for anyone who wrote ``method="auto"``.
DEFAULT_MESH_SUBSTITUTIVE_METHOD: str = "auto"


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
      viewport-relative thresholds, strict-ascending in
      [0, ``MAX_COVERAGE_FRACTION``] — 1.0 is a whole-object ladder's finest
      anchor, above it holds a level until the object is larger still),
      ``coarsen_dims``,
      ``max_aspect`` (per-splat anisotropy cap on the coarse levels, default 3.0;
      ``None`` disables — see :func:`luxar.gsplats.lift._cap_aspect`).
      Unrecognized keys raise. LOD switch thresholds are otherwise auto-derived by
      :func:`derive_coverage_fractions` (``sqrt(N_i/N_finest)``, re-anchored at
      fills-screen when the insertion point is partition-bound) — no method
      selector or per-dataset anchor knob.
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
            f"substitutive_lod for {geometry}: method must be one of "
            f"{sorted(SUBSTITUTIVE_METHODS)}; got {method!r}"
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
                f"(one strictly-ascending value in [0, {MAX_COVERAGE_FRACTION:g}] "
                "per LOD level)"
            )
        _assert_strict_ascending(
            explicit_coverage, "substitutive_lod=dict(coverage_fractions=...)"
        )
        if explicit_coverage[0] < 0.0 or explicit_coverage[-1] > MAX_COVERAGE_FRACTION:
            raise ValueError(
                "substitutive_lod=dict(coverage_fractions=...): values must lie in "
                f"[0, {MAX_COVERAGE_FRACTION:g}] (coarsest→finest); got "
                f"{explicit_coverage}. The upper bound is 1/FILL_FACTOR — the "
                "coverage metric a screen-filling object produces; 1.0 is a "
                "whole-object ladder's finest anchor (~a quarter of the viewport "
                "diagonal)."
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


# ─────────────────────────────────────────────────────────────────────
# Shared additive-LOD axis resolver (Points + Lines)
# ─────────────────────────────────────────────────────────────────────
#
# Points and Lines share the exact same ``additive_lod=`` kwarg vocabulary
# (both order elements/polylines and slice into cumulative levels). Keep ONE
# implementation here so the per-geometry resolvers
# (``resolve_additive_axis_points`` / ``resolve_additive_axis_lines``) can never
# drift.

#: Defaults for ``additive_lod=True`` / ``additive_lod=dict()``. This is the
#: only place the values are written: ``points.DEFAULT_METHOD`` /
#: ``lines.DEFAULT_METHOD`` (and their ``DEFAULT_N_LODS``) alias these, so the
#: per-geometry function defaults and the resolver can't drift apart.
DEFAULT_ADDITIVE_METHOD: Literal["random"] = "random"
DEFAULT_ADDITIVE_N_LODS: int = 4


def resolve_additive_axis(spec: Any, geometry: str) -> Optional[dict]:
    """Normalize the ``additive_lod=`` kwarg into a spec dict (or ``None``).

    Geometry-agnostic — ``geometry`` ("Points"/"Lines") only flavours the
    "unrecognized keys" error message. See
    :func:`resolve_additive_axis_points` / :func:`resolve_additive_axis_lines`
    for the full value vocabulary.
    """
    if spec is None or spec is False:
        return None
    if spec is True:
        return {
            "method": DEFAULT_ADDITIVE_METHOD,
            "n_lods": DEFAULT_ADDITIVE_N_LODS,
            "counts": None,
            "seed": None,
            "salience_kind": "size",
        }
    if not isinstance(spec, dict):
        raise TypeError(
            f"additive_lod must be None, bool, or dict; got {type(spec).__name__}"
        )
    kwargs = dict(spec)
    # ``recompute`` is gsplats-only; tolerate but don't act on it.
    kwargs.pop("recompute", None)

    method = kwargs.pop("method", DEFAULT_ADDITIVE_METHOD)
    if method not in (
        "random",
        "salience",
        "spatial-uniform",
        "poisson-disk",
    ):
        raise ValueError(
            "method must be one of 'random' / 'salience' / 'spatial-uniform' "
            f"/ 'poisson-disk'; got {method!r}"
        )

    n_lods = int(kwargs.pop("n_lods", DEFAULT_ADDITIVE_N_LODS))
    if n_lods < 1:
        raise ValueError(f"n_lods must be >= 1, got {n_lods}")

    # Accept ``breakpoints`` as an alias for ``counts`` (the energy:
    # vocabulary reads more naturally as "breakpoints") — but only one.
    counts = kwargs.pop("counts", None)
    breakpoints = kwargs.pop("breakpoints", None)
    if counts is not None and breakpoints is not None:
        raise ValueError(
            "additive_lod: pass either 'counts' OR 'breakpoints', not both"
        )
    if breakpoints is not None:
        counts = breakpoints
    if counts is not None and not isinstance(counts, str):
        counts = [int(c) for c in counts]
    if counts is not None:
        # Validate everything size-independent here, at resolve time. Under a
        # substitutive ladder the resolved spec is handed to a kind=lod group
        # whose wrapper is created BEFORE its children are written, so deferring
        # this to the children's writes would raise only after that group
        # exists on disk — leaving a partial group (and a duplicate-name error
        # on retry). Same messages as the write path's re-validation.
        from ....utils.lod_breakpoints import validate_element_breakpoints

        validate_element_breakpoints(counts)

    seed = kwargs.pop("seed", None)
    if seed is not None:
        seed = int(seed)

    salience_kind = kwargs.pop("salience_kind", "size")
    if salience_kind not in ("size", "energy"):
        raise ValueError(
            f"salience_kind must be 'size' or 'energy'; got {salience_kind!r}"
        )

    if kwargs:
        raise ValueError(
            f"additive_lod for {geometry}: unrecognized keys "
            f"{sorted(kwargs)}. Valid keys: method, n_lods, counts, "
            f"breakpoints, seed, salience_kind, recompute."
        )

    return {
        "method": method,
        "n_lods": n_lods,
        "counts": counts,
        "seed": seed,
        "salience_kind": salience_kind,
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
