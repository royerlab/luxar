"""Finalize-time harmonization of ``amplitude_data_range`` over a gsplat tree.

**The defect (#1691).** ``write_gsplat_arrays`` derives each node's
``amplitude_data_range`` as ``[min(a), p99.9(a)]`` of *that node's own*
amplitudes, and the viewer reads the window per node. That is correct for one
flat leaf and wrong for every multi-node gsplat structure:

* On a ``kind=lod`` group of substitutive levels, a coarse level's merged
  representatives carry the same total mass in far fewer splats, so its p99.9
  sits ~4.5x above the finest level's while its *typical* amplitude only grows
  ~1.1x. At a LOD switch the object therefore re-tones AND pops ~2x in
  brightness.
* ``kind=partition`` compounds it: measured adjacent tiles of one object were
  windowed at ``[0.40, 4.04]`` and ``[0.0036, 0.0355]`` — a ~100x difference
  across a single seam.

**The fix.** After the whole tree is on disk, take ONE reference window per
gsplat structure and hand it down, per-level-scaled where — and only where — the
*representation* changed:

* **LOD levels are scaled by the mass-weighted mean amplitude ratio**
  ``child.mwma / reference.mwma``. A substitutive reduction conserves mass and
  redistributes it over fewer splats, so the amplitude a unit of mass carries is
  exactly what moves; the mass-weighted mean amplitude measures it. Measured on
  a rasterized orthographic proxy of the real colormap render, scored against
  the finest level: per-level windows (the old behaviour) gave luminance ratios
  0.49 / 0.51 / 0.71 and chromaticity L1 0.187 / 0.183 / 0.100; one shared
  window scaled by this ratio gave 1.04 / 1.01 / 1.00 and 0.048 / 0.028 / 0.028.
  The estimator also tracked the swept optimum (1.19/1.10/1.09 predicted vs
  1.22/1.06/1.06 measured), whereas the p99.9 the window itself uses did not at
  all (4.51/3.95/2.36). The ratio is **clamped** to ``[1/10, 10]`` — see
  :data:`_SCALE_BOUND`.
* **Partition parts share the window VERBATIM**, unscaled. Parts are disjoint
  spatial pieces of ONE object with no representation change between them — a
  dim tile really is dim, and there is nothing to normalize away. Measured:
  per-part windows scored luminance 0.62 / chroma 0.123 with a 5.6x spread in
  window tops; one shared window scored 1.00 / 0.0001.

(The luminance / chromaticity tables above are the measurements recorded on
issue #1691 itself.)

**Where the reference comes from.** For a ``kind=lod`` group it is the *finest*
child that actually carries a USABLE window — see :func:`_lod_reference` for why
that is not always the literal finest child. For a ``kind=partition`` the parts
are aggregated: ``lo`` is the true union minimum, ``hi`` the **count-weighted
mean** of the usable part tops — see :func:`_combine` for why that rule and not
``max``, and for what it costs.

**Usable means ``hi > lo``.** A degenerate ``[x, x]`` window is not a window: it
reads as identity in the viewer, and :func:`_stamp` refuses to write one. If
such a node were allowed to *donate*, the whole structure would silently keep
the per-level windows this pass exists to remove. That shape ships — a constant
amplitude set gets ``[x, x]`` from the writer, and ``gsplats/interop`` gives
EVERY imported classical splat file (INRIA PLY, ``.splat``, SPZ, SuperSplat)
``amplitudes = 1`` — so ``gsplat import`` → ``gsplat lod --recipe levels`` lands
on it directly.

**Legacy fallback.** A store written before the two mass statistics existed (or
one hand-edited to drop them) carries no ratio to scale by. Such a level shares
the reference window verbatim (scale 1.0) rather than guessing. That is not
provably better than what it replaces — a self-consistent legacy sibling can end
up clipped by the shared window — but it is the honest answer when there is no
ratio to scale by, and it is what the partition arm does anyway: the structure
ends up on ONE window.

**Cross-recipe consequence.** A ``levels`` structure's reference top is one
level's real p99.9, while an ``overview`` / ``adaptive`` one is a *pooled*
estimate over parts. The same splats can therefore tone slightly differently
depending on the topology they were written in (measured 222.34 vs 160.50 on one
dataset). That is inherent to pooling, not a bug in either arm.

The pass **only ever overwrites an existing** ``amplitude_data_range`` — it
never creates one on a node that does not already carry it. The on-disk attr set
therefore stays exactly what the writers produce (a scalar-amplitude leaf writes
no window; ``kind=lod`` / ``kind=partition`` group nodes carry none), and this is
strictly a value correction. It never raises either: a malformed or hand-edited
store leaves the pass a no-op for that subtree rather than failing a compile.
Note that :func:`_assign` writes INCREMENTALLY, so a failure partway through one
structure leaves that structure **partially** rewritten — a mix of harmonized
and writer-derived windows, not a clean rollback. The warning reports how many
nodes had already been written when it gave up.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import zarr
from arbol import aprint

#: The reference window carried down a structure: ``(lo, hi)``.
_Window = Tuple[float, float]

#: Node ``type`` values that name a real scene node. A child group whose type is
#: not one of these (a ``labels`` / overlay / auxiliary subgroup, a ``pipeline``
#: or ``fitting`` bucket) is never a LOD level or a partition part, so it must
#: not be enumerated as one — mistaking it for the FINEST child would pick the
#: whole structure's reference window off something that is not content.
_NODE_TYPES = frozenset({"group", "gsplats", "points", "lines", "mesh"})
_GSPLAT_BOOKKEEPING_GROUPS = frozenset({"fitting", "provenance", "pipeline"})

#: Bound on the per-level window rescale, mirroring
#: ``gsplats.lod.substitutive._MASS_SCALE_BOUND``. The measured LOD ratios this
#: corrects are ~1.1-1.2; ``mwma`` is a second-moment ratio and therefore not
#: robust on the heavy-tailed amplitude distributions gsplat fits produce, so a
#: 10x window rescale is far more likely to be a degenerate statistic than a real
#: representation change. An out-of-bound ratio is **clamped to the bound**, not
#: discarded: the bound is insurance against a pathological statistic, and
#: clamping caps the damage in both directions, whereas falling back to 1.0
#: applies the full uncorrected error (measured at a genuine ratio of 0.02,
#: scale 1.0 leaves the level windowed 50x too wide — it renders black — while
#: clamping to 0.1 caps that at 5x).
_SCALE_BOUND = 10.0


class _Summary(NamedTuple):
    """Aggregate amplitude statistics of a subtree.

    ``sq`` is the total self-energy ``mass · mwma``, kept instead of ``mwma``
    itself so summaries combine by plain addition. ``n`` is the **pooling
    weight** :func:`_combine` aggregates window tops by — the splat count, or
    ``1`` for a node that declares none (see :func:`_pool_weight`).
    """

    mass: float
    sq: float
    n: int
    lo: Optional[float]
    hi: Optional[float]
    has_stats: bool
    has_gsplats: bool

    @property
    def mwma(self) -> float:
        """Mass-weighted mean amplitude (``0.0`` for a mass-less subtree)."""
        return self.sq / self.mass if self.mass > 0.0 else 0.0

    @property
    def has_window(self) -> bool:
        """Did this subtree yield a USABLE ``(lo, hi)`` reference window?

        Finiteness is not enough: a degenerate ``[x, x]`` is refused by
        :func:`_stamp` on the way out, so accepting it on the way IN would leave
        the structure un-harmonized and unlogged. See the module docstring.
        """
        return self.lo is not None and self.hi is not None and self.hi > self.lo


_NOTHING = _Summary(0.0, 0.0, 0, None, None, False, False)

#: Memoized :func:`_summarize` results, keyed on the group's store path. One
#: cache per pass: ``_assign`` re-walks the same subtrees the root summary
#: already covered, and on an 8000-part merge of 6-rung ladders that doubles a
#: five-figure count of ``use_consolidated=False`` metadata reads.
_Cache = Dict[str, _Summary]


class _Progress:
    """Mutable per-structure bookkeeping for one :func:`_assign` walk.

    ``written`` is threaded through a mutable object rather than returned so the
    count survives an exception mid-walk (the warning reports how much of the
    structure was already rewritten). ``clamped`` / ``extreme`` roll the
    :data:`_SCALE_BOUND` hits up into ONE console line per structure — a 40-rung
    ladder or a thousand-part ``adaptive`` merge would otherwise flood finalize.
    """

    def __init__(self) -> None:
        self.written = 0
        self.clamped = 0
        self.extreme = 1.0

    def clamp(self, ratio: float) -> None:
        """Record one out-of-bound ratio, keeping the most extreme seen."""
        self.clamped += 1
        if abs(math.log(ratio)) > abs(math.log(self.extreme)):
            self.extreme = ratio


def _finite(value: object) -> Optional[float]:
    """``value`` as a finite float, or ``None`` (covers bools, strings, NaN)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    fv = float(value)
    return fv if math.isfinite(fv) else None


def _count(value: object) -> int:
    """``value`` as a non-negative splat count, or ``0``."""
    fv = _finite(value)
    if fv is None or fv < 0.0:
        return 0
    return int(fv)


def _pool_weight(attrs: dict) -> int:
    """The weight this node's window top is pooled by in :func:`_combine`.

    An ABSENT ``n_splats`` means "weight unknown", not "weight zero": weighting
    it 0 silently erases that node from the pool entirely. It is pooled at 1
    instead. A ``n_splats`` that is PRESENT and 0 is a genuinely empty node and
    keeps weight 0 — "present and zero" and "absent" must stay distinguishable.
    """
    if "n_splats" not in attrs:
        return 1
    return _count(attrs["n_splats"])


def _window_of(attrs: dict) -> Tuple[Optional[float], Optional[float]]:
    """The ``amplitude_data_range`` of an attrs dict as ``(lo, hi)`` floats."""
    raw = attrs.get("amplitude_data_range")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None, None
    lo, hi = _finite(raw[0]), _finite(raw[1])
    if lo is None or hi is None:
        return None, None
    return lo, hi


def _child_nodes(group: "zarr.Group") -> List[Tuple[str, "zarr.Group", dict]]:
    """Every child group that is a scene node, as ``(name, group, attrs)``.

    Name-agnostic on purpose: ``Node.add_lod_group`` / ``add_partition_group``
    are PUBLIC and the child names are then the author's
    (``examples/partition_of_lod_example.py`` names its levels ``lod_coarse`` /
    ``lod_fine``), so a ``child_<i>`` / ``part_<i>`` name filter would silently
    skip every hand-authored structure. The ``type`` filter is what keeps a
    non-content subgroup out — see :data:`_NODE_TYPES`. The three standalone
    gsplat bookkeeping groups are excluded even if arbitrary fitting metadata
    gives one a scene-node ``type``.
    """
    out: List[Tuple[str, "zarr.Group", dict]] = []
    for name in group.group_keys():
        text = str(name)
        if text in _GSPLAT_BOOKKEEPING_GROUPS:
            continue
        child = group[text]
        attrs = dict(child.attrs)
        if attrs.get("type") in _NODE_TYPES:
            out.append((text, child, attrs))
    return out


def _numeric_suffix(name: str, prefix: str) -> Optional[int]:
    """The integer ``<i>`` of a ``prefix<i>`` name, or ``None``."""
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def _lod_children(group: "zarr.Group") -> List[Tuple[str, "zarr.Group", dict]]:
    """A ``kind=lod`` group's children ordered COARSEST → FINEST.

    Ascending ``child_index`` is coarsest→finest because ``child_index`` records
    on-disk *insertion* order (``Node.__init__``, and ``gsplat_tree`` for the
    standalone writer) and every producer inserts coarsest first:
    ``core/group/gsplats_pipeline/lod_dispatch.py`` walks the substitutive
    levels through an explicit ``order = range(n_sub - 1, -1, -1)``, and
    ``core/group/adders/points.py`` builds ``coarse_first =
    list(reversed(coarse))``.

    Priority: (a) ``child_index`` when EVERY candidate carries one, (b) the
    numeric suffix of a ``child_<i>`` name when every candidate has one — never
    alphabetically, or ``child_10`` would sort before ``child_2`` and a
    >=10-level ladder would pick the wrong finest child — (c) sorted name as a
    last resort.

    Rule (c) assumes alphabetically-last is finest, the same convention
    ``finalize/lod_backfill.py`` already applies. It is unreachable from any
    Python producer — ``core/node/node.py`` always stamps ``child_index`` — and
    exists only for a hand-edited or third-party store that supplies neither key.
    """
    kids = _child_nodes(group)
    indices = [_finite(attrs.get("child_index")) for _, _, attrs in kids]
    if kids and all(i is not None for i in indices):
        order: List[float] = [i for i in indices if i is not None]
        return [kid for _, kid in sorted(zip(order, kids), key=lambda p: p[0])]
    suffixes = [_numeric_suffix(name, "child_") for name, _, _ in kids]
    if kids and all(s is not None for s in suffixes):
        nums: List[int] = [s for s in suffixes if s is not None]
        return [kid for _, kid in sorted(zip(nums, kids), key=lambda p: p[0])]
    return sorted(kids, key=lambda kid: kid[0])


def _indexed_children(
    group: "zarr.Group", prefix: str
) -> List[Tuple[int, "zarr.Group"]]:
    """``prefix<i>`` child groups sorted NUMERICALLY by their index suffix.

    Only for ``additive_<i>`` sub-LODs, whose names are writer-owned (no public
    API lets a caller name them) — unlike LOD levels and partition parts, which
    go through :func:`_child_nodes`.
    """
    out: List[Tuple[int, "zarr.Group"]] = []
    for name in group.group_keys():
        index = _numeric_suffix(str(name), prefix)
        if index is not None:
            out.append((index, group[str(name)]))
    out.sort(key=lambda item: item[0])
    return out


def _weighted_top(tops: Sequence[Tuple[float, int]]) -> Optional[float]:
    """Count-weighted mean of ``(hi, weight)`` pairs (unweighted if all 0)."""
    if not tops:
        return None
    total = sum(weight for _, weight in tops)
    if total > 0:
        return sum(value * weight for value, weight in tops) / total
    return sum(value for value, _ in tops) / len(tops)


def _combine(parts: Sequence[_Summary]) -> _Summary:
    """Aggregate sibling summaries: masses add; the window is pooled.

    ``lo`` is ``min(part lows)`` — the true union minimum, exact and a
    legitimate window bottom. ``hi`` is the **count-weighted mean** of the
    *usable* part tops (weights = each part's splat count), NOT their max.

    Both candidate rules are biased, in opposite directions, and neither
    recovers the union's true ``p99.9``:

    * ``max`` over part tops drifts UPWARD without bound in the part count —
      measured against the union's true p99.9: 1.05x at 2 parts, 1.58x at 256,
      3.40x at 5000, and ~1500x on 60 dim tiles plus one small bright one, at
      which point the whole object renders black. ``write_partition_streaming``
      routinely emits thousands of parts, so this is not a corner case.
    * the count-weighted mean is biased slightly LOW, because a part's own
      ``p99.9`` is itself a downward-biased estimate of the union's (measured
      ~0.81x on 16 similar tiles, and 13.8x low on that same pathological case).

    The mean is chosen because its bias does **not** grow with the part count,
    and because its failure mode is the recoverable one: it clips
    outlier-bright content, which is precisely what a ``p99.9`` window does by
    design. It is neither unbiased nor a consistent estimator of what a flat
    store would have derived — do not read it as one.

    A part windowed ``[x, x]`` carries no window information at all, so it is
    excluded from the mean rather than dragged in at full weight (measured: a
    100 000-splat part windowed ``[1.0, 1.0]`` next to a 1 000-splat
    ``[0.01, 50.0]`` pooled to ``[0.01, 1.485]``, over-brightening the second
    part 34x and clipping everything above 1.49 to white). Its ``lo`` still
    counts toward the union minimum, which is a real observation. If no usable
    top survives, the combine has no window.

    (``write_gsplat_leaf`` aggregates an additive ladder as ``[min lo, max hi]``
    instead, and rightly so: a ladder's sub-LODs are disjoint *increments of one
    set* whose union is the thing being windowed, whereas partition parts are
    disjoint *pieces* each of which already windowed itself.)

    ``has_stats`` requires EVERY informative sibling to have them — a partial
    set cannot be scaled against, so the caller falls back to sharing the window
    verbatim.
    """
    has_gsplats = any(p.has_gsplats for p in parts)
    # A sibling that yielded neither a window nor statistics contributes no
    # information; excluding it keeps an unrecognised wrapper from poisoning
    # ``has_stats`` for the real siblings.
    real = [p for p in parts if p.has_window or p.has_stats]
    if not real:
        return _NOTHING._replace(has_gsplats=has_gsplats)
    los = [p.lo for p in real if p.lo is not None]
    tops = [(p.hi, p.n) for p in real if p.has_window and p.hi is not None]
    return _Summary(
        mass=sum(p.mass for p in real),
        sq=sum(p.sq for p in real),
        n=sum(p.n for p in real),
        lo=min(los) if los else None,
        hi=_weighted_top(tops),
        has_stats=all(p.has_stats for p in real),
        has_gsplats=has_gsplats,
    )


def _leaf_summary(attrs: dict) -> _Summary:
    """Summary of one ``type == "gsplats"`` leaf from its attrs."""
    mass = _finite(attrs.get("amplitude_mass"))
    mwma = _finite(attrs.get("amplitude_mass_weighted_mean"))
    lo, hi = _window_of(attrs)
    # "The statistics are PRESENT", not "the mass is positive": a legitimately
    # mass-less leaf (all-zero amplitudes) is stamped ``0.0`` / ``0.0``, and
    # treating that as missing used to drop the whole enclosing structure to
    # scale 1.0. The per-node ``mwma > 0`` guard in ``_level_scale`` handles it.
    has_stats = mass is not None and mwma is not None
    total_mass = max(mass, 0.0) if mass is not None and has_stats else 0.0
    return _Summary(
        mass=total_mass,
        sq=total_mass * mwma if (has_stats and mwma is not None) else 0.0,
        n=_pool_weight(attrs),
        lo=lo,
        hi=hi,
        has_stats=has_stats,
        has_gsplats=True,
    )


def _lod_reference(summaries: Sequence[_Summary]) -> Optional[_Summary]:
    """The finest child summary that actually carries a usable window.

    Walks FINEST → coarsest and takes the first usable one, rather than the
    literal finest child, because three shipped shapes put an un-windowable node
    there: ``add_points(substitutive_lod=…)`` / ``add_lines(substitutive_lod=…)``
    build ``[coarse gsplats levels …, points/lines finest]`` (the ``composed/``
    E2E fixture); a level whose amplitude is a scalar/broadcast writes no
    ``amplitude_data_range`` at all; and a constant-amplitude level (every
    imported classical splat file) writes a DEGENERATE ``[x, x]`` one. Taking
    the finest child unconditionally left the ENTIRE structure un-harmonized in
    all three cases — silently, since nothing could then be written.

    The donor supplies BOTH the reference window and the reference ``mwma`` the
    siblings are scaled against, so the two always come from the same content.
    """
    for summary in reversed(summaries):
        if summary.has_window:
            return summary
    return None


def _lod_summary(summaries: Sequence[_Summary]) -> _Summary:
    """A ``kind=lod`` group's summary: its reference child's, plus containment."""
    has_gsplats = any(s.has_gsplats for s in summaries)
    ref = _lod_reference(summaries)
    if ref is None:
        return _NOTHING._replace(has_gsplats=has_gsplats)
    return ref._replace(has_gsplats=has_gsplats)


def _summarize(group: "zarr.Group", cache: _Cache) -> _Summary:
    """Post-order summary of a subtree, memoized on the group's store path."""
    key = group.path
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = _summarize_uncached(group, cache)
    cache[key] = result
    return result


def _summarize_uncached(group: "zarr.Group", cache: _Cache) -> _Summary:
    attrs = dict(group.attrs)
    kind = attrs.get("kind")

    if kind == "lod":
        return _lod_summary(
            [_summarize(child, cache) for _, child, _ in _lod_children(group)]
        )

    if kind == "partition":
        return _combine(
            [_summarize(child, cache) for _, child, _ in _child_nodes(group)]
        )

    if attrs.get("type") == "gsplats":
        return _leaf_summary(attrs)

    # A plain group (or an unrecognised node): defensively recurse and combine
    # like a partition, so a wrapper nobody anticipated does not silently drop
    # the leaves underneath it.
    return _combine([_summarize(child, cache) for _, child, _ in _child_nodes(group)])


def _stamp(group: "zarr.Group", window: _Window) -> int:
    """Overwrite an EXISTING ``amplitude_data_range`` with ``window``.

    Never creates the attr: a node that carries no window today (a
    scalar-amplitude leaf, a group wrapper) must not start carrying one — this
    pass corrects values, it does not extend the format. Refuses anything that
    is not a finite ``lo < hi``, leaving the writer's value in place: a
    degenerate ``[x, x]`` reads as identity in the viewer (an un-windowed level)
    and an inverted ``lo > hi`` inverts the colormap — both worse than the
    per-node window this pass replaces. Skips the write when the stored value
    already equals the new one (the finest level always does, and so does a
    partition part that happened to match), so an unchanged node costs no
    ``zarr.json`` rewrite — which is also what makes a second run of the pass a
    silent no-op. Returns the number of attrs written (0 or 1).
    """
    attrs = dict(group.attrs)
    if "amplitude_data_range" not in attrs:
        return 0
    lo, hi = float(window[0]), float(window[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        return 0
    if _window_of(attrs) == (lo, hi):
        return 0
    group.attrs["amplitude_data_range"] = [lo, hi]
    return 1


def _level_scale(
    summary: _Summary, ref: Optional[_Summary], progress: _Progress
) -> float:
    """The LOD level's window scale ``summary.mwma / ref.mwma``, guarded.

    Falls back to the documented verbatim ``1.0`` only when there is no usable
    ratio at all — missing statistics on either side, a non-positive ``mwma``,
    or a non-finite quotient. A ratio that is real but lands outside
    ``[1/_SCALE_BOUND, _SCALE_BOUND]`` is **clamped** to the bound and recorded
    on ``progress``; discarding it would apply the full uncorrected error
    instead of capping it (see :data:`_SCALE_BOUND`).
    """
    if ref is None or not (summary.has_stats and ref.has_stats):
        return 1.0
    own, reference = summary.mwma, ref.mwma
    if not (own > 0.0 and reference > 0.0):
        return 1.0
    scale = own / reference
    if not math.isfinite(scale) or scale <= 0.0:
        return 1.0
    lower, upper = 1.0 / _SCALE_BOUND, _SCALE_BOUND
    if scale < lower or scale > upper:
        progress.clamp(scale)
        return min(max(scale, lower), upper)
    return scale


def _assign(
    group: "zarr.Group", window: _Window, cache: _Cache, progress: _Progress
) -> None:
    """Hand ``window`` down a subtree, scaling per LOD level.

    Accumulates the number of ``amplitude_data_range`` attrs rewritten into
    ``progress.written`` as it goes — the writes are incremental, so an
    exception partway through leaves both a partially rewritten structure and an
    accurate count of how far it got.
    """
    attrs = dict(group.attrs)
    kind = attrs.get("kind")

    if kind == "lod":
        children = _lod_children(group)
        if not children:
            return
        summaries = [_summarize(child, cache) for _, child, _ in children]
        ref = _lod_reference(summaries)
        for (_, child, _), summary in zip(children, summaries):
            scale = _level_scale(summary, ref, progress)
            _assign(child, (window[0] * scale, window[1] * scale), cache, progress)
        return

    if kind == "partition":
        # Verbatim: parts are disjoint pieces of one object, not a change of
        # representation, so there is nothing to normalize away.
        for _, child, _ in _child_nodes(group):
            _assign(child, window, cache, progress)
        return

    if attrs.get("type") == "gsplats":
        # An additive ladder's sub-LOD groups are prefix increments of the SAME
        # content the parent describes — same window, no scaling.
        progress.written += _stamp(group, window)
        for _, sub in _indexed_children(group, "additive_"):
            progress.written += _stamp(sub, window)
        return

    for _, child, _ in _child_nodes(group):
        _assign(child, window, cache, progress)


def _report(path: str, kind: str, window: _Window, progress: _Progress) -> None:
    """The at-most-two console lines one harmonized structure emits.

    Both are ROLLED UP per structure: the clamp line in particular would
    otherwise be one ``aprint`` per level, and a 40-rung ladder or a
    thousand-part ``adaptive`` merge would flood the finalize console.
    """
    if progress.written:
        aprint(
            f"  🎚️  Harmonized amplitude_data_range over kind={kind} gsplats "
            f"structure {path}: [{window[0]:.4g}, {window[1]:.4g}] on "
            f"{progress.written} node(s)"
        )
    if progress.clamped:
        aprint(
            f"  ⚠️  amplitude_data_range: {progress.clamped} LOD level(s) of "
            f"{path} had a mass-weighted amplitude ratio outside "
            f"[{1.0 / _SCALE_BOUND:g}, {_SCALE_BOUND:g}] and were clamped to it "
            f"(most extreme ratio seen: {progress.extreme:.4g})"
        )


def _harmonize_structure(
    group: "zarr.Group", kind: str, summary: _Summary, cache: _Cache
) -> None:
    """Hand one structure's reference window down its whole subtree."""
    lo, hi = summary.lo, summary.hi
    # ``summary.has_window``, spelled out so mypy narrows both to ``float``.
    if lo is None or hi is None or not (hi > lo):
        return
    path = group.path or "/"
    progress = _Progress()
    try:
        _assign(group, (lo, hi), cache, progress)
    except Exception as exc:  # pragma: no cover - defensive
        aprint(
            f"  ⚠️  amplitude_data_range harmonization failed partway for "
            f"{path}: {exc} — {progress.written} node(s) had already been "
            f"rewritten, so this structure is left partially harmonized"
        )
        return
    _report(path, kind, (lo, hi), progress)


def _visit(group: "zarr.Group", cache: _Cache) -> None:
    """Find the maximal gsplat structure roots under ``group`` and harmonize them.

    Per-subtree containment: a malformed / hand-edited structure must not fail a
    compile over a display window, nor stop the sibling structures from being
    harmonized. The windows in that subtree simply stay as the writers left them.
    """
    try:
        kind = dict(group.attrs).get("kind")
        # The summary doubles as the containment test (``has_gsplats``), so the
        # tree is not walked a second time purely to classify a root.
        summary = _summarize(group, cache) if kind in ("lod", "partition") else None
        children = list(group.group_keys())
    except Exception as exc:  # pragma: no cover - defensive
        aprint(f"  ⚠️  amplitude_data_range harmonization skipped: {exc}")
        return
    if summary is not None and summary.has_gsplats:
        _harmonize_structure(group, str(kind), summary, cache)
        return
    for name in children:
        # Opening a child is itself fallible (a truncated / hand-edited node);
        # containment is per subtree, so one bad child must not abort the walk
        # over its siblings.
        try:
            child = group[str(name)]
        except Exception as exc:  # pragma: no cover - defensive
            aprint(
                f"  ⚠️  amplitude_data_range harmonization skipped for "
                f"{group.path or '/'}/{name}: {exc}"
            )
            continue
        _visit(child, cache)


def harmonize_gsplat_amplitude_windows(store: zarr.Group) -> None:
    """Put every node of a gsplat structure on ONE colormap window.

    Walks the zarr tree and, for each **maximal gsplat structure root** — a
    ``kind in {"lod", "partition"}`` group whose subtree holds at least one
    ``type == "gsplats"`` leaf — rewrites every ``amplitude_data_range`` beneath
    it from a single reference window: the finest usably-windowed LOD content's,
    scaled per level by the (clamped) mass-weighted mean amplitude ratio, shared
    verbatim across partition parts. The walk does not descend past a structure
    root looking for more roots, so a nested ladder is harmonized as part of its
    parent structure rather than independently.

    A plain gsplats leaf on its own is a no-op (there is nothing to harmonize),
    and points/lines/mesh nodes and their ``scalar_data_range`` are never
    touched. See the module docstring for the measurement behind the rule.

    Containment is per SUBTREE, not per node: a malformed or hand-edited
    structure is skipped with a warning instead of failing the compile, and its
    siblings are still harmonized (see :func:`_visit`). It is **not**
    transactional — see :func:`_assign`.
    """
    _visit(store, {})
