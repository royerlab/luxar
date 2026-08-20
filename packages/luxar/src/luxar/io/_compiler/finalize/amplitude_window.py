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
  all (4.51/3.95/2.36). The ratio is **bounded** to ``[1/10, 10]`` — see
  :data:`_SCALE_BOUND`.
* **Partition parts share the window VERBATIM**, unscaled. Parts are disjoint
  spatial pieces of ONE object with no representation change between them — a
  dim tile really is dim, and there is nothing to normalize away. Measured:
  per-part windows scored luminance 0.62 / chroma 0.123 with a 5.6x spread in
  window tops; one shared window scored 1.00 / 0.0001.

**Where the reference comes from.** For a ``kind=lod`` group it is the *finest*
child that actually has a window — see :func:`_lod_reference` for why that is
not always the literal finest child. For a ``kind=partition`` the parts are
aggregated: ``lo`` is the true union minimum, ``hi`` the **count-weighted mean**
of the part tops — see :func:`_combine`.

**Legacy fallback.** A store written before the two mass statistics existed (or
one hand-edited to drop them) carries no ratio to scale by. Such a level shares
the reference window verbatim (scale 1.0) rather than guessing: that is already
strictly better than the per-level windows it would otherwise keep, and it is
what a partition does anyway.

The pass **only ever overwrites an existing** ``amplitude_data_range`` — it
never creates one on a node that does not already carry it. The on-disk attr set
therefore stays exactly what the writers produce (a scalar-amplitude leaf writes
no window; ``kind=lod`` / ``kind=partition`` group nodes carry none), and this is
strictly a value correction. It never raises either: a malformed or hand-edited
store leaves the pass a no-op for that subtree rather than failing a compile.
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

#: Bound on the per-level window rescale, mirroring
#: ``gsplats.lod.substitutive._MASS_SCALE_BOUND`` (which likewise SKIPS an
#: analogous mass rescale, with a warning, rather than trusting a wild ratio).
#: The measured LOD ratios this corrects are ~1.1-1.2; ``mwma`` is a
#: second-moment ratio and therefore not robust on the heavy-tailed amplitude
#: distributions gsplat fits produce, so a 10x window rescale is far more likely
#: to be a degenerate statistic than a real representation change — and applying
#: it would be a LARGER switch pop than the ~2x defect this pass exists to fix.
_SCALE_BOUND = 10.0


class _Summary(NamedTuple):
    """Aggregate amplitude statistics of a subtree.

    ``sq`` is the total self-energy ``mass · mwma``, kept instead of ``mwma``
    itself so summaries combine by plain addition. ``n`` is the total splat
    count, the weight :func:`_combine` aggregates window tops by.
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
        """Did this subtree yield a usable ``(lo, hi)`` reference window?"""
        return self.lo is not None and self.hi is not None


_NOTHING = _Summary(0.0, 0.0, 0, None, None, False, False)

#: Memoized :func:`_summarize` results, keyed on the group's store path. One
#: cache per pass: ``_assign`` re-walks the same subtrees the root summary
#: already covered, and on an 8000-part merge of 6-rung ladders that doubles a
#: five-figure count of ``use_consolidated=False`` metadata reads.
_Cache = Dict[str, _Summary]


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
    non-content subgroup out — see :data:`_NODE_TYPES`.
    """
    out: List[Tuple[str, "zarr.Group", dict]] = []
    for name in group.group_keys():
        text = str(name)
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
    last resort for a hand-authored group that supplies neither.
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
    """Count-weighted mean of ``(hi, n_splats)`` pairs (unweighted if all 0)."""
    if not tops:
        return None
    total = sum(weight for _, weight in tops)
    if total > 0:
        return sum(value * weight for value, weight in tops) / total
    return sum(value for value, _ in tops) / len(tops)


def _combine(parts: Sequence[_Summary]) -> _Summary:
    """Aggregate sibling summaries: masses add; the window is pooled.

    ``lo`` is ``min(part lows)`` — the true union minimum, exact and a
    legitimate window bottom. ``hi`` is the **count-weighted mean** of the part
    tops (weights = each part's splat count), NOT their max: a per-part top is
    that part's own ``p99.9``, so the max of N of them grows with N and drifts
    toward the global maximum — the exact value ``p99.9`` exists to avoid
    (measured against the union's true p99.9: 1.05x at 2 parts, 1.14x at 16,
    1.58x at 256, 3.40x at 5000, and 1551x for 60 dim tiles plus one small
    bright one, which renders the whole object black). The batch-fit merge
    (``write_partition_streaming``) routinely emits thousands of parts.

    When the parts are similar, each part's ``p99.9`` estimates the same
    population quantile, so the count-weighted mean is a consistent estimator of
    the window a FLAT store of the same splats would have derived — and unlike
    the max, its expectation does not grow with the part count. The honest
    trade-off: a part whose own top sits far above the shared window clips more
    than its own brightest 0.1%. That is exactly what a flat store does to that
    same content.

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
    tops = [(p.hi, p.n) for p in real if p.hi is not None]
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
    # empty leaf (``n_splats == 0``) is stamped ``0.0`` / ``0.0``, and treating
    # that as missing used to drop the whole enclosing structure to scale 1.0.
    # The per-node ``mwma > 0`` guard in ``_level_scale`` handles it locally.
    has_stats = mass is not None and mwma is not None
    total_mass = max(mass, 0.0) if mass is not None and has_stats else 0.0
    return _Summary(
        mass=total_mass,
        sq=total_mass * mwma if (has_stats and mwma is not None) else 0.0,
        n=_count(attrs.get("n_splats")),
        lo=lo,
        hi=hi,
        has_stats=has_stats,
        has_gsplats=True,
    )


def _lod_reference(summaries: Sequence[_Summary]) -> Optional[_Summary]:
    """The finest child summary that actually carries a window.

    Walks FINEST → coarsest and takes the first usable one, rather than the
    literal finest child, because two shipped shapes put an un-windowed node
    there: ``add_points(substitutive_lod=…)`` / ``add_lines(substitutive_lod=…)``
    build ``[coarse gsplats levels …, points/lines finest]`` (the ``composed/``
    E2E fixture), and a level whose amplitude is a scalar/broadcast writes no
    ``amplitude_data_range`` at all. Taking the finest child unconditionally
    left the ENTIRE structure un-harmonized in both cases.

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
    ``zarr.json`` rewrite. Returns the number of attrs written (0 or 1).
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


def _level_scale(summary: _Summary, ref: Optional[_Summary], path: str) -> float:
    """The LOD level's window scale ``summary.mwma / ref.mwma``, guarded.

    Falls back to the documented verbatim ``1.0`` whenever the ratio is not a
    trustworthy positive finite number, or lands outside
    ``[1/_SCALE_BOUND, _SCALE_BOUND]``.
    """
    if ref is None or not (summary.has_stats and ref.has_stats):
        return 1.0
    own, reference = summary.mwma, ref.mwma
    if not (own > 0.0 and reference > 0.0):
        return 1.0
    scale = own / reference
    if not math.isfinite(scale) or scale <= 0.0:
        return 1.0
    if not (1.0 / _SCALE_BOUND <= scale <= _SCALE_BOUND):
        aprint(
            f"  ⚠️  amplitude_data_range: LOD window rescale skipped for "
            f"{path or '/'} — mass-weighted amplitude ratio {scale:.4g} is "
            f"outside [{1.0 / _SCALE_BOUND:g}, {_SCALE_BOUND:g}]; sharing the "
            f"reference window verbatim instead"
        )
        return 1.0
    return scale


def _assign(group: "zarr.Group", window: _Window, cache: _Cache) -> int:
    """Hand ``window`` down a subtree, scaling per LOD level.

    Returns how many ``amplitude_data_range`` attrs were rewritten.
    """
    attrs = dict(group.attrs)
    kind = attrs.get("kind")

    if kind == "lod":
        children = _lod_children(group)
        if not children:
            return 0
        summaries = [_summarize(child, cache) for _, child, _ in children]
        ref = _lod_reference(summaries)
        written = 0
        for (_, child, _), summary in zip(children, summaries):
            scale = _level_scale(summary, ref, child.path)
            written += _assign(child, (window[0] * scale, window[1] * scale), cache)
        return written

    if kind == "partition":
        # Verbatim: parts are disjoint pieces of one object, not a change of
        # representation, so there is nothing to normalize away.
        return sum(_assign(child, window, cache) for _, child, _ in _child_nodes(group))

    if attrs.get("type") == "gsplats":
        # An additive ladder's sub-LOD groups are prefix increments of the SAME
        # content the parent describes — same window, no scaling.
        return _stamp(group, window) + sum(
            _stamp(sub, window) for _, sub in _indexed_children(group, "additive_")
        )

    return sum(_assign(child, window, cache) for _, child, _ in _child_nodes(group))


def harmonize_gsplat_amplitude_windows(store: zarr.Group) -> None:
    """Put every node of a gsplat structure on ONE colormap window.

    Walks the zarr tree and, for each **maximal gsplat structure root** — a
    ``kind in {"lod", "partition"}`` group whose subtree holds at least one
    ``type == "gsplats"`` leaf — rewrites every ``amplitude_data_range`` beneath
    it from a single reference window: the finest windowed LOD content's, scaled
    per level by the (bounded) mass-weighted mean amplitude ratio, shared
    verbatim across partition parts. The walk does not descend past a structure
    root looking for more roots, so a nested ladder is harmonized as part of its
    parent structure rather than independently.

    A plain gsplats leaf on its own is a no-op (there is nothing to harmonize),
    and points/lines/mesh nodes and their ``scalar_data_range`` are never
    touched. See the module docstring for the measurement behind the rule.
    """
    cache: _Cache = {}

    def harmonize(group: "zarr.Group", kind: str, summary: _Summary) -> None:
        if summary.lo is None or summary.hi is None:
            return
        written = _assign(group, (summary.lo, summary.hi), cache)
        if written:
            aprint(
                f"  🎚️  Harmonized amplitude_data_range over kind={kind} gsplats "
                f"structure {group.path or '/'}: "
                f"[{summary.lo:.4g}, {summary.hi:.4g}] on {written} node(s)"
            )

    def visit(group: "zarr.Group") -> None:
        # Per-subtree containment: a malformed / hand-edited structure must not
        # fail a compile over a display window, nor stop the sibling structures
        # from being harmonized. The windows in that subtree simply stay as the
        # writers left them.
        try:
            kind = dict(group.attrs).get("kind")
            # The summary doubles as the containment test (``has_gsplats``), so
            # the tree is not walked a second time purely to classify a root.
            summary = _summarize(group, cache) if kind in ("lod", "partition") else None
            children = list(group.group_keys())
        except Exception as exc:  # pragma: no cover - defensive
            aprint(f"  ⚠️  amplitude_data_range harmonization skipped: {exc}")
            return
        if summary is not None and summary.has_gsplats:
            try:
                harmonize(group, str(kind), summary)
            except Exception as exc:  # pragma: no cover - defensive
                aprint(
                    f"  ⚠️  amplitude_data_range harmonization skipped for "
                    f"{group.path or '/'}: {exc}"
                )
            return
        for name in children:
            visit(group[name])

    visit(store)
