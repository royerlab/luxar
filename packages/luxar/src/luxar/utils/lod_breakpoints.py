"""luxar.utils.lod_breakpoints – streaming-ladder breakpoint math, shared by all
three geometries.

An additive (streaming) LOD ladder cuts an importance-ordered element sequence
into cumulative prefixes so a viewer can paint a coarse prefix immediately and
refine as later chunks arrive. The **cut geometry** — how big the first chunk is
and how the rest grow — is a property of the network and the payload, not of the
geometry type, so Points, Lines and GSplats must derive identical cuts from an
identical spec. That shared math lives here.

Dependency-free by design (stdlib only): ``luxar.gsplats`` already depends on
``luxar.utils``, and ``luxar.core`` may too, so this is the one place both can
reach without introducing a new import direction.

The ``stream:<c>`` spec
-----------------------
``c`` is the first chunk's element count, normally derived from a download-time
budget via :func:`streaming_chunk_splats`. Cumulative cuts then double —
``[c, 2c, 4c, …, N]`` — so first paint costs one chunk and each refinement
doubles the resident set. Because the spec is resolved against *this* ``n``, the
same string adapts to every level, part and leaf of a tree.

The ``equi-energy:<n>`` spec
----------------------------
Cuts at equal shares of cumulative ENERGY along the ladder's own ordering rather
than at counts: on a contribution-first ordering the first rung is the few
heaviest elements and each later rung is fatter in count for the same light, so
first paint is fast and the slow rungs are the ones that matter least. Fat late
rungs are split at :data:`DEFAULT_MAX_ADDITIVE_COMMIT` (see
:func:`equi_energy_cuts`). Every geometry consumes it: GSplats against their
self-energy, Points against ``luminance × radius³``, Lines against
``luminance × Σ length·width²`` (per polyline, capped in vertices).

Doubling has one failure mode, and it is size-driven rather than spec-driven: the
FINAL increment of a pure doubling ladder is ``N`` minus the largest doubling
below it, so it grows with ``N`` and approaches ``N/2`` in the worst case however
small ``c`` is. Past a few million elements that last commit blocks the main
thread, so a large leaf wants :func:`capped_stream_cuts` — the same geometric
head, then equal steps of a fixed ceiling — instead of :func:`stream_cuts`.
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Union

#: Breakpoint specification for an additive ladder. Either a string form
#: (``"equal-count"``, ``"stream:<c>"``, ``"equi-energy:<n>"``,
#: ``"energy:<fractions>"`` for Points/Lines) or an explicit cumulative count /
#: fraction sequence.
BreakpointSpec = Union[str, Sequence[int], Sequence[float]]

#: Prefix of the equal-energy rung spec (see :func:`equi_energy_cuts`).
EQUI_ENERGY_PREFIX = "equi-energy:"

#: Assumed downlink for streaming-breakpoint sizing when the caller gives none —
#: a conservative "typical broadband" figure that also covers good 4G.
DEFAULT_BANDWIDTH_MBPS = 25.0

#: First rung for :func:`capped_stream_cuts` when the caller gives none. Small
#: enough to land in a single zarr chunk, so time-to-first-pixel on an eager
#: coarsest level is one range request.
DEFAULT_CAPPED_FIRST_CHUNK = 2_000

#: Ceiling on ONE additive increment, for :func:`capped_stream_cuts`. This is
#: the number that makes a multi-million-element leaf streamable: no single
#: commit may block the main thread, whatever the level is worth in total. Set
#: below the 1,000,000 that ``scripts/check_demo_ladders.py`` fails a level at
#: (``DEFAULT_MAX_LEVEL_ELEMENTS``), with margin.
DEFAULT_MAX_ADDITIVE_COMMIT = 900_000

#: Deepest default ladder for a sliced node: rung 0 carries at least 1/8 of
#: the node, hence the same share of every resident slice under uniform mixing.
#: See ``luxar.demos._lod_policy.SLICED_LADDER_MAX_DEPTH`` for the measured
#: viewer-gate rationale and the non-uniform-slice caveat behind this value.
DEFAULT_SLICED_LADDER_MAX_DEPTH = 8

#: Hard cap on the number of levels a ``stream:<c>`` ladder may produce. The
#: geometric doubling schedule gives ~log2(N/c) levels, so 16 covers c·2^15
#: splats (≈ 460 M at c=14 k) — far beyond realistic leaves. On hitting the cap
#: the last cut jumps straight to N.
DEFAULT_STREAM_MAX_LEVELS = 16


def streaming_chunk_splats(
    target_ms: float,
    bandwidth_mbps: float,
    bytes_per_splat: float,
) -> int:
    """Element count whose download takes ``target_ms`` at ``bandwidth_mbps``.

    Pure sizing math for the ``stream:<c>`` breakpoint spec:
    ``bandwidth_mbps × 125_000 B/s/Mbps × target_ms/1000 ÷ bytes_per_splat``.
    E.g. 200 ms @ 25 Mbps @ 45 B/splat → ~13.9 k splats.
    """
    if target_ms <= 0:
        raise ValueError(f"target_ms must be positive; got {target_ms}")
    if bandwidth_mbps <= 0:
        raise ValueError(f"bandwidth_mbps must be positive; got {bandwidth_mbps}")
    if bytes_per_splat <= 0:
        raise ValueError(f"bytes_per_splat must be positive; got {bytes_per_splat}")
    return max(
        1, round(bandwidth_mbps * 125_000.0 * (target_ms / 1000.0) / bytes_per_splat)
    )


def scaled_streaming_chunk(
    first_chunk: int, *, slice_count: int = 1, part_count: int = 1
) -> int:
    """Scale a whole-node first chunk to the viewer's resident slice.

    A sliced node exposes one of ``slice_count`` hidden coordinates at a time,
    while every child of a partition is drawn. Since ``stream:<c>`` is expanded
    independently per part, each part receives ``c`` elements. The per-part
    chunk is therefore ``first_chunk * slice_count / part_count``.
    """
    if first_chunk < 1:
        raise ValueError(f"first_chunk must be >= 1; got {first_chunk}")
    if slice_count < 1:
        raise ValueError(f"slice_count must be >= 1; got {slice_count}")
    if part_count < 1:
        raise ValueError(f"part_count must be >= 1; got {part_count}")
    return max(1, round(first_chunk * slice_count / part_count))


def sliced_ladder_first_chunk(
    first_chunk: int,
    *,
    elements: int,
    slices: int,
    max_depth: int = DEFAULT_SLICED_LADDER_MAX_DEPTH,
) -> int:
    """Floor a sliced node's first chunk at a useful resident share.

    ``slices`` is a predicate, not a divisor: requiring
    ``first_chunk / slices >= elements / (max_depth * slices)`` cancels the
    slice count, so any value above 1 applies the same whole-node share floor.
    See ``luxar.demos._lod_policy.stream_ladder`` for the measured rationale and
    the non-uniform-slice caveat behind that contract.
    """
    if first_chunk < 1:
        raise ValueError(f"first_chunk must be >= 1; got {first_chunk}")
    if elements < 1:
        raise ValueError(f"elements must be >= 1; got {elements}")
    if slices < 1:
        raise ValueError(f"slices must be >= 1; got {slices}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1; got {max_depth}")
    if slices == 1:
        return first_chunk
    return max(first_chunk, -(-elements // max_depth))


def hidden_coordinate_count(positions: Any, hidden_cols: Sequence[int]) -> int:
    """How many distinct hidden coordinates a node's elements actually occupy.

    The divisor a sliced node's first rung is spread over: the viewer shows one
    hidden coordinate at a time, so a rung sized against the whole node arrives
    divided by this (#2374/#2376).

    Counts distinct OCCURRING COMBINATIONS across all hidden columns, not the
    product of each column's cardinality. A node stacked on time *and* channel is
    sliced only by the pairs that occur, and on sparse data the product
    overstates badly — measured on ``biodiversity_planetary_scale``, the product
    of two axes gives 140 against 126 actually populated. Nor is it the declared
    ``Dimension`` range/step: ``drosophila_embryogenesis`` declares 500
    timepoints and its coarsest rung carries data at 499, and a coordinate with
    no elements costs no bytes and divides no rung.

    Returns at least 1, so the result is always safe as a divisor or multiplier.
    """
    import numpy as np

    cols = [int(c) for c in hidden_cols]
    if not cols:
        return 1
    arr = np.asarray(positions)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return 1
    if any(not 0 <= c < arr.shape[1] for c in cols):
        return 1
    unique = (
        np.unique(arr[:, cols[0]])
        if len(cols) == 1
        else np.unique(arr[:, cols], axis=0)
    )
    return max(1, len(unique))


def parse_stream_chunk(spec: str) -> int:
    """Extract ``c`` from a ``"stream:<c>"`` spec, validating it is >= 1."""
    body = spec[len("stream:") :]
    try:
        c = int(body)
    except ValueError as e:
        raise ValueError(
            f"stream breakpoints must be 'stream:<c>' with integer c >= 1; got {spec!r}"
        ) from e
    if c < 1:
        raise ValueError(f"stream first-chunk size must be >= 1; got {c}")
    return c


def parse_equi_energy_rungs(spec: str) -> int:
    """Extract ``n`` from an ``"equi-energy:<n>"`` spec, validating it is >= 1."""
    body = spec[len(EQUI_ENERGY_PREFIX) :]
    try:
        n = int(body)
    except ValueError as e:
        raise ValueError(
            "equi-energy breakpoints must be 'equi-energy:<n>' with integer "
            f"n >= 1; got {spec!r}"
        ) from e
    if n < 1:
        raise ValueError(f"equi-energy rung count must be >= 1; got {n}")
    return n


def _cumulative_energy(energy_in_order: Sequence[float]) -> List[float] | None:
    """Cumulative non-negative energy, or ``None`` when degenerate (non-finite
    or summing to nothing) so the caller falls back to equal-count cuts."""
    cum = 0.0
    out: List[float] = []
    for e in energy_in_order:
        v = float(e)
        if not math.isfinite(v):
            return None
        cum += max(v, 0.0)
        out.append(cum)
    if not out or out[-1] <= 0.0:
        return None
    return out


def _first_index_reaching(cum: Sequence[float], target: float) -> int:
    """Smallest index whose cumulative value reaches ``target`` (binary search)."""
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _equal_energy_cuts(cum_energy: Sequence[float], n_rungs: int) -> List[int]:
    """Cumulative counts at which the cumulative energy first reaches ``k/n``.

    Cuts that coincide (one element carrying several shares) collapse, so the
    result can be shorter than ``n_rungs``; it always ends at ``len(cum)``.
    """
    n = len(cum_energy)
    total = cum_energy[-1]
    cuts: List[int] = []
    for k in range(1, n_rungs + 1):
        cut = _first_index_reaching(cum_energy, total * k / n_rungs) + 1
        if not cuts or cut > cuts[-1]:
            cuts.append(cut)
    if cuts[-1] < n:
        cuts.append(n)
    return cuts


def _equal_count_cuts(n: int, n_rungs: int) -> List[int]:
    raw = sorted({round((k + 1) * n / n_rungs) for k in range(n_rungs)})
    cuts = [c for c in raw if 0 < c < n]
    cuts.append(n)
    return cuts


def _split_by_commit_cap(
    cuts: Sequence[int], weights: Sequence[int] | None, max_commit: int
) -> List[int]:
    """Split every increment whose payload exceeds ``max_commit``.

    A greedy walk: elements are accumulated until adding the next one would
    exceed the cap, then a cut is placed. Every emitted increment therefore
    weighs at most ``max_commit`` — except an element that alone outweighs the
    cap, which becomes a rung of its own (it cannot be split).
    """
    out: List[int] = []
    prev = 0
    for end in cuts:
        acc = 0
        for i in range(prev, end):
            w = 1 if weights is None else int(weights[i])
            if acc > 0 and acc + w > max_commit:
                out.append(i)
                acc = 0
            acc += w
        if not out or end > out[-1]:
            out.append(end)
        prev = end
    return out


def equi_energy_cuts(
    energy_in_order: Sequence[float],
    n_rungs: int,
    *,
    weights: Sequence[int] | None = None,
    max_commit: int = DEFAULT_MAX_ADDITIVE_COMMIT,
) -> List[int]:
    """Cumulative cuts at equal shares of cumulative ENERGY, then commit-capped.

    The rung boundary ``k`` (``k = 1..n_rungs``) is the first element at which
    the cumulative energy along the ladder's OWN ordering reaches ``k/n_rungs``
    of the total. On a contribution-first ordering (``self_energy`` for splats,
    ``salience_kind="energy"`` for points/lines) this makes the first rungs FEW
    elements but the perceptually heaviest ones, and the late rungs progressively
    fatter in count — each carries the same light, so the elements that take
    longest to arrive are exactly the ones whose absence is least visible. An
    equal-count ladder does the opposite: its first rung is ``N/n`` elements
    whatever they are worth.

    The equal-energy tail is exactly where a heavy-tailed dataset puts most of
    its ELEMENTS, so a late rung can be millions of them. Any increment whose
    payload exceeds ``max_commit`` is therefore split into capped steps —
    the same ceiling :func:`capped_stream_cuts` respects, so no single commit
    blocks the main thread. ``weights`` is the per-element payload used for that
    cap (default 1 each; Lines pass vertices-per-polyline so the cap is in the
    vertex currency, as for ``stream:``). The split rungs share one energy
    band, so their ``e(k)`` stamps rise less per rung than the equal-energy
    ones — expected, and honest.

    Shares that coincide on one element collapse into one cut (a single element
    carrying half the energy yields no empty rung between), and degenerate
    energy (all zero, non-finite) falls back to equal-count cuts — the same
    fallback the ``energy:`` fraction resolvers use — so a ladder is always
    produced.

    Args:
        energy_in_order: Per-element non-negative energy, ALREADY permuted into
            the ladder order (element 0 paints first).
        n_rungs: Number of equal-energy rungs requested (before cap splitting).
        weights: Optional per-element payload sizes, same order, for the cap.
        max_commit: Ceiling on any single increment, in ``weights`` units.

    Returns:
        Strictly increasing cumulative cuts ending at ``len(energy_in_order)``.

    Raises:
        ValueError: ``n_rungs`` or ``max_commit`` is not positive, or ``weights``
            has the wrong length.
    """
    n = len(energy_in_order)
    if n_rungs < 1:
        raise ValueError(f"n_rungs must be >= 1; got {n_rungs}")
    if max_commit < 1:
        raise ValueError(f"max_commit must be >= 1; got {max_commit}")
    if weights is not None and len(weights) != n:
        raise ValueError(
            f"weights has {len(weights)} entries but energy has {n}; they must match"
        )
    if n <= 1:
        return [n]
    cum = _cumulative_energy(energy_in_order)
    cuts = (
        _equal_count_cuts(n, n_rungs)
        if cum is None
        else _equal_energy_cuts(cum, n_rungs)
    )
    return _split_by_commit_cap(cuts, weights, max_commit)


def validate_element_breakpoints(spec: BreakpointSpec) -> None:
    """Size-independent validation of a Points/Lines ``counts``/``breakpoints``
    value, meant for RESOLVE time — before any group is written to disk.

    Checks exactly the properties that doom the spec for EVERY element count
    ``n`` (the write path still validates against the actual ``n``): the string
    vocabulary (``stream:<c>`` / ``equi-energy:<n>`` / ``energy:<fractions>``),
    the stream first-chunk size, the equi-energy rung count, the energy
    fractions parsing and non-emptiness, and list
    non-emptiness. Raises :class:`ValueError`, reusing the write path's messages
    where it has one, so failing earlier never changes what a caller must catch.
    """
    if isinstance(spec, str):
        if spec.startswith("stream:"):
            parse_stream_chunk(spec)
            return
        if spec.startswith(EQUI_ENERGY_PREFIX):
            parse_equi_energy_rungs(spec)
            return
        if not spec.startswith("energy:"):
            raise ValueError(
                f"unrecognized breakpoints string {spec!r}; expected "
                "'energy:<fractions>' (e.g. 'energy:0.5,0.9,1.0'), "
                "'equi-energy:<n>' (e.g. 'equi-energy:4') or "
                "'stream:<c>' (e.g. 'stream:40000')"
            )
        try:
            fracs = [float(s) for s in spec[len("energy:") :].split(",") if s.strip()]
        except ValueError as e:
            raise ValueError(f"energy: fractions must be numbers; got {spec!r}") from e
        if not fracs:
            raise ValueError("energy: fractions must be non-empty")
        return
    if len(spec) == 0:
        raise ValueError("counts must be a non-empty list of integers")


def stream_cuts(
    n: int, chunk: int, max_levels: int = DEFAULT_STREAM_MAX_LEVELS
) -> List[int]:
    """Cumulative cuts for a geometric streaming ladder over ``n`` elements.

    Returns ``[c, 2c, 4c, …, n]``, so increments are ``[c, c, 2c, …]`` — first
    paint costs ``chunk`` elements, then each refinement doubles the resident
    set. Silently clamps rather than raising on small ``n`` (deliberate: per-part
    N is unknowable to the caller writing the spec). A final increment smaller
    than ``chunk/2`` folds into the previous cut so no sliver level is emitted.
    """
    if n <= chunk:
        return [n]
    cuts: List[int] = []
    cum = chunk
    while cum < n and len(cuts) < max_levels - 1:
        cuts.append(cum)
        cum *= 2
    if cuts and (n - cuts[-1]) < chunk / 2:
        cuts.pop()
    cuts.append(n)
    return cuts


def capped_stream_cuts(
    n: int,
    chunk: int = DEFAULT_CAPPED_FIRST_CHUNK,
    max_commit: int = DEFAULT_MAX_ADDITIVE_COMMIT,
) -> List[int]:
    """Cumulative cuts that double early, then step by a fixed cap.

    :func:`stream_cuts` doubles all the way to ``n``, so its final increment is
    ``n`` minus the largest doubling below it — a quantity that grows with ``n``
    and approaches ``n/2`` at worst, **whatever the first chunk is**. Measured at
    ``chunk=2_000``: a 6,248,730-element leaf commits 2,152,730 in one go, twice
    the 1,000,000 at which ``scripts/check_demo_ladders.py`` fails a level. That
    is the shape behind #1812, where a demo was diagnosed as needing a row cap
    and lost 87% of its catalogue: capping ``n`` shrank the last commit without
    changing its geometry.

    This keeps the geometric ramp, which is what makes first paint cheap, but
    stops before the next doubling would exceed ``max_commit`` and finishes in
    equal steps of that size. The largest commit is therefore ``max_commit`` at
    **any** ``n``; with the defaults the geometric head totals 1,024,000.

    One list serves every level, part and leaf of a tree, because
    ``_validate_counts`` clamps a cumulative list to the node's own ``n`` and
    stops there — so a small coarse level simply takes the geometric head while
    the finest takes the whole schedule. That is also why this returns a list
    rather than adding a ``"capped-stream:<c>:<max>"`` spec string: the clamping
    already gives per-node adaptation, which is the only thing a string form
    would buy.

    Unlike :func:`stream_cuts`, which returns ``[0]`` for an empty input so a
    downstream builder always receives one cut, this public explicit-count
    helper rejects ``n < 1`` because its result must be strictly increasing and
    directly valid as ``counts=``.

    Args:
        n: Element count of the node the ladder is for, in its own payload
            currency (points, vertices, splats).
        chunk: First rung, i.e. the time-to-first-pixel payload.
        max_commit: Ceiling on any single increment.

    Returns:
        Strictly increasing cumulative cuts ending at ``n``.

    Raises:
        ValueError: ``n``, ``chunk``, or ``max_commit`` is not positive.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1; got {chunk}")
    if max_commit < 1:
        raise ValueError(f"max_commit must be >= 1; got {max_commit}")
    chunk = min(chunk, max_commit)
    if n <= chunk:
        return [n]
    cuts: List[int] = []
    cum = chunk
    # Geometric head: double while the NEXT increment still fits the cap. The
    # bound is 2*max_commit because the increment arriving at `cum` is cum/2.
    while cum < n and cum <= 2 * max_commit:
        cuts.append(cum)
        cum *= 2
    cum = cuts[-1] if cuts else 0
    while cum + max_commit < n:
        cum += max_commit
        cuts.append(cum)
    cuts.append(n)
    return cuts


def sibling_aware_stream_breakpoints(
    breakpoints: BreakpointSpec,
    leaf_n: int,
    compression_factor: int,
) -> BreakpointSpec:
    """Raise a ``stream:C`` ladder's first chunk for a leaf that has a
    COARSER SIBLING in its lod group.

    Measured pathology (h2afva vrefit, 23.4M splats): with every level's
    geometric ladder starting at the SAME small base chunk, the point where a
    finer level's committed content catches up with its coarser sibling —
    whether by count, energy, or measured L² quality — structurally lands
    ``log2(sibling_total / base)`` sequential network passes into the ladder,
    i.e. always 2-3 chunks from the END. Upgrades therefore feel like
    "waits until fully loaded".

    Fix the geometry instead of the currency: a leaf whose group contains a
    coarser sibling starts its ladder at ``ceil(leaf_n / (2·K))`` — half the
    sibling's expected size — so the catch-up fires at chunk 1-2 by
    construction (energy-ordered first chunks of that size carry ~70%+ of the
    leaf's energy on real data, comfortably past the viewer's committed-energy
    switch threshold). The user's ``stream:C`` base still applies wherever it
    is LARGER, and — crucially — the group's COARSEST leaf must NOT go through
    this helper: it is the eager default level whose small first chunk is the
    fast-first-paint path.

    ``leaf_n`` is in the level's own payload currency (splats for GSplats,
    points for Points, vertices for Lines); the ratio is what matters, and every
    lift is linear in that currency, so one helper serves all three.

    Non-``stream:`` specs (equal-count, explicit counts, energy fractions)
    pass through untouched — their chunk structure has no shared-base
    pathology (e.g. equal-count crosses the sibling at chunk 1 already).
    """
    if not (isinstance(breakpoints, str) and breakpoints.startswith("stream:")):
        return breakpoints
    try:
        user_base = int(breakpoints[len("stream:") :])
    except ValueError:
        return breakpoints
    sibling_base = math.ceil(leaf_n / (2.0 * max(2, compression_factor)))
    return f"stream:{max(user_base, sibling_base)}"
