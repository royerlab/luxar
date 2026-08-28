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

Doubling has one failure mode, and it is size-driven rather than spec-driven: the
FINAL increment of a pure doubling ladder is ``N`` minus the largest doubling
below it, so it grows with ``N`` and approaches ``N/2`` in the worst case however
small ``c`` is. Past a few million elements that last commit blocks the main
thread, so a large leaf wants :func:`capped_stream_cuts` — the same geometric
head, then equal steps of a fixed ceiling — instead of :func:`stream_cuts`.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Union

#: Breakpoint specification for an additive ladder. Either a string form
#: (``"equal-count"``, ``"stream:<c>"``, ``"energy:<fractions>"`` for
#: Points/Lines) or an explicit cumulative count / fraction sequence.
BreakpointSpec = Union[str, Sequence[int], Sequence[float]]

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


def validate_element_breakpoints(spec: BreakpointSpec) -> None:
    """Size-independent validation of a Points/Lines ``counts``/``breakpoints``
    value, meant for RESOLVE time — before any group is written to disk.

    Checks exactly the properties that doom the spec for EVERY element count
    ``n`` (the write path still validates against the actual ``n``): the string
    vocabulary (``stream:<c>`` / ``energy:<fractions>``), the stream first-chunk
    size, the energy fractions parsing and non-emptiness, and list
    non-emptiness. Raises :class:`ValueError`, reusing the write path's messages
    where it has one, so failing earlier never changes what a caller must catch.
    """
    if isinstance(spec, str):
        if spec.startswith("stream:"):
            parse_stream_chunk(spec)
            return
        if not spec.startswith("energy:"):
            raise ValueError(
                f"unrecognized breakpoints string {spec!r}; expected "
                "'energy:<fractions>' (e.g. 'energy:0.5,0.9,1.0') or "
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

    Args:
        n: Element count of the node the ladder is for, in its own payload
            currency (points, vertices, splats).
        chunk: First rung, i.e. the time-to-first-pixel payload.
        max_commit: Ceiling on any single increment.

    Returns:
        Strictly increasing cumulative cuts ending at ``n``.

    Raises:
        ValueError: ``chunk`` or ``max_commit`` is not positive.
    """
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1; got {chunk}")
    if max_commit < 1:
        raise ValueError(f"max_commit must be >= 1; got {max_commit}")
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
