"""Shared chunk-bounds helpers for the per-geometry ordering modules."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Padding added to barrier/discrete-dimension chunk bounds. This is ONLY a
# float-boundary safety margin — the query "reach" (how far a slice query
# selects around a category) lives entirely in the reader's per-dimension
# tolerance (see luxar-viewer tolerance-computer.ts, discrete = 0.25 × step).
# It used to be 0.5 (half a step); combined with the reader's own half-step
# tolerance that summed to a full step and made a single-category query (e.g.
# one timepoint) pull in the entire neighbouring category. Keep this tiny.
#
# KNOWN LIMIT (small steps): the pad is absolute while the reader's reach is
# step-scaled (0.25 x step), so a chunk at category `c` is pulled into a query
# for `c + 1` as soon as `c + pad >= (c + step) - 0.25*step`, i.e. as soon as
#   pad >= 0.75 * step   <=>   step <= pad / 0.75.
# At the bare epsilon that is 1e-3 / 0.75 = ~1.3e-3: for pathological discrete
# steps below that the over-fetch returns. Step metadata is not plumbed into
# these bound builders; discrete/categorical dims with milli-scale steps are
# not a supported layout (rescale the axis instead).
#   Since the quantisation pad (#1655) the threshold is worse on a NON-GRIDDED
#   barrier axis, because the compiler adds `coord_slack` on top of this
#   epsilon: the effective pad is `1e-3 + extent/131070`, i.e. 8.63e-3 at an
#   axis extent of 1000 — 8.6x wider, so by the SAME `pad / 0.75` rule steps
#   below 8.63e-3 / 0.75 = ~1.2e-2 over-fetch there. An ordinary stacked
#   integer time/channel axis is unaffected: it is GRIDDED, its slack is
#   exactly 0, and it keeps the plain 1e-3.
#
# KNOWN LIMIT (large coordinates): the outward float32 store below never lets a
# pad vanish, so the pad a barrier axis EFFECTIVELY gets is
# ``max(_BARRIER_BOUND_EPS, up to one float32 ULP at |x|)`` — 1e-3 near the
# origin, but 2.0 at 2e7 and 8.0 at 1e8. The two ends are asymmetric at an exact
# power of two, where the ULP below the binade boundary is half the one above:
# at |x| = 2**23 the low end moves 0.5 and the high end 1.0. Integers stay
# exactly float32 representable through 2**24, so a unit-step categorical axis
# with large absolute values (a millisecond timestamp, an acquisition index offset into an
# experiment) is a legitimate layout there and will over-fetch a whole
# neighbouring category. That is the deliberate trade — over-fetching a
# neighbour beats dropping the chunk at its own category value — but re-base
# such an axis near the origin if the extra traffic matters.
_BARRIER_BOUND_EPS = 1e-3


def _normalise_slice_dims(slice_dims: Optional[Sequence[int]], ndim: int) -> set[int]:
    """Coerce a ``slice_dims`` argument to a validated set of column indices.

    One sanitiser for all four bound builders, which otherwise diverged two ways
    behind this shared module. ``gsplats.py`` coerced with
    ``set(int(d) for d in slice_dims)`` and then INDEXED ``mins[d]``, so a
    positive out-of-range index blew up with ``IndexError`` deep in the loop;
    ``points.py`` and both ``lines.py`` builders used a membership test on both
    sides, so out-of-range entries were ignored outright.

    A POSITIVE out-of-range index is an ERROR rather than an ignored entry: it
    means a categorical axis silently loses its barrier treatment and gets the
    geometric extent expansion instead, which bleeds a chunk into its
    neighbour's category — a wrong answer with nothing to notice.

    A NEGATIVE index is rejected too, even though Python resolves it. It was
    NOT a bounds bug: the old gsplats builder σ-expanded the last column and then
    overwrote ``mins[-1]``/``maxs[-1]`` with the tight barrier interval, so its
    output for ``[-1]`` was bitwise identical to ``[ndim - 1]``. It is rejected
    because nothing ELSE in the pipeline resolves it that way:

    * ``_compound_sort`` puts the last column in BOTH sets. ``ordering_dims``
      complements over ``range(ndim)``, which never contains ``-1``, so the
      categorical axis is lexsorted as a barrier AND Morton/Hilbert-coded —
      wasted curve bits, and emitted metadata whose ``ordering_dims ∪
      slice_dims`` does not match ``ndim`` — malformed enough that the viewer's
      points chunk-index loader logs a "Dimension coverage mismatch" for that
      exact shape (``data/points/chunk-index-loader.ts``; the gsplats loader has
      no such check, so on gsplats it passes unremarked).
    * The raw ``-1`` is persisted into the store's ``slice_dims`` attr, and the
      viewer's ``readBarrierDims``
      (``data/gsplats/gsplats-spatial-index-loader.ts``) rejects the attr
      WHOLESALE on any entry outside ``[0, ndim)``, falling back to the scene
      dimensions' ``discrete`` flags
      (``data/loaders/spatial-query/tolerance-computer.ts``) — so the node
      silently loses its authoritative barrier classification.

    No in-tree caller can produce a bad index today (they either ``enumerate``
    the actual dims or complement over ``range(ndim)``; the batch merge
    orchestrator derives its single entry from the manifest's own axis count),
    but ``GSplatData.save(barrier_dims=...)`` is public API a user can drive
    directly, so this is a real door rather than dead code.

    Args:
        slice_dims: Barrier/discrete column indices, or ``None``
        ndim: Number of columns in the coordinate array

    Returns:
        The indices as a ``set`` of plain ``int``.

    Raises:
        ValueError: If an entry is outside ``[0, ndim)``.
    """
    if slice_dims is None:
        return set()
    out: set[int] = set()
    for d in slice_dims:
        idx = int(d)
        if not 0 <= idx < ndim:
            raise ValueError(
                f"slice_dims entry {d!r} is out of range for {ndim}-dimensional "
                f"data (valid indices are 0..{ndim - 1}). A barrier index the "
                "bound builders cannot resolve would silently cost that "
                "categorical axis its tight bounds and let a chunk bleed into "
                "the neighbouring category."
            )
        out.add(idx)
    return out


def _normalise_coord_slack(coord_slack: Optional[np.ndarray], ndim: int) -> np.ndarray:
    """Coerce a ``coord_slack`` argument to a validated ``(ndim,)`` float64 pad.

    ``coord_slack`` is the per-axis distance the STORE can move a coordinate
    away from the value the bound builder was handed — under the default AUTO
    encoding the coordinates are written as per-axis uint16 fixed point, so a
    decoded coordinate can sit up to half a quantum outside a bound computed
    from the authored one, and the reader then never fetches that chunk
    (issue #1655). The compiler asks the encoder for it
    (:meth:`~luxar.encoding.encoder.ArrayEncoder.coordinate_round_trip_slack`)
    and hands it to the builders; ``None`` means "no displacement", i.e. zeros.

    Validated here rather than in each builder for the same reason as
    :func:`_normalise_slice_dims`: three builders take this argument and must
    not diverge on what they accept. A wrong-LENGTH array is an error rather
    than a broadcast, because the entries are positional per-axis quantities —
    silently padding the wrong axis is exactly the failure this parameter
    exists to prevent. A NEGATIVE entry would TIGHTEN the bound and drop
    geometry; a non-finite one would poison every bound on that axis.

    The ``dtype=np.float64`` coercion below accepts float32 (and a plain list)
    rather than rejecting it the way :func:`_store_outward_f32_array` rejects a
    float32 interval. The asymmetry is deliberate and is documented because it
    otherwise reads as an inconsistency worth "fixing": that helper refuses
    because a pad already lost to float32 ARITHMETIC cannot be recovered by
    rounding outward, whereas this is a per-axis value a caller merely stored
    narrowly, and it is added into a float64 accumulator before any store. No
    in-tree caller needs it — the encoder's predicate returns float64 — but the
    three builders are public through ``luxar.io.ordering``, so this is the
    direct-caller door, pinned by
    ``io/tests/test_ordering_properties.py::test_normalise_coord_slack_accepts_none_and_float32``.

    Args:
        coord_slack: Per-axis outward pad, shape ``(ndim,)``, or ``None``
        ndim: Number of columns in the coordinate array

    Returns:
        The pad as a ``(ndim,)`` float64 array; all zeros when ``None``.

    Raises:
        ValueError: If the length is wrong, or an entry is negative or
            non-finite.
    """
    if coord_slack is None:
        return np.zeros(ndim, dtype=np.float64)
    slack = np.asarray(coord_slack, dtype=np.float64)
    if slack.shape != (ndim,):
        raise ValueError(
            f"coord_slack has shape {slack.shape} but the data is "
            f"{ndim}-dimensional (expected ({ndim},)). These are positional "
            "per-axis pads, so a length mismatch would pad the wrong axis and "
            "leave the one that needed it short — the exact under-fetch this "
            "argument exists to prevent."
        )
    if not np.all(np.isfinite(slack)):
        raise ValueError(
            f"coord_slack must be finite (got {coord_slack!r}): a non-finite "
            "pad poisons every chunk bound on that axis."
        )
    if np.any(slack < 0.0):
        raise ValueError(
            f"coord_slack must be non-negative (got {coord_slack!r}): a "
            "negative pad TIGHTENS the stored bound, which drops geometry the "
            "reader can no longer find."
        )
    return slack


def _normalise_scalar_slack(scalar_slack: Optional[float]) -> float:
    """Validate one non-negative finite footprint pad; ``None`` means zero."""
    if scalar_slack is None:
        return 0.0
    slack = float(scalar_slack)
    if not np.isfinite(slack):
        raise ValueError(f"scalar_slack must be finite (got {scalar_slack!r})")
    if slack < 0.0:
        raise ValueError(f"scalar_slack must be non-negative (got {scalar_slack!r})")
    return slack


def _store_outward_f32(lo: float, hi: float) -> tuple[np.float32, np.float32]:
    """Narrow a float64 interval to float32 OUTWARD (``lo`` down, ``hi`` up).

    ``chunk_bounds`` is a float32 array, but every pad the geometry builders add
    is a small ABSOLUTE quantity — a point radius (``DEFAULT_POINT_RADIUS`` or an
    authored one), a gsplat's ``coverage_sigma·σ`` extent, a line's endpoint
    width, ``_BARRIER_BOUND_EPS`` on a categorical axis — while the coordinate it
    is added to can be large. Past ``|x| ~ 2**23`` such a pad is under half a
    float32 ULP, so a round-to-nearest store throws it away entirely and the
    stored bound is TIGHTER than the footprint the renderer draws — precisely
    what the pad exists to prevent. The hole is not new and is not specific to
    one geometry: the removed scale-relative fudge vanished the same way whenever
    1% of a chunk's own range fell under half an ULP (0.1 against a half-ULP of
    1.0 at ``|x| = 2e7``, say), and so does a σ of 0.1 or a unit line width.
    Stepping one ULP outward whenever the cast moved a bound the wrong way closes
    it for every path at once: if a pad ``r`` was lost to rounding then ``r`` was
    below half an ULP, so one ULP outward is strictly more than ``r``.

    Callers must accumulate the interval in float64 first — an extent already
    lost to float32 arithmetic before the store is invisible here.

    ``over='ignore'``: past the float32 ceiling the cast overflows to ``±inf``,
    which is the RIGHT answer for a bound (an infinite bound over-fetches, it
    never drops a chunk), but the ``RuntimeWarning`` numpy raises for it is fatal
    under the ``-W error`` blocks several io tests wrap whole compiles in.
    """
    with np.errstate(over="ignore"):
        lo32 = np.float32(lo)
        if float(lo32) > lo:
            lo32 = np.nextafter(lo32, np.float32(-np.inf))
        hi32 = np.float32(hi)
        if float(hi32) < hi:
            hi32 = np.nextafter(hi32, np.float32(np.inf))
    return lo32, hi32


def _store_outward_f32_array(
    lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised :func:`_store_outward_f32` over a whole chunk's dimensions.

    Same rule, applied to an ALREADY-REDUCED ``(d,)`` vector: the cast, the
    comparison and the (unconditionally evaluated, then selected) ``nextafter``
    step are a handful of short NumPy passes over d elements instead of a Python
    call per dimension — measured faster from d=3 up (11.9 µs vs 16.2 µs at d=3,
    13.4 vs 68.9 at d=16). This is only about
    the outward-store BRANCHING, and is NOT a reason to vectorise the min/max
    REDUCE that produces ``lo``/``hi`` as well: both lines builders reduce per
    dimension on purpose, because NumPy's outer-axis reduce over a 3-or-4-element
    inner row is several times slower than one reduce per column (measured
    end-to-end at 1M vertices/segments and repo-default chunk sizes: ~4-6x for
    ``compute_vertex_chunk_bounds``, ~2.5-3.5x for
    ``compute_segment_chunk_bounds``, whose per-chunk gather dominates more of
    the work; bitwise-identical output either way).
    ``compute_chunk_bounds_gsplats`` keeps its ``min(axis=0)``
    only because it materialises the padded ``(n, d)`` array anyway.

    Args:
        lo: Per-dimension lower bounds, float64, shape ``(d,)``
        hi: Per-dimension upper bounds, float64, shape ``(d,)``

    Returns:
        The same intervals as float32, each end rounded away from the interval.

    Raises:
        TypeError: If either input is not float64. Accepting float32 here would
            silently defeat the whole point — a pad already rounded away by
            float32 arithmetic upstream cannot be recovered by rounding outward,
            and the resulting under-tight bound only shows up in tests that
            probe large coordinate magnitudes.
    """
    if lo.dtype != np.float64 or hi.dtype != np.float64:
        raise TypeError(
            "_store_outward_f32_array requires float64 bounds (got "
            f"lo={lo.dtype}, hi={hi.dtype}): accumulate the interval in float64 "
            "first, or the pad this helper exists to preserve is already gone."
        )
    # over='ignore' for the same reason as the scalar form: past the float32
    # ceiling the cast overflows to ±inf, which is the right answer for a bound
    # (over-fetch, never a dropped chunk), but its RuntimeWarning is fatal under
    # the ``-W error`` blocks several io tests wrap whole compiles in.
    with np.errstate(over="ignore"):
        lo32 = lo.astype(np.float32)
        hi32 = hi.astype(np.float32)
        lo32 = np.where(
            lo32.astype(np.float64) > lo,
            np.nextafter(lo32, np.float32(-np.inf)),
            lo32,
        )
        hi32 = np.where(
            hi32.astype(np.float64) < hi,
            np.nextafter(hi32, np.float32(np.inf)),
            hi32,
        )
    return lo32, hi32
