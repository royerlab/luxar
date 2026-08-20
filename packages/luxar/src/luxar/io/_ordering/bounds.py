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
# step-scaled (0.25 x step), so for pathological discrete steps below ~1.3e-3
# the pad reaches past the neighbour category's quarter-step boundary and the
# over-fetch returns. Step metadata is not plumbed into these bound
# builders; discrete/categorical dims with milli-scale steps are not a
# supported layout (rescale the axis instead).
#
# KNOWN LIMIT (large coordinates): the outward float32 store below never lets a
# pad vanish, so the pad a barrier axis EFFECTIVELY gets is
# ``max(_BARRIER_BOUND_EPS, one float32 ULP at |x|)`` — 1e-3 near the origin, but
# 1.0 at |x| = 2**23, 2.0 at 2e7, 8.0 at 1e8. Integers stay exactly float32
# representable through 2**24, so a unit-step categorical axis with large
# absolute values (a millisecond timestamp, an acquisition index offset into an
# experiment) is a legitimate layout there and will over-fetch a whole
# neighbouring category. That is the deliberate trade — over-fetching a
# neighbour beats dropping the chunk at its own category value — but re-base
# such an axis near the origin if the extra traffic matters.
_BARRIER_BOUND_EPS = 1e-3


def _normalise_slice_dims(slice_dims: Optional[Sequence[int]], ndim: int) -> set[int]:
    """Coerce a ``slice_dims`` argument to a validated set of column indices.

    One sanitiser for all four bound builders, which otherwise diverged three
    ways behind this shared module (a range filter that silently ignored a bad
    index, an ``int()`` coercion that raised ``IndexError`` deep in the loop, and
    a plain membership test that ignored negatives too).

    An out-of-range index is an ERROR rather than an ignored entry: it means a
    categorical axis silently loses its barrier treatment and gets the geometric
    extent expansion instead, which bleeds a chunk into its neighbour's
    category — a wrong answer with nothing to notice. No in-tree caller can
    produce one (they all build the list by ``enumerate``-ing the actual dims),
    so this only fires on a genuinely new bug.

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
    """
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
    comparison and the conditional ``nextafter`` step are four short NumPy passes
    over d elements instead of a Python call per dimension. This is only about
    the outward-store BRANCHING, and is NOT a reason to vectorise the min/max
    REDUCE that produces ``lo``/``hi`` as well: both lines builders reduce per
    dimension on purpose, because NumPy's outer-axis reduce over a 3-or-4-element
    inner row is several times slower than one reduce per column (measured 5-9x
    end-to-end at 1M elements and repo-default chunk sizes, bitwise-identical
    output either way). ``compute_chunk_bounds_gsplats`` keeps its ``min(axis=0)``
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
