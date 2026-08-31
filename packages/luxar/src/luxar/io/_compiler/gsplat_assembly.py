"""GSplat assembly: validation, spatial ordering, array writes, group attrs.

Private support module shared by the scene compiler
(:class:`~luxar.io.compiler.LuxarZarrCompiler`) and the root-agnostic node-tree
writer (:mod:`luxar.io._compiler.gsplat_tree`). Holds the gsplat-specific pipeline
both sequence: validate inputs, (optionally) spatially order, write the zarr
arrays, then stamp the group attributes. Because both paths call these same four
free functions, a scene leaf is byte-identical to a standalone one.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import zarr
from arbol import aprint
from numpy.typing import NDArray

from luxar._zarr_compat import create_array

from ...core.dimensions import Dimensions
from ...core.group.compositing import (
    IDENTITY_COMPOSITING_ATTRS,
    WRITER_STAMPED_APPEARANCE_DEFAULTS,
)
from ...encoding import (
    COORDINATE_LEVELS,
    ArrayEncoder,
    EncodingMode,
    SemanticType,
    gridded_axis_step,
)
from ...encoding.compression import resolve_compressor
from ...typing_utils.constants import (
    COORDINATE_U16_MAX_EXTENT,
    DEFAULT_TRUNCATION_RADIUS,
)
from .chunking import calculate_intelligent_chunks
from .colormap import write_colormap_lut_if_needed
from .context import DatasetCtx, OrderingCtx
from .dataset_writers.colors import write_colors
from .dataset_writers.scalars import write_positive_scalar
from .node_common import warn_if_over_element_cap

#: Worst-case round-trip center displacement, measured in the splat's OWN
#: marginal σ on that axis, above which the splat counts as *unrepresentable*
#: there. Applied as ``step / 2 > MAX_CENTER_DISPLACEMENT_SIGMAS * sigma``,
#: since half a grid step is the largest error uint16 rounding can produce.
#:
#: ``AUTO``/``MEMORY`` store centers as per-axis uint16 fixed point, so the step
#: on axis *i* is ``(hi_i - lo_i) / 65535`` — a property of the axis EXTENT, not
#: of how sharp the splats are. Left on that grid, a degenerate axis (σ floored
#: to 1e-7 by ``combine_as_new_dimension(..., sigma=0)``) would sit thousands of
#: σ from it, and every such splat would stop matching the slice query it
#: belongs to — silently, since the axis endpoints quantize exactly and so look
#: fine. The encoder's grid snap prevents exactly that for the common stacked
#: case; this rail covers what the snap cannot reach.
#:
#: This rail is the BACKSTOP, not the primary defence. A *gridded* axis — the
#: common stacked/categorical case, whose distinct values all sit on one regular
#: grid — is stored exactly by the encoder's own grid snap
#: (:func:`~luxar.encoding.gridded_axis_step`) at no cost in bytes, so the rail
#: skips it. What is left for the rail is a degenerate sub-population on a
#: NON-gridded axis: a ``sigma=0`` track stack merged into a fit whose time axis
#: is continuous, or any axis with more distinct values than uint16 has levels.
#: There the snap declines and the splats really are destroyed.
#:
#: The number is keyed on *harm*, not on jitter. A splat moved by less than one
#: σ is still centred inside its own core, and the displacement is bounded by
#: the splat's OWN width rather than by the axis extent — it renders essentially
#: where it did. That is not a promise that every query answer is preserved: a
#: slice query matches out to roughly ``tolerance + coverage_sigma·σ``, so a
#: splat sitting on that boundary and pushed the permitted 1.0 σ outward stops
#: matching. Nor is the bound joint — the test is PER AXIS, so a *d*-dimensional
#: center may move a full σ on every axis at once (√d σ in Mahalanobis terms for
#: a diagonal Σ — 2 σ at *d* = 4, more when the axes are correlated) with every
#: per-axis check still passing. What 1 σ does rule out is the failure this rail
#: exists for: past its own core the center leaves the footprint the splat was
#: fitted to describe altogether, which is how a merged-in track ends up
#: thousands of σ from its own coordinate and vanishes from every query. A tighter
#: line — the obvious alternative being 0.25 σ of *jitter*, i.e. 0.125 σ of
#: displacement — would charge 2× the centers bytes for sub-voxel error: measured
#: on 200k splats over an 8192-voxel axis with 5% pinned at the fitter's
#: ``sqrt(1/12) ≈ 0.2887`` σ floor, the step is
#: 0.125 voxel, so the worst displacement is 0.0625 voxel = 0.217 σ — 12.9% of
#: splats would be "unrepresentable" at that 0.125 σ-of-displacement line, against
#: 0.03% here.
MAX_CENTER_DISPLACEMENT_SIGMAS = 1.0

#: Largest fraction of the splats that may be unrepresentable (per
#: :data:`MAX_CENTER_DISPLACEMENT_SIGMAS`) on one axis before the centers
#: escalate to float32. Tripping this gate is NECESSARY but not SUFFICIENT: the
#: escalation also requires that the encoder have no exact path of its own for
#: the array. A GRIDDED axis is snapped and stored exactly
#: (:func:`_center_quantization_offender`), and a LUT-eligible centers array is
#: stored verbatim at ~1 B/value (:func:`_resolve_centers_encoding_mode`); either
#: stands the rail down.
#:
#: The per-splat criterion alone cannot drive the escalation, because a
#: *minimum* over splats is an outlier statistic: any real fit contains a few
#: needle Gaussians (an SPZ import decodes scales as ``exp(u8/16 - 10)``, floor
#: 4.5e-5; a random-Cholesky test fixture draws σ from ``U(0, 1)``), and one
#: such splat in 20,000 would flip the whole array. Displacing a handful of
#: needles is cosmetically negligible, and not worth doubling every centers
#: array on disk (measured: +96% on centers, +54% on a 600k-splat store).
#:
#: 0.1% is set by what the *displacement* criterion above leaves behind.
#: Benign populations collapse to essentially nothing under it — measured worst
#: axis: 0.015% for a 20k random-``U(0, 1)``-Cholesky fixture, 0.0005% for a
#: 200k SPZ-like log-normal scale distribution (0.001% with an added σ=1e-6
#: needle), 0.03% for the 8192-voxel light-sheet case above — so 0.1% still
#: clears every one of them by 3× or more. It also catches what a looser 1% gate
#: would miss — a *minority* degenerate sub-population: merging a
#: 2,000-splat ``sigma=0`` track stack into a 300,000-splat fit with a real
#: σ_t = 3.0 leaves 0.662% of the splats destroyed (max displacement 1,373 σ),
#: and at 1% the rail would stay silent while only 10% of the tracks still
#: landed on their own frame.
#:
#: The residual, stated plainly, because a population gate always has one: up to
#: 0.1% of any store can still be destroyed with NO warning — 1,650 splats on a
#: 1.65M-splat fit. Reproduced end to end: 150,000 splats on a real time axis
#: (σ_t = 3.0, extent 4) with 100 ``sigma=0`` track splats merged in is 0.0667%,
#: under the gate, so saving under ``AUTO`` leaves the rail silent and 61 of the
#: 100 tracks are knocked off their integer frame (worst offset 305 σ). Lowering
#: the fraction further is not the fix, because displacement *magnitude* does
#: not separate the two populations: a ``U(0, 1)`` Cholesky draw on perfectly
#: ordinary data can produce σ = 1e-9 and thousands of σ of displacement, so an
#: absolute-magnitude second tier would only reintroduce the false positives
#: this gate exists to prevent. A dataset carrying a small sub-population that
#: must land exactly on its own coordinate — a track stack, a categorical axis —
#: should be saved with :attr:`~luxar.encoding.EncodingMode.PRECISION` instead
#: of relying on the rail to notice it.
#:
#: Small *N* is the other edge of a fraction: ``fraction > 0.001`` means ONE
#: unrepresentable splat trips the whole array whenever N ≤ 999 (measured: N =
#: 10 / 100 / 500 / 999 escalate on a single bad splat, N = 1000 / 1001 / 2000
#: stay silent). The "a few needle splats cannot flip an otherwise ordinary
#: array" guarantee therefore holds only above 1,000 splats — which is also
#: where the bytes it protects begin to matter.
#:
#: Two consequences of the surrounding design sharpen that residual, and both
#: are deliberate:
#:
#: * A GRIDDED stacked axis keeps uint16 (the encoder's snap stores it exactly)
#:   instead of escalating the whole array to float32. Escalating it would
#:   *incidentally* rescue sub-gate needle populations on the OTHER axes of such
#:   a store — centers are one array with one encoding, so escalating for the
#:   time axis would make every axis exact. That side effect is deliberately not
#:   bought: a stacked store's spatial axes are held to this gate like any other
#:   array's.
#: * The decision is per WRITE, and a partitioned/laddered store writes each
#:   part separately (see :func:`write_gsplat_arrays`), so it is taken per part.
MAX_UNREPRESENTABLE_SPLAT_FRACTION = 0.001


class _CentersEncodingPlan(NamedTuple):
    """Resolved centers mode and the caller mode it was resolved against."""

    requested_mode: EncodingMode
    resolved_mode: EncodingMode


def _axis_center_offender(
    axis: int,
    column: NDArray[np.float32],
    lo: float,
    extent: float,
    chol: NDArray[np.float32],
    n_rows: int,
    n_splats: int,
    min_fraction: float,
) -> Optional[Tuple[int, float, int, float, float]]:
    """Evaluate ONE axis for :func:`_center_quantization_offender`.

    Returns that function's ``(axis, step, n_unrepresentable, fraction,
    sigma_median)`` tuple when this axis offends by MORE than ``min_fraction``
    (the running worst, so a later axis must beat an earlier one), or ``None``.

    The three declines:

    1. A constant axis (``hi == lo``) has a zero step and can never be violated;
       a non-finite extent is not something this rail can reason about (the
       value validators own that), so it is left to the encoder.
    2. The population gate. Below it the axis is not an offender at all.
    3. The grid check — an axis the encoder will store exactly is not an
       offender either. It runs LAST purely for SPEED, not for correctness:
       ``np.unique`` per axis is real time (0.30 s of a 2.85 s encode on 5M×3
       coordinates) and a tripped axis is rare, so testing only what has already
       passed the gate avoids paying it on every axis of every leaf. Run
       eagerly it would return the same offender.

    What IS load-bearing is that a declining axis must not raise the running
    worst — hence ``None`` at every decline, rather than a tuple the caller
    would fold into ``worst_fraction``. A gridded degenerate axis at 100% that
    set the bar on its way out would mask a genuinely broken non-gridded axis at
    0.2% behind a threshold it had no business setting.
    """
    step = extent / COORDINATE_LEVELS
    if not np.isfinite(step) or step <= 0.0:
        return None
    start = axis * (axis + 1) // 2
    row = chol[:, start : start + axis + 1]
    # Marginal σ on this axis: Σ[i,i] = Σ_{j≤i} L[i,j]². Accumulated in float64
    # straight out of the (usually float32) input via ``dtype=`` rather than
    # from a float64 copy of the whole Cholesky — the reduction is identical,
    # and a transient copy is hundreds of MB on a multi-million-splat leaf.
    sigma = np.sqrt(np.einsum("ij,ij->i", row, row, dtype=np.float64))
    # Multiply rather than divide: σ == 0 (a true delta axis) is
    # unrepresentable for any step > 0, and a NaN σ compares False and is
    # left to the value validators.
    unrepresentable = 0.5 * step > MAX_CENTER_DISPLACEMENT_SIGMAS * sigma
    n_bad_rows = int(np.count_nonzero(unrepresentable))
    if n_bad_rows == 0:
        return None
    # A uniform Cholesky is one broadcast row standing in for every splat,
    # so the row fraction IS the splat fraction in both layouts.
    fraction = n_bad_rows / n_rows
    if fraction <= min_fraction:
        return None
    # Ask whether the encoder is going to grid-snap this axis and store it
    # exactly anyway. If it is, there is nothing to protect: escalating would
    # double the centers bytes and warn about data that round-trips bit for bit.
    # The column is upcast HERE, one axis at a time, so the common (no offender)
    # path never pays for a float64 copy of the centers.
    if (
        gridded_axis_step(
            np.asarray(column, dtype=np.float64), lo, extent, COORDINATE_LEVELS
        )
        is not None
    ):
        return None
    return (
        axis,
        step,
        n_bad_rows if n_rows == n_splats else int(round(fraction * n_splats)),
        fraction,
        float(np.median(sigma[unrepresentable])),
    )


def _center_quantization_offender(
    centers: NDArray[np.float32],
    cholesky_factors: NDArray[np.float32],
    n_dims: int,
) -> Optional[Tuple[int, float, int, float, float]]:
    """Find the axis whose uint16 center grid is too coarse for its own splats.

    A splat is *unrepresentable* on axis *i* when the worst-case round-trip
    displacement there — half the grid step — exceeds
    :data:`MAX_CENTER_DISPLACEMENT_SIGMAS` times that splat's own marginal σ on
    that axis, i.e. when quantization can push the center out of its own core.
    An axis offends when MORE THAN :data:`MAX_UNREPRESENTABLE_SPLAT_FRACTION` of
    the splats are unrepresentable on it — a population test, not a minimum, so
    a few needle splats cannot flip an otherwise ordinary array (see that
    constant) — AND the encoder is not going to store that axis exactly anyway.

    Returns ``(axis, step, n_unrepresentable, fraction, sigma_median)`` for the
    worst offender — the axis with the largest affected *fraction* — or ``None``
    when no axis offends. ``sigma_median`` is the median marginal σ of the
    unrepresentable splats, i.e. a representative of what is being displaced.
    The per-axis half lives in :func:`_axis_center_offender`.

    An axis the encoder will GRID-SNAP is not an offender. The encoder widens
    ``hi`` on an axis whose distinct values all sit on one regular grid so the
    quantization grid coincides with the data's own, and then every value
    round-trips bit-exactly at uint16 — for free, since ``lo``/``hi`` are stored
    per axis regardless. Escalating such an axis would double the centers bytes
    and emit a warning for data that was never at risk, so the same predicate the
    encoder uses (:func:`~luxar.encoding.gridded_axis_step`, deliberately shared
    rather than re-derived) is consulted here and the axis skipped. It is
    consulted LAZILY, only for an axis that has already tripped the population
    gate: ``np.unique`` per axis is real time (0.30 s of a 2.85 s encode on 5M×3
    coordinates) and a tripped axis is rare.

    ``None`` is also returned when the centers are not ``(N, n_dims)`` — the
    same decline-rather-than-raise policy the Cholesky-shape arm below states,
    and for the same reason — or when some axis already spans
    :data:`~luxar.typing_utils.constants.COORDINATE_U16_MAX_EXTENT`: the
    encoder's own extent rail then stores the whole array as float32 anyway, so
    there is nothing left for this one to protect and it would only replace a
    warning that names the real cause with one blaming a degenerate σ.

    ``cholesky_factors`` is the packed row-major lower-triangular ``(N, k)``
    form (or a single broadcast ``(1, k)`` row when the Cholesky is uniform), so
    row *i* of the matrix occupies packed columns ``[i(i+1)/2, i(i+1)/2 + i]``
    and the marginal variance is ``Σ[i,i] = Σ_{j≤i} L[i,j]²``. Summing those
    slices directly costs one pass over ``N·k`` and avoids materializing the
    ``(N, d, d)`` array :meth:`~luxar.gsplats.GSplatData.marginal_sigmas` builds.
    """
    n_splats = centers.shape[0]
    if n_splats == 0 or n_dims == 0:
        return None

    # The packed (N, k) layout is load-bearing: handed an UNPACKED (N, d, d)
    # Cholesky this would slice the middle axis, read σ = 0 everywhere and
    # escalate unconditionally — a silently wrong answer, which is the exact bug
    # class this rail exists to prevent. Both call sites go through
    # validate_gsplat_inputs so it cannot happen today; decline rather than
    # raise, because this is an opportunistic size/precision rail and not a
    # validator (the validators own shape errors, and own them earlier).
    chol = np.asarray(cholesky_factors)
    if chol.ndim != 2 or chol.shape[1] < n_dims * (n_dims + 1) // 2:
        return None
    n_rows = chol.shape[0]
    if n_rows == 0:
        return None

    arr = np.asarray(centers)
    if arr.ndim != 2 or arr.shape[1] != n_dims:
        # Neither this rail nor the encoder's snap reasons about a non-(N, d)
        # coordinate array — both are per-axis, and the encoder falls through to
        # its generic per-column quantizer. An axis count that disagrees with
        # ``n_dims`` is the same kind of precondition breach as an unpacked
        # Cholesky above. Unreachable from either call site
        # (validate_gsplat_inputs normalizes to (N, d) and derives n_dims from
        # it); declining keeps the per-axis indexing below honest — and declines
        # rather than raising, for the reason given above.
        return None
    # Derived EXACTLY as `_encode_coordinate` derives them (which upcasts to
    # float64 first, then reduces), because the grid replay below is only a
    # guarantee about what the encoder will do if it is fed the encoder's own
    # numbers. Reducing FIRST and widening the (d,) result is bit-identical —
    # a float32→float64 cast is exact and order-preserving, so it commutes with
    # min/max — and skips a full float64 copy of the centers, which is 53 MB on
    # a 1.65M×4 leaf and is paid on every leaf of every ladder and partition.
    lo = np.min(arr, axis=0).astype(np.float64)
    extents = np.max(arr, axis=0).astype(np.float64) - lo
    # Defer to the encoder's own extent rail: at/above COORDINATE_U16_MAX_EXTENT
    # it already stores the centers as float32 — exactly, so nothing is lost —
    # and its warning names the real cause (an axis too wide for a unit step)
    # instead of this one blaming a degenerate sigma. Escalating here as well
    # would only replace that message with a misleading one and cost the array
    # its content dedup, which is safe for a purely extent-driven fallback
    # (that verdict is a function of the centers bytes alone).
    if float(np.max(extents)) >= COORDINATE_U16_MAX_EXTENT:
        return None

    worst: Optional[Tuple[int, float, int, float, float]] = None
    worst_fraction = MAX_UNREPRESENTABLE_SPLAT_FRACTION
    for axis in range(n_dims):
        found = _axis_center_offender(
            axis,
            arr[:, axis],
            float(lo[axis]),
            float(extents[axis]),
            chol,
            n_rows,
            n_splats,
            worst_fraction,
        )
        if found is None:
            continue
        worst_fraction = found[3]
        worst = found
    return worst


def _resolve_centers_encoding_mode(
    centers: NDArray[np.float32],
    cholesky_factors: NDArray[np.float32],
    n_dims: int,
    mode: EncodingMode,
    encoder: ArrayEncoder,
) -> EncodingMode:
    """Escalate the centers encoding to ``PRECISION`` when uint16 cannot hold it.

    Only ``AUTO``/``MEMORY`` quantize coordinates (``PRECISION`` is already
    exact and ``CUSTOM`` is rejected downstream for COORDINATE), so only those
    two are checked. The escalation is centers-only: the Cholesky, amplitude and
    color tiers keep whatever the caller asked for.

    The rail must stand down wherever the encoder is ALREADY going to store the
    centers exactly, and the encoder has two such paths. An axis it will
    grid-snap never reaches here at all (see
    :func:`_center_quantization_offender`). The other is LUT encoding, which is
    tried before the dtype encoder and is exact *and* smaller than float32 —
    :meth:`~luxar.encoding.encoder.ArrayEncoder.encodes_as_lut` is asked only
    once an offender has been found, so its ``np.unique`` pass costs nothing on
    the common path.
    """
    if mode not in (EncodingMode.AUTO, EncodingMode.MEMORY):
        return mode
    offender = _center_quantization_offender(centers, cholesky_factors, n_dims)
    if offender is None:
        return mode
    if encoder.encodes_as_lut(centers, SemanticType.COORDINATE):
        # Exact already, at ~1 B/value: escalating would quadruple the bytes and
        # warn about a displacement the encoder was never going to apply.
        return mode
    axis, step, n_bad, fraction, sigma_median = offender
    n_splats = centers.shape[0]
    warnings.warn(
        f"GSplat centers: axis {axis} has a uint16 fixed-point step of "
        f"{step:.4g}, so quantizing can displace a center along it by up to "
        f"{step / 2:.4g}. For {n_bad} of {n_splats} splats ({fraction:.2%}; "
        f"median sigma of those {sigma_median:.4g}) that worst case exceeds "
        f"{MAX_CENTER_DISPLACEMENT_SIGMAS:g}·sigma, i.e. it can move the center "
        "clear of the splat's own core and out of a slice query that used to "
        "match it. The encoder cannot store this axis exactly either (it is not "
        "on a regular grid, or it has more distinct values than uint16 has "
        "levels), so the centers are stored as float32 (exact) instead, and the "
        "Cholesky/amplitude/color tiers are unchanged. "
        "A figure near 100% means the axis itself is degenerate (sigma ~ 0); a "
        "small figure usually means a degenerate sub-population was merged onto "
        "a continuous axis (e.g. a sigma=0 track stack merged into a fit).",
        UserWarning,
        stacklevel=2,
    )
    return EncodingMode.PRECISION


def _centers_mode_for_write(
    plan: Optional[_CentersEncodingPlan],
    centers: NDArray[np.float32],
    cholesky_factors: NDArray[np.float32],
    n_dims: int,
    ctx: DatasetCtx,
) -> EncodingMode:
    """Reuse an ordering-time mode only for the same requested write mode."""
    if plan is not None and plan.requested_mode == ctx.encoding_mode:
        return plan.resolved_mode
    return _resolve_centers_encoding_mode(
        centers, cholesky_factors, n_dims, ctx.encoding_mode, ctx.encoder
    )


def validate_gsplat_inputs(
    centers: NDArray[np.float32],
    amplitudes: Union[NDArray[np.float32], float],
    cholesky_factors: NDArray[np.float32],
    colors: Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]] = None,
    *,
    check_values: bool = True,
) -> Tuple[
    NDArray[np.float32],
    Union[NDArray[np.float32], float],
    NDArray[np.float32],
    Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
    int,
    int,
    bool,
]:
    """Validate and normalize gsplat inputs.

    ``check_values=False`` skips the O(N) value scans (finiteness / sign /
    color checks) and keeps only the cheap shape normalization. It is for
    callers that already ran the full check on the SAME arrays — the per-level
    write after ``preflight_validate_leaf`` — so each leaf is value-scanned
    exactly once instead of two or three times.

    Returns:
        (centers, amplitudes, cholesky_factors, colors,
         n_splats, n_dims, cholesky_is_uniform)
    """
    from ...validation.base import (
        _validate_numeric_finite_values,
        validate_cholesky_for_writing,
        validate_colors_for_writing,
        validate_positions_for_writing,
    )
    from .node_common import validate_broadcast_color

    if check_values:
        n_splats, n_dims = validate_positions_for_writing(centers)
    else:
        # Shape/finiteness already checked on these arrays by the caller.
        n_splats, n_dims = centers.shape
    expected_k = n_dims * (n_dims + 1) // 2
    cholesky_is_uniform = False

    if cholesky_factors.ndim == 1:
        if cholesky_factors.shape[0] != expected_k:
            raise ValueError(
                f"Cholesky factors shape mismatch: expected ({expected_k},), "
                f"got {cholesky_factors.shape}"
            )
        cholesky_factors = cholesky_factors.reshape(1, expected_k)
        cholesky_is_uniform = True
    elif cholesky_factors.shape != (n_splats, expected_k):
        raise ValueError(
            f"Cholesky factors shape mismatch: expected ({n_splats}, {expected_k}), "
            f"got {cholesky_factors.shape}"
        )

    # Finiteness + positive-diagonal gate (shape is normalized above). Mirrors
    # the radii/widths validators: a NaN or non-positive diagonal used to pass
    # the shape-only check and either die deep in the encoder after centers were
    # written or be silently clamped to a degenerate covariance.
    if check_values:
        validate_cholesky_for_writing(cholesky_factors, n_dims)

    # Validate amplitudes (finiteness first, mirroring radii/widths — a NaN
    # would silently pass `< 0` since `nan < 0` is False and corrupt the store).
    if isinstance(amplitudes, np.ndarray):
        if amplitudes.shape[0] != n_splats:
            raise ValueError(
                f"Amplitudes shape {amplitudes.shape} doesn't match n_splats {n_splats}"
            )
        if check_values:
            _validate_numeric_finite_values(amplitudes, "amplitudes")
            if np.any(amplitudes < 0):
                min_val = float(np.min(amplitudes))
                raise ValueError(
                    f"Amplitudes must be non-negative (>= 0). "
                    f"Found minimum value: {min_val:.3f}"
                )
    elif isinstance(amplitudes, (int, float)):
        if not np.isfinite(amplitudes):
            raise ValueError(f"Amplitude must be finite. Got {amplitudes}")
        if amplitudes < 0:
            raise ValueError(f"Amplitude must be non-negative (>= 0). Got {amplitudes}")

    # Colors: the same check write_gsplat_arrays historically ran POST-write;
    # running it here puts colors in the pre-group gate on every path (flat
    # write_gsplats and the leaf preflight alike). GSplats accept RGBA — the
    # alpha column is per-splat opacity. Broadcast list/tuple colors get the
    # same pre-write gate Points/Lines use (validate_broadcast_color) — a NaN
    # or wrong-length tuple would otherwise be discovered only in write_colors,
    # after centers/amplitudes/Cholesky were already on disk.
    if check_values:
        if isinstance(colors, np.ndarray):
            validate_colors_for_writing(colors, n_splats, channels=(3, 4))
        elif isinstance(colors, (list, tuple)):
            validate_broadcast_color(colors)

    return (
        centers,
        amplitudes,
        cholesky_factors,
        colors,
        n_splats,
        n_dims,
        cholesky_is_uniform,
    )


def apply_gsplat_spatial_ordering(
    centers: NDArray[np.float32],
    amplitudes: Union[NDArray[np.float32], float],
    cholesky_factors: NDArray[np.float32],
    colors: Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
    label_ids: Optional[np.ndarray],
    n_splats: int,
    n_dims: int,
    cholesky_is_uniform: bool,
    ctx: OrderingCtx,
    coverage_sigma: float = DEFAULT_TRUNCATION_RADIUS,
    barrier_dims: Optional[Sequence[int]] = None,
    *,
    dataset_ctx: Optional[DatasetCtx] = None,
) -> Tuple[
    NDArray[np.float32],
    Union[NDArray[np.float32], float],
    NDArray[np.float32],
    Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
    Optional[np.ndarray],
    Optional[Dict[str, Any]],
    Optional[_CentersEncodingPlan],
]:
    """Apply spatial ordering to gsplat arrays.

    ``barrier_dims`` names categorical/barrier center columns (e.g. time,
    channel) that ordering must group by first so a chunk never straddles a
    category — the gsplat analogue of Points' discrete ``slice_dims``. When
    ``None``, the barrier is auto-detected from the centers
    (:func:`~luxar.io.ordering.detect_barrier_dims`); pass ``[]`` to force pure
    spatial ordering. Callers that know the exact barrier (LOD ``coarsen_dims``
    complement, or a scene's discrete dims) should pass it explicitly.

    ``dataset_ctx`` lets the ordering path resolve the centers' actual encoding
    mode once, ask the encoder for its per-axis round-trip slack, and return a
    plan the array writer may reuse when its requested mode matches. A direct
    caller that omits it gets bounds against the authored centers and leaves
    mode resolution to the write step.

    Returns:
        (centers, amplitudes, cholesky_factors, colors, ordering_data,
        centers_encoding_plan), where ordering_data is None if ordering was not
        applied and centers_encoding_plan is None if no dataset context was supplied.
    """
    ordering_data = None
    centers_encoding_plan = None
    if ctx.enable_spatial_index and n_splats > 0:
        from ..ordering import (
            compute_chunk_bounds_gsplats,
            detect_barrier_dims,
            sort_splats_spatial,
        )

        slice_dims = (
            list(barrier_dims)
            if barrier_dims is not None
            else detect_barrier_dims(centers)
        )

        aprint(f"  🔍 Applying {ctx.ordering_method} ordering to gsplats...")
        sort_indices, ordering_metadata = sort_splats_spatial(
            centers, method=ctx.ordering_method, slice_dims=slice_dims
        )

        centers = centers[sort_indices]
        if not cholesky_is_uniform:
            cholesky_factors = cholesky_factors[sort_indices]
        if isinstance(amplitudes, np.ndarray):
            amplitudes = amplitudes[sort_indices]
        if colors is not None and isinstance(colors, np.ndarray):
            if colors.shape[0] > 1:
                colors = colors[sort_indices]
        if label_ids is not None:
            label_ids = label_ids[sort_indices]

        chunk_size = resolve_gsplat_chunk_size(n_splats, n_dims)

        centers_mode = (
            _resolve_centers_encoding_mode(
                centers,
                cholesky_factors,
                n_dims,
                dataset_ctx.encoding_mode,
                dataset_ctx.encoder,
            )
            if dataset_ctx is not None
            else None
        )
        if dataset_ctx is not None and centers_mode is not None:
            centers_encoding_plan = _CentersEncodingPlan(
                dataset_ctx.encoding_mode, centers_mode
            )
        coord_slack = (
            # Centers encode with the default allow_lut=True, so the query must
            # ask about the same write path.
            dataset_ctx.encoder.coordinate_round_trip_slack(
                centers, centers_mode, allow_lut=True
            )
            if dataset_ctx is not None and centers_mode is not None
            else None
        )

        chunk_bounds = compute_chunk_bounds_gsplats(
            centers,
            cholesky_factors,
            chunk_size,
            coverage_sigma=coverage_sigma,
            slice_dims=slice_dims,
            coord_slack=coord_slack,
        )

        ordering_data = {
            "sort_order": sort_indices,
            "chunk_bounds": chunk_bounds,
            "chunk_size": chunk_size,
            **ordering_metadata,
        }

        aprint(
            f"  ✓ Spatial ordering complete: {ordering_metadata['ordering']} "
            f"with {len(chunk_bounds)} chunks"
        )

    return (
        centers,
        amplitudes,
        cholesky_factors,
        colors,
        label_ids,
        ordering_data,
        centers_encoding_plan,
    )


def resolve_gsplat_chunk_size(n_splats: int, n_dims: int) -> int:
    """Return the canonical spatial-index chunk atom for one gsplat leaf."""
    from ...typing_utils import TARGET_CHUNK_BYTES

    expected_k = n_dims * (n_dims + 1) // 2
    bytes_per_splat = n_dims * 4 + 4 + expected_k * 4 + 16
    return min(max(1024, TARGET_CHUNK_BYTES // bytes_per_splat), n_splats)


def compute_amplitude_mass_stats(
    amplitudes: Union[NDArray[np.float32], float],
    chol_diag: NDArray[np.float32],
    n_splats: int,
) -> Tuple[float, float]:
    """Total integral mass and mass-weighted mean amplitude of a splat set.

    Two cheap O(N) statistics that let the finalize pass
    (:func:`~luxar.io._compiler.finalize.amplitude_window.
    harmonize_gsplat_amplitude_windows`) put every level of an LOD ladder on ONE
    colormap window:

    * **mass** ``Σᵢ aᵢ·|Σᵢ|^½`` — the integral of the mixture (the shared
      ``(2π)^{D/2}`` constant is dropped, exactly as in
      ``gsplats.lod.substitutive._subset_mass`` and
      ``gsplats.lod.additive._mass_score``). ``|Σ|^½ = Π diag(L)``.
    * **mass-weighted mean amplitude** ``Σᵢ aᵢ²·|Σᵢ|^½ / Σᵢ aᵢ·|Σᵢ|^½`` — total
      self-energy over total mass, i.e. the amplitude a unit of mass typically
      carries. This is the estimator that tracks how a substitutive reduction
      re-scales the *rendered* amplitude scalar (a coarse level packs the same
      mass into fewer, brighter splats); a robust upper percentile does not.

    Amplitudes are the **RAW** ones, deliberately — not the alpha-effective
    ``A·α`` the LOD orderers rank by. ``amplitude_data_range`` windows the raw
    amplitude scalar the shader reads, so the statistic that rescales that
    window must be in the same units.

    Scalar (broadcast) amplitudes and a uniform single-row Cholesky are
    broadcast to ``n_splats`` first, so the totals are true totals rather than
    one splat's. Returns ``(0.0, 0.0)`` for an empty set or a non-positive mass.
    """
    if n_splats <= 0:
        return 0.0, 0.0
    # ``.abs()`` on the diagonal, matching both implementations cited above: a
    # negative pivot would otherwise flip the sign of that splat's determinant
    # and CANCEL mass against its neighbours instead of adding to it. Taken
    # AFTER the product rather than per element — ``Π|xᵢ| == |Π xᵢ|`` bit for
    # bit in IEEE (the sign is a separate field), and reducing straight out of
    # the float32 input with ``dtype=`` allocates only the (N,) result instead
    # of two full (N, d) float64 copies, which on a multi-million-splat leaf is
    # hundreds of MB of transient the writer does not need.
    det_sqrt = np.abs(np.prod(np.asarray(chol_diag), axis=-1, dtype=np.float64))
    det_sqrt = np.asarray(det_sqrt, dtype=np.float64).reshape(-1)
    if det_sqrt.shape[0] != n_splats:
        # Uniform (single-row) Cholesky: every splat shares one covariance.
        det_sqrt = np.broadcast_to(det_sqrt[:1], (n_splats,))
    if isinstance(amplitudes, np.ndarray):
        # No shape fallback here: ``validate_gsplat_inputs`` rejects an
        # amplitude array whose length is not ``n_splats`` on BOTH its
        # ``check_values`` paths, so the only broadcast case is the true scalar
        # below. Silently taking ``amps[:1]`` would answer a wrong total.
        amps = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    else:
        amps = np.full(n_splats, float(amplitudes), dtype=np.float64)
    mass = float(np.sum(amps * det_sqrt))
    if not math.isfinite(mass) or mass <= 0.0:
        return 0.0, 0.0
    self_energy = float(np.sum(amps * amps * det_sqrt))
    mwma = self_energy / mass
    if not math.isfinite(self_energy) or not math.isfinite(mwma):
        return 0.0, 0.0
    return mass, mwma


#: Metadata keys ``apply_gsplat_group_attrs`` re-stamps onto the colormap-bearing
#: node when the array writer produced them (see the call site for why).
_OPTIONAL_AMPLITUDE_ATTRS: Tuple[str, ...] = (
    "amplitude_data_range",
    "amplitude_mass",
    "amplitude_mass_weighted_mean",
)


def amplitude_mass_stats_attrs(
    amplitudes: Union[NDArray[np.float32], float],
    chol_diag: NDArray[np.float32],
    n_splats: int,
) -> Dict[str, float]:
    """:func:`compute_amplitude_mass_stats` as the attrs it is stamped under.

    Always BOTH keys, never an empty dict. There is nothing to skip:
    :func:`compute_amplitude_mass_stats` already normalizes every non-finite
    path to ``(0.0, 0.0)``, so a bare ``NaN`` / ``Infinity`` token — which is not
    JSON, and would cost a strict reader (the viewer) the whole store; see
    :func:`~luxar.io._compiler.gsplat_tree.json_safe_value` — cannot reach here.

    Stamping unconditionally is also what keeps "present and zero" (a mass-less
    splat set) distinguishable from "absent" (a legacy store), which the
    finalize-time window harmonization relies on.
    """
    mass, mwma = compute_amplitude_mass_stats(amplitudes, chol_diag, n_splats)
    return {"amplitude_mass": mass, "amplitude_mass_weighted_mean": mwma}


def _copy_present(
    group: zarr.Group, metadata: dict[str, Any], keys: Sequence[str]
) -> None:
    """Copy whichever of ``keys`` ``metadata`` carries onto ``group.attrs``."""
    for key in keys:
        if key in metadata:
            group.attrs[key] = metadata[key]


def _write_label_channel(
    group: zarr.Group,
    label_ids: Optional[np.ndarray],
    label_vocabulary: Optional[Dict[int, str]],
    ordering_data: Optional[Dict[str, Any]],
    ctx: DatasetCtx,
) -> dict[str, Any]:
    if label_ids is None:
        return {}
    chunks = calculate_intelligent_chunks(
        label_ids.shape,
        spatial_index_data=ordering_data,
        dtype=label_ids.dtype,
        per_array_bytes=True,
    )
    ctx.encoder.encode(
        data=label_ids,
        zarr_group=group,
        name="label_ids",
        semantic_type=SemanticType.INDEX,
        mode=ctx.encoding_mode,
        chunks=chunks,
        compressor=ctx.compressor,
        deduplicate=False,
        allow_lut=False,
    )
    return {
        "label_vocabulary": {
            str(label_id): name for label_id, name in (label_vocabulary or {}).items()
        }
    }


def _apply_label_metadata(group: zarr.Group, metadata: Dict[str, Any]) -> None:
    has_label_ids = metadata.get("has_label_ids", False)
    group.attrs["has_label_ids"] = has_label_ids
    group.attrs.update(
        {"label_vocabulary": metadata["label_vocabulary"]} if has_label_ids else {}
    )


def write_gsplat_arrays(
    group: zarr.Group,
    centers: NDArray[np.float32],
    amplitudes: Union[NDArray[np.float32], float],
    cholesky_factors: NDArray[np.float32],
    colors: Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
    label_ids: Optional[np.ndarray],
    label_vocabulary: Optional[Dict[int, str]],
    n_splats: int,
    n_dims: int,
    cholesky_is_uniform: bool,
    ordering_data: Optional[Dict[str, Any]],
    centers_encoding_plan: Optional[_CentersEncodingPlan],
    ctx: DatasetCtx,
) -> dict[str, Any]:
    """Write gsplat arrays to a zarr group and return metadata.

    This is the core array-writing routine used by both single-LOD
    and multi-LOD writers. Inputs (colors included) must already have been
    validated via :func:`validate_gsplat_inputs` — both callers run it in
    their pre-group gate, so nothing is re-scanned here.

    Every per-splat array sizes its own first-axis chunk to its own dtype byte
    budget, rounded DOWN to a multiple of the spatial ``chunk_size`` atom. One
    zarr chunk therefore holds several whole index chunks, and no index chunk's
    row range ever straddles a zarr chunk boundary — which is all the viewer's
    row-range reads require. This is deliberately NOT a mode: both callers (the
    scene compiler and the standalone
    ``.gsplats.zarr`` tree writer) get the identical layout, which is what keeps
    the scene ⇄ standalone parity invariant in
    ``tests/test_scene_leaf_parity.py`` true.

    The **centers** encoding carries one extra rail on top of the encoder's own
    extent check and grid snap: under ``AUTO``/``MEMORY`` half the per-axis
    uint16 grid step (the worst-case round-trip displacement) is compared
    against each splat's own marginal σ on that axis, and centers fall back to
    float32 (with a ``UserWarning``) when MORE THAN
    :data:`MAX_UNREPRESENTABLE_SPLAT_FRACTION` of the splats could be displaced
    by over :data:`MAX_CENTER_DISPLACEMENT_SIGMAS` of their own σ — far enough
    to leave the core the splat was fitted to describe. It is a population
    test, so a few needle splats (which every real fit has) keep the uint16
    size win.

    That rail is the BACKSTOP, and tripping the population gate is necessary but
    not sufficient: it stands down wherever the encoder is going to store the
    array exactly anyway, which it has THREE ways of doing. (1) The common
    degenerate case — a stacked or categorical axis built with
    ``combine_as_new_dimension(..., sigma=0)`` — is normally *gridded*, and the
    encoder snaps its quantization grid onto the data's own spacing so it
    round-trips bit-exactly at uint16 for free; the rail skips every such axis
    (:func:`_center_quantization_offender` asks the encoder's own
    :func:`~luxar.encoding.gridded_axis_step`). (2) A LUT-eligible centers array
    is stored verbatim at ~1 B/value, which is exact AND smaller than float32
    (:meth:`~luxar.encoding.encoder.ArrayEncoder.encodes_as_lut`, asked by
    :func:`_resolve_centers_encoding_mode`). (3) An axis at or above
    :data:`~luxar.typing_utils.constants.COORDINATE_U16_MAX_EXTENT` already
    falls to float32 under the encoder's own extent rail, whose warning names
    the real cause. What is left for the rail is a degenerate sub-population on
    a NON-gridded, non-LUT axis, e.g. a ``sigma=0`` track stack merged into a
    fit whose time axis is continuous. Only the centers escalate; the
    Cholesky/amplitude/color tiers are untouched, and an escalated write also
    opts OUT of content dedup (see below).

    The rail's verdict is per WRITE, i.e. per leaf, so a partitioned or
    laddered node decides PART BY PART: an 8-tile ``tiles`` recipe emits up to
    eight separate warnings, and within one logical node the centers dtype can
    differ between parts — a part whose degenerate splats fall under the
    population gate keeps uint16 (and stays damaged) while its sibling
    escalates. This is not collapsed, deliberately: the function is handed only
    a :class:`~luxar.io._compiler.context.DatasetCtx` and a single group, and
    the standalone tree writer has no whole-tree warning hook to collapse into.

    Returns:
        Metadata dict with n_splats, ndim, has_colors, amplitude_range,
        amplitude_mass, amplitude_mass_weighted_mean, center_bounds, and
        ordering info.
    """
    # Write centers
    chunks_centers = calculate_intelligent_chunks(
        centers.shape,
        spatial_index_data=ordering_data,
        dtype=centers.dtype,
        per_array_bytes=True,
    )
    # Reuse the ordering-time sigma-rail verdict only when it was resolved for
    # this write mode; otherwise resolve against the actual writer context.
    centers_mode = _centers_mode_for_write(
        centers_encoding_plan, centers, cholesky_factors, n_dims, ctx
    )
    # An escalated write must bypass the encoder's content-dedup registry.
    # Dedup is keyed on the centers BYTES alone, but the rail makes the chosen
    # mode depend on a SIBLING array (cholesky_factors) the registry knows
    # nothing about — so two nodes with byte-identical centers and different
    # Cholesky would collapse onto whichever was written first, and an
    # escalated node ref'ing an already-registered uint16 target would silently
    # inherit the exact quantization the rail just refused (issue #1748
    # verbatim, plus a warning claiming it had been prevented). Skipping the
    # registry is sufficient and is the whole fix: `ArrayRefRegistry.check` both
    # looks up AND registers, so an escalated array that never calls it neither
    # resolves to a lossy target nor becomes a target itself. Non-escalated
    # nodes keep the size win.
    centers_escalated = centers_mode != ctx.encoding_mode
    ctx.encoder.encode(
        data=centers,
        zarr_group=group,
        name="centers",
        semantic_type=SemanticType.COORDINATE,
        mode=centers_mode,
        chunks=chunks_centers,
        compressor=ctx.compressor,
        deduplicate=not centers_escalated,
    )

    # Compute amplitude range up-front for the layer-control
    # metadata block below. The helper recomputes max internally;
    # we surface min here because the GSplat metadata dict needs
    # both bounds.
    if isinstance(amplitudes, (int, float)):
        amplitude_min = amplitude_max = float(amplitudes)
    else:
        amplitude_min, amplitude_max = (
            float(np.min(amplitudes)),
            float(np.max(amplitudes)),
        )

    # Canonical POSITIVE_SCALAR helper — shared with Points "radii"
    # and Lines "widths" so the default-precision selection is
    # symmetric across all three geometries.
    write_positive_scalar(
        group=group,
        data=amplitudes,
        name="amplitudes",
        spatial_index_data=ordering_data,
        n_elements=n_splats,
        ctx=ctx,
        log_label_singular="amplitude",
        per_array_bytes=True,
    )

    # Display range for the viewer's colormap window. GSplat amplitudes are
    # heavily right-skewed (a few bright cells over a dim background), so the
    # raw max makes a [0, max] LUT window map ~99% of splats to near-black. Use
    # a robust upper (p99.9) so the signal spans the LUT; the brightest 0.1%
    # clip to white (fine for additive rendering). The user can still widen it
    # via the layer display-range slider.
    amp_data_range: Optional[List[float]] = None
    if isinstance(amplitudes, np.ndarray) and amplitudes.size > 0:
        lo = float(amplitudes.min())
        hi = float(np.percentile(amplitudes, 99.9))
        if not (hi > lo):  # degenerate (constant / tiny) — fall back to max
            hi = float(amplitudes.max())
        amp_data_range = [lo, hi]
        group.attrs["amplitude_data_range"] = amp_data_range

    # Split cholesky_factors into the diagonal (positive, scale-like) and the
    # off-diagonal (signed, zero-centred) so each can be encoded/quantised
    # independently on disk. They are recombined into the packed (N, k) form
    # immediately on read (Python reader + viewer loader), so nothing downstream
    # of the storage boundary sees the split. Split UP HERE because the mass
    # statistics below need the diagonal too (|Σ|^½ = Π diag(L)) — one split,
    # two consumers.
    from ...gsplats.utils.trils import split_tril

    chol_diag, chol_offdiag = split_tril(cholesky_factors, n_dims)

    # Mass statistics for the finalize-time colormap-window harmonization
    # (see ``compute_amplitude_mass_stats``). Stamped HERE, alongside
    # ``amplitude_data_range``, so the "lightweight" ``additive_<i>`` sub-LOD
    # groups carry them too.
    mass_stats = amplitude_mass_stats_attrs(amplitudes, chol_diag, n_splats)
    group.attrs.update(mass_stats)

    if cholesky_is_uniform:
        n_elems_chol: Optional[int] = n_splats
        chunks_diag: Optional[tuple] = None
        chunks_offdiag: Optional[tuple] = None
    else:
        n_elems_chol = None
        # Each half is sized to ITS OWN row width — the diagonal is (N, d), the
        # off-diagonal (N, k-d) — which is what GSPLATS_ZARR_FORMAT.md §8
        # tabulates. Deriving one row count from the packed (N, k) shape gave
        # both halves roughly half their byte budget (for 3D: 2,340 rows where
        # each 3-column half affords 4,680), doubling their request count. The
        # two arrays are read independently — a separate row-range zarr slice
        # each, in the Python reader and in the viewer loader alike — so they
        # need no common row count, only the atom alignment that
        # ``calculate_intelligent_chunks`` gives each of them.
        chunks_diag = calculate_intelligent_chunks(
            chol_diag.shape,
            spatial_index_data=ordering_data,
            dtype=chol_diag.dtype,
            per_array_bytes=True,
        )
        # 1D gsplats have no off-diagonal terms: that array is never written, so
        # leave its chunks unset rather than divide by a zero row width.
        chunks_offdiag = (
            calculate_intelligent_chunks(
                chol_offdiag.shape,
                spatial_index_data=ordering_data,
                dtype=chol_offdiag.dtype,
                per_array_bytes=True,
            )
            if chol_offdiag.shape[1] > 0
            else None
        )

    # One joint call: the encoder owns the whole pair policy (same tier for
    # both halves, AUTO's u8→u16 certificate escalation, the 1D no-offdiag
    # case, and deduplicate=False so the per-channel scales stay in each
    # array's own attrs for the viewer).
    ctx.encoder.encode_cholesky_split(
        zarr_group=group,
        diag=chol_diag,
        offdiag=chol_offdiag,
        ndim=n_dims,
        mode=ctx.encoding_mode,
        n_elements=n_elems_chol,
        chunks_diag=chunks_diag,
        chunks_offdiag=chunks_offdiag,
        compressor=ctx.compressor,
    )

    # Compute metadata
    if n_splats > 0:
        center_min = centers.min(axis=0).tolist()
        center_max = centers.max(axis=0).tolist()
    else:
        center_min = [0.0] * n_dims
        center_max = [0.0] * n_dims

    metadata: dict[str, Any] = {
        "n_splats": n_splats,
        "ndim": n_dims,
        "has_colors": False,
        "has_label_ids": label_ids is not None,
        "amplitude_range": {"min": amplitude_min, "max": amplitude_max},
        "center_bounds": {"min": center_min, "max": center_max},
    }
    metadata.update(mass_stats)
    if amp_data_range is not None:
        # Robust display window; propagated onto the colormap-bearing group by
        # apply_gsplat_group_attrs (the colormap node is often a parent of the
        # array leaf, e.g. an additive-ladder level), so the viewer reads the
        # colormap and its display range from the SAME node.
        metadata["amplitude_data_range"] = amp_data_range

    if ordering_data is not None:
        metadata.update(
            {
                "ordering": ordering_data["ordering"],
                "ordering_min": ordering_data["ordering_min"],
                "ordering_max": ordering_data["ordering_max"],
                "ordering_bits_per_dim": ordering_data["ordering_bits_per_dim"],
                "chunk_size": ordering_data["chunk_size"],
                # Barrier/spatial dim split (mirrors Points/Lines ordering attrs)
                # so the categorical axes an ordering grouped by are inspectable.
                "slice_dims": ordering_data.get("slice_dims", []),
                "ordering_dims": ordering_data.get("ordering_dims", []),
            }
        )
    else:
        metadata["ordering"] = "none"

    # Write colors via the canonical COLOR helper (shared with
    # Points + Lines). The helper handles the tuple/list/ndarray
    # branch, picks the right `color_mode`, and writes
    # `color_data_range` attrs identically across geometries.
    if colors is not None:
        # RGBA accepted: the alpha column is per-splat opacity (consumed by
        # every blending mode; mapped into optical depth in volumetric — see
        # VOLUMETRIC_BLENDING_SPEC.md). Validated pre-group by
        # validate_gsplat_inputs, not here.
        write_colors(
            group=group,
            colors=colors,
            spatial_index_data=ordering_data,
            n_elements=n_splats,
            ctx=ctx,
            per_array_bytes=True,
        )
        metadata["has_colors"] = True

    metadata.update(
        _write_label_channel(group, label_ids, label_vocabulary, ordering_data, ctx)
    )

    # Write chunk_bounds
    if ordering_data is not None:
        chunk_bounds = ordering_data["chunk_bounds"]
        if len(chunk_bounds) > 0:
            create_array(
                group,
                "chunk_bounds",
                data=chunk_bounds,
                chunks=(chunk_bounds.shape[0], n_dims, 2),
                dtype=np.float32,
                compressor=resolve_compressor(ctx.compressor, np.float32),
                overwrite=True,
            )
            aprint(f"  ✓ Chunk bounds written: {len(chunk_bounds)} chunks")

    return metadata


def inherited_gsplat_colormap(
    group: zarr.Group,
    store: zarr.Group,
    inherited_colormap: Optional[str] = None,
) -> Optional[str]:
    """The ``colormap`` an ANCESTOR of ``group`` authored, or ``None``.

    The single rule behind the gray default (see
    :func:`apply_gsplat_group_attrs`): a leaf only gets the manufactured
    ``"gray"`` when nothing above it authored a palette. Without this the
    shadow was structural — ``attrs`` there is the leaf's OWN bag, so a
    ``colormap`` set on an enclosing Group (or on the scene / gsplats root)
    was always beaten by a ``"gray"`` sitting nearer the leaf, and the
    viewer's root→leaf composition (``data/attrs-composer.ts``) could never
    see it (#1600).

    Two write paths reach it, with two different mechanics — hence the one
    function taking both:

    * **already on disk** (the scene compiler): a Group node's attrs are
      written by ``LuxarZarrCompiler.write_group`` at construction time,
      i.e. BEFORE any child leaf exists, so walking ``group.path`` upward
      through ``store`` finds them. Consequently the palette must be authored
      when the ancestor is created; setting ``group.attrs["colormap"]`` after
      its children were written cannot retroactively suppress their gray.
    * **still in flight** (the standalone ``.gsplats.zarr`` tree writer): a
      ``kind=lod`` / ``kind=partition`` wrapper writes its own attrs only
      AFTER its children, so nothing is on disk to walk. There the value
      rides DOWN the recursion in ``inherited_colormap`` — the same channel
      ``coverage_fraction`` / ``child_index`` already use.

    Both are consulted, which is exactly right for a standalone subtree
    grafted into a scene: the in-flight argument covers the wrapper chain
    inside the subtree, the disk walk covers the scene groups above it.

    No authored-vs-manufactured distinction is needed. Only LEAF groups ever
    receive a manufactured ``"gray"`` (``apply_gsplat_group_attrs`` is the one
    place that stamps it, and it is called only on gsplats leaf groups —
    ``kind=lod`` / ``kind=partition`` wrappers are groups written straight
    through by ``gsplat_tree``/``write_group``), and a leaf is never an
    ancestor of another node. So every ``colormap`` this walk can find was
    authored by a caller.

    Args:
        group: The leaf group about to be stamped.
        store: The store ROOT the walk is relative to (never walks above it).
        inherited_colormap: A palette handed down by an in-flight ancestor.

    Returns:
        The nearest inherited palette name, or ``None`` when there is none.
    """
    if inherited_colormap is not None:
        return inherited_colormap

    root_path = (store.path or "").strip("/")
    node_path = (group.path or "").strip("/")
    if root_path:
        if node_path == root_path:
            node_path = ""
        elif node_path.startswith(root_path + "/"):
            node_path = node_path[len(root_path) + 1 :]
        else:
            # Not under this root — nothing this store can tell us.
            return None
    if not node_path:
        return None

    segments = node_path.split("/")
    minimum_depth = int(getattr(store, "attrs", {}).get("type") == "scene")
    # Nearest ancestor first (the leaf itself is excluded: its own attrs are
    # the caller's `attrs` bag, checked separately).
    for depth in range(len(segments) - 1, minimum_depth - 1, -1):
        prefix = "/".join(segments[:depth])
        try:
            ancestor = store[prefix] if prefix else store
        except KeyError:
            continue
        ancestor_attrs = getattr(ancestor, "attrs", None)
        if ancestor_attrs is None:
            continue
        value = ancestor_attrs.get("colormap")
        if value is not None:
            return str(value)
    return None


def apply_gsplat_group_attrs(
    group: zarr.Group,
    metadata: dict[str, Any],
    attrs: dict[str, Any],
    store: zarr.Group,
    scene_tone_mapping: Optional[str],
    lut_tone_mapping_warned: bool,
    inherited_colormap: Optional[str] = None,
    warn_on_missing_tone_mapping: bool = True,
) -> bool:
    """Set standard gsplats group attributes and rendering defaults.

    Mutates both group.attrs and attrs dict in-place. The colormap-LUT
    tone-mapping warn flag is threaded by value (see
    :func:`~luxar.io._compiler.colormap.write_colormap_lut_if_needed`) and the
    updated flag is returned for the caller to store back.

    ``inherited_colormap`` is an in-flight ancestor's palette on the standalone
    tree path; see :func:`inherited_gsplat_colormap`.

    Returns:
        The updated ``lut_tone_mapping_warned`` flag.
    """
    # Default colormap if no colors and no colormap — and only when no ancestor
    # authored one, or the manufactured value would SHADOW it (it sits nearer
    # the leaf, and the viewer composes nearest-setter-wins). See
    # `inherited_gsplat_colormap`.
    if (
        not metadata.get("has_colors")
        and "colormap" not in attrs
        and inherited_gsplat_colormap(group, store, inherited_colormap) is None
    ):
        attrs["colormap"] = WRITER_STAMPED_APPEARANCE_DEFAULTS["colormap"]

    # Write colormap LUT if colormap is a custom array
    lut_tone_mapping_warned = write_colormap_lut_if_needed(
        group,
        attrs,
        scene_tone_mapping,
        lut_tone_mapping_warned,
        warn_on_missing_tone_mapping=warn_on_missing_tone_mapping,
    )

    # Process transform if present
    if "transform" in attrs:
        from ...core.transforms import prepare_transform_for_zarr

        attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])

    # Validate nd_transform if present
    if "nd_transform" in attrs:
        from ...validation.nd_transforms import validate_nd_transform

        dims = None
        if "scene_dimensions" in store.attrs:
            dims = Dimensions.from_dict(store.attrs["scene_dimensions"])
        attrs["nd_transform"] = validate_nd_transform(attrs["nd_transform"], dims)

    # Set rendering defaults. `blending_mode` is deliberately NOT stamped:
    # it has no identity value, so a stamped default would OVERRIDE an
    # ancestor-set mode under the viewer's nearest-setter-wins composition
    # (see node_common.apply_default_render_attrs). Unset leaves inherit;
    # the viewer defaults to "additive".
    #
    # The identity values come from WRITER_STAMPED_APPEARANCE_DEFAULTS rather
    # than from literals here: `gsplat merge` has to tell a stamp made HERE
    # from a value the author chose, and a second copy of the numbers is a
    # drift waiting to happen. `truncation_radius` is not an appearance attr
    # (it is a per-leaf footprint, see COMPOSITING_ATTRS) and keeps its own.
    for key, default in [
        *(
            (key, WRITER_STAMPED_APPEARANCE_DEFAULTS[key])
            for key in IDENTITY_COMPOSITING_ATTRS
        ),
        ("truncation_radius", DEFAULT_TRUNCATION_RADIUS),
    ]:
        if key not in attrs:
            attrs[key] = default

    # Write all attrs, then override with authoritative metadata
    group.attrs.update(attrs)
    group.attrs["type"] = "gsplats"
    group.attrs["n_splats"] = metadata["n_splats"]
    warn_if_over_element_cap("gsplats", metadata["n_splats"], group.name)
    group.attrs["ndim"] = metadata["ndim"]
    group.attrs["has_colors"] = metadata["has_colors"]
    _apply_label_metadata(group, metadata)
    group.attrs["amplitude_range"] = metadata["amplitude_range"]
    # Robust display window on the SAME node as the colormap (set above), so the
    # viewer reads colormap + range together. Without this, an additive-ladder
    # level carries the colormap but not the range (that lives on its sublods),
    # and the viewer falls back to [0, 1] → a near-black render. The mass
    # statistics ride along for the same reason: the finalize-time window
    # harmonization reads them off whichever node carries the window it is
    # about to correct (see finalize/amplitude_window.py).
    _copy_present(group, metadata, _OPTIONAL_AMPLITUDE_ATTRS)
    group.attrs["center_bounds"] = metadata["center_bounds"]
    group.attrs["ordering"] = metadata["ordering"]

    if metadata["ordering"] != "none":
        for key in [
            "ordering_min",
            "ordering_max",
            "ordering_bits_per_dim",
            "chunk_size",
            "slice_dims",
            "ordering_dims",
        ]:
            if key in metadata:
                group.attrs[key] = metadata[key]
    else:
        default_chunk_size = min(1024, max(64, metadata["n_splats"]))
        group.attrs["chunk_size"] = default_chunk_size

    # Position bounds
    center_bounds = metadata["center_bounds"]
    position_bounds = {"min": center_bounds["min"], "max": center_bounds["max"]}
    group.attrs["position_bounds"] = position_bounds
    metadata["position_bounds"] = position_bounds

    return lut_tone_mapping_warned
