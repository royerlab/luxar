"""GSplat assembly: validation, spatial ordering, array writes, group attrs.

Private support module shared by the scene compiler
(:class:`~luxar.io.compiler.LuxarZarrCompiler`) and the root-agnostic node-tree
writer (:mod:`luxar.io._compiler.gsplat_tree`). Holds the gsplat-specific pipeline
both sequence: validate inputs, (optionally) spatially order, write the zarr
arrays, then stamp the group attributes. Because both paths call these same four
free functions, a scene leaf is byte-identical to a standalone one.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import zarr
from arbol import aprint
from numpy.typing import NDArray

from luxar._zarr_compat import create_array

from ...core.dimensions import Dimensions
from ...encoding import EncodingMode, SemanticType
from ...encoding.compression import resolve_compressor
from ...typing_utils.constants import DEFAULT_TRUNCATION_RADIUS
from .chunking import calculate_intelligent_chunks
from .colormap import write_colormap_lut_if_needed
from .context import DatasetCtx, OrderingCtx
from .dataset_writers.colors import write_colors
from .dataset_writers.scalars import write_positive_scalar

#: Worst-case round-trip center displacement, measured in the splat's OWN
#: marginal σ on that axis, above which the splat counts as *unrepresentable*
#: there. Applied as ``step / 2 > MAX_CENTER_DISPLACEMENT_SIGMAS * sigma``,
#: since half a grid step is the largest error uint16 rounding can produce.
#:
#: ``AUTO``/``MEMORY`` store centers as per-axis uint16 fixed point, so the step
#: on axis *i* is ``(hi_i - lo_i) / 65535`` — a property of the axis EXTENT, not
#: of how sharp the splats are. A stacked axis (``combine_as_new_dimension``
#: with ``sigma=0``, whose σ is floored to 1e-7) has an extent of one unit per
#: frame and essentially no width, so the grid step lands thousands of σ away
#: from the integer frame coordinate and every interior frame stops matching a
#: slice query — silently, since the endpoints quantize exactly and so look fine.
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
#: fitted to describe altogether, which is how a stacked frame ends up thousands
#: of σ from its integer coordinate and vanishes from every query. A tighter
#: line (this rail shipped at 0.25 σ of *jitter*, i.e. 0.125 σ of displacement)
#: charges 2× the centers bytes for sub-voxel error: measured on 200k splats
#: over an 8192-voxel axis with 5% pinned at the fitter's
#: ``sqrt(1/12) ≈ 0.2887`` σ floor, the step is
#: 0.125 voxel, so the worst displacement is 0.0625 voxel = 0.217 σ — 12.9% of
#: splats "unrepresentable" at 0.25 σ, and 0.03% here.
MAX_CENTER_DISPLACEMENT_SIGMAS = 1.0

#: Largest fraction of the splats that may be unrepresentable (per
#: :data:`MAX_CENTER_DISPLACEMENT_SIGMAS`) on one axis before the centers
#: escalate to float32.
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
#: clears every one of them by 3× or more. What it now also catches is a
#: *minority* degenerate sub-population, which a 1% gate missed: merging a
#: 2,000-splat ``sigma=0`` track stack into a 300,000-splat fit with a real
#: σ_t = 3.0 leaves 0.662% of the splats destroyed (max displacement 1,373 σ)
#: — under the old gate the rail stayed silent and only 10% of the tracks still
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
MAX_UNREPRESENTABLE_SPLAT_FRACTION = 0.001

#: Number of quantization intervals of a uint16 fixed-point grid (2**16 - 1).
_UINT16_LEVELS = 65535.0


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
    constant).

    Returns ``(axis, step, n_unrepresentable, fraction, sigma_median)`` for the
    worst offender — the axis with the largest affected *fraction* — or ``None``
    when no axis offends. ``sigma_median`` is the median marginal σ of the
    unrepresentable splats, i.e. a representative of what is being displaced.

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
    chol = np.asarray(cholesky_factors, dtype=np.float64)
    if chol.ndim != 2 or chol.shape[1] < n_dims * (n_dims + 1) // 2:
        return None
    n_rows = chol.shape[0]
    if n_rows == 0:
        return None

    lo = np.min(centers, axis=0).astype(np.float64)
    hi = np.max(centers, axis=0).astype(np.float64)
    steps = (hi - lo) / _UINT16_LEVELS

    worst: Optional[Tuple[int, float, int, float, float]] = None
    worst_fraction = MAX_UNREPRESENTABLE_SPLAT_FRACTION
    for axis in range(n_dims):
        step = float(steps[axis])
        # A constant axis (hi == lo) has a zero step and can never be violated;
        # a non-finite extent is not something this rail can reason about (the
        # value validators own that) so it is left to the encoder.
        if not np.isfinite(step) or step <= 0.0:
            continue
        start = axis * (axis + 1) // 2
        row = chol[:, start : start + axis + 1]
        sigma = np.sqrt(np.sum(row * row, axis=1))
        # Multiply rather than divide: σ == 0 (a true delta axis) is
        # unrepresentable for any step > 0, and a NaN σ compares False and is
        # left to the value validators.
        unrepresentable = 0.5 * step > MAX_CENTER_DISPLACEMENT_SIGMAS * sigma
        n_bad_rows = int(np.count_nonzero(unrepresentable))
        if n_bad_rows == 0:
            continue
        # A uniform Cholesky is one broadcast row standing in for every splat,
        # so the row fraction IS the splat fraction in both layouts.
        fraction = n_bad_rows / n_rows
        if fraction > worst_fraction:
            worst_fraction = fraction
            worst = (
                axis,
                step,
                n_bad_rows if n_rows == n_splats else int(round(fraction * n_splats)),
                fraction,
                float(np.median(sigma[unrepresentable])),
            )
    return worst


def _resolve_centers_encoding_mode(
    centers: NDArray[np.float32],
    cholesky_factors: NDArray[np.float32],
    n_dims: int,
    mode: EncodingMode,
) -> EncodingMode:
    """Escalate the centers encoding to ``PRECISION`` when uint16 cannot hold it.

    Only ``AUTO``/``MEMORY`` quantize coordinates (``PRECISION`` is already
    exact and ``CUSTOM`` is rejected downstream for COORDINATE), so only those
    two are checked. The escalation is centers-only: the Cholesky, amplitude and
    color tiers keep whatever the caller asked for.
    """
    if mode not in (EncodingMode.AUTO, EncodingMode.MEMORY):
        return mode
    offender = _center_quantization_offender(centers, cholesky_factors, n_dims)
    if offender is None:
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
        "match it. The centers are therefore stored as float32 (exact) instead; "
        "the Cholesky/amplitude/color tiers are unchanged. A figure near 100% "
        "means the axis itself is degenerate (sigma ~ 0), the usual cause being "
        "a stacked/categorical axis built with "
        "combine_as_new_dimension(..., sigma=0).",
        UserWarning,
        stacklevel=2,
    )
    return EncodingMode.PRECISION


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
    n_splats: int,
    n_dims: int,
    cholesky_is_uniform: bool,
    ctx: OrderingCtx,
    coverage_sigma: float = DEFAULT_TRUNCATION_RADIUS,
    barrier_dims: Optional[Sequence[int]] = None,
) -> Tuple[
    NDArray[np.float32],
    Union[NDArray[np.float32], float],
    NDArray[np.float32],
    Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
    Optional[Dict[str, Any]],
]:
    """Apply spatial ordering to gsplat arrays.

    ``barrier_dims`` names categorical/barrier center columns (e.g. time,
    channel) that ordering must group by first so a chunk never straddles a
    category — the gsplat analogue of Points' discrete ``slice_dims``. When
    ``None``, the barrier is auto-detected from the centers
    (:func:`~luxar.io.ordering.detect_barrier_dims`); pass ``[]`` to force pure
    spatial ordering. Callers that know the exact barrier (LOD ``coarsen_dims``
    complement, or a scene's discrete dims) should pass it explicitly.

    Returns:
        (centers, amplitudes, cholesky_factors, colors, ordering_data)
        where ordering_data is None if ordering was not applied.
    """
    ordering_data = None
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

        from ...typing_utils import TARGET_CHUNK_BYTES

        expected_k = n_dims * (n_dims + 1) // 2
        bytes_per_splat = n_dims * 4 + 4 + expected_k * 4 + 16
        chunk_size = max(1024, TARGET_CHUNK_BYTES // bytes_per_splat)
        chunk_size = min(chunk_size, n_splats)

        chunk_bounds = compute_chunk_bounds_gsplats(
            centers,
            cholesky_factors,
            chunk_size,
            coverage_sigma=coverage_sigma,
            slice_dims=slice_dims,
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

    return centers, amplitudes, cholesky_factors, colors, ordering_data


def write_gsplat_arrays(
    group: zarr.Group,
    centers: NDArray[np.float32],
    amplitudes: Union[NDArray[np.float32], float],
    cholesky_factors: NDArray[np.float32],
    colors: Optional[Union[NDArray[np.float32], List[float], Tuple[float, ...]]],
    n_splats: int,
    n_dims: int,
    cholesky_is_uniform: bool,
    ordering_data: Optional[Dict[str, Any]],
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
    extent check: under ``AUTO``/``MEMORY`` half the per-axis uint16 grid step
    (the worst-case round-trip displacement) is compared against each splat's
    own marginal σ on that axis, and centers fall back to float32 (with a
    ``UserWarning``) when MORE THAN
    :data:`MAX_UNREPRESENTABLE_SPLAT_FRACTION` of the splats could be displaced
    by over :data:`MAX_CENTER_DISPLACEMENT_SIGMAS` of their own σ — far enough
    to leave the core the splat was fitted to describe. It is a population
    test, so a few needle splats (which every real fit has) keep the uint16
    size win, while a degenerate axis — where every splat fails — does not.
    Only the centers escalate; the Cholesky/amplitude/color tiers are
    untouched. An escalated write also opts OUT of content dedup (see below).

    Returns:
        Metadata dict with n_splats, ndim, has_colors, amplitude_range,
        center_bounds, and ordering info.
    """
    # Write centers
    chunks_centers = calculate_intelligent_chunks(
        centers.shape,
        spatial_index_data=ordering_data,
        dtype=centers.dtype,
        per_array_bytes=True,
    )
    # Sigma rail: a lossy (uint16 fixed-point) center grid is only legitimate
    # when half its step is small against the splats' own σ on that axis. See
    # MAX_CENTER_DISPLACEMENT_SIGMAS — this is the single shared choke point
    # where centers AND cholesky_factors are both in hand.
    centers_mode = _resolve_centers_encoding_mode(
        centers, cholesky_factors, n_dims, ctx.encoding_mode
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

    # Write cholesky_factors as two arrays. The diagonal (positive, scale-like)
    # and the off-diagonal (signed, zero-centred) are split so each can be
    # encoded/quantised independently on disk. They are recombined into the
    # packed (N, k) form immediately on read (Python reader + viewer loader),
    # so nothing downstream of the storage boundary sees the split.
    from ...gsplats.utils.trils import split_tril

    chol_diag, chol_offdiag = split_tril(cholesky_factors, n_dims)

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
        "amplitude_range": {"min": amplitude_min, "max": amplitude_max},
        "center_bounds": {"min": center_min, "max": center_max},
    }
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


def apply_gsplat_group_attrs(
    group: zarr.Group,
    metadata: dict[str, Any],
    attrs: dict[str, Any],
    store: zarr.Group,
    scene_tone_mapping: Optional[str],
    lut_tone_mapping_warned: bool,
) -> bool:
    """Set standard gsplats group attributes and rendering defaults.

    Mutates both group.attrs and attrs dict in-place. The colormap-LUT
    tone-mapping warn flag is threaded by value (see
    :func:`~luxar.io._compiler.colormap.write_colormap_lut_if_needed`) and the
    updated flag is returned for the caller to store back.

    Returns:
        The updated ``lut_tone_mapping_warned`` flag.
    """
    # Default colormap if no colors and no colormap
    if not metadata.get("has_colors") and "colormap" not in attrs:
        attrs["colormap"] = "gray"

    # Write colormap LUT if colormap is a custom array
    lut_tone_mapping_warned = write_colormap_lut_if_needed(
        group, attrs, scene_tone_mapping, lut_tone_mapping_warned
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
    for key, default in [
        ("opacity", 1.0),
        ("absorption", 1.0),
        ("gamma", 1.0),
        ("intensity", 1.0),
        ("offset", 0.0),
        ("truncation_radius", DEFAULT_TRUNCATION_RADIUS),
    ]:
        if key not in attrs:
            attrs[key] = default

    # Write all attrs, then override with authoritative metadata
    group.attrs.update(attrs)
    group.attrs["type"] = "gsplats"
    group.attrs["n_splats"] = metadata["n_splats"]
    group.attrs["ndim"] = metadata["ndim"]
    group.attrs["has_colors"] = metadata["has_colors"]
    group.attrs["amplitude_range"] = metadata["amplitude_range"]
    # Robust display window on the SAME node as the colormap (set above), so the
    # viewer reads colormap + range together. Without this, an additive-ladder
    # level carries the colormap but not the range (that lives on its sublods),
    # and the viewer falls back to [0, 1] → a near-black render.
    if "amplitude_data_range" in metadata:
        group.attrs["amplitude_data_range"] = metadata["amplitude_data_range"]
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
