"""Predict how bright a gsplat set renders, and pick an exposure that fits.

WHY THIS EXISTS
===============

Under ``additive`` (and ``luminous``) blending the viewer SUMS every splat that
overlaps a pixel. The per-splat amplitude normalisation of
:mod:`luxar.core.group.gsplats_pipeline.amplitude_norm` puts each splat's
amplitude near 1.0, but says nothing about how many of them stack along the
view ray: a dense fit of a thick specimen sums tens of splats per pixel, and the
opening view is a saturated white blob even though every splat is "in range".

This module predicts that sum without rendering. Each splat contributes its 2-D
marginal footprint on the image plane with peak ``w * RIF(T) * sigma_ray``
(``sigma_ray = 1 / sqrt((Sigma^-1)_aa)`` along the view axis ``a``; ``RIF`` is
the shader's truncated ray-integral factor,
:func:`luxar.gsplats.lift.compute_ray_integral_factor`) — the quantity the
shader integrates — and the footprints are rasterized onto a ~384 px grid
spanning the projected extent. The 99.9th percentile over occupied pixels is the
"projected p99.9" an exposure is chosen against. Measured on five real fits
(DNA, A549, two Allen cells, cells3d) the prediction tracked the rendered HDR sum
within 1.0-1.3x (the render erring dim), and dimming the amplitudes so it lands
at 2.0 removed the saturation under ACES on all of them (1.0 looked dim; 4.0
started saturating the densest).

Hidden (non-displayed) axes are weighted, not masked: each splat counts with its
Gaussian marginal along that axis at the viewer's opening position (a discrete
axis opens at its minimum, a continuous one at its centre — the policy of the
viewer's ``SceneDimsManager.defaultPosition``).

The cost is bounded and the per-splat quantities are built ONCE per plan: a set
larger than ``max_splats`` (2M) is subsampled FIRST — ranked by
``w * |det L|^(1/d)``, the brightest half kept outright, the rest drawn
uniformly with mass-preserving reweighting — and only the kept splats get
covariances. Footprints are evaluated on stencils capped at ``max_radius``
pixels, a larger splat being rasterized on a coarser grid level and bilinearly
upsampled. Sampling noise biases the p99.9 slightly UP (+6% at 2x, +18% at 4x
subsampling), i.e. toward a dimmer, never a more saturated, view. Pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from luxar.gsplats.tree import GSplatNode

__all__ = [
    "DEFAULT_TARGET_BRIGHTNESS",
    "LINEAR_TARGET_BRIGHTNESS",
    "ExposurePlan",
    "HiddenAxis",
    "ProjectedSumStats",
    "SIDE_CAP",
    "SUMMING_BLENDING_MODES",
    "amplitude_unit_attrs",
    "collect_default_splats",
    "data_unit_display_attrs",
    "default_target_brightness",
    "emission_weights",
    "plan_projected_exposure",
    "predict_projected_sum",
    "stamped_amplitude_window",
]

#: Predicted projected p99.9 the default view is dimmed to under a filmic tone
#: mapping. Chosen by eye on the five-dataset sweep under ACES: 1.0 dim, 2.0
#: best, 4.0 saturating cells3d.
DEFAULT_TARGET_BRIGHTNESS = 2.0

#: The target under a CLIPPING tone mapping (``None`` / ``Linear``): everything
#: above 1.0 clips, so 2.0 left 2-7.5% of pixels white there.
LINEAR_TARGET_BRIGHTNESS = 0.9

#: Tone mappings that clip at 1.0 rather than roll off.
_CLIPPING_TONE_MAPPINGS = frozenset({"none", "linear"})

#: Blending modes whose pixel value is a SUM over splats — the only ones whose
#: brightness grows with overlap, and so the only ones auto exposure applies to.
SUMMING_BLENDING_MODES = frozenset({"additive", "luminous"})

#: An orbit to a side view (or a step to another timepoint) may reach at most
#: this multiple of the target at p99.9. Side views of a thick specimen sum 2-8x
#: more splats than the opening view; without a cap one orbit lands at 16-60 (a
#: white blob again). With it they stay at <= ~8 under ACES while most opening
#: views keep the target.
SIDE_CAP = 4.0

#: The side views and other timepoints are a coarse guard, so they are predicted
#: on a coarser grid than the opening view.
SIDE_VIEW_GRID = 256

#: How many timepoints of the first hidden discrete axis, besides the opening
#: one, join the cap term.
EXTRA_TIMEPOINTS = 4

#: Pixels below this (in emitted-weight units) count as empty background.
DEFAULT_OCCUPANCY_FLOOR = 2e-3

#: Splat weights along a hidden axis below this are dropped before rasterizing.
_SLICE_WEIGHT_FLOOR = 1e-4


def default_target_brightness(tone_mapping: Optional[str]) -> float:
    """The auto-exposure target for a tone mapping (``None`` = viewer default ACES).

    2.0 under a filmic operator (ACES / AgX / Neutral / Reinhard / Cineon),
    0.9 under a clipping one (``"None"`` / ``"Linear"``).
    """
    if tone_mapping is not None and tone_mapping.lower() in _CLIPPING_TONE_MAPPINGS:
        return LINEAR_TARGET_BRIGHTNESS
    return DEFAULT_TARGET_BRIGHTNESS


@dataclass(frozen=True)
class HiddenAxis:
    """A non-displayed center column and where the viewer opens it.

    Attributes:
        column: Center column index.
        position: Opening coordinate (a discrete axis' minimum, a continuous
            axis' centre).
        discrete: Whether the axis is discrete (stepped).
        step: Discrete step (ignored when continuous).
        span: ``(min, max)`` of the axis.
    """

    column: int
    position: float
    discrete: bool
    step: float
    span: Tuple[float, float]


@dataclass(frozen=True)
class ProjectedSumStats:
    """Distribution of the predicted per-pixel projected sum.

    Attributes:
        p99: 99th percentile over occupied pixels.
        p999: 99.9th percentile over occupied pixels — the exposure statistic.
        max: Maximum pixel value.
        n_occupied: Pixels above the occupancy floor.
        pixel_size: Grid pixel size in center units.
        n_splats: Splats rasterized (after any subsampling).
    """

    p99: float
    p999: float
    max: float
    n_occupied: int
    pixel_size: float
    n_splats: int


_EMPTY = ProjectedSumStats(0.0, 0.0, 0.0, 0, 0.0, 0)


@dataclass(frozen=True)
class ExposurePlan:
    """Outcome of :func:`plan_projected_exposure`.

    Attributes:
        factor: Brightness factor to apply on top of the input amplitudes,
            always in ``(0, 1]`` — auto exposure never brightens.
        predicted_before: Opening-view projected p99.9 of the input amplitudes.
        predicted_after: Opening-view projected p99.9 after ``factor``.
        target: The requested projected p99.9.
        side_p999: Highest projected p99.9 among the side views and other
            timepoints (input amplitudes), or ``None`` when there are none.
        side_capped: Whether that cap term, not the opening view, set
            ``factor``.
        cap_source: What set the cap when ``side_capped``: ``"side views"`` or
            ``"other timepoints"``.
    """

    factor: float
    predicted_before: float
    predicted_after: float
    target: float
    side_p999: Optional[float] = None
    side_capped: bool = False
    cap_source: Optional[str] = None


def _packed_index(i: int, j: int) -> int:
    """Index of ``L[i, j]`` (``j <= i``) in the row-major packed lower triangle."""
    return i * (i + 1) // 2 + j


def _packed_ndim(n_packed: int) -> int:
    """Dimension ``d`` of a packed lower triangle with ``d(d+1)/2`` entries."""
    return int(round((np.sqrt(8 * n_packed + 1) - 1) / 2))


def _ray_factor(trunc: float) -> float:
    """The shader's truncated ray-integral factor (sqrt(2 pi) as T -> inf)."""
    from luxar.gsplats.lift import compute_ray_integral_factor

    return compute_ray_integral_factor(trunc)


def _rank(weights: np.ndarray, cholesky: np.ndarray) -> np.ndarray:
    """Brightness rank ``w * |det L|^(1/d)`` (float64, one value per splat)."""
    ndim = _packed_ndim(cholesky.shape[1])
    log_det = np.zeros(weights.size, dtype=np.float64)
    for i in range(ndim):
        log_det += np.log(np.abs(cholesky[:, _packed_index(i, i)]) + 1e-30)
    return weights * np.exp(log_det / max(ndim, 1))


def _subsample(
    rank: np.ndarray, max_splats: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Indices + reweighting factors bounding the set to ``max_splats``.

    The brightest half (by ``rank``) is kept outright — an isolated bright splat
    reweighted by the sampling ratio would invent a spike; the rest are drawn
    uniformly and reweighted by the inverse sampling rate, so the expected sum
    is unchanged.
    """
    n = rank.size
    if n <= max_splats:
        return np.arange(n), np.ones(n)
    keep = max_splats // 2
    order = np.argpartition(-rank, keep)
    head, tail = order[:keep], order[keep:]
    rng = np.random.default_rng(seed)
    drawn = rng.choice(tail, size=max_splats - keep, replace=False)
    scale = np.concatenate(
        [np.ones(keep), np.full(drawn.size, tail.size / max(drawn.size, 1))]
    )
    return np.concatenate([head, drawn]), scale


def _covariance_rows(cholesky: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    """``(n, len(columns), d)`` rows of ``L`` for ``columns`` (float64)."""
    ndim = _packed_ndim(cholesky.shape[1])
    rows = np.zeros((cholesky.shape[0], len(columns), ndim), dtype=np.float64)
    for r, i in enumerate(columns):
        for j in range(i + 1):
            rows[:, r, j] = cholesky[:, _packed_index(i, j)]
    return rows


@dataclass
class _Prepared:
    """Per-splat quantities shared by every view of one plan (subsampled)."""

    centers: np.ndarray  # (n, D) float64
    cov: np.ndarray  # (n, k, k) display marginal covariance, k = 2 or 3
    hidden_var: np.ndarray  # (n, H) marginal variances of the hidden columns
    base: np.ndarray  # (n,) weight x reweighting x ray factor
    dims: List[int]
    hidden: List[int]
    trunc: float
    n_source: int


def _prepare(
    centers: np.ndarray,
    cholesky: np.ndarray,
    weights: np.ndarray,
    dims: Sequence[int],
    hidden: Sequence[int],
    *,
    trunc: float,
    max_splats: int,
    seed: int,
) -> _Prepared:
    """Subsample, then build covariances for the kept splats only."""
    weights = np.asarray(weights, dtype=np.float64).ravel()
    cholesky = np.asarray(cholesky)
    idx, reweight = _subsample(_rank(weights, cholesky), max_splats, seed)
    chol = np.asarray(cholesky[idx], dtype=np.float64)
    rows = _covariance_rows(chol, dims)
    cov = rows @ np.transpose(rows, (0, 2, 1))
    hidden_var = np.empty((idx.size, len(hidden)), dtype=np.float64)
    if hidden:
        hrows = _covariance_rows(chol, hidden)
        hidden_var[:] = np.einsum("nij,nij->ni", hrows, hrows)
    return _Prepared(
        centers=np.asarray(centers[idx], dtype=np.float64),
        cov=cov,
        hidden_var=hidden_var,
        base=weights[idx] * reweight * _ray_factor(trunc),
        dims=list(dims),
        hidden=list(hidden),
        trunc=trunc,
        n_source=weights.size,
    )


def _upsample(img: np.ndarray, level: int, shape: Tuple[int, int]) -> np.ndarray:
    """Bilinearly resample a level-``level`` image onto the base grid."""
    out = img
    for axis, n in enumerate(shape):
        coarse = out.shape[axis]
        pos = (np.arange(n) + 0.5) / (1 << level) - 0.5
        i0 = np.clip(np.floor(pos).astype(np.int64), 0, coarse - 1)
        i1 = np.clip(i0 + 1, 0, coarse - 1)
        w = np.clip(pos - i0, 0.0, 1.0)
        a = np.take(out, i0, axis=axis)
        b = np.take(out, i1, axis=axis)
        shape_w = [1, 1]
        shape_w[axis] = n
        w = w.reshape(shape_w)
        out = a * (1.0 - w) + b * w
    return out


def _rasterize_level(
    uv: np.ndarray,
    inv2: np.ndarray,
    peak: np.ndarray,
    radius: np.ndarray,
    px: float,
    shape: Tuple[int, int],
    trunc: float,
    chunk_elems: int = 1 << 22,
) -> np.ndarray:
    """Sum truncated 2-D Gaussian footprints, grouped by stencil radius.

    Works in pixel units and float32 (the statistic is a percentile; the image
    accumulates in float64), evaluating ``exp`` only inside the footprint.
    """
    nx, ny = shape
    img = np.zeros(nx * ny, dtype=np.float64)
    ci = uv / px
    i0 = np.floor(ci).astype(np.int64)
    frac = (ci - i0 - 0.5).astype(np.float32)  # pixel center offset = d - frac
    q = (inv2 * (px * px)).astype(np.float32)  # precision in pixel units
    t2 = np.float32(trunc * trunc)
    for r in np.unique(radius):
        sel = np.flatnonzero(radius == r)
        offs = np.arange(-int(r), int(r) + 1)
        di = np.repeat(offs, offs.size)
        dj = np.tile(offs, offs.size)
        df_i = di.astype(np.float32)
        df_j = dj.astype(np.float32)
        step = max(1, chunk_elems // di.size)
        for start in range(0, sel.size, step):
            k = sel[start : start + step]
            dx = df_i - frac[k, 0:1]
            dy = df_j - frac[k, 1:2]
            m = q[k, 0:1] * dx * dx + 2.0 * q[k, 1:2] * dx * dy + q[k, 2:3] * dy * dy
            ii = i0[k, 0:1] + di
            jj = i0[k, 1:2] + dj
            ok = (m <= t2) & (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
            rows = np.nonzero(ok)[0]
            vals = peak[k][rows] * np.exp(-0.5 * m[ok])
            img += np.bincount((ii * ny + jj)[ok], weights=vals, minlength=nx * ny)
    return img.reshape(nx, ny)


def _ray_geometry(
    cov: np.ndarray, view_axis: Optional[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Image-plane 2x2 covariance and ``sigma_ray`` for each splat."""
    if view_axis is None:  # a planar scene: the phantom third axis
        plane = cov
        det = plane[:, 0, 0] * plane[:, 1, 1] - plane[:, 0, 1] ** 2
        return plane, np.maximum(det, 0.0) ** 0.25
    keep = [i for i in range(3) if i != view_axis]
    plane = cov[:, keep][:, :, keep]
    # (Sigma^-1)_aa = det(plane) / det(Sigma) — the cofactor, no batched inverse.
    det_plane = plane[:, 0, 0] * plane[:, 1, 1] - plane[:, 0, 1] ** 2
    det_full = (
        cov[:, 0, 0] * (cov[:, 1, 1] * cov[:, 2, 2] - cov[:, 1, 2] ** 2)
        - cov[:, 0, 1] * (cov[:, 0, 1] * cov[:, 2, 2] - cov[:, 1, 2] * cov[:, 0, 2])
        + cov[:, 0, 2] * (cov[:, 0, 1] * cov[:, 1, 2] - cov[:, 1, 1] * cov[:, 0, 2])
    )
    sigma_ray = np.sqrt(np.maximum(det_full, 0.0) / np.maximum(det_plane, 1e-300))
    return plane, sigma_ray


def _rasterize(
    uv: np.ndarray,
    plane: np.ndarray,
    peak: np.ndarray,
    *,
    grid: int,
    trunc: float,
    max_radius: int,
    occupancy_floor: float,
) -> ProjectedSumStats:
    """Rasterize image-plane footprints and take the occupied percentiles."""
    if peak.size == 0:
        return _EMPTY
    sd = np.sqrt(np.maximum(plane[:, [0, 1], [0, 1]].max(axis=1), 0.0))
    pad = trunc * float(np.median(sd))
    lo = uv.min(axis=0) - pad
    hi = uv.max(axis=0) + pad
    px = float((hi - lo).max() / grid) or 1.0
    shape = (
        int(np.ceil((hi[0] - lo[0]) / px)) + 1,
        int(np.ceil((hi[1] - lo[1]) / px)) + 1,
    )
    det = np.maximum(plane[:, 0, 0] * plane[:, 1, 1] - plane[:, 0, 1] ** 2, 1e-300)
    inv2 = np.stack(
        [plane[:, 1, 1] / det, -plane[:, 0, 1] / det, plane[:, 0, 0] / det], axis=1
    )
    radius = np.ceil(trunc * sd / px).astype(np.int64)
    level = np.zeros(radius.size, dtype=np.int64)
    big = radius > max_radius
    level[big] = np.ceil(np.log2(radius[big] / max_radius)).astype(np.int64)

    img = np.zeros(shape, dtype=np.float64)
    rel = uv - lo
    for lv in np.unique(level):
        sel = level == lv
        scale = 1 << int(lv)
        lv_shape = (-(-shape[0] // scale), -(-shape[1] // scale))
        lv_radius = np.minimum(np.ceil(radius[sel] / scale), max_radius).astype(
            np.int64
        )
        part = _rasterize_level(
            rel[sel], inv2[sel], peak[sel], lv_radius, px * scale, lv_shape, trunc
        )
        img += part if lv == 0 else _upsample(part, int(lv), shape)

    occupied = img[img > occupancy_floor]
    if occupied.size == 0:
        return ProjectedSumStats(0.0, 0.0, float(img.max()), 0, px, int(peak.size))
    p99, p999 = np.percentile(occupied, [99.0, 99.9])
    return ProjectedSumStats(
        float(p99),
        float(p999),
        float(img.max()),
        int(occupied.size),
        px,
        int(peak.size),
    )


def _view(
    prep: _Prepared,
    view_axis: Optional[int],
    slice_weight: Optional[np.ndarray],
    *,
    grid: int,
    max_radius: int,
    occupancy_floor: float,
) -> ProjectedSumStats:
    """Projected-sum statistics of the prepared set seen down ``view_axis``."""
    local = None if view_axis is None else prep.dims.index(int(view_axis))
    plane, sigma_ray = _ray_geometry(prep.cov, local)
    peak = prep.base * sigma_ray
    live = peak > 0
    if slice_weight is not None:
        peak = peak * slice_weight
        live &= slice_weight > _SLICE_WEIGHT_FLOOR
    plane_cols = [prep.dims[i] for i in range(len(prep.dims)) if i != local]
    uv = prep.centers[:, plane_cols]
    return _rasterize(
        uv[live],
        plane[live],
        peak[live],
        grid=grid,
        trunc=prep.trunc,
        max_radius=max_radius,
        occupancy_floor=occupancy_floor,
    )


def _check_dims(
    ndim: int, display_dims: Optional[Sequence[int]], view_axis: Optional[int]
) -> List[int]:
    """Validate the displayed columns and the view axis; return the columns."""
    dims = list(display_dims) if display_dims is not None else list(range(min(3, ndim)))
    if len(dims) not in (2, 3):
        raise ValueError(f"need 2 or 3 displayed dims, got {dims}")
    if len(dims) == 3 and view_axis not in dims:
        raise ValueError(f"view_axis {view_axis} is not displayed ({dims})")
    return dims


def predict_projected_sum(
    centers: np.ndarray,
    cholesky: np.ndarray,
    weights: np.ndarray,
    view_axis: Optional[int] = None,
    *,
    display_dims: Optional[Sequence[int]] = None,
    grid: int = 384,
    trunc: float = DEFAULT_TRUNCATION_RADIUS,
    max_radius: int = 16,
    max_splats: int = 2_000_000,
    occupancy_floor: float = DEFAULT_OCCUPANCY_FLOOR,
    seed: int = 0,
) -> ProjectedSumStats:
    """Predict the per-pixel additive sum of a splat set seen down one axis.

    Args:
        centers: ``(N, D)`` splat centers.
        cholesky: ``(N, D(D+1)/2)`` packed lower-triangular Cholesky factors.
        weights: ``(N,)`` emitted gain per splat — amplitude times colour/LUT
            gain times opacity (see :func:`emission_weights`).
        view_axis: Center column the camera looks along; must be one of
            ``display_dims``. ``None`` for a planar (two displayed dims) scene.
        display_dims: The center columns the scene displays (2 or 3). Default:
            the first three columns (two for a 2-D set).
        grid: Pixels along the longer side of the projected extent.
        trunc: Footprint truncation in sigmas (the shader's cutoff); also sets
            the ray-integral factor.
        max_radius: Stencil radius cap in pixels; larger footprints are drawn
            on a coarser level and upsampled.
        max_splats: Above this the set is subsampled (mass-preserving) BEFORE
            any covariance is built.
        occupancy_floor: Pixels at or below this are background and excluded
            from the percentiles.
        seed: Subsampling seed.

    Returns:
        The percentiles of the predicted sum over occupied pixels.

    Raises:
        ValueError: On mismatched shapes, a 1-D display, or a ``view_axis`` that
            is not displayed.
    """
    centers = np.asarray(centers)
    n, ndim = centers.shape
    if np.asarray(weights).size != n or np.asarray(cholesky).shape[0] != n:
        raise ValueError("centers, cholesky and weights must have the same length")
    dims = _check_dims(ndim, display_dims, view_axis)
    if n == 0:
        return _EMPTY
    prep = _prepare(
        centers,
        cholesky,
        weights,
        dims,
        [],
        trunc=trunc,
        max_splats=max_splats,
        seed=seed,
    )
    return _view(
        prep,
        view_axis if len(dims) == 3 else None,
        None,
        grid=grid,
        max_radius=max_radius,
        occupancy_floor=occupancy_floor,
    )


def _lut_index(
    amps: np.ndarray, data_range: Optional[Tuple[float, float]], n_lut: int
) -> np.ndarray:
    """LUT indices through the compiler's window, or the midpoint if degenerate.

    A degenerate window (``hi <= lo`` after the compiler's p99.9 -> max fallback)
    is what the viewer maps to the LUT MIDPOINT, so a constant-amplitude store
    is predicted at that colour rather than at black.
    """
    if data_range is None:
        lo = float(amps.min())
        hi = float(np.percentile(amps, 99.9))
        if not hi > lo:
            hi = float(amps.max())
    else:
        lo, hi = float(data_range[0]), float(data_range[1])
    if not hi - lo > 1e-10:
        return np.full(amps.size, (n_lut - 1) // 2 + 1 if n_lut > 1 else 0)
    t = np.clip((amps - lo) / (hi - lo), 0.0, 1.0)
    return np.minimum((t * (n_lut - 1) + 0.5).astype(np.int64), n_lut - 1)


def emission_weights(
    amplitudes: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    colormap: Optional[str] = None,
    opacity: float = 1.0,
    data_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Per-splat emitted gain: amplitude x colour (or LUT) gain x opacity.

    With per-splat ``colors`` the gain is the brightest RGB channel (times the
    alpha, when present). Otherwise the amplitude indexes the colormap LUT
    (default ``gray``) through ``data_range`` — the window the compiler stamps as
    ``amplitude_data_range`` (see :func:`stamped_amplitude_window`; default
    ``[min, p99.9]`` of ``amplitudes``) — and the gain is the LUT colour's
    brightest channel. The window scales with the amplitudes, so the weights are
    linear in them, which is what lets one factor hit a target.
    """
    amps = np.asarray(amplitudes, dtype=np.float64).ravel()
    if amps.size == 0:
        return amps
    if colors is not None:
        col = np.asarray(colors, dtype=np.float64)
        col = col.reshape(amps.size, -1)
        if np.issubdtype(np.asarray(colors).dtype, np.integer):
            col = col / 255.0
        gain = col[:, :3].max(axis=1)
        if col.shape[1] >= 4:
            gain = gain * col[:, 3]
        colour_weighted: np.ndarray = amps * gain * opacity
        return colour_weighted
    from luxar.colormaps.registry import resolve_colormap

    lut = resolve_colormap(colormap or "gray").astype(np.float64) / 255.0
    idx = _lut_index(amps, data_range, lut.shape[0])
    weighted: np.ndarray = amps * lut[idx].max(axis=1) * opacity
    return weighted


def _window(amps: np.ndarray) -> Tuple[float, float]:
    """The writer's per-node window: ``[min, p99.9]``, ``max`` when degenerate."""
    lo = float(amps.min())
    hi = float(np.percentile(amps, 99.9))
    if not hi > lo:
        hi = float(amps.max())
    return lo, hi


def stamped_amplitude_window(
    node: "GSplatNode", scale: float = 1.0
) -> Optional[Tuple[float, float]]:
    """The ``amplitude_data_range`` the compiler stamps on the opening leaves.

    Mirrors the writer and the finalize harmonisation
    (``io/_compiler/gsplat_tree.py`` / ``finalize/amplitude_window.py``): each
    additive rung is windowed ``[min, p99.9]`` and a laddered leaf takes
    ``[min lo, max hi]`` over its rungs; a partition shares one window — the
    union minimum and the splat-count-weighted mean of the usable (``hi > lo``)
    part tops. A substitutive group uses its finest child, which is the
    harmonisation reference. ``scale`` is the amplitude factor applied before
    writing (the window scales with it). ``None`` for an empty tree.
    """
    from luxar.gsplats.tree import iter_default_leaves

    parts: List[Tuple[float, float, int]] = []
    for leaf in iter_default_leaves(node):
        rungs = [
            _window(np.asarray(sub.amplitudes, dtype=np.float64) * scale)
            for sub in leaf.additive_sublods
            if np.asarray(sub.amplitudes).size
        ]
        if rungs:
            parts.append(
                (min(r[0] for r in rungs), max(r[1] for r in rungs), leaf.n_splats)
            )
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0][0], parts[0][1]
    lo = min(p[0] for p in parts)
    usable = [(hi, n) for plo, hi, n in parts if hi > plo]
    if not usable:
        return lo, lo
    total = sum(n for _, n in usable)
    hi = (
        sum(h * n for h, n in usable) / total
        if total > 0
        else sum(h for h, _ in usable) / len(usable)
    )
    return lo, hi


def collect_default_splats(
    node: "GSplatNode",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Pool the default-rendered splats of a tree: centers, chol, amps, colors.

    Every partition part, the finest child of each substitutive LOD group and
    all additive sub-LODs of those leaves — the same selection the per-splat
    normalisation pools, i.e. what the opening view draws. Arrays keep their
    stored dtype (no float64 copy of the whole set).
    """
    from luxar.gsplats.tree import iter_default_leaves

    subs = [s for leaf in iter_default_leaves(node) for s in leaf.additive_sublods]
    if not subs:
        empty = np.empty((0, 0))
        return empty, empty, np.empty(0), None
    centers = np.concatenate([np.asarray(s.centers) for s in subs])
    chol = np.concatenate([np.asarray(s.cholesky_factors) for s in subs])
    amps = np.concatenate([np.asarray(s.amplitudes, dtype=np.float64) for s in subs])
    colors = None
    if all(s.colors is not None for s in subs):
        colors = np.concatenate([np.asarray(s.colors) for s in subs])
    return centers, chol, amps, colors


def _slice_weight(
    prep: _Prepared, axes: Sequence[HiddenAxis], override: Dict[int, float]
) -> Optional[np.ndarray]:
    """Gaussian marginal weight of each splat at the hidden-axis positions.

    A discrete axis' variance is floored at ``(step / 4)^2`` so a splat on the
    grid (or jittered within a quarter step) counts while the neighbouring
    timepoint (``exp(-8)``) does not; a continuous axis' at ``1e-6`` of its span
    squared.
    """
    if not axes:
        return None
    weight = np.ones(prep.centers.shape[0], dtype=np.float64)
    for axis in axes:
        h = prep.hidden.index(axis.column)
        position = override.get(axis.column, axis.position)
        if axis.discrete:
            floor = (0.25 * (axis.step or 1.0)) ** 2
        else:
            floor = 1e-6 * max(axis.span[1] - axis.span[0], 1e-12) ** 2
        var = np.maximum(prep.hidden_var[:, h], floor)
        delta = prep.centers[:, axis.column] - position
        weight *= np.exp(-0.5 * delta * delta / var)
    return weight


def _other_timepoints(axes: Sequence[HiddenAxis]) -> Tuple[Optional[int], List[float]]:
    """Up to :data:`EXTRA_TIMEPOINTS` evenly spaced positions on the first
    hidden discrete axis, excluding its opening position."""
    for axis in axes:
        if not axis.discrete or not axis.step:
            continue
        lo, hi = axis.span
        n = int(round((hi - lo) / axis.step)) + 1
        if n <= 1:
            return None, []
        picks = np.unique(np.round(np.linspace(0, n - 1, EXTRA_TIMEPOINTS + 1)))
        positions = [lo + int(k) * axis.step for k in picks]
        return axis.column, [p for p in positions if abs(p - axis.position) > 1e-9]
    return None, []


def _cap_term(
    prep: _Prepared,
    view_axis: Optional[int],
    axes: Sequence[HiddenAxis],
    opening: Optional[np.ndarray],
    stats_kwargs: Dict[str, object],
) -> Tuple[Optional[float], Optional[str]]:
    """Highest p99.9 over the side views and other timepoints, and its source."""
    side_kwargs = {**stats_kwargs, "grid": SIDE_VIEW_GRID}
    best: Optional[float] = None
    source: Optional[str] = None
    if view_axis is not None:
        for axis in prep.dims:
            if axis == view_axis:
                continue
            p = _view(prep, axis, opening, **side_kwargs).p999  # type: ignore[arg-type]
            if best is None or p > best:
                best, source = p, "side views"
    column, positions = _other_timepoints(axes)
    for position in positions:
        weight = (
            _slice_weight(prep, axes, {column: position})
            if column is not None
            else None
        )
        p = _view(prep, view_axis, weight, **side_kwargs).p999  # type: ignore[arg-type]
        if best is None or p > best:
            best, source = p, "other timepoints"
    return best, source


def plan_projected_exposure(
    centers: np.ndarray,
    cholesky: np.ndarray,
    amplitudes: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    view_axis: Optional[int],
    display_dims: Sequence[int],
    hidden: Sequence[HiddenAxis] = (),
    colormap: Optional[str] = None,
    opacity: float = 1.0,
    data_range: Optional[Tuple[float, float]] = None,
    target: float = DEFAULT_TARGET_BRIGHTNESS,
    trunc: float = DEFAULT_TRUNCATION_RADIUS,
    grid: int = 384,
    max_radius: int = 16,
    max_splats: int = 2_000_000,
    occupancy_floor: float = DEFAULT_OCCUPANCY_FLOOR,
    seed: int = 0,
) -> ExposurePlan:
    """Choose a dimming factor so the opening view's projected p99.9 <= target.

    ``amplitudes`` are the values the scene would render at gain 1: the
    stored data-unit amplitudes (``gsplat convert``, where the factor becomes
    the node's ``display_gain``) or per-splat normalised ones (the Python
    adders, where it multiplies the stored values). Hidden axes (``hidden``) are weighted at the viewer's opening
    position. The factor is
    ``min(1, target / p999_open, SIDE_CAP * target / max(cap views))``, the cap
    views being the other two displayed axes and up to four more timepoints of
    the first hidden discrete axis: the opening view lands on ``target`` unless
    an orbit or a timepoint step would then exceed ``SIDE_CAP`` times it. Data
    already dim enough is left alone, never brightened. The per-splat arrays
    are built once and shared by every view.
    """
    if not (np.isfinite(target) and target > 0):
        raise ValueError(f"target brightness must be positive, got {target!r}")
    centers = np.asarray(centers)
    dims = _check_dims(centers.shape[1], display_dims, view_axis)
    if len(dims) == 2:
        view_axis = None
    if centers.shape[0] == 0:
        return ExposurePlan(1.0, 0.0, 0.0, float(target))
    weights = emission_weights(
        amplitudes, colors, colormap=colormap, opacity=opacity, data_range=data_range
    )
    prep = _prepare(
        centers,
        cholesky,
        weights,
        dims,
        [a.column for a in hidden],
        trunc=trunc,
        max_splats=max_splats,
        seed=seed,
    )
    stats_kwargs: Dict[str, object] = {
        "grid": grid,
        "max_radius": max_radius,
        "occupancy_floor": occupancy_floor,
    }
    opening = _slice_weight(prep, hidden, {})
    before = _view(prep, view_axis, opening, **stats_kwargs).p999  # type: ignore[arg-type]
    factor = min(1.0, target / before) if before > 0 else 1.0
    side, source = _cap_term(prep, view_axis, hidden, opening, stats_kwargs)
    capped = False
    if side is not None and side > 0 and SIDE_CAP * target / side < factor:
        factor = SIDE_CAP * target / side
        capped = True
    return ExposurePlan(
        factor=factor,
        predicted_before=before,
        predicted_after=before * factor,
        target=float(target),
        side_p999=side,
        side_capped=capped,
        cap_source=source if capped else None,
    )


def amplitude_unit_attrs(stats: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``amplitude_unit`` (+ ``amplitude_offset``) for a store's amplitudes.

    A fit records its normalisation basis (``intensity_range``), and its
    amplitudes are then in source data units above the subtracted level
    (``image_min``, see :mod:`luxar.gsplats.fit_basis`): ``counts`` for an
    integer source, ``data units`` otherwise, then ``above background`` when a
    resolved background floor was subtracted or ``above minimum`` when the level
    is the data's own minimum (``--floor none``; it may be negative for a float
    source), with that level as ``amplitude_offset`` (omitted when it is below
    1e-6 of the range). A store recording
    no basis — an imported classical splat file, a merge of disagreeing fits —
    gets ``a.u.`` and no offset: nothing says what its numbers mean.

    With a basis, ``source_data_range`` records ``[0, intensity_range]`` in
    amplitude units. After intensity scaling the amplitudes and range are in
    arbitrary units; the raw source floor is then omitted from the scene.
    """
    import numpy as np

    from luxar.gsplats.fit_basis import source_intensity_span

    stats = stats or {}
    if stats.get("intensity_range") is None:
        return {"amplitude_unit": "a.u."}
    try:
        integer = np.dtype(stats.get("source_dtype")).kind in "iu"
    except (TypeError, ValueError):
        integer = False
    scaled = stats.get("intensity_scale", 1.0) != 1.0
    unit = "a.u." if scaled else ("counts" if integer else "data units")
    level = _subtracted_level(stats)
    attrs: Dict[str, Any] = {}
    width = float(stats["intensity_range"])
    if not scaled and level is not None and abs(level) >= 1e-6 * abs(width):
        # `floor` is a number exactly when a background level was RESOLVED and
        # subtracted; otherwise the subtracted level is the data's own minimum
        # (or a low percentile) — which a float source may put below zero.
        background = isinstance(stats.get("floor"), (int, float)) and not isinstance(
            stats.get("floor"), bool
        )
        unit += " above background" if background else " above minimum"
        attrs["amplitude_offset"] = float(level)
    attrs["amplitude_unit"] = unit
    span = source_intensity_span(stats)
    if span is not None:
        attrs["source_data_range"] = [0.0, float(span)]
    return attrs


def _subtracted_level(stats: Mapping[str, Any]) -> Optional[float]:
    """The level the fit subtracted, for LABELLING: ``image_min``, else ``floor``.

    Unlike :func:`~luxar.gsplats.fit_basis.fit_image_min` (which reverses the
    shift on a raw volume and treats a negative level as unrecorded), any FINITE
    level is accepted: a float source whose minimum is negative still had that
    minimum subtracted, and its amplitudes are "above minimum" by that much.
    """
    import math

    for key in ("image_min", "floor"):
        value = stats.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            return float(value)
    return None


def data_unit_display_attrs(
    stats: Optional[Mapping[str, Any]],
    *,
    gain: Optional[float] = None,
    amplitudes: Optional[np.ndarray] = None,
    plan: Optional["ExposurePlan"] = None,
) -> Dict[str, Any]:
    """The data-unit display attrs ``luxar gsplat convert`` stamps, for Python.

    Returns ``display_gain``, ``amplitude_unit`` and — when the store records
    them — ``amplitude_offset`` and ``source_data_range``, ready to pass to
    ``add_gsplats_from_data`` / ``add_gsplats_from_file`` together with
    ``normalize_amplitudes=False`` so the scene keeps the fit's data units.

    The gain is ``gain`` when given; otherwise the legacy per-splat
    normalisation of ``amplitudes`` (their p99.9 → 1.0 when above 1.0) times
    ``plan.factor`` from :func:`plan_projected_exposure` (pass ``amplitudes ×
    per-splat factor`` to the planner, as convert does), else 1.0.

    Example:
        >>> from luxar.gsplats.exposure import data_unit_display_attrs
        >>> node, stats = load_gsplat_node(path, include_stats=True)
        >>> attrs = data_unit_display_attrs(stats, amplitudes=amps, plan=plan)
        >>> scene.add_gsplats_from_file("g", path, normalize_amplitudes=False, **attrs)
    """
    if gain is None:
        gain = 1.0
        if amplitudes is not None:
            from luxar.core.group.gsplats_pipeline.amplitude_norm import (
                _sample_reference,
                resolve_factor,
            )

            flat = np.asarray(amplitudes, dtype=np.float64).ravel()
            gain = resolve_factor(True, _sample_reference(flat)) or 1.0
        if plan is not None:
            gain *= float(plan.factor)
    if not (np.isfinite(gain) and gain > 0.0):
        raise ValueError(f"display gain must be positive and finite, got {gain!r}")
    return {"display_gain": float(gain), **amplitude_unit_attrs(stats)}
