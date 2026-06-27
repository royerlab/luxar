"""Blind-spot cross-validation for Gaussian-splat model selection.

Implements the manuscript's calibration protocol (Supp. Doc. 2,
``splat_count_vs_quality``):

1. Mask 5% of voxels with a deterministic Bernoulli draw (seed=42).
2. Replace masked voxels with the median of their 3^D donut neighbourhood
   (centre excluded) — Noise2Self self-supervision.
3. Fit a Gaussian-splat model on the donut-filled volume at each ``K`` in
   a sweep; the optimiser never sees the original noisy values at masked
   positions.
4. Evaluate held-out PSNR against the *original* (pre-fill) values at the
   masked positions.
5. The ``K`` that maximises held-out PSNR is the principled splat budget
   — capacity beyond ``K*`` memorises noise rather than signal.

As a free byproduct, an ensemble noise-floor estimator (Laplacian +
Haar HH + background MAD) places each dataset in absolute terms.

The protocol is purely additive: ``fit_gaussian_splats`` is called
unchanged at each ``K``; this module owns mask generation, donut fill,
held-out evaluation, K-grid construction, peak detection, and noise-floor
estimation.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

# =============================================================================
# CV mask
# =============================================================================


def cv_mask(
    shape: Tuple[int, ...],
    fraction: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Deterministic Bernoulli boolean mask for blind-spot cross-validation.

    Defaults match Batson & Royer (2019) and the Luxar manuscript: 5% of
    voxels are held out with seed 42.

    Parameters
    ----------
    shape : tuple of int
        Output array shape.
    fraction : float, default=0.05
        Probability of any voxel being marked True (held out).
    seed : int, default=42
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray of bool, shape ``shape``
        ``True`` at held-out positions, ``False`` elsewhere.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    rng = np.random.RandomState(seed)
    return rng.rand(*shape) < fraction


# =============================================================================
# Donut-median fill
# =============================================================================


def donut_median_fill(
    V: np.ndarray,
    mask: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """Replace masked voxels with the median of their donut neighbourhood.

    The donut is the ``(2r+1)^D`` cube around each masked voxel with the
    centre excluded — 26 neighbours in 3D when ``r=1``. Operates on
    arrays of arbitrary dimension (works for 2D, 3D, 4D, ...). Edge
    voxels use ``mode='reflect'`` padding.

    Vectorised: gathers donut values for all masked positions at once
    via stacked shifted-index lookups against a single padded copy of
    ``V``. Memory cost: ``(2r+1)^D - 1`` floats per masked voxel.

    Parameters
    ----------
    V : np.ndarray
        Volume to fill.
    mask : np.ndarray of bool, same shape as ``V``
        ``True`` at positions to replace.
    radius : int, default=1
        Donut half-width. Default ``1`` → ``3^D`` neighbourhood, matching
        the manuscript.

    Returns
    -------
    np.ndarray, same shape and dtype as ``V``
        Copy of ``V`` with masked voxels replaced by donut medians.
        Unmasked voxels are unchanged.
    """
    if V.shape != mask.shape:
        raise ValueError(f"V shape {V.shape} != mask shape {mask.shape}")
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")

    D = V.ndim
    # Donut footprint: all (2r+1)^D offsets except (0,...,0)
    offsets = [
        o
        for o in itertools.product(range(-radius, radius + 1), repeat=D)
        if any(c != 0 for c in o)
    ]

    masked_idx = np.nonzero(mask)
    n_masked = masked_idx[0].size if len(masked_idx) > 0 else 0

    if n_masked == 0:
        out_empty: np.ndarray = V.copy()
        return out_empty

    # Pad with reflect so edge voxels have full neighbourhoods
    V_pad = np.pad(V, radius, mode="reflect")

    n_donut = len(offsets)
    donut_values = np.empty((n_donut, n_masked), dtype=V.dtype)

    for k, offset in enumerate(offsets):
        shifted = tuple(masked_idx[d] + radius + offset[d] for d in range(D))
        donut_values[k] = V_pad[shifted]

    median_values = np.median(donut_values, axis=0)

    V_filled: np.ndarray = V.copy()
    V_filled[masked_idx] = median_values.astype(V.dtype, copy=False)
    return V_filled


# =============================================================================
# Held-out PSNR
# =============================================================================


def held_out_psnr(
    V_hat: np.ndarray,
    V_original: np.ndarray,
    mask: np.ndarray,
    data_range: Optional[float] = None,
) -> float:
    """PSNR of reconstruction at masked voxels vs the original (pre-fill) values.

    Parameters
    ----------
    V_hat : np.ndarray
        Reconstructed volume from the splat fit.
    V_original : np.ndarray
        The unmodified original volume (NOT the donut-filled one).
    mask : np.ndarray of bool
        Held-out mask. Must broadcast to the volume shape.
    data_range : float, optional
        Dynamic range for PSNR. If ``None``, uses
        ``V_original.max() - V_original.min()`` over the whole volume.

    Returns
    -------
    float
        PSNR in dB. ``+inf`` when MSE is zero, ``nan`` when mask is empty.
    """
    if V_hat.shape != V_original.shape:
        raise ValueError(
            f"shape mismatch: V_hat {V_hat.shape} vs V_original {V_original.shape}"
        )
    if mask.shape != V_original.shape:
        raise ValueError(f"mask shape {mask.shape} != volume shape {V_original.shape}")

    held_pred = V_hat[mask]
    held_true = V_original[mask]

    if held_pred.size == 0:
        return float("nan")

    mse = float(np.mean((held_pred - held_true) ** 2))
    if mse == 0.0:
        return float("inf")
    if data_range is None:
        data_range = float(V_original.max() - V_original.min())
    if data_range == 0.0:
        return float("inf")
    return float(10.0 * math.log10(data_range**2 / mse))


# =============================================================================
# Content-aware metrics (regime-robust extensions)
# =============================================================================
#
# The min--max-range held-out PSNR above is *correct for raw/noisy data at a
# manageable scale* (the manuscript regime), but it is **background-dominated**
# on large sparse volumes: a trivial predict-zero reconstruction already scores
# 40-60 dB, so the absolute PSNR — and the shape of the K-sweep curve — is
# dominated by empty space rather than signal fidelity. The two helpers below
# give regime-robust alternatives:
#   * gain-over-baseline: dB the fit beats predict-zero (plateaus meaningfully),
#   * foreground-restricted PSNR: error only where there is signal.
# Both are *additive* — the default K* selection still uses the min--max metric.


def predict_zero_baseline_mse(V_original: np.ndarray, mask: np.ndarray) -> float:
    """MSE of the trivial all-zeros reconstruction at masked voxels.

    This is the "free" error floor any fit must beat. On sparse data it is
    small (most masked voxels are background ~0), which is exactly why the
    raw held-out PSNR looks deceptively high.
    """
    held = V_original[mask]
    if held.size == 0:
        return float("nan")
    return float(np.mean(held.astype(np.float64) ** 2))


def held_out_gain_db(held_mse: float, baseline_mse: float) -> float:
    """dB improvement of the fit over the predict-zero baseline.

    ``10 * log10(baseline_mse / held_mse)``. 0 dB means "no better than
    predicting zeros"; this metric plateaus meaningfully (it is not inflated
    by trivially-reconstructed background), so it is the recommended
    K*-selection metric for sparse / noise-free data.
    """
    if not math.isfinite(baseline_mse) or baseline_mse <= 0.0:
        return float("nan")
    if held_mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10(baseline_mse / held_mse))


def _otsu_threshold(V: np.ndarray) -> float:
    """Otsu threshold (subsampled for speed); falls back to ``V.min()``."""
    v = np.asarray(V)
    flat = v.reshape(-1)
    if flat.size > 5_000_000:  # cap histogram sample on gigavoxel volumes
        flat = flat[:: max(1, flat.size // 5_000_000)]
    try:
        from skimage.filters import threshold_otsu  # lazy: optional dep

        return float(threshold_otsu(flat.astype(np.float32, copy=False)))  # type: ignore[no-untyped-call]
    except Exception:
        return float(v.min())


def foreground_mask_otsu(V: np.ndarray) -> np.ndarray:
    """Boolean foreground mask via Otsu's threshold (``V > thr``)."""
    v = np.asarray(V)
    return v > _otsu_threshold(v)


def held_out_psnr_foreground(
    V_hat: np.ndarray,
    V_original: np.ndarray,
    held_mask: np.ndarray,
    foreground_mask: np.ndarray,
    data_range: Optional[float] = None,
) -> float:
    """Held-out PSNR restricted to voxels that are both held out AND foreground.

    Strips the background-domination from :func:`held_out_psnr` so the curve
    reflects how well actual signal (not empty space) is reconstructed.
    Returns ``nan`` when the held-out∩foreground set is empty.
    """
    sel = held_mask & foreground_mask
    pred = V_hat[sel]
    true = V_original[sel]
    if pred.size == 0:
        return float("nan")
    mse = float(np.mean((pred - true) ** 2))
    if mse == 0.0:
        return float("inf")
    if data_range is None:
        data_range = float(V_original.max() - V_original.min())
    if data_range == 0.0:
        return float("inf")
    return float(10.0 * math.log10(data_range**2 / mse))


# =============================================================================
# Content / feature estimation (shared metric for density + planner)
# =============================================================================


def count_features(
    V: np.ndarray,
    method: str = "peaks",
    *,
    threshold_abs: Optional[float] = None,
    **kwargs: Any,
) -> int:
    """Estimate the feature content of a volume — the predictor of splat need.

    The empirical investigation found local-maxima count (``peaks``) the best
    predictor of how many splats a region needs (better than intensity-sum or
    foreground-count), so it is the default. ``edges`` (summed Sobel gradient
    magnitude, thresholded) suits non-punctate structure (filaments,
    membranes); ``intensity`` is a robust foreground-voxel count.

    Parameters
    ----------
    V : np.ndarray
        Input volume.
    method : {"peaks", "edges", "intensity"}, default="peaks"
        Feature estimator. Pluggable so non-nuclear data can choose ``edges``.
    threshold_abs : float, optional
        Absolute detection level (the value :func:`feature_threshold` returns).
        When given, every method counts at this *shared* level instead of a
        per-volume relative one — so counts on different crops compose (required
        when ranking sliding windows; a per-crop relative threshold lets a
        faint-noise window out-score a real one). ``peaks``/``edges`` threshold
        the blurred / gradient field at it; ``intensity`` counts ``V > thr``
        (strict ``>``, matching :func:`foreground_mask_otsu`).
    **kwargs
        Forwarded to the underlying estimator (e.g. ``radius``,
        ``threshold_rel`` for ``peaks``).

    Returns
    -------
    int
        A non-negative feature count.
    """
    v = np.asarray(V, dtype=np.float32)
    if method == "peaks":
        from luxar.gsplats.seeds.utils import count_local_maxima

        return int(count_local_maxima(v, threshold_abs=threshold_abs, **kwargs))
    if method == "edges":
        from luxar.gsplats.seeds.edges import _compute_nd_sobel_magnitude

        mag = np.asarray(_compute_nd_sobel_magnitude(v))
        if threshold_abs is not None:
            thr = float(threshold_abs)
        else:
            m = float(mag.max())
            if m <= 0.0:
                return 0
            thr = float(kwargs.get("threshold_rel", 0.1)) * m
        return int(np.count_nonzero(mag >= thr))
    if method == "intensity":
        if threshold_abs is not None:
            return int(np.count_nonzero(v > float(threshold_abs)))
        return int(np.count_nonzero(foreground_mask_otsu(v)))
    raise ValueError(
        f"unknown feature method {method!r}; use 'peaks', 'edges', or 'intensity'"
    )


def feature_threshold(
    V: np.ndarray, method: str = "peaks", threshold_rel: float = 0.1
) -> float:
    """The *exact absolute intensity level* :func:`count_features` thresholds at.

    Single source of truth for the cal→planner contract: the calibration records
    this so the planner's ``scan_content`` counts on the identical scale (the
    detectors threshold relative to a *blurred* / gradient / Otsu level, NOT the
    raw max, so a naive ``0.1*max`` drifts — badly with hot outliers).

    * ``peaks``     → ``threshold_rel * max(soft_blur(V))`` (matches count_local_maxima)
    * ``edges``     → ``threshold_rel * max(|∇V|)``
    * ``intensity`` → the Otsu cut
    """
    v = np.asarray(V, dtype=np.float32)
    if v.size == 0:
        return 0.0
    if method == "peaks":
        from luxar.gsplats.seeds.utils import soft_blur_nd

        return float(threshold_rel * float(np.asarray(soft_blur_nd(v)).max()))
    if method == "edges":
        from luxar.gsplats.seeds.edges import _compute_nd_sobel_magnitude

        return float(
            threshold_rel * float(np.asarray(_compute_nd_sobel_magnitude(v)).max())
        )
    if method == "intensity":
        return _otsu_threshold(v)
    raise ValueError(
        f"unknown feature method {method!r}; use 'peaks', 'edges', or 'intensity'"
    )


def _robust_feature_level(
    V: np.ndarray,
    method: str = "peaks",
    threshold_rel: float = 0.1,
    q: float = 99.9,
    **_: Any,
) -> float:
    """Outlier-robust analogue of :func:`feature_threshold` for window ranking.

    Identical in spirit to :func:`feature_threshold` but bases the level on the
    ``q``-th percentile of the (method-transformed) field instead of its **max**,
    so a handful of hot voxels (dead/stuck pixels, cosmic-ray hits) cannot set the
    global signal level above genuine content — which would otherwise gate every
    real window to zero and let the lone-outlier window win
    (:func:`select_calibration_region`). Used ONLY for ranking; the cal→planner
    contract still records the max-based :func:`feature_threshold` on the chosen
    crop, so this does not perturb the transferred density.
    """
    v = np.asarray(V, dtype=np.float32)
    if v.size == 0:
        return 0.0

    def _subsample(field: np.ndarray) -> np.ndarray:
        flat = field.reshape(-1)
        if flat.size > 5_000_000:  # bound the percentile cost on gigavoxel volumes
            flat = flat[:: max(1, flat.size // 5_000_000)]
        return flat

    if method == "peaks":
        from luxar.gsplats.seeds.utils import soft_blur_nd

        field = _subsample(np.asarray(soft_blur_nd(v)))
        return float(threshold_rel * float(np.percentile(field, q)))
    if method == "edges":
        from luxar.gsplats.seeds.edges import _compute_nd_sobel_magnitude

        field = _subsample(np.asarray(_compute_nd_sobel_magnitude(v)))
        return float(threshold_rel * float(np.percentile(field, q)))
    if method == "intensity":
        flat = _subsample(v)
        hi = float(np.percentile(flat, q))
        return _otsu_threshold(np.minimum(flat, hi))  # Otsu on winsorised values
    raise ValueError(
        f"unknown feature method {method!r}; use 'peaks', 'edges', or 'intensity'"
    )


@dataclass
class RegionSelection:
    """Provenance of an auto-selected calibration sub-region."""

    origin: List[int]
    """Top-left corner of the crop in the original volume's coordinates."""
    size: List[int]
    """Crop shape actually used (clamped per-axis to the volume)."""
    strategy: str
    """``densest`` | ``median`` | ``whole``."""
    n_features: int
    """Feature count inside the chosen crop."""
    score: float
    """Feature *density* (features / voxel) used to rank candidate windows."""


def select_calibration_region(
    V: np.ndarray,
    region_size: int = 256,
    strategy: str = "densest",
    feature: str = "peaks",
    **feature_kwargs: Any,
) -> Tuple[np.ndarray, RegionSelection]:
    """Pick a content-rich sub-region to calibrate at the *fitting* scale.

    The manuscript calibrates on crops ≤~20 M voxels; on a large sparse volume
    the held-out metric is background-dominated and the absolute K wrong-scale.
    This slides non-overlapping ``region_size`` windows, scores each by feature
    density (:func:`count_features`), and returns the chosen crop + provenance.

    ``strategy="densest"`` picks the highest-density window (worst case for
    splat budget); ``"median"`` picks the median-density window (representative,
    avoids the single brightest outlier). For volumes no larger than
    ``region_size`` on every axis the whole volume is returned
    (``strategy="whole"``).
    """
    v = np.asarray(V)
    if v.ndim != 3:
        raise ValueError(
            f"select_calibration_region expects a 3-D volume, got shape {v.shape}"
        )
    shape = v.shape
    size = [int(min(region_size, s)) for s in shape]

    def _origins(n: int, t: int) -> List[int]:
        os_ = list(range(0, n - t + 1, t))
        if not os_ or os_[-1] != n - t:
            os_.append(max(0, n - t))
        return os_

    # whole-volume short-circuit (no cross-window ranking needed)
    if all(size[d] == shape[d] for d in range(v.ndim)):
        n = count_features(v, method=feature, **feature_kwargs)
        return v, RegionSelection(
            origin=[0] * v.ndim,
            size=list(size),
            strategy="whole",
            n_features=int(n),
            score=float(n) / max(1, v.size),
        )

    # One *shared, outlier-robust* absolute level for all windows. Counting each
    # window at this fixed level (not its own relative max) makes counts compose
    # across windows — and using a high percentile rather than the raw max means a
    # lone hot voxel can't raise the bar above genuine signal and gate every real
    # window to zero (which previously let the single-outlier window win).
    global_thr = _robust_feature_level(v, feature, **feature_kwargs)

    axis_origins = [_origins(shape[d], size[d]) for d in range(v.ndim)]
    # Each candidate ranked primarily by feature count, ties broken by total
    # intensity (so equal-count windows prefer the one with more signal, not a
    # faint blob tail). `score` reported = feature density (features / voxel).
    candidates: List[Tuple[int, float, List[int], int]] = []
    for origin in itertools.product(*axis_origins):
        sl = tuple(slice(origin[d], origin[d] + size[d]) for d in range(v.ndim))
        crop = v[sl]
        # Count at the shared absolute level: a background window scores 0
        # naturally (no voxel reaches the level), so no separate gate is needed —
        # and the previous gate compared a *raw* crop max against a *blurred*-field
        # threshold (a units mismatch that mis-gated sharp single-voxel features).
        n = count_features(
            crop, method=feature, threshold_abs=global_thr, **feature_kwargs
        )
        candidates.append((int(n), float(crop.sum()), list(origin), int(n)))

    if not candidates:  # pragma: no cover - defensive
        n = count_features(v, method=feature, **feature_kwargs)
        return v, RegionSelection([0] * v.ndim, list(size), "whole", int(n), 0.0)

    candidates.sort(key=lambda c: (c[0], c[1]))
    if strategy == "densest":
        _, _, best_origin, n = candidates[-1]
    elif strategy == "median":
        _, _, best_origin, n = candidates[len(candidates) // 2]
    else:
        raise ValueError(f"unknown strategy {strategy!r}; use 'densest' or 'median'")

    sl = tuple(slice(best_origin[d], best_origin[d] + size[d]) for d in range(v.ndim))
    crop = v[sl]
    return crop, RegionSelection(
        origin=list(best_origin),
        size=list(size),
        strategy=strategy,
        n_features=int(n),
        score=float(n) / max(1, crop.size),
    )


# =============================================================================
# Noise-floor estimation
# =============================================================================


@dataclass
class NoiseFloor:
    """Ensemble noise-floor estimate for a [0, 1]-normalised volume.

    All ``sigma_*`` fields are noise standard deviations in the volume's
    intensity units. ``psnr_max_db`` is the corresponding PSNR ceiling
    assuming a ``data_range = 1.0``.
    """

    sigma_hat: float
    """Ensemble estimate (median of available high-pass estimators)."""

    sigma_laplacian: float
    """Discrete Laplacian MAD (Immerkaer 1996, kernel-norm = ``2D(2D+1)``)."""

    sigma_haar: float
    """Haar HH-subband MAD over slice-pairs (Donoho & Johnstone 1994)."""

    sigma_background: float
    """MAD of voxels in the bottom 10% intensity percentile."""

    psnr_max_db: float
    """``-20 log10(sigma_hat)`` for [0,1] data; ``+inf`` when ``sigma_hat == 0``."""


def _laplacian_mad(V: np.ndarray) -> float:
    """Sigma estimate via the discrete-Laplacian MAD (Immerkaer 1996).

    The Laplacian kernel is the sum of axis-wise centred second
    differences. Its squared L2 norm is ``K = 2 D (2D + 1)`` where ``D``
    is the array dimensionality (``D = 3 → K = 42``).
    """
    D = V.ndim
    L = np.zeros_like(V, dtype=np.float64)
    for axis in range(D):
        forward = np.roll(V, -1, axis=axis)
        backward = np.roll(V, 1, axis=axis)
        L += forward - 2.0 * V + backward
    K = 2.0 * D * (2.0 * D + 1.0)
    mad = float(np.median(np.abs(L)))
    return mad / (0.6745 * math.sqrt(K))


def _haar_mad(V: np.ndarray) -> float:
    """Sigma estimate via the Haar HH-subband MAD over the last two axes.

    For a slice ``Y``, the HH coefficient is
    ``D[y,x] = Y[y,x] - Y[y,x+1] - Y[y+1,x] + Y[y+1,x+1]``;
    its MAD divided by ``0.6745 * 2`` is an unbiased ``sigma`` estimate
    under independent Gaussian noise. For >2D arrays, the statistic is
    pooled over all ``Y``-``X`` slices along the leading axes.
    """
    if V.ndim < 2:
        return float("nan")
    if V.ndim == 2:
        slices: List[np.ndarray] = [V]
    else:
        leading = V.shape[:-2]
        slices = [V[idx] for idx in itertools.product(*(range(s) for s in leading))]

    diffs: List[np.ndarray] = []
    for sl in slices:
        if sl.shape[0] < 2 or sl.shape[1] < 2:
            continue
        d = sl[:-1, :-1] - sl[:-1, 1:] - sl[1:, :-1] + sl[1:, 1:]
        diffs.append(d.reshape(-1))
    if not diffs:
        return float("nan")
    d_all = np.concatenate(diffs)
    mad = float(np.median(np.abs(d_all)))
    return mad / (0.6745 * 2.0)


def _background_mad(V: np.ndarray, percentile: float = 10.0) -> float:
    """Sigma estimate from voxels below the ``percentile``-th intensity.

    Sensitive to detector noise floors and pre-processing clamps; tends
    to *under*-estimate when the dark tail is quantised or clipped, so
    the ensemble takes the median across estimators rather than the mean.
    """
    threshold = float(np.percentile(V, percentile))
    bg = V[V <= threshold]
    if bg.size == 0:
        return float("nan")
    med = float(np.median(bg))
    mad = float(np.median(np.abs(bg - med)))
    return mad / 0.6745


def estimate_noise_floor(V: np.ndarray) -> NoiseFloor:
    """Three-estimator ensemble noise-floor estimate.

    Returns the median of the (Laplacian, Haar, background) estimators
    that finite-valued — robust to one outlier on the low side
    (typical when the dark tail is quantised, e.g. ``acto3d_heart_nuclei``
    in the manuscript).

    The PSNR ceiling assumes ``data_range = 1.0`` (the [0, 1]
    normalisation enforced by ``fit_gaussian_splats``). When ``sigma_hat``
    is exactly zero (saturation at float32 precision), the ceiling is
    ``+inf``; callers can clamp to a conservative finite value.
    """
    sl = _laplacian_mad(V)
    sh = _haar_mad(V)
    sb = _background_mad(V)
    candidates = [s for s in (sl, sh, sb) if not (math.isnan(s) or math.isinf(s))]
    if not candidates:
        sigma_hat = float("nan")
    else:
        sigma_hat = float(np.median(candidates))

    if sigma_hat == 0.0:
        psnr_max_db = float("inf")
    elif math.isnan(sigma_hat):
        psnr_max_db = float("nan")
    else:
        psnr_max_db = float(-20.0 * math.log10(sigma_hat))

    return NoiseFloor(
        sigma_hat=sigma_hat,
        sigma_laplacian=sl,
        sigma_haar=sh,
        sigma_background=sb,
        psnr_max_db=psnr_max_db,
    )


# =============================================================================
# K-grid construction
# =============================================================================


def build_k_grid(
    explicit: Optional[Sequence[int]] = None,
    n_points: int = 10,
    k_min: int = 1_000,
    k_max: int = 512_000,
    progression: str = "exp",
    power: int = 2,
) -> List[int]:
    """Construct a sweep grid of splat counts.

    When ``explicit`` is provided it takes precedence; otherwise
    ``n_points`` values are placed between ``k_min`` and ``k_max``
    according to ``progression``:

    * ``"exp"`` — log-spaced (geometric). Default. Matches the
      manuscript's ``{1K, 2K, ..., 512K}`` at ``n_points=10``,
      ``k_min=1000``, ``k_max=512000``.
    * ``"power"`` — polynomial: ``K_i = k_min + (k_max - k_min) *
      (i/(N-1))**power``. Denser at low K when ``power > 1``.

    Duplicates from rounding are removed but the sequence is kept
    monotonic. Endpoints are guaranteed to be exactly ``k_min`` and
    ``k_max``.
    """
    if explicit is not None:
        explicit_out = [int(k) for k in explicit]
        if len(explicit_out) < 2:
            raise ValueError(
                f"explicit grid needs at least 2 points, got {len(explicit_out)}"
            )
        if any(k <= 0 for k in explicit_out):
            raise ValueError(f"all K values must be positive, got {explicit_out}")
        return explicit_out

    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if k_min <= 0 or k_max <= 0:
        raise ValueError(f"k_min and k_max must be positive, got ({k_min}, {k_max})")
    if k_max <= k_min:
        raise ValueError(f"k_max must be > k_min, got ({k_min}, {k_max})")

    if progression == "exp":
        log_min = math.log(k_min)
        log_max = math.log(k_max)
        raw = [
            int(round(math.exp(log_min + (log_max - log_min) * i / (n_points - 1))))
            for i in range(n_points)
        ]
    elif progression == "power":
        if power < 1:
            raise ValueError(f"power must be >= 1, got {power}")
        raw = [
            int(round(k_min + (k_max - k_min) * (i / (n_points - 1)) ** power))
            for i in range(n_points)
        ]
    else:
        raise ValueError(f"unknown progression {progression!r}; use 'exp' or 'power'")

    # Pin endpoints exactly (rounding may drift)
    raw[0] = k_min
    raw[-1] = k_max

    # Deduplicate while preserving order; ensure strictly monotonic
    parametric_out: List[int] = []
    for k in raw:
        if not parametric_out or k > parametric_out[-1]:
            parametric_out.append(k)
        else:
            parametric_out.append(parametric_out[-1] + 1)
    return parametric_out


# =============================================================================
# Peak detection
# =============================================================================


@dataclass
class HeldOutPeak:
    """Detected K* and qualitative shape of the held-out PSNR curve."""

    k_star: int
    """The recommended splat count."""

    type: Literal["peak", "plateau", "signal_limited"]
    """``peak`` — clear interior maximum; ``plateau`` — flat top, smallest K
    within 0.3 dB returned; ``signal_limited`` — monotone-rising through
    the largest tested K (no peak in sampled range)."""

    confidence_db: float
    """For ``peak``: margin to the second-best K in dB.
    For ``plateau``: spread across the in-tolerance plateau.
    For ``signal_limited``: total dB rise across the sweep."""


def find_k_star(
    k_values: Sequence[int],
    held_out_psnr_values: Sequence[float],
) -> HeldOutPeak:
    """Detect the held-out PSNR peak via the manuscript's hybrid rule.

    Hybrid rule (``splat_count_vs_quality`` §4.2):

    1. **Peak**: the argmax is strictly interior AND both ``mean(pre-argmax)``
       and ``mean(post-argmax)`` are at least 0.1 dB below the peak. Return
       the argmax.
    2. **Signal-limited**: the argmax is the last K, the curve rose by
       ≥ 0.3 dB across the sweep, AND it is *still climbing at the top*
       (the last finite step is ≥ 0.1 dB). Return the last K. The tail
       check stops a flat-topped plateau (whose noisy max lands on the
       last K) from being misread as signal-limited.
    3. **Plateau**: otherwise. Return the smallest K within 0.3 dB of the
       maximum — the onset of diminishing returns.
    """
    k_arr = np.asarray(list(k_values), dtype=int)
    psnr_arr = np.asarray(list(held_out_psnr_values), dtype=float)
    if k_arr.size != psnr_arr.size:
        raise ValueError(
            f"k_values has {k_arr.size} entries but psnr has {psnr_arr.size}"
        )
    n = k_arr.size
    if n < 2:
        raise ValueError(f"need at least 2 K values, got {n}")

    finite = np.isfinite(psnr_arr)
    if not finite.any():
        raise ValueError("all held-out PSNR values are non-finite")

    argmax = int(np.argmax(np.where(finite, psnr_arr, -np.inf)))
    peak = float(psnr_arr[argmax])

    # 1. Peak
    pre = psnr_arr[:argmax]
    post = psnr_arr[argmax + 1 :]
    pre_finite = pre[np.isfinite(pre)]
    post_finite = post[np.isfinite(post)]
    if pre_finite.size > 0 and post_finite.size > 0:
        pre_gap = peak - float(np.mean(pre_finite))
        post_gap = peak - float(np.mean(post_finite))
        if pre_gap >= 0.1 and post_gap >= 0.1:
            sorted_psnr = np.sort(psnr_arr[finite])
            margin = (
                float(sorted_psnr[-1] - sorted_psnr[-2])
                if sorted_psnr.size >= 2
                else float(peak)
            )
            return HeldOutPeak(
                k_star=int(k_arr[argmax]),
                type="peak",
                confidence_db=margin,
            )

    # 2. Signal-limited: argmax is the last K, the curve rose meaningfully overall,
    #    AND it is still climbing at the top (the last finite step is not flat).
    #    The tail check prevents a flat-topped *plateau* whose noisy maximum merely
    #    lands on the last K from being mislabelled signal-limited — which would also
    #    inflate any K*-derived splat density (observed on deconvolved tile cal).
    fin_idx = np.flatnonzero(finite)
    first_finite_psnr = float(psnr_arr[fin_idx[0]])
    # "Still climbing at the top" = mean per-step rise over the trailing run of
    # *adjacent* finite K's (up to 3 steps). Adjacency guards against a NaN/inf gap
    # reading as a climb (M1); averaging guards against a single sub-0.1 dB final
    # step demoting a steadily-climbing curve to plateau (M2).
    tail_rise = 0.0
    if fin_idx.size >= 2 and int(fin_idx[-1] - fin_idx[-2]) == 1:
        run = [int(fin_idx[-1])]
        for j in range(fin_idx.size - 2, -1, -1):
            if int(fin_idx[j + 1] - fin_idx[j]) == 1 and len(run) < 4:
                run.append(int(fin_idx[j]))
            else:
                break
        seg = psnr_arr[np.array(sorted(run))]
        tail_rise = float(np.mean(np.diff(seg))) if seg.size >= 2 else 0.0
    if argmax == n - 1 and (peak - first_finite_psnr) >= 0.3 and tail_rise >= 0.1:
        return HeldOutPeak(
            k_star=int(k_arr[-1]),
            type="signal_limited",
            confidence_db=float(peak - first_finite_psnr),
        )

    # 3. Plateau
    threshold = peak - 0.3
    above_idx = np.where(np.where(finite, psnr_arr, -np.inf) >= threshold)[0]
    smallest_above = int(above_idx[0])
    plateau_spread = float(
        peak - psnr_arr[above_idx][np.isfinite(psnr_arr[above_idx])].min()
    )
    return HeldOutPeak(
        k_star=int(k_arr[smallest_above]),
        type="plateau",
        confidence_db=plateau_spread,
    )


# =============================================================================
# Transferable splat density (cal -> planner interface)
# =============================================================================


@dataclass
class SplatDensity:
    """Transferable splat budget derived from one calibration.

    The investigation found splats-to-saturate scales **sub-linearly** with
    feature content (``K ~ features^alpha``, alpha≈0.44; n_peaks the best
    predictor). This packages K* + the reference feature count + the exponent
    so any tile can get a budget via :meth:`predict_k` without re-calibrating —
    the cal→planner interface. Assumes tiles of roughly the reference scale
    ("calibrate at the scale you fit at").
    """

    feature_method: str
    n_features_reference: int
    k_star_reference: int
    saturation_exponent: float
    """Sub-linear exponent ``alpha`` in ``K ~ features^alpha`` (default 0.44)."""
    saturation_cap: int
    """Effective K beyond which the reference region overfits / plateaus."""
    splats_per_feature: float
    """Linear reference density ``k_star / n_features`` (for reporting)."""
    feature_threshold: float = 0.0
    """Absolute intensity threshold used to count ``n_features_reference``. The
    planner must scan at this same absolute level so its per-box counts are on the
    same scale as the reference (threshold-relative-to-local-max does not compose
    across regions, especially with hot outliers)."""

    def predict_k(self, n_features: int) -> int:
        """Predict the splat budget for a region with ``n_features`` features."""
        if self.n_features_reference <= 0:
            k = float(self.k_star_reference)
        else:
            ratio = max(0.0, float(n_features)) / float(self.n_features_reference)
            k = float(self.k_star_reference) * (ratio**self.saturation_exponent)
        return int(min(max(round(k), 0), self.saturation_cap))


@dataclass
class ExponentFit:
    """Multi-scale fit of the saturation exponent ``alpha`` in ``K ~ features^alpha``.

    The single-scale calibration assumes the empirical default ``alpha=0.44``;
    this *measures* it by calibrating K* at several region scales (each a
    different feature count) and regressing ``log K*`` on ``log n_features``.
    The slope is ``alpha``; the per-scale ``(n_features, k_star)`` points and the
    fit ``r_squared`` are kept for reporting / provenance.
    """

    alpha: float
    """Fitted sub-linear exponent (the regression slope in log-log space)."""
    intercept: float
    """Log-space intercept ``log C`` (so ``K = exp(intercept)·features^alpha``)."""
    r_squared: float
    """Goodness-of-fit of the log-log regression (1 = perfect power law). **NaN**
    when it cannot be assessed: fewer than 3 *distinct* feature counts (a line
    through 2 points is trivially perfect), or zero K* variance (a flat/degenerate
    fit). A NaN here means "treat the exponent as provisional"."""
    n_points: int
    """Total scales measured (before collapsing duplicate feature counts)."""
    n_distinct: int
    """Distinct feature counts actually regressed (the meaningful sample size).
    ``< 3`` ⇒ ``r_squared`` is NaN (under-determined)."""
    scales: List[int]
    """Region edge lengths (voxels), one per distinct regressed point."""
    n_features: List[int]
    """Feature count of each distinct regressed point (shared-threshold count)."""
    k_star: List[int]
    """Effective K* at each distinct regressed point."""


def fit_saturation_exponent(
    points: "Sequence[Tuple[int, float, float]]",
) -> Optional[ExponentFit]:
    """Least-squares fit of ``alpha`` in ``K ~ features^alpha`` (log-log regression).

    ``points`` is a sequence of ``(scale, n_features, k_star)`` triples (one per
    calibrated region scale). Duplicate feature counts (scales that clamped to the
    same crop) are collapsed to one point. Returns ``None`` when fewer than two
    *distinct* feature counts remain (no spread to fit a slope).

    ``r_squared`` is set to **NaN** when it cannot be meaningfully assessed —
    fewer than three distinct points (a 2-point line is always perfect), or zero
    K* variance (a degenerate flat fit, ``alpha≈0``) — so a caller's
    goodness-of-fit gate is not fooled by a structural ``R²==1.0``.
    """
    clean = [
        (int(s), float(nf), float(k))
        for (s, nf, k) in points
        if np.isfinite(nf) and np.isfinite(k) and nf > 0 and k > 0
    ]
    if len(clean) < 2:
        return None

    # Collapse duplicate feature counts (clamped/identical crops → one region):
    # keep the smallest scale and the mean K* per distinct feature count, so the
    # regression — and its reported sample size — reflect distinct evidence only.
    by_feat: Dict[float, List[Tuple[int, float]]] = {}
    for s, nf, k in clean:
        by_feat.setdefault(nf, []).append((s, k))
    distinct_feats = sorted(by_feat)
    if len(distinct_feats) < 2:
        return None  # degenerate: every scale had the same feature count

    scales_out = [min(s for s, _ in by_feat[feat]) for feat in distinct_feats]
    nf_arr = np.asarray(distinct_feats, dtype=float)
    ks_arr = np.asarray(
        [float(np.mean([k for _, k in by_feat[f]])) for f in distinct_feats],
        dtype=float,
    )

    x = np.log(nf_arr)
    y = np.log(ks_arr)
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha, intercept = float(sol[0]), float(sol[1])
    pred = alpha * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    n_distinct = len(distinct_feats)
    # A 2-point line is trivially perfect (ss_res==0); a flat fit (ss_tot==0) is
    # degenerate, not perfect. Report NaN rather than a misleading R²==1.0.
    if n_distinct < 3 or ss_tot == 0.0:
        r2 = float("nan")
    else:
        r2 = 1.0 - ss_res / ss_tot
    return ExponentFit(
        alpha=alpha,
        intercept=intercept,
        r_squared=float(r2),
        n_points=len(clean),
        n_distinct=n_distinct,
        scales=scales_out,
        n_features=[int(v) for v in nf_arr],
        k_star=[int(round(v)) for v in ks_arr],
    )


# =============================================================================
# Parametric rate-distortion model
# =============================================================================


@dataclass
class RDModel:
    """Parametric fit of held-out error vs K: ``error ≈ floor + a·K^-beta``.

    Lets us extrapolate the sweep cheaply and, crucially, flag when a curve is
    still climbing at K_max (``converged_fraction`` < 1) — the cheap detector
    that would have caught the original ``signal_limited`` false alarm without
    an expensive dense high-K sweep.
    """

    floor: float
    a: float
    beta: float
    rmse: float
    n_points: int
    converged_fraction: float
    """Fraction of the achievable error drop realised by K_max (1 = converged)."""

    def predict_error(self, k: float) -> float:
        return float(self.floor + self.a * (float(k) ** (-self.beta)))

    def k_for_error(self, target_error: float) -> float:
        """Invert the model: smallest K reaching ``target_error`` (inf if < floor)."""
        if target_error <= self.floor or self.a <= 0.0 or self.beta <= 0.0:
            return float("inf")
        return float((self.a / (target_error - self.floor)) ** (1.0 / self.beta))


def fit_rd_model(
    k_values: Sequence[float], error_values: Sequence[float]
) -> Optional[RDModel]:
    """Fit ``error ≈ floor + a·K^-beta`` (least-squares). ``None`` if <3 finite
    points or the fit fails. ``error_values`` should be held-out MSE."""
    k = np.asarray(list(k_values), dtype=float)
    e = np.asarray(list(error_values), dtype=float)
    finite = np.isfinite(k) & np.isfinite(e) & (k > 0) & (e > 0)
    k, e = k[finite], e[finite]
    if k.size < 3:
        return None
    try:
        from scipy.optimize import curve_fit
    except Exception:  # pragma: no cover - scipy is a dependency
        return None

    def _model(kk: np.ndarray, floor: float, a: float, beta: float) -> np.ndarray:
        return np.asarray(floor + a * np.power(kk, -beta), dtype=float)

    floor0 = max(0.0, float(e.min()) * 0.5)
    a0 = float(max(e.max() - floor0, 1e-12)) * (float(k.min()) ** 0.5)
    p0 = [floor0, a0, 0.5]
    bounds = ([0.0, 0.0, 1e-2], [float(e.max()) if e.max() > 0 else 1.0, np.inf, 5.0])
    try:
        popt, _ = curve_fit(_model, k, e, p0=p0, bounds=bounds, maxfev=10_000)
    except Exception:
        return None
    floor, a, beta = (float(x) for x in popt)
    pred = _model(k, floor, a, beta)
    rmse = float(np.sqrt(np.mean((pred - e) ** 2)))
    e_kmin = float(_model(np.array([k.min()]), floor, a, beta)[0])
    e_kmax = float(_model(np.array([k.max()]), floor, a, beta)[0])
    denom = e_kmin - floor
    converged = float((e_kmin - e_kmax) / denom) if denom > 1e-12 else 1.0
    return RDModel(
        floor=floor,
        a=a,
        beta=beta,
        rmse=rmse,
        n_points=int(k.size),
        converged_fraction=float(np.clip(converged, 0.0, 1.0)),
    )


# =============================================================================
# Result container
# =============================================================================


@dataclass
class CalibrationResult:
    """Output of :func:`calibrate`. Serialisable to JSON."""

    k_values_requested: List[int]
    """K values passed to the sweep."""

    k_values_effective: List[int]
    """Post-cull splat counts actually realised at each K."""

    held_out_psnr_db: List[float]
    """PSNR at masked positions, against the original pre-fill values."""

    train_psnr_db: List[float]
    """PSNR at unmasked positions, against the original values."""

    held_out_mse: List[float]
    """MSE at masked positions, against the original values."""

    full_psnr_db: List[float]
    """PSNR over the whole volume against the original — for cross-run comparison."""

    full_ssim: List[float]
    """SSIM over the whole volume against the original."""

    held_out_peak: HeldOutPeak
    """The recommended K* and curve type."""

    noise_floor: NoiseFloor
    """Ensemble noise-floor estimate for the input volume."""

    fit_times_seconds: List[float]
    """Wall-clock time per fit, in seconds."""

    splat_paths: Optional[List[str]]
    """Per-K ``.gsplats.zarr`` paths when ``--keep-fits`` is set; else ``None``."""

    mask_seed: int
    mask_fraction: float
    donut_radius: int

    fit_config: Dict[str, Any]
    """Fit kwargs that were applied (sans the per-K ``seeds`` value)."""

    volume_shape: List[int]
    volume_dtype: str
    timestamp: str

    # --- regime-robust extensions (all optional; old cal.json still loads) ---
    held_out_psnr_fg_db: List[float] = field(default_factory=list)
    """Foreground-restricted held-out PSNR (background-domination removed)."""
    held_out_gain_db: List[float] = field(default_factory=list)
    """dB the fit beats the predict-zero baseline (plateaus meaningfully)."""
    predict_zero_baseline_mse: float = float("nan")
    """MSE of the trivial all-zeros reconstruction at masked voxels."""
    k_star_metric: str = "psnr_minmax"
    """Metric used for ``held_out_peak_selected`` (psnr_minmax|psnr_foreground|gain)."""
    held_out_peak_selected: Optional[HeldOutPeak] = None
    """K* under ``k_star_metric`` (None when it equals the default psnr_minmax)."""
    calibration_region: Optional[Dict[str, Any]] = None
    """Provenance when an auto-selected sub-region was calibrated (else None)."""
    original_volume_shape: Optional[List[int]] = None
    """Shape of the full input before any region crop (disambiguates cropped PSNR)."""
    splat_density: Optional[Dict[str, Any]] = None
    """Transferable :class:`SplatDensity` (as dict) for the planner."""
    rd_model: Optional[Dict[str, Any]] = None
    """Parametric :class:`RDModel` (as dict) of held-out error vs K."""
    not_converged: bool = False
    """True when the held-out curve was still climbing at K_max (RD model)."""
    exponent_fit: Optional[Dict[str, Any]] = None
    """Multi-scale :class:`ExponentFit` (as dict) when ``cal --fit-exponent`` ran;
    its ``alpha`` is also written into ``splat_density.saturation_exponent``."""

    def to_json(self, path: Path) -> None:
        """Serialise to JSON. Non-finite floats become ``null``."""

        def _safe(v: Any) -> Any:
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return None
            if isinstance(v, np.generic):
                return _safe(v.item())
            if isinstance(v, (list, tuple)):
                return [_safe(x) for x in v]
            if isinstance(v, dict):
                return {str(k): _safe(x) for k, x in v.items()}
            return v

        data = asdict(self)
        data = _safe(data)
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "CalibrationResult":
        """Load from JSON. ``null`` floats become ``nan``."""
        raw = json.loads(Path(path).read_text())

        def _hydrate_float_list(xs: List[Any]) -> List[float]:
            return [float("nan") if x is None else float(x) for x in xs]

        peak_raw = raw["held_out_peak"]
        peak = HeldOutPeak(
            k_star=int(peak_raw["k_star"]),
            type=peak_raw["type"],
            confidence_db=float(peak_raw["confidence_db"]),
        )
        nf_raw = raw["noise_floor"]
        nf = NoiseFloor(
            sigma_hat=float(nf_raw["sigma_hat"])
            if nf_raw["sigma_hat"] is not None
            else float("nan"),
            sigma_laplacian=float(nf_raw["sigma_laplacian"])
            if nf_raw["sigma_laplacian"] is not None
            else float("nan"),
            sigma_haar=float(nf_raw["sigma_haar"])
            if nf_raw["sigma_haar"] is not None
            else float("nan"),
            sigma_background=float(nf_raw["sigma_background"])
            if nf_raw["sigma_background"] is not None
            else float("nan"),
            psnr_max_db=float(nf_raw["psnr_max_db"])
            if nf_raw["psnr_max_db"] is not None
            else float("inf"),
        )
        sel_raw = raw.get("held_out_peak_selected")
        peak_selected = (
            HeldOutPeak(
                k_star=int(sel_raw["k_star"]),
                type=sel_raw["type"],
                confidence_db=float(sel_raw["confidence_db"]),
            )
            if sel_raw
            else None
        )
        baseline = raw.get("predict_zero_baseline_mse")
        return cls(
            k_values_requested=[int(x) for x in raw["k_values_requested"]],
            k_values_effective=[int(x) for x in raw["k_values_effective"]],
            held_out_psnr_db=_hydrate_float_list(raw["held_out_psnr_db"]),
            train_psnr_db=_hydrate_float_list(raw["train_psnr_db"]),
            held_out_mse=_hydrate_float_list(raw["held_out_mse"]),
            full_psnr_db=_hydrate_float_list(raw["full_psnr_db"]),
            full_ssim=_hydrate_float_list(raw["full_ssim"]),
            held_out_peak=peak,
            noise_floor=nf,
            fit_times_seconds=_hydrate_float_list(raw["fit_times_seconds"]),
            splat_paths=raw.get("splat_paths"),
            mask_seed=int(raw["mask_seed"]),
            mask_fraction=float(raw["mask_fraction"]),
            donut_radius=int(raw["donut_radius"]),
            fit_config=dict(raw.get("fit_config", {})),
            volume_shape=[int(x) for x in raw["volume_shape"]],
            volume_dtype=str(raw["volume_dtype"]),
            timestamp=str(raw["timestamp"]),
            # --- regime-robust extensions (default-hydrated for old files) ---
            held_out_psnr_fg_db=_hydrate_float_list(raw.get("held_out_psnr_fg_db", [])),
            held_out_gain_db=_hydrate_float_list(raw.get("held_out_gain_db", [])),
            predict_zero_baseline_mse=float("nan")
            if baseline is None
            else float(baseline),
            k_star_metric=str(raw.get("k_star_metric", "psnr_minmax")),
            held_out_peak_selected=peak_selected,
            calibration_region=raw.get("calibration_region"),
            original_volume_shape=(
                [int(x) for x in raw["original_volume_shape"]]
                if raw.get("original_volume_shape") is not None
                else None
            ),
            splat_density=_rehydrate_nan_dict(raw.get("splat_density")),
            rd_model=_rehydrate_nan_dict(raw.get("rd_model")),
            not_converged=bool(raw.get("not_converged", False)),
            exponent_fit=_rehydrate_nan_dict(raw.get("exponent_fit")),
        )


# =============================================================================
# Top-level driver
# =============================================================================


ProgressCallback = Callable[[int, int, str], None]
"""``(index, total, message)`` callback emitted before/after each fit."""


def calibrate(
    V: np.ndarray,
    k_grid: Sequence[int],
    *,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    mask_seed: int = 42,
    mask_fraction: float = 0.05,
    donut_radius: int = 1,
    keep_fits: Optional[Path] = None,
    progress_callback: Optional[ProgressCallback] = None,
    k_star_metric: str = "psnr_minmax",
    feature_method: str = "peaks",
    saturation_exponent: float = 0.44,
    compute_rd_model: bool = True,
) -> CalibrationResult:
    """Run a blind-spot CV sweep over ``k_grid`` on volume ``V``.

    Pipeline (one execution per call):

    1. Generate a deterministic CV mask, donut-fill ``V`` to make
       ``V_filled``.
    2. Fit a Gaussian-splat model with ``fit_gaussian_splats`` at each
       ``K`` in ``k_grid``, using ``V_filled`` as the target. ``fit_kwargs``
       are forwarded verbatim except ``seeds`` (overridden per-K).
    3. Render each fit back to volume; compute held-out / train / full
       PSNR (and full SSIM) against the original ``V``.
    4. Estimate the noise floor on ``V``.
    5. Detect K* via :func:`find_k_star`.

    Parameters
    ----------
    V : np.ndarray
        Input volume (``ndim >= 2``). Will be passed verbatim to the
        fitter, which handles its own normalisation.
    k_grid : sequence of int
        Splat counts to evaluate.
    fit_kwargs : dict, optional
        Forwarded to ``fit_gaussian_splats`` (preset / config / device /
        cull_retention / verbose / ...). The ``seeds`` key is always
        overridden per-K.
    mask_seed, mask_fraction, donut_radius
        Mask construction parameters; the defaults match the manuscript.
    keep_fits : Path, optional
        Directory to persist the per-K ``.gsplats.zarr`` outputs. If
        ``None``, the fitted splats are not saved (memory only during
        the run).
    progress_callback : callable, optional
        Invoked as ``(i, n, msg)`` before each fit and after metrics.

    Returns
    -------
    CalibrationResult
    """
    # Lazy imports — keep module-load cheap for tests that only touch utilities
    import torch

    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.metrics import compute_psnr, compute_ssim
    from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

    if V.ndim < 2:
        raise ValueError(f"V must be at least 2D, got shape {V.shape}")
    if len(k_grid) < 1:
        raise ValueError("k_grid must contain at least one K value")

    fit_kwargs = dict(fit_kwargs or {})
    fit_kwargs.pop("seeds", None)
    # cv runs many fits — keep their internal logging quiet by default
    fit_kwargs.setdefault("verbose", False)

    if keep_fits is not None:
        keep_fits = Path(keep_fits)
        keep_fits.mkdir(parents=True, exist_ok=True)

    # 1. Mask + donut fill
    mask = cv_mask(V.shape, fraction=mask_fraction, seed=mask_seed)
    V_filled = donut_median_fill(V, mask, radius=donut_radius)

    # Pre-compute originals on torch for in-loop PSNR/SSIM (kept on CPU to
    # avoid VRAM contention with the fitter; the metric calls are O(volume)
    # and dominated by the rendered tensor's location, not this one).
    V_orig_t = torch.from_numpy(V.astype(np.float32, copy=False))
    data_range = float(V.max() - V.min())
    if data_range == 0.0:
        data_range = 1.0
    mask_t = torch.from_numpy(mask)

    # Regime-robust extras: predict-zero baseline (the "free" error floor) and a
    # foreground mask so we can strip background-domination from the held-out
    # metric. Both kept on the same device as the rendered tensor in-loop.
    baseline_mse = predict_zero_baseline_mse(V, mask)
    fg_mask = foreground_mask_otsu(V)
    fg_t = torch.from_numpy(fg_mask)

    k_values_eff: List[int] = []
    held_psnr: List[float] = []
    held_psnr_fg: List[float] = []
    held_gain: List[float] = []
    train_psnr: List[float] = []
    held_mse: List[float] = []
    full_psnr: List[float] = []
    full_ssim: List[float] = []
    fit_times: List[float] = []
    splat_paths: List[str] = []

    # Resolve the render device ONCE and hoist the loop-invariant device copies
    # (original volume + masks) out of the per-K loop — they were re-copied to the
    # GPU and never freed every iteration (M9). The held∩foreground mask and the
    # train (unmasked) mask are also invariant, so build them once.
    device_pref = fit_kwargs.get("device")
    render_device = device_pref
    if render_device in (None, "auto"):
        render_device = (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
                else "cpu"
            )
        )
    ref_dev = V_orig_t.to(render_device)
    mask_dev = mask_t.to(render_device)
    fg_dev = fg_t.to(render_device)
    train_mask_dev = ~mask_dev
    held_fg_dev = mask_dev & fg_dev

    # 2-3. Fit at each K
    for i, K in enumerate(k_grid):
        if progress_callback is not None:
            progress_callback(i, len(k_grid), f"fit K={K}")
        t0 = time.perf_counter()
        splats = fit_gaussian_splats(V_filled, seeds=int(K), **fit_kwargs)
        elapsed = time.perf_counter() - t0
        fit_times.append(elapsed)

        with torch.no_grad():
            rendered = render_to_volume_tensor(
                splats, shape=V.shape, device=render_device
            )
            held_pred = rendered[mask_dev]
            held_true = ref_dev[mask_dev]
            train_pred = rendered[train_mask_dev]
            train_true = ref_dev[train_mask_dev]

            held_mse_val = float(torch.mean((held_pred - held_true) ** 2).item())
            held_mse.append(held_mse_val)
            held_psnr_val = (
                float("inf")
                if held_mse_val == 0.0
                else float(10.0 * math.log10(data_range**2 / held_mse_val))
            )
            held_psnr.append(held_psnr_val)

            # Foreground-restricted held-out PSNR (background-domination removed)
            fg_pred = rendered[held_fg_dev]
            fg_true = ref_dev[held_fg_dev]
            if fg_pred.numel() == 0:
                held_psnr_fg.append(float("nan"))
            else:
                fg_mse = float(torch.mean((fg_pred - fg_true) ** 2).item())
                held_psnr_fg.append(
                    float("inf")
                    if fg_mse == 0.0
                    else float(10.0 * math.log10(data_range**2 / fg_mse))
                )
            # Gain over the predict-zero baseline (plateaus meaningfully)
            held_gain.append(held_out_gain_db(held_mse_val, baseline_mse))

            train_mse = float(torch.mean((train_pred - train_true) ** 2).item())
            train_psnr_val = (
                float("inf")
                if train_mse == 0.0
                else float(10.0 * math.log10(data_range**2 / train_mse))
            )
            train_psnr.append(train_psnr_val)

            full_psnr_val = compute_psnr(rendered, ref_dev, data_range=data_range)
            full_psnr.append(float(full_psnr_val))
            full_ssim_val = compute_ssim(rendered, ref_dev, data_range=data_range)
            full_ssim.append(float(full_ssim_val))

            # free only the per-iteration tensors; the hoisted device copies persist
            del rendered, held_pred, held_true, train_pred, train_true, fg_pred, fg_true

        eff_k = int(splats.n_splats)
        k_values_eff.append(eff_k)

        if keep_fits is not None:
            out_path = keep_fits / f"k{int(K):08d}.gsplats.zarr"
            # The fitter's own metrics (psnr_db, ssim, time_seconds, ...) ride
            # along via splats.stats; calibration-specific fields would be
            # dropped by GSplatData.save()'s fitting-info whitelist anyway, so
            # we keep them only in the global cal.json (which references this
            # file via splat_paths).
            splats.save(out_path, include_fitting_info=True)
            splat_paths.append(str(out_path))

        if progress_callback is not None:
            progress_callback(
                i,
                len(k_grid),
                f"K={K} eff={eff_k} train={train_psnr_val:.2f}dB held={held_psnr_val:.2f}dB t={elapsed:.1f}s",
            )

        del splats
        if str(render_device).startswith("cuda"):
            torch.cuda.empty_cache()

    # release the hoisted device copies
    del ref_dev, mask_dev, fg_dev, train_mask_dev, held_fg_dev
    if str(render_device).startswith("cuda"):
        torch.cuda.empty_cache()

    # 4. Noise floor (on the original volume; needs [0, 1] for the PSNR ceiling
    # to be meaningful — the manuscript and fit_gaussian_splats both work in
    # that range, so we pre-normalise)
    Vn = V.astype(np.float32, copy=False)
    vmin = float(Vn.min())
    vmax = float(Vn.max())
    if vmax > vmin:
        Vn_norm = (Vn - vmin) / (vmax - vmin)
    else:
        Vn_norm = Vn - vmin
    noise_floor = estimate_noise_floor(Vn_norm)

    # 5. Peak detection — default metric is min--max PSNR (manuscript behaviour,
    #    and the report/back-compat depend on `peak`). NEVER replaced.
    peak = find_k_star(list(k_grid), held_psnr)

    # 5b. Optional regime-robust K* under a different metric (purely additive).
    _metric_curves: Dict[str, List[float]] = {
        "psnr_minmax": held_psnr,
        "psnr_foreground": held_psnr_fg,
        "gain": held_gain,
    }
    if k_star_metric not in _metric_curves:
        raise ValueError(
            f"unknown k_star_metric {k_star_metric!r}; use "
            "'psnr_minmax', 'psnr_foreground', or 'gain'"
        )
    peak_selected: Optional[HeldOutPeak] = None
    if k_star_metric != "psnr_minmax":
        curve = _metric_curves[k_star_metric]
        if bool(np.isfinite(np.asarray(curve, dtype=float)).any()):
            peak_selected = find_k_star(list(k_grid), curve)

    # 5c. Parametric R-D model on held-out MSE vs effective K + convergence flag.
    rd: Optional[RDModel] = None
    not_converged = False
    if compute_rd_model:
        rd = fit_rd_model([float(k) for k in k_values_eff], held_mse)
        if rd is not None:
            # converged_fraction is structurally < 1 for any power law (a clean
            # beta~0.5 sweep lands ~0.9); only flag genuinely splat-starved curves
            # that are still steeply climbing at K_max.
            not_converged = rd.converged_fraction < 0.85

    # 5d. Transferable splat density: feature count of the calibrated volume +
    #     the (regime-appropriate) K* mapped to its effective splat count.
    selected_peak = peak_selected if peak_selected is not None else peak
    n_features = count_features(V, method=feature_method)
    try:
        star_idx = list(k_grid).index(selected_peak.k_star)
        k_star_eff = int(k_values_eff[star_idx])
    except (ValueError, IndexError):
        k_star_eff = int(selected_peak.k_star)
    # Cap per-box budgets at the largest *measured* effective K (the empirical
    # ceiling), regardless of convergence — collapsing it to k_star_eff would
    # truncate denser-than-reference tiles to the reference budget (M3).
    saturation_cap = int(k_values_eff[-1]) if k_values_eff else k_star_eff
    # Exact absolute level the detector counted at (method-aware: blurred-max for
    # peaks, Otsu for intensity, sobel-max for edges) — recorded so the planner's
    # scan_content counts on the identical scale (a naive 0.1*raw_max drifts).
    feat_thr = feature_threshold(V, feature_method)
    density = SplatDensity(
        feature_method=feature_method,
        n_features_reference=int(n_features),
        k_star_reference=k_star_eff,
        saturation_exponent=float(saturation_exponent),
        saturation_cap=int(max(saturation_cap, k_star_eff)),
        splats_per_feature=(
            float(k_star_eff) / n_features if n_features > 0 else float("nan")
        ),
        feature_threshold=feat_thr,
    )

    return CalibrationResult(
        k_values_requested=[int(k) for k in k_grid],
        k_values_effective=k_values_eff,
        held_out_psnr_db=held_psnr,
        train_psnr_db=train_psnr,
        held_out_mse=held_mse,
        full_psnr_db=full_psnr,
        full_ssim=full_ssim,
        held_out_peak=peak,
        noise_floor=noise_floor,
        fit_times_seconds=fit_times,
        splat_paths=splat_paths if keep_fits is not None else None,
        mask_seed=mask_seed,
        mask_fraction=mask_fraction,
        donut_radius=donut_radius,
        fit_config={k: v for k, v in fit_kwargs.items() if _json_safe(v)},
        volume_shape=list(V.shape),
        volume_dtype=str(V.dtype),
        timestamp=datetime.now(timezone.utc).isoformat(),
        held_out_psnr_fg_db=held_psnr_fg,
        held_out_gain_db=held_gain,
        predict_zero_baseline_mse=baseline_mse,
        k_star_metric=k_star_metric,
        held_out_peak_selected=peak_selected,
        splat_density=asdict(density),
        rd_model=asdict(rd) if rd is not None else None,
        not_converged=not_converged,
    )


def calibrate_saturation_exponent(
    V: np.ndarray,
    scales: Sequence[int],
    *,
    k_grid: Sequence[int],
    fit_kwargs: Optional[Dict[str, Any]] = None,
    feature_method: str = "peaks",
    region_strategy: str = "densest",
    k_star_metric: str = "psnr_minmax",
    mask_seed: int = 42,
    mask_fraction: float = 0.05,
    progress_callback: Optional[ProgressCallback] = None,
) -> Optional[ExponentFit]:
    """Measure ``alpha`` in ``K ~ features^alpha`` by calibrating at several scales.

    For each edge length in ``scales`` a content-rich sub-region of that size is
    selected (:func:`select_calibration_region`) and calibrated
    (:func:`calibrate`, RD model skipped) to obtain its K*. Feature counts are
    taken at a **single shared absolute level** derived from the full volume
    (:func:`_robust_feature_level`) so the per-scale counts compose — a per-crop
    relative threshold would make the slope (``alpha``) inconsistent. ``K*`` is
    detected with the same ``k_star_metric`` the caller uses for the main sweep,
    so the regressed K* and the reported anchor are the same definition. The
    points are regressed in log-log space by :func:`fit_saturation_exponent`.

    Returns the :class:`ExponentFit`, or ``None`` when fewer than two scales yield
    distinct feature counts (e.g. every scale collapsed to the whole volume).
    Runtime is roughly ``len(scales)`` × a single :func:`calibrate` sweep.
    """
    # One shared absolute feature level for ALL scales (counts compose; a per-crop
    # relative level would bias the regression slope — see select_calibration_region).
    shared_thr = _robust_feature_level(np.asarray(V), feature_method)
    points: List[Tuple[int, float, float]] = []
    n = len(scales)
    for i, scale in enumerate(scales):
        crop, _region = select_calibration_region(
            V, region_size=int(scale), strategy=region_strategy, feature=feature_method
        )
        n_feat = float(
            count_features(crop, method=feature_method, threshold_abs=shared_thr)
        )
        if progress_callback is not None:
            progress_callback(i, n, f"scale {scale}: region n_features≈{int(n_feat)}")
        result = calibrate(
            crop,
            k_grid=k_grid,
            fit_kwargs=fit_kwargs,
            mask_seed=mask_seed,
            mask_fraction=mask_fraction,
            k_star_metric=k_star_metric,
            feature_method=feature_method,
            compute_rd_model=False,
        )
        density = result.splat_density or {}
        k_star = float(density.get("k_star_reference", result.held_out_peak.k_star))
        points.append((int(scale), n_feat, k_star))
    return fit_saturation_exponent(points)


def _rehydrate_nan_dict(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Restore float-symmetry on reload: ``to_json`` writes NaN/inf as ``null``,
    so a ``None`` inside a nested ``splat_density`` / ``rd_model`` dict means a
    non-finite float. Map it back to ``nan`` so consumers (planner) don't choke
    on ``float(None)``. Keys (all-string) are never None, so this is safe."""
    if d is None:
        return None
    return {k: (float("nan") if v is None else v) for k, v in d.items()}


def _json_safe(v: Any) -> bool:
    """Return True iff ``v`` survives ``json.dumps`` (used to filter fit_config)."""
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


__all__ = [
    "CalibrationResult",
    "HeldOutPeak",
    "NoiseFloor",
    "ProgressCallback",
    "RDModel",
    "RegionSelection",
    "SplatDensity",
    "build_k_grid",
    "calibrate",
    "count_features",
    "cv_mask",
    "donut_median_fill",
    "estimate_noise_floor",
    "find_k_star",
    "fit_rd_model",
    "foreground_mask_otsu",
    "held_out_gain_db",
    "held_out_psnr",
    "held_out_psnr_foreground",
    "predict_zero_baseline_mse",
    "select_calibration_region",
]
