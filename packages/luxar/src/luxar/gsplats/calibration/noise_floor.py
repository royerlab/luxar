"""Ensemble noise-floor estimation and background-pedestal estimation.

A free byproduct of the calibration sweep: a three-estimator (Laplacian +
Haar HH + background MAD) noise-floor estimate that places each dataset in
absolute terms, plus the DC-offset / floor estimator used before fitting.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import List

import numpy as np


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


@dataclass(frozen=True)
class FloorEstimate:
    """Resolved floor level and the estimator branch that produced it."""

    level: float
    strategy: str


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


def _histogram_mode(values: np.ndarray) -> float:
    hi = float(np.percentile(values, 95.0))
    low_band = values[values <= hi].astype(np.float64, copy=False)
    hist, edges = np.histogram(low_band, bins=512)
    index = int(hist.argmax())
    return 0.5 * (float(edges[index]) + float(edges[index + 1]))


def _otsu_threshold(values: np.ndarray) -> float:
    hist, edges = np.histogram(values.astype(np.float64, copy=False), bins=512)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weights = np.cumsum(hist, dtype=np.float64)
    moments = np.cumsum(hist * centers, dtype=np.float64)
    total_weight = weights[-1]
    total_moment = moments[-1]
    denominator = weights[:-1] * (total_weight - weights[:-1])
    score = np.zeros_like(denominator)
    valid = denominator > 0.0
    score[valid] = (
        total_moment * weights[:-1][valid] - moments[:-1][valid] * total_weight
    ) ** 2 / denominator[valid]
    return float(edges[int(score.argmax()) + 1])


def _specimen_floor(values: np.ndarray, auto_level: float) -> FloorEstimate:
    hi = float(np.percentile(values, 95.0))
    low_band = values[values <= hi]
    threshold = _otsu_threshold(low_band)
    lower = low_band[low_band <= threshold]
    upper = low_band[low_band > threshold]
    min_class_size = max(2, int(math.ceil(0.02 * low_band.size)))
    if lower.size < min_class_size or upper.size < min_class_size:
        return FloorEstimate(auto_level, "specimen-fallback-auto")

    lower_median = float(np.median(lower))
    upper_median = float(np.median(upper))
    separation = upper_median - lower_median
    lower_mad = float(np.median(np.abs(lower - lower_median)))
    upper_mad = float(np.median(np.abs(upper - upper_median)))
    if separation <= 0.0 or max(lower_mad, upper_mad) > 0.1 * separation:
        return FloorEstimate(auto_level, "specimen-fallback-auto")

    candidate = _histogram_mode(upper)
    if not threshold < candidate <= hi:
        return FloorEstimate(auto_level, "specimen-fallback-auto")
    return FloorEstimate(candidate, "specimen")


def estimate_floor_result(V: np.ndarray, method: str = "mode") -> FloorEstimate:
    """Estimate a floor and report which estimator branch produced it."""
    V = np.asarray(V)
    values = V[V != 0.0] if np.any(V != 0.0) else V
    if values.size == 0:
        return FloorEstimate(float(np.min(V)), method)
    if method == "percentile":
        return FloorEstimate(float(np.percentile(values, 10.0)), "percentile")
    if method not in ("mode", "specimen"):
        raise ValueError(f"estimate_floor: unknown method {method!r}")
    mode = float(min(_histogram_mode(values), float(np.median(values))))
    if method == "specimen":
        return _specimen_floor(values, mode)
    return FloorEstimate(mode, "mode")


def floor_strategy_for(
    V: np.ndarray, floor: "str | float | None", applied_floor: "float | None"
) -> "str | None":
    """Return specimen estimator provenance only when a floor was applied."""
    if applied_floor is None or not isinstance(floor, str):
        return None
    if floor.strip().lower() != "specimen":
        return None
    return estimate_floor_result(V, method="specimen").strategy


def estimate_floor(V: np.ndarray, method: str = "mode") -> float:
    """Estimate the background pedestal / DC offset to subtract before fitting.

    A constant background is the worst case for a localized Gaussian-splat
    basis, so subtracting it before normalisation is the single
    highest-leverage preprocessing step on real microscopy.

    Parameters
    ----------
    V : np.ndarray
        Input volume (any shape / dtype convertible to float).
    method : {"mode", "specimen", "percentile"}
        ``"mode"`` (default): histogram mode of the low-intensity bulk (the
        pedestal peak), capped at the median so an image that is *mostly*
        signal can never have real signal subtracted. On clean data with no
        pedestal ``mode ≈ min(V)`` → effectively a no-op → backward-compatible.
        ``"percentile"``: the 10th intensity percentile (cheaper; matches the
        :func:`_background_mad` threshold).
        ``"specimen"``: opt-in bimodal-background mode. Otsu splits the same
        sub-p95 low band used by ``mode``; when both populations are compact and
        separated, the upper population's mode is returned. Otherwise it falls
        back to ``mode``.

    Notes
    -----
    Exact-zero voxels (masked / out-of-FOV padding) are excluded so padding
    does not dominate the histogram. This function materializes ``V``; use
    :func:`luxar.gsplats.fitting.preprocessing.resolve_volume_floor` for a lazy
    whole volume.
    """
    return estimate_floor_result(V, method).level
