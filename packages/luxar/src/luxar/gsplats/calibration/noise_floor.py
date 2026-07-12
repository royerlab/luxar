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


def estimate_floor(V: np.ndarray, method: str = "mode") -> float:
    """Estimate the background pedestal / DC offset to subtract before fitting.

    A constant background is the worst case for a localized Gaussian-splat
    basis, so subtracting it before normalisation is the single
    highest-leverage preprocessing step on real microscopy (see
    ``docs/handoffs/floor-suppression-handoff.md``).

    Parameters
    ----------
    V : np.ndarray
        Input volume (any shape / dtype convertible to float).
    method : {"mode", "percentile"}
        ``"mode"`` (default): histogram mode of the low-intensity bulk (the
        pedestal peak), capped at the median so an image that is *mostly*
        signal can never have real signal subtracted. On clean data with no
        pedestal ``mode ≈ min(V)`` → effectively a no-op → backward-compatible.
        ``"percentile"``: the 10th intensity percentile (cheaper; matches the
        :func:`_background_mad` threshold).

    Notes
    -----
    Exact-zero voxels (masked / out-of-FOV padding) are excluded so padding
    does not dominate the histogram.
    """
    Vf = V[V != 0.0] if np.any(V != 0.0) else V
    if Vf.size == 0:
        return float(np.min(V))
    if method == "percentile":
        return float(np.percentile(Vf, 10.0))
    if method != "mode":
        raise ValueError(f"estimate_floor: unknown method {method!r}")
    hi = float(np.percentile(Vf, 95.0))
    hist, edges = np.histogram(Vf[Vf <= hi], bins=512)
    i = int(hist.argmax())
    mode = 0.5 * (float(edges[i]) + float(edges[i + 1]))
    return float(min(mode, float(np.median(Vf))))
