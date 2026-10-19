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

_SPECIMEN_MIN_CLASS_FRACTION = 0.02
_SPECIMEN_MAX_MAD_SEPARATION_RATIO = 0.10
_SPECIMEN_MAX_LOWER_P90_DEVIATION_RATIO = 0.20
_SPECIMEN_MAX_MODE_MAD_MULTIPLE = 4.0


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


#: Leading-axis slabs of :func:`estimate_masked_noise` are sized to about this
#: many voxels, so the high-pass temporaries stay bounded on a large reference.
_MASKED_SIGMA_SLAB_VOXELS = 1 << 24

#: Local-intensity bins the in-mask sigma is pooled over (shot noise grows with
#: the signal, so one median under-reads the RMS sigma an MSE averages).
_MASKED_SIGMA_BINS = 8

#: Fewest blocks a bin needs to contribute its own MAD.
_MASKED_SIGMA_MIN_BIN = 50

#: Above this ratio of the coarse-scale to the fine-scale sigma, the fine Haar
#: band holds signal, not just noise, and the estimate is not trusted. White
#: noise gives 1.0 (2x2 averaging halves sigma, and the coarse value is doubled
#: back); a smooth specimen stays within ~1.15 (A549 CTC t000: 1.16), while
#: structure near the voxel scale pushes it to 1.5-4. Structure finer than ~0.7
#: voxel in-plane (sparse undersampled puncta, hot pixels) has a flat spectrum
#: and keeps the ratio near 1: the check cannot see it.
MASKED_SIGMA_MAX_SCALE_RATIO = 1.3


@dataclass(frozen=True)
class MaskedNoise:
    """Noise estimate inside a mask (:func:`estimate_masked_noise`).

    Attributes:
        sigma: RMS noise sigma, pooled over local-intensity bins (``nan`` when
            undetermined).
        sigma_median: The single foreground MAD sigma (no binning).
        scale_ratio: Coarse-scale (2x2-averaged, rescaled) over fine-scale
            sigma; ~1 when the finest band is noise only.
        reliable: ``scale_ratio`` is finite and at most
            :data:`MASKED_SIGMA_MAX_SCALE_RATIO`.
        n_blocks: 2x2 blocks the estimate used (after subsampling).
    """

    sigma: float
    sigma_median: float
    scale_ratio: float
    reliable: bool
    n_blocks: int


def _block_bands(
    V: np.ndarray, M: np.ndarray, stride: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """In-mask HH coefficients and 2x2 means over ``(planes, y, x)`` slabs.

    Every ``stride``-th selected block of each slab is kept, so the returned
    arrays never exceed the caller's sample budget. Also returns the number of
    blocks selected before subsampling.
    """
    per_plane = V.shape[1] * V.shape[2]
    slab = max(1, _MASKED_SIGMA_SLAB_VOXELS // per_plane)
    hh: List[np.ndarray] = []
    ll: List[np.ndarray] = []
    total = 0
    for lo in range(0, V.shape[0], slab):
        sel = M[lo : lo + slab, :-1, :-1]
        n_sel = int(np.count_nonzero(sel))
        if not n_sel:
            continue
        total += n_sel
        Y = V[lo : lo + slab].astype(np.float32, copy=False)
        a, b = Y[:, :-1, :-1], Y[:, :-1, 1:]
        c, d = Y[:, 1:, :-1], Y[:, 1:, 1:]
        hh.append((a - b - c + d)[sel][::stride])
        ll.append(((a + b + c + d) * 0.25)[sel][::stride])
    if not hh:
        empty = np.empty(0, np.float32)
        return empty, empty, 0
    return np.concatenate(hh), np.concatenate(ll), total


def _mad_sigma(values: np.ndarray) -> float:
    """HH sigma from ``values``; a std when quantised data make the MAD zero."""
    mad = float(np.median(np.abs(values)))
    if mad > 0.0:
        return mad / (0.6745 * 2.0)
    return float(np.std(values)) / 2.0


def _binned_sigma(hh: np.ndarray, ll: np.ndarray) -> float:
    """RMS of per-intensity-bin MAD sigmas, weighted by block count."""
    edges = np.quantile(ll, np.linspace(0.0, 1.0, _MASKED_SIGMA_BINS + 1))
    idx = np.clip(
        np.searchsorted(edges, ll, side="right") - 1, 0, _MASKED_SIGMA_BINS - 1
    )
    var, count = 0.0, 0
    for k in range(_MASKED_SIGMA_BINS):
        values = hh[idx == k]
        if values.size < _MASKED_SIGMA_MIN_BIN:
            continue
        var += values.size * _mad_sigma(values) ** 2
        count += values.size
    return math.sqrt(var / count) if count else _mad_sigma(hh)


def _sample_stride(mask3: np.ndarray, max_samples: int) -> int:
    n = int(np.count_nonzero(mask3[:, :-1, :-1]))
    return max(1, -(-n // max(1, max_samples)))


def _coarse_hh(V: np.ndarray, M: np.ndarray, max_samples: int) -> np.ndarray:
    """Build 2x2-averaged planes one slab at a time for the scale check."""
    ny, nx = (V.shape[1] // 2) * 2, (V.shape[2] // 2) * 2
    coarse_m = M[:, :ny:2, :nx:2]
    stride = _sample_stride(coarse_m, max_samples)
    slab = max(1, _MASKED_SIGMA_SLAB_VOXELS // ((ny // 2) * (nx // 2)))
    bands: List[np.ndarray] = []
    for lo in range(0, V.shape[0], slab):
        fine = V[lo : lo + slab, :ny, :nx].astype(np.float32, copy=False)
        coarse_v = fine.reshape(fine.shape[0], ny // 2, 2, nx // 2, 2).mean(axis=(2, 4))
        hh, _, _ = _block_bands(coarse_v, coarse_m[lo : lo + slab], stride)
        if hh.size:
            bands.append(hh)
    return np.concatenate(bands) if bands else np.empty(0, np.float32)


def estimate_masked_noise(
    V: np.ndarray, mask: np.ndarray, *, max_samples: int = 4_000_000
) -> MaskedNoise:
    """Noise sigma inside ``mask`` from the in-plane Haar HH band, with a check.

    For the question "how large is the noise WHERE the signal is".
    Fluorescence noise is signal-dependent (shot noise), so the foreground's
    sigma is larger than the background's, and a whole-volume estimate
    (dominated by the background) under-reads it. Each 2x2 in-plane block whose
    top-left voxel is in ``mask`` contributes ``Y[y,x] - Y[y,x+1] - Y[y+1,x] +
    Y[y+1,x+1]``, whose MAD / (0.6745 * 2) is sigma under independent noise. The
    blocks are binned by local intensity and the per-bin sigmas pooled as an
    RMS, which is what an MSE averages. In-plane only: a coarse axial step would
    leak structure into a z difference.

    The HH band is also signal wherever the specimen has structure near the
    voxel scale (spots at the PSF size, sub-voxel puncta), and then the
    estimate reads that structure as noise. ``scale_ratio`` exposes it: the same
    sigma measured on the 2x2-averaged planes (times 2) matches the fine one for
    noise and exceeds it when the band holds signal; above
    :data:`MASKED_SIGMA_MAX_SCALE_RATIO` the result is flagged unreliable. The
    check is conservative (a clean, smooth, high-SNR specimen is often flagged
    too) and blind to structure finer than ~0.7 voxel in-plane, whose flat
    spectrum keeps the ratio near 1 while sigma reads several times too high.

    ``mask`` has ``V``'s shape; at most ``max_samples`` blocks are used (evenly
    strided per slab, so memory stays bounded). ``sigma`` is ``nan`` when ``V``
    has fewer than two axes or the mask holds no complete block.
    """
    nan = float("nan")
    undetermined = MaskedNoise(nan, nan, nan, False, 0)
    if V.ndim < 2 or V.shape[-1] < 2 or V.shape[-2] < 2:
        return undetermined
    if mask.shape != V.shape:
        raise ValueError(f"mask shape {mask.shape} does not match {V.shape}")
    V3 = V.reshape((-1,) + V.shape[-2:])
    M3 = mask.reshape((-1,) + mask.shape[-2:]).astype(bool, copy=False)
    hh, ll, _ = _block_bands(V3, M3, _sample_stride(M3, max_samples))
    if hh.size == 0:
        return undetermined
    fine = _mad_sigma(hh)
    sigma = _binned_sigma(hh, ll)
    ratio = nan
    ny, nx = (V3.shape[1] // 2) * 2, (V3.shape[2] // 2) * 2
    if ny >= 4 and nx >= 4 and fine > 0.0:
        coarse = _coarse_hh(V3, M3, max_samples)
        if coarse.size:
            ratio = 2.0 * _mad_sigma(coarse) / fine
    reliable = bool(math.isfinite(ratio) and ratio <= MASKED_SIGMA_MAX_SCALE_RATIO)
    return MaskedNoise(sigma, fine, ratio, reliable, int(hh.size))


def masked_noise_sigma(
    V: np.ndarray, mask: np.ndarray, *, max_samples: int = 4_000_000
) -> float:
    """``estimate_masked_noise(V, mask).sigma``: the in-mask RMS noise sigma."""
    return estimate_masked_noise(V, mask, max_samples=max_samples).sigma


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
    # Float64 is mandatory: 512 bins can be narrower than float32 spacing
    # for a narrow high-valued band, which NumPy rejects (#1671).
    low_band = values[values <= hi].astype(np.float64, copy=False)
    hist, edges = np.histogram(low_band, bins=512)
    index = int(hist.argmax())
    return 0.5 * (float(edges[index]) + float(edges[index + 1]))


def _otsu_threshold(values: np.ndarray) -> float:
    # Same float32-spacing constraint as _histogram_mode (#1671).
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
    signal_threshold = _otsu_threshold(values)
    low_band = values[values <= signal_threshold]
    threshold = _otsu_threshold(low_band)
    lower = low_band[low_band <= threshold]
    upper = low_band[low_band > threshold]
    min_class_size = max(
        2, int(math.ceil(_SPECIMEN_MIN_CLASS_FRACTION * low_band.size))
    )
    if lower.size < min_class_size or upper.size < min_class_size:
        return FloorEstimate(auto_level, "specimen-fallback-auto")

    lower_median = float(np.median(lower))
    upper_median = float(np.median(upper))
    separation = upper_median - lower_median
    lower_mad = float(np.median(np.abs(lower - lower_median)))
    upper_mad = float(np.median(np.abs(upper - upper_median)))
    lower_tail = float(np.percentile(np.abs(lower - lower_median), 90.0))
    if (
        separation <= 0.0
        or max(lower_mad, upper_mad) > _SPECIMEN_MAX_MAD_SEPARATION_RATIO * separation
        or lower_tail > _SPECIMEN_MAX_LOWER_P90_DEVIATION_RATIO * separation
    ):
        return FloorEstimate(auto_level, "specimen-fallback-auto")

    candidate = _histogram_mode(upper)
    candidate_ceiling = upper_median + _SPECIMEN_MAX_MODE_MAD_MULTIPLE * upper_mad
    if not threshold < candidate <= candidate_ceiling:
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
        ``"specimen"``: opt-in bimodal-background mode. Otsu first excludes the
        bright signal class, then splits the remaining background band; when
        both populations are compact and separated, the upper population's mode
        is returned. Otherwise it falls back to ``mode``.

    Notes
    -----
    Exact-zero voxels (masked / out-of-FOV padding) are excluded so padding
    does not dominate the histogram. This function materializes ``V``; use
    :func:`luxar.gsplats.fitting.preprocessing.resolve_volume_floor` for a lazy
    whole volume.
    """
    return estimate_floor_result(V, method).level
