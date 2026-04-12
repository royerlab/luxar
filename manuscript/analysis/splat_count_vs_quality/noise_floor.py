#!/usr/bin/env python3
"""Noise floor estimation for microscopy volumes.

Provides multiple robust estimators of the noise standard deviation σ,
from which the theoretical PSNR ceiling is derived:

    PSNR_max = -20 * log10(σ)    (for data normalised to [0, 1])

Methods:
  1. Laplacian MAD (Immerkaer-style) — robust to signal edges
  2. Haar finite-difference MAD — no dependencies, equivalent to Haar wavelet HH
  3. Wavelet MAD (optional, requires pywt) — Donoho & Johnstone estimator
  4. Background region MAD — uses lowest-intensity voxels

The ensemble estimate is the median of all available methods.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

# Consistent MAD-to-sigma conversion factor (assuming Gaussian noise)
_MAD_SIGMA = 0.6745


# ---------------------------------------------------------------------------
# Individual estimators
# ---------------------------------------------------------------------------


def estimate_noise_sigma_laplacian(volume: np.ndarray) -> float:
    """Immerkaer-style noise estimation via discrete Laplacian + MAD.

    The Laplacian is a high-pass filter that amplifies noise relative to
    smooth signal. Using MAD instead of variance makes it robust to strong
    edges in the image.
    """
    from scipy.ndimage import laplace

    L = laplace(volume.astype(np.float64))

    # Kernel sum-of-squares for the discrete Laplacian in nD:
    # The kernel has -2*ndim at center and +1 at each face-adjacent neighbor.
    # sum(k_i^2) = (2*ndim)^2 + 2*ndim*1^2 = 4*ndim^2 + 2*ndim = 2*ndim*(2*ndim+1)
    ndim = volume.ndim
    kernel_ss = 2 * ndim * (2 * ndim + 1)

    # Robust sigma estimate via MAD
    mad = float(np.median(np.abs(L - np.median(L))))
    sigma_L = mad / _MAD_SIGMA
    sigma = sigma_L / math.sqrt(kernel_ss)

    return sigma


def estimate_noise_sigma_mad_haar(volume: np.ndarray) -> float:
    """Haar-equivalent finite-difference MAD estimator (no dependencies).

    For each z-slice, computes the 2D diagonal detail:
        d[y,x] = img[y,x] - img[y,x+1] - img[y+1,x] + img[y+1,x+1]

    This is algebraically equivalent to the Haar wavelet HH subband * 2.
    For noise N(0, σ²), var(d) = 4σ², so σ = MAD(|d|) / 0.6745 / 2.
    """
    details = []
    for z in range(volume.shape[0]):
        sl = volume[z].astype(np.float64)
        d = sl[:-1, :-1] - sl[:-1, 1:] - sl[1:, :-1] + sl[1:, 1:]
        details.append(d.ravel())

    all_d = np.concatenate(details)
    mad = float(np.median(np.abs(all_d)))
    sigma = mad / _MAD_SIGMA / 2.0

    return sigma


def estimate_noise_sigma_mad_wavelet(volume: np.ndarray) -> float | None:
    """Donoho & Johnstone MAD estimator using finest wavelet coefficients.

    Applies 2D Haar (db1) wavelet decomposition per z-slice and collects
    the HH (diagonal detail) coefficients. Returns None if pywt is not
    installed.
    """
    try:
        import pywt
    except ImportError:
        return None

    all_hh = []
    for z in range(volume.shape[0]):
        sl = volume[z].astype(np.float64)
        _, (_, _, hh) = pywt.dwt2(sl, "db1")
        all_hh.append(hh.ravel())

    coeffs = np.concatenate(all_hh)
    sigma = float(np.median(np.abs(coeffs)) / _MAD_SIGMA)

    return sigma


def estimate_noise_sigma_background(
    volume: np.ndarray,
    percentile: float = 10.0,
    min_voxels: int = 1000,
) -> float | None:
    """Estimate noise from the lowest-intensity (background) voxels.

    Assumes that voxels below the given percentile are pure noise /
    background. Returns None if too few background voxels are found.
    """
    threshold = np.percentile(volume, percentile)
    bg_mask = volume <= threshold

    if bg_mask.sum() < min_voxels:
        return None

    bg = volume[bg_mask].astype(np.float64)
    mad = float(np.median(np.abs(bg - np.median(bg))))
    sigma = mad / _MAD_SIGMA

    return sigma


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


def estimate_noise_floor(
    volume: np.ndarray, data_range: float = 1.0
) -> dict[str, Any]:
    """Estimate the noise floor using an ensemble of robust methods.

    Parameters
    ----------
    volume : np.ndarray
        3D float32 volume, normalised to [0, 1].
    data_range : float
        Dynamic range of the data (default: 1.0 for normalised data).

    Returns
    -------
    dict with keys:
        sigma_laplacian, sigma_mad_haar, sigma_mad_wavelet (or None),
        sigma_background (or None), sigma_ensemble, noise_variance,
        psnr_max_db, methods_used (list of str).
    """
    s_lap = estimate_noise_sigma_laplacian(volume)
    s_haar = estimate_noise_sigma_mad_haar(volume)
    s_wav = estimate_noise_sigma_mad_wavelet(volume)
    s_bg = estimate_noise_sigma_background(volume)

    # Collect all valid estimates
    estimates = {"laplacian": s_lap, "mad_haar": s_haar}
    if s_wav is not None:
        estimates["mad_wavelet"] = s_wav
    if s_bg is not None:
        estimates["background"] = s_bg

    # Ensemble: median of all estimates
    sigma_ensemble = float(np.median(list(estimates.values())))

    # Guard against zero noise
    if sigma_ensemble <= 0:
        psnr_max = float("inf")
        noise_var = 0.0
    else:
        noise_var = sigma_ensemble ** 2
        psnr_max = -20.0 * math.log10(sigma_ensemble / data_range)

    return {
        "sigma_laplacian": s_lap,
        "sigma_mad_haar": s_haar,
        "sigma_mad_wavelet": s_wav,
        "sigma_background": s_bg,
        "sigma_ensemble": sigma_ensemble,
        "noise_variance": noise_var,
        "psnr_max_db": psnr_max,
        "methods_used": sorted(estimates.keys()),
    }


# ---------------------------------------------------------------------------
# TSV loader (used by plot scripts)
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"


def load_noise_floor(dataset_key: str) -> dict[str, float] | None:
    """Load noise floor for a dataset from the shared TSV.

    Returns dict with psnr_max_db, noise_variance, sigma_ensemble,
    or None if not available.
    """
    import pandas as pd

    tsv_path = RESULTS_DIR / "noise_floor.tsv"
    if not tsv_path.exists():
        return None

    df = pd.read_csv(tsv_path, sep="\t")
    row = df[df["dataset"] == dataset_key]
    if row.empty:
        return None

    return {
        "psnr_max_db": float(row.iloc[0]["psnr_max_db"]),
        "noise_variance": float(row.iloc[0]["noise_variance"]),
        "sigma_ensemble": float(row.iloc[0]["sigma_ensemble"]),
    }
