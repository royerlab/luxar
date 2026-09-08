"""Held-out reconstruction metrics for the calibration sweep.

The min--max-range held-out PSNR is correct for raw/noisy data at a
manageable scale (the manuscript regime), but it is background-dominated on
large sparse volumes: a trivial predict-zero reconstruction already scores
40-60 dB. The gain-over-baseline, foreground-restricted, and foreground-weighted
variants here are regime-robust alternatives; all are *additive* — the default
K* selection still uses the min--max metric.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _psnr_db(mse: float, data_range: float) -> float:
    """PSNR in dB from an MSE and a dynamic range.

    Single source of truth for the ``10·log10(data_range²/mse)`` formula used
    throughout the calibration sweep. ``+inf`` when either the MSE or the
    dynamic range is zero (a perfect / degenerate reconstruction).
    """
    if mse == 0.0:
        return float("inf")
    if data_range == 0.0:
        return float("inf")
    return float(10.0 * math.log10(data_range**2 / mse))


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
    return _psnr_db(mse, data_range)


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
    predicting zeros". Because the baseline is constant across K, this curve
    differs from min--max PSNR only by a constant and selects the same K*.
    """
    if not math.isfinite(baseline_mse) or baseline_mse <= 0.0:
        return float("nan")
    if held_mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10(baseline_mse / held_mse))


def held_out_psnr_fg_weighted(
    V_hat: np.ndarray,
    V_original: np.ndarray,
    held_mask: np.ndarray,
    foreground_mask: np.ndarray,
    fg_bg_ratio: float = 1.0,
    data_range: Optional[float] = None,
) -> float:
    """Held-out PSNR with controlled foreground/background total weight.

    Foreground voxels receive unit weight. Background voxels receive
    ``n_fg / (fg_bg_ratio * n_bg)`` using counts from the held-out subset, so
    ``fg_bg_ratio=1`` gives the two strata exactly equal total weight. Returns
    ``nan`` when either held-out stratum is empty.
    """
    if not math.isfinite(fg_bg_ratio) or fg_bg_ratio <= 0.0:
        raise ValueError(f"fg_bg_ratio must be finite and > 0, got {fg_bg_ratio}")
    if V_hat.shape != V_original.shape:
        raise ValueError(
            f"shape mismatch: V_hat {V_hat.shape} vs V_original {V_original.shape}"
        )
    if held_mask.shape != V_original.shape or foreground_mask.shape != V_original.shape:
        raise ValueError("held and foreground masks must match the volume shape")

    held_fg = held_mask & foreground_mask
    held_bg = held_mask & ~foreground_mask
    n_fg = int(np.count_nonzero(held_fg))
    n_bg = int(np.count_nonzero(held_bg))
    if n_fg == 0 or n_bg == 0:
        return float("nan")

    fg_error = np.asarray(V_hat[held_fg] - V_original[held_fg], dtype=np.float64)
    bg_error = np.asarray(V_hat[held_bg] - V_original[held_bg], dtype=np.float64)
    bg_weight = n_fg / (fg_bg_ratio * n_bg)
    weighted_mse = float(
        (np.sum(fg_error**2) + bg_weight * np.sum(bg_error**2))
        / (n_fg + bg_weight * n_bg)
    )
    if data_range is None:
        data_range = float(V_original.max() - V_original.min())
    return _psnr_db(weighted_mse, data_range)


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
    return _psnr_db(mse, data_range)
