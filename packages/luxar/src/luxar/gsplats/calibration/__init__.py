"""Blind-spot cross-validation for Gaussian-splat model selection.

Implements the manuscript's calibration protocol (Supp. Doc. 2,
``splat_count_vs_quality``):

1. Mask 5% of voxels with a deterministic Bernoulli draw (seed=42).
2. Replace masked voxels with the median of the unmasked voxels in their
   3^D donut neighbourhood (centre excluded) — Noise2Self self-supervision.
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
unchanged at each ``K``; this package owns mask generation, donut fill,
held-out evaluation, K-grid construction, peak detection, and noise-floor
estimation.

This module is a package split by phase (masking / metrics / content /
noise_floor / curve_analysis / result / driver); it re-exports the full public
surface so ``luxar.gsplats.calibration.X`` keeps resolving as before.
"""

from __future__ import annotations

from .content import (
    RegionSelection,
    _otsu_threshold,
    _robust_feature_level,
    count_features,
    feature_threshold,
    foreground_mask_otsu,
    foreground_mask_otsu_smoothed,
    select_calibration_region,
)
from .curve_analysis import (
    ExponentFit,
    HeldOutPeak,
    RDModel,
    SplatDensity,
    build_k_grid,
    find_k_star,
    fit_rd_model,
    fit_saturation_exponent,
)
from .driver import (
    ProgressCallback,
    calibrate,
    calibrate_saturation_exponent,
)
from .masking import cv_mask, donut_median_fill
from .metrics import (
    held_out_gain_db,
    held_out_psnr,
    held_out_psnr_fg_weighted,
    held_out_psnr_foreground,
    predict_zero_baseline_mse,
)
from .noise_floor import (
    FloorEstimate,
    NoiseFloor,
    estimate_floor,
    estimate_floor_result,
    estimate_noise_floor,
)
from .result import CalibrationResult, _json_safe, _rehydrate_nan_dict

__all__ = [
    "CalibrationResult",
    "ExponentFit",
    "FloorEstimate",
    "HeldOutPeak",
    "NoiseFloor",
    "ProgressCallback",
    "RDModel",
    "RegionSelection",
    "SplatDensity",
    "_json_safe",
    "_otsu_threshold",
    "_rehydrate_nan_dict",
    "_robust_feature_level",
    "build_k_grid",
    "calibrate",
    "calibrate_saturation_exponent",
    "count_features",
    "cv_mask",
    "donut_median_fill",
    "estimate_floor",
    "estimate_floor_result",
    "estimate_noise_floor",
    "feature_threshold",
    "find_k_star",
    "fit_rd_model",
    "fit_saturation_exponent",
    "foreground_mask_otsu",
    "foreground_mask_otsu_smoothed",
    "held_out_gain_db",
    "held_out_psnr",
    "held_out_psnr_fg_weighted",
    "held_out_psnr_foreground",
    "predict_zero_baseline_mse",
    "select_calibration_region",
]
