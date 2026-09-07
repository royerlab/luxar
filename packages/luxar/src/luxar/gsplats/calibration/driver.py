"""Top-level calibration drivers.

:func:`calibrate` runs one blind-spot CV sweep over a K-grid; it wires together
the masking, metric, content, noise-floor, and curve-analysis pieces.
:func:`calibrate_saturation_exponent` runs the sweep at several region scales to
measure the density exponent ``alpha``.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .content import (
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
    find_k_star,
    fit_rd_model,
    fit_saturation_exponent,
)
from .masking import cv_mask, donut_median_fill
from .metrics import _psnr_db, held_out_gain_db, predict_zero_baseline_mse
from .noise_floor import estimate_noise_floor
from .result import CalibrationResult, _json_safe

ProgressCallback = Callable[[int, int, str], None]
"""``(index, total, message)`` callback emitted before/after each fit."""

_K_STAR_METRICS = frozenset(
    {"psnr_minmax", "psnr_foreground", "psnr_fg_weighted", "gain"}
)


def _foreground_weighting(
    mask_dev: Any, foreground_dev: Any, fg_bg_ratio: float
) -> Tuple[Any, Any, int, int, float]:
    """Build held-out stratum selectors and their background weight."""
    foreground = foreground_dev[mask_dev]
    background = ~foreground
    n_foreground = int(foreground.sum().item())
    n_background = int(background.sum().item())
    weight = (
        n_foreground / (fg_bg_ratio * n_background)
        if n_foreground > 0 and n_background > 0
        else float("nan")
    )
    return foreground, background, n_foreground, n_background, weight


def _validate_fg_bg_ratio(fg_bg_ratio: float) -> None:
    """Reject ratios that cannot define finite positive stratum weights."""
    if not np.isfinite(fg_bg_ratio) or fg_bg_ratio <= 0.0:
        raise ValueError(f"fg_bg_ratio must be finite and > 0, got {fg_bg_ratio}")


def _foreground_weighted_psnr(
    errors: Any,
    foreground: Any,
    background: Any,
    n_foreground: int,
    n_background: int,
    background_weight: float,
    data_range: float,
) -> float:
    """Reduce held-out squared errors under the configured stratum weights."""
    if not math.isfinite(background_weight):
        return float("nan")
    mse = float(
        (errors[foreground].sum() + background_weight * errors[background].sum()).item()
        / (n_foreground + background_weight * n_background)
    )
    return _psnr_db(mse, data_range)


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
    fg_bg_ratio: float = 1.0,
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
    from luxar.gsplats.utils.device import resolve_torch_device

    if V.ndim < 2:
        raise ValueError(f"V must be at least 2D, got shape {V.shape}")
    if len(k_grid) < 1:
        raise ValueError("k_grid must contain at least one K value")
    _validate_fg_bg_ratio(fg_bg_ratio)
    if k_star_metric not in _K_STAR_METRICS:
        raise ValueError(
            f"unknown k_star_metric {k_star_metric!r}; use "
            "'psnr_minmax', 'psnr_foreground', 'psnr_fg_weighted', or 'gain'"
        )

    fit_kwargs = dict(fit_kwargs or {})
    fit_kwargs.pop("seeds", None)
    # cv runs many fits — keep their internal logging quiet by default
    fit_kwargs.setdefault("verbose", False)

    # Background floor / DC-offset suppression: subtract ONCE from V so the fit
    # target, the render reference, AND the held-out truth are all on the same
    # floor-suppressed scale. (A per-fit floor would subtract only inside each
    # fit, mismatching the raw held-out reference and tanking the PSNR.) Default
    # is on ("auto"); K* is thus measured the same way you will fit.
    from luxar.gsplats.fitting.preprocessing import _resolve_floor
    from luxar.gsplats.fitting.validation import _validate_floor

    floor_spec = fit_kwargs.pop("floor", "auto")
    _validate_floor(floor_spec)  # cal bypasses prepare_fit_config's validation
    applied_floor = _resolve_floor(V, floor_spec)
    # A floor at/above the brightest voxel would clip the whole volume to 0
    # (empty signal → non-finite held-out PSNR). Refuse it, mirroring the
    # single-pass guard in _normalize_data. `auto` can't trigger this (mode is
    # capped at the median); only an explicit too-high float/percentile can.
    if applied_floor is not None and applied_floor >= float(V.max()):
        import warnings

        warnings.warn(
            f"floor {applied_floor:.6g} >= volume max {float(V.max()):.6g}; "
            "ignoring (would erase all signal).",
            stacklevel=2,
        )
        applied_floor = None
    if applied_floor is not None:
        V = np.clip(V.astype(np.float32, copy=False) - applied_floor, 0.0, None)
    # V is already floor-subtracted; the per-K fits must not subtract again.
    fit_kwargs["floor"] = "none"

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
    fg_weighted_mask, fg_otsu_threshold = foreground_mask_otsu_smoothed(V)
    fg_t = torch.from_numpy(fg_mask)
    fg_weighted_t = torch.from_numpy(fg_weighted_mask)

    k_values_eff: List[int] = []
    held_psnr: List[float] = []
    held_psnr_fg: List[float] = []
    held_psnr_fg_weighted: List[float] = []
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
    render_device = str(resolve_torch_device(fit_kwargs.get("device")))
    ref_dev = V_orig_t.to(render_device)
    mask_dev = mask_t.to(render_device)
    fg_dev = fg_t.to(render_device)
    fg_weighted_dev = fg_weighted_t.to(render_device)
    train_mask_dev = ~mask_dev
    held_fg_dev = mask_dev & fg_dev
    (
        held_weighted_fg_sel,
        held_weighted_bg_sel,
        n_weighted_fg,
        n_weighted_bg,
        bg_weight,
    ) = _foreground_weighting(
        mask_dev,
        fg_weighted_dev,
        fg_bg_ratio,
    )

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
            held_psnr_val = _psnr_db(held_mse_val, data_range)
            held_psnr.append(held_psnr_val)

            # Foreground-restricted held-out PSNR (background-domination removed)
            fg_pred = rendered[held_fg_dev]
            fg_true = ref_dev[held_fg_dev]
            if fg_pred.numel() == 0:
                held_psnr_fg.append(float("nan"))
            else:
                fg_mse = float(torch.mean((fg_pred - fg_true) ** 2).item())
                held_psnr_fg.append(_psnr_db(fg_mse, data_range))
            # Gain over the predict-zero baseline (plateaus meaningfully)
            held_gain.append(held_out_gain_db(held_mse_val, baseline_mse))

            errors = (held_pred - held_true) ** 2
            held_psnr_fg_weighted.append(
                _foreground_weighted_psnr(
                    errors,
                    held_weighted_fg_sel,
                    held_weighted_bg_sel,
                    n_weighted_fg,
                    n_weighted_bg,
                    bg_weight,
                    data_range,
                )
            )
            del errors

            train_mse = float(torch.mean((train_pred - train_true) ** 2).item())
            train_psnr_val = _psnr_db(train_mse, data_range)
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
    del (
        ref_dev,
        mask_dev,
        fg_dev,
        fg_weighted_dev,
        train_mask_dev,
        held_fg_dev,
        held_weighted_fg_sel,
        held_weighted_bg_sel,
    )
    if str(render_device).startswith("cuda"):
        torch.cuda.empty_cache()

    # 4. Noise floor (on the calibrated volume — floor-subtracted above if a
    # floor was applied, i.e. the same scale the fits and K* use; needs [0, 1]
    # for the PSNR ceiling to be meaningful — the manuscript and
    # fit_gaussian_splats both work in that range, so we pre-normalise)
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
        "psnr_fg_weighted": held_psnr_fg_weighted,
        "gain": held_gain,
    }
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
        fit_config={
            **{k: v for k, v in fit_kwargs.items() if _json_safe(v)},
            "floor_subtracted": applied_floor,
        },
        volume_shape=list(V.shape),
        volume_dtype=str(V.dtype),
        timestamp=datetime.now(timezone.utc).isoformat(),
        held_out_psnr_fg_db=held_psnr_fg,
        held_out_psnr_fg_weighted_db=held_psnr_fg_weighted,
        foreground_mask_fraction=float(np.mean(fg_weighted_mask)),
        foreground_otsu_threshold=float(fg_otsu_threshold),
        fg_bg_ratio=float(fg_bg_ratio),
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
    fg_bg_ratio: float = 1.0,
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
            fg_bg_ratio=fg_bg_ratio,
            feature_method=feature_method,
            compute_rd_model=False,
        )
        density = result.splat_density or {}
        k_star = float(density.get("k_star_reference", result.held_out_peak.k_star))
        points.append((int(scale), n_feat, k_star))
    return fit_saturation_exponent(points)
