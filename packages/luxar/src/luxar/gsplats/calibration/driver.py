"""Top-level calibration drivers.

:func:`calibrate` runs one blind-spot CV sweep over a K-grid; it wires together
the masking, metric, content, noise-floor, and curve-analysis pieces.
:func:`calibrate_saturation_exponent` runs the sweep at several region scales to
measure the density exponent ``alpha``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
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
    fit_rd_model,
    fit_saturation_exponent,
)
from .masking import cv_mask, donut_median_fill
from .metrics import _psnr_db, held_out_gain_db, predict_zero_baseline_mse
from .noise_floor import estimate_masked_noise, estimate_noise_floor
from .result import CalibrationResult, _json_safe
from .sweep import (
    PointContext,
    ProgressCallback,
    curves,
    extend_below,
    peaks,
    run_sweep,
)
from .sweep_control import SweepControl

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


def _sum_float(t: Any) -> float:
    """Sum a tensor in float64 where the device supports it (MPS has no float64)."""
    import torch

    acc = torch.float32 if t.device.type == "mps" else torch.float64
    return float(t.sum(dtype=acc).item())


def _per_k_metrics(
    rendered: Any,
    ref: Any,
    held_mask: Any,
    held_fg_sel: Any,
    weighting: Tuple[Any, Any, int, int, float],
    *,
    data_range: float,
    baseline_mse: float,
    compute_ssim: Callable[..., float],
    robust_range: Optional[float] = None,
) -> Dict[str, float]:
    """Score one rendered fit against the reference, at bounded peak memory.

    ``rendered`` is CONSUMED: after SSIM it is overwritten in place with the
    squared error, so the only full-volume temporaries are SSIM's own (tiled
    past a fixed budget on CPU/MPS — see ``metrics._should_tile_ssim``). The
    held-out set is gathered (5 % of the volume); the train MSE is the total
    squared error minus the held-out part rather than a boolean gather of the
    other 95 %, which used to cost ~2x the volume in ``train_pred`` /
    ``train_true`` copies. Sums accumulate in float64 (float32 on MPS), so the
    values agree with the former ``torch.mean`` over the gathered sets to
    float32 rounding, not bit for bit.
    """
    numel = rendered.numel()
    full_ssim = float(compute_ssim(rendered, ref, data_range=data_range))
    # The robust-peak SSIM `fit` / `compare` headline (one hot pixel stretches
    # the full range, and SSIM's constants with it, towards SSIM = 1).
    # The two ranges arrive as float32- and float64-derived floats; equal to
    # float32 precision means the same constants, so skip the redundant pass.
    full_ssim_robust = (
        full_ssim
        if robust_range is None or math.isclose(robust_range, data_range, rel_tol=1e-6)
        else float(compute_ssim(rendered, ref, data_range=robust_range))
    )
    err = rendered.sub_(ref).square_()
    total_sse = _sum_float(err)
    held_err = err[held_mask]
    n_held = held_err.numel()
    held_sse = _sum_float(held_err)
    n_train = numel - n_held
    nan = float("nan")
    held_mse = held_sse / n_held if n_held else nan
    train_mse = max(total_sse - held_sse, 0.0) / n_train if n_train else nan

    fg_err = held_err[held_fg_sel]
    held_mse_fg = _sum_float(fg_err) / fg_err.numel() if fg_err.numel() else nan
    held_psnr_fg = _psnr_db(held_mse_fg, data_range) if fg_err.numel() else nan
    del fg_err
    fg_sel, bg_sel, n_fg, n_bg, bg_weight = weighting
    held_psnr_fg_weighted = _foreground_weighted_psnr(
        held_err, fg_sel, bg_sel, n_fg, n_bg, bg_weight, data_range
    )
    del held_err
    # Same convention as metrics.compute_psnr (inf at zero error), without
    # another full-volume pass.
    full_psnr = _psnr_db(total_sse / numel, data_range)
    peak = data_range if robust_range is None else robust_range
    return {
        "held_mse": held_mse,
        "held_psnr": _psnr_db(held_mse, data_range),
        "held_psnr_robust": _psnr_db(held_mse, peak),
        "held_psnr_fg_robust": (
            _psnr_db(held_mse_fg, peak) if math.isfinite(held_mse_fg) else nan
        ),
        "held_psnr_fg": held_psnr_fg,
        "held_gain": held_out_gain_db(held_mse, baseline_mse),
        "held_psnr_fg_weighted": held_psnr_fg_weighted,
        "train_psnr": _psnr_db(train_mse, data_range),
        "full_psnr": full_psnr,
        "full_ssim": full_ssim,
        "full_ssim_robust": full_ssim_robust,
    }


class CalibrationInterrupted(KeyboardInterrupt):
    """Raised by :func:`calibrate` on Ctrl-C after at least one finished point.

    ``result`` is the calibration of the points measured so far
    (``status="interrupted"``), so a caller can write it out and report the
    best-so-far K* instead of losing the whole sweep. It subclasses
    ``KeyboardInterrupt`` so code that does not know about it still stops.
    """

    def __init__(self, result: CalibrationResult) -> None:
        super().__init__("calibration interrupted")
        self.result = result


def _resolve_cal_floor(
    V: np.ndarray, floor_spec: Any
) -> Tuple[np.ndarray, Optional[float], Optional[str]]:
    """Subtract the background floor ONCE: ``(V, applied_floor, floor_strategy)``.

    The fit target, the render reference AND the held-out truth must share one
    floor-suppressed scale (a per-fit floor would subtract only inside each fit,
    mismatching the raw held-out reference and tanking the PSNR). A floor at or
    above the brightest voxel would clip the whole volume to 0, so it is refused
    with a warning, mirroring the single-pass guard in ``_normalize_data``.
    """
    from luxar.gsplats.fitting.preprocessing import _resolve_floor_result
    from luxar.gsplats.fitting.validation import _validate_floor

    _validate_floor(floor_spec)  # cal bypasses prepare_fit_config's validation
    floor_result = _resolve_floor_result(V, floor_spec)
    applied_floor = None if floor_result is None else float(floor_result.level)
    if applied_floor is not None and applied_floor >= float(V.max()):
        import warnings

        warnings.warn(
            f"floor {applied_floor:.6g} >= volume max {float(V.max()):.6g}; "
            "ignoring (would erase all signal).",
            stacklevel=3,
        )
        applied_floor = None
    floor_strategy = (
        floor_result.strategy
        if floor_result is not None
        and applied_floor is not None
        and isinstance(floor_spec, str)
        and floor_spec.strip().lower() == "specimen"
        else None
    )
    if applied_floor is not None:
        V = np.clip(V.astype(np.float32, copy=False) - applied_floor, 0.0, None)
    return V, applied_floor, floor_strategy


def _noise_floor_of(V: np.ndarray) -> Any:
    """Noise floor of the calibrated (floor-subtracted) volume, on ``[0, 1]``.

    The PSNR ceiling is only meaningful in the range the manuscript and
    ``fit_gaussian_splats`` work in, so the volume is pre-normalised.
    """
    Vn = V.astype(np.float32, copy=False)
    vmin, vmax = float(Vn.min()), float(Vn.max())
    Vn_norm = (Vn - vmin) / (vmax - vmin) if vmax > vmin else Vn - vmin
    return estimate_noise_floor(Vn_norm)


def _rms_noise_of(V: np.ndarray) -> Tuple[float, float]:
    """``(sigma, scale_ratio)``: the RMS noise a held-out MSE averages.

    Measured on the floor-subtracted volume, the very reference the held-out
    scores are computed against, with the intensity-binned in-plane estimator
    ``compare`` uses inside its mask, here over the whole volume. Shot noise
    makes the bright voxels dominate an MSE, so this RMS, not the median-voxel
    ``sigma_hat``, is the noise the held-out PSNR is limited by. On the two
    tester volumes the best held-out score sits within about 0.1 dB of it (on
    one the curve is still climbing; on the other the estimator's own scale
    check flags fine structure in the band), where ``sigma_hat`` put the
    ceiling 4 and 31 dB higher.
    """
    noise = estimate_masked_noise(V, np.ones(V.shape, dtype=bool))
    return float(noise.sigma), float(noise.scale_ratio)


def _ceiling_db(sigma: float, peak: float) -> float:
    """``20 log10(peak / sigma)``; ``inf`` at zero noise, ``nan`` if unknown."""
    if not (math.isfinite(sigma) and math.isfinite(peak)) or peak <= 0.0:
        return float("nan")
    if sigma <= 0.0:
        return float("inf")
    return 20.0 * math.log10(peak / sigma)


@dataclass
class _ResultParts:
    """Everything a :class:`CalibrationResult` needs besides the per-K records.

    Measured once before the sweep, so a result can be assembled after every
    point (checkpoint), on interrupt, or at the end.
    """

    V: np.ndarray
    noise_floor: Any
    n_features: int
    feature_threshold: float
    fit_config: Dict[str, Any]
    mask_seed: int
    mask_fraction: float
    donut_radius: int
    keep_fits: bool
    foreground_mask_fraction: float
    foreground_otsu_threshold: float
    fg_bg_ratio: float
    baseline_mse: float
    data_range: float
    robust_data_range: float
    noise_sigma_rms: float
    noise_scale_ratio: float
    k_star_metric: str
    feature_method: str
    saturation_exponent: float
    compute_rd_model: bool
    control: SweepControl
    record_sweep: bool


def _density(
    parts: _ResultParts, selected: HeldOutPeak, curve: Dict[str, List[Any]]
) -> SplatDensity:
    """Transferable splat density: feature count + K* as an effective count."""
    ks, eff = list(curve["k"]), list(curve["eff"])
    k_star_eff = (
        int(eff[ks.index(selected.k_star)])
        if selected.k_star in ks
        else int(selected.k_star)
    )
    saturation_cap, measured = _saturation_cap(parts, ks, eff, k_star_eff)
    n_features = parts.n_features
    return SplatDensity(
        feature_method=parts.feature_method,
        n_features_reference=int(n_features),
        k_star_reference=k_star_eff,
        saturation_exponent=float(parts.saturation_exponent),
        saturation_cap=int(max(saturation_cap, k_star_eff)),
        splats_per_feature=(
            float(k_star_eff) / n_features if n_features > 0 else float("nan")
        ),
        # Exact absolute level the detector counted at, recorded so the
        # planner's scan_content counts on the identical scale.
        feature_threshold=parts.feature_threshold,
        saturation_cap_measured=measured,
    )


def _saturation_cap(
    parts: _ResultParts, ks: List[Any], eff: List[Any], k_star_eff: int
) -> Tuple[int, bool]:
    """``(cap, measured)``: the per-box budget ceiling for the density transfer.

    The effective count at the TOP of the planned grid (the empirical ceiling),
    regardless of convergence — collapsing it to ``k_star_eff`` would truncate
    denser-than-reference tiles to the reference budget (M3). When early
    stopping skipped the grid's top, that count was never measured: it is
    estimated from the largest measured point's kept/requested ratio (on A549
    the measured top would have been 2972 instead of the full grid's 48371,
    under-seeding every busy box).
    """
    if not eff:
        return k_star_eff, True
    cap = int(eff[-1])
    skipped = [int(k) for k in parts.control.k_values_skipped]
    if not skipped or not ks or int(ks[-1]) <= 0:
        return cap, True
    ratio = float(eff[-1]) / float(ks[-1])
    return max(cap, int(round(max(skipped) * ratio))), False


def _rd(
    parts: _ResultParts, curve: Dict[str, List[Any]]
) -> Tuple[Optional[RDModel], bool]:
    """Parametric R-D model on held-out MSE vs effective K + convergence flag."""
    if not parts.compute_rd_model or len(curve["k"]) < 3:
        return None, False
    rd = fit_rd_model([float(k) for k in curve["eff"]], list(curve["held_mse"]))
    # converged_fraction is structurally < 1 for any power law (a clean beta~0.5
    # sweep lands ~0.9); only flag genuinely splat-starved curves that are still
    # steeply climbing at K_max.
    return rd, bool(rd is not None and rd.converged_fraction < 0.85)


def _assemble(
    parts: _ResultParts, records: Sequence[Dict[str, Any]], status: str
) -> CalibrationResult:
    """A :class:`CalibrationResult` from the records measured so far."""
    curve = curves(records)
    peak, peak_selected = peaks(curve, parts.k_star_metric)
    selected = peak_selected or peak
    rd, not_converged = _rd(parts, curve)
    ks = [int(k) for k in curve["k"]]
    stored = [int(x) for x in curve["n_stored"]]
    return CalibrationResult(
        k_values_requested=ks,
        k_values_effective=[int(x) for x in curve["eff"]],
        held_out_psnr_db=list(curve["held_psnr"]),
        train_psnr_db=list(curve["train_psnr"]),
        held_out_mse=list(curve["held_mse"]),
        full_psnr_db=list(curve["full_psnr"]),
        full_ssim=list(curve["full_ssim"]),
        full_ssim_robust=list(curve["full_ssim_robust"]),
        held_out_peak=peak,
        noise_floor=parts.noise_floor,
        fit_times_seconds=list(curve["fit_s"]),
        splat_paths=(
            [
                str(r["path"])
                for r in sorted(records, key=lambda r: r["k"])
                if r.get("path")
            ]
            if parts.keep_fits
            else None
        ),
        mask_seed=parts.mask_seed,
        mask_fraction=parts.mask_fraction,
        donut_radius=parts.donut_radius,
        fit_config=parts.fit_config,
        volume_shape=list(parts.V.shape),
        volume_dtype=str(parts.V.dtype),
        timestamp=datetime.now(timezone.utc).isoformat(),
        held_out_psnr_fg_db=list(curve["held_psnr_fg"]),
        held_out_psnr_fg_weighted_db=list(curve["held_psnr_fg_weighted"]),
        foreground_mask_fraction=parts.foreground_mask_fraction,
        foreground_otsu_threshold=parts.foreground_otsu_threshold,
        fg_bg_ratio=parts.fg_bg_ratio,
        held_out_gain_db=list(curve["held_gain"]),
        predict_zero_baseline_mse=parts.baseline_mse,
        k_star_metric=parts.k_star_metric,
        held_out_peak_selected=peak_selected,
        splat_density=asdict(_density(parts, selected, curve)),
        rd_model=asdict(rd) if rd is not None else None,
        not_converged=not_converged,
        sweep=parts.control.to_dict() if parts.record_sweep else None,
        status=status,
        held_out_psnr_robust_db=list(curve["held_psnr_robust"]),
        held_out_psnr_fg_robust_db=list(curve["held_psnr_fg_robust"]),
        data_range=parts.data_range,
        robust_data_range=parts.robust_data_range,
        k_values_fitted=[int(x) for x in curve["n_fitted"]],
        k_values_stored=stored,
        noise_sigma_rms=parts.noise_sigma_rms,
        noise_scale_ratio=parts.noise_scale_ratio,
        psnr_ceiling_robust_db=_ceiling_db(
            parts.noise_sigma_rms, parts.robust_data_range
        ),
        k_star_stored=stored[ks.index(selected.k_star)]
        if selected.k_star in ks
        else None,
    )


def _prepare_fit_kwargs(
    fit_kwargs: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Any, Optional[float], Optional[float], float]:
    """``(fit_kwargs, floor_spec, cull_retention, stored_min_extent, share)``.

    The per-K fits run with ``floor="none"`` (cal subtracts the floor once),
    ``min_extent=0`` — measure the UNPRUNED K, since a content fit sized from
    post-prune counts would prune again (~K*(1-p)^2) — and with the cumulative
    cull applied by the sweep itself (see :class:`~.sweep.PointContext`). The
    configured speck-pruning settings are kept to report what ``fit`` would
    store.
    """
    from luxar.gsplats.fitting.splat_hygiene import (
        DEFAULT_MIN_EXTENT_MAX_MASS_SHARE,
        DEFAULT_MIN_EXTENT_VOX,
    )

    fit_kwargs = dict(fit_kwargs or {})
    fit_kwargs.pop("seeds", None)
    # cv runs many fits — keep their internal logging quiet by default
    fit_kwargs.setdefault("verbose", False)
    floor_spec = fit_kwargs.pop("floor", "auto")
    cull_retention = fit_kwargs.pop("cull_retention", 0.999)
    stored_min_extent = fit_kwargs.get("min_extent", DEFAULT_MIN_EXTENT_VOX)
    share = float(
        fit_kwargs.get("min_extent_max_mass_share", DEFAULT_MIN_EXTENT_MAX_MASS_SHARE)
    )
    fit_kwargs["floor"] = "none"
    fit_kwargs["min_extent"] = 0.0
    return fit_kwargs, floor_spec, cull_retention, stored_min_extent, share


def _validate_calibrate_args(
    V: np.ndarray, k_grid: Sequence[int], fg_bg_ratio: float, k_star_metric: str
) -> None:
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
    sweep_control: Optional[SweepControl] = None,
    checkpoint: Optional[Callable[[CalibrationResult], None]] = None,
) -> CalibrationResult:
    """Run a blind-spot CV sweep over ``k_grid`` on volume ``V``.

    Pipeline (one execution per call):

    1. Generate a deterministic CV mask, donut-fill ``V`` to make
       ``V_filled``; estimate the noise floor and feature count of ``V``.
    2. Fit a Gaussian-splat model with ``fit_gaussian_splats`` at each
       ``K`` in ``k_grid``, using ``V_filled`` as the target. ``fit_kwargs``
       are forwarded verbatim except ``seeds`` (overridden per-K).
    3. Render each fit back to volume; compute held-out / train / full
       PSNR (and full SSIM) against the original ``V``.
    4. Detect K* via :func:`find_k_star`.

    Held-out PSNR is reported on two peaks: the full range ``max - min`` (the
    historical ``held_out_psnr_db``) and the ROBUST peak ``p99.99 - min`` that
    ``fit`` and ``compare`` headline (``held_out_psnr_robust_db``). The peak is
    one constant per volume, so the two curves differ by a constant offset and
    every K* rule (which compares dB differences) picks the same K.

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
        Invoked as ``(i, n, msg)`` before each fit and after metrics; the
        after-metrics line carries the elapsed time and an ETA for the rest
        of the grid (an estimate; early stopping may also end the sweep sooner).
    sweep_control : SweepControl, optional
        Early stopping (end the sweep once the K*-metric has not improved by
        ``early_stop_min_gain_db`` for ``early_stop_patience`` consecutive
        points) and the one-shot downward extension when K* lands on the
        smallest grid point. ``None`` (the default) runs exactly ``k_grid``;
        when given, it is updated with what the sweep did and recorded in
        ``CalibrationResult.sweep``.
    checkpoint : callable, optional
        Called with a ``status="partial"`` result after every finished point
        (``gsplat cal`` writes it to ``cal.json``), so a crash keeps the
        measured points.

    Returns
    -------
    CalibrationResult

    Raises
    ------
    CalibrationInterrupted
        On Ctrl-C once at least one point finished; carries the partial result.
    """
    # Lazy imports — keep module-load cheap for tests that only touch utilities
    import torch

    from luxar.gsplats.metrics import robust_peak
    from luxar.gsplats.utils.device import resolve_torch_device

    _validate_calibrate_args(V, k_grid, fg_bg_ratio, k_star_metric)
    fit_kwargs, floor_spec, cull_retention, stored_min_extent, share = (
        _prepare_fit_kwargs(fit_kwargs)
    )
    # Floor subtracted ONCE (default "auto"), so K* is measured the way you fit.
    V, applied_floor, floor_strategy = _resolve_cal_floor(V, floor_spec)
    noise_sigma_rms, noise_scale_ratio = _rms_noise_of(V)

    if keep_fits is not None:
        keep_fits = Path(keep_fits)
        keep_fits.mkdir(parents=True, exist_ok=True)

    # 1. Mask + donut fill
    mask = cv_mask(V.shape, fraction=mask_fraction, seed=mask_seed)
    V_filled = donut_median_fill(V, mask, radius=donut_radius)

    V_orig_t = torch.from_numpy(V.astype(np.float32, copy=False))
    data_range = float(V.max() - V.min()) or 1.0
    # The robust peak `fit` / `compare` headline: p99.99 of the reference minus
    # its minimum (the floor, on this floor-subtracted basis).
    robust_range = float(robust_peak(V_orig_t)) - float(V.min())
    if not robust_range > 0.0:
        robust_range = data_range

    # Regime-robust extras: predict-zero baseline (the "free" error floor) and a
    # foreground mask so we can strip background-domination from the held-out
    # metric.
    baseline_mse = predict_zero_baseline_mse(V, mask)
    fg_weighted_mask, fg_otsu_threshold = foreground_mask_otsu_smoothed(V)

    # Resolve the render device ONCE and hoist the loop-invariant device copies
    # (original volume + masks) out of the per-K loop (M9). Only the held-out
    # mask stays a full-volume selector; the foreground strata are reduced to
    # HELD-OUT-sized selectors (5 % of the volume) up front.
    render_device = str(
        resolve_torch_device(
            fit_kwargs.get("device"),
            use_cuda=fit_kwargs.get("use_cuda", True),
            use_metal=fit_kwargs.get("use_metal", True),
        )
    )
    mask_dev = torch.from_numpy(mask).to(render_device)
    ctx = PointContext(
        V_filled=V_filled,
        fit_kwargs=fit_kwargs,
        render_device=render_device,
        ref_dev=V_orig_t.to(render_device),
        mask_dev=mask_dev,
        held_fg_sel=torch.from_numpy(foreground_mask_otsu(V)).to(render_device)[
            mask_dev
        ],
        weighting=_foreground_weighting(
            mask_dev, torch.from_numpy(fg_weighted_mask).to(render_device), fg_bg_ratio
        ),
        data_range=data_range,
        robust_range=robust_range,
        baseline_mse=baseline_mse,
        keep_fits=keep_fits,
        cull_retention=cull_retention,
        stored_min_extent=stored_min_extent,
        stored_min_extent_max_mass_share=share,
    )
    control = sweep_control or SweepControl(early_stop=False, extend_below=False)
    parts = _ResultParts(
        V=V,
        noise_floor=_noise_floor_of(V),
        n_features=int(count_features(V, method=feature_method)),
        feature_threshold=feature_threshold(V, feature_method),
        fit_config={
            **{k: v for k, v in fit_kwargs.items() if _json_safe(v)},
            "cull_retention": cull_retention,
            "floor_subtracted": applied_floor,
            **(
                {"floor_strategy": floor_strategy} if floor_strategy is not None else {}
            ),
        },
        mask_seed=mask_seed,
        mask_fraction=mask_fraction,
        donut_radius=donut_radius,
        keep_fits=keep_fits is not None,
        foreground_mask_fraction=float(np.mean(fg_weighted_mask)),
        foreground_otsu_threshold=float(fg_otsu_threshold),
        fg_bg_ratio=float(fg_bg_ratio),
        baseline_mse=baseline_mse,
        data_range=data_range,
        robust_data_range=robust_range,
        noise_sigma_rms=noise_sigma_rms,
        noise_scale_ratio=noise_scale_ratio,
        k_star_metric=k_star_metric,
        feature_method=feature_method,
        saturation_exponent=saturation_exponent,
        compute_rd_model=compute_rd_model,
        control=control,
        record_sweep=sweep_control is not None,
    )
    del fg_weighted_mask

    def on_point(records: List[Dict[str, Any]]) -> None:
        if checkpoint is not None:
            checkpoint(_assemble(parts, records, "partial"))

    # 2-3. Fit at each K (stopping early on a plateau), then extend below the
    # grid once if K* landed on its smallest point.
    records: List[Dict[str, Any]] = []
    try:
        run_sweep(
            ctx,
            list(k_grid),
            control,
            k_star_metric,
            progress_callback,
            records=records,
            on_point=on_point,
        )
        extend_below(ctx, records, control, k_star_metric, progress_callback, on_point)
    except KeyboardInterrupt:
        control.interrupted = True
        if records:
            raise CalibrationInterrupted(
                _assemble(parts, records, "interrupted")
            ) from None
        raise
    finally:
        del ctx
        if render_device.startswith("cuda"):
            torch.cuda.empty_cache()
    return _assemble(parts, records, "complete")


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
