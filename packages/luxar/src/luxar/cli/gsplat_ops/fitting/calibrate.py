"""Implementation helper for gsplat calibration command."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from arbol import aprint, asection

from ..._traceback import exit_with_error

if TYPE_CHECKING:
    from luxar.gsplats.calibration import (
        CalibrationResult,
        HeldOutPeak,
        SweepControl,
    )


def _print_k_star_scope_caveat(
    calibration_region: Optional[dict], whole: Optional[dict] = None
) -> None:
    """Qualify the ``★ Recommended K*`` headline when it is region-scoped.

    Under ``--auto-region`` / ``--region`` K* was measured on the crop, so it is
    a REGION budget (roughly tile-scale) — feeding it to ``fit --seeds`` (a
    whole-volume budget, divided across tiles) would under-seed the volume. Say
    so, and print the whole-volume budget ``cal`` derived (``whole``, the
    content planner's density transfer summed over the input) when it has one.
    Prints nothing for a whole-volume calibration, including
    ``strategy="whole"``: that means the volume was no bigger than
    ``--region-size`` on every axis, so the "crop" IS the whole volume and its
    K* is exactly what ``--seeds`` wants.
    """
    if calibration_region is not None and calibration_region.get("strategy") == "whole":
        return
    if calibration_region is None and whole is None:
        return
    aprint("    ⚠ region-scoped (measured on the crop), NOT a whole-volume budget.")
    if whole is not None:
        aprint(
            f"    Whole volume: fit --seeds {whole['seeds']:,}  (the density "
            f"transfer summed over {whole['n_boxes']} crop-sized boxes, "
            f"{whole['n_features_whole']:,} features vs "
            f"{whole['n_features_crop']:,} in the crop), or transfer the "
            "density per box with `fit --tiling content --cal <this json>`."
        )
    else:
        aprint(
            "    Transfer it with `fit --tiling content --cal <this json>`; do "
            "not pass it to `fit --seeds`."
        )


def _print_grid_warnings(
    peak: "HeldOutPeak", prefix: str, skip_high_edge: bool = False
) -> None:
    """Print the K-grid adequacy warnings ``find_k_star`` attached to ``peak``
    (K* on a grid edge, or a curve too flat to resolve K).

    ``skip_high_edge`` drops the "largest grid point" one when the
    not-converged line will say it (with the R-D-model alternative) anyway;
    ``cal.json`` keeps it either way.
    """
    from luxar.gsplats.calibration.curve_analysis import HIGH_EDGE_WARNING

    for warning in peak.warnings:
        if not (skip_high_edge and warning.startswith(HIGH_EDGE_WARNING)):
            aprint(f"{prefix}{warning}")


def _print_regime_warnings(result: "CalibrationResult") -> None:
    """Regime warnings — warn-by-default, no behaviour change."""
    import math

    sig = result.noise_floor.sigma_hat
    if math.isfinite(sig) and sig < 1e-4:
        aprint(
            "  ⚠ median-voxel σ̂≈0 (noise-free/deconvolved): the blind-spot "
            "peak may not appear; prefer --k-star-metric psnr_fg_weighted "
            "(and --auto-region)."
        )
    if result.not_converged:
        aprint(
            "  ⚠ held-out curve still climbing at K_max (not converged): "
            "extend --k-max or use the R-D-model extrapolation."
        )


def _selected_summary(
    result: "CalibrationResult",
) -> tuple["HeldOutPeak", str, list[float]]:
    """Return the effective peak, metric label, and displayed metric curve."""
    selected_peak = result.held_out_peak_selected or result.held_out_peak
    selected_metric = result.k_star_metric
    if selected_metric != "psnr_minmax" and result.held_out_peak_selected is None:
        selected_metric = "psnr_minmax (fallback: selected curve is undefined)"
        selected_curve = result.held_out_psnr_db
        return selected_peak, selected_metric, selected_curve
    selected_curve = {
        "psnr_minmax": result.held_out_psnr_db,
        "psnr_foreground": result.held_out_psnr_fg_db,
        "psnr_fg_weighted": result.held_out_psnr_fg_weighted_db,
        "gain": result.held_out_gain_db,
    }.get(result.k_star_metric, result.held_out_psnr_db)
    if len(selected_curve) != len(result.k_values_requested):
        selected_curve = result.held_out_psnr_db
    return selected_peak, selected_metric, selected_curve


def _resolve_cal_region(region: Optional[str], auto_region: bool) -> Optional[tuple]:
    """Parse ``--region``; it and ``--auto-region`` both pick the crop."""
    from luxar.cli.gsplat_ops.voxel_frame import parse_region

    spans = parse_region(region)
    if spans is not None and auto_region:
        raise typer.BadParameter(
            "--region and --auto-region both choose the calibration crop; use one"
        )
    return spans


def _cal_input_frame(
    *,
    input_path: Path,
    source_info: dict,
    volume_shape: tuple,
    voxel_size: Optional[str],
    physical: bool,
    preset: str,
    config: Optional[Path],
    array_key: Optional[str],
    axes: Optional[str],
    region: Optional[tuple],
) -> dict:
    """The spacing and crop a calibration ran on, as recorded in ``cal.json``.

    Resolved exactly as ``gsplat fit`` resolves them (``--voxel-size`` > config
    > ``--physical`` metadata), so ``cal`` and the ``fit`` it calibrates agree.
    """
    from luxar.cli.gsplat_config import load_fit_config
    from luxar.cli.gsplat_ops.voxel_frame import (
        FitFrame,
        format_spans,
        normalized_region,
        parse_voxel_size,
        resolve_voxel_spacing,
    )

    spans = normalized_region(region, volume_shape) if region is not None else None
    spacing = resolve_voxel_spacing(
        input_path=input_path,
        source_info=source_info,
        ndim=len(volume_shape),
        explicit=parse_voxel_size(voxel_size),
        physical=physical,
        config=load_fit_config(preset=preset, config_path=config),
        array_key=array_key,
        axes=axes,
    )
    if spans is not None:
        aprint(f"Region: {format_spans(spans)} of the input")
    if spacing is None and spans is None:
        return {}
    return FitFrame(spacing=spacing, physical=False, region=spans).stamps()


def _apply_cal_spacing(fit_kwargs: dict, input_frame: dict) -> None:
    """Fit every K with the resolved spacing, but keep its output on the grid.

    The spacing changes the fit itself (anisotropic initial sigma, physical-space
    sigma bounds), so ``cal`` uses it to measure the K that ``fit --physical``
    will actually run. The held-out score renders each fit on the volume's
    VOXEL grid, so the splats must stay in voxel coordinates
    (``output_space="voxel"``); a real-space fit would render off-grid and score
    garbage. K* is a splat COUNT, so it is unaffected by the output units.
    """
    if "voxel_size" in input_frame:
        fit_kwargs["voxel_size"] = list(input_frame["voxel_size"])
    if fit_kwargs.get("voxel_size") is not None:
        fit_kwargs["output_space"] = "voxel"


def _resolve_cal_grid(
    volume: "Any",
    *,
    k_grid: Optional[str],
    n_grid: int,
    k_min: Optional[int],
    k_max: Optional[int],
    progression: str,
    power: int,
    floor_spec: "Any",
    early_stop: bool,
    extend_below: bool,
) -> "tuple[list[int], SweepControl]":
    """The sweep grid and its control, narrated.

    ``--k-grid`` wins; else ``--k-min``/``--k-max`` (each defaulting to the
    historical 1000 / 512000 when only the other is given); else both bounds
    come from the volume (:func:`~luxar.gsplats.calibration.auto_k_bounds`).
    """
    from luxar.gsplats.calibration import SweepControl, auto_k_bounds, build_k_grid

    control = SweepControl(early_stop=early_stop, extend_below=extend_below)
    explicit: Optional[list[int]] = None
    if k_grid is not None:
        explicit = sorted({int(x.strip()) for x in k_grid.split(",") if x.strip()})
        control.grid_source = "explicit"
    elif k_min is not None or k_max is not None:
        k_min = 1_000 if k_min is None else k_min
        k_max = 512_000 if k_max is None else k_max
        control.grid_source = "bounds"
    else:
        bounds = auto_k_bounds(volume, floor=floor_spec, n_points=n_grid)
        k_min, k_max = bounds.k_min, bounds.k_max
        control.grid_source = "auto"
        control.auto_grid = bounds.to_dict()
        aprint(bounds.summary())
    ks = build_k_grid(
        explicit=explicit,
        n_points=n_grid,
        k_min=k_min if k_min is not None else 1_000,
        k_max=k_max if k_max is not None else 512_000,
        progression=progression,
        power=power,
    )
    aprint(f"K grid ({len(ks)} points): {ks}")
    if early_stop:
        aprint(
            f"Early stop: after {control.early_stop_patience} points without a "
            f"> {control.early_stop_min_gain_db:g} dB gain, or one point "
            f">= {control.early_stop_decline_db:g} dB below the best (--no-early-stop "
            "runs the whole grid)"
        )
    return ks, control


def _parse_exponent_scales(exponent_scales: Optional[str]) -> "list[int]":
    """``--exponent-scales`` as integers (default ``128,192,256``)."""
    if exponent_scales:
        try:
            return [int(x) for x in exponent_scales.split(",") if x.strip()]
        except ValueError as e:
            raise typer.BadParameter(
                "--exponent-scales must be comma-separated integers; "
                f"got {exponent_scales!r}"
            ) from e
    return [128, 192, 256]


def _fit_exponent_into(
    result: "CalibrationResult",
    volume_full: "Any",
    ks: "list[int]",
    *,
    exponent_scales: Optional[str],
    saturation_exponent: float,
    fit_kwargs: dict,
    feature_metric: str,
    region_strategy: str,
    k_star_metric: str,
    fg_bg_ratio: float,
    mask_seed: int,
    mask_fraction: float,
    progress: "Any",
    enabled: bool = True,
) -> None:
    """Measure the saturation exponent alpha over several region scales.

    Calibrates K* at each scale and regresses log K* on log n_features; the
    fitted alpha overrides the assumed default in ``result.splat_density``
    (which the planner / ``fit --tiling content`` read). A no-op unless
    ``enabled`` (``--fit-exponent``).
    """
    import math
    from dataclasses import asdict as _asdict_fit

    if not enabled:
        return

    from luxar.gsplats.calibration import calibrate_saturation_exponent

    _on_progress = progress

    raw_scales = _parse_exponent_scales(exponent_scales)
    if any(s <= 0 for s in raw_scales):
        raise typer.BadParameter(
            f"--exponent-scales must be positive; got {raw_scales}"
        )
    # Dedupe (preserve order) — duplicate scales just waste a K-sweep
    # and collapse to one regression point.
    scales = list(dict.fromkeys(raw_scales))
    if len(scales) < len(raw_scales):
        aprint(f"⚠ --exponent-scales: dropped duplicates → {scales}")
    # Drop scales larger than every spatial axis: they clamp to the
    # whole volume and collapse to identical feature counts (the silent
    # regression killer the agent review flagged).
    _max_dim = int(max(volume_full.shape))
    _too_big = [s for s in scales if s > _max_dim]
    if _too_big:
        scales = [s for s in scales if s <= _max_dim]
        aprint(
            f"⚠ --exponent-scales: dropped {_too_big} > volume "
            f"({_max_dim} vox) — they clamp to the whole volume."
        )
    if len(scales) < 2:
        raise typer.BadParameter(
            "--exponent-scales needs ≥2 distinct scales within the "
            f"volume ({_max_dim} vox) to regress an exponent; "
            f"got {scales}."
        )
    with asection(f"Fitting saturation exponent over {len(scales)} scale(s)"):
        aprint(
            f"⚠ --fit-exponent runs {len(scales)} extra K-sweeps "
            f"(scales={scales}); this multiplies runtime accordingly."
        )
        efit = calibrate_saturation_exponent(
            volume_full,
            scales,
            k_grid=ks,
            fit_kwargs=fit_kwargs,
            feature_method=feature_metric,
            region_strategy=region_strategy,
            k_star_metric=k_star_metric,
            fg_bg_ratio=fg_bg_ratio,
            mask_seed=mask_seed,
            mask_fraction=mask_fraction,
            progress_callback=_on_progress,
        )
    if efit is None:
        aprint(
            "⚠ exponent fit failed (need ≥2 scales with distinct "
            f"feature counts); keeping α={saturation_exponent}."
        )
    else:
        # Always record the fit for provenance/inspection.
        result.exponent_fit = _asdict_fit(efit)
        r2 = efit.r_squared
        r2_str = "n/a" if not math.isfinite(r2) else f"{r2:.3f}"
        alpha_usable = math.isfinite(efit.alpha) and efit.alpha > 0.0
        if not alpha_usable:
            # A non-positive / non-finite slope means the power law
            # didn't hold (e.g. K* flat across scales → α≈0, which
            # would collapse predict_k to a constant). Keep the default
            # rather than silently disabling the density transfer.
            aprint(
                f"⚠ degenerate exponent (α={efit.alpha:.3g}, R²={r2_str}, "
                f"{efit.n_distinct} distinct scales); keeping default "
                f"α={saturation_exponent}. (Fit recorded for inspection.)"
            )
        else:
            if result.splat_density is not None:
                result.splat_density["saturation_exponent"] = efit.alpha
            aprint(
                f"Fitted α={efit.alpha:.3f} (R²={r2_str}, "
                f"{efit.n_distinct} distinct scales); was "
                f"{saturation_exponent}"
            )
            if not math.isfinite(r2) or r2 < 0.5:
                aprint(
                    "⚠ low/unassessable confidence (need ≥3 distinct "
                    "scales with varied feature counts and K*); treat "
                    "α as provisional — consider more/varied "
                    "--exponent-scales or keep the default."
                )


def _run_cal_sweep(
    volume: "Any",
    ks: "list[int]",
    output_json: Path,
    *,
    stamp: "Any",
    **calibrate_kwargs: "Any",
) -> "tuple[CalibrationResult, bool]":
    """Run the sweep, checkpointing ``cal.json`` after every point.

    Returns ``(result, interrupted)``. On Ctrl-C after at least one finished
    point the measured points come back (``status="interrupted"``) so the
    caller still writes them and prints the best-so-far K*; a crash leaves the
    last ``status="partial"`` checkpoint on disk.
    """
    from luxar.gsplats.calibration import CalibrationInterrupted, calibrate

    output_json.parent.mkdir(parents=True, exist_ok=True)

    def _checkpoint(partial: "CalibrationResult") -> None:
        stamp(partial).to_json(output_json)

    try:
        return calibrate(
            volume, k_grid=ks, checkpoint=_checkpoint, **calibrate_kwargs
        ), False
    except CalibrationInterrupted as exc:
        n = len(exc.result.k_values_requested)
        aprint(
            f"⚠ Interrupted after {n} point(s): writing what was measured "
            f"(status=interrupted); best so far is K*={_selected_summary(exc.result)[0].k_star:,}."
        )
        return exc.result, True


def _suggest_whole_volume_seeds(
    result: "CalibrationResult",
    volume_full: "Any",
    load_full: "Any",
    *,
    region_spans: Optional[tuple],
    interrupted: bool,
    probe_size: "Any" = lambda: None,
) -> None:
    """Store ``result.whole_volume_seeds`` for a crop calibration.

    Only for ``--auto-region`` (``volume_full`` is the loaded input) or
    ``--region`` (the input was never loaded whole, so ``load_full`` reads it
    once here). Sums the content planner's per-box density transfer over
    crop-sized boxes of the whole input
    (:func:`~luxar.gsplats.calibration.whole_volume_seeds`); a failure only
    skips the suggestion.
    """
    from luxar.gsplats.calibration import whole_volume_seeds
    from luxar.gsplats.calibration.whole_volume import MAX_WHOLE_INPUT_VOXELS

    region = result.calibration_region
    cropped = region_spans is not None or region is not None
    if interrupted or not cropped or result.splat_density is None:
        return
    if region is not None and region.get("strategy") == "whole":
        return  # the "crop" IS the whole volume; K* already is the budget
    if region_spans is not None:
        # The whole input was never loaded: read it once, but only when its
        # header says it is small enough (an OOM here must not be risked).
        n = probe_size()
        if n is None or n > MAX_WHOLE_INPUT_VOXELS:
            size = "of unknown size" if n is None else f"{n:,} elements"
            aprint(
                f"⚠ whole-volume --seeds suggestion skipped: the input is {size} "
                f"(limit {MAX_WHOLE_INPUT_VOXELS:,}). Transfer the density with "
                "`fit --tiling content --cal <this json>`."
            )
            return
        volume_full = None
    try:
        with asection("Sizing a whole-volume --seeds budget"):
            V_full = volume_full if volume_full is not None else load_full()
            # The same K* the content planner transfers: the density's
            # effective (post-cull) reference count.
            result.whole_volume_seeds = whole_volume_seeds(
                V_full,
                crop_shape=result.volume_shape,
                k_star_seeds=int(result.splat_density["k_star_reference"]),
                splat_density=result.splat_density,
                floor_subtracted=result.fit_config.get("floor_subtracted"),
            )
    except (MemoryError, ValueError, OSError) as exc:
        aprint(f"⚠ whole-volume --seeds suggestion skipped ({exc})")


def _run_post_sweep(
    result: "CalibrationResult", steps: "Any", *, skip: bool = False
) -> bool:
    """Run the post-sweep steps; on Ctrl-C keep the complete sweep.

    Returns whether they were interrupted. The sweep itself finished, so the
    result keeps ``status="complete"``; ``sweep.post_sweep_interrupted`` says a
    later step (``--fit-exponent``, the whole-volume budget) did not. With
    ``skip`` (the sweep itself was interrupted) nothing runs and ``True`` is
    returned.
    """
    if skip:
        return True
    try:
        steps()
    except KeyboardInterrupt:
        aprint("⚠ Interrupted after the sweep: cal.json keeps the complete sweep.")
        if result.sweep is not None:
            result.sweep["post_sweep_interrupted"] = True
        return True
    return False


def _resolve_keep_fits(
    keep_fits: Optional[Path],
    pdf_report: Optional[Path],
    stack: contextlib.ExitStack,
    output_json: Path,
) -> "tuple[Optional[Path], bool]":
    """``(directory for the per-K fits or None, whether it is temporary)``.

    ``--keep-fits`` wins. ``--pdf`` alone keeps the fits in a temporary
    directory that ``stack`` deletes once the report is written: the report's
    slice page renders K_min / K* / K_max, and K* is only known at the end.
    The directory sits beside ``cal.json``, not in ``$TMPDIR``: a run killed
    without cleanup (a Slurm timeout) leaves it where the user will see it,
    rather than filling a small node-local ``/tmp``.
    """
    import tempfile

    if keep_fits is not None:
        keep_fits = Path(keep_fits)
        keep_fits.mkdir(parents=True, exist_ok=True)
        aprint(f"Per-K fits will be saved under {keep_fits}")
        return keep_fits, False
    if pdf_report is None:
        return None, False
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(
        stack.enter_context(
            tempfile.TemporaryDirectory(
                prefix=f"{output_json.stem}-fits-", dir=output_json.parent
            )
        )
    )
    aprint(
        f"--pdf: per-K fits are kept in a temporary directory ({temp}) for the "
        "report's slice page and deleted afterwards (pass --keep-fits DIR to "
        "keep them)."
    )
    return temp, True


def _write_cal_outputs(
    result: "CalibrationResult",
    volume: "Any",
    output_json: Path,
    pdf_report: Optional[Path],
    *,
    splat_paths: Optional["list[str]"] = None,
) -> None:
    """Write ``cal.json`` and, when asked, the PDF report.

    ``splat_paths`` (the per-K fits the report's slice page renders) defaults
    to ``result.splat_paths``; it is passed separately when the fits live in a
    temporary directory that ``cal.json`` must not reference.
    """
    if splat_paths is None:
        splat_paths = result.splat_paths
    # 5. Write JSON
    with asection("Writing results"):
        output_json.parent.mkdir(parents=True, exist_ok=True)
        result.to_json(output_json)
        aprint(f"Wrote {output_json}")

    # 6. Optional PDF
    if pdf_report is not None:
        with asection("Generating PDF report"):
            try:
                from luxar.gsplats.calibration_report import (
                    render_calibration_report,
                )

                pdf_report.parent.mkdir(parents=True, exist_ok=True)
                render_calibration_report(
                    result=result,
                    volume=volume,
                    output_path=pdf_report,
                    splat_paths=splat_paths,
                )
                aprint(f"Wrote {pdf_report}")
            except ImportError as exc:
                aprint(f"Skipping PDF report: optional dependency missing ({exc})")
                aprint(f"Install matplotlib to enable --pdf: {PDF_REPORT_INSTALL_HINT}")


def _robust_offset_db(result: "CalibrationResult") -> float:
    """dB to ADD to a full-range PSNR to restate it on the robust peak."""
    return result.robust_offset_db()


def _fmt_db(x: float) -> str:
    import math

    if math.isnan(x):
        return "   nan"
    if math.isinf(x):
        return "   inf"
    return f"{x:6.2f}"


def _ssim_column(result: "CalibrationResult") -> "tuple[list[float], str]":
    """The table's SSIM column and what the robust-peak header line covers.

    SSIM on the robust peak like PSNR; a ``cal.json`` from before
    ``full_ssim_robust`` was recorded falls back to the full-range SSIM, and
    the header then claims the robust peak for PSNR only.
    """
    if result.full_ssim_robust:
        return result.full_ssim_robust, "PSNR and SSIM"
    return result.full_ssim, "PSNR (SSIM on the full range)"


def _print_cal_table(result: "CalibrationResult") -> "tuple[HeldOutPeak, str]":
    """The per-K table on the robust peak; returns ``(selected peak, metric)``.

    The marker and the K*-metric column track the metric the user selected
    (falling back to the legacy min--max peak when no metric switch).
    """
    selected_peak, selected_metric, selected_curve = _selected_summary(result)
    off = _robust_offset_db(result)
    # `gain` is a ratio of two MSEs: no peak, nothing to shift.
    metric_off = 0.0 if result.k_star_metric == "gain" else off
    fg = result.held_out_psnr_fg_db or [float("nan")] * len(selected_curve)
    stored = result.k_values_stored or [0] * len(selected_curve)
    ssim, measures = _ssim_column(result)
    gap = (
        f"the full-range (max - floor) PSNR figures in cal.json are {-off:.2f} dB higher"
        if abs(off) >= 0.01
        else "here the full-range (max - floor) peak coincides with it"
    )
    aprint(
        f"  {measures} on the robust peak (p99.99 - floor), as `fit` / `compare` "
        f"report them; {gap}."
    )
    aprint(
        "    K_req     kept   stored   held-out  fg held-out  K*-metric    train"
        "   SSIM   fit (s)"
    )
    aprint("    " + "-" * 86)
    for i, k_req in enumerate(result.k_values_requested):
        marker = "★" if k_req == selected_peak.k_star else " "
        aprint(
            f"  {marker} {k_req:7d}  {result.k_values_effective[i]:7d}  "
            f"{stored[i]:7d}   {_fmt_db(result.held_out_psnr_db[i] + off)}   "
            f"{_fmt_db(fg[i] + off)}     {_fmt_db(selected_curve[i] + metric_off)}"
            f"   {_fmt_db(result.train_psnr_db[i] + off)}  "
            f"{ssim[i]:5.3f}  {result.fit_times_seconds[i]:7.1f}"
        )
    aprint(
        "    kept = after the cumulative cull (the scored model); stored = what "
        "`fit --seeds K` keeps after its default speck pruning + cull."
    )
    aprint("")
    return selected_peak, selected_metric


def _print_stored_note(result: "CalibrationResult", k_star: int) -> None:
    """K* is a SEEDING budget; say what a fit at it will actually store."""
    if result.k_star_stored is None:
        return
    preset = result.fit_config.get("preset") or "this calibration's preset"
    aprint(
        f"  fit --seeds {k_star:,} will store about {result.k_star_stored:,} "
        "splats after the default speck pruning + cull (estimated from cal's "
        f"{preset} fits of the blind-spot target; a raw-volume fit, or another "
        "preset, can keep fewer)."
    )


def _noise_lines(result: "CalibrationResult") -> "list[str]":
    """The summary's noise line(s), on the SAME robust-peak basis as the table.

    Prefers the RMS noise (what a held-out MSE averages, on the
    floor-subtracted reference the table is scored against). A ``cal.json`` that predates it only has the median-voxel ensemble
    ``sigma_hat`` on the full range: that one is restated on the robust peak
    and labelled as a median, which under shot noise reads far too optimistic.
    """
    import math

    from luxar.gsplats.calibration.noise_floor import MASKED_SIGMA_MAX_SCALE_RATIO

    sigma, ceil_db = result.noise_sigma_rms, result.psnr_ceiling_robust_db
    if math.isfinite(sigma) and not math.isnan(ceil_db):
        peak = result.robust_data_range
        share = (
            f" ({100.0 * sigma / peak:.1f}% of the robust peak)"
            if math.isfinite(peak) and peak > 0
            else ""
        )
        ceil_str = f"{ceil_db:.1f} dB" if math.isfinite(ceil_db) else "unbounded"
        lines = [
            f"  Noise:          RMS σ = {sigma:.4g}{share} → held-out PSNR "
            f"ceiling ≈ {ceil_str} (robust peak, as in the table)"
        ]
        ratio = result.noise_scale_ratio
        if math.isfinite(ratio) and ratio > MASKED_SIGMA_MAX_SCALE_RATIO:
            lines.append(
                "                  (fine structure also reads as noise here "
                f"[scale ratio {ratio:.1f}], so the ceiling may read low)"
            )
        return lines
    nf_sigma, nf_ceil = result.noise_floor.sigma_hat, result.noise_floor.psnr_max_db
    sigma_str = f"{nf_sigma:.4f}" if math.isfinite(nf_sigma) else "—"
    if math.isfinite(nf_ceil):
        ceil_str = f"{nf_ceil + _robust_offset_db(result):.1f} dB"
    else:
        ceil_str = ">60 dB" if math.isinf(nf_ceil) else "—"
    return [
        f"  Noise floor:    median-voxel σ = {sigma_str} → ceiling {ceil_str} "
        "(robust peak; a median under-reads shot noise, so this is optimistic)"
    ]


def _print_cal_summary(
    result: "CalibrationResult", input_path: Path, elapsed: float, *, quiet: bool
) -> None:
    """The terminal summary table (or, under ``--quiet``, its warnings)."""

    if not quiet:
        aprint("\n" + "═" * 64)
        aprint(f"  CALIBRATION  —  {input_path.name}")
        aprint("═" * 64)
        aprint(f"  Volume:         {tuple(result.volume_shape)} {result.volume_dtype}")
        # Surface region provenance: under --auto-region the Volume / PSNR_full
        # below are CROP-scoped, not whole-volume (M12).
        if result.calibration_region is not None:
            reg = result.calibration_region
            aprint(
                f"  Region:         [{reg['strategy']}] origin={reg['origin']} "
                f"of full {tuple(result.original_volume_shape or [])}"
            )
            aprint("                  (Volume / PSNR_full above are for this crop)")
        for line in _noise_lines(result):
            aprint(line)
        aprint("")
        selected_peak, selected_metric = _print_cal_table(result)
        # Headline = the K* under the metric the user actually selected.
        aprint(
            f"  ★ Recommended K* = {selected_peak.k_star:,}  "
            f"(metric: {selected_metric}, type: {selected_peak.type}, "
            f"confidence: {selected_peak.confidence_db:.2f} dB)"
        )
        _print_stored_note(result, selected_peak.k_star)
        if result.status != "complete":
            aprint(
                f"  ⚠ {result.status.upper()}: measured {len(result.k_values_requested)} "
                "point(s); this K* is the best so far, not the sweep's answer."
            )
        _print_k_star_scope_caveat(result.calibration_region, result.whole_volume_seeds)
        _print_grid_warnings(
            selected_peak, "    ⚠ ", skip_high_edge=result.not_converged
        )
        # Operating point (point of diminishing returns) — equals K* for peak/
        # plateau; for a signal-limited curve it is the earlier knee (K* stays
        # the max-K budget anchor).
        if selected_peak.k_knee and selected_peak.k_knee != selected_peak.k_star:
            aprint(
                f"    operating point (diminishing returns) = "
                f"{selected_peak.k_knee:,} splats"
            )
        if result.held_out_peak_selected is not None:
            truncated = bool((result.sweep or {}).get("minmax_curve_truncated"))
            aprint(
                f"    (psnr_minmax K* = {result.held_out_peak.k_star:,}, "
                f"type: {result.held_out_peak.type}"
                f"{', on the early-stopped curve' if truncated else ''})"
            )
        if result.splat_density is not None:
            sd = result.splat_density
            aprint(
                f"  Density: {sd['k_star_reference']:,} splats / "
                f"{sd['n_features_reference']:,} {sd['feature_method']} features"
                f"  →  K ~ features^{sd['saturation_exponent']:.2f}"
            )
        _print_regime_warnings(result)
        aprint(f"  Total wall-clock: {elapsed:.1f} s")
        aprint("═" * 64)
    else:
        # --quiet drops the table, never the grid-adequacy warnings: a K*
        # on the grid edge is a bound, and that must not go unsaid.
        _print_grid_warnings(_selected_summary(result)[0], "⚠ ")


#: How to get matplotlib, which only the ``--pdf`` report (and the per-K slice
#: montages it draws from ``--keep-fits``) needs.
PDF_REPORT_INSTALL_HINT = (
    "pip install matplotlib (or install the luxar[demos] extra, which provides it)"
)


def _require_pdf_report_dependencies(pdf_report: Optional[Path]) -> None:
    """Exit before any work when ``--pdf`` is set but matplotlib is missing."""
    if pdf_report is None:
        return
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        aprint(
            f"Error: --pdf needs matplotlib, which is not installed ({exc}). "
            f"Install it with: {PDF_REPORT_INSTALL_HINT}; or drop --pdf."
        )
        raise typer.Exit(1) from exc


def run_calibrate_command(
    *,
    input_path: Path,
    output_json: Path,
    k_grid: Optional[str],
    n_grid: int,
    k_min: Optional[int],
    k_max: Optional[int],
    progression: str,
    power: int,
    mask_seed: int,
    mask_fraction: float,
    preset: str,
    config: Optional[Path],
    floor: Optional[str],
    device: Optional[str],
    channel: Optional[int],
    timepoint: Optional[int],
    array_key: Optional[str],
    axes: Optional[str],
    k_star_metric: str,
    fg_bg_ratio: float,
    auto_region: bool,
    region_size: int,
    region_strategy: str,
    feature_metric: str,
    saturation_exponent: float,
    rd_model: bool,
    fit_exponent: bool,
    exponent_scales: Optional[str],
    pdf_report: Optional[Path],
    keep_fits: Optional[Path],
    quiet: bool,
    voxel_size: Optional[str] = None,
    physical: bool = False,
    region_spec: Optional[str] = None,
    early_stop: bool = True,
    extend_below: bool = True,
) -> None:
    """Run calibration command implementation."""
    # A usage error, raised before the generic handler below so Typer reports
    # it as one (exit 2), as `fit` does.
    region_spans = _resolve_cal_region(region_spec, auto_region)
    # Before the sweep, not after it: the report is the last step of a run
    # that can take hours.
    _require_pdf_report_dependencies(pdf_report)
    try:
        from luxar.cli.gsplat_config import load_fit_config, load_volume
        from luxar.cli.gsplat_ops.voxel_frame import region_rank_guard
        from luxar.gsplats.calibration import (
            input_element_count,
            select_calibration_region,
        )
        from luxar.gsplats.utils.cpu_threads import cpu_threads_for

        # Resolve the device flags once for both CPU thread setup and the sweep.
        fit_kwargs = load_fit_config(
            preset=preset,
            config_path=config,
            cli_overrides={"device": device, "floor": floor},
        )
        # Calibration runs many fits; keep them quiet.
        fit_kwargs["verbose"] = False
        # 2. Load volume (reuse the loader used by `compare` and `fit`)
        with (
            contextlib.ExitStack() as stack,
            asection(f"Calibration: {input_path.name}"),
            cpu_threads_for(
                fit_kwargs["device"],
                use_cuda=fit_kwargs["use_cuda"],
                use_metal=fit_kwargs["use_metal"],
            ),
        ):
            with asection("Loading volume"):
                source_info: dict = {}
                with region_rank_guard():
                    volume = load_volume(
                        input_path,
                        channel=channel,
                        timepoint=timepoint,
                        array_key=array_key,
                        axes=axes,
                        info=source_info,
                        region=region_spans,
                    )
                input_frame = _cal_input_frame(
                    input_path=input_path,
                    source_info=source_info,
                    volume_shape=tuple(volume.shape),
                    voxel_size=voxel_size,
                    physical=physical,
                    preset=preset,
                    config=config,
                    array_key=array_key,
                    axes=axes,
                    region=region_spans,
                )

            # Optionally calibrate on a content-rich sub-region (the manuscript
            # itself crops to ~20 M voxels; this automates that at tile scale).
            original_shape = list(volume.shape)
            # Keep the pre-crop volume so --fit-exponent can select its own
            # per-scale regions from the full data (independent of --auto-region).
            volume_full = volume
            region_info: Optional[dict] = None
            if auto_region:
                from dataclasses import asdict as _asdict

                with asection("Selecting content-rich calibration region"):
                    volume, region = select_calibration_region(
                        volume,
                        region_size=region_size,
                        strategy=region_strategy,
                        feature=feature_metric,
                    )
                    region_info = _asdict(region)
                    aprint(
                        f"Region [{region.strategy}]: origin={region.origin} "
                        f"size={region.size} n_features={region.n_features}"
                    )

            aprint(
                f"Mask: {mask_fraction * 100:.1f}% (seed={mask_seed}); donut radius=1"
            )
            # 1. Resolve the K grid — after the load and crop, because the
            # default bounds are derived from the volume being calibrated.
            ks, control = _resolve_cal_grid(
                volume,
                k_grid=k_grid,
                n_grid=n_grid,
                k_min=k_min,
                k_max=k_max,
                progression=progression,
                power=power,
                floor_spec=fit_kwargs.get("floor", "auto"),
                early_stop=early_stop,
                extend_below=extend_below,
            )
            aprint(
                f"K*-metric: {k_star_metric}; fg:bg={fg_bg_ratio:g}; "
                f"feature metric: {feature_metric}"
            )

            # The resolved spacing shapes every K's fit (see _apply_cal_spacing).
            _apply_cal_spacing(fit_kwargs, input_frame)

            # 3. Prepare the optional per-K outputs. The PDF's slice page needs
            # the per-K fits, so --pdf alone keeps them in a temporary directory
            # (said now, not after a sweep that can take hours).
            keep_fits, temp_fits = _resolve_keep_fits(
                keep_fits, pdf_report, stack, output_json
            )

            # 4. Run sweep
            def _on_progress(i: int, n: int, msg: str) -> None:
                aprint(f"  [{i + 1}/{n}] {msg}")

            # Region provenance: calibrate() works on whatever array it is
            # handed; the CLI owns the crop, so it stamps the provenance — on
            # every per-point checkpoint too, so a crash leaves a usable file.
            def _stamp(r: "CalibrationResult") -> "CalibrationResult":
                r.original_volume_shape = original_shape
                r.input_frame = input_frame or None
                r.fit_config.setdefault("preset", preset)
                if temp_fits:
                    r.splat_paths = None  # deleted with the temporary directory
                if region_info is not None:
                    r.calibration_region = region_info
                return r

            with asection(f"Sweeping {len(ks)} fits"):
                t0 = time.perf_counter()
                result, interrupted = _run_cal_sweep(
                    volume,
                    ks,
                    output_json,
                    stamp=_stamp,
                    fit_kwargs=fit_kwargs,
                    mask_seed=mask_seed,
                    mask_fraction=mask_fraction,
                    keep_fits=keep_fits,
                    progress_callback=_on_progress,
                    k_star_metric=k_star_metric,
                    fg_bg_ratio=fg_bg_ratio,
                    feature_method=feature_metric,
                    saturation_exponent=saturation_exponent,
                    compute_rd_model=rd_model,
                    sweep_control=control,
                )
                elapsed = time.perf_counter() - t0
            splat_paths = result.splat_paths  # before _stamp drops temporary ones
            _stamp(result)
            # The sweep is done: write the COMPLETE cal.json now, so a later
            # step (exponent fit, whole-input read) can never leave it partial.
            result.to_json(output_json)

            def _post_sweep() -> None:
                # 4b. Optional multi-scale fit of the saturation exponent alpha
                #     (overrides the assumed default in splat_density).
                _fit_exponent_into(
                    result,
                    volume_full,
                    ks,
                    exponent_scales=exponent_scales,
                    saturation_exponent=saturation_exponent,
                    fit_kwargs=fit_kwargs,
                    feature_metric=feature_metric,
                    region_strategy=region_strategy,
                    k_star_metric=k_star_metric,
                    fg_bg_ratio=fg_bg_ratio,
                    mask_seed=mask_seed,
                    mask_fraction=mask_fraction,
                    progress=_on_progress,
                    enabled=fit_exponent,
                )

                # 4c. A crop's K* is a crop budget: restate it for the whole input.
                _suggest_whole_volume_seeds(
                    result,
                    volume_full,
                    lambda: load_volume(
                        input_path,
                        channel=channel,
                        timepoint=timepoint,
                        array_key=array_key,
                        axes=axes,
                    ),
                    region_spans=region_spans,
                    interrupted=False,
                    probe_size=lambda: input_element_count(input_path, array_key),
                )

            interrupted = _run_post_sweep(result, _post_sweep, skip=interrupted)

            # 5-6. Write JSON (+ optional PDF). Temporary fits are deleted with
            # the stack, so cal.json does not point at them (see _stamp).
            _write_cal_outputs(
                result, volume, output_json, pdf_report, splat_paths=splat_paths
            )

        # 7. Print formatted table
        _print_cal_summary(result, input_path, elapsed, quiet=quiet)
        if interrupted:
            raise typer.Exit(130)

    except (typer.Exit, typer.BadParameter):
        raise
    except Exception as exc:
        exit_with_error(f"Error: {exc}", exc)
