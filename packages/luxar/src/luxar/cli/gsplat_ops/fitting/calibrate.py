"""Implementation helper for gsplat calibration command."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
from arbol import aprint, asection

from ..._traceback import exit_with_error

if TYPE_CHECKING:
    from luxar.gsplats.calibration import CalibrationResult, HeldOutPeak


def _print_k_star_scope_caveat(calibration_region: Optional[dict]) -> None:
    """Qualify the ``★ Recommended K*`` headline when it is region-scoped.

    Under ``--auto-region`` K* was measured on the crop, so it is a REGION
    budget (roughly tile-scale) — feeding it to ``fit --seeds`` (a whole-volume
    budget, divided across tiles) would under-seed the volume. Point at the
    density transfer instead. Prints nothing for a whole-volume calibration,
    including ``strategy="whole"``: that means the volume was no bigger than
    ``--region-size`` on every axis, so the "crop" IS the whole volume and its
    K* is exactly what ``--seeds`` wants.
    """
    if calibration_region is None or calibration_region.get("strategy") == "whole":
        return
    aprint(
        "    ⚠ region-scoped (measured on the crop above), NOT a "
        "whole-volume budget: do not pass it to `fit --seeds`; "
        "transfer it with `fit --tiling content --cal <this json>`."
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


def run_calibrate_command(
    *,
    input_path: Path,
    output_json: Path,
    k_grid: Optional[str],
    n_grid: int,
    k_min: int,
    k_max: int,
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
) -> None:
    """Run calibration command implementation."""
    try:
        import math

        from luxar.cli.gsplat_config import load_fit_config, load_volume
        from luxar.gsplats.calibration import (
            build_k_grid,
            calibrate,
            select_calibration_region,
        )

        # 1. Resolve K grid
        explicit: Optional[list[int]] = None
        if k_grid is not None:
            explicit = [int(x.strip()) for x in k_grid.split(",") if x.strip()]
        ks = build_k_grid(
            explicit=explicit,
            n_points=n_grid,
            k_min=k_min,
            k_max=k_max,
            progression=progression,
            power=power,
        )

        # 2. Load volume (reuse the loader used by `compare` and `fit`)
        with asection(f"Calibration: {input_path.name}"):
            with asection("Loading volume"):
                volume = load_volume(
                    input_path,
                    channel=channel,
                    timepoint=timepoint,
                    array_key=array_key,
                    axes=axes,
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
            aprint(f"K grid ({len(ks)} points): {ks}")
            aprint(
                f"K*-metric: {k_star_metric}; fg:bg={fg_bg_ratio:g}; "
                f"feature metric: {feature_metric}"
            )

            # 3. Build fit kwargs from preset + YAML config + CLI overrides
            fit_kwargs = load_fit_config(
                preset=preset,
                config_path=config,
                cli_overrides={"device": device, "floor": floor},
            )
            # Calibration runs many fits — keep them quiet
            fit_kwargs["verbose"] = False

            if keep_fits is not None:
                keep_fits = Path(keep_fits)
                keep_fits.mkdir(parents=True, exist_ok=True)
                aprint(f"Per-K fits will be saved under {keep_fits}")

            # 4. Run sweep
            def _on_progress(i: int, n: int, msg: str) -> None:
                aprint(f"  [{i + 1}/{n}] {msg}")

            with asection(f"Sweeping {len(ks)} fits"):
                t0 = time.perf_counter()
                result = calibrate(
                    volume,
                    k_grid=ks,
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
                )
                elapsed = time.perf_counter() - t0

            # Record region provenance (calibrate() works on whatever array it
            # is handed; the CLI owns the crop, so it stamps the provenance).
            result.original_volume_shape = original_shape
            if region_info is not None:
                result.calibration_region = region_info

            # 4b. Optional multi-scale fit of the saturation exponent alpha.
            #     Calibrates K* at several region scales and regresses log K* on
            #     log n_features; the fitted alpha overrides the assumed default
            #     in splat_density (which the planner / fit --tiling content read).
            if fit_exponent:
                from dataclasses import asdict as _asdict_fit

                from luxar.gsplats.calibration import calibrate_saturation_exponent

                if exponent_scales:
                    try:
                        raw_scales = [
                            int(x) for x in exponent_scales.split(",") if x.strip()
                        ]
                    except ValueError as e:
                        raise typer.BadParameter(
                            "--exponent-scales must be comma-separated integers; "
                            f"got {exponent_scales!r}"
                        ) from e
                else:
                    raw_scales = [128, 192, 256]
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
                with asection(
                    f"Fitting saturation exponent over {len(scales)} scale(s)"
                ):
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
                            splat_paths=result.splat_paths,
                        )
                        aprint(f"Wrote {pdf_report}")
                    except ImportError as exc:
                        aprint(
                            f"Skipping PDF report: optional dependency missing ({exc})"
                        )
                        aprint(
                            "Install matplotlib to enable --pdf: pip install matplotlib"
                        )

        # 7. Print formatted table
        if not quiet:
            aprint("\n" + "═" * 64)
            aprint(f"  CALIBRATION  —  {input_path.name}")
            aprint("═" * 64)
            aprint(
                f"  Volume:         {tuple(result.volume_shape)} {result.volume_dtype}"
            )
            # Surface region provenance: under --auto-region the Volume / PSNR_full
            # below are CROP-scoped, not whole-volume (M12).
            if result.calibration_region is not None:
                reg = result.calibration_region
                aprint(
                    f"  Region:         [{reg['strategy']}] origin={reg['origin']} "
                    f"of full {tuple(result.original_volume_shape or [])}"
                )
                aprint("                  (Volume / PSNR_full above are for this crop)")
            sigma = result.noise_floor.sigma_hat
            ceil_db = result.noise_floor.psnr_max_db
            sigma_str = f"{sigma:.4f}" if math.isfinite(sigma) else "—"
            ceil_str = (
                f"{ceil_db:.1f} dB"
                if math.isfinite(ceil_db)
                else (">60 dB" if math.isinf(ceil_db) else "—")
            )
            aprint(f"  Noise floor:    σ = {sigma_str}, PSNR ceiling = {ceil_str}")
            aprint("")
            # The headline / table marker track the metric the user selected
            # (falls back to the legacy min--max peak when no metric switch).
            selected_peak, selected_metric, selected_curve = _selected_summary(result)
            aprint(
                "    K_req     K_eff    PSNR_train    K*-metric     PSNR_full   SSIM_full   fit (s)"
            )
            aprint("    " + "-" * 76)
            for i, k_req in enumerate(result.k_values_requested):
                k_eff = result.k_values_effective[i]
                pt = result.train_psnr_db[i]
                ph = selected_curve[i]
                pf = result.full_psnr_db[i]
                sf = result.full_ssim[i]
                ft = result.fit_times_seconds[i]

                def _f(x: float) -> str:
                    if math.isnan(x):
                        return "  nan"
                    if math.isinf(x):
                        return "  inf"
                    return f"{x:6.2f}"

                marker = "★" if k_req == selected_peak.k_star else " "
                aprint(
                    f"  {marker} {k_req:7d}  {k_eff:7d}    {_f(pt)} dB     {_f(ph)} dB    {_f(pf)} dB    {sf:5.3f}    {ft:6.1f}"
                )
            aprint("")
            # Headline = the K* under the metric the user actually selected.
            aprint(
                f"  ★ Recommended K* = {selected_peak.k_star:,}  "
                f"(metric: {selected_metric}, type: {selected_peak.type}, "
                f"confidence: {selected_peak.confidence_db:.2f} dB)"
            )
            _print_k_star_scope_caveat(result.calibration_region)
            # Operating point (point of diminishing returns) — equals K* for peak/
            # plateau; for a signal-limited curve it is the earlier knee (K* stays
            # the max-K budget anchor).
            if selected_peak.k_knee and selected_peak.k_knee != selected_peak.k_star:
                aprint(
                    f"    operating point (diminishing returns) = "
                    f"{selected_peak.k_knee:,} splats"
                )
            if result.held_out_peak_selected is not None:
                aprint(
                    f"    (psnr_minmax K* = {result.held_out_peak.k_star:,}, "
                    f"type: {result.held_out_peak.type})"
                )
            if result.splat_density is not None:
                sd = result.splat_density
                aprint(
                    f"  Density: {sd['k_star_reference']:,} splats / "
                    f"{sd['n_features_reference']:,} {sd['feature_method']} features"
                    f"  →  K ~ features^{sd['saturation_exponent']:.2f}"
                )
            # Regime warnings — warn-by-default, no behaviour change
            sig = result.noise_floor.sigma_hat
            if math.isfinite(sig) and sig < 1e-4:
                aprint(
                    "  ⚠ σ̂≈0 (noise-free/deconvolved): the blind-spot peak may not "
                    "appear; prefer --k-star-metric psnr_fg_weighted "
                    "(and --auto-region)."
                )
            if result.not_converged:
                aprint(
                    "  ⚠ held-out curve still climbing at K_max (not converged): "
                    "extend --k-max or use the R-D-model extrapolation."
                )
            aprint(f"  Total wall-clock: {elapsed:.1f} s")
            aprint("═" * 64)

    except typer.Exit:
        raise
    except Exception as exc:
        exit_with_error(f"Error: {exc}", exc)
