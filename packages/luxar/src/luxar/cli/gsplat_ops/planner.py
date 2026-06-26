"""``luxar gsplat plan`` — content-aware fit planning (and optional fitting).

Consumes the splats-per-feature density emitted by ``luxar gsplat cal`` (the
calibration→planner interface) and produces a content-balanced, size-bounded
``FitPlan`` (boxes + per-box budgets). With ``--fit`` it also runs the plan and
saves the merged ``.gsplats.zarr`` — the full Phase-2 pipeline in one command.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
from arbol import aprint, asection

if TYPE_CHECKING:
    from luxar.gsplats.calibration import SplatDensity


def plan_command(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input volume (.zarr, .zarr.zip, .tiff, .npy, .npz)"
    ),
    plan_json: Path = typer.Argument(..., help="Output FitPlan JSON (boxes + budgets)"),
    cal: Optional[Path] = typer.Option(
        None,
        "--cal",
        help="Calibration JSON (from `gsplat cal`) — supplies the splats-per-feature "
        "density used to size per-box budgets. Required unless --k-star-ref/"
        "--n-features-ref are given explicitly.",
    ),
    # explicit density override (when no cal.json is available)
    k_star_ref: Optional[int] = typer.Option(
        None, "--k-star-ref", help="Reference K* (effective splats) for the density."
    ),
    n_features_ref: Optional[int] = typer.Option(
        None, "--n-features-ref", help="Reference feature count for the density."
    ),
    saturation_exponent: float = typer.Option(
        0.44, "--saturation-exponent", help="Sub-linear exponent alpha (K~feat^alpha)."
    ),
    saturation_cap: Optional[int] = typer.Option(
        None, "--saturation-cap", help="Per-box splat cap (default: 4x k_star_ref)."
    ),
    feature_threshold: Optional[float] = typer.Option(
        None,
        "--feature-threshold",
        help="Absolute intensity threshold for feature counting (the level the "
        "calibration counted at). Taken from --cal automatically; pass explicitly "
        "when using --k-star-ref/--n-features-ref so box counts match the reference.",
    ),
    # planner knobs
    feature_metric: str = typer.Option(
        "peaks", "--feature-metric", help="Content metric: peaks | edges | intensity."
    ),
    cell: int = typer.Option(16, "--cell", help="Content-scan cell size (voxels)."),
    target_features: Optional[int] = typer.Option(
        None,
        "--target-features",
        help="Features per leaf to split toward (default: density's reference count).",
    ),
    min_leaf: int = typer.Option(256, "--min-leaf", help="Minimum leaf edge (voxels)."),
    max_leaf: int = typer.Option(512, "--max-leaf", help="Maximum leaf edge (voxels)."),
    overlap: int = typer.Option(32, "--overlap", help="Per-box halo for seamless fit."),
    # volume loader pass-through
    channel: Optional[int] = typer.Option(None, "--channel", "-c"),
    timepoint: Optional[int] = typer.Option(None, "--timepoint"),
    array_key: Optional[str] = typer.Option(None, "--array-key"),
    # optional fit
    fit: bool = typer.Option(
        False, "--fit", help="Also fit the plan and save the merged .gsplats.zarr."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output .gsplats.zarr (required with --fit)."
    ),
    preset: str = typer.Option("standard", "--preset", help="Fit preset (with --fit)."),
    device: Optional[str] = typer.Option(None, "--device", "-d"),
) -> None:
    """Plan a content-balanced tiling of a volume; optionally fit it."""
    try:
        from luxar.cli.gsplat_config import load_fit_config, load_volume
        from luxar.gsplats.planner import plan_volume

        with asection(f"Plan: {input_path.name}"):
            with asection("Loading volume"):
                volume = load_volume(
                    input_path,
                    channel=channel,
                    timepoint=timepoint,
                    array_key=array_key,
                )

            # 1. Resolve the splats-per-feature density (cal.json or explicit flags)
            density = _resolve_density(
                cal,
                k_star_ref,
                n_features_ref,
                saturation_exponent,
                saturation_cap,
                feature_metric,
                feature_threshold,
            )
            aprint(
                f"Density: {density.k_star_reference:,} splats / "
                f"{density.n_features_reference:,} {density.feature_method} features "
                f"→ K ~ features^{density.saturation_exponent:.2f} (cap {density.saturation_cap:,})"
            )

            # 2. Scan + plan
            with asection("Scanning content + planning"):
                t0 = time.perf_counter()
                plan = plan_volume(
                    volume,
                    density,
                    feature_method=feature_metric,
                    cell=cell,
                    target_features=target_features,
                    min_leaf=min_leaf,
                    max_leaf=max_leaf,
                    overlap=overlap,
                    device=device,
                )
                plan_json.parent.mkdir(parents=True, exist_ok=True)
                plan.to_json(plan_json)
                med, mx = plan.overlap_fraction()
                aprint(
                    f"Plan: {plan.n_boxes} boxes, total budget {plan.total_budget:,} "
                    f"splats, overlap median {med:.0%} / max {mx:.0%}  "
                    f"({time.perf_counter() - t0:.1f}s)"
                )
                aprint(f"Wrote {plan_json}")

            # 3. Optional fit
            if fit:
                if output is None:
                    raise typer.BadParameter("--fit requires --output/-o")
                from luxar.gsplats.planner import fit_planned

                fit_kwargs = load_fit_config(
                    preset=preset, config_path=None, cli_overrides={"device": device}
                )
                fit_kwargs.pop("seeds", None)
                # device is passed explicitly to fit_planned; drop it from the
                # forwarded kwargs to avoid a duplicate keyword argument.
                fit_kwargs.pop("device", None)
                fit_kwargs["verbose"] = False
                with asection(f"Fitting {plan.n_boxes} boxes"):
                    t0 = time.perf_counter()

                    def _prog(i: int, n: int, msg: str) -> None:
                        aprint(f"  [{i + 1}/{n}] {msg}")

                    merged = fit_planned(
                        volume,
                        plan,
                        device=device,
                        progress_callback=_prog,
                        **fit_kwargs,
                    )
                    output.parent.mkdir(parents=True, exist_ok=True)
                    merged.save(output, include_fitting_info=True)
                    aprint(
                        f"Fit {merged.n_splats:,} splats from {plan.n_boxes} boxes "
                        f"in {time.perf_counter() - t0:.1f}s → {output}"
                    )
    except typer.Exit:
        raise
    except Exception as exc:
        aprint(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def _resolve_density(
    cal: Optional[Path],
    k_star_ref: Optional[int],
    n_features_ref: Optional[int],
    saturation_exponent: float,
    saturation_cap: Optional[int],
    feature_metric: str,
    feature_threshold: Optional[float] = None,
) -> "SplatDensity":
    from luxar.gsplats.calibration import CalibrationResult, SplatDensity

    if cal is not None:
        result = CalibrationResult.from_json(cal)
        if result.splat_density is None:
            raise typer.BadParameter(
                f"{cal} has no splat_density (re-run `gsplat cal` to emit it)."
            )
        dens = SplatDensity(**result.splat_density)
        if feature_threshold is not None:  # explicit override
            dens.feature_threshold = float(feature_threshold)
        return dens
    if k_star_ref is None or n_features_ref is None:
        raise typer.BadParameter(
            "provide --cal CAL.json, or both --k-star-ref and --n-features-ref."
        )
    cap = saturation_cap if saturation_cap is not None else int(k_star_ref * 4)
    return SplatDensity(
        feature_method=feature_metric,
        n_features_reference=int(n_features_ref),
        k_star_reference=int(k_star_ref),
        saturation_exponent=float(saturation_exponent),
        saturation_cap=int(cap),
        splats_per_feature=float(k_star_ref) / max(1, int(n_features_ref)),
        feature_threshold=float(feature_threshold or 0.0),
    )


def register_planner_commands(app: "typer.Typer") -> None:
    """Register the planner commands onto ``app_gsplat``."""
    app.command("plan")(plan_command)


__all__ = ["plan_command", "register_planner_commands"]
