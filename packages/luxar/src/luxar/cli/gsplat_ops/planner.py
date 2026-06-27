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
    feature_metric: Optional[str] = typer.Option(
        None,
        "--feature-metric",
        help="Content metric: peaks | edges | intensity. Defaults to the density's "
        "calibrated metric (from --cal); must match it or budgets mis-scale. "
        "With --k-star-ref it defaults to 'peaks'.",
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
    jobs: str = typer.Option(
        "1",
        "--jobs",
        "-j",
        help="With --fit: number of boxes to fit concurrently as subprocesses on "
        "one GPU (int, or 'auto' to size from free VRAM). Default 1 = sequential.",
    ),
    keep_boxes: bool = typer.Option(
        False,
        "--keep-boxes",
        help="With --fit --jobs>1: keep the per-box temporary .gsplats.zarr "
        "outputs (and .empty markers) instead of deleting them after the merge.",
    ),
    fit_box: Optional[int] = typer.Option(
        None,
        "--fit-box",
        hidden=True,
        help="Internal worker mode: fit ONLY box i of an existing plan.json and "
        "save it to --output (used by --jobs>1 subprocess workers).",
    ),
) -> None:
    """Plan a content-balanced tiling of a volume; optionally fit it."""
    try:
        from luxar.cli.gsplat_config import load_fit_config, load_volume
        from luxar.gsplats.planner import plan_volume

        # Internal worker mode: fit ONLY box `fit_box` of an EXISTING plan.json
        # and write it to --output (or a sibling .empty marker for a 0-splat box).
        # Reuses _fit_one_box so the parallel path's per-box logic is identical to
        # the sequential fit_planned. No scan / no plan rewrite.
        if fit_box is not None:
            from luxar.gsplats.gsplat_data import GSplatData
            from luxar.gsplats.planner import FitPlan
            from luxar.gsplats.planner.fit_planned import _fit_one_box

            if output is None:
                raise typer.BadParameter("--fit-box requires --output/-o")
            plan = FitPlan.from_json(plan_json)
            if fit_box < 0 or fit_box >= len(plan.boxes):
                raise typer.BadParameter(
                    f"--fit-box {fit_box} out of range [0, {len(plan.boxes)})"
                )
            volume = load_volume(
                input_path, channel=channel, timepoint=timepoint, array_key=array_key
            )
            fit_kwargs = load_fit_config(
                preset=preset, config_path=None, cli_overrides={"device": device}
            )
            fit_kwargs.pop("seeds", None)
            fit_kwargs.pop("device", None)
            fit_kwargs.setdefault("cull_retention", 0.999)
            fit_kwargs["verbose"] = False
            fit_kwargs["device"] = device
            cap = int(plan.density.get("saturation_cap", 0)) if plan.density else 0
            c, a, k = _fit_one_box(
                volume, plan.boxes[fit_box], int(plan.overlap), cap, **fit_kwargs
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            if c.shape[0] == 0:
                # the gsplats writer rejects empty stores -> drop an .empty marker
                Path(str(output) + ".empty").write_text("")
            else:
                GSplatData(centers=c, amplitudes=a, cholesky_factors=k).save(
                    output, include_fitting_info=False
                )
            return

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

            # Reconcile the scan metric with the density's calibrated metric — they
            # MUST match or per-box budgets are silently mis-scaled (predict_k
            # divides by a reference counted with a different detector).
            scan_metric = feature_metric or density.feature_method
            if (
                feature_metric is not None
                and cal is not None
                and feature_metric != density.feature_method
            ):
                aprint(
                    f"⚠ --feature-metric '{feature_metric}' differs from the calibrated "
                    f"density.feature_method '{density.feature_method}' — per-box budgets "
                    f"will be mis-scaled. Use matching metrics."
                )

            # 2. Scan + plan
            with asection("Scanning content + planning"):
                t0 = time.perf_counter()
                plan = plan_volume(
                    volume,
                    density,
                    feature_method=scan_metric,
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

            # 3. Optional fit (sequential, or concurrent box subprocesses with -j)
            if fit:
                if output is None:
                    raise typer.BadParameter("--fit requires --output/-o")
                from luxar.gsplats.fit_tiled_parallel import resolve_jobs
                from luxar.gsplats.planner.fit_planned_parallel import (
                    max_padded_box_voxels,
                )

                n_budgeted = sum(1 for b in plan.boxes if b.budget > 0)
                try:
                    n_jobs = resolve_jobs(
                        jobs,
                        tile_voxels=max_padded_box_voxels(plan),
                        num_tiles=max(1, n_budgeted),
                        device=device,
                    )
                except ValueError:
                    aprint(f"Error: --jobs must be an integer or 'auto', got '{jobs}'")
                    raise typer.Exit(1)

                output.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.perf_counter()
                if n_jobs > 1:
                    # Concurrent: one subprocess per box (own CUDA context). Workers
                    # re-read the plan_json we just wrote and re-load the volume.
                    from luxar.gsplats.planner.fit_planned_parallel import (
                        _default_worker_cmd_builder,
                        fit_planned_parallel,
                    )

                    builder = _default_worker_cmd_builder(
                        input_path,
                        plan_json,
                        preset=preset,
                        device=device,
                        channel=channel,
                        timepoint=timepoint,
                        array_key=array_key,
                    )
                    tmp_dir = output.parent / f".{output.name}.boxes"
                    merged = fit_planned_parallel(
                        plan,
                        jobs=n_jobs,
                        tmp_dir=tmp_dir,
                        worker_cmd_builder=builder,
                        keep_boxes=keep_boxes,
                    )
                else:
                    from luxar.gsplats.planner import fit_planned

                    fit_kwargs = load_fit_config(
                        preset=preset,
                        config_path=None,
                        cli_overrides={"device": device},
                    )
                    fit_kwargs.pop("seeds", None)
                    # device is passed explicitly to fit_planned; drop it from the
                    # forwarded kwargs to avoid a duplicate keyword argument.
                    fit_kwargs.pop("device", None)
                    fit_kwargs["verbose"] = False
                    with asection(f"Fitting {plan.n_boxes} boxes"):

                        def _prog(i: int, n: int, msg: str) -> None:
                            aprint(f"  [{i + 1}/{n}] {msg}")

                        merged = fit_planned(
                            volume,
                            plan,
                            device=device,
                            progress_callback=_prog,
                            **fit_kwargs,
                        )
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
    feature_metric: Optional[str],
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
        feature_method=feature_metric or "peaks",
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
