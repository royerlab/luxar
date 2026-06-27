"""Content-aware tiled fitting — the implementation of ``gsplat fit --tiling content``.

Consumes the splats-per-feature density emitted by ``luxar gsplat cal`` (the
calibration→planner interface) to build a content-balanced, size-bounded
``FitPlan`` (boxes + per-box budgets), then fits it (sequential or parallel box
subprocesses) and saves the result. ``fit_volume`` calls :func:`run_content_fit`
when ``--tiling content`` is selected; this is no longer a standalone command.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from arbol import aprint, asection

if TYPE_CHECKING:
    from luxar.gsplats.calibration import SplatDensity


def _save_fit_result(result: Any, output: Path) -> None:
    """Save either a flat ``GSplatData`` leaf or a ``kind=partition`` tree node."""
    from luxar.gsplats.gsplat_data import GSplatData

    if isinstance(result, GSplatData):
        result.save(output, include_fitting_info=True)
    else:  # a GSplatNode (partition / leaf tree) has no flat-matrix equivalent
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        write_gsplats_tree(output, result)


def run_content_fit(
    input_path: Path,
    output: Optional[Path],
    *,
    volume: Any = None,
    # transferable density (cal.json or explicit reference)
    cal: Optional[Path] = None,
    k_star_ref: Optional[int] = None,
    n_features_ref: Optional[int] = None,
    saturation_exponent: float = 0.44,
    saturation_cap: Optional[int] = None,
    feature_threshold: Optional[float] = None,
    feature_metric: Optional[str] = None,
    # planner knobs
    cell: int = 16,
    target_features: Optional[int] = None,
    min_leaf: int = 256,
    max_leaf: int = 512,
    overlap: int = 32,
    # fit
    preset: Optional[str] = None,
    device: Optional[str] = None,
    jobs: str = "1",
    keep_boxes: bool = False,
    flat: bool = False,
    # plan I/O
    plan: Optional[Path] = None,
    plan_only: bool = False,
    plan_box: Optional[int] = None,
    # volume loader
    channel: Optional[int] = None,
    timepoint: Optional[int] = None,
    array_key: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """Content-aware tiled fit: scan → BSP plan → budgeted fit → save.

    Modes (selected by the flags ``fit_volume`` forwards):

    * ``plan_box`` set — internal worker: fit ONLY box ``plan_box`` of an
      existing ``--plan`` and write its leaf (or a ``.empty`` marker).
    * ``plan_only`` — write the ``FitPlan`` JSON to ``output`` and stop.
    * otherwise — build/load a plan and fit it (sequential, or ``-j`` parallel
      box subprocesses), saving a ``kind=partition`` (one part per box) unless
      ``flat``.
    """
    from luxar.cli.gsplat_config import load_fit_config, load_volume
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.planner import FitPlan, fit_planned, plan_volume
    from luxar.gsplats.planner.fit_planned import _fit_one_box

    def _load_vol() -> Any:
        if volume is not None:
            return volume
        return load_volume(
            input_path, channel=channel, timepoint=timepoint, array_key=array_key
        )

    def _fit_kwargs() -> dict:
        fk = load_fit_config(
            preset=preset, config_path=None, cli_overrides={"device": device}
        )
        fk.pop("seeds", None)
        fk.pop("device", None)
        fk.setdefault("cull_retention", 0.999)
        fk["verbose"] = False
        return fk

    # ── worker mode: fit ONE box of an existing plan (a -j parallel subprocess) ──
    if plan_box is not None:
        if output is None:
            raise typer.BadParameter("--plan-box requires --output/-o")
        if plan is None:
            raise typer.BadParameter("--plan-box requires --plan PLAN.json")
        fitplan = FitPlan.from_json(plan)
        if plan_box < 0 or plan_box >= len(fitplan.boxes):
            raise typer.BadParameter(
                f"--plan-box {plan_box} out of range [0, {len(fitplan.boxes)})"
            )
        vol = _load_vol()
        fk = _fit_kwargs()
        fk["device"] = device
        cap = int(fitplan.density.get("saturation_cap", 0)) if fitplan.density else 0
        c, a, k = _fit_one_box(
            vol, fitplan.boxes[plan_box], int(fitplan.overlap), cap, **fk
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if c.shape[0] == 0:
            Path(str(output) + ".empty").write_text("")  # writer rejects empty stores
        else:
            GSplatData(centers=c, amplitudes=a, cholesky_factors=k).save(
                output, include_fitting_info=False
            )
        return

    # ── resolve density ──
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

    vol = _load_vol()

    # ── obtain a plan: load --plan, or scan + plan ──
    created_plan = False
    if plan is not None:
        fitplan = FitPlan.from_json(plan)
        plan_json_path: Path = Path(plan)
    else:
        scan_metric = feature_metric or density.feature_method
        if (
            feature_metric is not None
            and cal is not None
            and feature_metric != density.feature_method
        ):
            aprint(
                f"⚠ --feature-metric '{feature_metric}' differs from the calibrated "
                f"density.feature_method '{density.feature_method}' — per-box budgets "
                "will be mis-scaled. Use matching metrics."
            )
        with asection("Scanning content + planning"):
            t0 = time.perf_counter()
            fitplan = plan_volume(
                vol,
                density,
                feature_method=scan_metric,
                cell=cell,
                target_features=target_features,
                min_leaf=min_leaf,
                max_leaf=max_leaf,
                overlap=overlap,
                device=device,
            )
            med, mx = fitplan.overlap_fraction()
            aprint(
                f"Plan: {fitplan.n_boxes} boxes, total budget {fitplan.total_budget:,} "
                f"splats, overlap median {med:.0%} / max {mx:.0%}  "
                f"({time.perf_counter() - t0:.1f}s)"
            )
        if plan_only:
            if output is None:
                raise typer.BadParameter(
                    "--plan-only requires --output/-o (plan JSON path)"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            fitplan.to_json(output)
            aprint(f"Wrote plan: {output}")
            return
        if output is None:
            raise typer.BadParameter("content fit requires --output/-o")
        # internal plan JSON the parallel workers re-read
        plan_json_path = output.parent / f".{output.name}.plan.json"
        plan_json_path.parent.mkdir(parents=True, exist_ok=True)
        fitplan.to_json(plan_json_path)
        created_plan = True

    if plan_only:  # --plan-only with an explicit --plan: nothing to compute
        aprint(f"Plan: {plan_json_path}")
        return
    if output is None:
        raise typer.BadParameter("content fit requires --output/-o")

    # ── fit (sequential or parallel box subprocesses); partition unless --flat ──
    from luxar.gsplats.fit_tiled_parallel import resolve_jobs
    from luxar.gsplats.planner.fit_planned_parallel import max_padded_box_voxels

    n_budgeted = sum(1 for b in fitplan.boxes if b.budget > 0)
    try:
        n_jobs = resolve_jobs(
            jobs,
            tile_voxels=max_padded_box_voxels(fitplan),
            num_tiles=max(1, n_budgeted),
            device=device,
        )
    except ValueError:
        aprint(f"Error: --jobs must be an integer or 'auto', got '{jobs}'")
        raise typer.Exit(1)

    partition = not flat
    t0 = time.perf_counter()
    if n_jobs > 1:
        from luxar.gsplats.planner.fit_planned_parallel import (
            _default_worker_cmd_builder,
            fit_planned_parallel,
        )

        builder = _default_worker_cmd_builder(
            input_path,
            plan_json_path,
            preset=preset or "standard",
            device=device,
            channel=channel,
            timepoint=timepoint,
            array_key=array_key,
        )
        tmp_dir = output.parent / f".{output.name}.boxes"
        with asection(f"Fitting {fitplan.n_boxes} boxes ({n_jobs} concurrent)"):
            result = fit_planned_parallel(
                fitplan,
                jobs=n_jobs,
                tmp_dir=tmp_dir,
                worker_cmd_builder=builder,
                keep_boxes=keep_boxes,
                partition=partition,
            )
    else:
        fk = _fit_kwargs()
        with asection(f"Fitting {fitplan.n_boxes} boxes"):

            def _prog(i: int, n: int, msg: str) -> None:
                aprint(f"  [{i + 1}/{n}] {msg}")

            result = fit_planned(
                vol,
                fitplan,
                device=device,
                partition=partition,
                progress_callback=_prog,
                **fk,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    _save_fit_result(result, output)
    kind = "partition" if partition else "leaf"
    aprint(
        f"Fit {fitplan.n_boxes} boxes → {kind} "
        f"in {time.perf_counter() - t0:.1f}s → {output}"
    )
    if created_plan and not keep_boxes:
        Path(plan_json_path).unlink(missing_ok=True)


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


__all__ = ["run_content_fit", "_resolve_density"]
