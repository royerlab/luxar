"""Content-aware tiled fitting — the implementation of ``gsplat fit --tiling content``.

Consumes the splats-per-feature density emitted by ``luxar gsplat cal`` (the
calibration→planner interface) to build a content-balanced, size-bounded
``FitPlan`` (boxes + per-box budgets), then fits it (sequential or parallel box
subprocesses) and saves the result. ``fit_volume`` calls :func:`run_content_fit`
when ``--tiling content`` is selected; this is no longer a standalone command.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import typer
from arbol import aprint, asection

from .fitting.fit_utils import _invocation_token

if TYPE_CHECKING:
    from luxar.gsplats.calibration import SplatDensity


def _parallel_staging_dir(output: Path, token: str) -> Path:
    """Per-invocation staging dir for the parallel content-box fit.

    ``fit_planned_parallel`` clean-slates (``rmtree`` + ``mkdir``) whatever
    ``tmp_dir`` it is handed, so a directory derived only from the output
    path would let two concurrent ``fit -j`` runs to the SAME output delete
    each other's in-progress boxes. Appending a unique per-invocation
    ``token`` gives each invocation its own dir, so it only ever cleans its
    OWN boxes.
    """
    return output.parent / f".{output.name}.boxes.{token}"


def _internal_plan_json(output: Path, token: str) -> Path:
    """Per-invocation path for the internal plan JSON the workers re-read.

    Tokenized so concurrent content fits to the SAME output never overwrite
    or unlink each other's plan mid-launch (the plan is written, re-read by
    every box worker, then unlinked on completion).
    """
    return output.parent / f".{output.name}.plan.{token}.json"


def _save_fit_result(
    result: Any,
    output: Path,
    *,
    compress: "Optional[Literal['zip', 'tar.gz']]" = None,
) -> None:
    """Save either a flat ``GSplatData`` leaf or a ``kind=partition`` tree node."""
    from luxar.gsplats.gsplat_data import GSplatData

    if isinstance(result, GSplatData):
        result.save(output, include_fitting_info=True, compress=compress)
    else:  # a GSplatNode (partition / leaf tree) has no flat-matrix equivalent
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        write_gsplats_tree(output, result, compress=compress)


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
    config: Optional[Path] = None,
    iters: Optional[int] = None,
    loss: Optional[str] = None,
    lr: Optional[float] = None,
    floor: Optional[str] = None,
    cull_retention: Optional[float] = None,
    device: Optional[str] = None,
    jobs: str = "1",
    keep_boxes: bool = False,
    flat: bool = False,
    recipe: Optional[str] = None,
    recipe_params: Any = None,
    compress: "Optional[Literal['zip', 'tar.gz']]" = None,
    # plan I/O
    plan: Optional[Path] = None,
    plan_only: bool = False,
    plan_box: Optional[int] = None,
    # volume loader
    channel: Optional[int] = None,
    timepoint: Optional[int] = None,
    array_key: Optional[str] = None,
    axes: Optional[str] = None,
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

    def _section(title: str) -> Any:
        # Skip the section header/indent when quiet; the body still runs.
        return asection(title) if verbose else contextlib.nullcontext()

    def _load_vol() -> Any:
        if volume is not None:
            return volume
        return load_volume(
            input_path,
            channel=channel,
            timepoint=timepoint,
            array_key=array_key,
            axes=axes,
        )

    def _fit_kwargs() -> dict:
        # Honor the user's fit knobs (None values are ignored by load_fit_config),
        # so `fit --tiling content --iters/--config/--loss/--lr` — and the Slurm
        # content worker, which re-invokes the same path with those flags baked in
        # — are not silently dropped.
        fk = load_fit_config(
            preset=preset,
            config_path=config,
            cli_overrides={
                "device": device,
                "n_iters": iters,
                "loss_type": loss,
                "lr": lr,
                "floor": floor,
            },
        )
        fk.pop("seeds", None)
        fk.pop("device", None)
        if cull_retention is not None:
            fk["cull_retention"] = cull_retention
        else:
            fk.setdefault("cull_retention", 0.999)  # content default (near-lossless)
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
    if verbose:
        aprint(
            f"Density: {density.k_star_reference:,} splats / "
            f"{density.n_features_reference:,} {density.feature_method} features "
            f"→ K ~ features^{density.saturation_exponent:.2f} "
            f"(cap {density.saturation_cap:,})"
        )

    vol = _load_vol()

    # ── obtain a plan: load --plan, or scan + plan ──
    created_plan = False
    # One unique per-invocation token shared by the internal plan JSON and
    # the parallel staging dir, so concurrent content fits to the SAME output
    # can't clobber each other's plan or in-progress boxes (issue #1040).
    token = _invocation_token()
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
        with _section("Scanning content + planning"):
            from luxar.gsplats.fitting.preprocessing import _resolve_floor

            t0 = time.perf_counter()
            # Scan the SAME floor-suppressed volume the boxes will fit: cal
            # records `density.feature_threshold` on the floor-subtracted volume,
            # so scanning the raw (pedestal-carrying) volume would count the whole
            # background as signal and flatten the content field. Subtract the
            # fit's own floor here so the scan scale matches the calibration.
            scan_vol = vol
            # `floor=None` means "not overridden" — the per-box worker then
            # defaults to "auto", so the scan must resolve "auto" too (matching
            # what the boxes will actually fit).
            scan_floor = _resolve_floor(vol, "auto" if floor is None else floor)
            if scan_floor is not None:
                scan_vol = np.clip(
                    np.asarray(vol, dtype=np.float32) - scan_floor, 0.0, None
                )
            fitplan = plan_volume(
                scan_vol,
                density,
                feature_method=scan_metric,
                cell=cell,
                target_features=target_features,
                min_leaf=min_leaf,
                max_leaf=max_leaf,
                overlap=overlap,
            )
            med, mx = fitplan.overlap_fraction()
            if verbose:
                aprint(
                    f"Plan: {fitplan.n_boxes} boxes, total budget "
                    f"{fitplan.total_budget:,} splats, overlap median "
                    f"{med:.0%} / max {mx:.0%}  "
                    f"({time.perf_counter() - t0:.1f}s)"
                )
        if plan_only:
            if output is None:
                raise typer.BadParameter(
                    "--plan-only requires --output/-o (plan JSON path)"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            fitplan.to_json(output)
            if verbose:
                aprint(f"Wrote plan: {output}")
            return
        if output is None:
            raise typer.BadParameter("content fit requires --output/-o")
        # internal plan JSON the parallel workers re-read (per-invocation)
        plan_json_path = _internal_plan_json(output, token)
        plan_json_path.parent.mkdir(parents=True, exist_ok=True)
        fitplan.to_json(plan_json_path)
        created_plan = True

    if plan_only:  # --plan-only with an explicit --plan: nothing to compute
        if verbose:
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
            floor=floor,
            channel=channel,
            timepoint=timepoint,
            array_key=array_key,
            axes=axes,
        )
        # Per-invocation staging (reusing the shared unique token): so
        # concurrent `fit -j` runs targeting one output can't clobber each
        # other's in-progress boxes — the helper clean-slates only its OWN dir.
        tmp_dir = _parallel_staging_dir(output, token)
        with _section(f"Fitting {fitplan.n_boxes} boxes ({n_jobs} concurrent)"):
            result = fit_planned_parallel(
                fitplan,
                jobs=n_jobs,
                tmp_dir=tmp_dir,
                worker_cmd_builder=builder,
                keep_boxes=keep_boxes,
                partition=partition,
                recipe=recipe,
                recipe_params=recipe_params,
                verbose=verbose,
            )
    else:
        fk = _fit_kwargs()
        with _section(f"Fitting {fitplan.n_boxes} boxes"):

            def _prog(i: int, n: int, msg: str) -> None:
                if verbose:
                    aprint(f"  [{i + 1}/{n}] {msg}")

            result = fit_planned(
                vol,
                fitplan,
                device=device,
                partition=partition,
                recipe=recipe,
                recipe_params=recipe_params,
                progress_callback=_prog,
                **fk,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    _save_fit_result(result, output, compress=compress)
    kind = "partition" if partition else "leaf"
    if verbose:
        aprint(
            f"Fit {fitplan.n_boxes} boxes → {kind} "
            f"in {time.perf_counter() - t0:.1f}s → {output}"
        )
    if created_plan and not keep_boxes:
        Path(plan_json_path).unlink(missing_ok=True)
    elif created_plan and verbose:
        # --keep-boxes retains the internal plan too; its token-suffixed name
        # is no longer predictable from the output path, so point at it.
        aprint(f"Kept plan at {plan_json_path}")


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
