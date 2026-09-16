"""Content-aware tiled fitting — the implementation of ``gsplat fit --tiling content``.

Consumes the splats-per-feature density emitted by ``luxar gsplat cal`` (the
calibration→planner interface) to build a content-balanced, size-bounded
``FitPlan`` (boxes + per-box budgets), then fits it (sequential or parallel box
subprocesses) and saves the result. ``fit_volume`` calls :func:`run_content_fit`
when ``--tiling content`` is selected; this is no longer a standalone command.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import typer
from arbol import aprint, asection

from .fitting.fit_utils import (
    _invocation_token,
    resolve_shared_floor,
    save_fit_output,
    validate_floor_spec,
)

if TYPE_CHECKING:
    from luxar.gsplats.calibration import SplatDensity
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.planner import PlanBox


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


def _require_plan_volume_shape(volume: Any, fitplan: Any) -> None:
    """Reject a plan whose boxes were built for a different voxel grid."""
    actual = tuple(int(size) for size in np.shape(volume))
    expected = tuple(int(size) for size in fitplan.volume_shape)
    if actual != expected:
        raise typer.BadParameter(
            f"volume shape {actual} does not match the plan grid {expected}"
        )


def _validate_parallel_content_frame(
    physical_coordinates: bool,
    jobs: str | int,
    *,
    plan_only: bool = False,
    plan_box: Optional[int] = None,
) -> None:
    """Reject a parallel mode whose assembler cannot scale the plan tree."""
    if plan_only or plan_box is not None:
        return
    try:
        n_jobs = int(jobs)
    except ValueError:
        return
    if physical_coordinates and n_jobs > 1:
        raise typer.BadParameter(
            "--physical-coordinates is not supported with parallel content fitting "
            "(-j/--jobs > 1); use -j 1."
        )


def _fill_source_dtype(fit_config: dict, source_dtype: Optional[str]) -> None:
    """Use the loader dtype unless configured; mirrors fit._stamp_source_dtype."""
    if not fit_config.get("source_dtype") and source_dtype:
        fit_config["source_dtype"] = source_dtype


def _stamp_content_box_output(
    result: "GSplatData",
    volume: Any,
    box: "PlanBox",
    device: Optional[str],
    *,
    verbose: bool,
    grid_scale: "Optional[tuple[float, ...]]" = None,
) -> None:
    """Describe a standalone content box after its halo splats were removed.

    The score measures the box in isolation. Neighbour contributions inside the
    core are absent, so the boundary shell is a lower bound and the result is not
    comparable to a whole-fit score. An internal marker from the parallel parent
    suppresses this whole block for disposable box outputs whose stamps would be
    scrubbed during merge.
    """
    from luxar.gsplats.planner.fit_planned_parallel import (
        _SKIP_CONTENT_BOX_STAMP_ENV,
    )

    if os.environ.get(_SKIP_CONTENT_BOX_STAMP_ENV) == "1":
        return

    from luxar.gsplats.fit_basis import fit_image_min, reference_on_fit_basis
    from luxar.gsplats.fitting.results import _occupied_fraction
    from luxar.gsplats.fitting.validation import _resolve_source_dtype
    from luxar.gsplats.gsplat_data import GSplatData, stamp_region_scoped_stats
    from luxar.gsplats.merged_quality import stamp_merged_quality

    z0, z1, y0, y1, x0, x1 = box.box
    shape = (z1 - z0, y1 - y0, x1 - x0)
    origin = np.asarray((z0, y0, x0), dtype=np.float32)
    if grid_scale is not None:
        origin = origin * np.asarray(grid_scale, dtype=np.float32)
    scored = GSplatData(
        centers=(result.centers - origin).astype(np.float32, copy=False),
        amplitudes=result.amplitudes,
        cholesky_factors=result.cholesky_factors,
        colors=result.colors,
        truncation_radius=result.truncation_radius,
    )
    core = np.asarray(volume[z0:z1, y0:y1, x0:x1], dtype=np.float32)
    stamp_merged_quality(
        scored,
        core,
        volume_shape=shape,
        grid_scale=grid_scale,
        device=device,
        verbose=verbose,
        image_min=fit_image_min(result.stats),
        stats=result.stats,
    )

    source_dtype = result.stats.get("source_dtype")
    _, source_itemsize = _resolve_source_dtype(core, source_dtype)
    intensity_range = float(result.stats.get("intensity_range", 1.0))
    normalized = reference_on_fit_basis(core, fit_image_min(result.stats))
    normalized = normalized / intensity_range
    stamp_region_scoped_stats(
        result.stats,
        source_shape=shape,
        fitted_shape=shape,
        n_splats=result.n_splats,
        occupancy=_occupied_fraction(normalized, int(np.prod(shape))),
        source_itemsize=source_itemsize,
    )


def _save_fit_result(
    result: Any,
    output: Path,
    *,
    compress: "Optional[Literal['zip', 'tar.gz']]" = None,
) -> None:
    """Save either a flat ``GSplatData`` leaf or a ``kind=partition`` tree node."""
    save_fit_output(result, output, compress=compress, verbose=False)


def _stamp_content_floor(
    result: Any,
    floor_level: "Optional[float]",
    floor_forward: "str | float",
) -> None:
    """Record the one level every content box subtracted (#1175).

    A content fit used to save NO record of the pedestal it removed: the merged
    result is built from fresh box nodes, and ``GSplatData.concatenate`` only
    carries a block its inputs AGREE on. ``floor_level`` is the one level
    `resolve_shared_floor` gave every box, so stamp it explicitly onto the flat
    leaf's ``stats``, or onto the root node's ``meta``, which
    ``write_gsplats_tree`` promotes into the store's ``pipeline/`` group.

    The exception is a NEGATIVE resolved level, which cannot be forwarded as a
    concrete ``--floor`` and is instead re-resolved per box (see
    :func:`resolve_shared_floor`). There is then no single level the artifact
    could honestly claim, so nothing is written.

    A level the BOXES already recorded wins over the planned one, and is left
    exactly as it stands. Since #1616 the boxes share one ``norm_range``, so their
    recorded bounds agree with each other instead of following each crop's own
    minimum. The applied level can still exceed the planned one when the shared
    low endpoint does, so the guard below refuses to overwrite a differing
    box-recorded ``floor`` — which would ship a store whose ``floor`` and
    ``image_min`` contradict each other and break the spec's
    ``image_min == floor`` invariant.
    """
    if isinstance(floor_forward, str) and floor_forward != "none":
        return
    from luxar.gsplats.gsplat_data import GSplatData

    target = result.stats if isinstance(result, GSplatData) else result.meta
    if "floor" in target and target["floor"] != floor_level:
        return
    target["floor"] = floor_level
    # Since #1616 the boxes share one norm_range, so their bounds agree with each
    # other. Still, drop any image_min that would contradict `floor` rather than
    # leave the invariant broken. Only meaningful when a floor WAS applied: with
    # none, `image_min` is just the normalization minimum and owes `floor` nothing.
    bound = target.get("image_min")
    if (
        floor_level is not None
        and isinstance(bound, (int, float))
        and not isinstance(bound, bool)
        and float(bound) != float(floor_level)
    ):
        target.pop("image_min", None)
        target.pop("intensity_range", None)


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
    norm_range: "Optional[tuple[float, float]]" = None,
    voxel_size: "Optional[tuple[float, ...]]" = None,
    physical_coordinates: bool = False,
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
    source_dtype: Optional[str] = None,
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
    from luxar.gsplats.planner import (
        CONTENT_CULL_RETENTION,
        FitPlan,
        fit_planned,
        plan_volume,
    )
    from luxar.gsplats.planner.fit_planned import (
        _ensure_planned_norm_range,
        _fit_one_box,
    )

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
                "norm_range": norm_range,
                "voxel_size": voxel_size,
                # `0.0` ("keep every splat") is not None, so it still wins here.
                "cull_retention": cull_retention,
            },
            # Content tiling's own baseline, layered just above the fitter's
            # harvested defaults: `--preset`, a `cull_retention:` in a `--config`
            # and `--cull-retention` all still win — only the fitter's 0.95 is
            # displaced. Shared with `fit_planned` so the CLI and the library
            # cannot disagree about what a preset-less content box is fitted at.
            command_defaults={"cull_retention": CONTENT_CULL_RETENTION},
        )
        _fill_source_dtype(fk, source_dtype)
        fk.pop("seeds", None)
        fk.pop("device", None)
        fk["verbose"] = False
        fk["_content_physical"] = physical_coordinates
        return fk

    # Validate the EFFECTIVE floor spec BEFORE the volume is read: a typo
    # (`--floor potato`), an out-of-range percentile (`--floor p150`) or a
    # negative `floor:` in a YAML --config must cost no read and surface as a
    # usage error, not a bare ValueError from inside the fit. (`--floor` itself
    # is not validated by Typer: it is a free-form string spec.)
    validate_floor_spec(_fit_kwargs().get("floor", "auto"))
    _validate_parallel_content_frame(
        physical_coordinates, jobs, plan_only=plan_only, plan_box=plan_box
    )

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
        _require_plan_volume_shape(vol, fitplan)
        fk = _fit_kwargs()
        fk["device"] = device
        # The parent resolved the level and forwarded it as a number — echoed
        # back untouched here (already guarded, so no re-read). A worker invoked
        # by hand (or from a manifest written before the level was resolved at
        # plan time) still carries a SPEC: resolve it against this whole
        # (t, c) volume, never the box crop, so the deterministic sampler makes
        # every box agree on one level anyway. EXCEPTION: a NEGATIVE resolved
        # level cannot be expressed as a concrete --floor, so resolve_shared_floor
        # hands the SPEC back and `_fit_one_box` re-resolves it against this box's
        # crop — per-box pedestals, for that one case only.
        _, fk["floor"] = resolve_shared_floor(
            vol, fk.get("floor", "auto"), guard_numeric=False, verbose=False
        )
        # The range has the same "resolve once against the whole (t, c) volume,
        # never the box crop" contract as the floor above; a hand-run worker (or
        # a manifest planned before the range existed) inherits none, so resolve
        # it here too rather than falling back to per-crop normalization.
        _ensure_planned_norm_range(vol, fk, False)
        cap = int(fitplan.density.get("saturation_cap", 0)) if fitplan.density else 0
        content_physical = bool(fk.pop("_content_physical", False))
        box_result = _fit_one_box(
            vol,
            fitplan.boxes[plan_box],
            int(fitplan.overlap),
            cap,
            content_physical=content_physical,
            **fk,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if box_result.n_splats == 0:
            Path(str(output) + ".empty").write_text("")  # writer rejects empty stores
        else:
            # Save the fitted dataset AS IS: rebuilding it from bare arrays
            # dropped the fit's truncation_radius and per-box stats (#1637),
            # and suppressing fitting info here discarded those stats on disk.
            from luxar.gsplats.tiling import resolve_grid_scale

            _stamp_content_box_output(
                box_result,
                vol,
                fitplan.boxes[plan_box],
                device,
                verbose=verbose,
                grid_scale=(
                    resolve_grid_scale(
                        vol.ndim,
                        voxel_size=fk.get("voxel_size"),
                        output_space=fk.get("output_space", "real"),
                    )
                    if content_physical
                    else None
                ),
            )
            box_result.save(output)
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
    loaded_fitplan = FitPlan.from_json(plan) if plan is not None else None
    if loaded_fitplan is not None:
        _require_plan_volume_shape(vol, loaded_fitplan)

    # ── background floor: ONE level for the whole volume ──
    # Boxes are core-kept and abutting, so a per-box estimate (what forwarding
    # the spec into `fit_gaussian_splats` produces — it resolves against the box
    # CROP) makes neighbouring boxes subtract different pedestals and normalize
    # by different ranges: visible brightness steps at box boundaries (#1174).
    # Resolved ONCE here instead, before anything is scanned or fitted, then
    # handed to the density scan, the sequential boxes, AND the -j box
    # subprocesses' argv. The spec comes out of the merged fit config, so a
    # `floor:` in --config/--preset is honoured rather than overridden.
    box_fit_kwargs = _fit_kwargs()
    floor_level, floor_forward = resolve_shared_floor(
        vol,
        box_fit_kwargs.get("floor", "auto"),
        guard_numeric=True,
        scope="every box",
        verbose=verbose,
    )
    # Normally a concrete number (or "none"), which is what fit_planned requires.
    # The one exception is a NEGATIVE resolved level, which cannot be expressed as
    # a concrete floor: the SPEC is forwarded and each box re-resolves it against
    # its own crop, so that case degenerates to the per-box pedestals #1174 is
    # about. Accepted rather than refused, because dark-frame-corrected data fit
    # fine before and refusing would make it unfittable.
    box_fit_kwargs["floor"] = floor_forward

    # One raw-input scale for every content box. A batch worker receives the
    # plan-time range through --norm-range; a direct content fit resolves it once
    # here against the whole selected volume. The floor remains a separate raw
    # zero point and _normalize_data combines the two without clipping the top.
    _ensure_planned_norm_range(vol, box_fit_kwargs, verbose)

    # ── obtain a plan: load --plan, or scan + plan ──
    created_plan = False
    # One unique per-invocation token shared by the internal plan JSON and
    # the parallel staging dir, so concurrent content fits to the SAME output
    # can't clobber each other's plan or in-progress boxes (issue #1040).
    token = _invocation_token()
    if loaded_fitplan is not None:
        fitplan = loaded_fitplan
        assert plan is not None
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
            t0 = time.perf_counter()
            # Scan the SAME floor-suppressed volume the boxes will fit: cal
            # records `density.feature_threshold` on the floor-subtracted volume,
            # so scanning the raw (pedestal-carrying) volume would count the whole
            # background as signal and flatten the content field. `floor_level`
            # is the very level every box subtracts, so plan and fits share one
            # basis by construction (it used to be re-resolved here, which also
            # forced a full-array np.percentile for `--floor pNN`).
            scan_vol = vol
            if floor_level is not None:
                scan_vol = np.clip(
                    np.asarray(vol, dtype=np.float32) - floor_level, 0.0, None
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
    from luxar.gsplats.fit_tiled_parallel import (
        report_auto_jobs,
        resolve_jobs,
    )
    from luxar.gsplats.planner.fit_planned_parallel import max_padded_box_voxels

    n_budgeted = sum(1 for b in fitplan.boxes if b.budget > 0)
    try:
        worker_limit = resolve_jobs(
            jobs,
            tile_voxels=max_padded_box_voxels(fitplan),
            num_tiles=max(1, n_budgeted),
            device=device,
        )
        n_jobs = worker_limit.count
    except ValueError:
        aprint(f"Error: --jobs must be an integer or 'auto', got '{jobs}'")
        raise typer.Exit(1)

    report_auto_jobs(jobs, worker_limit)
    if physical_coordinates and n_jobs > 1 and created_plan and not keep_boxes:
        Path(plan_json_path).unlink(missing_ok=True)
    _validate_parallel_content_frame(physical_coordinates, n_jobs)

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
            # VERBATIM, not `preset or "standard"`: the sequential path hands
            # `load_fit_config` the CLI value as-is, and defaulting to "standard"
            # here would layer that preset's n_iters (5000 vs 1000) and
            # cull_retention on every box — `-j N` fitting differently from
            # `-j 1` (#1637).
            preset=preset,
            # The run's fit configuration, so a box worker resolves the same fit
            # config as the sequential path. `truncate:` lives only in a YAML
            # --config (no preset sets it, and there is no --truncate flag), so
            # without this the boxes silently fit at the default (#1637).
            config=config,
            iters=iters,
            loss=loss,
            lr=lr,
            cull_retention=cull_retention,
            device=device,
            # The RESOLVED level, not the spec: each worker would otherwise
            # re-estimate on its own box crop (#1174).
            floor=floor_forward,
            norm_range=box_fit_kwargs.get("norm_range"),
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
                volume=vol,
                device=device,
                keep_boxes=keep_boxes,
                partition=partition,
                recipe=recipe,
                recipe_params=recipe_params,
                verbose=verbose,
            )
    else:
        fk = box_fit_kwargs  # already carries the one resolved floor level
        fk.pop("verbose", None)
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
                verbose=verbose,
                **fk,
            )

    _stamp_content_floor(result, floor_level, floor_forward)

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
        # --keep-tiles retains the internal plan too; its token-suffixed name
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
