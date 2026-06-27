"""``luxar gsplat slurm-fit`` — cluster-scale fitting via Slurm.

Owns the ``app_batch`` Typer sub-app and its commands; the aggregator
(``cli/gsplat_commands.py``) mounts it via ``add_typer``. Extracted from the
former monolithic ``gsplat_commands.py`` (package-refactor-plan P3/P4/P6).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import typer
from arbol import aprint, asection

if TYPE_CHECKING:
    import numpy as np

    from luxar.gsplats.lod.recipes import RecipeParams


app_batch = typer.Typer(
    help="Fit a whole nD dataset across its axes on a Slurm cluster "
    "(the cluster-scale sibling of `gsplat fit`)."
)


@app_batch.command("submit")
def batch_submit(
    input_path: Path = typer.Argument(..., exists=True, help="Input OME-Zarr dataset"),
    output_dir: Path = typer.Argument(..., help="Output directory for batch results"),
    # Tiling
    tile_size: Optional[int] = typer.Option(
        None,
        "--tile-size",
        help="Tile size in voxels (auto from GPU profile if omitted)",
    ),
    tile_overlap: int = typer.Option(32, "--overlap", help="Tile overlap in voxels"),
    tiling: str = typer.Option(
        "uniform",
        "--tiling",
        help="Spatial decomposition fanned across the Slurm array: 'uniform' (a "
        "regular tile grid, the default) or 'content' (a content-balanced box "
        "plan built once from a representative timepoint and REUSED for every "
        "(t,c) — the cluster sibling of `fit --tiling content`). Content mode "
        "needs a density (--cal or --k-star-ref/--n-features-ref) and no GPU "
        "profile.",
        rich_help_panel="Content-aware tiling",
    ),
    plan_timepoint: Optional[int] = typer.Option(
        None,
        "--plan-timepoint",
        help="[--tiling content] Representative timepoint to scan for the shared "
        "box plan (default: the first selected timepoint).",
        rich_help_panel="Content-aware tiling",
    ),
    cal: Optional[Path] = typer.Option(
        None,
        "--cal",
        help="[--tiling content] Calibration JSON (gsplat cal) supplying the "
        "splats-per-feature density.",
        rich_help_panel="Content-aware tiling",
    ),
    k_star_ref: Optional[int] = typer.Option(
        None,
        "--k-star-ref",
        help="[--tiling content] Reference K* (with --n-features-ref) if no --cal.",
        rich_help_panel="Content-aware tiling",
    ),
    n_features_ref: Optional[int] = typer.Option(
        None,
        "--n-features-ref",
        help="[--tiling content] Reference feature count (with --k-star-ref).",
        rich_help_panel="Content-aware tiling",
    ),
    saturation_exponent: float = typer.Option(
        0.44,
        "--saturation-exponent",
        help="[--tiling content] Sub-linear exponent alpha in K~features^alpha.",
        rich_help_panel="Content-aware tiling",
    ),
    saturation_cap: Optional[int] = typer.Option(
        None,
        "--saturation-cap",
        help="[--tiling content] Per-box budget cap.",
        rich_help_panel="Content-aware tiling",
    ),
    feature_threshold: Optional[float] = typer.Option(
        None,
        "--feature-threshold",
        help="[--tiling content] Absolute feature-detection level.",
        rich_help_panel="Content-aware tiling",
    ),
    feature_metric: Optional[str] = typer.Option(
        None,
        "--feature-metric",
        help="[--tiling content] Content metric: peaks | edges | intensity.",
        rich_help_panel="Content-aware tiling",
    ),
    cell: int = typer.Option(
        16,
        "--cell",
        help="[--tiling content] Coarse feature-grid cell size (voxels).",
        rich_help_panel="Content-aware tiling",
    ),
    target_features: Optional[int] = typer.Option(
        None,
        "--target-features",
        help="[--tiling content] Target features per box (BSP split threshold).",
        rich_help_panel="Content-aware tiling",
    ),
    min_leaf: int = typer.Option(
        256,
        "--min-leaf",
        help="[--tiling content] Minimum box edge length (voxels).",
        rich_help_panel="Content-aware tiling",
    ),
    max_leaf: int = typer.Option(
        512,
        "--max-leaf",
        help="[--tiling content] Maximum box edge length (voxels).",
        rich_help_panel="Content-aware tiling",
    ),
    # Fit params
    preset: str = typer.Option("standard", "--preset", help="Fitting preset"),
    config: Optional[Path] = typer.Option(None, "--config", help="YAML fit config"),
    seeds: Optional[str] = typer.Option(None, "--seeds", help="Seed count or ratio"),
    iters: Optional[int] = typer.Option(
        None, "--iters", "-n", help="Max optimization iterations (overrides preset)"
    ),
    # Progressive fitting
    batch_progressive: bool = typer.Option(
        False,
        "--progressive",
        help="Use progressive fitting per tile (multi-LOD). "
        "Combine with --parallel for better GPU utilization.",
    ),
    batch_splats_per_pass: Optional[int] = typer.Option(
        None, "--splats-per-pass", help="Max splats per progressive pass"
    ),
    batch_psnr_patience: Optional[float] = typer.Option(
        None, "--psnr-patience", help="PSNR patience for progressive fitting (dB)"
    ),
    batch_max_passes: Optional[int] = typer.Option(
        None, "--max-passes", help="Max progressive passes per tile"
    ),
    # Post-fit culling
    batch_cull_retention: Optional[float] = typer.Option(
        None,
        "--cull-retention",
        help="After fitting each tile, remove the weakest splats that "
        "collectively contribute less than (1 - value) of the total amplitude. "
        "Default 0.95 (discard bottom 5%%). Set to 0 to keep every splat.",
    ),
    # Denoising
    batch_denoise: bool = typer.Option(
        False,
        "--denoise",
        help="Denoise volumes before fitting (NLM). Auto-calibrates h per channel.",
    ),
    batch_denoise_h: Optional[float] = typer.Option(
        None, "--denoise-h", help="Manual NLM h (skip calibration)"
    ),
    batch_denoise_2d: bool = typer.Option(
        False, "--denoise-2d", help="Use 2D NLM (slice-by-slice) instead of 3D"
    ),
    batch_denoise_patch_size: int = typer.Option(
        3, "--denoise-patch-size", help="NLM patch size"
    ),
    batch_denoise_search_distance: int = typer.Option(
        5, "--denoise-search-distance", help="NLM search distance"
    ),
    batch_denoise_backend: str = typer.Option(
        "auto", "--denoise-backend", help="NLM backend"
    ),
    batch_calibration_samples: int = typer.Option(
        5, "--calibration-samples", help="Timepoints to sample for h calibration"
    ),
    batch_preprocess: Optional[bool] = typer.Option(
        None,
        "--preprocess/--no-preprocess",
        help="Write denoised volumes to zarr before fitting (default: off, denoise per-tile on-the-fly).",
    ),
    # Slurm params
    partition: Optional[str] = typer.Option(
        None, "--partition", "-p", help="Slurm partition"
    ),
    max_concurrent: Optional[int] = typer.Option(
        None,
        "--max-concurrent",
        help="Maximum simultaneous Slurm array tasks (limits cluster usage). "
        "Maps to --array=0-N%%MAX. No limit if omitted.",
    ),
    preemptible: bool = typer.Option(
        False,
        "--preemptible",
        help="Also submit tasks on a preemptible partition for extra throughput. "
        "Auto-detects the preemptible partition. Preempted tasks are automatically "
        "requeued. Uses atomic tile writes to handle interruptions safely.",
    ),
    preemptible_partition_opt: Optional[str] = typer.Option(
        None,
        "--preemptible-partition",
        help="Explicit preemptible partition name (skip auto-detection).",
    ),
    preemptible_concurrent: Optional[int] = typer.Option(
        None,
        "--preemptible-concurrent",
        help="Max concurrent tasks on preemptible partition. "
        "Defaults to same as --max-concurrent.",
    ),
    account: Optional[str] = typer.Option(None, "--account", "-A"),
    qos: Optional[str] = typer.Option(None, "--qos"),
    gpus: int = typer.Option(1, "--gpus", help="GPUs per task"),
    cpus: int = typer.Option(4, "--cpus", help="CPUs per task"),
    mem: int = typer.Option(32, "--mem", help="Memory per task (GB)"),
    time_limit: Optional[str] = typer.Option(
        None, "--time", help="Wall time per task override (HH:MM:SS)"
    ),
    gpu_name_opt: Optional[str] = typer.Option(
        None, "--gpu", help="GPU name from profile (auto-detect if omitted)"
    ),
    gpu_mem: Optional[int] = typer.Option(
        None, "--gpu-mem", help="Target GPU memory in GB (picks closest profile)"
    ),
    # Merge
    channel_colors: Optional[str] = typer.Option(
        None, "--channel-colors", help="Hex colors for per-channel merge"
    ),
    merge_recipe: Optional[str] = typer.Option(
        None,
        "--merge-recipe",
        help=(
            "Per-part LOD recipe applied to each spatial tile-part by the merge "
            "job: 'additive' (partitioned topology) or 'substitutive' (mosaic). "
            "Default: bare-leaf parts. The merge sbatch script invokes "
            "`slurm-fit merge --recipe <r>` with the knobs below."
        ),
    ),
    merge_n_lods: Optional[int] = typer.Option(
        None, "--merge-n-lods", help="Additive ladder depth for --merge-recipe."
    ),
    merge_compression_factor: Optional[int] = typer.Option(
        None, "--merge-compression-factor", help="Substitutive K for --merge-recipe."
    ),
    merge_levels: Optional[int] = typer.Option(
        None, "--merge-levels", help="Substitutive level count for --merge-recipe."
    ),
    merge_substitutive_method: Optional[str] = typer.Option(
        None,
        "--merge-substitutive-method",
        help="Substitutive coarsening method for --merge-recipe.",
    ),
    merge_coarsen_dims: Optional[str] = typer.Option(
        None,
        "--merge-coarsen-dims",
        help="Comma-separated center-column indices --merge-recipe substitutive "
        "may coarsen over (the rest stay hard barriers). Default: spatial dims "
        "only (the stacked-timepoint axis is a barrier).",
    ),
    # Dataset structure override
    axes: Optional[str] = typer.Option(
        None,
        "--axes",
        help=(
            "Comma-separated axis names overriding auto-detection, e.g. "
            "'time,camera,channel,z,y,x'. Recognised special names: "
            "time/t (timepoint), channel/c/ch/camera/cam (channel), "
            "z/y/x/depth/height/width (spatial)."
        ),
    ),
    timepoints_slice: Optional[str] = typer.Option(
        None,
        "--timepoints",
        help=(
            "Python-style slice to select timepoints, e.g. "
            "'0:10' (first 10), '::10' (every 10th), '100:200:5' (100-200 step 5). "
            "Default: all timepoints."
        ),
    ),
    channels_slice: Optional[str] = typer.Option(
        None,
        "--channels",
        help=(
            "Python-style slice to select channels, e.g. "
            "'0:2' (first 2 channels), '::2' (every other). "
            "Default: all channels."
        ),
    ),
    # Packing
    tasks_per_job: Optional[int] = typer.Option(
        None,
        "--tasks-per-job",
        help=(
            "Number of fitting tasks to run per Slurm job. "
            "Auto-calculated from GPU capacity when omitted. "
            "Packing multiple small volumes per GPU reduces scheduling overhead."
        ),
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel/--sequential",
        help=(
            "Run packed tasks concurrently (--parallel) or one by one "
            "(--sequential, default). Parallel mode launches multiple fit "
            "processes sharing the same GPU — higher throughput but uses "
            "more GPU memory."
        ),
    ),
    # Array selection
    array_key: Optional[str] = typer.Option(
        None,
        "--array-key",
        help=(
            "Key path to a specific array within the zarr store, e.g. "
            "'h2afva/fused'. Useful when a store contains multiple groups "
            "with different arrays. Auto-selects largest array if omitted."
        ),
    ),
    # Control
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show the plan without submitting (default: submit to Slurm).",
    ),
) -> None:
    """Plan and submit a cluster Gaussian splat fitting job over an nD dataset.

    Discovers T/C/spatial structure from OME-Zarr metadata, loads a GPU
    profile to auto-select tile size, and generates Slurm array + merge
    jobs.

    Submits to Slurm by default. Pass --dry-run to show the plan without
    submitting.

    A GPU profile from `luxar gsplat benchmark` is used to auto-select tile
    size; pass --tile-size to skip the profile requirement.

    Examples:
        luxar gsplat slurm-fit submit data.ome.zarr output/ -p gpu --dry-run

        luxar gsplat slurm-fit submit data.ome.zarr output/ --partition gpu

        luxar gsplat slurm-fit submit data.ome.zarr out/ -p gpu --tile-size 256 --preset hifi

        luxar gsplat slurm-fit submit keller.zarr.zip out/ -p gpu --tile-size 128 \\
            --axes time,camera,channel,z,y,x
    """
    if partition is None:
        aprint("Error: --partition is required")
        raise typer.Exit(1)

    # Normalize + validate --tiling up front (mirrors `gsplat fit`'s
    # _resolve_tiling) so a typo fails loudly instead of silently submitting a
    # large uniform array in the wrong mode.
    tiling = tiling.lower()
    if tiling not in ("uniform", "content"):
        aprint(f"Error: --tiling must be uniform|content, got {tiling!r}")
        raise typer.Exit(1)

    try:
        import datetime
        import subprocess

        from luxar.cli.gsplat_config import (
            PRESETS,
            decode_flat_channel_index,
            discover_ome_zarr_shape,
        )
        from luxar.gsplats.batch.env_capture import (
            capture_environment,
            generate_env_preamble,
            get_slurm_scheduler_info,
            is_slurm_mps_available,
        )
        from luxar.gsplats.batch.manifest import (
            BatchJob,
            BatchManifest,
            output_filename,
            save_manifest,
        )
        from luxar.gsplats.batch.slurm_gen import (
            generate_fit_sbatch,
            generate_merge_sbatch,
        )
        from luxar.gsplats.batch.time_estimate import (
            estimate_slurm_time_limit,
            estimate_tile_wall_seconds,
        )
        from luxar.gsplats.gpu_profile import (
            get_gpu_summary,
            get_gpu_throughput_table,
            load_profiles,
        )
        from luxar.gsplats.tiling import compute_tile_specs

        # 1. Load GPU profile (required only for auto tile-size)
        axes_list = [a.strip() for a in axes.split(",")] if axes else None
        summary = get_gpu_summary(
            gpu_name=gpu_name_opt,
            gpu_mem=float(gpu_mem) if gpu_mem else None,
        )
        if summary is None and tile_size is None and tiling != "content":
            aprint("Error: No GPU benchmark profile found.")
            aprint("")
            aprint("Option A — run the benchmark first (recommended):")
            aprint(
                "  luxar gsplat benchmark --slurm --partition "
                + (partition or "<partition>")
            )
            aprint("")
            aprint("Option B — skip the profile by providing a tile size explicitly:")
            aprint("  luxar gsplat slurm-fit submit ... --tile-size 128")
            raise typer.Exit(1)

        recs = (summary or {}).get("recommendations", {})
        peak = recs.get("peak_throughput_3d", {})

        # Resolve GPU name for display
        profiles = load_profiles()
        resolved_gpu = gpu_name_opt
        if summary is not None and resolved_gpu is None:
            for name, entry in profiles.get("gpus", {}).items():
                if entry.get("summary") == summary:
                    resolved_gpu = name
                    break
        if resolved_gpu is None and profiles.get("gpus"):
            resolved_gpu = next(iter(profiles["gpus"]))
        resolved_gpu = resolved_gpu or "unknown"

        # 2. Discover dataset shape
        # Helper: parse Python-style slice string "start:stop:step"
        def _parse_slice(s: str, max_val: int) -> list[int]:
            parts = s.split(":")
            if len(parts) == 1:
                # Single index
                return [int(parts[0])]
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1]) if len(parts) > 1 and parts[1] else max_val
            step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
            return list(range(start, stop, step))

        with asection("Discovering dataset shape"):
            ome_info = discover_ome_zarr_shape(
                input_path, axes_override=axes_list, array_key=array_key
            )
            n_t_full = ome_info.n_timepoints
            n_c_full = ome_info.n_channels
            spatial = ome_info.spatial_shape
            aprint(f"Axes: {ome_info.axes}")
            aprint(f"Shape: {ome_info.shape}")
            aprint(
                f"T={n_t_full}, C={n_c_full}, spatial={'x'.join(str(s) for s in spatial)}"
            )

            # Apply --timepoints / --channels slicing
            t_indices = (
                _parse_slice(timepoints_slice, n_t_full)
                if timepoints_slice
                else list(range(n_t_full))
            )
            c_indices = (
                _parse_slice(channels_slice, n_c_full)
                if channels_slice
                else list(range(n_c_full))
            )
            if not t_indices:
                raise ValueError("--timepoints selected no timepoints")
            if not c_indices:
                raise ValueError("--channels selected no channels")
            bad_t = [idx for idx in t_indices if idx < 0 or idx >= n_t_full]
            bad_c = [idx for idx in c_indices if idx < 0 or idx >= n_c_full]
            if bad_t:
                raise ValueError(
                    f"--timepoints selected out-of-range indices {bad_t}; valid range is 0..{n_t_full - 1}"
                )
            if bad_c:
                raise ValueError(
                    f"--channels selected out-of-range flat channel indices {bad_c}; valid range is 0..{n_c_full - 1}"
                )
            n_t = len(t_indices)
            n_c = len(c_indices)
            if timepoints_slice or channels_slice:
                aprint(f"Sliced: T={n_t} (of {n_t_full}), C={n_c} (of {n_c_full})")

        # 3. Decompose the spatial volume into the slots fanned across the array.
        import math

        mode = "content" if tiling == "content" else "uniform"
        content_plan = None  # the shared FitPlan in content mode
        plan_path_str: Optional[str] = None
        peak_shape = peak.get("shape", [])
        oom = (summary or {}).get("oom_boundaries", {}).get("3d", {})
        max_shape = oom.get("max_successful_shape", peak_shape)
        total_voxels = math.prod(spatial)

        if mode == "content":
            # Build ONE content-balanced box plan from a representative (t, c) and
            # reuse it for every (t, c) — each array task fits one box of this plan
            # (`fit --tiling content --plan plan.json --plan-box $K`). No GPU
            # profile needed; tile_size is irrelevant (the sbatch omits it).
            from luxar.cli.gsplat_config import load_volume
            from luxar.cli.gsplat_ops.planner import _resolve_density
            from luxar.gsplats.planner import plan_volume
            from luxar.gsplats.planner.fit_planned_parallel import (
                max_padded_box_voxels,
            )

            rep_t = plan_timepoint if plan_timepoint is not None else t_indices[0]
            rep_c = c_indices[0]
            with asection(f"Content plan (scan t={rep_t}, c={rep_c})"):
                rep_vol = load_volume(
                    input_path, channel=rep_c, timepoint=rep_t, array_key=array_key
                )
                density = _resolve_density(
                    cal,
                    k_star_ref,
                    n_features_ref,
                    saturation_exponent,
                    saturation_cap,
                    feature_metric,
                    feature_threshold,
                )
                content_plan = plan_volume(
                    rep_vol,
                    density,
                    feature_method=(feature_metric or density.feature_method),
                    cell=cell,
                    target_features=target_features,
                    min_leaf=min_leaf,
                    max_leaf=max_leaf,
                    overlap=tile_overlap,
                )
                # Drop budget<=0 boxes (matches the local `fit_planned` skip): the
                # array fits one box PER task, so a 0-budget box would just emit an
                # empty marker. Filtering keeps --plan-box K indexing the SAME plan
                # the workers read, with no empty array tasks.
                import dataclasses as _dc

                kept_boxes = [b for b in content_plan.boxes if b.budget > 0]
                if not kept_boxes:
                    aprint("Error: content plan has no boxes with budget > 0")
                    raise typer.Exit(1)
                content_plan = _dc.replace(content_plan, boxes=kept_boxes)
                output_dir.mkdir(parents=True, exist_ok=True)
                plan_path_obj = output_dir.resolve() / "plan.json"
                content_plan.to_json(plan_path_obj)
                plan_path_str = str(plan_path_obj)
                med, mx = content_plan.overlap_fraction()
                aprint(
                    f"Plan: {content_plan.n_boxes} boxes, total budget "
                    f"{content_plan.total_budget:,} splats, overlap median "
                    f"{med:.0%} / max {mx:.0%} → {plan_path_obj}"
                )
            n_tiles = content_plan.n_boxes
            needs_tiling = n_tiles > 1
            tile_size = 0  # sentinel; unused by the content sbatch
            tile_voxels = max_padded_box_voxels(content_plan)
            auto_tile = False  # no GPU-profile auto-sizing in content mode
        else:
            # Uniform: pick the largest tile that fits in GPU memory (auto from the
            # benchmark profile, or an explicit --tile-size).
            auto_tile = tile_size is None
            if auto_tile:
                assert summary is not None
                max_safe_voxels = math.prod(max_shape) if max_shape else 256**3
                if total_voxels <= max_safe_voxels:
                    # Whole volume fits — make the stride exceed every dim so
                    # compute_tile_specs yields exactly one tile.
                    tile_size = max(spatial) + tile_overlap
                else:
                    tile_edge = int(max_safe_voxels ** (1.0 / len(spatial)))
                    tile_size = min(tile_edge, max(spatial))
            assert tile_size is not None  # narrowed by branches above

            # compute_tile_specs is authoritative (overlap can add tiles even when
            # volume_shape == tile_size).
            specs = compute_tile_specs(spatial, tile_size, tile_overlap)
            n_tiles = len(specs)
            needs_tiling = n_tiles > 1
            if needs_tiling:
                tile_voxels = tile_size ** len(spatial)
            else:
                tile_voxels = total_voxels

        total_tasks = n_t * n_c * n_tiles

        throughput_table = get_gpu_throughput_table(gpu_name=resolved_gpu)

        preset_config = PRESETS.get(preset, PRESETS["standard"])
        n_iters = iters if iters is not None else preset_config.get("n_iters", 3000)

        if throughput_table:
            est_seconds = estimate_tile_wall_seconds(
                tile_voxels, n_iters, throughput_table
            )
        else:
            est_seconds = 600.0

        # 5b. Compute tasks-per-job packing
        #
        # When each volume is small relative to GPU capacity, we pack
        # multiple fitting tasks sequentially into one Slurm job to
        # reduce scheduling overhead (fewer array elements to launch).
        # 5b-i. Query scheduler for smart packing defaults
        sched_info = get_slurm_scheduler_info()
        uses_backfill = sched_info["uses_backfill"]
        no_job_limit = sched_info["max_jobs_per_user"] is None

        if tasks_per_job is None:
            max_safe = math.prod(max_shape) if max_shape else tile_voxels

            if parallel:
                # Each concurrent fit holds the volume tensor + model params
                # + optimizer state.  ~2× the raw volume is a safe estimate.
                packing = max(1, int(max_safe / max(tile_voxels * 2, 1)))
            else:
                packing = max(1, int(max_safe / max(tile_voxels, 1)))

            # On backfill clusters with no job limit, prefer shorter jobs
            # (more jobs = more backfill opportunities = faster throughput).
            # Cap packing lower so individual jobs stay short.
            if uses_backfill and no_job_limit:
                if parallel:
                    # Parallel: already short, keep the memory-based packing
                    packing = min(packing, 4)
                else:
                    # Sequential: each extra task adds wall-time.
                    # Keep jobs under ~5 min for best backfill scheduling.
                    if est_seconds > 0:
                        max_tasks_for_5min = max(1, int(300 / est_seconds))
                        packing = min(packing, max_tasks_for_5min)
                    packing = min(packing, 3)
            else:
                packing = min(packing, 10)

            tasks_per_job = packing
        tasks_per_job = max(1, tasks_per_job)

        n_slurm_jobs = math.ceil(total_tasks / tasks_per_job)
        if parallel:
            # Parallel: all tasks run at once, so wall time ≈ 1 task
            est_seconds_per_job = est_seconds * 1.2  # 20% overhead for contention
        else:
            est_seconds_per_job = est_seconds * tasks_per_job
        slurm_time = time_limit or estimate_slurm_time_limit(est_seconds_per_job)
        total_gpu_hours = est_seconds * total_tasks / 3600.0

        # 6. Build manifest
        fit_args = {}
        if seeds:
            fit_args["seeds"] = seeds
        if iters is not None:
            fit_args["iters"] = str(iters)
        if config:
            fit_args["config"] = str(config)
        if batch_progressive:
            fit_args["progressive"] = ""  # boolean flag, no value
        if batch_splats_per_pass is not None:
            fit_args["splats-per-pass"] = str(batch_splats_per_pass)
        if batch_psnr_patience is not None:
            fit_args["psnr-patience"] = str(batch_psnr_patience)
        if batch_max_passes is not None:
            fit_args["max-passes"] = str(batch_max_passes)
        if batch_cull_retention is not None:
            fit_args["cull-retention"] = str(batch_cull_retention)

        # Denoise mode detection
        denoise_mode = None
        denoised_zarr_path = None
        if batch_denoise:
            if batch_preprocess is True:
                denoise_mode = "preprocess"
            elif batch_preprocess is False:
                denoise_mode = "on-the-fly"
            else:
                # Default: on-the-fly (denoise per-tile inside each fit task).
                # Use --preprocess to write denoised zarr separately.
                denoise_mode = "on-the-fly"
            aprint(f"Denoise mode: {denoise_mode}")

            if denoise_mode == "preprocess":
                denoised_zarr_path = str(output_dir.resolve() / "denoised.zarr")

            # For on-the-fly mode, pass denoise flags to fit tasks
            if denoise_mode == "on-the-fly":
                fit_args["denoise"] = ""
                if batch_denoise_2d:
                    fit_args["denoise-2d"] = ""
                if batch_denoise_patch_size != 3:
                    fit_args["denoise-patch-size"] = str(batch_denoise_patch_size)
                if batch_denoise_search_distance != 5:
                    fit_args["denoise-search-distance"] = str(
                        batch_denoise_search_distance
                    )
                if batch_denoise_backend != "auto":
                    fit_args["denoise-backend"] = batch_denoise_backend
                # Note: --denoise-h is passed at runtime from h_values JSON

        colors_list = None
        if channel_colors:
            colors_list = [c.strip() for c in channel_colors.split(",")]

        # Per-part LOD recipe for the merge job (stored in the manifest; the merge
        # sbatch script turns it into `slurm-fit merge --recipe ...`).
        merge_recipe_args: dict = {}
        if merge_recipe is not None:
            from luxar.gsplats.lod.recipes import PER_PART_RECIPES

            if merge_recipe not in PER_PART_RECIPES:
                raise typer.BadParameter(
                    f"--merge-recipe {merge_recipe!r} is not supported; choose from "
                    f"{', '.join(sorted(PER_PART_RECIPES))} (the composed recipes "
                    "re-partition their input, but each tile is already one part)."
                )
            if merge_n_lods is not None:
                merge_recipe_args["n-lods"] = str(merge_n_lods)
            if merge_compression_factor is not None:
                merge_recipe_args["compression-factor"] = str(merge_compression_factor)
            if merge_levels is not None:
                merge_recipe_args["levels"] = str(merge_levels)
            if merge_substitutive_method is not None:
                # Validate now (fail-fast) so a bad method is caught before the
                # Slurm fit array runs, not hours later in the merge job.
                from luxar.cli.lod import _VALID_SUBSTITUTIVE_METHODS

                sm_norm = merge_substitutive_method.strip().replace("-", "_")
                if sm_norm not in _VALID_SUBSTITUTIVE_METHODS:
                    raise typer.BadParameter(
                        f"--merge-substitutive-method must be one of "
                        f"{list(_VALID_SUBSTITUTIVE_METHODS)}; "
                        f"got {merge_substitutive_method!r}"
                    )
                merge_recipe_args["substitutive-method"] = sm_norm
            if merge_coarsen_dims is not None:
                merge_recipe_args["coarsen-dims"] = merge_coarsen_dims

        # Preemptible partition detection
        preempt_partition: Optional[str] = None
        if preemptible:
            if preemptible_partition_opt:
                preempt_partition = preemptible_partition_opt
            else:
                from luxar.gsplats.batch.env_capture import (
                    detect_preemptible_gpu_partition,
                )

                preempt_partition = detect_preemptible_gpu_partition()

            if preempt_partition is None:
                aprint(
                    "No preemptible GPU partition found on this cluster.\n"
                    "  Checked all partitions for: preemptible naming + GPU resources.\n"
                    "  Use --preemptible-partition to specify one explicitly.\n"
                    "  Continuing with guaranteed partition only."
                )
            else:
                from luxar.gsplats.batch.env_capture import validate_partition_access

                if not validate_partition_access(preempt_partition):
                    aprint(
                        f"Cannot submit to preemptible partition '{preempt_partition}'.\n"
                        f"  Your account may not have access.\n"
                        f"  To check: sacctmgr show assoc user=$USER partition={preempt_partition}\n"
                        "  Continuing with guaranteed partition only."
                    )
                    preempt_partition = None
                else:
                    aprint(f"Preemptible partition: {preempt_partition}")

        manifest = BatchManifest(
            version=1,
            created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            input_path=str(input_path.resolve()),
            output_dir=str(output_dir.resolve()),
            array_key=array_key,
            n_timepoints=n_t,
            n_channels=n_c,
            channel_axes=ome_info.channel_axes,
            channel_shape=ome_info.channel_shape,
            spatial_shape=spatial,
            mode=mode,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            n_tiles=n_tiles,
            plan_path=plan_path_str,
            total_tasks=total_tasks,
            preset=preset,
            fit_args=fit_args,
            gpu_name=resolved_gpu,
            estimated_seconds_per_task=est_seconds,
            slurm_time_limit=slurm_time,
            slurm_partition=partition,
            slurm_account=account,
            slurm_qos=qos,
            slurm_gpus=gpus,
            slurm_cpus=cpus,
            slurm_mem_gb=mem,
            tasks_per_job=tasks_per_job,
            parallel_tasks_per_job=parallel,
            max_concurrent=max_concurrent,
            preemptible=preempt_partition is not None,
            preemptible_partition=preempt_partition,
            preemptible_max_concurrent=(
                (preemptible_concurrent or max_concurrent)
                if preempt_partition
                else None
            ),
            timepoint_indices=t_indices if timepoints_slice else None,
            channel_indices=c_indices if channels_slice else None,
            channel_colors=colors_list,
            merge_recipe=merge_recipe,
            merge_recipe_args=merge_recipe_args,
            denoise=batch_denoise,
            denoise_2d=batch_denoise_2d,
            denoise_h=batch_denoise_h,
            denoise_patch_size=batch_denoise_patch_size,
            denoise_search_distance=batch_denoise_search_distance,
            denoise_backend=batch_denoise_backend,
            denoise_mode=denoise_mode,
            denoised_zarr_path=denoised_zarr_path,
            calibration_samples=batch_calibration_samples,
        )

        # Build job list. Store real dataset indices in filenames so status,
        # merge, and generated Slurm scripts agree when --timepoints/--channels
        # select non-contiguous values.
        jobs = []
        t_width_base = max(t_indices) + 1
        c_width_base = max(c_indices) + 1
        for task_id in range(total_tasks):
            t_seq = task_id // (n_c * n_tiles)
            r = task_id % (n_c * n_tiles)
            c_seq = r // n_tiles
            k = r % n_tiles
            t_real = t_indices[t_seq]
            c_real = c_indices[c_seq]
            jobs.append(
                BatchJob(
                    task_id=task_id,
                    timepoint=t_real,
                    channel=c_real,
                    tile_index=k,
                    output_filename=output_filename(
                        t_real,
                        c_real,
                        k,
                        t_width_base,
                        c_width_base,
                        n_tiles,
                        label="box" if mode == "content" else "tile",
                    ),
                    estimated_wall_seconds=est_seconds,
                    channel_coords=decode_flat_channel_index(
                        c_real, ome_info.channel_shape
                    ),
                )
            )
        manifest.jobs = jobs

        # 7. Capture environment + generate scripts
        env = capture_environment()
        preamble = generate_env_preamble(env)
        fit_script = generate_fit_sbatch(manifest, preamble)
        merge_script = generate_merge_sbatch(manifest, preamble)

        # Generate preemptible fit script if enabled
        preempt_fit_script = None
        if preempt_partition:
            preempt_fit_script = generate_fit_sbatch(
                manifest,
                preamble,
                partition_override=preempt_partition,
                max_concurrent_override=manifest.preemptible_max_concurrent,
                requeue=True,
                job_name="luxar-fit-preempt",
            )

        # Generate denoise scripts if needed
        calibrate_script = None
        denoise_script = None
        if batch_denoise:
            from luxar.gsplats.batch.slurm_gen import (
                generate_calibrate_sbatch,
                generate_denoise_sbatch,
            )

            if batch_denoise_h is None:
                calibrate_script = generate_calibrate_sbatch(manifest, preamble)
            if denoise_mode == "preprocess":
                denoise_script = generate_denoise_sbatch(manifest, preamble)

        # 8. Print plan (always)
        spatial_str = "x".join(str(s) for s in spatial)
        peak_gvs = peak.get("gvoxel_per_s", "?")
        peak_shape_str = "x".join(str(s) for s in peak.get("shape", []))

        aprint("")
        aprint("=" * 60)
        aprint("BATCH PLAN")
        aprint("=" * 60)
        aprint(f"  Input: {input_path.name} (T={n_t}, C={n_c}, spatial={spatial_str})")
        aprint(f"  GPU: {resolved_gpu} (peak: {peak_gvs} GV/s at {peak_shape_str})")
        if mode == "content":
            aprint(
                f"  Decomposition: content plan, {n_tiles} boxes/volume "
                f"(overlap={tile_overlap}); shared across all (t,c)"
            )
        elif needs_tiling:
            aprint(
                f"  Tile: {tile_size}^{len(spatial)}"
                f" ({'auto' if auto_tile else 'manual'})"
                f", overlap={tile_overlap}, {n_tiles} tiles/volume"
            )
        else:
            aprint("  Tile: not needed (volume fits in GPU memory)")
        slot = "boxes" if mode == "content" else "tiles"
        aprint(
            f"  Jobs: {n_t} x {n_c} x {n_tiles} = {total_tasks} fitting tasks "
            f"({slot})"
        )
        if tasks_per_job > 1:
            mode = "parallel" if parallel else "sequential"
            mps_note = ""
            if parallel:
                if is_slurm_mps_available():
                    mps_note = " [MPS available]"
                else:
                    mps_note = " [bash background processes]"
            aprint(
                f"  Packing: {tasks_per_job} tasks/job ({mode}) → {n_slurm_jobs} Slurm jobs{mps_note}"
            )
        else:
            aprint(f"  Slurm array: {total_tasks} jobs (1 task each)")
        if uses_backfill:
            sched_note = "backfill scheduler — short jobs get scheduled fastest"
            if no_job_limit:
                sched_note += ", no job count limit"
            aprint(f"  Scheduler: {sched_note}")
        aprint(
            f"  Est. time/task: ~{est_seconds / 60:.0f} min"
            f" (preset: {preset}, {n_iters} iters)"
        )
        if tasks_per_job > 1:
            if parallel:
                aprint(
                    f"  Est. time/job: ~{est_seconds_per_job / 60:.0f} min ({tasks_per_job} tasks in parallel)"
                )
            else:
                aprint(
                    f"  Est. time/job: ~{est_seconds_per_job / 60:.0f} min ({tasks_per_job} tasks × {est_seconds / 60:.0f} min)"
                )
        aprint(f"  Est. total GPU-hours: {total_gpu_hours:.0f} h")
        aprint(f"  Slurm --time: {slurm_time}")
        aprint(f"  Partition: {partition}, GPUs: {gpus}, CPUs: {cpus}, Mem: {mem}G")
        aprint(f"  Output: {output_dir}")
        aprint("")

        if dry_run:
            aprint("Dry run -- omit --dry-run to actually submit.")
            raise typer.Exit(0)

        # 9. Submit
        out = output_dir.resolve()
        (out / "tiles").mkdir(parents=True, exist_ok=True)
        (out / "merged").mkdir(parents=True, exist_ok=True)
        (out / "logs").mkdir(parents=True, exist_ok=True)

        fit_path = out / "fit_array.sbatch"
        merge_path = out / "merge.sbatch"
        env_path = out / "env_snapshot.sh"

        fit_path.write_text(fit_script)
        merge_path.write_text(merge_script)
        env_path.write_text(preamble)
        if calibrate_script:
            (out / "calibrate.sbatch").write_text(calibrate_script)
        if denoise_script:
            (out / "denoise_array.sbatch").write_text(denoise_script)
        if preempt_fit_script:
            (out / "fit_array_preempt.sbatch").write_text(preempt_fit_script)
        save_manifest(manifest, out)

        def _parse_job_id(stdout: str) -> Optional[int]:
            for word in stdout.strip().split():
                if word.isdigit():
                    return int(word)
            return None

        # Submit calibration job (if needed)
        calibrate_job_id = None
        if calibrate_script:
            aprint("Submitting calibration job...")
            result = subprocess.run(
                ["sbatch", str(out / "calibrate.sbatch")],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                aprint(f"Error submitting calibration job: {result.stderr}")
                raise typer.Exit(1)
            calibrate_job_id = _parse_job_id(result.stdout)
            manifest.calibrate_job_id = calibrate_job_id
            aprint(f"  Calibration job: {calibrate_job_id}")

        # Submit denoise preprocessing array (if preprocess mode)
        denoise_job_id = None
        if denoise_script:
            aprint("Submitting denoise preprocessing array...")
            dep_cmd = ["sbatch"]
            if calibrate_job_id:
                dep_cmd.append(f"--dependency=afterok:{calibrate_job_id}")
            dep_cmd.append(str(out / "denoise_array.sbatch"))
            result = subprocess.run(dep_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                aprint(f"Error submitting denoise job: {result.stderr}")
                raise typer.Exit(1)
            denoise_job_id = _parse_job_id(result.stdout)
            manifest.denoise_job_id = denoise_job_id
            denoise_n_t = (
                len(manifest.timepoint_indices)
                if manifest.timepoint_indices
                else manifest.n_timepoints
            )
            denoise_n_c = (
                len(manifest.channel_indices)
                if manifest.channel_indices
                else manifest.n_channels
            )
            denoise_total = denoise_n_t * denoise_n_c
            aprint(f"  Denoise array job: {denoise_job_id} ({denoise_total} tasks)")

        # Submit fitting array (depends on denoise or calibrate)
        fit_dep_id = denoise_job_id or calibrate_job_id
        aprint("Submitting fitting array job...")
        fit_cmd = ["sbatch"]
        if fit_dep_id:
            fit_cmd.append(f"--dependency=afterok:{fit_dep_id}")
        fit_cmd.append(str(fit_path))
        result = subprocess.run(fit_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            aprint(f"Error submitting fit job: {result.stderr}")
            raise typer.Exit(1)

        fit_job_id = _parse_job_id(result.stdout)
        aprint(f"  Fitting array job: {fit_job_id} ({total_tasks} tasks)")

        # Submit preemptible fit array (if enabled)
        preemptible_job_id = None
        if preempt_fit_script:
            aprint("Submitting preemptible fitting array...")
            preempt_cmd = ["sbatch"]
            if fit_dep_id:
                preempt_cmd.append(f"--dependency=afterok:{fit_dep_id}")
            preempt_cmd.append(str(out / "fit_array_preempt.sbatch"))
            result = subprocess.run(preempt_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                preemptible_job_id = _parse_job_id(result.stdout)
                manifest.preemptible_job_id = preemptible_job_id
                aprint(
                    f"  Preemptible array job: {preemptible_job_id} "
                    f"({total_tasks} tasks on {preempt_partition}, requeue)"
                )
            else:
                aprint(
                    f"  Warning: preemptible submission failed: {result.stderr}\n"
                    "  Continuing with guaranteed partition only."
                )

        # Merge depends on ALL fit arrays
        merge_deps = [jid for jid in [fit_job_id, preemptible_job_id] if jid]
        merge_cmd = ["sbatch"]
        if merge_deps:
            dep_str = ":".join(str(jid) for jid in merge_deps)
            merge_cmd.append(f"--dependency=afterok:{dep_str}")
        merge_cmd.append(str(merge_path))

        result = subprocess.run(merge_cmd, capture_output=True, text=True)
        merge_job_id = None
        if result.returncode == 0:
            merge_job_id = _parse_job_id(result.stdout)
            dep_info = " + ".join(str(j) for j in merge_deps)
            aprint(f"  Merge job: {merge_job_id} (depends on {dep_info})")
        else:
            aprint(f"  Warning: merge job submission failed: {result.stderr}")

        manifest.array_job_id = fit_job_id
        manifest.merge_job_id = merge_job_id
        save_manifest(manifest, out)

        aprint(f"\nManifest: {out / 'manifest.json'}")
        aprint(f"Check status: luxar gsplat slurm-fit status {out}")

    except typer.Exit:
        raise
    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app_batch.command("status")
def batch_status_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Check status of a batch fitting job.

    Reads the manifest, checks for output files, and queries sacct/squeue
    for job states.

    Examples:
        luxar gsplat slurm-fit status output_dir/
    """
    try:
        from luxar.gsplats.batch.manifest import load_manifest
        from luxar.gsplats.batch.status import (
            check_batch_status,
            format_status_report,
        )

        manifest = load_manifest(output_dir)
        status = check_batch_status(output_dir)
        aprint(format_status_report(status, manifest, verbose=verbose))

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1)


@app_batch.command("validate")
def batch_validate_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    fix: bool = typer.Option(
        False, "--fix", help="Delete corrupt/incomplete tiles so they get re-fitted"
    ),
) -> None:
    """Validate integrity of all tiles in a batch output directory.

    Checks each tile for completeness (metadata, arrays, shapes).
    Reports OK, MISSING, CORRUPT, and STALE_TMP counts.

    Use --fix to delete corrupt tiles and leftover .tmp directories,
    so they get re-fitted on the next submit.

    Examples:
        luxar gsplat slurm-fit validate output_dir/

        luxar gsplat slurm-fit validate output_dir/ --fix
    """
    try:
        from luxar.gsplats.batch.manifest import load_manifest

        manifest = load_manifest(output_dir)
        tiles_dir = output_dir / "tiles"

        if not tiles_dir.exists():
            aprint("No tiles directory found.")
            raise typer.Exit(1)

        # Build expected tile list from manifest
        expected_tiles = [job.output_filename for job in manifest.jobs]
        aprint(f"Checking {len(expected_tiles)} expected tiles...")

        ok = 0
        missing = 0
        empty = 0
        corrupt = 0
        unmigrated = 0
        stale_tmp = 0
        corrupt_reasons: list[str] = []
        unmigrated_reasons: list[str] = []

        for tile_name in expected_tiles:
            tile_path = tiles_dir / tile_name
            tmp_path = tiles_dir / f"{tile_name}.tmp"

            # Check for stale .tmp
            if tmp_path.is_dir():
                stale_tmp += 1
                if fix:
                    shutil.rmtree(tmp_path)
                    aprint(f"  Deleted: {tile_name}.tmp")

            if not tile_path.is_dir():
                # A `<tile>.empty` marker = the task ran and legitimately produced
                # 0 splats (content boxes / sparse tiles); that is NOT missing.
                if (tiles_dir / f"{tile_name}.empty").exists():
                    empty += 1
                else:
                    missing += 1
                continue

            # Validate tile integrity
            reason = _validate_tile(tile_path)
            if reason == "ok":
                ok += 1
            elif reason.startswith("unsupported_format_version"):
                # Recoverable, NOT corrupt: an unmigrated legacy tile. Never
                # delete it under --fix — it converts via `gsplat migrate-format`.
                unmigrated += 1
                unmigrated_reasons.append(f"  {tile_name}: {reason}")
            else:
                corrupt += 1
                corrupt_reasons.append(f"  {tile_name}: {reason}")
                if fix:
                    shutil.rmtree(tile_path)
                    aprint(f"  Deleted corrupt: {tile_name} ({reason})")

        # Summary
        aprint("")
        aprint(f"  OK:         {ok}")
        aprint(f"  EMPTY:      {empty}")
        aprint(f"  MISSING:    {missing}")
        aprint(f"  CORRUPT:    {corrupt}")
        aprint(f"  UNMIGRATED: {unmigrated}")
        aprint(f"  STALE_TMP:  {stale_tmp}")

        if corrupt_reasons and not fix:
            aprint("")
            aprint("Corrupt tiles:")
            for r in corrupt_reasons:
                aprint(r)
            aprint("")
            aprint("Run with --fix to delete corrupt tiles.")

        if unmigrated_reasons:
            aprint("")
            aprint("Unmigrated (legacy-format) tiles — NOT deleted:")
            for r in unmigrated_reasons:
                aprint(r)
            aprint("")
            aprint("Convert each with `luxar gsplat migrate-format <tile> <out>`.")

        if fix and (corrupt > 0 or stale_tmp > 0):
            aprint(f"\nFixed: deleted {corrupt} corrupt + {stale_tmp} stale .tmp")
            aprint("Resubmit to re-fit deleted tiles.")

    except typer.Exit:
        raise
    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1) from e


def _validate_leaf_arrays(node_dir: Path, label: str) -> str:
    """Check a v3.0 gsplats leaf's required array sub-dirs (no decode)."""
    for arr_name in ("centers", "amplitudes", "cholesky_factors"):
        arr_dir = node_dir / arr_name
        if not arr_dir.is_dir():
            return f"missing_{arr_name}@{label}"
        if not (arr_dir / ".zarray").exists():
            return f"no_zarray_{arr_name}@{label}"
    return "ok"


def _validate_node_dir(node_dir: Path, label: str) -> str:
    """Structurally validate a v3.0 node subtree on disk (no array decode)."""
    import json

    zattrs_path = node_dir / ".zattrs"
    if not zattrs_path.exists():
        # Every node (root, child_<i>, part_<i>) must carry its .zattrs; a
        # metadata-stripped node is corrupt, not a bare single-set leaf.
        return f"no_zattrs@{label}"
    try:
        attrs = json.loads(zattrs_path.read_text())
    except (json.JSONDecodeError, OSError):
        return f"corrupt_zattrs@{label}"

    kind = attrs.get("kind")
    if kind in ("lod", "partition"):
        prefix = "child_" if kind == "lod" else "part_"
        children = sorted(
            d for d in node_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)
        )
        if not children:
            return f"{kind}_no_children@{label}"
        for child in children:
            reason = _validate_node_dir(child, f"{label}/{child.name}")
            if reason != "ok":
                return reason
        return "ok"

    # Leaf: a single splat set, or an additive ladder (additive_<i>/ subgroups).
    n_additive = int(attrs.get("n_additive_sublods", 1))
    if n_additive > 1:
        for i in range(n_additive):
            reason = _validate_leaf_arrays(
                node_dir / f"additive_{i}", f"{label}/additive_{i}"
            )
            if reason != "ok":
                return reason
        return "ok"
    return _validate_leaf_arrays(node_dir, label)


def _validate_tile(tile_path: Path) -> str:
    """Validate a single v3.0 tile's integrity. Returns 'ok' or a reason string.

    Walks the node-tree structure (leaf / kind=lod / kind=partition) checking for
    the consolidated metadata, the format header, and the presence of every
    required array — without decoding any data. A non-v3.0 tile is reported (so
    ``slurm-fit validate --fix`` never silently deletes an unmigrated tile).
    """
    import json

    # .zmetadata is written last by consolidate_metadata — best completeness signal.
    if not (tile_path / ".zmetadata").exists():
        return "no_zmetadata (save incomplete)"

    zattrs_path = tile_path / ".zattrs"
    if not zattrs_path.exists():
        return "no_zattrs"
    try:
        attrs = json.loads(zattrs_path.read_text())
    except (json.JSONDecodeError, OSError):
        return "corrupt_zattrs"

    if attrs.get("format_type") != "gsplats_zarr":
        return f"bad_format_type: {attrs.get('format_type')}"

    version = attrs.get("format_version")
    if version != "3.0":
        # Not corrupt — just unmigrated. Surface it instead of classifying it as
        # corrupt (which would let --fix delete a recoverable tile).
        return f"unsupported_format_version: {version} (run gsplat migrate-format)"

    return _validate_node_dir(tile_path, ".")


@app_batch.command("cancel")
def batch_cancel_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
) -> None:
    """Cancel all Slurm jobs for a batch fitting run.

    Reads the manifest to find job IDs (calibrate, denoise, fit array,
    merge) and cancels them via scancel.

    Examples:
        luxar gsplat slurm-fit cancel output_dir/
    """
    import subprocess

    try:
        from luxar.gsplats.batch.manifest import load_manifest

        manifest = load_manifest(output_dir)

        job_ids = []
        for attr in (
            "calibrate_job_id",
            "denoise_job_id",
            "array_job_id",
            "merge_job_id",
        ):
            jid = getattr(manifest, attr, None)
            if jid is not None:
                job_ids.append(str(jid))

        if not job_ids:
            aprint("No job IDs found in manifest — nothing to cancel.")
            raise typer.Exit(0)

        aprint(f"Cancelling {len(job_ids)} job(s): {', '.join(job_ids)}")
        result = subprocess.run(["scancel"] + job_ids, capture_output=True, text=True)
        if result.returncode == 0:
            aprint("All jobs cancelled.")
        else:
            # scancel may warn about already-completed jobs — that's fine
            aprint(f"scancel output: {result.stderr.strip()}")
            aprint("Cancel command sent (some jobs may have already completed).")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1)


def _build_merge_recipe_params(
    stored: dict,
    *,
    n_lods: Optional[int],
    compression_factor: Optional[int],
    levels: Optional[int],
    substitutive_method: Optional[str],
    coarsen_dims: Optional[str],
) -> "RecipeParams":
    """Build a ``RecipeParams`` for the per-part merge recipe.

    Each knob is resolved CLI-first, then the value recorded at plan time
    (``manifest.merge_recipe_args``, string-valued), then the ``RecipeParams``
    default. ``coarsen_dims`` is parsed from a comma string to a tuple of ints;
    leaving it unset lets the merge default it per part (spatial dims only).
    """
    from luxar.cli.lod import _VALID_SUBSTITUTIVE_METHODS
    from luxar.gsplats.lod.recipes import RecipeParams

    def _resolve(key: str, cli: Any, cast: Callable[[Any], Any]) -> Any:
        if cli is not None:
            return cli
        raw = stored.get(key)
        return cast(raw) if raw is not None else None

    def _parse_dims(raw: Any) -> tuple:
        try:
            return tuple(int(x) for x in str(raw).split(",") if x.strip() != "")
        except ValueError as e:
            raise typer.BadParameter(
                f"--coarsen-dims must be comma-separated integers; got {raw!r}"
            ) from e

    overrides: dict = {}
    nl = _resolve("n-lods", n_lods, int)
    if nl is not None:
        overrides["n_lods"] = nl
    cf = _resolve("compression-factor", compression_factor, int)
    if cf is not None:
        overrides["compression_factor"] = cf
    lv = _resolve("levels", levels, int)
    if lv is not None:
        overrides["levels"] = lv
    sm = _resolve("substitutive-method", substitutive_method, str)
    if sm is not None:
        # Normalise hyphens to underscores so the documented CLI spelling
        # (`kmeans-lloyd`) maps to the canonical method name (`kmeans_lloyd`),
        # then validate up front — both halves of the `gsplat lod` contract
        # (cli/lod.py:381-386). Validating here means a bad method fails cleanly
        # BEFORE the streaming writer overwrites final.gsplats.zarr, rather than
        # raising deep in the merge and leaving a stub a non-`--force` re-run skips.
        sm_norm = sm.strip().replace("-", "_")
        if sm_norm not in _VALID_SUBSTITUTIVE_METHODS:
            raise typer.BadParameter(
                f"--substitutive-method must be one of "
                f"{list(_VALID_SUBSTITUTIVE_METHODS)}; got {sm!r}"
            )
        overrides["substitutive_method"] = sm_norm
    cd = coarsen_dims if coarsen_dims is not None else stored.get("coarsen-dims")
    if cd is not None:
        overrides["coarsen_dims"] = _parse_dims(cd)

    return RecipeParams(**overrides)


@app_batch.command("merge")
def batch_merge_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    channel_colors: Optional[str] = typer.Option(
        None, "--channel-colors", help="Hex colors for channel merge"
    ),
    force: bool = typer.Option(False, "--force", help="Re-merge even if outputs exist"),
    flat: bool = typer.Option(
        False,
        "--flat",
        help=(
            "Concatenate all tiles into a single flat leaf (legacy). Default is a "
            "memory-safe kind=partition with one part per spatial tile."
        ),
    ),
    recipe: Optional[str] = typer.Option(
        None,
        "--recipe",
        help=(
            "Per-part LOD recipe applied to each spatial tile-part as it streams: "
            "'additive' (each part a prefix-sum ladder → partitioned topology) or "
            "'substitutive' (each part its own coarse↔fine lod group → mosaic). "
            "Default: bare-leaf parts (no per-part LOD). Closes the tiled-data LOD "
            "gap without re-loading the whole volume. Falls back to the recipe "
            "recorded at plan time. Mutually exclusive with --flat."
        ),
    ),
    n_lods: Optional[int] = typer.Option(
        None, "--n-lods", help="Additive ladder depth (additive recipe)."
    ),
    compression_factor: Optional[int] = typer.Option(
        None, "-K", "--compression-factor", help="Substitutive reduction factor."
    ),
    levels: Optional[int] = typer.Option(
        None, "-L", "--levels", help="Substitutive level count (substitutive recipe)."
    ),
    substitutive_method: Optional[str] = typer.Option(
        None, "--substitutive-method", help="Substitutive coarsening method."
    ),
    coarsen_dims: Optional[str] = typer.Option(
        None,
        "--coarsen-dims",
        help=(
            "Comma-separated center-column indices substitutive coarsening may "
            "merge over; the rest become hard barriers. Default: spatial dims only "
            "(stacked-timepoint axis is a barrier)."
        ),
    ),
) -> None:
    """Run the merge step for a completed batch job.

    Normally runs as a dependent Slurm job, but this command allows
    running it manually or re-running if the merge job failed.

    By default the tiles are assembled into a ``kind=partition`` file (one part
    per spatial tile) — streamed tile-by-tile so peak memory is a single
    tile-region, and the spatial structure is preserved for per-part frustum
    culling. Pass ``--flat`` for the legacy single-leaf concatenation (reloads
    every tile into memory).

    Pass ``--recipe`` to give each tile-part its own LOD ladder as it streams —
    the memory-safe way to add level-of-detail to tiled output (the canonical
    ``cal → fit → lod`` chain otherwise can't, since ``lod`` rejects a partition).

    Examples:
        luxar gsplat slurm-fit merge output_dir/

        luxar gsplat slurm-fit merge output_dir/ --flat

        luxar gsplat slurm-fit merge output_dir/ --recipe additive --n-lods 6

        luxar gsplat slurm-fit merge output_dir/ --recipe substitutive -K 4 -L 3

        luxar gsplat slurm-fit merge output_dir/ --channel-colors "#ff0080,#00ff00"
    """
    try:
        from luxar.cli.gsplat_config import parse_hex_color
        from luxar.gsplats.batch.manifest import load_manifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results

        manifest = load_manifest(output_dir)

        colors = None
        color_source = channel_colors or (
            ",".join(manifest.channel_colors) if manifest.channel_colors else None
        )
        if color_source:
            colors = [parse_hex_color(c.strip()) for c in color_source.split(",")]

        # Resolve the per-part recipe + its knobs, CLI overriding the values
        # recorded at plan time (manifest.merge_recipe / merge_recipe_args).
        eff_recipe = recipe or manifest.merge_recipe
        recipe_params = None
        if eff_recipe is not None:
            recipe_params = _build_merge_recipe_params(
                manifest.merge_recipe_args,
                n_lods=n_lods,
                compression_factor=compression_factor,
                levels=levels,
                substitutive_method=substitutive_method,
                coarsen_dims=coarsen_dims,
            )

        with asection(f"Merging batch results: {output_dir}"):
            final_path = merge_batch_results(
                manifest=manifest,
                output_dir=output_dir,
                channel_colors=colors,
                force=force,
                flat=flat,
                recipe=eff_recipe,
                recipe_params=recipe_params,
            )
            aprint(f"\nFinal output: {final_path}")

    except (typer.Exit, typer.BadParameter):
        # Usage errors (e.g. an invalid --substitutive-method) surface cleanly
        # instead of being swallowed into an "Error: ..." traceback below.
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


# ── Hidden batch worker commands for denoise pipeline ────────────


@app_batch.command("denoise-calibrate", hidden=True)
def batch_denoise_calibrate_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
) -> None:
    """[Internal] Run NLM calibration for batch denoise pipeline.

    Reads manifest, calibrates h per channel, writes results back.
    Called by the calibration Slurm job.
    """
    try:
        import json

        from luxar.gsplats.batch.manifest import load_manifest, save_manifest
        from luxar.gsplats.preprocessing.denoise_pipeline import calibrate_all_channels

        manifest = load_manifest(output_dir)

        if not manifest.denoise:
            aprint("Error: denoise not enabled in manifest")
            raise typer.Exit(1)

        with asection("NLM Calibration"):
            h_values = calibrate_all_channels(
                input_path=Path(manifest.input_path),
                n_timepoints=manifest.n_timepoints,
                n_channels=manifest.n_channels,
                channel_indices=(
                    manifest.channel_indices
                    if manifest.channel_indices
                    else list(range(manifest.n_channels))
                ),
                timepoint_indices=manifest.timepoint_indices,
                array_key=manifest.array_key,
                calibration_samples=manifest.calibration_samples,
                patch_size=manifest.denoise_patch_size,
                search_distance=manifest.denoise_search_distance,
                backend=manifest.denoise_backend,
                h_override=manifest.denoise_h,
            )

            # Write h_values to manifest (string keys for JSON)
            manifest.denoise_h_values = {str(k): v for k, v in h_values.items()}
            save_manifest(manifest, output_dir)

            # Also write standalone JSON for easy reading by other jobs
            h_path = output_dir / "denoise_h_values.json"
            h_path.write_text(json.dumps(h_values, indent=2))

            aprint(f"Calibrated h values: {h_values}")
            aprint(f"Saved to {h_path}")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1) from e


@app_batch.command("denoise-preprocess", hidden=True)
def batch_denoise_preprocess_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    task_id: int = typer.Argument(..., help="Array task ID (encodes T*n_c + C)"),
) -> None:
    """[Internal] Denoise one (T,C) volume for batch preprocess pipeline.

    Called by the denoise Slurm array job, one task per (timepoint, channel).
    """
    try:
        import json

        import zarr

        from luxar.cli.gsplat_config import load_volume
        from luxar.gsplats.batch.manifest import load_manifest
        from luxar.gsplats.preprocessing.denoise_pipeline import denoise_volume_array

        manifest = load_manifest(output_dir)

        # Read calibrated h values
        h_path = output_dir / "denoise_h_values.json"
        if not h_path.exists():
            aprint("Error: denoise_h_values.json not found. Run calibration first.")
            raise typer.Exit(1)
        h_values = json.loads(h_path.read_text())

        # Decode task_id -> (t_idx, c_idx) within selected indices
        n_c = manifest.n_channels
        t_idx = task_id // n_c
        c_idx = task_id % n_c

        # Map to real dataset indices
        t_indices = manifest.timepoint_indices or list(range(manifest.n_timepoints))
        c_indices = manifest.channel_indices or list(range(manifest.n_channels))
        t_real = t_indices[t_idx]
        c_real = c_indices[c_idx]

        h = h_values.get(str(c_real), 0.04)

        with asection(f"Denoising T={t_real} C={c_real} (h={h:.4f})"):
            # Load volume
            volume = load_volume(
                Path(manifest.input_path),
                channel=c_real if manifest.n_channels > 1 else None,
                timepoint=t_real if manifest.n_timepoints > 1 else None,
                array_key=manifest.array_key,
            )
            aprint(f"Loaded: shape={volume.shape}")

            # Denoise
            denoised = denoise_volume_array(
                volume,
                h=h,
                patch_size=manifest.denoise_patch_size,
                search_distance=manifest.denoise_search_distance,
                backend=manifest.denoise_backend,
                use_2d=manifest.denoise_2d,
            )

            # Write to denoised.zarr
            zarr_path = output_dir / "denoised.zarr"
            store = zarr.open(str(zarr_path), mode="a")

            spatial = denoised.shape
            full_shape = (len(t_indices), len(c_indices), *spatial)
            chunks = (1, 1, *[min(s, 128) for s in spatial])

            if "data" not in store:
                store.create_dataset(
                    "data",
                    shape=full_shape,
                    chunks=chunks,
                    dtype=np.float32,
                )
            store["data"][t_idx, c_idx] = denoised
            aprint(f"Written to denoised.zarr[{t_idx}, {c_idx}]")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1) from e
