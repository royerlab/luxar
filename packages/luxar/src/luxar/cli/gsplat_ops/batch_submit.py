"""Implementation helper for ``batch-fit submit`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from arbol import aprint

from .batch_submit_packing import resolve_tasks_per_job
from .batch_submit_plan_output import print_batch_submit_plan
from .batch_submit_preemptible import resolve_preemptible_partition
from .batch_submit_slurm import submit_batch_jobs


def run_batch_submit(
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
        help="[--tiling content] Scan ONLY this single timepoint for the shared "
        "box plan. Default (unset): max-project up to --plan-samples timepoints so "
        "boxes cover any region with signal at ANY timepoint (avoids holes where "
        "content moves over time).",
        rich_help_panel="Content-aware tiling",
    ),
    plan_samples: int = typer.Option(
        16,
        "--plan-samples",
        help="[--tiling content] Max number of evenly-spaced timepoints to "
        "max-project when building the shared box plan (default 16; ignored when "
        "--plan-timepoint pins a single timepoint).",
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
    floor: Optional[str] = typer.Option(
        None,
        "--floor",
        help="Background floor / DC-offset suppression per tile (default: "
        "auto): auto | pN | <float> | none. Unset lets a `floor:` in "
        "--config apply, else defaults to auto. See `gsplat fit --help`.",
    ),
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
        "When omitted, each tile uses the per-fit default (0.95 for uniform "
        "tiles, 0.999 near-lossless for content boxes). Set to 0 to keep every "
        "splat.",
    ),
    # Denoising
    batch_denoise: bool = typer.Option(
        False,
        "--denoise",
        help="Denoise volumes before fitting (NLM). Auto-calibrates h per channel.",
        rich_help_panel="Denoising",
    ),
    batch_denoise_h: Optional[float] = typer.Option(
        None,
        "--denoise-h",
        help="Manual NLM h (skip calibration)",
        rich_help_panel="Denoising",
    ),
    batch_denoise_2d: bool = typer.Option(
        False,
        "--denoise-2d",
        help="Use 2D NLM (slice-by-slice) instead of 3D",
        rich_help_panel="Denoising",
    ),
    batch_denoise_patch_size: int = typer.Option(
        3,
        "--denoise-patch-size",
        help="NLM patch size",
        rich_help_panel="Denoising",
    ),
    batch_denoise_search_distance: int = typer.Option(
        5,
        "--denoise-search-distance",
        help="NLM search distance",
        rich_help_panel="Denoising",
    ),
    batch_denoise_backend: str = typer.Option(
        "auto",
        "--denoise-backend",
        help="NLM backend",
        rich_help_panel="Denoising",
    ),
    batch_calibration_samples: int = typer.Option(
        5,
        "--calibration-samples",
        help="Timepoints to sample for h calibration",
        rich_help_panel="Denoising",
    ),
    batch_preprocess: Optional[bool] = typer.Option(
        None,
        "--preprocess/--no-preprocess",
        help="Write denoised volumes to zarr before fitting (default: off, denoise per-tile on-the-fly).",
        rich_help_panel="Denoising",
    ),
    # Slurm params
    partition: Optional[str] = typer.Option(
        None,
        "--partition",
        "-p",
        help="Slurm partition (required)",
        rich_help_panel="Slurm resources",
    ),
    max_concurrent: Optional[int] = typer.Option(
        None,
        "--max-concurrent",
        help="Maximum simultaneous Slurm array tasks (limits cluster usage). "
        "Maps to --array=0-N%%MAX. No limit if omitted.",
        rich_help_panel="Slurm resources",
    ),
    preemptible: bool = typer.Option(
        False,
        "--preemptible",
        help="Also submit tasks on a preemptible partition for extra throughput. "
        "Auto-detects the preemptible partition. Preempted tasks are automatically "
        "requeued. Uses atomic tile writes to handle interruptions safely.",
        rich_help_panel="Slurm resources",
    ),
    preemptible_partition_opt: Optional[str] = typer.Option(
        None,
        "--preemptible-partition",
        help="Explicit preemptible partition name (skip auto-detection).",
        rich_help_panel="Slurm resources",
    ),
    preemptible_concurrent: Optional[int] = typer.Option(
        None,
        "--preemptible-concurrent",
        help="Max concurrent tasks on preemptible partition. "
        "Defaults to same as --max-concurrent.",
        rich_help_panel="Slurm resources",
    ),
    account: Optional[str] = typer.Option(
        None, "--account", "-A", rich_help_panel="Slurm resources"
    ),
    qos: Optional[str] = typer.Option(None, "--qos", rich_help_panel="Slurm resources"),
    gpus: int = typer.Option(
        1, "--gpus", help="GPUs per task", rich_help_panel="Slurm resources"
    ),
    cpus: int = typer.Option(
        4, "--cpus", help="CPUs per task", rich_help_panel="Slurm resources"
    ),
    mem: int = typer.Option(
        32, "--mem", help="Memory per task (GB)", rich_help_panel="Slurm resources"
    ),
    time_limit: Optional[str] = typer.Option(
        None,
        "--time",
        help="Wall time per task override (HH:MM:SS)",
        rich_help_panel="Slurm resources",
    ),
    gpu_name_opt: Optional[str] = typer.Option(
        None,
        "--gpu",
        help="GPU name from profile (auto-detect if omitted)",
        rich_help_panel="Slurm resources",
    ),
    gpu_mem: Optional[int] = typer.Option(
        None,
        "--gpu-mem",
        help="Target GPU memory in GB (picks closest profile)",
        rich_help_panel="Slurm resources",
    ),
    # Merge
    channel_colors: Optional[str] = typer.Option(
        None,
        "--channel-colors",
        help="Hex colors for per-channel merge",
        rich_help_panel="Merge LOD",
    ),
    merge_recipe: Optional[str] = typer.Option(
        None,
        "--merge-recipe",
        help=(
            "Per-part LOD recipe applied to each spatial tile-part by the merge "
            "job: 'stream' (tiles topology) or 'levels' (adaptive). "
            "Default: bare-leaf parts. The merge sbatch script invokes "
            "`batch-fit merge --recipe <r>` with the knobs below."
        ),
        rich_help_panel="Merge LOD",
    ),
    merge_n_lods: Optional[int] = typer.Option(
        None,
        "--merge-n-lods",
        help="Additive ladder depth for --merge-recipe.",
        rich_help_panel="Merge LOD",
    ),
    merge_additive_method: Optional[str] = typer.Option(
        None,
        "--merge-additive-method",
        help="Additive ladder method for --merge-recipe stream "
        "(auto (default) | greedy | self_energy).",
        rich_help_panel="Merge LOD",
    ),
    merge_breakpoints: Optional[str] = typer.Option(
        None,
        "--merge-breakpoints",
        help="Additive ladder breakpoints for --merge-recipe stream "
        "('equal-count' | 'stream:C' | 'counts:...' | 'energy:...').",
        rich_help_panel="Merge LOD",
    ),
    merge_target_ms: Optional[float] = typer.Option(
        None,
        "--merge-target-ms",
        min=1.0,
        help="[--merge-recipe stream] streaming sizing: derive 'stream:<c>' "
        "breakpoints so each part's first additive chunk downloads in ~this "
        "many ms at --merge-bandwidth-mbps. Mutually exclusive with "
        "--merge-breakpoints.",
        rich_help_panel="Merge LOD",
    ),
    merge_bandwidth_mbps: Optional[float] = typer.Option(
        None,
        "--merge-bandwidth-mbps",
        min=0.1,
        help="Assumed downlink for --merge-target-ms sizing (default 25).",
        rich_help_panel="Merge LOD",
    ),
    merge_bytes_per_splat: Optional[float] = typer.Option(
        None,
        "--merge-bytes-per-splat",
        min=0.1,
        help="Override the on-wire bytes/splat for --merge-target-ms sizing "
        "(default: analytic estimate for the merged parts).",
        rich_help_panel="Merge LOD",
    ),
    merge_compression_factor: Optional[int] = typer.Option(
        None,
        "--merge-compression-factor",
        help="Substitutive K for --merge-recipe.",
        rich_help_panel="Merge LOD",
    ),
    merge_levels: Optional[int] = typer.Option(
        None,
        "--merge-levels",
        help="Substitutive level count for --merge-recipe.",
        rich_help_panel="Merge LOD",
    ),
    merge_substitutive_method: Optional[str] = typer.Option(
        None,
        "--merge-substitutive-method",
        help="Substitutive coarsening method for --merge-recipe.",
        rich_help_panel="Merge LOD",
    ),
    merge_coarsen_dims: Optional[str] = typer.Option(
        None,
        "--merge-coarsen-dims",
        help="Comma-separated center-column indices --merge-recipe levels "
        "may coarsen over (the rest stay hard barriers). Default: spatial dims "
        "only (the stacked-timepoint axis is a barrier).",
        rich_help_panel="Merge LOD",
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
        luxar gsplat batch-fit submit data.ome.zarr output/ -p gpu --dry-run

        luxar gsplat batch-fit submit data.ome.zarr output/ --partition gpu

        luxar gsplat batch-fit submit data.ome.zarr out/ -p gpu --tile-size 256 --preset hifi

        luxar gsplat batch-fit submit keller.zarr.zip out/ -p gpu --tile-size 128 \\
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
        import math

        from luxar.cli.gsplat_config import PRESETS
        from luxar.cli.gsplat_ops.batch_planning import (
            ContentKnobs,
            DenoiseConfig,
            FitConfig,
            MergeConfig,
            plan_batch,
        )
        from luxar.gsplats.batch.env_capture import (
            capture_environment,
            generate_env_preamble,
            get_slurm_scheduler_info,
            is_slurm_mps_available,
        )
        from luxar.gsplats.batch.slurm_gen import (
            generate_fit_sbatch,
            generate_merge_sbatch,
        )
        from luxar.gsplats.batch.time_estimate import estimate_slurm_time_limit
        from luxar.gsplats.gpu_profile import (
            get_gpu_summary,
            get_gpu_throughput_table,
            load_profiles,
        )

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
            aprint("  luxar gsplat batch-fit submit ... --tile-size 128")
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

        # 1b. GPU-profile-derived sizing inputs (uniform auto tile-size + ETA).
        peak_shape = peak.get("shape", [])
        oom = (summary or {}).get("oom_boundaries", {}).get("3d", {})
        max_shape = oom.get("max_successful_shape", peak_shape)
        throughput_table = get_gpu_throughput_table(gpu_name=resolved_gpu)

        # 2-6. Discover + decompose + build the manifest/jobs. This whole half is
        # shared verbatim with `batch-fit run` (the local runner) via plan_batch.
        fit_cfg = FitConfig(
            preset=preset,
            seeds=seeds,
            iters=iters,
            config=config,
            floor=floor,
            progressive=batch_progressive,
            splats_per_pass=batch_splats_per_pass,
            psnr_patience=batch_psnr_patience,
            max_passes=batch_max_passes,
            cull_retention=batch_cull_retention,
        )
        denoise_cfg = DenoiseConfig(
            denoise=batch_denoise,
            denoise_h=batch_denoise_h,
            denoise_2d=batch_denoise_2d,
            patch_size=batch_denoise_patch_size,
            search_distance=batch_denoise_search_distance,
            backend=batch_denoise_backend,
            calibration_samples=batch_calibration_samples,
            preprocess=batch_preprocess,
        )
        content_cfg = ContentKnobs(
            cal=cal,
            k_star_ref=k_star_ref,
            n_features_ref=n_features_ref,
            saturation_exponent=saturation_exponent,
            saturation_cap=saturation_cap,
            feature_threshold=feature_threshold,
            feature_metric=feature_metric,
            cell=cell,
            target_features=target_features,
            min_leaf=min_leaf,
            max_leaf=max_leaf,
            plan_timepoint=plan_timepoint,
            plan_samples=plan_samples,
        )
        merge_cfg = MergeConfig(
            recipe=merge_recipe,
            channel_colors=channel_colors,
            n_lods=merge_n_lods,
            additive_method=merge_additive_method,
            breakpoints=merge_breakpoints,
            target_ms=merge_target_ms,
            bandwidth_mbps=merge_bandwidth_mbps,
            bytes_per_splat=merge_bytes_per_splat,
            compression_factor=merge_compression_factor,
            levels=merge_levels,
            substitutive_method=merge_substitutive_method,
            coarsen_dims=merge_coarsen_dims,
        )

        # Merge-recipe knobs (incl. --merge-target-ms sizing) are resolved
        # INSIDE plan_batch, after shape discovery — so the ladder is sized
        # with the true merged ndim, matching `batch-fit merge`.
        plan = plan_batch(
            input_path=input_path,
            output_dir=output_dir,
            tiling=tiling,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            axes_list=axes_list,
            array_key=array_key,
            timepoints_slice=timepoints_slice,
            channels_slice=channels_slice,
            fit=fit_cfg,
            denoise=denoise_cfg,
            content=content_cfg,
            merge=merge_cfg,
            max_shape=max_shape,
            throughput_table=throughput_table,
            resolved_gpu=resolved_gpu,
        )
        manifest = plan.manifest
        n_t = manifest.n_timepoints
        n_c = manifest.n_channels
        spatial = manifest.spatial_shape
        mode = manifest.mode
        n_tiles = manifest.n_tiles
        total_tasks = manifest.total_tasks
        tile_voxels = plan.tile_voxels
        needs_tiling = plan.needs_tiling
        est_seconds = manifest.estimated_seconds_per_task
        denoise_mode = manifest.denoise_mode
        # auto_tile is purely for the printed plan (manifest.tile_size is resolved).
        auto_tile = (tile_size is None) and mode != "content"
        tile_size = manifest.tile_size
        preset_config = PRESETS.get(preset, PRESETS["standard"])
        n_iters = iters if iters is not None else preset_config.get("n_iters", 3000)

        # 5b. Compute tasks-per-job packing
        #
        # When each volume is small relative to GPU capacity, we pack
        # multiple fitting tasks sequentially into one Slurm job to
        # reduce scheduling overhead (fewer array elements to launch).
        # 5b-i. Query scheduler for smart packing defaults
        sched_info = get_slurm_scheduler_info()
        uses_backfill = sched_info["uses_backfill"]
        no_job_limit = sched_info["max_jobs_per_user"] is None

        tasks_per_job = resolve_tasks_per_job(
            tasks_per_job,
            parallel=parallel,
            uses_backfill=uses_backfill,
            no_job_limit=no_job_limit,
            max_shape=max_shape,
            tile_voxels=tile_voxels,
            est_seconds=est_seconds,
        )

        n_slurm_jobs = math.ceil(total_tasks / tasks_per_job)
        if parallel:
            # Parallel: all tasks run at once, so wall time ≈ 1 task
            est_seconds_per_job = est_seconds * 1.2  # 20% overhead for contention
        else:
            est_seconds_per_job = est_seconds * tasks_per_job
        slurm_time = time_limit or estimate_slurm_time_limit(est_seconds_per_job)
        total_gpu_hours = est_seconds * total_tasks / 3600.0

        # 6. Build manifest — fit_args, denoise mode, channel colors, and the
        # per-part merge-recipe args were all resolved inside plan_batch above;
        # the manifest is plan.manifest. We only add the Slurm-specific fields
        # (partition, packing, preemptible) here.

        # Preemptible partition detection
        preempt_partition = resolve_preemptible_partition(
            preemptible=preemptible,
            preemptible_partition_opt=preemptible_partition_opt,
        )

        # plan_batch built the manifest + jobs (dataset/decomposition/fit/merge/
        # denoise fields). Stamp the Slurm-specific fields onto it here.
        manifest.slurm_time_limit = slurm_time
        manifest.slurm_partition = partition
        manifest.slurm_account = account
        manifest.slurm_qos = qos
        manifest.slurm_gpus = gpus
        manifest.slurm_cpus = cpus
        manifest.slurm_mem_gb = mem
        manifest.tasks_per_job = tasks_per_job
        manifest.parallel_tasks_per_job = parallel
        manifest.max_concurrent = max_concurrent
        manifest.preemptible = preempt_partition is not None
        manifest.preemptible_partition = preempt_partition
        manifest.preemptible_max_concurrent = (
            (preemptible_concurrent or max_concurrent) if preempt_partition else None
        )

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
        print_batch_submit_plan(
            input_name=input_path.name,
            n_t=n_t,
            n_c=n_c,
            spatial=spatial,
            resolved_gpu=resolved_gpu,
            peak_gvs=peak.get("gvoxel_per_s", "?"),
            peak_shape=peak.get("shape", []),
            mode=mode,
            n_tiles=n_tiles,
            tile_overlap=tile_overlap,
            needs_tiling=needs_tiling,
            tile_size=tile_size,
            auto_tile=auto_tile,
            total_tasks=total_tasks,
            tasks_per_job=tasks_per_job,
            parallel=parallel,
            n_slurm_jobs=n_slurm_jobs,
            mps_available_fn=is_slurm_mps_available,
            uses_backfill=uses_backfill,
            no_job_limit=no_job_limit,
            est_seconds=est_seconds,
            preset=preset,
            n_iters=n_iters,
            est_seconds_per_job=est_seconds_per_job,
            total_gpu_hours=total_gpu_hours,
            slurm_time=slurm_time,
            partition=partition,
            gpus=gpus,
            cpus=cpus,
            mem=mem,
            output_dir=output_dir,
        )

        if dry_run:
            aprint("Dry run -- omit --dry-run to actually submit.")
            raise typer.Exit(0)

        # 9. Submit
        submit_batch_jobs(
            output_dir=output_dir,
            manifest=manifest,
            fit_script=fit_script,
            merge_script=merge_script,
            preamble=preamble,
            calibrate_script=calibrate_script,
            denoise_script=denoise_script,
            preempt_fit_script=preempt_fit_script,
            total_tasks=total_tasks,
            preempt_partition=preempt_partition,
        )

    except typer.Exit:
        raise
    except typer.BadParameter:
        # A user-input error (bad --merge-recipe knob, empty content plan, …):
        # let Typer/Click render it cleanly instead of dumping a traceback.
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
