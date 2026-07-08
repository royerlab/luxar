"""``luxar gsplat batch-fit`` — cluster-scale fitting via Slurm.

Owns the ``app_batch`` Typer sub-app and its commands; the aggregator
(``cli/gsplat_commands.py``) mounts it via ``add_typer``. Extracted from the
former monolithic ``gsplat_commands.py`` (package-refactor-plan P3/P4/P6).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import typer
from arbol import aprint, asection

if TYPE_CHECKING:
    from luxar.gsplats.lod.recipes import RecipeParams

from luxar.encoding.compression import WIDTH_AWARE_DEFAULT, resolve_compressor

app_batch = typer.Typer(
    help="Fit a whole nD dataset across its axes — locally across GPUs "
    "(`batch-fit run`) or on a Slurm cluster (`batch-fit submit`). The "
    "scheduler-agnostic, scaled-up sibling of `gsplat fit`."
)

# Re-exported from batch_planning (single source); kept importable here for
# back-compat with callers/tests that import it from this module.
from luxar.cli.gsplat_ops.batch_planning import (  # noqa: E402,F401
    _select_plan_timepoints,
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
    floor: str = typer.Option(
        "auto",
        "--floor",
        help="Background floor / DC-offset suppression per tile (on by "
        "default): auto | pN | <float> | none. See `gsplat fit --help`.",
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
        import subprocess

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
        from luxar.gsplats.batch.manifest import save_manifest
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

        # 6. Build manifest — fit_args, denoise mode, channel colors, and the
        # per-part merge-recipe args were all resolved inside plan_batch above;
        # the manifest is plan.manifest. We only add the Slurm-specific fields
        # (partition, packing, preemptible) here.

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
            f"  Jobs: {n_t} x {n_c} x {n_tiles} = {total_tasks} fitting tasks ({slot})"
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
        aprint(f"Check status: luxar gsplat batch-fit status {out}")

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


@app_batch.command("run")
def batch_run(
    input_path: Path = typer.Argument(..., exists=True, help="Input OME-Zarr dataset"),
    output_dir: Path = typer.Argument(..., help="Output directory for batch results"),
    # Tiling
    tile_size: Optional[int] = typer.Option(
        None,
        "--tile-size",
        help="Tile size in voxels (uniform mode). Required for a multi-tile "
        "uniform fit unless a GPU benchmark profile is available.",
    ),
    tile_overlap: int = typer.Option(32, "--overlap", help="Tile overlap in voxels"),
    tiling: str = typer.Option(
        "uniform",
        "--tiling",
        help="Spatial decomposition: 'uniform' (a regular tile grid) or 'content' "
        "(a content-balanced box plan built once from a representative timepoint "
        "and reused for every (t,c)). Content needs a density "
        "(--cal or --k-star-ref/--n-features-ref).",
        rich_help_panel="Content-aware tiling",
    ),
    plan_timepoint: Optional[int] = typer.Option(
        None,
        "--plan-timepoint",
        help="[--tiling content] Scan ONLY this timepoint for the shared box plan "
        "(default: max-project up to --plan-samples timepoints).",
        rich_help_panel="Content-aware tiling",
    ),
    plan_samples: int = typer.Option(
        16,
        "--plan-samples",
        help="[--tiling content] Max evenly-spaced timepoints to max-project for "
        "the shared box plan.",
        rich_help_panel="Content-aware tiling",
    ),
    cal: Optional[Path] = typer.Option(
        None,
        "--cal",
        help="[--tiling content] Calibration JSON supplying the density.",
        rich_help_panel="Content-aware tiling",
    ),
    k_star_ref: Optional[int] = typer.Option(
        None, "--k-star-ref", rich_help_panel="Content-aware tiling"
    ),
    n_features_ref: Optional[int] = typer.Option(
        None, "--n-features-ref", rich_help_panel="Content-aware tiling"
    ),
    saturation_exponent: float = typer.Option(
        0.44, "--saturation-exponent", rich_help_panel="Content-aware tiling"
    ),
    saturation_cap: Optional[int] = typer.Option(
        None, "--saturation-cap", rich_help_panel="Content-aware tiling"
    ),
    feature_threshold: Optional[float] = typer.Option(
        None, "--feature-threshold", rich_help_panel="Content-aware tiling"
    ),
    feature_metric: Optional[str] = typer.Option(
        None,
        "--feature-metric",
        help="peaks | edges | intensity",
        rich_help_panel="Content-aware tiling",
    ),
    cell: int = typer.Option(16, "--cell", rich_help_panel="Content-aware tiling"),
    target_features: Optional[int] = typer.Option(
        None, "--target-features", rich_help_panel="Content-aware tiling"
    ),
    min_leaf: int = typer.Option(
        256, "--min-leaf", rich_help_panel="Content-aware tiling"
    ),
    max_leaf: int = typer.Option(
        512, "--max-leaf", rich_help_panel="Content-aware tiling"
    ),
    # Fit params
    preset: str = typer.Option("standard", "--preset", help="Fitting preset"),
    config: Optional[Path] = typer.Option(None, "--config", help="YAML fit config"),
    floor: str = typer.Option(
        "auto",
        "--floor",
        help="Background floor / DC-offset suppression per tile (on by "
        "default): auto | pN | <float> | none. See `gsplat fit --help`.",
    ),
    seeds: Optional[str] = typer.Option(None, "--seeds", help="Seed count or ratio"),
    iters: Optional[int] = typer.Option(
        None, "--iters", "-n", help="Max optimization iterations (overrides preset)"
    ),
    batch_progressive: bool = typer.Option(
        False, "--progressive", help="Progressive fitting per tile (multi-LOD)."
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
    batch_cull_retention: Optional[float] = typer.Option(
        None,
        "--cull-retention",
        help="Per-tile amplitude retention after fitting (default per-fit; 0 keeps all).",
    ),
    # Denoising
    batch_denoise: bool = typer.Option(
        False,
        "--denoise",
        help="Denoise volumes before fitting (NLM, on-the-fly per tile).",
        rich_help_panel="Denoising",
    ),
    batch_denoise_h: Optional[float] = typer.Option(
        None, "--denoise-h", help="Manual NLM h", rich_help_panel="Denoising"
    ),
    batch_denoise_2d: bool = typer.Option(
        False, "--denoise-2d", help="2D NLM", rich_help_panel="Denoising"
    ),
    batch_denoise_patch_size: int = typer.Option(
        3, "--denoise-patch-size", rich_help_panel="Denoising"
    ),
    batch_denoise_search_distance: int = typer.Option(
        5, "--denoise-search-distance", rich_help_panel="Denoising"
    ),
    batch_denoise_backend: str = typer.Option(
        "auto", "--denoise-backend", rich_help_panel="Denoising"
    ),
    # Local GPUs
    gpus: str = typer.Option(
        "auto",
        "--gpus",
        help="GPUs to use: 'auto' (every visible card above a free-VRAM floor, "
        "skipping small cards) | 'all' (every visible card) | 'cpu' | an explicit "
        "list like '0,1,3'.",
        rich_help_panel="Local GPUs",
    ),
    jobs_per_gpu: str = typer.Option(
        "auto",
        "--jobs-per-gpu",
        help="Concurrent fit workers per GPU. 'auto' sizes each card from its own "
        "free VRAM; an integer applies uniformly. Use 1 to be safe on small cards.",
        rich_help_panel="Local GPUs",
    ),
    no_resume: bool = typer.Option(
        False,
        "--no-resume",
        help="Re-fit every task even if its output already exists "
        "(default: resume — skip completed tiles).",
        rich_help_panel="Local GPUs",
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
        help="Per-part LOD recipe applied to each tile-part at merge: 'stream' "
        "(tiles) or 'levels' (adaptive). Default: bare-leaf parts.",
        rich_help_panel="Merge LOD",
    ),
    merge_n_lods: Optional[int] = typer.Option(
        None, "--merge-n-lods", rich_help_panel="Merge LOD"
    ),
    merge_additive_method: Optional[str] = typer.Option(
        None, "--merge-additive-method", rich_help_panel="Merge LOD"
    ),
    merge_breakpoints: Optional[str] = typer.Option(
        None, "--merge-breakpoints", rich_help_panel="Merge LOD"
    ),
    merge_target_ms: Optional[float] = typer.Option(
        None, "--merge-target-ms", min=1.0, rich_help_panel="Merge LOD"
    ),
    merge_bandwidth_mbps: Optional[float] = typer.Option(
        None, "--merge-bandwidth-mbps", min=0.1, rich_help_panel="Merge LOD"
    ),
    merge_bytes_per_splat: Optional[float] = typer.Option(
        None, "--merge-bytes-per-splat", min=0.1, rich_help_panel="Merge LOD"
    ),
    merge_compression_factor: Optional[int] = typer.Option(
        None, "--merge-compression-factor", rich_help_panel="Merge LOD"
    ),
    merge_levels: Optional[int] = typer.Option(
        None, "--merge-levels", rich_help_panel="Merge LOD"
    ),
    merge_substitutive_method: Optional[str] = typer.Option(
        None, "--merge-substitutive-method", rich_help_panel="Merge LOD"
    ),
    merge_coarsen_dims: Optional[str] = typer.Option(
        None, "--merge-coarsen-dims", rich_help_panel="Merge LOD"
    ),
    # Dataset structure / selection
    axes: Optional[str] = typer.Option(
        None,
        "--axes",
        help="Comma-separated axis names overriding auto-detection, e.g. "
        "'time,channel,z,y,x'.",
    ),
    timepoints_slice: Optional[str] = typer.Option(
        None, "--timepoints", help="Python-style slice to select timepoints."
    ),
    channels_slice: Optional[str] = typer.Option(
        None, "--channels", help="Python-style slice to select channels."
    ),
    array_key: Optional[str] = typer.Option(
        None, "--array-key", help="Key path to an array within the zarr store."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show the plan without fitting."
    ),
) -> None:
    """Fit a whole nD dataset locally across multiple GPUs, then merge.

    The local (non-Slurm) sibling of `batch-fit submit`: plans the decomposition
    once (uniform tiles or a shared content box plan), fits every (t,c,slot) task
    with a multi-GPU subprocess pool (one worker pinned per GPU via
    CUDA_VISIBLE_DEVICES, per-GPU concurrency sized from free VRAM), then runs the
    memory-safe streaming merge to a single kind=partition .gsplats.zarr. Resumable
    — re-running skips tiles already on disk.

    Examples:
        luxar gsplat batch-fit run vol.zarr out/ --gpus all --tile-size 256

        luxar gsplat batch-fit run vol.zarr out/ --tiling content --cal cal.json \\
            --gpus auto --merge-recipe stream --n-lods 4 -K...

        luxar gsplat batch-fit run vol.zarr out/ --gpus cpu   # CPU fallback
    """
    tiling = tiling.lower()
    if tiling not in ("uniform", "content"):
        aprint(f"Error: --tiling must be uniform|content, got {tiling!r}")
        raise typer.Exit(1)

    try:
        from luxar.cli.gsplat_config import parse_hex_color
        from luxar.cli.gsplat_ops.batch_planning import (
            ContentKnobs,
            DenoiseConfig,
            FitConfig,
            MergeConfig,
            plan_batch,
        )
        from luxar.gsplats.batch.local_runner import run_batch_local
        from luxar.gsplats.gpu_profile import (
            get_gpu_summary,
            get_gpu_throughput_table,
            load_profiles,
        )
        from luxar.gsplats.utils.device import resolve_gpu_selection

        axes_list = [a.strip() for a in axes.split(",")] if axes else None

        # Optional GPU profile — only used to auto-size uniform tiles; not required.
        summary = get_gpu_summary()
        resolved_gpu = "local"
        max_shape = None
        throughput_table = None
        if summary is not None:
            recs = summary.get("recommendations", {})
            peak = recs.get("peak_throughput_3d", {})
            oom = summary.get("oom_boundaries", {}).get("3d", {})
            max_shape = oom.get("max_successful_shape", peak.get("shape", []))
            profiles = load_profiles()
            for name, entry in profiles.get("gpus", {}).items():
                if entry.get("summary") == summary:
                    resolved_gpu = name
                    break
            throughput_table = get_gpu_throughput_table(gpu_name=resolved_gpu)

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
            preprocess=False,  # local runner denoises on-the-fly per tile
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

        # Resolve merge colors + recipe params for the streaming merge.
        colors = None
        if manifest.channel_colors:
            colors = [parse_hex_color(c.strip()) for c in manifest.channel_colors]
        recipe_params = None
        if manifest.merge_recipe is not None:
            recipe_params = _build_merge_recipe_params(manifest.merge_recipe_args)

        # Plan summary (which GPUs, how many tasks already done).
        try:
            sel = resolve_gpu_selection(gpus)
            gpu_desc = "CPU" if not sel else f"GPU(s) {sel}"
        except ValueError as exc:
            gpu_desc = f"<{exc}>"
        slot = "boxes" if manifest.mode == "content" else "tiles"
        aprint("")
        aprint("=" * 60)
        aprint("LOCAL BATCH FIT")
        aprint("=" * 60)
        aprint(
            f"  Input: {input_path.name} "
            f"(T={manifest.n_timepoints}, C={manifest.n_channels}, "
            f"spatial={'x'.join(str(s) for s in manifest.spatial_shape)})"
        )
        aprint(
            f"  Decomposition: {manifest.mode}, {manifest.n_tiles} {slot}/volume "
            f"(overlap={tile_overlap})"
        )
        aprint(
            f"  Tasks: {manifest.n_timepoints} x {manifest.n_channels} x "
            f"{manifest.n_tiles} = {manifest.total_tasks} fits"
        )
        aprint(f"  Devices: {gpu_desc} (--jobs-per-gpu {jobs_per_gpu})")
        if manifest.merge_recipe:
            aprint(f"  Merge recipe: {manifest.merge_recipe}")
        aprint(f"  Output: {output_dir}")
        aprint("")

        if dry_run:
            aprint("Dry run -- omit --dry-run to actually fit.")
            raise typer.Exit(0)

        final_path = run_batch_local(
            manifest,
            output_dir,
            gpus=gpus,
            jobs_per_gpu=jobs_per_gpu,
            resume=not no_resume,
            channel_colors=colors,
            recipe=manifest.merge_recipe,
            recipe_params=recipe_params,
        )
        aprint(f"\nFinal output: {final_path}")
        aprint(f"Inspect: luxar gsplat info {final_path}")
        aprint(f"Validate tiles: luxar gsplat batch-fit validate {output_dir}")

    except typer.Exit:
        raise
    except typer.BadParameter:
        # User-input error: render cleanly (no traceback).
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
        luxar gsplat batch-fit status output_dir/
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
        luxar gsplat batch-fit validate output_dir/

        luxar gsplat batch-fit validate output_dir/ --fix
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
    """Check a v3.x gsplats leaf's required array sub-dirs (no decode)."""
    import json

    # Cholesky factors are stored as the v3.1 split (``cholesky_factors_diag``,
    # optionally + ``cholesky_factors_offdiag``) or a single v3.0
    # ``cholesky_factors`` array. The diagonal is the marker for the split.
    diag_dir = node_dir / "cholesky_factors_diag"
    is_split = diag_dir.is_dir()
    chol_name = "cholesky_factors_diag" if is_split else "cholesky_factors"
    for arr_name in ("centers", "amplitudes", chol_name):
        arr_dir = node_dir / arr_name
        if not arr_dir.is_dir():
            return f"missing_{arr_name}@{label}"
        if not (arr_dir / ".zarray").exists():
            return f"no_zarray_{arr_name}@{label}"

    # v3.1 split: for d > 1 the off-diagonal array is mandatory — only d == 1
    # omits it. A leaf with the diagonal but no off-diagonal is a partial /
    # corrupt write; surface it (recoverable via re-fit) rather than passing.
    if is_split:
        try:
            d = int(json.loads((diag_dir / ".zarray").read_text())["shape"][1])
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,  # shape is null / scalar / non-subscriptable
            ValueError,
        ):
            return f"no_zarray_cholesky_factors_diag@{label}"
        if d > 1 and not (node_dir / "cholesky_factors_offdiag" / ".zarray").exists():
            return f"missing_cholesky_factors_offdiag@{label}"
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
    ``batch-fit validate --fix`` never silently deletes an unmigrated tile).
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

    from luxar.gsplats.io.save_gsplats import SUPPORTED_FORMAT_VERSIONS

    version = attrs.get("format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
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
        luxar gsplat batch-fit cancel output_dir/
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


# Per-part merge recipes accept only ``additive`` / ``substitutive`` (the
# composed topologies are reached via `gsplat flatten` → `gsplat lod`). These
# mirror lod.py's ``_OPTION_TOKENS`` / ``_ALLOWED_TOKENS`` so the merge path
# rejects — rather than silently ignores — a knob irrelevant to (or given
# without) a recipe. ``--coarsen-dims`` belongs to the substitutive (mosaic)
# per-part lod group.
_MERGE_OPTION_TOKENS = {
    "--n-lods": "additive",
    "--additive-method": "additive",
    "--breakpoints": "additive",
    "--target-ms": "additive",
    "--bandwidth-mbps": "additive",
    "--bytes-per-splat": "additive",
    "--compression-factor": "substitutive",
    "--levels": "substitutive",
    "--substitutive-method": "substitutive",
    "--coarsen-dims": "substitutive",
}
_MERGE_ALLOWED_TOKENS = {
    "stream": frozenset({"additive"}),
    "levels": frozenset({"substitutive"}),
}


def _build_merge_recipe_params(
    stored: dict,
    *,
    n_lods: Optional[int] = None,
    additive_method: Optional[str] = None,
    breakpoints: Optional[str] = None,
    compression_factor: Optional[int] = None,
    levels: Optional[int] = None,
    substitutive_method: Optional[str] = None,
    coarsen_dims: Optional[str] = None,
) -> "RecipeParams":
    """Build a ``RecipeParams`` for the per-part merge recipe.

    Each knob is resolved CLI-first, then the value recorded at plan time
    (``manifest.merge_recipe_args``, string-valued), then the ``RecipeParams``
    default. ``coarsen_dims`` is parsed from a comma string to a tuple of ints;
    leaving it unset lets the merge default it per part (spatial dims only). The
    additive knobs (``additive_method`` / ``breakpoints``) bring
    the merge recipe to parity with ``fit --recipe`` and ``gsplat lod``.

    NOTE: the L2 refine knobs (``refine`` / ``refine_iters`` on ``RecipeParams``)
    are deliberately NOT exposed at the merge yet — per-part refits across
    hundreds of tiles need their own perf validation first; the RecipeParams
    defaults ("none") keep the merge byte-identical to before.
    """
    from luxar.cli.lod import (
        _VALID_ADDITIVE_METHODS,
        _VALID_SUBSTITUTIVE_METHODS,
        _parse_lod_breakpoints,
    )
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
    am = _resolve("additive-method", additive_method, str)
    if am is not None:
        am_norm = am.strip().replace("-", "_")
        if am_norm not in _VALID_ADDITIVE_METHODS:
            raise typer.BadParameter(
                f"--additive-method must be one of "
                f"{list(_VALID_ADDITIVE_METHODS)}; got {am!r}"
            )
        overrides["additive_method"] = am_norm
    bp = _resolve("breakpoints", breakpoints, str)
    if bp is not None:
        overrides["breakpoints"] = _parse_lod_breakpoints(bp)
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


def _measure_tiles_bytes_per_splat(
    tiles_dir: Path, tile_names: List[str]
) -> Tuple[Optional[float], int]:
    """Measure the real on-wire bytes/splat from completed tile stores.

    The tiles enumerated by the manifest exist on disk at merge time, so
    ``--target-ms`` can be sized against their true average storage cost
    rather than the analytic estimate. Sums :func:`measure_store_bytes` and
    the stored splat counts (every ``centers`` array's ``.zarray`` row count —
    covers leaves, additive sublods, and lod levels without decoding data)
    over every completed tile. Returns ``(bytes_per_splat, n_tiles_measured)``;
    ``(None, 0)`` when nothing could be measured (missing/empty tiles) —
    callers then fall back to the analytic estimate.
    """
    import json

    from luxar.cli.lod import measure_store_bytes

    total_bytes = 0
    total_splats = 0
    n_measured = 0
    for name in tile_names:
        tile = tiles_dir / name
        if not tile.is_dir():
            continue
        tile_bytes = measure_store_bytes(tile)
        tile_splats = 0
        for zarray in tile.rglob("centers/.zarray"):
            try:
                tile_splats += int(json.loads(zarray.read_text())["shape"][0])
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                continue
        if tile_bytes > 0 and tile_splats > 0:
            total_bytes += tile_bytes
            total_splats += tile_splats
            n_measured += 1
    if total_bytes <= 0 or total_splats <= 0:
        return None, 0
    return total_bytes / total_splats, n_measured


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
            "'stream' (each part a prefix-sum ladder → tiles topology) or "
            "'levels' (each part its own coarse↔fine lod group → adaptive). "
            "Default: bare-leaf parts (no per-part LOD). Closes the tiled-data LOD "
            "gap without re-loading the whole volume. Falls back to the recipe "
            "recorded at plan time. Mutually exclusive with --flat."
        ),
    ),
    no_recipe: bool = typer.Option(
        False,
        "--no-recipe",
        help=(
            "Force a recipe-less merge (bare-leaf parts), overriding any "
            "merge_recipe recorded at plan time. Use this to merge without LOD "
            "when the manifest defaulted to a recipe."
        ),
    ),
    n_lods: Optional[int] = typer.Option(
        None, "--n-lods", help="Additive ladder depth (stream recipe)."
    ),
    additive_method: Optional[str] = typer.Option(
        None,
        "--additive-method",
        help="Additive ladder method: auto (default) | greedy | self_energy "
        "(stream recipe).",
    ),
    breakpoints: Optional[str] = typer.Option(
        None,
        "--breakpoints",
        help="Additive ladder breakpoints: 'equal-count' (default), 'stream:C', "
        "'counts:...' or 'energy:...' (stream recipe).",
    ),
    target_ms: Optional[float] = typer.Option(
        None,
        "--target-ms",
        min=1.0,
        help="[stream recipe] streaming sizing: derive 'stream:<c>' "
        "breakpoints so each part's first additive chunk downloads in ~this "
        "many ms at --bandwidth-mbps. Mutually exclusive with --breakpoints.",
    ),
    bandwidth_mbps: Optional[float] = typer.Option(
        None,
        "--bandwidth-mbps",
        min=0.1,
        help="Assumed downlink for --target-ms sizing (default 25).",
    ),
    bytes_per_splat: Optional[float] = typer.Option(
        None,
        "--bytes-per-splat",
        min=0.1,
        help="Override the on-wire bytes/splat for --target-ms sizing "
        "(default: analytic estimate for the merged parts).",
    ),
    compression_factor: Optional[int] = typer.Option(
        None, "-K", "--compression-factor", help="Substitutive reduction factor."
    ),
    levels: Optional[int] = typer.Option(
        None, "-L", "--levels", help="Substitutive level count (levels recipe)."
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
        luxar gsplat batch-fit merge output_dir/

        luxar gsplat batch-fit merge output_dir/ --flat

        luxar gsplat batch-fit merge output_dir/ --recipe stream --n-lods 6

        luxar gsplat batch-fit merge output_dir/ --recipe levels -K 4 -L 3

        luxar gsplat batch-fit merge output_dir/ --no-recipe   # override a manifest recipe

        luxar gsplat batch-fit merge output_dir/ --channel-colors "#ff0080,#00ff00"
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

        from luxar.cli.lod import reject_irrelevant_recipe_options
        from luxar.gsplats.lod.recipes import PER_PART_RECIPES

        # ── usage validation (up front, before the streaming writer runs) ──
        if recipe is not None and no_recipe:
            raise typer.BadParameter("--recipe and --no-recipe are mutually exclusive.")
        if flat and recipe is not None:
            raise typer.BadParameter(
                "--flat and --recipe are mutually exclusive; --flat concatenates "
                "all tiles into a single bare leaf (no per-part LOD)."
            )

        # Resolve the per-part recipe + its knobs, CLI overriding the values
        # recorded at plan time (manifest.merge_recipe / merge_recipe_args).
        # --no-recipe (or --flat) forces a recipe-less merge regardless of the
        # manifest default.
        # NOTE: the uniform+per-part-LOD warning now fires inside
        # merge_batch_results (the library boundary), so every caller — this CLI,
        # the Slurm merge job, and any direct API use — gets it exactly once.
        from luxar.gsplats.lod.recipes import (
            LEGACY_RECIPE_NAMES,
            canonical_recipe_name,
        )

        if recipe in LEGACY_RECIPE_NAMES:
            raise typer.BadParameter(
                f"recipe {recipe!r} was renamed to "
                f"{LEGACY_RECIPE_NAMES[recipe]!r}; use --recipe "
                f"{LEGACY_RECIPE_NAMES[recipe]}."
            )
        # Manifests written before the rename carry legacy spellings —
        # translate those silently (data compat, not CLI compat).
        manifest_recipe = (
            canonical_recipe_name(manifest.merge_recipe)
            if manifest.merge_recipe
            else None
        )
        eff_recipe = None if (no_recipe or flat) else (recipe or manifest_recipe)

        # Validate the effective recipe NAME before the knob-relevance check —
        # mirrors `gsplat lod`'s RECIPE_NAMES guard (cli/lod.py). Without this an
        # unknown recipe (a typo like `--recipe addative`, or a stale manifest
        # value) reaches reject_irrelevant_recipe_options, whose allowed-token
        # lookup returns empty and misreports a VALID knob as "not used by
        # --recipe addative" — hiding the real error (the recipe name).
        if eff_recipe is not None and eff_recipe not in PER_PART_RECIPES:
            raise typer.BadParameter(
                f"unknown per-part recipe {eff_recipe!r}; choose from "
                f"{', '.join(sorted(PER_PART_RECIPES))}"
                + (" (recorded at plan time in the manifest)" if recipe is None else "")
            )

        # Reject recipe knobs that are irrelevant to (or given without) the
        # effective recipe — previously such knobs were silently dropped. The
        # no-recipe hint depends on WHY there's no recipe: a forced recipe-less
        # merge (--no-recipe/--flat) must not tell the user to "pass --recipe"
        # (it would contradict the flag they just typed).
        if no_recipe or flat:
            forced = "--no-recipe" if no_recipe else "--flat"
            no_recipe_hint = (
                f"{forced} forces a recipe-less (bare-leaf) merge — drop these "
                f"knobs, or drop {forced} and pass --recipe stream|levels "
                f"for per-part LOD."
            )
        else:
            no_recipe_hint = (
                "Pass --recipe stream|levels (without one the merge "
                "writes bare-leaf parts, so these knobs would be ignored)."
            )
        reject_irrelevant_recipe_options(
            eff_recipe,
            {
                "--n-lods": n_lods,
                "--additive-method": additive_method,
                "--breakpoints": breakpoints,
                "--target-ms": target_ms,
                "--bandwidth-mbps": bandwidth_mbps,
                "--bytes-per-splat": bytes_per_splat,
                "--compression-factor": compression_factor,
                "--levels": levels,
                "--substitutive-method": substitutive_method,
                "--coarsen-dims": coarsen_dims,
            },
            _MERGE_OPTION_TOKENS,
            _MERGE_ALLOWED_TOKENS,
            no_recipe_hint=no_recipe_hint,
        )

        # Streaming trio → a concrete stream:<c> breakpoints string. Bytes/splat
        # is MEASURED from the completed tile stores when possible (they exist
        # on disk at merge time), falling back to the analytic estimate for the
        # merged parts (spatial dims + the stacked-timepoint axis when
        # timepoints were stacked; colors when a multi-channel color merge will
        # write them). Mutually exclusive with an explicit --breakpoints; the
        # supporting knobs need --target-ms.
        from luxar.cli.lod import validate_streaming_knobs

        validate_streaming_knobs(
            target_ms, bandwidth_mbps, bytes_per_splat, breakpoints
        )
        eff_breakpoints = breakpoints
        if target_ms is not None:
            from luxar.cli.lod import (
                estimate_bytes_per_splat,
                resolve_streaming_breakpoints,
            )

            merged_ndim = len(manifest.spatial_shape) + (
                1 if manifest.n_timepoints > 1 else 0
            )
            merged_has_colors = bool(colors) and manifest.n_channels > 1
            measured, n_measured = _measure_tiles_bytes_per_splat(
                output_dir / "tiles",
                [job.output_filename for job in manifest.jobs],
            )
            eff_breakpoints = resolve_streaming_breakpoints(
                target_ms,
                bandwidth_mbps,
                bytes_per_splat,
                measured_bps=measured,
                measured_label=(f"measured from {n_measured} completed tile store(s)"),
                analytic_bps=estimate_bytes_per_splat(
                    merged_ndim, has_colors=merged_has_colors
                ),
            )

        recipe_params = None
        if eff_recipe is not None:
            recipe_params = _build_merge_recipe_params(
                manifest.merge_recipe_args,
                n_lods=n_lods,
                additive_method=additive_method,
                breakpoints=eff_breakpoints,
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
                    # string dtype: numpy is only imported under TYPE_CHECKING
                    # in this module (dtype=np.float32 here was a latent
                    # NameError before this change).
                    dtype="float32",
                    compressor=resolve_compressor(WIDTH_AWARE_DEFAULT, "float32"),
                )
            store["data"][t_idx, c_idx] = denoised
            aprint(f"Written to denoised.zarr[{t_idx}, {c_idx}]")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1) from e
