"""Implementation helper for ``batch-fit submit`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from arbol import aprint

from luxar.utils.lod_methods import GSPLAT_ADDITIVE_CHOICES_HELP

from .help_text import BATCH_PROGRESSIVE_HELP
from .plan_configs import build_plan_configs
from .submit_pipeline import (
    generate_all_scripts,
    resolve_gpu_context,
    resolve_packing,
    stamp_slurm_fields,
    validate_tiling_arg,
)
from .submit_plan_output import print_batch_submit_plan
from .submit_slurm import submit_batch_jobs


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
    physical: bool = typer.Option(
        False,
        "--physical",
        help="Fit centers and covariance in physical coordinates using the selected "
        "OME-Zarr coordinateTransformations scale. A voxel_size in --config wins.",
    ),
    floor: Optional[str] = typer.Option(
        None,
        "--floor",
        help="Background floor / DC-offset suppression (default: auto): "
        "auto | pN | <float> | none. Resolved to ONE GLOBAL LEVEL at plan "
        "time — the MINIMUM of the levels measured on a bounded set of evenly "
        "spaced (t, c) slices spanning the whole store (up to 4 timepoints, "
        "always including t=0 and t=T-1 when T > 1, x up to 4 channel-like "
        "coordinates) — recorded in the manifest, and subtracted by every "
        "(timepoint, channel) task and every tile/box. It is deliberately NOT "
        "re-estimated per timepoint: that would be a time-varying pedestal, "
        "i.e. brightness flicker across the merged partition. A minimum cannot "
        "clip a SAMPLED slice to zero (which would drop it silently from the "
        "merge); a dimmer NON-sampled slice still can, since bounded sampling "
        "bounds only what it samples — pass --floor none or an explicit "
        "numeric --floor N if a particular slice must survive. Unset lets a "
        "`floor:` in --config "
        "apply, else defaults to auto. See `gsplat fit --help`.",
    ),
    seeds: Optional[str] = typer.Option(
        None,
        "--seeds",
        help="Seed count or ratio. Under uniform tiling an integer is a "
        "WHOLE-VOLUME budget per (t, c) volume: each task divides it across "
        "that volume's tiles rather than fitting the full count per tile. "
        "Under --tiling content it is ignored (per-box budgets come from the "
        "density plan).",
    ),
    iters: Optional[int] = typer.Option(
        None, "--iters", "-n", help="Max optimization iterations (overrides preset)"
    ),
    # Progressive fitting
    batch_progressive: bool = typer.Option(
        False,
        "--progressive",
        help=BATCH_PROGRESSIVE_HELP
        + " Combine with --parallel for better GPU utilization.",
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
        "When omitted, each tile uses its fitting preset's default, which is "
        "0.999 (near-lossless) for every preset. Set to 0 to keep every "
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
        "Maps to --array=0-N%MAX. No limit if omitted.",
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
    gpus_per_task: int = typer.Option(
        1,
        "--gpus-per-task",
        help="GPUs per task (becomes `#SBATCH --gpus-per-task`). Distinct from `batch-fit run --gpus`, which SELECTS local devices.",
        rich_help_panel="Slurm resources",
    ),
    cpus: int = typer.Option(
        4,
        "--cpus",
        help="CPUs per fit task; --parallel multiplies the Slurm request by resolved packing",
        rich_help_panel="Slurm resources",
    ),
    mem: int = typer.Option(
        32,
        "--mem",
        help="Memory per fit task (GB); --parallel multiplies the Slurm request by resolved packing",
        rich_help_panel="Slurm resources",
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
        "--merge-add-method",
        help="Additive ladder method for --merge-recipe stream "
        f"({GSPLAT_ADDITIVE_CHOICES_HELP}). Default auto.",
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
    merge_refine: Optional[str] = typer.Option(
        None,
        "--merge-refine",
        help="[--merge-recipe levels] refine each per-tile coarse level at merge: "
        "none (default) | l2 (against its fine input) | volume (re-open THIS "
        "input and re-fit each tile against its own crop of it — highest "
        "fidelity). 'volume' needs --axes recorded and a single channel.",
        rich_help_panel="Merge LOD",
    ),
    merge_refine_iters: Optional[int] = typer.Option(
        None,
        "--merge-refine-iters",
        help="[--merge-recipe levels] refinement steps per level (default 120 "
        "for l2, 300 for volume).",
        rich_help_panel="Merge LOD",
    ),
    merge_substitutive_method: Optional[str] = typer.Option(
        None,
        "--merge-subst-method",
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
            "processes with per-worker GPU/thread/quality-budget isolation and "
            "scales the CPU/RAM allocation by the resolved packing."
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

    tiling = validate_tiling_arg(tiling)

    try:
        from luxar.cli.gsplat_config import PRESETS
        from luxar.cli.gsplat_ops.batch.planning import plan_batch
        from luxar.gsplats.batch.env_capture import is_slurm_mps_available

        # 1. Load GPU profile (required only for auto tile-size)
        axes_list = [a.strip() for a in axes.split(",")] if axes else None
        gpu = resolve_gpu_context(gpu_name_opt, gpu_mem, tile_size, tiling, partition)
        peak = gpu.peak
        resolved_gpu = gpu.resolved_gpu
        max_shape = gpu.max_shape
        throughput_table = gpu.throughput_table

        # 2-6. Discover + decompose + build the manifest/jobs. This whole half is
        # shared verbatim with `batch-fit run` (the local runner) via plan_batch.
        cfgs = build_plan_configs(
            preset=preset,
            seeds=seeds,
            iters=iters,
            config=config,
            floor=floor,
            physical=physical,
            batch_progressive=batch_progressive,
            batch_splats_per_pass=batch_splats_per_pass,
            batch_psnr_patience=batch_psnr_patience,
            batch_max_passes=batch_max_passes,
            batch_cull_retention=batch_cull_retention,
            batch_denoise=batch_denoise,
            batch_denoise_h=batch_denoise_h,
            batch_denoise_2d=batch_denoise_2d,
            batch_denoise_patch_size=batch_denoise_patch_size,
            batch_denoise_search_distance=batch_denoise_search_distance,
            batch_denoise_backend=batch_denoise_backend,
            batch_calibration_samples=batch_calibration_samples,
            batch_preprocess=batch_preprocess,
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
            merge_recipe=merge_recipe,
            channel_colors=channel_colors,
            merge_n_lods=merge_n_lods,
            merge_additive_method=merge_additive_method,
            merge_breakpoints=merge_breakpoints,
            merge_target_ms=merge_target_ms,
            merge_bandwidth_mbps=merge_bandwidth_mbps,
            merge_bytes_per_splat=merge_bytes_per_splat,
            merge_compression_factor=merge_compression_factor,
            merge_levels=merge_levels,
            merge_refine=merge_refine,
            merge_refine_iters=merge_refine_iters,
            merge_substitutive_method=merge_substitutive_method,
            merge_coarsen_dims=merge_coarsen_dims,
        )
        fit_cfg = cfgs.fit
        denoise_cfg = cfgs.denoise
        content_cfg = cfgs.content
        merge_cfg = cfgs.merge

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
        packing = resolve_packing(
            tasks_per_job,
            parallel=parallel,
            max_shape=max_shape,
            tile_voxels=tile_voxels,
            est_seconds=est_seconds,
            total_tasks=total_tasks,
            time_limit=time_limit,
            partition=partition,
            gpus_per_task=gpus_per_task,
            cpus=cpus,
            mem=mem,
        )
        tasks_per_job = packing.tasks_per_job
        uses_backfill = packing.uses_backfill
        no_job_limit = packing.no_job_limit
        n_slurm_jobs = packing.n_slurm_jobs
        est_seconds_per_job = packing.est_seconds_per_job
        slurm_time = packing.slurm_time
        total_gpu_hours = packing.total_gpu_hours

        # 6. Build manifest — fit_args, denoise mode, channel colors, and the
        # per-part merge-recipe args were all resolved inside plan_batch above;
        # the manifest is plan.manifest. We only add the Slurm-specific fields
        # (partition, packing, preemptible) here.
        preempt_partition = stamp_slurm_fields(
            manifest,
            packing=packing,
            partition=partition,
            account=account,
            qos=qos,
            gpus_per_task=gpus_per_task,
            parallel=parallel,
            max_concurrent=max_concurrent,
            preemptible=preemptible,
            preemptible_partition_opt=preemptible_partition_opt,
            preemptible_concurrent=preemptible_concurrent,
        )

        # 7. Capture environment + generate scripts
        scripts = generate_all_scripts(
            manifest,
            preempt_partition=preempt_partition,
            batch_denoise=batch_denoise,
            batch_denoise_h=batch_denoise_h,
            denoise_mode=denoise_mode,
        )

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
            packing_limit=packing.limiting_resource,
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
            gpus_per_task=gpus_per_task,
            cpus_per_task=packing.slurm_cpus,
            mem_gb_per_task=packing.slurm_mem_gb,
            cpus_total=packing.slurm_cpus_total,
            mem_gb_total=packing.slurm_mem_gb_total,
            output_dir=output_dir,
        )

        if dry_run:
            aprint("Dry run -- omit --dry-run to actually submit.")
            raise typer.Exit(0)

        # 9. Submit
        submit_batch_jobs(
            output_dir=output_dir,
            manifest=manifest,
            fit_script=scripts.fit_script,
            merge_script=scripts.merge_script,
            preamble=scripts.preamble,
            calibrate_script=scripts.calibrate_script,
            denoise_script=scripts.denoise_script,
            floor_script=scripts.floor_script,
            preempt_fit_script=scripts.preempt_fit_script,
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
        raise typer.Exit(1) from e
