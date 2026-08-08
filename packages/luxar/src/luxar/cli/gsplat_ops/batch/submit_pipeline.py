"""Pipeline helpers for ``batch-fit submit`` (extracted from ``submit``).

Each helper owns one stage of the submit pipeline — tiling validation, GPU
profile resolution, plan-config assembly, tasks-per-job packing, Slurm-field
stamping, and sbatch script generation — so ``run_batch_submit`` reads as the
stage sequence plus the plan/print/submit tail it keeps in its body. Sibling
of the four existing ``submit_*`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import typer
from arbol import aprint

from .submit_packing import resolve_tasks_per_job
from .submit_preemptible import resolve_preemptible_partition


def validate_tiling_arg(tiling: str) -> str:
    """Normalize + validate ``--tiling`` up front.

    Mirrors ``gsplat fit``'s ``_resolve_tiling`` so a typo fails loudly
    instead of silently submitting a large uniform array in the wrong mode.
    """
    tiling = tiling.lower()
    if tiling not in ("uniform", "content"):
        aprint(f"Error: --tiling must be uniform|content, got {tiling!r}")
        raise typer.Exit(1)
    return tiling


@dataclass
class GpuContext:
    """GPU-profile-derived sizing inputs for the submit plan.

    ``summary``/``peak`` feed the printed plan; ``max_shape`` and
    ``throughput_table`` drive uniform auto tile-size + ETA inside
    ``plan_batch``; ``resolved_gpu`` is the display/profile-lookup name.
    """

    summary: Optional[dict]
    peak: dict
    resolved_gpu: str
    max_shape: Any
    throughput_table: Any


def resolve_gpu_context(
    gpu_name_opt: Optional[str],
    gpu_mem: Optional[int],
    tile_size: Optional[int],
    tiling: str,
    partition: Optional[str],
) -> GpuContext:
    """Load the GPU benchmark profile (required only for auto tile-size).

    Raises ``typer.Exit(1)`` with remediation options when no profile exists
    and neither ``--tile-size`` nor content tiling sidesteps the requirement.
    """
    from luxar.gsplats.gpu_profile import (
        get_gpu_summary,
        get_gpu_throughput_table,
        load_profiles,
    )

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

    # GPU-profile-derived sizing inputs (uniform auto tile-size + ETA).
    peak_shape = peak.get("shape", [])
    oom = (summary or {}).get("oom_boundaries", {}).get("3d", {})
    max_shape = oom.get("max_successful_shape", peak_shape)
    throughput_table = get_gpu_throughput_table(gpu_name=resolved_gpu)

    return GpuContext(
        summary=summary,
        peak=peak,
        resolved_gpu=resolved_gpu,
        max_shape=max_shape,
        throughput_table=throughput_table,
    )


@dataclass
class PlanConfigs:
    """The four ``plan_batch`` config dataclasses, assembled from CLI flags."""

    fit: Any
    denoise: Any
    content: Any
    merge: Any


def build_plan_configs(
    *,
    preset: str,
    seeds: Optional[str],
    iters: Optional[int],
    config: Optional[Path],
    floor: Optional[str],
    batch_progressive: bool,
    batch_splats_per_pass: Optional[int],
    batch_psnr_patience: Optional[float],
    batch_max_passes: Optional[int],
    batch_cull_retention: Optional[float],
    batch_denoise: bool,
    batch_denoise_h: Optional[float],
    batch_denoise_2d: bool,
    batch_denoise_patch_size: int,
    batch_denoise_search_distance: int,
    batch_denoise_backend: str,
    batch_calibration_samples: int,
    batch_preprocess: Optional[bool],
    cal: Optional[Path],
    k_star_ref: Optional[int],
    n_features_ref: Optional[int],
    saturation_exponent: float,
    saturation_cap: Optional[int],
    feature_threshold: Optional[float],
    feature_metric: Optional[str],
    cell: int,
    target_features: Optional[int],
    min_leaf: int,
    max_leaf: int,
    plan_timepoint: Optional[int],
    plan_samples: int,
    merge_recipe: Optional[str],
    channel_colors: Optional[str],
    merge_n_lods: Optional[int],
    merge_additive_method: Optional[str],
    merge_breakpoints: Optional[str],
    merge_target_ms: Optional[float],
    merge_bandwidth_mbps: Optional[float],
    merge_bytes_per_splat: Optional[float],
    merge_compression_factor: Optional[int],
    merge_levels: Optional[int],
    merge_substitutive_method: Optional[str],
    merge_coarsen_dims: Optional[str],
) -> PlanConfigs:
    """Map the submit CLI flags onto the ``plan_batch`` config dataclasses."""
    from luxar.cli.gsplat_ops.batch.planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
    )

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
    return PlanConfigs(
        fit=fit_cfg, denoise=denoise_cfg, content=content_cfg, merge=merge_cfg
    )


@dataclass
class PackingPlan:
    """Resolved tasks-per-job packing + the derived Slurm sizing figures."""

    tasks_per_job: int
    uses_backfill: bool
    no_job_limit: bool
    n_slurm_jobs: int
    est_seconds_per_job: float
    slurm_time: str
    total_gpu_hours: float


def resolve_packing(
    tasks_per_job: Optional[int],
    *,
    parallel: bool,
    max_shape: Any,
    tile_voxels: int,
    est_seconds: float,
    total_tasks: int,
    time_limit: Optional[str],
) -> PackingPlan:
    """Compute tasks-per-job packing and the derived job-count/time figures.

    When each volume is small relative to GPU capacity, we pack multiple
    fitting tasks sequentially into one Slurm job to reduce scheduling
    overhead (fewer array elements to launch). Queries the scheduler for
    smart packing defaults first.
    """
    import math

    from luxar.gsplats.batch.env_capture import get_slurm_scheduler_info
    from luxar.gsplats.batch.time_estimate import estimate_slurm_time_limit

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

    return PackingPlan(
        tasks_per_job=tasks_per_job,
        uses_backfill=uses_backfill,
        no_job_limit=no_job_limit,
        n_slurm_jobs=n_slurm_jobs,
        est_seconds_per_job=est_seconds_per_job,
        slurm_time=slurm_time,
        total_gpu_hours=total_gpu_hours,
    )


def stamp_slurm_fields(
    manifest: Any,
    *,
    packing: PackingPlan,
    partition: Optional[str],
    account: Optional[str],
    qos: Optional[str],
    gpus: int,
    cpus: int,
    mem: int,
    parallel: bool,
    max_concurrent: Optional[int],
    preemptible: bool,
    preemptible_partition_opt: Optional[str],
    preemptible_concurrent: Optional[int],
) -> Optional[str]:
    """Stamp the Slurm-specific fields onto the planned manifest.

    ``plan_batch`` built the manifest + jobs (dataset/decomposition/fit/merge/
    denoise fields); only the Slurm-specific fields (partition, packing,
    preemptible) are added here. Returns the resolved preemptible partition
    (``None`` when preemptible submission is off).
    """
    # Preemptible partition detection
    preempt_partition = resolve_preemptible_partition(
        preemptible=preemptible,
        preemptible_partition_opt=preemptible_partition_opt,
    )

    manifest.slurm_time_limit = packing.slurm_time
    manifest.slurm_partition = partition
    manifest.slurm_account = account
    manifest.slurm_qos = qos
    manifest.slurm_gpus = gpus
    manifest.slurm_cpus = cpus
    manifest.slurm_mem_gb = mem
    manifest.tasks_per_job = packing.tasks_per_job
    manifest.parallel_tasks_per_job = parallel
    manifest.max_concurrent = max_concurrent
    manifest.preemptible = preempt_partition is not None
    manifest.preemptible_partition = preempt_partition
    manifest.preemptible_max_concurrent = (
        (preemptible_concurrent or max_concurrent) if preempt_partition else None
    )
    return preempt_partition


@dataclass
class Scripts:
    """Environment preamble + the generated sbatch scripts for the batch."""

    preamble: str
    fit_script: str
    merge_script: str
    preempt_fit_script: Optional[str]
    calibrate_script: Optional[str]
    denoise_script: Optional[str]


def generate_all_scripts(
    manifest: Any,
    *,
    preempt_partition: Optional[str],
    batch_denoise: bool,
    batch_denoise_h: Optional[float],
    denoise_mode: Optional[str],
) -> Scripts:
    """Capture the environment and generate every sbatch script the run needs.

    Always the fit + merge scripts; a requeue-enabled preemptible fit script
    when a preemptible partition was resolved; calibrate/denoise scripts when
    ``--denoise`` is on (calibration only when no manual ``h``; a preprocess
    denoise pass only in ``preprocess`` denoise mode).
    """
    from luxar.gsplats.batch.env_capture import (
        capture_environment,
        generate_env_preamble,
    )
    from luxar.gsplats.batch.slurm_gen import (
        generate_fit_sbatch,
        generate_merge_sbatch,
    )

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

    return Scripts(
        preamble=preamble,
        fit_script=fit_script,
        merge_script=merge_script,
        preempt_fit_script=preempt_fit_script,
        calibrate_script=calibrate_script,
        denoise_script=denoise_script,
    )
