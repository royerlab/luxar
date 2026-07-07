"""Orchestration helper for local ``batch-fit run`` execution."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from arbol import aprint


def run_batch_local_orchestration(
    *,
    input_path: Path,
    output_dir: Path,
    tiling: str,
    tile_size: Optional[int],
    tile_overlap: int,
    axes: Optional[str],
    array_key: Optional[str],
    timepoints_slice: Optional[str],
    channels_slice: Optional[str],
    preset: str,
    config: Optional[Path],
    seeds: Optional[str],
    iters: Optional[int],
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
    gpus: str,
    jobs_per_gpu: str,
    no_resume: bool,
    dry_run: bool,
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
) -> None:
    """Build a local batch plan, print summary, and execute local workers."""
    from luxar.cli.gsplat_config import parse_hex_color
    from luxar.cli.gsplat_ops.batch_planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )
    from luxar.cli.gsplat_ops.batch_recipe_args import (
        build_merge_recipe_params as _build_merge_recipe_params_impl,
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
        recipe_params = _build_merge_recipe_params_impl(manifest.merge_recipe_args)

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
