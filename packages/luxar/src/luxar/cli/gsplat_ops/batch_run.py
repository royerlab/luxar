"""Implementation helper for ``batch-fit run`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from arbol import aprint

from luxar.cli.gsplat_ops.batch_recipe_args import (
    build_merge_recipe_params as _build_merge_recipe_params_impl,
)


def run_batch_run(
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
