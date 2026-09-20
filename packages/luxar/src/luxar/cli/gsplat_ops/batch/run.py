"""Implementation helper for ``batch-fit run`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from arbol import aprint

from .help_text import BATCH_PROGRESSIVE_HELP
from .plan_configs import build_plan_configs
from .run_orchestration import run_batch_local_orchestration


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
        0.44,
        "--saturation-exponent",
        help="Sub-linear exponent for content budgets and uniform tile weights.",
        rich_help_panel="Content-aware tiling",
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
    batch_progressive: bool = typer.Option(
        False,
        "--progressive",
        help=BATCH_PROGRESSIVE_HELP,
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
    batch_calibration_samples: int = typer.Option(
        5,
        "--calibration-samples",
        help="Timepoints to sample for h calibration",
        rich_help_panel="Denoising",
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
        help="Concurrent fit workers per GPU. 'auto' shares GPU memory, host RAM, "
        "available CPU threads, and a configurable host-wide hard cap; an integer "
        "applies uniformly. Use 1 to be safe on small cards.",
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
        None, "--merge-add-method", rich_help_panel="Merge LOD"
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
        None, "--merge-subst-method", rich_help_panel="Merge LOD"
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
    CUDA_VISIBLE_DEVICES, concurrency bounded by GPU memory and shared host
    resources), then runs the memory-safe streaming merge to a single kind=partition
    .gsplats.zarr. Resumable — re-running skips tiles already on disk.

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
        # ONE flat-flags -> config-objects mapping, shared verbatim with
        # `batch-fit submit`. Inlining a second copy here is what let
        # --calibration-samples exist on submit and silently not on run.
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
            # `--preprocess` is submit-only by design: writing denoised.zarr up
            # front is a separate dependent Slurm job, and the local runner
            # denoises per tile on the fly. planning.py's own error text points
            # users at `batch-fit submit` for it.
            batch_preprocess=False,
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

        run_batch_local_orchestration(
            input_path=input_path,
            output_dir=output_dir,
            tiling=tiling,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            axes=axes,
            array_key=array_key,
            timepoints_slice=timepoints_slice,
            channels_slice=channels_slice,
            cfgs=cfgs,
            gpus=gpus,
            jobs_per_gpu=jobs_per_gpu,
            no_resume=no_resume,
            dry_run=dry_run,
        )

    except typer.Exit:
        raise
    except typer.BadParameter:
        # User-input error: render cleanly (no traceback).
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e
