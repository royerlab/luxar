"""Implementation helper for gsplat fit command."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import typer
from arbol import aprint, asection

from .fitting_fit_utils import (
    FitPipelineCtx,
    assemble_fit_config,
    dispatch_parallel_tiled,
    fit_progressive,
    fit_sequential_tiled,
    fit_single_tile,
    maybe_denoise_full_volume,
    rescale_and_save,
    resolve_denoise_h,
    validate_and_build_recipe,
    warn_ignored_density_flags,
)
from .fitting_fit_utils import (
    resolve_tiling as _resolve_tiling_impl,
)


def run_fit_volume(
    input_path: Optional[Path] = typer.Argument(
        None, help="Input volume (.npy/.npz/.tiff/.zarr)"
    ),
    output_path: Optional[Path] = typer.Argument(
        None, help="Output .gsplats.zarr path"
    ),
    # Common flags
    seeds: Optional[str] = typer.Option(
        None,
        "--seeds",
        "-s",
        help="Seed count (int), compression ratio (float 0-1), or 'auto'",
    ),
    iters: Optional[int] = typer.Option(
        None, "--iters", "-n", help="Max optimization iterations"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-d", help="Device: auto/cpu/cuda/mps"
    ),
    preset: Optional[str] = typer.Option(
        None, "--preset", help="Parameter preset: draft/standard/hifi/ultra/n2s"
    ),
    loss: Optional[str] = typer.Option(
        None, "--loss", help="Loss function: l1/mse/poisson"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", help="YAML config file for full parameter control"
    ),
    dump_config: bool = typer.Option(
        False, "--dump-config", help="Print default YAML config and exit"
    ),
    compress: Optional[Literal["zip", "tar.gz"]] = typer.Option(
        None, "--compress", "-c", help="Compress output archive"
    ),
    # Input selection for multi-array formats
    channel: Optional[int] = typer.Option(
        None,
        "--channel",
        help="Channel index for 5D OME-ZARR",
        rich_help_panel="Input selection",
    ),
    timepoint: Optional[int] = typer.Option(
        None,
        "--timepoint",
        help="Timepoint index for 5D OME-ZARR",
        rich_help_panel="Input selection",
    ),
    array_key: Optional[str] = typer.Option(
        None,
        "--array-key",
        help="Array key within .npz or .zarr",
        rich_help_panel="Input selection",
    ),
    axes: Optional[str] = typer.Option(
        None,
        "--axes",
        help="Per-dimension axis labels overriding the positional "
        "TCZYX/CZYX/ZYX heuristic, e.g. 'z,c,y,x' or 't,z,y,x'. Use when your "
        "data's axis order differs. Time/channel axes are sliced (by "
        "--timepoint/--channel) and dropped; spatial axes kept in the given order.",
        rich_help_panel="Input selection",
    ),
    # Frequently used fit params
    lr: Optional[float] = typer.Option(None, "--lr", help="Learning rate"),
    floor: Optional[str] = typer.Option(
        None,
        "--floor",
        help="Background floor / DC-offset suppression before normalization "
        "(default: auto). auto = histogram-mode estimate (capped at median; "
        "no-op on clean data) | pN = Nth percentile (e.g. p10) | <float> = "
        "fixed value | none = disable (hard-min normalization). Unset lets a "
        "`floor:` in --config/preset apply, else defaults to auto.",
    ),
    seed_method: Optional[str] = typer.Option(
        None, "--seed-method", help="Seed generation method"
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Verbose output"),
    # Downscaling
    downscale: Optional[str] = typer.Option(
        None,
        "--downscale",
        help="Downsample volume by integer factor before fitting. "
        "Single value (e.g., '4') or per-axis comma-separated (e.g., '1,4,4'). "
        "Anti-alias Gaussian blur applied before decimation.",
    ),
    # Tiled fitting
    tiling: str = typer.Option(
        "auto",
        "--tiling",
        help="Decomposition: auto | none | uniform | content. auto = whole "
        "volume if it fits one tile, else uniform (or content when a density "
        "--cal/--k-star-ref is given). Replaces the old --tiled.",
        rich_help_panel="Tiling",
    ),
    flat: bool = typer.Option(
        False,
        "--flat",
        help="Tiled fits emit a kind=partition (one part per tile/box) by "
        "default for viewer frustum culling; --flat merges to a single leaf.",
        rich_help_panel="Tiling",
    ),
    tile_size: int = typer.Option(
        256,
        "--tile-size",
        help="Tile size in voxels (per axis)",
        rich_help_panel="Tiling",
    ),
    tile_overlap: int = typer.Option(
        32,
        "--overlap",
        help="Overlap between tiles in voxels",
        rich_help_panel="Tiling",
    ),
    tile: Optional[str] = typer.Option(
        None,
        "--tile",
        help="Fit single tile N/M (e.g., '3/16' = tile index 3 of 16 total)",
        rich_help_panel="Tiling",
    ),
    jobs: str = typer.Option(
        "1",
        "--jobs",
        "-j",
        help="With --tiling uniform/content: number of tiles/boxes to fit "
        "concurrently as subprocesses "
        "on one GPU (int, or 'auto' to size from free VRAM). Default 1 = "
        "sequential. Ignored with --tiling none or --tile.",
        rich_help_panel="Tiling",
    ),
    keep_tiles: bool = typer.Option(
        False,
        "--keep-tiles",
        help="With --tiling --jobs>1: keep the per-tile/box temporary .gsplats.zarr "
        "outputs (and any .empty markers for skipped tiles) instead of "
        "deleting them after the merge.",
        rich_help_panel="Tiling",
    ),
    allow_empty_tile: bool = typer.Option(
        False,
        "--allow-empty-tile",
        hidden=True,
        help="Single-tile mode only: if the tile has no signal (0 splats), "
        "write an empty marker and exit 0 instead of erroring. Used internally "
        "by parallel --tiling uniform --jobs so an empty tile is skipped at merge.",
    ),
    # Per-part LOD (tiled fits only): give each tile/box-part its own LOD ladder
    # at fit time instead of a separate `gsplat lod` pass (which rejects a
    # partition). stream -> tiles topology; levels -> adaptive.
    recipe: Optional[str] = typer.Option(
        None,
        "-r",
        "--recipe",
        help="Per-part LOD for a tiled partition: stream (each part a "
        "prefix-sum ladder -> 'tiles' topology) or levels (each part its own "
        "coarse<->fine lod group -> 'adaptive'). Requires a tiled fit "
        "(--tiling uniform/content) and a partition output (not --flat).",
        rich_help_panel="Per-part LOD",
    ),
    recipe_n_lods: Optional[int] = typer.Option(
        None,
        "--n-lods",
        help="[--recipe stream] Number of additive sub-LODs per part (default 4).",
        rich_help_panel="Per-part LOD",
    ),
    recipe_additive_method: Optional[str] = typer.Option(
        None,
        "-m",
        "--additive-method",
        help="[--recipe stream] auto (default: greedy at small N, "
        "self_energy for large parts) | greedy ((1-1/e)-optimal) | "
        "self_energy (cheap O(N log N) for very large parts).",
        rich_help_panel="Per-part LOD",
    ),
    recipe_breakpoints: Optional[str] = typer.Option(
        None,
        "-b",
        "--breakpoints",
        help="[--recipe stream] additive ladder breakpoints: 'equal-count' "
        "(default), 'stream:C' (geometric streaming ladder, sized per part), "
        "'counts:500,2000,...' or 'energy:0.5,0.9,...'.",
        rich_help_panel="Per-part LOD",
    ),
    recipe_target_ms: Optional[float] = typer.Option(
        None,
        "--target-ms",
        min=1.0,
        help="[--recipe stream] streaming sizing: derive 'stream:<c>' "
        "breakpoints so each part's first additive chunk downloads in ~this "
        "many ms at --bandwidth-mbps (analytic bytes/splat estimate; override "
        "with --bytes-per-splat). Mutually exclusive with --breakpoints.",
        rich_help_panel="Per-part LOD",
    ),
    recipe_bandwidth_mbps: Optional[float] = typer.Option(
        None,
        "--bandwidth-mbps",
        min=0.1,
        help="[--recipe stream] assumed downlink for --target-ms sizing "
        "(default 25, a typical broadband connection).",
        rich_help_panel="Per-part LOD",
    ),
    recipe_bytes_per_splat: Optional[float] = typer.Option(
        None,
        "--bytes-per-splat",
        min=0.1,
        help="[--recipe stream] override the on-wire bytes/splat used by "
        "--target-ms sizing (default: analytic estimate).",
        rich_help_panel="Per-part LOD",
    ),
    recipe_compression_factor: Optional[int] = typer.Option(
        None,
        "-K",
        "--compression-factor",
        help="[--recipe levels] per-level coarsening factor K (default 4).",
        rich_help_panel="Per-part LOD",
    ),
    recipe_levels: Optional[int] = typer.Option(
        None,
        "-L",
        "--levels",
        help="[--recipe levels] number of substitutive levels L (default 3).",
        rich_help_panel="Per-part LOD",
    ),
    recipe_substitutive_method: Optional[str] = typer.Option(
        None,
        "--substitutive-method",
        help="[--recipe levels] auto (default) / kmeans-lloyd / greedy / greedy-lloyd.",
        rich_help_panel="Per-part LOD",
    ),
    recipe_coarsen_dims: Optional[str] = typer.Option(
        None,
        "--coarsen-dims",
        help="[--recipe levels] comma-separated center-column indices "
        "coarsening may merge over; the rest become hard barriers (default: all "
        "spatial dims).",
        rich_help_panel="Per-part LOD",
    ),
    # Content-aware tiling (--tiling content): transferable density + planner knobs
    cal: Optional[Path] = typer.Option(
        None,
        "--cal",
        help="Calibration JSON (gsplat cal) supplying the splats-per-feature density.",
        rich_help_panel="Content-aware tiling",
    ),
    k_star_ref: Optional[int] = typer.Option(
        None,
        "--k-star-ref",
        help="Reference K* (effective splats) instead of --cal.",
        rich_help_panel="Content-aware tiling",
    ),
    n_features_ref: Optional[int] = typer.Option(
        None,
        "--n-features-ref",
        help="Reference feature count for the density.",
        rich_help_panel="Content-aware tiling",
    ),
    saturation_exponent: float = typer.Option(
        0.44,
        "--saturation-exponent",
        help="Sub-linear exponent alpha (K~feat^alpha).",
        rich_help_panel="Content-aware tiling",
    ),
    saturation_cap: Optional[int] = typer.Option(
        None,
        "--saturation-cap",
        help="Per-box splat cap (default 4x k_star_ref).",
        rich_help_panel="Content-aware tiling",
    ),
    feature_threshold: Optional[float] = typer.Option(
        None,
        "--feature-threshold",
        help="Absolute feature-count threshold (taken from --cal automatically).",
        rich_help_panel="Content-aware tiling",
    ),
    feature_metric: Optional[str] = typer.Option(
        None,
        "--feature-metric",
        help="Content metric: peaks | edges | intensity.",
        rich_help_panel="Content-aware tiling",
    ),
    cell: int = typer.Option(
        16,
        "--cell",
        help="Content-scan cell size (voxels).",
        rich_help_panel="Content-aware tiling",
    ),
    target_features: Optional[int] = typer.Option(
        None,
        "--target-features",
        help="Features per content-box to split toward.",
        rich_help_panel="Content-aware tiling",
    ),
    min_leaf: int = typer.Option(
        256,
        "--min-leaf",
        help="Minimum content-box edge (voxels).",
        rich_help_panel="Content-aware tiling",
    ),
    max_leaf: int = typer.Option(
        512,
        "--max-leaf",
        help="Maximum content-box edge (voxels).",
        rich_help_panel="Content-aware tiling",
    ),
    # Plan I/O (content tiling)
    plan: Optional[Path] = typer.Option(
        None,
        "--plan",
        help="Fit a precomputed FitPlan JSON (skip scan/plan).",
        rich_help_panel="Plan I/O",
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only",
        help="With --tiling content: write the FitPlan JSON to OUTPUT and stop.",
        rich_help_panel="Plan I/O",
    ),
    plan_box: Optional[int] = typer.Option(
        None,
        "--plan-box",
        hidden=True,
        help="Internal worker: fit only box i of --plan (used by content -j).",
    ),
    # Progressive fitting
    progressive: bool = typer.Option(
        False,
        "--progressive",
        help="Enable progressive fitting: fit in multiple passes on residuals, "
        "producing a multi-LOD result. Each pass adds detail to the previous. "
        "Tip: for tiled batch jobs, combine with --parallel to improve GPU utilization.",
        rich_help_panel="Progressive fitting",
    ),
    max_splats_per_pass: int = typer.Option(
        5000,
        "--splats-per-pass",
        help="Maximum splats per progressive pass (actual may be fewer after culling)",
        rich_help_panel="Progressive fitting",
    ),
    psnr_patience: float = typer.Option(
        0.5,
        "--psnr-patience",
        help="Stop progressive fitting if ΔPSNR between passes < this value (dB)",
        rich_help_panel="Progressive fitting",
    ),
    max_passes: Optional[int] = typer.Option(
        None,
        "--max-passes",
        help="Maximum number of progressive passes (default: unlimited, stops by budget or PSNR patience)",
        rich_help_panel="Progressive fitting",
    ),
    # Post-fit culling
    cull_retention: Optional[float] = typer.Option(
        None,
        "--cull-retention",
        help="After fitting, remove the weakest splats that collectively "
        "contribute less than (1 - value) of the total amplitude. "
        "For example, 0.95 (the default) discards splats in the bottom 5%% "
        "of cumulative amplitude — typically removing 10-30%% of splats with "
        "negligible quality loss. Set to 0 to keep every splat.",
    ),
    # Denoising
    denoise: bool = typer.Option(
        False,
        "--denoise",
        help="Denoise volume before fitting (NLM)",
        rich_help_panel="Denoising",
    ),
    denoise_h: Optional[float] = typer.Option(
        None,
        "--denoise-h",
        help="Manual NLM h value (skip auto-calibration)",
        rich_help_panel="Denoising",
    ),
    denoise_2d: bool = typer.Option(
        False,
        "--denoise-2d",
        help="Use 2D NLM (slice-by-slice) instead of 3D",
        rich_help_panel="Denoising",
    ),
    denoise_patch_size: int = typer.Option(
        3,
        "--denoise-patch-size",
        help="NLM patch size (odd integer)",
        rich_help_panel="Denoising",
    ),
    denoise_search_distance: int = typer.Option(
        5,
        "--denoise-search-distance",
        help="NLM search window half-size",
        rich_help_panel="Denoising",
    ),
    denoise_backend: str = typer.Option(
        "auto",
        "--denoise-backend",
        help="NLM backend: auto/cuda/pytorch/skimage",
        rich_help_panel="Denoising",
    ),
) -> None:
    """Fit Gaussian splats to a volume.

    Reconstructs an n-dimensional image/volume as a set of oriented Gaussian
    splats. Use presets for quick configuration or a YAML config file for
    full control over all ~35 parameters.

    Presets:
        draft    - Fast preview (2000 iters)
        standard - Balanced quality/speed (5000 iters)
        hifi     - High quality (10000 iters)
        ultra    - Maximum quality (20000 iters)

    Examples:
        luxar gsplat fit volume.npy splats.gsplats.zarr --preset draft --seeds 1000

        luxar gsplat fit volume.tiff splats.gsplats.zarr --preset standard --seeds 8000

        luxar gsplat fit --dump-config --preset hifi > config.yaml
        luxar gsplat fit volume.zarr splats.gsplats.zarr --config config.yaml

        luxar gsplat fit data.zarr splats.gsplats.zarr --channel 1 --timepoint 0

        luxar gsplat fit large.zarr splats.gsplats.zarr --tiling uniform --tile-size 256 --overlap 32

        luxar gsplat fit large.zarr tile_3.gsplats.zarr --tile 3/16 --tile-size 256 --overlap 32

        luxar gsplat fit volume.tiff splats.gsplats.zarr --progressive --seeds 10000

        luxar gsplat fit volume.tiff splats.gsplats.zarr --progressive --seeds 50000 --splats-per-pass 5000 --psnr-patience 0.3
    """
    from luxar.cli.gsplat_config import (
        dump_default_config,
        load_volume,
    )

    # Handle --dump-config: print and exit (no input/output needed)
    if dump_config:
        typer.echo(dump_default_config(preset or "standard"))
        raise typer.Exit(0)

    # Validate required args (optional for --dump-config)
    if input_path is None:
        aprint("Error: Missing argument 'INPUT_PATH'")
        raise typer.Exit(1)
    if output_path is None:
        aprint("Error: Missing argument 'OUTPUT_PATH'")
        raise typer.Exit(1)
    if not input_path.exists():
        aprint(f"Error: Input file not found: {input_path}")
        raise typer.Exit(1)

    try:
        from luxar.gsplats import fit_gaussian_splats

        with asection(f"Fitting Gaussian Splats: {input_path.name}"):
            # 1. Load volume
            with asection("Loading volume"):
                volume = load_volume(
                    input_path, channel, timepoint, array_key, axes=axes
                )
                aprint(f"Volume shape: {volume.shape}")

            # 1a. Resolve the decomposition. `--tiling auto` → none (fits one
            # tile) / uniform / content (when a density is supplied). `content`
            # folds in the former `gsplat plan`; the rest drive the uniform
            # branches below via the local `tiled` flag (the old --tiled bool).
            _has_density = (
                cal is not None
                or k_star_ref is not None
                or plan is not None
                or plan_box is not None
            )
            resolved_tiling = _resolve_tiling_impl(
                tiling, volume.shape, tile_size, _has_density
            )

            # Pipeline ctx: the parameter state the extracted helpers consume.
            ctx = FitPipelineCtx(
                input_path=input_path,
                output_path=output_path,
                seeds=seeds,
                iters=iters,
                device=device,
                preset=preset,
                loss=loss,
                config=config,
                compress=compress,
                channel=channel,
                timepoint=timepoint,
                array_key=array_key,
                axes=axes,
                lr=lr,
                floor=floor,
                seed_method=seed_method,
                verbose=verbose,
                downscale=downscale,
                resolved_tiling=resolved_tiling,
                flat=flat,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                tile=tile,
                jobs=jobs,
                keep_tiles=keep_tiles,
                allow_empty_tile=allow_empty_tile,
                recipe=recipe,
                recipe_n_lods=recipe_n_lods,
                recipe_additive_method=recipe_additive_method,
                recipe_breakpoints=recipe_breakpoints,
                recipe_target_ms=recipe_target_ms,
                recipe_bandwidth_mbps=recipe_bandwidth_mbps,
                recipe_bytes_per_splat=recipe_bytes_per_splat,
                recipe_compression_factor=recipe_compression_factor,
                recipe_levels=recipe_levels,
                recipe_substitutive_method=recipe_substitutive_method,
                recipe_coarsen_dims=recipe_coarsen_dims,
                cal=cal,
                k_star_ref=k_star_ref,
                n_features_ref=n_features_ref,
                feature_threshold=feature_threshold,
                feature_metric=feature_metric,
                target_features=target_features,
                plan_only=plan_only,
                plan_box=plan_box,
                progressive=progressive,
                max_splats_per_pass=max_splats_per_pass,
                psnr_patience=psnr_patience,
                max_passes=max_passes,
                cull_retention=cull_retention,
                denoise=denoise,
                denoise_h=denoise_h,
                denoise_2d=denoise_2d,
                denoise_patch_size=denoise_patch_size,
                denoise_search_distance=denoise_search_distance,
                denoise_backend=denoise_backend,
            )

            warn_ignored_density_flags(ctx)

            # Per-part LOD recipe (tiled partition only): validate + build params.
            recipe_params: "Any" = validate_and_build_recipe(ctx, volume.ndim)

            if resolved_tiling == "content":
                from luxar.cli.gsplat_ops.planner import run_content_fit

                # Flags the content path does not implement — warn loudly rather
                # than silently ignore (the fit knobs below ARE honored).
                _unsupported = [
                    name
                    for name, on in (
                        ("--denoise", denoise),
                        ("--downscale", downscale is not None),
                        ("--progressive", progressive),
                    )
                    if on
                ]
                if _unsupported and plan_box is None:
                    aprint(
                        f"⚠ {', '.join(_unsupported)} are not supported with "
                        "--tiling content and are ignored."
                    )
                # --seeds is superseded by the content plan (per-box budgets from
                # the density), not unsupported — note it so the user isn't
                # surprised the explicit count had no effect.
                if seeds is not None and plan_box is None:
                    aprint(
                        "⚠ --seeds is ignored with --tiling content; per-box "
                        "budgets come from the density plan (use --cal / "
                        "--k-star-ref to size them)."
                    )

                run_content_fit(
                    input_path,
                    output_path,
                    volume=volume,
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
                    overlap=tile_overlap,
                    preset=preset,
                    config=config,
                    iters=iters,
                    loss=loss,
                    lr=lr,
                    floor=floor,
                    cull_retention=cull_retention,
                    device=device,
                    jobs=jobs,
                    keep_boxes=keep_tiles,
                    flat=flat,
                    recipe=recipe,
                    recipe_params=recipe_params,
                    compress=compress,
                    plan=plan,
                    plan_only=plan_only,
                    plan_box=plan_box,
                    channel=channel,
                    timepoint=timepoint,
                    array_key=array_key,
                    axes=axes,
                    verbose=verbose,
                )
                raise typer.Exit(0)
            tiled = resolved_tiling == "uniform"

            # 1b. Denoise (if requested)
            # Resolve effective_h (calibrate if needed), then either:
            # - Denoise full volume now (non-tiled fitting)
            # - Pass h + params through to fit_tile (tiled fitting, per-tile denoise)
            ctx.denoise_effective_h = resolve_denoise_h(ctx, volume)

            # For non-tiled paths, denoise the full volume now.
            # For tiled paths, denoise is deferred to per-tile (see fit_tile).
            is_tiled = (tile is not None) or tiled
            volume = maybe_denoise_full_volume(ctx, volume, is_tiled)

            # 2-5. Merged config + parsed seeds + effective downscale
            fit_config, parsed_seeds, effective_downscale = assemble_fit_config(
                ctx, is_tiled
            )

            # 5b. Parallel tiled fitting: spawn one subprocess per tile (branch
            # BEFORE the in-memory downscale below — see dispatch_parallel_tiled).
            if dispatch_parallel_tiled(
                ctx, volume, fit_config, effective_downscale, recipe_params
            ):
                raise typer.Exit(0)

            # For tiled modes, downscale the volume before tiling
            tiled_downscale_factors = None
            if effective_downscale is not None and (tile is not None or tiled):
                from luxar.gsplats.fitting.downscale import (
                    downscale_volume,
                    normalize_downscale,
                )

                tiled_downscale_factors = normalize_downscale(
                    effective_downscale, volume.ndim
                )
                if tiled_downscale_factors is not None:
                    original_shape = volume.shape
                    volume = downscale_volume(volume, tiled_downscale_factors)
                    aprint(
                        f"Downscaled volume: {original_shape} -> {volume.shape} "
                        f"(factors={tiled_downscale_factors})"
                    )

            # 6. Fit
            if tile is not None:
                # Single-tile mode (Slurm-ready)
                result = fit_single_tile(ctx, volume, fit_config, parsed_seeds)

            elif tiled:
                # Full tiled fitting
                result = fit_sequential_tiled(
                    ctx,
                    volume,
                    fit_config,
                    parsed_seeds,
                    tiled_downscale_factors,
                    recipe_params,
                )

            elif progressive:
                # Progressive fitting: multiple passes on residuals
                result = fit_progressive(ctx, volume, fit_config, parsed_seeds)

            else:
                # Standard fitting (downscale handled inside fit_gaussian_splats)
                with asection("Optimization"):
                    result = fit_gaussian_splats(
                        volume,
                        seeds=parsed_seeds,
                        downscale=effective_downscale,
                        **fit_config,
                    )

            # 7. Rescale (if downscaled tiled) + save
            result, n_splats, is_leaf = rescale_and_save(
                ctx, result, tiled_downscale_factors
            )

        time_s = result.stats.get("time_seconds", 0) if is_leaf else 0
        aprint(f"\nDone: {n_splats:,} splats in {time_s:.1f}s")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
