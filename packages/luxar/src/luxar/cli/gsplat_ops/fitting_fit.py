"""Implementation helper for gsplat fit command."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import typer
from arbol import aprint, asection

from .fitting_fit_utils import (
    build_fit_recipe_params as _build_fit_recipe_params_impl,
)
from .fitting_fit_utils import (
    resolve_tiling as _resolve_tiling_impl,
)
from .fitting_fit_utils import (
    save_fit_output as _save_fit_output_impl,
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
        None, "--preset", help="Parameter preset: draft/standard/hifi/ultra"
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
    floor: str = typer.Option(
        "auto",
        "--floor",
        help="Background floor / DC-offset suppression before normalization "
        "(on by default). auto = histogram-mode estimate (capped at median; "
        "no-op on clean data) | pN = Nth percentile (e.g. p10) | <float> = "
        "fixed value | none = disable (hard-min normalization).",
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
        load_fit_config,
        load_volume,
        parse_seeds,
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

            # Content density knobs only apply to content tiling — warn if the
            # decomposition didn't resolve to content (e.g. an explicit
            # --tiling uniform/none), so the flags aren't silently no-ops.
            if resolved_tiling != "content" and plan_box is None:
                _density_flags = [
                    name
                    for name, on in (
                        ("--cal", cal is not None),
                        ("--k-star-ref", k_star_ref is not None),
                        ("--n-features-ref", n_features_ref is not None),
                        ("--feature-threshold", feature_threshold is not None),
                        ("--feature-metric", feature_metric is not None),
                        ("--target-features", target_features is not None),
                    )
                    if on
                ]
                if _density_flags:
                    aprint(
                        f"⚠ {', '.join(_density_flags)} apply only to "
                        f"--tiling content; ignored under --tiling {resolved_tiling}."
                    )

            # Per-part LOD recipe (tiled partition only): validate + build params.
            recipe_params: "Any" = None
            if recipe is not None:
                if flat:
                    raise typer.BadParameter(
                        "--recipe needs a partition output; it is incompatible "
                        "with --flat (which merges to a single leaf)."
                    )
                if resolved_tiling == "none":
                    raise typer.BadParameter(
                        "--recipe needs a tiled fit (--tiling uniform/content); a "
                        "whole-volume fit is a single leaf. Run `gsplat lod` on it "
                        "instead."
                    )
                if tile is not None:
                    raise typer.BadParameter(
                        "--recipe is applied when the parts are merged; it cannot "
                        "be combined with single-tile --tile (a worker fits one "
                        "bare leaf)."
                    )
                if plan_only or plan_box is not None:
                    raise typer.BadParameter(
                        "--recipe is incompatible with --plan-only / --plan-box."
                    )
                recipe_params = _build_fit_recipe_params_impl(
                    recipe,
                    n_lods=recipe_n_lods,
                    additive_method=recipe_additive_method,
                    breakpoints=recipe_breakpoints,
                    target_ms=recipe_target_ms,
                    bandwidth_mbps=recipe_bandwidth_mbps,
                    bytes_per_splat=recipe_bytes_per_splat,
                    compression_factor=recipe_compression_factor,
                    levels=recipe_levels,
                    substitutive_method=recipe_substitutive_method,
                    coarsen_dims=recipe_coarsen_dims,
                    device=device,
                    volume_ndim=volume.ndim,
                )
                from luxar.gsplats.lod.recipes import uniform_per_part_lod_warning

                _w = uniform_per_part_lod_warning(resolved_tiling, recipe)
                if _w:
                    aprint(f"⚠ {_w}")

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
            _denoise_effective_h: Optional[float] = None
            if denoise:
                if denoise_h is not None:
                    _denoise_effective_h = denoise_h
                    aprint(f"Denoise: using manual h={_denoise_effective_h:.4f}")
                else:
                    import torch

                    from luxar.gsplats.preprocessing import calibrate_nlm_h
                    from luxar.gsplats.preprocessing.denoise_pipeline import (
                        normalize_volume,
                    )
                    from luxar.gsplats.utils.device import resolve_torch_device

                    with asection("Calibrating NLM h"):
                        norm_vol, _, _ = normalize_volume(volume)
                        t_vol = torch.from_numpy(norm_vol)
                        # Auto-select CUDA > MPS > CPU when --device is omitted.
                        dev = (
                            resolve_torch_device(device)
                            if device
                            else resolve_torch_device()
                        )
                        _denoise_effective_h = calibrate_nlm_h(
                            t_vol,
                            patch_size=denoise_patch_size,
                            search_distance=denoise_search_distance,
                            backend=denoise_backend,
                            device=dev,
                            use_2d_slice=True,
                        )
                        aprint(f"Calibrated h={_denoise_effective_h:.4f}")

            # For non-tiled paths, denoise the full volume now.
            # For tiled paths, denoise is deferred to per-tile (see fit_tile).
            is_tiled = (tile is not None) or tiled
            if denoise and _denoise_effective_h is not None and not is_tiled:
                from luxar.gsplats.preprocessing.denoise_pipeline import (
                    denoise_volume_array,
                )

                with asection("Denoising (NLM)"):
                    volume = denoise_volume_array(
                        volume,
                        h=_denoise_effective_h,
                        patch_size=denoise_patch_size,
                        search_distance=denoise_search_distance,
                        backend=denoise_backend,
                        device=device,
                        use_2d=denoise_2d,
                    )
                    aprint(f"Denoised volume shape: {volume.shape}")

            # 2. Parse downscale option
            parsed_downscale = None
            if downscale is not None:
                ds_parts = [int(x.strip()) for x in downscale.split(",")]
                parsed_downscale = (
                    ds_parts[0] if len(ds_parts) == 1 else tuple(ds_parts)
                )

            # 3. Build merged config
            cli_overrides = {
                "n_iters": iters,
                "device": device,
                "loss_type": loss,
                "lr": lr,
                "floor": floor,
                "seed_method": seed_method,
                "verbose": verbose,
                "cull_retention": cull_retention,
            }
            fit_config = load_fit_config(preset, config, cli_overrides)

            if preset:
                aprint(f"Preset: {preset}")
            if config:
                aprint(f"Config: {config}")
            aprint(f"Iterations: {fit_config.get('n_iters')}")

            # 4. Parse seeds
            parsed_seeds = parse_seeds(seeds)
            if parsed_seeds is not None:
                aprint(f"Seeds: {parsed_seeds}")
            else:
                aprint("Seeds: auto")

            # 4b. Inject per-tile denoise params for tiled fitting
            if denoise and _denoise_effective_h is not None and is_tiled:
                fit_config["_denoise_h"] = _denoise_effective_h
                fit_config["_denoise_params"] = {
                    "patch_size": denoise_patch_size,
                    "search_distance": denoise_search_distance,
                    "backend": denoise_backend,
                    "device": device,
                    "use_2d": denoise_2d,
                }
                aprint(f"Denoise: per-tile on-the-fly (h={_denoise_effective_h:.4f})")

            # 5. Apply downscaling
            # Pop downscale from fit_config to avoid "multiple values" conflict
            # (get_fit_defaults extracts it from the fit_gaussian_splats signature)
            fc_downscale = fit_config.pop("downscale", None)
            # CLI --downscale flag takes priority over YAML/preset config
            effective_downscale = (
                parsed_downscale if parsed_downscale is not None else fc_downscale
            )

            # 5b. Parallel tiled fitting: spawn one subprocess per tile.
            # Branch BEFORE the in-memory downscale below — the parent skips the
            # in-memory downscale (it only needs the shape to compute the grid);
            # each worker re-invokes `fit --tile i/M`, loading and downscaling
            # its own region and rescaling back to original coords, then we
            # reload + merge. (The parent still holds the volume loaded above —
            # only its shape is used here.) When --jobs resolves to 1 (e.g.
            # `-j auto` on a CPU/MPS box, or an explicit `-j 0/1`), fall through
            # to the in-process sequential path instead of spawning a subprocess.
            if tiled and tile is None and jobs != "1":
                import math

                from luxar.gsplats.fit_tiled_parallel import (
                    build_worker_cmd,
                    fit_tiled_parallel,
                    luxar_argv0,
                    resolve_jobs,
                )
                from luxar.gsplats.fitting.downscale import normalize_downscale
                from luxar.gsplats.tiling import compute_tile_specs

                # Compute the tile grid on the POST-downscale shape (shape math
                # only — decimation is volume[::f]) so the parent and workers
                # agree on the tile count M.
                ds_factors = (
                    normalize_downscale(effective_downscale, volume.ndim)
                    if effective_downscale is not None
                    else None
                )
                if ds_factors is not None:
                    grid_shape = tuple(
                        len(range(0, s, f)) for s, f in zip(volume.shape, ds_factors)
                    )
                else:
                    grid_shape = tuple(volume.shape)

                specs = compute_tile_specs(grid_shape, tile_size, tile_overlap)
                n_tiles = len(specs)
                tile_voxels = max((int(math.prod(s.shape)) for s in specs), default=1)

                try:
                    n_jobs = resolve_jobs(
                        jobs,
                        tile_voxels=tile_voxels,
                        num_tiles=n_tiles,
                        device=device,
                    )
                except ValueError:
                    aprint(f"Error: --jobs must be an integer or 'auto', got '{jobs}'")
                    raise typer.Exit(1)

                # Only spawn workers when there is genuine concurrency to gain.
                # Otherwise (n_jobs == 1) fall through to the sequential tiled
                # path below — no subprocess overhead for a single worker.
                if n_jobs > 1:
                    aprint(
                        f"Parallel tiled fitting: {n_tiles} tiles, grid={grid_shape}, "
                        f"{n_jobs} concurrent worker(s)"
                    )

                    # Format downscale for worker argv (scalar or per-axis).
                    ds_arg: Optional[str] = None
                    if effective_downscale is not None:
                        if isinstance(effective_downscale, (list, tuple)):
                            ds_arg = ",".join(str(int(x)) for x in effective_downscale)
                        else:
                            ds_arg = str(int(effective_downscale))

                    argv0 = luxar_argv0()

                    def _worker_cmd(i: int, m: int, out_path: Path) -> list[str]:
                        return build_worker_cmd(
                            argv0,
                            input_path,
                            out_path,
                            i,
                            m,
                            tile_size,
                            tile_overlap,
                            seeds=seeds,
                            iters=iters,
                            device=device,
                            preset=preset,
                            config=config,
                            loss=loss,
                            lr=lr,
                            floor=floor,
                            seed_method=seed_method,
                            downscale=ds_arg,
                            channel=channel,
                            timepoint=timepoint,
                            array_key=array_key,
                            axes=axes,
                            progressive=progressive,
                            max_splats_per_pass=max_splats_per_pass,
                            psnr_patience=psnr_patience,
                            max_passes=max_passes,
                            denoise=denoise,
                            denoise_h=_denoise_effective_h,
                            denoise_patch_size=denoise_patch_size,
                            denoise_search_distance=denoise_search_distance,
                            denoise_backend=denoise_backend,
                            denoise_2d=denoise_2d,
                            # Empty (windowed-to-zero) tiles must not crash the
                            # whole run: the worker writes an .empty marker and
                            # exits 0; the orchestrator skips it at merge.
                            allow_empty_tile=True,
                        )

                    tmp_dir = output_path.parent / f".{output_path.name}.tiles"
                    merge_cull = fit_config.get("cull_retention")

                    with asection("Optimization (parallel tiles)"):
                        result = fit_tiled_parallel(
                            num_tiles=n_tiles,
                            jobs=n_jobs,
                            tmp_dir=tmp_dir,
                            worker_cmd_builder=_worker_cmd,
                            volume_shape=grid_shape,
                            tile_size=tile_size,
                            overlap=tile_overlap,
                            progressive=progressive,
                            cull_retention=merge_cull,
                            verbose=verbose,
                            keep_tiles=keep_tiles,
                            partition=not flat,
                            recipe=recipe,
                            recipe_params=recipe_params,
                        )

                    with asection(f"Saving to {output_path.name}"):
                        n_splats = _save_fit_output_impl(
                            result, output_path, compress=compress, verbose=verbose
                        )

                    aprint(f"\nDone: {n_splats:,} splats")
                    raise typer.Exit(0)

                aprint("--jobs resolved to 1 worker; using sequential tiled fitting")

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
                from luxar.gsplats.fit_tiled_gsplats import fit_tile
                from luxar.gsplats.tiling import compute_tile_specs

                tile_parts = tile.split("/")
                if len(tile_parts) != 2:
                    aprint("Error: --tile must be N/M format (e.g., '3/16')")
                    raise typer.Exit(1)
                try:
                    tile_idx, tile_total = int(tile_parts[0]), int(tile_parts[1])
                except ValueError:
                    aprint("Error: --tile N/M requires integer values")
                    raise typer.Exit(1)

                specs = compute_tile_specs(volume.shape, tile_size, tile_overlap)
                if tile_total != len(specs):
                    aprint(
                        f"Note: --tile specifies {tile_total} tiles but "
                        f"grid has {len(specs)} tiles for this volume. "
                        f"Using actual grid count."
                    )
                if tile_idx < 0 or tile_idx >= len(specs):
                    aprint(
                        f"Error: tile index {tile_idx} out of range [0, {len(specs)})"
                    )
                    raise typer.Exit(1)

                # Extract params that are explicit in fit_tile to avoid
                # "got multiple values" conflicts with **fit_config
                fc_voxel_size = fit_config.pop("voxel_size", None)
                fc_output_space = fit_config.pop("output_space", "real")

                with asection(
                    f"Fitting tile {tile_idx}/{len(specs)} "
                    f"grid={specs[tile_idx].grid_index}"
                ):
                    result = fit_tile(
                        volume,
                        specs[tile_idx],
                        voxel_size=fc_voxel_size,
                        output_space=fc_output_space,
                        progressive=progressive,
                        max_splats_per_pass=max_splats_per_pass,
                        psnr_patience=psnr_patience,
                        max_passes=max_passes,
                        seeds=parsed_seeds,
                        **fit_config,
                    )

            elif tiled:
                # Full tiled fitting
                from luxar.gsplats.fit_tiled_gsplats import fit_tiled

                # Extract params that are explicit in fit_tiled to avoid
                # "got multiple values" conflicts with **fit_config
                fc_voxel_size = fit_config.pop("voxel_size", None)
                fc_output_space = fit_config.pop("output_space", "real")
                fc_verbose = fit_config.pop("verbose", True)

                # Partition by default (one part per tile), unless --flat. With
                # --downscale this sequential path rescales a flat merged result
                # back to original coords below, so partition is only offered
                # here when not downscaling (use -j>1 for a downscaled partition,
                # whose workers rescale themselves).
                seq_partition = (not flat) and tiled_downscale_factors is None
                if (not flat) and tiled_downscale_factors is not None:
                    if recipe is not None:
                        raise typer.BadParameter(
                            "--recipe needs a partition, but the sequential tiled "
                            "path writes a flat leaf under --downscale. Use -j>1 "
                            "(parallel tiles) for a downscaled partition with LOD."
                        )
                    aprint(
                        "Note: --downscale on the sequential tiled path writes a "
                        "flat leaf; use -j>1 for a downscaled partition."
                    )
                result = fit_tiled(
                    volume,
                    tile_size=tile_size,
                    overlap=tile_overlap,
                    voxel_size=fc_voxel_size,
                    output_space=fc_output_space,
                    verbose=fc_verbose,
                    progressive=progressive,
                    max_splats_per_pass=max_splats_per_pass,
                    psnr_patience=psnr_patience,
                    max_passes=max_passes,
                    seeds=parsed_seeds,
                    partition=seq_partition,
                    recipe=recipe,
                    recipe_params=recipe_params,
                    **fit_config,
                )

            elif progressive:
                # Progressive fitting: multiple passes on residuals
                from luxar.gsplats.fit_progressive_gsplats import (
                    fit_progressive_gaussian_splats,
                )

                # max_splats = seeds (total budget), or use seeds as max
                prog_max_splats = (
                    parsed_seeds
                    if isinstance(parsed_seeds, int)
                    else fit_config.pop("seeds", 50000)
                )
                # Map --iters to iters_per_pass for progressive mode
                prog_iters = fit_config.pop("n_iters", 1000)
                # Remove params that progressive handles differently
                fit_config.pop("downscale", None)
                fit_config.pop("seeds", None)
                # voxel_size/output_space are passed through — progressive
                # handles them internally (voxel space for passes, converts final result)

                with asection("Progressive Optimization"):
                    result = fit_progressive_gaussian_splats(
                        volume,
                        max_splats=prog_max_splats,
                        max_splats_per_pass=max_splats_per_pass,
                        iters_per_pass=prog_iters,
                        psnr_patience=psnr_patience,
                        max_passes=max_passes,
                        **fit_config,
                    )

            else:
                # Standard fitting (downscale handled inside fit_gaussian_splats)
                with asection("Optimization"):
                    result = fit_gaussian_splats(
                        volume,
                        seeds=parsed_seeds,
                        downscale=effective_downscale,
                        **fit_config,
                    )

            # Rescale tiled results back to original coordinates if downscaled
            if tiled_downscale_factors is not None and result.n_splats > 0:
                from luxar.gsplats.fitting.downscale import (
                    rescale_centers,
                    rescale_cholesky_packed,
                )
                from luxar.gsplats.gsplat_data import GSplatData

                result = GSplatData(
                    centers=rescale_centers(result.centers, tiled_downscale_factors),
                    amplitudes=result.amplitudes,
                    cholesky_factors=rescale_cholesky_packed(
                        result.cholesky_factors, tiled_downscale_factors
                    ),
                    colors=result.colors,
                    stats=result.stats,
                )
                aprint(f"Rescaled {result.n_splats} splats to original coordinates")

            # 7. Save
            from luxar.gsplats.gsplat_data import GSplatData

            is_leaf = isinstance(result, GSplatData)
            with asection(f"Saving to {output_path.name}"):
                if (
                    is_leaf
                    and allow_empty_tile
                    and tile is not None
                    and result.n_splats == 0
                ):
                    # Empty tile (windowed to near-zero signal): the gsplats
                    # writer enforces a no-empty policy, so instead of erroring
                    # we drop an .empty marker that the parallel orchestrator
                    # treats as a legitimately-skipped tile at merge time.
                    marker = Path(str(output_path) + ".empty")
                    marker.write_text("0 splats\n")
                    aprint("Empty tile (0 splats): wrote marker, skipped save")
                    n_splats = 0
                else:
                    # leaf → .save; partition node → write_gsplats_tree
                    n_splats = _save_fit_output_impl(
                        result, output_path, compress=compress, verbose=verbose
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
