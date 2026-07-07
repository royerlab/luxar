"""``luxar gsplat fitting`` commands (extracted from gsplat_commands.py).

Each command is a plain function; ``register_fitting_commands(app)`` wires them onto
the shared ``app_gsplat`` Typer (package-refactor-plan P3/P4/P6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import typer

from .fitting_calibrate import run_calibrate_command
from .fitting_denoise_render import (
    run_denoise_volume_cmd,
    run_render_to_file,
)
from .fitting_fit import run_fit_volume
from .fitting_fit_utils import (
    build_fit_recipe_params as _build_fit_recipe_params_impl,
)
from .fitting_fit_utils import (
    resolve_tiling as _resolve_tiling_impl,
)
from .fitting_fit_utils import (
    save_fit_output as _save_fit_output_impl,
)


def denoise_volume_cmd(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input volume (.npy/.npz/.tiff/.zarr/.zarr.zip)"
    ),
    output_path: Path = typer.Argument(..., help="Output path (.npy/.zarr)"),
    # Denoise params
    h: Optional[float] = typer.Option(
        None, "--h", help="Manual NLM h value (skip auto-calibration)"
    ),
    patch_size: int = typer.Option(
        3, "--patch-size", help="NLM patch size (odd integer)"
    ),
    search_distance: int = typer.Option(
        5, "--search-distance", help="NLM search window half-size"
    ),
    backend: str = typer.Option(
        "auto", "--backend", help="NLM backend: auto/cuda/pytorch/skimage"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-d", help="Device: auto/cpu/cuda/mps"
    ),
    denoise_2d: bool = typer.Option(
        False, "--denoise-2d", help="Denoise slice-by-slice (2D) instead of 3D"
    ),
    # Input selection
    channel: Optional[int] = typer.Option(None, "--channel", help="Channel index"),
    timepoint: Optional[int] = typer.Option(
        None, "--timepoint", help="Timepoint index"
    ),
    array_key: Optional[str] = typer.Option(
        None, "--array-key", help="Array key within zarr store"
    ),
) -> None:
    """Denoise a volume using Non-Local Means.

    Auto-calibrates the denoising strength h using Noise2Self unless --h is
    provided.  Runs locally (no Slurm).  For batch denoising on HPC, use
    ``luxar gsplat batch-fit submit --denoise``.

    Examples:
        luxar gsplat denoise volume.zarr denoised.zarr

        luxar gsplat denoise volume.zarr denoised.npy --h 0.03

        luxar gsplat denoise data.zarr.zip out.zarr --channel 0 --timepoint 5 --denoise-2d
    """
    return run_denoise_volume_cmd(
        input_path=input_path,
        output_path=output_path,
        h=h,
        patch_size=patch_size,
        search_distance=search_distance,
        backend=backend,
        device=device,
        denoise_2d=denoise_2d,
        channel=channel,
        timepoint=timepoint,
        array_key=array_key,
    )


def _resolve_tiling(
    tiling: str, shape: "tuple[int, ...]", tile_size: int, has_density: bool
) -> str:
    """Back-compat wrapper around fit tiling-strategy resolution helpers."""
    return _resolve_tiling_impl(tiling, shape, tile_size, has_density)


def _build_fit_recipe_params(
    recipe: str,
    *,
    n_lods: Optional[int],
    additive_method: Optional[str],
    breakpoints: Optional[str],
    target_ms: Optional[float] = None,
    bandwidth_mbps: Optional[float] = None,
    bytes_per_splat: Optional[float] = None,
    compression_factor: Optional[int],
    levels: Optional[int],
    substitutive_method: Optional[str],
    coarsen_dims: Optional[str],
    device: Optional[str],
    volume_ndim: int,
) -> "Any":
    """Back-compat wrapper around fit recipe-argument parsing helpers."""
    return _build_fit_recipe_params_impl(
        recipe,
        n_lods=n_lods,
        additive_method=additive_method,
        breakpoints=breakpoints,
        target_ms=target_ms,
        bandwidth_mbps=bandwidth_mbps,
        bytes_per_splat=bytes_per_splat,
        compression_factor=compression_factor,
        levels=levels,
        substitutive_method=substitutive_method,
        coarsen_dims=coarsen_dims,
        device=device,
        volume_ndim=volume_ndim,
    )


def _save_fit_output(
    result: Any,
    output_path: Path,
    *,
    compress: "Optional[Literal['zip', 'tar.gz']]",
    verbose: bool,
) -> int:
    """Back-compat wrapper around fit output saving helper."""
    return _save_fit_output_impl(
        result,
        output_path,
        compress=compress,
        verbose=verbose,
    )


def fit_volume(
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
    return run_fit_volume(
        input_path=input_path,
        output_path=output_path,
        seeds=seeds,
        iters=iters,
        device=device,
        preset=preset,
        loss=loss,
        config=config,
        dump_config=dump_config,
        compress=compress,
        channel=channel,
        timepoint=timepoint,
        array_key=array_key,
        axes=axes,
        lr=lr,
        seed_method=seed_method,
        verbose=verbose,
        downscale=downscale,
        tiling=tiling,
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
        saturation_exponent=saturation_exponent,
        saturation_cap=saturation_cap,
        feature_threshold=feature_threshold,
        feature_metric=feature_metric,
        cell=cell,
        target_features=target_features,
        min_leaf=min_leaf,
        max_leaf=max_leaf,
        plan=plan,
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


def render_to_file(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    output_path: Path = typer.Argument(..., help="Output file (.npy or .tiff)"),
    shape: Optional[str] = typer.Option(
        None,
        "--shape",
        help="Output shape as comma-separated ints (e.g., '128,128,128')",
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-d", help="Device: auto/cpu/cuda/mps"
    ),
    truncate: Optional[float] = typer.Option(
        None,
        "--truncate",
        "-t",
        help="Truncation radius in sigma. Defaults to dataset's stored value.",
    ),
) -> None:
    """Render Gaussian splats back to a volume.

    Useful for quality comparison with original data. Auto-detects the
    output format from file extension (.npy or .tiff).

    If --shape is not given, it is auto-computed from the bounding box.

    Examples:
        luxar gsplat render fitted.gsplats.zarr rendered.npy
        luxar gsplat render fitted.gsplats.zarr rendered.tiff --shape 256,256,256
        luxar gsplat render fitted.gsplats.zarr rendered.npy --device cuda
    """
    return run_render_to_file(
        input_path=input_path,
        output_path=output_path,
        shape=shape,
        device=device,
        truncate=truncate,
    )


def calibrate_command(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input volume (.zarr, .zarr.zip, .tiff, .npy, .npz)"
    ),
    output_json: Path = typer.Argument(
        ..., help="Output JSON with full sweep curves and recommended K*"
    ),
    # K grid
    k_grid: Optional[str] = typer.Option(
        None,
        "--k-grid",
        help="Explicit comma-separated K values (e.g. '1000,4000,16000'). Overrides --n-grid/--k-min/--k-max.",
    ),
    n_grid: int = typer.Option(
        10, "--n-grid", help="Number of K values when --k-grid is not given"
    ),
    k_min: int = typer.Option(1_000, "--k-min", help="Smallest K in the sweep"),
    k_max: int = typer.Option(512_000, "--k-max", help="Largest K in the sweep"),
    progression: str = typer.Option(
        "exp",
        "--progression",
        help="Spacing of the K grid: 'exp' (geometric/log-spaced) or 'power' (polynomial)",
    ),
    power: int = typer.Option(
        2,
        "--power",
        help="Exponent when --progression power (1=linear, 2=quadratic, ...)",
    ),
    # CV mask
    mask_seed: int = typer.Option(
        42, "--mask-seed", help="RNG seed for the held-out mask"
    ),
    mask_fraction: float = typer.Option(
        0.05, "--mask-fraction", help="Fraction of voxels to hold out (default 5%)"
    ),
    # Fit configuration (delegated to existing config loader)
    preset: str = typer.Option(
        "n2s",
        "--preset",
        help=(
            "Fit preset: draft, standard, hifi, ultra, n2s. "
            "Default 'n2s' matches the manuscript's blind-spot protocol "
            "(n_iters=20000, early_stop_patience=500, cull_retention=0.999) "
            "so the held-out PSNR curve has enough optimiser budget to enter "
            "the overfit regime at high K. Lower presets undertrain at high K "
            "and bias K* upward."
        ),
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", help="YAML overrides for fit parameters"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-d", help="Device: auto/cpu/cuda/mps"
    ),
    # Volume loader pass-through (matches `compare` and `fit`)
    channel: Optional[int] = typer.Option(
        None, "--channel", "-c", help="Channel index for OME-Zarr inputs"
    ),
    timepoint: Optional[int] = typer.Option(
        None, "--timepoint", help="Timepoint index for OME-Zarr inputs"
    ),
    array_key: Optional[str] = typer.Option(
        None, "--array-key", help="Array key within .npz / nested zarr"
    ),
    axes: Optional[str] = typer.Option(
        None,
        "--axes",
        help="Per-dimension axis labels overriding the TCZYX/CZYX/ZYX heuristic "
        "(e.g. 'z,c,y,x'). Time/channel axes are sliced by --timepoint/--channel "
        "and dropped; spatial axes kept in the given order.",
    ),
    # Regime-robust extensions (all opt-in; defaults preserve manuscript behaviour)
    k_star_metric: str = typer.Option(
        "psnr_minmax",
        "--k-star-metric",
        help=(
            "Metric for K* selection: psnr_minmax (default, manuscript) | "
            "psnr_foreground | gain. For sparse/noise-free data, 'gain' "
            "(dB over the predict-zero baseline) is far more reliable than the "
            "background-dominated min--max PSNR."
        ),
    ),
    auto_region: bool = typer.Option(
        False,
        "--auto-region/--no-auto-region",
        help=(
            "Calibrate on an auto-selected content-rich sub-region (recommended "
            "for large/sparse volumes: calibrate at the scale you fit at)."
        ),
    ),
    region_size: int = typer.Option(
        256,
        "--region-size",
        help="Edge length of the auto-selected calibration region (voxels).",
    ),
    region_strategy: str = typer.Option(
        "densest", "--region-strategy", help="Auto-region pick: densest | median."
    ),
    feature_metric: str = typer.Option(
        "peaks",
        "--feature-metric",
        help=(
            "Content metric for region density + splat-density transfer: "
            "peaks (default, best for nuclei) | edges | intensity."
        ),
    ),
    saturation_exponent: float = typer.Option(
        0.44,
        "--saturation-exponent",
        help=(
            "Sub-linear exponent alpha in the K~features^alpha density transfer "
            "(empirical default 0.44; adjustable/discoverable)."
        ),
    ),
    rd_model: bool = typer.Option(
        True,
        "--rd-model/--no-rd-model",
        help="Fit a parametric error-vs-K model (extrapolation + 'not-converged' flag).",
    ),
    fit_exponent: bool = typer.Option(
        False,
        "--fit-exponent",
        help=(
            "Measure the saturation exponent alpha (K~features^alpha) instead of "
            "assuming the default 0.44: calibrate K* at several region scales "
            "(--exponent-scales) and regress log K* on log n_features. The fitted "
            "alpha is written into splat_density. WARNING: multiplies runtime by "
            "the number of scales (each is a full K-sweep)."
        ),
    ),
    exponent_scales: Optional[str] = typer.Option(
        None,
        "--exponent-scales",
        help=(
            "Comma-separated region edge lengths for --fit-exponent "
            "(default '128,192,256'). Each yields one (n_features, K*) point."
        ),
    ),
    # Optional outputs
    pdf_report: Optional[Path] = typer.Option(
        None,
        "--pdf",
        help="Generate calibration PDF report (rate-distortion + slice montages + CV curves)",
    ),
    keep_fits: Optional[Path] = typer.Option(
        None,
        "--keep-fits",
        help="Directory to persist per-K .gsplats.zarr fits for later inspection",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress the terminal summary table (the JSON file is still written)",
    ),
) -> None:
    """Calibrate splat count via blind-spot cross-validation.

    Sweeps Gaussian-splat fits over a grid of K values, evaluates held-out
    PSNR at masked voxels, and reports the recommended K* (the held-out
    peak) plus the dataset's noise-floor PSNR ceiling.

    The fit at each K runs against a 5%-donut-median-filled volume so the
    optimiser never sees the original noisy values at masked positions —
    this is the Noise2Self protocol from Batson & Royer (2019), as used
    in the Luxar manuscript's model-selection analysis.

    Canonical end-to-end pipeline: ``cal`` → ``fit --seeds K*`` →
    ``lod --recipe stream`` (or ``levels`` / ``tiles`` / ...) for a
    streaming-ready multi-resolution dataset. Use ``--fit-exponent`` to measure
    the density exponent that ``fit --tiling content`` consumes.

    Examples:
        luxar gsplat cal kidney_dapi.tiff cal.json
        luxar gsplat cal volume.zarr cal.json --n-grid 5 --k-max 128000 --preset draft
        luxar gsplat cal volume.zarr cal.json --k-grid '1000,4000,16000,64000,256000'
        luxar gsplat cal volume.tiff cal.json --pdf report.pdf --keep-fits fits/
    """
    return run_calibrate_command(
        input_path=input_path,
        output_json=output_json,
        k_grid=k_grid,
        n_grid=n_grid,
        k_min=k_min,
        k_max=k_max,
        progression=progression,
        power=power,
        mask_seed=mask_seed,
        mask_fraction=mask_fraction,
        preset=preset,
        config=config,
        device=device,
        channel=channel,
        timepoint=timepoint,
        array_key=array_key,
        axes=axes,
        k_star_metric=k_star_metric,
        auto_region=auto_region,
        region_size=region_size,
        region_strategy=region_strategy,
        feature_metric=feature_metric,
        saturation_exponent=saturation_exponent,
        rd_model=rd_model,
        fit_exponent=fit_exponent,
        exponent_scales=exponent_scales,
        pdf_report=pdf_report,
        keep_fits=keep_fits,
        quiet=quiet,
    )


def register_fitting_commands(app: typer.Typer) -> None:
    """Register the fitting commands onto ``app_gsplat``."""
    # Workflow order: fit and cal (the core pre-/fit steps) lead; render and
    # denoise (utilities) follow.
    app.command("fit")(fit_volume)
    app.command("cal")(calibrate_command)
    app.command("render")(render_to_file)
    app.command("denoise")(denoise_volume_cmd)
