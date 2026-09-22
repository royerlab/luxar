"""Implementation helper for gsplat fit command."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import typer
from arbol import aprint, asection

from luxar.utils.lod_methods import GSPLAT_ADDITIVE_CHOICES_HELP

from .fit_utils import (
    CONTENT_UNSUPPORTED_FIT_FLAGS,
    FitPipelineCtx,
    assemble_fit_config,
    dispatch_parallel_tiled,
    fit_progressive,
    fit_sequential_tiled,
    fit_single_tile,
    maybe_denoise_full_volume,
    reject_rescaled_volume_refit,
    rescale_and_save,
    resolve_denoise_h,
    resolve_floor_with_calibration,
    validate_and_build_recipe,
    warn_ignored_density_flags,
)
from .fit_utils import (
    resolve_tiling as _resolve_tiling_impl,
)


def _parse_norm_range(value: Optional[str]) -> "Optional[tuple[float, float]]":
    """Parse the internal ``--norm-range LO,HI`` worker handoff."""
    if value is None:
        return None
    try:
        parts = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise typer.BadParameter("--norm-range must be LO,HI") from exc
    if len(parts) != 2:
        raise typer.BadParameter("--norm-range must be LO,HI")
    from luxar.gsplats.fitting.validation import _validate_norm_range

    norm_range = (parts[0], parts[1])
    try:
        _validate_norm_range(norm_range)
    except ValueError as exc:
        raise typer.BadParameter(f"--norm-range: {exc}") from exc
    return norm_range


def _parse_voxel_size(value: Optional[str]) -> "Optional[tuple[float, ...]]":
    """Parse the internal ``--voxel-size`` worker handoff."""
    if value is None:
        return None
    try:
        return tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise typer.BadParameter(
            "--voxel-size must be comma-separated numbers"
        ) from exc


def _parse_tile_region(value: Optional[str]) -> "Optional[tuple[slice, ...]]":
    """Parse the internal ``--tile-region START:STOP,...`` worker handoff."""
    if value is None:
        return None
    spans = []
    try:
        for raw in value.split(","):
            start, stop = raw.split(":", 1)
            spans.append(slice(int(start), int(stop)))
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(
            "--tile-region must be START:STOP[,START:STOP...]"
        ) from exc
    if any(span.start < 0 or span.stop <= span.start for span in spans):
        raise typer.BadParameter(
            "--tile-region spans must be non-empty and non-negative"
        )
    return tuple(spans)


def _parse_tile_volume_shape(value: Optional[str]) -> "Optional[tuple[int, ...]]":
    if value is None:
        return None
    try:
        shape = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise typer.BadParameter(
            "--tile-volume-shape must be comma-separated integers"
        ) from exc
    if not shape or any(size <= 0 for size in shape):
        raise typer.BadParameter("--tile-volume-shape entries must be positive")
    return shape


def _parse_tile_worker_metadata(
    *,
    tile: Optional[str],
    tile_region: Optional[str],
    tile_volume_shape: Optional[str],
    tile_nonempty_count: Optional[int],
    tile_seed_count: Optional[int],
    fold_tile_slivers: Optional[bool],
) -> "tuple[Optional[tuple[slice, ...]], Optional[tuple[int, ...]]]":
    region = _parse_tile_region(tile_region)
    shape = _parse_tile_volume_shape(tile_volume_shape)
    if (region is None) != (shape is None):
        raise typer.BadParameter(
            "--tile-region and --tile-volume-shape must be provided together"
        )
    if region is not None and tile is None:
        raise typer.BadParameter("--tile-region requires --tile")
    if tile_nonempty_count is not None and tile_nonempty_count <= 0:
        raise typer.BadParameter("--tile-nonempty-count must be positive")
    if tile_seed_count is not None and tile_seed_count < 0:
        raise typer.BadParameter("--tile-seed-count must be non-negative")
    # A per-tile count needs a tile, not necessarily a tile-local READ plan:
    # `batch-fit` pairs it with --tile-region, but the direct `fit -j` parent
    # hands its workers an exact count while each still reads the whole volume.
    if tile_seed_count is not None and tile is None:
        raise typer.BadParameter("--tile-seed-count requires --tile")
    # Tri-state since #2838: unset means "fold", which is what every other
    # producer does. Only an EXPLICIT flag is a usage error off the --tile path,
    # where there is no grid choice left to make.
    if fold_tile_slivers is not None and tile is None:
        raise typer.BadParameter(
            "--fold-tile-slivers/--no-fold-tile-slivers requires --tile"
        )
    return region, shape


def _stamp_source_dtype(fit_config: dict, source_info: dict) -> None:
    """Carry the loader-observed source dtype into the fit config.

    ``fit_config`` is forwarded (as ``**fit_config``) by every fit branch to
    ``fit_gaussian_splats``, so a tile worker stamps the same source dtype as a
    whole-volume fit. The CLI's ``load_volume`` returns float32 whatever the file
    holds, so this is the only place the on-disk element type still exists.

    A dtype the USER put in the config wins, and there is no CLI flag to override
    it: ``load_fit_config`` passes arbitrary YAML keys through, so a
    ``source_dtype: uint16`` in a ``--config`` file is a deliberate statement
    about a file whose stored type the loader can no longer see (a float32 .npy
    exported from a 16-bit acquisition). Only a TRUTHY existing value counts as a
    choice — ``get_fit_defaults()`` injects a signature-derived
    ``source_dtype: None``, which must still be filled in from the loader.
    """
    if fit_config.get("source_dtype"):
        return
    if source_info.get("source_dtype"):
        fit_config["source_dtype"] = source_info["source_dtype"]


def _declare_pre_downscale_source(
    fit_config: dict, original_shape: "tuple[int, ...]", *, single_tile: bool
) -> None:
    """Declare the pre-downscale grid when the CLI decimates before tiling.

    The tiled fitter measures the array it is HANDED, and on the tiled paths the
    command downscales the volume itself first. Left alone the merged result
    would therefore record the decimated working copy as its source — a
    compression ratio quoted against a grid the acquisition never had, with
    ``fitted_shape`` equal to it so ``info`` cannot even show the decimation.
    (The whole-volume path has no such problem: ``fit_gaussian_splats``
    downscales internally, so it measures the caller's array.)

    Not for ``--tile i/M``: that writes ONE crop, whose source is its own
    sub-volume, not the whole acquisition.

    A ``source_shape`` already in the config is a deliberate statement by the
    user (``--config`` passes arbitrary keys through) and wins, exactly as
    :func:`_stamp_source_dtype` treats ``source_dtype``.
    """
    if single_tile or fit_config.get("source_shape"):
        return
    fit_config["source_shape"] = [int(s) for s in original_shape]


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
        help=(
            "Seed count (int), compression ratio (float in (0,1]), or 'auto'. "
            "An integer is a WHOLE-VOLUME budget (what a default `gsplat cal` "
            "reports): a tiled fit divides it across the tiles that contain "
            "signal instead of giving every tile the full count. A K below "
            "the non-empty tile count gives one seed per such tile. A ratio is "
            "scale-free and is applied per tile unchanged."
        ),
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
        "data's axis order differs. Time/channel axes are sliced and dropped; "
        "--channel is a flat row-major index across all channel-like axes, and "
        "more than one time axis is rejected. Spatial axes stay in the given order.",
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
        "fixed value | none or 0 = disable (hard-min normalization). auto and "
        "pN ignore exact-zero padding. Negative user levels are rejected; a "
        "negative estimate from dark-frame-corrected data is preserved. The "
        "erase-all guard compares against a `norm_range:` supplied in "
        "--config; "
        "otherwise it uses the data maximum, not the configured normalization "
        "percentile's high endpoint. A configured "
        "norm_percentile may still raise the applied low endpoint above the "
        "requested floor. Unset lets a "
        "`floor:` in --config/preset apply, else defaults to auto. Under any "
        "--tiling the spec is resolved against the WHOLE volume — never a tile "
        "or box crop — so every tile/box works from the same level.",
    ),
    norm_range: Optional[str] = typer.Option(
        None,
        "--norm-range",
        hidden=True,
        help="Internal worker handoff: raw-input normalization range LO,HI.",
    ),
    voxel_size: Optional[str] = typer.Option(
        None,
        "--voxel-size",
        hidden=True,
        help="Internal worker handoff: comma-separated physical voxel spacing.",
    ),
    physical_coordinates: bool = typer.Option(
        False,
        "--physical-coordinates",
        hidden=True,
        help="Internal batch worker handoff: enable physical content-box geometry.",
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
        "volume unless BOTH some dimension exceeds --tile-size AND the volume "
        "has more than 64 M voxels; then uniform (or content when a density "
        "--cal/--k-star-ref is given). Replaces the old --tiled.",
        rich_help_panel="Tiling",
    ),
    flat: bool = typer.Option(
        False,
        "--flat",
        help="Tiled fits emit a kind=partition (one part per tile/box) by "
        "default for viewer frustum culling; --flat merges to a single leaf "
        "instead. Uniform and content merges record bounded whole-volume "
        "quality metrics for both shapes (override with "
        "LUXAR_TILED_QUALITY_MAX_GB).",
        rich_help_panel="Tiling",
    ),
    tile_size: int = typer.Option(
        256,
        "--tile-size",
        help="Tile size in voxels (per axis). A trailing sliver is folded into "
        "its neighbour, so one tile per axis can span up to "
        "tile_size + overlap - 1 voxels (256/32 -> 287, i.e. 1.41x the voxels "
        "of a full tile in 3D); size this for that worst case.",
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
    tile_region: Optional[str] = typer.Option(None, "--tile-region", hidden=True),
    tile_volume_shape: Optional[str] = typer.Option(
        None, "--tile-volume-shape", hidden=True
    ),
    tile_nonempty_count: Optional[int] = typer.Option(
        None, "--tile-nonempty-count", hidden=True
    ),
    tile_seed_count: Optional[int] = typer.Option(
        None, "--tile-seed-count", hidden=True
    ),
    fold_tile_slivers: Optional[bool] = typer.Option(
        None, "--fold-tile-slivers/--no-fold-tile-slivers", hidden=True
    ),
    floor_resolved: bool = typer.Option(
        False,
        "--floor-resolved",
        hidden=True,
        help="The numeric --floor on this command line is a LEVEL a parent "
        "already resolved against the whole volume, not a user request — apply "
        "it verbatim instead of re-guarding it against this worker's sub-volume "
        "(#1174/#2838). Read by single-tile mode (every `-j N` and batch-fit "
        "worker); inert elsewhere and for a non-numeric --floor, which is "
        "guarded wherever it first becomes a level.",
    ),
    jobs: str = typer.Option(
        "1",
        "--jobs",
        "-j",
        help="With --tiling uniform/content: number of tiles/boxes to fit "
        "concurrently as subprocesses "
        "on one GPU (int, or 'auto' to size from GPU memory, host RAM, and "
        "available CPU threads). Default 1 = "
        "sequential. Ignored with --tiling none or --tile.",
        rich_help_panel="Tiling",
    ),
    keep_tiles: bool = typer.Option(
        False,
        "--keep-tiles",
        help="With --tiling --jobs>1: keep the per-tile/box temporary .gsplats.zarr "
        "outputs (and any .empty markers for skipped tiles) instead of "
        "deleting them after the merge; retained content-box workers also render "
        "and stamp their own quality/source-grid block.",
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
        "--add-method",
        help=f"[--recipe stream] {GSPLAT_ADDITIVE_CHOICES_HELP}. auto (the default) is "
        "greedy ((1-1/e)-optimal) at small N, self_energy (cheap O(N log N)) "
        "for large parts. radial reveals outward from the bbox center.",
        rich_help_panel="Per-part LOD",
    ),
    recipe_breakpoints: Optional[str] = typer.Option(
        None,
        "-b",
        "--breakpoints",
        help="[--recipe stream] additive ladder breakpoints: 'equal-count' "
        "(default), 'stream:C' (geometric streaming ladder, sized per part), "
        "'counts:500,2000,...', 'equi-energy:<n>' or 'energy:0.5,0.9,...'.",
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
        "--subst-method",
        help="[--recipe levels] auto (default) / kmeans-lloyd / greedy / greedy-lloyd.",
        rich_help_panel="Per-part LOD",
    ),
    recipe_refine: Optional[str] = typer.Option(
        None,
        "--refine",
        help="[--recipe levels] refine each per-tile coarse level: none "
        "(default) | l2 (against its fine input) | volume (re-fit against the "
        "volume being fitted, cropped to each tile — highest fidelity). No "
        "--target is needed: the volume is already in hand and the splats are "
        "in its voxel frame.",
        rich_help_panel="Per-part LOD",
    ),
    recipe_refine_iters: Optional[int] = typer.Option(
        None,
        "--refine-iters",
        help="[--recipe levels] refinement steps per level (default 120 for "
        "--refine l2, 300 for --refine volume).",
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
        help="Calibration JSON (gsplat cal) supplying the splats-per-feature "
        "density. Under --tiling content it also supplies the background floor "
        "when --floor is unset, so the fit runs on the intensity scale the "
        "density was measured on. That level is ABSOLUTE: reusing one cal.json "
        "across a timelapse applies the calibrated timepoint's pedestal to "
        "every other one, where --floor auto re-estimates per volume.",
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
        help="Sub-linear exponent alpha (K~feat^alpha) for content planning "
        "and the split of a uniform integer --seeds budget: each non-empty "
        "tile weighs Hann voxels x (mean above-floor intensity / ceiling)^alpha "
        "unless the legacy foreground weights track mass at >= 0.95 correlation; "
        "in mass mode, a tile "
        "holding >= 1/4 of the equal mass share never gets < 1/4 of the equal "
        "seed share.",
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
        help="Enable progressive fitting: optimize in several passes, each pass "
        "fitting new splats to the residual of the previous ones. The passes are "
        "an optimization schedule, not a level-of-detail structure: the result is "
        "ONE flat splat set (build a streaming ladder afterwards with "
        "`luxar gsplat lod --recipe stream`). "
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
        "A --tile k/M worker applies this independently to its tile; batch-fit "
        "does the same before merging, so --flat approximates rather than exactly "
        "matches one global cull over the merged splats. "
        "For example, 0.95 — what a bare fit falls through to — discards splats "
        "in the bottom 5% of cumulative amplitude; how many splats that is "
        "depends on how heavy-tailed the data is, and on a sparse volume it can "
        "be most of them. Every --preset sets 0.999 instead "
        "(near-lossless), and so does --tiling content even without a preset. "
        "Set to 0 to keep every splat.",
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
    parsed_norm_range = _parse_norm_range(norm_range)
    parsed_voxel_size = _parse_voxel_size(voxel_size)
    parsed_tile_region, parsed_tile_volume_shape = _parse_tile_worker_metadata(
        tile=tile,
        tile_region=tile_region,
        tile_volume_shape=tile_volume_shape,
        tile_nonempty_count=tile_nonempty_count,
        tile_seed_count=tile_seed_count,
        fold_tile_slivers=fold_tile_slivers,
    )

    try:
        from luxar.gsplats import fit_gaussian_splats

        with asection(f"Fitting Gaussian Splats: {input_path.name}"):
            # 1. Load volume
            with asection("Loading volume"):
                # `load_volume` returns float32 whatever the file holds, so the
                # stored element type is knowable only from it. It is what the
                # fit records as its source size, and hence the denominator of
                # any compression ratio quoted about the result: a 16-bit
                # acquisition measured as float32 would report half the real
                # compression.
                source_info: dict = {}
                volume = load_volume(
                    input_path,
                    channel,
                    timepoint,
                    array_key,
                    axes=axes,
                    info=source_info,
                    region=parsed_tile_region,
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
                tiling,
                parsed_tile_volume_shape or volume.shape,
                tile_size,
                _has_density,
            )

            # A calibration records the floor it subtracted, and measured its
            # density's feature_threshold on that scale — so consume it when the
            # user said nothing about --floor, rather than silently fitting on a
            # different scale than the density was calibrated for (#1175).
            # ONLY under `--tiling content` (the resolver enforces it): that is
            # the only mode where --cal is honoured at all, and
            # `warn_ignored_density_flags` below says so out loud for every
            # other mode. Adopting a floor from a flag the very next line calls
            # ignored would be the CLI contradicting itself.
            floor = resolve_floor_with_calibration(
                cal, floor, config, tiling=resolved_tiling, verbose=verbose
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
                norm_range=parsed_norm_range,
                voxel_size=parsed_voxel_size,
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
                recipe_compression_factor=recipe_compression_factor,
                recipe_levels=recipe_levels,
                recipe_substitutive_method=recipe_substitutive_method,
                recipe_coarsen_dims=recipe_coarsen_dims,
                recipe_refine=recipe_refine,
                recipe_refine_iters=recipe_refine_iters,
                cal=cal,
                k_star_ref=k_star_ref,
                n_features_ref=n_features_ref,
                feature_threshold=feature_threshold,
                feature_metric=feature_metric,
                target_features=target_features,
                saturation_exponent=saturation_exponent,
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
            recipe_params: "Any" = validate_and_build_recipe(ctx, volume.ndim, volume)

            if resolved_tiling == "content":
                from luxar.cli.gsplat_ops.planner import run_content_fit

                # Flags the content path does not implement — warn loudly rather
                # than silently ignore (the fit knobs below ARE honored).
                enabled_unsupported = {
                    "--denoise": denoise,
                    "--downscale": downscale is not None,
                    "--progressive": progressive,
                }
                _unsupported = [
                    flag
                    for flag in CONTENT_UNSUPPORTED_FIT_FLAGS
                    if enabled_unsupported[flag]
                ]
                if _unsupported:
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
                    norm_range=parsed_norm_range,
                    voxel_size=parsed_voxel_size,
                    physical_coordinates=physical_coordinates,
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
                    source_dtype=source_info.get("source_dtype"),
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
            _stamp_source_dtype(fit_config, source_info)

            # A per-tile volume re-fit needs the tile grid and the splats in ONE
            # coordinate frame, which --downscale and a real-space voxel_size
            # each break independently (see the helper).
            reject_rescaled_volume_refit(
                recipe_params,
                effective_downscale,
                voxel_size=fit_config.get("voxel_size"),
                output_space=fit_config.get("output_space", "real"),
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
                    _declare_pre_downscale_source(
                        fit_config, original_shape, single_tile=tile is not None
                    )

            # 6. Fit
            if tile is not None:
                # Single-tile mode (Slurm-ready)
                result = fit_single_tile(
                    ctx,
                    volume,
                    fit_config,
                    parsed_seeds,
                    full_volume_shape=parsed_tile_volume_shape,
                    nonempty_tiles=tile_nonempty_count,
                    tile_seed_count=tile_seed_count,
                    # Unset folds: the `--tile k/M` worker must build the same
                    # grid the sequential, -j N and batch-fit producers do, or
                    # M itself means something different here (#2838).
                    fold_tile_slivers=(
                        True if fold_tile_slivers is None else fold_tile_slivers
                    ),
                    preselected_tile=parsed_tile_region is not None,
                    floor_resolved=floor_resolved,
                )

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

    except (typer.Exit, typer.BadParameter):
        # BadParameter is a usage error (e.g. an invalid --floor spec, rejected
        # before anything is read): let Typer render it as one instead of burying
        # it under a traceback from the generic handler below. Kept in the same
        # handler as Exit so this stays one branch (the C901 ratchet counts them).
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e
