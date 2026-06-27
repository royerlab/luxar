"""``luxar gsplat fitting`` commands (extracted from gsplat_commands.py).

Each command is a plain function; ``register_fitting_commands(app)`` wires them onto
the shared ``app_gsplat`` Typer (package-refactor-plan P3/P4/P6).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import typer
from arbol import aprint, asection

from ..utils import format_memory_size

if TYPE_CHECKING:
    import numpy as np


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
    ``luxar gsplat slurm-fit submit --denoise``.

    Examples:
        luxar gsplat denoise volume.zarr denoised.zarr

        luxar gsplat denoise volume.zarr denoised.npy --h 0.03

        luxar gsplat denoise data.zarr.zip out.zarr --channel 0 --timepoint 5 --denoise-2d
    """
    try:
        from luxar.cli.gsplat_config import load_volume
        from luxar.gsplats.preprocessing.denoise_pipeline import (
            denoise_volume_array,
            normalize_volume,
        )

        with asection("Loading volume"):
            volume = load_volume(
                input_path, channel=channel, timepoint=timepoint, array_key=array_key
            )
            aprint(f"Shape: {volume.shape}, dtype: {volume.dtype}")

        # Calibrate h if not provided
        effective_h: float
        if h is not None:
            effective_h = h
            aprint(f"Using manual h={effective_h:.4f}")
        else:
            import torch

            from luxar.gsplats.preprocessing import calibrate_nlm_h
            from luxar.gsplats.utils.device import resolve_torch_device

            with asection("Auto-calibrating h (Noise2Self)"):
                norm_vol, _, _ = normalize_volume(volume)
                t_vol = torch.from_numpy(norm_vol)
                # Auto-select CUDA > MPS > CPU when --device is omitted.
                dev = resolve_torch_device(device) if device else resolve_torch_device()
                effective_h = calibrate_nlm_h(
                    t_vol,
                    patch_size=patch_size,
                    search_distance=search_distance,
                    backend=backend,
                    device=dev,
                    use_2d_slice=True,
                )
                aprint(f"Calibrated h={effective_h:.4f}")

        with asection("Denoising (NLM)"):
            denoised = denoise_volume_array(
                volume,
                h=effective_h,
                patch_size=patch_size,
                search_distance=search_distance,
                backend=backend,
                device=device,
                use_2d=denoise_2d,
            )
            aprint(f"Denoised shape: {denoised.shape}")

        # Save
        with asection(f"Saving to {output_path.name}"):
            suffix = output_path.suffix.lower()
            if suffix == ".npy":
                np.save(output_path, denoised)
            elif suffix in (".zarr",):
                import zarr

                zarr.save(str(output_path), denoised)
            else:
                np.save(output_path, denoised)
            aprint("Done")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1) from e


def _resolve_tiling(
    tiling: str, shape: "tuple[int, ...]", tile_size: int, has_density: bool
) -> str:
    """Resolve ``--tiling`` to a concrete strategy: ``none | uniform | content``.

    ``auto`` fits the whole volume when it fits in a single tile, else uniform —
    or content when a transferable density (``--cal`` / ``--k-star-ref`` / a
    ``--plan`` / ``--plan-box``) is available to size content-balanced boxes.
    """
    t = tiling.lower()
    if t not in ("auto", "none", "uniform", "content"):
        raise typer.BadParameter(
            f"--tiling must be one of auto|none|uniform|content, got {tiling!r}"
        )
    if t != "auto":
        return t
    large = any(int(s) > int(tile_size) for s in shape)
    if not large:
        return "none"
    return "content" if has_density else "uniform"


def _build_fit_recipe_params(
    recipe: str,
    *,
    n_lods: Optional[int],
    additive_method: Optional[str],
    breakpoints: Optional[str],
    compression_factor: Optional[int],
    levels: Optional[int],
    substitutive_method: Optional[str],
    coarsen_dims: Optional[str],
    lod_method: Optional[str],
    device: Optional[str],
    volume_ndim: int,
) -> "Any":
    """Validate the per-part ``--recipe`` knobs and build a ``RecipeParams``.

    Mirrors the ``gsplat lod`` vocabulary (reusing its breakpoint parser and the
    valid-method sets) so a fit-time per-part ladder is identical to a separate
    ``gsplat lod`` pass. Only the two :data:`PER_PART_RECIPES` are accepted.
    """
    from luxar.cli.lod import (
        _VALID_ADDITIVE_METHODS,
        _VALID_SUBSTITUTIVE_METHODS,
        _parse_lod_breakpoints,
    )
    from luxar.gsplats.lod.recipes import PER_PART_RECIPES, RecipeParams

    if recipe not in PER_PART_RECIPES:
        raise typer.BadParameter(
            f"--recipe must be one of {list(PER_PART_RECIPES)} for a fit "
            f"(additive → partitioned, substitutive → mosaic); got {recipe!r}. "
            f"For other topologies run `gsplat lod` on a flat (--flat) fit."
        )

    # Reject knobs that don't apply to the chosen recipe (mirrors `gsplat lod`,
    # which raises on irrelevant options rather than silently dropping them).
    additive_only = {
        "--n-lods": n_lods,
        "--additive-method": additive_method,
        "--breakpoints": breakpoints,
    }
    substitutive_only = {
        "--compression-factor": compression_factor,
        "--levels": levels,
        "--substitutive-method": substitutive_method,
        "--coarsen-dims": coarsen_dims,
        "--lod-method": lod_method,
    }
    irrelevant = substitutive_only if recipe == "additive" else additive_only
    provided = [flag for flag, val in irrelevant.items() if val is not None]
    if provided:
        other = "substitutive" if recipe == "additive" else "additive"
        raise typer.BadParameter(
            f"option(s) {', '.join(provided)} are not used by --recipe {recipe} "
            f"(they configure --recipe {other}). Remove them or switch recipe."
        )

    add_norm = (additive_method or "greedy").strip().replace("-", "_")
    if add_norm not in _VALID_ADDITIVE_METHODS:
        raise typer.BadParameter(
            f"--additive-method must be one of {list(_VALID_ADDITIVE_METHODS)}; "
            f"got {additive_method!r}"
        )
    sub_norm = (substitutive_method or "auto").strip().replace("-", "_")
    if sub_norm not in _VALID_SUBSTITUTIVE_METHODS:
        raise typer.BadParameter(
            f"--substitutive-method must be one of "
            f"{list(_VALID_SUBSTITUTIVE_METHODS)}; got {substitutive_method!r}"
        )
    if lod_method is not None and lod_method not in ("extent", "count"):
        raise typer.BadParameter(
            f"--lod-method must be 'extent' or 'count'; got {lod_method!r}"
        )

    bp = _parse_lod_breakpoints(breakpoints) if breakpoints else "equal-count"

    parsed_coarsen: Optional[tuple] = None
    if coarsen_dims is not None:
        try:
            idxs = sorted({int(t) for t in coarsen_dims.split(",") if t.strip() != ""})
        except ValueError as e:
            raise typer.BadParameter(
                f"--coarsen-dims must be comma-separated integers; got {coarsen_dims!r}"
            ) from e
        if not idxs:
            raise typer.BadParameter("--coarsen-dims must list >=1 index")
        for i in idxs:
            if i < 0 or i >= volume_ndim:
                raise typer.BadParameter(
                    f"--coarsen-dims index {i} out of range for {volume_ndim}D data"
                )
        parsed_coarsen = tuple(idxs) if len(idxs) < volume_ndim else None

    return RecipeParams(
        n_lods=n_lods if n_lods is not None else 4,
        additive_method=add_norm,  # type: ignore[arg-type]
        breakpoints=bp,  # type: ignore[arg-type]
        compression_factor=(
            compression_factor if compression_factor is not None else 4
        ),
        levels=levels if levels is not None else 3,
        substitutive_method=sub_norm,
        coarsen_dims=parsed_coarsen,
        lod_method=lod_method if lod_method is not None else "extent",
        device=device or "auto",
    )


def _save_fit_output(
    result: Any,
    output_path: Path,
    *,
    compress: "Optional[Literal['zip', 'tar.gz']]",
    verbose: bool,
) -> int:
    """Save a flat ``GSplatData`` leaf or a ``kind=partition`` tree node.

    Returns the splat count for the summary line.
    """
    from luxar.gsplats.gsplat_data import GSplatData

    if isinstance(result, GSplatData):
        result.save(output_path, compress=compress)
        n = int(result.n_splats)
    else:  # a partition / tree node has no flat-matrix equivalent
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        write_gsplats_tree(output_path, result, compress=compress)
        n = int(getattr(result, "n_splats", 0))
    if verbose:
        aprint(f"Saved {n:,} splats")
        if output_path.exists():
            aprint(f"File size: {format_memory_size(output_path.stat().st_size)}")
    return n


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
        None, "--channel", help="Channel index for 5D OME-ZARR",
        rich_help_panel="Input selection",
    ),
    timepoint: Optional[int] = typer.Option(
        None, "--timepoint", help="Timepoint index for 5D OME-ZARR",
        rich_help_panel="Input selection",
    ),
    array_key: Optional[str] = typer.Option(
        None, "--array-key", help="Array key within .npz or .zarr",
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
        32, "--overlap", help="Overlap between tiles in voxels",
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
    # partition). additive -> partitioned topology; substitutive -> mosaic.
    recipe: Optional[str] = typer.Option(
        None,
        "-r",
        "--recipe",
        help="Per-part LOD for a tiled partition: additive (each part a "
        "prefix-sum ladder -> 'partitioned' topology) or substitutive (each "
        "part its own coarse<->fine lod group -> 'mosaic'). Requires a tiled "
        "fit (--tiling uniform/content) and a partition output (not --flat).",
        rich_help_panel="Per-part LOD",
    ),
    recipe_n_lods: Optional[int] = typer.Option(
        None,
        "--n-lods",
        help="[--recipe additive] Number of additive sub-LODs per part (default 4).",
        rich_help_panel="Per-part LOD",
    ),
    recipe_additive_method: Optional[str] = typer.Option(
        None,
        "-m",
        "--additive-method",
        help="[--recipe additive] greedy (default, (1-1/e)-optimal) or "
        "self_energy (cheap O(N log N) for very large parts).",
        rich_help_panel="Per-part LOD",
    ),
    recipe_breakpoints: Optional[str] = typer.Option(
        None,
        "-b",
        "--breakpoints",
        help="[--recipe additive] additive ladder breakpoints: 'equal-count' "
        "(default), 'counts:500,2000,...' or 'energy:0.5,0.9,...'.",
        rich_help_panel="Per-part LOD",
    ),
    recipe_compression_factor: Optional[int] = typer.Option(
        None,
        "-K",
        "--compression-factor",
        help="[--recipe substitutive] per-level coarsening factor K (default 4).",
        rich_help_panel="Per-part LOD",
    ),
    recipe_levels: Optional[int] = typer.Option(
        None,
        "-L",
        "--levels",
        help="[--recipe substitutive] number of substitutive levels L (default 3).",
        rich_help_panel="Per-part LOD",
    ),
    recipe_substitutive_method: Optional[str] = typer.Option(
        None,
        "--substitutive-method",
        help="[--recipe substitutive] auto (default) / kmeans-lloyd / greedy / "
        "greedy-lloyd.",
        rich_help_panel="Per-part LOD",
    ),
    recipe_coarsen_dims: Optional[str] = typer.Option(
        None,
        "--coarsen-dims",
        help="[--recipe substitutive] comma-separated center-column indices "
        "coarsening may merge over; the rest become hard barriers (default: all "
        "spatial dims).",
        rich_help_panel="Per-part LOD",
    ),
    recipe_lod_method: Optional[str] = typer.Option(
        None,
        "--lod-method",
        help="[--recipe substitutive] LOD switch threshold: extent (default, "
        "physically-anchored T·W/r) or count (legacy √N proxy).",
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
        False, "--denoise", help="Denoise volume before fitting (NLM)",
        rich_help_panel="Denoising",
    ),
    denoise_h: Optional[float] = typer.Option(
        None, "--denoise-h", help="Manual NLM h value (skip auto-calibration)",
        rich_help_panel="Denoising",
    ),
    denoise_2d: bool = typer.Option(
        False, "--denoise-2d", help="Use 2D NLM (slice-by-slice) instead of 3D",
        rich_help_panel="Denoising",
    ),
    denoise_patch_size: int = typer.Option(
        3, "--denoise-patch-size", help="NLM patch size (odd integer)",
        rich_help_panel="Denoising",
    ),
    denoise_search_distance: int = typer.Option(
        5, "--denoise-search-distance", help="NLM search window half-size",
        rich_help_panel="Denoising",
    ),
    denoise_backend: str = typer.Option(
        "auto", "--denoise-backend", help="NLM backend: auto/cuda/pytorch/skimage",
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
            resolved_tiling = _resolve_tiling(
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
                recipe_params = _build_fit_recipe_params(
                    recipe,
                    n_lods=recipe_n_lods,
                    additive_method=recipe_additive_method,
                    breakpoints=recipe_breakpoints,
                    compression_factor=recipe_compression_factor,
                    levels=recipe_levels,
                    substitutive_method=recipe_substitutive_method,
                    coarsen_dims=recipe_coarsen_dims,
                    lod_method=recipe_lod_method,
                    device=device,
                    volume_ndim=volume.ndim,
                )

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
                            seed_method=seed_method,
                            downscale=ds_arg,
                            channel=channel,
                            timepoint=timepoint,
                            array_key=array_key,
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
                        n_splats = _save_fit_output(
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
                    n_splats = _save_fit_output(
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
    try:
        import numpy as np

        from luxar.cli.gsplat_config import parse_shape
        from luxar.gsplats.gsplat_data import GSplatData

        with asection(f"Rendering: {input_path.name}"):
            with asection("Loading gsplat dataset"):
                data = GSplatData.load(input_path, include_stats=False)
                ndim = data.ndim
                aprint(f"Loaded {data.n_splats:,} splats ({ndim}D)")

            # Resolve truncation radius from dataset if not explicitly set
            if truncate is None:
                truncate = data.truncation_radius

            if shape is not None:
                output_shape = parse_shape(shape)
            else:
                mins = data.centers.min(axis=0)
                maxs = data.centers.max(axis=0)
                output_shape = tuple(int(maxs[i] - mins[i]) + 1 for i in range(ndim))
                aprint(f"Auto shape from bounding box: {output_shape}")

            with asection(f"Rendering to {output_shape}"):
                volume = data.render_to_volume(
                    shape=output_shape,
                    device=device,
                    truncate=truncate,
                )
                aprint(
                    f"Rendered: {volume.shape}, "
                    f"range [{volume.min():.4f}, {volume.max():.4f}]"
                )

            suffix = output_path.suffix.lower()
            with asection(f"Saving to {output_path.name}"):
                if suffix == ".npy":
                    np.save(str(output_path), volume)
                elif suffix in (".tiff", ".tif"):
                    try:
                        import tifffile
                    except ImportError:
                        aprint("tifffile not installed.")
                        aprint("Install with: pip install luxar[io]")
                        raise typer.Exit(1)
                    tifffile.imwrite(str(output_path), volume)
                else:
                    aprint(f"Unknown extension '{suffix}', saving as NumPy .npy")
                    np.save(str(output_path), volume)

                if output_path.exists():
                    aprint(
                        f"File size: {format_memory_size(output_path.stat().st_size)}"
                    )

        aprint(f"\nSaved: {output_path}")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


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
    ``lod --recipe additive`` (or ``substitutive`` / ``partitioned`` / ...) for a
    streaming-ready multi-resolution dataset. Use ``--fit-exponent`` to measure
    the density exponent that ``fit --tiling content`` consumes.

    Examples:
        luxar gsplat cal kidney_dapi.tiff cal.json
        luxar gsplat cal volume.zarr cal.json --n-grid 5 --k-max 128000 --preset draft
        luxar gsplat cal volume.zarr cal.json --k-grid '1000,4000,16000,64000,256000'
        luxar gsplat cal volume.tiff cal.json --pdf report.pdf --keep-fits fits/
    """
    try:
        import math

        from luxar.cli.gsplat_config import load_fit_config, load_volume
        from luxar.gsplats.calibration import (
            build_k_grid,
            calibrate,
            select_calibration_region,
        )

        # 1. Resolve K grid
        explicit: Optional[list[int]] = None
        if k_grid is not None:
            explicit = [int(x.strip()) for x in k_grid.split(",") if x.strip()]
        ks = build_k_grid(
            explicit=explicit,
            n_points=n_grid,
            k_min=k_min,
            k_max=k_max,
            progression=progression,
            power=power,
        )

        # 2. Load volume (reuse the loader used by `compare` and `fit`)
        with asection(f"Calibration: {input_path.name}"):
            with asection("Loading volume"):
                volume = load_volume(
                    input_path,
                    channel=channel,
                    timepoint=timepoint,
                    array_key=array_key,
                    axes=axes,
                )

            # Optionally calibrate on a content-rich sub-region (the manuscript
            # itself crops to ~20 M voxels; this automates that at tile scale).
            original_shape = list(volume.shape)
            # Keep the pre-crop volume so --fit-exponent can select its own
            # per-scale regions from the full data (independent of --auto-region).
            volume_full = volume
            region_info: Optional[dict] = None
            if auto_region:
                from dataclasses import asdict as _asdict

                with asection("Selecting content-rich calibration region"):
                    volume, region = select_calibration_region(
                        volume,
                        region_size=region_size,
                        strategy=region_strategy,
                        feature=feature_metric,
                    )
                    region_info = _asdict(region)
                    aprint(
                        f"Region [{region.strategy}]: origin={region.origin} "
                        f"size={region.size} n_features={region.n_features}"
                    )

            aprint(
                f"Mask: {mask_fraction * 100:.1f}% (seed={mask_seed}); donut radius=1"
            )
            aprint(f"K grid ({len(ks)} points): {ks}")
            aprint(f"K*-metric: {k_star_metric}; feature metric: {feature_metric}")

            # 3. Build fit kwargs from preset + YAML config + CLI overrides
            fit_kwargs = load_fit_config(
                preset=preset,
                config_path=config,
                cli_overrides={"device": device},
            )
            # Calibration runs many fits — keep them quiet
            fit_kwargs["verbose"] = False

            if keep_fits is not None:
                keep_fits = Path(keep_fits)
                keep_fits.mkdir(parents=True, exist_ok=True)
                aprint(f"Per-K fits will be saved under {keep_fits}")

            # 4. Run sweep
            def _on_progress(i: int, n: int, msg: str) -> None:
                aprint(f"  [{i + 1}/{n}] {msg}")

            with asection(f"Sweeping {len(ks)} fits"):
                t0 = time.perf_counter()
                result = calibrate(
                    volume,
                    k_grid=ks,
                    fit_kwargs=fit_kwargs,
                    mask_seed=mask_seed,
                    mask_fraction=mask_fraction,
                    keep_fits=keep_fits,
                    progress_callback=_on_progress,
                    k_star_metric=k_star_metric,
                    feature_method=feature_metric,
                    saturation_exponent=saturation_exponent,
                    compute_rd_model=rd_model,
                )
                elapsed = time.perf_counter() - t0

            # Record region provenance (calibrate() works on whatever array it
            # is handed; the CLI owns the crop, so it stamps the provenance).
            result.original_volume_shape = original_shape
            if region_info is not None:
                result.calibration_region = region_info

            # 4b. Optional multi-scale fit of the saturation exponent alpha.
            #     Calibrates K* at several region scales and regresses log K* on
            #     log n_features; the fitted alpha overrides the assumed default
            #     in splat_density (which the planner / fit --tiling content read).
            if fit_exponent:
                from dataclasses import asdict as _asdict_fit

                from luxar.gsplats.calibration import calibrate_saturation_exponent

                scales = (
                    [int(x) for x in exponent_scales.split(",") if x.strip()]
                    if exponent_scales
                    else [128, 192, 256]
                )
                with asection(
                    f"Fitting saturation exponent over {len(scales)} scale(s)"
                ):
                    aprint(
                        f"⚠ --fit-exponent runs {len(scales)} extra K-sweeps "
                        f"(scales={scales}); this multiplies runtime accordingly."
                    )
                    efit = calibrate_saturation_exponent(
                        volume_full,
                        scales,
                        k_grid=ks,
                        fit_kwargs=fit_kwargs,
                        feature_method=feature_metric,
                        region_strategy=region_strategy,
                        k_star_metric=k_star_metric,
                        mask_seed=mask_seed,
                        mask_fraction=mask_fraction,
                        progress_callback=_on_progress,
                    )
                if efit is None:
                    aprint(
                        "⚠ exponent fit failed (need ≥2 scales with distinct "
                        f"feature counts); keeping α={saturation_exponent}."
                    )
                else:
                    # Always record the fit for provenance/inspection.
                    result.exponent_fit = _asdict_fit(efit)
                    r2 = efit.r_squared
                    r2_str = "n/a" if not math.isfinite(r2) else f"{r2:.3f}"
                    alpha_usable = math.isfinite(efit.alpha) and efit.alpha > 0.0
                    if not alpha_usable:
                        # A non-positive / non-finite slope means the power law
                        # didn't hold (e.g. K* flat across scales → α≈0, which
                        # would collapse predict_k to a constant). Keep the default
                        # rather than silently disabling the density transfer.
                        aprint(
                            f"⚠ degenerate exponent (α={efit.alpha:.3g}, R²={r2_str}, "
                            f"{efit.n_distinct} distinct scales); keeping default "
                            f"α={saturation_exponent}. (Fit recorded for inspection.)"
                        )
                    else:
                        if result.splat_density is not None:
                            result.splat_density["saturation_exponent"] = efit.alpha
                        aprint(
                            f"Fitted α={efit.alpha:.3f} (R²={r2_str}, "
                            f"{efit.n_distinct} distinct scales); was "
                            f"{saturation_exponent}"
                        )
                        if not math.isfinite(r2) or r2 < 0.5:
                            aprint(
                                "⚠ low/unassessable confidence (need ≥3 distinct "
                                "scales with varied feature counts and K*); treat "
                                "α as provisional — consider more/varied "
                                "--exponent-scales or keep the default."
                            )

            # 5. Write JSON
            with asection("Writing results"):
                output_json.parent.mkdir(parents=True, exist_ok=True)
                result.to_json(output_json)
                aprint(f"Wrote {output_json}")

            # 6. Optional PDF
            if pdf_report is not None:
                with asection("Generating PDF report"):
                    try:
                        from luxar.gsplats.calibration_report import (
                            render_calibration_report,
                        )

                        pdf_report.parent.mkdir(parents=True, exist_ok=True)
                        render_calibration_report(
                            result=result,
                            volume=volume,
                            output_path=pdf_report,
                            splat_paths=result.splat_paths,
                        )
                        aprint(f"Wrote {pdf_report}")
                    except ImportError as exc:
                        aprint(
                            f"Skipping PDF report: optional dependency missing ({exc})"
                        )
                        aprint(
                            "Install matplotlib to enable --pdf: pip install matplotlib"
                        )

        # 7. Print formatted table
        if not quiet:
            aprint("\n" + "═" * 64)
            aprint(f"  CALIBRATION  —  {input_path.name}")
            aprint("═" * 64)
            aprint(
                f"  Volume:         {tuple(result.volume_shape)} {result.volume_dtype}"
            )
            # Surface region provenance: under --auto-region the Volume / PSNR_full
            # below are CROP-scoped, not whole-volume (M12).
            if result.calibration_region is not None:
                reg = result.calibration_region
                aprint(
                    f"  Region:         [{reg['strategy']}] origin={reg['origin']} "
                    f"of full {tuple(result.original_volume_shape or [])}"
                )
                aprint("                  (Volume / PSNR_full above are for this crop)")
            sigma = result.noise_floor.sigma_hat
            ceil_db = result.noise_floor.psnr_max_db
            sigma_str = f"{sigma:.4f}" if math.isfinite(sigma) else "—"
            ceil_str = (
                f"{ceil_db:.1f} dB"
                if math.isfinite(ceil_db)
                else (">60 dB" if math.isinf(ceil_db) else "—")
            )
            aprint(f"  Noise floor:    σ = {sigma_str}, PSNR ceiling = {ceil_str}")
            aprint("")
            # The headline / table marker track the metric the user selected
            # (falls back to the legacy min--max peak when no metric switch).
            selected_peak = result.held_out_peak_selected or result.held_out_peak
            aprint(
                "    K_req     K_eff    PSNR_train  PSNR_held-out   PSNR_full   SSIM_full   fit (s)"
            )
            aprint("    " + "-" * 76)
            for i, k_req in enumerate(result.k_values_requested):
                k_eff = result.k_values_effective[i]
                pt = result.train_psnr_db[i]
                ph = result.held_out_psnr_db[i]
                pf = result.full_psnr_db[i]
                sf = result.full_ssim[i]
                ft = result.fit_times_seconds[i]

                def _f(x: float) -> str:
                    if math.isnan(x):
                        return "  nan"
                    if math.isinf(x):
                        return "  inf"
                    return f"{x:6.2f}"

                marker = "★" if k_req == selected_peak.k_star else " "
                aprint(
                    f"  {marker} {k_req:7d}  {k_eff:7d}    {_f(pt)} dB     {_f(ph)} dB    {_f(pf)} dB    {sf:5.3f}    {ft:6.1f}"
                )
            aprint("")
            # Headline = the K* under the metric the user actually selected.
            aprint(
                f"  ★ Recommended K* = {selected_peak.k_star:,}  "
                f"(metric: {result.k_star_metric}, type: {selected_peak.type}, "
                f"confidence: {selected_peak.confidence_db:.2f} dB)"
            )
            if result.held_out_peak_selected is not None:
                aprint(
                    f"    (psnr_minmax K* = {result.held_out_peak.k_star:,}, "
                    f"type: {result.held_out_peak.type})"
                )
            if result.splat_density is not None:
                sd = result.splat_density
                aprint(
                    f"  Density: {sd['k_star_reference']:,} splats / "
                    f"{sd['n_features_reference']:,} {sd['feature_method']} features"
                    f"  →  K ~ features^{sd['saturation_exponent']:.2f}"
                )
            # Regime warnings — warn-by-default, no behaviour change
            sig = result.noise_floor.sigma_hat
            if math.isfinite(sig) and sig < 1e-4:
                aprint(
                    "  ⚠ σ̂≈0 (noise-free/deconvolved): the blind-spot peak may not "
                    "appear; prefer --k-star-metric gain (and --auto-region)."
                )
            if result.not_converged:
                aprint(
                    "  ⚠ held-out curve still climbing at K_max (not converged): "
                    "extend --k-max or use the R-D-model extrapolation."
                )
            aprint(f"  Total wall-clock: {elapsed:.1f} s")
            aprint("═" * 64)

    except typer.Exit:
        raise
    except Exception as exc:
        aprint(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def register_fitting_commands(app: typer.Typer) -> None:
    """Register the fitting commands onto ``app_gsplat``."""
    # Workflow order: fit and cal (the core pre-/fit steps) lead; render and
    # denoise (utilities) follow.
    app.command("fit")(fit_volume)
    app.command("cal")(calibrate_command)
    app.command("render")(render_to_file)
    app.command("denoise")(denoise_volume_cmd)
