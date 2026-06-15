"""``luxar gsplat fitting`` commands (extracted from gsplat_commands.py).

Each command is a plain function; ``register_fitting_commands(app)`` wires them onto
the shared ``app_gsplat`` Typer (package-refactor-plan P3/P4/P6).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

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
    ``luxar gsplat batch plan --denoise``.

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
        None, "--channel", help="Channel index for 5D OME-ZARR"
    ),
    timepoint: Optional[int] = typer.Option(
        None, "--timepoint", help="Timepoint index for 5D OME-ZARR"
    ),
    array_key: Optional[str] = typer.Option(
        None, "--array-key", help="Array key within .npz or .zarr"
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
    tiled: bool = typer.Option(
        False, "--tiled", help="Enable tiled fitting for large volumes"
    ),
    tile_size: int = typer.Option(
        256, "--tile-size", help="Tile size in voxels (per axis)"
    ),
    tile_overlap: int = typer.Option(
        32, "--overlap", help="Overlap between tiles in voxels"
    ),
    tile: Optional[str] = typer.Option(
        None,
        "--tile",
        help="Fit single tile N/M (e.g., '3/16' = tile index 3 of 16 total)",
    ),
    # Progressive fitting
    progressive: bool = typer.Option(
        False,
        "--progressive",
        help="Enable progressive fitting: fit in multiple passes on residuals, "
        "producing a multi-LOD result. Each pass adds detail to the previous. "
        "Tip: for tiled batch jobs, combine with --parallel to improve GPU utilization.",
    ),
    max_splats_per_pass: int = typer.Option(
        5000,
        "--splats-per-pass",
        help="Maximum splats per progressive pass (actual may be fewer after culling)",
    ),
    psnr_patience: float = typer.Option(
        0.5,
        "--psnr-patience",
        help="Stop progressive fitting if ΔPSNR between passes < this value (dB)",
    ),
    max_passes: Optional[int] = typer.Option(
        None,
        "--max-passes",
        help="Maximum number of progressive passes (default: unlimited, stops by budget or PSNR patience)",
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
        False, "--denoise", help="Denoise volume before fitting (NLM)"
    ),
    denoise_h: Optional[float] = typer.Option(
        None, "--denoise-h", help="Manual NLM h value (skip auto-calibration)"
    ),
    denoise_2d: bool = typer.Option(
        False, "--denoise-2d", help="Use 2D NLM (slice-by-slice) instead of 3D"
    ),
    denoise_patch_size: int = typer.Option(
        3, "--denoise-patch-size", help="NLM patch size (odd integer)"
    ),
    denoise_search_distance: int = typer.Option(
        5, "--denoise-search-distance", help="NLM search window half-size"
    ),
    denoise_backend: str = typer.Option(
        "auto", "--denoise-backend", help="NLM backend: auto/cuda/pytorch/skimage"
    ),
) -> None:
    """Fit Gaussian splats to a volume.

    Reconstructs an n-dimensional image/volume as a set of oriented Gaussian
    splats. Use presets for quick configuration or a YAML config file for
    full control over all ~35 parameters.

    Presets:
        draft    - Fast preview (500 iters)
        standard - Balanced quality/speed (3000 iters)
        hifi     - High quality (6000 iters)
        ultra    - Maximum quality (10000 iters)

    Examples:
        luxar gsplat fit volume.npy splats.gsplats.zarr --preset draft --seeds 1000

        luxar gsplat fit volume.tiff splats.gsplats.zarr --preset standard --seeds 8000

        luxar gsplat fit --dump-config --preset hifi > config.yaml
        luxar gsplat fit volume.zarr splats.gsplats.zarr --config config.yaml

        luxar gsplat fit data.zarr splats.gsplats.zarr --channel 1 --timepoint 0

        luxar gsplat fit large.zarr splats.gsplats.zarr --tiled --tile-size 256 --overlap 32

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
                volume = load_volume(input_path, channel, timepoint, array_key)
                aprint(f"Volume shape: {volume.shape}")

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
            with asection(f"Saving to {output_path.name}"):
                result.save(output_path, compress=compress)
                n_splats = result.n_splats
                aprint(f"Saved {n_splats:,} splats")
                if output_path.exists():
                    aprint(
                        f"File size: {format_memory_size(output_path.stat().st_size)}"
                    )

        time_s = result.stats.get("time_seconds", 0)
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
    ``lod additive`` (or ``lod substitutive``) for a streaming-ready
    multi-resolution dataset.

    Examples:
        luxar gsplat cal kidney_dapi.tiff cal.json
        luxar gsplat cal volume.zarr cal.json --n-grid 5 --k-max 128000 --preset draft
        luxar gsplat cal volume.zarr cal.json --k-grid '1000,4000,16000,64000,256000'
        luxar gsplat cal volume.tiff cal.json --pdf report.pdf --keep-fits fits/
    """
    try:
        import math

        from luxar.cli.gsplat_config import load_fit_config, load_volume
        from luxar.gsplats.calibration import build_k_grid, calibrate

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
                )

            aprint(
                f"Mask: {mask_fraction * 100:.1f}% (seed={mask_seed}); donut radius=1"
            )
            aprint(f"K grid ({len(ks)} points): {ks}")

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
                )
                elapsed = time.perf_counter() - t0

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

                marker = "★" if k_req == result.held_out_peak.k_star else " "
                aprint(
                    f"  {marker} {k_req:7d}  {k_eff:7d}    {_f(pt)} dB     {_f(ph)} dB    {_f(pf)} dB    {sf:5.3f}    {ft:6.1f}"
                )
            aprint("")
            aprint(
                f"  ★ Recommended K* = {result.held_out_peak.k_star:,}  "
                f"(type: {result.held_out_peak.type}, "
                f"confidence: {result.held_out_peak.confidence_db:.2f} dB)"
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
    app.command("denoise")(denoise_volume_cmd)
    app.command("fit")(fit_volume)
    app.command("render")(render_to_file)
    app.command("cal")(calibrate_command)
