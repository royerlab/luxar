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


def fit_volume(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around ``run_fit_volume``."""
    return run_fit_volume(*args, **kwargs)


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
    floor: Optional[str] = typer.Option(
        None,
        "--floor",
        help="Background floor / DC-offset suppression (default: auto), so K* "
        "is measured on floor-suppressed data (matches how you will fit). "
        "auto | pN | <float> | none. Unset lets a `floor:` in --config/preset "
        "apply, else defaults to auto. See `gsplat fit --help`.",
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
        floor=floor,
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
    app.command("fit")(run_fit_volume)
    app.command("cal")(calibrate_command)
    app.command("render")(render_to_file)
    app.command("denoise")(denoise_volume_cmd)
