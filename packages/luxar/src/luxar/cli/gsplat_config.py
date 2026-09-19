"""Configuration system for gsplat CLI commands.

Provides:
- Fitting presets (draft/standard/hifi/ultra/n2s)
- YAML config loading with priority chain
- Commented YAML config dump
- Volume file loaders (.npy, .npz, .tiff, .zarr, imageio fallback)
- OME-Zarr shape discovery
- Utility parsers for CLI arguments
"""

from __future__ import annotations

import inspect
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import yaml
from arbol import aprint

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class FitPreset(str, Enum):
    """Fitting quality presets."""

    DRAFT = "draft"
    STANDARD = "standard"
    HIFI = "hifi"
    ULTRA = "ultra"


# All presets share the manuscript's blind-spot / Noise2Self protocol as their
# baseline (``cull_retention=0.999`` — i.e. no silent post-fit culling) and
# differ in optimiser budget (``n_iters`` + ``early_stop_patience``) plus
# ``max_eccentricity`` (10 → 20 from ``draft`` to ``ultra``).
#
# Why this matters: the previous presets defaulted to the fitter's
# ``cull_retention=0.95``, which silently dropped 5% of splats by amplitude
# after every fit. Combined with too few iterations at high K, this made
# ``luxar gsplat cal`` report artificially rising held-out PSNR ("signal
# limited") instead of the true plateau / overfit curve the manuscript's
# supp_doc/splat_count_vs_quality/run_noise2self.py recorded.  Aligning all
# presets to the paper's profile fixes the cal protocol; users picking a
# faster preset trade convergence quality but not protocol validity.
#
# ``n2s`` is the canonical name for the protocol — ``ultra``'s optimiser budget,
# but at ``max_eccentricity=10``, kept explicit so it's clear which preset is the
# "paper reference".
PRESETS: Dict[str, Dict[str, Any]] = {
    "draft": {
        "n_iters": 2_000,
        "early_stop_patience": 200,
        "max_eccentricity": 10.0,
        "cull_retention": 0.999,
    },
    "standard": {
        "n_iters": 5_000,
        "early_stop_patience": 300,
        "max_eccentricity": 10.0,
        "cull_retention": 0.999,
    },
    "hifi": {
        "n_iters": 10_000,
        "early_stop_patience": 400,
        "max_eccentricity": 15.0,
        "cull_retention": 0.999,
    },
    "ultra": {
        "n_iters": 20_000,
        "early_stop_patience": 500,
        "max_eccentricity": 20.0,
        "cull_retention": 0.999,
    },
    # Canonical alias for the manuscript's blind-spot protocol; ``ultra``'s budget
    # (n_iters=20000, early_stop_patience=500, cull_retention=0.999) but at
    # max_eccentricity=10 (``ultra`` uses 20), and the default for
    # ``luxar gsplat cal``.  Kept named so users can pick the "paper" preset
    # explicitly when reproducing supp_doc results.
    "n2s": {
        "n_iters": 20_000,
        "early_stop_patience": 500,
        "max_eccentricity": 10.0,
        "cull_retention": 0.999,
    },
}

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def get_fit_defaults() -> Dict[str, Any]:
    """Extract default parameter values from fit_gaussian_splats signature."""
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats

    sig = inspect.signature(fit_gaussian_splats)
    defaults = {}
    for name, param in sig.parameters.items():
        if name == "V":
            continue
        if param.default is not inspect.Parameter.empty:
            defaults[name] = param.default
    return defaults


def load_fit_config(
    preset: Optional[str] = None,
    config_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    command_defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a merged fit config from preset, YAML file, and CLI overrides.

    Priority chain (highest wins):
        CLI flags > YAML config > preset > command defaults > function defaults

    Args:
        preset: Preset name ("draft", "standard", "hifi", "ultra", "n2s") or None
        config_path: Path to YAML config file or None
        cli_overrides: Dict of CLI-provided values (None values are ignored)
        command_defaults: Per-command defaults that displace the harvested
            function defaults but yield to preset / YAML / CLI (None values are
            ignored, so a caller can pass a sentinel-free dict). Lets one command
            carry a different baseline from a bare ``fit`` — e.g. the
            content-tiling path's near-lossless ``cull_retention``.

    Returns:
        Merged config dict ready to pass as ``**kwargs`` to fit_gaussian_splats
    """
    # Start with function defaults
    config = get_fit_defaults()

    # Layer per-command defaults (skip None so a caller may pass a sparse dict)
    if command_defaults:
        for key, value in command_defaults.items():
            if value is not None:
                config[key] = value

    # Layer preset
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Choose from: {list(PRESETS.keys())}"
            )
        config.update(PRESETS[preset])

    # Layer YAML config
    if config_path is not None:
        yaml_config = _load_yaml_config(config_path)
        config.update(yaml_config)

    # Layer CLI overrides (skip None values — Typer sentinel for "not provided")
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                config[key] = value

    # Remove 'seeds' — handled separately by the CLI.
    config.pop("seeds", None)

    # Note: we do NOT whitelist to only known params, because
    # fit_gaussian_splats accepts **seed_kwargs (e.g., num_scales,
    # percentile_thresh, spacing) which are valid but not in the
    # explicit parameter list. All config keys pass through.

    return config


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load and validate a YAML config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(data)}")

    return data


# ---------------------------------------------------------------------------
# Config dump
# ---------------------------------------------------------------------------


def dump_default_config(preset: str = "standard") -> str:
    """Generate a fully-commented YAML config with defaults from a preset.

    Args:
        preset: Base preset to use for default values

    Returns:
        YAML string with all parameters and comments
    """
    p = PRESETS.get(preset, PRESETS["standard"])
    d = get_fit_defaults()
    # Merge: preset overrides defaults
    vals = {**d, **p}

    def _fmt(v: Any) -> str:
        """Format a Python value as YAML."""
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(str(x) for x in v) + "]"
        if isinstance(v, float):
            return f"{v}"
        return str(v)

    lines = [
        "# ============================================================",
        "# Luxar Gaussian Splat Fitting Configuration",
        "# ============================================================",
        f"# Base preset: {preset}",
        "# Priority: CLI flags > YAML config > preset > command defaults > function defaults",
        "#",
        "# Usage:",
        "#   luxar gsplat fit volume.npy output.gsplats.zarr --config this_file.yaml",
        "#   luxar gsplat fit volume.npy output.gsplats.zarr --preset hifi --config overrides.yaml",
        "#",
        "# Seed generation kwargs (num_scales, percentile_thresh, spacing, etc.)",
        "# can also be set here and will be forwarded to the seed generator.",
        "",
        "# --- Basic Parameters ---",
        f"n_iters: {_fmt(vals.get('n_iters'))}            # Max optimization iterations",
        f"lr: {_fmt(vals.get('lr'))}                      # Adam learning rate",
        f'loss_type: "{vals.get("loss_type", "l1")}"        # Loss function: l1, mse, or poisson',
        f'seed_method: "{vals.get("seed_method", "auto")}"  # Seed method: auto, edges, grid, decomposition',
        "",
        "# --- Preprocessing ---",
        f"norm_percentile: {_fmt(vals.get('norm_percentile'))}  # Percentile clipping (0=full range, >0=robust)",
        f'floor: "{vals.get("floor", "auto")}"                # Background/DC suppression: auto | pN (e.g. p10) | <float> | none',
        f"downscale: {_fmt(vals.get('downscale'))}            # Downsample by integer factor (null=disabled, e.g. 4 or [1,4,4])",
        "",
        "# --- Regularization ---",
        f"asymmetric_penalty: {_fmt(vals.get('asymmetric_penalty'))}  # Over-prediction penalty (null=disabled)",
        f"l1_amp: {_fmt(vals.get('l1_amp'))}              # L1 on amplitudes (null=auto: 0.1*lr)",
        f"l1_diag: {_fmt(vals.get('l1_diag'))}            # L1 on Cholesky diagonals (null=auto: 0.01*lr)",
        "",
        "# --- Shape Constraints ---",
        f"sigma_min_diag: {_fmt(vals.get('sigma_min_diag'))}  # Min Cholesky diagonal",
        f"sigma_max_diag: {_fmt(vals.get('sigma_max_diag'))}  # Max Cholesky diagonal (null=unbounded)",
        f"amp_max: {_fmt(vals.get('amp_max'))}            # Max amplitude (null=auto: 1.0)",
        f"max_eccentricity: {_fmt(vals.get('max_eccentricity'))}  # Max axis ratio (null=no constraint)",
        f"truncate: {_fmt(vals.get('truncate'))}          # Truncation radius in sigma",
        "",
        "# --- Convergence ---",
        f"max_abs_error: {_fmt(vals.get('max_abs_error'))}  # Absolute error threshold (null=auto: 0.01)",
        f"rel_l2_target: {_fmt(vals.get('rel_l2_target'))}  # Relative L2 threshold (null=disabled)",
        f"gradient_clip: {_fmt(vals.get('gradient_clip'))}   # Gradient norm clipping (null=disabled)",
        f'scheduler_type: "{vals.get("scheduler_type", "plateau")}"  # LR scheduler: plateau or exponential',
        f"patience: {_fmt(vals.get('patience'))}          # Iterations before LR reduction",
        f"lr_reduction_factor: {_fmt(vals.get('lr_reduction_factor'))}  # LR multiplier on plateau",
        f"early_stop_patience: {_fmt(vals.get('early_stop_patience'))}  # Stop after N iters without improvement",
        "",
        "# --- Dynamic Operations ---",
        f"enable_dynamic_ops: {_fmt(vals.get('enable_dynamic_ops'))}  # Enable splat relocation during optimization",
        f"dynamic_ops_verbose: {_fmt(vals.get('dynamic_ops_verbose'))}  # Verbose logging for dynamic ops",
        "",
        "# --- Post-Processing ---",
        f"cull_retention: {_fmt(vals.get('cull_retention', 0.95))}  # Post-fit cumulative culling (0-1, null=disabled)",
        f"voxel_footprint_correction: {_fmt(vals.get('voxel_footprint_correction'))}  # Inflate covariances by voxel footprint",
        "",
        "# --- Boundary Containment ---",
        f"boundary_penalty: {_fmt(vals.get('boundary_penalty'))}  # Boundary penalty weight (null=disabled)",
        f"clip_to_bounds: {_fmt(vals.get('clip_to_bounds'))}  # Hard clip splats to volume bounds",
        "",
        "# --- Anisotropic Voxels ---",
        f"voxel_size: {_fmt(vals.get('voxel_size'))}      # Physical spacing (null=isotropic)",
        f'output_space: "{vals.get("output_space", "real")}"  # Output coords: real or voxel',
        "",
        "# --- Performance ---",
        f"sort_splats_enabled: {_fmt(vals.get('sort_splats_enabled'))}  # Periodic Morton-code sorting for GPU cache locality",
        f"sort_splats_interval: {_fmt(vals.get('sort_splats_interval'))}  # Sort every N iterations (also sorts at iteration 0)",
        "",
        "# --- Hardware ---",
        f"use_metal: {_fmt(vals.get('use_metal'))}        # Metal acceleration (macOS Apple Silicon)",
        f"use_cuda: {_fmt(vals.get('use_cuda'))}          # Custom CUDA kernels (NVIDIA GPUs)",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Volume loading, OME-Zarr discovery, and dimension inference now live in the
# domain layer (luxar.io / luxar.core); the gsplat CLI re-exports them so its
# command modules keep a single import site. `load_volume` is wrapped to convert
# the domain ImportError (missing optional reader) into a clean Typer exit.
# ---------------------------------------------------------------------------

from luxar.core.dimension_inference import (  # noqa: E402
    build_dimensions_from_data,
)
from luxar.io.ome_zarr import (  # noqa: E402
    OMEZarrInfo,
    discover_ome_zarr_shape,
)
from luxar.io.volume import (  # noqa: E402
    decode_flat_channel_index,
)
from luxar.io.volume import (  # noqa: E402
    load_volume as _load_volume_impl,
)

__all__ = [
    "OMEZarrInfo",
    "build_dimensions_from_data",
    "decode_flat_channel_index",
    "discover_ome_zarr_shape",
    "dump_default_config",
    "get_fit_defaults",
    "load_fit_config",
    "load_volume",
    "parse_hex_color",
    "parse_seeds",
    "parse_shape",
    "FitPreset",
    "PRESETS",
]


def load_volume(
    path: Path,
    channel: Optional[int] = None,
    timepoint: Optional[int] = None,
    array_key: Optional[str] = None,
    axes: Optional[str] = None,
    info: Optional[Dict[str, Any]] = None,
    region: Optional[Tuple[slice, ...]] = None,
) -> np.ndarray:
    """CLI wrapper around :func:`luxar.io.volume.load_volume`.

    Converts the domain-layer :class:`ImportError` (raised when an optional
    reader such as tifffile/imageio is missing) into a clean ``typer.Exit(1)``
    with an install hint, so the CLI shows a friendly message instead of a
    traceback. All loading behaviour is identical to the domain function.
    """
    import typer

    try:
        return _load_volume_impl(
            path,
            channel=channel,
            timepoint=timepoint,
            array_key=array_key,
            axes=axes,
            info=info,
            region=region,
        )
    except ImportError as exc:
        aprint(str(exc))
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------


def parse_seeds(value: Optional[str]) -> Union[int, float, None]:
    """Parse the --seeds CLI argument.

    Args:
        value: "auto" | None | integer string | float string

    Returns:
        None (auto), int (exact count), or float (compression ratio)
    """
    if value is None or value.lower() == "auto":
        return None
    if "." in value:
        ratio = float(value)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"Compression ratio must be in (0, 1], got {ratio}")
        return ratio
    return int(value)


def parse_hex_color(s: str) -> Tuple[float, float, float]:
    """Parse a hex color string to RGB floats.

    Args:
        s: Hex color like "#ff0080" or "ff0080"

    Returns:
        Tuple of (R, G, B) floats in [0, 1]
    """
    s = s.strip().lstrip("#")
    if not re.match(r"^[0-9a-fA-F]{6}$", s):
        raise ValueError(f"Invalid hex color: '#{s}'. Expected format: #rrggbb")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


def parse_shape(s: str) -> Tuple[int, ...]:
    """Parse a comma-separated shape string.

    Args:
        s: Shape like "128,128,128"

    Returns:
        Tuple of ints
    """
    parts = [p.strip() for p in s.split(",")]
    return tuple(int(p) for p in parts)
