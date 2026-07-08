"""Configuration system for gsplat CLI commands.

Provides:
- Fitting presets (draft/standard/hifi/ultra)
- YAML config loading with priority chain
- Commented YAML config dump
- Volume file loaders (.npy, .npz, .tiff, .zarr, imageio fallback)
- OME-Zarr shape discovery
- Utility parsers for CLI arguments
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
# vary only in optimiser budget (``n_iters`` + ``early_stop_patience``).
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
# ``n2s`` is the canonical name for the protocol — same numbers as ``ultra``,
# kept explicit so it's clear which preset is the "paper reference".
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
    # Canonical alias for the manuscript's blind-spot protocol; identical to
    # ``ultra`` (n_iters=20000, early_stop_patience=500, cull_retention=0.999)
    # and the default for ``luxar gsplat cal``.  Kept named so users can pick
    # the "paper" preset explicitly when reproducing supp_doc results.
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
) -> Dict[str, Any]:
    """Build a merged fit config from preset, YAML file, and CLI overrides.

    Priority chain (highest wins):
        CLI flags > YAML config > preset > function defaults

    Args:
        preset: Preset name ("draft", "standard", "hifi", "ultra") or None
        config_path: Path to YAML config file or None
        cli_overrides: Dict of CLI-provided values (None values are ignored)

    Returns:
        Merged config dict ready to pass as **kwargs to fit_gaussian_splats
    """
    # Start with function defaults
    config = get_fit_defaults()

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
        "# Priority: CLI flags > YAML config > preset > function defaults",
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
# Volume loading
# ---------------------------------------------------------------------------


def decode_flat_channel_index(
    channel: int, channel_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Decode a flat channel task index into folded channel-axis coordinates.

    For data with multiple non-spatial, channel-like axes (for example
    ``camera`` and ``channel``), batch planning treats each axis combination as
    one flat channel task. This helper uses row-major order to map the flat
    index back to per-axis coordinates.
    """
    if channel < 0:
        raise ValueError(f"channel index must be non-negative, got {channel}")
    if not channel_shape:
        if channel == 0:
            return ()
        raise ValueError("channel index > 0 is invalid when there are no channel axes")

    total = 1
    for size in channel_shape:
        if size <= 0:
            raise ValueError(
                f"channel axis sizes must be positive, got {channel_shape}"
            )
        total *= size
    if channel >= total:
        raise ValueError(
            f"flat channel index {channel} is out of range for channel_shape={channel_shape} "
            f"(total={total})"
        )

    coords: List[int] = []
    remaining = channel
    for dim_size in reversed(channel_shape):
        coords.insert(0, remaining % dim_size)
        remaining //= dim_size
    return tuple(coords)


def _apply_axes_spec(
    arr: np.ndarray,
    axes: str,
    channel: Optional[int],
    timepoint: Optional[int],
) -> np.ndarray:
    """Collapse a non-canonically-ordered nD array to its spatial volume.

    ``axes`` is a comma-separated label per array dimension (e.g.
    ``"z,c,y,x"`` or ``"t,z,y,x"``). Recognised: time (``t``/``time``),
    channel (``c``/``channel``/``ch``/``camera``/``cam``), spatial
    (``z``/``y``/``x``/``depth``/``height``/``width``). Each time/channel axis is
    indexed (by ``timepoint``/``channel``, default 0) and dropped; the remaining
    spatial axes are kept in their given order. This is the single-volume
    counterpart of ``batch-fit submit --axes`` — it lets ``fit``/``cal`` consume
    data whose axis order isn't the assumed TCZYX/CZYX/ZYX.
    """
    labels = [a.strip().lower() for a in axes.split(",") if a.strip() != ""]
    if len(labels) != arr.ndim:
        raise ValueError(
            f"--axes has {len(labels)} labels but the array is {arr.ndim}D "
            f"(shape {arr.shape}); give one label per dimension."
        )

    def _kind(label: str) -> str:
        if label in ("t", "time"):
            return "t"
        if label in ("c", "channel", "ch", "camera", "cam"):
            return "c"
        if label in ("z", "y", "x", "depth", "height", "width"):
            return "s"
        raise ValueError(
            f"--axes label {label!r} not recognised; use time/t, "
            "channel/c/ch/camera/cam, or z/y/x (depth/height/width)."
        )

    kinds = [_kind(label) for label in labels]
    index: list = [slice(None)] * arr.ndim
    for i, k in enumerate(kinds):
        if k in ("t", "c"):
            which, idx = (
                ("--timepoint", timepoint) if k == "t" else ("--channel", channel)
            )
            idx = 0 if idx is None else int(idx)
            size = arr.shape[i]
            if not (0 <= idx < size):
                raise ValueError(
                    f"{which} index {idx} is out of range for the '{labels[i]}' "
                    f"axis of size {size} (valid 0..{size - 1})."
                )
            index[i] = idx
    return np.asarray(arr[tuple(index)])


def load_volume(
    path: Path,
    channel: Optional[int] = None,
    timepoint: Optional[int] = None,
    array_key: Optional[str] = None,
    axes: Optional[str] = None,
) -> np.ndarray:
    """Load a volume from various file formats.

    Supported formats:
        .npy         — NumPy binary (numpy, base dep)
        .npz         — NumPy compressed (numpy, base dep)
        .zarr        — Zarr array/group, including OME-ZARR 5D (zarr, base dep)
        .tiff / .tif — TIFF image (tifffile, optional: pip install luxar[io])
        other        — Fallback via imageio (optional: pip install luxar[io])

    Args:
        path: Path to the volume file
        channel: Channel index for 4D/5D+ OME-ZARR data. If None, defaults
            to 0 when slicing is needed; for 4D arrays, ``None`` returns
            the array as-is.
        timepoint: Timepoint index for 5D+ OME-ZARR data. If None, defaults
            to 0 when slicing is needed; for 4D arrays, ``None`` returns
            the array as-is.
        array_key: Array key within .npz or .zarr files
        axes: Explicit per-dimension axis labels (e.g. ``"z,c,y,x"``) overriding
            the positional TCZYX/CZYX/ZYX heuristic — for data whose axis order
            differs. Time/channel axes are sliced (by ``timepoint``/``channel``)
            and dropped; spatial axes are kept in the given order.

    Returns:
        Volume as float32 numpy array (>=2D)
    """
    import typer

    suffix = path.suffix.lower()

    if suffix == ".npy":
        aprint(f"Loading NumPy array: {path.name}")
        volume = np.load(str(path))

    elif suffix == ".npz":
        aprint(f"Loading NumPy archive: {path.name}")
        with np.load(str(path)) as npz:
            keys = list(npz.keys())
            if array_key:
                if array_key not in keys:
                    raise ValueError(
                        f"Key '{array_key}' not found in {path.name}. "
                        f"Available keys: {keys}"
                    )
                volume = np.array(npz[array_key])
            else:
                volume = np.array(npz[keys[0]])
                if len(keys) > 1:
                    aprint(f"  Using first array '{keys[0]}' (available: {keys})")

    elif suffix == ".zarr" or (suffix == ".zip" and path.stem.endswith(".zarr")):
        # Handles both plain .zarr directories and .zarr.zip archives.
        # zarr natively supports ZipStore so no extraction needed. With an
        # explicit --axes the raw array is loaded and sliced by _apply_axes_spec
        # below (bypassing the positional TCZYX/CZYX heuristic).
        volume = _load_zarr_volume(
            path, channel, timepoint, array_key, raw=axes is not None
        )

    elif suffix in (".tiff", ".tif"):
        try:
            import tifffile
        except ImportError:
            aprint("tifffile not installed.")
            aprint("Install with: pip install luxar[io]")
            raise typer.Exit(1)
        aprint(f"Loading TIFF: {path.name}")
        volume = tifffile.imread(str(path))

    else:
        try:
            import imageio.v3 as iio
        except ImportError:
            aprint(f"Cannot load '{suffix}' files — imageio not installed.")
            aprint("Install with: pip install luxar[io]")
            raise typer.Exit(1)
        aprint(f"Loading via imageio: {path.name}")
        volume = iio.imread(str(path))

    # Explicit axis spec (overrides the positional heuristic): slice/drop the
    # time & channel axes and keep the spatial axes in the given order.
    if axes is not None:
        # Pass `volume` as-is (a lazy zarr array for .zarr inputs) so
        # _apply_axes_spec slices the time/channel axes BEFORE materializing —
        # do NOT np.asarray() here or a huge nD movie loads fully into RAM.
        volume = _apply_axes_spec(volume, axes, channel, timepoint)
        # The spec already fixed the shape (time/channel dropped, spatial kept) —
        # do NOT squeeze, or a deliberately-kept size-1 spatial axis (e.g. a
        # single z-plane via --axes z,y,x) would be silently dropped.
        volume = np.asarray(volume, dtype=np.float32)
    else:
        # Post-process: drop incidental size-1 dims from the positional heuristic.
        volume = np.asarray(volume, dtype=np.float32)
        volume = np.squeeze(volume)

    if volume.ndim < 2:
        raise ValueError(
            f"Volume must be at least 2D after squeezing, got {volume.ndim}D "
            f"with shape {volume.shape}"
        )

    aprint(f"  Shape: {volume.shape}, dtype: float32")
    return volume


def _find_all_arrays(group: Any, prefix: str = "") -> list:
    """Recursively find all arrays in a zarr group, returning (key_path, array) pairs."""
    import zarr

    results = []
    for k in group.keys():
        item = group[k]
        key_path = f"{prefix}/{k}" if prefix else k
        if isinstance(item, zarr.Array):
            results.append((key_path, item))
        elif isinstance(item, zarr.Group):
            results.extend(_find_all_arrays(item, key_path))
    return results


def _load_zarr_volume(
    path: Path,
    channel: Optional[int],
    timepoint: Optional[int],
    array_key: Optional[str],
    raw: bool = False,
) -> np.ndarray:
    """Load a volume from a zarr store, handling OME-ZARR conventions.

    With ``raw=True`` the full array is returned WITHOUT the positional
    TCZYX/CZYX slicing — the caller (``load_volume`` with an explicit ``--axes``)
    applies its own axis spec instead.
    """
    import zarr

    aprint(f"Loading Zarr: {path.name}")
    store = zarr.open(str(path), mode="r")

    # Navigate to the target array
    if isinstance(store, zarr.Array):
        arr = store
    elif isinstance(store, zarr.Group):
        if array_key is not None:
            try:
                arr = store[array_key]
            except KeyError:
                available = list(store.keys())
                raise ValueError(
                    f"Array key '{array_key}' not found in {path}. "
                    f"Available keys: {available}"
                )
            aprint(f"  Using array '{array_key}'")
        elif "0" in store:
            # OME-ZARR convention: "0" is highest resolution
            aprint("  Detected OME-ZARR layout (using resolution level '0')")
            arr = store["0"]
        else:
            # Find the largest array in the group, searching recursively
            # into sub-groups (e.g. h2afva/fused, mezzo/fused).
            arrays = _find_all_arrays(store)
            if not arrays:
                raise ValueError(f"No arrays found in zarr group: {path}")
            best_key = max(arrays, key=lambda kv: int(np.prod(kv[1].shape)))[0]
            arr = store[best_key]
            aprint(f"  Using array '{best_key}'")
    else:
        raise ValueError(f"Unexpected zarr object type: {type(store)}")

    shape = arr.shape
    ndim = len(shape)
    aprint(f"  Raw array shape: {shape} ({ndim}D)")

    if raw:
        # Explicit --axes path: hand back the LAZY zarr array (NOT np.array(arr)) so
        # the caller's _apply_axes_spec slices the requested timepoint/channel BEFORE
        # materializing — otherwise a whole nD movie (e.g. a 329-timepoint stack,
        # >1 TiB) would be loaded into RAM just to extract one 3D volume.
        return arr  # type: ignore[no-any-return]

    # Slice the array down to a 2D/3D spatial volume.
    # For nD data where ndim > 5, consume leading dimensions using
    # timepoint and channel indices (defaulting to 0 for each).
    if ndim >= 6:
        # Generic >5D: treat first dim as T, fold all leading non-spatial
        # dimensions before the final 3 spatial axes into one flat channel index.
        t = timepoint if timepoint is not None else 0
        remaining_non_spatial = ndim - 4  # -1 for time, -3 for spatial
        channel_shape = tuple(shape[1 : 1 + remaining_non_spatial])
        if channel is None:
            channel_coords = tuple(0 for _ in channel_shape)
        else:
            channel_coords = decode_flat_channel_index(channel, channel_shape)
        idx = [t, *channel_coords]
        aprint(f"  Slicing {ndim}D: indices {idx} → 3D spatial")
        volume = np.array(arr[tuple(idx)])
    elif ndim == 5:
        t = timepoint if timepoint is not None else 0
        c = channel if channel is not None else 0
        aprint(f"  Slicing 5D (TCZYX): T={t}, C={c}")
        volume = np.array(arr[t, c, :, :, :])
    elif ndim == 4:
        if channel is not None:
            aprint(f"  Slicing 4D (CZYX): C={channel}")
            volume = np.array(arr[channel, :, :, :])
        elif timepoint is not None:
            aprint(f"  Slicing 4D (TZYX): T={timepoint}")
            volume = np.array(arr[timepoint, :, :, :])
        else:
            aprint("  4D array — using as-is (use --channel or --timepoint to slice)")
            volume = np.array(arr)
    else:
        volume = np.array(arr)

    return volume


# ---------------------------------------------------------------------------
# OME-Zarr shape discovery
# ---------------------------------------------------------------------------


@dataclass
class OMEZarrInfo:
    """Metadata about an OME-Zarr dataset's structure."""

    axes: List[str]
    """Axis labels, e.g. ``["t", "c", "z", "y", "x"]``."""

    shape: Tuple[int, ...]
    """Full array shape at highest resolution."""

    n_timepoints: int
    """Size of the T dimension (1 if absent)."""

    n_channels: int
    """Number of flat channel tasks (product of channel-like axes, or 1)."""

    channel_axes: List[str]
    """Axis labels folded into the flat channel task index."""

    channel_shape: Tuple[int, ...]
    """Shape of axes folded into the flat channel task index."""

    spatial_shape: Tuple[int, ...]
    """ZYX (or YX) portion of the shape."""

    spatial_axes: List[str]
    """Spatial axis labels, e.g. ``["z", "y", "x"]``."""

    voxel_size: Optional[Tuple[float, ...]] = None
    """Physical spacing from coordinateTransformations (spatial axes only)."""

    unit: Optional[str] = None
    """Physical unit string (e.g. ``"micrometer"``)."""

    resolution_levels: int = 1
    """Number of multiscale levels."""

    path: Optional[Path] = None
    """Path to the zarr store."""


def discover_ome_zarr_shape(
    path: Path,
    axes_override: Optional[List[str]] = None,
    array_key: Optional[str] = None,
) -> OMEZarrInfo:
    """Discover the shape and axis structure of an OME-Zarr dataset.

    Parses ``.zattrs`` ``multiscales`` metadata (NGFF v0.4+). Falls back
    to a custom ``axes`` attribute, then to a shape-based heuristic
    (5D→TCZYX, 4D→CZYX, 3D→ZYX) for non-NGFF zarr stores.

    Accepts both plain ``.zarr`` directories and ``.zarr.zip`` archives —
    zarr's ZipStore handles the latter transparently.

    Args:
        path: Path to the ``.zarr`` store or ``.zarr.zip`` archive.
        axes_override: Explicit axis labels (e.g. ``["time","channel","z","y","x"]``).
            Overrides all auto-detection when provided.
        array_key: Key path to a specific array within the zarr store
            (e.g. ``"h2afva/fused"``).  When provided, skips auto-selection
            and navigates directly to this array.

    Returns:
        :class:`OMEZarrInfo` with discovered metadata.

    Raises:
        ValueError: If the zarr store has no arrays, ``array_key`` is not
            found, or the store is unreadable.
    """
    import zarr

    store = zarr.open(str(path), mode="r")

    # Navigate to the group/array
    if isinstance(store, zarr.Array):
        arr = store
        attrs: Dict[str, Any] = dict(getattr(store, "attrs", {}))
    elif isinstance(store, zarr.Group):
        attrs = dict(store.attrs)
        if array_key is not None:
            # User-specified array key (may be nested, e.g. "h2afva/fused")
            try:
                arr = store[array_key]
            except KeyError:
                available = list(store.keys())
                raise ValueError(
                    f"Array key '{array_key}' not found in {path}. "
                    f"Available keys: {available}"
                )
        elif "0" in store:
            # OME-NGFF standard: resolution level "0" is highest resolution
            arr = store["0"]
        else:
            # Find the largest array, searching recursively into sub-groups
            arrays = _find_all_arrays(store)
            if not arrays:
                raise ValueError(f"No arrays found in zarr group: {path}")
            # Pick the array with the most elements
            arr = max(arrays, key=lambda kv: int(np.prod(kv[1].shape)))[1]
    else:
        raise ValueError(f"Unexpected zarr object type: {type(store)}")

    shape = tuple(arr.shape)
    ndim = len(shape)

    # User-supplied axes override: skip all auto-detection
    if axes_override is not None:
        if len(axes_override) != ndim:
            raise ValueError(
                f"--axes has {len(axes_override)} labels but array is {ndim}D "
                f"(shape {shape}). Provide exactly {ndim} comma-separated axis names."
            )
        return _parse_custom_axes_attr(axes_override, shape, path)

    # Try NGFF multiscales metadata
    multiscales = attrs.get("multiscales")
    if multiscales and isinstance(multiscales, list) and len(multiscales) > 0:
        ms = multiscales[0]
        return _parse_ngff_metadata(ms, shape, path, store)

    # Try custom axes attribute (e.g. Keller-lab zarr.zip files store
    # axes = ['time', 'camera', 'channel', 'z', 'y', 'x'])
    custom_axes = attrs.get("axes")
    if custom_axes and isinstance(custom_axes, list) and len(custom_axes) == ndim:
        return _parse_custom_axes_attr(custom_axes, shape, path)

    # Fallback: heuristic based on ndim
    return _heuristic_ome_info(shape, ndim, path)


def _parse_ngff_metadata(
    ms: Dict[str, Any],
    shape: Tuple[int, ...],
    path: Path,
    store: Any,
) -> OMEZarrInfo:
    """Parse NGFF v0.4+ multiscales metadata."""
    axes_raw = ms.get("axes", [])
    axes = [a["name"] if isinstance(a, dict) else str(a) for a in axes_raw]

    # Identify T, C, spatial axes
    t_idx: Optional[int] = None
    c_idx: Optional[int] = None
    spatial_indices: List[int] = []
    spatial_axes: List[str] = []

    for i, a in enumerate(axes_raw):
        if isinstance(a, dict):
            atype = a.get("type", "").lower()
            aname = a.get("name", "").lower()
        else:
            atype = ""
            aname = str(a).lower()

        if atype == "time" or aname == "t":
            t_idx = i
        elif atype == "channel" or aname == "c":
            c_idx = i
        elif atype == "space" or aname in ("z", "y", "x"):
            spatial_indices.append(i)
            spatial_axes.append(aname)
        else:
            # Unknown axis — treat as spatial
            spatial_indices.append(i)
            spatial_axes.append(aname)

    n_t = shape[t_idx] if t_idx is not None else 1
    channel_axes = [axes[c_idx]] if c_idx is not None else []
    channel_shape = (shape[c_idx],) if c_idx is not None else ()
    n_c = shape[c_idx] if c_idx is not None else 1
    spatial_shape = tuple(shape[i] for i in spatial_indices)

    # Extract voxel_size from coordinateTransformations
    voxel_size = None
    unit = None
    datasets = ms.get("datasets", [])
    if datasets:
        transforms = datasets[0].get("coordinateTransformations", [])
        for t in transforms:
            if t.get("type") == "scale":
                scale = t.get("scale", [])
                # Extract spatial dimensions only
                if spatial_indices and len(scale) == len(shape):
                    voxel_size = tuple(float(scale[i]) for i in spatial_indices)
                elif len(scale) == len(spatial_indices):
                    voxel_size = tuple(float(s) for s in scale)

    # Extract unit from axes metadata
    for a in axes_raw:
        if isinstance(a, dict) and a.get("type") == "space":
            u = a.get("unit")
            if u:
                unit = u
                break

    # Count resolution levels
    n_levels = len(datasets) if datasets else 1

    return OMEZarrInfo(
        axes=axes,
        shape=shape,
        n_timepoints=n_t,
        n_channels=n_c,
        channel_axes=channel_axes,
        channel_shape=channel_shape,
        spatial_shape=spatial_shape,
        spatial_axes=spatial_axes,
        voxel_size=voxel_size,
        unit=unit,
        resolution_levels=n_levels,
        path=path,
    )


def _parse_custom_axes_attr(
    axes: List[str], shape: Tuple[int, ...], path: Path
) -> OMEZarrInfo:
    """Build OMEZarrInfo from a custom ``axes`` list attribute.

    Recognises common axis name conventions:
      - T: ``time``, ``t``
      - C: ``channel``, ``c``, ``ch``
      - Camera / extra non-spatial dims (``camera``, ``cam``, ``view``,
        ``angle``): folded into the channel count so each combination
        becomes its own fitting task.
      - Spatial: ``z``, ``y``, ``x``, ``depth``, ``height``, ``width``
        (and any unrecognised leftover axes)
    """
    _SPATIAL = {"z", "y", "x", "depth", "height", "width"}
    _TIME = {"time", "t"}
    _CHANNEL = {"channel", "c", "ch"}
    _CAMERA = {"camera", "cam", "view", "angle"}

    t_idx: Optional[int] = None
    channel_indices: List[int] = []  # channel + camera axes
    spatial_indices: List[int] = []

    for i, ax in enumerate(axes):
        ax_l = ax.lower()
        if ax_l in _TIME:
            t_idx = i
        elif ax_l in _CHANNEL or ax_l in _CAMERA:
            channel_indices.append(i)
        elif ax_l in _SPATIAL:
            spatial_indices.append(i)
        else:
            # Unknown axis — treat as spatial
            spatial_indices.append(i)

    n_t = shape[t_idx] if t_idx is not None else 1
    channel_shape = tuple(shape[i] for i in channel_indices)
    channel_axes = [axes[i] for i in channel_indices]
    n_c = 1
    for size in channel_shape:
        n_c *= size

    spatial_shape = tuple(shape[i] for i in spatial_indices)
    spatial_axes = [axes[i] for i in spatial_indices]

    return OMEZarrInfo(
        axes=axes,
        shape=shape,
        n_timepoints=n_t,
        n_channels=n_c,
        channel_axes=channel_axes,
        channel_shape=channel_shape,
        spatial_shape=spatial_shape,
        spatial_axes=spatial_axes,
        path=path,
    )


def _heuristic_ome_info(shape: Tuple[int, ...], ndim: int, path: Path) -> OMEZarrInfo:
    """Fallback OME info based on shape heuristics."""
    if ndim == 5:
        # Assume TCZYX
        return OMEZarrInfo(
            axes=["t", "c", "z", "y", "x"],
            shape=shape,
            n_timepoints=shape[0],
            n_channels=shape[1],
            channel_axes=["c"],
            channel_shape=(shape[1],),
            spatial_shape=shape[2:],
            spatial_axes=["z", "y", "x"],
            path=path,
        )
    elif ndim == 4:
        # Assume CZYX (could be TZYX — user can override)
        return OMEZarrInfo(
            axes=["c", "z", "y", "x"],
            shape=shape,
            n_timepoints=1,
            n_channels=shape[0],
            channel_axes=["c"],
            channel_shape=(shape[0],),
            spatial_shape=shape[1:],
            spatial_axes=["z", "y", "x"],
            path=path,
        )
    elif ndim == 3:
        return OMEZarrInfo(
            axes=["z", "y", "x"],
            shape=shape,
            n_timepoints=1,
            n_channels=1,
            channel_axes=[],
            channel_shape=(),
            spatial_shape=shape,
            spatial_axes=["z", "y", "x"],
            path=path,
        )
    elif ndim == 2:
        return OMEZarrInfo(
            axes=["y", "x"],
            shape=shape,
            n_timepoints=1,
            n_channels=1,
            channel_axes=[],
            channel_shape=(),
            spatial_shape=shape,
            spatial_axes=["y", "x"],
            path=path,
        )
    else:
        # Generic nD — all spatial
        axes = [f"dim{i}" for i in range(ndim)]
        return OMEZarrInfo(
            axes=axes,
            shape=shape,
            n_timepoints=1,
            n_channels=1,
            channel_axes=[],
            channel_shape=(),
            spatial_shape=shape,
            spatial_axes=axes,
            path=path,
        )


# ---------------------------------------------------------------------------
# Dimension building (reused by convert and view)
# ---------------------------------------------------------------------------


def build_dimensions_from_data(
    centers: np.ndarray,
) -> Any:
    """Build Dimensions object from gsplat center bounding box.

    Args:
        centers: Splat center positions (N, D)

    Returns:
        Dimensions with ranges matching the data extent
    """
    from luxar import Dimension, Dimensions

    ndim = centers.shape[1]
    mins = centers.min(axis=0)
    maxs = centers.max(axis=0)

    # Ensure range is valid (min < max) — add epsilon for degenerate dims
    for i in range(ndim):
        if maxs[i] <= mins[i]:
            maxs[i] = mins[i] + 1.0

    if ndim == 2:
        dims = Dimensions.default_2d()
        for i, dim in enumerate(dims.dimensions):
            dim.range = (float(mins[i]), float(maxs[i]))
        return dims

    if ndim == 3:
        dims = Dimensions.default_3d()
        for i, dim in enumerate(dims.dimensions):
            dim.range = (float(mins[i]), float(maxs[i]))
        return dims

    # nD: first 3 displayed, rest non-displayed
    dim_list = []
    for i in range(ndim):
        dim_list.append(
            Dimension(
                name=f"dim{i}",
                unit="voxel",
                range=(float(mins[i]), float(maxs[i])),
                step=1.0,
                display=(i < 3),
            )
        )
    return Dimensions(dimensions=dim_list)


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
