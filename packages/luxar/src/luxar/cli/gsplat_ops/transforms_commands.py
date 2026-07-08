"""``luxar gsplat transforms`` commands (extracted from gsplat_commands.py).

Each command is a plain function; ``register_transforms_commands(app)`` wires them onto
the shared ``app_gsplat`` Typer (package-refactor-plan P3/P4/P6).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import typer

from .transforms_additive import run_additive_dataset as _run_additive_dataset_impl
from .transforms_cull import run_cull_dataset as _run_cull_dataset_impl
from .transforms_filter_slice import (
    run_filter_dataset as _run_filter_dataset_impl,
)
from .transforms_filter_slice import run_slice_dataset as _run_slice_dataset_impl
from .transforms_merge import run_merge_datasets as _run_merge_datasets_impl
from .transforms_parsing import (
    parse_bbox as _parse_bbox_impl,
)
from .transforms_parsing import (
    parse_csv_floats as _parse_csv_floats_impl,
)
from .transforms_parsing import (
    parse_slices as _parse_slices_impl,
)
from .transforms_partition_flatten import (
    run_flatten_dataset as _run_flatten_dataset_impl,
)
from .transforms_partition_flatten import (
    run_partition_dataset as _run_partition_dataset_impl,
)
from .transforms_transform import run_transform_dataset

if TYPE_CHECKING:
    pass


def cull_dataset(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    output_path: Path = typer.Argument(..., help="Output .gsplats.zarr dataset"),
    method: str = typer.Option(
        "auto",
        "--method",
        "-m",
        help=(
            "Culling method: "
            "error_budget (most principled, needs --target), "
            "redundancy (GPU, no target), "
            "cumulative (fast, keep top N%% amplitude), "
            "amplitude_percentile (remove bottom X%%), "
            "combined (low amp OR large vol), "
            "auto (error_budget if --target, redundancy if --shape, else cumulative)"
        ),
    ),
    target_path: Optional[Path] = typer.Option(
        None,
        "--target",
        exists=True,
        help="[error_budget] Original volume the splats were fitted to. "
        "Required for error_budget mode, which compares the reconstruction "
        "against this reference to decide which splats are dispensable.",
    ),
    volume_shape: Optional[str] = typer.Option(
        None,
        "--shape",
        help="[redundancy] Volume shape as comma-separated ints (e.g. '41,512,512'). "
        "Required for redundancy mode, which renders the splats to measure "
        "each one's fractional contribution — no target volume needed.",
    ),
    # --- error_budget params ---
    error_percentile: float = typer.Option(
        99.0,
        "--error-percentile",
        "-p",
        help="[error_budget] Controls aggressiveness. The error budget is set "
        "at this percentile of the existing residual |target - reconstruction|. "
        "Higher = more conservative. 99 means 'allow each splat's removal to "
        "increase local error by up to the 99th-percentile error level'.",
        min=0.0,
        max=100.0,
    ),
    error_tolerance: float = typer.Option(
        1.0,
        "--error-tolerance",
        help="[error_budget] Multiplier on the error budget. "
        ">1 allows more removal, <1 is more conservative.",
        min=0.0,
    ),
    # --- redundancy params ---
    redundancy_threshold: float = typer.Option(
        0.01,
        "--redundancy-threshold",
        help="[redundancy] A splat is removed if it never contributes more "
        "than this fraction of the local signal anywhere in its support. "
        "0.01 means 'remove splats contributing < 1%% of the signal at every "
        "point they cover — other splats already handle those regions'.",
        min=0.0,
        max=1.0,
    ),
    # --- heuristic params ---
    retention: float = typer.Option(
        0.95,
        "--retention",
        "-r",
        help="[cumulative] Keep the top splats that account for this fraction "
        "of the total amplitude. 0.95 keeps 95%% of total signal, discarding "
        "the weakest ~10-30%% of splats. Fast but ignores spatial overlap.",
        min=0.0,
        max=1.0,
    ),
    amplitude_percentile: float = typer.Option(
        5.0,
        "--amplitude-percentile",
        "-a",
        help="[amplitude_percentile / combined] Remove splats in the bottom "
        "X percentile of amplitude. 5.0 removes the weakest 5%% by amplitude.",
        min=0.0,
        max=100.0,
    ),
    volume_percentile: float = typer.Option(
        95.0,
        "--volume-percentile",
        "-v",
        help="[combined] Also remove splats above this volume percentile. "
        "Targets artifacts: unusually large, diffuse splats.",
        min=0.0,
        max=100.0,
    ),
    # --- common params ---
    truncate: Optional[float] = typer.Option(
        None,
        "--truncate",
        help="Gaussian truncation radius in standard deviations. "
        "Defaults to the value stored in the dataset (typically 3.0). "
        "Affects AABB size for per-splat evaluation in GPU-based modes.",
    ),
    max_iters: int = typer.Option(
        8,
        "--max-iters",
        help="[error_budget / redundancy] Max binary-search iterations for "
        "the joint compounding check, which tightens the threshold if "
        "joint removal of all candidates exceeds the budget due to overlap.",
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-d", help="Device: auto/cpu/cuda/mps"
    ),
    channel: Optional[int] = typer.Option(
        None, "--channel", "-c", help="Channel index for OME-Zarr target"
    ),
    timepoint: Optional[int] = typer.Option(
        None, "--timepoint", help="Timepoint index for OME-Zarr target"
    ),
    encoding_mode: Literal["auto", "precision", "memory"] = typer.Option(
        "auto", "--encoding", "-e", help="Encoding mode for output"
    ),
    compress: Optional[Literal["zip", "tar.gz"]] = typer.Option(
        None, "--compress", help="Compress output"
    ),
) -> None:
    """Remove splats that contribute negligibly to the reconstruction.

    Five methods are available, from fastest to most principled:

    \b
    HEURISTIC METHODS (fast, no rendering):
      cumulative            Keep top splats that account for --retention of
                            total amplitude. Fast but ignores spatial overlap.
      amplitude_percentile  Remove bottom --amplitude-percentile by amplitude.
      combined              Remove (low amplitude OR large volume) artifacts.

    \b
    CONTRIBUTION-BASED METHODS (GPU rendering, spatially aware):
      redundancy            No target needed. Measures each splat's fractional
                            contribution g_j/V_pred. Removes splats below
                            --redundancy-threshold everywhere in their support.
      error_budget          Most principled. Requires --target. Measures the
                            error *increase* from removing each splat against
                            the fitting residual. Robust to noise.

    \b
    AUTO MODE (default):
      Selects error_budget if --target is given, redundancy if --shape is
      given, cumulative otherwise.

    Examples:
        luxar gsplat cull input.gsplats.zarr output.gsplats.zarr
        luxar gsplat cull input.gsplats.zarr output.gsplats.zarr -m cumulative -r 0.90
        luxar gsplat cull input.gsplats.zarr output.gsplats.zarr -m redundancy --shape 41,512,512
        luxar gsplat cull input.gsplats.zarr output.gsplats.zarr -m error_budget --target volume.npy
        luxar gsplat cull input.gsplats.zarr output.gsplats.zarr --target vol.tiff -p 95
    """
    return _run_cull_dataset_impl(
        input_path=input_path,
        output_path=output_path,
        method=method,
        target_path=target_path,
        volume_shape=volume_shape,
        error_percentile=error_percentile,
        error_tolerance=error_tolerance,
        redundancy_threshold=redundancy_threshold,
        retention=retention,
        amplitude_percentile=amplitude_percentile,
        volume_percentile=volume_percentile,
        truncate=truncate,
        max_iters=max_iters,
        device=device,
        channel=channel,
        timepoint=timepoint,
        encoding_mode=encoding_mode,
        compress=compress,
    )


def _parse_bbox(s: str, ndim: int) -> list[tuple[float, float]]:
    """Back-compat wrapper around shared bbox parsing helper."""
    return _parse_bbox_impl(s, ndim)


def filter_dataset(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    output_path: Path = typer.Argument(..., help="Output .gsplats.zarr dataset"),
    # Bounding box
    bbox: Optional[str] = typer.Option(
        None, "--bbox", help="Bounding box: 'min0,max0,min1,max1,...' (pairs per dim)"
    ),
    # Volume
    volume_min: Optional[float] = typer.Option(
        None, "--volume-min", help="Minimum volume threshold"
    ),
    volume_max: Optional[float] = typer.Option(
        None, "--volume-max", help="Maximum volume threshold"
    ),
    volume_normalized: bool = typer.Option(
        False,
        "--volume-normalized",
        help="Interpret volume thresholds as 0-1 normalized",
    ),
    # Amplitude
    amplitude_min: Optional[float] = typer.Option(
        None, "--amplitude-min", help="Minimum amplitude threshold"
    ),
    amplitude_max: Optional[float] = typer.Option(
        None, "--amplitude-max", help="Maximum amplitude threshold"
    ),
    amplitude_normalized: bool = typer.Option(
        False,
        "--amplitude-normalized",
        help="Interpret amplitude thresholds as 0-1 normalized",
    ),
    # Eccentricity
    eccentricity_min: Optional[float] = typer.Option(
        None, "--eccentricity-min", help="Minimum eccentricity (1.0 = sphere)"
    ),
    eccentricity_max: Optional[float] = typer.Option(
        None, "--eccentricity-max", help="Maximum eccentricity"
    ),
    # Mass
    mass_min: Optional[float] = typer.Option(
        None, "--mass-min", help="Minimum mass (amplitude * volume)"
    ),
    mass_max: Optional[float] = typer.Option(None, "--mass-max", help="Maximum mass"),
    mass_normalized: bool = typer.Option(
        False, "--mass-normalized", help="Interpret mass thresholds as 0-1 normalized"
    ),
    # Per-axis sigma
    sigma_axis: Optional[int] = typer.Option(
        None, "--sigma-axis", help="Axis index for per-axis sigma filtering"
    ),
    sigma_min: Optional[float] = typer.Option(
        None, "--sigma-min", help="Minimum marginal sigma on --sigma-axis"
    ),
    sigma_max: Optional[float] = typer.Option(
        None, "--sigma-max", help="Maximum marginal sigma on --sigma-axis"
    ),
    # Truncation
    truncate: Optional[float] = typer.Option(
        None,
        "--truncate",
        help="Sigma truncation for volume computation. Defaults to dataset's stored value.",
    ),
    # Output options
    encoding_mode: Literal["auto", "precision", "memory"] = typer.Option(
        "auto", "--encoding", "-e", help="Encoding mode for output"
    ),
    compress: Optional[Literal["zip", "tar.gz"]] = typer.Option(
        None, "--compress", "-c", help="Compress output as .zip or .tar.gz"
    ),
) -> None:
    """Filter splats by multiple criteria (AND logic).

    All filter options are optional. Only specified criteria are applied.
    Multiple criteria combine with AND — a splat must satisfy all
    active criteria to be kept.

    Criteria:
        --bbox: Spatial bounding box (filter by center position)
        --amplitude-min/max: Intensity thresholds
        --volume-min/max: Size thresholds (characteristic length * truncate)
        --eccentricity-min/max: Shape (1.0 = sphere, higher = elongated)
        --mass-min/max: Amplitude * volume (physical importance)
        --sigma-axis + --sigma-min/max: Per-axis standard deviation

    Use --*-normalized flags to interpret thresholds as 0-1 fractions
    of the dataset's [min, max] range.

    Examples:
        # Keep only bright splats
        luxar gsplat filter input.gsplats.zarr output.gsplats.zarr \\
            --amplitude-min 0.1

        # Crop to a 3D bounding box
        luxar gsplat filter input.gsplats.zarr output.gsplats.zarr \\
            --bbox "0,50,0,50,0,50"

        # Remove elongated outliers and large splats
        luxar gsplat filter input.gsplats.zarr output.gsplats.zarr \\
            --eccentricity-max 5.0 --volume-max 100

        # Keep top 50% by amplitude (normalized)
        luxar gsplat filter input.gsplats.zarr output.gsplats.zarr \\
            --amplitude-min 0.5 --amplitude-normalized

        # With compression
        luxar gsplat filter input.gsplats.zarr output.gsplats.zarr.zip \\
            --amplitude-min 0.1 --compress zip
    """
    return _run_filter_dataset_impl(
        input_path=input_path,
        output_path=output_path,
        bbox=bbox,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_normalized=volume_normalized,
        amplitude_min=amplitude_min,
        amplitude_max=amplitude_max,
        amplitude_normalized=amplitude_normalized,
        eccentricity_min=eccentricity_min,
        eccentricity_max=eccentricity_max,
        mass_min=mass_min,
        mass_max=mass_max,
        mass_normalized=mass_normalized,
        sigma_axis=sigma_axis,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        truncate=truncate,
        encoding_mode=encoding_mode,
        compress=compress,
    )


def partition_dataset(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    output_path: Path = typer.Argument(
        ..., help="Output .gsplats.zarr (a single kind=partition file)"
    ),
    max_elements: Optional[int] = typer.Option(
        None,
        "--max-elements",
        "-m",
        help="Max splats per spatial part (the BSP recurses until each part "
        "is at or below this).",
    ),
    parts: Optional[int] = typer.Option(
        None,
        "--parts",
        "-n",
        help="Convenience: target ~N parts (sets --max-elements to "
        "ceil(n_splats / N)).",
    ),
    rule: Literal["median", "midpoint", "sah"] = typer.Option(
        "median", "--rule", help="BSP split rule (median | midpoint | sah)."
    ),
    encoding_mode: Literal["auto", "precision", "memory"] = typer.Option(
        "auto", "--encoding", "-e", help="Encoding mode for output"
    ),
    compress: Optional[Literal["zip", "tar.gz"]] = typer.Option(
        None, "--compress", "-c", help="Compress output as .zip or .tar.gz"
    ),
) -> None:
    """Spatially partition a Gaussian splat dataset into one partitioned file.

    Splits the splats by position (BSP) into a single kind=partition
    .gsplats.zarr (one spatial part per region) for per-part frustum culling.

    Recursively splits the splats by position so each ``part_<i>`` holds at
    most ``--max-elements`` splats (each part carries its own ``position_bounds``
    for per-part frustum culling in the viewer). Supply ``--max-elements``
    directly, or ``--parts N`` to target ~N parts. The output is a single
    self-contained partition file — open it with ``luxar gsplat view`` or embed
    it in a scene.

    Examples:
        # At most 50k splats per spatial part
        luxar gsplat partition input.gsplats.zarr out.gsplats.zarr --max-elements 50000

        # Target ~4 parts
        luxar gsplat partition input.gsplats.zarr out.gsplats.zarr --parts 4

        # Surface-area-heuristic splits, compressed
        luxar gsplat partition input.gsplats.zarr out.gsplats.zarr -m 50000 --rule sah -c zip
    """
    return _run_partition_dataset_impl(
        input_path=input_path,
        output_path=output_path,
        max_elements=max_elements,
        parts=parts,
        rule=rule,
        encoding_mode=encoding_mode,
        compress=compress,
    )


def flatten_dataset(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr (any shape: leaf, lod, partition)"
    ),
    output_path: Path = typer.Argument(
        ..., help="Output .gsplats.zarr (a single flat, matrix-shaped leaf)"
    ),
    encoding_mode: Literal["auto", "precision", "memory"] = typer.Option(
        "auto", "--encoding", "-e", help="Encoding mode for output"
    ),
    compress: Optional[Literal["zip", "tar.gz"]] = typer.Option(
        None, "--compress", "-c", help="Compress output as .zip or .tar.gz"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite the output if it already exists."
    ),
) -> None:
    """Collapse any gsplat tree into a single flat (matrix-shaped) leaf.

    Concatenates the **default-rendered** splats of every spatial part — a
    partition renders all parts; a kind=lod group renders only its finest
    (default) level, so coarse LOD representatives are not double-counted — into
    one flat ``.gsplats.zarr``. This is the bridge from a tiled ``kind=partition``
    (e.g. a ``batch-fit merge`` output, which ``gsplat lod`` and the composed
    recipes can't load directly) to a matrix-shaped input the recipe builders
    accept.

    A leaf or matrix-shaped lod group flattens to its full finest splat set; a
    partition (or partitioned/mosaic topology) is merged across all parts.

    Examples:
        # Tiled batch-fit merge → flat → overview LOD
        luxar gsplat flatten merged.gsplats.zarr flat.gsplats.zarr
        luxar gsplat lod flat.gsplats.zarr out.gsplats.zarr --recipe overview
    """
    return _run_flatten_dataset_impl(
        input_path=input_path,
        output_path=output_path,
        encoding_mode=encoding_mode,
        compress=compress,
        overwrite=overwrite,
    )


def additive_dataset(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr (any shape: leaf, lod, partition)"
    ),
    output_path: Path = typer.Argument(
        ..., help="Output .gsplats.zarr (same tree shape; every leaf laddered)"
    ),
    n_lods: Optional[int] = typer.Option(
        None,
        "--n-lods",
        min=1,
        help="Additive levels per leaf for 'equal-count' breakpoints (default 4). "
        "Ignored when --breakpoints/--target-ms determine the level count.",
    ),
    method: Optional[str] = typer.Option(
        None,
        "--method",
        "-m",
        help="Additive ordering per leaf: auto (default; greedy at small N, "
        "self_energy above) | greedy | self_energy | mass | amplitude | "
        "spectral | random.",
    ),
    breakpoints: Optional[str] = typer.Option(
        None,
        "--breakpoints",
        "-b",
        help="'equal-count' (default) | 'stream:C' (geometric streaming ladder, "
        "first chunk C splats then doubling; sized per leaf) | "
        "'counts:N1,N2,...' (clamped per leaf) | 'energy:f1,f2,...'.",
    ),
    target_ms: Optional[float] = typer.Option(
        None,
        "--target-ms",
        min=1.0,
        help="Streaming sizing: derive 'stream:<c>' breakpoints so each leaf's "
        "first additive chunk downloads in ~this many ms at --bandwidth-mbps "
        "(bytes/splat measured from the input store; override with "
        "--bytes-per-splat). Mutually exclusive with --breakpoints.",
    ),
    bandwidth_mbps: Optional[float] = typer.Option(
        None,
        "--bandwidth-mbps",
        min=0.1,
        help="Assumed downlink for --target-ms sizing (default 25, a typical "
        "broadband connection).",
    ),
    bytes_per_splat: Optional[float] = typer.Option(
        None,
        "--bytes-per-splat",
        min=0.1,
        help="Override the on-wire bytes/splat used by --target-ms sizing "
        "(default: measured from the input store).",
    ),
    encoding_mode: Literal["auto", "precision", "memory"] = typer.Option(
        "auto", "--encoding", "-e", help="Encoding mode for output"
    ),
    compress: Optional[Literal["zip", "tar.gz"]] = typer.Option(
        None, "--compress", "-c", help="Compress output as .zip or .tar.gz"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite the output if it already exists."
    ),
) -> None:
    """Give every leaf of a gsplat tree an additive (streaming) LOD ladder.

    Walks the tree structure-preservingly — substitutive ``kind=lod`` levels,
    ``kind=partition`` parts, adaptive/overview groups all keep their shape —
    and rebuilds each leaf with an additive prefix-sum ladder, WITHOUT
    recomputing the (expensive, GPU-built) substitutive/partition structure.
    The per-leaf counterpart of ``gsplat lod --recipe stream`` (which needs
    a flat input), and the inverse companion of ``gsplat flatten``.

    Breakpoints are sized per leaf: ``--target-ms``/``--bandwidth-mbps`` derive
    a ``stream:<c>`` geometric ladder (first chunk ~target-ms of download,
    then doubling — the viewer streams additive sub-LODs progressively, so
    the first chunk sets first-paint latency); explicit ``counts:`` lists are
    clamped to each leaf's size. A leaf that already has a ladder is rebuilt
    from its flattened union (prior ordering discarded).

    Examples:
        # Substitutive pyramid -> pyramid with ~200ms streaming ladders per level
        luxar gsplat additive sub.gsplats.zarr pyr.gsplats.zarr --target-ms 200

        # Explicit geometric ladder, 14k first chunk
        luxar gsplat additive in.gsplats.zarr out.gsplats.zarr -b stream:14000

        # Classic 4-level equal-count ladders on every leaf
        luxar gsplat additive in.gsplats.zarr out.gsplats.zarr --n-lods 4
    """
    return _run_additive_dataset_impl(
        input_path=input_path,
        output_path=output_path,
        n_lods=n_lods,
        method=method,
        breakpoints=breakpoints,
        target_ms=target_ms,
        bandwidth_mbps=bandwidth_mbps,
        bytes_per_splat=bytes_per_splat,
        encoding_mode=encoding_mode,
        compress=compress,
        overwrite=overwrite,
    )


def _parse_slices(s: str, ndim: int) -> list[slice]:
    """Back-compat wrapper around shared range parsing helper."""
    return _parse_slices_impl(s, ndim)


def slice_dataset(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    output_path: Path = typer.Argument(..., help="Output .gsplats.zarr dataset"),
    ranges: str = typer.Argument(
        ..., help="Numpy-style ranges per dimension: '0:50, :, 10:90'"
    ),
    # Output options
    encoding_mode: Literal["auto", "precision", "memory"] = typer.Option(
        "auto", "--encoding", "-e", help="Encoding mode for output"
    ),
    compress: Optional[Literal["zip", "tar.gz"]] = typer.Option(
        None, "--compress", "-c", help="Compress output as .zip or .tar.gz"
    ),
) -> None:
    """Slice splats by coordinate ranges (numpy-style syntax).

    Keeps splats whose center coordinates fall within the specified ranges
    per dimension. Uses float coordinate values, not integer indices.

    Range syntax (per dimension, comma-separated):
        lo:hi   — keep centers in [lo, hi]
        lo:     — keep centers >= lo
        :hi     — keep centers <= hi
        :       — keep all (no constraint)

    Examples:
        # Crop x to [0,50], keep all y, crop z to [10,90]
        luxar gsplat slice input.gsplats.zarr output.gsplats.zarr "0:50, :, 10:90"

        # Keep only first half of x-range
        luxar gsplat slice input.gsplats.zarr output.gsplats.zarr ":50, :, :"

        # Float coordinates work
        luxar gsplat slice input.gsplats.zarr output.gsplats.zarr "1.5:42.7, :, -3.2:100"

        # With compression
        luxar gsplat slice input.gsplats.zarr output.gsplats.zarr.zip "0:50, :, :" --compress zip
    """
    return _run_slice_dataset_impl(
        input_path=input_path,
        output_path=output_path,
        ranges=ranges,
        encoding_mode=encoding_mode,
        compress=compress,
    )


def _parse_csv_floats(value: str, expected: int, name: str) -> list[float]:
    """Back-compat wrapper around shared float-list parser."""
    return _parse_csv_floats_impl(value, expected, name)


def transform_dataset(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    output_path: Path = typer.Argument(..., help="Output .gsplats.zarr dataset"),
    # Spatial transforms
    scale_factors: Optional[str] = typer.Option(
        None,
        "--scale",
        "-s",
        help="Per-axis scale factors, comma-separated (e.g. '1,1,1,4')",
    ),
    translate_offset: Optional[str] = typer.Option(
        None,
        "--translate",
        "-t",
        help="Per-axis translation, comma-separated (e.g. '0,0,0,100')",
    ),
    rotate_x_deg: Optional[float] = typer.Option(
        None, "--rotate-x", help="Rotate around X axis (degrees, 3D spatial dims only)"
    ),
    rotate_y_deg: Optional[float] = typer.Option(
        None, "--rotate-y", help="Rotate around Y axis (degrees, 3D spatial dims only)"
    ),
    rotate_z_deg: Optional[float] = typer.Option(
        None, "--rotate-z", help="Rotate around Z axis (degrees, 3D spatial dims only)"
    ),
    center: bool = typer.Option(
        False, "--center", help="Center at amplitude-weighted centroid"
    ),
    # Intensity transforms
    scale_intensity_factor: Optional[float] = typer.Option(
        None, "--scale-intensity", help="Scale amplitudes by factor (e.g. 0.01)"
    ),
    normalize_intensity: Optional[float] = typer.Option(
        None,
        "--normalize-intensity",
        help="Normalize amplitudes so max equals this value (e.g. 1.0)",
    ),
    # Output options
    encoding_mode: Literal["auto", "precision", "memory"] = typer.Option(
        "auto", "--encoding", "-e", help="Encoding mode for output"
    ),
    compress: Optional[Literal["zip", "tar.gz"]] = typer.Option(
        None, "--compress", "-c", help="Compress output as .zip or .tar.gz"
    ),
) -> None:
    """Apply spatial and intensity transforms to a Gaussian splat dataset.

    Multiple transforms can be combined in one command. They are applied
    in a fixed order: scale → rotate → translate → center → scale-intensity →
    normalize-intensity.

    Examples:
        # Fix microscopy anisotropy (Z=4x) and normalize brightness
        luxar gsplat transform in.gsplats.zarr out.gsplats.zarr \\
            --scale 4,1,1,1 --normalize-intensity 1.0 --center

        # Translate and recenter
        luxar gsplat transform in.gsplats.zarr out.gsplats.zarr \\
            --translate 0,100,0 --center

        # Rotate 90 degrees around Z axis
        luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --rotate-z 90

        # Just recenter at centroid
        luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --center

        # Normalize amplitudes to [0, 1]
        luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --normalize-intensity 1.0

        # Scale brightness only
        luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --scale-intensity 0.5
    """
    return run_transform_dataset(
        input_path=input_path,
        output_path=output_path,
        scale_factors=scale_factors,
        translate_offset=translate_offset,
        rotate_x_deg=rotate_x_deg,
        rotate_y_deg=rotate_y_deg,
        rotate_z_deg=rotate_z_deg,
        center=center,
        scale_intensity_factor=scale_intensity_factor,
        normalize_intensity=normalize_intensity,
        encoding_mode=encoding_mode,
        compress=compress,
    )


def merge_datasets(
    inputs: list[Path] = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr datasets (2 or more)"
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output .gsplats.zarr path"
    ),
    as_dimension: bool = typer.Option(
        False, "--as-dimension", help="Stack along a new dimension (e.g., time)"
    ),
    values: Optional[str] = typer.Option(
        None, "--values", help="Comma-separated coordinate values for --as-dimension"
    ),
    sigma: float = typer.Option(
        0.0, "--sigma", help="Sigma in new dimension for --as-dimension (0=discrete)"
    ),
    channel_colors: Optional[str] = typer.Option(
        None,
        "--channel-colors",
        help="Comma-separated hex colors (e.g., '#ff0080,#00ff80')",
    ),
    compress: Optional[Literal["zip", "tar.gz"]] = typer.Option(
        None, "--compress", "-c", help="Compress output archive"
    ),
    encoding: Literal["auto", "precision", "memory"] = typer.Option(
        "auto", "--encoding", "-e", help="Encoding mode"
    ),
) -> None:
    """Merge multiple Gaussian splat datasets into one.

    Three modes (mutually exclusive):

    1. Concatenation (default): Simple merge of all splats.

    2. New dimension (--as-dimension): Stack along new dimension (e.g., time).

    3. Channel colors (--channel-colors): Per-dataset color assignment.

    Examples:
        luxar gsplat merge a.gsplats.zarr b.gsplats.zarr -o merged.gsplats.zarr

        luxar gsplat merge t0.zarr t1.zarr t2.zarr -o 4d.zarr --as-dimension

        luxar gsplat merge ch0.zarr ch1.zarr -o multi.zarr \\
            --channel-colors "#ff0080,#00ff00"
    """
    return _run_merge_datasets_impl(
        inputs=inputs,
        output_path=output_path,
        as_dimension=as_dimension,
        values=values,
        sigma=sigma,
        channel_colors=channel_colors,
        compress=compress,
        encoding=encoding,
    )


def register_transforms_commands(app: typer.Typer) -> None:
    """Register the transforms commands onto ``app_gsplat``."""
    # Lead with the most common editing ops; partition/slice are more advanced.
    app.command("transform")(transform_dataset)
    app.command("merge")(merge_datasets)
    app.command("cull")(cull_dataset)
    app.command("filter")(filter_dataset)
    app.command("slice")(slice_dataset)
    app.command("partition")(partition_dataset)
    app.command("flatten")(flatten_dataset)
    app.command("additive")(additive_dataset)
