"""``luxar gsplat transforms`` commands (extracted from gsplat_commands.py).

Each command is a plain function; ``register_transforms_commands(app)`` wires them onto
the shared ``app_gsplat`` Typer (package-refactor-plan P3/P4/P6).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional

import typer
from arbol import aprint, asection

from ..utils import format_memory_size
from .encoding import _resolve_encoding_mode

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
    try:
        from luxar.gsplats.gsplat_data import GSplatData

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        with asection(f"Culling: {input_path.name}"):
            with asection("Loading dataset"):
                data = GSplatData.load(input_path, include_stats=True)
                n_original = data.n_splats
                aprint(f"Loaded {n_original:,} splats ({data.ndim}D)")

            # Resolve truncation radius from dataset if not explicitly set
            if truncate is None:
                truncate = data.truncation_radius

            # Load target volume if provided
            target_np = None
            if target_path is not None:
                from luxar.cli.gsplat_config import load_volume

                with asection("Loading target volume"):
                    target_np = load_volume(
                        target_path, channel=channel, timepoint=timepoint
                    )
                    aprint(f"Target shape: {target_np.shape}")
                    if len(target_np.shape) != data.ndim:
                        aprint(
                            f"Dimension mismatch: gsplats are {data.ndim}D "
                            f"but target is {len(target_np.shape)}D"
                        )
                        raise typer.Exit(1)

            # Parse --shape if provided
            parsed_shape = None
            if volume_shape is not None:
                parsed_shape = tuple(int(x.strip()) for x in volume_shape.split(","))

            # Resolve method
            resolved = method
            if resolved == "auto":
                if target_np is not None:
                    resolved = "error_budget"
                elif parsed_shape is not None:
                    resolved = "redundancy"
                else:
                    resolved = "cumulative"

            with asection(f"Culling (method={resolved})"):
                culled_data = data.cull(
                    target=target_np,
                    method=resolved,
                    shape=parsed_shape,
                    truncate=truncate,
                    error_percentile=error_percentile,
                    error_tolerance=error_tolerance,
                    redundancy_threshold=redundancy_threshold,
                    max_binary_search_iters=max_iters,
                    device=device,
                    retention=retention,
                    amplitude_percentile=amplitude_percentile,
                    volume_percentile=volume_percentile,
                    verbose=True,
                )

                n_culled = n_original - culled_data.n_splats
                aprint("\nResults:")
                aprint(f"  Original: {n_original:,} splats")
                aprint(
                    f"  Removed:  {n_culled:,} ({100 * n_culled / max(n_original, 1):.1f}%)"
                )
                aprint(f"  Kept:     {culled_data.n_splats:,} splats")
                aprint(f"  Method:   {resolved}")

                # Compression stats
                ndim = data.ndim
                tril = ndim * (ndim + 1) // 2
                floats_per_splat = ndim + tril + 1
                bits_orig = n_original * floats_per_splat * 32
                bits_culled = culled_data.n_splats * floats_per_splat * 32
                if target_np is not None:
                    vol_bits = int(target_np.size) * 32
                    aprint(
                        f"  Compression: {vol_bits / max(bits_orig, 1):.1f}x -> {vol_bits / max(bits_culled, 1):.1f}x"
                    )

                if resolved in ("error_budget", "redundancy"):
                    aprint(
                        f"  Error budget: {culled_data.stats.get('error_budget', 'N/A')}"
                    )
                    aprint(
                        f"  Joint check iters: {culled_data.stats.get('phase2_iterations', 'N/A')}"
                    )
                if resolved in ("cumulative", "amplitude_percentile", "combined"):
                    amp_ret = culled_data.stats.get("amplitude_retention")
                    if amp_ret is not None:
                        aprint(f"  Amplitude retention: {100 * amp_ret:.2f}%")

            with asection("Saving"):
                culled_data.save(
                    output_path,
                    encoding_mode=encoding_mode_obj,
                    compress=compress,
                )
                aprint(f"Saved to {output_path}")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def _parse_bbox(s: str, ndim: int) -> list[tuple[float, float]]:
    """Parse a bbox string 'min0,max0,min1,max1,...' into list of (min, max) pairs."""
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 2 * ndim:
        raise typer.BadParameter(
            f"bbox needs {2 * ndim} values for {ndim}D data, got {len(parts)}"
        )
    pairs = [(parts[2 * i], parts[2 * i + 1]) for i in range(ndim)]
    for i, (lo, hi) in enumerate(pairs):
        if lo > hi:
            raise typer.BadParameter(f"bbox dimension {i} has min ({lo}) > max ({hi})")
    return pairs


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
    try:
        from luxar.gsplats.gsplat_data import GSplatData

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        with asection(f"Filtering: {input_path.name}"):
            # Load
            with asection("Loading dataset"):
                data = GSplatData.load(input_path, include_stats=True)
                n_original = data.n_splats
                ndim = data.ndim
                aprint(f"Loaded {n_original:,} splats ({ndim}D)")

            # Resolve truncation radius from dataset if not explicitly set
            if truncate is None:
                truncate = data.truncation_radius

            # Parse bbox
            bbox_parsed = None
            if bbox is not None:
                bbox_parsed = _parse_bbox(bbox, ndim)

            # Show active criteria
            with asection("Active criteria"):
                if bbox_parsed is not None:
                    aprint(f"  bbox: {bbox_parsed}")
                if amplitude_min is not None or amplitude_max is not None:
                    norm = " (normalized)" if amplitude_normalized else ""
                    aprint(f"  amplitude: [{amplitude_min}, {amplitude_max}]{norm}")
                if volume_min is not None or volume_max is not None:
                    norm = " (normalized)" if volume_normalized else ""
                    aprint(
                        f"  volume: [{volume_min}, {volume_max}]{norm} (truncate={truncate})"
                    )
                if eccentricity_min is not None or eccentricity_max is not None:
                    aprint(f"  eccentricity: [{eccentricity_min}, {eccentricity_max}]")
                if mass_min is not None or mass_max is not None:
                    norm = " (normalized)" if mass_normalized else ""
                    aprint(f"  mass: [{mass_min}, {mass_max}]{norm}")
                if sigma_axis is not None:
                    aprint(f"  sigma axis {sigma_axis}: [{sigma_min}, {sigma_max}]")

            # Filter
            with asection("Filtering"):
                filtered_data = data.filter_by(
                    bbox=bbox_parsed,
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
                )
                n_filtered = filtered_data.n_splats
                n_removed = n_original - n_filtered

                aprint("\nResults:")
                aprint(f"  Original splats: {n_original:,}")
                aprint(f"  Filtered splats: {n_filtered:,}")
                aprint(
                    f"  Removed:         {n_removed:,} ({100 * n_removed / max(n_original, 1):.1f}%)"
                )

            # Save
            with asection(f"Saving to {output_path.name}"):
                if n_filtered == 0:
                    aprint("⚠ No splats remain after filtering — skipping save")
                else:
                    filtered_data.save(
                        output_path,
                        encoding_mode=encoding_mode_obj,
                        include_fitting_info=True,
                        compress=compress,
                    )
                    aprint(f"Saved filtered dataset: {output_path}")

                    if output_path.exists():
                        aprint(
                            f"  Size: {format_memory_size(output_path.stat().st_size)}"
                        )

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


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
    try:
        import math

        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import iter_leaves

        if max_elements is None and parts is None:
            aprint("❌ Error: specify --max-elements N (or --parts N)")
            raise typer.Exit(1)
        if max_elements is not None and parts is not None:
            aprint("❌ Error: --max-elements and --parts are mutually exclusive")
            raise typer.Exit(1)

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        with asection(f"Partitioning: {input_path.name}"):
            with asection("Loading dataset"):
                data = GSplatData.load(input_path, include_stats=True)
                aprint(f"Loaded {data.n_splats:,} splats ({data.ndim}D)")

            # --parts N → target ~N parts via ceil(n / N).
            resolved_max = (
                max_elements
                if max_elements is not None
                else max(1, math.ceil(data.n_splats / int(parts)))  # type: ignore[arg-type]
            )

            with asection("Spatial BSP partition"):
                partition_node = data.to_spatial_partition(
                    max_elements=resolved_max, rule=rule
                )
                part_sizes = [leaf.n_splats for leaf in iter_leaves(partition_node)]
                aprint(
                    f"Produced {len(part_sizes)} spatial parts "
                    f"(rule={rule}, max_elements={resolved_max:,}): sizes={part_sizes}"
                )

            with asection(f"Saving to {output_path.name}"):
                write_gsplats_tree(
                    output_path,
                    partition_node,
                    encoding_mode=encoding_mode_obj,
                    compress=compress,
                )
                aprint(
                    f"  Saved kind=partition file: {output_path} "
                    f"({data.n_splats:,} splats in {len(part_sizes)} parts)"
                )

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


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
        # Tiled batch-fit merge → flat → multiscale LOD
        luxar gsplat flatten merged.gsplats.zarr flat.gsplats.zarr
        luxar gsplat lod flat.gsplats.zarr out.gsplats.zarr --recipe multiscale
    """
    try:
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import iter_default_leaves

        if output_path.exists() and not overwrite:
            aprint(f"❌ Error: {output_path} exists; pass --overwrite to replace it.")
            raise typer.Exit(1)

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        with asection(f"Flattening: {input_path.name}"):
            with asection("Loading tree"):
                node, stats = load_gsplat_node(input_path, include_stats=True)

            # One flat GSplatData per default-rendered leaf (finest level only,
            # all parts). `.flattened()` collapses each leaf's additive ladder to
            # a single full set so `concatenate` (which requires a matching
            # substitutive depth) merges them cleanly.
            parts = [
                GSplatData.from_tree(leaf).flattened()
                for leaf in iter_default_leaves(node)
            ]
            if not parts:
                aprint("❌ Error: input tree has no leaves")
                raise typer.Exit(1)

            flat = GSplatData.concatenate(parts)
            # concatenate() builds a fresh stats dict from the first input; keep
            # the root-level provenance/fitting stats from the source tree.
            if stats:
                flat = GSplatData.from_additive_sublods(
                    list(flat.additive_sublods), stats=stats
                )
            aprint(
                f"Flattened {len(parts)} leaf/leaves → {flat.n_splats:,} splats "
                f"({flat.ndim}D, single matrix-shaped leaf)"
            )

            with asection(f"Saving to {output_path.name}"):
                if output_path.exists() and overwrite:
                    import shutil

                    if output_path.is_dir():
                        shutil.rmtree(output_path)
                    else:
                        output_path.unlink()
                flat.save(
                    output_path,
                    encoding_mode=encoding_mode_obj,
                    compress=compress,
                )
                aprint(f"  Saved flat file: {output_path} ({flat.n_splats:,} splats)")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def _parse_slices(s: str, ndim: int) -> list[slice]:
    """Parse numpy-style range string into list of slices.

    Format: "lo:hi, :, lo:" where each dimension is separated by comma.
    Empty start/stop means unbounded (None).

    Examples:
        "0:50, :, 10:90"  → [slice(0,50), slice(None,None), slice(10,90)]
        ":50, 20:, :"     → [slice(None,50), slice(20,None), slice(None,None)]
        "1.5:42.7, :, :"  → [slice(1.5,42.7), slice(None,None), slice(None,None)]
    """
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != ndim:
        raise typer.BadParameter(
            f"Expected {ndim} ranges for {ndim}D data, got {len(parts)}"
        )
    slices = []
    for part in parts:
        if ":" not in part:
            raise typer.BadParameter(
                f"Invalid range '{part}': expected 'lo:hi', 'lo:', ':hi', or ':'"
            )
        lo_str, hi_str = part.split(":", 1)
        lo = float(lo_str.strip()) if lo_str.strip() else None
        hi = float(hi_str.strip()) if hi_str.strip() else None
        slices.append(slice(lo, hi))
    return slices


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
    try:
        from luxar.gsplats.gsplat_data import GSplatData

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        with asection(f"Slicing: {input_path.name}"):
            # Load
            with asection("Loading dataset"):
                data = GSplatData.load(input_path, include_stats=True)
                n_original = data.n_splats
                aprint(f"Loaded {n_original:,} splats ({data.ndim}D)")

            # Parse ranges
            slices = _parse_slices(ranges, data.ndim)
            with asection("Ranges"):
                for i, s in enumerate(slices):
                    lo = s.start if s.start is not None else "-inf"
                    hi = s.stop if s.stop is not None else "inf"
                    aprint(f"  dim {i}: [{lo}, {hi}]")

            # Slice
            with asection("Slicing"):
                sliced_data = data.slice_by(slices)
                n_sliced = sliced_data.n_splats
                n_removed = n_original - n_sliced

                aprint("\nResults:")
                aprint(f"  Original splats: {n_original:,}")
                aprint(f"  Sliced splats:   {n_sliced:,}")
                aprint(
                    f"  Removed:         {n_removed:,} ({100 * n_removed / max(n_original, 1):.1f}%)"
                )

            # Save
            with asection(f"Saving to {output_path.name}"):
                if n_sliced == 0:
                    aprint("⚠ No splats remain after slicing — skipping save")
                else:
                    sliced_data.save(
                        output_path,
                        encoding_mode=encoding_mode_obj,
                        include_fitting_info=True,
                        compress=compress,
                    )
                    aprint(f"Saved sliced dataset: {output_path}")

                    if output_path.exists():
                        aprint(
                            f"  Size: {format_memory_size(output_path.stat().st_size)}"
                        )

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def _parse_csv_floats(value: str, expected: int, name: str) -> list[float]:
    """Parse comma-separated float values and validate count matches expected dimensions."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != expected:
        raise typer.BadParameter(
            f"--{name} expects {expected} comma-separated values (one per dimension), "
            f"got {len(parts)}: '{value}'"
        )
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise typer.BadParameter(f"--{name} values must be numbers: {e}") from e


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
    try:
        from dataclasses import replace

        import numpy as np

        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import (
            amplitude_weighted_centroid,
            center_bounds,
            global_amplitude_max,
            is_matrix_shaped,
            map_leaves,
            node_ndim,
            total_splats,
        )

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        # Check that at least one transform is requested
        has_transform = any(
            [
                scale_factors,
                translate_offset,
                rotate_x_deg is not None,
                rotate_y_deg is not None,
                rotate_z_deg is not None,
                center,
                scale_intensity_factor is not None,
                normalize_intensity is not None,
            ]
        )
        if not has_transform:
            aprint("❌ No transforms specified. Use --help to see available options.")
            raise typer.Exit(1)

        with asection(f"Transforming: {input_path.name}"):
            # Load the raw node tree so partitions / nested trees are preserved.
            # A matrix-shaped tree (a leaf, or a lod group of leaves) flattens to a
            # GSplatData exactly as `GSplatData.load` would; a kind=partition (or
            # otherwise nested) tree is transformed leaf-by-leaf, keeping its shape.
            with asection("Loading dataset"):
                node, stats = load_gsplat_node(input_path, include_stats=True)
                d = node_ndim(node)
                matrix_shaped = is_matrix_shaped(node)
                shape_desc = "leaf/matrix" if matrix_shaped else type(node).__name__
                aprint(
                    f"Loaded {total_splats(node):,} splats ({d}D, {shape_desc})"
                )

            transforms_applied: list[str] = []

            # ── Parse the geometry transforms once (shared by both code paths) ──
            scale_matrix = None
            if scale_factors is not None:
                factors = _parse_csv_floats(scale_factors, d, "scale")
                scale_matrix = np.diag(factors)
                transforms_applied.append(f"scale({scale_factors})")

            rot_matrix = None
            has_rotation = any(
                r is not None for r in [rotate_x_deg, rotate_y_deg, rotate_z_deg]
            )
            if has_rotation:
                # nD convention: the last 3 dims are spatial (XYZ); any preceding
                # dims (e.g. time) are left unrotated.
                if d < 3:
                    aprint(
                        f"❌ Rotation requires at least 3 spatial dimensions, got {d}D data"
                    )
                    raise typer.Exit(1)
                rot3 = np.eye(3, dtype=np.float64)
                if rotate_x_deg is not None:
                    rad = np.radians(rotate_x_deg)
                    c, s = np.cos(rad), np.sin(rad)
                    rot3 = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ rot3
                    transforms_applied.append(f"rotate_x({rotate_x_deg}°)")
                if rotate_y_deg is not None:
                    rad = np.radians(rotate_y_deg)
                    c, s = np.cos(rad), np.sin(rad)
                    rot3 = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ rot3
                    transforms_applied.append(f"rotate_y({rotate_y_deg}°)")
                if rotate_z_deg is not None:
                    rad = np.radians(rotate_z_deg)
                    c, s = np.cos(rad), np.sin(rad)
                    rot3 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ rot3
                    transforms_applied.append(f"rotate_z({rotate_z_deg}°)")
                rot_matrix = np.eye(d, dtype=np.float64)
                rot_matrix[d - 3 :, d - 3 :] = rot3

            translate_vec = None
            if translate_offset is not None:
                offsets = _parse_csv_floats(translate_offset, d, "translate")
                translate_vec = np.array(offsets, dtype=np.float64)
                transforms_applied.append(f"translate({translate_offset})")

            if center:
                transforms_applied.append("center")
            if scale_intensity_factor is not None:
                transforms_applied.append(f"scale_intensity({scale_intensity_factor})")
            if normalize_intensity is not None:
                transforms_applied.append(f"normalize({normalize_intensity})")

            if matrix_shaped:
                # ── Flat path: a leaf / matrix tree → the GSplatData methods ──
                data = GSplatData.from_tree(node, stats=stats)
                if scale_matrix is not None:
                    with asection("Applying scale"):
                        aprint(f"Scale factors: {list(np.diagonal(scale_matrix))}")
                        data = data.transform(scale_matrix)
                if rot_matrix is not None:
                    with asection("Applying rotation"):
                        data = data.transform(rot_matrix)
                if translate_vec is not None:
                    with asection("Applying translation"):
                        aprint(f"Translation: {list(translate_vec)}")
                        data = data.translate(translate_vec)
                if center:
                    with asection("Centering at centroid"):
                        data = data.center_at_centroid()
                        aprint("Centered at amplitude-weighted centroid")
                if scale_intensity_factor is not None:
                    with asection("Scaling intensity"):
                        aprint(f"Intensity scale factor: {scale_intensity_factor}")
                        data = data.scale_intensity(scale_intensity_factor)
                if normalize_intensity is not None:
                    with asection("Normalizing intensity"):
                        current_max = float(data.amplitudes.max())
                        aprint(
                            f"Current max: {current_max:.4f} → "
                            f"target max: {normalize_intensity}"
                        )
                        data = data.normalize_intensity(normalize_intensity)
                result_node = data.tree
            else:
                # ── Tree-walking path: kind=partition / nested (structure kept) ──
                aprint(
                    "Tree-structured input — transforming each part in place "
                    "(structure preserved)."
                )

                from luxar.gsplats.tree import GSplatLeaf, GSplatNode

                def _leaf_op(
                    op: "Callable[[GSplatData], GSplatData]",
                ) -> "Callable[[GSplatLeaf], GSplatNode]":
                    def _fn(leaf: "GSplatLeaf") -> "GSplatNode":
                        # GSplatData.transform/translate/... rebuild a fresh leaf with
                        # empty meta; restore the source leaf's provenance verbatim.
                        # A stale extent-derived min_pixel_size is scrubbed AFTER all
                        # transforms (from leaf AND group nodes) — see below.
                        new_leaf = op(GSplatData.from_tree(leaf)).tree
                        return replace(new_leaf, meta=dict(leaf.meta))

                    return _fn

                if scale_matrix is not None:
                    with asection("Applying scale"):
                        aprint(f"Scale factors: {list(np.diagonal(scale_matrix))}")
                        node = map_leaves(
                            node, _leaf_op(lambda gd: gd.transform(scale_matrix))
                        )
                if rot_matrix is not None:
                    with asection("Applying rotation"):
                        node = map_leaves(
                            node, _leaf_op(lambda gd: gd.transform(rot_matrix))
                        )
                if translate_vec is not None:
                    with asection("Applying translation"):
                        aprint(f"Translation: {list(translate_vec)}")
                        node = map_leaves(
                            node, _leaf_op(lambda gd: gd.translate(translate_vec))
                        )
                if center:
                    with asection("Centering at centroid"):
                        centroid = amplitude_weighted_centroid(node)
                        if centroid is not None:
                            node = map_leaves(
                                node, _leaf_op(lambda gd: gd.translate(-centroid))
                            )
                            aprint("Centered at global amplitude-weighted centroid")
                if scale_intensity_factor is not None:
                    with asection("Scaling intensity"):
                        aprint(f"Intensity scale factor: {scale_intensity_factor}")
                        node = map_leaves(
                            node,
                            _leaf_op(lambda gd: gd.scale_intensity(scale_intensity_factor)),
                        )
                if normalize_intensity is not None:
                    with asection("Normalizing intensity"):
                        current_max = global_amplitude_max(node)
                        aprint(
                            f"Current global max: {current_max:.4f} → "
                            f"target max: {normalize_intensity}"
                        )
                        if current_max > 0:
                            factor = normalize_intensity / current_max
                            node = map_leaves(
                                node, _leaf_op(lambda gd: gd.scale_intensity(factor))
                            )
                # A geometry transform (scale/rotate/translate/center) invalidates
                # the extent-derived min_pixel_size LOD-switch threshold on EVERY
                # node — leaves AND group nodes (a multiscale partition child, a
                # mosaic per-part lod group). Scrub it from the whole tree so the
                # writer re-derives it from the transformed extents; intensity-only
                # transforms leave it intact (the extents are unchanged).
                geometry_changed = (
                    scale_matrix is not None
                    or rot_matrix is not None
                    or translate_vec is not None
                    or center
                )
                if geometry_changed:
                    from luxar.gsplats.tree import without_meta_key

                    node = without_meta_key(node, "min_pixel_size")
                result_node = node

            # Summary
            aprint(f"\nTransforms applied: {' → '.join(transforms_applied)}")

            # Print new bounding box (from the result tree's center bounds)
            with asection("Result bounding box"):
                bounds = center_bounds(result_node)
                if bounds is not None:
                    lo_all, hi_all = bounds
                    for i in range(d):
                        lo, hi = float(lo_all[i]), float(hi_all[i])
                        aprint(f"  Dim {i}: [{lo:.4f}, {hi:.4f}]  range: {hi - lo:.4f}")

            # Save (color SDR/HDR is auto-detected by the writer). The flat path
            # keeps GSplatData.save (carries fitting provenance, unchanged); the
            # tree path uses the v3.0 tree writer so a partition stays a partition
            # on disk.
            with asection(f"Saving to {output_path.name}"):
                if matrix_shaped:
                    data.save(
                        output_path,
                        encoding_mode=encoding_mode_obj,
                        include_fitting_info=True,
                        compress=compress,
                    )
                else:
                    write_gsplats_tree(
                        output_path,
                        result_node,
                        encoding_mode=encoding_mode_obj,
                        compress=compress,
                    )
                aprint(f"Saved: {output_path}")

                if output_path.exists():
                    aprint(f"  Size: {format_memory_size(output_path.stat().st_size)}")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


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
    try:
        from luxar.cli.gsplat_config import parse_hex_color
        from luxar.gsplats.gsplat_data import GSplatData

        if len(inputs) < 2:
            aprint("Error: At least 2 input datasets required for merge")
            raise typer.Exit(1)

        if as_dimension and channel_colors:
            aprint("Error: --as-dimension and --channel-colors are mutually exclusive")
            raise typer.Exit(1)

        with asection(f"Merging {len(inputs)} datasets"):
            datasets: list[GSplatData] = []
            total_splats = 0
            for inp in inputs:
                with asection(f"Loading {inp.name}"):
                    ds = GSplatData.load(inp, include_stats=False)
                    aprint(f"{ds.n_splats:,} splats ({ds.ndim}D)")
                    datasets.append(ds)
                    total_splats += ds.n_splats
            aprint(f"Total input splats: {total_splats:,}")

            if channel_colors:
                color_strs = [c.strip() for c in channel_colors.split(",")]
                if len(color_strs) != len(datasets):
                    aprint(
                        f"Error: {len(color_strs)} colors but {len(datasets)} datasets"
                    )
                    raise typer.Exit(1)
                colors = [parse_hex_color(c) for c in color_strs]
                with asection("Merging with channel colors"):
                    merged = GSplatData.merge_with_channel_colors(datasets, colors)

            elif as_dimension:
                if values is not None:
                    dim_values: list[float] = [
                        float(v.strip()) for v in values.split(",")
                    ]
                    if len(dim_values) != len(datasets):
                        aprint(
                            f"Error: {len(dim_values)} values but "
                            f"{len(datasets)} datasets"
                        )
                        raise typer.Exit(1)
                else:
                    dim_values = [float(i) for i in range(len(datasets))]
                with asection(f"Stacking along new dimension (sigma={sigma})"):
                    aprint(f"  Values: {dim_values}")
                    merged = GSplatData.combine_as_new_dimension(
                        datasets, values=dim_values, sigma=sigma
                    )
                    aprint(f"  Result: {merged.ndim}D ({merged.n_splats:,} splats)")

            else:
                with asection("Concatenating"):
                    merged = GSplatData.concatenate(datasets)

            with asection(f"Saving to {output_path.name}"):
                # Color SDR/HDR is auto-detected by the writer.
                merged.save(
                    output_path,
                    encoding_mode=_resolve_encoding_mode(encoding),
                    compress=compress,
                )
                aprint(f"Saved {merged.n_splats:,} splats ({merged.ndim}D)")

        aprint(f"\nDone: {merged.n_splats:,} splats merged")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


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
