"""CLI commands for Gaussian splat operations."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional

import typer
from arbol import aprint, asection

app_gsplat = typer.Typer(help="Gaussian splat tools")


def _ascii_histogram(
    data: "np.ndarray", bins: int = 40, width: int = 60, title: str = "Distribution"
) -> str:
    """Create ASCII histogram from data array."""
    import numpy as np

    # Compute histogram
    counts, bin_edges = np.histogram(data, bins=bins)
    max_count = counts.max()

    if max_count == 0:
        return f"{title}: No data"

    lines = [f"\n{title}:"]
    lines.append("─" * (width + 10))

    # Create bars
    for i, count in enumerate(counts):
        bar_length = int((count / max_count) * width)
        bar = "█" * bar_length
        bin_start = bin_edges[i]

        # Format bin labels
        if bin_start < 0.01 or bin_start > 1000:
            label = f"{bin_start:.2e}"
        else:
            label = f"{bin_start:.4f}"

        lines.append(f"{label:>10s} │ {bar} {count}")

    lines.append("─" * (width + 10))

    return "\n".join(lines)


def _compute_splat_volumes(cholesky_factors: "np.ndarray", ndim: int) -> "np.ndarray":
    """Compute volumes at 3-sigma for each splat."""
    import numpy as np

    # Extract diagonal elements from packed Cholesky factors
    # For nD: positions are at cumsum([1,2,3,...,ndim]) - 1
    diag_indices = np.cumsum(np.arange(1, ndim + 1)) - 1
    diag_elements = cholesky_factors[:, diag_indices]

    # Volume ∝ det(Σ)^(1/2) = |det(L)| = |product of diagonal elements|
    det_L = np.prod(diag_elements, axis=1)
    det_Sigma = det_L**2

    # Volume of nD ellipsoid at 3-sigma
    # V = (2π)^(n/2) * det(Σ)^(1/2) * 3^n / Γ(n/2 + 1)
    # For simplicity, use det(Σ)^(1/2) * 3^n as proxy
    volumes = np.abs(det_Sigma) ** 0.5 * (3**ndim)

    return volumes


def _print_statistics_table(data: "np.ndarray", label: str) -> None:
    """Print formatted statistics table."""
    import numpy as np

    stats = {
        "Min": np.min(data),
        "Max": np.max(data),
        "Mean": np.mean(data),
        "Median": np.median(data),
        "Std": np.std(data),
        "P5": np.percentile(data, 5),
        "P25": np.percentile(data, 25),
        "P75": np.percentile(data, 75),
        "P95": np.percentile(data, 95),
    }

    aprint(f"\n{label} Statistics:")
    aprint("─" * 40)
    for key, value in stats.items():
        if value < 0.01 or value > 1000:
            aprint(f"  {key:<10s}: {value:>12.4e}")
        else:
            aprint(f"  {key:<10s}: {value:>12.6f}")


@app_gsplat.command("info")
def info_dataset(
    path: Path = typer.Argument(
        ..., exists=True, help="Path to .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    show_histograms: bool = typer.Option(
        True, "--histograms/--no-histograms", help="Show ASCII histograms"
    ),
    bins: int = typer.Option(
        40, "--bins", "-b", help="Number of bins for histograms", min=10, max=100
    ),
) -> None:
    """Show detailed information about a Gaussian splat dataset.

    Displays comprehensive statistics including:
    - Number of splats and dimensions
    - Bounding box in each dimension
    - Amplitude distribution with statistics and histogram
    - Volume distribution (size at 3-sigma) with statistics and histogram
    - Sharpness distribution
    - Color information (if present)
    - Metadata (fitting info, provenance, etc.)

    Examples:
        # Basic info with histograms
        luxar gsplat info dataset.gsplats.zarr.zip

        # Info without histograms (faster)
        luxar gsplat info dataset.gsplats.zarr.zip --no-histograms

        # More detailed histograms
        luxar gsplat info dataset.gsplats.zarr.zip --bins 60

    Args:
        path: Path to .gsplats.zarr or compressed archive
        show_histograms: Whether to display ASCII histograms
        bins: Number of bins for histogram plots
    """
    try:
        import numpy as np

        from luxar.gsplats.gsplat_data import GSplatData

        with asection(f"Loading dataset: {path.name}"):
            data = GSplatData.load(path, include_stats=True)
            n_splats = len(data.amplitudes)
            ndim = data.centers.shape[1]

            aprint(f"✓ Loaded {n_splats:,} splats ({ndim}D)")

        # ================================================================
        # Basic Information
        # ================================================================
        aprint("\n" + "═" * 70)
        aprint("DATASET INFORMATION")
        aprint("═" * 70)

        aprint(f"\nFile: {path.name}")
        file_size = path.stat().st_size
        if file_size < 1024:
            aprint(f"Size: {file_size} bytes")
        elif file_size < 1024 * 1024:
            aprint(f"Size: {file_size / 1024:.1f} KB")
        else:
            aprint(f"Size: {file_size / (1024 * 1024):.2f} MB")

        aprint(f"\nSplats: {n_splats:,}")
        aprint(f"Dimensions: {ndim}D")
        aprint(f"Has Colors: {'Yes' if data.colors is not None else 'No'}")
        aprint("Has Sharpness: Yes")  # Always present after load

        # ================================================================
        # Bounding Box
        # ================================================================
        aprint("\n" + "─" * 70)
        aprint("BOUNDING BOX")
        aprint("─" * 70)

        mins = data.centers.min(axis=0)
        maxs = data.centers.max(axis=0)
        ranges = maxs - mins

        for i in range(ndim):
            aprint(
                f"  Dim {i}: [{mins[i]:>10.4f}, {maxs[i]:>10.4f}]  range: {ranges[i]:.4f}"
            )

        total_volume = np.prod(ranges)
        aprint(f"\nTotal Volume: {total_volume:.4e}")

        # ================================================================
        # Amplitude Statistics
        # ================================================================
        aprint("\n" + "─" * 70)
        aprint("AMPLITUDE ANALYSIS")
        aprint("─" * 70)

        _print_statistics_table(data.amplitudes, "Amplitude")

        # Total amplitude
        total_amp = np.sum(data.amplitudes)
        aprint(f"\nTotal Amplitude: {total_amp:.4e}")

        # Top contributors
        sorted_amps = np.sort(data.amplitudes)[::-1]
        cumsum = np.cumsum(sorted_amps)
        cumsum_norm = cumsum / cumsum[-1]

        # Find how many splats contribute to 50%, 90%, 95%, 99%
        for threshold in [0.50, 0.90, 0.95, 0.99]:
            n_contrib = np.searchsorted(cumsum_norm, threshold) + 1
            pct = (n_contrib / n_splats) * 100
            aprint(
                f"  Top {n_contrib:,} splats ({pct:.1f}%) contribute {threshold * 100:.0f}% of total amplitude"
            )

        if show_histograms:
            aprint(
                _ascii_histogram(
                    data.amplitudes, bins=bins, title="Amplitude Distribution"
                )
            )

        # ================================================================
        # Volume Statistics (3-sigma)
        # ================================================================
        aprint("\n" + "─" * 70)
        aprint("VOLUME ANALYSIS (3-Sigma)")
        aprint("─" * 70)

        volumes = _compute_splat_volumes(data.cholesky_factors, ndim)
        _print_statistics_table(volumes, "Volume")

        if show_histograms:
            aprint(
                _ascii_histogram(volumes, bins=bins, title="Volume Distribution (3σ)")
            )

        # ================================================================
        # Sharpness Statistics
        # ================================================================
        aprint("\n" + "─" * 70)
        aprint("SHARPNESS ANALYSIS")
        aprint("─" * 70)

        _print_statistics_table(data.sharpnesses, "Sharpness")

        # Count standard Gaussians (s=2.0)
        n_standard = np.sum(np.abs(data.sharpnesses - 2.0) < 1e-5)
        pct_standard = (n_standard / n_splats) * 100
        aprint(f"\nStandard Gaussians (s=2.0): {n_standard:,} ({pct_standard:.1f}%)")

        if show_histograms and n_standard < n_splats:  # Only if there's variation
            aprint(
                _ascii_histogram(
                    data.sharpnesses, bins=bins, title="Sharpness Distribution"
                )
            )

        # ================================================================
        # Color Information
        # ================================================================
        if data.colors is not None:
            aprint("\n" + "─" * 70)
            aprint("COLOR INFORMATION")
            aprint("─" * 70)

            aprint(f"\nColor dtype: {data.colors.dtype}")
            aprint(f"Color range: [{data.colors.min():.4f}, {data.colors.max():.4f}]")

            # Per-channel statistics
            for i, channel_name in enumerate(["Red", "Green", "Blue"]):
                channel_data = data.colors[:, i]
                aprint(f"\n{channel_name} Channel:")
                aprint(f"  Mean: {np.mean(channel_data):.4f}")
                aprint(f"  Std:  {np.std(channel_data):.4f}")

        # ================================================================
        # Metadata
        # ================================================================
        if data.stats:
            aprint("\n" + "─" * 70)
            aprint("METADATA")
            aprint("─" * 70)

            # Display important metadata
            important_keys = [
                "n_splats",
                "ndim",
                "ordering",
                "format_version",
                "timestamp",
                "luxar_gsplats_version",
                "description",
                "fitter_name",
                "n_iters",
                "final_loss",
                "convergence_time",
                "pruned",
                "pruning_method",
                "n_original",
                "n_removed",
                "amplitude_retention",
            ]

            displayed_keys = set()
            for key in important_keys:
                if key in data.stats:
                    value = data.stats[key]
                    if isinstance(value, float):
                        aprint(f"  {key}: {value:.6f}")
                    else:
                        aprint(f"  {key}: {value}")
                    displayed_keys.add(key)

            # Display remaining metadata
            remaining = set(data.stats.keys()) - displayed_keys
            if remaining:
                aprint("\nAdditional Metadata:")
                for key in sorted(remaining):
                    if key not in [
                        "movie_frames",
                        "movie_shape",
                        "config",
                        "provenance",
                    ]:
                        value = data.stats[key]
                        if isinstance(value, (dict, list)):
                            aprint(
                                f"  {key}: {type(value).__name__} with {len(value)} items"
                            )
                        else:
                            aprint(f"  {key}: {value}")

        # ================================================================
        # Summary
        # ================================================================
        aprint("\n" + "═" * 70)
        aprint("SUMMARY")
        aprint("═" * 70)

        aprint(f"\n✓ Dataset contains {n_splats:,} Gaussian splats in {ndim}D")
        aprint(f"✓ Total amplitude: {total_amp:.4e}")
        aprint(f"✓ Bounding box volume: {total_volume:.4e}")
        aprint(f"✓ Mean splat volume (3σ): {np.mean(volumes):.4e}")

        # Pruning recommendation
        n_for_95pct = np.searchsorted(cumsum_norm, 0.95) + 1
        if n_for_95pct < n_splats * 0.5:  # If less than 50% needed for 95%
            removable = n_splats - n_for_95pct
            pct_removable = (removable / n_splats) * 100
            aprint("\n💡 Pruning Suggestion:")
            aprint(
                f"   You could remove {removable:,} splats ({pct_removable:.1f}%) while retaining 95% of amplitude"
            )
            aprint(
                f"   Command: luxar gsplat prune {path.name} pruned.gsplats.zarr.zip --method cumulative --retention 0.95"
            )

    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app_gsplat.command("napari")
def napari_viewer(
    path: Path = typer.Argument(
        ..., exists=True, help="Path to .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
) -> None:
    """Open a Gaussian splat dataset in napari for inspection.

    This renders the splats back to a volume and displays them in napari.

    Args:
        path: Path to .gsplats.zarr or .gsplats.zarr.zip dataset
    """
    try:
        import napari
    except ImportError:
        aprint("❌ napari not installed")
        aprint("💡 Install with: pip install napari[all]")
        raise typer.Exit(1)

    try:
        from luxar.gsplats.gsplat_data import GSplatData

        with asection(f"Loading gsplat dataset: {path.name}"):
            # Load dataset
            data = GSplatData.load(path, include_stats=True)

            n_splats = len(data.amplitudes)
            ndim = data.centers.shape[1]
            aprint(f"Loaded {n_splats:,} splats ({ndim}D)")

            # Determine volume shape from bounding box
            mins = data.centers.min(axis=0)
            maxs = data.centers.max(axis=0)
            shape = tuple(int(maxs[i] - mins[i]) + 1 for i in range(ndim))

            aprint(f"Bounding box: {shape}")

            # Render to volume
            with asection("Rendering to volume"):
                aprint("This may take a moment for large datasets...")
                volume = data.render_to_volume(
                    shape=shape,
                    device=None,  # Auto-detect best device (cuda/mps/cpu)
                    truncate=3.0,
                )
                aprint(f"Rendered to {volume.shape}")

            # Open in napari
            with asection("Opening napari"):
                viewer = napari.Viewer(title=f"GSplats: {path.name}")

                # Add rendered volume
                viewer.add_image(
                    volume,
                    name="Rendered GSplats",
                    colormap="viridis",
                    blending="additive",
                )

                # Add splat centers as points for reference
                viewer.add_points(
                    data.centers,
                    name="Splat Centers",
                    size=2,
                    opacity=0.3,
                    face_color="red",
                )

                # Show stats in console
                aprint("\nDataset Statistics:")
                if data.stats:
                    for key, value in data.stats.items():
                        if key not in [
                            "movie_frames",
                            "movie_shape",
                            "config",
                            "provenance",
                        ]:
                            aprint(f"  {key}: {value}")

                napari.run()

    except Exception as e:
        aprint(f"❌ Error opening dataset: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app_gsplat.command("view")
def quick_view(
    path: Path = typer.Argument(
        ..., exists=True, help="Path to .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Data server port"),
    viewer_port: int = typer.Option(5173, "--viewer-port", help="Viewer port"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser"),
) -> None:
    """Quick view of a Gaussian splat dataset in the Luxar web viewer.

    This converts the .gsplats.zarr to Luxar's scene format, serves it,
    and opens the viewer. The scene is created in a temporary directory.

    The dataset is automatically centered at its centroid for optimal viewing.

    Args:
        path: Path to .gsplats.zarr or .gsplats.zarr.zip dataset
        port: Port for data server
        viewer_port: Port for viewer
        open_browser: Whether to open browser automatically
    """
    try:
        import threading

        from luxar import Dimensions, LuxarZarrCompiler
        from luxar.cli.main import _serve_data, _serve_viewer
        from luxar.cli.utils import (
            build_viewer,
            check_viewer_built,
            find_available_port,
        )
        from luxar.encoding import EncodingMode
        from luxar.gsplats.gsplat_data import GSplatData

        # Check viewer is built
        if not check_viewer_built():
            aprint("🔨 Building viewer...")
            if not build_viewer():
                aprint("❌ Failed to build viewer")
                raise typer.Exit(1)

        with asection(f"Quick View: {path.name}"):
            # Load gsplat data
            with asection("Loading gsplat dataset"):
                data = GSplatData.load(path, include_stats=False)
                n_splats = len(data.amplitudes)
                ndim = data.centers.shape[1]
                aprint(f"Loaded {n_splats:,} splats ({ndim}D)")

            # Center at origin for better default view
            aprint("Centering at centroid for better view...")
            data = data.center_at_centroid()

            # Create temporary Luxar scene
            temp_dir = Path(tempfile.mkdtemp(prefix="luxar_gsplat_view_"))
            scene_path = temp_dir / "gsplat_scene.zarr"

            with asection("Converting to Luxar scene"):
                aprint(f"Output: {scene_path}")

                # Determine dimensions (centered around origin after centering)

                mins = data.centers.min(axis=0)
                maxs = data.centers.max(axis=0)

                if ndim == 3:
                    dims = Dimensions.default_3d()
                    # Update ranges to match actual centered data
                    for i, dim in enumerate(dims.dimensions):
                        dim.range = (float(mins[i]), float(maxs[i]))
                elif ndim == 2:
                    dims = Dimensions.default_2d()
                    for i, dim in enumerate(dims.dimensions):
                        dim.range = (float(mins[i]), float(maxs[i]))
                else:
                    # Create nD dimensions
                    from luxar import Dimension

                    dims_list = []
                    for i in range(ndim):
                        dims_list.append(
                            Dimension(
                                name=f"dim{i}",
                                unit="voxel",
                                range=(float(mins[i]), float(maxs[i])),
                                step=1.0,
                                display=(i < 3),  # Display first 3 dimensions
                            )
                        )
                    dims = Dimensions(dimensions=dims_list)

                with LuxarZarrCompiler(
                    scene_path, encoding_mode=EncodingMode.MEMORY
                ) as compiler:
                    scene = compiler.create_scene(dimensions=dims)

                    scene.attrs["title"] = f"GSplats: {path.name}"
                    scene.attrs["description"] = (
                        f"Quick view of Gaussian splat dataset from {path.name}"
                    )

                    # Add gsplats
                    scene.add_gsplats_from_data(
                        name="gsplats",
                        result=data,
                        opacity=1.0,
                        blending_mode="additive",
                    )

                aprint(f"Scene created: {scene_path}")

            # Find available ports
            actual_port = find_available_port(port)
            actual_viewer_port = find_available_port(viewer_port)

            if actual_port is None or actual_viewer_port is None:
                aprint("❌ Could not find available ports")
                raise typer.Exit(1)

            with asection("Starting servers"):
                # Start data server in background
                data_thread = threading.Thread(
                    target=_serve_data,
                    args=(scene_path, "127.0.0.1", actual_port, None, None, 0.0, 0.0),
                    daemon=True,
                )
                data_thread.start()
                time.sleep(1)

                # Construct data URL
                data_url = f"http://127.0.0.1:{actual_port}/{scene_path.name}"

                # Serve viewer (this blocks)
                aprint("\n🎉 Viewer ready!")
                _serve_viewer("127.0.0.1", actual_viewer_port, data_url, open_browser)

    except KeyboardInterrupt:
        aprint("\n🛑 Shutting down viewer...")
        # Cleanup temp directory
        if "temp_dir" in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        # Cleanup temp directory
        if "temp_dir" in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise typer.Exit(1)


@app_gsplat.command("prune")
def prune_dataset(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    output_path: Path = typer.Argument(..., help="Output .gsplats.zarr dataset"),
    method: Literal["cumulative", "amplitude_percentile", "combined"] = typer.Option(
        "cumulative",
        "--method",
        "-m",
        help="Pruning strategy: cumulative (retain X% amplitude), amplitude_percentile (remove bottom X%), or combined (low amp OR large vol)",
    ),
    retention: float = typer.Option(
        0.95,
        "--retention",
        "-r",
        help="For cumulative method: fraction of amplitude to retain (0.0-1.0)",
        min=0.0,
        max=1.0,
    ),
    amplitude_percentile: float = typer.Option(
        5.0,
        "--amplitude-percentile",
        "-a",
        help="For amplitude_percentile/combined methods: bottom percentile to remove (0-100)",
        min=0.0,
        max=100.0,
    ),
    volume_percentile: float = typer.Option(
        95.0,
        "--volume-percentile",
        "-v",
        help="For combined method: remove splats above this volume percentile (0-100)",
        min=0.0,
        max=100.0,
    ),
    encoding_mode: Literal["auto", "precision", "memory"] = typer.Option(
        "auto",
        "--encoding",
        "-e",
        help="Encoding mode for output",
    ),
    compress: Optional[Literal["zip", "tar.gz"]] = typer.Option(
        None,
        "--compress",
        "-c",
        help="Compress output as .zip or .tar.gz archive",
    ),
    napari: bool = typer.Option(
        False,
        "--napari",
        "-n",
        help="Open napari comparison viewer after pruning (original vs pruned)",
    ),
) -> None:
    """Prune low-impact splats from a Gaussian splat dataset.

    Pruning removes splats that contribute minimally to the reconstruction,
    reducing file size and rendering cost while preserving quality.

    Methods:
        cumulative (RECOMMENDED): Keep top splats that contribute X% of total amplitude.
            Use --retention to control quality (0.90-0.99, default 0.95).
            Example: --method cumulative --retention 0.95 (keeps 95% of signal)

        amplitude_percentile: Remove bottom X percentile by amplitude.
            Use --amplitude-percentile to set threshold (0-100).
            Example: --method amplitude_percentile --amplitude-percentile 10

        combined: Remove splats with (low amplitude OR large volume).
            Useful for removing artifacts. Use --amplitude-percentile and --volume-percentile.
            Example: --method combined --amplitude-percentile 5 --volume-percentile 95

    Examples:
        # Recommended: Keep 95% of amplitude (removes ~80-90% of splats)
        luxar gsplat prune input.gsplats.zarr.zip output.gsplats.zarr.zip \\
            --method cumulative --retention 0.95

        # With compression (auto-detect format from output extension)
        luxar gsplat prune input.gsplats.zarr.zip output.gsplats.zarr.zip \\
            --method cumulative --retention 0.95 --compress zip

        # Compare before/after in napari
        luxar gsplat prune input.gsplats.zarr.zip output.gsplats.zarr.zip \\
            --method cumulative --retention 0.95 --napari

        # More aggressive: Keep 90% of amplitude
        luxar gsplat prune input.gsplats.zarr output.gsplats.zarr \\
            --method cumulative --retention 0.90

        # Remove artifacts (low amp OR large volume outliers)
        luxar gsplat prune input.gsplats.zarr output.gsplats.zarr \\
            --method combined --amplitude-percentile 5 --volume-percentile 95

    Args:
        input_path: Input .gsplats.zarr or .gsplats.zarr.zip dataset
        output_path: Output path for pruned dataset
        method: Pruning strategy
        retention: Amplitude retention fraction (for cumulative method)
        amplitude_percentile: Bottom percentile to remove (for amplitude_percentile/combined)
        volume_percentile: Volume percentile threshold (for combined method)
        encoding_mode: Encoding mode for output (auto/precision/memory)
        compress: Optional compression format (zip or tar.gz)
        napari: Open napari to compare original vs pruned
    """
    try:
        from luxar.encoding import EncodingMode
        from luxar.gsplats.gsplat_data import GSplatData

        # Map string to EncodingMode
        encoding_map = {
            "auto": EncodingMode.AUTO,
            "precision": EncodingMode.PRECISION,
            "memory": EncodingMode.MEMORY,
        }
        encoding_mode_obj = encoding_map[encoding_mode]

        with asection(f"Pruning: {input_path.name}"):
            # Load dataset
            with asection("Loading dataset"):
                data = GSplatData.load(input_path, include_stats=True)
                n_original = len(data.amplitudes)
                total_amp_original = data.amplitudes.sum()
                aprint(f"Loaded {n_original:,} splats")
                aprint(f"Total amplitude: {total_amp_original:.6f}")

            # Prune
            with asection(f"Pruning ({method})"):
                if method == "cumulative":
                    aprint(f"Target retention: {retention * 100:.0f}%")
                    pruned_data = data.prune(
                        method="cumulative",
                        target_retention=retention,
                    )
                elif method == "amplitude_percentile":
                    aprint(f"Removing bottom {amplitude_percentile}%")
                    pruned_data = data.prune(
                        method="amplitude_percentile",
                        amplitude_percentile=amplitude_percentile,
                    )
                elif method == "combined":
                    aprint(
                        f"Removing: (amp < p{amplitude_percentile}) OR (vol > p{volume_percentile})"
                    )
                    pruned_data = data.prune(
                        method="combined",
                        amplitude_percentile=amplitude_percentile,
                        volume_percentile=volume_percentile,
                    )

                n_pruned = len(pruned_data.amplitudes)
                n_removed = n_original - n_pruned
                total_amp_pruned = pruned_data.amplitudes.sum()
                amp_retention = total_amp_pruned / total_amp_original

                aprint("\nResults:")
                aprint(f"  Original splats: {n_original:,}")
                aprint(f"  Pruned splats:   {n_pruned:,}")
                aprint(
                    f"  Removed:         {n_removed:,} ({100 * n_removed / n_original:.1f}%)"
                )
                aprint(f"  Amplitude retention: {100 * amp_retention:.2f}%")

            # Save pruned dataset
            with asection(f"Saving to {output_path.name}"):
                pruned_data.save(
                    output_path,
                    encoding_mode=encoding_mode_obj,
                    include_fitting_info=True,
                    compress=compress,
                )
                aprint(f"Saved pruned dataset: {output_path}")

                # Show file size
                if output_path.exists():
                    output_size = output_path.stat().st_size
                    if output_size < 1024 * 1024:
                        aprint(f"Output size: {output_size / 1024:.1f} KB")
                    else:
                        aprint(f"Output size: {output_size / (1024 * 1024):.2f} MB")

        aprint(
            f"\n✅ Done! Removed {n_removed:,} splats, retained {100 * amp_retention:.1f}% amplitude"
        )

        # Open napari comparison if requested
        if napari:
            try:
                import napari as napari_module
            except ImportError:
                aprint("\n⚠️  napari not installed, skipping comparison view")
                aprint("Install with: pip install napari[all]")
                return

            with asection("Opening napari comparison"):
                aprint("Rendering both datasets to volumes for comparison...")

                # Determine rendering shape from original data
                import numpy as np

                mins_orig = data.centers.min(axis=0)
                maxs_orig = data.centers.max(axis=0)
                shape_orig = tuple(
                    int(maxs_orig[i] - mins_orig[i]) + 1 for i in range(len(mins_orig))
                )

                # Render original
                aprint(f"Rendering original ({n_original:,} splats)...")
                volume_orig = data.render_to_volume(
                    shape=shape_orig, device=None, truncate=3.0
                )

                # Render pruned
                aprint(f"Rendering pruned ({n_pruned:,} splats)...")
                volume_pruned = pruned_data.render_to_volume(
                    shape=shape_orig, device=None, truncate=3.0
                )

                # Compute difference
                volume_diff = np.abs(volume_orig - volume_pruned)

                # Open napari
                viewer = napari_module.Viewer(
                    title=f"Pruning Comparison: {input_path.name}"
                )

                # Add layers
                viewer.add_image(
                    volume_orig,
                    name=f"Original ({n_original:,} splats)",
                    colormap="green",
                    blending="additive",
                    visible=True,
                )
                viewer.add_image(
                    volume_pruned,
                    name=f"Pruned ({n_pruned:,} splats, {100 * amp_retention:.1f}% amp)",
                    colormap="magenta",
                    blending="additive",
                    visible=True,
                )
                viewer.add_image(
                    volume_diff,
                    name="Difference (|orig - pruned|)",
                    colormap="red",
                    blending="additive",
                    visible=False,
                )

                # Add splat centers as points
                viewer.add_points(
                    data.centers,
                    name="Original Centers",
                    size=2,
                    opacity=0.3,
                    face_color="green",
                    visible=False,
                )
                viewer.add_points(
                    pruned_data.centers,
                    name="Pruned Centers",
                    size=2,
                    opacity=0.3,
                    face_color="magenta",
                    visible=False,
                )

                aprint("\n✓ Napari opened with comparison layers:")
                aprint("  • Green: Original dataset")
                aprint("  • Magenta: Pruned dataset")
                aprint("  • Red: Absolute difference (hidden by default)")
                aprint("  • Points: Splat centers (hidden by default)")
                aprint("\nToggle layers to compare datasets!")

                napari_module.run()

    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
