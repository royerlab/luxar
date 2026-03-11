"""CLI commands for Gaussian splat operations."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import typer
from arbol import aprint, asection

if TYPE_CHECKING:
    import numpy as np

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


# ═══════════════════════════════════════════════════════════════════════
# filter — Filter splats by multiple criteria
# ═══════════════════════════════════════════════════════════════════════


def _parse_bbox(s: str, ndim: int) -> list[tuple[float, float]]:
    """Parse a bbox string 'min0,max0,min1,max1,...' into list of (min, max) pairs."""
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 2 * ndim:
        raise typer.BadParameter(
            f"bbox needs {2 * ndim} values for {ndim}D data, got {len(parts)}"
        )
    return [(parts[2 * i], parts[2 * i + 1]) for i in range(ndim)]


@app_gsplat.command("filter")
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
    # Sharpness
    sharpness_min: Optional[float] = typer.Option(
        None, "--sharpness-min", help="Minimum sharpness"
    ),
    sharpness_max: Optional[float] = typer.Option(
        None, "--sharpness-max", help="Maximum sharpness"
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
    truncate: float = typer.Option(
        3.0, "--truncate", help="Sigma truncation for volume computation"
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
        --sharpness-min/max: Sharpness values (2.0 = standard Gaussian)
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
        from luxar.encoding import EncodingMode
        from luxar.gsplats.gsplat_data import GSplatData

        encoding_map = {
            "auto": EncodingMode.AUTO,
            "precision": EncodingMode.PRECISION,
            "memory": EncodingMode.MEMORY,
        }
        encoding_mode_obj = encoding_map[encoding_mode]

        with asection(f"Filtering: {input_path.name}"):
            # Load
            with asection("Loading dataset"):
                data = GSplatData.load(input_path, include_stats=True)
                n_original = data.n_splats
                ndim = data.ndim
                aprint(f"Loaded {n_original:,} splats ({ndim}D)")

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
                if sharpness_min is not None or sharpness_max is not None:
                    aprint(f"  sharpness: [{sharpness_min}, {sharpness_max}]")
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
                    sharpness_min=sharpness_min,
                    sharpness_max=sharpness_max,
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
                        output_size = output_path.stat().st_size
                        if output_size < 1024 * 1024:
                            aprint(f"  Size: {output_size / 1024:.1f} KB")
                        else:
                            aprint(f"  Size: {output_size / (1024 * 1024):.1f} MB")

    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


# ═══════════════════════════════════════════════════════════════════════
# split — Split dataset into multiple parts
# ═══════════════════════════════════════════════════════════════════════


@app_gsplat.command("split")
def split_dataset(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    output_dir: Path = typer.Argument(..., help="Output directory for split parts"),
    # Split mode (exactly one required)
    parts: Optional[int] = typer.Option(
        None, "--parts", "-n", help="Split into N roughly equal parts"
    ),
    indices: Optional[str] = typer.Option(
        None, "--indices", help="Split at splat indices: '100,500,1000'"
    ),
    # Output options
    encoding_mode: Literal["auto", "precision", "memory"] = typer.Option(
        "auto", "--encoding", "-e", help="Encoding mode for output"
    ),
    compress: Optional[Literal["zip", "tar.gz"]] = typer.Option(
        None, "--compress", "-c", help="Compress output as .zip or .tar.gz"
    ),
) -> None:
    """Split a Gaussian splat dataset into multiple parts.

    Two modes (exactly one required):

    1. Equal parts (--parts N): Split into N roughly equal parts.

    2. At indices (--indices): Split at specific splat indices.

    Output files are named part_000.gsplats.zarr, part_001.gsplats.zarr, etc.
    in the specified output directory.

    Examples:
        # Split into 4 equal parts
        luxar gsplat split input.gsplats.zarr output_dir/ --parts 4

        # Split at specific indices (produces 3 parts: [0:100], [100:500], [500:])
        luxar gsplat split input.gsplats.zarr output_dir/ --indices "100,500"

        # Split with compression
        luxar gsplat split input.gsplats.zarr output_dir/ --parts 3 --compress zip
    """
    try:
        from luxar.encoding import EncodingMode
        from luxar.gsplats.gsplat_data import GSplatData

        # Validate mode
        if parts is None and indices is None:
            aprint("❌ Error: Specify --parts N or --indices '100,500,...'")
            raise typer.Exit(1)
        if parts is not None and indices is not None:
            aprint("❌ Error: --parts and --indices are mutually exclusive")
            raise typer.Exit(1)

        encoding_map = {
            "auto": EncodingMode.AUTO,
            "precision": EncodingMode.PRECISION,
            "memory": EncodingMode.MEMORY,
        }
        encoding_mode_obj = encoding_map[encoding_mode]

        with asection(f"Splitting: {input_path.name}"):
            # Load
            with asection("Loading dataset"):
                data = GSplatData.load(input_path, include_stats=True)
                aprint(f"Loaded {data.n_splats:,} splats ({data.ndim}D)")

            # Split
            with asection("Splitting"):
                if parts is not None:
                    aprint(f"Mode: {parts} equal parts")
                    split_parts = data.split(parts)
                else:
                    idx_list = [int(x.strip()) for x in indices.split(",")]
                    aprint(f"Mode: split at indices {idx_list}")
                    split_parts = data.split(idx_list)

                aprint(f"Produced {len(split_parts)} parts:")
                for i, part in enumerate(split_parts):
                    aprint(f"  part_{i:03d}: {part.n_splats:,} splats")

            # Filter out empty parts
            nonempty_parts = [
                (i, part) for i, part in enumerate(split_parts) if part.n_splats > 0
            ]
            if len(nonempty_parts) < len(split_parts):
                n_empty = len(split_parts) - len(nonempty_parts)
                aprint(f"  Skipping {n_empty} empty part(s)")

            # Save
            output_dir.mkdir(parents=True, exist_ok=True)
            with asection(f"Saving to {output_dir}"):
                ext = ".gsplats.zarr"
                for i, part in nonempty_parts:
                    out_path = output_dir / f"part_{i:03d}{ext}"
                    part.save(
                        out_path,
                        encoding_mode=encoding_mode_obj,
                        include_fitting_info=True,
                        compress=compress,
                    )
                    aprint(f"  Saved {out_path.name} ({part.n_splats:,} splats)")

    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


# ═══════════════════════════════════════════════════════════════════════
# slice — Slice by coordinate ranges (numpy-style syntax)
# ═══════════════════════════════════════════════════════════════════════


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


@app_gsplat.command("slice")
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
        from luxar.encoding import EncodingMode
        from luxar.gsplats.gsplat_data import GSplatData

        encoding_map = {
            "auto": EncodingMode.AUTO,
            "precision": EncodingMode.PRECISION,
            "memory": EncodingMode.MEMORY,
        }
        encoding_mode_obj = encoding_map[encoding_mode]

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
                        output_size = output_path.stat().st_size
                        if output_size < 1024 * 1024:
                            aprint(f"  Size: {output_size / 1024:.1f} KB")
                        else:
                            aprint(f"  Size: {output_size / (1024 * 1024):.1f} MB")

    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


# ═══════════════════════════════════════════════════════════════════════
# fit — Fit Gaussian splats to a volume
# ═══════════════════════════════════════════════════════════════════════


@app_gsplat.command("fit")
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
        None, "--preset", help="Parameter preset: draft/standard/hifi"
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
) -> None:
    """Fit Gaussian splats to a volume.

    Reconstructs an n-dimensional image/volume as a set of oriented Gaussian
    splats. Use presets for quick configuration or a YAML config file for
    full control over all ~35 parameters.

    Presets:
        draft    - Fast preview (500 iters, aggressive culling)
        standard - Balanced quality/speed (3000 iters)
        hifi     - Maximum quality (6000 iters, learnable sharpness)

    Examples:
        luxar gsplat fit volume.npy splats.gsplats.zarr --preset draft --seeds 1000

        luxar gsplat fit volume.tiff splats.gsplats.zarr --preset standard --seeds 8000

        luxar gsplat fit --dump-config --preset hifi > config.yaml
        luxar gsplat fit volume.zarr splats.gsplats.zarr --config config.yaml

        luxar gsplat fit data.zarr splats.gsplats.zarr --channel 1 --timepoint 0
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

            # 2. Build merged config
            cli_overrides = {
                "n_iters": iters,
                "device": device,
                "loss_type": loss,
                "lr": lr,
                "seed_method": seed_method,
                "verbose": verbose,
            }
            fit_config = load_fit_config(preset, config, cli_overrides)

            if preset:
                aprint(f"Preset: {preset}")
            if config:
                aprint(f"Config: {config}")
            aprint(f"Iterations: {fit_config.get('n_iters')}")

            # 3. Parse seeds
            parsed_seeds = parse_seeds(seeds)
            if parsed_seeds is not None:
                aprint(f"Seeds: {parsed_seeds}")
            else:
                aprint("Seeds: auto")

            # 4. Fit
            with asection("Optimization"):
                result = fit_gaussian_splats(volume, seeds=parsed_seeds, **fit_config)

            # 5. Save
            with asection(f"Saving to {output_path.name}"):
                result.save(output_path, compress=compress)
                n_splats = result.n_splats
                aprint(f"Saved {n_splats:,} splats")
                if output_path.exists():
                    size = output_path.stat().st_size
                    if size < 1024 * 1024:
                        aprint(f"File size: {size / 1024:.1f} KB")
                    else:
                        aprint(f"File size: {size / (1024 * 1024):.2f} MB")

        time_s = result.stats.get("time_seconds", 0)
        aprint(f"\nDone: {n_splats:,} splats in {time_s:.1f}s")

    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


# ═══════════════════════════════════════════════════════════════════════
# convert — Convert .gsplats.zarr to Luxar scene
# ═══════════════════════════════════════════════════════════════════════


@app_gsplat.command("convert")
def convert_to_scene(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    output_path: Path = typer.Argument(..., help="Output .zarr scene path"),
    center: bool = typer.Option(
        True, "--center/--no-center", help="Center at amplitude-weighted centroid"
    ),
    scale_intensity: Optional[float] = typer.Option(
        None, "--scale-intensity", help="Scale amplitudes by factor (e.g., 0.1)"
    ),
    opacity: float = typer.Option(1.0, "--opacity", help="Opacity (0.0-1.0)"),
    blending_mode: str = typer.Option(
        "additive", "--blending-mode", help="Blending: additive/normal/max/opaque"
    ),
    encoding: Literal["auto", "precision", "memory"] = typer.Option(
        "auto", "--encoding", "-e", help="Encoding mode"
    ),
) -> None:
    """Convert a .gsplats.zarr dataset to a Luxar scene for the web viewer.

    Creates a persistent Luxar scene zarr that can be served with
    ``luxar serve``. By default, centers the data at the centroid.

    Examples:
        luxar gsplat convert fitted.gsplats.zarr scene.zarr
        luxar gsplat convert fitted.gsplats.zarr scene.zarr --no-center
        luxar gsplat convert fitted.gsplats.zarr scene.zarr --scale-intensity 0.1
    """
    try:
        from luxar import LuxarZarrCompiler
        from luxar.cli.gsplat_config import build_dimensions_from_data
        from luxar.encoding import EncodingMode
        from luxar.gsplats.gsplat_data import GSplatData

        encoding_map = {
            "auto": EncodingMode.AUTO,
            "precision": EncodingMode.PRECISION,
            "memory": EncodingMode.MEMORY,
        }

        with asection(f"Converting: {input_path.name} -> {output_path.name}"):
            with asection("Loading gsplat dataset"):
                data = GSplatData.load(input_path, include_stats=False)
                aprint(f"Loaded {data.n_splats:,} splats ({data.ndim}D)")

            if center:
                aprint("Centering at amplitude-weighted centroid")
                data = data.center_at_centroid()

            if scale_intensity is not None:
                aprint(f"Scaling intensity by {scale_intensity}")
                data = data.scale_intensity(scale_intensity)

            with asection("Creating Luxar scene"):
                dims = build_dimensions_from_data(data.centers)
                with LuxarZarrCompiler(
                    output_path, encoding_mode=encoding_map[encoding]
                ) as compiler:
                    scene = compiler.create_scene(dimensions=dims)
                    scene.add_gsplats_from_data(
                        name="gsplats",
                        result=data,
                        opacity=opacity,
                        blending_mode=blending_mode,
                    )

            aprint(f"\nScene saved: {output_path}")
            aprint(f"Serve with: luxar serve {output_path} --viewer")

    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


# ═══════════════════════════════════════════════════════════════════════
# render — Render gsplats to a volume file
# ═══════════════════════════════════════════════════════════════════════


@app_gsplat.command("render")
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
    truncate: float = typer.Option(
        3.0, "--truncate", "-t", help="Truncation radius in sigma"
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
                    size = output_path.stat().st_size
                    if size < 1024 * 1024:
                        aprint(f"File size: {size / 1024:.1f} KB")
                    else:
                        aprint(f"File size: {size / (1024 * 1024):.2f} MB")

        aprint(f"\nSaved: {output_path}")

    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


# ═══════════════════════════════════════════════════════════════════════
# merge — Combine multiple gsplat datasets
# ═══════════════════════════════════════════════════════════════════════


@app_gsplat.command("merge")
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
        from luxar.encoding import EncodingMode
        from luxar.gsplats.gsplat_data import GSplatData

        encoding_map = {
            "auto": EncodingMode.AUTO,
            "precision": EncodingMode.PRECISION,
            "memory": EncodingMode.MEMORY,
        }

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
                # Determine color_mode for float32 colors
                save_color_mode = None
                if merged.colors is not None:
                    import numpy as np

                    if np.issubdtype(merged.colors.dtype, np.floating):
                        save_color_mode = (
                            "hdr" if np.any(merged.colors > 1.0) else "sdr"
                        )

                merged.save(
                    output_path,
                    encoding_mode=encoding_map[encoding],
                    compress=compress,
                    color_mode=save_color_mode,
                )
                aprint(f"Saved {merged.n_splats:,} splats ({merged.ndim}D)")

        aprint(f"\nDone: {merged.n_splats:,} splats merged")

    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
