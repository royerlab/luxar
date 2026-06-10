"""CLI commands for Gaussian splat operations."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import typer
from arbol import aprint, asection

from .utils import _DEFAULT_CORS_ORIGIN, format_memory_size

if TYPE_CHECKING:
    import numpy as np

app_gsplat = typer.Typer(help="Gaussian splat tools")


def _resolve_encoding_mode(
    mode: str,
) -> Any:
    """Convert a string encoding mode to an EncodingMode enum value."""
    from luxar.encoding import EncodingMode

    _ENCODING_MAP = {
        "auto": EncodingMode.AUTO,
        "precision": EncodingMode.PRECISION,
        "memory": EncodingMode.MEMORY,
    }
    return _ENCODING_MAP[mode]


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
    volumes: np.ndarray = np.abs(det_Sigma) ** 0.5 * (3**ndim)

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
        aprint(f"Size: {format_memory_size(path.stat().st_size)}")

        aprint(f"\nSplats: {n_splats:,}")
        aprint(f"Dimensions: {ndim}D")
        aprint(f"Has Colors: {'Yes' if data.colors is not None else 'No'}")

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
                "psnr_db",
                "ssim",
                "mse",
                "convergence_time",
                "culled",
                "culling_method",
                "n_original",
                "n_culled",
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
            aprint("\n💡 Culling Suggestion:")
            aprint(
                f"   You could remove {removable:,} splats ({pct_removable:.1f}%) while retaining 95% of amplitude"
            )
            aprint(
                f"   Command: luxar gsplat cull {path.name} culled.gsplats.zarr.zip --method cumulative --retention 0.95"
            )

    except typer.Exit:
        raise
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
                    truncate=data.truncation_radius,
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

    except typer.Exit:
        raise
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
    cors_origin: str = typer.Option(
        _DEFAULT_CORS_ORIGIN,
        "--cors-origin",
        help=(
            "Allowed CORS origin. Default 'local' allows localhost/127.0.0.1/::1. "
            "Use '*' to allow any origin without credentials."
        ),
    ),
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
        cors_origin: Allowed CORS origin for both servers (default "local").
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
                    args=(
                        scene_path,
                        "127.0.0.1",
                        actual_port,
                        None,  # bandwidth_mbps
                        None,  # latency_ms
                        0.0,  # jitter_percent
                        0.0,  # packet_loss_rate
                        False,  # allow_sensitive_path
                        cors_origin,
                    ),
                    daemon=True,
                )
                data_thread.start()
                time.sleep(1)

                # Construct data URL
                data_url = f"http://127.0.0.1:{actual_port}/{scene_path.name}"

                # Serve viewer (this blocks)
                aprint("\n🎉 Viewer ready!")
                _serve_viewer(
                    "127.0.0.1",
                    actual_viewer_port,
                    data_url,
                    open_browser,
                    cors_origin,
                )

    except KeyboardInterrupt:
        aprint("\n🛑 Shutting down viewer...")
        # Cleanup temp directory
        if "temp_dir" in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        # Cleanup temp directory
        if "temp_dir" in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise typer.Exit(1)


@app_gsplat.command("cull")
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
    pairs = [(parts[2 * i], parts[2 * i + 1]) for i in range(ndim)]
    for i, (lo, hi) in enumerate(pairs):
        if lo > hi:
            raise typer.BadParameter(f"bbox dimension {i} has min ({lo}) > max ({hi})")
    return pairs


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


# ═══════════════════════════════════════════════════════════════════════
# partition — Partition dataset into multiple parts
# ═══════════════════════════════════════════════════════════════════════


@app_gsplat.command("partition")
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
    """Spatially partition a Gaussian splat dataset (BSP) into ONE
    ``kind=partition`` ``.gsplats.zarr`` file.

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


# ═══════════════════════════════════════════════════════════════════════
# transform — Apply spatial and intensity transforms to gsplat datasets
# ═══════════════════════════════════════════════════════════════════════


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


@app_gsplat.command("transform")
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
        import numpy as np

        from luxar.gsplats.gsplat_data import GSplatData

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
            # Load
            with asection("Loading dataset"):
                data = GSplatData.load(input_path, include_stats=True)
                aprint(f"Loaded {data.n_splats:,} splats ({data.ndim}D)")

            d = data.ndim
            transforms_applied: list[str] = []

            # 1. Scale (per-axis)
            if scale_factors is not None:
                factors = _parse_csv_floats(scale_factors, d, "scale")
                with asection("Applying scale"):
                    aprint(f"Scale factors: {factors}")
                    scale_matrix = np.diag(factors)
                    data = data.transform(scale_matrix)
                    transforms_applied.append(f"scale({scale_factors})")

            # 2. Rotations (3D spatial dims only)
            has_rotation = any(
                r is not None for r in [rotate_x_deg, rotate_y_deg, rotate_z_deg]
            )
            if has_rotation:
                # Determine spatial dimensions
                # For nD data, we assume the last 3 dims are spatial (XYZ)
                # and any preceding dims are non-spatial (e.g., time)
                if d < 3:
                    aprint(
                        f"❌ Rotation requires at least 3 spatial dimensions, got {d}D data"
                    )
                    raise typer.Exit(1)

                with asection("Applying rotation"):
                    # Build 3x3 rotation matrix
                    rot3 = np.eye(3, dtype=np.float64)
                    if rotate_x_deg is not None:
                        rad = np.radians(rotate_x_deg)
                        c, s = np.cos(rad), np.sin(rad)
                        rx = np.array(
                            [[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64
                        )
                        rot3 = rx @ rot3
                        aprint(f"Rotate X: {rotate_x_deg}°")
                        transforms_applied.append(f"rotate_x({rotate_x_deg}°)")

                    if rotate_y_deg is not None:
                        rad = np.radians(rotate_y_deg)
                        c, s = np.cos(rad), np.sin(rad)
                        ry = np.array(
                            [[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64
                        )
                        rot3 = ry @ rot3
                        aprint(f"Rotate Y: {rotate_y_deg}°")
                        transforms_applied.append(f"rotate_y({rotate_y_deg}°)")

                    if rotate_z_deg is not None:
                        rad = np.radians(rotate_z_deg)
                        c, s = np.cos(rad), np.sin(rad)
                        rz = np.array(
                            [[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64
                        )
                        rot3 = rz @ rot3
                        aprint(f"Rotate Z: {rotate_z_deg}°")
                        transforms_applied.append(f"rotate_z({rotate_z_deg}°)")

                    # Embed 3x3 rotation into (d, d) identity matrix
                    # Rotation applies to the last 3 dimensions
                    full_matrix = np.eye(d, dtype=np.float64)
                    full_matrix[d - 3 :, d - 3 :] = rot3
                    data = data.transform(full_matrix)

            # 3. Translate
            if translate_offset is not None:
                offsets = _parse_csv_floats(translate_offset, d, "translate")
                with asection("Applying translation"):
                    aprint(f"Translation: {offsets}")
                    data = data.translate(np.array(offsets, dtype=np.float64))
                    transforms_applied.append(f"translate({translate_offset})")

            # 4. Center at centroid
            if center:
                with asection("Centering at centroid"):
                    data = data.center_at_centroid()
                    aprint("Centered at amplitude-weighted centroid")
                    transforms_applied.append("center")

            # 5. Scale intensity
            if scale_intensity_factor is not None:
                with asection("Scaling intensity"):
                    aprint(f"Intensity scale factor: {scale_intensity_factor}")
                    data = data.scale_intensity(scale_intensity_factor)
                    transforms_applied.append(
                        f"scale_intensity({scale_intensity_factor})"
                    )

            # 6. Normalize intensity
            if normalize_intensity is not None:
                with asection("Normalizing intensity"):
                    current_max = float(data.amplitudes.max())
                    aprint(
                        f"Current max: {current_max:.4f} → target max: {normalize_intensity}"
                    )
                    data = data.normalize_intensity(normalize_intensity)
                    transforms_applied.append(f"normalize({normalize_intensity})")

            # Summary
            aprint(f"\nTransforms applied: {' → '.join(transforms_applied)}")

            # Print new bounding box
            with asection("Result bounding box"):
                for i in range(d):
                    lo = data.centers[:, i].min()
                    hi = data.centers[:, i].max()
                    aprint(f"  Dim {i}: [{lo:.4f}, {hi:.4f}]  range: {hi - lo:.4f}")

            # Save (color SDR/HDR is auto-detected by the writer)
            with asection(f"Saving to {output_path.name}"):
                data.save(
                    output_path,
                    encoding_mode=encoding_mode_obj,
                    include_fitting_info=True,
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


# ═══════════════════════════════════════════════════════════════════════
# denoise — Denoise a volume using Non-Local Means
# ═══════════════════════════════════════════════════════════════════════


@app_gsplat.command("denoise")
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
        from luxar.gsplats.gsplat_data import GSplatData

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
                    output_path, encoding_mode=_resolve_encoding_mode(encoding)
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

    except typer.Exit:
        raise
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


# ═══════════════════════════════════════════════════════════════════════
# compare — Quality metrics against a reference volume
# ═══════════════════════════════════════════════════════════════════════


@app_gsplat.command("compare")
def compare_quality(
    gsplats_path: Path = typer.Argument(
        ..., exists=True, help="Path to .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    reference_path: Path = typer.Argument(
        ..., exists=True, help="Reference volume (.npy, .tiff, .zarr, etc.)"
    ),
    shape: Optional[str] = typer.Option(
        None,
        "--shape",
        help="Output shape as comma-separated ints (overrides reference shape)",
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
    channel: Optional[int] = typer.Option(
        None, "--channel", "-c", help="Channel index for OME-Zarr reference"
    ),
    timepoint: Optional[int] = typer.Option(
        None, "--timepoint", help="Timepoint index for OME-Zarr reference"
    ),
    output_json: Optional[Path] = typer.Option(
        None, "--output-json", "-j", help="Write metrics to JSON file"
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress terminal output (useful with --output-json)",
    ),
) -> None:
    """Compare Gaussian splat reconstruction quality against a reference volume.

    Renders the gsplats back to a volume and computes PSNR, SSIM, MSE,
    relative L2 error, and maximum absolute error.  All heavy computation
    runs on GPU when available.

    Examples:
        luxar gsplat compare fitted.gsplats.zarr original.tiff
        luxar gsplat compare fitted.gsplats.zarr original.npy --device cuda
        luxar gsplat compare fitted.gsplats.zarr original.zarr --output-json metrics.json
        luxar gsplat compare fitted.gsplats.zarr original.zarr -j metrics.json -q
    """
    try:
        import json

        import torch

        from luxar.cli.gsplat_config import load_volume, parse_shape
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.metrics import compute_quality_metrics
        from luxar.gsplats.rendering.volume_rendering import (
            auto_detect_device,
            render_to_volume_tensor,
        )

        with asection("Quality Comparison"):
            # Load gsplat dataset
            with asection("Loading gsplat dataset"):
                data = GSplatData.load(gsplats_path, include_stats=False)
                n_splats = data.n_splats
                ndim = data.ndim
                aprint(f"Loaded {n_splats:,} splats ({ndim}D)")

            # Resolve truncation radius from dataset if not explicitly set
            if truncate is None:
                truncate = data.truncation_radius

            # Load reference volume
            with asection("Loading reference volume"):
                ref_np = load_volume(
                    reference_path, channel=channel, timepoint=timepoint
                )
                ref_shape = ref_np.shape

            # Determine rendering shape
            if shape is not None:
                render_shape = parse_shape(shape)
            else:
                render_shape = ref_shape
            aprint(f"Comparison shape: {render_shape}")

            if len(render_shape) != ndim:
                aprint(
                    f"Dimension mismatch: gsplats are {ndim}D but "
                    f"reference/shape is {len(render_shape)}D"
                )
                raise typer.Exit(1)

            if shape is not None and tuple(render_shape) != tuple(ref_shape):
                aprint(
                    f"Error: --shape {render_shape} does not match "
                    f"reference shape {ref_shape}. "
                    f"Omit --shape to use the reference shape."
                )
                raise typer.Exit(1)

            # Select device
            dev = device if device else auto_detect_device()

            # Render gsplats to tensor (stays on GPU)
            with torch.no_grad():
                with asection(f"Rendering on {dev}"):
                    rendered_t = render_to_volume_tensor(
                        data, shape=render_shape, device=dev, truncate=truncate
                    )
                    aprint(
                        f"Rendered: {rendered_t.shape}, "
                        f"range [{rendered_t.min().item():.4f}, {rendered_t.max().item():.4f}]"
                    )

                # Upload reference to same device
                ref_t = torch.from_numpy(ref_np).to(rendered_t.device)

                # Compute metrics (all on GPU)
                with asection("Computing metrics"):
                    metrics = compute_quality_metrics(rendered_t, ref_t)

            # Compression ratio (handle zarr directories and archives)
            def _total_size(p: Path) -> int:
                if p.is_file():
                    return p.stat().st_size
                # Directory: sum all file sizes recursively
                return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

            gsplats_size = _total_size(gsplats_path)
            ref_size = _total_size(reference_path)
            if ref_size > 0 and gsplats_size > 0:
                metrics["compression_ratio"] = ref_size / gsplats_size

        # Print table
        if not quiet:
            aprint("\n" + "=" * 50)
            aprint("QUALITY COMPARISON")
            aprint("=" * 50)
            aprint(f"\nReference:  {reference_path.name} {ref_shape}")
            aprint(f"GSplats:    {gsplats_path.name} ({n_splats:,} splats)")
            aprint("")
            aprint(f"  MSE:             {metrics['mse']:.6g}")
            aprint(f"  PSNR:            {metrics['psnr_db']:.2f} dB")
            aprint(f"  SSIM:            {metrics['ssim']:.4f}")
            aprint(f"  Rel L2:          {metrics['rel_l2']:.6g}")
            aprint(f"  Max Abs Error:   {metrics['max_abs_error']:.6g}")
            if "compression_ratio" in metrics:
                aprint(f"  Compression:     {metrics['compression_ratio']:.1f}x")
            aprint("=" * 50)

        # JSON output
        if output_json is not None:
            import math

            # Replace non-finite floats (inf/nan) with None for valid JSON
            safe_metrics = {
                k: (v if isinstance(v, (int, str)) or math.isfinite(v) else None)
                for k, v in metrics.items()
            }
            json_data = {
                "gsplats": str(gsplats_path),
                "reference": str(reference_path),
                "n_splats": n_splats,
                "ndim": ndim,
                "shape": list(render_shape),
                **safe_metrics,
            }
            with open(output_json, "w") as f:
                json.dump(json_data, f, indent=2)
            if not quiet:
                aprint(f"\nMetrics written to: {output_json}")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


# ═══════════════════════════════════════════════════════════════════════
# cal — Calibrate splat count K via blind-spot cross-validation
# ═══════════════════════════════════════════════════════════════════════


@app_gsplat.command("cal")
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


# ═══════════════════════════════════════════════════════════════════════
# migrate-format — Convert legacy layouts to the v3.0 node-tree format
# ═══════════════════════════════════════════════════════════════════════


@app_gsplat.command("migrate-format")
def migrate_format_command(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        help="Legacy .gsplats.zarr (v1.0 / v1.1 / v2.0), .gsplats.zarr.zip/.tar.gz, "
        "or a substitutive directory (with manifest.json + level_<i>.gsplats.zarr).",
    ),
    output_path: Path = typer.Argument(..., help="Output .gsplats.zarr (v3.0)."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite output if it exists."
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress the trailing 'wrote …' summary."
    ),
) -> None:
    """Convert a legacy .gsplats.zarr layout to the v3.0 node-tree format.

    Four input shapes are auto-detected:

    \b
    * v1.0  .gsplats.zarr (single flat splat set)
    * v1.1  .gsplats.zarr (multi-LOD additive, /splats/lod_<i>/ subgroups)
    * v2.0  .gsplats.zarr (2-D substitutive_<s>/additive_<a> matrix)
    * substitutive directory (manifest.json + level_<i>.gsplats.zarr files)

    All migrate to a single v3.0 ``.gsplats.zarr`` node subtree.
    """
    try:
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.migrate import migrate_format

        with asection(f"Migrating {input_path.name} → v3.0"):
            detected = migrate_format(input_path, output_path, overwrite=overwrite)
            aprint(f"Detected legacy format: {detected}")

            # Post-write read-back: confirm the output is a loadable v3.0 file
            # rather than reporting success blind.
            import zarr

            verify = GSplatData.load(output_path, include_stats=False)
            out_attrs = dict(zarr.open_group(str(output_path), mode="r").attrs)
            fmt = out_attrs.get("format_version")
            if fmt != "3.0":
                aprint(f"❌ Migration produced format_version={fmt!r}, expected '3.0'")
                raise typer.Exit(1)
            if not quiet:
                aprint(
                    f"✓ Verified v3.0 output: {verify.n_splats:,} splats, "
                    f"{verify.n_substitutive} substitutive level(s) → "
                    f"{output_path}"
                )
    except typer.Exit:
        raise
    except FileNotFoundError as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1)
    except ValueError as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1)
    except Exception as exc:
        aprint(f"Error: {exc}")
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


# ═══════════════════════════════════════════════════════════════════════
# benchmark — GPU performance profiling
# ═══════════════════════════════════════════════════════════════════════


@app_gsplat.command("benchmark")
def benchmark_gpu(
    force: bool = typer.Option(
        False, "--force", help="Re-run even if a profile exists for this GPU"
    ),
    list_gpus: bool = typer.Option(False, "--list", help="List profiled GPUs and exit"),
    sweep: bool = typer.Option(
        True, "--sweep/--no-sweep", help="Include splat count sweep"
    ),
    shape: Optional[str] = typer.Option(
        None, "--shape", help="Sweep volume shape (e.g., '512,512,512')"
    ),
    slurm: bool = typer.Option(False, "--slurm", help="Submit as a one-shot Slurm job"),
    partition: Optional[str] = typer.Option(
        None, "--partition", help="Slurm partition (for --slurm)"
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Verbose output"),
) -> None:
    """Benchmark CUDA kernels and build a GPU performance profile.

    Runs the CUDA benchmark suite to characterize GPU throughput, OOM
    boundaries, and optimal operating points. Results are stored in
    ~/.luxar/gpu_profiles.yaml for use by `luxar gsplat batch`.

    Multiple runs on the same GPU are aggregated (averaged throughput,
    conservative OOM boundaries).

    Examples:
        luxar gsplat benchmark
        luxar gsplat benchmark --list
        luxar gsplat benchmark --force
        luxar gsplat benchmark --slurm --partition gpu
    """
    from luxar.gsplats.gpu_profile import (
        PROFILE_PATH,
        load_profiles,
    )

    # --list mode
    if list_gpus:
        profiles = load_profiles()
        gpus = profiles.get("gpus", {})
        if not gpus:
            aprint("No GPU profiles found.")
            aprint("Run `luxar gsplat benchmark` to create one.")
            raise typer.Exit(0)

        with asection("Profiled GPUs"):
            for name, entry in gpus.items():
                info = entry.get("info", {})
                summary = entry.get("summary", {})
                n_runs = len(entry.get("runs", []))
                recs = summary.get("recommendations", {})
                peak = recs.get("peak_throughput_3d", {})

                aprint(f"\n{name}")
                aprint(f"  Memory: {info.get('total_memory_gb', '?')} GB")
                aprint(f"  Compute: sm_{info.get('compute_capability', '?')}")
                aprint(f"  Benchmark runs: {n_runs}")
                if peak:
                    shape_str = "x".join(str(s) for s in peak.get("shape", []))
                    aprint(
                        f"  Peak 3D: {peak.get('gvoxel_per_s', '?')} GV/s "
                        f"at {shape_str}"
                    )
                oom = summary.get("oom_boundaries", {}).get("3d", {})
                if oom.get("max_successful_shape"):
                    shape_str = "x".join(str(s) for s in oom["max_successful_shape"])
                    aprint(f"  Max safe 3D: {shape_str}")

        raise typer.Exit(0)

    # --slurm mode: submit a one-shot job
    if slurm:
        if not partition:
            aprint("Error: --partition is required with --slurm")
            raise typer.Exit(1)

        import subprocess
        import tempfile

        from luxar.gsplats.batch.env_capture import (
            capture_environment,
            generate_env_preamble,
        )

        env = capture_environment()
        preamble = generate_env_preamble(env)

        script = (
            "#!/bin/bash\n"
            f"#SBATCH --job-name=luxar-benchmark\n"
            f"#SBATCH --partition={partition}\n"
            "#SBATCH --ntasks=1\n"
            "#SBATCH --gpus-per-task=1\n"
            "#SBATCH --cpus-per-task=4\n"
            "#SBATCH --mem=32G\n"
            "#SBATCH --time=00:30:00\n"
            "#SBATCH --output=luxar-benchmark.out\n"
            "#SBATCH --error=luxar-benchmark.err\n\n"
            f"{preamble}\n\n"
            "luxar gsplat benchmark --force"
            f"{' --no-sweep' if not sweep else ''}"
            f"{' --shape ' + shape if shape else ''}\n"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sbatch", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            aprint(f"Submitting benchmark job to partition '{partition}'...")
            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                aprint(f"Error submitting job: {result.stderr}")
                raise typer.Exit(1)
            aprint(result.stdout.strip())
            aprint(f"Profile will be saved to: {PROFILE_PATH}")
        finally:
            import os

            os.unlink(script_path)
        raise typer.Exit(0)

    # Direct benchmark run
    try:
        import torch
    except ImportError:
        aprint("Error: PyTorch is required for GPU benchmarking.")
        raise typer.Exit(1)

    if not torch.cuda.is_available():
        aprint("Error: CUDA is not available on this system.")
        aprint("Run `luxar gsplat benchmark --slurm` to benchmark on a GPU node.")
        raise typer.Exit(1)

    gpu_name_detected = torch.cuda.get_device_properties(0).name

    # Check if profile already exists
    if not force:
        profiles = load_profiles()
        if gpu_name_detected in profiles.get("gpus", {}):
            n_runs = len(profiles["gpus"][gpu_name_detected].get("runs", []))
            aprint(f"Profile already exists for {gpu_name_detected} ({n_runs} runs).")
            aprint("Use --force to add another run, or --list to view.")
            raise typer.Exit(0)

    try:
        from luxar.gsplats.models.gsplats.cuda.benchmark import (
            generate_profile,
            run_benchmark,
            run_splat_sweep,
        )
    except ImportError:
        aprint("Error: CUDA splatting backend is not compiled.")
        aprint("Build it with: make build-cuda")
        raise typer.Exit(1)

    with asection(f"Benchmarking GPU: {gpu_name_detected}"):
        results = run_benchmark(verbose=verbose)

        sweep_results = None
        sweep_shape_parsed = None
        if sweep:
            if shape:
                sweep_shape_parsed = tuple(int(s) for s in shape.split(","))
            elif results:
                best_gvs = 0.0
                for _label, r in results.items():
                    if r["dim"] == 3 and r.get("gvoxel_per_s_fp32") is not None:
                        if r["gvoxel_per_s_fp32"] > best_gvs:
                            best_gvs = r["gvoxel_per_s_fp32"]
                            sweep_shape_parsed = r["shape"]
            if sweep_shape_parsed is None:
                sweep_shape_parsed = (512, 512, 512)

            sweep_results = run_splat_sweep(shape=sweep_shape_parsed, verbose=verbose)

        profile_path = generate_profile(
            benchmark_results=results or {},
            sweep_results=sweep_results,
            sweep_shape=sweep_shape_parsed,
        )

    aprint(f"\nProfile saved to: {profile_path}")
    aprint("View with: luxar gsplat benchmark --list")


# ═══════════════════════════════════════════════════════════════════════
# batch — HPC batch fitting via Slurm
# ═══════════════════════════════════════════════════════════════════════

app_batch = typer.Typer(help="HPC batch fitting for large OME-Zarr datasets")
app_gsplat.add_typer(app_batch, name="batch")


@app_batch.command("plan")
def batch_plan(
    input_path: Path = typer.Argument(..., exists=True, help="Input OME-Zarr dataset"),
    output_dir: Path = typer.Argument(..., help="Output directory for batch results"),
    # Tiling
    tile_size: Optional[int] = typer.Option(
        None,
        "--tile-size",
        help="Tile size in voxels (auto from GPU profile if omitted)",
    ),
    tile_overlap: int = typer.Option(32, "--overlap", help="Tile overlap in voxels"),
    # Fit params
    preset: str = typer.Option("standard", "--preset", help="Fitting preset"),
    config: Optional[Path] = typer.Option(None, "--config", help="YAML fit config"),
    seeds: Optional[str] = typer.Option(None, "--seeds", help="Seed count or ratio"),
    iters: Optional[int] = typer.Option(
        None, "--iters", "-n", help="Max optimization iterations (overrides preset)"
    ),
    # Progressive fitting
    batch_progressive: bool = typer.Option(
        False,
        "--progressive",
        help="Use progressive fitting per tile (multi-LOD). "
        "Combine with --parallel for better GPU utilization.",
    ),
    batch_splats_per_pass: Optional[int] = typer.Option(
        None, "--splats-per-pass", help="Max splats per progressive pass"
    ),
    batch_psnr_patience: Optional[float] = typer.Option(
        None, "--psnr-patience", help="PSNR patience for progressive fitting (dB)"
    ),
    batch_max_passes: Optional[int] = typer.Option(
        None, "--max-passes", help="Max progressive passes per tile"
    ),
    # Post-fit culling
    batch_cull_retention: Optional[float] = typer.Option(
        None,
        "--cull-retention",
        help="After fitting each tile, remove the weakest splats that "
        "collectively contribute less than (1 - value) of the total amplitude. "
        "Default 0.95 (discard bottom 5%%). Set to 0 to keep every splat.",
    ),
    # Denoising
    batch_denoise: bool = typer.Option(
        False,
        "--denoise",
        help="Denoise volumes before fitting (NLM). Auto-calibrates h per channel.",
    ),
    batch_denoise_h: Optional[float] = typer.Option(
        None, "--denoise-h", help="Manual NLM h (skip calibration)"
    ),
    batch_denoise_2d: bool = typer.Option(
        False, "--denoise-2d", help="Use 2D NLM (slice-by-slice) instead of 3D"
    ),
    batch_denoise_patch_size: int = typer.Option(
        3, "--denoise-patch-size", help="NLM patch size"
    ),
    batch_denoise_search_distance: int = typer.Option(
        5, "--denoise-search-distance", help="NLM search distance"
    ),
    batch_denoise_backend: str = typer.Option(
        "auto", "--denoise-backend", help="NLM backend"
    ),
    batch_calibration_samples: int = typer.Option(
        5, "--calibration-samples", help="Timepoints to sample for h calibration"
    ),
    batch_preprocess: Optional[bool] = typer.Option(
        None,
        "--preprocess/--no-preprocess",
        help="Write denoised volumes to zarr before fitting (default: off, denoise per-tile on-the-fly).",
    ),
    # Slurm params
    partition: Optional[str] = typer.Option(
        None, "--partition", "-p", help="Slurm partition"
    ),
    max_concurrent: Optional[int] = typer.Option(
        None,
        "--max-concurrent",
        help="Maximum simultaneous Slurm array tasks (limits cluster usage). "
        "Maps to --array=0-N%%MAX. No limit if omitted.",
    ),
    preemptible: bool = typer.Option(
        False,
        "--preemptible",
        help="Also submit tasks on a preemptible partition for extra throughput. "
        "Auto-detects the preemptible partition. Preempted tasks are automatically "
        "requeued. Uses atomic tile writes to handle interruptions safely.",
    ),
    preemptible_partition_opt: Optional[str] = typer.Option(
        None,
        "--preemptible-partition",
        help="Explicit preemptible partition name (skip auto-detection).",
    ),
    preemptible_concurrent: Optional[int] = typer.Option(
        None,
        "--preemptible-concurrent",
        help="Max concurrent tasks on preemptible partition. "
        "Defaults to same as --max-concurrent.",
    ),
    account: Optional[str] = typer.Option(None, "--account", "-A"),
    qos: Optional[str] = typer.Option(None, "--qos"),
    gpus: int = typer.Option(1, "--gpus", help="GPUs per task"),
    cpus: int = typer.Option(4, "--cpus", help="CPUs per task"),
    mem: int = typer.Option(32, "--mem", help="Memory per task (GB)"),
    time_limit: Optional[str] = typer.Option(
        None, "--time", help="Wall time per task override (HH:MM:SS)"
    ),
    gpu_name_opt: Optional[str] = typer.Option(
        None, "--gpu", help="GPU name from profile (auto-detect if omitted)"
    ),
    gpu_mem: Optional[int] = typer.Option(
        None, "--gpu-mem", help="Target GPU memory in GB (picks closest profile)"
    ),
    # Merge
    channel_colors: Optional[str] = typer.Option(
        None, "--channel-colors", help="Hex colors for per-channel merge"
    ),
    # Dataset structure override
    axes: Optional[str] = typer.Option(
        None,
        "--axes",
        help=(
            "Comma-separated axis names overriding auto-detection, e.g. "
            "'time,camera,channel,z,y,x'. Recognised special names: "
            "time/t (timepoint), channel/c/ch/camera/cam (channel), "
            "z/y/x/depth/height/width (spatial)."
        ),
    ),
    timepoints_slice: Optional[str] = typer.Option(
        None,
        "--timepoints",
        help=(
            "Python-style slice to select timepoints, e.g. "
            "'0:10' (first 10), '::10' (every 10th), '100:200:5' (100-200 step 5). "
            "Default: all timepoints."
        ),
    ),
    channels_slice: Optional[str] = typer.Option(
        None,
        "--channels",
        help=(
            "Python-style slice to select channels, e.g. "
            "'0:2' (first 2 channels), '::2' (every other). "
            "Default: all channels."
        ),
    ),
    # Packing
    tasks_per_job: Optional[int] = typer.Option(
        None,
        "--tasks-per-job",
        help=(
            "Number of fitting tasks to run per Slurm job. "
            "Auto-calculated from GPU capacity when omitted. "
            "Packing multiple small volumes per GPU reduces scheduling overhead."
        ),
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel/--sequential",
        help=(
            "Run packed tasks concurrently (--parallel) or one by one "
            "(--sequential, default). Parallel mode launches multiple fit "
            "processes sharing the same GPU — higher throughput but uses "
            "more GPU memory."
        ),
    ),
    # Array selection
    array_key: Optional[str] = typer.Option(
        None,
        "--array-key",
        help=(
            "Key path to a specific array within the zarr store, e.g. "
            "'h2afva/fused'. Useful when a store contains multiple groups "
            "with different arrays. Auto-selects largest array if omitted."
        ),
    ),
    # Control
    submit: bool = typer.Option(
        False, "--submit", help="Actually submit to Slurm (default: dry-run)"
    ),
) -> None:
    """Plan and optionally submit a batch Gaussian splat fitting job.

    Discovers T/C/spatial structure from OME-Zarr metadata, loads a GPU
    profile to auto-select tile size, and generates Slurm array + merge
    jobs.

    By default shows the plan without submitting. Pass --submit to submit.

    A GPU profile from `luxar gsplat benchmark` is used to auto-select tile
    size; pass --tile-size to skip the profile requirement.

    Examples:
        luxar gsplat batch data.ome.zarr output/ --partition gpu --tile-size 128

        luxar gsplat batch data.ome.zarr output/ --partition gpu --submit

        luxar gsplat batch data.ome.zarr output/ -p gpu --tile-size 256 --preset hifi

        luxar gsplat batch keller.zarr.zip out/ -p gpu --tile-size 128 \\
            --axes time,camera,channel,z,y,x
    """
    if partition is None:
        aprint("Error: --partition is required")
        raise typer.Exit(1)

    try:
        import datetime
        import subprocess

        from luxar.cli.gsplat_config import (
            PRESETS,
            decode_flat_channel_index,
            discover_ome_zarr_shape,
        )
        from luxar.gsplats.batch.env_capture import (
            capture_environment,
            generate_env_preamble,
            get_slurm_scheduler_info,
            is_slurm_mps_available,
        )
        from luxar.gsplats.batch.manifest import (
            BatchJob,
            BatchManifest,
            output_filename,
            save_manifest,
        )
        from luxar.gsplats.batch.slurm_gen import (
            generate_fit_sbatch,
            generate_merge_sbatch,
        )
        from luxar.gsplats.batch.time_estimate import (
            estimate_slurm_time_limit,
            estimate_tile_wall_seconds,
        )
        from luxar.gsplats.gpu_profile import (
            get_gpu_summary,
            get_gpu_throughput_table,
            load_profiles,
        )
        from luxar.gsplats.tiling import compute_tile_specs

        # 1. Load GPU profile (required only for auto tile-size)
        axes_list = [a.strip() for a in axes.split(",")] if axes else None
        summary = get_gpu_summary(
            gpu_name=gpu_name_opt,
            gpu_mem=float(gpu_mem) if gpu_mem else None,
        )
        if summary is None and tile_size is None:
            aprint("Error: No GPU benchmark profile found.")
            aprint("")
            aprint("Option A — run the benchmark first (recommended):")
            aprint(
                "  luxar gsplat benchmark --slurm --partition "
                + (partition or "<partition>")
            )
            aprint("")
            aprint("Option B — skip the profile by providing a tile size explicitly:")
            aprint("  luxar gsplat batch ... --tile-size 128")
            raise typer.Exit(1)

        recs = (summary or {}).get("recommendations", {})
        peak = recs.get("peak_throughput_3d", {})

        # Resolve GPU name for display
        profiles = load_profiles()
        resolved_gpu = gpu_name_opt
        if summary is not None and resolved_gpu is None:
            for name, entry in profiles.get("gpus", {}).items():
                if entry.get("summary") == summary:
                    resolved_gpu = name
                    break
        if resolved_gpu is None and profiles.get("gpus"):
            resolved_gpu = next(iter(profiles["gpus"]))
        resolved_gpu = resolved_gpu or "unknown"

        # 2. Discover dataset shape
        # Helper: parse Python-style slice string "start:stop:step"
        def _parse_slice(s: str, max_val: int) -> list[int]:
            parts = s.split(":")
            if len(parts) == 1:
                # Single index
                return [int(parts[0])]
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1]) if len(parts) > 1 and parts[1] else max_val
            step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
            return list(range(start, stop, step))

        with asection("Discovering dataset shape"):
            ome_info = discover_ome_zarr_shape(
                input_path, axes_override=axes_list, array_key=array_key
            )
            n_t_full = ome_info.n_timepoints
            n_c_full = ome_info.n_channels
            spatial = ome_info.spatial_shape
            aprint(f"Axes: {ome_info.axes}")
            aprint(f"Shape: {ome_info.shape}")
            aprint(
                f"T={n_t_full}, C={n_c_full}, spatial={'x'.join(str(s) for s in spatial)}"
            )

            # Apply --timepoints / --channels slicing
            t_indices = (
                _parse_slice(timepoints_slice, n_t_full)
                if timepoints_slice
                else list(range(n_t_full))
            )
            c_indices = (
                _parse_slice(channels_slice, n_c_full)
                if channels_slice
                else list(range(n_c_full))
            )
            if not t_indices:
                raise ValueError("--timepoints selected no timepoints")
            if not c_indices:
                raise ValueError("--channels selected no channels")
            bad_t = [idx for idx in t_indices if idx < 0 or idx >= n_t_full]
            bad_c = [idx for idx in c_indices if idx < 0 or idx >= n_c_full]
            if bad_t:
                raise ValueError(
                    f"--timepoints selected out-of-range indices {bad_t}; valid range is 0..{n_t_full - 1}"
                )
            if bad_c:
                raise ValueError(
                    f"--channels selected out-of-range flat channel indices {bad_c}; valid range is 0..{n_c_full - 1}"
                )
            n_t = len(t_indices)
            n_c = len(c_indices)
            if timepoints_slice or channels_slice:
                aprint(f"Sliced: T={n_t} (of {n_t_full}), C={n_c} (of {n_c_full})")

        # 3. Pick tile size
        #
        # The goal is to choose the largest tile that fits in GPU memory.
        # For anisotropic volumes (e.g. 108×1352×532) the old logic
        # `min(peak_shape[0], *spatial)` would cap at the smallest dim (108),
        # producing hundreds of tiny tiles even when the whole volume fits.
        #
        # New logic: compare total spatial voxels against max safe voxel
        # count from the benchmark.  If the volume fits, skip tiling entirely.
        import math

        auto_tile = tile_size is None

        # Compute max safe shape from GPU profile (used by both auto-tile
        # and tasks-per-job packing).  Falls back to a conservative default.
        peak_shape = peak.get("shape", [])
        oom = (summary or {}).get("oom_boundaries", {}).get("3d", {})
        max_shape = oom.get("max_successful_shape", peak_shape)
        total_voxels = math.prod(spatial)

        if auto_tile:
            assert summary is not None

            max_safe_voxels = math.prod(max_shape) if max_shape else 256**3

            if total_voxels <= max_safe_voxels:
                # Whole volume fits — set tile_size large enough that
                # stride (= tile_size - overlap) exceeds every spatial dim,
                # guaranteeing compute_tile_specs produces exactly 1 tile.
                tile_size = max(spatial) + tile_overlap
            else:
                # Volume is too large — tile it.  Use the cube root of max
                # safe voxels as the isotropic tile edge length, clamped to
                # the largest spatial dim.
                tile_edge = int(max_safe_voxels ** (1.0 / len(spatial)))
                tile_size = min(tile_edge, max(spatial))

        assert tile_size is not None  # narrowed by branches above

        # 4. Compute tile grid — always use compute_tile_specs to get the
        # authoritative tile count (overlap can create extra tiles even when
        # volume_shape == tile_size).
        specs = compute_tile_specs(spatial, tile_size, tile_overlap)
        n_tiles = len(specs)
        needs_tiling = n_tiles > 1

        total_tasks = n_t * n_c * n_tiles

        # 5. Estimate wall time per task
        if needs_tiling:
            tile_voxels = tile_size ** len(spatial)
        else:
            # Single tile — use the actual volume size
            tile_voxels = 1
            for s in spatial:
                tile_voxels *= s

        throughput_table = get_gpu_throughput_table(gpu_name=resolved_gpu)

        preset_config = PRESETS.get(preset, PRESETS["standard"])
        n_iters = iters if iters is not None else preset_config.get("n_iters", 3000)

        if throughput_table:
            est_seconds = estimate_tile_wall_seconds(
                tile_voxels, n_iters, throughput_table
            )
        else:
            est_seconds = 600.0

        # 5b. Compute tasks-per-job packing
        #
        # When each volume is small relative to GPU capacity, we pack
        # multiple fitting tasks sequentially into one Slurm job to
        # reduce scheduling overhead (fewer array elements to launch).
        # 5b-i. Query scheduler for smart packing defaults
        sched_info = get_slurm_scheduler_info()
        uses_backfill = sched_info["uses_backfill"]
        no_job_limit = sched_info["max_jobs_per_user"] is None

        if tasks_per_job is None:
            max_safe = math.prod(max_shape) if max_shape else tile_voxels

            if parallel:
                # Each concurrent fit holds the volume tensor + model params
                # + optimizer state.  ~2× the raw volume is a safe estimate.
                packing = max(1, int(max_safe / max(tile_voxels * 2, 1)))
            else:
                packing = max(1, int(max_safe / max(tile_voxels, 1)))

            # On backfill clusters with no job limit, prefer shorter jobs
            # (more jobs = more backfill opportunities = faster throughput).
            # Cap packing lower so individual jobs stay short.
            if uses_backfill and no_job_limit:
                if parallel:
                    # Parallel: already short, keep the memory-based packing
                    packing = min(packing, 4)
                else:
                    # Sequential: each extra task adds wall-time.
                    # Keep jobs under ~5 min for best backfill scheduling.
                    if est_seconds > 0:
                        max_tasks_for_5min = max(1, int(300 / est_seconds))
                        packing = min(packing, max_tasks_for_5min)
                    packing = min(packing, 3)
            else:
                packing = min(packing, 10)

            tasks_per_job = packing
        tasks_per_job = max(1, tasks_per_job)

        n_slurm_jobs = math.ceil(total_tasks / tasks_per_job)
        if parallel:
            # Parallel: all tasks run at once, so wall time ≈ 1 task
            est_seconds_per_job = est_seconds * 1.2  # 20% overhead for contention
        else:
            est_seconds_per_job = est_seconds * tasks_per_job
        slurm_time = time_limit or estimate_slurm_time_limit(est_seconds_per_job)
        total_gpu_hours = est_seconds * total_tasks / 3600.0

        # 6. Build manifest
        fit_args = {}
        if seeds:
            fit_args["seeds"] = seeds
        if iters is not None:
            fit_args["iters"] = str(iters)
        if config:
            fit_args["config"] = str(config)
        if batch_progressive:
            fit_args["progressive"] = ""  # boolean flag, no value
        if batch_splats_per_pass is not None:
            fit_args["splats-per-pass"] = str(batch_splats_per_pass)
        if batch_psnr_patience is not None:
            fit_args["psnr-patience"] = str(batch_psnr_patience)
        if batch_max_passes is not None:
            fit_args["max-passes"] = str(batch_max_passes)
        if batch_cull_retention is not None:
            fit_args["cull-retention"] = str(batch_cull_retention)

        # Denoise mode detection
        denoise_mode = None
        denoised_zarr_path = None
        if batch_denoise:
            if batch_preprocess is True:
                denoise_mode = "preprocess"
            elif batch_preprocess is False:
                denoise_mode = "on-the-fly"
            else:
                # Default: on-the-fly (denoise per-tile inside each fit task).
                # Use --preprocess to write denoised zarr separately.
                denoise_mode = "on-the-fly"
            aprint(f"Denoise mode: {denoise_mode}")

            if denoise_mode == "preprocess":
                denoised_zarr_path = str(output_dir.resolve() / "denoised.zarr")

            # For on-the-fly mode, pass denoise flags to fit tasks
            if denoise_mode == "on-the-fly":
                fit_args["denoise"] = ""
                if batch_denoise_2d:
                    fit_args["denoise-2d"] = ""
                if batch_denoise_patch_size != 3:
                    fit_args["denoise-patch-size"] = str(batch_denoise_patch_size)
                if batch_denoise_search_distance != 5:
                    fit_args["denoise-search-distance"] = str(
                        batch_denoise_search_distance
                    )
                if batch_denoise_backend != "auto":
                    fit_args["denoise-backend"] = batch_denoise_backend
                # Note: --denoise-h is passed at runtime from h_values JSON

        colors_list = None
        if channel_colors:
            colors_list = [c.strip() for c in channel_colors.split(",")]

        # Preemptible partition detection
        preempt_partition: Optional[str] = None
        if preemptible:
            if preemptible_partition_opt:
                preempt_partition = preemptible_partition_opt
            else:
                from luxar.gsplats.batch.env_capture import (
                    detect_preemptible_gpu_partition,
                )

                preempt_partition = detect_preemptible_gpu_partition()

            if preempt_partition is None:
                aprint(
                    "No preemptible GPU partition found on this cluster.\n"
                    "  Checked all partitions for: preemptible naming + GPU resources.\n"
                    "  Use --preemptible-partition to specify one explicitly.\n"
                    "  Continuing with guaranteed partition only."
                )
            else:
                from luxar.gsplats.batch.env_capture import validate_partition_access

                if not validate_partition_access(preempt_partition):
                    aprint(
                        f"Cannot submit to preemptible partition '{preempt_partition}'.\n"
                        f"  Your account may not have access.\n"
                        f"  To check: sacctmgr show assoc user=$USER partition={preempt_partition}\n"
                        "  Continuing with guaranteed partition only."
                    )
                    preempt_partition = None
                else:
                    aprint(f"Preemptible partition: {preempt_partition}")

        manifest = BatchManifest(
            version=1,
            created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            input_path=str(input_path.resolve()),
            output_dir=str(output_dir.resolve()),
            array_key=array_key,
            n_timepoints=n_t,
            n_channels=n_c,
            channel_axes=ome_info.channel_axes,
            channel_shape=ome_info.channel_shape,
            spatial_shape=spatial,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            n_tiles=n_tiles,
            total_tasks=total_tasks,
            preset=preset,
            fit_args=fit_args,
            gpu_name=resolved_gpu,
            estimated_seconds_per_task=est_seconds,
            slurm_time_limit=slurm_time,
            slurm_partition=partition,
            slurm_account=account,
            slurm_qos=qos,
            slurm_gpus=gpus,
            slurm_cpus=cpus,
            slurm_mem_gb=mem,
            tasks_per_job=tasks_per_job,
            parallel_tasks_per_job=parallel,
            max_concurrent=max_concurrent,
            preemptible=preempt_partition is not None,
            preemptible_partition=preempt_partition,
            preemptible_max_concurrent=(
                (preemptible_concurrent or max_concurrent)
                if preempt_partition
                else None
            ),
            timepoint_indices=t_indices if timepoints_slice else None,
            channel_indices=c_indices if channels_slice else None,
            channel_colors=colors_list,
            denoise=batch_denoise,
            denoise_2d=batch_denoise_2d,
            denoise_h=batch_denoise_h,
            denoise_patch_size=batch_denoise_patch_size,
            denoise_search_distance=batch_denoise_search_distance,
            denoise_backend=batch_denoise_backend,
            denoise_mode=denoise_mode,
            denoised_zarr_path=denoised_zarr_path,
            calibration_samples=batch_calibration_samples,
        )

        # Build job list. Store real dataset indices in filenames so status,
        # merge, and generated Slurm scripts agree when --timepoints/--channels
        # select non-contiguous values.
        jobs = []
        t_width_base = max(t_indices) + 1
        c_width_base = max(c_indices) + 1
        for task_id in range(total_tasks):
            t_seq = task_id // (n_c * n_tiles)
            r = task_id % (n_c * n_tiles)
            c_seq = r // n_tiles
            k = r % n_tiles
            t_real = t_indices[t_seq]
            c_real = c_indices[c_seq]
            jobs.append(
                BatchJob(
                    task_id=task_id,
                    timepoint=t_real,
                    channel=c_real,
                    tile_index=k,
                    output_filename=output_filename(
                        t_real, c_real, k, t_width_base, c_width_base, n_tiles
                    ),
                    estimated_wall_seconds=est_seconds,
                    channel_coords=decode_flat_channel_index(
                        c_real, ome_info.channel_shape
                    ),
                )
            )
        manifest.jobs = jobs

        # 7. Capture environment + generate scripts
        env = capture_environment()
        preamble = generate_env_preamble(env)
        fit_script = generate_fit_sbatch(manifest, preamble)
        merge_script = generate_merge_sbatch(manifest, preamble)

        # Generate preemptible fit script if enabled
        preempt_fit_script = None
        if preempt_partition:
            preempt_fit_script = generate_fit_sbatch(
                manifest,
                preamble,
                partition_override=preempt_partition,
                max_concurrent_override=manifest.preemptible_max_concurrent,
                requeue=True,
                job_name="luxar-fit-preempt",
            )

        # Generate denoise scripts if needed
        calibrate_script = None
        denoise_script = None
        if batch_denoise:
            from luxar.gsplats.batch.slurm_gen import (
                generate_calibrate_sbatch,
                generate_denoise_sbatch,
            )

            if batch_denoise_h is None:
                calibrate_script = generate_calibrate_sbatch(manifest, preamble)
            if denoise_mode == "preprocess":
                denoise_script = generate_denoise_sbatch(manifest, preamble)

        # 8. Print plan (always)
        spatial_str = "x".join(str(s) for s in spatial)
        peak_gvs = peak.get("gvoxel_per_s", "?")
        peak_shape_str = "x".join(str(s) for s in peak.get("shape", []))

        aprint("")
        aprint("=" * 60)
        aprint("BATCH PLAN")
        aprint("=" * 60)
        aprint(f"  Input: {input_path.name} (T={n_t}, C={n_c}, spatial={spatial_str})")
        aprint(f"  GPU: {resolved_gpu} (peak: {peak_gvs} GV/s at {peak_shape_str})")
        if needs_tiling:
            aprint(
                f"  Tile: {tile_size}^{len(spatial)}"
                f" ({'auto' if auto_tile else 'manual'})"
                f", overlap={tile_overlap}, {n_tiles} tiles/volume"
            )
        else:
            aprint("  Tile: not needed (volume fits in GPU memory)")
        aprint(f"  Jobs: {n_t} x {n_c} x {n_tiles} = {total_tasks} fitting tasks")
        if tasks_per_job > 1:
            mode = "parallel" if parallel else "sequential"
            mps_note = ""
            if parallel:
                if is_slurm_mps_available():
                    mps_note = " [MPS available]"
                else:
                    mps_note = " [bash background processes]"
            aprint(
                f"  Packing: {tasks_per_job} tasks/job ({mode}) → {n_slurm_jobs} Slurm jobs{mps_note}"
            )
        else:
            aprint(f"  Slurm array: {total_tasks} jobs (1 task each)")
        if uses_backfill:
            sched_note = "backfill scheduler — short jobs get scheduled fastest"
            if no_job_limit:
                sched_note += ", no job count limit"
            aprint(f"  Scheduler: {sched_note}")
        aprint(
            f"  Est. time/task: ~{est_seconds / 60:.0f} min"
            f" (preset: {preset}, {n_iters} iters)"
        )
        if tasks_per_job > 1:
            if parallel:
                aprint(
                    f"  Est. time/job: ~{est_seconds_per_job / 60:.0f} min ({tasks_per_job} tasks in parallel)"
                )
            else:
                aprint(
                    f"  Est. time/job: ~{est_seconds_per_job / 60:.0f} min ({tasks_per_job} tasks × {est_seconds / 60:.0f} min)"
                )
        aprint(f"  Est. total GPU-hours: {total_gpu_hours:.0f} h")
        aprint(f"  Slurm --time: {slurm_time}")
        aprint(f"  Partition: {partition}, GPUs: {gpus}, CPUs: {cpus}, Mem: {mem}G")
        aprint(f"  Output: {output_dir}")
        aprint("")

        if not submit:
            aprint("Dry run -- pass --submit to actually submit.")
            raise typer.Exit(0)

        # 9. Submit
        out = output_dir.resolve()
        (out / "tiles").mkdir(parents=True, exist_ok=True)
        (out / "merged").mkdir(parents=True, exist_ok=True)
        (out / "logs").mkdir(parents=True, exist_ok=True)

        fit_path = out / "fit_array.sbatch"
        merge_path = out / "merge.sbatch"
        env_path = out / "env_snapshot.sh"

        fit_path.write_text(fit_script)
        merge_path.write_text(merge_script)
        env_path.write_text(preamble)
        if calibrate_script:
            (out / "calibrate.sbatch").write_text(calibrate_script)
        if denoise_script:
            (out / "denoise_array.sbatch").write_text(denoise_script)
        if preempt_fit_script:
            (out / "fit_array_preempt.sbatch").write_text(preempt_fit_script)
        save_manifest(manifest, out)

        def _parse_job_id(stdout: str) -> Optional[int]:
            for word in stdout.strip().split():
                if word.isdigit():
                    return int(word)
            return None

        # Submit calibration job (if needed)
        calibrate_job_id = None
        if calibrate_script:
            aprint("Submitting calibration job...")
            result = subprocess.run(
                ["sbatch", str(out / "calibrate.sbatch")],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                aprint(f"Error submitting calibration job: {result.stderr}")
                raise typer.Exit(1)
            calibrate_job_id = _parse_job_id(result.stdout)
            manifest.calibrate_job_id = calibrate_job_id
            aprint(f"  Calibration job: {calibrate_job_id}")

        # Submit denoise preprocessing array (if preprocess mode)
        denoise_job_id = None
        if denoise_script:
            aprint("Submitting denoise preprocessing array...")
            dep_cmd = ["sbatch"]
            if calibrate_job_id:
                dep_cmd.append(f"--dependency=afterok:{calibrate_job_id}")
            dep_cmd.append(str(out / "denoise_array.sbatch"))
            result = subprocess.run(dep_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                aprint(f"Error submitting denoise job: {result.stderr}")
                raise typer.Exit(1)
            denoise_job_id = _parse_job_id(result.stdout)
            manifest.denoise_job_id = denoise_job_id
            denoise_n_t = (
                len(manifest.timepoint_indices)
                if manifest.timepoint_indices
                else manifest.n_timepoints
            )
            denoise_n_c = (
                len(manifest.channel_indices)
                if manifest.channel_indices
                else manifest.n_channels
            )
            denoise_total = denoise_n_t * denoise_n_c
            aprint(f"  Denoise array job: {denoise_job_id} ({denoise_total} tasks)")

        # Submit fitting array (depends on denoise or calibrate)
        fit_dep_id = denoise_job_id or calibrate_job_id
        aprint("Submitting fitting array job...")
        fit_cmd = ["sbatch"]
        if fit_dep_id:
            fit_cmd.append(f"--dependency=afterok:{fit_dep_id}")
        fit_cmd.append(str(fit_path))
        result = subprocess.run(fit_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            aprint(f"Error submitting fit job: {result.stderr}")
            raise typer.Exit(1)

        fit_job_id = _parse_job_id(result.stdout)
        aprint(f"  Fitting array job: {fit_job_id} ({total_tasks} tasks)")

        # Submit preemptible fit array (if enabled)
        preemptible_job_id = None
        if preempt_fit_script:
            aprint("Submitting preemptible fitting array...")
            preempt_cmd = ["sbatch"]
            if fit_dep_id:
                preempt_cmd.append(f"--dependency=afterok:{fit_dep_id}")
            preempt_cmd.append(str(out / "fit_array_preempt.sbatch"))
            result = subprocess.run(preempt_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                preemptible_job_id = _parse_job_id(result.stdout)
                manifest.preemptible_job_id = preemptible_job_id
                aprint(
                    f"  Preemptible array job: {preemptible_job_id} "
                    f"({total_tasks} tasks on {preempt_partition}, requeue)"
                )
            else:
                aprint(
                    f"  Warning: preemptible submission failed: {result.stderr}\n"
                    "  Continuing with guaranteed partition only."
                )

        # Merge depends on ALL fit arrays
        merge_deps = [jid for jid in [fit_job_id, preemptible_job_id] if jid]
        merge_cmd = ["sbatch"]
        if merge_deps:
            dep_str = ":".join(str(jid) for jid in merge_deps)
            merge_cmd.append(f"--dependency=afterok:{dep_str}")
        merge_cmd.append(str(merge_path))

        result = subprocess.run(merge_cmd, capture_output=True, text=True)
        merge_job_id = None
        if result.returncode == 0:
            merge_job_id = _parse_job_id(result.stdout)
            dep_info = " + ".join(str(j) for j in merge_deps)
            aprint(f"  Merge job: {merge_job_id} (depends on {dep_info})")
        else:
            aprint(f"  Warning: merge job submission failed: {result.stderr}")

        manifest.array_job_id = fit_job_id
        manifest.merge_job_id = merge_job_id
        save_manifest(manifest, out)

        aprint(f"\nManifest: {out / 'manifest.json'}")
        aprint(f"Check status: luxar gsplat batch status {out}")

    except typer.Exit:
        raise
    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app_batch.command("status")
def batch_status_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Check status of a batch fitting job.

    Reads the manifest, checks for output files, and queries sacct/squeue
    for job states.

    Examples:
        luxar gsplat batch status output_dir/
    """
    try:
        from luxar.gsplats.batch.manifest import load_manifest
        from luxar.gsplats.batch.status import (
            check_batch_status,
            format_status_report,
        )

        manifest = load_manifest(output_dir)
        status = check_batch_status(output_dir)
        aprint(format_status_report(status, manifest, verbose=verbose))

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1)


@app_batch.command("validate")
def batch_validate_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    fix: bool = typer.Option(
        False, "--fix", help="Delete corrupt/incomplete tiles so they get re-fitted"
    ),
) -> None:
    """Validate integrity of all tiles in a batch output directory.

    Checks each tile for completeness (metadata, arrays, shapes).
    Reports OK, MISSING, CORRUPT, and STALE_TMP counts.

    Use --fix to delete corrupt tiles and leftover .tmp directories,
    so they get re-fitted on the next submit.

    Examples:
        luxar gsplat batch validate output_dir/

        luxar gsplat batch validate output_dir/ --fix
    """
    try:
        from luxar.gsplats.batch.manifest import load_manifest

        manifest = load_manifest(output_dir)
        tiles_dir = output_dir / "tiles"

        if not tiles_dir.exists():
            aprint("No tiles directory found.")
            raise typer.Exit(1)

        # Build expected tile list from manifest
        expected_tiles = [job.output_filename for job in manifest.jobs]
        aprint(f"Checking {len(expected_tiles)} expected tiles...")

        ok = 0
        missing = 0
        corrupt = 0
        unmigrated = 0
        stale_tmp = 0
        corrupt_reasons: list[str] = []
        unmigrated_reasons: list[str] = []

        for tile_name in expected_tiles:
            tile_path = tiles_dir / tile_name
            tmp_path = tiles_dir / f"{tile_name}.tmp"

            # Check for stale .tmp
            if tmp_path.is_dir():
                stale_tmp += 1
                if fix:
                    shutil.rmtree(tmp_path)
                    aprint(f"  Deleted: {tile_name}.tmp")

            if not tile_path.is_dir():
                missing += 1
                continue

            # Validate tile integrity
            reason = _validate_tile(tile_path)
            if reason == "ok":
                ok += 1
            elif reason.startswith("unsupported_format_version"):
                # Recoverable, NOT corrupt: an unmigrated legacy tile. Never
                # delete it under --fix — it converts via `gsplat migrate-format`.
                unmigrated += 1
                unmigrated_reasons.append(f"  {tile_name}: {reason}")
            else:
                corrupt += 1
                corrupt_reasons.append(f"  {tile_name}: {reason}")
                if fix:
                    shutil.rmtree(tile_path)
                    aprint(f"  Deleted corrupt: {tile_name} ({reason})")

        # Summary
        aprint("")
        aprint(f"  OK:         {ok}")
        aprint(f"  MISSING:    {missing}")
        aprint(f"  CORRUPT:    {corrupt}")
        aprint(f"  UNMIGRATED: {unmigrated}")
        aprint(f"  STALE_TMP:  {stale_tmp}")

        if corrupt_reasons and not fix:
            aprint("")
            aprint("Corrupt tiles:")
            for r in corrupt_reasons:
                aprint(r)
            aprint("")
            aprint("Run with --fix to delete corrupt tiles.")

        if unmigrated_reasons:
            aprint("")
            aprint("Unmigrated (legacy-format) tiles — NOT deleted:")
            for r in unmigrated_reasons:
                aprint(r)
            aprint("")
            aprint("Convert each with `luxar gsplat migrate-format <tile> <out>`.")

        if fix and (corrupt > 0 or stale_tmp > 0):
            aprint(f"\nFixed: deleted {corrupt} corrupt + {stale_tmp} stale .tmp")
            aprint("Resubmit to re-fit deleted tiles.")

    except typer.Exit:
        raise
    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1) from e


def _validate_leaf_arrays(node_dir: Path, label: str) -> str:
    """Check a v3.0 gsplats leaf's required array sub-dirs (no decode)."""
    for arr_name in ("centers", "amplitudes", "cholesky_factors"):
        arr_dir = node_dir / arr_name
        if not arr_dir.is_dir():
            return f"missing_{arr_name}@{label}"
        if not (arr_dir / ".zarray").exists():
            return f"no_zarray_{arr_name}@{label}"
    return "ok"


def _validate_node_dir(node_dir: Path, label: str) -> str:
    """Structurally validate a v3.0 node subtree on disk (no array decode)."""
    import json

    zattrs_path = node_dir / ".zattrs"
    if not zattrs_path.exists():
        # Every node (root, child_<i>, part_<i>) must carry its .zattrs; a
        # metadata-stripped node is corrupt, not a bare single-set leaf.
        return f"no_zattrs@{label}"
    try:
        attrs = json.loads(zattrs_path.read_text())
    except (json.JSONDecodeError, OSError):
        return f"corrupt_zattrs@{label}"

    kind = attrs.get("kind")
    if kind in ("lod", "partition"):
        prefix = "child_" if kind == "lod" else "part_"
        children = sorted(
            d for d in node_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)
        )
        if not children:
            return f"{kind}_no_children@{label}"
        for child in children:
            reason = _validate_node_dir(child, f"{label}/{child.name}")
            if reason != "ok":
                return reason
        return "ok"

    # Leaf: a single splat set, or an additive ladder (additive_<i>/ subgroups).
    n_additive = int(attrs.get("n_additive_sublods", 1))
    if n_additive > 1:
        for i in range(n_additive):
            reason = _validate_leaf_arrays(
                node_dir / f"additive_{i}", f"{label}/additive_{i}"
            )
            if reason != "ok":
                return reason
        return "ok"
    return _validate_leaf_arrays(node_dir, label)


def _validate_tile(tile_path: Path) -> str:
    """Validate a single v3.0 tile's integrity. Returns 'ok' or a reason string.

    Walks the node-tree structure (leaf / kind=lod / kind=partition) checking for
    the consolidated metadata, the format header, and the presence of every
    required array — without decoding any data. A non-v3.0 tile is reported (so
    ``batch validate --fix`` never silently deletes an unmigrated tile).
    """
    import json

    # .zmetadata is written last by consolidate_metadata — best completeness signal.
    if not (tile_path / ".zmetadata").exists():
        return "no_zmetadata (save incomplete)"

    zattrs_path = tile_path / ".zattrs"
    if not zattrs_path.exists():
        return "no_zattrs"
    try:
        attrs = json.loads(zattrs_path.read_text())
    except (json.JSONDecodeError, OSError):
        return "corrupt_zattrs"

    if attrs.get("format_type") != "gsplats_zarr":
        return f"bad_format_type: {attrs.get('format_type')}"

    version = attrs.get("format_version")
    if version != "3.0":
        # Not corrupt — just unmigrated. Surface it instead of classifying it as
        # corrupt (which would let --fix delete a recoverable tile).
        return f"unsupported_format_version: {version} (run gsplat migrate-format)"

    return _validate_node_dir(tile_path, ".")


@app_batch.command("cancel")
def batch_cancel_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
) -> None:
    """Cancel all Slurm jobs for a batch fitting run.

    Reads the manifest to find job IDs (calibrate, denoise, fit array,
    merge) and cancels them via scancel.

    Examples:
        luxar gsplat batch cancel output_dir/
    """
    import subprocess

    try:
        from luxar.gsplats.batch.manifest import load_manifest

        manifest = load_manifest(output_dir)

        job_ids = []
        for attr in (
            "calibrate_job_id",
            "denoise_job_id",
            "array_job_id",
            "merge_job_id",
        ):
            jid = getattr(manifest, attr, None)
            if jid is not None:
                job_ids.append(str(jid))

        if not job_ids:
            aprint("No job IDs found in manifest — nothing to cancel.")
            raise typer.Exit(0)

        aprint(f"Cancelling {len(job_ids)} job(s): {', '.join(job_ids)}")
        result = subprocess.run(["scancel"] + job_ids, capture_output=True, text=True)
        if result.returncode == 0:
            aprint("All jobs cancelled.")
        else:
            # scancel may warn about already-completed jobs — that's fine
            aprint(f"scancel output: {result.stderr.strip()}")
            aprint("Cancel command sent (some jobs may have already completed).")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1)


@app_batch.command("merge")
def batch_merge_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    channel_colors: Optional[str] = typer.Option(
        None, "--channel-colors", help="Hex colors for channel merge"
    ),
    force: bool = typer.Option(False, "--force", help="Re-merge even if outputs exist"),
) -> None:
    """Run the merge step for a completed batch job.

    Normally runs as a dependent Slurm job, but this command allows
    running it manually or re-running if the merge job failed.

    Examples:
        luxar gsplat batch merge output_dir/

        luxar gsplat batch merge output_dir/ --channel-colors "#ff0080,#00ff00"
    """
    try:
        from luxar.cli.gsplat_config import parse_hex_color
        from luxar.gsplats.batch.manifest import load_manifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results

        manifest = load_manifest(output_dir)

        colors = None
        color_source = channel_colors or (
            ",".join(manifest.channel_colors) if manifest.channel_colors else None
        )
        if color_source:
            colors = [parse_hex_color(c.strip()) for c in color_source.split(",")]

        with asection(f"Merging batch results: {output_dir}"):
            final_path = merge_batch_results(
                manifest=manifest,
                output_dir=output_dir,
                channel_colors=colors,
                force=force,
            )
            aprint(f"\nFinal output: {final_path}")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


# ── Hidden batch worker commands for denoise pipeline ────────────


@app_batch.command("denoise-calibrate", hidden=True)
def batch_denoise_calibrate_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
) -> None:
    """[Internal] Run NLM calibration for batch denoise pipeline.

    Reads manifest, calibrates h per channel, writes results back.
    Called by the calibration Slurm job.
    """
    try:
        import json

        from luxar.gsplats.batch.manifest import load_manifest, save_manifest
        from luxar.gsplats.preprocessing.denoise_pipeline import calibrate_all_channels

        manifest = load_manifest(output_dir)

        if not manifest.denoise:
            aprint("Error: denoise not enabled in manifest")
            raise typer.Exit(1)

        with asection("NLM Calibration"):
            h_values = calibrate_all_channels(
                input_path=Path(manifest.input_path),
                n_timepoints=manifest.n_timepoints,
                n_channels=manifest.n_channels,
                channel_indices=(
                    manifest.channel_indices
                    if manifest.channel_indices
                    else list(range(manifest.n_channels))
                ),
                timepoint_indices=manifest.timepoint_indices,
                array_key=manifest.array_key,
                calibration_samples=manifest.calibration_samples,
                patch_size=manifest.denoise_patch_size,
                search_distance=manifest.denoise_search_distance,
                backend=manifest.denoise_backend,
                h_override=manifest.denoise_h,
            )

            # Write h_values to manifest (string keys for JSON)
            manifest.denoise_h_values = {str(k): v for k, v in h_values.items()}
            save_manifest(manifest, output_dir)

            # Also write standalone JSON for easy reading by other jobs
            h_path = output_dir / "denoise_h_values.json"
            h_path.write_text(json.dumps(h_values, indent=2))

            aprint(f"Calibrated h values: {h_values}")
            aprint(f"Saved to {h_path}")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1) from e


@app_batch.command("denoise-preprocess", hidden=True)
def batch_denoise_preprocess_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    task_id: int = typer.Argument(..., help="Array task ID (encodes T*n_c + C)"),
) -> None:
    """[Internal] Denoise one (T,C) volume for batch preprocess pipeline.

    Called by the denoise Slurm array job, one task per (timepoint, channel).
    """
    try:
        import json

        import zarr

        from luxar.cli.gsplat_config import load_volume
        from luxar.gsplats.batch.manifest import load_manifest
        from luxar.gsplats.preprocessing.denoise_pipeline import denoise_volume_array

        manifest = load_manifest(output_dir)

        # Read calibrated h values
        h_path = output_dir / "denoise_h_values.json"
        if not h_path.exists():
            aprint("Error: denoise_h_values.json not found. Run calibration first.")
            raise typer.Exit(1)
        h_values = json.loads(h_path.read_text())

        # Decode task_id -> (t_idx, c_idx) within selected indices
        n_c = manifest.n_channels
        t_idx = task_id // n_c
        c_idx = task_id % n_c

        # Map to real dataset indices
        t_indices = manifest.timepoint_indices or list(range(manifest.n_timepoints))
        c_indices = manifest.channel_indices or list(range(manifest.n_channels))
        t_real = t_indices[t_idx]
        c_real = c_indices[c_idx]

        h = h_values.get(str(c_real), 0.04)

        with asection(f"Denoising T={t_real} C={c_real} (h={h:.4f})"):
            # Load volume
            volume = load_volume(
                Path(manifest.input_path),
                channel=c_real if manifest.n_channels > 1 else None,
                timepoint=t_real if manifest.n_timepoints > 1 else None,
                array_key=manifest.array_key,
            )
            aprint(f"Loaded: shape={volume.shape}")

            # Denoise
            denoised = denoise_volume_array(
                volume,
                h=h,
                patch_size=manifest.denoise_patch_size,
                search_distance=manifest.denoise_search_distance,
                backend=manifest.denoise_backend,
                use_2d=manifest.denoise_2d,
            )

            # Write to denoised.zarr
            zarr_path = output_dir / "denoised.zarr"
            store = zarr.open(str(zarr_path), mode="a")

            spatial = denoised.shape
            full_shape = (len(t_indices), len(c_indices), *spatial)
            chunks = (1, 1, *[min(s, 128) for s in spatial])

            if "data" not in store:
                store.create_dataset(
                    "data",
                    shape=full_shape,
                    chunks=chunks,
                    dtype=np.float32,
                )
            store["data"][t_idx, c_idx] = denoised
            aprint(f"Written to denoised.zarr[{t_idx}, {c_idx}]")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1) from e


# ═══════════════════════════════════════════════════════════════════════
# lod — Build LOD ladders from a fitted gsplat dataset
# ═══════════════════════════════════════════════════════════════════════

app_lod = typer.Typer(help="Build LOD ladders from a fitted gsplat dataset")
app_gsplat.add_typer(app_lod, name="lod")


def _parse_lod_breakpoints(spec: str) -> "str | list[int] | list[float]":
    """Parse the ``--breakpoints`` CLI string into the form expected by
    :func:`luxar.gsplats.lod.make_additive_lod`.

    Accepted forms:
      - ``equal-count``  → literal ``'equal-count'``.
      - ``counts:5,10,15,20`` → ``[5, 10, 15, 20]`` (cumulative splat counts).
      - ``energy:0.5,0.9,0.99,1.0`` → ``[0.5, 0.9, 0.99, 1.0]``
        (cumulative energy fractions in (0, 1]).
    """
    s = spec.strip()
    if s == "equal-count":
        return "equal-count"
    if s.startswith("counts:"):
        body = s[len("counts:") :]
        try:
            values_int = [int(p.strip()) for p in body.split(",") if p.strip()]
        except ValueError as e:
            raise typer.BadParameter(
                f"counts breakpoints must be ints; got {body!r}"
            ) from e
        if not values_int:
            raise typer.BadParameter("counts breakpoints list is empty")
        return values_int
    if s.startswith("energy:"):
        body = s[len("energy:") :]
        try:
            values_flt = [float(p.strip()) for p in body.split(",") if p.strip()]
        except ValueError as e:
            raise typer.BadParameter(
                f"energy breakpoints must be floats; got {body!r}"
            ) from e
        if not values_flt:
            raise typer.BadParameter("energy breakpoints list is empty")
        return values_flt
    raise typer.BadParameter(
        f"breakpoints must be 'equal-count', 'counts:...', or 'energy:...'; "
        f"got {spec!r}"
    )


@app_lod.command("additive")
def lod_additive(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr (single- or multi-LOD)"
    ),
    output_path: Path = typer.Argument(..., help="Output .gsplats.zarr (multi-LOD)"),
    n_lods: int = typer.Option(
        4,
        "--n-lods",
        help="Number of LOD levels when --breakpoints=equal-count",
        min=1,
    ),
    method: str = typer.Option(
        "greedy",
        "--method",
        "-m",
        help=(
            "Ordering method: greedy (default), self_energy, mass, "
            "amplitude, spectral, random. greedy is provably (1-1/e)-optimal "
            "at every prefix; self_energy is the recommended O(N log N) "
            "fallback for very large N."
        ),
    ),
    breakpoints: str = typer.Option(
        "equal-count",
        "--breakpoints",
        "-b",
        help=(
            "How to slice the ordered set into LODs. "
            "'equal-count' uses --n-lods bins of equal size. "
            "'counts:5,10,15,20' uses explicit cumulative splat counts. "
            "'energy:0.5,0.9,1.0' uses cumulative energy fractions in (0, 1]."
        ),
    ),
    truncation_sigmas: float = typer.Option(
        3.0,
        "--truncation-sigmas",
        help="Mahalanobis cutoff (in sigmas) for sparse-Gram pruning. "
        "Larger keeps more pairs (slower but tighter); smaller is faster.",
    ),
    max_n_dense: int = typer.Option(
        2000,
        "--max-n-dense",
        help="At N <= this, greedy uses a dense Gram + scan-greedy. "
        "Above it, sparse Gram + lazy greedy.",
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed for --method=random."
    ),
    substitutive_level: Optional[int] = typer.Option(
        None,
        "--substitutive-level",
        help=(
            "When the input is a multi-substitutive pyramid (output of "
            "`lod substitutive` or `lod pyramid`), build the additive ladder "
            "on this substitutive level only. Other substitutive levels are "
            "carried through unchanged. Defaults to the default substitutive "
            "level (= 0, the finest)."
        ),
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite output if it exists."
    ),
    encoding_mode: Literal["auto", "precision", "memory"] = typer.Option(
        "auto",
        "--encoding",
        "-e",
        help="Encoding mode for output",
    ),
    compress: Optional[str] = typer.Option(
        None,
        "--compress",
        help="Optional output compression: 'zip' or 'tar.gz' for an archive; "
        "omit for a plain .gsplats.zarr directory.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress per-LOD progress lines and the trailing summary "
        "(top-level asection headers still appear).",
    ),
) -> None:
    """Build an additive LOD ladder from a fitted gsplat dataset.

    The original splats are reordered by the chosen method, then sliced into
    ``--n-lods`` (or as many levels as ``--breakpoints`` implies) so that
    ``additive_prefix(k)`` is the best L^2 approximation of the full scene
    at that splat budget. On a multi-substitutive input, the new ladder
    replaces the chosen substitutive level (default: the finest level);
    other substitutive levels are passed through unchanged.

    Input must be a pre-fitted .gsplats.zarr (output of ``luxar gsplat fit``).
    The canonical end-to-end pipeline is: ``cal`` → ``fit --seeds K*`` →
    ``lod additive``.

    \b
    Examples:
        luxar gsplat lod additive in.gsplats.zarr out.gsplats.zarr
        luxar gsplat lod additive in.gsplats.zarr out.gsplats.zarr --n-lods 6
        luxar gsplat lod additive in.gsplats.zarr out.gsplats.zarr \\
            --breakpoints energy:0.5,0.9,0.99,1.0
        luxar gsplat lod additive in.gsplats.zarr out.gsplats.zarr \\
            --breakpoints counts:1000,5000,25000
        luxar gsplat lod additive in.gsplats.zarr out.gsplats.zarr \\
            --method self_energy
    """
    try:
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.lod import make_additive_lod

        method_norm = method.strip().replace("-", "_")
        valid_methods = {
            "greedy",
            "self_energy",
            "mass",
            "amplitude",
            "spectral",
            "random",
        }
        if method_norm not in valid_methods:
            raise typer.BadParameter(
                f"--method must be one of {sorted(valid_methods)}, got {method!r}"
            )

        bp = _parse_lod_breakpoints(breakpoints)
        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        if compress not in (None, "zip", "tar.gz"):
            raise typer.BadParameter(
                f"--compress must be 'zip' or 'tar.gz'; got {compress!r}"
            )

        if output_path.exists() and not overwrite:
            raise typer.BadParameter(
                f"Output {output_path} exists; pass --overwrite to replace it."
            )

        with asection(f"LOD additive: {input_path.name}"):
            with asection("Loading dataset"):
                data = GSplatData.load(input_path, include_stats=True)
                aprint(f"Loaded {data.n_splats:,} splats ({data.ndim}D)")
                if data.n_additive_sublods > 1:
                    aprint(
                        f"Input is multi-LOD ({data.n_additive_sublods} LODs); "
                        "operating on the flattened concatenation."
                    )

            with asection(f"Ordering ({method_norm})"):
                t0 = time.time()
                target_sub = (
                    data.default_substitutive
                    if substitutive_level is None
                    else int(substitutive_level)
                )
                if not (0 <= target_sub < data.n_substitutive):
                    raise typer.BadParameter(
                        f"--substitutive-level {target_sub} is out of bounds "
                        f"for input with n_substitutive={data.n_substitutive}"
                    )
                ladder = make_additive_lod(
                    data,
                    n_lods=n_lods,
                    method=method_norm,  # type: ignore[arg-type]
                    breakpoints=bp,  # type: ignore[arg-type]
                    truncation_sigmas=truncation_sigmas,
                    max_n_dense=max_n_dense,
                    seed=seed,
                    substitutive_level=target_sub,
                )
                # The ladder we just produced lives on `target_sub`; report counts
                # from that level (other substitutive levels were carried over).
                touched_level = ladder.substitutive_levels[target_sub]
                aprint(
                    f"Built {touched_level.n_additive_lods}-level ladder on "
                    f"substitutive level {target_sub} in {time.time() - t0:.2f}s"
                )
                cuts = ladder.stats.get("lod_cutpoints", [])
                kind = ladder.stats.get("lod_breakpoints_kind", "?")
                if not quiet:
                    aprint(f"Cutpoints ({kind}): {cuts}")
                    for level, sublod in enumerate(touched_level.additive_sublods):
                        aprint(f"  LOD {level}: {sublod.n_splats:,} splats")

            with asection("Saving"):
                if output_path.exists() and overwrite:
                    if output_path.is_dir():
                        shutil.rmtree(output_path)
                    else:
                        output_path.unlink()
                ladder.save(
                    output_path,
                    encoding_mode=encoding_mode_obj,
                    compress=compress,  # type: ignore[arg-type]
                )
                if not quiet:
                    aprint(f"Saved to {output_path}")

    except typer.Exit:
        raise
    except typer.BadParameter:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


_VALID_SUBSTITUTIVE_METHODS = (
    "auto",
    "kmeans",
    "kmeans_lloyd",
    "greedy",
    "greedy_lloyd",
)


@app_lod.command("substitutive")
def lod_substitutive(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr (single- or multi-LOD)"
    ),
    output_path: Path = typer.Argument(
        ...,
        help=(
            "Output .gsplats.zarr (v3.0) holding the substitutive hierarchy as "
            "a ``kind=lod`` group of ``child_<i>/`` leaves (coarsest→finest). "
            "Loadable with ``luxar gsplat info``."
        ),
    ),
    compression_factor: int = typer.Option(
        4,
        "--K",
        "-K",
        help="Per-level compression factor (each level shrinks by this).",
        min=2,
    ),
    levels: int = typer.Option(
        3,
        "--L",
        "-L",
        help="Number of coarser levels to produce.",
        min=1,
    ),
    method: str = typer.Option(
        "auto",
        "--method",
        "-m",
        help=(
            "Partition algorithm. auto (default): greedy for levels <=5000 "
            "splats (best quality, fast there), kmeans_lloyd above. Also: "
            "kmeans_lloyd (large-N workhorse), kmeans (no Lloyd), greedy "
            "(quality-leading at small N), greedy_lloyd."
        ),
    ),
    lloyd_iterations: int = typer.Option(
        5,
        "--lloyd-iters",
        help="Max Lloyd refinement passes per level (only for *_lloyd methods).",
        min=0,
    ),
    candidate_bins_k: int = typer.Option(
        12,
        "--candidate-bins-k",
        help="Top-k spatial-hash candidates per splat during Lloyd.",
        min=1,
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="PyTorch device: auto | cpu | cuda | mps.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help=(
            "Accepted for API stability; has no effect. The Morton warm start "
            "and synchronous Lloyd pass are deterministic."
        ),
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite output directory if it exists."
    ),
    encoding_mode: Literal["auto", "precision", "memory"] = typer.Option(
        "auto",
        "--encoding",
        "-e",
        help="Encoding mode for output .gsplats.zarr levels.",
    ),
    compress: Optional[str] = typer.Option(
        None,
        "--compress",
        help=(
            "Optional output compression: 'zip' or 'tar.gz'. "
            "Omit for plain .gsplats.zarr directories."
        ),
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress per-level progress lines and the trailing summary "
        "(top-level asection headers still appear).",
    ),
) -> None:
    """Build a substitutive LOD hierarchy from a fitted gsplat dataset.

    Each coarser substitutive level contains synthesised representative
    splats that *replace* the previous level (compression factor K per
    step). The full hierarchy is written to a single v3.0 .gsplats.zarr
    file (a ``kind=lod`` group of ``child_<i>/`` leaves), loadable
    independently and renderable level-by-level by the viewer.

    Input must be a pre-fitted .gsplats.zarr (output of ``luxar gsplat fit``).
    The canonical end-to-end pipeline is: ``cal`` → ``fit --seeds K*`` →
    ``lod substitutive``. For a 2-D ``[N, M]`` pyramid combining
    substitutive + additive in one shot, use ``luxar gsplat lod pyramid``.

    \b
    Examples:
        luxar gsplat lod substitutive in.gsplats.zarr out.gsplats.zarr
        luxar gsplat lod substitutive in.gsplats.zarr out.gsplats.zarr --K 4 --L 3
        luxar gsplat lod substitutive in.gsplats.zarr out.gsplats.zarr \\
            --method kmeans-lloyd --lloyd-iters 5
        luxar gsplat lod substitutive in.gsplats.zarr out.gsplats.zarr \\
            --method greedy_lloyd --K 2
    """
    try:
        import shutil
        import time

        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.lod import make_substitutive_lod

        method_norm = method.strip().replace("-", "_")
        if method_norm not in _VALID_SUBSTITUTIVE_METHODS:
            raise typer.BadParameter(
                f"--method must be one of {sorted(_VALID_SUBSTITUTIVE_METHODS)}, "
                f"got {method!r}"
            )

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)
        if compress not in (None, "zip", "tar.gz"):
            raise typer.BadParameter(
                f"--compress must be 'zip' or 'tar.gz'; got {compress!r}"
            )

        if output_path.exists() and not overwrite:
            raise typer.BadParameter(
                f"Output {output_path} exists; pass --overwrite to replace it."
            )

        with asection(f"LOD substitutive: {input_path.name}"):
            with asection("Loading dataset"):
                data = GSplatData.load(input_path, include_stats=True)
                aprint(f"Loaded {data.n_splats:,} splats ({data.ndim}D)")
                if data.n_additive_sublods > 1:
                    aprint(
                        f"Input is multi-LOD ({data.n_additive_sublods} LODs); "
                        "operating on the flattened concatenation."
                    )

            with asection(
                f"Reducing ({method_norm}, K={compression_factor}, L={levels})"
            ):
                t0 = time.time()
                pyramid = make_substitutive_lod(
                    data,
                    compression_factor=compression_factor,
                    levels=levels,
                    method=method_norm,  # type: ignore[arg-type]
                    lloyd_iterations=lloyd_iterations,
                    candidate_bins_k=candidate_bins_k,
                    device=device,
                    seed=seed,
                    verbose=False,
                )
                aprint(
                    f"Built {pyramid.n_substitutive}-level pyramid "
                    f"in {time.time() - t0:.2f}s"
                )
                if not quiet:
                    for s, lev in enumerate(pyramid.substitutive_levels):
                        aprint(f"  level {s}: {lev.n_splats_total:,} splats")

            with asection("Saving"):
                if output_path.exists() and overwrite:
                    if output_path.is_dir():
                        shutil.rmtree(output_path)
                    else:
                        output_path.unlink()
                pyramid.save(
                    output_path,
                    encoding_mode=encoding_mode_obj,
                    compress=compress,  # type: ignore[arg-type]
                )
                if not quiet:
                    aprint(f"Wrote {output_path}")

    except typer.Exit:
        raise
    except typer.BadParameter:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


# ═══════════════════════════════════════════════════════════════════════
# lod pyramid — Build the 2-D LOD matrix (substitutive × additive) in one shot
# ═══════════════════════════════════════════════════════════════════════


def _parse_substitutive_spec(spec: str) -> tuple[int, int]:
    """Parse ``--substitutive K=4,L=3`` into ``(K, L)``."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    kv = {}
    for p in parts:
        if "=" not in p:
            raise typer.BadParameter(
                f"--substitutive must be 'K=<int>,L=<int>'; got {spec!r}"
            )
        k, _, v = p.partition("=")
        kv[k.strip().upper()] = v.strip()
    if "K" not in kv or "L" not in kv:
        raise typer.BadParameter(
            f"--substitutive must specify both K and L; got {spec!r}"
        )
    try:
        K = int(kv["K"])
        L = int(kv["L"])
    except ValueError as exc:
        raise typer.BadParameter(
            f"--substitutive K and L must be integers; got {spec!r}"
        ) from exc
    if K < 2 or L < 1:
        raise typer.BadParameter(
            f"--substitutive requires K>=2 and L>=1; got K={K}, L={L}"
        )
    return K, L


@app_lod.command("pyramid")
def lod_pyramid(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr (fitted)."
    ),
    output_path: Path = typer.Argument(
        ...,
        help=(
            "Output .gsplats.zarr (v3.0) carrying the full 2-D pyramid: a "
            "``kind=lod`` group of ``child_<i>/`` leaves, each with an "
            "``additive_<a>/`` ladder."
        ),
    ),
    substitutive: str = typer.Option(
        "K=4,L=3",
        "--substitutive",
        help=(
            "Substitutive axis spec as ``K=<int>,L=<int>`` where K is the "
            "per-level compression factor and L is the number of coarser "
            "substitutive levels (so n_substitutive = L+1)."
        ),
    ),
    additive: int = typer.Option(
        4,
        "--additive",
        help="Number of additive sub-LODs per substitutive level.",
        min=1,
    ),
    substitutive_method: str = typer.Option(
        "auto",
        "--substitutive-method",
        help=(
            "Substitutive partition algorithm "
            "(auto | kmeans_lloyd | kmeans | greedy | greedy_lloyd). "
            "auto picks greedy for small levels, kmeans_lloyd for large."
        ),
    ),
    additive_method: str = typer.Option(
        "greedy",
        "--additive-method",
        help=(
            "Additive ordering algorithm "
            "(greedy | self_energy | mass | amplitude | spectral | random)."
        ),
    ),
    breakpoints: str = typer.Option(
        "equal-count",
        "--breakpoints",
        "-b",
        help=(
            "Additive breakpoints: 'equal-count', 'counts:N1,N2,...', or "
            "'energy:f1,f2,...'. Applied to every substitutive level."
        ),
    ),
    lloyd_iterations: int = typer.Option(
        5,
        "--lloyd-iters",
        help="Max Lloyd refinement passes per substitutive level.",
        min=0,
    ),
    candidate_bins_k: int = typer.Option(
        12,
        "--candidate-bins-k",
        help="Top-k spatial-hash candidates per splat during Lloyd.",
        min=1,
    ),
    truncation_sigmas: float = typer.Option(
        3.0,
        "--truncation-sigmas",
        help="Mahalanobis cutoff for additive sparse-Gram pruning.",
    ),
    max_n_dense: int = typer.Option(
        2000,
        "--max-n-dense",
        help="Additive greedy switches to dense Gram at N <= this.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="PyTorch device for substitutive reduction: auto | cpu | cuda | mps.",
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Shared RNG seed (per-axis offsets are added internally)."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite output if it exists."
    ),
    encoding_mode: Literal["auto", "precision", "memory"] = typer.Option(
        "auto",
        "--encoding",
        "-e",
        help="Encoding mode for output.",
    ),
    compress: Optional[str] = typer.Option(
        None,
        "--compress",
        help="Optional output compression: 'zip' or 'tar.gz'.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress per-level enumeration; top-level asection headers remain.",
    ),
) -> None:
    """Build the full 2-D LOD pyramid (substitutive × additive) in one shot.

    Equivalent to running ``lod substitutive`` then ``lod additive
    --substitutive-level`` for every substitutive level, but written as
    a single composite invocation that produces one v3.0 .gsplats.zarr
    file with shape ``[L+1, additive]``.

    \b
    Examples:
        luxar gsplat lod pyramid in.gsplats.zarr out.gsplats.zarr \\
            --substitutive K=4,L=3 --additive 4
        luxar gsplat lod pyramid in.gsplats.zarr out.gsplats.zarr \\
            --substitutive K=2,L=2 --additive 3 \\
            --substitutive-method kmeans_lloyd --additive-method greedy
    """
    try:
        import shutil
        import time

        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.lod import make_lod_pyramid

        sub_norm = substitutive_method.strip().replace("-", "_")
        if sub_norm not in _VALID_SUBSTITUTIVE_METHODS:
            raise typer.BadParameter(
                f"--substitutive-method must be one of "
                f"{sorted(_VALID_SUBSTITUTIVE_METHODS)}, got {substitutive_method!r}"
            )
        add_norm = additive_method.strip().replace("-", "_")
        valid_additive = {
            "greedy",
            "self_energy",
            "mass",
            "amplitude",
            "spectral",
            "random",
        }
        if add_norm not in valid_additive:
            raise typer.BadParameter(
                f"--additive-method must be one of {sorted(valid_additive)}, "
                f"got {additive_method!r}"
            )
        K, L = _parse_substitutive_spec(substitutive)
        bp = _parse_lod_breakpoints(breakpoints)
        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)
        if compress not in (None, "zip", "tar.gz"):
            raise typer.BadParameter(
                f"--compress must be 'zip' or 'tar.gz'; got {compress!r}"
            )
        if output_path.exists() and not overwrite:
            raise typer.BadParameter(
                f"Output {output_path} exists; pass --overwrite to replace it."
            )

        with asection(f"LOD pyramid: {input_path.name}"):
            with asection("Loading dataset"):
                data = GSplatData.load(input_path, include_stats=True)
                aprint(f"Loaded {data.n_splats:,} splats ({data.ndim}D)")

            with asection(
                f"Building pyramid (K={K}, L={L}, additive={additive}, "
                f"sub={sub_norm}, add={add_norm})"
            ):
                t0 = time.time()
                pyramid = make_lod_pyramid(
                    data,
                    compression_factor=K,
                    levels=L,
                    substitutive_method=sub_norm,  # type: ignore[arg-type]
                    lloyd_iterations=lloyd_iterations,
                    candidate_bins_k=candidate_bins_k,
                    device=device,
                    n_additive_lods=additive,
                    additive_method=add_norm,  # type: ignore[arg-type]
                    breakpoints=bp,  # type: ignore[arg-type]
                    truncation_sigmas=truncation_sigmas,
                    max_n_dense=max_n_dense,
                    seed=seed,
                    verbose=False,
                )
                aprint(
                    f"Built {pyramid.n_substitutive}-level pyramid in "
                    f"{time.time() - t0:.2f}s"
                )
                if not quiet:
                    for s, lev in enumerate(pyramid.substitutive_levels):
                        aprint(
                            f"  substitutive {s}: {lev.n_splats_total:,} splats, "
                            f"{lev.n_additive_lods} additive sub-LOD(s)"
                        )

            with asection("Saving"):
                if output_path.exists() and overwrite:
                    if output_path.is_dir():
                        shutil.rmtree(output_path)
                    else:
                        output_path.unlink()
                pyramid.save(
                    output_path,
                    encoding_mode=encoding_mode_obj,
                    compress=compress,  # type: ignore[arg-type]
                )
                if not quiet:
                    aprint(f"Wrote {output_path}")

    except typer.Exit:
        raise
    except typer.BadParameter:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e
