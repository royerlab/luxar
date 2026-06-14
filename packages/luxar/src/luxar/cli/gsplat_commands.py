"""CLI commands for Gaussian splat operations."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import typer
from arbol import aprint, asection

from .gsplat_ops.encoding import _resolve_encoding_mode
from .utils import _DEFAULT_CORS_ORIGIN, format_memory_size

if TYPE_CHECKING:
    import numpy as np

app_gsplat = typer.Typer(help="Gaussian splat tools")

# `_resolve_encoding_mode` is re-exported (imported above) for any callers that
# referenced it from this module before the gsplat_ops/ split.
__all__ = ["app_gsplat", "_resolve_encoding_mode"]


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
            try:
                data = GSplatData.load(path, include_stats=True)
            except ValueError:
                # GSplatData.load raises for two distinct reasons: (a) a valid
                # v3.0 partition/nested tree that has no flat GSplatData form, or
                # (b) a legacy/invalid file the v3.0 reader rejects. Disambiguate
                # by probing the raw tree (instead of brittle substring matching
                # on the message — the v3.0 rejection text contains "node-tree").
                from luxar.gsplats.io.load_gsplats import load_gsplat_node

                try:
                    load_gsplat_node(path)  # succeeds only for a valid v3.0 tree
                except ValueError as load_exc:
                    # Legacy/invalid → surface the actionable message (which names
                    # `luxar gsplat migrate-format`) without a traceback.
                    aprint(f"❌ {load_exc}")
                    raise typer.Exit(1) from None
                # Valid v3.0 partition/nested tree → report its shape.
                _print_gsplat_tree_summary(path)
                return
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

    The standalone ``.gsplats.zarr`` is a v3.0 node subtree — exactly what the
    viewer renders inside a scene — so it is served **directly** (``?src=``) and
    opened with no scene-compile round-trip. This works for every tree shape:
    a single leaf, an additive ladder, a ``kind=lod`` substitutive hierarchy, a
    ``kind=partition`` split, and arbitrary nestings. The viewer frames the
    camera on the file's ``position_bounds``, so no centroid mutation is needed.

    Compressed archives (``.gsplats.zarr.zip`` / ``.gsplats.zarr.tar.gz``) are
    extracted to a temporary directory before serving.

    Args:
        path: Path to .gsplats.zarr (or .zip/.tar.gz) dataset
        port: Port for data server
        viewer_port: Port for viewer
        open_browser: Whether to open browser automatically
        cors_origin: Allowed CORS origin for both servers (default "local").
    """
    try:
        import threading

        from luxar.cli.main import _serve_data, _serve_viewer
        from luxar.cli.utils import (
            build_viewer,
            check_viewer_built,
            find_available_port,
        )
        from luxar.gsplats.io.load_gsplats import _extract_compressed_zarr

        # Check viewer is built
        if not check_viewer_built():
            aprint("🔨 Building viewer...")
            if not build_viewer():
                aprint("❌ Failed to build viewer")
                raise typer.Exit(1)

        with asection(f"Quick View: {path.name}"):
            # Resolve to an on-disk .gsplats.zarr directory: extract archives to
            # a temp dir, otherwise serve the directory in place. No GSplatData
            # round-trip — the viewer consumes the node tree directly, which is
            # the only path that supports partition/nested roots.
            if str(path).endswith((".zip", ".tar.gz")):
                aprint("Extracting compressed dataset...")
                serve_target = _extract_compressed_zarr(path)
                temp_dir = serve_target.parent
            elif path.is_dir():
                serve_target = path
            else:
                aprint(f"❌ Not a .gsplats.zarr directory or archive: {path}")
                raise typer.Exit(1)

            aprint(f"Serving node tree directly: {serve_target.name}")

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
                        serve_target,
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

                # Construct data URL (no trailing slash — see CLAUDE.md gotcha)
                data_url = f"http://127.0.0.1:{actual_port}/{serve_target.name}"

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
    output_path: Path = typer.Argument(..., help="Output .luxar.zarr scene path"),
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
        luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr
        luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr --no-center
        luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr --scale-intensity 0.1
    """
    try:
        import numpy as np

        from luxar import LuxarZarrCompiler
        from luxar.cli.gsplat_config import build_dimensions_from_data
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import center_bounds, is_matrix_shaped

        with asection(f"Converting: {input_path.name} -> {output_path.name}"):
            with asection("Loading gsplat dataset"):
                # Peek at the on-disk shape. A matrix-shaped tree (leaf / additive
                # ladder / kind=lod of leaves) round-trips through GSplatData and
                # supports --center / --scale-intensity. A partition / nested tree
                # has no flat GSplatData equivalent: it is grafted node-for-node.
                node, _ = load_gsplat_node(input_path)
                matrix = is_matrix_shaped(node)

            if matrix:
                data = GSplatData.from_tree(node)
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
            else:
                kind = (
                    "partition"
                    if node.__class__.__name__ == "GSplatPartition"
                    else "nested LOD"
                )
                aprint(f"Grafting a {kind} node tree (no flat-data transforms apply)")
                if scale_intensity is not None:
                    aprint(
                        "⚠️  --scale-intensity is ignored for a partition/nested "
                        "file (re-author intensity upstream with `gsplat transform`)."
                    )
                # --center defaults True; it does not apply to a graft (the file's
                # own coordinates are preserved), so note it rather than fail.
                if center:
                    aprint(
                        "ℹ️  --center is ignored for a partition/nested file; the "
                        "node tree keeps its authored coordinates."
                    )
                bounds = center_bounds(node)
                if bounds is None:
                    raise ValueError("Could not derive bounds from the node tree")
                bmin, bmax = bounds
                box = np.array([bmin, bmax], dtype=np.float32)

                with asection("Creating Luxar scene"):
                    dims = build_dimensions_from_data(box)
                    with LuxarZarrCompiler(
                        output_path, encoding_mode=_resolve_encoding_mode(encoding)
                    ) as compiler:
                        scene = compiler.create_scene(dimensions=dims)
                        scene.add_gsplats_from_file(
                            name="gsplats",
                            path=input_path,
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
# batch — HPC batch fitting via Slurm (commands live in gsplat_ops/batch.py)
# ═══════════════════════════════════════════════════════════════════════

# `batch_validate_cmd` / `_validate_tile` are re-exported (consumed by tests via
# `luxar.cli.gsplat_commands`); keep them importable here.
from .gsplat_ops.batch import (  # noqa: E402,F401
    _validate_tile,
    app_batch,
    batch_validate_cmd,
)

app_gsplat.add_typer(app_batch, name="batch")


# Used by `info` (inspect group, still in this module) to summarise nested trees.
def _print_gsplat_tree_summary(path: Path) -> None:
    """Report the node-tree shape of a partition / nested .gsplats.zarr.

    These have no flat ``GSplatData`` (``gsplat info``'s normal path), so we
    walk the node tree and print its structure (kind, parts/levels, per-leaf
    splat counts, total, ndim, bounds) instead of failing.
    """
    import shutil

    import zarr

    from luxar.gsplats.io.load_gsplats import _extract_compressed_zarr
    from luxar.gsplats.tree import (
        GSplatLodGroup,
        GSplatPartition,
        iter_leaves,
        node_ndim,
        total_splats,
    )
    from luxar.io._compiler.gsplat_tree import read_gsplat_node

    zarr_path = path
    tmp = None
    try:
        if path.is_file():  # compressed archive
            zarr_path = _extract_compressed_zarr(path)
            tmp = zarr_path.parent
        root = zarr.open_group(str(zarr_path), mode="r")
        node = read_gsplat_node(root, root)

        aprint("\n" + "═" * 70)
        aprint("DATASET INFORMATION (node tree)")
        aprint("═" * 70)
        aprint(f"\nFile: {path.name}")
        aprint(f"Size: {format_memory_size(path.stat().st_size)}")
        kind = (
            "partition"
            if isinstance(node, GSplatPartition)
            else ("lod" if isinstance(node, GSplatLodGroup) else "leaf")
        )
        aprint(f"\nRoot kind: {kind}")
        aprint(f"Dimensions: {node_ndim(node)}D")
        aprint(f"Total splats (all leaves): {total_splats(node):,}")
        if isinstance(node, (GSplatPartition, GSplatLodGroup)):
            child_word = "part" if isinstance(node, GSplatPartition) else "level"
            aprint(f"{child_word.capitalize()}s: {len(node.children)}")
            for i, leaf in enumerate(iter_leaves(node)):
                aprint(f"  leaf {i}: {leaf.n_splats:,} splats")
        pb = root.attrs.get("position_bounds")
        if pb:
            aprint(f"Position bounds: min={pb.get('min')} max={pb.get('max')}")
    finally:
        if tmp is not None and tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
# lod — Build a representation topology (recipe) from a fitted gsplat dataset
# ═══════════════════════════════════════════════════════════════════════

# The unified `lod --recipe {flat,additive,partitioned,multiscale,substitutive,
# pyramid}` command lives in `cli/lod.py` (a thin wrapper over the pure recipe
# builders in `gsplats/lod/recipes.py`). It replaces the former
# `lod additive/substitutive/pyramid` subcommands.
from .lod import register_lod_command  # noqa: E402

register_lod_command(app_gsplat)
