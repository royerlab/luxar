"""``luxar gsplat inspect`` commands.

Each command is a plain function; ``register_inspect_commands(app)`` wires them onto
the shared ``app_gsplat`` Typer.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
from arbol import aprint, asection

from ..common_options import CorsOriginOption, make_port_option
from ..utils import _DEFAULT_CORS_ORIGIN, format_memory_size

if TYPE_CHECKING:
    import numpy as np


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


def quick_view(
    path: Path = typer.Argument(
        ..., exists=True, help="Path to .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    port: int = make_port_option(8000, "Data server port"),
    viewer_port: int = typer.Option(5173, "--viewer-port", help="Viewer port"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser"),
    cors_origin: CorsOriginOption = _DEFAULT_CORS_ORIGIN,
) -> None:
    """Quick view of a Gaussian splat dataset in the Luxar web viewer.

    The standalone ``.gsplats.zarr`` is a v3.3 node subtree — exactly what the
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

        from luxar.cli.serving import _serve_data, _serve_viewer
        from luxar.cli.utils import (
            ensure_viewer_built,
            pick_port,
            wait_for_server,
        )
        from luxar.gsplats.io._archive import extract_compressed_zarr

        # Check viewer is built
        if not ensure_viewer_built():
            raise typer.Exit(1)

        with asection(f"Quick View: {path.name}"):
            # Resolve to an on-disk .gsplats.zarr directory: extract archives to
            # a temp dir, otherwise serve the directory in place. No GSplatData
            # round-trip — the viewer consumes the node tree directly, which is
            # the only path that supports partition/nested roots.
            if str(path).endswith((".zip", ".tar.gz")):
                aprint("Extracting compressed dataset...")
                serve_target = extract_compressed_zarr(path)
                temp_dir = serve_target.parent
            elif path.is_dir():
                serve_target = path
            else:
                aprint(f"❌ Not a .gsplats.zarr directory or archive: {path}")
                raise typer.Exit(1)

            aprint(f"Serving node tree directly: {serve_target.name}")

            # Find available ports
            actual_port = pick_port(port, label="data")
            actual_viewer_port = pick_port(viewer_port, label="viewer")

            if actual_port is None or actual_viewer_port is None:
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
                if not wait_for_server("127.0.0.1", actual_port, data_thread):
                    aprint("⚠️  Data server did not become ready.")

                # Construct data URL (no trailing slash — see CLAUDE.md gotcha).
                # The store is mounted at the server root: no name suffix.
                data_url = f"http://127.0.0.1:{actual_port}"

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


def _print_gsplat_tree_summary(path: Path) -> None:
    """Report the node-tree shape of a partition / nested .gsplats.zarr.

    These have no flat ``GSplatData`` (``gsplat info``'s normal path), so we
    walk the node tree and print its structure (kind, parts/levels, per-leaf
    splat counts, total, ndim, bounds) instead of failing.
    """
    import shutil

    import zarr

    from luxar.gsplats.io._archive import extract_compressed_zarr
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
            zarr_path = extract_compressed_zarr(path)
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


def annotate_quality(
    path: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to an UNCOMPRESSED .gsplats.zarr directory (in-place; "
        ".zip/.tar.gz stores must be unpacked first).",
    ),
    with_quality: bool = typer.Option(
        False,
        "--with-quality",
        help="Also measure per-level mixture-L2 quality Q vs each lod group's "
        "finest content (loads full splat arrays — slower; the free e(k)/w "
        "stamps alone already enable the viewer's energy-threshold upgrades).",
    ),
    max_pair_splats: int = typer.Option(
        2_000_000,
        "--max-pair-splats",
        min=1,
        help="Subsample cap per mixture for the Q measurement.",
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Compute device for Q: auto|cpu|cuda|mps."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compute and print the stamps, write nothing."
    ),
) -> None:
    """Retrofit Q·e quality stamps onto an existing .gsplats.zarr, in place.

    Stamps, without refitting or re-laddering:

    - lod_stats.energy_fraction_cum per additive sub-LOD (cumulative committed
      self-energy fraction e(k); cheap O(N))
    - level_stats.reference_energy per leaf (the aggregation weight w)
    - level_stats.quality per substitutive level (--with-quality; measured
      mixture-L2 Q vs the group's finest content)

    The root content_hash is re-stamped so the viewer's persistent cache
    invalidates automatically.

    Examples:
        luxar gsplat annotate-quality data.gsplats.zarr
        luxar gsplat annotate-quality data.gsplats.zarr --with-quality
        luxar gsplat annotate-quality data.gsplats.zarr --dry-run
    """
    from luxar.gsplats.lod.annotate import annotate_quality_store

    try:
        with asection(f"Annotating {path.name}{' (dry run)' if dry_run else ''}"):
            report = annotate_quality_store(
                path,
                with_quality=with_quality,
                max_pair_splats=max_pair_splats,
                device=device or "auto",
                dry_run=dry_run,
            )
            aprint(f"Leaves stamped: {len(report.leaves)}")
            for leaf in report.leaves:
                e_str = ", ".join(f"{e:.3f}" for e in leaf.energy_fraction_cum)
                aprint(
                    f"  {leaf.path or '/'}: {leaf.n_splats:,} splats, "
                    f"w={leaf.reference_energy:.4g}, e(k)=[{e_str}]"
                )
            if with_quality:
                aprint(f"Levels measured: {len(report.levels)}")
                for lev in report.levels:
                    aprint(
                        f"  {lev.path or '/'}: {lev.n_splats:,} splats, "
                        f"Q={lev.quality:.4f}, w={lev.reference_energy:.4g}"
                    )
            if dry_run:
                aprint("Dry run: nothing written.")
            else:
                aprint("✓ Stamps written; content_hash refreshed.")
    except typer.Exit:
        raise
    except ValueError as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1)


def register_inspect_commands(app: typer.Typer) -> None:
    """Register the inspect commands onto ``app_gsplat``."""
    app.command("info")(info_dataset)
    app.command("napari")(napari_viewer)
    app.command("view")(quick_view)
    app.command("compare")(compare_quality)
    app.command("annotate-quality")(annotate_quality)
