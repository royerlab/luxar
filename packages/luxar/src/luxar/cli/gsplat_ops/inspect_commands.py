"""``luxar gsplat inspect`` commands.

Each command is a plain function; ``register_inspect_commands(app)`` wires them onto
the shared ``app_gsplat`` Typer.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from arbol import aprint, asection

from ..common_options import CorsOriginOption, make_port_option
from ..utils import _DEFAULT_CORS_ORIGIN, format_memory_size

if TYPE_CHECKING:
    import numpy as np

    from luxar.gsplats.doctor import DoctorReport, Finding, StoreKind
    from luxar.gsplats.gsplat_data import GSplatData


_IMPORTANT_FITTING_KEYS = (
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
    "foreground_psnr_db",
    "foreground_threshold",
    "foreground_fraction",
    "ssim",
    "mse",
    # Amplitudes are background-relative, so these normalization stamps are
    # interpretation-critical rather than generic trailing metadata (#1175).
    "floor",
    "image_min",
    "image_max",
    "intensity_range",
    "convergence_time",
    "culled",
    "culling_method",
    "n_original",
    "n_culled",
    "amplitude_retention",
)


def _part_provenance_depth(records: list[Any]) -> int:
    depth = 1
    for record in records:
        if not isinstance(record, dict):
            continue
        fitting = record.get("fitting")
        nested = fitting.get("part_provenance") if isinstance(fitting, dict) else None
        if isinstance(nested, list):
            depth = max(depth, 1 + _part_provenance_depth(nested))
    return depth


def _print_fitting_value(
    key: str, value: Any, *, show_full_provenance: bool = False
) -> None:
    if (
        key == "part_provenance"
        and isinstance(value, list)
        and not show_full_provenance
    ):
        depth = _part_provenance_depth(value)
        if depth == 3:
            suffix = ", nested channels × timepoints"
        elif depth > 1:
            suffix = f", nested component records ({depth} levels)"
        else:
            suffix = ""
        part_word = "part" if len(value) == 1 else "parts"
        aprint(f"  {key}: {len(value)} {part_word}{suffix}")
        return
    if isinstance(value, float):
        aprint(f"  {key}: {value:.6f}")
    else:
        aprint(f"  {key}: {value}")


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


def _compute_splat_volumes(
    cholesky_factors: "np.ndarray", ndim: int, truncate: float
) -> "np.ndarray":
    """Compute volumes at ``truncate``-sigma for each splat.

    ``truncate`` is REQUIRED and comes from the dataset's own
    ``truncation_radius``: this used to be a hardcoded 3, so a dataset fitted at
    the canonical 2.75 was measured at a support it never had (the report was
    ``(3/2.75)**ndim`` too large — 30% at 3D).
    """
    import numpy as np

    # Extract diagonal elements from packed Cholesky factors
    # For nD: positions are at cumsum([1,2,3,...,ndim]) - 1
    diag_indices = np.cumsum(np.arange(1, ndim + 1)) - 1
    diag_elements = cholesky_factors[:, diag_indices]

    # Volume ∝ det(Σ)^(1/2) = |det(L)| = |product of diagonal elements|
    det_L = np.prod(diag_elements, axis=1)
    det_Sigma = det_L**2

    # Volume of nD ellipsoid at T-sigma (T = the dataset's truncation radius)
    # V = (2π)^(n/2) * det(Σ)^(1/2) * T^n / Γ(n/2 + 1)
    # For simplicity, use det(Σ)^(1/2) * T^n as proxy
    volumes: np.ndarray = np.abs(det_Sigma) ** 0.5 * (float(truncate) ** ndim)

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


def _normalized_amplitude_cdf(amplitudes: "np.ndarray") -> "Optional[np.ndarray]":
    """Return the descending amplitude CDF with a non-saturating accumulator."""
    import numpy as np

    sorted_amplitudes = np.sort(amplitudes)[::-1]
    cumulative = np.cumsum(sorted_amplitudes, dtype=np.float64)
    total = cumulative[-1]
    if total <= 0:
        return None
    cumulative /= total
    return cumulative


def _print_dataset_metadata(
    stats: dict[str, Any],
    source_grid_keys: tuple[str, ...],
    *,
    show_full_provenance: bool,
) -> None:
    """Print metadata, excluding stats already reported by the source-grid block."""
    aprint("\n" + "─" * 70)
    aprint("METADATA")
    aprint("─" * 70)

    displayed_keys = set()
    for key in _IMPORTANT_FITTING_KEYS:
        if key in stats:
            _print_fitting_value(
                key,
                stats[key],
                show_full_provenance=show_full_provenance,
            )
            displayed_keys.add(key)

    # The source-volume block already reported its own keys (and RECOMPUTED
    # voxels/splat from the stored splats), so re-dumping them here would quote
    # one quantity twice with two different numbers. Only the keys it actually
    # reported are suppressed: when that block bailed out (no `source_shape`)
    # it returns nothing and the stamps still surface here.
    remaining = set(stats) - displayed_keys - set(source_grid_keys)
    if remaining:
        aprint("\nAdditional Metadata:")
        for key in sorted(remaining):
            if key not in ["movie_frames", "movie_shape", "config", "provenance"]:
                value = stats[key]
                if key == "part_provenance":
                    _print_fitting_value(
                        key,
                        value,
                        show_full_provenance=show_full_provenance,
                    )
                elif isinstance(value, (dict, list)):
                    aprint(f"  {key}: {type(value).__name__} with {len(value)} items")
                else:
                    aprint(f"  {key}: {value}")


def _load_info_data(
    path: Path, *, full_provenance: bool
) -> tuple["Optional[GSplatData]", bool]:
    """Load a flat dataset, or print a valid tree summary or load failure."""
    from luxar.gsplats.gsplat_data import GSplatData

    with asection(f"Loading dataset: {path.name}"):
        try:
            data = GSplatData.load(path, include_stats=True)
        except ValueError:
            # GSplatData.load raises for two distinct reasons: (a) a valid v3.0
            # partition/nested tree that has no flat GSplatData form, or (b) a
            # legacy/invalid file the v3.0 reader rejects. Probe the raw tree
            # instead of matching text: the rejection itself says "node-tree".
            from luxar.gsplats.io.load_gsplats import load_gsplat_node

            try:
                load_gsplat_node(path)  # succeeds only for a valid v3.0 tree
            except ValueError as load_exc:
                aprint(f"❌ {load_exc}")
                return None, False
            _print_gsplat_tree_summary(path, show_full_provenance=full_provenance)
            return None, True

        n_splats = len(data.amplitudes)
        ndim = data.centers.shape[1]
        aprint(f"✓ Loaded {n_splats:,} splats ({ndim}D)")
        return data, True


def _print_color_information(colors: "Optional[np.ndarray]") -> None:
    """Print per-channel color statistics when colors are present."""
    if colors is None:
        return

    import numpy as np

    aprint("\n" + "─" * 70)
    aprint("COLOR INFORMATION")
    aprint("─" * 70)

    aprint(f"\nColor dtype: {colors.dtype}")
    aprint(f"Color range: [{colors.min():.4f}, {colors.max():.4f}]")

    for i, channel_name in enumerate(["Red", "Green", "Blue"]):
        channel_data = colors[:, i]
        aprint(f"\n{channel_name} Channel:")
        aprint(f"  Mean: {np.mean(channel_data):.4f}")
        aprint(f"  Std:  {np.std(channel_data):.4f}")


def _info_report(
    path: Path,
    *,
    show_histograms: bool,
    bins: int,
    full_provenance: bool,
) -> bool:
    """Print a Gaussian splat dataset report and return whether it succeeded."""
    try:
        import numpy as np

        from luxar.gsplats.utils.alpha import effective_amplitudes

        data, report_succeeded = _load_info_data(path, full_provenance=full_provenance)
        if data is None:
            return report_succeeded
        n_splats = len(data.amplitudes)
        ndim = data.centers.shape[1]

        # ================================================================
        # Basic Information
        # ================================================================
        aprint("\n" + "═" * 70)
        aprint("DATASET INFORMATION")
        aprint("═" * 70)

        stored_bytes = _store_size(path)
        aprint(f"\nFile: {path.name}")
        aprint(f"Size: {format_memory_size(stored_bytes)}")

        aprint(f"\nSplats: {n_splats:,}")
        aprint(f"Dimensions: {ndim}D")
        aprint(f"Has Colors: {'Yes' if data.colors is not None else 'No'}")
        source_grid_keys = _print_source_grid(data, stored_bytes)

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
        cumsum_norm = _normalized_amplitude_cdf(effective_amplitudes(data))

        # Find how many splats contribute to 50%, 90%, 95%, 99%
        if cumsum_norm is not None:
            for threshold in [0.50, 0.90, 0.95, 0.99]:
                n_contrib = np.searchsorted(cumsum_norm, threshold) + 1
                pct = (n_contrib / n_splats) * 100
                aprint(
                    f"  Top {n_contrib:,} splats ({pct:.1f}%) contribute "
                    f"{threshold * 100:.0f}% of total rendered amplitude (A·α)"
                )

        if show_histograms:
            aprint(
                _ascii_histogram(
                    data.amplitudes, bins=bins, title="Amplitude Distribution"
                )
            )

        # ================================================================
        # Volume Statistics (at the dataset's own truncation radius)
        # ================================================================
        # `:g` so the canonical 2.75 reads "2.75" and an integral 3.0 reads "3".
        truncate = data.truncation_radius
        aprint("\n" + "─" * 70)
        aprint(f"VOLUME ANALYSIS ({truncate:g}-Sigma)")
        aprint("─" * 70)

        volumes = _compute_splat_volumes(data.cholesky_factors, ndim, truncate)
        _print_statistics_table(volumes, "Volume")

        if show_histograms:
            aprint(
                _ascii_histogram(
                    volumes,
                    bins=bins,
                    title=f"Volume Distribution ({truncate:g}σ)",
                )
            )

        _print_color_information(data.colors)

        # ================================================================
        # Metadata
        # ================================================================
        if data.stats:
            _print_dataset_metadata(
                data.stats,
                source_grid_keys,
                show_full_provenance=full_provenance,
            )

        # ================================================================
        # Summary
        # ================================================================
        aprint("\n" + "═" * 70)
        aprint("SUMMARY")
        aprint("═" * 70)

        aprint(f"\n✓ Dataset contains {n_splats:,} Gaussian splats in {ndim}D")
        aprint(f"✓ Total amplitude: {total_amp:.4e}")
        aprint(f"✓ Bounding box volume: {total_volume:.4e}")
        aprint(f"✓ Mean splat volume ({truncate:g}σ): {np.mean(volumes):.4e}")

        # Pruning recommendation
        n_for_95pct = (
            np.searchsorted(cumsum_norm, 0.95) + 1
            if cumsum_norm is not None
            else n_splats
        )
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

        return True
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


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
    full_provenance: bool = typer.Option(
        False,
        "--full-provenance",
        help="Print the complete nested fitting/part_provenance record",
    ),
) -> None:
    """Show detailed information about a Gaussian splat dataset.

    Displays comprehensive statistics including:
    - Number of splats and dimensions
    - Bounding box in each dimension
    - Amplitude distribution with statistics and histogram
    - Volume distribution (size at the dataset's own truncation radius, in
      sigmas) with statistics and histogram
    - Color information (if present)
    - Metadata (fitting info, provenance, etc.)

    Examples:
        # Basic info with histograms
        luxar gsplat info dataset.gsplats.zarr.zip

        # Info without histograms (faster)
        luxar gsplat info dataset.gsplats.zarr.zip --no-histograms

        # More detailed histograms
        luxar gsplat info dataset.gsplats.zarr.zip --bins 60

        # Print the complete nested per-part fitting record
        luxar gsplat info dataset.gsplats.zarr.zip --full-provenance

    Args:
        path: Path to .gsplats.zarr or compressed archive
        show_histograms: Whether to display ASCII histograms
        bins: Number of bins for histogram plots
        full_provenance: Whether to print the complete nested component record
    """
    if not _info_report(
        path,
        show_histograms=show_histograms,
        bins=bins,
        full_provenance=full_provenance,
    ):
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
        from luxar.gsplats.io.load_gsplats import load_default_gsplats

        with asection(f"Loading gsplat dataset: {path.name}"):
            # Load dataset
            data = load_default_gsplats(path, include_stats=True)

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


def _resolve_view_target(path: Path) -> tuple[Path, Optional[Path]]:
    """Resolve a ``gsplat view`` argument to a servable store + its temp dir.

    Through :func:`~luxar.gsplats.io._archive.resolve_store_path`, the same call
    the loader makes, so this command cannot disagree with it about which node an
    archive holds. A store is served over HTTP as a DIRECTORY, so the archive
    file itself is never a usable target here (no ``flat_zip_in_place``).

    The pre-check keys on the SUFFIX, not on the file type: anything not named
    ``*.zip`` / ``*.tar.gz`` must be a directory or it is refused HERE rather
    than by the resolver, which only rejects a REGULAR file — so a FIFO or a
    device node (both of which pass typer's ``exists=True``) would otherwise be
    handed to the data server as the mount root, which fails deep inside a
    background thread while the command sits blocked on a viewer serving
    nothing. The converse is not covered and never was: a DIRECTORY named
    ``x.zip`` takes the archive branch and dies in the extractor. An unusable or
    unsafe ARCHIVE is a real error and propagates.
    """
    from luxar.gsplats.io._archive import resolve_store_path

    if str(path).endswith((".zip", ".tar.gz")):
        aprint("Extracting compressed dataset...")
    elif not path.is_dir():
        aprint(f"❌ Not a .gsplats.zarr directory or archive: {path}")
        raise typer.Exit(1)
    return resolve_store_path(path)


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

    The standalone ``.gsplats.zarr`` is a v3.4 node subtree — exactly what the
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
    # Bound before the `try` so the `finally` below can test it: the store
    # resolver answers None for a directory store, and `rmtree(None)` raises —
    # turning a real error into an unrelated one.
    temp_dir: Optional[Path] = None
    try:
        import threading

        from luxar.cli.serving import _serve_data, _serve_viewer
        from luxar.cli.utils import (
            dataset_title,
            ensure_viewer_built,
            pick_port,
            wait_for_server,
        )

        # Check viewer is built
        if not ensure_viewer_built():
            raise typer.Exit(1)

        with asection(f"Quick View: {path.name}"):
            # Resolve to an on-disk .gsplats.zarr directory: extract archives to
            # a temp dir, otherwise serve the directory in place. No GSplatData
            # round-trip — the viewer consumes the node tree directly, which is
            # the only path that supports partition/nested roots.
            serve_target, temp_dir = _resolve_view_target(path)

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
                    title=dataset_title(path),
                )

    except KeyboardInterrupt:
        aprint("\n🛑 Shutting down viewer...")
    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
    finally:
        # Every exit removes the extraction, not just the two error handlers
        # that used to: a NORMAL return from `_serve_viewer`, and the
        # `typer.Exit` a failed `pick_port` raises AFTER the archive has been
        # extracted, both left a full uncompressed copy of the dataset behind
        # in /tmp.
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _print_quality_comparison(
    metrics: dict[str, Any],
    *,
    reference_name: str,
    ref_shape: Any,
    gsplats_name: str,
    n_splats: int,
) -> None:
    """Render the ``gsplat compare`` results table."""
    aprint("\n" + "=" * 50)
    aprint("QUALITY COMPARISON")
    aprint("=" * 50)
    aprint(f"\nReference:  {reference_name} {ref_shape}")
    aprint(f"GSplats:    {gsplats_name} ({n_splats:,} splats)")
    aprint("")
    aprint(f"  MSE:             {metrics['mse']:.6g}")
    aprint(f"  PSNR:            {metrics['psnr_db']:.2f} dB")
    if "foreground_psnr_db" in metrics:
        # The share is not decoration: on a 99%-empty volume the global PSNR
        # above is largely a score for reproducing the emptiness, and this line
        # says how little of the volume the honest number was taken over.
        aprint(
            f"  PSNR foreground: {metrics['foreground_psnr_db']:.2f} dB "
            f"(over {metrics['foreground_fraction'] * 100:.2f}% of voxels, "
            f"Otsu > {metrics['foreground_threshold']:.4g})"
        )
    aprint(f"  SSIM:            {metrics['ssim']:.4f}")
    aprint(f"  Rel L2:          {metrics['rel_l2']:.6g}")
    aprint(f"  Max Abs Error:   {metrics['max_abs_error']:.6g}")
    if "compression_ratio" in metrics:
        aprint(f"  Compression:     {metrics['compression_ratio']:.1f}x")
    aprint("=" * 50)


def _load_gsplats_for_comparison(path: Path) -> tuple["GSplatData", int, int]:
    """Materialize the tree selection that the renderer shows by default.

    Root stats come along on both paths so the caller can resolve the
    normalization basis (#1173). They remain unscrubbed because this temporary
    flat dataset is never persisted.
    """
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.load_gsplats import load_gsplat_node
    from luxar.gsplats.tree import iter_default_leaves, total_splats

    node, stats = load_gsplat_node(path, include_stats=True)
    n_leaves = sum(1 for _ in iter_default_leaves(node))
    data = GSplatData.from_default_selection(node, stats=stats)
    return data, n_leaves, total_splats(node)


def _announce_unscored_comparison_splats(
    *, n_stored_splats: int, n_scored_splats: int
) -> None:
    """Report coarse LOD splats excluded from the default-rendered selection."""
    if n_stored_splats > n_scored_splats:
        aprint(
            f"Skipped {n_stored_splats - n_scored_splats:,} splats in coarse "
            "LOD levels; compression ratio covers the whole store"
        )


def _validate_image_min_override(image_min: Optional[float]) -> None:
    if image_min is not None and (not math.isfinite(image_min) or image_min < 0.0):
        raise typer.BadParameter(
            "must be a finite, non-negative level", param_hint="--image-min"
        )


def _reference_on_dataset_basis(
    reference: "np.ndarray", stats: Any, image_min: Optional[float]
) -> "np.ndarray":
    from luxar.gsplats.fit_basis import (
        MISSING_BASIS_HINT,
        fit_image_min,
        reference_on_fit_basis,
    )

    resolved_min = float(image_min) if image_min is not None else fit_image_min(stats)
    if resolved_min is None:
        aprint(f"WARNING: {MISSING_BASIS_HINT}")
    elif resolved_min == 0.0:
        aprint("Basis: fit removed no background (image_min=0)")
    else:
        source = "--image-min" if image_min is not None else "dataset"
        reference = reference_on_fit_basis(reference, resolved_min)
        aprint(
            f"Basis: reference shifted by image_min={resolved_min:.6g} "
            f"(from {source}); scores are background-relative, matching the render"
        )
    return reference


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
        help="Render shape as comma-separated ints. NOT an override: metrics need "
        "both volumes on the same grid, so a value that differs from the "
        "reference shape exits 1. Omit it to use the reference shape.",
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
    image_min: Optional[float] = typer.Option(
        None,
        "--image-min",
        help="Background level the fit subtracted, for a dataset that does not "
        "record one. A render is background-relative, so the reference is "
        "shifted by this before scoring. Normally read from the dataset; pass it "
        "only for stores fitted before the level was persisted.",
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
    runs on GPU when available. Partition and nested stores are scored over
    their default-rendered selection: all parts and each LOD group's finest
    level. The compression ratio covers the whole store, including coarse
    levels that are not scored.

    Examples:
        luxar gsplat compare fitted.gsplats.zarr original.tiff
        luxar gsplat compare fitted.gsplats.zarr original.npy --device cuda
        luxar gsplat compare fitted.gsplats.zarr original.zarr --output-json metrics.json
        luxar gsplat compare fitted.gsplats.zarr original.zarr -j metrics.json -q
    """
    _validate_image_min_override(image_min)

    try:
        import json

        import torch

        from luxar.cli.gsplat_config import load_volume, parse_shape
        from luxar.gsplats.metrics import compute_quality_metrics
        from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor
        from luxar.gsplats.utils.device import resolve_torch_device

        with asection("Quality Comparison"):
            # Load gsplat dataset
            with asection("Loading gsplat dataset"):
                # Stats come along (`include_stats=True` inside the helper) so
                # the normalization basis is reachable: a render is
                # background-relative and the reference is raw, and without the
                # stored `image_min` the two cannot be reconciled (#1173). This
                # is metadata only — no extra array decode.
                data, n_leaves, n_stored_splats = _load_gsplats_for_comparison(
                    gsplats_path
                )
                n_splats = data.n_splats
                ndim = data.ndim
                aprint(f"Loaded {n_splats:,} splats ({ndim}D)")
                aprint(f"Materialized {n_leaves:,} default-rendered leaf/leaves")
                _announce_unscored_comparison_splats(
                    n_stored_splats=n_stored_splats, n_scored_splats=n_splats
                )

            # Resolve truncation radius from dataset if not explicitly set
            if truncate is None:
                truncate = data.truncation_radius

            # Load reference volume
            with asection("Loading reference volume"):
                ref_np = load_volume(
                    reference_path, channel=channel, timepoint=timepoint
                )
                ref_shape = ref_np.shape

                # Put the reference on the render's basis before anything scores
                # it. An explicit --image-min wins over the stored level so a
                # pre-provenance store is still comparable.
                ref_np = _reference_on_dataset_basis(ref_np, data.stats, image_min)

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
            dev = str(resolve_torch_device(device))

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

                # Upload reference to same device, on the FIT's basis. Scoring
                # a background-relative render against a pedestal-bearing
                # reference charges the fit for background it never claimed to
                # represent, which is the artifact this command most needed
                # fixing (#1173).
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
            _print_quality_comparison(
                metrics,
                reference_name=reference_path.name,
                ref_shape=ref_shape,
                gsplats_name=gsplats_path.name,
                n_splats=n_splats,
            )

        # JSON output
        if output_json is not None:
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


def _print_normalization_block(root: Any) -> None:
    """List a store's ``pipeline/`` normalization block, if it has one (#1175).

    The tree branch of ``info`` has no ``stats`` dict and no "Additional
    Metadata" dump, so without this a ``kind=partition`` — the tiled CLI's
    DEFAULT output — showed nothing at all about the background level its fit
    subtracted, even though amplitudes are relative to exactly that level.
    """
    from luxar.gsplats.io.save_gsplats import NORMALIZATION_STATS_KEYS

    if "pipeline" not in root:
        return
    attrs = dict(root["pipeline"].attrs)
    present = [(k, attrs[k]) for k in NORMALIZATION_STATS_KEYS if k in attrs]
    if not present:
        return
    aprint("\nNormalization (pipeline/):")
    for key, value in present:
        aprint(f"  {key}: {value}")


def _print_gsplat_tree_summary(
    path: Path, *, show_full_provenance: bool = False
) -> None:
    """Report the node-tree shape of a partition / nested .gsplats.zarr.

    These have no flat ``GSplatData`` (``gsplat info``'s normal path), so we
    walk the node tree and print its structure (kind, parts/levels, per-leaf
    splat counts, total, ndim, bounds) instead of failing.
    """
    import shutil

    from luxar._zarr_compat import open_group as zarr_open_group
    from luxar.gsplats.io._archive import resolve_store_path
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
            # The same resolver `load_gsplat_node` gated on just above: `info`
            # uses that load purely as a gate and then re-resolves here, so the
            # two must not be able to disagree about which node the archive holds
            # (they did for a FLAT archive — the gate saw one node, this saw an
            # arbitrary child).
            zarr_path, tmp = resolve_store_path(path)
        root = zarr_open_group(str(zarr_path), mode="r")
        node = read_gsplat_node(root, root)

        aprint("\n" + "═" * 70)
        aprint("DATASET INFORMATION (node tree)")
        aprint("═" * 70)
        aprint(f"\nFile: {path.name}")
        # Same measurement as the flat report's "Size:" line — a directory store's
        # own stat() is the ~4 KB directory entry, not the chunks in it, and a
        # partition is the shape most likely to BE a directory.
        aprint(f"Size: {format_memory_size(_store_size(path))}")
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
        if "fitting" in root:
            aprint("\nFitting (fitting/):")
            fitting = dict(root["fitting"].attrs)
            displayed_keys = set()
            for key in _IMPORTANT_FITTING_KEYS:
                if key in fitting:
                    _print_fitting_value(
                        key,
                        fitting[key],
                        show_full_provenance=show_full_provenance,
                    )
                    displayed_keys.add(key)
            for key in sorted(set(fitting) - displayed_keys):
                _print_fitting_value(
                    key,
                    fitting[key],
                    show_full_provenance=show_full_provenance,
                )
        _print_normalization_block(root)
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
                # `w=none` where no weight was written: a REVEAL ladder carries
                # neither half of the e/w pair, and printing the number we
                # computed would read as one that reached the store.
                w_str = (
                    "none"
                    if leaf.reference_energy is None
                    else f"{leaf.reference_energy:.4g}"
                )
                aprint(
                    f"  {leaf.path or '/'}: {leaf.n_splats:,} splats, "
                    f"w={w_str}, e(k)=[{e_str}]"
                )
            if with_quality:
                aprint(f"Levels measured: {len(report.levels)}")
                for lev in report.levels:
                    lw = (
                        "none"
                        if lev.reference_energy is None
                        else f"{lev.reference_energy:.4g}"
                    )
                    aprint(
                        f"  {lev.path or '/'}: {lev.n_splats:,} splats, "
                        f"Q={lev.quality:.4f}, w={lw}"
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


_SEVERITY_MARK = {"error": "❌", "warning": "⚠️ ", "note": "ℹ️ "}


def _print_finding(finding: "Finding") -> None:
    """One doctor finding: what it is, what it costs, and where it stands."""
    mark = _SEVERITY_MARK.get(finding.severity, "•")
    aprint(f"{mark} [{finding.path}] {finding.summary}")
    if finding.detail:
        aprint(f"     {finding.detail}")
    if finding.fixed:
        aprint(f"     ✅ fixed: {finding.remedy}")
    elif finding.fixable:
        aprint(f"     🔧 fixable: {finding.remedy} — re-run with --fix")
    elif finding.remedy:
        aprint(f"     → {finding.remedy}")


def _print_report(report: "DoctorReport") -> None:
    """Every finding, then whatever the repairs left standing."""
    if not report.findings:
        aprint(f"✅ No problems found ({len(report.checks_run)} check(s) run).")
    for finding in report.findings:
        _print_finding(finding)
    if report.residual:
        # A repair is not always a cure — removing a misleading tree from parts
        # that cannot be ordered exactly leaves a lesser condition behind. This
        # is the store as it stands AFTER the repairs, and what the exit code
        # keys on.
        aprint("\nStill standing after the repairs:")
        for finding in report.residual:
            _print_finding(finding)


def _resolve_doctor_store_kind(path: Path) -> "StoreKind":
    """Classify a doctor input while keeping CLI errors concise."""
    from luxar.gsplats.doctor import resolve_store_kind

    try:
        return resolve_store_kind(path)
    except typer.Exit:
        raise
    except Exception as exc:
        aprint(f"❌ {exc}")
        raise typer.Exit(1) from None


def _print_doctor_info(
    path: Path,
    store_kind: "StoreKind",
    *,
    histograms: bool,
    bins: int,
    full_provenance: bool,
) -> None:
    """Print the optional report appropriate for a doctor input."""
    if store_kind == "gsplats":
        _info_report(
            path,
            show_histograms=histograms,
            bins=bins,
            full_provenance=full_provenance,
        )
    elif store_kind == "scene":
        aprint("ℹ️ The gsplat info report does not apply to a Luxar scene.")
    else:
        aprint(
            "ℹ️ Could not classify this store from its metadata; "
            "skipping the optional info report."
        )


def doctor(
    path: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to a .gsplats.zarr dataset or .luxar.zarr scene (or .zip/.tar.gz)",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Repair what can be repaired, in place. Requires an UNCOMPRESSED "
        "zarr directory. Without this, doctor only reports.",
    ),
    info: bool = typer.Option(
        True,
        "--info/--no-info",
        help="Also print the full `gsplat info` report above the diagnosis for "
        "standalone .gsplats.zarr inputs.",
    ),
    histograms: bool = typer.Option(
        False,
        "--histograms/--no-histograms",
        help="Include the info report's ASCII histograms (implies --info).",
    ),
    bins: int = typer.Option(
        40,
        "--bins",
        "-b",
        help="Number of bins for info-report histograms.",
        min=10,
        max=100,
    ),
    full_provenance: bool = typer.Option(
        False,
        "--full-provenance",
        help="Print the complete nested fitting/part_provenance record in the "
        "info report (implies --info).",
    ),
    json_out: Optional[Path] = typer.Option(
        None, "--json", help="Write the findings to a JSON file as well."
    ),
) -> None:
    """Examine Luxar partition metadata and optionally repair it.

    A dataset can load perfectly and still be missing something a later Luxar
    learned to record, or be carrying metadata that went stale under an edit —
    conditions that are invisible in the viewer, because the scene still renders,
    just not as well as it should. Doctor names those, explains what each costs,
    and with --fix repairs the ones whose correct value can be recovered from the
    store itself. No re-fitting, and the root content_hash is re-stamped so the
    viewer's cache invalidates.

    Currently diagnosed:

    - Partition split planes (bsp_tree): missing, or stale after a transform.
      Without them the viewer orders parts by centroid, which is not a valid
      painter's order — it pops at the seams under `normal`/`volumetric`
      blending. Recovered from the part boxes when those are disjoint.

    Exits non-zero when a problem is left standing, so it can gate a pipeline.

    Examples:
        luxar gsplat doctor data.gsplats.zarr
        luxar gsplat doctor scene.luxar.zarr --no-info
        luxar gsplat doctor data.gsplats.zarr --fix
        luxar gsplat doctor data.gsplats.zarr --histograms --bins 60
        luxar gsplat doctor data.gsplats.zarr --full-provenance
        luxar gsplat doctor data.gsplats.zarr --no-info --json report.json
    """
    import json as _json

    from luxar.gsplats.doctor import diagnose_store

    if histograms or full_provenance:
        info = True
    store_kind = _resolve_doctor_store_kind(path)
    if info:
        _print_doctor_info(
            path,
            store_kind,
            histograms=histograms,
            bins=bins,
            full_provenance=full_provenance,
        )

    with asection(f"Diagnosing: {path.name}"):
        try:
            report = diagnose_store(path, fix=fix)
        except typer.Exit:
            raise
        except Exception as exc:
            aprint(f"❌ {exc}")
            raise typer.Exit(1) from None

        _print_report(report)

        if json_out is not None:
            json_out.write_text(_json.dumps(report.as_dict(), indent=2))
            aprint(f"Wrote {json_out}")

        unresolved = report.unresolved
        if unresolved:
            aprint(f"\n{len(unresolved)} problem(s) outstanding.")
            raise typer.Exit(1)
        if report.fix and any(f.fixed for f in report.findings):
            aprint("\n✅ All diagnosed problems repaired.")


def register_inspect_commands(app: typer.Typer) -> None:
    """Register the inspect commands onto ``app_gsplat``."""
    app.command("info")(info_dataset)
    app.command("doctor")(doctor)
    app.command("napari")(napari_viewer)
    app.command("view")(quick_view)
    app.command("compare")(compare_quality)
    app.command("annotate-quality")(annotate_quality)


def _store_size(path: Path) -> int:
    """Bytes a dataset occupies: an archive's own size, a directory store's total.

    ``Path.stat().st_size`` on a `.gsplats.zarr` directory reports the directory
    entry (typically 4 KB), not the chunks inside it — off by orders of magnitude,
    and it would contradict the compression line printed from the same number.
    """
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return 0


#: ``stats`` keys :func:`_print_source_grid` reports itself. ``info``'s
#: "Additional Metadata" dump skips exactly these once the source block has run,
#: so one quantity is never quoted twice in one report: the block RECOMPUTES
#: ``voxels/splat`` from the splats actually stored, and on any dataset whose
#: count changed after the fit (post-fit culling is on by default) the stamped
#: ``voxels_per_splat`` disagrees with it.
_SOURCE_GRID_STATS_KEYS = (
    "source_shape",
    "source_declared",
    "source_dtype",
    "source_voxels",
    "source_bytes",
    "source_stored_bytes",
    "fitted_shape",
    "fitted_voxels",
    "occupancy",
    "voxels_per_splat",
)


def _voxels_per_splat(stats: dict, n_splats: int) -> Optional[float]:
    """Voxels per splat for the splats actually IN the file, else the stamp.

    Recomputed rather than read from ``voxels_per_splat``: post-fit culling (on
    by default) and any later ``cull``/``decimate`` change the count without
    restamping, and a figure that contradicts the "Splats:" line printed just
    above would be worse than none. The stamp is the fallback for a store that
    carries it without a ``fitted_voxels`` denominator (a third-party stamp),
    which would otherwise go unreported.
    """
    fitted_voxels = stats.get("fitted_voxels")
    if fitted_voxels and n_splats:
        return float(fitted_voxels) / n_splats
    stamped = stats.get("voxels_per_splat")
    return float(stamped) if stamped else None


def _print_source_grid(data: Any, stored_bytes: int) -> tuple[str, ...]:
    """Report what the splats are a representation of, when the fit recorded it.

    Silent for a dataset fitted before these stamps existed: the source grid is
    genuinely unknown there, and a compression ratio invented from the bounding
    box would be a guess presented as a measurement.

    ``stored_bytes`` is the size the caller already measured for its "Size:"
    line, handed over rather than re-measured: a directory store is sized by
    walking every chunk file, and the two numbers must be the same one anyway.

    Returns the ``stats`` keys this block has now reported —
    :data:`_SOURCE_GRID_STATS_KEYS` when it ran, empty when it bailed out. The
    caller suppresses exactly those from its catch-all metadata dump, so bailing
    out here leaves them to be printed there rather than dropping them.
    """
    stats = getattr(data, "stats", None) or {}
    shape = stats.get("source_shape")
    if not shape:
        return ()
    voxels = stats.get("source_voxels")
    dtype = stats.get("source_dtype")
    aprint(
        f"Source volume: {' x '.join(str(int(s)) for s in shape)}"
        + (f" {dtype}" if dtype else "")
        + (f" ({voxels:,} voxels)" if voxels else "")
        # A grid the producer STATED (because it preprocessed before fitting) is
        # not a grid measured from the array the fitter saw, and the compression
        # ratio printed below rests on it. Saying so beside the number is the
        # only place a reader of this report would look.
        + (" [declared by the producer]" if stats.get("source_declared") else "")
    )
    fitted = stats.get("fitted_shape")
    if fitted and list(fitted) != list(shape):
        aprint(
            "  fitted at:   "
            + " x ".join(str(int(s)) for s in fitted)
            + " (downscaled before fitting)"
        )
    occ = stats.get("occupancy")
    if occ is not None:
        aprint(
            f"  occupancy:   {100 * float(occ):.3f}% of voxels above "
            "1% of the intensity range"
        )
    n_splats = len(data.amplitudes) if data.amplitudes is not None else 0
    per_splat = _voxels_per_splat(stats, n_splats)
    if per_splat is not None:
        aprint(f"  voxels/splat: {per_splat:,.0f}")
    src_stored = stats.get("source_stored_bytes")
    if src_stored:
        aprint(f"  stored source: {format_memory_size(src_stored)} (as downloaded)")

    def _ratio(n: int, d: int) -> str:
        r = n / d
        # A whole-number ratio reads best, but a stored artifact LARGER than its
        # source is a real outcome (few voxels, many splats) and must not round
        # to a nonsensical "0:1".
        return f"{r:,.0f}" if r >= 10 else f"{r:.2g}"

    # TWO ratios, each labelled with its basis. `source_bytes` is the DECODED
    # array while the splat store on disk is compressed, so that ratio compares
    # unlike things and flatters the splats by whatever the source codec was
    # already achieving. It is still the number volumetric compression is
    # normally quoted against, so both are printed rather than either alone:
    # quoting only the first invites reading it as the second.
    src_bytes = stats.get("source_bytes")
    lines = []
    if src_bytes and stored_bytes:
        lines.append(
            f"{_ratio(src_bytes, stored_bytes)}:1 vs raw voxels "
            f"({format_memory_size(src_bytes)} -> {format_memory_size(stored_bytes)})"
        )
    if src_stored and stored_bytes:
        lines.append(
            f"{_ratio(src_stored, stored_bytes)}:1 vs the stored source "
            f"({format_memory_size(src_stored)} -> {format_memory_size(stored_bytes)})"
        )
    for i, line in enumerate(lines):
        aprint(f"  compression: {line}" if i == 0 else f"               {line}")
    return _SOURCE_GRID_STATS_KEYS
