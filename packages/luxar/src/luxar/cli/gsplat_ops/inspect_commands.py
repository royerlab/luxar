"""``luxar gsplat inspect`` commands.

Each command is a plain function; ``register_inspect_commands(app)`` wires them onto
the shared ``app_gsplat`` Typer.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from arbol import aprint, asection

from .._traceback import exit_with_error, report_error
from ..common_options import CorsOriginOption, HostOption, make_port_option
from ..utils import (
    _DEFAULT_CORS_ORIGIN,
    format_memory_size,
    local_input_path,
    store_size,
)

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
    "psnr_robust_db",
    "psnr_db",
    "foreground_psnr_robust_db",
    "foreground_psnr_db",
    "foreground_threshold",
    "foreground_fraction",
    "ssim_robust",
    "ssim",
    "mse",
    # Amplitudes are background-relative, so these normalization stamps are
    # interpretation-critical rather than generic trailing metadata (#1175).
    "floor",
    "image_min",
    "image_max",
    "intensity_range",
    "voxel_size",
    "intensity_scale",
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


def _part_provenance_count(records: list[Any]) -> int:
    """Count collapsed summary parts, falling back to one per record."""
    count = 0
    for record in records:
        part_count = record.get("part_count") if isinstance(record, dict) else None
        if isinstance(part_count, int) and not isinstance(part_count, bool):
            count += max(1, part_count)
        else:
            count += 1
    return count


def _print_fitting_value(
    key: str, value: Any, *, show_full_provenance: bool = False
) -> None:
    if key in {"fit_provenance", "batch_provenance"} and not show_full_provenance:
        aprint(f"  {key}: recorded (use --provenance for a summary)")
        return
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
        part_count = _part_provenance_count(value)
        part_word = "part" if part_count == 1 else "parts"
        aprint(f"  {key}: {part_count} {part_word}{suffix}")
        return
    if isinstance(value, float):
        aprint(f"  {key}: {value:.6f}")
    else:
        aprint(f"  {key}: {value}")


def _fit_leaves(
    fitting: dict[str, Any], label: str = "fit"
) -> list[tuple[str, dict[str, Any]]]:
    """Walk component records without treating a merged score as a leaf fit."""
    parts = fitting.get("part_provenance")
    if not isinstance(parts, list):
        return [(label, fitting)]
    leaves: list[tuple[str, dict[str, Any]]] = []
    for part in parts:
        if not isinstance(part, dict) or not isinstance(part.get("fitting"), dict):
            continue
        name = part.get("slot_id", part.get("coordinate", "?"))
        leaves.extend(_fit_leaves(part["fitting"], f"{label}/{name}"))
    return leaves


def _provenance_stages(record: Any) -> dict[str, float]:
    import math

    timings = record.get("timings", {}) if isinstance(record, dict) else {}
    return {
        stage: float(value)
        for stage, value in timings.items()
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
    }


def _provenance_quality(outcome: dict[str, Any]) -> float | None:
    import math

    score = outcome.get("foreground_psnr_db", outcome.get("psnr_db"))
    if isinstance(score, (int, float)) and math.isfinite(score):
        return float(score)
    return None


def _maybe_print_provenance(fitting: dict[str, Any], enabled: bool) -> None:
    if enabled:
        _print_provenance_summary(fitting)


def _print_batch_stages(batch: Any) -> None:
    if isinstance(batch, dict):
        for stage, seconds in batch.get("timings", {}).items():
            aprint(f"  batch {stage}: {float(seconds):.2f} s")


def _print_provenance_summary(fitting: dict[str, Any]) -> None:
    """Readable timings and outcome spread from old or versioned fit records."""
    import statistics

    leaves = _fit_leaves(fitting)
    if not leaves:
        aprint("\nFit provenance: no fitted parts")
        return
    stage_totals: dict[str, float] = {}
    durations: list[tuple[float, str]] = []
    stop_reasons: Counter[str] = Counter()
    cull_total = 0
    prune_total = 0
    quality: list[float] = []
    for label, leaf in leaves:
        record = leaf.get("fit_provenance", {})
        outcome = record.get("outcome", leaf) if isinstance(record, dict) else leaf
        timings = _provenance_stages(record)
        duration = sum(timings.values())
        for stage, value in timings.items():
            stage_totals[stage] = stage_totals.get(stage, 0.0) + value
        if duration == 0:
            duration = float(leaf.get("time_seconds") or 0)
        durations.append((duration, label))
        stop_reasons[str(outcome.get("early_stop_reason") or "iteration limit")] += 1
        cull_total += int(outcome.get("n_culled") or 0)
        prune_total += int(outcome.get("min_extent_removed") or 0)
        score = _provenance_quality(outcome)
        if score is not None:
            quality.append(score)
    aprint(f"\nFit provenance: {len(leaves)} fit(s)")
    aprint(f"  Fit task time (sum): {sum(d for d, _ in durations):.2f} s")
    for stage, seconds in sorted(stage_totals.items()):
        aprint(f"  {stage}: {seconds:.2f} s")
    if isinstance(fitting.get("save_seconds"), (int, float)):
        aprint(f"  saving: {float(fitting['save_seconds']):.2f} s")
    _print_batch_stages(fitting.get("batch_provenance"))
    if len(durations) > 1:
        aprint(
            "  Slowest: "
            + ", ".join(
                f"{label} {seconds:.2f}s"
                for seconds, label in sorted(durations, reverse=True)[:5]
            )
        )
    aprint(
        "  Stop reasons: "
        + ", ".join(
            f"{reason} {count}" for reason, count in sorted(stop_reasons.items())
        )
    )
    aprint(f"  Culled: {cull_total}; min-extent pruned: {prune_total}")
    if quality:
        aprint(
            "  Quality dB (min/median/max): "
            f"{min(quality):.2f}/{statistics.median(quality):.2f}/{max(quality):.2f}"
        )


def _is_short_numeric_list(value: Any) -> bool:
    """Whether a metadata list is compact and readable without summarizing."""
    return (
        isinstance(value, list)
        and len(value) <= 8
        and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        )
    )


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
                elif _is_short_numeric_list(value):
                    _print_fitting_value(key, value)
                elif isinstance(value, (dict, list)):
                    aprint(f"  {key}: {type(value).__name__} with {len(value)} items")
                else:
                    aprint(f"  {key}: {value}")


def _is_node_tree(path: Path) -> bool:
    """Whether an already-resolved store has no flat ``GSplatData`` representation.

    Partitions and nested lod groups are trees; a lod whose children are all
    leaves is matrix-shaped and must keep the normal flat info report. Returns
    False for anything unreadable so the normal load path reports its error.
    """
    from luxar._zarr_compat import open_group as zarr_open_group

    try:
        root = zarr_open_group(str(path), mode="r")
        kind = root.attrs.get("kind")
        if kind == "partition":
            return True
        if kind != "lod":
            return False
        return any(
            root[name].attrs.get("kind") is not None
            for name in root
            if str(name).startswith("child_")
        )
    except Exception:
        return False


#: Inputs `gsplat fit` accepts that are IMAGES, not splat stores.
_IMAGE_VOLUME_SUFFIXES = (".tif", ".tiff", ".npy", ".npz")
#: Root attributes a `.gsplats.zarr` (any format version) carries.
_GSPLAT_ROOT_ATTRS = ("format_type", "format_version", "kind", "type")


def _image_volume_reason(path: Path) -> Optional[str]:
    """Why ``path`` looks like an image volume rather than a splat store, or None.

    ``gsplat info`` is the first thing people point at a raw stack, and the
    loader's own error for that ("missing centers", a zarr KeyError) never says
    the useful thing: fit it first. Conservative on purpose — anything carrying
    a gsplats root attribute or ``centers``/``amplitudes`` arrays is left to the
    normal loader (and its own error).
    """
    from luxar._zarr_compat import read_array_meta, read_node_attrs

    name = path.name.lower()
    for suffix in _IMAGE_VOLUME_SUFFIXES:
        if name.endswith(suffix):
            return f"a {suffix} file"
    if not path.is_dir():
        return None
    if read_array_meta(path) is not None:
        return "a bare zarr array"
    attrs = read_node_attrs(path)
    if attrs is None or any(key in attrs for key in _GSPLAT_ROOT_ATTRS):
        return None
    if "multiscales" in attrs or "ome" in attrs:
        return "an OME-Zarr image"
    if (path / "centers").exists() or (path / "amplitudes").exists():
        return None
    return "a zarr store with no Gaussian-splat structure"


def _report_image_volume(path: Path, reason: str) -> None:
    aprint(
        f"❌ {path.name} is an image volume ({reason}), not a .gsplats.zarr — "
        f"fit it first with `luxar gsplat fit {path.name} "
        f"<output>.gsplats.zarr`, then run `gsplat info` on the output."
    )


def _load_info_data(
    path: Path, *, full_provenance: bool, provenance: bool = False
) -> tuple["Optional[GSplatData]", bool]:
    """Load a flat dataset, or print a valid tree summary or load failure."""
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io._archive import resolve_store_path

    with asection(f"Loading dataset: {path.name}"):
        zarr_path = path
        tmp = None
        reason = _image_volume_reason(path)
        if reason is not None:
            _report_image_volume(path, reason)
            return None, False
        try:
            if path.is_file():
                from tarfile import TarError
                from zipfile import BadZipFile

                try:
                    zarr_path, tmp = resolve_store_path(path)
                except (
                    ValueError,
                    OSError,
                    EOFError,
                    BadZipFile,
                    TarError,
                ) as resolve_exc:
                    aprint(f"❌ {resolve_exc}")
                    return None, False

            if zarr_path != path:
                reason = _image_volume_reason(zarr_path)
                if reason is not None:
                    _report_image_volume(path, reason)
                    return None, False

            # Ask the store what shape it is BEFORE loading anything. A partition
            # or nested lod tree has no flat GSplatData form, so the flat load below
            # would decode every array of every leaf and then raise -- and `info`
            # would throw all of it away. Reading one root attribute costs nothing
            # and is the difference between 0 and ~115 MB of decode on a 6.4 MB
            # store (audit A14-02).
            if _is_node_tree(zarr_path):
                try:
                    _print_gsplat_tree_summary(
                        zarr_path,
                        show_full_provenance=full_provenance,
                        show_provenance=provenance,
                        source_path=path,
                    )
                except ValueError as summary_exc:
                    aprint(f"❌ {summary_exc}")
                    return None, False
                return None, True

            try:
                data = GSplatData.load(zarr_path, include_stats=True)
            except ValueError as load_exc:
                # Not a tree and not loadable: a legacy or invalid file the v3.0
                # reader rejects.
                aprint(f"❌ {load_exc}")
                return None, False

            n_splats = len(data.amplitudes)
            ndim = data.centers.shape[1]
            aprint(f"✓ Loaded {n_splats:,} splats ({ndim}D)")
            return data, True
        finally:
            if tmp is not None and tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)


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


def _print_categorical_label_information(data: "GSplatData") -> None:
    """Print categorical vocabulary size when labels are present."""
    if data.label_ids is None:
        return
    aprint(f"Categorical labels: {len(data.label_vocabulary or {})} vocabulary entries")


def _info_report(
    path: Path,
    *,
    show_histograms: bool,
    bins: int,
    full_provenance: bool,
    provenance: bool = False,
) -> bool:
    """Print a Gaussian splat dataset report and return whether it succeeded."""
    try:
        import numpy as np

        from luxar.gsplats.utils.alpha import effective_amplitudes

        data, report_succeeded = _load_info_data(
            path, full_provenance=full_provenance, provenance=provenance
        )
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

        stored_bytes = store_size(path)
        aprint(f"\nFile: {path.name}")
        aprint(f"Size: {format_memory_size(stored_bytes)}")

        aprint(f"\nSplats: {n_splats:,}")
        aprint(f"Dimensions: {ndim}D")
        aprint(f"Has Colors: {'Yes' if data.colors is not None else 'No'}")
        _print_categorical_label_information(data)
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
            _maybe_print_provenance(data.stats, provenance)

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
        report_error(f"❌ Error: {e}", e)
        return False


def info_dataset(
    path: Path = typer.Argument(
        ...,
        callback=local_input_path,
        help="Path to .gsplats.zarr dataset (or .zip/.tar.gz)",
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
    provenance: bool = typer.Option(
        False, "--provenance", help="Summarize fit stages, slow tiles, and outcomes"
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
        provenance=provenance,
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
        exit_with_error(f"❌ Error opening dataset: {e}", e)


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


def _build_view_scene(store: Path, opts: Any) -> tuple[Path, Path]:
    """Build ``gsplat convert``'s scene for ``store`` in a fresh temp directory.

    Returns ``(temp_dir, scene_path)``; the caller removes ``temp_dir``. The
    directory is created BEFORE the build, so a build that fails part-way still
    leaves the caller something to clean up — hence the cleanup here on error.
    """
    import tempfile
    import time

    from .scene_build import build_gsplat_scene

    scene_dir = Path(tempfile.mkdtemp(prefix="luxar_gsplat_view_"))
    scene_path = scene_dir / f"{store.name.split('.')[0] or 'gsplats'}.luxar.zarr"
    start = time.perf_counter()
    try:
        with asection("Building a temporary scene (as `gsplat convert` would)"):
            build_gsplat_scene(store, scene_path, opts)
    except BaseException:
        shutil.rmtree(scene_dir, ignore_errors=True)
        raise
    aprint(
        f"Built a temporary scene in {time.perf_counter() - start:.1f}s "
        f"(removed on exit; --raw serves the node tree as is)"
    )
    return scene_dir, scene_path


def quick_view(
    path: Path = typer.Argument(
        ..., exists=True, help="Path to .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    host: HostOption = "127.0.0.1",
    port: int = make_port_option(8000, "Data server port"),
    viewer_port: int = typer.Option(5173, "--viewer-port", help="Viewer port"),
    open_browser: Optional[bool] = typer.Option(
        None,
        "--open/--no-open",
        help="Open a browser (default: yes, where one can be seen).",
    ),
    cors_origin: CorsOriginOption = _DEFAULT_CORS_ORIGIN,
    raw: bool = typer.Option(
        False,
        "--raw",
        help="Serve the .gsplats.zarr node tree as is (no temporary scene: no "
        "image orientation, no auto exposure, no display gain and no unit "
        "labels; amplitudes as stored).",
    ),
    auto_exposure: bool = typer.Option(
        True,
        "--auto-exposure/--no-auto-exposure",
        help="As for `gsplat convert`: dim splat brightness so the opening view "
        "is not saturated (side views and other timepoints capped at 4x the "
        "target).",
    ),
    target_brightness: Optional[float] = typer.Option(
        None,
        "--target-brightness",
        min=1e-6,
        help="Auto-exposure target for the predicted projected p99.9 "
        "(default 2.0, the ACES value).",
    ),
    dim_order: Optional[str] = typer.Option(
        None,
        "--dim-order",
        help="Name each center column (e.g. 'z,y,x'), as for `gsplat convert`.",
    ),
    view: Optional[str] = typer.Option(
        None, "--view", help="Opening view: xy (default for image fits), xz, or yz."
    ),
    colormap: Optional[str] = typer.Option(
        None, "--colormap", help="Colormap for scalar amplitudes (default gray)."
    ),
) -> None:
    """Quick view of a Gaussian splat dataset in the Luxar web viewer.

    By default the store is turned into a TEMPORARY scene through exactly the
    ``gsplat convert`` code path — image-convention orientation and opening
    camera, auto exposure, the same appearance defaults — written to a temp
    directory that is removed on exit, and that scene is served. So the first
    look matches what ``convert`` would publish. The build re-writes the splats
    (a partition / nested tree is grafted node for node, which still copies its
    arrays), so it costs a few seconds on a large store.

    ``--raw`` keeps the old behaviour: the standalone ``.gsplats.zarr`` is a v3.4
    node subtree the viewer renders directly (``?src=``), with no scene-compile
    round-trip — every tree shape works, framed on the file's
    ``position_bounds`` — but amplitudes are the stored (raw) ones and the axes
    are positional.

    Compressed archives (``.gsplats.zarr.zip`` / ``.gsplats.zarr.tar.gz``) are
    extracted to a temporary directory before serving.

    Like ``luxar serve``, both servers bind loopback unless ``--host`` says
    otherwise (``0.0.0.0`` to view from another machine — the printed and
    opened URLs then name this machine's LAN address).

    Args:
        path: Path to .gsplats.zarr (or .zip/.tar.gz) dataset
        host: Bind address for both servers (default loopback).
        port: Port for data server
        viewer_port: Port for viewer
        open_browser: Whether to open browser automatically
        cors_origin: Allowed CORS origin for both servers (default "local").
        raw: Serve the node tree directly instead of a temporary scene.
        auto_exposure: Auto exposure for the temporary scene.
        target_brightness: Auto-exposure target.
        dim_order: Column names, as for ``gsplat convert``.
        view: Opening view preset, as for ``gsplat convert``.
        colormap: Colormap for scalar amplitudes.
    """
    # Bound before the `try` so the `finally` below can test it: the store
    # resolver answers None for a directory store, and `rmtree(None)` raises —
    # turning a real error into an unrelated one.
    temp_dir: Optional[Path] = None
    scene_dir: Optional[Path] = None
    # SIGTERM / SIGHUP (a harness timeout, `luxar stop`, a closed terminal —
    # also mid-build) unwind through the `finally` below instead of killing the
    # process, so the temp scene and any extraction are always removed.
    from contextlib import ExitStack

    from luxar.cli.demo_runs import exit_on_termination_signals

    signals = ExitStack()
    signals.enter_context(exit_on_termination_signals())
    try:
        import threading

        from luxar.cli import demo_runs
        from luxar.cli.serving import _serve_data, _serve_viewer, _warn_if_lan_exposed
        from luxar.cli.utils import (
            advertised_host,
            dataset_title,
            ensure_viewer_built,
            pick_port,
            wait_for_server,
        )

        _warn_if_lan_exposed(host, cors_origin)

        # Check viewer is built
        if not ensure_viewer_built():
            raise typer.Exit(1)

        with asection(f"Quick View: {path.name}"):
            # Resolve to an on-disk .gsplats.zarr directory: extract archives to
            # a temp dir, otherwise serve the directory in place. No GSplatData
            # round-trip — the viewer consumes the node tree directly, which is
            # the only path that supports partition/nested roots.
            serve_target, temp_dir = _resolve_view_target(path)
            if raw:
                aprint(f"Serving node tree directly: {serve_target.name}")
            else:
                from .scene_build import SceneBuildOptions

                scene_dir, serve_target = _build_view_scene(
                    serve_target,
                    SceneBuildOptions(
                        auto_exposure=auto_exposure,
                        target_brightness=target_brightness,
                        dim_order=dim_order,
                        view=view,
                        colormap=colormap,
                    ),
                )

            # Find available ports
            actual_port = pick_port(port, host, label="data")
            actual_viewer_port = pick_port(viewer_port, host, label="viewer")

            if actual_port is None or actual_viewer_port is None:
                raise typer.Exit(1)

            with asection("Starting servers"):
                # Start data server in background
                data_thread = threading.Thread(
                    target=_serve_data,
                    args=(
                        serve_target,
                        host,
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
                if not wait_for_server(host, actual_port, data_thread):
                    aprint("⚠️  Data server did not become ready.")

                # Construct data URL (no trailing slash — see CLAUDE.md gotcha).
                # The store is mounted at the server root: no name suffix. A
                # wildcard bind is not dialable, hence `advertised_host`.
                data_url = f"http://{advertised_host(host)}:{actual_port}"

                # Serve viewer (this blocks)
                aprint("\n🎉 Viewer ready!")
                with demo_runs.registered_server(
                    f"gsplat view {path.name}", "gsplat view"
                ):
                    _serve_viewer(
                        host,
                        actual_viewer_port,
                        data_url,
                        open_browser is not False,
                        cors_origin,
                        title=dataset_title(path),
                        open_explicit=open_browser is True,
                    )

    except KeyboardInterrupt:
        aprint("\n🛑 Shutting down viewer...")
    except typer.Exit:
        raise
    except Exception as e:
        exit_with_error(f"❌ Error: {e}", e)
    finally:
        # Every exit removes the extraction, not just the two error handlers
        # that used to: a NORMAL return from `_serve_viewer`, and the
        # `typer.Exit` a failed `pick_port` raises AFTER the archive has been
        # extracted, both left a full uncompressed copy of the dataset behind
        # in /tmp.
        for scratch in (temp_dir, scene_dir):
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)
        signals.close()


def compare_quality(
    gsplats_path: Path = typer.Argument(
        ...,
        callback=local_input_path,
        help="Path to .gsplats.zarr dataset (or .zip/.tar.gz)",
    ),
    reference_path: Path = typer.Argument(
        ...,
        callback=local_input_path,
        help="Reference volume (.npy, .tiff, .zarr, etc.)",
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
    voxel_size: Optional[str] = typer.Option(
        None,
        "--voxel-size",
        help="Spacing the splat centers are scaled by, one value or one per axis "
        "(e.g. '2.0,0.4,0.4'). Normally read from the dataset's voxel_size stamp, "
        "which a physical-output fit records; pass it for a physical store "
        "without the stamp, or '1' to disable the mapping.",
    ),
    at_fit_resolution: bool = typer.Option(
        False,
        "--at-fit-resolution",
        help="Score a --downscale fit on the grid it was fitted on: the reference "
        "is decimated the same way (anti-aliased, strided) and the splats mapped "
        "onto that grid. Default: score at the source resolution.",
    ),
    whole_volume: bool = typer.Option(
        False,
        "--whole-volume",
        help="Score a --region crop fit over the WHOLE reference. By default a "
        "crop fit compared against its full source volume is scored on the "
        "region it fitted (fitting/source_region), so the foreground mask "
        "matches the one the fit stamped; outside the region every structure "
        "counts as error.",
    ),
    mask: Optional[Path] = typer.Option(
        None,
        "--mask",
        exists=True,
        help="Foreground mask volume, same shape as the reference (voxels > 0 are "
        "foreground). Overrides the default smoothed-Otsu foreground.",
    ),
    fg_threshold: Optional[float] = typer.Option(
        None,
        "--fg-threshold",
        help="Foreground = reference > this value, in the reference's own "
        "intensity units. Overrides the default smoothed-Otsu foreground.",
    ),
    channel: Optional[int] = typer.Option(
        None,
        "--channel",
        "-c",
        help="Channel index (zarr/OME-Zarr reference, or any format with --axes)",
    ),
    timepoint: Optional[int] = typer.Option(
        None,
        "--timepoint",
        help="Timepoint index (zarr/OME-Zarr reference, or any format with --axes)",
    ),
    array_key: Optional[str] = typer.Option(
        None, "--array-key", help="Array key within .npz or .zarr"
    ),
    axes: Optional[str] = typer.Option(
        None,
        "--axes",
        help="Per-dimension axis labels overriding the positional "
        "TCZYX/CZYX/ZYX heuristic, e.g. 'z,c,y,x' or 't,z,y,x' — the same "
        "selection `gsplat fit` used. Required to pick a --channel/--timepoint "
        "out of a TIFF or NumPy file.",
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

    Renders the gsplats back onto the reference's voxel grid and computes PSNR,
    SSIM, MSE, relative L2 error and maximum absolute error, globally and over
    the foreground. The headline PSNR and SSIM use a robust peak (reference
    p99.99 minus the fit's floor), which one hot pixel cannot move; the
    classical full-range figures are reported beside them. A physical-output fit is mapped back to voxels
    with its stored voxel_size. Partition and nested stores are scored over
    their default-rendered selection: all parts and each LOD group's finest
    level. The compression ratio covers the whole store, including coarse
    levels that are not scored. A --region crop fit is scored on the region it
    fitted (--whole-volume for all of the reference). Beside the foreground
    PSNR it prints the PSNR a noise-free model would score against this noisy
    reference (from the noise sigma measured in the foreground) and a verdict
    on the gap: an estimate that fine real texture can fool, not a pass mark.

    Examples:
        luxar gsplat compare fitted.gsplats.zarr original.tiff
        luxar gsplat compare fitted.gsplats.zarr original.npy --device cuda
        luxar gsplat compare fitted.gsplats.zarr stack.tif --axes c,z,y,x -c 1
        luxar gsplat compare fitted.gsplats.zarr original.zarr --mask cells.tif
        luxar gsplat compare fitted.gsplats.zarr original.zarr -j metrics.json -q
    """
    from .compare_ops import CompareOptions, run_compare

    run_compare(
        CompareOptions(
            gsplats_path=gsplats_path,
            reference_path=reference_path,
            shape=shape,
            device=device,
            truncate=truncate,
            image_min=image_min,
            channel=channel,
            timepoint=timepoint,
            array_key=array_key,
            axes=axes,
            mask=mask,
            fg_threshold=fg_threshold,
            voxel_size=voxel_size,
            at_fit_resolution=at_fit_resolution,
            whole_volume=whole_volume,
            output_json=output_json,
            quiet=quiet,
        )
    )


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
    path: Path,
    *,
    show_full_provenance: bool = False,
    show_provenance: bool = False,
    source_path: Optional[Path] = None,
) -> None:
    """Report the node-tree shape of a partition / nested .gsplats.zarr.

    These have no flat ``GSplatData`` (``gsplat info``'s normal path), so we
    walk the node tree and print its structure (kind, parts/levels, per-leaf
    splat counts, total, ndim, bounds) instead of failing.
    """
    from luxar._zarr_compat import open_group as zarr_open_group
    from luxar.gsplats.io._archive import resolve_store_path
    from luxar.gsplats.io.load_gsplats import read_gsplat_root_stats
    from luxar.gsplats.io.tree_summary import read_gsplat_tree_summary

    zarr_path = path
    display_path = source_path or path
    tmp = None
    try:
        if path.is_file():  # compressed archive
            zarr_path, tmp = resolve_store_path(path)
        root = zarr_open_group(str(zarr_path), mode="r")
        read_gsplat_root_stats(root, include_stats=False)
        # Metadata only: attrs plus array SHAPES, never a chunk. Raises
        # ValueError for a group that is not a gsplat node, which is the same
        # contract the full reader had -- so this doubles as the validity gate
        # that used to cost a whole extra decode of the tree.
        summary = read_gsplat_tree_summary(root)

        aprint("\n" + "═" * 70)
        aprint("DATASET INFORMATION (node tree)")
        aprint("═" * 70)
        aprint(f"\nFile: {display_path.name}")
        # Same measurement as the flat report's "Size:" line — a directory store's
        # own stat() is the ~4 KB directory entry, not the chunks in it, and a
        # partition is the shape most likely to BE a directory.
        aprint(f"Size: {format_memory_size(store_size(display_path))}")
        aprint(f"\nRoot kind: {summary.kind}")
        aprint(f"Dimensions: {summary.ndim}D")
        aprint(f"Total splats (all leaves): {summary.n_splats:,}")
        if summary.kind in ("partition", "lod"):
            child_word = "part" if summary.kind == "partition" else "level"
            aprint(f"{child_word.capitalize()}s: {len(summary.children)}")
            for i, leaf_count in enumerate(summary.leaf_counts):
                aprint(f"  leaf {i}: {leaf_count:,} splats")
        aprint(
            "Full load would materialise: "
            f"{format_memory_size(summary.estimated_decoded_bytes)} (estimated)"
        )
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
            _maybe_print_provenance(fitting, show_provenance)
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
    aprint(f"{mark} [{finding.path or '/'}] {finding.summary}")
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
        exit_with_error(f"❌ {exc}", exc)


def _print_doctor_info(
    path: Path,
    store_kind: "StoreKind",
    *,
    histograms: bool,
    bins: int,
    full_provenance: bool,
) -> bool:
    """Print the optional report appropriate for a doctor input."""
    if store_kind == "gsplats":
        return _info_report(
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
    return True


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
        help="Number of bins for the info report's histograms; only applies "
        "with --histograms.",
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

    - Unsupported standalone gsplat format versions. These stores cannot be
      opened by the current reader and must be converted with `migrate-format`.
    - Supported-version standalone gsplat stores with an incomplete tree or
      inconsistent leaf-array metadata. Restore or regenerate these stores.
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
        report_succeeded = _print_doctor_info(
            path,
            store_kind,
            histograms=histograms,
            bins=bins,
            full_provenance=full_provenance,
        )
        if not report_succeeded:
            aprint("ℹ️ Info report failed; continuing to the diagnosis.")

    with asection(f"Diagnosing: {path.name}"):
        try:
            report = diagnose_store(path, fix=fix)
        except typer.Exit:
            raise
        except Exception as exc:
            exit_with_error(f"❌ {exc}", exc)

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

    # TWO ratios, each labelled with its basis (one shared definition, see
    # `compression_ratio`): `source_bytes` is the DECODED array while the splat
    # store on disk is compressed, so the first compares unlike things and
    # flatters the splats by whatever the source codec was already achieving.
    # It is still the number volumetric compression is normally quoted
    # against, so both are printed rather than either alone.
    from luxar.gsplats.fitting.visualization import compression_lines

    lines = compression_lines(stats.get("source_bytes"), stored_bytes, src_stored)
    for i, line in enumerate(lines):
        aprint(f"  compression: {line}" if i == 0 else f"               {line}")
    return _SOURCE_GRID_STATS_KEYS
