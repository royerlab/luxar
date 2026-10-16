"""
Visualization helpers for Gaussian splat fitting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from arbol import aprint, asection


def _format_bytes(n_bytes: float) -> str:
    """Human-readable size, formatted exactly like ``cli.utils.format_memory_size``.

    Duplicated rather than imported: ``gsplats`` sits below ``cli`` in the
    import-linter layering contract.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024.0:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f} PB"


def compression_ratio(
    source_bytes: Optional[float], stored_bytes: Optional[float]
) -> Optional[float]:
    """THE compression ratio Luxar quotes: decoded source bytes / stored bytes.

    ``source_bytes`` is the source array's DECODED size in its own element type
    (``voxels x itemsize`` — the ``source_bytes`` fit stat), and ``stored_bytes``
    the bytes the ``.gsplats.zarr`` occupies ON DISK (``luxar.cli.utils.
    store_size``). ``gsplat info``, ``fit``, ``compare`` and ``cull`` all quote
    this one definition, so the same store gets the same number everywhere.
    Returns None when either side is unknown or non-positive.
    """
    if not source_bytes or not stored_bytes or source_bytes <= 0 or stored_bytes <= 0:
        return None
    return float(source_bytes) / float(stored_bytes)


def format_compression_ratio(ratio: float) -> str:
    """``"1,234:1"`` for a whole-number-sized ratio, ``"0.52:1"`` below 10.

    A stored artifact LARGER than its source (few voxels, many splats) is a real
    outcome and must not round to a nonsensical ``"0:1"``.
    """
    return f"{ratio:,.0f}:1" if ratio >= 10 else f"{ratio:.2g}:1"


def compression_lines(
    source_bytes: Optional[int],
    stored_bytes: Optional[int],
    source_stored_bytes: Optional[int] = None,
) -> List[str]:
    """The compression figures, each labelled with its basis.

    The primary figure is :func:`compression_ratio` (vs raw voxels). When the
    source's own STORED size is known (``source_stored_bytes``, e.g. a
    compressed download), a second line quotes the ratio against that too:
    the raw-voxel figure flatters the splats by whatever the source codec was
    already achieving, so quoting it alone invites reading it as the second.
    """
    lines: List[str] = []
    raw = compression_ratio(source_bytes, stored_bytes)
    if raw is not None:
        lines.append(
            f"{format_compression_ratio(raw)} vs raw voxels "
            f"({_format_bytes(float(source_bytes or 0))} -> "
            f"{_format_bytes(float(stored_bytes or 0))})"
        )
    vs_stored = compression_ratio(source_stored_bytes, stored_bytes)
    if vs_stored is not None:
        lines.append(
            f"{format_compression_ratio(vs_stored)} vs the stored source "
            f"({_format_bytes(float(source_stored_bytes or 0))} -> "
            f"{_format_bytes(float(stored_bytes or 0))})"
        )
    return lines


def display_compression_analysis(
    source_bytes: Optional[int],
    stored_bytes: Optional[int],
    *,
    n_splats: int,
    source_voxels: Optional[int] = None,
    source_stored_bytes: Optional[int] = None,
) -> Optional[float]:
    """Display the compression of a SAVED splat store against its source.

    Uses the single :func:`compression_ratio` definition (decoded source bytes
    over the store's bytes on disk), so the figure matches ``gsplat info``. It
    needs a written store: an in-memory estimate from float32 parameter counts
    is a different (and flattering-or-not, depending on the encoding) number,
    which is why the fitter no longer prints one before saving.

    Parameters
    ----------
    source_bytes : int or None
        Decoded source size (``voxels x itemsize``), e.g. ``stats["source_bytes"]``.
    stored_bytes : int or None
        Bytes the saved store occupies on disk.
    n_splats : int
        Splats in the saved store (after any post-fit cull).
    source_voxels : int, optional
        Voxel count of the source, for the bits-per-voxel line.
    source_stored_bytes : int, optional
        The source's own on-disk size, for the "vs the stored source" line.

    Returns
    -------
    float or None
        The compression ratio, or None when it is not computable.
    """
    with asection("Compression Analysis"):
        lines = compression_lines(source_bytes, stored_bytes, source_stored_bytes)
        if not lines:
            reason = (
                "source size not recorded"
                if not source_bytes
                else "store size unavailable"
            )
            aprint(f"Compression ratio: unknown ({reason})")
            return None
        for k, line in enumerate(lines):
            aprint(
                f"Compression ratio: {line}" if k == 0 else f"                   {line}"
            )
        if source_voxels and stored_bytes:
            src_bits = 8.0 * float(source_bytes or 0) / float(source_voxels)
            aprint(
                f"Bits per voxel: {8.0 * stored_bytes / source_voxels:.3f} "
                f"(source: {src_bits:.3f})"
            )
        aprint(f"Splats stored: {n_splats:,}")
        return compression_ratio(source_bytes, stored_bytes)


def show_optimization_movie(movie_frames: Dict[str, Any], shape: tuple) -> None:
    """
    Display napari viewer with optimization movie showing target, reconstruction, and residual over time.

    Parameters
    ----------
    movie_frames : dict
        Dictionary containing movie frame data
    shape : tuple
        Shape of the original data
    """
    try:
        import napari

        aprint("🎬 Creating optimization movie visualization...")

        # Convert to 4D arrays (time, y, x) for 2D or (time, z, y, x) for 3D
        # Target is stored once (not per-frame) — broadcast to match frame count
        reconstruction_stack = np.array(movie_frames["reconstruction"])
        target_single = movie_frames["target"]  # single array, not a list
        target_stack = np.broadcast_to(
            target_single[np.newaxis], reconstruction_stack.shape
        )
        residual_stack = np.array(movie_frames["residual"])
        splat_centers_list = movie_frames["splat_centers"]
        iterations = movie_frames["iterations"]

        # Create napari viewer with time series
        viewer = napari.Viewer(title=f"Optimization Movie ({len(iterations)} frames)")

        # Add image stacks as layers
        viewer.add_image(
            target_stack,
            name="Target",
            colormap="magma",
            contrast_limits=[0, float(target_stack.max())],
        )

        viewer.add_image(
            reconstruction_stack,
            name="Reconstruction",
            colormap="magma",
            contrast_limits=[
                0,
                float(target_stack.max()),
            ],  # Same as target for fair comparison
        )

        viewer.add_image(
            residual_stack,
            name="Residual",
            colormap="inferno",  # Better visibility than hot
            contrast_limits=[0, max(1e-12, float(residual_stack.max()))],
        )

        # Add splat centers as points that change over time
        # Create a stack of points data for napari (time, n_points, n_dims)
        # Pad all frames to have the same number of points (use max)
        max_splats = max(len(centers) for centers in splat_centers_list)
        d = len(shape)

        # Create padded points array: (n_frames, max_splats, d)
        points_stack = np.full((len(splat_centers_list), max_splats, d), np.nan)
        for i, centers in enumerate(splat_centers_list):
            n_centers = len(centers)
            if n_centers > 0:
                points_stack[i, :n_centers, :] = centers

        # # Add points layer (napari will handle NaN values automatically)
        # viewer.add_points(
        #     points_stack,
        #     name="Splat Centers",
        #     size=3,
        #     face_color="cyan",
        #     border_color="white",
        #     border_width=1,
        # )

        # Set up the time slider
        viewer.dims.axis_labels = ["iteration"] + [
            f"dim_{i}" for i in range(len(shape))
        ]

        # Add text overlay with movie information
        info_text = "Optimization Movie\n"
        info_text += f"Frames: {len(iterations)}\n"
        info_text += f"Iterations: {iterations[0]} → {iterations[-1]}\n"
        info_text += f"Shape: {shape}\n\n"
        info_text += "Use time slider to scrub through optimization\n"
        info_text += "Toggle layers to compare target/reconstruction/residual"

        viewer.text_overlay.text = info_text
        viewer.text_overlay.visible = True

        aprint(
            f"🎬 Movie ready: {len(iterations)} frames from iterations {iterations[0]} to {iterations[-1]}"
        )
        aprint("Use the time slider to scrub through optimization progress!")
        aprint("Toggle layer visibility to compare target/reconstruction/residual")
        aprint("Close napari window to continue...")

        # Run napari - blocks until window is closed
        # Note: Timing is shown BEFORE this call to avoid confusion from blocking
        napari.run()

    except ImportError:
        aprint("⚠ napari not available for movie visualization")
    except Exception as e:
        aprint(f"⚠ Movie visualization error: {e}")
