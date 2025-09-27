"""
Visualization helpers for Gaussian splat fitting.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from arbol import aprint, asection


def display_compression_analysis(V: np.ndarray, params: np.ndarray, amps: np.ndarray) -> None:
    """
    Calculate and display compression ratio analysis.

    Compares the storage requirements of the original image vs the Gaussian splat representation.

    Parameters
    ----------
    V : np.ndarray
        Original input image/volume
    params : np.ndarray
        Fitted parameters [centers, packed_cholesky]
    amps : np.ndarray
        Fitted amplitudes
    """
    with asection("Compression Analysis"):
        # Original image storage (assuming float32)
        original_bytes = V.size * 4  # 4 bytes per float32
        original_bits = original_bytes * 8

        # Gaussian splat representation storage
        # params contains: centers (d floats) + packed L matrix (tril_size(d) floats)
        # amps contains: amplitudes (1 float per splat)
        n_splats = len(amps)
        d = len(V.shape)

        from luxar.gsplats.utils.trils import tril_size

        floats_per_splat = d + tril_size(d) + 1  # centers + covariance + amplitude
        splat_bytes = n_splats * floats_per_splat * 4  # 4 bytes per float32
        splat_bits = splat_bytes * 8

        # Calculate compression metrics
        compression_ratio = (
            original_bytes / splat_bytes if splat_bytes > 0 else float("inf")
        )
        compression_percent = (
            (1.0 - splat_bytes / original_bytes) * 100.0 if original_bytes > 0 else 0.0
        )
        bits_per_pixel = splat_bits / V.size

        aprint(f"Original image: {original_bytes:,} bytes ({original_bits:,} bits)")
        aprint(f"Splat representation: {splat_bytes:,} bytes ({splat_bits:,} bits)")
        aprint(f"Compression ratio: {compression_ratio:.2f}:1")
        aprint(f"Space savings: {compression_percent:.1f}%")
        aprint(f"Bits per pixel: {bits_per_pixel:.3f} (original: 32.000)")
        aprint(
            f"Storage efficiency: {n_splats} splats ({floats_per_splat} floats each)"
        )


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

        # Convert lists to 4D arrays (time, y, x) for 2D or (time, z, y, x) for 3D
        target_stack = np.array(movie_frames["target"])
        reconstruction_stack = np.array(movie_frames["reconstruction"])
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
            contrast_limits=[0, float(target_stack.max())],  # Same as target for fair comparison
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
        aprint("Close window to continue...")

        # Run napari - blocks until window is closed
        napari.run()

    except ImportError:
        aprint("⚠ napari not available for movie visualization")
    except Exception as e:
        aprint(f"⚠ Movie visualization error: {e}")
