#!/usr/bin/env python3
"""GSplats Demo: 4D Multi-Channel Cells (scikit-image cells3d)

Demonstrates the ``dim_order`` feature: fitting 3D Gaussian splats per channel,
then embedding them into a 4D scene with a categorical Channel dimension.

================================================================================
DIM_ORDER SHOWCASE — 3D SPLATS IN A 4D SCENE
================================================================================

This demo uses the cells3d dataset from scikit-image — a two-channel 3D
fluorescence microscopy volume of cells (membranes + nuclei).

The key feature demonstrated here is ``dim_order``: each channel is fitted
independently as 3D splats, then added to a 4D scene (X, Y, Z, Channel) using
``dim_order=["z", "y", "x"]`` to map the 3D data columns to the correct scene
dimensions. The Channel dimension is filled with a fixed value per channel.

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source: scikit-image sample data (``skimage.data.cells3d()``)
Shape: (60, 2, 256, 256) — (Z, Channel, Y, X), uint16
Channel 0: Cell membranes
Channel 1: Cell nuclei (fluorescent stain)
Origin: Allen Institute for Cell Science

How to Cite:
------------
scikit-image: image processing in Python.
van der Walt et al. (2014). PeerJ 2:e453. DOI: 10.7717/peerj.453

WORKFLOW:
=========

1. **Load** cells3d from scikit-image (60 × 2 × 256 × 256)
2. **Fit** each channel independently as 3D Gaussian splats
3. **Create 4D scene** with dimensions [X, Y, Z, Channel]
4. **Add splats** using ``dim_order=["z", "y", "x"]`` with per-channel colors
5. **Visualize** — Channel slider toggles between membrane and nuclear views

USAGE:
======
    python demo_gsplats_4d_cells3d_multichannel.py [--no-cache] [--no-serve] [--serve-only]

Options:
    --no-cache:   Force re-fitting (ignore cached results)
    --no-serve:   Don't auto-launch viewer after scene creation
    --serve-only: Skip loading/fitting, just serve existing scene

Output:
    - Scene saved to: demos/gsplats_4d_cells3d_multichannel.zarr
    - Automatically opens in browser at http://localhost:8000

"""

import sys
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.demos import launch_viewer, warn_if_no_cuda_gpu
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Fitting parameters
N_SEEDS = 15000  # Splats per channel
N_ITERS = 8000  # Optimization iterations

# Voxel spacing (Z, Y, X) in micrometres for cells3d
# Original: (0.29, 0.065, 0.065) µm, 4x downsampled in Y/X → (0.29, 0.26, 0.26) µm
VOXEL_SIZE_ZYX = (0.29, 0.26, 0.26)

# Channel configuration
CHANNELS = [
    {"index": 0, "name": "Membranes", "color": (0.0, 1.0, 0.3)},  # Green
    {"index": 1, "name": "Nuclei", "color": (0.5, 0.3, 1.0)},  # Purple
]

# Cache directory
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_cells3d"

# Parse command-line flags
NO_CACHE = "--no-cache" in sys.argv
NO_SERVE = "--no-serve" in sys.argv
SERVE_ONLY = "--serve-only" in sys.argv

# Setup
Arbol.max_depth = 5

# Global device (auto-detected on first fit)
DEVICE = None


# =============================================================================
# Data Loading
# =============================================================================


def load_cells3d():
    """Load cells3d dataset from scikit-image.

    Returns:
        list[np.ndarray]: One 3D volume per channel, shape (Z, Y, X), float32 [0, 1].
    """
    with asection("Loading cells3d dataset"):
        try:
            from skimage.data import cells3d
        except ImportError:
            raise ImportError(
                "scikit-image is required for this demo.\n"
                "Install with: pip install scikit-image"
            )

        # cells3d() returns (60, 2, 256, 256) — (Z, Channel, Y, X), uint16
        raw = cells3d()
        aprint(f"Raw data shape: {raw.shape}, dtype: {raw.dtype}")
        aprint(
            f"  Axes: (Z={raw.shape[0]}, C={raw.shape[1]}, Y={raw.shape[2]}, X={raw.shape[3]})"
        )

        volumes = []
        for ch_config in CHANNELS:
            ch_idx = ch_config["index"]
            ch_name = ch_config["name"]

            V = raw[:, ch_idx, :, :].astype(np.float32)
            # Normalize to [0, 1]
            V = (V - V.min()) / (V.max() - V.min() + 1e-8)

            volumes.append(V)
            aprint(f"  {ch_name}: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

        aprint(f"Loaded {len(volumes)} channels")
        return volumes


# =============================================================================
# GSplats Fitting
# =============================================================================


def fit_channel(volume, channel_name, cache_file):
    """Fit gsplats to a single channel, using cache if available.

    Args:
        volume: 3D volume (Z, Y, X), float32 [0, 1]
        channel_name: Human-readable channel name
        cache_file: Path to .gsplats.zarr.zip cache file

    Returns:
        GSplatData with fitted 3D splats
    """
    # Check cache
    if cache_file.exists() and not NO_CACHE:
        aprint(f"Loading cached fit for {channel_name}")
        try:
            result = GSplatData.load(cache_file, include_stats=False)
            aprint(f"  Loaded {len(result.amplitudes):,} cached splats")
            return result
        except Exception as e:
            aprint(f"  Cache load failed: {e}, re-fitting...")

    # Auto-detect device
    global DEVICE
    if DEVICE is None:
        from luxar.utils.demos import detect_device

        DEVICE = detect_device()

    from luxar.gsplats import fit_gaussian_splats

    aprint(f"Fitting {channel_name} ({N_ITERS} iterations, {N_SEEDS} seeds)...")

    result = fit_gaussian_splats(
        volume,
        seeds=N_SEEDS,
        n_iters=N_ITERS,
        device=DEVICE,
        verbose=True,
        enable_dynamic_ops=True,
        voxel_size=VOXEL_SIZE_ZYX,
    )

    n_splats = len(result.amplitudes)
    aprint(f"  Fitted {n_splats:,} splats")

    # Cache result in compressed zarr format
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    aprint(f"  Caching to {cache_file.name}")
    result.save(
        cache_file,
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
    )

    return result


def fit_all_channels(volumes):
    """Fit gsplats to all channels."""
    with asection("Fitting GSplats per channel"):
        gsplats_list = []

        for i, (volume, ch_config) in enumerate(zip(volumes, CHANNELS)):
            ch_name = ch_config["name"]
            cache_file = CACHE_DIR / f"cells3d_ch{i}.gsplats.zarr.zip"

            with asection(f"Channel {i}: {ch_name}"):
                gsplats = fit_channel(volume, ch_name, cache_file)
                gsplats_list.append(gsplats)

        return gsplats_list


# =============================================================================
# Scene Creation — dim_order showcase
# =============================================================================


def create_luxar_scene(gsplats_list, output_path=None):
    """Create 4D Luxar scene using dim_order to embed 3D splats.

    This is the key part of the demo: each channel was fitted as 3D (Z, Y, X),
    but the scene has 4 dimensions (X, Y, Z, Channel). We use ``dim_order`` to
    map the 3D data columns to the correct scene dimensions and ``fill`` to
    assign each channel's splats to their Channel index.
    """
    if output_path is None:
        output_path = get_demos_output_dir() / "gsplats_4d_cells3d_multichannel.zarr"

    with asection("Creating 4D Luxar Scene"):
        aprint(f"Output: {output_path.name}")

        # Define 4D scene: 3 spatial + 1 categorical Channel dimension
        # Spatial ranges are NOT set — the compiler infers them from
        # actual data (which is centroid-centered, not in voxel coords).
        dims = Dimensions(
            [
                Dimension("x", unit="px", display=True),
                Dimension("y", unit="px", display=True),
                Dimension("z", unit="px", display=True),
                Dimension(
                    "channel",
                    display=False,
                    discrete=True,
                    range=(0, len(CHANNELS) - 1),
                    step=1.0,
                    categories=[ch["name"] for ch in CHANNELS],
                ),
            ]
        )

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
            )

            scene.attrs["title"] = "GSplats: 4D Cells (dim_order demo)"
            scene.attrs["description"] = """
4D Multi-Channel Gaussian Splatting — cells3d (scikit-image)
============================================================

Demonstrates the dim_order feature: 3D Gaussian splats fitted per channel,
embedded into a 4D scene with a categorical Channel dimension.

Data Source:
  - scikit-image cells3d sample dataset
  - Allen Institute for Cell Science
  - Shape: (60, 2, 256, 256) — (Z, Channel, Y, X)

Channels:
  - Green: Cell membranes (Channel 0)
  - Purple: Cell nuclei (Channel 1)

Navigation:
  - Use the Channel slider to switch between membrane and nuclear views
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan

dim_order usage:
  Each channel was fitted as 3D splats (Z, Y, X), then added to this
  4D scene using dim_order=["z", "y", "x"] with fill={"channel": i}.
  The Cholesky covariance factors are automatically embedded from 3D to 4D.
            """

            # Compute shared centroid across ALL channels so they stay aligned
            all_centers = [g.centers for g in gsplats_list]
            all_amps = [g.amplitudes for g in gsplats_list]
            total_amp = sum(a.sum() for a in all_amps)
            if total_amp > 0:
                shared_centroid = (
                    sum(c.T @ a for c, a in zip(all_centers, all_amps)) / total_amp
                )
            else:
                shared_centroid = np.mean(np.concatenate(all_centers, axis=0), axis=0)

            # Add each channel as a separate gsplats node using dim_order
            for i, (gsplats, ch_config) in enumerate(zip(gsplats_list, CHANNELS)):
                ch_name = ch_config["name"]
                color = ch_config["color"]

                with asection(f"Adding {ch_name} (Channel {i})"):
                    # Transform: shared centroid so channels stay aligned
                    gsplats = gsplats.translate(-shared_centroid)
                    gsplats = gsplats.scale_intensity(0.1)

                    n_splats = len(gsplats.amplitudes)

                    # Assign channel color to all splats
                    colors = np.tile(np.array(color, dtype=np.float32), (n_splats, 1))

                    # KEY: Use dim_order to map 3D data → 4D scene
                    # Data columns are [Z, Y, X] from fitting a (Z, Y, X) volume
                    # Scene dimensions are [x, y, z, channel]
                    # dim_order tells the API which scene dim each data column maps to
                    scene.add_gsplats(
                        name=f"gsplats_{ch_name.lower()}",
                        centers=gsplats.centers,
                        amplitudes=gsplats.amplitudes,
                        cholesky_factors=gsplats.cholesky_factors,
                        colors=colors,
                        sharpness=gsplats.sharpnesses,
                        dim_order=["z", "y", "x"],
                        fill={"channel": float(i)},
                        fill_sigma={"channel": 0},
                        extend_to_all=[],  # Only visible at own Channel value
                        opacity=1.0,
                        blending_mode="additive",
                    )
                    aprint(f"  Added {n_splats:,} splats at Channel={i}")

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    warn_if_no_cuda_gpu()
    aprint("=" * 70)
    aprint("GSplats Demo: 4D Multi-Channel Cells (dim_order showcase)")
    aprint("=" * 70)
    aprint("3D per-channel fitting + dim_order embedding into 4D scene")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_4d_cells3d_multichannel.zarr"

    # Serve-only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    # Load data
    volumes = load_cells3d()

    # Fit gsplats per channel (with caching)
    gsplats_list = fit_all_channels(volumes)

    # Report
    with asection("Fitting Summary"):
        for i, (gsplats, ch_config) in enumerate(zip(gsplats_list, CHANNELS)):
            aprint(
                f"  {ch_config['name']}: {len(gsplats.amplitudes):,} splats, "
                f"{gsplats.centers.shape[1]}D"
            )

    # Create 4D scene using dim_order
    scene_path = create_luxar_scene(gsplats_list)

    # Launch viewer
    if not NO_SERVE:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
