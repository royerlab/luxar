#!/usr/bin/env python3
"""GSplats Demo: Multi-Channel 3D Microscopy Visualization

Demonstrates Gaussian Splatting with per-channel colors for multi-channel
microscopy data visualization.

================================================================================
MULTI-CHANNEL GAUSSIAN SPLATTING
================================================================================

This demo shows how to fit Gaussian splats to each channel of a multi-channel
microscopy volume separately, then merge them with distinct colors for
visualization.

WORKFLOW:
=========

1. **Load multi-channel data** from Image Data Resource (IDR)
   - Channel 0: First fluorescent marker
   - Channel 1: DAPI (DNA stain showing cell nuclei)

2. **Fit each channel independently**
   - Each channel gets its own set of Gaussian splats
   - Captures channel-specific structures

3. **Merge with channel colors**
   - Channel 0: Magenta (1.0, 0.0, 0.5)
   - Channel 1: Cyan (0.0, 1.0, 0.5)
   - Uses GSplatData.merge_with_channel_colors()

4. **Visualize** in the Luxar viewer
   - Additive blending shows channel overlap
   - Distinct colors reveal co-localization

USAGE:
======
    python demo_gsplats_3d_multichannel.py [--no-cache] [--no-serve]

Options:
    --no-cache: Force re-fitting even if cached results exist
    --no-serve: Don't auto-launch viewer after scene creation
    --serve-only: Skip fitting, just serve existing scene

"""

# Enable MPS→CPU fallback for unsupported PyTorch ops (must be before torch import)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
from pathlib import Path

import numpy as np
import zarr
from arbol import Arbol, aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.models.gsplats.metal import is_metal_available
from luxar.utils.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

ZARR_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
TARGET_SIZE = 128  # Downscale to manageable size
TIME_POINT = 0  # First time point

# Channel configuration with colors
CHANNELS = [
    {"index": 0, "name": "Channel 0", "color": (1.0, 0.0, 0.5)},  # Magenta
    {"index": 1, "name": "DAPI", "color": (0.0, 1.0, 0.5)},  # Cyan
]

# Fitting parameters
N_ITERS = 6000  # Good balance of quality vs speed
DEVICE = None  # Auto-detect (cuda/mps/cpu)

# Cache paths
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_multichannel"

# Parse command line flags
NO_CACHE = "--no-cache" in sys.argv
NO_SERVE = "--no-serve" in sys.argv
SERVE_ONLY = "--serve-only" in sys.argv

# Setup
Arbol.max_depth = 3
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Loading
# =============================================================================


def load_multichannel_data():
    """Load and preprocess multi-channel microscopy data from IDR."""
    with asection("Loading multi-channel microscopy data"):
        aprint(f"Source: {ZARR_URL}")
        aprint(f"Target size: {TARGET_SIZE}³ voxels per channel")

        try:
            import fsspec
            from scipy.ndimage import zoom

            # Open remote zarr
            mapper = fsspec.get_mapper(ZARR_URL)
            store = zarr.open_group(mapper, mode="r")
            data = store["0"]  # Highest resolution
            full_shape = data.shape

            aprint(f"Full data shape: {full_shape}")

            if len(full_shape) != 5:
                raise ValueError(f"Expected 5D data (T×C×Z×Y×X), got shape {full_shape}")

            n_time, n_channels, z_size, y_size, x_size = full_shape
            aprint("Format: OME-ZARR 5D (T×C×Z×Y×X)")
            aprint(f"  Time points: {n_time}")
            aprint(f"  Channels: {n_channels}")
            aprint(f"  Spatial: {z_size}×{y_size}×{x_size}")

            # Load each channel
            volumes = []
            for ch_config in CHANNELS:
                ch_idx = ch_config["index"]
                ch_name = ch_config["name"]

                if ch_idx >= n_channels:
                    aprint(f"Warning: Channel {ch_idx} not available (only {n_channels} channels)")
                    continue

                aprint(f"Loading T={TIME_POINT}, C={ch_idx} ({ch_name})...")
                V = np.array(data[TIME_POINT, ch_idx, :, :, :], dtype=np.float32)

                # Downscale
                zoom_factors = [TARGET_SIZE / s for s in V.shape]
                V = zoom(V, zoom_factors, order=1)

                # Normalize to [0, 1]
                V = (V - V.min()) / (V.max() - V.min() + 1e-8)
                V = V.astype(np.float32)

                volumes.append(V)
                aprint(f"  {ch_name}: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

            aprint(f"Loaded {len(volumes)} channels")
            return volumes

        except Exception as e:
            aprint(f"Remote loading failed: {e}")
            aprint("Creating synthetic fallback data...")

            # Fallback: synthetic multi-channel data
            volumes = []
            shape = (TARGET_SIZE, TARGET_SIZE, TARGET_SIZE)

            for ch_idx, ch_config in enumerate(CHANNELS):
                V = np.zeros(shape, dtype=np.float32)

                # Add channel-specific blobs at different positions
                np.random.seed(42 + ch_idx)  # Reproducible per channel
                for _ in range(10 + ch_idx * 5):
                    center = [np.random.uniform(15, s - 15) for s in shape]
                    sigma = np.random.uniform(4, 10)
                    amplitude = np.random.uniform(0.5, 1.0)

                    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
                    dist_sq = sum((g - c) ** 2 for g, c in zip(grids, center))
                    V += amplitude * np.exp(-dist_sq / (2 * sigma**2))

                V = np.clip(V, 0, 1).astype(np.float32)
                volumes.append(V)
                aprint(f"  {ch_config['name']}: {V.shape}")

            return volumes


# =============================================================================
# GSplats Fitting
# =============================================================================


def fit_channel(volume, channel_name, cache_file):
    """Fit gsplats to a single channel, using cache if available."""
    # Check cache
    if cache_file.exists() and not NO_CACHE:
        aprint(f"Loading cached fit for {channel_name}")
        try:
            cache = np.load(cache_file)
            result = GSplatData(
                centers=cache["centers"],
                cholesky_factors=cache["cholesky_factors"],
                amplitudes=cache["amplitudes"],
                sharpnesses=cache["sharpnesses"],
                stats={},
            )
            aprint(f"  Loaded {len(result.amplitudes)} cached splats")
            return result
        except Exception as e:
            aprint(f"  Cache load failed: {e}, re-fitting...")

    # Auto-detect best device
    global DEVICE
    if DEVICE is None:
        import torch

        if is_metal_available() and torch.backends.mps.is_available():
            DEVICE = "mps"
            aprint("Using MPS device (Metal acceleration)")
        elif torch.cuda.is_available():
            DEVICE = "cuda"
            aprint("Using CUDA device")
        else:
            DEVICE = "cpu"
            aprint("Using CPU device")

    # Fit gsplats
    aprint(f"Fitting {channel_name} ({N_ITERS} iterations)...")

    result = fit_gaussian_splats(
        volume,
        seeds=8000,
        n_iters=N_ITERS,
        device=DEVICE,
        verbose=True,
        napari_movie=False,
        max_eccentricity=8.0,
    )

    n_splats = len(result.amplitudes)
    aprint(f"  Fitted {n_splats} splats")

    # Cache result
    aprint(f"  Caching to {cache_file.name}")
    np.savez(
        cache_file,
        centers=result.centers,
        cholesky_factors=result.cholesky_factors,
        amplitudes=result.amplitudes,
        sharpnesses=result.sharpnesses,
    )

    return result


def fit_all_channels(volumes):
    """Fit gsplats to all channels."""
    with asection("Fitting GSplats per channel"):
        gsplats_list = []

        for i, (volume, ch_config) in enumerate(zip(volumes, CHANNELS)):
            ch_name = ch_config["name"]
            cache_file = CACHE_DIR / f"gsplats_ch{i}_{ch_name.lower().replace(' ', '_')}.npz"

            with asection(f"Channel {i}: {ch_name}"):
                gsplats = fit_channel(volume, ch_name, cache_file)
                gsplats_list.append(gsplats)

        return gsplats_list


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(merged_gsplats, output_path: Path | None = None):
    """Create Luxar scene with merged multi-channel gsplats."""
    if output_path is None:
        output_path = get_demos_output_dir() / "gsplats_3d_multichannel.zarr"

    with asection("Creating Luxar Scene"):
        aprint(f"Output: {output_path.name}")

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Add scene metadata
            scene.attrs["title"] = "GSplats: Multi-Channel 3D Microscopy"
            scene.attrs["description"] = """
Multi-Channel Gaussian Splatting Demo
======================================

This scene demonstrates multi-channel microscopy visualization using
Gaussian splats with per-channel colors.

Each channel was fitted independently then merged:
- Magenta: Channel 0
- Cyan: Channel 1 (DAPI - nuclear stain)

Color overlap indicates co-localization of markers.

Controls:
- Mouse drag to rotate
- Scroll to zoom
- Right-click drag to pan
- 'C' to toggle fly controls
            """

            # Add merged gsplats
            aprint(f"Adding {len(merged_gsplats.amplitudes)} merged gsplats...")
            scene.add_gsplats_from_data(
                name="multichannel_gsplats",
                result=merged_gsplats,
                opacity=1.0,
                blending_mode="additive",
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: Multi-Channel 3D Microscopy")
    aprint("=" * 70)
    aprint("Per-channel fitting + color-coded merge + Web visualization")
    aprint("")

    # Determine output path
    output_path = get_demos_output_dir() / "gsplats_3d_multichannel.zarr"

    # Serve only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
            return
        else:
            aprint(f"Scene not found: {output_path}")
            aprint("Run without --serve-only to generate first")
            return

    # Load multi-channel data
    volumes = load_multichannel_data()

    if len(volumes) < 2:
        aprint("Error: Need at least 2 channels for this demo")
        return

    # Fit each channel
    gsplats_list = fit_all_channels(volumes)

    # Merge with channel colors
    with asection("Merging channels with colors"):
        channel_colors = [ch["color"] for ch in CHANNELS[: len(gsplats_list)]]
        aprint(f"Channel colors: {channel_colors}")

        merged = GSplatData.merge_with_channel_colors(
            gsplats_list,
            channel_colors=channel_colors,
        )

        aprint(f"Merged: {len(merged.amplitudes)} total splats")
        aprint(f"  Per channel: {merged.stats.get('splats_per_channel', 'N/A')}")

    # Apply transformations for web viewer
    with asection("Applying transformations"):
        aprint("Centering at center-of-mass...")
        merged = merged.center_at_centroid()

        aprint("Reducing brightness by 10x...")
        merged = merged.scale_intensity(0.1)

    # Create scene
    scene_path = create_luxar_scene(merged, output_path)

    # Summary
    total_splats = len(merged.amplitudes)
    total_voxels = sum(v.size for v in volumes)
    volume_bytes = total_voxels * 4  # float32
    # 11 floats per splat + 3 for color = 14
    splats_bytes = total_splats * 14 * 4
    compression = volume_bytes / splats_bytes

    aprint("\n" + "=" * 70)
    aprint("Multi-Channel Compression Summary")
    aprint("=" * 70)
    aprint(f"Channels: {len(volumes)}")
    aprint(f"Total voxels: {total_voxels:,}")
    aprint(f"Total splats: {total_splats:,}")
    aprint(f"Raw size: {volume_bytes / 1024 / 1024:.2f} MB")
    aprint(f"Splat size: {splats_bytes / 1024:.2f} KB")
    aprint(f"Compression ratio: {compression:.1f}:1")
    aprint("=" * 70)

    # Launch viewer
    if NO_SERVE:
        aprint(f"Dataset generated at {scene_path}")
    else:
        aprint("\nLaunching viewer in 2 seconds...")
        time.sleep(2)
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
