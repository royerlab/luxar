#!/usr/bin/env python3
"""GSplats Demo: Multi-Channel 3D Organoid Microscopy - Full Compute Pipeline from IDR

FULL COMPUTE VERSION - Fetches data from IDR and fits Gaussian splats from scratch.
This is the complete workflow but takes time to run (network download + GPU fitting).

For a QUICK START demo using precomputed gsplats, use:
    python demo_gsplats_3d_organoid_multichannel_precomputed.py

================================================================================
MULTI-CHANNEL GAUSSIAN SPLATTING - FULL PIPELINE
================================================================================

This demo shows the complete workflow for multi-channel Gaussian splatting:
- Fetching real microscopy data from Image Data Resource (IDR)
- Fitting Gaussian splats to each channel independently
- Merging channels with distinct colors for visualization

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Image ID: 6001240 (idr6001240)
Source: Image Data Resource (IDR) - https://idr.openmicroscopy.org/
Study: idr0062 - Intestinal organoid development and nuclear segmentation
Format: OME-ZARR 5D (Time × Channel × Z × Y × X)
Data Type: High-resolution 3D light microscopy of mouse intestinal organoid

Original Authors & Study:
--------------------------
Principal Investigator: Prisca Liberali
Institution: Friedrich Miescher Institute for Biomedical Research (FMI)

This data is part of research on intestinal organoid development, nuclear
segmentation, and symmetry breaking in organoids.

How to Cite:
------------
If you use this dataset, please cite:

1. Original Research:
   Blin, G., et al. (2019). "A conserved role for β-catenin in
   organ-specific branching morphogenesis."
   (Or related publications from Liberali lab associated with IDR study idr0062)

2. Image Data Resource (IDR):
   Williams, E. et al. (2017). "The Image Data Resource: a bioimage data
   integration and publication platform."
   Nature Methods, 14(8), 775-781.
   DOI: 10.1038/nmeth.4326

3. Data Accession:
   IDR study idr0062, Image 6001240
   URL: https://idr.openmicroscopy.org/webclient/?show=image-6001240

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
    python demo_gsplats_3d_organoid_multichannel_from_idr.py [--no-cache] [--no-serve]

Options:
    --no-cache: Force re-fitting even if cached results exist
    --no-serve: Don't auto-launch viewer after scene creation
    --serve-only: Skip fitting, just serve existing scene

Note: This demo fetches data from IDR and performs full GSplat fitting (slow).
For a quick demo with precomputed gsplats, use:
    python demo_gsplats_3d_organoid_multichannel_precomputed.py

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
from luxar.utils.demos import launch_viewer, warn_if_no_cuda_gpu
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

ZARR_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
TARGET_SIZE = 256  # Downscale to manageable size
TIME_POINT = 0  # First time point

# Channel configuration with colors
CHANNELS = [
    {"index": 0, "name": "Channel 0", "color": (1.0, 0.0, 0.5)},  # Magenta
    {"index": 1, "name": "DAPI", "color": (0.0, 1.0, 0.5)},  # Cyan
]

# Fitting parameters
N_ITERS = 6000  # Good balance of quality vs speed
N_SEEDS = 16000
DEVICE = None  # Auto-detect (cuda/mps/cpu)

# Cache paths
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_multichannel"

# Parse command line flags
NO_CACHE = "--no-cache" in sys.argv
NO_SERVE = "--no-serve" in sys.argv
SERVE_ONLY = "--serve-only" in sys.argv

# Setup
Arbol.max_depth = 10
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Loading
# =============================================================================


def load_multichannel_data():
    """Load and preprocess multi-channel microscopy data from IDR.

    Data Source: Image Data Resource (IDR) study idr0062, Image 6001240
    Original Authors: Prisca Liberali lab, FMI
    Citation: Blin et al. (2019) + Williams et al. (2017) Nature Methods 14(8):775-781
    """
    with asection("Loading multi-channel microscopy data"):
        aprint(f"Source: {ZARR_URL}")
        aprint("Dataset: IDR idr0062, Image 6001240 (Liberali lab, FMI)")
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
                raise ValueError(
                    f"Expected 5D data (T×C×Z×Y×X), got shape {full_shape}"
                )

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
                    aprint(
                        f"Warning: Channel {ch_idx} not available (only {n_channels} channels)"
                    )
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
            # Load from zarr.zip format
            result = GSplatData.load(cache_file, include_stats=False)
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
        seeds=N_SEEDS,
        n_iters=N_ITERS,
        device=DEVICE,
        verbose=True,
        napari_movie=False,  # Show convergence animation
        movie_every=50,  # Record frame every 50 iterations (not every iteration!)
        movie_max_frames=200,  # Limit to 200 frames max (~40 GB → ~0.4 GB)
        enable_dynamic_ops=True,
        dynamic_ops_verbose=False,  # Show when splats are relocated
    )

    n_splats = len(result.amplitudes)
    aprint(f"  Fitted {n_splats} splats")

    # Cache result in compressed zarr format
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
            cache_file = CACHE_DIR / f"organoids_gsplats_ch{i}.gsplats.zarr.zip"

            with asection(f"Channel {i}: {ch_name}"):
                gsplats = fit_channel(volume, ch_name, cache_file)
                gsplats_list.append(gsplats)

        return gsplats_list


# =============================================================================
# Napari Viewing
# =============================================================================


def view_with_napari(volumes, gsplats_list, channel_configs):
    """Open original volumes and gsplat renderings in napari for comparison."""
    try:
        import napari
    except ImportError:
        aprint("napari not installed, skipping napari view")
        aprint("Install with: pip install napari[all]")
        return

    def _colormap_for_channel(idx, name):
        if idx == 0:
            return "magenta"
        if idx == 1:
            return "cyan"
        return "gray"

    with asection("Opening in napari"):
        aprint("Preparing napari visualization...")

        rendered_volumes = []
        for idx, (volume, gsplats, ch_config) in enumerate(
            zip(volumes, gsplats_list, channel_configs)
        ):
            ch_name = ch_config["name"]
            aprint(f"Rendering gsplats for {ch_name}...")
            rendered = gsplats.render_to_volume(
                shape=(dim_len * 2 for dim_len in volume.shape)
            )
            rendered_volumes.append(rendered)

        aprint("Launching napari...")
        viewer = napari.Viewer(title="GSplats vs Original - Organoid Channels")

        for idx, (volume, rendered, ch_config) in enumerate(
            zip(volumes, rendered_volumes, channel_configs)
        ):
            ch_name = ch_config["name"]
            cmap = _colormap_for_channel(idx, ch_name)

            viewer.add_image(
                volume,
                name=f"Original {ch_name}",
                colormap=cmap,
                opacity=1.0,
                blending="additive",
            )
            viewer.add_image(
                rendered,
                name=f"GSplats {ch_name}",
                colormap=cmap,
                opacity=1.0,
                blending="additive",
            )

        aprint("Napari opened - toggle layers to compare channels")
        napari.run()


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(merged_gsplats, output_path: Path | None = None):
    """Create Luxar scene with merged multi-channel gsplats."""
    if output_path is None:
        output_path = (
            get_demos_output_dir() / "gsplats_3d_organoid_multichannel_from_idr.zarr"
        )

    with asection("Creating Luxar Scene"):
        aprint(f"Output: {output_path.name}")

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Add scene metadata
            scene.attrs["title"] = "GSplats: Multi-Channel 3D Organoids"
            scene.attrs["description"] = """
Multi-Channel Gaussian Splatting - Organoid Microscopy
=======================================================

This scene demonstrates multi-channel microscopy visualization using
Gaussian splats with per-channel colors.

Data Source:
  - Image Data Resource (IDR) study idr0062, Image 6001240
  - High-resolution 3D microscopy of mouse intestinal organoid
  - Original research: Prisca Liberali lab, FMI
  - Citation: Blin et al. (2019) + Williams et al. (2017) Nat Methods 14(8):775-781

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
    warn_if_no_cuda_gpu()
    aprint("=" * 70)
    aprint("GSplats Demo: Multi-Channel 3D Organoid Microscopy")
    aprint("=" * 70)
    aprint("Per-channel fitting + color-coded merge + Web visualization")
    aprint("")

    # Determine output path
    output_path = (
        get_demos_output_dir() / "gsplats_3d_organoid_multichannel_from_idr.zarr"
    )

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

    # Open in napari before merging/transforming for proper alignment
    view_with_napari(volumes, gsplats_list, CHANNELS[: len(gsplats_list)])

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
