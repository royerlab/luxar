#!/usr/bin/env python3
"""GSplats Demo: Multi-Channel 3D Organoid Microscopy - Full Compute Pipeline from IDR

Visualises multi-channel 3D microscopy data as Gaussian splats.
By default, uses precomputed gsplats from Git LFS (fast).
Use --recompute to fetch data from IDR and fit from scratch (slow, needs GPU).

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

3. **Add each channel as a separate layer**
   - Channel 0: Magenta colormap
   - Channel 1: Cyan colormap (DAPI)
   - Each channel is a toggleable layer in the viewer

4. **Visualize** in the Luxar viewer
   - Additive blending shows channel overlap
   - Toggle layers to inspect individual channels

USAGE:
======
    python demo_gsplats_3d_organoid_multichannel.py [--recompute] [--no-serve] [--no-napari]

Options:
    --recompute:      Force re-fitting from scratch (download + GPU fitting)
    --no-serve:       Don't auto-launch viewer after scene creation
    --no-napari:      Skip napari visualization (useful for headless/CI)
    --serve-only:     Skip fitting, just serve existing scene
    --show-roundtrip: Show matplotlib comparison of original vs reconstructed volumes

By default, precomputed GSplats are loaded from package data (Git LFS).
Use --recompute to re-fit from scratch (requires network + GPU).

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
from luxar.core.viewer_config import ViewerConfig
from luxar.encoding import EncodingMode
from luxar.gsplats import fit_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.models.gsplats.metal import is_metal_available
from luxar.utils.demos import (
    launch_viewer,
    load_precomputed_gsplats,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

ZARR_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
TARGET_SIZE = 256  # Only used by the synthetic fallback (edge cube size when
# the IDR load fails); the real path loads at native resolution (no resample).
TIME_POINT = 0  # First time point

# Channel configuration with colors
CHANNELS = [
    {"index": 0, "name": "Channel 0", "color": (1.0, 0.0, 0.5)},  # Magenta
    {"index": 1, "name": "DAPI", "color": (0.0, 1.0, 0.5)},  # Cyan
]

# Fit parameters (fixed-K, seeds=K*)
MAX_SPLATS = 22000
DEVICE = None  # Auto-detect (cuda/mps/cpu)

# Cache paths
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_multichannel"

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
NO_NAPARI = "--no-napari" in sys.argv
SHOW_ROUNDTRIP = "--show-roundtrip" in sys.argv

# Setup
Arbol.max_depth = 10
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Loading
# =============================================================================


def load_multichannel_data():
    """Load and preprocess multi-channel microscopy data from IDR.

    Returns the IDR volumes at native resolution (no zoom resample) — this
    matches the manuscript's supp_doc dataset for organoid_ch0.  An earlier
    version of this loader force-resampled to TARGET_SIZE^3 via bilinear
    ``scipy.ndimage.zoom``; that smoothed away high-frequency noise and
    pushed the held-out PSNR ceiling ~5 dB above the paper's reference,
    biasing K* upward.  Native resolution at this image (≤20M voxels) is
    plenty tractable.

    Data Source: Image Data Resource (IDR) study idr0062, Image 6001240
    Original Authors: Prisca Liberali lab, FMI
    Citation: Blin et al. (2019) + Williams et al. (2017) Nature Methods 14(8):775-781
    """
    with asection("Loading multi-channel microscopy data"):
        aprint(f"Source: {ZARR_URL}")
        aprint("Dataset: IDR idr0062, Image 6001240 (Liberali lab, FMI)")
        aprint("Resolution: native (no zoom resample)")

        try:
            import fsspec

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

            # Load each channel at native resolution.
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
    """Fit gsplats to a single channel (always fits — caller handles precomputed)."""
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

    # Fit gsplats progressively
    aprint(f"Fitting {channel_name} (fixed-K joint fit: seeds={MAX_SPLATS})...")

    result = fit_gaussian_splats(
        volume,
        seeds=MAX_SPLATS,
        device=DEVICE,
        verbose=True,
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
        zip_deflate=True,
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
                shape=tuple(dim_len * 2 for dim_len in volume.shape)
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


CHANNEL_COLORMAPS = ["magenta", "cyan"]


def create_luxar_scene(gsplats_list, output_path: Path | None = None):
    """Create Luxar scene with per-channel gsplat layers."""
    if output_path is None:
        output_path = (
            get_demos_output_dir() / "gsplats_3d_organoid_multichannel.luxar.zarr"
        )

    with asection("Creating Luxar Scene"):
        aprint(f"Output: {output_path.name}")

        # Compute shared centroid across all channels (amplitude-weighted)
        all_centers = [g.centers for g in gsplats_list]
        all_amps = [g.amplitudes for g in gsplats_list]
        total_amp = sum(a.sum() for a in all_amps)
        if total_amp > 0:
            shared_centroid = (
                sum(c.T @ a for c, a in zip(all_centers, all_amps)) / total_amp
            )
        else:
            shared_centroid = np.mean(np.concatenate(all_centers, axis=0), axis=0)

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            # Neutral tone-mapping keeps the per-channel colormap hues faithful
            # (the viewer's default ACES shifts scientific LUT colors).
            scene = compiler.create_scene(
                dimensions=Dimensions.default_3d(),
                viewer_config=ViewerConfig(tone_mapping="Neutral"),
            )

            # Add scene metadata
            scene.attrs["title"] = "GSplats: Multi-Channel 3D Organoids"
            scene.attrs["description"] = """
Multi-Channel Gaussian Splatting - Organoid Microscopy
=======================================================

This scene demonstrates multi-channel microscopy visualization using
Gaussian splats with per-channel colors as separate layers.

Data Source:
  - Image Data Resource (IDR) study idr0062, Image 6001240
  - High-resolution 3D microscopy of mouse intestinal organoid
  - Original research: Prisca Liberali lab, FMI
  - Citation: Blin et al. (2019) + Williams et al. (2017) Nat Methods 14(8):775-781

Each channel is a separate layer with its own colormap:
- Magenta: Channel 0
- Cyan: Channel 1 (DAPI - nuclear stain)

Toggle layers in the viewer to inspect individual channels.

Controls:
- Mouse drag to rotate
- Scroll to zoom
- Right-click drag to pan
- 'C' to toggle fly controls
            """

            # Add each channel as a separate layer
            for i, (gsplats, ch_config) in enumerate(
                zip(gsplats_list, CHANNELS[: len(gsplats_list)])
            ):
                ch_name = ch_config["name"]
                colormap = CHANNEL_COLORMAPS[i]

                # Center using shared centroid and reduce brightness
                centered = gsplats.translate(-shared_centroid)
                centered = centered.scale_intensity(0.1)

                aprint(
                    f"Adding {ch_name} ({len(centered.amplitudes)} splats, "
                    f"colormap={colormap})..."
                )
                scene.add_gsplats_from_data(
                    name=f"ch{i}_{ch_name.lower().replace(' ', '_')}",
                    result=centered,
                    opacity=1.0,
                    blending_mode="additive",
                    layer=True,
                    colormap=colormap,
                )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Round-Trip Visualisation
# =============================================================================


def show_roundtrip_comparison(
    volumes: list[np.ndarray],
    gsplats_list: list[GSplatData],
) -> None:
    """Show original vs round-trip reconstructed volumes side by side."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        aprint(
            "matplotlib is required for --show-roundtrip. Install with: pip install matplotlib"
        )
        return

    n_channels = len(volumes)

    with asection("Round-trip reconstruction comparison"):
        reconstructions = []
        for i, (volume, gsplats, ch_config) in enumerate(
            zip(volumes, gsplats_list, CHANNELS[:n_channels])
        ):
            with asection(f"Rendering Ch{i}: {ch_config['name']}"):
                recon = gsplats.render_to_volume(shape=volume.shape, device=DEVICE)
                reconstructions.append(recon)
                mse = float(np.mean((volume - recon) ** 2))
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
                aprint(f"  PSNR: {psnr:.2f} dB, MSE: {mse:.6g}")

        fig, axes = plt.subplots(
            n_channels, 3, figsize=(14, 4.5 * n_channels), squeeze=False
        )

        for i, (volume, recon, ch_config) in enumerate(
            zip(volumes, reconstructions, CHANNELS[:n_channels])
        ):
            mid_z = volume.shape[0] // 2
            orig_slice = volume[mid_z]
            recon_slice = recon[mid_z]
            diff_slice = np.abs(orig_slice - recon_slice)

            mse = float(np.mean((volume - recon) ** 2))
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

            axes[i, 0].imshow(orig_slice, cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_title(f"Original — {ch_config['name']}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(recon_slice, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title(f"Reconstructed (PSNR {psnr:.1f} dB)")
            axes[i, 1].axis("off")

            im = axes[i, 2].imshow(diff_slice, cmap="inferno", vmin=0, vmax=0.3)
            axes[i, 2].set_title("|Difference|")
            axes[i, 2].axis("off")
            fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Round-Trip Comparison — z-slice {mid_z}  "
            f"({sum(len(g.amplitudes) for g in gsplats_list):,} total splats)",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: Multi-Channel 3D Organoid Microscopy")
    aprint("=" * 70)
    aprint("Per-channel fitting + colormap layers + Web visualization")
    aprint("")

    # Determine output path
    output_path = get_demos_output_dir() / "gsplats_3d_organoid_multichannel.luxar.zarr"

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

    # Try loading precomputed data (from Git LFS / local cache)
    precomputed = load_precomputed_gsplats(
        "gsplats_multichannel",
        [
            "organoids_gsplats_ch0.gsplats.zarr.zip",
            "organoids_gsplats_ch1.gsplats.zarr.zip",
        ],
        recompute=RECOMPUTE,
    )

    volumes = None

    if precomputed is not None:
        gsplats_list = precomputed
    else:
        # --recompute path: download raw data, fit from scratch
        warn_if_no_cuda_gpu()
        volumes = load_multichannel_data()

        if len(volumes) < 2:
            aprint("Error: Need at least 2 channels for this demo")
            return

        gsplats_list = fit_all_channels(volumes)

    # Optional round-trip visualisation
    if SHOW_ROUNDTRIP:
        if volumes is not None:
            show_roundtrip_comparison(volumes, gsplats_list)
        else:
            aprint(
                "Cannot show round-trip: original volumes not available "
                "(loaded from precomputed cache). Re-run with --recompute."
            )

    # Create scene (centering + intensity scaling happen inside)
    scene_path = create_luxar_scene(gsplats_list, output_path)

    # Summary (only if volumes were loaded)
    if volumes is not None:
        total_splats = sum(len(g.amplitudes) for g in gsplats_list)
        total_voxels = sum(v.size for v in volumes)
        volume_bytes = total_voxels * 4  # float32
        # 11 floats per splat (no per-splat color, using colormaps)
        splats_bytes = total_splats * 11 * 4
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

    # Open in napari for visual comparison
    if not NO_NAPARI and volumes is not None:
        view_with_napari(volumes, gsplats_list, CHANNELS[: len(gsplats_list)])

    # Launch viewer
    if NO_SERVE:
        aprint(f"Dataset generated at {scene_path}")
    else:
        aprint("\nLaunching viewer in 2 seconds...")
        time.sleep(2)
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
