#!/usr/bin/env python3
"""GSplats Demo: 3D Multi-Channel Cells (scikit-image cells3d, Layers + BOP LUTs)

Two-channel scikit-image ``cells3d`` fluorescence volume (cell membranes +
nuclei) fitted per channel as 3D Gaussian splats and shown as independent,
toggleable **layers**, each coloured by a BOP (Blue-Orange-Purple) microscopy
LUT. Fully self-contained — skimage downloads the sample on first use — so it is
the cheapest end-to-end gsplat demo to run from scratch.

================================================================================
LAYERS + BOP LUT SHOWCASE
================================================================================

Each channel is fitted independently as 3D splats, centered on a shared
amplitude-weighted centroid so the channels stay aligned, and added as a
``layer=True`` gsplats node. The viewer's Layers panel (press **L**) exposes a
per-channel visibility toggle, display-range window, gamma, and blending mode.
Colours come from Luxar's built-in BOP LUTs (``bop_orange`` for membranes,
``bop_blue`` for nuclei), applied at display time so you can switch colormaps
interactively.

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
3. **Create 3D scene** with per-channel layer nodes
4. **Add splats** with ``layer=True`` and a BOP LUT per channel
5. **Visualize** — toggle layers in the Layers panel (press L)

USAGE:
======
    python demo_gsplats_3d_cells3d_multichannel.py [--recompute] [--no-serve] [--serve-only]

Options:
    --recompute:  Force re-fitting from scratch (requires GPU)
    --no-serve:   Don't auto-launch viewer after scene creation
    --serve-only: Skip loading/fitting, just serve existing scene

By default, precomputed GSplats are loaded from package data (Git LFS).
Use --recompute to re-fit from scratch.

Output:
    - Scene saved to: demos/gsplats_3d_cells3d_multichannel.luxar.zarr
    - Automatically opens in browser
"""

from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.encoding import EncodingMode
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

# Progressive fitting parameters
MAX_SPLATS = 6000
MAX_SPLATS_PER_PASS = 1000
ITERS_PER_PASS = 3000
PSNR_PATIENCE = 0.2

# Voxel spacing (Z, Y, X) in micrometres for cells3d
# Original: (0.29, 0.065, 0.065) µm, 4x downsampled in Y/X → (0.29, 0.26, 0.26) µm
VOXEL_SIZE_ZYX = (0.29, 0.26, 0.26)

# Channel configuration — each channel is fitted separately and shown as a
# toggleable layer coloured by a BOP (Blue-Orange-Purple) microscopy LUT.
# The viewer applies the colormap at display time (interactive switching).
CHANNELS = [
    {"index": 0, "name": "Membranes", "colormap": "bop_orange"},
    {"index": 1, "name": "Nuclei", "colormap": "bop_blue"},
]

# Cache directory
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_cells3d"

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

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
            f"  Axes: (Z={raw.shape[0]}, C={raw.shape[1]}, "
            f"Y={raw.shape[2]}, X={raw.shape[3]})"
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
    """Fit gsplats to a single channel (always fits — caller handles precomputed).

    Args:
        volume: 3D volume (Z, Y, X), float32 [0, 1]
        channel_name: Human-readable channel name
        cache_file: Path to .gsplats.zarr.zip cache file

    Returns:
        GSplatData with fitted 3D splats
    """
    # Auto-detect device
    global DEVICE
    if DEVICE is None:
        from luxar.utils.demos import detect_device

        DEVICE = detect_device()

    from luxar.gsplats import fit_progressive_gaussian_splats

    aprint(
        f"Fitting {channel_name} (progressive: max_splats={MAX_SPLATS}, "
        f"{MAX_SPLATS_PER_PASS}/pass, {ITERS_PER_PASS} iters/pass, "
        f"psnr_patience={PSNR_PATIENCE})..."
    )

    result = fit_progressive_gaussian_splats(
        volume,
        max_splats=MAX_SPLATS,
        max_splats_per_pass=MAX_SPLATS_PER_PASS,
        iters_per_pass=ITERS_PER_PASS,
        psnr_patience=PSNR_PATIENCE,
        device=DEVICE,
        verbose=True,
        enable_dynamic_ops=True,
        voxel_size=VOXEL_SIZE_ZYX,
    )

    n_splats = len(result.amplitudes)
    aprint(f"  Fitted {n_splats:,} splats")

    # Cache result in compressed zarr format (AUTO = certified near-lossless,
    # the current default; centers→u16, Cholesky→certified u8).
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    aprint(f"  Caching to {cache_file.name}")
    result.save(
        cache_file,
        encoding_mode=EncodingMode.AUTO,
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
            cache_file = CACHE_DIR / f"cells3d_ch{i}.gsplats.zarr.zip"

            with asection(f"Channel {i}: {ch_name}"):
                gsplats = fit_channel(volume, ch_name, cache_file)
                gsplats_list.append(gsplats)

        return gsplats_list


# =============================================================================
# Scene Creation — per-channel layers with BOP LUTs
# =============================================================================


def create_luxar_scene(gsplats_list, output_path=None):
    """Create a 3D Luxar scene with one layer-enabled gsplats node per channel.

    Each channel is centered on a shared amplitude-weighted centroid (so the
    channels stay aligned) and added with ``layer=True`` and a BOP LUT. The
    viewer's Layers panel toggles channels and adjusts their display range,
    gamma, and blending mode live.
    """
    if output_path is None:
        output_path = (
            get_demos_output_dir() / "gsplats_3d_cells3d_multichannel.luxar.zarr"
        )

    with asection("Creating 3D Luxar Scene (per-channel layers)"):
        aprint(f"Output: {output_path.name}")

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            # Neutral tone-mapping keeps the per-channel BOP LUT hues faithful
            # (the viewer's default ACES shifts scientific LUT colors).
            scene = compiler.create_scene(
                dimensions=Dimensions.default_3d(),
                viewer_config=ViewerConfig(tone_mapping="Neutral"),
            )

            scene.attrs["title"] = "GSplats: 3D Cells Multi-Channel (BOP layers)"
            scene.attrs["description"] = """
Multi-Channel Gaussian Splatting — cells3d (scikit-image)
=========================================================

Two-channel fluorescence microscopy rendered as Gaussian splats, one
toggleable layer per channel with a BOP (Blue-Orange-Purple) LUT.

Data Source:
  - scikit-image cells3d sample dataset
  - Allen Institute for Cell Science
  - Shape: (60, 2, 256, 256) — (Z, Channel, Y, X)

Channels (Layers panel — press L):
  - bop_orange: Cell membranes (Channel 0)
  - bop_blue:   Cell nuclei    (Channel 1)

Controls:
  - Press L for the Layers panel (toggle channels, display range, gamma)
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
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

            # Add each channel as a layer-enabled gsplats node with a BOP LUT
            for i, (gsplats, ch_config) in enumerate(zip(gsplats_list, CHANNELS)):
                ch_name = ch_config["name"]
                colormap = ch_config["colormap"]

                with asection(f"Adding {ch_name} (layer, {colormap})"):
                    # Transform: shared centroid so channels stay aligned
                    gsplats = gsplats.translate(-shared_centroid)
                    gsplats = gsplats.scale_intensity(0.1)

                    n_splats = len(gsplats.amplitudes)

                    # colormap= applies a BOP LUT at display time; layer=True
                    # exposes the node in the Layers panel. Data columns are
                    # [Z, Y, X] from fitting a (Z, Y, X) volume, mapped to the
                    # scene's [x, y, z] via dim_order.
                    scene.add_gsplats(
                        name=f"gsplats_{ch_name.lower()}",
                        centers=gsplats.centers,
                        amplitudes=gsplats.amplitudes,
                        cholesky_factors=gsplats.cholesky_factors,
                        dim_order=["z", "y", "x"],
                        opacity=1.0,
                        blending_mode="additive",
                        layer=True,
                        colormap=colormap,
                    )
                    aprint(f"  Added {n_splats:,} splats with colormap='{colormap}'")

            # --- Overlays ---
            scene.add_text(
                "Cells3D Multichannel",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
            )
            scene.add_text(
                "scikit-image • 2 channels • BOP LUTs",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: 3D Multi-Channel Cells (Layers + BOP LUTs)")
    aprint("=" * 70)
    aprint("3D per-channel fitting + per-channel BOP-LUT layers")
    aprint("")

    output_path = (
        get_demos_output_dir() / "gsplats_3d_cells3d_multichannel.luxar.zarr"
    )

    # Serve-only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    # Try loading precomputed data (from Git LFS / local cache)
    precomputed = load_precomputed_gsplats(
        "gsplats_cells3d",
        ["cells3d_ch0.gsplats.zarr.zip", "cells3d_ch1.gsplats.zarr.zip"],
        recompute=RECOMPUTE,
    )

    if precomputed is not None:
        gsplats_list = precomputed
    else:
        # --recompute path: load raw data, fit from scratch
        warn_if_no_cuda_gpu()
        volumes = load_cells3d()
        gsplats_list = fit_all_channels(volumes)

    # Report
    with asection("Fitting Summary"):
        for i, (gsplats, ch_config) in enumerate(zip(gsplats_list, CHANNELS)):
            aprint(
                f"  {ch_config['name']}: {len(gsplats.amplitudes):,} splats, "
                f"{gsplats.centers.shape[1]}D"
            )

    # Create 3D scene with per-channel layers
    scene_path = create_luxar_scene(gsplats_list)

    # Launch viewer
    if not NO_SERVE:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
