#!/usr/bin/env python3
"""GSplats Demo: 2D Whole-Slide Pathology Image (CMU-1, OpenSlide)

Visualises a large H&E-stained whole-slide pathology image as 2D Gaussian
splats with per-channel (R/G/B) layers, using tiled fitting to handle
the full gigapixel resolution.

================================================================================
2D GAUSSIAN SPLATTING — WHOLE-SLIDE PATHOLOGY IMAGE (TILED)
================================================================================

This demo shows the complete workflow for tiled multi-channel 2D Gaussian
splatting on a real gigapixel digital pathology dataset:
- Downloading the Aperio SVS whole-slide image from OpenSlide test data
- Splitting into R, G, B channels (with brightness inversion for brightfield)
- Tiled fitting of 2D Gaussian splats per channel (Hann cosine apodization)
- Adding each channel as a separate layer with colormap for visualisation

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source:  OpenSlide test data — CMU-1.svs
URL:    https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/
Format: Aperio SVS (pyramidal TIFF, JPEG-compressed 256x256 tiles)
Size:   46,000 x 32,914 pixels (1.5 gigapixels), ~169 MB download
Stain:  Hematoxylin & Eosin (H&E) — brightfield histology

Imaging:
--------
Scanner:      Aperio ScanScope
Magnification: 20x
Specimen:     Human tissue section

Channels (derived from RGB):
-----------------------------
  0: Red   — Eosin / cytoplasm / connective tissue
  1: Green — Intermediate (both stains contribute)
  2: Blue  — Hematoxylin / nuclei / basophilic structures

License:
--------
CC0 1.0 Universal (Public Domain Dedication)

How to Cite:
------------
If you use this dataset, please cite:
  OpenSlide: A Vendor-Neutral Software Foundation for Digital Pathology
  Adam Goode, Benjamin Gilbert, Jan Harkes, Drazen Jukic, M. Satyanarayanan
  Journal of Pathology Informatics 2013, 4:27
  https://openslide.org

WORKFLOW:
=========

1. **Download** SVS whole-slide image from OpenSlide (~169 MB)
2. **Read** full-resolution level from the pyramidal TIFF
3. **Split** into R, G, B channels and invert (brightfield -> dark-on-light)
4. **Tiled fit** 2D Gaussian splats per channel (Hann cosine apodization
   for seamless tile stitching) with GPU acceleration
5. **Add** each channel as a separate layer with colormap
6. **Visualise** in the Luxar web viewer

USAGE:
======
    python demo_gsplats_2d_cmu1_pathology.py [OPTIONS]

Options:
    --recompute:        Force re-fitting from scratch (download + GPU fitting)
    --no-serve:         Generate scene without launching viewer
    --serve-only:       Just serve a previously generated scene
    --show-roundtrip:   Show matplotlib comparison of original vs reconstructed images
    --target-size=N:    Downsample target for longest axis (default: 0 = full res)
    --tile-size=N:      Tile size in pixels for tiled fitting (default: 4096)
    --overlap=N:        Tile overlap in pixels (default: 512)
    --seeds-per-tile=N: Gaussian seeds per tile (default: 255000, per tile)

By default, the demo works at full resolution (46,000 x 32,914 pixels)
using tiled fitting.

Output:
    - Scene saved to:  datasets/demos/gsplats_2d_cmu1_pathology.luxar.zarr
    - Automatically opens in browser
"""

DEMO_META = {
    "key": "gsplats_2d_cmu1_pathology",
    "title": "2D Whole-Slide Pathology Image (CMU-1, OpenSlide)",
    "description": "H&E whole-slide pathology image (CMU-1, 1.5 gigapixel) as tiled 2D Gaussian splats.",
    "category": "medical",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 150,  # approx
        "compute": "medium",
        "gpu": "optional",
        "local_data": "git-lfs",
    },
    "caches": ["gsplats_cmu1_pathology"],
    "outputs": ["gsplats_2d_cmu1_pathology"],
}

# Enable MPS->CPU fallback for unsupported PyTorch ops (must be before torch import)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import UIConfig, ViewerConfig
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.demos import (
    launch_viewer,
    load_precomputed_gsplats,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
from luxar.utils.download import download_with_checksum
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# OpenSlide test data — direct HTTP download (no auth required)
DOWNLOAD_URL = (
    "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs"
)
EXPECTED_FILE_SIZE = 177_552_579  # 169.3 MB
EXPECTED_SHA256 = "00a3d54482cd707abf254fe69dccc8d06b8ff757a1663f1290c23418c480eb30"

# Original image specs
ORIGINAL_WIDTH = 46_000
ORIGINAL_HEIGHT = 32_914
MAGNIFICATION = 20  # 20x objective

# Channel configuration — RGB split from brightfield H&E
N_CHANNELS = 3
CHANNELS = [
    {"index": 0, "name": "Red (Eosin/Tissue)", "color": (1.0, 0.2, 0.2)},
    {"index": 1, "name": "Green", "color": (0.2, 1.0, 0.2)},
    {"index": 2, "name": "Blue (Hematoxylin/Nuclei)", "color": (0.2, 0.2, 1.0)},
]

# Tiled fitting parameters
TILE_SIZE = 4096  # Pixels per tile axis — fits comfortably in GPU memory
OVERLAP = 512  # Overlap for Hann cosine apodization (seamless stitching)
SEEDS_PER_TILE = 500000  # Seeds per tile
N_ITERS = 6_000

# Cache location
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_cmu1_pathology"

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
SHOW_ROUNDTRIP = "--show-roundtrip" in sys.argv

# Parse optional CLI overrides
TARGET_SIZE = 0  # Default: full resolution (46,000 x 32,914)
for _arg in sys.argv:
    if _arg.startswith("--target-size="):
        TARGET_SIZE = int(_arg.split("=")[1])
    elif _arg.startswith("--tile-size="):
        TILE_SIZE = int(_arg.split("=")[1])
    elif _arg.startswith("--overlap="):
        OVERLAP = int(_arg.split("=")[1])
    elif _arg.startswith("--seeds-per-tile="):
        SEEDS_PER_TILE = int(_arg.split("=")[1])

# Arbol logging depth
Arbol.max_depth = 10
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Auto-detected on first fit
DEVICE = None


# =============================================================================
# Data Loading
# =============================================================================


def load_cmu1_image() -> list[np.ndarray]:
    """Download and load the CMU-1 whole-slide image as RGB channels.

    Downloads the SVS file, reads the full-resolution level (or a suitable
    pyramid level if --target-size is set), splits into R/G/B, inverts
    (brightfield), and normalises.

    Returns:
        List of 3 channel images (2D float32, normalised to [0, 1]).
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError(
            "tifffile is required for this demo.\nInstall with: pip install tifffile"
        )

    from scipy.ndimage import zoom

    # Download
    svs_path = CACHE_DIR / "CMU-1.svs"
    download_with_checksum(
        DOWNLOAD_URL,
        svs_path,
        expected_sha256=EXPECTED_SHA256,
        expected_size=EXPECTED_FILE_SIZE,
    )

    # Read SVS — find best pyramid level
    with asection("Loading whole-slide image"):
        aprint(f"Reading {svs_path.name}...")

        with tifffile.TiffFile(str(svs_path)) as tif:
            aprint(f"Pages: {len(tif.pages)}")
            for i, page in enumerate(tif.pages):
                aprint(f"  Page {i}: {page.shape}, dtype={page.dtype}")

            if TARGET_SIZE > 0:
                # Find smallest page that is >= TARGET_SIZE on longest axis
                best_page_idx = 0
                best_shape = tif.pages[0].shape

                for i, page in enumerate(tif.pages):
                    shape = page.shape
                    if len(shape) < 2:
                        continue
                    longest = max(shape[0], shape[1])
                    if longest >= TARGET_SIZE:
                        best_longest = max(best_shape[0], best_shape[1])
                        if longest < best_longest:
                            best_page_idx = i
                            best_shape = shape
            else:
                # Full resolution — first page
                best_page_idx = 0
                best_shape = tif.pages[0].shape

            aprint(f"Selected page {best_page_idx}: {best_shape}")
            data = tif.pages[best_page_idx].asarray()
            aprint(f"Loaded: {data.shape}, dtype={data.dtype}")

    # Convert to float32 and split channels
    with asection("Processing RGB channels"):
        if data.ndim == 2:
            data = np.stack([data, data, data], axis=-1)

        if data.ndim != 3 or data.shape[2] < 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {data.shape}")

        # Take only first 3 channels (ignore alpha if present)
        data = data[:, :, :3].astype(np.float32)

        # Normalise to [0, 1]
        if data.max() > 1.0:
            data = data / 255.0

        h, w = data.shape[0], data.shape[1]
        aprint(f"Image: {w}x{h} pixels, RGB float32")

        # Optional downsampling
        longest = max(h, w)
        if TARGET_SIZE > 0 and longest > TARGET_SIZE:
            scale = TARGET_SIZE / longest
            target_h = int(h * scale)
            target_w = int(w * scale)
            aprint(f"Downsampling to {target_w}x{target_h}")
        else:
            target_h, target_w = h, w
            scale = 1.0

        # Invert brightness: H&E brightfield has white background and dark tissue.
        # For gsplat fitting, we want tissue structures to be bright (high amplitude).
        aprint("Inverting brightness (brightfield -> dark background)")
        data = 1.0 - data

        # Split and optionally downsample each channel
        channels = []
        for ch_idx, ch_config in enumerate(CHANNELS):
            ch = data[:, :, ch_idx]

            if scale < 1.0:
                zoom_factors = (target_h / ch.shape[0], target_w / ch.shape[1])
                ch = zoom(ch, zoom_factors, order=1)

            # Re-normalise to [0, 1]
            vmin, vmax = ch.min(), ch.max()
            if vmax - vmin > 1e-8:
                ch = (ch - vmin) / (vmax - vmin)
            ch = ch.astype(np.float32)

            channels.append(ch)
            aprint(
                f"  Ch{ch_idx} ({ch_config['name']}): {ch.shape}, "
                f"range [{ch.min():.3f}, {ch.max():.3f}]"
            )

        del data  # Free memory

    return channels


# =============================================================================
# Tiled GSplats Fitting
# =============================================================================


def fit_channel_tiled(
    image: np.ndarray,
    channel_name: str,
    cache_file: Path,
) -> GSplatData:
    """Fit 2D gsplats to a single channel using tiled fitting.

    Uses Hann cosine apodization for seamless tile stitching across the
    full-resolution image.

    Args:
        image: 2D float32 image (H, W), normalised to [0, 1].
        channel_name: Name for logging.
        cache_file: Where to cache the fitted result.

    Returns:
        Fitted GSplatData with 2D centers in pixel coordinates.
    """
    from luxar.gsplats import fit_tiled

    global DEVICE
    if DEVICE is None:
        from luxar.utils.demos import detect_device

        DEVICE = detect_device()

    h, w = image.shape
    aprint(f"Tiled fitting {channel_name}: {w}x{h} px")
    aprint(f"  Tile size: {TILE_SIZE}, overlap: {OVERLAP}")
    aprint(f"  Seeds/tile: {SEEDS_PER_TILE:,}, iters: {N_ITERS}")
    aprint(f"  Device: {DEVICE}")

    result = fit_tiled(
        image,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        seeds=SEEDS_PER_TILE,
        n_iters=N_ITERS,
        device=DEVICE,
        verbose=True,
        enable_dynamic_ops=True,
    )

    n_splats = len(result.amplitudes)
    aprint(f"  Fitted {n_splats:,} splats across all tiles")

    # Cache result
    aprint(f"  Caching to {cache_file.name}")
    result.save(
        cache_file,
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )

    return result


def fit_all_channels(
    images: list[np.ndarray],
) -> list[GSplatData]:
    """Fit 2D gsplats to all RGB channels using tiled fitting."""
    with asection("Tiled fitting of 2D GSplats per channel"):
        gsplats_list = []

        for i, (image, ch_config) in enumerate(zip(images, CHANNELS)):
            ch_name = ch_config["name"]
            cache_file = CACHE_DIR / f"cmu1_ch{i}.gsplats.zarr.zip"

            # Check per-channel cache
            if cache_file.exists() and not RECOMPUTE:
                with asection(f"Channel {i}: {ch_name} (cached)"):
                    gsplats = GSplatData.load(cache_file, include_stats=False)
                    aprint(f"Loaded {len(gsplats.amplitudes):,} splats from cache")
                    gsplats_list.append(gsplats)
                    continue

            with asection(f"Channel {i}: {ch_name}"):
                gsplats = fit_channel_tiled(image, ch_name, cache_file)
                gsplats_list.append(gsplats)

        return gsplats_list


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(
    gsplats_list: list[GSplatData], output_path: Path | None = None
) -> Path:
    """Create Luxar scene with per-channel 2D gsplats as separate layers.

    Args:
        gsplats_list: List of per-channel GSplatData objects (one per RGB channel).
        output_path: Output .zarr path (default: demos output dir).

    Returns:
        Path to saved scene.
    """
    if output_path is None:
        output_path = get_demos_output_dir() / "gsplats_2d_cmu1_pathology.luxar.zarr"

    # Channel -> colormap mapping
    CHANNEL_COLORMAPS = ["red", "green", "blue"]

    with asection("Creating Luxar Scene"):
        aprint(f"Output: {output_path.name}")

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            dims = Dimensions(
                [
                    Dimension("x", unit="px", display=True),
                    Dimension("y", unit="px", display=True),
                ]
            )
            # 2D data: start in orthographic mode with scale bar visible.
            # Neutral tone-mapping keeps the H&E R/G/B colors faithful — the
            # viewer's default ACES shifts hues away from true histology color.
            viewer_config = ViewerConfig(
                control_type="ortho",
                tone_mapping="Neutral",
                ui=UIConfig(show_scale_bar=True),
            )
            scene = compiler.create_scene(dimensions=dims, viewer_config=viewer_config)

            scene.attrs["title"] = "GSplats 2D: Whole-Slide Pathology (CMU-1, H&E)"
            scene.attrs["description"] = f"""
2D Gaussian Splatting - Whole-Slide Pathology Image
======================================================

H&E-stained human tissue section from digital pathology,
represented as 2D Gaussian splats with per-channel layers.

Fitted using tiled Gaussian splatting with Hann cosine apodization
for seamless stitching across {ORIGINAL_WIDTH:,} x {ORIGINAL_HEIGHT:,} pixels.

Data Source:
  - OpenSlide test data: CMU-1.svs (CC0 public domain)
  - Aperio ScanScope, 20x magnification
  - Resolution: {ORIGINAL_WIDTH:,} x {ORIGINAL_HEIGHT:,} pixels (1.5 gigapixels)

Tiled Fitting:
  - Tile size: {TILE_SIZE} px, overlap: {OVERLAP} px
  - Seeds per tile: {SEEDS_PER_TILE:,}, iterations: {N_ITERS:,}

Channels (inverted brightfield RGB, each a separate layer):
  - Red:   Eosin / cytoplasm / connective tissue
  - Green: Mixed contribution from both stains
  - Blue:  Hematoxylin / nuclei / basophilic structures

Controls:
  - Press L to open the Layers panel
  - Click eye icon to toggle channel visibility
  - Adjust [min, max] display range per channel
  - Mouse drag to pan, scroll to zoom
            """

            # Compute shared centroid across ALL channels so they stay aligned
            with asection("Computing shared centroid"):
                all_centers = [g.centers for g in gsplats_list]
                all_amps = [g.amplitudes for g in gsplats_list]
                total_amp = sum(a.sum() for a in all_amps)
                if total_amp > 0:
                    shared_centroid = (
                        sum(c.T @ a for c, a in zip(all_centers, all_amps)) / total_amp
                    )
                else:
                    shared_centroid = np.mean(
                        np.concatenate(all_centers, axis=0), axis=0
                    )
                aprint(f"  Shared centroid: {shared_centroid}")

            # Add each channel as a layer-enabled gsplats node
            for i, (gsplats, ch_config) in enumerate(
                zip(gsplats_list, CHANNELS[: len(gsplats_list)])
            ):
                ch_name = ch_config["name"]
                colormap = CHANNEL_COLORMAPS[i]

                with asection(f"Adding {ch_name} (layer)"):
                    # Transform: shared centroid so channels stay aligned
                    gsplats = gsplats.translate(-shared_centroid)
                    gsplats = gsplats.scale_intensity(0.1)

                    n_splats = len(gsplats.amplitudes)

                    scene.add_gsplats(
                        name=f"gsplats_{colormap}",
                        centers=gsplats.centers,
                        amplitudes=gsplats.amplitudes,
                        cholesky_factors=gsplats.cholesky_factors,
                        dim_order=["x", "y"],
                        opacity=1.0,
                        # Stays additive while the other bioimaging gsplat
                        # demos are volumetric: this fit is strictly 2D, so
                        # there is no depth structure for volumetric to
                        # resolve. Viewed face-on (the default under this
                        # demo's control_type="ortho") every splat shares one
                        # view-depth plane and the depth sorter takes its
                        # identity-ordering branch, so volumetric would just
                        # composite in storage order. Additive blending is
                        # order-independent, so for a 2D fit the additive sum
                        # IS the reconstruction regardless of the camera.
                        blending_mode="additive",
                        layer=True,
                        colormap=colormap,
                    )
                    aprint(f"  Added {n_splats:,} splats with colormap='{colormap}'")

            # --- Overlays ---
            # Title
            scene.add_text(
                "Whole-Slide Pathology (CMU-1)",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Info
            scene.add_text(
                "46K\u00d732K \u2022 H&E stain \u2022 20x",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Round-Trip Visualisation
# =============================================================================


def show_roundtrip_comparison(
    images: list[np.ndarray],
    gsplats_list: list[GSplatData],
) -> None:
    """Show original vs round-trip reconstructed 2D images side by side."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        aprint(
            "matplotlib is required for --show-roundtrip. Install with: pip install matplotlib"
        )
        return

    n_channels = len(images)

    with asection("Round-trip reconstruction comparison"):
        reconstructions = []
        for i, (image, gsplats, ch_config) in enumerate(
            zip(images, gsplats_list, CHANNELS[:n_channels])
        ):
            with asection(f"Rendering Ch{i}: {ch_config['name']}"):
                recon = gsplats.render_to_volume(shape=image.shape, device=DEVICE)
                reconstructions.append(recon)
                mse = float(np.mean((image - recon) ** 2))
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
                aprint(f"  PSNR: {psnr:.2f} dB, MSE: {mse:.6g}")

        fig, axes = plt.subplots(
            n_channels, 3, figsize=(14, 4.5 * n_channels), squeeze=False
        )

        for i, (image, recon, ch_config) in enumerate(
            zip(images, reconstructions, CHANNELS[:n_channels])
        ):
            diff = np.abs(image - recon)

            mse = float(np.mean((image - recon) ** 2))
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

            axes[i, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_title(f"Original — {ch_config['name']}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(recon, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title(f"Reconstructed (PSNR {psnr:.1f} dB)")
            axes[i, 1].axis("off")

            im = axes[i, 2].imshow(diff, cmap="inferno", vmin=0, vmax=0.3)
            axes[i, 2].set_title("|Difference|")
            axes[i, 2].axis("off")
            fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Round-Trip Comparison  "
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
    aprint("GSplats Demo: 2D Whole-Slide Pathology (CMU-1, H&E)")
    aprint("=" * 70)
    aprint("Tiled per-channel RGB fitting + per-channel layers + Web viewer")
    if TARGET_SIZE > 0:
        aprint(f"Target size: {TARGET_SIZE} px (longest axis)")
    else:
        aprint(f"Full resolution: {ORIGINAL_WIDTH:,} x {ORIGINAL_HEIGHT:,} px")
    aprint(
        f"Tile: {TILE_SIZE} px, overlap: {OVERLAP} px, seeds/tile: {SEEDS_PER_TILE:,}"
    )
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_2d_cmu1_pathology.luxar.zarr"

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
        "gsplats_cmu1_pathology",
        [
            "cmu1_ch0.gsplats.zarr.zip",
            "cmu1_ch1.gsplats.zarr.zip",
            "cmu1_ch2.gsplats.zarr.zip",
        ],
        recompute=RECOMPUTE,
    )

    images = None

    if precomputed is not None:
        gsplats_list = precomputed
    else:
        # --recompute path: download raw data, fit from scratch
        warn_if_no_cuda_gpu()
        images = load_cmu1_image()

        if len(images) < N_CHANNELS:
            aprint(f"Error: Need {N_CHANNELS} channels, got {len(images)}")
            return

        # Tiled fitting per channel
        gsplats_list = fit_all_channels(images)

    # Optional round-trip visualisation
    if SHOW_ROUNDTRIP:
        if images is not None:
            show_roundtrip_comparison(images, gsplats_list)
        else:
            aprint(
                "Cannot show round-trip: original images not available "
                "(loaded from precomputed cache). Re-run with --recompute."
            )

    # Create scene with per-channel layers
    scene_path = create_luxar_scene(gsplats_list, output_path)

    # Summary
    if images is not None:
        total_splats = sum(len(g.amplitudes) for g in gsplats_list)
        total_pixels = sum(img.size for img in images)
        pixel_bytes = total_pixels * 4  # float32
        # Floats per 2D splat: d + d*(d+1)/2 + 2 + 3 (colors) = 2 + 3 + 2 + 3 = 10
        splats_bytes = total_splats * 10 * 4
        compression = pixel_bytes / splats_bytes if splats_bytes > 0 else 0

        aprint("")
        aprint("=" * 70)
        aprint("2D Tiled Multi-Channel Compression Summary")
        aprint("=" * 70)
        aprint(f"Channels:          {N_CHANNELS}")
        aprint(f"Total pixels:      {total_pixels:,}")
        aprint(f"Total splats:      {total_splats:,}")
        aprint(f"Raw pixel data:    {pixel_bytes / (1024 * 1024):.1f} MB")
        aprint(f"Splat data size:   {splats_bytes / 1024:.1f} KB")
        aprint(f"Compression ratio: {compression:.1f}:1")
        aprint("=" * 70)

    # Launch viewer
    if NO_SERVE:
        aprint(f"Dataset generated at {scene_path}")
    else:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
