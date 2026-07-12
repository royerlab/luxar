#!/usr/bin/env python3
"""GSplats Demo: 2D 12-Channel CODEX Pancreas (Multiplexed Fluorescence)

Visualises a 12-channel multiplexed immunofluorescence image of human pancreas
tissue as 2D Gaussian splats with per-channel layers, using tiled fitting.

================================================================================
2D GAUSSIAN SPLATTING — CODEX MULTIPLEXED PANCREAS (12 CHANNELS, TILED)
================================================================================

This demo shows the complete workflow for tiled multi-channel 2D Gaussian
splatting on a real multiplexed immunofluorescence dataset:
- Downloading 12-channel CODEX data from Zenodo
- Extracting and normalising each fluorescence channel
- Tiled fitting of 2D Gaussian splats per channel (Hann cosine apodization)
- Adding each channel as a separate layer with colormap
- Visualising in the Luxar web viewer

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source:  Zenodo record 7742474
URL:    https://zenodo.org/records/7742474
Format: 12 separate 16-bit TIFF files in a ZIP archive
Size:   25,816 x 18,440 pixels (476 megapixels) per channel
Pixel:  0.325 x 0.325 um
Depth:  16-bit unsigned integer

Imaging:
--------
Method:     CODEX (CO-Detection by indEXing) multiplexed immunofluorescence
Microscope: Leica system with CODEX Processor v1.7.0.6
Specimen:   Human pancreas 5 um tissue section

Channels (12 protein markers):
-------------------------------
  Hoechst      — Nuclear stain (DNA, all cells)
  CGC          — Chromogranin C (endocrine granules)
  Beta-Catenin — Cell-cell junctions (Wnt signalling)
  CPEP         — C-peptide (insulin-producing beta cells)
  VIM          — Vimentin (mesenchymal / stromal cells)
  KRT19        — Keratin 19 (ductal epithelium)
  PECAM-1      — CD31 (endothelial cells / vasculature)
  SST          — Somatostatin (delta cells)
  E-Cadherin   — Epithelial cell junctions
  CHGA         — Chromogranin A (neuroendocrine)
  ACTA2        — Alpha smooth muscle actin
  IAPP         — Islet amyloid polypeptide (beta cells)

License:
--------
Creative Commons Attribution 4.0 International (CC BY 4.0)

WORKFLOW:
=========

1. **Download** ZIP archive from Zenodo (~5.9 GB)
2. **Extract** 12 TIFF channel files
3. **Normalise** each channel (uint16 -> float32 [0, 1])
4. **Tiled fit** 2D Gaussian splats per channel with GPU acceleration
5. **Add** each channel as a separate layer with colormap
6. **Visualise** in the Luxar web viewer

USAGE:
======
    python demo_gsplats_2d_codex_pancreas.py [OPTIONS]

Options:
    --recompute:        Force re-fitting from scratch
    --no-serve:         Generate scene without launching viewer
    --serve-only:       Just serve a previously generated scene
    --show-roundtrip:   Show matplotlib comparison of original vs reconstructed images
    --target-size=N:    Downsample target for longest axis (default: 0 = full res)
    --tile-size=N:      Tile size in pixels for tiled fitting (default: 4096)
    --overlap=N:        Tile overlap in pixels (default: 512)
    --seeds-per-tile=N: Gaussian seeds per tile (default: 255000, per tile)

By default, the demo works at full resolution (25,816 x 18,440 pixels)
using tiled fitting. Each channel is fitted and cached independently,
so interrupted runs can resume from where they left off.

Output:
    - Scene saved to:  datasets/demos/gsplats_2d_codex_pancreas.luxar.zarr
    - Automatically opens in browser
"""

# Enable MPS->CPU fallback for unsupported PyTorch ops (must be before torch import)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import zipfile
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import UIConfig, ViewerConfig
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.demos import (
    launch_viewer,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
from luxar.utils.download import robust_download
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Zenodo download
DOWNLOAD_URL = "https://zenodo.org/records/7742474/files/human_pancreas_codex.zip"
EXPECTED_FILE_SIZE = 6_294_399_806  # 5.9 GB

# Image specs
ORIGINAL_WIDTH = 25_816
ORIGINAL_HEIGHT = 18_440
PIXEL_SIZE = 0.325  # um per pixel

# 12 fluorescence channels with biologically meaningful colours
N_CHANNELS = 12
CHANNELS = [
    {
        "file": "reg001_cyc001_ch001_Hoechst.tiff",
        "name": "Hoechst (Nuclei)",
        "color": (0.0, 0.4, 1.0),
        "colormap": "blue",
    },
    {
        "file": "reg001_cyc002_ch003_CGC.tiff",
        "name": "CGC (Endocrine)",
        "color": (1.0, 0.8, 0.0),
        "colormap": "yellow",
    },
    {
        "file": "reg001_cyc002_ch004_BETA-CATENIN.tiff",
        "name": "Beta-Catenin (Junctions)",
        "color": (0.0, 1.0, 0.6),
        "colormap": "green",
    },
    {
        "file": "reg001_cyc003_ch003_CPEP.tiff",
        "name": "CPEP (Beta Cells)",
        "color": (1.0, 0.0, 0.4),
        "colormap": "red",
    },
    {
        "file": "reg001_cyc003_ch004_VIM.tiff",
        "name": "VIM (Mesenchymal)",
        "color": (0.6, 0.0, 1.0),
        "colormap": "magenta",
    },
    {
        "file": "reg001_cyc004_ch003_KRT19.tiff",
        "name": "KRT19 (Ductal)",
        "color": (0.0, 1.0, 1.0),
        "colormap": "cyan",
    },
    {
        "file": "reg001_cyc005_ch002_PECAM-1.tiff",
        "name": "PECAM-1 (Vasculature)",
        "color": (1.0, 0.0, 0.0),
        "colormap": "red",
    },
    {
        "file": "reg001_cyc005_ch003_SST.tiff",
        "name": "SST (Delta Cells)",
        "color": (1.0, 0.5, 0.0),
        "colormap": "orange",
    },
    {
        "file": "reg001_cyc006_ch003_E-CADHERIN.tiff",
        "name": "E-Cadherin (Epithelial)",
        "color": (0.5, 1.0, 0.0),
        "colormap": "green",
    },
    {
        "file": "reg001_cyc006_ch004_CHGA.tiff",
        "name": "CHGA (Endocrine)",
        "color": (1.0, 1.0, 0.0),
        "colormap": "yellow",
    },
    {
        "file": "reg001_cyc007_ch002_ACTA2.tiff",
        "name": "ACTA2 (Smooth Muscle)",
        "color": (1.0, 0.0, 1.0),
        "colormap": "magenta",
    },
    {
        "file": "reg001_cyc007_ch003_IAPP.tiff",
        "name": "IAPP (Beta Cells)",
        "color": (0.0, 1.0, 0.3),
        "colormap": "green",
    },
]

# Tiled fitting parameters
TILE_SIZE = 4096
OVERLAP = 512
SEEDS_PER_TILE = 255000
N_ITERS = 6_000

# Cache location
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_codex_pancreas"

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
SHOW_ROUNDTRIP = "--show-roundtrip" in sys.argv

# Parse optional CLI overrides
TARGET_SIZE = 0  # Default: full resolution
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


def ensure_extracted() -> Path:
    """Download the ZIP and extract TIFF files to cache.

    Returns:
        Path to the directory containing extracted TIFF files.
    """
    extract_dir = CACHE_DIR / "tiffs"
    zip_path = CACHE_DIR / "human_pancreas_codex.zip"

    # Check if all TIFFs already extracted
    all_present = extract_dir.exists() and all(
        (extract_dir / ch["file"]).exists() for ch in CHANNELS
    )
    if all_present:
        aprint("All TIFF files already extracted")
        return extract_dir

    # Download if needed
    robust_download(
        DOWNLOAD_URL,
        zip_path,
        expected_size=EXPECTED_FILE_SIZE,
    )

    # Extract TIFFs (skip __MACOSX junk)
    with asection("Extracting TIFF files"):
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            for ch in CHANNELS:
                fname = ch["file"]
                target = extract_dir / fname
                if target.exists():
                    aprint(f"  Already extracted: {fname}")
                    continue
                aprint(f"  Extracting {fname}...")
                with zf.open(fname) as src, open(target, "wb") as dst:
                    import shutil

                    shutil.copyfileobj(src, dst)
        aprint(f"Extracted {N_CHANNELS} channels to {extract_dir}")

    return extract_dir


def load_channel(tiff_dir: Path, ch_config: dict) -> np.ndarray:
    """Load a single fluorescence channel from TIFF.

    Args:
        tiff_dir: Directory containing extracted TIFF files.
        ch_config: Channel config dict with 'file' and 'name' keys.

    Returns:
        2D float32 image (H, W), normalised to [0, 1].
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError(
            "tifffile is required for this demo.\nInstall with: pip install tifffile"
        )

    from scipy.ndimage import zoom

    fname = ch_config["file"]
    path = tiff_dir / fname

    data = tifffile.imread(str(path))
    aprint(f"  Loaded {fname}: {data.shape}, dtype={data.dtype}")

    # Convert to float32 and normalise to [0, 1]
    data = data.astype(np.float32)
    vmax = data.max()
    if vmax > 0:
        data = data / vmax
    # No inversion — fluorescence: bright = signal

    # Optional downsampling
    h, w = data.shape[:2]
    longest = max(h, w)
    if TARGET_SIZE > 0 and longest > TARGET_SIZE:
        scale = TARGET_SIZE / longest
        target_h = int(h * scale)
        target_w = int(w * scale)
        zoom_factors = (target_h / h, target_w / w)
        data = zoom(data, zoom_factors, order=1)

    data = data.astype(np.float32)
    aprint(
        f"  {ch_config['name']}: {data.shape}, "
        f"range [{data.min():.3f}, {data.max():.3f}]"
    )

    return data


# =============================================================================
# Tiled GSplats Fitting
# =============================================================================


def fit_channel_tiled(
    image: np.ndarray,
    channel_name: str,
    cache_file: Path,
) -> GSplatData:
    """Fit 2D gsplats to a single channel using tiled fitting.

    Args:
        image: 2D float32 image (H, W), normalised to [0, 1].
        channel_name: Name for logging.
        cache_file: Where to cache the fitted result.

    Returns:
        Fitted GSplatData with 2D centers.
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
        voxel_size=PIXEL_SIZE,
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


def fit_all_channels(tiff_dir: Path) -> list[GSplatData]:
    """Load and fit all 12 channels, one at a time to save memory."""
    with asection(f"Tiled fitting of {N_CHANNELS} fluorescence channels"):
        gsplats_list = []

        for i, ch_config in enumerate(CHANNELS):
            ch_name = ch_config["name"]
            cache_file = CACHE_DIR / f"codex_ch{i:02d}.gsplats.zarr.zip"

            # Check per-channel cache
            if cache_file.exists() and not RECOMPUTE:
                with asection(f"Channel {i}: {ch_name} (cached)"):
                    gsplats = GSplatData.load(cache_file, include_stats=False)
                    aprint(f"Loaded {len(gsplats.amplitudes):,} splats from cache")
                    gsplats_list.append(gsplats)
                    continue

            with asection(f"Channel {i}: {ch_name}"):
                # Load one channel at a time (908 MB each)
                image = load_channel(tiff_dir, ch_config)
                gsplats = fit_channel_tiled(image, ch_name, cache_file)
                gsplats_list.append(gsplats)
                del image  # Free 908 MB

        return gsplats_list


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(
    gsplats_list: list[GSplatData], output_path: Path | None = None
) -> Path:
    """Create Luxar scene with per-channel 2D gsplat layers.

    Each channel is added as a separate gsplats node with ``layer=True``
    and a ``colormap``, so the viewer's Layers panel (press L) provides
    per-channel visibility, display-range, gamma, and colormap controls.

    Args:
        gsplats_list: List of per-channel GSplatData (one per channel).
        output_path: Output .zarr path (default: demos output dir).

    Returns:
        Path to saved scene.
    """
    if output_path is None:
        output_path = get_demos_output_dir() / "gsplats_2d_codex_pancreas.luxar.zarr"

    with asection("Creating Luxar Scene"):
        aprint(f"Output: {output_path.name}")

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                ]
            )
            # 2D data: start in orthographic mode with scale bar visible
            viewer_config = ViewerConfig(
                control_type="ortho",
                ui=UIConfig(show_scale_bar=True),
            )
            scene = compiler.create_scene(dimensions=dims, viewer_config=viewer_config)

            ch_list = "\n".join(f"  - {ch['name']}" for ch in CHANNELS)
            scene.attrs["title"] = "GSplats 2D: CODEX Pancreas (12-Channel Multiplexed)"
            scene.attrs["description"] = f"""
2D Gaussian Splatting - CODEX Multiplexed Pancreas
======================================================

12-channel multiplexed immunofluorescence of human pancreas tissue,
represented as 2D Gaussian splats with per-channel layers.

Fitted using tiled Gaussian splatting with Hann cosine apodization
for seamless stitching across {ORIGINAL_WIDTH:,} x {ORIGINAL_HEIGHT:,} pixels.

Data Source:
  - Zenodo record 7742474 (CC BY 4.0)
  - CODEX multiplexed imaging, Leica microscope
  - Resolution: {ORIGINAL_WIDTH:,} x {ORIGINAL_HEIGHT:,} pixels (476 MP)
  - Pixel size: {PIXEL_SIZE} um

Tiled Fitting:
  - Tile size: {TILE_SIZE} px, overlap: {OVERLAP} px
  - Seeds per tile: {SEEDS_PER_TILE:,}, iterations: {N_ITERS:,}

Channels ({N_CHANNELS} protein markers):
{ch_list}

Controls:
  - Press L to open the Layers panel
  - Toggle visibility, adjust display range, gamma, colormap per channel
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
            for i, (gsplats, ch_config) in enumerate(zip(gsplats_list, CHANNELS)):
                ch_name = ch_config["name"]
                colormap = ch_config["colormap"]

                with asection(f"Adding {ch_name} (layer)"):
                    # Transform: shared centroid so channels stay aligned
                    gsplats = gsplats.translate(-shared_centroid)
                    gsplats = gsplats.scale_intensity(0.1)

                    n_splats = len(gsplats.amplitudes)

                    scene.add_gsplats(
                        name=f"gsplats_{ch_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                        centers=gsplats.centers,
                        amplitudes=gsplats.amplitudes,
                        cholesky_factors=gsplats.cholesky_factors,
                        dim_order=["y", "x"],
                        opacity=1.0,
                        blending_mode="additive",
                        layer=True,
                        colormap=colormap,
                    )
                    aprint(f"  Added {n_splats:,} splats with colormap='{colormap}'")

            # --- Overlays ---
            # Title
            scene.add_text(
                "CODEX Pancreas",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Info
            scene.add_text(
                "12-channel multiplexed fluorescence",
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

# Max channels to show in round-trip comparison (full set is 12, which is huge)
_ROUNDTRIP_MAX_CHANNELS = 3


def show_roundtrip_comparison(
    tiff_dir: Path,
    gsplats_list: list[GSplatData],
) -> None:
    """Show original vs round-trip reconstructed 2D images side by side.

    Reloads a subset of channels from disk to avoid keeping all 12 in memory.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        aprint(
            "matplotlib is required for --show-roundtrip. Install with: pip install matplotlib"
        )
        return

    n_show = min(_ROUNDTRIP_MAX_CHANNELS, len(gsplats_list))

    with asection(
        f"Round-trip reconstruction comparison ({n_show}/{len(gsplats_list)} channels)"
    ):
        images = []
        reconstructions = []
        for i in range(n_show):
            ch_config = CHANNELS[i]
            gsplats = gsplats_list[i]

            with asection(f"Ch{i}: {ch_config['name']}"):
                image = load_channel(tiff_dir, ch_config)
                recon = gsplats.render_to_volume(shape=image.shape, device=DEVICE)
                images.append(image)
                reconstructions.append(recon)

                mse = float(np.mean((image - recon) ** 2))
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
                aprint(f"  PSNR: {psnr:.2f} dB, MSE: {mse:.6g}")

        fig, axes = plt.subplots(n_show, 3, figsize=(14, 4.5 * n_show), squeeze=False)

        for i in range(n_show):
            image = images[i]
            recon = reconstructions[i]
            ch_config = CHANNELS[i]
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
            f"Round-Trip Comparison ({n_show} of {N_CHANNELS} channels)  "
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
    aprint("GSplats Demo: 2D CODEX Pancreas (12-Channel Multiplexed)")
    aprint("=" * 70)
    aprint("Tiled per-channel fitting + per-channel layers + Web viewer")
    if TARGET_SIZE > 0:
        aprint(f"Target size: {TARGET_SIZE} px (longest axis)")
    else:
        aprint(f"Full resolution: {ORIGINAL_WIDTH:,} x {ORIGINAL_HEIGHT:,} px")
    aprint(f"Channels: {N_CHANNELS}")
    aprint(
        f"Tile: {TILE_SIZE} px, overlap: {OVERLAP} px, seeds/tile: {SEEDS_PER_TILE:,}"
    )
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_2d_codex_pancreas.luxar.zarr"

    # Serve-only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    # Download and extract
    warn_if_no_cuda_gpu()
    tiff_dir = ensure_extracted()

    # Tiled fitting per channel (one at a time to save memory)
    gsplats_list = fit_all_channels(tiff_dir)

    if len(gsplats_list) < N_CHANNELS:
        aprint(f"Error: Need {N_CHANNELS} channels, got {len(gsplats_list)}")
        return

    # Optional round-trip visualisation
    if SHOW_ROUNDTRIP:
        show_roundtrip_comparison(tiff_dir, gsplats_list)

    # Create scene (centering and intensity scaling happen per-channel inside)
    scene_path = create_luxar_scene(gsplats_list, output_path)

    # Summary
    total_splats = sum(len(g.amplitudes) for g in gsplats_list)
    total_pixels = ORIGINAL_WIDTH * ORIGINAL_HEIGHT * N_CHANNELS
    pixel_bytes = total_pixels * 2  # uint16 original
    # Floats per 2D splat: d + d*(d+1)/2 + 2 + 3 (colors) = 2 + 3 + 2 + 3 = 10
    splats_bytes = total_splats * 10 * 4
    compression = pixel_bytes / splats_bytes if splats_bytes > 0 else 0

    aprint("")
    aprint("=" * 70)
    aprint("12-Channel 2D Tiled Compression Summary")
    aprint("=" * 70)
    aprint(f"Channels:          {N_CHANNELS}")
    aprint(f"Pixels/channel:    {ORIGINAL_WIDTH * ORIGINAL_HEIGHT:,}")
    aprint(f"Total pixels:      {total_pixels:,}")
    aprint(f"Total splats:      {total_splats:,}")
    aprint(f"Raw data (uint16): {pixel_bytes / (1024**2):.1f} MB")
    aprint(f"Splat data:        {splats_bytes / (1024**2):.1f} MB")
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
