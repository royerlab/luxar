#!/usr/bin/env python3
"""GSplats Demo: 3-Channel Mouse Embryo Heart (Acto3D / Zeiss Lightsheet 7)

Visualises a 3-channel light-sheet microscopy volume of an E13.5 mouse embryo
heart as Gaussian splats with per-channel colours.

================================================================================
MULTI-CHANNEL GAUSSIAN SPLATTING — MOUSE EMBRYO HEART
================================================================================

This demo shows the complete workflow for multi-channel Gaussian splatting
on a real light-sheet microscopy dataset of a developing mouse heart:
- Downloading the raw TIFF from Google Drive
- Splitting into 3 fluorescence channels
- Fitting Gaussian splats to each channel independently
- Merging channels with distinct colours for visualisation

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source:  Acto3D sample data (https://github.com/Acto3D/Acto3D)
Format: Multi-page TIFF, 8-bit, 3-channel interleaved
Size:   960 x 960 x 597 voxels (W x H x Z), ~1.65 GB
Voxel:  1.0635 x 1.0635 x 2.4009 um (XY x Z)

Imaging:
--------
Microscope:  Zeiss Lightsheet 7
Specimen:    E13.5 mouse embryo heart
Processing:  XY downsampled 0.5x, converted to 8-bit

Channels:
---------
  0: SYTOX Green        — Nuclear stain (all nuclei)
  1: Tomato lectin DyLight 594 — Vasculature / endothelium
  2: Anti-TNNI3 Alexa Fluor 633 — Cardiac troponin (cardiomyocytes)

How to Cite:
------------
If you use this dataset, please cite:
  Acto3D — https://github.com/Acto3D/Acto3D

WORKFLOW:
=========

1. **Download** multi-page TIFF from Google Drive (1.65 GB)
2. **Split** into 3 channels and downsample for fitting
3. **Fit** Gaussian splats per channel with GPU acceleration
4. **Merge** with per-channel colours (blue/red/green)
5. **Visualise** in the Luxar web viewer

USAGE:
======
    python demo_gsplats_3d_acto3d_heart.py [--recompute] [--no-serve] [--serve-only] [--target-size=N]

Options:
    --recompute:      Force re-fitting from scratch (download + GPU fitting)
    --no-serve:       Generate scene without launching viewer
    --serve-only:     Just serve a previously generated scene
    --target-size=N:  Downsample target per axis (default: 256)
    --data-path=PATH: Use a manually downloaded TIFF instead of Google Drive

By default, precomputed GSplats are loaded from package data (Git LFS).
Use --recompute to re-fit from scratch (requires network + GPU).

Output:
    - Scene saved to:  datasets/demos/gsplats_3d_acto3d_heart.zarr
    - Automatically opens in browser
"""

# Enable MPS->CPU fallback for unsupported PyTorch ops (must be before torch import)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
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

# Google Drive file ID for the Acto3D heart dataset
GDRIVE_FILE_ID = "1VHiLkK2O1ZrWoWX4ahPwnZfNgDQ242Kz"
EXPECTED_FILE_SIZE = 1_730_000_000  # ~1.65 GB (approximate)

# Original volume specs
N_CHANNELS = 3
ORIGINAL_SHAPE = (597, 960, 960)  # Z, Y, X per channel (already 0.5x downsampled)
ORIGINAL_VOXEL_SIZE = (2.4009, 1.0635, 1.0635)  # Z, Y, X in um

# Channel configuration with colours
CHANNELS = [
    {"index": 0, "name": "SYTOX Green (Nuclei)", "color": (0.0, 0.5, 1.0)},  # Blue
    {
        "index": 1,
        "name": "Tomato Lectin (Vasculature)",
        "color": (1.0, 0.2, 0.0),
    },  # Red
    {"index": 2, "name": "TNNI3 (Cardiac Tissue)", "color": (0.0, 1.0, 0.3)},  # Green
]

# Fitting parameters (progressive)
MAX_SPLATS = 6000
MAX_SPLATS_PER_PASS = 1000
ITERS_PER_PASS = 3000
PSNR_PATIENCE = 0.2

# Cache location
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_acto3d_heart"

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

# Parse --target-size=N
TARGET_SIZE = 256
for _arg in sys.argv:
    if _arg.startswith("--target-size="):
        TARGET_SIZE = int(_arg.split("=")[1])

# Arbol logging depth
Arbol.max_depth = 10
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Auto-detected on first fit
DEVICE = None


# =============================================================================
# Google Drive Download
# =============================================================================


def _download_from_google_drive(file_id: str, output_path: Path) -> Path:
    """Download a large file from Google Drive, handling the virus-scan confirmation.

    Args:
        file_id: Google Drive file ID.
        output_path: Where to save the downloaded file.

    Returns:
        Path to downloaded file.
    """
    import requests

    # Check if already downloaded
    if output_path.exists() and output_path.stat().st_size > EXPECTED_FILE_SIZE * 0.9:
        aprint(f"File already downloaded: {output_path.name}")
        aprint(f"  Size: {output_path.stat().st_size / (1024**3):.2f} GB")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    session = requests.Session()

    with asection("Downloading from Google Drive"):
        aprint(f"File ID: {file_id}")
        aprint(f"Expected size: ~{EXPECTED_FILE_SIZE / (1024**3):.2f} GB")

        # Strategy 1: Direct download with confirm=t (works for many large files)
        response = session.get(url, params={"confirm": "t"}, stream=True, timeout=60)

        # Strategy 2: Check cookies for download_warning token
        if response.headers.get("content-type", "").startswith("text/html"):
            aprint("Trying cookie-based confirmation...")
            confirm_token = None
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    confirm_token = value
                    break
            if confirm_token:
                response = session.get(
                    url,
                    params={"confirm": confirm_token},
                    stream=True,
                    timeout=60,
                )

        # Strategy 3: Parse the HTML confirmation page for form action + inputs
        if response.headers.get("content-type", "").startswith("text/html"):
            import html as html_mod
            import re

            aprint("Parsing confirmation page for download form...")
            page_html = response.text

            # Extract form action URL
            action_match = re.search(r'action="([^"]*)"', page_html)
            # Extract hidden form inputs (id, export, confirm, uuid)
            form_inputs = dict(
                re.findall(
                    r'<input[^>]*name="([^"]*)"[^>]*value="([^"]*)"',
                    page_html,
                )
            )

            if action_match and form_inputs:
                action_url = html_mod.unescape(action_match.group(1))
                aprint(f"Found download form with {len(form_inputs)} params")
                response = session.get(
                    action_url,
                    params=form_inputs,
                    stream=True,
                    timeout=60,
                )
            else:
                # Strategy 4: Try the usercontent endpoint directly
                aprint("Trying usercontent endpoint...")
                uc_url = (
                    f"https://drive.usercontent.google.com/download"
                    f"?id={file_id}&export=download&confirm=t"
                )
                response = session.get(uc_url, stream=True, timeout=60)

        response.raise_for_status()

        # Get total size from headers
        total_size = int(response.headers.get("content-length", 0))
        if total_size > 0:
            aprint(f"Download size: {total_size / (1024**3):.2f} GB")

        # Stream to disk
        downloaded = 0
        last_report_mb = 0
        start_time = time.time()
        chunk_size = 1024 * 1024  # 1 MB

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress every 100 MB
                    progress_mb = downloaded / (1024 * 1024)
                    if progress_mb - last_report_mb >= 100:
                        elapsed = time.time() - start_time
                        rate = (
                            downloaded / (1024 * 1024) / elapsed if elapsed > 0 else 0
                        )
                        if total_size > 0:
                            pct = downloaded / total_size * 100
                            aprint(
                                f"  {downloaded / (1024**3):.2f} / "
                                f"{total_size / (1024**3):.2f} GB "
                                f"({pct:.0f}%) - {rate:.1f} MB/s"
                            )
                        else:
                            aprint(
                                f"  {downloaded / (1024**3):.2f} GB - {rate:.1f} MB/s"
                            )
                        last_report_mb = progress_mb

        final_size = output_path.stat().st_size
        aprint(f"Download complete: {final_size / (1024**3):.2f} GB")

        # Sanity check — Google Drive HTML error pages are small
        if final_size < 1_000_000:
            output_path.unlink()
            raise RuntimeError(
                "Downloaded file is too small — likely a Google Drive error page. "
                "Try downloading manually from: "
                f"https://drive.google.com/file/d/{file_id}/view?usp=sharing "
                f"and pass --data-path=<path> (or place it at {output_path})"
            )

    return output_path


# =============================================================================
# Data Loading
# =============================================================================


def load_acto3d_heart_data() -> tuple[list[np.ndarray], tuple[float, ...]]:
    """Download and load the 3-channel Acto3D heart dataset.

    Returns:
        Tuple of (list of 3 channel volumes, effective voxel size after downsampling).
        Each volume is float32, normalised to [0, 1], shape ~(TARGET_SIZE,)*3.
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError(
            "tifffile is required for this demo.\nInstall with: pip install tifffile"
        )

    from scipy.ndimage import zoom

    # Download
    tiff_path = CACHE_DIR / "acto3d_heart_E13_5.tif"

    # Allow --data-path override for manual downloads
    for arg in sys.argv:
        if arg.startswith("--data-path="):
            tiff_path = Path(arg.split("=", 1)[1])
            aprint(f"Using manually provided data: {tiff_path}")
            break
    else:
        _download_from_google_drive(GDRIVE_FILE_ID, tiff_path)

    # Load TIFF
    with asection("Loading multi-channel TIFF"):
        aprint(f"Reading {tiff_path.name}...")
        data = tifffile.imread(str(tiff_path))
        aprint(f"Raw shape: {data.shape}, dtype: {data.dtype}")

        # Detect channel layout
        # Expected: (Z*C, H, W) = (1791, 1920, 1920) interleaved
        # or (C, Z, H, W) = (3, 597, 1920, 1920)
        # or (Z, C, H, W) = (597, 3, 1920, 1920)
        if data.ndim == 3:
            # Interleaved: (Z*C, H, W) — most common for multi-page TIFF
            n_planes = data.shape[0]
            if n_planes % N_CHANNELS == 0:
                z_size = n_planes // N_CHANNELS
                aprint(
                    f"Interleaved layout: {n_planes} planes -> {z_size} Z x {N_CHANNELS} ch"
                )
                # Zeiss multi-channel TIFFs are typically sequential:
                # all Z slices of ch0, then all Z slices of ch1, etc.
                data = data.reshape(N_CHANNELS, z_size, data.shape[1], data.shape[2])
                aprint(f"Reshaped to (C, Z, Y, X): {data.shape}")
            else:
                raise ValueError(
                    f"Cannot split {n_planes} planes into {N_CHANNELS} channels. "
                    f"Expected multiple of {N_CHANNELS}."
                )
            channels_axis = 0
        elif data.ndim == 4:
            # Find the channel axis (smallest dim that equals N_CHANNELS)
            if data.shape[0] == N_CHANNELS:
                channels_axis = 0
                aprint(f"Layout: (C, Z, Y, X) = {data.shape}")
            elif data.shape[1] == N_CHANNELS:
                channels_axis = 1
                aprint(f"Layout: (Z, C, Y, X) = {data.shape}")
            else:
                raise ValueError(
                    f"Cannot find channel axis (size {N_CHANNELS}) in shape {data.shape}"
                )
        else:
            raise ValueError(f"Expected 3D or 4D TIFF, got {data.ndim}D: {data.shape}")

        # Split channels (copy to release reference to full array)
        volumes = []
        for ch_idx, ch_config in enumerate(CHANNELS):
            if channels_axis == 0:
                V = np.array(data[ch_idx])
            else:
                V = np.array(data[:, ch_idx])

            aprint(f"  Ch{ch_idx} ({ch_config['name']}): {V.shape}")
            volumes.append(V)

        del data  # Free the full TIFF (~6.6 GB)

    # Downsample each channel
    with asection(f"Downsampling to ~{TARGET_SIZE}^3"):
        downsampled = []
        for ch_idx, (V, ch_config) in enumerate(zip(volumes, CHANNELS)):
            zoom_factors = tuple(TARGET_SIZE / s for s in V.shape)
            V = zoom(V.astype(np.float32), zoom_factors, order=1)

            # Normalise to [0, 1]
            vmin, vmax = V.min(), V.max()
            V = (V - vmin) / (vmax - vmin + 1e-8)
            V = V.astype(np.float32)

            downsampled.append(V)
            aprint(
                f"  Ch{ch_idx} ({ch_config['name']}): {V.shape}, "
                f"range [{V.min():.3f}, {V.max():.3f}]"
            )

    # Compute effective voxel size after downsampling
    effective_voxel_size = tuple(
        orig_vox * (orig_dim / TARGET_SIZE)
        for orig_vox, orig_dim in zip(ORIGINAL_VOXEL_SIZE, ORIGINAL_SHAPE)
    )
    aprint(
        f"Effective voxel size: {tuple(f'{v:.2f}' for v in effective_voxel_size)} um"
    )

    return downsampled, effective_voxel_size


# =============================================================================
# GSplats Fitting
# =============================================================================


def fit_channel(
    volume: np.ndarray,
    channel_name: str,
    cache_file: Path,
    voxel_size: tuple[float, ...],
) -> GSplatData:
    """Fit gsplats to a single channel.

    Args:
        volume: 3D float32 volume (Z, Y, X), normalised to [0, 1].
        channel_name: Name for logging.
        cache_file: Where to cache the fitted result.
        voxel_size: Physical voxel size (Z, Y, X) in um.

    Returns:
        Fitted GSplatData.
    """
    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats

    global DEVICE
    if DEVICE is None:
        from luxar.utils.demos import detect_device

        DEVICE = detect_device()

    aprint(
        f"Fitting {channel_name} (progressive: "
        f"max_splats={MAX_SPLATS}, {MAX_SPLATS_PER_PASS}/pass, "
        f"{ITERS_PER_PASS} iters/pass, patience={PSNR_PATIENCE} dB)..."
    )
    aprint(f"  Volume: {volume.shape}, Device: {DEVICE}")

    result = fit_progressive_gaussian_splats(
        volume,
        max_splats=MAX_SPLATS,
        max_splats_per_pass=MAX_SPLATS_PER_PASS,
        iters_per_pass=ITERS_PER_PASS,
        psnr_patience=PSNR_PATIENCE,
        device=DEVICE,
        voxel_size=voxel_size,
        verbose=True,
        enable_dynamic_ops=True,
    )

    n_splats = len(result.amplitudes)
    aprint(f"  Fitted {n_splats:,} splats")

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
    volumes: list[np.ndarray],
    voxel_size: tuple[float, ...],
) -> list[GSplatData]:
    """Fit gsplats to all 3 channels."""
    with asection("Fitting GSplats per channel"):
        gsplats_list = []

        for i, (volume, ch_config) in enumerate(zip(volumes, CHANNELS)):
            ch_name = ch_config["name"]
            cache_file = CACHE_DIR / f"acto3d_heart_ch{i}.gsplats.zarr.zip"

            with asection(f"Channel {i}: {ch_name}"):
                gsplats = fit_channel(volume, ch_name, cache_file, voxel_size)
                gsplats_list.append(gsplats)

        return gsplats_list


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(
    merged_gsplats: GSplatData, output_path: Path | None = None
) -> Path:
    """Create Luxar scene with merged multi-channel gsplats.

    Args:
        merged_gsplats: Merged GSplatData with per-channel colours.
        output_path: Output .zarr path (default: demos output dir).

    Returns:
        Path to saved scene.
    """
    if output_path is None:
        output_path = get_demos_output_dir() / "gsplats_3d_acto3d_heart.zarr"

    with asection("Creating Luxar Scene"):
        aprint(f"Output: {output_path.name}")

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                ]
            )
            scene = compiler.create_scene(dimensions=dims)

            scene.attrs["title"] = "GSplats: Mouse Embryo Heart E13.5 (Acto3D)"
            scene.attrs["description"] = """
3-Channel Gaussian Splatting - Mouse Embryo Heart
=====================================================

Light-sheet fluorescence microscopy of an E13.5 mouse embryo heart,
represented as Gaussian splats with per-channel colours.

Data Source:
  - Acto3D sample data (https://github.com/Acto3D/Acto3D)
  - Zeiss Lightsheet 7, 960 x 960 x 597 voxels, 3 channels
  - Voxel size: 1.06 x 1.06 x 2.40 um (XY x Z)

Channels:
  - Blue:  SYTOX Green (nuclei)
  - Red:   Tomato lectin DyLight 594 (vasculature)
  - Green: Anti-TNNI3 Alexa Fluor 633 (cardiac tissue)

Colour overlap indicates co-localisation of markers.

Controls:
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
            """

            aprint(f"Adding {len(merged_gsplats.amplitudes):,} merged gsplats...")
            scene.add_gsplats_from_data(
                name="heart_multichannel",
                result=merged_gsplats,
                opacity=1.0,
                blending_mode="additive",
            )

            # --- Overlays ---
            scene.add_text(
                "Mouse Embryo Heart (E13.5)",
                position=(0.5, 0.02),
                font_size=0.026,
                anchor="top-center",
                color="rgba(255,255,255,0.85)",
                stroke_color="black",
                stroke_width=0.002,
            )

            # Channel legend
            scene.add_html(
                '<div style="font-size:1.3vh;line-height:1.7;background:rgba(0,0,0,0.5);padding:0.5vh;border-radius:3px">'
                '<div style="font-weight:bold;color:#ccc;margin-bottom:0.3vh">Channels</div>'
                '<div><span style="color:#0088ff">\u2588</span> SYTOX Green (nuclei)</div>'
                '<div><span style="color:#ff3300">\u2588</span> Tomato Lectin (vasculature)</div>'
                '<div><span style="color:#00ff4d">\u2588</span> Anti-TNNI3 (cardiac tissue)</div>'
                "</div>",
                position=(0.02, 0.97),
                anchor="bottom-left",
            )

            scene.add_text(
                "Light-sheet \u2022 1.06\u00d71.06\u00d72.40 \u03bcm",
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
    aprint("GSplats Demo: 3-Channel Mouse Embryo Heart E13.5 (Acto3D)")
    aprint("=" * 70)
    aprint("Per-channel fitting + colour-coded merge + Web visualisation")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_3d_acto3d_heart.zarr"

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
        "gsplats_acto3d_heart",
        [
            "acto3d_heart_ch0.gsplats.zarr.zip",
            "acto3d_heart_ch1.gsplats.zarr.zip",
            "acto3d_heart_ch2.gsplats.zarr.zip",
        ],
        recompute=RECOMPUTE,
    )

    volumes = None

    if precomputed is not None:
        gsplats_list = precomputed
    else:
        # --recompute path: download raw data, fit from scratch
        warn_if_no_cuda_gpu()
        volumes, voxel_size = load_acto3d_heart_data()

        if len(volumes) < N_CHANNELS:
            aprint(f"Error: Need {N_CHANNELS} channels, got {len(volumes)}")
            return

        gsplats_list = fit_all_channels(volumes, voxel_size)

    # Merge with channel colours
    with asection("Merging channels with colours"):
        channel_colors = [ch["color"] for ch in CHANNELS[: len(gsplats_list)]]
        aprint(f"Channel colours: {channel_colors}")

        merged = GSplatData.merge_with_channel_colors(
            gsplats_list,
            channel_colors=channel_colors,
        )

        aprint(f"Merged: {len(merged.amplitudes):,} total splats")

    # Apply transformations for web viewer
    with asection("Applying transformations"):
        aprint("Centering at centre-of-mass...")
        merged = merged.center_at_centroid()

        aprint("Reducing brightness by 10x...")
        merged = merged.scale_intensity(0.1)

    # Create scene
    scene_path = create_luxar_scene(merged, output_path)

    # Summary
    if volumes is not None:
        total_splats = len(merged.amplitudes)
        total_voxels = sum(v.size for v in volumes)
        volume_bytes = total_voxels * 4  # float32
        # Floats per splat: d + d*(d+1)/2 + 2 + 3 (colors) = 3 + 6 + 2 + 3 = 14
        splats_bytes = total_splats * 14 * 4
        compression = volume_bytes / splats_bytes if splats_bytes > 0 else 0

        aprint("")
        aprint("=" * 70)
        aprint("Multi-Channel Compression Summary")
        aprint("=" * 70)
        aprint(f"Channels:          {len(volumes)}")
        aprint(f"Total voxels:      {total_voxels:,}")
        aprint(f"Total splats:      {total_splats:,}")
        aprint(f"Raw volume size:   {volume_bytes / (1024 * 1024):.1f} MB")
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
