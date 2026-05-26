#!/usr/bin/env python3
"""GSplats Demo: OpenCell MAP4 — Microtubule-Associated Protein (Fluorescence Microscopy)

Visualises a 2-channel 3D confocal stack of endogenously tagged MAP4 protein
in human HEK293T cells as Gaussian splats with per-channel colours.

================================================================================
MULTI-CHANNEL GAUSSIAN SPLATTING — OPENCELL MAP4 (CYTOSKELETON)
================================================================================

This demo fits Gaussian splats to a 2-channel confocal z-stack from the
OpenCell project, showing the microtubule-associated protein MAP4 alongside
Hoechst-stained nuclei.

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source:  OpenCell (https://opencell.sf.czbiohub.org/)
Target: MAP4 — Microtubule associated protein 4 (ENSG00000047849)
URL:    https://opencell.sf.czbiohub.org/target/828
Format: Multi-page TIFF, 16-bit, 2-channel (ZCYX)
Size:   51 x 600 x 600 voxels (Z x Y x X), ~70 MB

Imaging:
--------
Cell line:  HEK293T (split-fluorescent protein tagging)
Modality:   3D spinning-disk confocal microscopy
Localization: Cytoskeleton (microtubule network)

Channels:
---------
  0: MAP4-GFP       — Target protein (microtubule network, cyan)
  1: Hoechst 33342  — Nuclear stain (blue)

How to Cite:
------------
Cho, N.H., Bhatt, D.P. et al. (2022). OpenCell: Endogenous tagging for the
cartography of human cellular organization. Science, 375(6585), eabi6983.
DOI: 10.1126/science.abi6983

WORKFLOW:
=========

1. **Load** 2-channel z-stack from local TIFF file
2. **Fit** Gaussian splats per channel
3. **Add** each channel as a separate layer with ``layer=True``
4. **Visualise** — press L to open the Layers panel for per-channel control

USAGE:
======
    python demo_gsplats_3d_opencell_map4.py [--recompute] [--no-serve] [--serve-only]

Options:
    --data-path=PATH:   Path to a local OpenCell TIFF (overrides auto-download)
    --recompute:        Force re-fitting from scratch
    --no-serve:         Generate scene without launching viewer
    --serve-only:       Just serve a previously generated scene
    --show-roundtrip:   Show matplotlib comparison of original vs reconstructed volumes

The TIFF is auto-downloaded from the OpenCell S3 bucket on first run.
You can also provide a local path with: --data-path=<path_to_tiff>

Output:
    - Scene saved to:  datasets/demos/gsplats_3d_opencell_map4.zarr
    - Automatically opens in browser
"""

# Enable MPS->CPU fallback for unsupported PyTorch ops (must be before torch import)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
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

N_CHANNELS = 2

# Channel configuration with colours
CHANNELS = [
    {"index": 0, "name": "MAP4-GFP (Microtubules)", "color": (0.0, 1.0, 0.8)},  # Cyan
    {"index": 1, "name": "Hoechst (Nuclei)", "color": (0.3, 0.3, 1.0)},  # Blue
]

# Progressive fitting parameters
MAX_SPLATS = 81000
MAX_SPLATS_PER_PASS = 16000
ITERS_PER_PASS = 5000
PSNR_PATIENCE = 0.1

# Data source URL (OpenCell S3 bucket)
TIFF_URL = (
    "https://czb-opencell.s3.amazonaws.com/microscopy/raw/"
    "MAP4_ENSG00000047849/"
    "OC-FOV_MAP4_ENSG00000047849_CID000828_FID00002848_stack.tif"
)

# Cache location
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_opencell_map4"

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
SHOW_ROUNDTRIP = "--show-roundtrip" in sys.argv

# Arbol logging depth
Arbol.max_depth = 10
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Auto-detected on first fit
DEVICE = None


# =============================================================================
# Data Loading
# =============================================================================


def load_opencell_data(tiff_path: Path) -> list[np.ndarray]:
    """Load the 2-channel OpenCell z-stack.

    Args:
        tiff_path: Path to the downloaded OpenCell TIFF.

    Returns:
        List of 2 channel volumes, each float32 normalised to [0, 1].
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError(
            "tifffile is required for this demo.\nInstall with: pip install tifffile"
        )

    with asection("Loading OpenCell TIFF"):
        aprint(f"Reading {tiff_path.name}...")
        data = tifffile.imread(str(tiff_path))
        aprint(f"Raw shape: {data.shape}, dtype: {data.dtype}")

        # Expected: (Z, C, Y, X) = (51, 2, 600, 600)
        if data.ndim == 4 and data.shape[1] == N_CHANNELS:
            aprint(f"Layout: (Z, C, Y, X) = {data.shape}")
            channels_axis = 1
        elif data.ndim == 4 and data.shape[0] == N_CHANNELS:
            aprint(f"Layout: (C, Z, Y, X) = {data.shape}")
            channels_axis = 0
        elif data.ndim == 3:
            n_planes = data.shape[0]
            if n_planes % N_CHANNELS == 0:
                z_size = n_planes // N_CHANNELS
                data = data.reshape(N_CHANNELS, z_size, data.shape[1], data.shape[2])
                aprint(f"Reshaped to (C, Z, Y, X): {data.shape}")
                channels_axis = 0
            else:
                raise ValueError(
                    f"Cannot split {n_planes} planes into {N_CHANNELS} channels."
                )
        else:
            raise ValueError(f"Unexpected shape: {data.shape}")

        # Split and normalise channels
        volumes = []
        for ch_idx, ch_config in enumerate(CHANNELS):
            if channels_axis == 0:
                V = np.array(data[ch_idx], dtype=np.float32)
            else:
                V = np.array(data[:, ch_idx], dtype=np.float32)

            # Robust normalisation (clip outliers for better contrast)
            p_low, p_high = np.percentile(V, [1, 99.5])
            V = np.clip(V, p_low, p_high)
            V = (V - p_low) / (p_high - p_low + 1e-8)

            aprint(
                f"  Ch{ch_idx} ({ch_config['name']}): {V.shape}, "
                f"range [{V.min():.3f}, {V.max():.3f}]"
            )
            volumes.append(V)

        del data

    return volumes


# =============================================================================
# GSplats Fitting
# =============================================================================


def fit_channel(
    volume: np.ndarray,
    channel_name: str,
    cache_file: Path,
) -> GSplatData:
    """Fit gsplats to a single channel.

    Args:
        volume: 3D float32 volume (Z, Y, X), normalised to [0, 1].
        channel_name: Name for logging.
        cache_file: Where to cache the fitted result.

    Returns:
        Fitted GSplatData.
    """
    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats

    global DEVICE
    if DEVICE is None:
        from luxar.utils.demos import detect_device

        DEVICE = detect_device()

    aprint(
        f"Fitting {channel_name} (progressive: max {MAX_SPLATS} splats, "
        f"{MAX_SPLATS_PER_PASS}/pass, {ITERS_PER_PASS} iters/pass)..."
    )
    aprint(f"  Volume: {volume.shape}, Device: {DEVICE}")

    result = fit_progressive_gaussian_splats(
        volume,
        max_splats=MAX_SPLATS,
        max_splats_per_pass=MAX_SPLATS_PER_PASS,
        iters_per_pass=ITERS_PER_PASS,
        psnr_patience=PSNR_PATIENCE,
        device=DEVICE,
        verbose=True,
        enable_dynamic_ops=True,
        cull_retention=0.99,
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


def fit_all_channels(volumes: list[np.ndarray]) -> list[GSplatData]:
    """Fit gsplats to all channels."""
    with asection("Fitting GSplats per channel"):
        gsplats_list = []

        for i, (volume, ch_config) in enumerate(zip(volumes, CHANNELS)):
            ch_name = ch_config["name"]
            cache_file = CACHE_DIR / f"opencell_map4_ch{i}.gsplats.zarr.zip"

            with asection(f"Channel {i}: {ch_name}"):
                gsplats = fit_channel(volume, ch_name, cache_file)
                gsplats_list.append(gsplats)

        return gsplats_list


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(
    gsplats_list: list[GSplatData], output_path: Path | None = None
) -> Path:
    """Create Luxar scene with per-channel gsplats layers.

    Each channel is added as a separate gsplats node with ``layer=True``,
    so the viewer's Layers panel (press L) provides per-channel visibility,
    display range, gamma, and blending mode controls.

    Args:
        gsplats_list: List of per-channel GSplatData (one per channel).
        output_path: Output .zarr path (default: demos output dir).

    Returns:
        Path to saved scene.
    """
    if output_path is None:
        output_path = get_demos_output_dir() / "gsplats_3d_opencell_map4.zarr"

    with asection("Creating Luxar Scene (per-channel layers)"):
        aprint(f"Output: {output_path.name}")

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            dims = Dimensions(
                [
                    Dimension("x", unit="px", display=True),
                    Dimension("y", unit="px", display=True),
                    Dimension("z", unit="px", display=True),
                ]
            )
            scene = compiler.create_scene(dimensions=dims)

            scene.attrs["title"] = "GSplats: OpenCell MAP4 (Microtubule Cytoskeleton)"
            scene.attrs["description"] = """
OpenCell MAP4 — Microtubule-Associated Protein
==================================================

3D confocal z-stack of endogenously GFP-tagged MAP4 protein in HEK293T
cells, represented as Gaussian splats with per-channel layers.

Data Source:
  - OpenCell (https://opencell.sf.czbiohub.org/target/828)
  - MAP4: Microtubule associated protein 4 (cytoskeleton)
  - Spinning-disk confocal, 51 z-slices x 600 x 600

Channels (each is a layer — press L):
  - Cyan: MAP4-GFP (microtubule network)
  - Blue: Hoechst 33342 (nuclei)

Citation:
  Cho et al. (2022). OpenCell. Science, 375(6585), eabi6983.

Controls:
  - Press L to open the Layers panel
  - Click eye icon to toggle channel visibility
  - Adjust [min, max] display range per channel
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
            """

            # Compute shared centroid so channels stay aligned
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

            # Add each channel as a separate layer
            for i, (gsplats, ch_config) in enumerate(
                zip(gsplats_list, CHANNELS[: len(gsplats_list)])
            ):
                ch_name = ch_config["name"]
                color = ch_config["color"]

                with asection(f"Adding {ch_name} (layer)"):
                    gsplats = gsplats.translate(-shared_centroid)
                    gsplats = gsplats.scale_intensity(0.1)

                    n_splats = len(gsplats.amplitudes)
                    colors = np.tile(np.array(color, dtype=np.float32), (n_splats, 1))

                    scene.add_gsplats(
                        name=f"gsplats_ch{i}",
                        centers=gsplats.centers,
                        amplitudes=gsplats.amplitudes,
                        cholesky_factors=gsplats.cholesky_factors,
                        colors=colors,
                        dim_order=["z", "y", "x"],
                        opacity=1.0,
                        blending_mode="additive",
                        layer=True,
                    )
                    aprint(f"  Added {n_splats:,} splats with layer=True")

            # Overlay annotations
            scene.add_text(
                "OpenCell MAP4",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                "Fluorescence microscopy",
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
    volumes: list[np.ndarray],
    gsplats_list: list[GSplatData],
) -> None:
    """Show original vs round-trip reconstructed volumes side by side.

    Renders each channel's gsplats back to a volume at the original resolution,
    then displays the middle z-slice of original, reconstructed, and absolute
    difference using matplotlib.

    Args:
        volumes: Original per-channel volumes (Z, Y, X), float32 in [0, 1].
        gsplats_list: Fitted GSplatData per channel.
    """
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

                # Compute basic quality metrics
                mse = float(np.mean((volume - recon) ** 2))
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
                aprint(f"  PSNR: {psnr:.2f} dB, MSE: {mse:.6g}")

        # Plot: 3 columns (original, reconstructed, |difference|) x n_channels rows
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
    aprint("GSplats Demo: OpenCell MAP4 — Microtubule Cytoskeleton")
    aprint("=" * 70)
    aprint("Per-channel fitting + Layers panel for per-channel control")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_3d_opencell_map4.zarr"

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
        "gsplats_opencell_map4",
        [
            "opencell_map4_ch0.gsplats.zarr.zip",
            "opencell_map4_ch1.gsplats.zarr.zip",
        ],
        recompute=RECOMPUTE,
    )

    volumes = None

    if precomputed is not None:
        gsplats_list = precomputed
    else:
        # Resolve TIFF: CLI arg > cached file > auto-download
        tiff_path = None
        for arg in sys.argv:
            if arg.startswith("--data-path="):
                tiff_path = Path(arg.split("=", 1)[1])
                break

        cached_tiff = CACHE_DIR / "opencell_map4_stack.tif"
        if tiff_path is not None and not cached_tiff.exists():
            import shutil

            aprint(f"Caching TIFF to {cached_tiff}")
            shutil.copy2(tiff_path, cached_tiff)
        elif tiff_path is None and cached_tiff.exists():
            tiff_path = cached_tiff
        elif tiff_path is None:
            # Auto-download from OpenCell S3 bucket
            import tempfile
            import urllib.request

            with asection("Downloading OpenCell MAP4 TIFF (~70 MB)"):
                aprint(f"URL: {TIFF_URL}")
                tmp_fd, tmp_path = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tif.tmp")
                os.close(tmp_fd)
                try:
                    urllib.request.urlretrieve(TIFF_URL, tmp_path)
                    os.replace(tmp_path, cached_tiff)
                except BaseException:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    raise
                aprint(f"Saved to {cached_tiff}")
            tiff_path = cached_tiff

        warn_if_no_cuda_gpu()
        volumes = load_opencell_data(tiff_path)

        if len(volumes) < N_CHANNELS:
            aprint(f"Error: Need {N_CHANNELS} channels, got {len(volumes)}")
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

    # Create scene with per-channel layers (centering + intensity scaling done inside)
    scene_path = create_luxar_scene(gsplats_list, output_path)

    # Summary
    if volumes is not None:
        total_splats = sum(len(g.amplitudes) for g in gsplats_list)
        total_voxels = sum(v.size for v in volumes)
        volume_bytes = total_voxels * 4
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
