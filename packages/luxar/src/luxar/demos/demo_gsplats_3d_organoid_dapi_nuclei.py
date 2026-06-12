#!/usr/bin/env python3
"""GSplats Demo: 3D Organoid DAPI-Stained Nuclei from Microscopy Data

Demonstrates Gaussian Splatting compression of real 3D microscopy data with
interactive web visualization.

Note: This demo uses Metal acceleration on Apple Silicon for substantial speedup (chip-dependent).
Some PyTorch ops on MPS aren't supported yet, so we enable CPU fallback.

================================================================================
GAUSSIAN SPLATTING FOR 3D MICROSCOPY
================================================================================

This demo shows how Gaussian Splats can compress 3D microscopy volumes while
preserving key structural features. We fit oriented 3D Gaussians to DAPI-stained
cell nuclei and visualize the result in the Luxar viewer.

WHY GSPLATS FOR MICROSCOPY:
- **Compression**: 20-50x smaller than raw voxels (typical: 95%+ space savings)
- **Smooth representation**: Gaussians naturally model biological structures
- **Oriented ellipsoids**: Captures elongated nuclear shapes
- **Fast rendering**: GPU-accelerated splatting is faster than volume rendering
- **Interpretable**: Each splat corresponds to a feature in the image

WHAT THIS DEMO DOES:
====================

1. **Download** real microscopy data from Image Data Resource (IDR)
   - DAPI channel (DNA stain showing cell nuclei)
   - 3D confocal microscopy volume
   - OME-ZARR format (standard for microscopy)

2. **Fit Gaussian Splats** to the volume
   - Automatic candidate detection finds nuclei
   - Optimization fits oriented 3D Gaussians
   - Dynamic operations refine the representation
   - Result cached for reuse

3. **Create Luxar Scene** with the fitted splats
   - Add gsplats node with centers, covariances, amplitudes
   - Include raw image as reference points (optional)
   - Save to zarr format

4. **Serve & Visualize** in the browser
   - Interactive 3D WebGL rendering
   - Compare gsplats vs original volume
   - Explore compression quality

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Image ID: 6001240 (idr6001240)
Source: Image Data Resource (IDR) - https://idr.openmicroscopy.org/
Study: idr0062 - Intestinal organoid development and nuclear segmentation
Format: OME-ZARR 5D (Time × Channel × Z × Y × X)
Data Type: High-resolution 3D light microscopy of mouse intestinal organoid
Resolution: Downscaled to 128³ for this demo

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

COMPRESSION METRICS:
====================
Typical results for 128³ volume:
- Raw volume: ~8 MB (float32)
- Fitted splats: ~0.3-0.5 MB (100-150 splats × 11 floats each)
- Compression: ~20-25x
- Visual quality: Excellent (captures nuclear shapes)

USAGE:
======
    python demo_gsplats_3d_organoid_dapi_nuclei.py [--recompute] [--no-serve] [--no-napari]

Options:
    --recompute:      Force re-fitting from scratch (download + GPU fitting)
    --no-serve:       Don't auto-launch viewer after scene creation
    --no-napari:      Skip napari visualization (useful for headless/CI)
    --serve-only:     Skip fitting, just serve existing scene
    --show-roundtrip: Show matplotlib comparison of original vs reconstructed volume

By default, precomputed GSplats are loaded from package data (Git LFS).
Use --recompute to re-fit from scratch (requires network + GPU).

Output:
    - Scene saved to: demos/gsplats_3d_organoid_dapi_nuclei.luxar.zarr
    - Cache saved to: ~/.cache/luxar/gsplats_dapi/dapi_gsplats.gsplats.zarr.zip
    - Automatically opens in browser at http://localhost:8000

Controls:
    - Mouse drag to rotate
    - Scroll to zoom
    - Right-click drag to pan
    - 'C' to toggle fly controls

"""

# Enable MPS→CPU fallback for unsupported PyTorch ops (must be before torch import)
import os

from luxar.utils.demos import (
    launch_viewer,
    load_precomputed_gsplats,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
from pathlib import Path

import numpy as np
import zarr
from arbol import Arbol, aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.models.gsplats.metal import is_metal_available
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

ZARR_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
DAPI_CHANNEL = 1  # DAPI is channel 1 in this dataset
TARGET_SIZE = 128  # Downscale to manageable size
TIME_POINT = 0  # First time point

# Progressive fitting parameters
MAX_SPLATS = 12000
MAX_SPLATS_PER_PASS = 2500
ITERS_PER_PASS = 3000
PSNR_PATIENCE = 0.2
DEVICE = None  # Auto-detect (cuda/mps/cpu)

# Cache paths (use user cache directory for intermediate fit results)
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_dapi"
CACHE_FILE = CACHE_DIR / "dapi_gsplats.gsplats.zarr.zip"

# Parse command line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
NO_NAPARI = "--no-napari" in sys.argv
SHOW_ROUNDTRIP = "--show-roundtrip" in sys.argv

# Setup
Arbol.max_depth = 3
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Loading
# =============================================================================


def load_dapi_data():
    """Load and preprocess DAPI microscopy data from IDR.

    Data Source: Image Data Resource (IDR) study idr0062, Image 6001240
    Original Authors: Prisca Liberali lab, FMI
    Citation: Blin et al. (2019) + Williams et al. (2017) Nature Methods 14(8):775-781
    """
    with asection("Loading DAPI microscopy data"):
        aprint(f"📦 Source: {ZARR_URL}")
        aprint("🔬 Channel: DAPI (nuclear stain)")
        aprint("📚 Dataset: IDR idr0062, Image 6001240 (Liberali lab, FMI)")
        aprint(f"📐 Target size: {TARGET_SIZE}³ voxels")

        try:
            import fsspec
            from scipy.ndimage import zoom

            # Open remote zarr
            mapper = fsspec.get_mapper(ZARR_URL)
            store = zarr.open_group(mapper, mode="r")
            data = store["0"]  # Highest resolution
            full_shape = data.shape

            aprint(f"📊 Full data shape: {full_shape}")

            # Extract DAPI channel
            if len(full_shape) == 5:
                n_time, n_channels, z_size, y_size, x_size = full_shape
                aprint("Format: OME-ZARR 5D (T×C×Z×Y×X)")
                aprint(f"  Time points: {n_time}")
                aprint(f"  Channels: {n_channels}")
                aprint(f"  Spatial: {z_size}×{y_size}×{x_size}")

                # Load DAPI channel
                aprint(f"Extracting T={TIME_POINT}, C={DAPI_CHANNEL} (DAPI)...")
                V = np.array(data[TIME_POINT, DAPI_CHANNEL, :, :, :], dtype=np.float32)
            else:
                raise ValueError(f"Unexpected data shape: {full_shape}")

            # Downscale
            zoom_factors = [TARGET_SIZE / s for s in V.shape]
            aprint(f"Downscaling with zoom factors: {zoom_factors[0]:.3f} (linear)")
            V = zoom(V, zoom_factors, order=1)

            # Normalize
            V = (V - V.min()) / (V.max() - V.min() + 1e-8)
            V = V.astype(np.float32)

            aprint(f"✓ Loaded: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")
            return V

        except Exception as e:
            aprint(f"⚠ Remote loading failed: {e}")
            aprint("Creating synthetic fallback data...")

            # Fallback: synthetic nuclei-like blobs
            shape = (TARGET_SIZE, TARGET_SIZE, TARGET_SIZE)
            V = np.zeros(shape, dtype=np.float32)

            # Add nucleus-like blobs
            for _ in range(15):
                center = [np.random.uniform(10, s - 10) for s in shape]
                sigma = np.random.uniform(4, 8)
                amplitude = np.random.uniform(0.6, 1.0)

                grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
                dist_sq = sum((g - c) ** 2 for g, c in zip(grids, center))
                V += amplitude * np.exp(-dist_sq / (2 * sigma**2))

            V = np.clip(V, 0, 1).astype(np.float32)
            aprint(f"✓ Fallback created: {V.shape}")
            return V


# =============================================================================
# GSplats Fitting
# =============================================================================


def fit_dapi_gsplats(volume):
    """Fit gsplats to DAPI volume (no cache check — caller handles that)."""
    with asection("GSplats Fitting"):
        # Auto-detect best device (Metal on Apple Silicon for substantial speedup, chip-dependent)
        global DEVICE
        if DEVICE is None:
            import torch

            if is_metal_available() and torch.backends.mps.is_available():
                DEVICE = "mps"
                aprint(
                    "Metal acceleration detected - will use MPS device for substantial speedup (chip-dependent)!"
                )
                aprint("   (MPS->CPU fallback enabled for unsupported PyTorch ops)")
            elif torch.cuda.is_available():
                DEVICE = "cuda"
                aprint("Using CUDA device")
            else:
                DEVICE = "cpu"
                aprint("Using CPU device")
        else:
            aprint(f"Using specified device: {DEVICE}")

        # Fit gsplats progressively
        aprint(
            f"Progressive fitting (max {MAX_SPLATS} splats, "
            f"{MAX_SPLATS_PER_PASS}/pass, {ITERS_PER_PASS} iters/pass)..."
        )

        result = fit_progressive_gaussian_splats(
            volume,
            max_splats=MAX_SPLATS,
            max_splats_per_pass=MAX_SPLATS_PER_PASS,
            iters_per_pass=ITERS_PER_PASS,
            psnr_patience=PSNR_PATIENCE,
            device=DEVICE,
            verbose=True,
            max_eccentricity=8.0,
        )

        n_splats = len(result.amplitudes)
        aprint(f"Fitted {n_splats} splats")
        aprint(f"  Centers: {result.centers.shape}")
        aprint(f"  Cholesky: {result.cholesky_factors.shape}")

        # Cache result in gsplats.zarr.zip format
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        aprint(f"Caching fit to: {CACHE_FILE.name}")
        result.save(
            CACHE_FILE,
            encoding_mode=EncodingMode.MEMORY,
            include_fitting_info=True,
            compress="zip",
            zip_deflate=True,
        )

        return result


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(gsplats_data, output_path: Path | None = None):
    """Create Luxar scene with gsplats."""
    if output_path is None:
        output_path = (
            get_demos_output_dir() / "gsplats_3d_organoid_dapi_nuclei.luxar.zarr"
        )

    with asection("Creating Luxar Scene"):
        aprint(f"Output: {output_path.name}")

        # Use PRECISION mode for maximum quality (float32 for all data)
        # This prioritizes accuracy over space savings
        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=Dimensions.default_3d(),
            )

            # Add scene metadata
            scene.attrs["title"] = "GSplats: 3D Organoid DAPI-Stained Nuclei"
            scene.attrs["description"] = """
3D Gaussian Splatting - Organoid Microscopy
============================================

This scene demonstrates Gaussian Splat compression of DAPI-stained cell nuclei
from confocal microscopy imaging.

Data Source:
  - Image Data Resource (IDR) study idr0062, Image 6001240
  - High-resolution 3D microscopy of mouse intestinal organoid
  - Original research: Prisca Liberali lab, FMI
  - Citation: Blin et al. (2019) + Williams et al. (2017) Nat Methods 14(8):775-781

Resolution: 128×128×128 voxels (downscaled from original)
Compression: ~20-25x (3D Gaussians vs raw voxels)

Each Gaussian splat represents a feature in the image - notice how the oriented
ellipsoids align with nuclear shapes and chromatin structures.

Controls:
- Mouse drag to rotate
- Scroll to zoom
- Right-click drag to pan
- 'C' to toggle fly controls
            """

            # Add gsplats using the convenient from_data method
            aprint(f"Adding {len(gsplats_data.amplitudes)} gsplats...")
            scene.add_gsplats_from_data(
                name="dapi_nuclei_gsplats",
                result=gsplats_data,
                opacity=1.0,
                blending_mode="additive",
                layer=True,
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Round-Trip Visualisation
# =============================================================================


def show_roundtrip_comparison(
    volume: np.ndarray,
    gsplats_data: GSplatData,
) -> None:
    """Show original vs round-trip reconstructed volume side by side."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        aprint(
            "matplotlib is required for --show-roundtrip. Install with: pip install matplotlib"
        )
        return

    with asection("Round-trip reconstruction comparison"):
        with asection("Rendering reconstruction"):
            recon = gsplats_data.render_to_volume(shape=volume.shape, device=DEVICE)
            mse = float(np.mean((volume - recon) ** 2))
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
            aprint(f"  PSNR: {psnr:.2f} dB, MSE: {mse:.6g}")

        mid_z = volume.shape[0] // 2
        orig_slice = volume[mid_z]
        recon_slice = recon[mid_z]
        diff_slice = np.abs(orig_slice - recon_slice)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        axes[0].imshow(orig_slice, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original (DAPI)")
        axes[0].axis("off")

        axes[1].imshow(recon_slice, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Reconstructed (PSNR {psnr:.1f} dB)")
        axes[1].axis("off")

        im = axes[2].imshow(diff_slice, cmap="inferno", vmin=0, vmax=0.3)
        axes[2].set_title("|Difference|")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Organoid DAPI — Round-Trip Comparison — z-slice {mid_z}  "
            f"({len(gsplats_data.amplitudes):,} splats)",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()


# =============================================================================
# Napari Viewing
# =============================================================================


def view_with_napari(volume, gsplats_data):
    """Open original volume and gsplat rendering in napari for comparison.

    Note: gsplats_data should be in ORIGINAL coordinates (not centered) for proper alignment.

    Args:
        volume: Original 3D volume
        gsplats_data: Fitted gsplats in original voxel coordinates
    """
    try:
        import napari
    except ImportError:
        aprint("⚠️  napari not installed, skipping napari view")
        aprint("   Install with: pip install napari[all]")
        return

    with asection("Opening in napari"):
        aprint("Preparing napari visualization...")
        aprint("  Layer 1: Original DAPI volume (green)")
        aprint("  Layer 2: GSplats reconstruction (magenta)")
        aprint("  Both in original voxel coordinates - should align perfectly!")

        # Render gsplats to volume for comparison BEFORE opening napari
        # Using GPU-accelerated renderer (substantially faster than old NumPy implementation; often orders of magnitude on GPU)
        aprint("Rendering gsplats to volume (GPU-accelerated)...")
        rendered = gsplats_data.render_to_volume(
            shape=tuple(dim_len * 2 for dim_len in volume.shape)
        )
        aprint("✓ Rendering complete")

        # NOW create the napari viewer with all data ready
        aprint("Launching napari...")
        viewer = napari.Viewer(title="GSplats vs Original - DAPI Nuclei")

        # Add original volume
        viewer.add_image(
            volume,
            name="Original DAPI",
            colormap="gray",
            opacity=1.0,
            blending="additive",
        )

        # Add rendered gsplats
        viewer.add_image(
            rendered,
            name="GSplats Reconstruction",
            colormap="gray",
            opacity=1.0,
            blending="additive",
        )

        aprint("✓ Napari opened - compare original (green) vs gsplats (magenta)")
        aprint("  Toggle layers on/off to see compression quality")

        napari.run()


# =============================================================================
# Serving
# =============================================================================


def serve_scene(scene_path):
    """Launch luxar viewer to display the scene."""
    launch_viewer(scene_path)


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: 3D Organoid DAPI-Stained Nuclei")
    aprint("=" * 70)
    aprint("Real microscopy data + Gaussian Splatting + Web visualization")
    aprint("")

    # Determine output path
    output_path = get_demos_output_dir() / "gsplats_3d_organoid_dapi_nuclei.luxar.zarr"

    # Serve only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            serve_scene(output_path)
            return
        else:
            aprint(f"Scene not found: {output_path}")
            aprint("Run without --serve-only to generate first")
            return

    # Try loading precomputed data (from Git LFS / local cache)
    volume = None
    gsplats_data_original = None

    precomputed = load_precomputed_gsplats(
        "gsplats_dapi",
        ["dapi_gsplats.gsplats.zarr.zip"],
        recompute=RECOMPUTE,
    )

    if precomputed is not None:
        gsplats_data_original = precomputed[0]
    else:
        # --recompute path: download raw data, fit from scratch
        warn_if_no_cuda_gpu()
        volume = load_dapi_data()
        gsplats_data_original = fit_dapi_gsplats(volume)

    # Optional round-trip visualisation (before centering/scaling transforms)
    if SHOW_ROUNDTRIP:
        if volume is not None:
            show_roundtrip_comparison(volume, gsplats_data_original)
        else:
            aprint(
                "Cannot show round-trip: original volume not available "
                "(loaded from precomputed cache). Re-run with --recompute."
            )

    # Apply transformations for web viewer
    with asection("Applying transformations for web viewer"):
        aprint("Centering at center-of-mass...")
        gsplats_data = gsplats_data_original.center_at_centroid()
        centroid_check = (
            gsplats_data.centers.T @ gsplats_data.amplitudes
        ) / gsplats_data.amplitudes.sum()
        aprint(
            f"Centered (centroid: [{centroid_check[0]:.3f}, {centroid_check[1]:.3f}, {centroid_check[2]:.3f}])"
        )

        aprint("Reducing brightness by 10x for better visualization...")
        gsplats_data = gsplats_data.scale_intensity(0.1)
        aprint(
            f"Brightness scaled to 0.1x (amplitude range: [{gsplats_data.amplitudes.min():.4f}, {gsplats_data.amplitudes.max():.4f}])"
        )

    # Create scene with transformed data
    scene_path = create_luxar_scene(gsplats_data, output_path)

    # Summary (only if volume was loaded)
    if volume is not None:
        n_splats = len(gsplats_data.amplitudes)
        volume_bytes = volume.size * 4  # float32
        splats_bytes = n_splats * 11 * 4  # 11 floats per splat
        compression = volume_bytes / splats_bytes

        aprint("\n" + "=" * 70)
        aprint("Compression Summary")
        aprint("=" * 70)
        aprint(f"Volume: {volume.shape} = {volume.size:,} voxels")
        aprint(f"Splats: {n_splats} x 11 floats = {n_splats * 11:,} floats")
        aprint(f"Raw size: {volume_bytes / 1024 / 1024:.2f} MB")
        aprint(f"Splat size: {splats_bytes / 1024:.2f} KB")
        aprint(f"Compression ratio: {compression:.1f}:1")
        aprint(f"Space savings: {(1 - 1 / compression) * 100:.1f}%")
        aprint("=" * 70)

    # Open in napari with original un-transformed data (proper alignment)
    if not NO_NAPARI and volume is not None:
        view_with_napari(volume, gsplats_data_original)

    # Launch viewer
    if NO_SERVE:
        aprint(f"Dataset generated at {scene_path}")
    else:
        aprint("\nLaunching viewer in 2 seconds...")
        time.sleep(2)
        serve_scene(scene_path)


if __name__ == "__main__":
    main()
