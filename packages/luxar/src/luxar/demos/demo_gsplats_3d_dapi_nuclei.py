#!/usr/bin/env python3
"""GSplats Demo: 3D DAPI-Stained Nuclei from Microscopy Data

Demonstrates Gaussian Splatting compression of real 3D microscopy data with
interactive web visualization.

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

DATA SOURCE:
============
Image Data Resource (IDR) - https://idr.openmicroscopy.org/
Dataset: 6001240 - DAPI-stained nuclei
Format: OME-ZARR 5D (Time × Channel × Z × Y × X)
Resolution: Downscaled to 128³ for this demo

COMPRESSION METRICS:
====================
Typical results for 128³ volume:
- Raw volume: ~8 MB (float32)
- Fitted splats: ~0.3-0.5 MB (100-150 splats × 11 floats each)
- Compression: ~20-25x
- Visual quality: Excellent (captures nuclear shapes)

USAGE:
======
    python demo_gsplats_3d_dapi_nuclei.py [--no-cache] [--no-serve]

Options:
    --no-cache: Force re-fitting even if cached result exists
    --no-serve: Don't auto-launch viewer after scene creation
    --serve-only: Skip fitting, just serve existing scene

Output:
    - Scene saved to: examples/gsplats_3d_dapi_nuclei_example.zarr
    - Cache saved to: examples/.cache/gsplats_dapi_fit.npz
    - Automatically opens in browser at http://localhost:8000

Controls:
    - Mouse drag to rotate
    - Scroll to zoom
    - Right-click drag to pan
    - 'C' to toggle fly controls

"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zarr
from arbol import Arbol, aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.fit_result import GSplatData
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig

# =============================================================================
# Configuration
# =============================================================================

ZARR_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
DAPI_CHANNEL = 1  # DAPI is channel 1 in this dataset
TARGET_SIZE = 128  # Downscale to manageable size
TIME_POINT = 0  # First time point

# Fitting parameters
N_ITERS = 1500  # Good balance of quality vs speed
DEVICE = None  # Auto-detect (cuda/mps/cpu)
TRUNCATE = 3.0  # 3-sigma truncation

# Output paths
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
CACHE_DIR = EXAMPLES_DIR / ".cache"
CACHE_FILE = CACHE_DIR / "gsplats_dapi_fit.npz"
OUTPUT_PATH = EXAMPLES_DIR / "gsplats_3d_dapi_nuclei_example.zarr"

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


def load_dapi_data():
    """Load and preprocess DAPI microscopy data from IDR."""
    with asection("Loading DAPI microscopy data"):
        aprint(f"📦 Source: {ZARR_URL}")
        aprint("🔬 Channel: DAPI (nuclear stain)")
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


def fit_or_load_gsplats(volume):
    """Fit gsplats to volume, using cache if available."""
    with asection("GSplats Fitting"):
        # Check cache
        if CACHE_FILE.exists() and not NO_CACHE:
            aprint(f"📦 Loading cached fit from: {CACHE_FILE.name}")
            try:
                cache = np.load(CACHE_FILE)
                # Reconstruct GSplatData from cache
                result = GSplatData(
                    centers=cache["centers"],
                    cholesky_factors=cache["cholesky_factors"],
                    amplitudes=cache["amplitudes"],
                    sharpnesses=cache["sharpnesses"],
                    stats={},  # Empty stats for cached data
                )
                aprint(f"✓ Loaded {len(result.amplitudes)} cached splats")
                return result
            except Exception as e:
                aprint(f"⚠ Cache load failed: {e}")
                aprint("Re-fitting...")

        # Fit gsplats
        aprint(f"Fitting Gaussian Splats ({N_ITERS} iterations)...")
        aprint(f"Device: {DEVICE or 'auto-detect'}")

        dynamic_config = DynamicOpsConfig()

        result = fit_gaussian_splats(
            volume,
            n_iters=N_ITERS,
            truncate=TRUNCATE,
            device=DEVICE,
            verbose=True,
            enable_dynamic_ops=True,
            dynamic_config=dynamic_config,
            napari_movie=False,  # No visualization during fitting
        )

        n_splats = len(result.amplitudes)
        aprint(f"✓ Fitted {n_splats} splats")
        aprint(f"  Centers: {result.centers.shape}")
        aprint(f"  Cholesky: {result.cholesky_factors.shape}")

        # Cache result
        aprint(f"💾 Caching fit to: {CACHE_FILE.name}")
        np.savez(
            CACHE_FILE,
            centers=result.centers,
            cholesky_factors=result.cholesky_factors,
            amplitudes=result.amplitudes,
            sharpnesses=result.sharpnesses,
        )

        return result


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(gsplats_data):
    """Create Luxar scene with gsplats."""
    with asection("Creating Luxar Scene"):
        aprint(f"Output: {OUTPUT_PATH.name}")

        with LuxarZarrCompiler(OUTPUT_PATH) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Add scene metadata
            scene.attrs["title"] = "GSplats: 3D DAPI-Stained Nuclei"
            scene.attrs["description"] = """
3D Gaussian Splatting Demo - Real Microscopy Data
==================================================

This scene demonstrates Gaussian Splat compression of DAPI-stained cell nuclei
from confocal microscopy imaging.

Data: Image Data Resource (IDR) dataset 6001240
Resolution: 128×128×128 voxels
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
                opacity=0.8,
                blending_mode="additive",
            )

        aprint(f"✓ Scene saved: {OUTPUT_PATH}")
        return OUTPUT_PATH


def add_reference_points(scene, volume, sample_rate=0.01):
    """Add sampled points from original volume as reference (optional)."""
    aprint(f"Adding reference points (sample rate: {sample_rate*100:.1f}%)...")

    # Sample volume at points above threshold
    threshold = np.percentile(volume, 90)
    coords = np.argwhere(volume > threshold)
    intensities = volume[coords[:, 0], coords[:, 1], coords[:, 2]]

    # Subsample
    n_total = len(coords)
    n_sample = int(n_total * sample_rate)
    indices = np.random.choice(n_total, size=n_sample, replace=False)

    sampled_coords = coords[indices].astype(np.float32)
    sampled_intensities = intensities[indices]

    # Normalize intensities to colors (grayscale)
    intensities_norm = (sampled_intensities - sampled_intensities.min()) / (
        sampled_intensities.max() - sampled_intensities.min() + 1e-8
    )
    colors = np.stack([intensities_norm] * 3, axis=1).astype(np.float32)

    scene.add_points(
        name="reference_points",
        positions=sampled_coords,
        colors=colors,
        radii=np.full(n_sample, 0.3, dtype=np.float32),
        sharpness=np.full(n_sample, 8.0, dtype=np.float32),
        opacity=0.3,
        blending_mode="additive",
    )

    aprint(f"✓ Added {n_sample:,} reference points")


# =============================================================================
# Napari Viewing
# =============================================================================


def view_with_napari(volume, gsplats_data):
    """Open original volume and gsplat rendering in napari for comparison.

    Args:
        volume: Original 3D volume
        gsplats_data: Fitted gsplats
    """
    try:
        import napari
    except ImportError:
        aprint("⚠️  napari not installed, skipping napari view")
        aprint("   Install with: pip install napari[all]")
        return

    with asection("Opening in napari"):
        aprint("Launching napari for comparison...")
        aprint("  Layer 1: Original DAPI volume")
        aprint("  Layer 2: GSplats rendering")

        viewer = napari.Viewer(title="GSplats vs Original - DAPI Nuclei")

        # Add original volume
        viewer.add_image(
            volume,
            name="Original DAPI",
            colormap="green",
            opacity=0.8,
            blending="additive",
        )

        # Render gsplats to volume for comparison
        aprint("Rendering gsplats to volume...")
        from luxar.gsplats.models.gsplats import render_gaussians_numpy

        rendered = render_gaussians_numpy(volume.shape, gsplats_data, truncate=3.0)

        viewer.add_image(
            rendered,
            name="GSplats Rendering",
            colormap="magenta",
            opacity=0.8,
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
    aprint("\n" + "=" * 70)
    aprint("🚀 Launching Luxar Viewer...")
    aprint("=" * 70)
    aprint(f"Scene: {scene_path}")
    aprint("URL: http://localhost:8000")
    aprint("\nPress Ctrl+C to stop the server")
    aprint("=" * 70 + "\n")

    try:
        # Launch luxar serve
        cmd = ["luxar", "serve", str(scene_path), "--viewer"]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        aprint("\n✓ Server stopped")
    except Exception as e:
        aprint(f"\n⚠ Error: {e}")
        aprint("\nYou can manually serve with:")
        aprint(f"  luxar serve {scene_path} --viewer")


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: 3D DAPI-Stained Nuclei")
    aprint("=" * 70)
    aprint("Real microscopy data + Gaussian Splatting + Web visualization")
    aprint("")

    # Serve only mode
    if SERVE_ONLY:
        if OUTPUT_PATH.exists():
            aprint("Serve-only mode: Launching viewer...")
            serve_scene(OUTPUT_PATH)
            return
        else:
            aprint(f"❌ Scene not found: {OUTPUT_PATH}")
            aprint("Run without --serve-only to generate first")
            return

    # Load data
    volume = load_dapi_data()

    # Fit or load gsplats
    gsplats_data = fit_or_load_gsplats(volume)

    # Apply transformations for better viewing
    with asection("Applying transformations"):
        aprint("Centering at center-of-mass...")
        gsplats_data = gsplats_data.center_at_centroid()
        centroid_check = (gsplats_data.centers.T @ gsplats_data.amplitudes) / gsplats_data.amplitudes.sum()
        aprint(f"✓ Centered (centroid: [{centroid_check[0]:.3f}, {centroid_check[1]:.3f}, {centroid_check[2]:.3f}])")

        aprint("Reducing brightness by 10x for better visualization...")
        gsplats_data = gsplats_data.scale_intensity(0.1)
        aprint(f"✓ Brightness scaled to 0.1x (amplitude range: [{gsplats_data.amplitudes.min():.4f}, {gsplats_data.amplitudes.max():.4f}])")

    # Create scene
    scene_path = create_luxar_scene(gsplats_data)

    # Open in napari for comparison
    view_with_napari(volume, gsplats_data)

    # Summary
    n_splats = len(gsplats_data.amplitudes)
    volume_bytes = volume.size * 4  # float32
    splats_bytes = n_splats * 11 * 4  # 11 floats per splat
    compression = volume_bytes / splats_bytes

    aprint("\n" + "=" * 70)
    aprint("📊 Compression Summary")
    aprint("=" * 70)
    aprint(f"Volume: {volume.shape} = {volume.size:,} voxels")
    aprint(f"Splats: {n_splats} × 11 floats = {n_splats * 11:,} floats")
    aprint(f"Raw size: {volume_bytes / 1024 / 1024:.2f} MB")
    aprint(f"Splat size: {splats_bytes / 1024:.2f} KB")
    aprint(f"Compression ratio: {compression:.1f}:1")
    aprint(f"Space savings: {(1 - 1/compression)*100:.1f}%")
    aprint("=" * 70)

    # Launch viewer
    if not NO_SERVE:
        aprint("\nLaunching viewer in 2 seconds...")
        time.sleep(2)
        serve_scene(scene_path)
    else:
        aprint(f"\n✓ Scene ready at: {scene_path}")
        aprint(f"To view: luxar serve {scene_path} --viewer")


if __name__ == "__main__":
    main()
