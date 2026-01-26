#!/usr/bin/env python3
"""GSplats Demo: Multi-Channel 3D Organoid Microscopy (Quick Start)

Quick-start demo using precomputed Gaussian splats from real organoid microscopy data.
Perfect for testing and demos without needing to run the full fitting pipeline.

================================================================================
MULTI-CHANNEL GAUSSIAN SPLATTING - PRECOMPUTED ORGANOIDS
================================================================================

This demo uses precomputed Gaussian splats fitted to multi-channel organoid
microscopy data. It's fast to run and demonstrates multi-channel visualization
with distinct colors for each channel.

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

1. **Load precomputed gsplats** from included .gsplats.zarr.zip files
   - Channel 0: First fluorescent marker
   - Channel 1: DAPI (DNA stain showing cell nuclei)
   - Pre-fitted using the full IDR dataset
   - Compressed zarr format for efficient storage

2. **Merge with channel colors**
   - Channel 0: Magenta (1.0, 0.0, 0.5)
   - Channel 1: Cyan (0.0, 1.0, 0.5)
   - Uses GSplatData.merge_with_channel_colors()

3. **Visualize** in the Luxar viewer
   - Additive blending shows channel overlap
   - Distinct colors reveal co-localization

PRECOMPUTED DATA:
=================

The included .gsplats.zarr.zip files contain pre-fitted Gaussian splats:
- organoids_gsplats_ch0.gsplats.zarr.zip: Channel 0 (magenta marker, ~609 KB)
- organoids_gsplats_ch1.gsplats.zarr.zip: Channel 1 (DAPI nuclei, ~630 KB)

These files are stored in Git LFS. Make sure you have pulled LFS files:
    git lfs pull

If you want to recompute from scratch (slow, requires network):
    python demo_gsplats_3d_organoid_multichannel_from_idr.py

USAGE:
======
    python demo_gsplats_3d_organoid_multichannel_precomputed.py [--no-serve]

Options:
    --no-serve: Don't auto-launch viewer after scene creation
    --serve-only: Skip loading, just serve existing scene

Output:
    - Scene saved to: demos/gsplats_3d_organoid_multichannel_precomputed.zarr
    - Automatically opens in browser at http://localhost:8000

"""

import sys
import time
from pathlib import Path

from arbol import Arbol, aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Precomputed data paths
DATA_DIR = Path(__file__).parent / "data"
GSPLATS_FILES = [
    DATA_DIR / "organoids_gsplats_ch0.gsplats.zarr.zip",
    DATA_DIR / "organoids_gsplats_ch1.gsplats.zarr.zip",
]

# Channel configuration with colors
CHANNELS = [
    {"index": 0, "name": "Channel 0", "color": (1.0, 0.0, 0.5)},  # Magenta
    {"index": 1, "name": "DAPI", "color": (0.0, 1.0, 0.5)},  # Cyan
]

# Parse command line flags
NO_SERVE = "--no-serve" in sys.argv
SERVE_ONLY = "--serve-only" in sys.argv

# Setup
Arbol.max_depth = 5


# =============================================================================
# Data Loading
# =============================================================================


def load_precomputed_gsplats():
    """Load precomputed Gaussian splats from .gsplats.zarr.zip files.

    Data Source: Image Data Resource (IDR) study idr0062, Image 6001240
    Original Authors: Prisca Liberali lab, FMI
    Citation: Blin et al. (2019) + Williams et al. (2017) Nature Methods 14(8):775-781

    Returns:
        list[GSplatData]: List of GSplatData objects, one per channel
    """
    with asection("Loading precomputed gsplats"):
        aprint("Source: Precomputed from IDR dataset 6001240")
        aprint("Dataset: IDR idr0062, Image 6001240 (Liberali lab, FMI)")
        aprint(f"Loading {len(GSPLATS_FILES)} channels from .gsplats.zarr.zip files")

        gsplats_list = []

        for i, (gsplat_file, ch_config) in enumerate(zip(GSPLATS_FILES, CHANNELS)):
            ch_name = ch_config["name"]

            if not gsplat_file.exists():
                raise FileNotFoundError(
                    f"Precomputed gsplat file not found: {gsplat_file}\n"
                    f"Make sure you have pulled Git LFS files:\n"
                    f"  git lfs pull\n"
                    f"Or run the full compute version:\n"
                    f"  python demo_gsplats_3d_organoid_multichannel_from_idr.py"
                )

            with asection(f"Channel {i}: {ch_name}"):
                aprint(f"Loading: {gsplat_file.name}")

                try:
                    # Load from .gsplats.zarr.zip format
                    gsplat_data = GSplatData.load(gsplat_file, include_stats=True)

                    n_splats = len(gsplat_data.amplitudes)
                    aprint(f"✓ Loaded {n_splats:,} splats")
                    aprint(f"  Centers shape: {gsplat_data.centers.shape}")
                    aprint(f"  Cholesky shape: {gsplat_data.cholesky_factors.shape}")

                    gsplats_list.append(gsplat_data)

                except Exception as e:
                    raise RuntimeError(f"Failed to load {gsplat_file.name}: {e}")

        aprint(f"\n✓ Loaded all {len(gsplats_list)} channels")
        return gsplats_list


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(merged_gsplats, output_path: Path | None = None):
    """Create Luxar scene with merged multi-channel gsplats."""
    if output_path is None:
        output_path = (
            get_demos_output_dir() / "gsplats_3d_organoid_multichannel_precomputed.zarr"
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
precomputed Gaussian splats with per-channel colors.

Data Source:
  - Image Data Resource (IDR) study idr0062, Image 6001240
  - High-resolution 3D microscopy of mouse intestinal organoid
  - Original research: Prisca Liberali lab, FMI
  - Citation: Blin et al. (2019) + Williams et al. (2017) Nat Methods 14(8):775-781

Each channel was fitted independently then merged:
- Magenta: Channel 0 (fluorescent marker)
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
                name="organoids_multichannel_gsplats",
                result=merged_gsplats,
                opacity=1.0,
                blending_mode="additive",
            )

        aprint(f"✓ Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: Multi-Channel 3D Organoids (Quick Start)")
    aprint("=" * 70)
    aprint("Precomputed gsplats + color-coded merge + Web visualization")
    aprint("")

    # Determine output path
    output_path = (
        get_demos_output_dir() / "gsplats_3d_organoid_multichannel_precomputed.zarr"
    )

    # Serve only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
            return
        else:
            aprint(f"⚠ Scene not found: {output_path}")
            aprint("Run without --serve-only to generate first")
            return

    # Load precomputed gsplats
    gsplats_list = load_precomputed_gsplats()

    # Merge with channel colors
    with asection("Merging channels with colors"):
        channel_colors = [ch["color"] for ch in CHANNELS[: len(gsplats_list)]]
        aprint(f"Channel colors: {channel_colors}")

        merged = GSplatData.merge_with_channel_colors(
            gsplats_list,
            channel_colors=channel_colors,
        )

        aprint(f"✓ Merged: {len(merged.amplitudes)} total splats")
        splats_per_channel = merged.stats.get("splats_per_channel")
        if splats_per_channel:
            aprint(f"  Per channel: {splats_per_channel}")

    # Apply transformations for web viewer
    with asection("Applying transformations"):
        aprint("Centering at center-of-mass...")
        merged = merged.center_at_centroid()

        aprint("Reducing brightness by 10x for better visualization...")
        merged = merged.scale_intensity(0.1)

    # Create scene
    scene_path = create_luxar_scene(merged, output_path)

    # Summary
    total_splats = len(merged.amplitudes)

    aprint("\n" + "=" * 70)
    aprint("Multi-Channel Organoid Visualization")
    aprint("=" * 70)
    aprint(f"Channels: {len(gsplats_list)}")
    aprint(f"Total splats: {total_splats:,}")
    aprint("Data source: IDR study idr0062, Image 6001240")
    aprint(f"Scene file: {scene_path.name}")
    aprint("=" * 70)

    # Launch viewer
    if NO_SERVE:
        aprint(f"\n✓ Dataset generated at {scene_path}")
    else:
        aprint("\nLaunching viewer in 2 seconds...")
        time.sleep(2)
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
