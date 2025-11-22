#!/usr/bin/env python3
"""
Human mitosis seed generation comparison demo.

This demo compares two seed generation methods on the scikit-image human mitosis dataset:
1. Multiscale Gaussian method (standard): Multi-scale detection with spatial redundancy
2. Decomposition method (new): Scale-hierarchical detection via image decomposition

Displays seed locations side-by-side in napari for visual comparison.
"""

import sys

import napari
from arbol import Arbol, aprint, asection
from skimage import color, data, img_as_float32

from luxar.gsplats.seeds import (
    combine_seeds,
    find_seeds_multiscale_decomposition,
    find_seeds_multiscale_gaussian,
)

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("🔬 Decomposition Seeds Demo (napari disabled)")
    aprint("Running all computations without napari visualization...")


# Setup Arbol
Arbol.max_depth = 4


with asection("Seed Generation Methods Comparison"):
    aprint("🔬 Comparing seed generation methods on human mitosis histology data")

    with asection("Loading and preprocessing data"):
        # Load human_mitosis and prepare grayscale image
        img = data.human_mitosis()  # RGB image
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = color.rgb2gray(img) * 100  # → float in [0, 100]
        V = img_as_float32(img)

        # Crop to smaller region for faster demo
        V = V[100:356, 100:356]  # 256x256

        aprint(f"Preprocessed image: {V.shape}")
        aprint(f"Data range: [{V.min():.4f}, {V.max():.4f}]")

    # ======================================================================
    # SEED GENERATION
    # ======================================================================

    with asection("Generating seeds"):
        # Method 1: Multiscale Gaussian
        with asection("Multiscale Gaussian method"):
            seeds_multiscale = find_seeds_multiscale_gaussian(V)
            aprint(f"Generated {len(seeds_multiscale)} seeds")

        # Method 2: Decomposition-based
        with asection("Decomposition method"):
            seeds_decomp = find_seeds_multiscale_decomposition(V, verbose=True)
            aprint(f"Generated {len(seeds_decomp)} seeds")

        # Method 3: Combined (as used in fit_gaussian_splats)
        with asection("Combined method"):
            seeds_combined = combine_seeds(
                seeds_decomp,  # Decomposition first (global structure)
                seeds_multiscale,  # Then multiscale (local features)
            )
            aprint(f"Generated {len(seeds_combined)} seeds")
            aprint(
                f"  ({len(seeds_decomp)} decomp + "
                f"{len(seeds_multiscale)} multiscale → "
                f"{len(seeds_combined)} after dedup)"
            )

    # ======================================================================
    # VISUALIZATION
    # ======================================================================

    if not NO_NAPARI:
        with asection("Launching napari visualization"):
            viewer = napari.Viewer(title="Seed Generation Methods Comparison")

            # Original image
            viewer.add_image(V, name="Original Mitosis Image", colormap="gray")

            # Method 1: Multiscale Gaussian seeds
            viewer.add_points(
                seeds_multiscale,
                name=f"Multiscale Gaussian ({len(seeds_multiscale)})",
                size=3,
                face_color="magenta",
                border_color="white",
                border_width=0.3,
                symbol="disc",
            )

            # Method 2: Decomposition seeds
            viewer.add_points(
                seeds_decomp,
                name=f"Decomposition ({len(seeds_decomp)})",
                size=3,
                face_color="cyan",
                border_color="white",
                border_width=0.3,
                symbol="disc",
                visible=False,  # Start hidden
            )

            # Method 3: Combined seeds (as used in fitting)
            viewer.add_points(
                seeds_combined,
                name=f"Combined ({len(seeds_combined)})",
                size=3,
                face_color="lime",
                border_color="black",
                border_width=0.3,
                symbol="disc",
            )

            # Summary
            aprint("\n" + "=" * 70)
            aprint("SEED GENERATION SUMMARY")
            aprint("=" * 70)
            aprint(
                f"Multiscale Gaussian: {len(seeds_multiscale):4d} seeds (magenta)"
            )
            aprint(
                f"Decomposition:       {len(seeds_decomp):4d} seeds (cyan)"
            )
            aprint(
                f"Combined:            {len(seeds_combined):4d} seeds (lime) ← Used in fitting"
            )
            aprint("=" * 70)
            aprint("\n✅ Napari viewer launched!")
            aprint(
                "   • Lime points show combined method (used in fit_gaussian_splats)"
            )
            aprint("   • Toggle layers to compare individual methods")

            napari.run()
    else:
        aprint("\n✅ Demo completed successfully (napari visualization disabled)")
