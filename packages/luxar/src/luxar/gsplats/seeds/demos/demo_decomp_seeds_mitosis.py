#!/usr/bin/env python3
"""
Human mitosis seed generation comparison demo.

This demo compares three seed generation methods on the scikit-image human mitosis dataset:
1. Grid method: Uniform grid seeding for baseline coverage
2. Decomposition method: Scale-hierarchical detection via image decomposition
3. Edges method: Edge-based seeding with anisotropic shapes

All methods return GSplatData with scale-informed Gaussian shapes.
Displays seed locations side-by-side in napari for visual comparison.
"""

import sys

import napari
from arbol import Arbol, aprint, asection
from skimage import color, data, img_as_float32

from luxar.gsplats.seeds import (
    combine_seeds,
    seed_from_decomposition,
    seed_from_edges,
    seed_from_grid,
)

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("Decomposition Seeds Demo (napari disabled)")
    aprint("Running all computations without napari visualization...")


# Setup Arbol
Arbol.max_depth = 4


with asection("Seed Generation Methods Comparison"):
    aprint("Comparing seed generation methods on human mitosis histology data")

    with asection("Loading and preprocessing data"):
        # Load human_mitosis and prepare grayscale image
        img = data.human_mitosis()  # RGB image
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = color.rgb2gray(img) * 100  # -> float in [0, 100]
        V = img_as_float32(img)

        # Crop to smaller region for faster demo
        V = V[100:356, 100:356]  # 256x256

        aprint(f"Preprocessed image: {V.shape}")
        aprint(f"Data range: [{V.min():.4f}, {V.max():.4f}]")

    # ======================================================================
    # SEED GENERATION (all methods return GSplatData with scale-informed shapes)
    # ======================================================================

    with asection("Generating seeds"):
        # Method 1: Grid seeding
        with asection("Grid method"):
            result_grid = seed_from_grid(V, spacing=8.0)
            seeds_grid = result_grid.centers
            aprint(f"Generated {len(seeds_grid)} seeds")
            aprint(
                f"  Scale info in cholesky_factors shape: {result_grid.cholesky_factors.shape}"
            )

        # Method 2: Decomposition-based
        with asection("Decomposition method"):
            result_decomp = seed_from_decomposition(V, verbose=True)
            seeds_decomp = result_decomp.centers
            aprint(f"Generated {len(seeds_decomp)} seeds")
            aprint(
                f"  Scale info in cholesky_factors shape: {result_decomp.cholesky_factors.shape}"
            )

        # Method 3: Edge-based seeding
        with asection("Edges method"):
            result_edges = seed_from_edges(V, min_distance=3.0)
            seeds_edges = result_edges.centers
            aprint(f"Generated {len(seeds_edges)} seeds")
            aprint(
                f"  Scale info in cholesky_factors shape: {result_edges.cholesky_factors.shape}"
            )

        # Method 4: Combined (as used in fit_gaussian_splats with method='auto')
        with asection("Combined method"):
            seeds_combined = combine_seeds(
                seeds_decomp,  # Decomposition first (global structure)
                seeds_edges,  # Then edges (boundaries)
                seeds_grid,  # Finally grid (coverage)
                min_distance=3.0,
            )
            aprint(f"Generated {len(seeds_combined)} seeds")
            aprint(
                f"  ({len(seeds_decomp)} decomp + "
                f"{len(seeds_edges)} edges + "
                f"{len(seeds_grid)} grid -> "
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

            # Method 1: Grid seeds
            viewer.add_points(
                seeds_grid,
                name=f"Grid ({len(seeds_grid)})",
                size=3,
                face_color="yellow",
                border_color="white",
                border_width=0.3,
                symbol="disc",
                visible=False,  # Start hidden
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

            # Method 3: Edge seeds
            viewer.add_points(
                seeds_edges,
                name=f"Edges ({len(seeds_edges)})",
                size=3,
                face_color="magenta",
                border_color="white",
                border_width=0.3,
                symbol="disc",
                visible=False,  # Start hidden
            )

            # Method 4: Combined seeds (as used in fitting)
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
            aprint(f"Grid:          {len(seeds_grid):4d} seeds (yellow)")
            aprint(f"Decomposition: {len(seeds_decomp):4d} seeds (cyan)")
            aprint(f"Edges:         {len(seeds_edges):4d} seeds (magenta)")
            aprint(
                f"Combined:      {len(seeds_combined):4d} seeds (lime) <- Used in fitting"
            )
            aprint("=" * 70)
            aprint("\nNapari viewer launched!")
            aprint(
                "   - Lime points show combined method (used in fit_gaussian_splats)"
            )
            aprint("   - Toggle layers to compare individual methods")

            napari.run()
    else:
        aprint("\nDemo completed successfully (napari visualization disabled)")
