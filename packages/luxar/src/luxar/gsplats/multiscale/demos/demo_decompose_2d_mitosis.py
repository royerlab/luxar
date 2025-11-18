#!/usr/bin/env python3
"""
Human Mitosis Multi-Scale Decomposition Demo

Demonstrates multi-scale decomposition on the scikit-image human mitosis
biological histology dataset, showing how cellular features are distributed
across different scales using napari visualization.
"""

import sys

import napari
import numpy as np
from arbol import aprint, asection
from skimage import color, data, img_as_float32

from luxar.gsplats.multiscale import (
    decompose_image,
    show_optimization_movie,
    upsample_for_visualization,
)

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("🔬 Human Mitosis Multi-Scale Decomposition Demo (napari disabled)")
    aprint("Running all computations without napari visualization...")

# ======= Demo knobs =======
SCALES = [1, 2, 4, 8, 16]  # Scale factors to use
N_ITERS = 5000  # Number of optimization iterations
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps"
# ==========================


with asection("Human Mitosis Multi-Scale Decomposition Demo"):
    aprint("🔬 Multi-scale decomposition of biological histology data")

    with asection("Loading and preprocessing data"):
        # Load human_mitosis image
        img = data.human_mitosis()  # RGB image
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = color.rgb2gray(img) * 100  # -> float in [0, 100]
        V = img_as_float32(img)

        # Crop the image to a smaller region for faster demo
        V = V[100:356, 100:356]  # Crop to 256x256

        aprint(f"Preprocessed human mitosis image: {V.shape}")
        aprint(f"Data range: [{V.min():.4f}, {V.max():.4f}]")

    with asection(f"Decomposing into {len(SCALES)} scales"):
        aprint(f"Scales: {SCALES}")
        aprint(f"Parameters: n_iters={N_ITERS}")

        # Decompose with movie recording enabled
        scales_list, stats = decompose_image(
            V,
            scales=SCALES,
            n_iters=N_ITERS,
            napari_movie=(not NO_NAPARI),
            movie_every=10,  # Record every 10 iterations for reasonable frame count
            movie_max_frames=None,  # No limit on frames
            device=DEVICE,
            verbose=True,
        )

        aprint("Decomposition complete!")
        aprint(f"Final reconstruction error: {stats['final_error']:.6e}")
        aprint(f"Time elapsed: {stats['time_seconds']:.2f} seconds")

    with asection("Preparing visualization data"):
        # Upsample all scales to original resolution for visualization
        # Use same interpolation as optimization
        scales_upsampled = []
        interpolation_mode = stats.get("interpolation", "cubic")
        aprint(
            f"  Using '{interpolation_mode}' interpolation for upsampling (matches optimization)"
        )
        for i, (scale, img_scale) in enumerate(zip(SCALES, scales_list)):
            aprint(f"  Upsampling scale {scale}x from {img_scale.shape} to {V.shape}")
            if img_scale.shape != V.shape:
                img_upsampled = upsample_for_visualization(
                    img_scale, V.shape, interpolation_mode
                )
            else:
                img_upsampled = img_scale
            scales_upsampled.append(img_upsampled)

        # Compute reconstruction
        reconstruction = np.sum(scales_upsampled, axis=0)

        # Compute residual
        residual = V - reconstruction
        abs_residual = np.abs(residual)

        aprint(f"Reconstruction MSE: {np.mean((reconstruction - V) ** 2):.6e}")
        aprint(f"Max absolute residual: {abs_residual.max():.6e}")

    # Energy distribution analysis
    energy_dist = stats["energy_distribution"]
    aprint("\n" + "=" * 60)
    aprint("Energy Distribution (coarse → fine):")
    aprint("=" * 60)
    for i, (scale, energy_frac) in enumerate(zip(SCALES, energy_dist)):
        energy_pct = energy_frac * 100
        bar_length = int(energy_pct / 2)  # Scale for visualization
        bar = "█" * bar_length
        aprint(f"Scale {scale:2d}x: {energy_pct:5.1f}% {bar}")
    aprint(f"Total: {sum(energy_dist) * 100:.1f}%")
    aprint("=" * 60)

    if not NO_NAPARI:
        # Napari visualization
        aprint("\nLaunching napari viewer...")
        viewer = napari.Viewer()

        # Determine contrast limits from original image (shared across all scales)
        contrast_limits = [0, float(V.max())]

        # Add original image
        viewer.add_image(
            V,
            name="human_mitosis (input)",
            colormap="magma",
            contrast_limits=contrast_limits,
        )

        # Add reconstruction
        viewer.add_image(
            reconstruction,
            name="reconstruction",
            colormap="magma",
            contrast_limits=contrast_limits,
            visible=True,
        )

        # Add each scale component (upsampled) - use same contrast limits for consistency
        for i, (scale, img_upsampled) in enumerate(zip(SCALES, scales_upsampled)):
            energy_pct = energy_dist[i] * 100
            viewer.add_image(
                img_upsampled,
                name=f"scale_{scale}x ({energy_pct:.1f}%)",
                colormap="viridis",
                contrast_limits=contrast_limits,
                blending="additive",
                opacity=0.8,
                visible=True,
            )

        # Add residual
        viewer.add_image(
            abs_residual,
            name="absolute_residual",
            colormap="inferno",
            contrast_limits=[0, max(1e-12, float(abs_residual.max()))],
            visible=True,
        )

        # Add signed residual for debugging
        viewer.add_image(
            residual,
            name="signed_residual",
            colormap="bwr",
            contrast_limits=[-abs_residual.max(), abs_residual.max()],
            visible=True,
        )

        # Enable tile/grid mode for side-by-side comparison
        viewer.grid.enabled = True
        viewer.grid.shape = (-1, 3)  # Auto rows, 3 columns

        # Set up text overlay with energy distribution
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            f"Human Mitosis Multi-Scale Decomposition | Scales: {SCALES} | "
            f"Energy: {' → '.join([f'{e:.1%}' for e in energy_dist])} | "
            f"Reconstruction MSE: {stats['final_error']:.6e}"
        )

        # Console tips
        aprint("\n📊 Visualization Tips:")
        aprint("  • Opened in tile/grid mode for side-by-side comparison")
        aprint("  • All scales use same contrast limits for consistent comparison")
        aprint("  • Toggle layers on/off to compare scales")
        aprint("  • 'human_mitosis (input)' = original histology image")
        aprint("  • 'reconstruction' = sum of all scales")
        aprint(f"  • 'scale_Nx' = individual scale components (N={SCALES})")
        aprint("  • 'absolute_residual' = |original - reconstruction|")
        aprint("  • Use additive blending to see scale contributions")

        aprint("\n🎯 Energy Distribution Insights:")
        coarse_energy = energy_dist[-1]
        fine_energy = energy_dist[0]
        if coarse_energy > 0.5:
            aprint(f"  ✓ Good: {coarse_energy:.1%} energy in coarsest scale")
            aprint("    → Low-frequency cellular structures captured at coarse resolution")
        elif fine_energy > 0.5:
            aprint(
                f"  ⚠ Warning: {fine_energy:.1%} energy in finest scale (trivial solution)"
            )
            aprint("    → Try increasing alpha or energy_weight")
        else:
            aprint("  → Energy well distributed across scales")

        aprint("\n📊 Compression Potential:")
        # Calculate bits per pixel for each scale
        FLOAT_BITS = 32
        IMAGE_BITS = V.size * FLOAT_BITS
        for i, scale in enumerate(SCALES):
            scale_pixels = scales_list[i].size
            scale_bits = scale_pixels * FLOAT_BITS
            compression_pct = 100.0 * (1.0 - (scale_bits / IMAGE_BITS))
            energy_pct = energy_dist[i] * 100
            aprint(
                f"  Scale {scale:2d}x: {scale_pixels:>7,d} pixels | "
                f"Size: {compression_pct:5.1f}% smaller | "
                f"Energy: {energy_pct:5.1f}%"
            )

        aprint("\n🔬 Biological Interpretation:")
        aprint("  • Coarse scales: Overall cell shapes and large-scale structures")
        aprint("  • Medium scales: Nuclear boundaries and chromatin patterns")
        aprint("  • Fine scales: Fine chromatin details and texture")

        napari.run()

        # Show optimization convergence movie
        if stats["movie_frames"] is not None:
            aprint("\n🎬 Showing optimization convergence movie...")
            # Pass interpolation mode from stats to ensure movie matches optimization
            interpolation_mode = stats.get("interpolation", "cubic")
            show_optimization_movie(
                stats["movie_frames"], V.shape, interpolation=interpolation_mode
            )

    else:
        aprint("\n✅ Demo completed successfully (napari visualization disabled)")
