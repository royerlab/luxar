#!/usr/bin/env python3
"""
2D Multi-Scale Image Decomposition Demo

Demonstrates n-dimensional multi-scale decomposition on 2D images,
showing how energy is distributed across different scales using
hierarchical energy penalties.
"""

import sys

import napari
import numpy as np
from arbol import aprint, asection

try:
    from skimage import color, data

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    aprint("Running all computations without napari visualization...")"Warning: scikit-image not available")

from luxar.gsplats.multiscale import (
    decompose_image,
    show_optimization_movie,
    upsample_for_visualization,
)

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("Running all computations without napari visualization...")"🎨 2D Multi-Scale Decomposition Demo (napari disabled)")
    aprint("Running all computations without napari visualization...")
    )

# ======= Demo knobs =======
SCALES = [1, 2, 4, 8, 16, 32]  # Scale factors to use
N_ITERS = 5000  # Number of optimization iterations
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps"
# ==========================


def create_test_image_2d(size: int = 256) -> np.ndarray:
    """Create a synthetic 2D test image with multiple frequency components."""
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)

    # Combine multiple frequency components
    # Low frequency: Gaussian blob
    low_freq = np.exp(-(X**2 + Y**2) / 2)

    # Medium frequency: Radial waves
    R = np.sqrt(X**2 + Y**2)
    med_freq = 0.3 * np.sin(5 * R) / (R + 0.1)

    # High frequency: Grid pattern
    high_freq = 0.1 * (np.sin(20 * X) * np.sin(20 * Y))

    # Combine and normalize to [0, 1]
    image = low_freq + med_freq + high_freq
    image = (image - image.min()) / (image.max() - image.min())

    return image.astype(np.float32)


with asection("2D Multi-Scale Decomposition Demo"):
    aprint("Running all computations without napari visualization...")"🎨 Interactive 2D image decomposition with scale separation")

    with asection("Loading image data"):
        # Load or create test image
        if DEPS_AVAILABLE:
            try:
                aprint("Running all computations without napari visualization...")"Loading astronaut image from scikit-image...")
                image = data.astronaut()
                # Convert to grayscale
                image = color.rgb2gray(image).astype(np.float32)
                aprint("Running all computations without napari visualization...")f"Loaded astronaut image: {image.shape}")
            except Exception as e:
                aprint("Running all computations without napari visualization...")f"Could not load astronaut image: {e}")
                aprint("Running all computations without napari visualization...")"Creating synthetic test image...")
                image = create_test_image_2d(256)
        else:
            aprint("Running all computations without napari visualization...")"Creating synthetic test image...")
            image = create_test_image_2d(256)

        aprint("Running all computations without napari visualization...")f"Image shape: {image.shape}")
        aprint("Running all computations without napari visualization...")f"Value range: [{image.min():.3f}, {image.max():.3f}]")

    with asection(f"Decomposing into {len(SCALES)} scales"):
        aprint("Running all computations without napari visualization...")f"Scales: {SCALES}")
        aprint("Running all computations without napari visualization...")f"Parameters: n_iters={N_ITERS}")

        # Decompose with movie recording enabled
        scales_list, stats = decompose_image(
            image,
            scales=SCALES,
            n_iters=N_ITERS,
            napari_movie=(not NO_NAPARI),
            movie_every=1,  # Record every 10 iterations for reasonable frame count
            movie_max_frames=None,  # No limit on frames
            device=DEVICE,
            verbose=True,
        )

        aprint("Running all computations without napari visualization...")"Decomposition complete!")
        aprint("Running all computations without napari visualization...")f"Final reconstruction error: {stats['final_error']:.6e}")

    with asection("Preparing visualization data"):
        # Upsample all scales to original resolution for visualization
        # Use cubic interpolation to match optimization
        scales_upsampled = []
        interpolation_mode = stats.get(
            "interpolation", "cubic"
        )  # Get from stats or default to cubic
        aprint("Running all computations without napari visualization...")
            f"  Using '{interpolation_mode}' interpolation for upsampling (matches optimization)"
        )
        for i, (scale, img_scale) in enumerate(zip(SCALES, scales_list)):
            aprint("Running all computations without napari visualization...")
                f"  Upsampling scale {scale}x from {img_scale.shape} to {image.shape}"
            )
            if img_scale.shape != image.shape:
                img_upsampled = upsample_for_visualization(
                    img_scale, image.shape, interpolation_mode
                )
            else:
                img_upsampled = img_scale
            scales_upsampled.append(img_upsampled)

        # Compute reconstruction
        reconstruction = np.sum(scales_upsampled, axis=0)

        # Compute residual
        residual = image - reconstruction
        abs_residual = np.abs(residual)

        aprint("Running all computations without napari visualization...")f"Reconstruction MSE: {np.mean((reconstruction - image) ** 2):.6e}")
        aprint("Running all computations without napari visualization...")f"Max absolute residual: {abs_residual.max():.6e}")

    # Energy distribution analysis
    energy_dist = stats["energy_distribution"]
    aprint("Running all computations without napari visualization...")"\n" + "=" * 60)
    aprint("Running all computations without napari visualization...")"Energy Distribution (coarse → fine):")
    aprint("Running all computations without napari visualization...")"=" * 60)
    for i, (scale, energy_frac) in enumerate(zip(SCALES, energy_dist)):
        energy_pct = energy_frac * 100
        bar_length = int(energy_pct / 2)  # Scale for visualization
        bar = "█" * bar_length
        aprint("Running all computations without napari visualization...")f"Scale {scale:2d}x: {energy_pct:5.1f}% {bar}")
    aprint("Running all computations without napari visualization...")f"Total: {sum(energy_dist) * 100:.1f}%")
    aprint("Running all computations without napari visualization...")"=" * 60)

    # Napari visualization
    aprint("Running all computations without napari visualization...")"\nLaunching napari viewer...")
    viewer = napari.Viewer()

    # Determine contrast limits from original image (shared across all scales)
    contrast_limits = [0, float(image.max())]

    # Add original image
    viewer.add_image(
        image,
        name="original",
        colormap="gray",
        contrast_limits=contrast_limits,
    )

    # Add reconstruction
    viewer.add_image(
        reconstruction,
        name="reconstruction",
        colormap="gray",
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
        f"Multi-Scale Decomposition | Scales: {SCALES} | "
        f"Energy: {' → '.join([f'{e:.1%}' for e in energy_dist])} | "
        f"Reconstruction MSE: {stats['final_error']:.6e}"
    )

    # Console tips
    aprint("Running all computations without napari visualization...")"\n📊 Visualization Tips:")
    aprint("Running all computations without napari visualization...")"  • Opened in tile/grid mode for side-by-side comparison")
    aprint("Running all computations without napari visualization...")"  • All scales use same contrast limits for consistent comparison")
    aprint("Running all computations without napari visualization...")"  • Toggle layers on/off to compare scales")
    aprint("Running all computations without napari visualization...")"  • 'original' = input image")
    aprint("Running all computations without napari visualization...")"  • 'reconstruction' = sum of all scales")
    aprint("Running all computations without napari visualization...")f"  • 'scale_Nx' = individual scale components (N={SCALES})")
    aprint("Running all computations without napari visualization...")"  • 'absolute_residual' = |original - reconstruction|")
    aprint("Running all computations without napari visualization...")"  • Use additive blending to see scale contributions")

    aprint("Running all computations without napari visualization...")"\n🎯 Energy Distribution Insights:")
    coarse_energy = energy_dist[-1]
    fine_energy = energy_dist[0]
    if coarse_energy > 0.5:
        aprint("Running all computations without napari visualization...")f"  ✓ Good: {coarse_energy:.1%} energy in coarsest scale")
    elif fine_energy > 0.5:
        aprint("Running all computations without napari visualization...")
            f"  ⚠ Warning: {fine_energy:.1%} energy in finest scale (trivial solution)"
        )
    else:
        aprint("Running all computations without napari visualization...")"  → Energy well distributed across scales")

    napari.run()

    # Show optimization convergence movie
    if stats["movie_frames"] is not None:
        aprint("Running all computations without napari visualization...")"\n🎬 Showing optimization convergence movie...")
        # Pass interpolation mode from stats to ensure movie matches optimization
        interpolation_mode = stats.get("interpolation", "cubic")
        show_optimization_movie(
            stats["movie_frames"], image.shape, interpolation=interpolation_mode
        )
