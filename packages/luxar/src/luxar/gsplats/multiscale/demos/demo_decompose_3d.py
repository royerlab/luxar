#!/usr/bin/env python3
"""
3D Multi-Scale Volume Decomposition Demo

Demonstrates multi-scale decomposition on 3D volumetric data,
with interactive napari visualization showing how 3D features
are distributed across different scales.
"""

import sys

import napari
import numpy as np
from arbol import aprint, asection

from luxar.gsplats.multiscale import (
    decompose_image,
    show_optimization_movie,
    upsample_for_visualization,
)

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("🧊 3D Multi-Scale Decomposition Demo (napari disabled)")
    aprint("Running 3D validation in headless mode...")

# ======= Demo knobs =======
SCALES = [1, 2, 4, 8, 16]  # Scale factors to use
N_ITERS = 5000  # Fewer iterations for 3D (computational cost)
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps"
VOLUME_SIZE = 64  # Size of 3D volume (keep small for demo)
# ==========================


def create_test_volume_3d(size: int = 64) -> np.ndarray:
    """
    Create a synthetic 3D test volume with multiple frequency components.

    Parameters
    ----------
    size : int
        Size of the volume (will be size x size x size)

    Returns
    -------
    np.ndarray
        3D volume with shape (size, size, size)
    """
    aprint(f"Creating synthetic {size}³ volume...")

    # Create coordinate grids
    z = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    x = np.linspace(-2, 2, size)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    # Low frequency: Large Gaussian blob
    R_low = np.sqrt(X**2 + Y**2 + Z**2)
    low_freq = np.exp(-(R_low**2) / 2)

    # Medium frequency: Multiple smaller blobs
    blob1 = 0.5 * np.exp(-((X - 0.8) ** 2 + (Y - 0.8) ** 2 + (Z - 0.8) ** 2) / 0.3)
    blob2 = 0.5 * np.exp(-((X + 0.8) ** 2 + (Y + 0.8) ** 2 + (Z + 0.8) ** 2) / 0.3)
    blob3 = 0.5 * np.exp(-((X - 0.8) ** 2 + (Y + 0.8) ** 2 + (Z - 0.8) ** 2) / 0.3)
    med_freq = blob1 + blob2 + blob3

    # High frequency: Radial oscillations
    R = np.sqrt(X**2 + Y**2 + Z**2)
    high_freq = 0.2 * np.sin(8 * R) / (R + 0.1)

    # Combine all frequencies
    volume = low_freq + med_freq + high_freq

    # Normalize to [0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min())

    aprint(f"Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
    aprint(f"Total energy: {volume.sum():.2f}")

    return volume.astype(np.float32)


with asection("3D Multi-Scale Decomposition Demo"):
    aprint(
        "🧊 Interactive 3D volumetric decomposition with scale separation"
    )

    with asection("Creating 3D test volume"):
        # Create test volume
        volume = create_test_volume_3d(size=VOLUME_SIZE)

        aprint(f"Volume shape: {volume.shape}")
        aprint(f"Volume size: {volume.nbytes / 1024 / 1024:.2f} MB")

    with asection(f"Decomposing into {len(SCALES)} scales"):
        aprint(f"Scales: {SCALES}")
        aprint(
            f"Parameters: n_iters={N_ITERS}"
        )

        # Decompose with movie recording enabled
        scales_list, stats = decompose_image(
            volume,
            interpolation='nearest',  # Fastest option (cubic is also practical for 3D now with Keys cubic)
            scales=SCALES,
            n_iters=N_ITERS,
            napari_movie=True,
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
        interpolation_mode = stats.get('interpolation', 'cubic')
        aprint(f"  Using '{interpolation_mode}' interpolation for upsampling (matches optimization)")
        for i, (scale, vol_scale) in enumerate(zip(SCALES, scales_list)):
            aprint(
                f"  Upsampling scale {scale}x from {vol_scale.shape} to {volume.shape}"
            )
            if vol_scale.shape != volume.shape:
                vol_upsampled = upsample_for_visualization(vol_scale, volume.shape, interpolation_mode)
            else:
                vol_upsampled = vol_scale
            scales_upsampled.append(vol_upsampled)

        # Compute reconstruction
        reconstruction = np.sum(scales_upsampled, axis=0)

        # Compute residual
        residual = volume - reconstruction
        abs_residual = np.abs(residual)

        aprint(f"Reconstruction MSE: {np.mean((reconstruction - volume) ** 2):.6e}")
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
        shape_str = "×".join(str(s) for s in scales_list[i].shape)
        aprint(f"Scale {scale:2d}x: {energy_pct:5.1f}% {bar} [{shape_str}]")
    aprint(f"Total: {sum(energy_dist) * 100:.1f}%")
    aprint("=" * 60)

    # Napari visualization (only if enabled)
    if not NO_NAPARI:
        aprint("\nLaunching 3D napari viewer...")
        viewer = napari.Viewer(ndisplay=3)  # Force 3D display

        # Determine contrast limits from original volume (shared across all scales)
        contrast_limits = [0, float(volume.max())]

        # Add original volume
        viewer.add_image(
            volume,
            name="original",
            colormap="viridis",
            contrast_limits=contrast_limits,
            rendering="mip",  # Maximum intensity projection
        )

        # Add reconstruction
        viewer.add_image(
            reconstruction,
            name="reconstruction",
            colormap="viridis",
            contrast_limits=contrast_limits,
            rendering="mip",
            visible=True,
        )

        # Add each scale component (upsampled) - use same contrast limits for consistency
        for i, (scale, vol_upsampled) in enumerate(zip(SCALES, scales_upsampled)):
            energy_pct = energy_dist[i] * 100
            viewer.add_image(
                vol_upsampled,
                name=f"scale_{scale}x ({energy_pct:.1f}%)",
                colormap="turbo",
                contrast_limits=contrast_limits,
                blending="additive",
                opacity=0.8,
                rendering="mip",
                visible=True,
            )

        # Add residual
        viewer.add_image(
            abs_residual,
            name="absolute_residual",
            colormap="inferno",
            contrast_limits=[0, max(1e-12, float(abs_residual.max()))],
            rendering="mip",
            visible=True,
        )

        # Add signed residual for debugging
        viewer.add_image(
            residual,
            name="signed_residual",
            colormap="bwr",
            contrast_limits=[-abs_residual.max(), abs_residual.max()],
            rendering="mip",
            visible=True,
        )

        # Try to label axes
        try:
            viewer.dims.axis_labels = ["z", "y", "x"]
        except Exception:
            pass

        # Set better 3D camera view
        viewer.camera.angles = (15, 25, 120)
        viewer.camera.zoom = 1.0

        # Enable tile/grid mode for side-by-side comparison
        viewer.grid.enabled = True
        viewer.grid.shape = (-1, 3)  # Auto rows, 3 columns

        # Set up text overlay with energy distribution
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            f"3D Multi-Scale Decomposition | Volume: {VOLUME_SIZE}³ | "
            f"Scales: {SCALES} | "
            f"Energy: {' → '.join([f'{e:.1%}' for e in energy_dist])} | "
            f"Reconstruction MSE: {stats['final_error']:.6e}"
        )

        # Console tips
        aprint("\n🎮 3D Navigation Tips:")
        aprint("  • Opened in tile/grid mode for side-by-side comparison")
        aprint("  • All scales use same contrast limits for consistent comparison")
        aprint("  • Use mouse + Shift to rotate the 3D view")
        aprint("  • Toggle layers on/off to compare scales")
        aprint("  • 'original' = input volume")
        aprint("  • 'reconstruction' = sum of all scales")
        aprint(f"  • 'scale_Nx' = individual scale components (N={SCALES})")
        aprint("  • MIP rendering = Maximum Intensity Projection")
        aprint("  • Use additive blending to see scale contributions")

        aprint("\n🎯 Energy Distribution Insights:")
        coarse_energy = energy_dist[-1]
        fine_energy = energy_dist[0]
        if coarse_energy > 0.5:
            aprint(f"  ✓ Good: {coarse_energy:.1%} energy in coarsest scale")
            aprint("    → Low-frequency 3D features captured at coarse resolution")
        elif fine_energy > 0.5:
            aprint(
                f"  ⚠ Warning: {fine_energy:.1%} energy in finest scale (trivial solution)"
            )
            aprint("    → Try increasing alpha or energy_weight")
        else:
            aprint("  → Energy well distributed across scales")

        aprint("\n📊 Compression Potential:")
        # Calculate bits per voxel for each scale
        FLOAT_BITS = 32
        VOLUME_BITS = volume.size * FLOAT_BITS
        for i, scale in enumerate(SCALES):
            scale_voxels = scales_list[i].size
            scale_bits = scale_voxels * FLOAT_BITS
            compression_pct = 100.0 * (1.0 - (scale_bits / VOLUME_BITS))
            energy_pct = energy_dist[i] * 100
            aprint(
                f"  Scale {scale:2d}x: {scale_voxels:>7,d} voxels | "
                f"Size: {compression_pct:5.1f}% smaller | "
                f"Energy: {energy_pct:5.1f}%"
            )

        napari.run()

        # Show optimization convergence movie
        if stats['movie_frames'] is not None:
            aprint("\n🎬 Showing optimization convergence movie...")
            # Pass interpolation mode from stats to ensure movie matches optimization
            interpolation_mode = stats.get('interpolation', 'cubic')
            show_optimization_movie(stats['movie_frames'], volume.shape, interpolation=interpolation_mode)

    # Console summary (always shown)
    aprint("\n" + "=" * 60)
    aprint("3D Decomposition Summary")
    aprint("=" * 60)
    aprint(f"Input volume: {VOLUME_SIZE}³ = {volume.size:,} voxels")
    aprint(f"Scales used: {SCALES}")
    aprint(f"Optimization: {N_ITERS} iterations in {stats['time_seconds']:.2f}s")
    aprint(f"Final MSE: {stats['final_error']:.6e}")
    aprint(f"Best MSE: {stats['best_error']:.6e}")
    aprint("")
    aprint("Energy distribution:")
    for i, (scale, energy_frac) in enumerate(zip(SCALES, energy_dist)):
        aprint(f"  {scale:2d}x: {energy_frac * 100:5.1f}%")
    aprint("=" * 60)

    if NO_NAPARI:
        aprint(
            "✅ 3D decomposition validation complete - synthetic volume approach working"
        )
