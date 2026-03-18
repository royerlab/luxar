#!/usr/bin/env python3
"""
Test script for 4D GSplats viewer integration and nD slicing.

Creates a 4D gsplats dataset with time-varying blobs to test:
- nD → 3D slicing
- Amplitude attenuation based on hidden dimension distance
- Dimension navigation in the viewer

Usage:
    python test_gsplats_viewer_4d.py

Then view with:
    luxar serve test_gsplats_4d_example.zarr --viewer
"""

from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.gsplats.fit_gsplats import fit_gaussian_splats


def create_4d_time_varying_blobs(shape=(8, 32, 32, 32), n_blobs=4):
    """
    Create 4D test data with time-varying Gaussian blobs.

    The blobs move or change intensity over time to test nD slicing.
    """
    aprint(f"Creating 4D test volume: {shape} (t, z, y, x)")

    V = np.zeros(shape, dtype=np.float32)
    t_steps, depth, height, width = shape

    # Create blobs that vary along the time dimension
    for blob_idx in range(n_blobs):
        # Base center in spatial dimensions (z, y, x)
        if blob_idx < 2:
            # Two blobs that move along x
            z_center = depth // 4 if blob_idx == 0 else 3 * depth // 4
            y_center = height // 2
            x_base = width // 4
        else:
            # Two blobs that move along y
            z_center = depth // 2
            y_base = height // 4
            x_center = width // 4 if blob_idx == 2 else 3 * width // 4

        # Blob size and intensity
        sigma_spatial = 3.0
        sigma_time = 1.5  # Narrow in time - only visible for a few time steps
        base_amplitude = 0.8

        for t in range(t_steps):
            # Position varies with time for some blobs
            if blob_idx < 2:
                # Move along x-axis over time
                x_center = x_base + int((width // 2) * (t / (t_steps - 1)))
                y_pos = y_center
            else:
                # Move along y-axis over time
                y_pos = y_base + int((height // 2) * (t / (t_steps - 1)))
                _x_pos = x_center  # Reserved for future use

            # Amplitude varies with time (Gaussian envelope in time)
            t_center = t_steps // 2
            time_attenuation = np.exp(-((t - t_center) ** 2) / (2 * sigma_time**2))
            amplitude = base_amplitude * time_attenuation

            # Create spatial grids for this time slice
            z_grid, y_grid, x_grid = np.meshgrid(
                np.arange(depth), np.arange(height), np.arange(width), indexing="ij"
            )

            # Compute spatial distance
            if blob_idx < 2:
                dist_sq = (
                    (z_grid - z_center) ** 2
                    + (y_grid - y_center) ** 2
                    + (x_grid - x_center) ** 2
                )
            else:
                dist_sq = (
                    (z_grid - z_center) ** 2
                    + (y_grid - y_pos) ** 2
                    + (x_grid - x_center) ** 2
                )

            # Add blob for this time step
            blob = amplitude * np.exp(-dist_sq / (2 * sigma_spatial**2))
            V[t] += blob

    # Add minimal noise
    V += np.random.normal(0, 0.005, shape)
    V = np.clip(V, 0, None).astype(np.float32)

    aprint(f"Volume range: [{V.min():.4f}, {V.max():.4f}]")
    aprint(f"Time slices: {t_steps}")
    return V


with asection("GSplats 4D Viewer Test"):
    aprint("Creating 4D test data for nD slicing test...")

    # Create 4D time-varying volume
    volume = create_4d_time_varying_blobs(shape=(8, 32, 32, 32), n_blobs=4)

    with asection("Fitting 4D Gaussian splats"):
        # Fit gsplats
        result = fit_gaussian_splats(
            volume,
            n_iters=300,  # Reduced for 4D
            truncate=3.0,
            device=None,
            verbose=True,
            enable_dynamic_ops=True,
            napari_movie=False,
        )

        aprint(f"✓ Fitted {len(result.amplitudes)} 4D splats")
        aprint(f"  Centers shape: {result.centers.shape} (should be N x 4)")
        aprint(f"  Cholesky shape: {result.cholesky_factors.shape} (should be N x 10)")
        aprint(f"  Amplitudes shape: {result.amplitudes.shape}")

    with asection("Creating Luxar scene with 4D dimensions"):
        # Create scene with 4 dimensions
        output_path = Path(__file__).parent / "test_gsplats_4d_example.zarr"

        dims = Dimensions(
            [
                Dimension(name="t", unit="frame", scale=1.0, step=1.0),
                Dimension(name="z", unit="px", scale=1.0),
                Dimension(name="y", unit="px", scale=1.0),
                Dimension(name="x", unit="px", scale=1.0),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Add 4D gsplats to scene
            scene.add_gsplats(
                name="test_gsplats_4d",
                centers=result.centers,
                cholesky_factors=result.cholesky_factors,
                amplitudes=result.amplitudes,
                colors=None,
                opacity=1.0,
                blending_mode="additive",
            )

        aprint(f"✓ Saved scene to: {output_path}")

    aprint("\n" + "=" * 60)
    aprint("4D test data created successfully!")
    aprint("=" * 60)
    aprint("\nTo view in the browser:")
    aprint(f"  luxar serve {output_path} --viewer")
    aprint("\nExpected behavior:")
    aprint("  • Initially shows 3D slice (displaying z, y, x)")
    aprint("  • Use dimension selector to navigate time (t dimension)")
    aprint("  • Splats should appear/disappear as you navigate through time")
    aprint("  • Splat intensity should attenuate based on distance from slice")
    aprint("  • Some splats move across the view as time changes")
    aprint("\nThis tests:")
    aprint("  ✓ nD → 3D slicing in gsplats-processor.ts")
    aprint("  ✓ Amplitude attenuation for hidden dimensions")
    aprint("  ✓ Spatial index queries with tolerance")
    aprint("  ✓ View updates during dimension navigation")
    aprint("=" * 60)
