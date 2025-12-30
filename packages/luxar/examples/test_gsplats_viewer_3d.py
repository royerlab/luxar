#!/usr/bin/env python3
"""
Test script for 3D GSplats viewer integration.

Creates a simple 3D gsplats dataset and saves it to a Luxar scene
for testing the TypeScript viewer implementation.

Usage:
    python test_gsplats_viewer_3d.py

Then view with:
    luxar serve test_gsplats_3d_example.zarr --viewer
"""

from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.gsplats.fit_gsplats import fit_gaussian_splats


def create_simple_3d_blobs(shape=(32, 32, 32), n_blobs=5):
    """Create simple 3D test data with Gaussian blobs."""
    aprint(f"Creating 3D test volume: {shape}")

    V = np.zeros(shape, dtype=np.float32)

    # Add well-separated Gaussian blobs for easy visual verification
    for i in range(n_blobs):
        # Position blobs in a grid pattern for easy identification
        if i < 4:
            # Four corners of a square in the middle of the volume
            centers = [
                [shape[0]//4, shape[1]//4, shape[2]//2],
                [3*shape[0]//4, shape[1]//4, shape[2]//2],
                [shape[0]//4, 3*shape[1]//4, shape[2]//2],
                [3*shape[0]//4, 3*shape[1]//4, shape[2]//2],
            ]
            center = centers[i]
        else:
            # One in the center
            center = [shape[0]//2, shape[1]//2, shape[2]//2]

        # Vary size and intensity
        sigma = 3.0 if i < 4 else 5.0
        amplitude = 0.8 if i < 4 else 1.0

        # Create 3D coordinate grids
        grids = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')

        # Compute 3D distance from center
        dist_sq = sum((g - c)**2 for g, c in zip(grids, center))

        # Add 3D Gaussian blob
        blob = amplitude * np.exp(-dist_sq / (2 * sigma**2))
        V += blob

    # Add minimal noise
    V += np.random.normal(0, 0.01, shape)
    V = np.clip(V, 0, None).astype(np.float32)

    aprint(f"Volume range: [{V.min():.4f}, {V.max():.4f}]")
    return V


with asection("GSplats 3D Viewer Test"):
    aprint("Creating test data for GSplats viewer...")

    # Create simple 3D test volume
    volume = create_simple_3d_blobs(shape=(32, 32, 32), n_blobs=5)

    with asection("Fitting Gaussian splats"):
        # Fit gsplats with reasonable settings
        result = fit_gaussian_splats(
            volume,
            n_iters=500,
            truncate=3.0,
            device=None,  # Auto-detect
            verbose=True,
            enable_dynamic_ops=True,
            napari_movie=False,  # No visualization during fitting
        )

        aprint(f"✓ Fitted {len(result.amplitudes)} splats")
        aprint(f"  Centers shape: {result.centers.shape}")
        aprint(f"  Cholesky shape: {result.cholesky_factors.shape}")
        aprint(f"  Amplitudes shape: {result.amplitudes.shape}")
        aprint(f"  Sharpness shape: {result.sharpnesses.shape}")

    with asection("Creating Luxar scene"):
        # Create scene with LuxarZarrCompiler
        output_path = Path(__file__).parent / 'test_gsplats_3d_example.zarr'

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            # Add gsplats to scene
            scene.add_gsplats(
                name='test_gsplats',
                centers=result.centers,
                cholesky_factors=result.cholesky_factors,
                amplitudes=result.amplitudes,
                sharpness=result.sharpnesses,
                colors=None,  # Default white color
                opacity=1.0,
                blending_mode='additive',
            )

        aprint(f"✓ Saved scene to: {output_path}")

    aprint("\n" + "="*60)
    aprint("Test data created successfully!")
    aprint("="*60)
    aprint("\nTo view in the browser:")
    aprint(f"  luxar serve {output_path} --viewer")
    aprint("\nExpected result:")
    aprint("  • 5 Gaussian splats visible")
    aprint("  • 4 splats at corners of a square")
    aprint("  • 1 larger splat in the center")
    aprint("  • All splats should be white (no colors)")
    aprint("  • Additive blending should show overlap")
    aprint("="*60)
