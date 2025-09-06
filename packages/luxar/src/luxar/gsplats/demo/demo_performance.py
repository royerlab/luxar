#!/usr/bin/env python
"""
Performance demonstration of Gaussian splatting with optimization features.
Shows convergence speed and quality metrics.
"""

import time

import napari
import numpy as np
from arbol import Arbol, aprint, asection
from skimage import data, filters

from luxar.gsplats.candidates import find_candidates_overcomplete_nd
from luxar.gsplats.fit_gsplats import GaussianSplatFitter
from luxar.gsplats.models.gsplats.gsplat_model import render_gaussians_numpy

# Setup Arbol
Arbol.max_depth = 3


def main():
    """Run performance demo with visual output."""

    with asection("Gaussian Splatting Performance Demo"):
        # Generate test data
        with asection("Data Generation"):
            blobs = data.binary_blobs(
                length=256,
                blob_size_fraction=0.06,
                n_dim=2,
                volume_fraction=0.18,
                rng=42
            ).astype(float)
            V = filters.gaussian(blobs, sigma=3.25).astype(np.float32)
            aprint(f"Image shape: {V.shape}")
            aprint(f"Data range: [{V.min():.3f}, {V.max():.3f}]")

        # Find candidates
        with asection("Candidate Generation"):
            candidates = find_candidates_overcomplete_nd(
                V,
                scales=(0.8, 1.2, 1.8, 2.6, 3.6),
                peaks_per_scale=900,
                percentile_thresh=90,
                min_dist=2.0,
            )
            aprint(f"Found {len(candidates)} candidate centers")

        # Fit with optimizations enabled
        with asection("Optimized Fitting"):
            start_time = time.time()

            # Use class interface for detailed stats
            fitter = GaussianSplatFitter()
            params, amps, stats = fitter.fit(
                V,
                centers_overcomplete=candidates,
                init_sigma_vox=1.6,
                n_iters=300,
                lr=0.2,
                loss_type="poisson",
                l1_amp=0.01,
                sigma_min_diag=[0.6, 0.6],
                truncate=3.0,
                verbose=True,
                early_stopping=True,
                early_stop_patience=20,
            )

            fit_time = time.time() - start_time

            aprint("\nOptimization Summary:")
            aprint(f"  Time: {fit_time:.2f} seconds")
            aprint(f"  Iterations: {stats['iterations']}/300")
            aprint(f"  Converged: {stats['converged']}")
            aprint(f"  Final loss: {stats['final_loss']:.5g}")
            aprint(f"  Active splats: {np.sum(amps > 0.01)}/{len(amps)}")

            if stats['converged']:
                saved_iters = 300 - stats['iterations']
                time_per_iter = fit_time / stats['iterations']
                time_saved = saved_iters * time_per_iter
                aprint(f"  ✓ Early stopping saved {saved_iters} iterations (~{time_saved:.1f}s)")

        # Prepare visualization
        with asection("Visualization Preparation"):
            # Render reconstruction
            reconstruction = render_gaussians_numpy(
                V.shape, params, amps, truncate=3.0
            )

            # Compute error metrics
            residual = V - reconstruction
            mse = np.mean(residual ** 2)
            rel_l2 = np.linalg.norm(residual) / (np.linalg.norm(V) + 1e-12)
            psnr = 10 * np.log10(1.0 / (mse + 1e-12))

            aprint("Reconstruction Quality:")
            aprint(f"  MSE: {mse:.5f}")
            aprint(f"  Relative L2 error: {rel_l2:.4f}")
            aprint(f"  PSNR: {psnr:.1f} dB")

            # Extract ellipse parameters for visualization
            d = 2
            centers = params[:, :d]

            # Create shapes for active splats
            active_mask = amps > 0.01
            active_centers = centers[active_mask]
            active_amps = amps[active_mask]

            aprint(f"Visualizing {len(active_centers)} active splats")

    # Launch napari viewer
    viewer = napari.Viewer()

    # Add layers
    viewer.add_image(
        V,
        name="Original",
        colormap="magma",
        contrast_limits=[0, float(V.max())]
    )

    viewer.add_image(
        reconstruction,
        name="Reconstruction",
        colormap="magma",
        contrast_limits=[0, float(V.max())],
        opacity=0.8,
    )

    viewer.add_image(
        np.abs(residual),
        name="Absolute Error",
        colormap="turbo",
        opacity=0.6,
        contrast_limits=[0, float(np.abs(residual).max())],
    )

    # Add splat centers
    viewer.add_points(
        active_centers,
        name="Active Splat Centers",
        size=3,
        border_color="cyan",
        face_color="transparent",
        properties={'amplitude': active_amps},
    )

    # Add text overlay with stats
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = (
        f"Optimization: {stats['iterations']} iterations in {fit_time:.1f}s | "
        f"Active splats: {len(active_centers)}/{len(candidates)} | "
        f"MSE: {mse:.5f} | Rel L2: {rel_l2:.4f} | PSNR: {psnr:.1f} dB"
    )

    aprint("\n" + "=" * 60)
    aprint("Visualization ready. Close napari window to exit.")
    aprint("=" * 60)

    napari.run()


if __name__ == "__main__":
    main()
