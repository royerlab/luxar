#!/usr/bin/env python
"""
Performance demonstration of Gaussian splatting with per-splat optimizer.
Shows convergence speed and quality metrics.
"""

import argparse
import time

import napari
import numpy as np
from arbol import Arbol, aprint, asection
from skimage import data, filters

from luxar.gsplats.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.models.gsplats.gsplat_model import render_gaussians_numpy

# Setup Arbol
Arbol.max_depth = 3


def main():
    """Run performance demo with visual output."""

    parser = argparse.ArgumentParser(
        description="Performance demo for Gaussian splatting"
    )
    parser.add_argument(
        "--no-napari",
        action="store_true",
        help="Disable napari visualization for headless testing",
    )
    parser.add_argument(
        "--n-iters", type=int, default=300, help="Number of optimization iterations"
    )
    args = parser.parse_args()

    with asection("Gaussian Splatting Performance Demo"):
        # Generate test data
        with asection("Data Generation"):
            blobs = data.binary_blobs(
                length=256,
                blob_size_fraction=0.06,
                n_dim=2,
                volume_fraction=0.18,
                rng=42,
            ).astype(float)
            V = filters.gaussian(blobs, sigma=3.25).astype(np.float32)
            aprint(f"Image shape: {V.shape}")
            aprint(f"Data range: [{V.min():.3f}, {V.max():.3f}]")

        # Configure dynamic operations
        dynamic_config = DynamicOpsConfig()
        aprint(f"Dynamic operations enabled (step_every={dynamic_config.step_every})")

        # Fit with per-splat optimizer
        with asection("Per-Splat Optimizer Fitting"):
            start_time = time.time()

            # Use simplified one-step API with auto-candidate generation
            params, amps, stats = fit_gaussian_splats(
                V,
                # centers_overcomplete auto-generated with intelligent defaults
                init_sigma_vox=1.6,
                n_iters=args.n_iters,
                lr=0.2,
                loss_type="poisson",
                l1_amp=0.01,
                sigma_min_diag=[0.6, 0.6],
                truncate=3.0,
                verbose=True,
                max_abs_error=0.01,  # Stop when max absolute error < 0.01
                # Dynamic operations
                enable_dynamic_ops=True,
                dynamic_config=dynamic_config,
                napari_movie=not args.no_napari,  # Record movie unless disabled
                movie_every=5,
            )

            fit_time = time.time() - start_time

            aprint("\nOptimization Summary:")
            aprint(f"  Time: {fit_time:.2f} seconds")
            aprint(f"  Iterations: {stats['iterations']}/{args.n_iters}")
            aprint(f"  Converged: {stats['converged']}")
            aprint(f"  Final loss: {stats['final_loss']:.5g}")
            aprint(f"  Active splats: {np.sum(amps > 0.01)}/{len(amps)}")
            aprint("  ✓ Per-splat optimizer with individual learning rates")

            if stats["converged"]:
                saved_iters = args.n_iters - stats["iterations"]
                time_per_iter = fit_time / stats["iterations"]
                time_saved = saved_iters * time_per_iter
                aprint(
                    f"  ✓ Early stopping saved {saved_iters} iterations (~{time_saved:.1f}s)"
                )

        # Prepare visualization
        with asection("Visualization Preparation"):
            # Render reconstruction
            reconstruction = render_gaussians_numpy(V.shape, params, amps, truncate=3.0)

            # Compute error metrics
            residual = V - reconstruction
            mse = np.mean(residual**2)
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

    # Show results in napari if enabled
    if not args.no_napari:
        # Launch napari viewer
        viewer = napari.Viewer()

        # Add layers
        viewer.add_image(
            V, name="Original", colormap="magma", contrast_limits=[0, float(V.max())]
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
            properties={"amplitude": active_amps},
        )

        # Add text overlay with stats
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            f"Per-Splat Optimizer: {stats['iterations']} iterations in {fit_time:.1f}s | "
            f"Active splats: {len(active_centers)}/{len(amps)} total | "
            f"MSE: {mse:.5f} | Rel L2: {rel_l2:.4f} | PSNR: {psnr:.1f} dB"
        )

        aprint("\n" + "=" * 60)
        aprint("Visualization ready. Close napari window to exit.")
        aprint("=" * 60)

        # Run napari
        napari.run()
    else:
        aprint("\nPerformance Demo Complete (napari disabled)")
        aprint(
            f"Per-splat optimizer: {stats['iterations']} iterations in {fit_time:.1f}s"
        )
        aprint(f"Active splats: {len(active_centers)}/{len(amps)} total")
        aprint(f"Quality: MSE={mse:.5f}, Rel L2={rel_l2:.4f}, PSNR={psnr:.1f} dB")


if __name__ == "__main__":
    main()
