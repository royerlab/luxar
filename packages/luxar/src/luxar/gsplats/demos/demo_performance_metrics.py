#!/usr/bin/env python
"""
Performance Metrics - Convergence Speed and Quality Analysis

**What this demo demonstrates:**
- Detailed performance metrics for standard Adam + fixed-pool relocation
- Convergence monitoring with early stopping
- Quality metrics: MSE, relative L2 error, PSNR
- Timing analysis: iterations per second, total time
- Active vs total splat counting (fixed-pool architecture)
- Memory-efficient movie recording (every 5 iterations)

**Key concepts:**
- Early stopping: Saves compute when max absolute error threshold is met
- Fixed-pool relocation: weak splats are moved to high-residual regions
  rather than added/removed, keeping optimizer tensor shapes constant
- Quality metrics:
  * MSE: Mean Squared Error (lower is better)
  * Relative L2: Normalized error relative to signal magnitude
  * PSNR: Peak Signal-to-Noise Ratio in dB (higher is better)
- Dynamic operations impact: Shows effect of periodic relocation

**Metrics displayed:**
- Optimization time (seconds)
- Iterations completed vs requested
- Early convergence detection and time saved
- Final loss value
- Active splat count (amplitude > 0.01 threshold)
- Reconstruction quality (MSE, rel L2, PSNR)

**Data source:** Synthetic 2D blobs (256×256, same as basic demo)
**Visualization:** Napari showing original, reconstruction, error, and active splat centers
**Command-line:** `--no-napari` for headless mode, `--n-iters N` to set iterations

**Use cases:**
- Benchmarking: Compare performance across hardware/settings
- Quality assessment: Understand trade-offs between time and accuracy
- Algorithm validation: Verify early stopping and convergence behavior

**Related demos:**
- demo_basic_fitting.py - Simpler introduction without detailed metrics
"""

import argparse
import time

import napari
import numpy as np
from arbol import Arbol, aprint, asection
from skimage import data, filters

from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy

# Setup Arbol
Arbol.max_depth = 3


def main() -> None:
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

        with asection("Adam + Fixed-Pool Relocation Fitting"):
            start_time = time.time()

            # Use simplified one-step API with auto-candidate generation
            result = fit_gaussian_splats(
                V,
                # seeds auto-generated with intelligent defaults
                n_iters=args.n_iters,
                verbose=True,
                max_abs_error=0.01,  # Stop when max absolute error < 0.01
                # Dynamic operations
                enable_dynamic_ops=True,
                dynamic_config=dynamic_config,
                napari_movie=not args.no_napari,  # Record movie unless disabled
                movie_every=5,
            )

            stats = result.stats
            fit_time = time.time() - start_time

            aprint("\nOptimization Summary:")
            aprint(f"  Time: {fit_time:.2f} seconds")
            aprint(f"  Iterations: {stats['iterations']}/{args.n_iters}")
            aprint(f"  Converged: {stats['converged']}")
            aprint(f"  Final loss: {stats['final_loss']:.5g}")
            aprint(
                f"  Active splats: {np.sum(result.amplitudes > 0.01)}/{len(result.amplitudes)}"
            )
            aprint("  ✓ Standard Adam with gradient-dilution-compensated LR")

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
            reconstruction = render_gaussians_numpy(V.shape, result, truncate=3.0)

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
            centers = result.centers

            # Create shapes for active splats
            active_mask = result.amplitudes > 0.01
            active_centers = centers[active_mask]
            active_amps = result.amplitudes[active_mask]

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
            f"Adam + Fixed-Pool Relocation: {stats['iterations']} iterations in {fit_time:.1f}s | "
            f"Active splats: {len(active_centers)}/{len(result.amplitudes)} total | "
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
            f"Adam + fixed-pool relocation: {stats['iterations']} iterations in {fit_time:.1f}s"
        )
        aprint(f"Active splats: {len(active_centers)}/{len(result.amplitudes)} total")
        aprint(f"Quality: MSE={mse:.5f}, Rel L2={rel_l2:.4f}, PSNR={psnr:.1f} dB")


if __name__ == "__main__":
    main()
