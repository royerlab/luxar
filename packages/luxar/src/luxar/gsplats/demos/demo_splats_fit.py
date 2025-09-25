#!/usr/bin/env python3
"""
High-level demo using fit_gaussian_splats() with per-splat optimizer.

This demo shows how to use the simplified fit_gaussian_splats() function
which internally uses the per-splat optimizer and dynamic operations.
"""

import argparse

import napari
import numpy as np
from arbol import aprint, asection
from skimage import data, filters

from luxar.gsplats.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.fit_gsplats import fit_gaussian_splats


def main():
    """Main demo with high-level fit function."""

    parser = argparse.ArgumentParser(
        description="High-level Gaussian splat fitting demo"
    )
    parser.add_argument(
        "--no-napari",
        action="store_true",
        help="Disable napari visualization for headless testing",
    )
    parser.add_argument(
        "--n-iters", type=int, default=1000, help="Number of optimization iterations"
    )
    parser.add_argument(
        "--disable-dynamic",
        action="store_true",
        help="Disable dynamic operations (dynamic ops are enabled by default)",
    )
    args = parser.parse_args()

    aprint("🚀 High-Level Gaussian Splat Fitting Demo")

    with asection("Creating test data"):
        # Create soft 2D "blobs" image (same as other demos)
        blobs = data.binary_blobs(
            length=256, blob_size_fraction=0.06, n_dim=2, volume_fraction=0.18, rng=42
        ).astype(float)
        V = filters.gaussian(blobs, sigma=3.25).astype(np.float32)
        aprint(f"Target image shape: {V.shape}")


    # Configure dynamic operations (enabled by default)
    dynamic_config = None
    enable_dynamic_ops = not args.disable_dynamic
    if enable_dynamic_ops:
        dynamic_config = DynamicOpsConfig()
        aprint(f"Dynamic operations enabled (step_every={dynamic_config.step_every})")
    else:
        aprint("Dynamic operations disabled")

    with asection(f"Fitting with per-splat optimizer ({args.n_iters} iterations)"):
        # Use the high-level fit function with per-splat optimizer
        params_full, amps, stats = fit_gaussian_splats(
            V=V,
            # centers_overcomplete auto-generated with intelligent defaults
            init_sigma_vox=1.6,
            n_iters=args.n_iters,
            lr=0.01,
            l1_amp=0.001,
            loss_type="l1",
            # asymmetric_penalty defaults to 10.0
            verbose=True,
            dynamic_ops_verbose=True,
            enable_dynamic_ops=enable_dynamic_ops,
            dynamic_config=dynamic_config,
            napari_movie=not args.no_napari,  # Record movie unless disabled
            movie_every=1,
        )

        aprint(f"Final parameters shape: {params_full.shape}")
        aprint(f"Final amplitudes shape: {amps.shape}")
        aprint(f"Final MSE: {stats.get('final_loss', 'N/A'):.6f}")

    # Show results in napari if enabled
    if not args.no_napari:
        with asection("Napari visualization"):
            # Reconstruct final image for display
            from luxar.gsplats.models.gsplats.gsplat_model import render_gaussians_numpy

            V_recon = render_gaussians_numpy(V.shape, params_full, amps)

            # Create napari viewer
            viewer = napari.Viewer(title="High-Level Fit Demo Results")

            # Add images
            viewer.add_image(V, name="Target", colormap="viridis", opacity=0.7)
            viewer.add_image(
                V_recon, name="Reconstruction", colormap="plasma", opacity=0.7
            )
            viewer.add_image(np.abs(V - V_recon), name="Residual", colormap="hot")

            # Add splat centers
            centers_np = params_full[:, :2]  # Extract center coordinates
            viewer.add_points(
                centers_np,
                name="Splat Centers",
                face_color="cyan",
                size=3,
                border_color="white",
                border_width=0.5,
            )

            # Add text info
            optimization_info = (
                f"Per-Splat Optimizer fit_gaussian_splats() Demo\n"
                f"Target: {V.shape} image\n"
                f"Splats: {len(amps)}\n"
                f"Iterations: {stats['iterations']}/{args.n_iters}\n"
                f"Time: {stats['time_seconds']:.2f}s\n"
                f"Final loss: {stats.get('final_loss', 'N/A'):.6f}\n"
                f"Dynamic ops: {'Enabled' if enable_dynamic_ops else 'Disabled'}\n\n"
                f"✅ High-level API with per-splat optimization\n"
                f"✅ Individual learning rates per splat\n"
                f"✅ Momentum preservation during topology changes"
            )

            viewer.text_overlay.text = optimization_info
            viewer.text_overlay.visible = True

            aprint("🎉 High-level demo complete!")
            aprint("Napari viewer opened - explore the results!")

            # Run napari
            napari.run()
    else:
        aprint("🎉 High-level demo complete (napari disabled)")


if __name__ == "__main__":
    main()
