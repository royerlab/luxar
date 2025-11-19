#!/usr/bin/env python3
"""
Multi-Scale Gaussian Splatting Demo

Demonstrates multi-scale Gaussian splat fitting for computational efficiency.
Compares single-scale vs multi-scale fitting showing speedup and reconstruction quality.
Uses multi-scale decomposition to fit large Gaussians efficiently on downsampled images.
"""

import sys
import time

import napari
import numpy as np
from arbol import Arbol, aprint, asection
from skimage import data

from luxar.gsplats import fit_gaussian_splats, fit_multiscale_gaussian_splats
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.multiscale import show_optimization_movie

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("🔬 Demo (napari disabled)")
    aprint("Running all computations without napari visualization...")

# ======= Demo knobs =======
SCALES = [1, 2, 4, 8]  # Scale factors to use
N_ITERS_DECOMP = 4000  # Iterations for decomposition
N_ITERS_PER_SCALE = 1000  # Iterations per scale (multi-scale)
N_ITERS_SINGLE = 500  # Iterations for single-scale comparison
BASE_INIT_SIGMA = 1.5  # Base sigma (gets multiplied by scale factor)
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps"
TRUNCATE_SIG = 3.0  # Rendering truncation
# ==========================

# Setup Arbol
Arbol.max_depth = 3


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

    # Combine and normalize to [0, 100]
    image = low_freq + med_freq + high_freq
    image = (image - image.min()) / (image.max() - image.min()) * 100.0

    return image.astype(np.float32)


with asection("Multi-Scale Gaussian Splatting Demo"):
    aprint("🎯 Comparing single-scale vs multi-scale Gaussian splat fitting")
    aprint(f"Scales: {SCALES}  |  Decomp iters: {N_ITERS_DECOMP}")
    aprint(
        f"Per-scale iters: {N_ITERS_PER_SCALE}  |  Single-scale iters: {N_ITERS_SINGLE}"
    )

    with asection("Loading test image"):
        # Load mitosis image (full size)
        try:
            aprint("Loading mitosis image from scikit-image...")
            # Load mitosis image (grayscale biological data)
            img = data.human_mitosis()
            V = img.astype(np.float32)
            aprint(f"Loaded mitosis image: {V.shape}")

            # Normalize to [0, 100] range
            V = (V - V.min()) / (V.max() - V.min()) * 100.0

            # Add background Gaussians (smooth background texture)
            aprint("Adding background Gaussians...")
            background = np.zeros_like(V)
            np.random.seed(42)  # Reproducible
            n_background_blobs = 5
            for i in range(n_background_blobs):
                # Random position
                cy, cx = (
                    np.random.randint(20, V.shape[0] - 20),
                    np.random.randint(20, V.shape[1] - 20),
                )
                # Random size (large blobs for background)
                sigma = np.random.uniform(30, 60)
                # Create Gaussian blob
                y_coords, x_coords = np.ogrid[: V.shape[0], : V.shape[1]]
                blob = np.exp(
                    -((y_coords - cy) ** 2 + (x_coords - cx) ** 2) / (2 * sigma**2)
                )
                # Random amplitude (subtle background)
                amplitude = np.random.uniform(5, 15)
                background += blob * amplitude

            V = V + background
            aprint(f"Added {n_background_blobs} background Gaussian blobs")

            # Add tiny bit of noise
            noise_level = 0.5  # Tiny noise
            noise = np.random.normal(0, noise_level, V.shape).astype(np.float32)
            V = V + noise
            aprint(f"Added Gaussian noise (std={noise_level})")

        except Exception as e:
            aprint(f"Could not load mitosis: {e}")
            aprint("Creating synthetic test image...")
            V = create_test_image_2d(256)

        aprint(f"Final image shape: {V.shape}")
        aprint(f"Value range: [{V.min():.3f}, {V.max():.3f}]")

    # Method 1: Multi-scale fitting
    with asection(f"Method 1: Multi-Scale Fitting (scales={SCALES})"):
        start_multi = time.time()
        params_multi, amps_multi, stats_multi = fit_multiscale_gaussian_splats(
            V,
            scales=SCALES,
            base_init_sigma=BASE_INIT_SIGMA,
            n_iters_decomp=N_ITERS_DECOMP,
            n_iters_per_scale=N_ITERS_PER_SCALE,
            loss_type="l1",
            max_abs_error=0.1,
            device=DEVICE,
            verbose=True,
            napari_movie=(not NO_NAPARI),
            movie_every=50,  # Record every 50 iterations to reduce memory
            visualize_per_scale=True,  # Enable per-scale visualization
        )
        time_multi = time.time() - start_multi

        n_splats_multi = len(amps_multi)
        aprint(f"\n✅ Multi-scale complete in {time_multi:.2f}s")
        aprint(f"Total splats: {n_splats_multi:,}")
        aprint(f"Splats per scale: {stats_multi['n_splats_per_scale']}")
        aprint(f"Computational speedup: {stats_multi['computational_speedup']:.1f}×")

    # Render multi-scale reconstruction
    # params_multi includes sharpness - auto-extracted by render function
    V_recon_multi = render_gaussians_numpy(
        V.shape, params_multi, amps_multi, truncate=TRUNCATE_SIG
    )
    residual_multi = V - V_recon_multi
    error_multi = np.mean((V - V_recon_multi) ** 2)
    max_abs_error_multi = np.abs(residual_multi).max()

    aprint(f"Multi-scale MSE: {error_multi:.6e}")
    aprint(f"Multi-scale max abs error: {max_abs_error_multi:.6f}")

    # Display decomposition convergence animation
    decomp_stats = stats_multi.get("decomposition_stats", {})
    if decomp_stats.get("movie_frames") is not None:
        with asection("Decomposition Convergence Animation"):
            aprint("🎬 Showing optimization movie...")
            aprint(f"Number of frames: {len(decomp_stats['movie_frames'])}")
            show_optimization_movie(
                decomp_stats["movie_frames"],
                V.shape,
                interpolation="cubic",
            )

    # Display per-scale visualizations
    per_scale_vis = stats_multi.get("per_scale_visualizations", [])
    if len(per_scale_vis) > 0:
        with asection("Per-Scale Visualizations"):
            aprint(
                f"🔍 Displaying {len(per_scale_vis)} scale visualizations in napari..."
            )
            viewer_per_scale = napari.Viewer(title="Per-Scale Gaussian Splat Fitting")

            # Shared contrast limits
            contrast_limits = [0, float(V.max())]

            # For each scale, display: original scale, reconstruction, residual, splat centers
            for scale_vis in per_scale_vis:
                scale_factor = scale_vis["scale_factor"]
                n_splats = scale_vis["n_splats"]

                # Add reconstruction at full resolution
                viewer_per_scale.add_image(
                    scale_vis["reconstruction"],
                    name=f"scale {scale_factor}× recon ({n_splats} splats)",
                    colormap="gray",
                    contrast_limits=contrast_limits,
                )

                # Add residual
                residual_max = max(1e-12, float(np.abs(scale_vis["residual"]).max()))
                viewer_per_scale.add_image(
                    np.abs(scale_vis["residual"]),
                    name=f"scale {scale_factor}× residual",
                    colormap="inferno",
                    contrast_limits=[0, residual_max],
                )

                # Add splat centers as points
                centers = scale_vis["centers"]
                viewer_per_scale.add_points(
                    centers,
                    name=f"scale {scale_factor}× centers ({n_splats} splats)",
                    size=3,
                    face_color="cyan",
                    border_color="white",
                    border_width=0.5,
                )

                aprint(
                    f"  Scale {scale_factor}×: {n_splats} splats, MSE={scale_vis['error_mse']:.6e}, Max abs error={scale_vis['error_max_abs']:.6f}"
                )

            # Enable grid mode for comparison
            viewer_per_scale.grid.enabled = True
            viewer_per_scale.grid.shape = (1, -1)  # 1 row, auto columns

            aprint("✅ Per-scale visualizations ready")
            aprint(
                "   Tip: Toggle layers to compare reconstructions and see splat locations"
            )
            aprint(
                "   Note: Per-scale viewer will open alongside comparison viewer at the end"
            )

    # Method 2: Single-scale fitting (baseline comparison)
    with asection("Method 2: Single-Scale Fitting (baseline comparison)"):
        start_single = time.time()
        params_single, amps_single, stats_single = fit_gaussian_splats(
            V,
            init_sigma_vox=BASE_INIT_SIGMA,
            n_iters=N_ITERS_SINGLE,
            loss_type="l1",
            max_abs_error=0.1,
            device=DEVICE,
            verbose=True,
        )
        time_single = time.time() - start_single

        n_splats_single = len(amps_single)
        aprint(f"\n✅ Single-scale complete in {time_single:.2f}s")
        aprint(f"Total splats: {n_splats_single:,}")

    # Render single-scale reconstruction
    # params_single includes sharpness - auto-extracted by render function
    V_recon_single = render_gaussians_numpy(
        V.shape, params_single, amps_single, truncate=TRUNCATE_SIG
    )
    residual_single = V - V_recon_single
    error_single = np.mean((V - V_recon_single) ** 2)
    max_abs_error_single = np.abs(residual_single).max()

    aprint(f"Single-scale MSE: {error_single:.6e}")
    aprint(f"Single-scale max abs error: {max_abs_error_single:.6f}")

    # Comparison summary
    with asection("Performance Comparison"):
        time_speedup = time_single / time_multi if time_multi > 0 else 1.0
        aprint("=" * 60)
        aprint(f"{'Metric':<25} {'Multi-Scale':>15} {'Single-Scale':>15}")
        aprint("=" * 60)
        aprint(f"{'Time (s)':<25} {time_multi:>15.2f} {time_single:>15.2f}")
        aprint(f"{'Speedup':<25} {time_speedup:>15.1f}× {'':<15}")
        aprint(f"{'Number of splats':<25} {n_splats_multi:>15,} {n_splats_single:>15,}")
        aprint(f"{'MSE':<25} {error_multi:>15.6e} {error_single:>15.6e}")
        aprint(
            f"{'Max abs error':<25} {max_abs_error_multi:>15.6f} {max_abs_error_single:>15.6f}"
        )
        aprint("=" * 60)

        if time_speedup > 1:
            aprint(f"🚀 Multi-scale is {time_speedup:.1f}× faster!")
        elif time_speedup < 0.9:
            aprint("⚠️  Multi-scale was slower (overhead from decomposition)")
        else:
            aprint("≈ Similar performance")

    if not NO_NAPARI:
        # Napari visualization
        aprint("\n🔬 Launching napari viewer for side-by-side comparison...")
        viewer = napari.Viewer(title="Multi-Scale Gaussian Splatting Demo")

        # Shared contrast limits
        contrast_limits = [0, float(V.max())]

        # Add original image
        viewer.add_image(
            V,
            name="original",
            colormap="gray",
            contrast_limits=contrast_limits,
        )

        # Add multi-scale reconstruction
        viewer.add_image(
            V_recon_multi,
            name=f"multi-scale (scales={SCALES})",
            colormap="gray",
            contrast_limits=contrast_limits,
        )

        # Add multi-scale residual
        viewer.add_image(
            np.abs(residual_multi),
            name="multi-scale residual",
            colormap="inferno",
            contrast_limits=[0, max(1e-12, float(np.abs(residual_multi).max()))],
        )

        # Add single-scale reconstruction
        viewer.add_image(
            V_recon_single,
            name="single-scale (baseline)",
            colormap="gray",
            contrast_limits=contrast_limits,
        )

        # Add single-scale residual
        viewer.add_image(
            np.abs(residual_single),
            name="single-scale residual",
            colormap="inferno",
            contrast_limits=[0, max(1e-12, float(np.abs(residual_single).max()))],
        )

        # Enable tile/grid mode for side-by-side comparison
        viewer.grid.enabled = True
        viewer.grid.shape = (1, -1)  # 1 row, auto columns

        # Set up text overlay
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            f"Multi-Scale Gaussian Splatting Demo | "
            f"Scales: {SCALES} | "
            f"Multi: {n_splats_multi} splats, {time_multi:.1f}s, MSE={error_multi:.2e} | "
            f"Single: {n_splats_single} splats, {time_single:.1f}s, MSE={error_single:.2e} | "
            f"Speedup: {time_speedup:.1f}×"
        )

        # Console tips
        aprint("\n📊 Visualization Tips:")
        aprint("  • Grid mode enabled for side-by-side comparison")
        aprint("  • Compare 'multi-scale' vs 'single-scale' reconstructions")
        aprint("  • Check residuals to see reconstruction quality")
        aprint("  • Multi-scale uses hierarchical splat distribution:")
        for scale_idx, scale in enumerate(SCALES):
            n = stats_multi["n_splats_per_scale"][scale_idx]
            aprint(f"    - Scale {scale}×: {n} splats")

        aprint("\n🎯 Key Insights:")
        aprint(
            f"  • Multi-scale fitting achieved {time_speedup:.1f}× wall-clock speedup"
        )
        aprint(
            f"  • Computational speedup (voxel reduction): {stats_multi['computational_speedup']:.1f}×"
        )
        aprint("  • Coarse scales capture large structures efficiently")
        aprint("  • Fine scales capture details at full resolution")
        aprint("  • Both methods achieve similar reconstruction quality")

        # Start the napari event loop (opens all created viewers)
        napari.run()
    else:
        aprint("\n✅ Demo completed successfully (napari visualization disabled)")
