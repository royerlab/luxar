#!/usr/bin/env python3
"""
Multi-Scale Gaussian Splatting Demo

Demonstrates multi-scale Gaussian splat fitting:
1. Fit Gaussian splats per scale with intermediate results
2. Show decomposed scale images via napari
3. Display each scale's target, reconstruction, residual (close napari to continue)
4. Show final combined reconstruction
"""

import numpy as np
from arbol import aprint, asection
from skimage import color, data, img_as_float32

from luxar.gsplats import fit_multiscale_gaussian_splats
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.multiscale import show_optimization_movie, upsample_for_visualization

# ======= Demo Configuration =======
SCALES = [1, 2, 4]  # Scale factors (must be powers of 2)
N_ITERS_DECOMP = 5000  # Decomposition needs many iterations to converge well
N_ITERS_PER_SCALE = 5000  # Per-scale fitting iterations
COMPRESSION_RATIO = 0.15  # 15% compression (float) - or use int for absolute seed count
# ==================================


def load_mitosis_image() -> np.ndarray:
    """Load and preprocess the Human Mitosis image (matching working demo)."""
    with asection("Loading Human Mitosis image"):
        img = data.human_mitosis()

        # Convert to grayscale if needed (match working demo)
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = color.rgb2gray(img) * 100
        V = img_as_float32(img)

        # Crop to 256×256 for faster demo (match working demo)
        V = V[100:356, 100:356]

        aprint(f"Image shape: {V.shape}")
        aprint(f"Value range: [{V.min():.4f}, {V.max():.4f}]")

    return V


def show_decomposition(
    V: np.ndarray, scale_images: list, scales: list, energy_dist: list
) -> None:
    """Show the decomposed scale images via napari (upsampled to full resolution)."""
    import napari

    with asection("Displaying Decomposition"):
        aprint(f"Opening napari to show {len(scales)} decomposed scales...")
        aprint("Upsampling all scales to full resolution for visualization...")

        # Upsample all scales to original resolution for proper comparison
        scales_upsampled = []
        for scale_factor, scale_img in zip(scales, scale_images):
            if scale_img.shape != V.shape:
                img_upsampled = upsample_for_visualization(scale_img, V.shape, "cubic")
            else:
                img_upsampled = scale_img
            scales_upsampled.append(img_upsampled)

        # Verify reconstruction
        reconstruction = np.sum(scales_upsampled, axis=0)
        mse = np.mean((V - reconstruction) ** 2)
        aprint(f"Reconstruction MSE: {mse:.6e}")
        aprint("Close the napari window to continue.")

        viewer = napari.Viewer(title="Multi-Scale Decomposition")

        # Use consistent contrast limits
        contrast_limits = [0, float(V.max())]

        # Add original image
        viewer.add_image(
            V, name="Original", colormap="gray", contrast_limits=contrast_limits
        )

        # Add reconstruction
        viewer.add_image(
            reconstruction,
            name="Reconstruction (sum of scales)",
            colormap="gray",
            contrast_limits=contrast_limits,
        )

        # Add each scale (upsampled)
        for i, (scale_factor, img_up) in enumerate(zip(scales, scales_upsampled)):
            energy_pct = energy_dist[i] * 100 if i < len(energy_dist) else 0
            viewer.add_image(
                img_up,
                name=f"Scale {scale_factor}× ({energy_pct:.1f}%)",
                colormap="viridis",
                contrast_limits=contrast_limits,
                blending="additive",
            )

        # Add residual
        residual = np.abs(V - reconstruction)
        viewer.add_image(
            residual,
            name="Residual (abs)",
            colormap="inferno",
            contrast_limits=[0, max(1e-6, float(residual.max()))],
        )

        viewer.grid.enabled = True
        viewer.grid.shape = (-1, 3)

        napari.run()


def show_scale_fitting(
    scale_factor: int,
    target: np.ndarray,
    reconstruction: np.ndarray,
    residual: np.ndarray,
    n_splats: int,
) -> None:
    """Show a single scale's target, reconstruction, and residual via napari."""
    import napari

    with asection(f"Scale {scale_factor}×"):
        aprint(f"Shape: {target.shape}, Splats: {n_splats}")
        aprint(f"MSE: {np.mean(residual**2):.6e}")
        aprint(f"Max abs error: {np.abs(residual).max():.4f}")
        aprint("Close napari window to continue to next scale...")

        viewer = napari.Viewer(
            title=f"Scale {scale_factor}× Fitting ({n_splats} splats)"
        )

        contrast_limits = [0, float(target.max())]

        viewer.add_image(
            target,
            name="Target (decomposed)",
            colormap="gray",
            contrast_limits=contrast_limits,
        )

        viewer.add_image(
            reconstruction,
            name="Reconstruction",
            colormap="gray",
            contrast_limits=contrast_limits,
        )

        viewer.add_image(
            np.abs(residual),
            name="Residual (abs)",
            colormap="inferno",
            contrast_limits=[0, max(1e-6, float(np.abs(residual).max()))],
        )

        # Enable grid view for side-by-side comparison
        viewer.grid.enabled = True
        viewer.grid.shape = (1, -1)

        napari.run()


def show_final_reconstruction(
    V: np.ndarray,
    reconstruction: np.ndarray,
    result,
) -> None:
    """Show the final combined reconstruction."""
    import napari

    residual = V - reconstruction

    with asection("Final Combined Reconstruction"):
        aprint(f"Total splats: {len(result.amplitudes):,}")
        aprint(f"Splats per scale: {result.stats['n_splats_per_scale']}")
        aprint(f"MSE: {np.mean(residual**2):.6e}")
        aprint(f"Max abs error: {np.abs(residual).max():.4f}")
        aprint("Opening napari for final comparison...")

        viewer = napari.Viewer(title="Final Multi-Scale Reconstruction")

        contrast_limits = [0, float(V.max())]

        viewer.add_image(
            V, name="Original", colormap="gray", contrast_limits=contrast_limits
        )

        viewer.add_image(
            reconstruction,
            name=f"Reconstruction ({len(result.amplitudes)} splats)",
            colormap="gray",
            contrast_limits=contrast_limits,
        )

        viewer.add_image(
            np.abs(residual),
            name="Residual (abs)",
            colormap="inferno",
            contrast_limits=[0, max(1e-6, float(np.abs(residual).max()))],
        )

        viewer.grid.enabled = True
        viewer.grid.shape = (1, -1)

        napari.run()


def main() -> None:
    """Run the multi-scale fitting demo."""
    with asection("Multi-Scale Gaussian Splatting Demo"):
        # Step 1: Load image
        V = load_mitosis_image()

        # Step 2: Multi-scale fitting with intermediate results
        # (decomposition happens internally, results returned via return_intermediate)
        with asection("Step 1: Multi-Scale Decomposition + Fitting"):
            aprint(f"Scales: {SCALES}")
            aprint(f"Decomposition iterations: {N_ITERS_DECOMP}")
            aprint(f"Fitting iterations per scale: {N_ITERS_PER_SCALE}")
            aprint(f"Compression ratio: {COMPRESSION_RATIO:.0%}")

            result = fit_multiscale_gaussian_splats(
                V,
                seeds=COMPRESSION_RATIO,  # Float = compression ratio (0.15 = 15%)
                scales=SCALES,
                n_iters_decomp=N_ITERS_DECOMP,
                n_iters_per_scale=N_ITERS_PER_SCALE,
                return_intermediate=True,
                napari_movie=True,
                verbose=True,
            )

            # Get decomposition results from intermediate data
            scale_images = result.stats.get("scale_images", [])
            decomp_stats = result.stats.get("decomposition_stats", {})
            energy_dist = decomp_stats.get("energy_distribution", [])
            aprint(f"Energy distribution: {[f'{e:.1%}' for e in energy_dist]}")

            # Show the decomposition optimization movie if recorded
            movie_frames = decomp_stats.get("movie_frames")
            if movie_frames and movie_frames.get("iterations"):
                interpolation = decomp_stats.get("interpolation", "cubic")
                aprint(
                    f"Showing decomposition optimization movie ({len(movie_frames['iterations'])} frames)..."
                )
                show_optimization_movie(
                    movie_frames, V.shape, interpolation=interpolation
                )
            else:
                aprint(
                    "No movie frames recorded (napari_movie may not have been enabled)"
                )

        # Step 3: Display decomposition
        with asection("Step 2: Viewing Decomposition"):
            show_decomposition(V, scale_images, SCALES, energy_dist)

        # Step 4: Display each scale's results
        with asection("Step 3: Per-Scale Results"):
            intermediate = result.stats.get("intermediate", [])

            for inter in intermediate:
                show_scale_fitting(
                    scale_factor=inter["scale_factor"],
                    target=inter["target"],
                    reconstruction=inter["reconstruction"],
                    residual=inter["residual"],
                    n_splats=len(inter["splats"].amplitudes),
                )

        # Step 5: Show final combined result
        with asection("Step 4: Combined Reconstruction"):
            final_reconstruction = render_gaussians_numpy(V.shape, result)

        show_final_reconstruction(V, final_reconstruction, result)

        aprint("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
