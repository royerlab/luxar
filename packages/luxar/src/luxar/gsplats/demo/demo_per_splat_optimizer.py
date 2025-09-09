#!/usr/bin/env python3
"""
Demo showcasing per-splat Adam optimizer with napari visualization.

This demo is equivalent to demo_splats.py but uses the new per-splat optimizer
to eliminate the global disruption problem during dynamic operations.
"""

import napari
import numpy as np
import torch
from arbol import aprint, asection
from skimage import data, filters

from luxar.gsplats.candidates import find_candidates_overcomplete_nd
from luxar.gsplats.dynamic_ops import DynamicOpsConfig, apply_dynamic_operations
from luxar.gsplats.models.gsplats.gsplat_model import (
    GaussianSplatModel,
    render_gaussians_numpy,
)
from luxar.gsplats.optim import create_per_splat_optimizer_setup
from luxar.gsplats.utils.trils import pack_tril

# ======= Demo Configuration =======
N_ITERS = 500
USE_DYNAMIC_OPS = True  # Enable dynamic operations with per-splat optimizer
RECORD_MOVIE = True  # Record optimization movie for napari visualization
MOVIE_EVERY = 5  # Record every N iterations
# ==================================


def ellipse_polygon_from_L(
    mu_yx: np.ndarray, L: np.ndarray, t: float = 2.0, n_pts: int = 64
) -> np.ndarray:
    """
    Build a polygon approximating the 2D ellipse from Cholesky factor L.
    """
    Sigma = L @ L.T  # (2,2)
    # Eigen-decompose Sigma for principal axes
    evals, evecs = np.linalg.eigh(Sigma)  # evals >= 0
    evals = np.clip(evals, 1e-12, None)
    # Radii along principal axes at level t: r_i = t * sqrt(lambda_i)
    radii = t * np.sqrt(evals)  # (2,)
    # Parametric angles
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2, n_pts)
    # Map unit circle -> ellipse in data coords: μ + R diag(r) circle
    pts = (evecs @ (radii[:, None] * circle)).T + mu_yx[None, :]
    return pts.astype(np.float32)


def main():
    """Main demo with per-splat optimizer and napari visualization."""

    aprint("🚀 Per-Splat Optimizer Demo with Napari")

    with asection("Creating test data"):
        # Create soft 2D "blobs" image (same as original demo)
        blobs = data.binary_blobs(
            length=256, blob_size_fraction=0.06, n_dim=2, volume_fraction=0.18, rng=42
        ).astype(float)
        V = filters.gaussian(blobs, sigma=3.25).astype(np.float32)
        aprint(f"Target image shape: {V.shape}")

    with asection("Finding candidates"):
        # Find overcomplete candidate centers
        centers = find_candidates_overcomplete_nd(
            V,
            scales=(0.8, 1.2, 1.8, 2.6, 3.6),
            peaks_per_scale=900,
            percentile_thresh=95,
            min_dist=2.0,
            add_intensity_grid=False,
        )
        aprint(f"Found {len(centers)} candidates")

    with asection("Setting up per-splat optimizer"):
        # Build model
        d = len(V.shape)
        N = len(centers)
        centers0 = np.array(centers, dtype=np.float32)
        L0 = np.stack([np.eye(d) * 1.6] * N).astype(np.float32)
        amps0 = np.full(N, 0.1, dtype=np.float32)

        model = GaussianSplatModel(
            shape=V.shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.5, 0.5],
            device=torch.device("cpu"),
        )

        # Create per-splat optimizer setup
        optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
            model, lr=0.2, scheduler_type="plateau", patience=10, factor=0.5
        )

        aprint(f"✓ Per-splat optimizer for {model.n_splats()} splats")
        aprint(
            f"✓ Individual learning rates: mean={optimizer.get_effective_learning_rates().mean():.6f}"
        )

    # Configure dynamic operations
    dynamic_config = None
    if USE_DYNAMIC_OPS:
        dynamic_config = DynamicOpsConfig()
        aprint(f"Dynamic operations enabled (step_every={dynamic_config.step_every})")

    with asection(f"Optimization with Per-Splat Adam ({N_ITERS} iterations)"):
        V_t = torch.tensor(V, dtype=torch.float32, device=torch.device("cpu"))

        loss_history = []
        best_loss = float("inf")

        # Movie recording setup
        movie_frames = None
        if RECORD_MOVIE:
            movie_frames = {
                "target": [],
                "reconstruction": [],
                "residual": [],
                "iterations": [],
            }
            aprint(f"🎬 Recording optimization movie (every {MOVIE_EVERY} iterations)")

        for it in range(1, N_ITERS + 1):
            # Standard optimization step
            optimizer.zero_grad()
            pred = model()
            loss = torch.nn.functional.mse_loss(pred, V_t)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            current_loss = loss.item()
            loss_history.append(current_loss)
            best_loss = min(best_loss, current_loss)

            # Record movie frames
            if RECORD_MOVIE and movie_frames is not None and it % MOVIE_EVERY == 0:
                with torch.no_grad():
                    target_frame = V_t.cpu().numpy()
                    pred_frame = pred.detach().cpu().numpy()
                    residual_frame = torch.abs(V_t - pred.detach()).cpu().numpy()

                    movie_frames["target"].append(target_frame)
                    movie_frames["reconstruction"].append(pred_frame)
                    movie_frames["residual"].append(residual_frame)
                    movie_frames["iterations"].append(it)

            # Dynamic operations (the key difference - no momentum loss!)
            if (
                USE_DYNAMIC_OPS
                and dynamic_config
                and it % dynamic_config.step_every == 0
            ):
                old_n = model.n_splats()

                _, _, operations_occurred = apply_dynamic_operations(
                    model,
                    optimizer,
                    scheduler,
                    V_t,
                    dynamic_config,
                    lr=0.2,
                    device=torch.device("cpu"),
                    verbose=True,
                    napari_debug=False,
                )

                if operations_occurred:
                    new_n = model.n_splats()
                    aprint(f"  → Topology change: {old_n} → {new_n} splats")

                    # Show per-splat optimizer benefits
                    lrs = optimizer.get_effective_learning_rates()
                    aprint(
                        f"  → LR stats: mean={lrs.mean():.6f}, std={lrs.std():.6f}, range=[{lrs.min():.6f}, {lrs.max():.6f}]"
                    )
                    aprint("  ✅ No global momentum loss - smooth continuation!")

            # Progress logging
            if it % max(1, N_ITERS // 10) == 0 or it <= 5:
                with torch.no_grad():
                    rel = torch.linalg.norm((pred - V_t).flatten()) / torch.linalg.norm(
                        V_t.flatten()
                    )
                aprint(
                    f"[{it:4d}/{N_ITERS}] loss={current_loss:.5g}  relL2={float(rel):.4f}  N={model.n_splats()}"
                )

    with asection("Extracting final results"):
        # Extract final parameters in correct format
        with torch.no_grad():
            centers, Ls, amps = model.current_params()
            params_full = []
            for i in range(len(amps)):
                L = Ls[i].detach().cpu().numpy()  # (d, d) lower triangular matrix
                center = centers[i].detach().cpu().numpy()  # (d,) center coordinates

                # Pack only the lower triangular elements of L
                packed_L = pack_tril(L[None, :, :])[0]  # Add batch dim, then remove
                params_full.append(np.concatenate([center, packed_L]))

            params_full = np.array(params_full)
            amps_final = amps.detach().cpu().numpy()

            # Generate reconstruction
            V_recon = render_gaussians_numpy(V.shape, params_full, amps_final)

        aprint(f"Final parameters shape: {params_full.shape}")
        aprint(f"Final amplitudes shape: {amps_final.shape}")
        aprint(f"Final MSE: {np.mean((V - V_recon) ** 2):.6f}")

        # Prepare movie data if recorded
        if RECORD_MOVIE and movie_frames:
            n_frames = len(movie_frames["iterations"])
            aprint(
                f"🎬 Movie ready: {n_frames} frames from iterations {movie_frames['iterations'][0]} to {movie_frames['iterations'][-1]}"
            )

    with asection("Napari visualization"):
        # Create napari viewer
        viewer = napari.Viewer(title="Per-Splat Optimizer Demo Results")

        # Add optimization movie if recorded
        if RECORD_MOVIE and movie_frames:
            # Stack frames into time series arrays
            target_stack = np.stack(movie_frames["target"])  # (n_frames, H, W)
            recon_stack = np.stack(movie_frames["reconstruction"])  # (n_frames, H, W)
            residual_stack = np.stack(movie_frames["residual"])  # (n_frames, H, W)

            # Add as time series images
            viewer.add_image(
                target_stack,
                name="🎬 Target Movie",
                colormap="viridis",
                opacity=0.8,
                scale=(MOVIE_EVERY, 1, 1),  # Scale time axis for proper spacing
            )
            viewer.add_image(
                recon_stack,
                name="🎬 Reconstruction Movie",
                colormap="plasma",
                opacity=0.8,
                scale=(MOVIE_EVERY, 1, 1),
            )
            viewer.add_image(
                residual_stack,
                name="🎬 Residual Movie",
                colormap="hot",
                opacity=0.8,
                scale=(MOVIE_EVERY, 1, 1),
            )

            aprint("🎬 Use the time slider to scrub through optimization progress!")

        # Add final static images
        viewer.add_image(V, name="Final Target", colormap="viridis", opacity=0.7)
        viewer.add_image(
            V_recon, name="Final Reconstruction", colormap="plasma", opacity=0.7
        )
        viewer.add_image(np.abs(V - V_recon), name="Final Residual", colormap="hot")

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

        # Add ellipses for splat shapes (sample of first 50 for performance)
        n_show = min(50, len(params_full))
        ellipse_data = []

        for i in range(n_show):
            if amps_final[i] > 0.01:  # Only show significant splats
                center_yx = params_full[i, :2]
                packed_L = params_full[i, 2:]

                # Reconstruct L from packed format
                L_recon = np.zeros((2, 2))
                L_recon[0, 0] = packed_L[0]  # L00
                L_recon[1, 0] = packed_L[1]  # L10
                L_recon[1, 1] = packed_L[2]  # L11

                # Generate ellipse polygon
                ellipse_pts = ellipse_polygon_from_L(
                    center_yx, L_recon, t=2.0, n_pts=32
                )
                ellipse_data.append(ellipse_pts)

        if ellipse_data:
            viewer.add_shapes(
                ellipse_data,
                shape_type="polygon",
                name=f"Splat Ellipses (top {len(ellipse_data)})",
                face_color="transparent",
                edge_color="yellow",
                edge_width=1,
                opacity=0.6,
            )

        aprint(f"✓ Napari viewer with {len(centers_np)} splat centers")
        aprint(f"✓ Showing {len(ellipse_data)} ellipse shapes")

        # Add text info
        movie_info = ""
        if RECORD_MOVIE and movie_frames:
            n_frames = len(movie_frames["iterations"])
            movie_info = f"""
🎬 Optimization Movie: {n_frames} frames
📹 Use time slider to see progress!
🔄 Toggle layer visibility to compare
"""

        info_text = f"""Per-Splat Optimizer Demo
Target: {V.shape} image
Splats: {len(amps_final)}
Final MSE: {np.mean((V - V_recon) ** 2):.6f}
Dynamic ops: {"Enabled" if USE_DYNAMIC_OPS else "Disabled"}{movie_info}
✅ No momentum loss during dynamic operations!
✅ Individual learning rates per splat
✅ Smooth optimization trajectory
        """

        viewer.text_overlay.text = info_text
        viewer.text_overlay.visible = True

        aprint("🎉 Per-splat optimizer demo complete!")
        if RECORD_MOVIE and movie_frames:
            aprint(
                "🎬 Movie recorded - use the time slider to watch optimization progress!"
            )
            aprint(
                "📹 Toggle layer visibility to compare target/reconstruction/residual"
            )
        aprint("Napari viewer opened - explore the results!")

        # Run napari
        napari.run()


if __name__ == "__main__":
    main()
