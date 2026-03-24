#!/usr/bin/env python3
"""
3D DAPI — Progressive fitting convergence movie for pass 1.

Runs 2 progressive passes on the DAPI dataset, then shows the optimization
convergence movie for pass 1 (fitting the residual) in napari. This lets
you see how splats converge on the sparse residual frame by frame.

Usage:
    python demo_3d_dapi_progressive_movie.py
    python demo_3d_dapi_progressive_movie.py --no-napari  # just compute, no viewer
"""

import sys

import numpy as np
import zarr
from arbol import Arbol, aprint, asection

from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData, GSplatLOD
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

NO_NAPARI = "--no-napari" in sys.argv

# ======= Demo knobs =======
MAX_SPLATS_PER_PASS = 1000
ITERS_PASS_0 = 3000  # Iterations for pass 0 (original volume)
ITERS_PASS_1 = 3000  # Iterations for pass 1 (residual) — with movie
MOVIE_EVERY = 50  # Record a movie frame every N iterations
DEVICE = None
TRUNCATE_SIG = 3.0
ZARR_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
DAPI_CHANNEL = 1
TARGET_SIZE = 128
# ==========================

Arbol.max_depth = 5


def _load_dapi_volume() -> np.ndarray:
    """Load DAPI volume from IDR, with synthetic fallback."""
    try:
        import fsspec

        aprint(f"Loading from {ZARR_URL}")
        mapper = fsspec.get_mapper(ZARR_URL)
        try:
            store = zarr.open_group(mapper, mode="r")
        except (zarr.errors.PathNotFoundError, zarr.errors.GroupNotFoundError):
            store = zarr.open_array(mapper, mode="r")

        data = store["0"]
        full_shape = data.shape
        aprint(f"OME-ZARR shape: {full_shape}")

        if len(full_shape) == 5:
            ch = min(DAPI_CHANNEL, full_shape[1] - 1)
            vol = np.array(data[0, ch, :, :, :], dtype=np.float32)
        elif len(full_shape) == 4:
            ch = min(DAPI_CHANNEL, full_shape[0] - 1)
            vol = np.array(data[ch, :, :, :], dtype=np.float32)
        else:
            vol = np.array(data[:, :, :], dtype=np.float32)

        if TARGET_SIZE:
            from scipy.ndimage import zoom

            factors = [TARGET_SIZE / s for s in vol.shape]
            vol = zoom(vol, factors, order=1)
            aprint(f"Downscaled to: {vol.shape}")

        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = ((vol - vmin) / (vmax - vmin)) * 100.0
        return vol.astype(np.float32)

    except Exception as exc:
        aprint(f"Remote load failed: {exc}, using synthetic fallback")
        rng = np.random.RandomState(42)
        shape = (TARGET_SIZE, TARGET_SIZE, TARGET_SIZE)
        vol = np.zeros(shape, dtype=np.float32)
        for _ in range(10):
            center = [rng.uniform(10, s - 10) for s in shape]
            sigma = rng.uniform(4, 8)
            amp = rng.uniform(60, 100)
            grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
            dist_sq = sum((g - c) ** 2 for g, c in zip(grids, center))
            vol += amp * np.exp(-dist_sq / (2 * sigma**2))
        return np.clip(vol, 0, 100).astype(np.float32)


with asection("Progressive fitting convergence movie (pass 1)"):
    # --- Load data ---
    with asection("Loading DAPI data"):
        V = _load_dapi_volume()
        aprint(f"Volume: {V.shape}, range: [{V.min():.2f}, {V.max():.2f}]")

    # --- Pass 0: fit original volume (no movie, just get the splats) ---
    with asection(f"Pass 0: fitting {MAX_SPLATS_PER_PASS} splats to original volume"):
        pass0_result = fit_gaussian_splats(
            V,
            seeds=MAX_SPLATS_PER_PASS,
            n_iters=ITERS_PASS_0,
            cull_ratio=1.0,
            truncate=TRUNCATE_SIG,
            device=DEVICE,
            verbose=True,
            max_eccentricity=6.0,
        )
        aprint(f"Pass 0: {pass0_result.n_splats} splats")

    # --- Compute residual for pass 1 (matching fit_progressive_gaussian_splats) ---
    with asection("Computing residual"):
        import torch

        with torch.no_grad():
            rendered = render_to_volume_tensor(
                pass0_result, shape=V.shape, device=DEVICE, truncate=TRUNCATE_SIG
            )
            V_tensor = torch.from_numpy(V).to(rendered.device)
            residual = torch.clamp(V_tensor - rendered, min=0).cpu().numpy()

        residual_max = float(residual.max())
        residual_nonzero_frac = float((residual > 0.1).sum()) / residual.size
        aprint(f"Residual (raw): max={residual_max:.2f}, non-zero={residual_nonzero_frac:.1%}")

        # Threshold the residual: zero out diffuse background
        # (same logic as fit_progressive_gaussian_splats)
        nonzero_vals = residual[residual > 0]
        if len(nonzero_vals) > 0:
            residual_threshold = float(np.median(nonzero_vals))
        else:
            residual_threshold = 0.0
        n_before = int((residual > 0).sum())
        residual[residual < residual_threshold] = 0.0
        n_after = int((residual > 0).sum())
        aprint(
            f"Thresholding: zeroed {n_before - n_after:,} of {n_before:,} voxels "
            f"(threshold={residual_threshold:.3f} = median of non-zero residual)"
        )
        aprint(f"Residual (thresholded): non-zero={n_after:,} ({100*n_after/residual.size:.1f}%)")

    # --- Pass 1: fit residual WITH convergence movie ---
    with asection(f"Pass 1: fitting {MAX_SPLATS_PER_PASS} splats to residual (with movie)"):
        pass1_result = fit_gaussian_splats(
            residual,
            seeds=MAX_SPLATS_PER_PASS,
            n_iters=ITERS_PASS_1,
            cull_ratio=1.0,
            truncate=TRUNCATE_SIG,
            device=DEVICE,
            verbose=True,
            max_eccentricity=6.0,
            seed_method="peaks",
            napari_movie=True,
            movie_every=MOVIE_EVERY,
            movie_max_frames=200,
        )
        aprint(f"Pass 1: {pass1_result.n_splats} splats")

    # --- Show convergence movie ---
    movie_frames = pass1_result.stats.get("movie_frames")

    if movie_frames is None:
        aprint("No movie frames recorded!")
    else:
        n_frames = len(movie_frames["reconstruction"])
        aprint(f"Recorded {n_frames} movie frames")

        if not NO_NAPARI and n_frames > 0:
            import napari

            # Build stacks: (time, z, y, x)
            recon_stack = np.array(movie_frames["reconstruction"])
            residual_stack = np.array(movie_frames["residual"])
            target_vol = movie_frames["target"]  # single 3D array
            iterations = movie_frames["iterations"]

            viewer = napari.Viewer(
                title=f"Pass 1 convergence movie ({n_frames} frames, "
                f"iterations {iterations[0]}..{iterations[-1]})",
                ndisplay=3,
            )

            # Target (the thresholded residual that pass 1 was actually fitting)
            viewer.add_image(
                target_vol,
                name="Target (thresholded residual)",
                colormap="magma",
                contrast_limits=[0, residual_max],
                rendering="mip",
            )

            # Reconstruction evolving over time
            viewer.add_image(
                recon_stack,
                name="Pass 1 reconstruction (convergence)",
                colormap="magma",
                contrast_limits=[0, residual_max],
                rendering="mip",
            )

            # Absolute fit residual over time
            viewer.add_image(
                residual_stack,
                name="Fit residual (|target - reconstruction|)",
                colormap="hot",
                contrast_limits=[0, residual_max * 0.5],
                rendering="mip",
                visible=False,
            )

            try:
                viewer.dims.axis_labels = ["iteration", "z", "y", "x"]
            except Exception:
                pass

            def _update_overlay(event=None) -> None:
                t = int(viewer.dims.current_step[0])
                if t < n_frames:
                    it = iterations[t]
                    viewer.text_overlay.visible = True
                    viewer.text_overlay.text = (
                        f"Iteration {it}  |  Frame {t + 1}/{n_frames}"
                    )

            _update_overlay()
            viewer.dims.events.current_step.connect(_update_overlay)

            viewer.camera.angles = (45, 45, 45)
            viewer.camera.zoom = 2.0

            aprint("Use the iteration slider to scrub through convergence.")
            aprint("Watch how splats appear and grow to cover the residual peaks.")
            napari.run()
        elif not NO_NAPARI:
            aprint("No frames to display")
        else:
            aprint(f"Movie recorded ({n_frames} frames), napari disabled")
