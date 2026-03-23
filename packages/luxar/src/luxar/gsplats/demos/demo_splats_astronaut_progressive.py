#!/usr/bin/env python3
"""
Astronaut progressive Gaussian splatting demo.

The astronaut image is a challenging target for Gaussian splatting: rich
textures, sharp edges, fine facial detail, and high-frequency patterns in the
helmet and suit.  Progressive fitting handles this well by starting with the
large-scale structure (face, helmet silhouette) and adding detail in later
passes (wrinkles, reflections, text on suit).

Features:
- Progressive residual fitting on a hard natural image
- Per-LOD visualization: scrub from coarse to full detail
- Comparison of progressive vs the original single-fit approach
- Compression analysis at each LOD level
"""

import sys

import numpy as np
from arbol import Arbol, aprint, asection
from skimage import color, data, img_as_float32

from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy

NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("Running all computations without napari visualization...")

# ======= Demo knobs =======
MAX_SPLATS = 4000  # Total splat budget
MAX_SPLATS_PER_PASS = 1000  # Max splats per pass
ITERS_PER_PASS = 3000  # Optimization iterations per pass
PSNR_PATIENCE = 0.2  # Stop if ΔPSNR < 0.2 dB between passes
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps"
TRUNCATE_SIG = 3.0  # Rendering support truncation
# ==========================

Arbol.max_depth = 4


with asection("Astronaut Progressive Gaussian Splatting Demo"):
    aprint("Complex photograph with faces, textures, and fine detail")

    # --- Load and preprocess ---
    with asection("Loading and preprocessing data"):
        img = data.astronaut()  # RGB (512, 512, 3)
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = color.rgb2gray(img)
        V = img_as_float32(img) * 100.0  # Scale to [0, 100]

        # Crop to astronaut face and helmet region
        V = V[80:400, 120:440]  # 320x320

        aprint(f"Image shape: {V.shape}, range: [{V.min():.4f}, {V.max():.4f}]")

    # --- Progressive fitting ---
    with asection("Progressive fitting"):
        result = fit_progressive_gaussian_splats(
            V,
            max_splats=MAX_SPLATS,
            max_splats_per_pass=MAX_SPLATS_PER_PASS,
            iters_per_pass=ITERS_PER_PASS,
            psnr_patience=PSNR_PATIENCE,
            truncate=TRUNCATE_SIG,
            device=DEVICE,
            verbose=True,
            max_eccentricity=4.0,
        )

    # --- Summary ---
    with asection("Results"):
        aprint(f"Total splats: {result.n_splats:,}")
        aprint(f"LOD levels: {result.n_lods}")
        aprint(f"Final PSNR: {result.stats.get('psnr_db', 0):.2f} dB")
        aprint(f"Stop reason: {result.stats.get('stop_reason', '?')}")
        aprint(f"Total time: {result.stats.get('time_seconds', 0):.1f}s")
        aprint("")

        psnrs = result.lod_psnrs()
        for i in range(result.n_lods):
            lod = result.at_lod(i)
            cumul = result.up_to_lod(i).n_splats
            aprint(
                f"  LOD {i}: +{lod.n_splats:,} splats "
                f"(total: {cumul:,}), "
                f"PSNR = {psnrs[i]:.2f} dB"
            )

    # --- Render each LOD level cumulatively ---
    with asection("Rendering LOD levels"):
        n_lods = result.n_lods
        stack_recon = np.zeros((n_lods,) + V.shape, dtype=np.float32)
        stack_resid = np.zeros_like(stack_recon)

        for level in range(n_lods):
            data_at_level = result.up_to_lod(level)
            rendered = render_gaussians_numpy(
                V.shape, data_at_level, truncate=TRUNCATE_SIG
            )
            stack_recon[level] = rendered
            stack_resid[level] = V - rendered

# --- Napari visualization ---
if not NO_NAPARI:
    import napari

    viewer = napari.Viewer(title="Astronaut Progressive GSplat Fitting")

    viewer.add_image(
        V,
        name="Astronaut (input)",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
    )
    viewer.add_image(
        stack_recon,
        name="Reconstruction (LOD levels)",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
    )
    viewer.add_image(
        np.abs(stack_resid),
        name="Absolute residual",
        colormap="hot",
        contrast_limits=[0, max(1e-12, float(np.abs(stack_resid).max()))],
    )

    try:
        viewer.dims.axis_labels = ["LOD level", "y", "x"]
    except Exception:
        pass

    def _update_overlay(event=None) -> None:
        t = int(viewer.dims.current_step[0])
        if t < n_lods:
            n_at_level = result.up_to_lod(t).n_splats
            psnr_val = psnrs[t]
            viewer.text_overlay.visible = True
            viewer.text_overlay.text = (
                f"LOD 0..{t}  |  {n_at_level:,} splats  |  "
                f"PSNR = {psnr_val:.2f} dB"
            )

    _update_overlay()
    viewer.dims.events.current_step.connect(_update_overlay)

    aprint("Ready. Use the LOD slider to see coarse-to-fine reconstruction.")
    aprint("Notice: LOD 0 captures face/helmet silhouette, later LODs add fine detail.")
    napari.run()
else:
    aprint("\nDemo completed successfully (napari visualization disabled)")
