#!/usr/bin/env python3
"""
Coins progressive Gaussian splatting demo.

Progressive (multi-pass) fitting on the scikit-image coins dataset.
Each pass fits splats to the residual of the previous approximation, building
a multi-LOD representation from coarse to fine.

Features:
- Progressive residual fitting on metallic textures with circular objects
- Per-LOD visualization: scrub from coarse to full detail
- Illumination gradients and fine surface texture capture
- Compression analysis at each LOD level

Data source: scikit-image coins (303x384 grayscale)
"""

import sys

import numpy as np
from arbol import Arbol, aprint, asection
from skimage import data, img_as_float32

from luxar.gsplats.demos._demo_common import psnr
from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
from luxar.gsplats.lod import make_additive_lod
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy

NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("Running all computations without napari visualization...")

# ======= Demo knobs =======
MAX_SPLATS = 6000  # Total splat budget (0.3 ratio = rich texture in single-shot)
MAX_SPLATS_PER_PASS = 1000  # Max splats per pass
ITERS_PER_PASS = 3000  # Optimization iterations per pass
PSNR_PATIENCE = 0.2  # Stop if ΔPSNR < 0.2 dB between passes
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps"
TRUNCATE_SIG = 3.0  # Rendering support truncation
N_LODS = 4  # Additive-LOD ladder size for the post-fit ordering
# ==========================

Arbol.max_depth = 4


with asection("Coins Progressive Gaussian Splatting Demo"):
    aprint("Progressive fitting on metallic textures with circular objects")

    # --- Load and preprocess ---
    with asection("Loading and preprocessing data"):
        img = data.coins()  # Grayscale (303, 384)
        V = img_as_float32(img) * 100.0  # Scale to [0, 100]
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
        )

    # --- Build the additive LOD ladder (post-fit, principled) ---
    with asection(f"Building additive LOD ladder ({N_LODS} levels)"):
        result = make_additive_lod(result, n_lods=N_LODS, method="greedy")
        aprint(f"Cutpoints: {result.stats.get('lod_cutpoints', [])}")

    # --- Summary ---
    with asection("Results"):
        aprint(f"Total splats: {result.n_splats:,}")
        aprint(f"LOD levels (additive ladder): {result.n_additive_sublods}")
        aprint(f"Final PSNR (fit): {result.stats.get('psnr_db', 0):.2f} dB")
        aprint(f"Stop reason: {result.stats.get('stop_reason', '?')}")
        aprint(f"Total time: {result.stats.get('time_seconds', 0):.1f}s")
        aprint(
            f"Fitting passes: {result.stats.get('n_passes', '?')} "
            f"(per-pass PSNRs: "
            f"{[f'{p:.2f}' for p in result.stats.get('pass_psnrs', [])]})"
        )

    # --- Render each LOD level cumulatively ---
    psnrs: list[float] = []
    with asection("Rendering LOD levels"):
        n_lods = result.n_additive_sublods
        stack_recon = np.zeros((n_lods,) + V.shape, dtype=np.float32)
        stack_resid = np.zeros_like(stack_recon)

        for level in range(n_lods):
            data_at_level = result.additive_prefix(level)
            rendered = render_gaussians_numpy(
                V.shape, data_at_level, truncate=TRUNCATE_SIG
            )
            stack_recon[level] = rendered
            stack_resid[level] = V - rendered
            psnr_val = psnr(rendered, V)
            psnrs.append(psnr_val)
            lod_n = result.additive_sublod(level).n_splats
            aprint(
                f"  LOD {level}: +{lod_n:,} splats "
                f"(total: {data_at_level.n_splats:,}), "
                f"PSNR = {psnr_val:.2f} dB"
            )

# --- Napari visualization ---
if not NO_NAPARI:
    import napari

    viewer = napari.Viewer(title="Coins Progressive GSplat Fitting")

    viewer.add_image(
        V,
        name="Input (coins)",
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
        colormap="copper",
        contrast_limits=[0, max(1e-12, float(np.abs(stack_resid).max()))],
    )

    try:
        viewer.dims.axis_labels = ["LOD level", "y", "x"]
    except Exception:
        pass

    def _update_overlay(event=None) -> None:
        t = int(viewer.dims.current_step[0])
        if t < n_lods:
            n_at_level = result.additive_prefix(t).n_splats
            psnr_val = psnrs[t]
            viewer.text_overlay.visible = True
            viewer.text_overlay.text = (
                f"LOD 0..{t}  |  {n_at_level:,} splats  |  PSNR = {psnr_val:.2f} dB"
            )

    _update_overlay()
    viewer.dims.events.current_step.connect(_update_overlay)

    aprint("Ready. Use the LOD slider to see coarse-to-fine reconstruction.")
    aprint("Notice: LOD 0 captures coin outlines, later LODs add metallic detail.")
    napari.run()
else:
    aprint("\nDemo completed successfully (napari visualization disabled)")
