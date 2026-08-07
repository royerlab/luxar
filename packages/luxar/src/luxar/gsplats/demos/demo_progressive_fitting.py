#!/usr/bin/env python3
"""
Progressive Gaussian splatting demo on the human mitosis dataset.

Demonstrates the progressive fitting approach (fit splats in multiple passes,
each targeting the residual of the previous approximation) followed by a
post-fit *additive-LOD* construction (greedy ordering + cutpoints) — see
:mod:`luxar.gsplats.lod.additive`.  The fitter itself returns a single
flattened ``GSplatData``; the LOD ladder is built explicitly here.

Features:
- Progressive residual fitting with automatic PSNR-based stopping.
- Principled additive LOD ladder (greedy / matching-pursuit ordering).
- Napari visualization: scrub through LOD levels to see coarse-to-fine.
- Per-LOD PSNR computed from the rendered prefix sums.
"""

import sys

import napari
import numpy as np
from arbol import Arbol, aprint, asection
from skimage import color, data, img_as_float32

from luxar.gsplats.demos._demo_common import psnr
from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
from luxar.gsplats.lod import make_additive_lod
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("Running all computations without napari visualization...")

# ======= Demo knobs =======
MAX_SPLATS = 3000  # Total splat budget
MAX_SPLATS_PER_PASS = 500  # Max splats per pass (actual may be lower after culling)
ITERS_PER_PASS = 2000  # Optimization iterations per pass
PSNR_PATIENCE = 0.1  # Stop if ΔPSNR < 0.1 dB between passes
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps"
TRUNCATE_SIG = 3.0  # Rendering support truncation (≈ ±3σ)
N_LODS = 4  # Number of additive-LOD levels for the post-fit ladder
# ==========================

Arbol.max_depth = 4


with asection("Progressive Gaussian Splatting Demo (Human Mitosis)"):
    # --- Load and preprocess ---
    with asection("Loading and preprocessing data"):
        img = data.human_mitosis()
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = color.rgb2gray(img) * 100
        V = img_as_float32(img)
        V = V[100:356, 100:356]  # Crop to 256x256
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

    # --- Render each LOD level cumulatively for visualization ---
    psnrs: list[float] = []
    with asection("Rendering LOD levels"):
        n_lods = result.n_additive_sublods
        stack_recon = np.zeros((n_lods,) + V.shape, dtype=np.float32)
        stack_resid = np.zeros_like(stack_recon)

        for level in range(n_lods):
            # Render splats up to this LOD level (additive prefix).
            data_at_level = result.additive_prefix(level)
            rendered = render_gaussians_numpy(
                V.shape, data_at_level, truncate=TRUNCATE_SIG
            )
            stack_recon[level] = rendered
            stack_resid[level] = V - rendered
            psnr_val = psnr(rendered, V)
            psnrs.append(psnr_val)
            aprint(
                f"LOD 0..{level}: {data_at_level.n_splats:,} splats, "
                f"PSNR = {psnr_val:.2f} dB"
            )

# --- Napari visualization ---
if not NO_NAPARI:
    viewer = napari.Viewer(title="Progressive GSplat Fitting")

    viewer.add_image(
        V,
        name="Input (human mitosis)",
        colormap="magma",
        contrast_limits=[0, float(V.max())],
    )
    viewer.add_image(
        stack_recon,
        name="Reconstruction (LOD levels)",
        colormap="magma",
        contrast_limits=[0, float(V.max())],
    )
    viewer.add_image(
        np.abs(stack_resid),
        name="Absolute residual",
        colormap="inferno",
        contrast_limits=[0, max(1e-12, float(np.abs(stack_resid).max()))],
    )

    # Axis labels
    try:
        viewer.dims.axis_labels = ["LOD level", "y", "x"]
    except Exception:
        pass

    # Overlay text showing LOD info
    def _update_overlay(event=None) -> None:
        t = int(viewer.dims.current_step[0])
        if t < n_lods:
            n_splats_at_level = result.additive_prefix(t).n_splats
            psnr_val = psnrs[t]
            viewer.text_overlay.visible = True
            viewer.text_overlay.text = (
                f"LOD 0..{t}  |  {n_splats_at_level:,} splats  |  "
                f"PSNR = {psnr_val:.2f} dB"
            )

    _update_overlay()
    viewer.dims.events.current_step.connect(_update_overlay)

    aprint("Ready. Use the LOD slider to see coarse-to-fine reconstruction.")
    napari.run()
else:
    aprint("\nDemo completed successfully (napari visualization disabled)")
