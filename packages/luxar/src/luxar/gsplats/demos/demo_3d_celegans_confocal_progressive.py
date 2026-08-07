#!/usr/bin/env python3
"""
3D C. elegans Confocal — Progressive Gaussian Splatting Demo

Progressive (multi-pass) fitting on a single timepoint from the C. elegans
embryo confocal dataset.  Each pass fits splats to the residual of the
previous approximation, building a multi-LOD representation from coarse
nuclei outlines to fine sub-nuclear detail.

Features:
- Progressive residual fitting with automatic PSNR-based stopping
- Anisotropic voxel spacing (5:1 Z-anisotropy: 0.75 um Z vs 0.15 um XY)
- Per-LOD 3D visualization with MIP rendering
- Compression analysis at each LOD level

Data source: Zenodo record 6460303
- Hirsch, P. et al. (2022). 3D+time nuclei tracking dataset of confocal
  fluorescence microscopy time series of C. elegans embryos.
- DOI: 10.5281/zenodo.6460303
- Sample: mskcc_confocal_s1
- Resolution: 0.75 x 0.15 x 0.15 um (Z x Y x X)
- Volume: 41 x 512 x 512 per timepoint

Prerequisites: Run the main C. elegans demo first to download and extract data:
    python packages/luxar/src/luxar/demos/demo_gsplats_4d_celegans_tracking.py --no-serve

Usage:
    python demo_3d_celegans_confocal_progressive.py [options]

Options:
    --no-napari:     Run without napari (compute only)
    --timepoint=N:   Which timepoint to use (default: 0)
"""

import sys
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar.gsplats.demos._demo_common import psnr
from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
from luxar.gsplats.lod import make_additive_lod
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.utils.trils import tril_size

# ======= Configuration =======
NO_NAPARI = "--no-napari" in sys.argv
TIMEPOINT = 0

# Progressive fitting parameters (5000 seeds in single-shot version)
MAX_SPLATS = 6000  # Total splat budget
MAX_SPLATS_PER_PASS = 1000  # Max splats per pass
ITERS_PER_PASS = 3000  # Optimization iterations per pass
PSNR_PATIENCE = 0.2  # Stop if DPSNR < 0.2 dB between passes
TRUNCATE_SIG = 3.0  # Rendering support truncation
DEVICE = None  # auto-detect
N_LODS = 4  # Additive-LOD ladder size for the post-fit ordering

# C. elegans imaging parameters
VOXEL_SIZE_ZYX = (0.75, 0.15, 0.15)  # um
SAMPLE_INDEX = 1
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_celegans"

for _arg in sys.argv:
    if _arg.startswith("--timepoint="):
        TIMEPOINT = int(_arg.split("=")[1])

Arbol.max_depth = 5

if NO_NAPARI:
    aprint("C. elegans Confocal Progressive (napari disabled)")

# ==========================


with asection("3D C. elegans Confocal — Progressive Gaussian Splatting"):
    aprint("Progressive fitting on real confocal microscopy data")
    aprint(f"Timepoint: {TIMEPOINT}  |  Voxel size (Z,Y,X): {VOXEL_SIZE_ZYX} um")

    # --- Load single timepoint ---
    with asection("Loading C. elegans volume"):
        sample_name = f"mskcc_confocal_s{SAMPLE_INDEX}"
        extract_dir = CACHE_DIR / "extracted"

        # Find TIFF files
        sample_dir = extract_dir / "mskcc-confocal" / sample_name
        if not sample_dir.is_dir():
            candidates = list(extract_dir.rglob(sample_name))
            candidates = [c for c in candidates if c.is_dir()]
            if candidates:
                sample_dir = candidates[0]
            else:
                raise FileNotFoundError(
                    "Sample directory not found. Run the main C. elegans demo first:\n"
                    "  hatch run python packages/luxar/src/luxar/demos/"
                    "demo_gsplats_4d_celegans_tracking.py --no-serve"
                )

        tiff_files = []
        for d in [sample_dir] + [s for s in sample_dir.iterdir() if s.is_dir()]:
            tiff_files.extend(
                f
                for f in d.iterdir()
                if f.is_file() and f.suffix.lower() in (".tif", ".tiff")
            )
        tiff_files = sorted(set(tiff_files))

        if TIMEPOINT >= len(tiff_files):
            raise ValueError(
                f"Timepoint {TIMEPOINT} requested but only {len(tiff_files)} available"
            )

        aprint(f"Found {len(tiff_files)} timepoints")
        aprint(f"Loading timepoint {TIMEPOINT}: {tiff_files[TIMEPOINT].name}")

        try:
            import tifffile
        except ImportError:
            raise ImportError("tifffile required: pip install tifffile")

        V = tifffile.imread(str(tiff_files[TIMEPOINT])).astype(np.float32)

        # Handle extra dimensions
        if V.ndim == 4 and V.shape[0] <= 4:
            V = V[0]
        elif V.ndim == 4 and V.shape[-1] <= 4:
            V = V[..., 0]

        if V.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {V.ndim}D: {V.shape}")

        # Normalise to [0, 100]
        vmin, vmax = V.min(), V.max()
        if vmax > vmin:
            V = ((V - vmin) / (vmax - vmin)) * 100.0
        else:
            V = np.zeros_like(V)

        aprint(f"Volume shape: {V.shape} (Z, Y, X)")
        aprint(f"Intensity range: [{V.min():.1f}, {V.max():.1f}]")
        aprint(f"Non-zero voxels: {np.count_nonzero(V):,} / {V.size:,}")

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
            voxel_size=VOXEL_SIZE_ZYX,
            output_space="voxel",  # Keep in voxel coords for napari overlay
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

        # Compression stats
        d = 3
        floats_per_splat = d + tril_size(d) + 1
        model_bits = result.n_splats * floats_per_splat * 32
        image_bits = V.size * 32
        fold = image_bits / max(model_bits, 1)
        aprint(f"Compression: {fold:.1f}x ({100 * (1 - model_bits / image_bits):.1f}%)")
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

    aprint("Launching napari viewer...")
    viewer = napari.Viewer(
        title=f"C. elegans t={TIMEPOINT} — Progressive GSplats", ndisplay=3
    )

    scale = list(VOXEL_SIZE_ZYX)

    viewer.add_image(
        V,
        name="Input volume",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
        rendering="mip",
        scale=scale,
    )
    viewer.add_image(
        stack_recon,
        name="Reconstruction (LOD levels)",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
        rendering="mip",
        scale=[1.0] + scale,
    )
    viewer.add_image(
        np.abs(stack_resid),
        name="Absolute residual (LOD levels)",
        colormap="gray",
        contrast_limits=[0, max(1e-12, float(np.abs(stack_resid).max()))],
        rendering="mip",
        scale=[1.0] + scale,
    )

    try:
        viewer.dims.axis_labels = ["LOD level", "z", "y", "x"]
    except Exception:
        pass

    viewer.camera.angles = (45, 45, 45)
    viewer.camera.zoom = 4.0

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

    aprint("")
    aprint("Controls:")
    aprint("  LOD slider: scrub from coarse (LOD 0) to full detail")
    aprint("  Mouse drag: rotate 3D view")
    aprint("  Toggle layers to compare input vs reconstruction")

    napari.run()
else:
    aprint("\nDemo completed (napari disabled)")
