#!/usr/bin/env python3
"""
3D C. elegans Confocal — Single Timepoint Fitting Inspection

**What this demo demonstrates:**
- 3D Gaussian splatting on a single timepoint from a C. elegans embryo confocal dataset
- Anisotropic voxel spacing (5:1 Z-anisotropy: 0.75 µm Z vs 0.15 µm XY)
- Fitting quality inspection via napari: input, reconstruction, residual
- Compression sweep to visualise quality vs splat count trade-off

**Data source:** Zenodo record 6460303
- Hirsch, P. et al. (2022). 3D+time nuclei tracking dataset of confocal
  fluorescence microscopy time series of C. elegans embryos.
- DOI: 10.5281/zenodo.6460303
- Sample: mskcc_confocal_s1
- Resolution: 0.75 × 0.15 × 0.15 µm (Z × Y × X)
- Volume: 41 × 512 × 512 per timepoint

**Prerequisites:** Run the main C. elegans demo first to download and extract data:
    python packages/luxar/src/luxar/demos/demo_gsplats_4d_celegans_tracking.py --no-serve

**Visualization:** 3D napari viewer with MIP rendering
- Input volume
- Final reconstruction (with PSNR/compression stats)
- Final residual
- Compression sweep with slider

**Usage:**
    python demo_3d_celegans_confocal.py [options]

Options:
    --no-napari:     Run without napari (compute only)
    --timepoint=N:   Which timepoint to use (default: 0)
    --seeds=N:       Number of seed splats (default: 3000)
    --iters=N:       Fitting iterations (default: 4000)
"""

import sys
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.utils.trils import tril_size, unpack_tril

# ======= Configuration =======
NO_NAPARI = "--no-napari" in sys.argv
TIMEPOINT = 0
NUM_SPLATS = 5000
N_ITERS = 6000
N_FRAMES = 30  # compression sweep steps
DEVICE = None  # auto-detect
USE_METAL = True
USE_CUDA = True

# C. elegans imaging parameters
VOXEL_SIZE_ZYX = (0.75, 0.15, 0.15)  # µm
SAMPLE_INDEX = 1
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_celegans"

for _arg in sys.argv:
    if _arg.startswith("--timepoint="):
        TIMEPOINT = int(_arg.split("=")[1])
    elif _arg.startswith("--seeds="):
        NUM_SPLATS = int(_arg.split("=")[1])
    elif _arg.startswith("--iters="):
        N_ITERS = int(_arg.split("=")[1])

Arbol.max_depth = 5

if NO_NAPARI:
    aprint("🔬 C. elegans Confocal — Single Timepoint Fitting (napari disabled)")

# ==========================


with asection("3D C. elegans Confocal — Gaussian Splatting Demo"):
    aprint("🪱 Real confocal microscopy: C. elegans embryo nuclei")
    aprint(f"Timepoint: {TIMEPOINT}  |  Seeds: {NUM_SPLATS}  |  Iters: {N_ITERS}")
    aprint(f"Voxel size (Z,Y,X): {VOXEL_SIZE_ZYX} µm")

    # --- Load single timepoint ---
    with asection("Loading C. elegans volume"):
        sample_name = f"mskcc_confocal_s{SAMPLE_INDEX}"
        extract_dir = CACHE_DIR / "extracted"

        # Find TIFF files
        sample_dir = extract_dir / "mskcc-confocal" / sample_name
        if not sample_dir.is_dir():
            # Try recursive search
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

        # Collect TIFF files
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

        # Normalise to [0, 100] (matching DAPI demo convention)
        vmin, vmax = V.min(), V.max()
        if vmax > vmin:
            V = ((V - vmin) / (vmax - vmin)) * 100.0
        else:
            V = np.zeros_like(V)

        aprint(f"Volume shape: {V.shape} (Z, Y, X)")
        aprint(f"Intensity range: [{V.min():.1f}, {V.max():.1f}]")
        aprint(f"Non-zero voxels: {np.count_nonzero(V):,} / {V.size:,}")

    # --- Fit Gaussian splats ---
    with asection(f"Fitting 3D Gaussian splats ({N_ITERS} iterations)"):
        aprint(f"Voxel size: {VOXEL_SIZE_ZYX} µm (5:1 Z-anisotropy)")

        result = fit_gaussian_splats(
            V,
            lr=0.01,
            seeds=NUM_SPLATS,
            n_iters=N_ITERS,
            device=DEVICE,
            use_metal=USE_METAL,
            use_cuda=USE_CUDA,
            verbose=True,
            enable_dynamic_ops=True,
            voxel_size=VOXEL_SIZE_ZYX,
            output_space="voxel",  # Keep in voxel coords for napari overlay
            napari_movie=(not NO_NAPARI),
            movie_every=max(1, N_ITERS // 10),
        )

        aprint(f"Fitted {len(result.amplitudes)} splats")

        if len(result.amplitudes) == 0:
            raise RuntimeError("No splats fitted.")

    # --- Reconstruction and quality ---
    with asection("Final reconstruction vs original"):
        V_final = render_gaussians_numpy(V.shape, result, truncate=3.0)
        residual_final = V - V_final

        mse = float(np.mean(residual_final**2))
        psnr = 10.0 * np.log10(float(V.max()) ** 2 / (mse + 1e-12))
        rel_l2 = float(np.linalg.norm(residual_final) / (np.linalg.norm(V) + 1e-12))
        max_err = float(np.abs(residual_final).max())

        n_final = len(result.amplitudes)
        d = 3
        floats_per_splat = d + tril_size(d) + 1 + 1  # centers + chol + amp + sharpness
        model_bits = n_final * floats_per_splat * 32
        image_bits = V.size * 32
        fold = image_bits / max(model_bits, 1)

        aprint(f"Splats: {n_final}")
        aprint(f"Compression: {fold:.1f}x")
        aprint(f"PSNR: {psnr:.2f} dB")
        aprint(f"Relative L2: {rel_l2:.4f}")
        aprint(f"Max absolute error: {max_err:.2f}")

    # --- Compression sweep ---
    L_full = unpack_tril(result.cholesky_factors, d)
    diag_prod = np.prod(np.stack([L_full[:, i, i] for i in range(d)], axis=1), axis=1)
    energy = (result.amplitudes**2) * (np.sqrt(np.pi) ** d) * diag_prod
    order = np.argsort(-energy)

    N = len(result.amplitudes)
    keep_counts = np.unique(
        np.linspace(1, N, num=min(N_FRAMES, N), endpoint=True).astype(int)
    )

    stack_recon = np.zeros((len(keep_counts),) + V.shape, dtype=np.float32)
    stack_resid = np.zeros_like(stack_recon)

    aprint("Computing compression sweep...")
    with asection("Compression sweep"):
        for i, K in enumerate(keep_counts):
            idx = order[:K]
            sub = GSplatData(
                centers=result.centers[idx],
                amplitudes=result.amplitudes[idx],
                cholesky_factors=result.cholesky_factors[idx],
                sharpnesses=result.sharpnesses[idx],
                stats={},
            )
            Vk = render_gaussians_numpy(V.shape, sub, truncate=3.0)
            stack_recon[i] = Vk
            stack_resid[i] = V - Vk

            if (i + 1) % 10 == 0 or i == len(keep_counts) - 1:
                err = np.linalg.norm(V - Vk) / (np.linalg.norm(V) + 1e-12)
                bits = int(K) * floats_per_splat * 32
                aprint(f"  K={K:4d}  {image_bits / max(bits, 1):.0f}x  relL2={err:.4f}")

if not NO_NAPARI:
    import napari

    aprint("Launching napari viewer...")
    viewer = napari.Viewer(
        title=f"C. elegans t={TIMEPOINT} — GSplats Fitting", ndisplay=3
    )

    # Scale for anisotropic voxels (napari uses ZYX order)
    scale = list(VOXEL_SIZE_ZYX)

    viewer.add_image(
        V,
        name="input volume",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
        rendering="mip",
        scale=scale,
    )
    viewer.add_image(
        V_final,
        name=f"reconstruction ({n_final} splats, PSNR {psnr:.1f} dB)",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
        rendering="mip",
        scale=scale,
    )
    viewer.add_image(
        np.abs(residual_final),
        name="residual (absolute)",
        colormap="gray",
        contrast_limits=[0, max(1e-12, max_err)],
        rendering="mip",
        scale=scale,
    )
    viewer.add_image(
        stack_recon,
        name="compression sweep",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
        rendering="mip",
        scale=[1.0] + scale,
    )
    viewer.add_image(
        np.abs(stack_resid),
        name="compression residual",
        colormap="gray",
        contrast_limits=[0, max(1e-12, float(np.abs(stack_resid).max()))],
        rendering="mip",
        scale=[1.0] + scale,
    )

    viewer.camera.angles = (45, 45, 45)
    viewer.camera.zoom = 4.0

    aprint(f"\nSplats: {n_final}  |  PSNR: {psnr:.1f} dB  |  {fold:.0f}x compression")
    aprint("Use the top slider to explore compression levels")
    aprint("Toggle layers to compare input vs reconstruction")

    napari.run()
else:
    aprint("\nDemo completed (napari disabled)")
