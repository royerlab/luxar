#!/usr/bin/env python3
"""
3D DAPI Microscopy — Progressive Gaussian Splatting Demo

Demonstrates progressive fitting on real 3D DAPI-stained nuclear microscopy
data from the Image Data Resource (IDR).  Each pass fits splats to the
residual of the previous approximation, building a multi-LOD representation
from coarse to fine.

**Features:**
- **GPU-accelerated NLM denoising** with Noise2Self auto-calibration (before fitting)
- Progressive residual fitting with automatic PSNR-based stopping
- Real OME-ZARR microscopy data from IDR (with synthetic fallback)
- Multi-LOD output: scrub through LOD levels to see coarse-to-fine
- Per-pass quality tracking and compression analysis
- GPU acceleration (CUDA / Metal / CPU fallback)

**Data source:** IDR (Image Data Resource)
- URL: https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr
- Type: OME-ZARR 5D (T × C × Z × Y × X), DAPI channel
- Processing: Downscaled to 128³ voxels

**Visualization (napari):**
- DAPI raw (noisy) volume and NLM-denoised volume (before/after comparison)
- NLM removed noise (absolute difference)
- LOD-level reconstruction stack: scrub from coarse to full detail
- Residual at each LOD level
- Text overlay with splat count and PSNR per LOD

**Controls:**
- Top slider: LOD level (coarsest → finest)
- Mouse drag: Rotate 3D view
- Mouse wheel: Zoom
- Toggle layers to compare input vs reconstruction
"""

import sys

import numpy as np
import zarr
from arbol import Arbol, aprint, asection

from luxar.gsplats.demos._demo_common import psnr
from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
from luxar.gsplats.lod import make_additive_lod
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.utils.trils import tril_size

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("Running all computations without napari visualization...")

# ======= Demo knobs =======
MAX_SPLATS = 6000  # Total splat budget
MAX_SPLATS_PER_PASS = 1000  # Max splats per pass (actual may be lower after culling)
ITERS_PER_PASS = 3000  # Optimization iterations per pass
PSNR_PATIENCE = 0.01  # Stop if ΔPSNR < 0.01 dB between passes
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps"
TRUNCATE_SIG = 3.0  # Rendering support truncation
ZARR_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
DAPI_CHANNEL = 1  # DAPI is typically channel 1 (0-indexed)
TARGET_SIZE = None  # Downscale to this cube size
TIME_POINT = 0  # First time point
# NLM denoising parameters
NLM_PATCH_SIZE = 3
NLM_PATCH_DISTANCE = 5
# Additive-LOD ladder size
N_LODS = 4
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

        # Extract DAPI channel from 5D (T,C,Z,Y,X) or 4D (C,Z,Y,X)
        if len(full_shape) == 5:
            ch = min(DAPI_CHANNEL, full_shape[1] - 1)
            V = np.array(data[TIME_POINT, ch, :, :, :], dtype=np.float32)
        elif len(full_shape) == 4:
            ch = min(DAPI_CHANNEL, full_shape[0] - 1)
            V = np.array(data[ch, :, :, :], dtype=np.float32)
        elif len(full_shape) == 3:
            V = np.array(data[:, :, :], dtype=np.float32)
        else:
            raise ValueError(f"Unexpected shape: {full_shape}")

        # Downscale
        if TARGET_SIZE:
            from scipy.ndimage import zoom

            factors = [TARGET_SIZE / s for s in V.shape]
            V = zoom(V, factors, order=1)
            aprint(f"Downscaled to: {V.shape}")

        # Normalize to [0, 100]
        vmin, vmax = V.min(), V.max()
        if vmax > vmin:
            V = ((V - vmin) / (vmax - vmin)) * 100.0
        else:
            V = np.ones_like(V) * 50.0

        return V.astype(np.float32)

    except Exception as e:
        aprint(f"Remote load failed: {e}")
        aprint("Falling back to synthetic nucleus-like data")

        fallback_size = TARGET_SIZE if TARGET_SIZE is not None else 128
        shape = (fallback_size, fallback_size, fallback_size)
        V = np.zeros(shape, dtype=np.float32)
        rng = np.random.RandomState(42)
        for _ in range(10):
            center = [rng.uniform(10, s - 10) for s in shape]
            sigma = rng.uniform(4.0, 8.0)
            amp = rng.uniform(60.0, 100.0)
            grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
            dist_sq = sum((g - c) ** 2 for g, c in zip(grids, center))
            V += amp * np.exp(-dist_sq / (2 * sigma**2))
        return np.clip(V, 0, 100).astype(np.float32)


with asection("3D DAPI Progressive Gaussian Splatting Demo"):
    aprint("Progressive fitting on real DAPI-stained nuclear microscopy data")

    # --- Load data ---
    with asection("Loading DAPI data"):
        V = _load_dapi_volume()
        aprint(f"Volume: {V.shape}, range: [{V.min():.2f}, {V.max():.2f}]")

    # --- NLM Denoising ---
    with asection("NLM Denoising (GPU-accelerated)"):
        import torch

        from luxar.gsplats.preprocessing import calibrate_nlm_h, denoise_nlm
        from luxar.gsplats.utils.device import resolve_torch_device

        denoise_device = str(resolve_torch_device())
        aprint(f"Denoising device: {denoise_device}")

        vol_tensor = torch.from_numpy(V)

        with asection("Calibrating NLM h (Noise2Self / J-invariant)"):
            h = calibrate_nlm_h(
                vol_tensor,
                h_range=(1.5, 2),
                patch_size=NLM_PATCH_SIZE,
                search_distance=NLM_PATCH_DISTANCE,
                use_2d_slice=False,
                device=denoise_device,
            )
            aprint(f"Calibrated h = {h:.6f}")

        with asection("Applying Non-Local Means denoising"):
            V_raw = V.copy()  # keep original for napari comparison
            V_denoised = denoise_nlm(
                vol_tensor.to(denoise_device),
                h=h,
                patch_size=NLM_PATCH_SIZE,
                search_distance=NLM_PATCH_DISTANCE,
            )
            if isinstance(V_denoised, torch.Tensor):
                V_denoised = V_denoised.cpu().numpy()
            V = V_denoised.astype(np.float32)
            aprint(f"Denoised volume range: [{V.min():.2f}, {V.max():.2f}]")

    # --- Show denoising comparison in napari (blocking) ---
    if not NO_NAPARI:
        import napari

        aprint("🔬 Launching napari to compare raw vs denoised...")
        aprint("   Close the napari window to continue with progressive fitting.")
        denoise_viewer = napari.Viewer(
            title="NLM Denoising — close to continue fitting", ndisplay=3
        )

        denoise_viewer.add_image(
            V_raw,
            name="DAPI (raw / noisy)",
            colormap="gray",
            contrast_limits=[0, float(V_raw.max())],
            rendering="mip",
        )
        denoise_viewer.add_image(
            V,
            name="DAPI (NLM denoised)",
            colormap="gray",
            contrast_limits=[0, float(V.max())],
            rendering="mip",
        )
        denoise_diff = np.abs(V_raw - V)
        denoise_viewer.add_image(
            denoise_diff,
            name="NLM removed noise",
            colormap="gray",
            contrast_limits=[0, max(1e-12, float(denoise_diff.max()))],
            rendering="mip",
            visible=False,
        )

        denoise_viewer.camera.angles = (45, 45, 45)
        denoise_viewer.camera.zoom = 2.0

        napari.run()  # blocks until user closes the window
        aprint("Napari closed — continuing with progressive fitting...")

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
            max_eccentricity=6.0,
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
        floats_per_splat = d + tril_size(d) + 1  # centers + cholesky + amplitude
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

# --- Full results napari viewer ---
if not NO_NAPARI:
    import napari

    aprint("🔬 Launching napari viewer with full results...")
    viewer = napari.Viewer(title="3D DAPI Progressive Gaussian Splatting", ndisplay=3)

    # Denoising layers
    viewer.add_image(
        V_raw,
        name="DAPI (raw / noisy)",
        colormap="gray",
        contrast_limits=[0, float(V_raw.max())],
        rendering="mip",
    )
    viewer.add_image(
        V,
        name="DAPI (NLM denoised)",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
        rendering="mip",
    )

    # Reconstruction layers
    viewer.add_image(
        stack_recon,
        name="Reconstruction (LOD levels)",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
        rendering="mip",
    )
    viewer.add_image(
        np.abs(stack_resid),
        name="Absolute residual (LOD levels)",
        colormap="gray",
        contrast_limits=[0, max(1e-12, float(np.abs(stack_resid).max()))],
        rendering="mip",
    )

    try:
        viewer.dims.axis_labels = ["LOD level", "z", "y", "x"]
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

    viewer.camera.angles = (45, 45, 45)
    viewer.camera.zoom = 2.0

    aprint("")
    aprint("Controls:")
    aprint("  LOD slider: scrub from coarse (LOD 0) to full detail")
    aprint("  Mouse drag: rotate 3D view")
    aprint("  Toggle layers to compare input vs reconstruction")
    aprint("")
    aprint("Layer guide:")
    aprint("  'DAPI (raw / noisy)' = original volume before denoising")
    aprint("  'DAPI (NLM denoised)' = after GPU-accelerated NLM denoising")
    aprint("  'Reconstruction (LOD levels)' = progressive reconstruction (slider)")
    aprint("  'Absolute residual (LOD levels)' = residual at each LOD")

    napari.run()
else:
    aprint("\nDemo completed successfully (napari visualization disabled)")
