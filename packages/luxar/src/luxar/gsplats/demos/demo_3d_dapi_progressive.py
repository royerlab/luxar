#!/usr/bin/env python3
"""
3D DAPI Microscopy — Progressive Gaussian Splatting Demo

Demonstrates progressive fitting on real 3D DAPI-stained nuclear microscopy
data from the Image Data Resource (IDR).  Each pass fits splats to the
residual of the previous approximation, building a multi-LOD representation
from coarse to fine.

**Features:**
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
- Input DAPI volume (MIP rendering)
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

from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatLOD
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.utils.trils import tril_size

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("Running all computations without napari visualization...")

# ======= Demo knobs =======
DEBUG = False  # Set to True to collect per-pass diagnostics (overshoot analysis)
MAX_SPLATS = 10000  # Total splat budget
MAX_SPLATS_PER_PASS = 1000  # Max splats per pass (actual may be lower after culling)
ITERS_PER_PASS = 3000  # Optimization iterations per pass
PSNR_PATIENCE = 0.01  # Stop if ΔPSNR < 0.01 dB between passes
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps"
TRUNCATE_SIG = 3.0  # Rendering support truncation
ZARR_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
DAPI_CHANNEL = 1  # DAPI is typically channel 1 (0-indexed)
TARGET_SIZE = None  # Downscale to this cube size
TIME_POINT = 0  # First time point
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
        V_raw = _load_dapi_volume()
        aprint(f"Volume: {V_raw.shape}, range: [{V_raw.min():.2f}, {V_raw.max():.2f}]")

    # --- NLM denoising ---
    with asection("NLM denoising (CUDA accelerated)"):
        import torch

        from luxar.gsplats.preprocessing import calibrate_nlm_h, denoise_nlm

        denoise_device = "cuda" if torch.cuda.is_available() else "cpu"

        # NLM expects [0, 1] range for proper h calibration
        v_max = float(V_raw.max())
        v_min = float(V_raw.min())
        v_range = max(v_max - v_min, 1e-6)
        V_norm = torch.from_numpy(
            ((V_raw - v_min) / v_range).astype(np.float32)
        ).to(denoise_device)

        # Auto-calibrate denoising strength via Noise2Self
        with asection("Calibrating h (Noise2Self)"):
            h_opt = calibrate_nlm_h(V_norm, device=denoise_device)
            aprint(f"Calibrated h = {h_opt:.4f}")
            # Ensure minimum denoising strength for visible effect
            h_min = 0.03
            if h_opt < h_min:
                aprint(f"h too small ({h_opt:.4f}), bumping to {h_min}")
                h_opt = h_min
            aprint(f"Using h = {h_opt:.4f}")

        # Denoise
        with asection("Denoising"):
            V_denoised_norm = denoise_nlm(V_norm, h=h_opt, search_distance=9, device=denoise_device)

            # Sanity check: is the output actually different?
            diff_check = (V_norm.cpu() - V_denoised_norm.cpu()).abs()
            aprint(
                f"Denoising diff: max={diff_check.max().item():.6f}, "
                f"mean={diff_check.mean().item():.6f}"
            )
            if diff_check.max().item() < 1e-6:
                aprint("WARNING: denoising had no effect! Check backend.")

            # Rescale back to original intensity range
            V = V_denoised_norm.cpu().numpy() * v_range + v_min
            aprint(f"Denoised: range: [{V.min():.2f}, {V.max():.2f}]")

        # Show raw vs denoised in napari before fitting
        if not NO_NAPARI:
            import napari

            viewer = napari.Viewer(
                title="NLM Denoising Result (close to continue)", ndisplay=3
            )
            viewer.add_image(
                V_raw,
                name="Raw DAPI",
                colormap="gray",
                contrast_limits=[0, float(V_raw.max())],
                rendering="mip",
            )
            viewer.add_image(
                V,
                name="Denoised DAPI (NLM)",
                colormap="gray",
                contrast_limits=[0, float(V.max())],
                rendering="mip",
            )
            # Difference (amplified) to visualize what was removed
            diff = V_raw - V
            diff_absmax = max(1e-12, float(np.abs(diff).max()))
            viewer.add_image(
                diff,
                name=f"Removed noise (h={h_opt:.4f})",
                colormap="PiYG",
                contrast_limits=[-diff_absmax, diff_absmax],
                rendering="mip",
            )
            aprint(f"Noise removed: max={diff_absmax:.3f}, std={float(diff.std()):.3f}")
            aprint("Close napari window to continue to fitting...")
            napari.run()

    # --- Debug: per-pass diagnostic collection ---
    debug_passes: list[dict] = []  # collected by callback

    def _debug_callback(
        pass_idx: int, lod_data: GSplatLOD, cumulative_psnr: float
    ) -> None:
        """Collect per-pass diagnostics for debug visualization."""
        if not DEBUG:
            return

        try:
            _debug_callback_inner(pass_idx, lod_data, cumulative_psnr)
        except Exception as exc:
            aprint(f"  DEBUG callback error (pass {pass_idx}): {exc}")

    def _debug_callback_inner(
        pass_idx: int, lod_data: GSplatLOD, cumulative_psnr: float
    ) -> None:
        from luxar.gsplats.gsplat_data import GSplatData

        # Accumulate all LODs seen so far (including this one)
        # We rebuild from debug_passes + current LOD
        all_lods = [p["lod"] for p in debug_passes] + [lod_data]
        accumulated = GSplatData.from_lods(all_lods)

        # Render accumulated reconstruction
        recon = render_gaussians_numpy(V.shape, accumulated, truncate=TRUNCATE_SIG)

        # Signed residual: positive = undershoot, negative = OVERSHOOT
        signed_residual = V - recon

        # Per-pass target (what this pass was asked to fit):
        # Pass 0 fitted the original V; later passes fitted clamped residual
        if pass_idx == 0:
            pass_target = V.copy()
        else:
            prev_lods = [p["lod"] for p in debug_passes]
            prev_accumulated = GSplatData.from_lods(prev_lods)
            prev_recon = render_gaussians_numpy(
                V.shape, prev_accumulated, truncate=TRUNCATE_SIG
            )
            pass_target = np.clip(V - prev_recon, 0, None)

        # Per-pass reconstruction (just this pass's splats)
        pass_only = GSplatData(
            centers=lod_data.centers,
            amplitudes=lod_data.amplitudes,
            cholesky_factors=lod_data.cholesky_factors,
        )
        pass_recon = render_gaussians_numpy(V.shape, pass_only, truncate=TRUNCATE_SIG)

        # Stats
        overshoot_frac = float((signed_residual < 0).sum()) / max(signed_residual.size, 1)
        overshoot_max = float(np.abs(np.minimum(signed_residual, 0)).max())
        undershoot_max = float(np.maximum(signed_residual, 0).max())

        aprint(
            f"  DEBUG pass {pass_idx}: "
            f"splats={lod_data.n_splats}, PSNR={cumulative_psnr:.2f} dB, "
            f"overshoot: {100 * overshoot_frac:.1f}% of voxels "
            f"(max={overshoot_max:.3f}), "
            f"undershoot max={undershoot_max:.3f}"
        )

        debug_passes.append(
            {
                "lod": lod_data,
                "pass_target": pass_target,
                "pass_recon": pass_recon,
                "cumul_recon": recon,
                "signed_residual": signed_residual,
                "psnr": cumulative_psnr,
                "overshoot_frac": overshoot_frac,
                "overshoot_max": overshoot_max,
            }
        )

        # Open napari viewer for this pass (close to continue to next pass)
        if not NO_NAPARI:
            import napari

            viewer = napari.Viewer(
                title=f"DEBUG Pass {pass_idx} | "
                f"{lod_data.n_splats} splats | "
                f"PSNR={cumulative_psnr:.2f} dB | "
                f"overshoot={100 * overshoot_frac:.1f}%",
                ndisplay=3,
            )
            sr_absmax = max(1e-12, overshoot_max, undershoot_max)

            viewer.add_image(
                V, name="Original", colormap="gray",
                contrast_limits=[0, float(V.max())], rendering="mip",
            )
            viewer.add_image(
                pass_target, name=f"Pass {pass_idx} target (clamped residual)",
                colormap="magma",
                contrast_limits=[0, float(V.max())], rendering="mip",
            )
            viewer.add_image(
                pass_recon, name=f"Pass {pass_idx} reconstruction (this pass only)",
                colormap="magma",
                contrast_limits=[0, float(V.max())], rendering="mip",
                visible=False,
            )
            viewer.add_image(
                recon, name=f"Cumulative reconstruction (passes 0..{pass_idx})",
                colormap="gray",
                contrast_limits=[0, float(V.max())], rendering="mip",
            )
            viewer.add_image(
                signed_residual,
                name="Signed residual (neg=OVERSHOOT)",
                colormap="PiYG",
                contrast_limits=[-sr_absmax, sr_absmax], rendering="mip",
            )
            overshoot_vol = np.clip(-signed_residual, 0, None)
            viewer.add_image(
                overshoot_vol, name="Overshoot only",
                colormap="hot",
                contrast_limits=[0, max(1e-12, float(overshoot_vol.max()))],
                rendering="mip",
            )

            viewer.camera.angles = (45, 45, 45)
            viewer.camera.zoom = 2.0

            aprint(f"  DEBUG: napari open for pass {pass_idx}. Close window to continue...")
            napari.run()  # Blocks until viewer is closed

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
            on_pass_complete=_debug_callback if DEBUG else None,
        )

    # --- Summary ---
    with asection("Results"):
        aprint(f"Total splats: {result.n_splats:,}")
        aprint(f"LOD levels: {result.n_lods}")
        aprint(f"Final PSNR: {result.stats.get('psnr_db', 0):.2f} dB")
        aprint(f"Stop reason: {result.stats.get('stop_reason', '?')}")
        aprint(f"Total time: {result.stats.get('time_seconds', 0):.1f}s")

        # Compression stats
        d = 3
        floats_per_splat = d + tril_size(d) + 1  # centers + cholesky + amplitude
        model_bits = result.n_splats * floats_per_splat * 32
        image_bits = V.size * 32
        fold = image_bits / max(model_bits, 1)
        aprint(f"Compression: {fold:.1f}x ({100 * (1 - model_bits / image_bits):.1f}%)")
        aprint("")

        psnrs = result.lod_psnrs()
        for i in range(result.n_lods):
            lod = result.at_lod(i)
            cumul_splats = result.up_to_lod(i).n_splats
            aprint(
                f"  LOD {i}: +{lod.n_splats:,} splats "
                f"(total: {cumul_splats:,}), "
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
            aprint(
                f"LOD 0..{level}: {data_at_level.n_splats:,} splats, "
                f"PSNR = {psnrs[level]:.2f} dB"
            )

# --- Napari visualization ---
if not NO_NAPARI:
    import napari

    aprint("Launching napari viewer...")
    viewer = napari.Viewer(
        title="3D DAPI Progressive Gaussian Splatting", ndisplay=3
    )
    aprint("Napari viewer created, adding layers...")

    viewer.add_image(
        V,
        name="DAPI (input)",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
        rendering="mip",
    )
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
            n_at_level = result.up_to_lod(t).n_splats
            psnr_val = psnrs[t]
            viewer.text_overlay.visible = True
            viewer.text_overlay.text = (
                f"LOD 0..{t}  |  {n_at_level:,} splats  |  "
                f"PSNR = {psnr_val:.2f} dB"
            )

    _update_overlay()
    viewer.dims.events.current_step.connect(_update_overlay)

    viewer.camera.angles = (45, 45, 45)
    viewer.camera.zoom = 2.0

    # --- Debug layers: per-pass diagnostics ---
    if DEBUG and debug_passes:
        n_debug = len(debug_passes)
        last = debug_passes[-1]

        # Final signed residual (3D volume — green=undershoot, pink=overshoot)
        sr = last["signed_residual"]
        sr_absmax = max(1e-12, float(np.abs(sr).max()))
        viewer.add_image(
            sr,
            name="DEBUG: final signed residual (neg=OVERSHOOT)",
            colormap="PiYG",
            contrast_limits=[-sr_absmax, sr_absmax],
            rendering="mip",
        )

        # Overshoot-only volume (where reconstruction > original)
        overshoot_vol = np.clip(-sr, 0, None)
        viewer.add_image(
            overshoot_vol,
            name="DEBUG: overshoot only",
            colormap="hot",
            contrast_limits=[0, max(1e-12, float(overshoot_vol.max()))],
            rendering="mip",
        )

        # Per-pass target and reconstruction as 4D stacks (slider = pass)
        target_stack = np.stack([p["pass_target"] for p in debug_passes])
        viewer.add_image(
            target_stack,
            name="DEBUG: per-pass target (residual input)",
            colormap="magma",
            contrast_limits=[0, float(V.max())],
            rendering="mip",
            visible=False,
        )

        pass_recon_stack = np.stack([p["pass_recon"] for p in debug_passes])
        viewer.add_image(
            pass_recon_stack,
            name="DEBUG: per-pass reconstruction (this pass only)",
            colormap="magma",
            contrast_limits=[0, float(V.max())],
            rendering="mip",
            visible=False,
        )

        aprint(f"\nDEBUG: {n_debug} diagnostic layers added")
        aprint("  'final signed residual': green=undershoot, pink=OVERSHOOT")
        aprint("  'overshoot only': bright = where reconstruction exceeds original")
        aprint("  'per-pass target/reconstruction': toggle visibility, use slider")
        for i, p in enumerate(debug_passes):
            aprint(
                f"  Pass {i}: overshoot={100 * p['overshoot_frac']:.1f}% "
                f"(max={p['overshoot_max']:.3f})"
            )

    aprint("")
    aprint("Controls:")
    aprint("  LOD slider: scrub from coarse (LOD 0) to full detail")
    aprint("  Mouse drag: rotate 3D view")
    aprint("  Toggle layers to compare input vs reconstruction")

    aprint("Starting napari event loop...")
    napari.run()
else:
    # Print debug summary even without napari
    if DEBUG and debug_passes:
        aprint("\nDEBUG: Per-pass overshoot analysis:")
        for i, p in enumerate(debug_passes):
            aprint(
                f"  Pass {i}: {p['lod'].n_splats} splats, "
                f"PSNR={p['psnr']:.2f} dB, "
                f"overshoot={100 * p['overshoot_frac']:.1f}% of voxels "
                f"(max={p['overshoot_max']:.3f})"
            )

    aprint("\nDemo completed successfully (napari visualization disabled)")
