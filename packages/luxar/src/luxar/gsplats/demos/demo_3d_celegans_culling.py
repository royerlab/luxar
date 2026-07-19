#!/usr/bin/env python3
"""
3D C. elegans — Contribution-Based Culling Demo

Demonstrates the contribution-based culling algorithm on pre-computed
Gaussian splats from the C. elegans confocal timelapse dataset.  Compares
the full (unculled) reconstruction against increasingly aggressive culling
levels (99%, 98%, 97%, 95%, 90% error percentile).

**What this demo shows:**
- Principled splat removal based on local reconstruction error budget
- How different culling aggressiveness levels trade off compression vs quality
- Per-timepoint compression ratios and PSNR for each culling level
- Side-by-side napari visualisation: unculled vs culled reconstructions

**Data source:** Pre-computed GSplats from Zenodo record 6460303
- C. elegans embryo confocal microscopy (mskcc_confocal_s1)
- 5 timepoints sampled across the 400-frame timelapse
- Volume: 41 x 512 x 512 per timepoint (Z x Y x X)

**Prerequisites:** Run the main C. elegans demo first to download and extract data:
    hatch run python packages/luxar/src/luxar/demos/demo_gsplats_4d_celegans_tracking.py --no-serve

**Usage:**
    hatch run python demo_3d_celegans_culling.py [options]

Options:
    --no-napari:  Run without napari (compute only, print stats table)
"""

import sys
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

# ======= Configuration =======

NO_NAPARI = "--no-napari" in sys.argv

# 5 timepoints distributed across the 400-frame timelapse
TIMEPOINTS = [0, 100, 200, 300, 399]

# Culling levels: error_percentile values (higher = more conservative = less culling)
# Ordered from least to most aggressive (slider goes left=conservative → right=aggressive)
# Higher percentile = larger error budget = more splats culled
CULL_LEVELS = [90.0, 95.0, 97.0, 98.0, 99.0, 99.5, 99.9]

# C. elegans imaging parameters
VOXEL_SIZE_ZYX = (0.75, 0.15, 0.15)  # um
SAMPLE_INDEX = 1
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_celegans"
TRUNCATE = 3.0

# Preprocessing parameters (must match the 4D demo fitting pipeline)
NLM_PATCH_SIZE = 3
NLM_SEARCH_DISTANCE = 5
CLAHE_TILE_SIZE = 16
CLAHE_CLIP_LIMIT = 2.0
from luxar.gsplats.utils.device import resolve_torch_device  # noqa: E402

DEVICE = str(resolve_torch_device())

Arbol.max_depth = 5

# ======= Helpers =======


def _physical_to_voxel(gsplats, voxel_size: tuple):  # noqa: ANN201
    """Convert GSplatData from physical coordinates back to voxel coordinates.

    The pre-computed splats were fitted with output_space='real', which
    scales centers by voxel_size and Cholesky rows by the corresponding
    voxel_size element.  This reverses that transformation so the splats
    can be rendered onto the original voxel grid.
    """
    from luxar.gsplats.gsplat_data import GSplatData

    vs = np.array(voxel_size, dtype=np.float32)
    ndim = gsplats.ndim

    centers_vox = gsplats.centers / vs

    chol = gsplats.cholesky_factors.copy()
    k = 0
    for row in range(ndim):
        for _col in range(row + 1):
            chol[:, k] /= vs[row]
            k += 1

    return GSplatData(
        centers=centers_vox,
        amplitudes=gsplats.amplitudes.copy(),
        cholesky_factors=chol,
        colors=gsplats.colors.copy() if gsplats.colors is not None else None,
        stats=dict(gsplats.stats),
    )


def _compute_compression(n_splats: int, ndim: int, volume_size: int) -> dict:
    """Compute compression statistics for a splat dataset."""
    tril_size = ndim * (ndim + 1) // 2
    floats_per_splat = ndim + tril_size + 1
    model_bits = n_splats * floats_per_splat * 32
    image_bits = volume_size * 32
    fold = image_bits / max(model_bits, 1)
    return {
        "n_splats": n_splats,
        "model_bits": model_bits,
        "image_bits": image_bits,
        "fold": fold,
        "floats_per_splat": floats_per_splat,
    }


# ======= Main =======

with asection("C. elegans — Contribution-Based Culling Demo"):
    aprint(f"Timepoints: {TIMEPOINTS}")
    aprint(f"Culling levels (error_percentile): {CULL_LEVELS}")

    # --- Load pre-computed splats ---
    from luxar.utils.demos import load_precomputed_bundle

    file_names = [f"celegans_s1_t{t:04d}.gsplats.zarr.zip" for t in TIMEPOINTS]

    gsplats_list = load_precomputed_bundle(
        "gsplats_celegans",
        "celegans_s1.gsplats.zarr.zip",
        file_names,
    )
    if gsplats_list is None:
        raise RuntimeError("Failed to load precomputed bundle")

    # Convert from physical coordinates (microns) back to voxel coordinates.
    # The pre-computed splats were fitted with output_space='real' and
    # voxel_size=(0.75, 0.15, 0.15), so centers and Cholesky factors are
    # in microns.  Rendering and culling need voxel coordinates.
    gsplats_list = [_physical_to_voxel(gs, VOXEL_SIZE_ZYX) for gs in gsplats_list]

    aprint(f"Loaded {len(gsplats_list)} timepoints (converted to voxel coordinates)")

    # --- Load and preprocess TIFF volumes ---
    # The pre-computed splats were fitted to NLM-denoised + CLAHE-enhanced
    # volumes normalised to [0,1].  We must apply the same preprocessing
    # here so the culling target matches what was actually fitted.
    with asection("Loading and preprocessing volumes"):
        try:
            import tifffile
        except ImportError:
            raise ImportError("tifffile required: pip install tifffile")

        import torch

        from luxar.gsplats.clahe import apply_clahe
        from luxar.gsplats.preprocessing import calibrate_nlm_h, denoise_nlm

        extract_dir = CACHE_DIR / "extracted"
        sample_name = f"mskcc_confocal_s{SAMPLE_INDEX}"
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
        aprint(f"Found {len(tiff_files)} TIFF timepoints")

        # NLM calibration on first timepoint (same as 4D demo)
        # Pass the full 3D volume — calibrate_nlm_h extracts a 2D middle slice
        with asection("Calibrating NLM denoiser"):
            V0 = tifffile.imread(str(tiff_files[TIMEPOINTS[0]])).astype(np.float32)
            if V0.ndim == 4 and V0.shape[0] <= 4:
                V0 = V0[0]
            elif V0.ndim == 4 and V0.shape[-1] <= 4:
                V0 = V0[..., 0]
            vmin, vmax = V0.min(), V0.max()
            if vmax > vmin:
                V0 = (V0 - vmin) / (vmax - vmin)
            nlm_h = calibrate_nlm_h(
                torch.from_numpy(V0),
                patch_size=NLM_PATCH_SIZE,
                search_distance=NLM_SEARCH_DISTANCE,
                use_2d_slice=True,
                device=DEVICE,
            )
            aprint(f"Calibrated NLM h = {nlm_h:.4f}")

        # Load, denoise, CLAHE each timepoint
        volumes = []
        for t in TIMEPOINTS:
            if t >= len(tiff_files):
                raise ValueError(
                    f"Timepoint {t} not available ({len(tiff_files)} files)"
                )

            with asection(f"Preprocessing t={t}"):
                V = tifffile.imread(str(tiff_files[t])).astype(np.float32)
                if V.ndim == 4 and V.shape[0] <= 4:
                    V = V[0]
                elif V.ndim == 4 and V.shape[-1] <= 4:
                    V = V[..., 0]

                # Normalise to [0, 1] (matching fitting pipeline)
                vmin, vmax = V.min(), V.max()
                if vmax > vmin:
                    V = (V - vmin) / (vmax - vmin)

                # NLM denoising
                V_tensor = torch.from_numpy(V)
                V_denoised = denoise_nlm(
                    V_tensor,
                    h=nlm_h,
                    patch_size=NLM_PATCH_SIZE,
                    search_distance=NLM_SEARCH_DISTANCE,
                    device=DEVICE,
                )

                # CLAHE
                V_enhanced = apply_clahe(
                    V_denoised,
                    tile_size=CLAHE_TILE_SIZE,
                    clip_limit=CLAHE_CLIP_LIMIT,
                )

                # Re-normalise to [0, 1]
                V_np = V_enhanced.cpu().numpy()
                rmin, rmax = V_np.min(), V_np.max()
                if rmax > rmin:
                    V_np = (V_np - rmin) / (rmax - rmin)

                volumes.append(V_np)
                aprint(
                    f"shape={V_np.shape}, range=[{V_np.min():.2f}, {V_np.max():.2f}]"
                )

    # --- Cull at each level and collect stats ---
    from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy

    all_results: list[list[dict]] = []
    unculled_renders: list[np.ndarray] = []

    with asection("Culling and rendering"):
        for ti, (t, gsplats, V) in enumerate(zip(TIMEPOINTS, gsplats_list, volumes)):
            tp_results = []

            with asection(f"Timepoint {t} ({gsplats.n_splats} splats)"):
                V_unculled = render_gaussians_numpy(V.shape, gsplats, truncate=TRUNCATE)
                unculled_renders.append(V_unculled)

                mse_full = float(np.mean((V - V_unculled) ** 2))
                psnr_full = 10.0 * np.log10(float(V.max()) ** 2 / (mse_full + 1e-12))
                comp_full = _compute_compression(gsplats.n_splats, gsplats.ndim, V.size)

                aprint(
                    f"Unculled: {gsplats.n_splats} splats, "
                    f"PSNR={psnr_full:.2f} dB, "
                    f"{comp_full['fold']:.1f}x compression"
                )

                for level in CULL_LEVELS:
                    culled = gsplats.cull(
                        V,
                        truncate=TRUNCATE,
                        error_percentile=level,
                    )

                    V_culled = render_gaussians_numpy(
                        V.shape, culled, truncate=TRUNCATE
                    )

                    mse = float(np.mean((V - V_culled) ** 2))
                    psnr = 10.0 * np.log10(float(V.max()) ** 2 / (mse + 1e-12))
                    comp = _compute_compression(culled.n_splats, culled.ndim, V.size)

                    n_removed = gsplats.n_splats - culled.n_splats
                    pct_removed = 100.0 * n_removed / max(gsplats.n_splats, 1)

                    aprint(
                        f"  p={level:5.1f}%: "
                        f"{culled.n_splats:5d} splats "
                        f"(-{n_removed}, -{pct_removed:.1f}%), "
                        f"PSNR={psnr:.2f} dB, "
                        f"{comp['fold']:.1f}x compression"
                    )

                    tp_results.append(
                        {
                            "level": level,
                            "culled": culled,
                            "rendered": V_culled,
                            "psnr": psnr,
                            "n_splats": culled.n_splats,
                            "n_removed": n_removed,
                            "pct_removed": pct_removed,
                            "compression_fold": comp["fold"],
                        }
                    )

            all_results.append(tp_results)

    # --- Summary table ---
    with asection("Summary"):
        header = f"{'Level':>8s}"
        for t in TIMEPOINTS:
            header += f"  |  t={t:>3d} splats   PSNR     comp"
        aprint(header)
        aprint("-" * len(header))

        row = f"{'full':>8s}"
        for ti in range(len(TIMEPOINTS)):
            gs = gsplats_list[ti]
            V = volumes[ti]
            V_u = unculled_renders[ti]
            mse = float(np.mean((V - V_u) ** 2))
            psnr = 10.0 * np.log10(float(V.max()) ** 2 / (mse + 1e-12))
            comp = _compute_compression(gs.n_splats, gs.ndim, V.size)
            row += f"  |  {gs.n_splats:>6d}  {psnr:>6.2f} dB  {comp['fold']:>5.1f}x"
        aprint(row)

        for li, level in enumerate(CULL_LEVELS):
            row = f"{'p=' + str(level) + '%':>8s}"
            for ti in range(len(TIMEPOINTS)):
                r = all_results[ti][li]
                row += (
                    f"  |  {r['n_splats']:>6d}  "
                    f"{r['psnr']:>6.2f} dB  "
                    f"{r['compression_fold']:>5.1f}x"
                )
            aprint(row)

# --- Napari visualization ---
if not NO_NAPARI:
    import napari

    with asection("Launching napari"):
        aprint("Building visualization stacks...")

        n_levels = len(CULL_LEVELS) + 1  # +1 for unculled
        n_tp = len(TIMEPOINTS)
        vol_shape = volumes[0].shape
        scale_zyx = list(VOXEL_SIZE_ZYX)

        # All stacks: 5D (n_tp, n_levels, Z, Y, X) so sliders align
        stack_recon = np.zeros((n_tp, n_levels) + vol_shape, dtype=np.float32)
        stack_resid = np.zeros_like(stack_recon)

        level_labels = ["full (unculled)"] + [f"p={lv}%" for lv in CULL_LEVELS]
        splat_counts = np.zeros((n_tp, n_levels), dtype=int)
        psnr_values = np.zeros((n_tp, n_levels))
        compression_folds = np.zeros((n_tp, n_levels))

        for ti in range(n_tp):
            V = volumes[ti]

            stack_recon[ti, 0] = unculled_renders[ti]
            stack_resid[ti, 0] = V - unculled_renders[ti]
            splat_counts[ti, 0] = gsplats_list[ti].n_splats
            mse = float(np.mean((V - unculled_renders[ti]) ** 2))
            psnr_values[ti, 0] = 10.0 * np.log10(float(V.max()) ** 2 / (mse + 1e-12))
            comp = _compute_compression(
                gsplats_list[ti].n_splats, gsplats_list[ti].ndim, V.size
            )
            compression_folds[ti, 0] = comp["fold"]

            for li, r in enumerate(all_results[ti]):
                stack_recon[ti, li + 1] = r["rendered"]
                stack_resid[ti, li + 1] = V - r["rendered"]
                splat_counts[ti, li + 1] = r["n_splats"]
                psnr_values[ti, li + 1] = r["psnr"]
                compression_folds[ti, li + 1] = r["compression_fold"]

        # Original volume: broadcast to 5D so all layers share sliders
        stack_original = np.stack(volumes, axis=0)[:, np.newaxis, ...]
        stack_original = np.broadcast_to(
            stack_original, (n_tp, n_levels) + vol_shape
        ).copy()

        scale_5d = [1.0, 1.0] + scale_zyx

        viewer = napari.Viewer(
            title="C. elegans — Contribution-Based Culling", ndisplay=3
        )

        viewer.add_image(
            stack_original,
            name="original (preprocessed) volume",
            colormap="gray",
            contrast_limits=[0, float(stack_original.max())],
            rendering="mip",
            scale=scale_5d,
        )

        viewer.add_image(
            stack_recon,
            name="reconstruction (culling sweep)",
            colormap="gray",
            contrast_limits=[0, float(stack_recon.max())],
            rendering="mip",
            scale=scale_5d,
        )

        viewer.add_image(
            np.abs(stack_resid),
            name="absolute residual (culling sweep)",
            colormap="gray",
            contrast_limits=[0, max(1e-12, float(np.abs(stack_resid).max()))],
            rendering="mip",
            scale=scale_5d,
        )

        def _update_overlay(event=None) -> None:
            steps = viewer.dims.current_step
            if len(steps) >= 2:
                ti = min(int(steps[0]), n_tp - 1)
                li = min(int(steps[1]), n_levels - 1)
                t_val = TIMEPOINTS[ti]
                lbl = level_labels[li]
                n_s = splat_counts[ti, li]
                psnr_v = psnr_values[ti, li]
                fold_v = compression_folds[ti, li]
                viewer.text_overlay.visible = True
                viewer.text_overlay.text = (
                    f"t={t_val}  |  {lbl}  |  "
                    f"{n_s:,} splats  |  "
                    f"PSNR={psnr_v:.2f} dB  |  "
                    f"{fold_v:.1f}x compression"
                )

        _update_overlay()
        viewer.dims.events.current_step.connect(_update_overlay)

        try:
            viewer.dims.axis_labels = [
                "timepoint",
                "culling level",
                "z",
                "y",
                "x",
            ]
        except Exception:
            pass

        viewer.camera.angles = (45, 45, 45)
        viewer.camera.zoom = 4.0

        aprint("")
        aprint("Controls:")
        aprint("  Slider 0 (top): timepoint")
        aprint("  Slider 1: culling level (0=full, 1..5 = 99%..90%)")
        aprint("  Toggle layers to compare original vs reconstruction")
        aprint("  Text overlay shows splat count, PSNR, and compression")
        aprint("")
        aprint("Culling levels:")
        for i, lbl in enumerate(level_labels):
            aprint(f"  {i}: {lbl}")

        napari.run()
else:
    aprint("\nDemo completed (napari disabled)")
