#!/usr/bin/env python3
"""
3D DAPI Microscopy - Real Biological Data from Image Data Resource

**What this demo demonstrates:**
- **GPU-accelerated Non-Local Means (NLM) denoising** with auto-calibrated h
- 3D Gaussian splatting on real DAPI-stained nuclear microscopy data
- **Metal acceleration on Apple Silicon (substantial speedup automatically; chip-dependent)**
- Remote zarr data loading from Image Data Resource (IDR)
- Automatic downscaling to manageable size (128³ voxels)
- OME-ZARR format handling (5D: T×C×Z×Y×X)
- Channel extraction (DAPI channel from multi-channel data)
- 3D ellipsoid fitting to real biological structures

**Key concepts:**
- **GPU-accelerated NLM denoising** with Noise2Self auto-calibration (before fitting)
- Real data challenges: Noise, irregular shapes, varying intensities
- OME-ZARR: Standard format for multi-dimensional microscopy data
- Remote loading: Uses fsspec to stream zarr data from IDR
- Nuclear morphology: DAPI reveals chromatin structure and nuclear shapes
- Biological validation: Tests algorithm on real scientific imaging data

**Data source:** IDR (Image Data Resource)
- URL: https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr
- Type: OME-ZARR 5D volume (Time × Channel × Z × Y × X)
- Channel: DAPI (channel 1, DNA stain showing nuclei)
- Processing: Downscaled to 128³ voxels via zoom interpolation
- Fallback: Creates synthetic nucleus-like blobs if remote load fails

**Visualization:** 3D napari viewer with MIP rendering (all layers gray/white LUT)
- DAPI raw (noisy) volume
- DAPI NLM-denoised volume (before/after comparison)
- NLM removed noise (absolute difference)
- Final reconstruction from fitted model (with PSNR/compression stats)
- Final residual: original minus reconstruction
- Compression sweep: reconstruction at varying splat counts (slider)
- Compression sweep residual (slider)

**Controls:**
- Top slider: Compression level (all splats → minimal)
- Mouse + Shift: Rotate 3D view
- Toggle layers to see individual components
- Observe ellipsoid alignment with nuclear structures

**Related demos:**
- demo_3d_synthetic_phantom.py - Controlled 3D synthetic data
- multiscale/demos/demo_decompose_3d_dapi_nuclei.py - Multi-scale at FULL res
"""

import sys

import numpy as np
import zarr
from arbol import Arbol, aprint, asection

from luxar.gsplats.demos._demo_common import ellipsoid_wireframe_from_L
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.models.gsplats.metal import is_metal_available
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.utils.trils import tril_size, unpack_tril

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("🧬 3D DAPI Gaussian Splatting Demo (napari disabled)")
    aprint("Running all computations without napari visualization...")

# ======= Demo knobs =======
NUM_SPLATS = 6000
N_ITERS = 6000
DEVICE = None  # None -> auto: CUDA on Linux with NVIDIA, MPS on macOS, CPU fallback
N_FRAMES = 30  # number of compression steps (<= #splats)
ZARR_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
DAPI_CHANNEL = 1  # DAPI is typically channel 1 (0-indexed)
TARGET_SIZE = None  # Downscale to this size for manageable computation
TIME_POINT = 0  # Use first time point
# Hardware acceleration (enabled by default, auto-detected)
USE_METAL = True  # Enable Metal acceleration on Apple Silicon (substantially faster; chip-dependent)
USE_CUDA = True  # Enable CUDA acceleration on NVIDIA GPUs (often orders of magnitude faster; GPU-dependent)
# NLM denoising parameters
NLM_PATCH_SIZE = 3
NLM_PATCH_DISTANCE = 5
# ==========================

# Setup Arbol
Arbol.max_depth = 5


with asection("3D DAPI Gaussian Splatting Demo"):
    aprint("🧬 Real microscopy data: DAPI-stained nuclei from IDR")
    aprint(f"📦 Data source: {ZARR_URL}")

    with asection("Loading DAPI data from zarr"):
        aprint("Loading data from remote zarr store...")
        aprint("Note: Remote data loading may take a moment...")
        try:
            # Open remote zarr store via fsspec
            import fsspec

            mapper = fsspec.get_mapper(ZARR_URL)

            # Try opening as a group first
            try:
                store = zarr.open_group(mapper, mode="r")
                aprint("Zarr group opened successfully")
            except (zarr.errors.PathNotFoundError, zarr.errors.GroupNotFoundError):
                # Try as direct array
                store = zarr.open_array(mapper, mode="r")
                aprint("Zarr array opened successfully")

            # OME-ZARR format: access the '0' array (highest resolution)
            data = store["0"]
            full_shape = data.shape
            aprint(f"OME-ZARR data shape: {full_shape}")
            aprint(f"Data type: {data.dtype}")

            # OME-ZARR typically uses (T, C, Z, Y, X) format
            if len(full_shape) == 5:
                n_time, n_channels, z_size, y_size, x_size = full_shape
                aprint(
                    f"OME-ZARR 5D: T={n_time} C={n_channels} Z={z_size}"
                    f" Y={y_size} X={x_size}"
                )

                # Extract DAPI channel
                if DAPI_CHANNEL >= n_channels:
                    aprint(
                        f"⚠ Channel {DAPI_CHANNEL} requested but {n_channels} available"
                    )
                    aprint("Using channel 0 instead")
                    DAPI_CHANNEL = 0

                aprint(
                    f"Extracting time={TIME_POINT}, channel={DAPI_CHANNEL} (DAPI)..."
                )

                # Load full volume for this channel and time point
                aprint(f"Loading full volume: Z={z_size}, Y={y_size}, X={x_size}")
                V = data[TIME_POINT, DAPI_CHANNEL, :, :, :]
                V = np.array(V, dtype=np.float32)

                if TARGET_SIZE:
                    # Downscale to target size using zoom
                    from scipy.ndimage import zoom

                    zoom_factors = [
                        TARGET_SIZE / z_size,
                        TARGET_SIZE / y_size,
                        TARGET_SIZE / x_size,
                    ]
                    aprint(
                        f"Zoom: Z={zoom_factors[0]:.3f} Y={zoom_factors[1]:.3f}"
                        f" X={zoom_factors[2]:.3f}"
                    )
                    V = zoom(V, zoom_factors, order=1)
                    aprint(f"Downscaled to: {V.shape}")

            elif len(full_shape) == 4:
                # (C, Z, Y, X) format
                n_channels, z_size, y_size, x_size = full_shape
                aprint(
                    f"Detected 4D: C={n_channels}, Z={z_size}, Y={y_size}, X={x_size}"
                )

                if DAPI_CHANNEL >= n_channels:
                    aprint(f"⚠ Warning: Using channel 0 instead of {DAPI_CHANNEL}")
                    DAPI_CHANNEL = 0

                aprint(f"Extracting channel={DAPI_CHANNEL} (DAPI)...")

                # Load full volume for this channel
                aprint(f"Loading full volume: Z={z_size}, Y={y_size}, X={x_size}")
                V = data[DAPI_CHANNEL, :, :, :]
                V = np.array(V, dtype=np.float32)

                if TARGET_SIZE:
                    # Downscale to target size using zoom
                    from scipy.ndimage import zoom

                    zoom_factors = [
                        TARGET_SIZE / z_size,
                        TARGET_SIZE / y_size,
                        TARGET_SIZE / x_size,
                    ]
                    aprint(
                        f"Zoom: Z={zoom_factors[0]:.3f} Y={zoom_factors[1]:.3f}"
                        f" X={zoom_factors[2]:.3f}"
                    )
                    V = zoom(V, zoom_factors, order=1)
                    aprint(f"Downscaled to: {V.shape}")

            elif len(full_shape) == 3:
                # Single channel, just ZYX
                z_size, y_size, x_size = full_shape
                aprint(f"Detected 3D: Z={z_size}, Y={y_size}, X={x_size}")

                # Load full volume
                aprint(f"Loading full volume: Z={z_size}, Y={y_size}, X={x_size}")
                V = data[:, :, :]
                V = np.array(V, dtype=np.float32)

                if TARGET_SIZE:
                    # Downscale to target size using zoom
                    from scipy.ndimage import zoom

                    zoom_factors = [
                        TARGET_SIZE / z_size,
                        TARGET_SIZE / y_size,
                        TARGET_SIZE / x_size,
                    ]
                    aprint(
                        f"Zoom: Z={zoom_factors[0]:.3f} Y={zoom_factors[1]:.3f}"
                        f" X={zoom_factors[2]:.3f}"
                    )
                    V = zoom(V, zoom_factors, order=1)
                    aprint(f"Downscaled to: {V.shape}")
            else:
                raise ValueError(f"Unexpected shape: {full_shape}. Expected 3D/4D/5D.")

            # Normalize to [0, 100] range for consistency with other demos
            V_min, V_max = V.min(), V.max()
            if V_max > V_min:
                V = ((V - V_min) / (V_max - V_min)) * 100.0
            else:
                aprint("⚠ Warning: Uniform data, using constant value")
                V = np.ones_like(V) * 50.0

            aprint(f"Loaded DAPI volume: {V.shape} = {V.size:,} voxels")
            aprint(f"Intensity range: [{V.min():.2f}, {V.max():.2f}]")

        except Exception as e:
            aprint(f"❌ Error loading zarr data: {e}")
            aprint("Falling back to synthetic phantom data for demo purposes")

            # Create synthetic data as fallback
            fallback_size = TARGET_SIZE if TARGET_SIZE is not None else 128
            shape_3d = (fallback_size, fallback_size, fallback_size)
            V = np.zeros(shape_3d, dtype=np.float32)

            # Add nucleus-like blobs
            n_nuclei = 10
            for i in range(n_nuclei):
                center = [np.random.uniform(5, s - 5) for s in shape_3d]
                sigma = np.random.uniform(4.0, 8.0)
                amplitude = np.random.uniform(60.0, 100.0)

                grids = np.meshgrid(*[np.arange(s) for s in shape_3d], indexing="ij")
                dist_sq = sum((g - c) ** 2 for g, c in zip(grids, center))
                blob = amplitude * np.exp(-dist_sq / (2 * sigma**2))
                V += blob

            V = np.clip(V, 0, 100).astype(np.float32)
            aprint(f"Created synthetic DAPI-like volume: {V.shape}")

    # ----- NLM Denoising -----
    with asection("NLM Denoising (GPU-accelerated)"):
        import torch

        from luxar.gsplats.preprocessing import calibrate_nlm_h, denoise_nlm
        from luxar.gsplats.utils.device import resolve_torch_device

        denoise_device = str(
            resolve_torch_device(use_cuda=USE_CUDA, use_metal=USE_METAL)
        )
        aprint(f"Denoising device: {denoise_device}")

        vol_tensor = torch.from_numpy(V)

        with asection("Calibrating NLM h (Noise2Self / J-invariant)"):
            h = calibrate_nlm_h(
                vol_tensor,
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

    # ----- Show denoising comparison in napari (blocking) -----
    if not NO_NAPARI:
        import napari

        aprint("🔬 Launching napari to compare raw vs denoised...")
        aprint("   Close the napari window to continue with splat fitting.")
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
        aprint("Napari closed — continuing with splat fitting...")

    # Device auto-detection: fitter will automatically select best backend:
    # - Linux + NVIDIA GPU: CUDA with custom kernels (often orders of magnitude faster, GPU-dependent)
    # - macOS + Apple Silicon: MPS with Metal acceleration (substantial speedup, chip-dependent)
    # - Fallback: CPU
    if DEVICE is None:
        import torch

        if USE_CUDA and torch.cuda.is_available():
            aprint(
                "🚀 CUDA available - will use custom CUDA kernels (substantial speedup, GPU-dependent!)"
            )
        elif USE_METAL and is_metal_available() and torch.backends.mps.is_available():
            aprint(
                "🚀 Metal available - will use MPS device (substantial speedup, chip-dependent!)"
            )
        else:
            aprint("Using CPU device (no GPU acceleration available)")
    else:
        aprint(f"Using specified device: {DEVICE}")

    with asection(f"Fitting 3D Gaussian splats ({N_ITERS} iterations)"):
        # Fit oriented (full-covariance) 3D Gaussians with auto-seed generation
        # Hardware acceleration is automatic:
        # - CUDA kernels on NVIDIA GPUs (often orders of magnitude faster, GPU-dependent)
        # - Metal kernels on Apple Silicon (substantial speedup, chip-dependent)
        result = fit_gaussian_splats(
            V,
            seeds=NUM_SPLATS,  # initial seed count
            n_iters=N_ITERS,
            device=DEVICE,
            use_metal=USE_METAL,  # Enable Metal acceleration (macOS)
            use_cuda=USE_CUDA,  # Enable CUDA acceleration (NVIDIA)
            use_fp16=True,
            verbose=True,
            max_abs_error=0.1,
            # convergence movie:
            napari_movie=(not NO_NAPARI),
            movie_every=int(N_ITERS / 30),
        )

        aprint(f"🎉 Fitted {len(result.amplitudes)} splats successfully")

        if len(result.amplitudes) == 0:
            raise RuntimeError(
                "No splats fitted; lower thresholds or increase iterations."
            )

    # ----- Final reconstruction vs original -----
    with asection("Final reconstruction vs original"):
        V_final = render_gaussians_numpy(V.shape, result, truncate=3.0)
        residual_final = V - V_final

        # Quality metrics
        mse_final = float(np.mean(residual_final**2))
        psnr_final = 10.0 * np.log10(float(V.max()) ** 2 / (mse_final + 1e-12))
        rel_error_final = float(
            np.linalg.norm(residual_final) / (np.linalg.norm(V) + 1e-12)
        )
        max_abs_error_final = float(np.abs(residual_final).max())

        n_splats_final = len(result.amplitudes)
        model_bits_final = n_splats_final * (3 + tril_size(3) + 1 + 1) * 32
        compression_final = 100.0 * (1.0 - model_bits_final / (V.size * 32))

        fold_compression_final = (V.size * 32) / max(model_bits_final, 1)

        aprint(f"Final model: {n_splats_final} splats")
        aprint(f"Compression: {compression_final:.1f}% ({fold_compression_final:.1f}x)")
        aprint(f"MSE: {mse_final:.4f}")
        aprint(f"PSNR: {psnr_final:.2f} dB")
        aprint(f"Relative L2 error: {rel_error_final:.4f}")
        aprint(f"Max absolute error: {max_abs_error_final:.4f}")

# ----- Compression ranking by approximate L2 energy -----
# For a 3D Gaussian, ||G||_2^2 = (sqrt(pi))^d * sqrt(det Σ).
# Here sqrt(det Σ) = prod(diag(L)) because Σ = L L^T.
d = 3

# Extract L for energy ranking
L_full = unpack_tril(result.cholesky_factors, d)  # (N, 3, 3)
diag_prod = np.prod(
    np.stack([L_full[:, 0, 0], L_full[:, 1, 1], L_full[:, 2, 2]], axis=1), axis=1
)  # ∏ diag(L)
energy_score = (result.amplitudes**2) * (np.sqrt(np.pi) ** d) * diag_prod
order = np.argsort(-energy_score)  # descending

aprint(f"📊 Ranking {len(result.amplitudes)} splats by L2 energy contribution")

# ----- Precompute reconstructions/residuals + wireframes per frame -----
N = len(result.amplitudes)
keep_counts = np.unique(
    np.linspace(1, N, num=min(N_FRAMES, N), endpoint=True).astype(int)
)

stack_recon = np.zeros((len(keep_counts),) + V.shape, dtype=np.float32)
stack_resid = np.zeros_like(stack_recon)
rel_err_frames = np.zeros(len(keep_counts), dtype=np.float32)

# Bit accounting (float32 for centers + packed L + amplitude)
FLOAT_BITS = 32
FLOATS_PER_SPLAT = d + tril_size(d) + 1  # centers(3) + packed L(6) + amp(1) = 10
BITS_PER_SPLAT = FLOATS_PER_SPLAT * FLOAT_BITS
IMAGE_BITS = V.size * FLOAT_BITS
NUM_VOXELS = V.size

model_bits_frames = np.zeros(len(keep_counts), dtype=np.float64)
bpp_frames = np.zeros(len(keep_counts), dtype=np.float64)
bit_compression_pct = np.zeros(len(keep_counts), dtype=np.float64)

wireframes_frames = []
centers_frames = []

aprint("🎬 Precomputing compression frames...")
with asection("Computing 3D reconstruction quality at different compression levels"):
    for i, K in enumerate(keep_counts):
        idx = order[:K]

        # Create sliced result for rendering
        result_idx = GSplatData(
            centers=result.centers[idx],
            amplitudes=result.amplitudes[idx],
            cholesky_factors=result.cholesky_factors[idx],
            stats={},  # Empty stats for rendering subset
        )

        # Render
        Vk = render_gaussians_numpy(V.shape, result_idx, truncate=3.0)
        stack_recon[i] = Vk
        stack_resid[i] = V - Vk
        rel_err_frames[i] = np.linalg.norm(V - Vk) / (np.linalg.norm(V) + 1e-12)

        model_bits_frames[i] = int(K) * BITS_PER_SPLAT
        bpp_frames[i] = model_bits_frames[i] / NUM_VOXELS
        bit_compression_pct[i] = 100.0 * (1.0 - (model_bits_frames[i] / IMAGE_BITS))

        # Build 3D wireframe ellipsoids
        Lk = L_full[idx]  # (K, 3, 3)
        Ck = result.centers[idx]  # (K, 3) - centers in ZYX order

        wireframes = []
        for j in range(len(idx)):
            wf = ellipsoid_wireframe_from_L(Ck[j], Lk[j], t=2.0, n_pts=32)
            wireframes.append(wf)

        wireframes_frames.append(wireframes)
        centers_frames.append(Ck)

        if (i + 1) % 5 == 0:
            aprint(
                f"Frame {i + 1:02d}/{len(keep_counts)}: {K} splats, "
                f"{bit_compression_pct[i]:.1f}% compression"
            )

# Console summary
aprint("📈 3D Compression Analysis Results:")
aprint(f"Raw volume bits (float32): {IMAGE_BITS:,}  |  raw bpv = 32.000")
for i, K in enumerate(keep_counts[::5]):  # Show every 5th frame
    idx = i * 5
    if idx < len(keep_counts):
        mb = int(model_bits_frames[idx])
        bp = bpp_frames[idx]
        re = rel_err_frames[idx]
        fold = IMAGE_BITS / max(mb, 1)
        aprint(
            f"Fr{idx:02d} K={K:4d} bits={mb:>10,} {fold:>7.1f}x "
            f"bpv={bp:.3f} L2={re:.4f}"
        )

if not NO_NAPARI:
    import napari

    aprint("🔬 Launching napari viewer with full results...")
    viewer = napari.Viewer(title="3D DAPI Gaussian Splatting Demo", ndisplay=3)

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

    # ----- Final reconstruction layers (post-culling model) -----
    viewer.add_image(
        V_final,
        name="final reconstruction",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
        rendering="mip",
    )

    viewer.add_image(
        np.abs(residual_final),
        name="final residual",
        colormap="gray",
        contrast_limits=[0, max(1e-12, max_abs_error_final)],
        rendering="mip",
    )

    # ----- Compression sweep layers (slider-controlled) -----
    # Add reconstruction stack
    viewer.add_image(
        stack_recon,
        name="reconstruction (compression, oriented 3D)",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
        rendering="mip",
    )

    # Add residual stack
    viewer.add_image(
        np.abs(stack_resid),
        name="absolute residual",
        colormap="gray",
        contrast_limits=[0, max(1e-12, float(np.abs(stack_resid).max()))],
        rendering="mip",
    )

    def _update_3d_layers(t_index: int) -> None:
        """Update wireframes and centers for current compression level."""

        # Update text overlay
        K = int(keep_counts[t_index])
        bits_model = int(model_bits_frames[t_index])
        fold = IMAGE_BITS / max(bits_model, 1)
        rel = float(rel_err_frames[t_index])
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            f"Splats: {K}/{N}  |  {fold:.1f}x compression  |  rel L2 error: {rel:.4f}"
        )

    # Initialize and wire slider
    _update_3d_layers(0)

    def _on_step_change(event=None) -> None:
        # Get the first dimension step (compression axis)
        if hasattr(viewer.dims, "current_step"):
            t = viewer.dims.current_step[0]
            _update_3d_layers(int(t))

    viewer.dims.events.current_step.connect(_on_step_change)

    # Set 3D rendering defaults
    viewer.camera.angles = (45, 45, 45)
    viewer.camera.zoom = 2.0

    aprint("")
    aprint("🎛️  3D Controls:")
    aprint("   • Use the top slider (axis 0) to explore compression levels")
    aprint("   • Rotate view with mouse drag")
    aprint("   • Zoom with mouse wheel")
    aprint("   • Toggle layers on/off to compare input vs reconstruction")
    aprint("")
    aprint("📊 Layer guide:")
    aprint("   • 'DAPI (raw / noisy)' = original volume before denoising")
    aprint("   • 'DAPI (NLM denoised)' = after GPU-accelerated NLM denoising")
    aprint(
        "   • 'final reconstruction' = fitted model output "
        f"({n_splats_final} splats, PSNR {psnr_final:.1f} dB)"
    )
    aprint("   • 'final residual' = denoised minus reconstruction")
    aprint("   • 'reconstruction (compression...)' = compression sweep (slider)")
    aprint("   • 'absolute residual' = compression sweep residual (slider)")
    aprint("")
    aprint("🔍 What to notice:")
    aprint("   • Toggle between 'raw' and 'NLM denoised' to see denoising effect")
    aprint("   • Toggle 'final reconstruction' to compare with denoised input")
    aprint("   • Toggle 'final residual' to see where the model struggles")
    aprint("   • Use the compression slider to see quality vs splat count")
    aprint("   • Alignment of ellipsoids with nuclear morphology")

    # Best compression and final error
    best_fold = IMAGE_BITS / max(int(model_bits_frames[-1]), 1)
    final_error = rel_err_frames[-1]
    aprint(f"  • All splats: {best_fold:.1f}x compression")
    aprint(f"  • Final relative error: {final_error:.4f}")
    aprint("  • 3D splats efficiently capture volumetric DAPI structures!")

    napari.run()
else:
    aprint("\n✅ Demo completed successfully (napari visualization disabled)")
