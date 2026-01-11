#!/usr/bin/env python3
"""
3D DAPI Microscopy - Real Biological Data from Image Data Resource

**What this demo demonstrates:**
- 3D Gaussian splatting on real DAPI-stained nuclear microscopy data
- **Metal acceleration on Apple Silicon (3-7x speedup automatically!)**
- Remote zarr data loading from Image Data Resource (IDR)
- Automatic downscaling to manageable size (128³ voxels)
- OME-ZARR format handling (5D: T×C×Z×Y×X)
- Channel extraction (DAPI channel from multi-channel data)
- 3D ellipsoid fitting to real biological structures

**Key concepts:**
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

**Visualization:** 3D napari viewer with MIP rendering
- DAPI input volume (gray colormap)
- Reconstruction (cyan colormap, 80% opacity)
- Absolute residual (red colormap)
- 3D wireframe ellipsoids (yellow, 30% opacity)
- Splat centers (lime green points)

**Controls:**
- Top slider: Compression level (all splats → minimal)
- Mouse + Shift: Rotate 3D view
- Toggle layers to see individual components
- Observe ellipsoid alignment with nuclear structures

**Related demos:**
- demo_3d_synthetic_phantom.py - Controlled 3D synthetic data
- multiscale/demos/demo_decompose_3d_dapi_nuclei.py - Multi-scale version at FULL resolution
"""

import sys

import napari
import numpy as np
import zarr
from arbol import Arbol, aprint, asection

from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.fit_result import GSplatData
from luxar.gsplats.models.gsplats.metal import is_metal_available
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.utils.trils import tril_size, unpack_tril

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("🧬 3D DAPI Gaussian Splatting Demo (napari disabled)")
    aprint("Running all computations without napari visualization...")

# ======= Demo knobs =======
N_ITERS = 6000
DEVICE = None  # None -> auto: CUDA on Linux with NVIDIA, MPS on macOS, CPU fallback
N_FRAMES = 30  # number of compression steps (<= #splats)
ZARR_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
DAPI_CHANNEL = 1  # DAPI is typically channel 1 (0-indexed)
TARGET_SIZE = 128  # Downscale to this size for manageable computation
TIME_POINT = 0  # Use first time point
# Hardware acceleration (enabled by default, auto-detected)
USE_METAL = True  # Enable Metal acceleration on Apple Silicon (3-7x faster!)
USE_CUDA = True  # Enable CUDA acceleration on NVIDIA GPUs (10-50x faster!)
# ==========================

# Setup Arbol
Arbol.max_depth = 4


# --- Helper: oriented 3D ellipsoid wireframe from covariance ---
def ellipsoid_wireframe_from_L(
    mu_zyx: np.ndarray, L: np.ndarray, t: float = 2.0, n_pts: int = 32
) -> np.ndarray:
    """
    Build a wireframe approximating the 3D ellipsoid corresponding to the level set
    (x-μ)^T Σ^{-1} (x-μ) = t^2, where Σ = L L^T (full cov in voxel units).

    Returns wireframe points as (n_wireframe_pts, 3) array of (z,y,x) coordinates.
    Creates circular wireframes along the three principal planes.
    """
    Sigma = L @ L.T  # (3,3)
    # Eigen-decompose Sigma for principal axes
    evals, evecs = np.linalg.eigh(Sigma)  # evals >= 0
    evals = np.clip(evals, 1e-12, None)
    # Radii along principal axes at level t: r_i = t * sqrt(lambda_i)
    radii = t * np.sqrt(evals)  # (3,)

    # Create wireframe circles in the three principal planes
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    wireframe_pts = []

    # XY plane (z=0 in principal coords)
    circle_xy = np.zeros((n_pts, 3))
    circle_xy[:, 0] = radii[0] * cos_theta  # x-axis in principal coords
    circle_xy[:, 1] = radii[1] * sin_theta  # y-axis in principal coords
    circle_xy[:, 2] = 0  # z-axis
    # Transform to data coordinates
    pts_xy = (evecs @ circle_xy.T).T + mu_zyx[None, :]
    wireframe_pts.append(pts_xy)

    # XZ plane (y=0 in principal coords)
    circle_xz = np.zeros((n_pts, 3))
    circle_xz[:, 0] = radii[0] * cos_theta
    circle_xz[:, 1] = 0
    circle_xz[:, 2] = radii[2] * sin_theta
    pts_xz = (evecs @ circle_xz.T).T + mu_zyx[None, :]
    wireframe_pts.append(pts_xz)

    # YZ plane (x=0 in principal coords)
    circle_yz = np.zeros((n_pts, 3))
    circle_yz[:, 0] = 0
    circle_yz[:, 1] = radii[1] * cos_theta
    circle_yz[:, 2] = radii[2] * sin_theta
    pts_yz = (evecs @ circle_yz.T).T + mu_zyx[None, :]
    wireframe_pts.append(pts_yz)

    # Combine all wireframes
    return np.vstack(wireframe_pts).astype(np.float32)


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
                    f"Detected OME-ZARR 5D: T={n_time}, C={n_channels}, Z={z_size}, Y={y_size}, X={x_size}"
                )

                # Extract DAPI channel
                if DAPI_CHANNEL >= n_channels:
                    aprint(
                        f"⚠ Warning: Requested channel {DAPI_CHANNEL} but only {n_channels} available"
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

                # Downscale to target size using zoom
                from scipy.ndimage import zoom

                zoom_factors = [
                    TARGET_SIZE / z_size,
                    TARGET_SIZE / y_size,
                    TARGET_SIZE / x_size,
                ]
                aprint(
                    f"Downscaling with zoom factors: Z={zoom_factors[0]:.3f}, Y={zoom_factors[1]:.3f}, X={zoom_factors[2]:.3f}"
                )
                V = zoom(V, zoom_factors, order=1)  # order=1 for linear interpolation
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

                # Downscale to target size using zoom
                from scipy.ndimage import zoom

                zoom_factors = [
                    TARGET_SIZE / z_size,
                    TARGET_SIZE / y_size,
                    TARGET_SIZE / x_size,
                ]
                aprint(
                    f"Downscaling with zoom factors: Z={zoom_factors[0]:.3f}, Y={zoom_factors[1]:.3f}, X={zoom_factors[2]:.3f}"
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

                # Downscale to target size using zoom
                from scipy.ndimage import zoom

                zoom_factors = [
                    TARGET_SIZE / z_size,
                    TARGET_SIZE / y_size,
                    TARGET_SIZE / x_size,
                ]
                aprint(
                    f"Downscaling with zoom factors: Z={zoom_factors[0]:.3f}, Y={zoom_factors[1]:.3f}, X={zoom_factors[2]:.3f}"
                )
                V = zoom(V, zoom_factors, order=1)
                aprint(f"Downscaled to: {V.shape}")
            else:
                raise ValueError(
                    f"Unexpected data shape: {full_shape}. Expected 3D, 4D, or 5D (OME-ZARR)."
                )

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
            shape_3d = (TARGET_SIZE, TARGET_SIZE, TARGET_SIZE)
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

    # Device auto-detection: fitter will automatically select best backend:
    # - Linux + NVIDIA GPU: CUDA with custom kernels (10-50x speedup)
    # - macOS + Apple Silicon: MPS with Metal acceleration (3-7x speedup)
    # - Fallback: CPU
    if DEVICE is None:
        import torch

        if USE_CUDA and torch.cuda.is_available():
            aprint("🚀 CUDA available - will use custom CUDA kernels (10-50x speedup!)")
        elif USE_METAL and is_metal_available() and torch.backends.mps.is_available():
            aprint("🚀 Metal available - will use MPS device (3-7x speedup!)")
        else:
            aprint("Using CPU device (no GPU acceleration available)")
    else:
        aprint(f"Using specified device: {DEVICE}")

    with asection(f"Fitting 3D Gaussian splats ({N_ITERS} iterations)"):
        # Fit oriented (full-covariance) 3D Gaussians with auto-seed generation
        # Hardware acceleration is automatic:
        # - CUDA kernels on NVIDIA GPUs (10-50x speedup)
        # - Metal kernels on Apple Silicon (3-7x speedup)
        result = fit_gaussian_splats(
            V,
            seeds=8000,  # initial seed count
            n_iters=N_ITERS,
            device=DEVICE,
            use_metal=USE_METAL,  # Enable Metal acceleration (macOS)
            use_cuda=USE_CUDA,  # Enable CUDA acceleration (NVIDIA)
            verbose=True,
            max_abs_error=0.1,
            # convergence movie:
            napari_movie=(not NO_NAPARI),
            movie_every=int(N_ITERS / 30),
        )

        aprint(f"🎉 Fitted {len(result.amplitudes)} splats successfully")

        if len(result.amplitudes) == 0:
            raise RuntimeError(
                "No splats were fitted; try lowering thresholds or increasing iterations."
            )

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

# Bit accounting (float32 for centers + packed L + sharpness + amplitude)
FLOAT_BITS = 32
FLOATS_PER_SPLAT = (
    d + tril_size(d) + 1 + 1
)  # centers(3) + packed L(6) + sharpness(1) + amp(1) = 11
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
            sharpnesses=result.sharpnesses[idx],
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
                f"Frame {i + 1:02d}/{len(keep_counts)}: {K} splats, {bit_compression_pct[i]:.1f}% compression"
            )

# Console summary
aprint("📈 3D Compression Analysis Results:")
aprint(f"Raw volume bits (float32): {IMAGE_BITS:,}  |  raw bpv = 32.000")
for i, K in enumerate(keep_counts[::5]):  # Show every 5th frame
    idx = i * 5
    if idx < len(keep_counts):
        aprint(
            f"Frame {idx:02d} | keep {K:4d} | model_bits={int(model_bits_frames[idx]):>10,d} "
            f"| compression={bit_compression_pct[idx]:6.1f}% | bpv={bpp_frames[idx]:6.3f} "
            f"| relL2={rel_err_frames[idx]:.4f}"
        )

if not NO_NAPARI:
    # ----- Napari viewer with "compression" slider -----
    aprint("🔬 Launching interactive 3D napari viewer...")
    viewer = napari.Viewer(title="3D DAPI Gaussian Splatting Demo", ndisplay=3)

    # Add original DAPI volume
    viewer.add_image(
        V,
        name="DAPI (input)",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
        rendering="mip",  # Maximum intensity projection
    )

    # Add reconstruction stack
    viewer.add_image(
        stack_recon,
        name="reconstruction (compression, oriented 3D)",
        colormap="cyan",
        contrast_limits=[0, float(V.max())],
        rendering="mip",
        opacity=0.8,
    )

    # Add residual stack
    viewer.add_image(
        np.abs(stack_resid),
        name="absolute residual",
        colormap="red",
        contrast_limits=[0, max(1e-12, float(np.abs(stack_resid).max()))],
        rendering="mip",
        opacity=0.5,
    )

    # Dynamic wireframe and points layers
    def _update_3d_layers(t_index: int) -> None:
        """Update wireframes and centers for current compression level."""
        # Clear existing wireframe and point layers
        for layer in list(viewer.layers):
            if "wireframe" in layer.name.lower() or "centers" in layer.name.lower():
                viewer.layers.remove(layer)

        # Get current frame data
        centers = centers_frames[t_index]

        # Add splat centers
        viewer.add_points(
            centers,
            name="splat centers (kept)",
            size=2.0,
            face_color="lime",
            border_color="lime",
            opacity=0.8,
        )

        # Update text overlay
        K = int(keep_counts[t_index])
        bits_model = int(model_bits_frames[t_index])
        bpp = float(bpp_frames[t_index])
        pct_bits = float(bit_compression_pct[t_index])
        rel = float(rel_err_frames[t_index])
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            f"🧬 3D DAPI Demo | Kept splats: {K}/{N}  |  Model bits: {bits_model:,}  "
            f"|  bpv: {bpp:.3f} (raw=32.0)  |  Compression: {pct_bits:.1f}%  "
            f"|  rel L2 error: {rel:.4f}"
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
    aprint("   • Yellow points = 3D ellipsoid wireframes")
    aprint("   • Lime points = splat centers")
    aprint("")
    aprint("🔍 What to notice:")
    aprint("   • How 3D nuclear structures are represented by oriented ellipsoids")
    aprint("   • Efficiency of 3D Gaussians for volumetric microscopy data")
    aprint("   • Trade-off between storage size and reconstruction fidelity")
    aprint("   • Alignment of ellipsoids with nuclear morphology")

    # Best compression and final error
    best_compression = bit_compression_pct.max()
    final_error = rel_err_frames[-1]
    aprint(f"  • Best compression: {best_compression:.1f}% bit reduction")
    aprint(f"  • Final relative error: {final_error:.4f}")
    aprint("  • 3D splats efficiently capture volumetric DAPI structures!")

    napari.run()
else:
    aprint("\n✅ Demo completed successfully (napari visualization disabled)")
