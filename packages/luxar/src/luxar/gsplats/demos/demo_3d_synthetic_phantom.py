#!/usr/bin/env python3
"""
3D Synthetic Phantom - Volumetric Compression with Ellipsoid Visualization

**What this demo demonstrates:**
- 3D Gaussian splatting on synthetic volumetric phantom data
- Full-covariance 3D ellipsoids (oriented, anisotropic splats)
- Interactive 3D compression analysis with napari slider
- Wireframe ellipsoid visualization showing 3D splat shapes
- Systematic phantom approach for 3D validation
- Bits per voxel (bpv) accounting for volumetric compression

**Key concepts:**
- 3D Gaussians: Each splat is a 3D oriented ellipsoid with 6 covariance parameters
- Phantom validation: Controlled 3D blobs with known properties for systematic testing
- Wireframe rendering: 3 principal plane circles visualize each 3D ellipsoid
- MIP rendering: Maximum Intensity Projection for better 3D visualization
- Energy ranking: Compression by L2 energy importance (same as 2D)

**Data source:** Synthetic 3D phantom (64³ voxels) with 15 controlled Gaussian blobs
**Visualization:** Interactive 3D napari viewer (ndisplay=3)
- Original input volume
- Reconstruction at current compression level
- Absolute residual volume
- 3D wireframe ellipsoids (cyan points)
- Splat centers (yellow points)

**Controls:**
- Top slider (axis 0): Adjust compression level
- Mouse + Shift: Rotate 3D view
- Mouse wheel: Zoom in/out
- Toggle layers to compare volumes

**Performance:** Reduced iterations (1000) for 3D computational cost
**Related demos:**
- demo_2d_synthetic_blobs.py - 2D version with same compression approach
- demo_3d_dapi_microscopy.py - Real microscopy data in 3D
- demo_4d_hypercube.py - 4D extension showing nD scalability
"""

import sys

import napari
import numpy as np
from arbol import Arbol, aprint, asection

from luxar.gsplats.demos._demo_common import ellipsoid_wireframe_from_L
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.utils.trils import tril_size, unpack_tril

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("🧊 3D Gaussian Splatting Demo (napari disabled)")
    aprint("Running 3D phantom validation in headless mode...")

# ======= Demo knobs =======
N_ITERS = 1000  # Number of optimization iterations (reduced for 3D computational cost)
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps:0"
N_FRAMES = 30  # number of compression steps (<= #splats)
TRUNCATE_SIG = 3.0  # rendering support truncation (≈ ±3σ)
SPACING = (1.0, 1.0, 1.0)  # (z, y, x); not needed for ranking here
# ==========================

# Setup Arbol
Arbol.max_depth = 3


with asection("3D Gaussian Splatting Demo"):
    aprint(
        "🧊 Interactive 3D volumetric compression analysis with wireframe visualization"
    )

    with asection("Creating 3D phantom data"):
        # Create 3D phantom with systematic Gaussian blobs (phantom approach from 4D demo)
        volume_size = 64  # Size for 3D demo
        shape_3d = (volume_size, volume_size, volume_size)
        aprint(f"Creating 3D phantom: {shape_3d}")

        V = np.zeros(shape_3d, dtype=np.float32)

        # Add multiple 3D Gaussian blobs with controlled characteristics
        n_blobs = 15  # Controlled number of features for systematic validation
        for i in range(n_blobs):
            # Random center in 3D space (avoid boundaries)
            center = [np.random.uniform(5, s - 5) for s in shape_3d]

            # Random size and intensity appropriate for 3D
            sigma = np.random.uniform(3.0, 8.0)  # 3D-appropriate blob sizes
            amplitude = np.random.uniform(0.6, 1.0)

            # Create 3D coordinate grids
            grids = np.meshgrid(*[np.arange(s) for s in shape_3d], indexing="ij")

            # Compute 3D distance from center
            dist_sq = sum((g - c) ** 2 for g, c in zip(grids, center))

            # Add 3D Gaussian blob to phantom
            blob = amplitude * np.exp(-dist_sq / (2 * sigma**2))
            V += blob

        # Add minimal noise for realism
        V += np.random.normal(0, 0.02, shape_3d)
        V = np.clip(V, 0, None).astype(np.float32)

        aprint(f"Created 3D phantom: {V.shape} = {V.size:,} voxels")
        aprint(f"Volume range: [{V.min():.4f}, {V.max():.4f}]")

    # Configure dynamic operations
    dynamic_config = DynamicOpsConfig()
    aprint(f"Dynamic operations enabled (step_every={dynamic_config.step_every})")

    with asection(f"Fitting 3D Gaussian splats ({N_ITERS} iterations)"):
        # Fit oriented (full-covariance) 3D Gaussians with auto-candidate generation
        result = fit_gaussian_splats(
            V,
            seeds=1000,
            n_iters=N_ITERS,
            truncate=TRUNCATE_SIG,
            device=DEVICE,
            verbose=True,
            napari_movie=True,
        )

        aprint(f"Fitted {len(result.amplitudes)} splats successfully")

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

aprint("Energy scores computed for compression ranking")

# ----- Precompute reconstructions/residuals + wireframe ellipsoids per frame -----
N = len(result.amplitudes)
keep_counts = np.unique(
    np.linspace(1, N, num=min(N_FRAMES, N), endpoint=True).astype(int)
)

aprint(f"Computing {len(keep_counts)} compression frames...")

stack_recon = np.zeros((len(keep_counts),) + V.shape, dtype=np.float32)
stack_resid = np.zeros_like(stack_recon)
rel_err_frames = np.zeros(len(keep_counts), dtype=np.float32)

# Bit accounting (float32 for all params + amps)
FLOAT_BITS = 32
FLOATS_PER_SPLAT = d + tril_size(d) + 1  # centers(3) + packed L(6) + amplitude(1) = 10
BITS_PER_SPLAT = FLOATS_PER_SPLAT * FLOAT_BITS
VOLUME_BITS = V.size * FLOAT_BITS
NUM_VOXELS = V.size

model_bits_frames = np.zeros(len(keep_counts), dtype=np.float64)
bpv_frames = np.zeros(len(keep_counts), dtype=np.float64)  # bits per voxel
bit_compression_pct = np.zeros(len(keep_counts), dtype=np.float64)

# 3D shapes data for each frame (wireframe ellipsoids)
wireframes_frames = []
centers_frames = []

for i, K in enumerate(keep_counts):
    if i % 5 == 0:
        aprint(f"  Processing frame {i + 1}/{len(keep_counts)} (K={K} splats)...")

    idx = order[:K]

    # Create sliced result for rendering
    result_idx = GSplatData(
        centers=result.centers[idx],
        amplitudes=result.amplitudes[idx],
        cholesky_factors=result.cholesky_factors[idx],
        stats={},  # Empty stats for rendering subset
    )

    # Reconstruction & residual
    Vk = render_gaussians_numpy(V.shape, result_idx, truncate=TRUNCATE_SIG)
    stack_recon[i] = Vk
    stack_resid[i] = V - Vk
    rel_err_frames[i] = np.linalg.norm(V - Vk) / (np.linalg.norm(V) + 1e-12)

    # Bit cost for current model
    model_bits_frames[i] = int(K) * BITS_PER_SPLAT
    bpv_frames[i] = model_bits_frames[i] / NUM_VOXELS  # bits per voxel
    bit_compression_pct[i] = 100.0 * (1.0 - (model_bits_frames[i] / VOLUME_BITS))

    # Wireframe ellipsoids & centers (2σ contour)
    Lk = L_full[idx]  # (K, 3, 3)
    Ck = result.centers[idx]  # (K, 3) voxel centers (z,y,x)

    # Create wireframes for each ellipsoid (limit to reasonable number for visualization)
    max_wireframes = min(K, 50)  # Limit wireframes for performance
    wireframes = []
    for j in range(max_wireframes):
        wireframe = ellipsoid_wireframe_from_L(Ck[j], Lk[j], t=2.0, n_pts=24)
        wireframes.append(wireframe)

    # Combine all wireframes into single array for napari
    if wireframes:
        wireframes_combined = np.vstack(wireframes)
    else:
        wireframes_combined = np.zeros((0, 3))

    wireframes_frames.append(wireframes_combined)
    centers_frames.append(Ck)

aprint("3D rendering and compression analysis complete!")

# 4) Napari viewer with 3D volumes and "compression" slider (only if napari enabled)
if not NO_NAPARI:
    aprint("Launching 3D napari viewer...")
    viewer = napari.Viewer(ndisplay=3)  # Force 3D display

    # Add the input volume
    viewer.add_image(
        V,
        name="input_volume",
        colormap="viridis",
        contrast_limits=[0, float(V.max())],
        opacity=1.0,  # Full opacity for clear comparison
        rendering="mip",  # Maximum intensity projection for better 3D visualization
    )

    # Add reconstruction volume stack
    lyr_recon = viewer.add_image(
        stack_recon,
        name="reconstruction (compression, 3D)",
        colormap="viridis",  # Match input colormap for consistency
        opacity=1.0,  # Full opacity for clear comparison
        contrast_limits=[0, float(V.max())],  # Same as input for fair comparison
        rendering="mip",
    )

    # Add residual volume stack
    lyr_resid = viewer.add_image(
        np.abs(stack_resid),
        name="absolute residual",
        colormap="inferno",  # Better visibility than turbo
        opacity=1.0,  # Full opacity for clear visualization
        contrast_limits=[0, max(1e-12, float(np.abs(stack_resid).max()))],
        rendering="mip",
    )

    # 3D wireframe ellipsoids that update with slider
    wireframe_layer = viewer.add_points(
        np.zeros((0, 3)),
        name="ellipsoid wireframes (kept)",
        size=1,
        border_color="cyan",
        face_color="cyan",
        opacity=0.6,
    )

    # Centers that update with slider
    pts = viewer.add_points(
        np.zeros((0, 3)),
        name="centers (kept)",
        size=4,
        border_color="yellow",
        face_color="transparent",
    )

    # Try to label axes (requires napari >= 0.4.18)
    try:
        viewer.dims.axis_labels = ["compression", "z", "y", "x"]
    except Exception:
        pass

    # Set better 3D camera view
    viewer.camera.angles = (15, 25, 120)  # Good 3D viewing angle
    viewer.camera.zoom = 0.8

    def _set_overlay_text_3d(t_index: int) -> None:
        """Update text overlay with 3D-specific information."""
        K = int(keep_counts[t_index])
        bits_model = int(model_bits_frames[t_index])
        bpv = float(bpv_frames[t_index])  # bits per voxel
        pct_bits = float(bit_compression_pct[t_index])
        rel = float(rel_err_frames[t_index])
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            f"3D Splats: {K}/{N}  |  Model bits: {bits_model:,}  "
            f"|  Model bpv: {bpv:.3f} (raw=32.000)  |  Bit compression: {pct_bits:.1f}%  "
            f"|  rel L2 err: {rel:.4f}  |  Volume: {volume_size}³ voxels"
        )

    def _update_layers_for_t_3d(t_index: int) -> None:
        """Update 3D layers for given time index."""
        # Update wireframe ellipsoids
        wireframe_layer.data = wireframes_frames[t_index]

        # Update centers
        pts.data = centers_frames[t_index]

        # Update text overlay
        _set_overlay_text_3d(t_index)

    # Initialize with first frame
    _update_layers_for_t_3d(0)

    # Hook slider to updates
    def _on_step_change_3d(event=None) -> None:
        t = viewer.dims.current_step[0]
        _update_layers_for_t_3d(int(t))

    viewer.dims.events.current_step.connect(_on_step_change_3d)

    # Console summary for 3D
    aprint("\n3D Gaussian Splat Compression Analysis")
    aprint(f"{'=' * 60}")
    aprint(f"Raw volume bits (float32): {VOLUME_BITS:,}  |  raw bpv = 32.000")
    aprint(f"Volume size: {volume_size}³ = {NUM_VOXELS:,} voxels")
    aprint(f"Bits per splat: {BITS_PER_SPLAT} (3 centers + 6 covariance + 1 amplitude)")
    aprint(f"{'=' * 60}")

    for i, K in enumerate(keep_counts):
        aprint(
            f"Frame {i:02d} | keep {K:4d} | model_bits={int(model_bits_frames[i]):>10,d} "
            f"| bit_compression={bit_compression_pct[i]:6.1f}% | bpv={bpv_frames[i]:6.3f} "
            f"| relL2={rel_err_frames[i]:.4f}"
        )

    aprint("\n🎮 3D Navigation Tips:")
    aprint("  • Use mouse + Shift to rotate the 3D view")
    aprint("  • Use the top slider (axis 0) to see compression progression")
    aprint("  • Yellow points = splat centers")
    aprint("  • Cyan wireframes = 2σ ellipsoid boundaries")
    aprint("  • Toggle layers on/off to compare input vs reconstruction")

    aprint("\n📊 Compression Insights:")
    best_compression: float = float(np.max(bit_compression_pct))
    final_error = rel_err_frames[-1]
    aprint(f"  • Best compression: {best_compression:.1f}% bit reduction")
    aprint(f"  • Final relative error: {final_error:.4f}")
    aprint("  • 3D splats can achieve good compression on volumetric data!")

    napari.run()
else:
    aprint("\n✅ Demo completed successfully (napari visualization disabled)")

# Console summary for 3D (always shown)
aprint("\n3D Gaussian Splat Compression Analysis")
aprint(f"{'=' * 60}")
aprint(f"Raw volume bits (float32): {VOLUME_BITS:,}  |  raw bpv = 32.000")
aprint(f"Volume size: {volume_size}³ = {NUM_VOXELS:,} voxels")
aprint(f"Bits per splat: {BITS_PER_SPLAT} (3 centers + 6 covariance + 1 amplitude)")
aprint(f"{'=' * 60}")

for i, K in enumerate(keep_counts):
    aprint(
        f"Frame {i:02d} | keep {K:4d} | model_bits={int(model_bits_frames[i]):>10,d} "
        f"| bit_compression={bit_compression_pct[i]:6.1f}% | bpv={bpv_frames[i]:6.3f} "
        f"| relL2={rel_err_frames[i]:.4f}"
    )

aprint("\n📊 3D Compression Insights:")
best_compression = np.max(bit_compression_pct)
final_error = rel_err_frames[-1]
aprint(f"  • Best compression: {best_compression:.1f}% bit reduction")
aprint(f"  • Final relative error: {final_error:.4f}")
aprint("  • 3D phantom splats demonstrate excellent volumetric compression!")

if NO_NAPARI:
    aprint(
        "✅ 3D phantom validation complete - systematic synthetic data approach working"
    )
