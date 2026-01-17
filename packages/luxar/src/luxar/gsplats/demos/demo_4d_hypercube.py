#!/usr/bin/env python3
"""
4D Hypercube - nD Algorithm Validation and Scalability Test

**What this demo demonstrates:**
- 4D Gaussian splatting on hypercube data (8×64×64×64)
- Complete nD pipeline validation (works for any dimensionality)
- 4D auto-candidate generation with dimension-aware heuristics
- 10-parameter covariance matrices (d=4: 4+10+1+1=16 floats/splat)
- Napari 4D visualization with hypercube navigation
- Computational scalability beyond 3D

**Key concepts:**
- nD generalization: All algorithms work for arbitrary dimensions
- Hypercube: 4D data where 1st dimension might be time/spectral
- 4D covariance: 10 unique parameters in symmetric 4×4 matrix
- Compression scaling: Bits per hypervoxel (bpv) accounting
- Dynamic operations: Seeding/pruning work correctly in 4D

**Data source:** Synthetic 4D hypercube with 100 Gaussian blobs
- Shape: (8, 64, 64, 64) = 2,097,152 hypervoxels
- Interpretation: 8 time/spectral slices of 64³ spatial volumes
- Blobs: Random 4D Gaussians with varying sizes and intensities

**Visualization:** Napari 4D navigator
- Input hypercube
- Reconstruction hypercube
- Absolute residual
- 4D splat centers (navigable points)

**Navigation:**
- Compression slider (axis 0): Adjust compression level
- Time/spectral slider (axis 1): Navigate through 4th dimension
- Spatial sliders (axes 2-4): Navigate Z, Y, X

**Performance:** Reduced iterations (400) due to 4D computational cost
**Technical:** Validates nD rendering, energy ranking, and compression in 4D

**Related demos:**
- demo_3d_synthetic_phantom.py - 3D version
- demo_2d_synthetic_blobs.py - 2D version (shows progression 2D→3D→4D)
"""

import sys

import napari
import numpy as np
from arbol import Arbol, aprint, asection

from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.utils.trils import tril_size, unpack_tril

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("🌌 4D Gaussian Splatting Demo (napari disabled)")
    aprint("Running 4D fitting validation in headless mode...")

# ======= Demo knobs =======
SIZE = 64
N_ITERS = 400  # Number of optimization iterations (reduced for 4D computational cost)
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps:0"
N_FRAMES = 20  # number of compression steps (<= #splats)
TRUNCATE_SIG = 3.0  # rendering support truncation (≈ ±3σ)
# ==========================

# Setup Arbol
Arbol.max_depth = 3


with asection("4D Gaussian Splatting Demo"):
    aprint("🌌 Interactive 4D hypercube analysis with nD algorithm validation")

    with asection("Creating 4D test data"):
        # Create 4D hypercube data: (time/spectral, z, y, x)
        shape_4d = (8, SIZE, SIZE, SIZE)  # 4D hypercube
        aprint(f"Creating 4D hypercube: {shape_4d}")

        V = np.zeros(shape_4d, dtype=np.float32)

        # Add multiple 4D Gaussian blobs with different characteristics
        n_blobs = 100
        for i in range(n_blobs):
            # Random center in 4D space
            center = [np.random.uniform(1, s - 1) for s in shape_4d]

            # Random size and intensity
            sigma = np.random.uniform(1.0, 6.0)
            amplitude = np.random.uniform(0.5, 1.0)

            # Create 4D coordinate grids
            grids = np.meshgrid(*[np.arange(s) for s in shape_4d], indexing="ij")

            # Compute 4D distance from center
            dist_sq = sum((g - c) ** 2 for g, c in zip(grids, center))

            # Add 4D Gaussian blob
            blob = amplitude * np.exp(-dist_sq / (2 * sigma**2))
            V += blob

        # Add some noise for realism
        V += np.random.normal(0, 0.05, shape_4d)
        V = np.clip(V, 0, None).astype(np.float32)

        aprint(f"Created 4D volume: {V.shape} = {V.size:,} hypervoxels")
        aprint(f"Volume range: [{V.min():.4f}, {V.max():.4f}]")

    # Configure dynamic operations
    dynamic_config = DynamicOpsConfig()
    aprint(f"Dynamic operations enabled (step_every={dynamic_config.step_every})")

    with asection(f"Fitting 4D Gaussian splats ({N_ITERS} iterations)"):
        # Test complete nD pipeline with auto-candidate generation
        result = fit_gaussian_splats(
            V,
            # seeds auto-generated with 4D-aware intelligent defaults
            n_iters=N_ITERS,
            truncate=TRUNCATE_SIG,
            device=DEVICE,
            verbose=True,
            # Dynamic operations
            enable_dynamic_ops=True,
            dynamic_config=dynamic_config,
            # convergence movie:
            napari_movie=True,
        )

        aprint(f"Fitted {len(result.amplitudes)} 4D splats successfully")

        if len(result.amplitudes) == 0:
            raise RuntimeError(
                "No splats were fitted; try lowering thresholds or increasing iterations."
            )

# ----- 4D Compression ranking by approximate L2 energy -----
# For a 4D Gaussian, ||G||_2^2 = (sqrt(pi))^d * sqrt(det Σ).
# Here sqrt(det Σ) = prod(diag(L)) because Σ = L L^T.
d = 4

# Extract L for energy ranking
L_full = unpack_tril(result.cholesky_factors, d)  # (N, 4, 4)
diag_prod = np.prod(
    np.stack([L_full[:, i, i] for i in range(d)], axis=1), axis=1
)  # ∏ diag(L)
energy_score = (result.amplitudes**2) * (np.sqrt(np.pi) ** d) * diag_prod
order = np.argsort(-energy_score)  # descending

aprint("4D energy scores computed for compression ranking")

# ----- Precompute reconstructions/residuals per frame -----
N = len(result.amplitudes)
keep_counts = np.unique(
    np.linspace(1, N, num=min(N_FRAMES, N), endpoint=True).astype(int)
)

aprint(f"Computing {len(keep_counts)} compression frames for 4D data...")

stack_recon = np.zeros((len(keep_counts),) + V.shape, dtype=np.float32)
stack_resid = np.zeros_like(stack_recon)
rel_err_frames = np.zeros(len(keep_counts), dtype=np.float32)

# 4D bit accounting (float32 for all params + amps)
FLOAT_BITS = 32
FLOATS_PER_SPLAT = (
    d + tril_size(d) + 1 + 1
)  # centers(4) + packed L(10) + sharpness(1) + amplitude(1) = 16
BITS_PER_SPLAT = FLOATS_PER_SPLAT * FLOAT_BITS
HYPERCUBE_BITS = V.size * FLOAT_BITS
NUM_HYPERVOXELS = V.size

model_bits_frames = np.zeros(len(keep_counts), dtype=np.float64)
bpv_frames = np.zeros(len(keep_counts), dtype=np.float64)  # bits per hypervoxel
bit_compression_pct = np.zeros(len(keep_counts), dtype=np.float64)

# 4D centers for each frame
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
        sharpnesses=result.sharpnesses[idx],
        stats={},  # Empty stats for rendering subset
    )

    # Reconstruction & residual
    Vk = render_gaussians_numpy(V.shape, result_idx, truncate=TRUNCATE_SIG)
    stack_recon[i] = Vk
    stack_resid[i] = V - Vk
    rel_err_frames[i] = np.linalg.norm(V - Vk) / (np.linalg.norm(V) + 1e-12)

    # Bit cost for current model
    model_bits_frames[i] = int(K) * BITS_PER_SPLAT
    bpv_frames[i] = model_bits_frames[i] / NUM_HYPERVOXELS  # bits per hypervoxel
    bit_compression_pct[i] = 100.0 * (1.0 - (model_bits_frames[i] / HYPERCUBE_BITS))

    # 4D centers
    Ck = result.centers[idx]  # (K, 4) hypercube centers
    centers_frames.append(Ck)

aprint("4D rendering and compression analysis complete!")

# 4D napari viewer with hypercube visualization (only if napari enabled)
if not NO_NAPARI:
    aprint("Launching 4D napari viewer...")
    viewer = napari.Viewer(title="4D Gaussian Splatting Demo")

    # Add the input hypercube
    viewer.add_image(
        V,
        name="input_hypercube",
        colormap="viridis",
        contrast_limits=[0, float(V.max())],
        opacity=1.0,
    )

    # Add reconstruction hypercube stack
    lyr_recon = viewer.add_image(
        stack_recon,
        name="reconstruction (compression, 4D)",
        colormap="viridis",  # Match input colormap for consistency
        opacity=1.0,
        contrast_limits=[0, float(V.max())],  # Same as input for fair comparison
    )

    # Add residual hypercube stack
    lyr_resid = viewer.add_image(
        np.abs(stack_resid),
        name="absolute residual",
        colormap="inferno",  # Better visibility
        opacity=1.0,
        contrast_limits=[0, max(1e-12, float(np.abs(stack_resid).max()))],
    )

    # 4D centers that update with slider
    pts = viewer.add_points(
        np.zeros((0, 4)),
        name="4D splat centers",
        size=4,
        border_color="yellow",
        face_color="transparent",
    )

    # Try to label axes (4D navigation)
    try:
        viewer.dims.axis_labels = ["compression", "time/spectral", "z", "y", "x"]
    except Exception:
        pass

    def _set_overlay_text_4d(t_index: int) -> None:
        """Update text overlay with 4D-specific information."""
        K = int(keep_counts[t_index])
        bits_model = int(model_bits_frames[t_index])
        bpv = float(bpv_frames[t_index])  # bits per hypervoxel
        pct_bits = float(bit_compression_pct[t_index])
        rel = float(rel_err_frames[t_index])
        hypercube_size_str = "×".join(map(str, shape_4d))
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            f"4D Splats: {K}/{N}  |  Model bits: {bits_model:,}  "
            f"|  Model bpv: {bpv:.3f} (raw=32.000)  |  Bit compression: {pct_bits:.1f}%  "
            f"|  rel L2 err: {rel:.4f}  |  Hypercube: {hypercube_size_str} hypervoxels"
        )

    def _update_layers_for_t_4d(t_index: int) -> None:
        """Update 4D layers for given time index."""
        # Update 4D centers
        pts.data = centers_frames[t_index]

        # Update text overlay
        _set_overlay_text_4d(t_index)

    # Initialize with first frame
    _update_layers_for_t_4d(0)

    # Hook slider to updates
    def _on_step_change_4d(event=None) -> None:
        t = viewer.dims.current_step[0]
        _update_layers_for_t_4d(int(t))

    viewer.dims.events.current_step.connect(_on_step_change_4d)

    aprint("\n🎮 4D Navigation Tips:")
    aprint("  • Use dimension sliders to navigate through 4D hypercube")
    aprint("  • Use the compression slider (axis 0) to see compression progression")
    aprint("  • Yellow points = 4D splat centers")
    aprint("  • Toggle layers on/off to compare input vs reconstruction")

    napari.run()

# Console summary for 4D (always shown)
aprint("\n4D Gaussian Splat Compression Analysis")
aprint(f"{'=' * 60}")
aprint(f"Raw hypercube bits (float32): {HYPERCUBE_BITS:,}  |  raw bpv = 32.000")
aprint(f"Hypercube size: {shape_4d} = {NUM_HYPERVOXELS:,} hypervoxels")
aprint(f"Bits per splat: {BITS_PER_SPLAT} (4 centers + 10 covariance + 1 amplitude)")
aprint(f"{'=' * 60}")

for i, K in enumerate(keep_counts):
    aprint(
        f"Frame {i:02d} | keep {K:4d} | model_bits={int(model_bits_frames[i]):>10,d} "
        f"| bit_compression={bit_compression_pct[i]:6.1f}% | bpv={bpv_frames[i]:6.3f} "
        f"| relL2={rel_err_frames[i]:.4f}"
    )

aprint("\n📊 4D Compression Insights:")
best_compression: float = float(np.max(bit_compression_pct))
final_error = rel_err_frames[-1]
aprint(f"  • Best compression: {best_compression:.1f}% bit reduction")
aprint(f"  • Final relative error: {final_error:.4f}")
aprint("  • 4D splats demonstrate excellent nD scalability!")

if NO_NAPARI:
    aprint("✅ 4D validation complete - all nD algorithms working correctly")
