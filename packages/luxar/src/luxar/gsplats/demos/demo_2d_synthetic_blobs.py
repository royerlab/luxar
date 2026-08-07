#!/usr/bin/env python3
"""
2D Synthetic Blobs - Interactive Compression Analysis

**What this demo demonstrates:**
- 2D Gaussian splatting on synthetic blob data with smooth features
- Interactive compression analysis via napari slider
- Full-covariance (oriented) 2D Gaussian splats with ellipse visualization
- Energy-based ranking for progressive quality/compression trade-off
- Bit-level compression accounting (bits per pixel)
- Oriented ellipse overlays showing splat shapes and orientations

**Key concepts:**
- Compression by importance: Keeps splats with highest L2 energy contribution
- Full covariance: Each splat is an oriented ellipse (not just isotropic circle)
- Interactive exploration: Slider lets you explore quality vs compression trade-offs
- L1 loss: Robust to outliers and preserves sharp features
- Dynamic operations: Auto seeding/pruning for optimal splat distribution

**Data source:** Synthetic 2D blobs generated with scikit-image binary_blobs + Gaussian smoothing
**Visualization:** Multi-layer napari view with compression slider
- Input image
- Reconstruction at current compression level
- Absolute residual (error map)
- Oriented 2σ ellipses showing splat shapes
- Splat centers

**Controls:**
- Top slider (axis 0): Adjust compression level from all splats to just one
- Toggle layers to compare input vs reconstruction
- Watch ellipse overlays to see how splats are distributed

**Related demos:**
- demo_basic_fitting.py - Simpler introduction to the API
- demo_splats_coins.py - Real image with similar compression features
- demo_3d_synthetic_phantom.py - 3D version with volumetric data
"""

import sys

import napari
import numpy as np
from arbol import aprint
from skimage import data, filters

from luxar.gsplats.demos._demo_common import ellipse_polygon_from_L
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.utils.trils import tril_size, unpack_tril

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("🔬 Demo (napari disabled)")
    aprint("Running all computations without napari visualization...")

# ======= Demo knobs =======
LOSS_TYPE = "l1"  # L1 loss for robust features
N_ITERS = 4000
DEVICE = None  # "mps:0"    # None -> auto; or "cuda"/"cpu"
N_FRAMES = 40  # number of compression steps (<= #splats)
TRUNCATE_SIG = 3.0  # rendering support truncation (≈ ±3σ)
SPACING = (1.0, 1.0)  # (row, col); not needed for ranking here
# ==========================


# 1) Make a soft 2D “blobs” image
blobs = data.binary_blobs(
    length=256, blob_size_fraction=0.06, n_dim=2, volume_fraction=0.18, rng=42
).astype(float)
V = filters.gaussian(blobs, sigma=3.25).astype(np.float32)

# 2) Fit oriented (full-covariance) Gaussians with auto-candidate generation
result = fit_gaussian_splats(
    V,
    seeds=500,
    n_iters=N_ITERS,
    loss_type=LOSS_TYPE,
    truncate=TRUNCATE_SIG,
    device=DEVICE,
    verbose=True,
    napari_movie=(not NO_NAPARI),
    movie_every=1,
)

if len(result.amplitudes) == 0:
    raise RuntimeError(
        "No splats were fitted; try lowering thresholds or increasing iterations."
    )

# ----- Compression ranking by approximate L2 energy -----
# For a general Gaussian, ||G||_2^2 = (sqrt(pi))^d * sqrt(det Σ).
# Here sqrt(det Σ) = prod(diag(L)) because Σ = L L^T.
d = 2

# Extract L for energy ranking
L_full = unpack_tril(result.cholesky_factors, d)  # (N, 2, 2)
diag_prod = np.prod(
    np.stack([L_full[:, 0, 0], L_full[:, 1, 1]], axis=1), axis=1
)  # ∏ diag(L)
energy_score = (result.amplitudes**2) * (np.sqrt(np.pi) ** d) * diag_prod
order = np.argsort(-energy_score)  # descending

# ----- Precompute reconstructions/residuals + oriented polygons per frame -----
N = len(result.amplitudes)
keep_counts = np.unique(
    np.linspace(1, N, num=min(N_FRAMES, N), endpoint=True).astype(int)
)

stack_recon = np.zeros((len(keep_counts),) + V.shape, dtype=np.float32)
stack_resid = np.zeros_like(stack_recon)
rel_err_frames = np.zeros(len(keep_counts), dtype=np.float32)

# Bit accounting (float32 for all params + amps)
FLOAT_BITS = 32
FLOATS_PER_SPLAT = d + tril_size(d) + 1  # centers(d) + packed L + amplitude
BITS_PER_SPLAT = FLOATS_PER_SPLAT * FLOAT_BITS
IMAGE_BITS = V.size * FLOAT_BITS
NUM_PIXELS = V.size

model_bits_frames = np.zeros(len(keep_counts), dtype=np.float64)
bpp_frames = np.zeros(len(keep_counts), dtype=np.float64)
bit_compression_pct = np.zeros(len(keep_counts), dtype=np.float64)

# Shapes data for each frame (polygons for oriented 2σ ellipses)
polygons_frames = []
centers_frames = []

for i, K in enumerate(keep_counts):
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
    bpp_frames[i] = model_bits_frames[i] / NUM_PIXELS
    bit_compression_pct[i] = 100.0 * (1.0 - (model_bits_frames[i] / IMAGE_BITS))

    # Polygons & centers (2σ contour)
    Lk = L_full[idx]  # (K, 2, 2)
    Ck = result.centers[idx]  # (K, 2) voxel centers (y,x)
    polys = [
        ellipse_polygon_from_L(Ck[j], Lk[j], t=2.0, n_pts=64) for j in range(len(idx))
    ]
    polygons_frames.append(polys)
    centers_frames.append(Ck)

# Console summary
aprint(f"Raw image bits (float32): {IMAGE_BITS:,}  |  raw bpp = 32.000")
for i, K in enumerate(keep_counts):
    aprint(
        f"Frame {i:02d} | keep {K:4d} | model_bits={int(model_bits_frames[i]):>10,d} "
        f"| bit_compression={bit_compression_pct[i]:6.1f}% | bpp={bpp_frames[i]:6.3f} "
        f"| relL2={rel_err_frames[i]:.4f}"
    )

if not NO_NAPARI:
    # Napari viewer with "compression" slider
    viewer = napari.Viewer()
    viewer.add_image(
        V,
        name="input",
        colormap="magma",
        contrast_limits=[0, float(V.max())],
    )

    lyr_recon = viewer.add_image(
        stack_recon,
        name="reconstruction (compression, oriented)",
        colormap="magma",
        contrast_limits=[0, float(V.max())],  # Same as input for fair comparison
    )
    lyr_resid = viewer.add_image(
        np.abs(stack_resid),
        name="absolute residual",
        colormap="inferno",  # Better visibility than turbo
        opacity=1.0,  # Full opacity for clear visualization
        contrast_limits=[0, max(1e-12, float(np.abs(stack_resid).max()))],
    )

    # Shapes & points that update with slider
    shapes = viewer.add_shapes(
        name="oriented 2σ ellipses (kept)",
        shape_type="polygon",
        edge_color="cyan",
        edge_width=1,
        face_color=[0, 0, 0, 0],
    )
    pts = viewer.add_points(
        np.zeros((0, 2)),
        name="centers (kept)",
        size=3,
        border_color="cyan",
        face_color="transparent",
    )

    # Try to label axes (requires napari >= 0.4.18)
    try:
        viewer.dims.axis_labels = ["compression", "y", "x"]
    except Exception:
        pass

    def _set_overlay_text(t_index: int) -> None:
        K = int(keep_counts[t_index])
        bits_model = int(model_bits_frames[t_index])
        bpp = float(bpp_frames[t_index])
        pct_bits = float(bit_compression_pct[t_index])
        rel = float(rel_err_frames[t_index])
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            f"Kept splats: {K}/{N}  |  Model bits: {bits_model:,}  "
            f"|  Model bpp: {bpp:.3f} (raw=32.000)  |  Bit compression: {pct_bits:.1f}%  "
            f"|  rel L2 err: {rel:.4f}"
        )

    def _update_layers_for_t(t_index: int) -> None:
        shapes.data = polygons_frames[t_index]  # list of (M_i, 2) polygons
        pts.data = centers_frames[t_index]
        _set_overlay_text(t_index)

    # Initialize
    _update_layers_for_t(0)

    # Hook slider to updates
    def _on_step_change(event=None) -> None:
        t = viewer.dims.current_step[0]
        _update_layers_for_t(int(t))

    viewer.dims.events.current_step.connect(_on_step_change)

    aprint(
        "Ready. Use the top slider (axis 0) to move from keeping all splats toward keeping just one."
    )
    napari.run()
else:
    aprint("\n✅ Demo completed successfully (napari visualization disabled)")
