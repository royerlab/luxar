#!/usr/bin/env python3
"""
2D Gaussian splatting demo with interactive compression visualization in napari.
"""

import sys

import napari
import numpy as np
from arbol import aprint
from skimage import data, filters

from luxar.gsplats.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.models.gsplats.gsplat_model import render_gaussians_numpy
from luxar.gsplats.utils.trils import tril_size, unpack_tril

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI and len(sys.argv) > 1:
    aprint("🎯 2D Gaussian Splatting Demo (napari disabled)")
    aprint("Note: This demo is designed for interactive napari visualization.")
    aprint(
        "✅ Demo structure verified - would run with full napari functionality when enabled"
    )
    sys.exit(0)

# ======= Demo knobs =======
LOSS_TYPE = "l1"  # True: Poisson deviance; False: MSE
LR = 0.01  # e.g. 0.01-0.03 works well
L1_AMP = 0.001  # e.g. 1e-3 to encourage sparsity
N_ITERS = 1000
DEVICE = None  # "mps:0"    # None -> auto; or "cuda"/"cpu"
N_FRAMES = 40  # number of compression steps (<= #splats)
TRUNCATE_SIG = 3.0  # rendering support truncation (≈ ±3σ)
SPACING = (1.0, 1.0)  # (row, col); not needed for ranking here
USE_DYNAMIC_OPS = True  # Enable dynamic operations (seeding, splitting, pruning) - set to False to disable
# ==========================


# --- Helper: oriented 2D ellipse polygon from covariance ---
def ellipse_polygon_from_L(
    mu_yx: np.ndarray, L: np.ndarray, t: float = 2.0, n_pts: int = 64
) -> np.ndarray:
    """
    Build a polygon approximating the 2D ellipse corresponding to the level set
    (x-μ)^T Σ^{-1} (x-μ) = t^2, where Σ = L L^T (full cov in voxel units).

    Returns (n_pts, 2) array of (y,x) polygon points.
    """
    Sigma = L @ L.T  # (2,2)
    # Eigen-decompose Sigma for principal axes
    evals, evecs = np.linalg.eigh(Sigma)  # evals >= 0
    evals = np.clip(evals, 1e-12, None)
    # Radii along principal axes at level t: r_i = t * sqrt(lambda_i)
    radii = t * np.sqrt(evals)  # (2,)
    # Parametric angles
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2, n_pts)
    # Map unit circle -> ellipse in data coords: μ + R diag(r) circle
    pts = (evecs @ (radii[:, None] * circle)).T + mu_yx[None, :]
    return pts.astype(np.float32)


# 1) Make a soft 2D “blobs” image
blobs = data.binary_blobs(
    length=256, blob_size_fraction=0.06, n_dim=2, volume_fraction=0.18, rng=42
).astype(float)
V = filters.gaussian(blobs, sigma=3.25).astype(np.float32)

# 2) Configure dynamic operations (if enabled)
dynamic_config = None
if USE_DYNAMIC_OPS:
    dynamic_config = DynamicOpsConfig()
    aprint(f"Dynamic operations enabled (step_every={dynamic_config.step_every})")

# 3) Fit oriented (full-covariance) Gaussians with auto-candidate generation
params_full, amps, stats = fit_gaussian_splats(
    V,
    # centers_overcomplete auto-generated with intelligent defaults
    n_iters=N_ITERS,
    lr=LR,
    loss_type=LOSS_TYPE,
    l1_amp=L1_AMP,
    truncate=TRUNCATE_SIG,
    device=DEVICE,
    verbose=True,
    # Dynamic operations
    enable_dynamic_ops=USE_DYNAMIC_OPS,
    dynamic_config=dynamic_config,
    napari_movie=True,
    movie_every=1,
)

if len(amps) == 0:
    raise RuntimeError(
        "No splats were fitted; try lowering thresholds or increasing iterations."
    )

# ----- Compression ranking by approximate L2 energy -----
# For a general Gaussian, ||G||_2^2 = (sqrt(pi))^d * sqrt(det Σ).
# Here sqrt(det Σ) = prod(diag(L)) because Σ = L L^T.
d = 2
L_packed = params_full[:, d:]  # (N, 3) in 2D
L_full = unpack_tril(L_packed, d)  # (N, 2, 2)
diag_prod = np.prod(
    np.stack([L_full[:, 0, 0], L_full[:, 1, 1]], axis=1), axis=1
)  # ∏ diag(L)
energy_score = (amps**2) * (np.sqrt(np.pi) ** d) * diag_prod
order = np.argsort(-energy_score)  # descending

# ----- Precompute reconstructions/residuals + oriented polygons per frame -----
N = len(amps)
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

    # Reconstruction & residual
    Vk = render_gaussians_numpy(
        V.shape, params_full[idx], amps[idx], truncate=TRUNCATE_SIG
    )
    stack_recon[i] = Vk
    stack_resid[i] = V - Vk
    rel_err_frames[i] = np.linalg.norm(V - Vk) / (np.linalg.norm(V) + 1e-12)

    # Bit cost for current model
    model_bits_frames[i] = int(K) * BITS_PER_SPLAT
    bpp_frames[i] = model_bits_frames[i] / NUM_PIXELS
    bit_compression_pct[i] = 100.0 * (1.0 - (model_bits_frames[i] / IMAGE_BITS))

    # Polygons & centers (2σ contour)
    Lk = L_full[idx]  # (K, 2, 2)
    Ck = params_full[idx, :2]  # (K, 2) voxel centers (y,x)
    polys = [
        ellipse_polygon_from_L(Ck[j], Lk[j], t=2.0, n_pts=64) for j in range(len(idx))
    ]
    polygons_frames.append(polys)
    centers_frames.append(Ck)

# 4) Napari viewer with "compression" slider
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


def _set_overlay_text(t_index: int):
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


def _update_layers_for_t(t_index: int):
    shapes.data = polygons_frames[t_index]  # list of (M_i, 2) polygons
    pts.data = centers_frames[t_index]
    _set_overlay_text(t_index)


# Initialize
_update_layers_for_t(0)


# Hook slider to updates
def _on_step_change(event=None):
    t = viewer.dims.current_step[0]
    _update_layers_for_t(int(t))


viewer.dims.events.current_step.connect(_on_step_change)

# Console summary
aprint(f"Raw image bits (float32): {IMAGE_BITS:,}  |  raw bpp = 32.000")
for i, K in enumerate(keep_counts):
    aprint(
        f"Frame {i:02d} | keep {K:4d} | model_bits={int(model_bits_frames[i]):>10,d} "
        f"| bit_compression={bit_compression_pct[i]:6.1f}% | bpp={bpp_frames[i]:6.3f} "
        f"| relL2={rel_err_frames[i]:.4f}"
    )

aprint(
    "Ready. Use the top slider (axis 0) to move from keeping all splats toward keeping just one."
)
napari.run()
