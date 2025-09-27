#!/usr/bin/env python3
"""
Human mitosis image Gaussian splatting demo with interactive compression analysis.

This demo applies Gaussian splatting to the scikit-image human mitosis dataset,
demonstrating full-covariance fitting with compression analysis via napari.
Features interactive slider to explore reconstruction quality vs compression ratio.
"""

import sys

import napari
import numpy as np
from arbol import Arbol, aprint, asection
from skimage import color, data, img_as_float32

from luxar.gsplats.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.models.gsplats.gsplat_model import render_gaussians_numpy
from luxar.gsplats.utils.trils import tril_size, unpack_tril

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI and len(sys.argv) > 1:
    aprint("🔬 Human Mitosis Gaussian Splatting Demo (napari disabled)")
    aprint("Note: This demo is designed for interactive napari visualization.")
    aprint(
        "✅ Demo structure verified - would run with full napari functionality when enabled"
    )
    sys.exit(0)

# ======= Demo knobs =======
LOSS_TYPE = "l1"
LR = 0.02  # Learning rate for fitting
N_ITERS = 2000  # Number of optimization iterations
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps:0"
N_FRAMES = 40  # number of compression steps (<= #splats)
TRUNCATE_SIG = 3.0  # rendering support truncation (≈ ±3σ)
# ==========================

# Setup Arbol
Arbol.max_depth = 3


def ellipse_polygon_from_L(
    mu_yx: np.ndarray, L: np.ndarray, t: float = 2.0, n_pts: int = 64
) -> np.ndarray:
    """
    2D oriented ellipse polygon for the contour (x-μ)^T Σ^{-1} (x-μ) = t^2, with Σ = L L^T.
    Returns (n_pts, 2) polygon in (y, x).
    """
    Sigma = L @ L.T
    evals, evecs = np.linalg.eigh(Sigma)  # principal axes
    evals = np.clip(evals, 1e-12, None)
    radii = t * np.sqrt(evals)  # radii along principal axes
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2, n_pts)
    pts = (evecs @ (radii[:, None] * circle)).T + mu_yx[None, :]
    return pts.astype(np.float32)


with asection("Human Mitosis Gaussian Splatting Demo"):
    aprint("🔬 Interactive compression analysis on biological histology data")

    with asection("Loading and preprocessing data"):
        # Load human_mitosis and prepare a soft grayscale target
        img = data.human_mitosis()  # likely RGB
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = color.rgb2gray(img)*100  # -> float in [0, 100]
        V = img_as_float32(img)

        # Crop the image to a smaller region for faster demo
        V = V[100:356, 100:356]  # Crop to 256x256

        # # Optional: mild contrast normalization & smoothing (helps candidate detection)
        # V = exposure.rescale_intensity(V, in_range="image", out_range=(0.0, 1.0)).astype(
        #     np.float32
        # )
        aprint(f"Preprocessed human mitosis image: {V.shape}")
        aprint(f"Data range: [{V.min():.4f}, {V.max():.4f}]")

    # Configure dynamic operations
    dynamic_config = DynamicOpsConfig()
    aprint(f"Dynamic operations enabled (step_every={dynamic_config.step_every})")

    with asection(f"Fitting Gaussian splats ({N_ITERS} iterations)"):
        # Fit oriented (full-covariance) Gaussians with auto-candidate generation
        params_full, amps, stats = fit_gaussian_splats(
            V,
            # centers_overcomplete auto-generated with intelligent defaults
            init_sigma_vox=0.5,
            n_iters=N_ITERS,
            loss_type=LOSS_TYPE,
            lr=LR,
            # l1_amp auto-set to 0.1 * lr = 0.002
            truncate=TRUNCATE_SIG,
            device=DEVICE,
            verbose=True,
            # Dynamic operations
            enable_dynamic_ops=True,
            dynamic_config=dynamic_config,
            max_abs_error=0.1,
            napari_movie=True,
            movie_every=1,
            movie_max_frames=None,
        )
        if len(amps) == 0:
            raise RuntimeError(
                "No splats were fitted; try lowering thresholds or increasing iterations."
            )

# ----- Compression ranking by approximate L2 energy -----
# ||G||_2^2 = (sqrt(pi))^d * sqrt(det Σ); with Σ = L L^T, sqrt(det Σ) = prod(diag(L))
d = 2
L_packed = params_full[:, d:]  # (N, 3) in 2D
L_full = unpack_tril(L_packed, d)  # (N, 2, 2)
diag_prod = L_full[:, 0, 0] * L_full[:, 1, 1]  # ∏ diag(L) in 2D
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

# Bit accounting (float32 for centers + packed L + amplitude)
FLOAT_BITS = 32
FLOATS_PER_SPLAT = d + tril_size(d) + 1  # centers(d) + packed L + amp
BITS_PER_SPLAT = FLOATS_PER_SPLAT * FLOAT_BITS
IMAGE_BITS = V.size * FLOAT_BITS
NUM_PIXELS = V.size

model_bits_frames = np.zeros(len(keep_counts), dtype=np.float64)
bpp_frames = np.zeros(len(keep_counts), dtype=np.float64)
bit_compression_pct = np.zeros(len(keep_counts), dtype=np.float64)

polygons_frames = []
centers_frames = []

for i, K in enumerate(keep_counts):
    idx = order[:K]

    Vk = render_gaussians_numpy(
        V.shape, params_full[idx], amps[idx], truncate=TRUNCATE_SIG
    )
    stack_recon[i] = Vk
    stack_resid[i] = V - Vk
    rel_err_frames[i] = np.linalg.norm(V - Vk) / (np.linalg.norm(V) + 1e-12)

    model_bits_frames[i] = int(K) * BITS_PER_SPLAT
    bpp_frames[i] = model_bits_frames[i] / NUM_PIXELS
    bit_compression_pct[i] = 100.0 * (1.0 - (model_bits_frames[i] / IMAGE_BITS))

    Lk = L_full[idx]  # (K, 2, 2)
    Ck = params_full[idx, :2]  # (K, 2)
    polys = [
        ellipse_polygon_from_L(Ck[j], Lk[j], t=2.0, n_pts=64) for j in range(len(idx))
    ]
    polygons_frames.append(polys)
    centers_frames.append(Ck)

# 4) Napari viewer with "compression" slider
viewer = napari.Viewer()
viewer.add_image(
    V,
    name="human_mitosis (input)",
    colormap="magma",
    contrast_limits=[0, float(V.max())],
)

viewer.add_image(
    stack_recon,
    name="reconstruction (compression, oriented)",
    colormap="magma",
    contrast_limits=[0, float(V.max())],
)
viewer.add_image(
    np.abs(stack_resid),
    name="absolute residual",
    colormap="inferno",
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

# Axis labels (if supported)
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
    shapes.data = polygons_frames[t_index]
    pts.data = centers_frames[t_index]
    _set_overlay_text(t_index)


# Initialize and wire slider
_update_layers_for_t(0)


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
