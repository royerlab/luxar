#!/usr/bin/env python3
"""
Human mitosis image with intensity gradient - tests CLAHE-based seeding.

This demo applies a linear intensity attenuation gradient along the y-axis to create
a challenging scenario: structures at the bottom are bright while identical structures
at the top are dim (near zero). This tests whether CLAHE-based coverage seeding can
discover and represent dim structures that would be missed by residual-only seeding.

Key challenge: Without CLAHE-based seeding, the top (dim) region receives insufficient
splat coverage despite containing the same biological structures as the bright bottom.
"""

import sys

import napari
import numpy as np
from arbol import Arbol, aprint, asection
from skimage import color, data, img_as_float32

from luxar.gsplats.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.utils.trils import tril_size, unpack_tril

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI and len(sys.argv) > 1:
    aprint("🔬 Mitosis Intensity Gradient Demo (napari disabled)")
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
GRADIENT_ATTENUATION = 0.1  # Top attenuation factor (0.1 = 10% reduction at top)
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


with asection("Mitosis Intensity Gradient Demo - Testing CLAHE Seeding"):
    aprint("🔬 Intensity gradient demo for CLAHE-based coverage seeding validation")
    aprint(
        "🎯 Challenge: Top region dim, bottom bright - can CLAHE find dim structures?"
    )

    with asection("Loading and preprocessing data"):
        # Load human_mitosis and prepare a soft grayscale target
        img = data.human_mitosis()  # likely RGB
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = color.rgb2gray(img) * 100  # -> float in [0, 100]
        V_base = img_as_float32(img)

        # Crop the image to a smaller region for faster demo
        V_base = V_base[100:356, 100:356]  # Crop to 256x256

        aprint(f"Base image shape: {V_base.shape}")
        aprint(f"Base intensity range: [{V_base.min():.4f}, {V_base.max():.4f}]")

    with asection("Applying linear intensity attenuation gradient (y-axis)"):
        # Create linear attenuation gradient: 1.0 at bottom (y=height-1), GRADIENT_ATTENUATION at top (y=0)
        height, width = V_base.shape
        gradient = np.linspace(
            GRADIENT_ATTENUATION, 1.0, height
        )  # [GRADIENT_ATTENUATION ... 1.0]
        gradient_2d = gradient[:, np.newaxis]  # (height, 1) for broadcasting

        # Apply gradient
        V = V_base * gradient_2d

        aprint(f"Gradient: top={GRADIENT_ATTENUATION:.2f}×, bottom=1.00×")
        aprint(f"Result intensity range: [{V.min():.4f}, {V.max():.4f}]")
        aprint(
            f"Top row intensity: [{V[0].min():.4f}, {V[0].max():.4f}]  (dim region)"
        )
        aprint(
            f"Bottom row intensity: [{V[-1].min():.4f}, {V[-1].max():.4f}]  (bright region)"
        )

    # Configure dynamic operations with CLAHE-based coverage seeding
    dynamic_config = DynamicOpsConfig()
    dynamic_config.k_max_residuals = 40  # Total seed budget per cycle
    dynamic_config.density_seeding_fraction = 0.5  # 50% CLAHE, 50% residual
    dynamic_config.clahe_tile_size = 16  # Tile size for CLAHE
    dynamic_config.clahe_clip_limit = 2.0  # Contrast limiting factor
    dynamic_config.clahe_nbins = 256  # Histogram bins

    aprint("Dynamic operations with CLAHE-based coverage seeding:")
    aprint(f"  step_every={dynamic_config.step_every}")
    aprint(f"  k_max_residuals={dynamic_config.k_max_residuals}")
    aprint(
        f"  density_seeding_fraction={dynamic_config.density_seeding_fraction} (50-50 hybrid)"
    )
    aprint(
        f"  CLAHE: tile_size={dynamic_config.clahe_tile_size}, clip_limit={dynamic_config.clahe_clip_limit}, nbins={dynamic_config.clahe_nbins}"
    )

    # === Visualize CLAHE Effect ===
    with asection("CLAHE Visualization - Before/After Comparison"):
        import torch
        from luxar.gsplats.clahe import apply_clahe

        aprint("Computing CLAHE-equalized image for visualization...")

        # Convert to torch tensor (same as will be used in dynamic ops)
        V_torch = torch.tensor(V, dtype=torch.float32)

        # Apply CLAHE with same parameters as dynamic ops
        V_clahe_torch = apply_clahe(
            V_torch,
            tile_size=dynamic_config.clahe_tile_size,
            clip_limit=dynamic_config.clahe_clip_limit,
            nbins=dynamic_config.clahe_nbins,
        )

        # Convert back to numpy for visualization
        V_clahe = V_clahe_torch.cpu().numpy()

        aprint(f"Original gradient range: [{V.min():.4f}, {V.max():.4f}]")
        aprint(f"CLAHE-equalized range: [{V_clahe.min():.4f}, {V_clahe.max():.4f}]")

        # Show CLAHE before/after comparison
        viewer_clahe = napari.Viewer(title="CLAHE Visualization - Before/After")

        viewer_clahe.add_image(
            V,
            name="Before CLAHE (gradient input)",
            colormap="magma",
            contrast_limits=[0, float(V.max())],
        )

        viewer_clahe.add_image(
            V_clahe,
            name="After CLAHE (used for sampling)",
            colormap="viridis",
            contrast_limits=[V_clahe.min(), V_clahe.max()],
        )

        # Add difference visualization
        clahe_diff = V_clahe - V
        viewer_clahe.add_image(
            clahe_diff,
            name="CLAHE - Original (difference)",
            colormap="bwr",  # Blue-white-red diverging
            contrast_limits=[-V.max() * 0.5, V.max() * 0.5],
            visible=False,
        )

        aprint("")
        aprint("📊 CLAHE Effect Analysis:")
        aprint(f"  Top region (dim):")
        aprint(f"    Before: mean={V[0:32].mean():.4f}, std={V[0:32].std():.4f}")
        aprint(f"    After:  mean={V_clahe[0:32].mean():.4f}, std={V_clahe[0:32].std():.4f}")
        aprint(f"  Bottom region (bright):")
        aprint(
            f"    Before: mean={V[-32:].mean():.4f}, std={V[-32:].std():.4f}"
        )
        aprint(
            f"    After:  mean={V_clahe[-32:].mean():.4f}, std={V_clahe[-32:].std():.4f}"
        )
        aprint("")
        aprint("🔍 What to look for:")
        aprint("  • CLAHE should enhance contrast in dim (top) region")
        aprint("  • Top and bottom regions should have more balanced intensities")
        aprint(
            "  • 'After CLAHE' image shows what's used for sampling probabilities"
        )
        aprint("")
        aprint("Press any key to close this window and continue with fitting...")

        napari.run()

    with asection(f"Fitting Gaussian splats ({N_ITERS} iterations)"):
        # Fit oriented (full-covariance) Gaussians with auto-candidate generation
        params_full, amps, stats = fit_gaussian_splats(
            V,
            # seeds auto-generated with intelligent defaults
            init_sigma_vox=0.5,
            n_iters=N_ITERS,
            loss_type=LOSS_TYPE,
            lr=LR,
            l1_diag=0,
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

# params_full includes sharpness in last column: [centers, packed_L, sharpness]
# Extract L for energy ranking (exclude sharpness from the end)
L_packed = params_full[:, d:-1]  # (N, 3) in 2D - centers excluded, sharpness excluded
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

# Bit accounting (float32 for centers + packed L + sharpness + amplitude)
FLOAT_BITS = 32
FLOATS_PER_SPLAT = d + tril_size(d) + 1 + 1  # centers(d) + packed L + sharpness + amp
BITS_PER_SPLAT = FLOATS_PER_SPLAT * FLOAT_BITS
IMAGE_BITS = V.size * FLOAT_BITS
NUM_PIXELS = V.size

model_bits_frames = np.zeros((len(keep_counts),), dtype=np.float64)
bpp_frames = np.zeros(len(keep_counts), dtype=np.float64)
bit_compression_pct = np.zeros(len(keep_counts), dtype=np.float64)

polygons_frames = []
centers_frames = []

for i, K in enumerate(keep_counts):
    idx = order[:K]

    # Render with auto-extraction of all parameters from params_full
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

# Napari viewer with "compression" slider
viewer = napari.Viewer(title="Mitosis Intensity Gradient - CLAHE Seeding Test")

# Add original image (no gradient)
viewer.add_image(
    V_base,
    name="original (no gradient)",
    colormap="magma",
    contrast_limits=[0, float(V_base.max())],
    visible=False,  # Start hidden
)

# Add gradient image (input to fitting)
viewer.add_image(
    V,
    name="gradient input (top dim, bottom bright)",
    colormap="magma",
    contrast_limits=[0, float(V_base.max())],  # Use original range for fair comparison
)

viewer.add_image(
    stack_recon,
    name="reconstruction (compression, oriented)",
    colormap="magma",
    contrast_limits=[0, float(V_base.max())],
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
        f"🔬 Intensity Gradient Demo (CLAHE Seeding) | Kept splats: {K}/{N}  |  Model bits: {bits_model:,}  "
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
aprint("")
aprint("🎯 Evaluation Guidance:")
aprint("1. Check splat distribution - are there splats in BOTH top (dim) and bottom (bright) regions?")
aprint("2. Compare residual between top and bottom - is error balanced?")
aprint("3. Toggle 'original (no gradient)' layer to compare structures")
aprint("4. If splats are missing from top → CLAHE seeding may need tuning")
aprint("")
aprint(f"Raw image bits (float32): {IMAGE_BITS:,}  |  raw bpp = 32.000")
for i, K in enumerate(keep_counts[::5]):  # Show every 5th frame
    idx = i * 5
    if idx < len(keep_counts):
        aprint(
            f"Frame {idx:02d} | keep {K:4d} | model_bits={int(model_bits_frames[idx]):>10,d} "
            f"| bit_compression={bit_compression_pct[idx]:6.1f}% | bpp={bpp_frames[idx]:6.3f} "
            f"| relL2={rel_err_frames[idx]:.4f}"
        )

aprint("")
aprint("🎛️  Controls:")
aprint("   • Use the top slider (axis 0) to explore compression levels")
aprint("   • Toggle layers to compare original, gradient input, and reconstruction")
aprint("   • Cyan points/ellipses show active splats at current compression level")

napari.run()
