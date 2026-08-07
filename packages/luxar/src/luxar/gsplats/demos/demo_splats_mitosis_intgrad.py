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

from luxar.gsplats.demos._demo_common import ellipse_polygon_from_L
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.utils.trils import tril_size, unpack_tril

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("🔬 Mitosis Intensity Gradient Demo (napari disabled)")
    aprint("Running all computations without napari visualization...")

# ======= Demo knobs =======
N_ITERS = 2000  # Number of optimization iterations
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps:0"
N_FRAMES = 40  # number of compression steps (<= #splats)
GRADIENT_ATTENUATION = 0.1  # Top attenuation factor (0.1 = 10% reduction at top)
# ==========================

# Setup Arbol
Arbol.max_depth = 3


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
        aprint(f"Top row intensity: [{V[0].min():.4f}, {V[0].max():.4f}]  (dim region)")
        aprint(
            f"Bottom row intensity: [{V[-1].min():.4f}, {V[-1].max():.4f}]  (bright region)"
        )

    # Configure dynamic operations (residual-based seeding only)
    dynamic_config = DynamicOpsConfig()

    aprint("Dynamic operations enabled (residual-based seeding):")
    aprint(f"  step_every={dynamic_config.step_every}")
    aprint(f"  k_max_residuals={dynamic_config.k_max_residuals}")
    aprint(f"  nms_radius_vox={dynamic_config.nms_radius_vox}")

    # === Visualize CLAHE Effect ===
    with asection("CLAHE Visualization - Initial Seed Detection"):
        import torch

        from luxar.gsplats.clahe import apply_clahe

        aprint("Computing CLAHE-equalized image for visualization...")
        aprint("(CLAHE is used for initial candidate detection, not dynamic seeding)")

        # Convert to torch tensor
        V_torch = torch.tensor(V, dtype=torch.float32)

        # Apply CLAHE with same parameters as used in candidates.py
        V_clahe_torch = apply_clahe(
            V_torch,
            tile_size=32,  # Same as candidates.py default
            clip_limit=16.0,  # Same as candidates.py default
            nbins=256,
        )

        # Convert back to numpy for visualization
        V_clahe = V_clahe_torch.cpu().numpy()

        aprint(f"Original gradient range: [{V.min():.4f}, {V.max():.4f}]")
        aprint(f"CLAHE-equalized range: [{V_clahe.min():.4f}, {V_clahe.max():.4f}]")

        aprint("")
        aprint("📊 CLAHE Effect Analysis:")
        aprint("  Top region (dim):")
        aprint(f"    Before: mean={V[0:32].mean():.4f}, std={V[0:32].std():.4f}")
        aprint(
            f"    After:  mean={V_clahe[0:32].mean():.4f}, std={V_clahe[0:32].std():.4f}"
        )
        aprint("  Bottom region (bright):")
        aprint(f"    Before: mean={V[-32:].mean():.4f}, std={V[-32:].std():.4f}")
        aprint(
            f"    After:  mean={V_clahe[-32:].mean():.4f}, std={V_clahe[-32:].std():.4f}"
        )
        aprint("")
        aprint("🔍 CLAHE Effect:")
        aprint("  • CLAHE enhances contrast in dim (top) region")
        aprint("  • Top and bottom regions have more balanced intensities")
        aprint("  • Used for initial peak detection to find dim structures")

        if not NO_NAPARI:
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

            aprint("Press any key to close this window and continue with fitting...")
            napari.run()

    with asection(f"Fitting Gaussian splats ({N_ITERS} iterations)"):
        # Fit oriented (full-covariance) Gaussians with auto-candidate generation
        result = fit_gaussian_splats(
            V,
            seeds=0.05,
            n_iters=N_ITERS,
            max_abs_error=0.05,
            device=DEVICE,
            verbose=True,
            napari_movie=(not NO_NAPARI),
            movie_every=1,
            movie_max_frames=None,
        )

        if len(result.amplitudes) == 0:
            raise RuntimeError(
                "No splats were fitted; try lowering thresholds or increasing iterations."
            )

# ----- Compression ranking by approximate L2 energy -----
# ||G||_2^2 = (sqrt(pi))^d * sqrt(det Σ); with Σ = L L^T, sqrt(det Σ) = prod(diag(L))
d = 2

# Extract L for energy ranking
L_full = unpack_tril(result.cholesky_factors, d)  # (N, 2, 2)
diag_prod = L_full[:, 0, 0] * L_full[:, 1, 1]  # ∏ diag(L) in 2D
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

# Bit accounting (float32 for centers + packed L + amplitude)
FLOAT_BITS = 32
FLOATS_PER_SPLAT = d + tril_size(d) + 1  # centers(d) + packed L + amp
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

    # Create sliced result for rendering
    result_idx = GSplatData(
        centers=result.centers[idx],
        amplitudes=result.amplitudes[idx],
        cholesky_factors=result.cholesky_factors[idx],
        stats={},  # Empty stats for rendering subset
    )

    # Render
    Vk = render_gaussians_numpy(V.shape, result_idx)
    stack_recon[i] = Vk
    stack_resid[i] = V - Vk
    rel_err_frames[i] = np.linalg.norm(V - Vk) / (np.linalg.norm(V) + 1e-12)

    model_bits_frames[i] = int(K) * BITS_PER_SPLAT
    bpp_frames[i] = model_bits_frames[i] / NUM_PIXELS
    bit_compression_pct[i] = 100.0 * (1.0 - (model_bits_frames[i] / IMAGE_BITS))

    Lk = L_full[idx]  # (K, 2, 2)
    Ck = result.centers[idx]  # (K, 2)
    polys = [
        ellipse_polygon_from_L(Ck[j], Lk[j], t=2.0, n_pts=64) for j in range(len(idx))
    ]
    polygons_frames.append(polys)
    centers_frames.append(Ck)

# Console summary
aprint("")
aprint("🎯 Evaluation Results:")
aprint(
    "1. Splat distribution - check if there are splats in BOTH top (dim) and bottom (bright) regions"
)
aprint("2. Residual balance - error should be balanced between top and bottom")
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

if not NO_NAPARI:
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
        contrast_limits=[
            0,
            float(V_base.max()),
        ],  # Use original range for fair comparison
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

    def _set_overlay_text(t_index: int) -> None:
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

    def _update_layers_for_t(t_index: int) -> None:
        shapes.data = polygons_frames[t_index]
        pts.data = centers_frames[t_index]
        _set_overlay_text(t_index)

    # Initialize and wire slider
    _update_layers_for_t(0)

    def _on_step_change(event=None) -> None:
        t = viewer.dims.current_step[0]
        _update_layers_for_t(int(t))

    viewer.dims.events.current_step.connect(_on_step_change)

    aprint("")
    aprint("🎛️  Controls:")
    aprint("   • Use the top slider (axis 0) to explore compression levels")
    aprint("   • Toggle layers to compare original, gradient input, and reconstruction")
    aprint("   • Cyan points/ellipses show active splats at current compression level")

    napari.run()
else:
    aprint("\n✅ Demo completed successfully (napari visualization disabled)")
