#!/usr/bin/env python3
"""
Astronaut image Gaussian splatting demo with interactive compression analysis.

This demo applies Gaussian splatting to the classic scikit-image astronaut photo,
demonstrating full-covariance fitting with compression analysis via napari.
Features interactive slider to explore reconstruction quality vs compression ratio
on a complex color photograph with rich textures and details.
"""

import sys

import napari
import numpy as np
from arbol import Arbol, aprint, asection
from skimage import color, data, img_as_float32

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
N_ITERS = 2000  # Same as mitosis
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps:0"
N_FRAMES = 50  # number of compression steps (<= #splats)
TRUNCATE_SIG = 3.0  # rendering support truncation (≈ ±3σ)
# ==========================

# Setup Arbol
Arbol.max_depth = 3


with asection("Astronaut Gaussian Splatting Demo"):
    aprint("🚀 Interactive compression analysis on classic astronaut photograph")
    aprint("📸 Complex color image with rich textures, faces, and spatial detail")

    with asection("Loading and preprocessing data"):
        # Load astronaut image and convert to grayscale for splatting
        img = data.astronaut()  # RGB (512, 512, 3)
        aprint(
            f"Original astronaut image: {img.shape}, range=[{img.min()}, {img.max()}]"
        )

        # Convert to grayscale using luminance weights for better feature preservation
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = color.rgb2gray(img)  # -> float in [0, 1]
        V = img_as_float32(img) * 100.0  # Scale to [0, 100] (same as mitosis)

        # Crop to focus on the astronaut's face and upper body for faster demo
        V = V[80:400, 120:440]  # Crop to 320x320 region with main subject

        aprint(f"Preprocessed astronaut image: {V.shape}")
        aprint(f"Data range: [{V.min():.4f}, {V.max():.4f}]")
        aprint("🎯 Focused on astronaut face and helmet for detailed reconstruction")

    with asection(f"Fitting Gaussian splats ({N_ITERS} iterations)"):
        # Fit oriented (full-covariance) Gaussians with auto-candidate generation
        result = fit_gaussian_splats(
            V,
            seeds=4000,
            n_iters=N_ITERS,
            truncate=TRUNCATE_SIG,
            device=DEVICE,
            verbose=True,
            max_abs_error=0.1,  # Same as mitosis
            napari_movie=(not NO_NAPARI),
            movie_every=1,  # Same as mitosis
            movie_max_frames=None,  # Same as mitosis
        )
        if len(result.amplitudes) == 0:
            raise RuntimeError(
                "No splats were fitted; try lowering thresholds or increasing iterations."
            )

        aprint(
            f"🎉 Fitted {len(result.amplitudes)} splats to reconstruct astronaut photograph"
        )

# ----- Compression ranking by approximate L2 energy -----
# ||G||_2^2 = (sqrt(pi))^d * sqrt(det Σ); with Σ = L L^T, sqrt(det Σ) = prod(diag(L))
d = 2

# Extract L for energy ranking
L_full = unpack_tril(result.cholesky_factors, d)  # (N, 2, 2)
diag_prod = L_full[:, 0, 0] * L_full[:, 1, 1]  # ∏ diag(L) in 2D
energy_score = (result.amplitudes**2) * (np.sqrt(np.pi) ** d) * diag_prod
order = np.argsort(-energy_score)  # descending

aprint(f"📊 Ranking {len(result.amplitudes)} splats by L2 energy contribution")

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

model_bits_frames = np.zeros(len(keep_counts), dtype=np.float64)
bpp_frames = np.zeros(len(keep_counts), dtype=np.float64)
bit_compression_pct = np.zeros(len(keep_counts), dtype=np.float64)

polygons_frames = []
centers_frames = []

aprint("🎬 Precomputing compression frames...")
with asection("Computing reconstruction quality at different compression levels"):
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
        Vk = render_gaussians_numpy(V.shape, result_idx, truncate=TRUNCATE_SIG)
        stack_recon[i] = Vk
        stack_resid[i] = V - Vk
        rel_err_frames[i] = np.linalg.norm(V - Vk) / (np.linalg.norm(V) + 1e-12)

        model_bits_frames[i] = int(K) * BITS_PER_SPLAT
        bpp_frames[i] = model_bits_frames[i] / NUM_PIXELS
        bit_compression_pct[i] = 100.0 * (1.0 - (model_bits_frames[i] / IMAGE_BITS))

        Lk = L_full[idx]  # (K, 2, 2)
        Ck = result.centers[idx]  # (K, 2)
        polys = [
            ellipse_polygon_from_L(Ck[j], Lk[j], t=2.0, n_pts=64)
            for j in range(len(idx))
        ]
        polygons_frames.append(polys)
        centers_frames.append(Ck)

        if (i + 1) % 10 == 0:
            aprint(
                f"Frame {i + 1:02d}/{len(keep_counts)}: {K} splats, {bit_compression_pct[i]:.1f}% compression"
            )

# Console summary
aprint("📈 Compression Analysis Results:")
aprint(f"Raw image bits (float32): {IMAGE_BITS:,}  |  raw bpp = 32.000")
for i, K in enumerate(keep_counts[::5]):  # Show every 5th frame to avoid spam
    idx = i * 5
    if idx < len(keep_counts):
        aprint(
            f"Frame {idx:02d} | keep {K:4d} | model_bits={int(model_bits_frames[idx]):>10,d} "
            f"| compression={bit_compression_pct[idx]:6.1f}% | bpp={bpp_frames[idx]:6.3f} "
            f"| relL2={rel_err_frames[idx]:.4f}"
        )

if not NO_NAPARI:
    # Napari viewer with "compression" slider
    aprint("🔬 Launching interactive napari viewer...")
    viewer = napari.Viewer(title="Astronaut Gaussian Splatting Demo")

    # Add original image
    viewer.add_image(
        V,
        name="astronaut (input)",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
    )

    # Add reconstruction stack
    viewer.add_image(
        stack_recon,
        name="reconstruction (compression, oriented)",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
    )

    # Add residual stack
    viewer.add_image(
        np.abs(stack_resid),
        name="absolute residual",
        colormap="hot",
        contrast_limits=[0, max(1e-12, float(np.abs(stack_resid).max()))],
    )

    # Shapes & points that update with slider
    shapes = viewer.add_shapes(
        name="oriented 2σ ellipses (kept)",
        shape_type="polygon",
        edge_color="lime",
        edge_width=1.2,
        face_color=[0, 0, 0, 0],
    )
    pts = viewer.add_points(
        np.zeros((0, 2)),
        name="centers (kept)",
        size=2.5,
        border_color="lime",
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
            f"🚀 Astronaut Demo | Kept splats: {K}/{N}  |  Model bits: {bits_model:,}  "
            f"|  bpp: {bpp:.3f} (raw=32.0)  |  Compression: {pct_bits:.1f}%  "
            f"|  rel L2 error: {rel:.4f}"
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

    aprint("🎛️  Controls:")
    aprint("   • Use the top slider (axis 0) to explore compression levels")
    aprint(
        "   • Move from keeping all splats toward keeping just the most important ones"
    )
    aprint("   • Watch how the reconstruction degrades with fewer splats")
    aprint("   • Observe the ellipse overlays showing splat orientations and sizes")
    aprint("")
    aprint("🔍 What to notice:")
    aprint("   • How facial features are preserved at different compression levels")
    aprint("   • The trade-off between file size and reconstruction quality")
    aprint("   • How oriented ellipses capture image structure efficiently")
    aprint("   • The role of L1 regularization in creating cleaner splat layouts")

    napari.run()
