#!/usr/bin/env python3
"""
Human mitosis image Gaussian splatting demo with EXPLICIT seed initialization.

This demo showcases explicit seeding where seeds are generated with
`seed_from_decomposition()`, `seed_from_grid()`, or `seed_from_edges()`,
which return GSplatData with scale-informed Gaussian shapes. The seeds are then
passed to `fit_gaussian_splats()` for optimization.

Key difference from demo_splats_mitosis.py:
- Seeds are generated explicitly before fitting
- Shows how scale information flows from seeding to fitting
- Demonstrates direct control over seed generation parameters
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
from luxar.gsplats.seeds import (
    generate_seeds,
    seed_from_decomposition,
    seed_from_edges,
    seed_from_grid,
)
from luxar.gsplats.utils.trils import tril_size, unpack_tril

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("Human Mitosis Demo with Explicit Seeding (napari disabled)")
    aprint("Running all computations without napari visualization...")

# ======= Demo knobs =======
N_ITERS = 2000  # Number of optimization iterations
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps:0"
N_FRAMES = 40  # number of compression steps (<= #splats)
TRUNCATE_SIG = 3.0  # rendering support truncation (approx +-3 sigma)
SEED_METHOD = "decomposition"  # "grid", "decomposition", "edges", or "auto"
# ==========================

# Setup Arbol
Arbol.max_depth = 4


with asection("Human Mitosis Demo with Explicit Seeding"):
    aprint("Demonstrating the new explicit seeding API")

    with asection("Loading and preprocessing data"):
        # Load human_mitosis and prepare a soft grayscale target
        img = data.human_mitosis()  # likely RGB
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = color.rgb2gray(img) * 100  # -> float in [0, 100]
        V = img_as_float32(img)

        # Crop the image to a smaller region for faster demo
        V = V[100:356, 100:356]  # Crop to 256x256

        aprint(f"Preprocessed human mitosis image: {V.shape}")
        aprint(f"Data range: [{V.min():.4f}, {V.max():.4f}]")

    # ========== Explicit seed generation ==========
    with asection(f"Generating seeds using '{SEED_METHOD}' method"):
        # Seeding returns GSplatData with scale-informed shapes.
        if SEED_METHOD == "grid":
            # Uniform grid seeding for baseline coverage
            seeds = seed_from_grid(
                V,
                spacing=8.0,
                sigma=4.0,
            )
            aprint("Used seed_from_grid() - uniform spatial coverage")

        elif SEED_METHOD == "decomposition":
            # Scale-hierarchical decomposition (most principled)
            seeds = seed_from_decomposition(
                V,
                scales=[1, 2, 4, 8, 16],
                ignore_finest_k=1,  # Skip finest scale (noise)
                threshold_rel=0.1,  # Relative threshold (0-1)
                min_distance=2.0,
            )
            aprint("Used seed_from_decomposition() - principled scale separation")

        elif SEED_METHOD == "edges":
            # Edge-based seeding with anisotropic shapes
            seeds = seed_from_edges(
                V,
                min_distance=2.0,
                edge_threshold_rel=0.1,
                structure_radius=3.0,
            )
            aprint("Used seed_from_edges() - anisotropic edge detection")

        elif SEED_METHOD == "auto":
            # Combined approach (decomposition + edges + grid)
            seeds = generate_seeds(
                V,
                method="auto",
                min_distance=2.0,
            )
            aprint("Used generate_seeds(method='auto') - principled combination")

        else:
            raise ValueError(f"Unknown seed method: {SEED_METHOD}")

        # The seeds object is GSplatData with scale-informed parameters!
        aprint(f"Generated {len(seeds.centers)} seeds")
        aprint(f"  centers shape: {seeds.centers.shape}")
        aprint(f"  cholesky_factors shape: {seeds.cholesky_factors.shape}")
        aprint(
            f"  amplitudes range: [{seeds.amplitudes.min():.4f}, {seeds.amplitudes.max():.4f}]"
        )

        # Show scale distribution from Cholesky factors
        L_seeds = unpack_tril(seeds.cholesky_factors, 2)  # (N, 2, 2)
        sigmas = np.sqrt(L_seeds[:, 0, 0] ** 2 + L_seeds[:, 1, 1] ** 2) / np.sqrt(2)
        aprint(
            f"  sigma range: [{sigmas.min():.2f}, {sigmas.max():.2f}] (from scale info)"
        )

    # Configure dynamic operations
    dynamic_config = DynamicOpsConfig()
    aprint(f"Dynamic operations enabled (step_every={dynamic_config.step_every})")

    with asection(f"Fitting Gaussian splats ({N_ITERS} iterations)"):
        # Pass the GSplatData seeds directly - the fitter will use
        # the scale-informed cholesky_factors for initialization!
        result = fit_gaussian_splats(
            V,
            seeds=seeds,  # <-- Pass GSplatData directly!
            n_iters=N_ITERS,
            truncate=TRUNCATE_SIG,
            device=DEVICE,
            verbose=True,
            lr_reduction_factor=0.95,
            # Dynamic operations
            enable_dynamic_ops=False,
            dynamic_config=dynamic_config,
            napari_movie=(not NO_NAPARI),
            movie_every=1,
            movie_max_frames=None,
        )

        if len(result.amplitudes) == 0:
            raise RuntimeError(
                "No splats were fitted; try lowering thresholds or increasing iterations."
            )

        aprint(
            f"Final splat count: {len(result.amplitudes)} (started with {len(seeds.centers)} seeds)"
        )

# ----- Compression ranking by approximate L2 energy -----
# ||G||_2^2 = (sqrt(pi))^d * sqrt(det Sigma); with Sigma = L L^T, sqrt(det Sigma) = prod(diag(L))
d = 2

# Extract L for energy ranking
L_full = unpack_tril(result.cholesky_factors, d)  # (N, 2, 2)
diag_prod = L_full[:, 0, 0] * L_full[:, 1, 1]  # prod diag(L) in 2D
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
        name="oriented 2 sigma ellipses (kept)",
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
            f"Kept splats: {K}/{N}  |  Model bits: {bits_model:,}  "
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

    aprint(
        "Ready. Use the top slider (axis 0) to move from keeping all splats toward keeping just one."
    )
    napari.run()
else:
    aprint("\nDemo completed successfully (napari visualization disabled)")
