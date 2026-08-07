#!/usr/bin/env python3
"""
Boundary Containment Demo - Constraining Splats Within Volume Bounds

**What this demo demonstrates:**
- The ``boundary_penalty`` parameter: a differentiable loss term that discourages
  splats from extending beyond the volume during optimization
- The ``clip_to_bounds`` parameter: a post-processing step that hard-clips splats
  to guarantee containment
- Side-by-side comparison of unconstrained vs constrained fitting
- Visualizing how edge splats behave with and without boundary containment

**Key concepts:**
- Gaussian splats have spatial extent beyond their center. A splat near the edge
  can have its bell curve extend outside the volume bounds.
- During fitting, the renderer clips out-of-bounds regions via AABB clamping,
  so the optimizer never "sees" the overflow — it effectively learns half-Gaussians
  that exploit edge truncation.
- ``boundary_penalty`` adds a smooth penalty: weight * mean(relu(radius - dist_to_edge)^2)
  This gently pushes edge splats inward during optimization.
- ``clip_to_bounds`` scales down Cholesky rows post-fit so that
  truncate * sqrt(Sigma_ii) <= distance_to_nearest_edge for every splat.

**Data source:** Nuclei channel of cells3d from scikit-image (single 2D slice)
**Visualization:** napari viewer comparing unconstrained, penalty-only, clip-only,
                   and combined approaches with oriented ellipse overlays

**Command-line:**
    python demo_boundary_containment.py [--no-napari]

**Related demos:**
- demo_basic_fitting.py - Simple introduction to the fitting API
- demo_2d_synthetic_blobs.py - Compression analysis with ellipse overlays
"""

import sys

import numpy as np
from arbol import Arbol, aprint, asection

from luxar.gsplats.demos._demo_common import ellipse_polygon_from_L
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.utils.trils import unpack_tril

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv

# ======= Demo knobs =======
N_ITERS = 2000
N_SEEDS = 300
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps"
TRUNCATE = 3.0
# ===========================

# Setup
Arbol.max_depth = 4


# =============================================================================
# Helpers
# =============================================================================


def count_out_of_bounds(result: GSplatData, shape, truncate: float = 3.0):
    """Count splats whose effective support extends beyond volume bounds.

    Returns (n_oob, total, max_overflow) where:
    - n_oob: number of splats with at least one dimension out of bounds
    - total: total number of splats
    - max_overflow: maximum overflow distance in any dimension (voxels)
    """
    d = result.centers.shape[1]
    L_full = unpack_tril(result.cholesky_factors, d)
    sigma_diag = np.sum(L_full * L_full, axis=2)  # (N, d)
    radii = truncate * np.sqrt(sigma_diag)  # (N, d)

    shape_arr = np.array(shape, dtype=np.float32)
    overflow_lo = np.maximum(radii - result.centers, 0)
    overflow_hi = np.maximum(radii - (shape_arr - 1.0 - result.centers), 0)
    overflow = np.maximum(overflow_lo, overflow_hi)

    oob_mask = np.any(overflow > 0.01, axis=1)  # tolerance for numerical noise
    n_oob = int(oob_mask.sum())
    max_overflow = float(overflow.max()) if overflow.size > 0 else 0.0

    return n_oob, len(result.amplitudes), max_overflow


def make_ellipse_polygons(result: GSplatData, t: float = 3.0):
    """Build oriented ellipse polygons at t-sigma contour for all splats."""
    d = result.centers.shape[1]
    L_full = unpack_tril(result.cholesky_factors, d)
    polys = []
    for k in range(len(result.amplitudes)):
        polys.append(ellipse_polygon_from_L(result.centers[k], L_full[k], t=t))
    return polys


# =============================================================================
# Data Loading
# =============================================================================


def load_slice():
    """Load a single 2D slice from the cells3d nuclei channel.

    Returns a float32 2D image normalized to [0, 1].
    """
    with asection("Loading cells3d slice"):
        try:
            from skimage.data import cells3d
        except ImportError:
            raise ImportError(
                "scikit-image is required for this demo.\n"
                "Install with: pip install scikit-image"
            )

        raw = cells3d()  # (60, 2, 256, 256) — (Z, C, Y, X)
        # Take nuclei channel (index 1), middle Z slice
        z_mid = raw.shape[0] // 2
        V = raw[z_mid, 1, :, :].astype(np.float32)
        V = (V - V.min()) / (V.max() - V.min() + 1e-8)
        aprint(f"Slice shape: {V.shape}, Z={z_mid}, channel=nuclei")
        return V


# =============================================================================
# Fitting Variants
# =============================================================================


def fit_variants(V):
    """Fit four variants: unconstrained, penalty-only, clip-only, combined.

    Returns dict mapping variant name to GSplatData.
    """
    common_kwargs = dict(
        seeds=N_SEEDS,
        n_iters=N_ITERS,
        truncate=TRUNCATE,
        device=DEVICE,
        verbose=True,
    )

    results = {}

    with asection("Variant 1: Unconstrained (baseline)"):
        results["unconstrained"] = fit_gaussian_splats(
            V,
            **common_kwargs,
        )

    with asection("Variant 2: Boundary penalty only (soft constraint)"):
        results["penalty"] = fit_gaussian_splats(
            V,
            boundary_penalty=1.0,
            **common_kwargs,
        )

    with asection("Variant 3: Clip-to-bounds only (hard post-processing)"):
        results["clip"] = fit_gaussian_splats(
            V,
            clip_to_bounds=True,
            **common_kwargs,
        )

    with asection("Variant 4: Combined (penalty + clip)"):
        results["combined"] = fit_gaussian_splats(
            V,
            boundary_penalty=1.0,
            clip_to_bounds=True,
            **common_kwargs,
        )

    return results


# =============================================================================
# Analysis
# =============================================================================


def analyze_results(results, V):
    """Print boundary containment analysis for each variant."""
    shape = V.shape

    with asection("Boundary Containment Analysis"):
        aprint(f"Volume shape: {shape}")
        aprint(f"Truncation radius: {TRUNCATE} sigma")
        aprint("")
        aprint(
            f"{'Variant':<25s} {'Splats':>7s} {'OOB':>5s} "
            f"{'OOB%':>7s} {'MaxOverflow':>12s} {'RelErr':>8s}"
        )
        aprint("-" * 70)

        for name, result in results.items():
            n_oob, total, max_overflow = count_out_of_bounds(result, shape, TRUNCATE)
            recon = render_gaussians_numpy(shape, result, truncate=TRUNCATE)
            rel_err = float(np.linalg.norm(V - recon) / (np.linalg.norm(V) + 1e-12))
            oob_pct = 100.0 * n_oob / max(total, 1)

            aprint(
                f"{name:<25s} {total:>7d} {n_oob:>5d} "
                f"{oob_pct:>6.1f}% {max_overflow:>11.2f}px {rel_err:>7.4f}"
            )


# =============================================================================
# Visualization
# =============================================================================


def show_napari(results, V):
    """Show side-by-side comparison in napari."""
    import napari

    shape = V.shape
    names = list(results.keys())

    # Stack reconstructions and residuals
    recons = np.stack(
        [render_gaussians_numpy(shape, results[n], truncate=TRUNCATE) for n in names]
    )
    residuals = np.abs(np.stack([V] * len(names)) - recons)

    viewer = napari.Viewer(title="Boundary Containment Demo")

    # Input image (no slider axis)
    viewer.add_image(
        V,
        name="input (nuclei slice)",
        colormap="magma",
        contrast_limits=[0, float(V.max())],
    )

    # Stacked reconstructions with variant slider
    viewer.add_image(
        recons,
        name="reconstructions",
        colormap="magma",
        contrast_limits=[0, float(V.max())],
    )

    viewer.add_image(
        residuals,
        name="absolute residual",
        colormap="inferno",
        opacity=0.8,
        contrast_limits=[0, max(1e-6, float(residuals.max()))],
    )

    # Colors per variant
    colors = {
        "unconstrained": "red",
        "penalty": "yellow",
        "clip": "cyan",
        "combined": "lime",
    }

    # Add ellipses and centers per variant (one layer each, toggled by slider)
    for idx, name in enumerate(names):
        result = results[name]
        polys = make_ellipse_polygons(result, t=TRUNCATE)
        color = colors.get(name, "white")

        if polys:
            viewer.add_shapes(
                polys,
                shape_type="polygon",
                edge_color=color,
                edge_width=0.8,
                face_color=[0, 0, 0, 0],
                name=f"ellipses ({name})",
                visible=(idx == 0),
            )

        viewer.add_points(
            result.centers,
            name=f"centers ({name})",
            face_color=color,
            size=2,
            visible=(idx == 0),
        )

    # Volume boundary rectangle
    h, w = shape
    boundary = np.array(
        [[0, 0], [0, w - 1], [h - 1, w - 1], [h - 1, 0]], dtype=np.float32
    )
    viewer.add_shapes(
        [boundary],
        shape_type="polygon",
        edge_color="white",
        edge_width=2,
        face_color=[0, 0, 0, 0],
        name="volume bounds",
    )

    # Label the slider axis
    try:
        viewer.dims.axis_labels = ["variant", "y", "x"]
    except Exception:
        pass

    # Text overlay with stats
    def _update_overlay(event=None):
        t = viewer.dims.current_step[0]
        name = names[t]
        result = results[name]
        n_oob, total, max_overflow = count_out_of_bounds(result, shape, TRUNCATE)
        recon = recons[t]
        rel_err = float(np.linalg.norm(V - recon) / (np.linalg.norm(V) + 1e-12))
        oob_pct = 100.0 * n_oob / max(total, 1)

        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            f"Variant: {name}\n"
            f"Splats: {total}  |  Out-of-bounds: {n_oob} ({oob_pct:.1f}%)\n"
            f"Max overflow: {max_overflow:.2f} px  |  Rel L2 error: {rel_err:.4f}"
        )

        # Toggle ellipse/center layers
        for i, n in enumerate(names):
            for layer in viewer.layers:
                if n in layer.name and (
                    "ellipses" in layer.name or "centers" in layer.name
                ):
                    layer.visible = i == t

    _update_overlay()
    viewer.dims.events.current_step.connect(_update_overlay)

    aprint("Use the variant slider (axis 0) to compare approaches:")
    aprint("  0 = unconstrained, 1 = penalty, 2 = clip, 3 = combined")
    aprint("Toggle layers to see ellipse overlays for each variant")
    napari.run()


# =============================================================================
# Main
# =============================================================================


def main():
    aprint("=" * 70)
    aprint("Boundary Containment Demo")
    aprint("=" * 70)
    aprint("Comparing unconstrained vs boundary-constrained Gaussian splatting")
    aprint("")

    # Load data
    V = load_slice()

    # Fit four variants
    results = fit_variants(V)

    # Print analysis table
    analyze_results(results, V)

    # Napari visualization
    if not NO_NAPARI:
        show_napari(results, V)
    else:
        aprint("\nDemo complete (napari disabled)")


if __name__ == "__main__":
    main()
