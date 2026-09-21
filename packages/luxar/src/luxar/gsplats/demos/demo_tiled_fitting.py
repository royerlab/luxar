#!/usr/bin/env python3
"""
Tiled Fitting Demo - 3x3 Tiled Gaussian Splatting on cells3d Max-Projection

**What this demo demonstrates:**
- Tiled fitting: splitting a large image into overlapping tiles with Hann
  cosine apodization, fitting each tile independently, and concatenating
- The partition-of-unity property: overlapping cosine windows sum to 1.0,
  eliminating visible seams at tile boundaries
- Side-by-side comparison of tiled vs non-tiled fitting quality
- Visualization of per-tile splat contributions and window weights

**Key concepts:**
- Large microscopy volumes can exceed GPU memory. Tiled fitting solves this
  by processing one tile at a time with cosine apodization for seamless blending.
- The ``--tile N/M`` CLI flag enables Slurm-ready single-tile fitting for
  cluster parallelism.
- The Hann window guarantees that two adjacent tiles' weights sum to exactly
  1.0 in the overlap zone, so no post-merge pruning is needed.

**Data source:** Max-projection of cells3d nuclei channel from scikit-image
**Visualization:** napari viewer comparing tiled vs non-tiled, with per-tile
                   boundaries, window weights, and residual maps

**Command-line:**
    python demo_tiled_fitting.py [--no-napari]

**Related demos:**
- demo_boundary_containment.py - Boundary containment constraints
- demo_basic_fitting.py - Simple introduction to the fitting API
"""

import sys

import numpy as np
from arbol import Arbol, aprint, asection

from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.fit_tiled_gsplats import fit_tile
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.tiling import compute_tile_specs, cosine_window

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv

# ======= Demo knobs =======
N_ITERS = 2000
SEEDS = 200  # seeds per tile (and for non-tiled)
TILE_SIZE = 128
OVERLAP = 32
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps"
# ===========================

Arbol.max_depth = 4


# =============================================================================
# Data Loading
# =============================================================================


def load_cells3d_maxproj() -> np.ndarray:
    """Load cells3d nuclei channel as a 2D max-projection.

    Returns a float32 2D image normalized to [0, 1].
    """
    with asection("Loading cells3d max-projection"):
        try:
            from skimage.data import cells3d
        except ImportError:
            raise ImportError(
                "scikit-image is required for this demo.\n"
                "Install with: pip install scikit-image"
            )

        raw = cells3d()  # (60, 2, 256, 256) — (Z, C, Y, X)
        # Nuclei channel (index 1), max-projection over Z
        nuclei = raw[:, 1, :, :].astype(np.float32)
        V = nuclei.max(axis=0)  # (256, 256)
        V = (V - V.min()) / (V.max() - V.min() + 1e-8)
        aprint(f"Max-projection shape: {V.shape}")
        aprint(f"Data range: [{V.min():.3f}, {V.max():.3f}]")
        return V


# =============================================================================
# Fitting
# =============================================================================


def fit_non_tiled(V: np.ndarray) -> GSplatData:
    """Fit the whole image without tiling (baseline)."""
    with asection("Non-tiled fitting (baseline)"):
        result = fit_gaussian_splats(
            V,
            seeds=SEEDS,
            n_iters=N_ITERS,
            device=DEVICE,
            verbose=True,
        )
        aprint(f"Fitted {result.n_splats} splats")
        return result


def fit_tiles_and_merge(V: np.ndarray):
    """Fit each tile individually, return both per-tile and merged results.

    This is equivalent to calling fit_tiled() but also retains per-tile
    results for visualization of individual tile contributions.
    """
    # fold_slivers matches fit_tiled's default, so this really is the same grid.
    specs = compute_tile_specs(
        V.shape, tile_size=TILE_SIZE, overlap=OVERLAP, fold_slivers=True
    )
    tile_results = []

    with asection(
        f"Tiled fitting: {len(specs)} tiles (tile_size={TILE_SIZE}, overlap={OVERLAP})"
    ):
        for spec in specs:
            with asection(f"Tile {spec.index + 1}/{len(specs)} grid={spec.grid_index}"):
                result = fit_tile(
                    V,
                    spec,
                    seeds=SEEDS,
                    n_iters=N_ITERS,
                    device=DEVICE,
                    verbose=True,
                )
                aprint(f"{result.n_splats} splats")
                tile_results.append(result)

    # Merge all tiles (same as fit_tiled does internally)
    merged = GSplatData.concatenate(tile_results)
    aprint(f"Total: {merged.n_splats} splats from {len(specs)} tiles")

    return specs, tile_results, merged


# =============================================================================
# Analysis
# =============================================================================


def compute_quality(V: np.ndarray, result: GSplatData, label: str):
    """Compute and print quality metrics."""
    recon = render_gaussians_numpy(V.shape, result)
    mse = float(np.mean((V - recon) ** 2))
    psnr = 10.0 * np.log10(V.max() ** 2 / (mse + 1e-12))
    rel_l2 = float(np.linalg.norm(V - recon) / (np.linalg.norm(V) + 1e-12))
    aprint(
        f"{label}: {result.n_splats} splats, PSNR={psnr:.1f} dB, rel_L2={rel_l2:.4f}"
    )
    return recon, psnr, rel_l2


# =============================================================================
# Main
# =============================================================================


def main():
    aprint("=" * 70)
    aprint("Tiled Fitting Demo (3x3 on cells3d max-projection)")
    aprint("=" * 70)
    aprint("")

    # Load data
    V = load_cells3d_maxproj()

    # Fit both ways
    result_baseline = fit_non_tiled(V)
    specs, tile_results, result_tiled = fit_tiles_and_merge(V)

    # Quality comparison
    with asection("Quality Comparison"):
        recon_baseline, psnr_baseline, _ = compute_quality(
            V, result_baseline, "Non-tiled"
        )
        recon_tiled, psnr_tiled, _ = compute_quality(V, result_tiled, "Tiled    ")
        aprint("")
        aprint(
            f"Note: non-tiled uses {SEEDS} seeds total; tiled uses "
            f"{SEEDS} seeds x {len(specs)} tiles = "
            f"{SEEDS * len(specs)} seeds total"
        )
        aprint(f"PSNR difference: {psnr_tiled - psnr_baseline:+.1f} dB")

    # Build per-tile reconstruction for visualization
    per_tile_recons = []
    for tile_result in tile_results:
        tile_recon = render_gaussians_numpy(V.shape, tile_result)
        per_tile_recons.append(tile_recon)

    # Build accumulated window weights for visualization
    accumulated_window = np.zeros(V.shape, dtype=np.float32)
    for spec in specs:
        w = cosine_window(spec)
        accumulated_window[spec.slices] += w

    # Napari visualization
    if not NO_NAPARI:
        import napari

        aprint("Launching napari viewer...")
        viewer = napari.Viewer(title="Tiled Fitting Demo (3x3)")

        # Input image
        viewer.add_image(
            V,
            name="input (cells3d nuclei max-proj)",
            colormap="magma",
            contrast_limits=[0, float(V.max())],
        )

        # Reconstructions
        viewer.add_image(
            recon_baseline,
            name=f"non-tiled ({result_baseline.n_splats} splats, "
            f"PSNR={psnr_baseline:.1f} dB)",
            colormap="magma",
            contrast_limits=[0, float(V.max())],
            visible=False,
        )
        viewer.add_image(
            recon_tiled,
            name=f"tiled ({result_tiled.n_splats} splats, PSNR={psnr_tiled:.1f} dB)",
            colormap="magma",
            contrast_limits=[0, float(V.max())],
            visible=False,
        )

        # Residuals
        resid_baseline = np.abs(V - recon_baseline)
        resid_tiled = np.abs(V - recon_tiled)
        rmax = max(float(resid_baseline.max()), float(resid_tiled.max()), 1e-6)
        viewer.add_image(
            resid_baseline,
            name="residual (non-tiled)",
            colormap="inferno",
            contrast_limits=[0, rmax],
            visible=False,
        )
        viewer.add_image(
            resid_tiled,
            name="residual (tiled)",
            colormap="inferno",
            contrast_limits=[0, rmax],
            visible=False,
        )

        # Cosine window weights (should be uniformly 1.0 everywhere)
        viewer.add_image(
            accumulated_window,
            name="window sum (should be 1.0)",
            colormap="viridis",
            contrast_limits=[0.9, 1.1],
            visible=False,
        )

        # Per-tile reconstructions stacked
        if per_tile_recons:
            stacked = np.stack(per_tile_recons)
            viewer.add_image(
                stacked,
                name="per-tile reconstructions",
                colormap="magma",
                contrast_limits=[0, float(V.max())],
                visible=False,
            )

        # Per-tile splat centers with different colors
        tile_colors = [
            "red",
            "cyan",
            "yellow",
            "lime",
            "magenta",
            "orange",
            "white",
            "deepskyblue",
            "coral",
        ]
        for i, (spec, tile_result) in enumerate(zip(specs, tile_results)):
            if tile_result.n_splats > 0:
                color = tile_colors[i % len(tile_colors)]
                viewer.add_points(
                    tile_result.centers,
                    name=f"tile {i} centers (grid={spec.grid_index})",
                    face_color=color,
                    size=3,
                    visible=False,
                )

        # Tile boundary rectangles
        boundary_polys = []
        for spec in specs:
            y0, x0 = spec.slices[0].start, spec.slices[1].start
            y1, x1 = spec.slices[0].stop, spec.slices[1].stop
            boundary_polys.append(
                np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=np.float32)
            )
        viewer.add_shapes(
            boundary_polys,
            shape_type="polygon",
            edge_color="white",
            edge_width=1.5,
            face_color=[0, 0, 0, 0],
            name="tile boundaries",
        )

        aprint("")
        aprint("Layers:")
        aprint("  - Toggle 'non-tiled' vs 'tiled' reconstructions to compare")
        aprint("  - Toggle residual layers to see error distribution")
        aprint("  - 'window sum' should be uniformly 1.0 (partition-of-unity)")
        aprint("  - 'per-tile reconstructions' shows each tile's contribution")
        aprint("  - Colored points show which splats belong to which tile")
        aprint("  - White rectangles show tile boundaries")

        napari.run()
    else:
        aprint("")
        aprint("Demo complete (napari disabled)")
        aprint(
            f"Window sum range: [{accumulated_window.min():.6f}, "
            f"{accumulated_window.max():.6f}] (should be [1.0, 1.0])"
        )


if __name__ == "__main__":
    main()
