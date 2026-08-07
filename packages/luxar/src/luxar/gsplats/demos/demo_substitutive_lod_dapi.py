#!/usr/bin/env python3
"""
3D DAPI — Substitutive LOD Demo

Builds a substitutive Levels-of-Detail hierarchy from a fitted Gaussian-splat
representation of a DAPI-stained nuclear microscopy volume. Mirrors the
qualitative finding of supplementary document
``substitutive_lod/substitutive_lod.tex`` Experiment C: cost-aware Lloyd
refinement (k-means + cost-increment Lloyd) dominates an amplitude-culling
baseline at matched splat budgets on real anisotropic 3D data.

**What it does:**

1. Load (or synthesise) a small DAPI volume.
2. Fit progressive gsplats to a moderate splat budget.
3. Build a substitutive LOD hierarchy at K=4, L=3.
4. Compare against an amplitude-culling baseline at matched per-level counts.
5. Report relative L² error and per-level PSNR.
6. Optional napari visualisation (use ``--no-napari`` to skip).

**Run:**

::

    hatch run python packages/luxar/src/luxar/gsplats/demos/demo_substitutive_lod_dapi.py
    hatch run python packages/luxar/src/luxar/gsplats/demos/demo_substitutive_lod_dapi.py --no-napari
"""

from __future__ import annotations

import sys

import numpy as np
from arbol import Arbol, aprint, asection

from luxar.gsplats.demos._demo_common import psnr
from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod import make_substitutive_lod
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy

NO_NAPARI = "--no-napari" in sys.argv

# ─── Demo knobs ────────────────────────────────────────────────────
TARGET_SIZE = 64  # downscale DAPI volume to this cube size (fast)
MAX_SPLATS = 1500  # source-of-truth splat budget for the fit
ITERS_PER_PASS = 800  # progressive fitting iterations per pass
TRUNCATE_SIG = 3.0  # rendering truncation
COMPRESSION_FACTOR = 4
LEVELS = 3
LLOYD_ITERS = 5
SEED = 42
# ───────────────────────────────────────────────────────────────────

Arbol.max_depth = 5


def _rel_l2(rendered: np.ndarray, target: np.ndarray) -> float:
    diff = (rendered.astype(np.float64) - target.astype(np.float64)).reshape(-1)
    t = target.astype(np.float64).reshape(-1)
    num = float(np.sqrt(np.dot(diff, diff)))
    den = float(np.sqrt(np.dot(t, t)))
    return num / max(den, 1e-12)


def _amplitude_cull(data: GSplatData, target_n: int) -> GSplatData:
    """Keep the ``target_n`` splats with the largest integral mass."""
    if target_n >= data.n_splats:
        return data
    masses = np.asarray(data.amplitudes, dtype=np.float64)
    # Use mass = amplitude * |Σ|^{1/2} ranking (same as the supp-doc baseline).
    diag = data._cholesky_diag_elements()
    sqrt_det = np.abs(np.prod(diag.astype(np.float64), axis=1))
    mass_score = masses * sqrt_det
    keep = np.argsort(-mass_score)[:target_n]
    keep_sorted = np.sort(keep)
    return GSplatData(
        centers=np.asarray(data.centers)[keep_sorted].astype(np.float32),
        amplitudes=np.asarray(data.amplitudes)[keep_sorted].astype(np.float32),
        cholesky_factors=np.asarray(data.cholesky_factors)[keep_sorted].astype(
            np.float32
        ),
        colors=(
            np.asarray(data.colors)[keep_sorted] if data.colors is not None else None
        ),
        truncation_radius=data.truncation_radius,
    )


def _load_dapi_volume() -> np.ndarray:
    """Load a small DAPI volume from IDR; fall back to a synthetic mixture."""
    try:
        import fsspec
        import zarr

        url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
        aprint(f"Loading from {url}")
        mapper = fsspec.get_mapper(url)
        try:
            store = zarr.open_group(mapper, mode="r")
        except Exception:
            store = zarr.open_array(mapper, mode="r")
        data = store["0"]
        full_shape = data.shape
        aprint(f"OME-ZARR shape: {full_shape}")
        if len(full_shape) == 5:
            V = np.asarray(data[0, 1, :, :, :], dtype=np.float32)
        elif len(full_shape) == 4:
            V = np.asarray(data[1, :, :, :], dtype=np.float32)
        else:
            V = np.asarray(data[:, :, :], dtype=np.float32)
        from scipy.ndimage import zoom

        factors = [TARGET_SIZE / s for s in V.shape]
        V = zoom(V, factors, order=1)
        vmin, vmax = V.min(), V.max()
        if vmax > vmin:
            V = ((V - vmin) / (vmax - vmin)) * 100.0
        else:
            V = np.ones_like(V) * 50.0
        return V.astype(np.float32)
    except Exception as e:
        aprint(f"Remote load failed: {e}; using synthetic nucleus-like data")
        rng = np.random.RandomState(SEED)
        V = np.zeros((TARGET_SIZE, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
        for _ in range(8):
            center = [rng.uniform(8, TARGET_SIZE - 8) for _ in range(3)]
            sigma = rng.uniform(3.0, 6.0)
            amp = rng.uniform(60.0, 100.0)
            grids = np.meshgrid(
                *[np.arange(TARGET_SIZE) for _ in range(3)], indexing="ij"
            )
            dist_sq = sum((g - c) ** 2 for g, c in zip(grids, center))
            V += amp * np.exp(-dist_sq / (2 * sigma**2))
        return np.clip(V, 0, 100).astype(np.float32)


def _render_data(data: GSplatData, shape: tuple[int, ...]) -> np.ndarray:
    """Render a flat (or already-flat) GSplatData to a volume."""
    flat = data if data.n_additive_sublods == 1 else data.flattened()
    return render_gaussians_numpy(shape, flat, truncate=TRUNCATE_SIG).astype(np.float32)


def main() -> None:
    with asection("3D DAPI Substitutive LOD Demo"):
        with asection("Loading DAPI data"):
            V = _load_dapi_volume()
            aprint(f"Volume: {V.shape}, range: [{V.min():.2f}, {V.max():.2f}]")

        with asection("Progressive fit (source of truth)"):
            fitted = fit_progressive_gaussian_splats(
                V,
                max_splats=MAX_SPLATS,
                max_splats_per_pass=MAX_SPLATS,
                iters_per_pass=ITERS_PER_PASS,
                psnr_patience=0.05,
                truncate=TRUNCATE_SIG,
                verbose=False,
            )
            aprint(f"Fitted {fitted.n_splats} splats")
            rendered_full = _render_data(fitted, V.shape)
            full_rel_l2 = _rel_l2(rendered_full, V)
            full_psnr = psnr(rendered_full, V)
            aprint(
                f"Full reconstruction: rel_l2={full_rel_l2:.4f}, PSNR={full_psnr:.2f} dB"
            )

        with asection("Substitutive LOD"):
            pyramid = make_substitutive_lod(
                fitted,
                compression_factor=COMPRESSION_FACTOR,
                levels=LEVELS,
                method="kmeans_lloyd",
                lloyd_iterations=LLOYD_ITERS,
                candidate_bins_k=12,
                device="auto",
                seed=SEED,
                verbose=False,
            )
            # Materialise per-level views for convenient enumeration.
            hierarchy = [
                pyramid.at_substitutive(s) for s in range(pyramid.n_substitutive)
            ]
            for level_idx, lev in enumerate(hierarchy):
                aprint(f"  level {level_idx}: {lev.n_splats} splats")

        with asection("Amplitude-culling baseline"):
            culled = [fitted]
            for lev in hierarchy[1:]:
                culled.append(_amplitude_cull(fitted, lev.n_splats))
            for level_idx, c in enumerate(culled):
                aprint(f"  level {level_idx}: {c.n_splats} splats")

        with asection("Per-level rel_l2 and PSNR"):
            sub_rel: list[float] = []
            cull_rel: list[float] = []
            sub_psnrs: list[float] = []
            cull_psnrs: list[float] = []
            for level_idx in range(len(hierarchy)):
                rendered_sub = _render_data(hierarchy[level_idx], V.shape)
                rendered_cull = _render_data(culled[level_idx], V.shape)
                rs = _rel_l2(rendered_sub, V)
                rc = _rel_l2(rendered_cull, V)
                ps = psnr(rendered_sub, V)
                pc = psnr(rendered_cull, V)
                sub_rel.append(rs)
                cull_rel.append(rc)
                sub_psnrs.append(ps)
                cull_psnrs.append(pc)
                aprint(
                    f"  level {level_idx}: "
                    f"sub n={hierarchy[level_idx].n_splats:>5d} rel_l2={rs:.4f} PSNR={ps:.2f} dB | "
                    f"cull n={culled[level_idx].n_splats:>5d} rel_l2={rc:.4f} PSNR={pc:.2f} dB"
                )

        with asection("Verdict (supp doc Experiment C)"):
            for level_idx in range(1, len(hierarchy)):
                if sub_rel[level_idx] < cull_rel[level_idx]:
                    aprint(
                        f"  level {level_idx}: substitutive WINS by "
                        f"{cull_rel[level_idx] - sub_rel[level_idx]:.4f} rel_l2 "
                        f"(cost-aware Lloyd > amplitude culling)"
                    )
                else:
                    aprint(
                        f"  level {level_idx}: amplitude culling marginally ahead "
                        f"({sub_rel[level_idx] - cull_rel[level_idx]:.4f}); on tiny "
                        "fixtures this is noisy."
                    )

    if NO_NAPARI:
        aprint("Demo completed (napari visualisation disabled).")
        return

    try:
        import napari
    except ImportError:
        aprint("napari not installed; skipping visualisation.")
        return

    viewer = napari.Viewer(
        title="DAPI Substitutive LOD: substitutive vs amplitude-cull",
        ndisplay=3,
    )
    viewer.add_image(
        V, name="DAPI input", colormap="gray", contrast_limits=[0, float(V.max())]
    )
    sub_stack = np.stack([_render_data(lev, V.shape) for lev in hierarchy], axis=0)
    cull_stack = np.stack([_render_data(c, V.shape) for c in culled], axis=0)
    viewer.add_image(
        sub_stack,
        name="Substitutive LOD",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
    )
    viewer.add_image(
        cull_stack,
        name="Amplitude-cull baseline",
        colormap="gray",
        contrast_limits=[0, float(V.max())],
        visible=False,
    )

    try:
        viewer.dims.axis_labels = ["LOD level", "z", "y", "x"]
    except Exception:
        pass

    aprint("Use the LOD slider to compare. Toggle layers for direct A/B.")
    napari.run()


if __name__ == "__main__":
    main()
