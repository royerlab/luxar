#!/usr/bin/env python3
"""CZ CELLxGENE Census — a very large single-cell scVI UMAP (LOD stress test).

A 3D UMAP of millions of human cells from the CZ CELLxGENE Census, embedded from
their precomputed **scVI latent** (50-d) with cuML UMAP, rendered in Luxar with
**substitutive Points LOD**. A categorical ``coloring`` dimension
switches the colour scheme (navigate with ``[`` / ``]``):

  0: Cell type        (hundreds of categories — hashed hue)
  1: Tissue           (tissue_general)
  2: Disease

Because the colorings are stacked along the non-displayed ``coloring`` dimension,
substitutive coarsening's **Auto default** (coarsen displayed dims, group by the
non-displayed dim) keeps every coarse Gaussian splat pure to a single colour — the
exact ``coarsen_dims`` barrier feature. With N≈10M cells × 3 colorings this is a
~30M-element scene that genuinely exercises the LOD machinery.

Data pipeline (heavy steps are GPU; see ``scripts``/README for the generator):
  CELLxGENE Census scVI latent  ->  cuML UMAP (3D)  ->  coords cache (NPZ)
  ->  Luxar substitutive-LOD scene.

The coords cache (``census_umap_<N>.npz``: ``coords`` (N,3) f32 + per-cell int
codes for cell_type / tissue_general / disease + a ``labels_json`` map) is the
shippable artifact; this script loads it and builds the scene.

Scales (measured): the substitutive-LOD **build** is cheap — 1M cells × 3 = 3M
elements in ~46 s on CPU; the full 10M × 3 = 30M elements in ~8 min / <16 GB RAM.
Generating the 10M scVI-UMAP *coords* needs a GPU (cuML; see scripts). This demo
ships a **1M-cell cache** (Git LFS) and builds a 3M-element scene by default —
which the web viewer renders cleanly. NB: the full 30M scene builds fine and is
barrier-pure at every level, but the current viewer eagerly streams the finest
level, so ~10M+ finest points can exhaust the browser — a viewer LOD-streaming
ceiling, not a scene defect. Override with ``CENSUS_UMAP_CACHE`` / ``CENSUS_UMAP_
MAX_CELLS`` / ``CENSUS_UMAP_DEVICE`` to build larger.
"""

from __future__ import annotations

DEMO_META = {
    "key": "cellxgene_census_umap",
    "title": "CELLxGENE Census UMAP",
    "description": "3D scVI UMAP of ~1M human cells from the CZ CELLxGENE Census (Points LOD stress test).",
    "category": "embeddings",
    "geometry": "points",
    "requirements": {
        "download_mb": 12,
        "compute": "medium",
        "gpu": "none",
        "local_data": "git-lfs",
    },
    "caches": [],
    "outputs": ["cellxgene_census_umap"],
}

import colorsys
import json
import os
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import (
    launch_viewer,
    parse_demo_flags,
    require_local_data,
    substitutive_lod_or_flat,
)
from luxar.utils._umap_utils import attribute_to_color
from luxar.utils.paths import get_demos_output_dir

# (code-field, display label) for the categorical `coloring` dimension.
COLORINGS = [
    ("cell_type", "Cell type"),
    ("tissue_general", "Tissue"),
    ("disease", "Disease"),
]
# Shipped runnable default: a 1M-cell subsample (Git LFS). Override with the
# CENSUS_UMAP_CACHE env var to point at a larger (e.g. 10M) regenerated cache.
DEFAULT_CACHE = Path(__file__).parent / "data" / "census_umap_1m.npz"

# Per-cell sphere radius in scene units, tied to the local cell spacing rather
# than picked by eye: at NORM_SPAN the 1M-cell cloud has a median
# nearest-neighbour distance of ~0.127, so a radius of ~0.4x that leaves
# adjacent cells just short of touching. Anything approaching the spacing
# itself fuses the cloud — at the old 0.35 each sphere swallowed a median of 13
# neighbours (p90 62) and every mid-density region clipped to opaque white,
# hiding both the UMAP filaments and the per-cell-type hues.
POINT_RADIUS = 0.05
# Coordinate span the UMAP is normalized into; POINT_RADIUS is calibrated for it.
NORM_SPAN = 140.0


def _hashed_hue(codes: np.ndarray) -> np.ndarray:
    """Stable, well-spread hue per integer category code (good for hundreds)."""
    out = np.zeros((len(codes), 3), dtype=np.float32)
    uniq = np.unique(codes)
    lut = {}
    for v in uniq:
        h = ((int(v) * 2654435761) % 360) / 360.0  # Knuth multiplicative hash
        s = 0.55 + 0.35 * (((int(v) >> 8) % 5) / 4.0)
        lut[int(v)] = colorsys.hsv_to_rgb(h, s, 0.95)
    for v in uniq:
        out[codes == v] = lut[int(v)]
    return out


def _colors_for(field: str, codes: np.ndarray, labels: list[str]) -> np.ndarray:
    """High-cardinality fields (cell_type) get hashed hues; smaller ones the
    curated categorical palette keyed by label."""
    if len(labels) > 64:
        return _hashed_hue(codes)
    names = np.array(labels, dtype=object)[codes]
    return attribute_to_color(names, field)


def normalize_coords(coords: np.ndarray, span: float = NORM_SPAN) -> np.ndarray:
    """Center the cloud and rescale it to fill a cube of side ``span``."""
    c = coords - coords.mean(axis=0)
    c /= np.abs(c).max() + 1e-9
    return (c * (span / 2)).astype(np.float32)


def load_cache(path: Path):
    """Load the coords cache: returns (coords, per-coloring int codes, labels)."""
    # Gate the shipped LFS npz so an unpulled pointer gives the "git lfs pull"
    # message instead of a cryptic np.load zip error.
    z = np.load(require_local_data(path), allow_pickle=False)
    labels = json.loads(str(z["labels_json"]))
    codes = {f"{c}": z[f"{c}_code"] for c, _ in COLORINGS}
    return z["coords"], codes, labels


def build_scene(
    cache: Path,
    output_path: Path,
    *,
    max_cells: int | None = None,
    device: str = "cpu",
    compression_factor: int = 4,
    levels: int = 4,
    seed: int = 0,
) -> int:
    """Build the substitutive-LOD Points scene from the coords cache.

    Optionally subsamples to ``max_cells``, stacks the three colorings along
    the categorical ``coloring`` dimension, and returns the per-coloring cell
    count.
    """
    coords, codes, labels = load_cache(cache)
    n = len(coords)
    if max_cells and n > max_cells:
        idx = np.random.default_rng(seed).choice(n, max_cells, replace=False)
        idx.sort()
        coords = coords[idx]
        codes = {k: v[idx] for k, v in codes.items()}
        n = max_cells
    coords = normalize_coords(coords)
    aprint(
        f"{n:,} cells × {len(COLORINGS)} colorings = {n * len(COLORINGS):,} elements"
    )

    color_arrays = {
        field: _colors_for(field, codes[field], labels[field]) for field, _ in COLORINGS
    }

    dims = Dimensions(
        [
            Dimension(
                "coloring",
                unit="",
                categories=[lbl for _, lbl in COLORINGS],
                display=False,
                description="Colour scheme for the CELLxGENE Census UMAP",
            ),
            Dimension("x", unit="UMAP", display=True),
            Dimension("y", unit="UMAP", display=True),
            Dimension("z", unit="UMAP", display=True),
        ]
    )

    # Stack the colorings along the categorical `coloring` dim — ONE node.
    positions = np.vstack(
        [
            np.column_stack([np.full(n, i, np.float32), coords])
            for i in range(len(COLORINGS))
        ]
    ).astype(np.float32)
    colors = np.vstack([color_arrays[f] for f, _ in COLORINGS]).astype(np.float32)
    radii = np.full(len(positions), POINT_RADIUS, dtype=np.float32)

    # Per-cell hover labels, aligned with the stacked `coloring` blocks: within
    # each block a point shows that coloring's category (cell type / tissue /
    # disease) for its cell — the metadata is already in `codes`/`labels`.
    hover_labels: list[str] = []
    for field, _ in COLORINGS:
        cat_names = labels[field]
        hover_labels.extend(
            cat_names[c] if 0 <= c < len(cat_names) else "?" for c in codes[field]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with asection(f"Building substitutive-LOD scene (device={device})"):
        with LuxarZarrCompiler(str(output_path)) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points(
                "cells",
                positions,
                colors=colors,
                radii=radii,
                sharpness=np.full(len(positions), 0.6, np.float32),
                labels=hover_labels,
                # Appearance tuned in the viewer's Layers panel and baked back
                # here. Volumetric emission-absorption is what makes a cloud
                # this dense readable: its compositing is order-dependent, so
                # near cells ABSORB the ones behind them and the UMAP lobes
                # read as depth-ordered structure rather than the flat
                # order-independent sum that additive (the default)
                # accumulates. This is a MIXED substitutive ladder (the coarse
                # levels are lifted gsplats), so the ray-mass unification left
                # kappa untouched: tau = kappa * rayMass with rayMass the same
                # peak-alpha the additive branch emits, and kappa=10 keeps
                # occlusion building up across overlapping cells instead of
                # saturating on any single one.
                # NB volumetric implies back-to-front depth sorting
                # (`needsDepthSort`), which this ~1M-point level now pays per
                # camera move; `?depthSort=0` opts out.
                # `intensity` here is the Layers panel's DISPLAY RANGE control,
                # which is STORED as intensity/offset (intensity = 1/(max-min),
                # offset = -min/(max-min)) — so 0.4235 is the window
                # [0, 2.361]. On this direct-colour node the shader then
                # applies it as a plain colour gain; it only becomes a
                # scalar-LUT window on colormapped nodes.
                opacity=0.39,
                intensity=0.4235,
                absorption=10.0,
                blending_mode="volumetric",
                # Expose the single cells node in the viewer's Layers panel.
                # With the substitutive-LOD wrapper this rides onto the
                # kind=lod group (not the per-level children), so the panel
                # shows one "cells" layer.
                layer=True,
                # Auto coarsen_dims = coarsen x/y/z, group by the `coloring`
                # barrier so coarse splats stay pure per colour.
                substitutive_lod=substitutive_lod_or_flat(
                    dict(
                        compression_factor=compression_factor,
                        levels=levels,
                        device=device,
                    )
                ),
            )
            for idx, (_, label) in enumerate(COLORINGS):
                scene.add_text(
                    f"CELLxGENE Census — colored by {label}",
                    position=(0.02, 0.97),
                    font_size=0.02,
                    anchor="bottom-left",
                    color="#ffcc44",
                    visible_range={"coloring": idx},
                    transition="fade",
                    transition_duration=0.2,
                )
            scene.add_text(
                f"CZ CELLxGENE Census — {n:,} human cells (scVI UMAP)",
                position=(0.02, 0.02),
                font_size=0.028,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
    aprint(f"\nScene saved to: {output_path}")
    return n


def main() -> None:
    """Resolve the cache, build the scene, and optionally launch the viewer."""
    flags = parse_demo_flags()
    cache = Path(os.environ.get("CENSUS_UMAP_CACHE", DEFAULT_CACHE))
    if not cache.exists():
        aprint(
            f"⚠ coords cache not found at {cache}\n"
            "  Generate it on a GPU box (see the module docstring / README):\n"
            "    python scripts/gen_census_umap.py --n 10000000 --out <cache>.npz\n"
            "  (needs cellxgene-census + cuml; ~96M primary human cells available),\n"
            "  then point this demo at it via CENSUS_UMAP_CACHE=<cache>.npz.\n"
            f"  Or pre-built scenes can be served directly with `luxar serve --viewer`."
        )
        return
    output_path = get_demos_output_dir() / "cellxgene_census_umap.luxar.zarr"
    # CPU build is tractable to ~1M cells; pass a GPU build (device='cuda',
    # higher max) for the full ~10M showcase — see the module docstring.
    max_cells = int(os.environ.get("CENSUS_UMAP_MAX_CELLS", "1000000"))
    device = os.environ.get("CENSUS_UMAP_DEVICE", "cpu")
    n = build_scene(cache, output_path, device=device, max_cells=max_cells)
    aprint(f"Built {n:,}-cell scene. To view: luxar serve --viewer {output_path}")
    if not flags.get("no_serve"):
        launch_viewer(output_path, open_browser=not flags.get("serve_only", False))


if __name__ == "__main__":
    main()
