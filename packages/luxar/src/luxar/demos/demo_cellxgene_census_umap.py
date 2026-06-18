#!/usr/bin/env python3
"""CZ CELLxGENE Census — a very large single-cell scVI UMAP (LOD stress test).

A 3D UMAP of millions of human cells from the CZ CELLxGENE Census, embedded from
their precomputed **scVI latent** (50-d) with cuML UMAP, rendered in Luxar with
**substitutive Points LOD** (PR #379/#390). A categorical ``coloring`` dimension
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

import colorsys
import json
import os
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer, parse_demo_flags
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


def normalize_coords(coords: np.ndarray, span: float = 140.0) -> np.ndarray:
    c = coords - coords.mean(axis=0)
    c /= np.abs(c).max() + 1e-9
    return (c * (span / 2)).astype(np.float32)


def load_cache(path: Path):
    z = np.load(path, allow_pickle=False)
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
    coords, codes, labels = load_cache(cache)
    n = len(coords)
    if max_cells and n > max_cells:
        idx = np.random.default_rng(seed).choice(n, max_cells, replace=False)
        idx.sort()
        coords = coords[idx]
        codes = {k: v[idx] for k, v in codes.items()}
        n = max_cells
    coords = normalize_coords(coords)
    aprint(f"{n:,} cells × {len(COLORINGS)} colorings = {n * len(COLORINGS):,} elements")

    color_arrays = {
        field: _colors_for(field, codes[field], labels[field])
        for field, _ in COLORINGS
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
    radii = np.full(len(positions), 0.35, dtype=np.float32)

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
                opacity=0.85,
                intensity=0.2,
                # Auto coarsen_dims = coarsen x/y/z, group by the `coloring`
                # barrier so coarse splats stay pure per colour.
                substitutive_lod=dict(
                    compression_factor=compression_factor,
                    levels=levels,
                    device=device,
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
