#!/usr/bin/env python3
"""CZ CELLxGENE Census — a very large single-cell scVI UMAP (LOD stress test).

A 3D UMAP of millions of human cells from the CZ CELLxGENE Census, embedded from
their precomputed **scVI latent** (50-d) with cuML UMAP, rendered in Luxar with
bounded additive Points streaming. A categorical ``coloring`` dimension
switches the colour scheme (navigate with ``[`` / ``]``):

  0: Cell type        (hundreds of categories — hashed hue)
  1: Tissue           (tissue_general)
  2: Disease

The colorings are stacked along the non-displayed ``coloring`` dimension, so the
viewer slices to ONE of them at a time — which is why the resident element count
is a third of the stored one, and why this scene needs no coarse levels. (It used
to demonstrate substitutive coarsening's ``coarsen_dims`` barrier, keeping each
coarse Gaussian pure to a single colour; ``demo_gsplats_lod_tribolium`` and
``demo_gsplats_lod_embryo_line`` are where that machinery is shown now.)

Data pipeline (heavy steps are GPU; see ``scripts``/README for the generator):
  CELLxGENE Census scVI latent  ->  cuML UMAP (3D)  ->  coords cache (NPZ)
  ->  Luxar streaming-ladder scene.

The coords cache (``census_umap_<N>.npz``: ``coords`` (N,3) f32 + per-cell int
codes for cell_type / tissue_general / disease + a ``labels_json`` map) is the
shippable artifact; this script loads it and builds the scene.

The scene carries an ADDITIVE ladder, not substitutive levels. Stacked over the
three colorings on a hidden axis, 3M elements are ~1M RESIDENT — five times under
the 5,591,040 Points cap — and the screen-area selector anchors a finest level at
half-screen occupancy, which this auto-fitted opening pose always satisfies. So
the coarse levels were bytes nobody fetched. See ``_lod_policy`` for the rule.

Scales (measured): the build is a write rather than a compute now that nothing is
coarsened — no torch/scipy, no CPU/GPU split, runs anywhere. Generating the 10M
scVI-UMAP *coords* still needs a GPU (cuML; see scripts). This demo resolves a
**1M-cell cache** through the checksum-verified dataset manifest and builds a
3M-element scene by default. The viewer eagerly converges an additive ladder to
100% of the selected coloring, so keep ``CENSUS_UMAP_MAX_CELLS`` at or below the
portable 5,591,040-Point node cap; larger values are silently clamped on a
4096-class GPU after one console warning. Point ``CENSUS_UMAP_CACHE`` at another
cache to rebuild from it.
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
        "local_data": None,
    },
    "caches": ["census_umap_1m"],
    "outputs": ["cellxgene_census_umap"],
    "citation": {
        "short": "CZ CELLxGENE Discover (CZI Cell Science Program 2025)",
        "ref": "CZI Cell Science Program 2025",
        "doi": "10.1093/nar/gkae1142",
    },
}

import colorsys
import json
import os
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    DatasetUnavailable,
    add_demo_caption,
    ensure_dataset,
    launch_viewer,
    parse_demo_flags,
    require_local_data,
    stamp_input_digests,
)
from luxar.demos._lod_policy import hidden_axis_stops, stream_ladder
from luxar.demos._support._umap_utils import attribute_to_color
from luxar.utils.paths import get_demos_output_dir

# (code-field, display label) for the categorical `coloring` dimension.
COLORINGS = [
    ("cell_type", "Cell type"),
    ("tissue_general", "Tissue"),
    ("disease", "Disease"),
]
# Hosted runnable default: a 1M-cell subsample. Override with CENSUS_UMAP_CACHE
# to point at a larger (e.g. 10M) regenerated cache.
DATASET = "census_umap_1m"

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
    # Guard a user-supplied override from producing a cryptic np.load zip error
    # when it is an unpulled LFS pointer.
    z = np.load(require_local_data(path), allow_pickle=False)
    labels = json.loads(str(z["labels_json"]))
    codes = {f"{c}": z[f"{c}_code"] for c, _ in COLORINGS}
    return z["coords"], codes, labels


def build_scene(
    cache: Path,
    output_path: Path,
    *,
    max_cells: int | None = None,
    seed: int = 0,
) -> int:
    """Build the streaming-ladder Points scene from the coords cache.

    Optionally subsamples to ``max_cells``, stacks the three colorings along
    the categorical ``coloring`` dimension, and returns the per-coloring cell
    count.

    Takes no ``device``/``compression_factor``/``levels`` any more: those were
    the substitutive-coarsening knobs, and this scene carries an additive ladder
    instead (see the ``add_points`` call). Dropping them also drops the
    torch+scipy import the coarsening write path needed, so a warm-cache build
    now runs anywhere.
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
    with asection("Building streaming-ladder scene"):
        with LuxarZarrCompiler(str(output_path)) as compiler:
            scene = compiler.create_scene(
                citation=DEMO_META["citation"],
                dimensions=dims,
                viewer_config=ViewerConfig(cinematic_mode=True),
            )
            stamp_input_digests(scene)
            scene.add_points(
                "cells",
                positions,
                colors=colors,
                radii=radii,
                sharpness=np.full(len(positions), 0.6, np.float32),
                labels=hover_labels,
                # Click a cell to look its term up in the EBI Ontology Lookup
                # Service, right-click to copy the term (#1917).
                #
                # No `keys=` here, unlike the sibling demos: this label is
                # ALREADY the bare ontology term with nothing appended, and it
                # is the right term per block — cell type, tissue or disease,
                # whichever the active coloring is. All three are OLS
                # vocabularies (CL / UBERON / MONDO), so one search URL serves
                # every block and the cloud pays no second CSR for 3 x N cells.
                link="https://www.ebi.ac.uk/ols4/search?q={hover_label}",
                copy="{hover_label}",
                # Appearance tuned in the viewer's Layers panel and baked back
                # here. Volumetric emission-absorption is what makes a cloud
                # this dense readable: its compositing is order-dependent, so
                # near cells ABSORB the ones behind them and the UMAP lobes
                # read as depth-ordered structure rather than the flat
                # order-independent sum that additive (the default)
                # accumulates. This being a MIXED substitutive ladder (the
                # coarse levels are lifted gsplats) is WHY the ray-mass
                # unification happened — one kappa has to serve both families —
                # and it deleted a point's world-thickness factor
                # (R * sqrt(pi/ln 100) = 0.0413 at this radius), multiplying
                # tau ~24x while leaving the stored 10 in place. 6.5 is a
                # re-tune, NOT that compensation — the spec's /24 would land
                # near 0.4, and this stays deliberately heavy because the
                # screening is what gives the lobes depth. The independence
                # runs one way: the display gain never enters tau (kappa is
                # screening strength only), but kappa still moves brightness,
                # through S(tau). On the finest Points level, at peak falloff
                # with the sprite at/above the 1.5 px floor, kappa=10 absorbed
                # 0.98 per cell and its own self-screening
                # S(tau) = (1 - e^-tau)/tau = 0.25 ate most of the emission;
                # 6.5 gives 0.92 and S = 0.36, ~1.45x more emission per cell at
                # peak. Coarse levels run higher tau (merged ray mass).
                # NB volumetric implies back-to-front depth sorting
                # (`needsDepthSort`), which this level's ~1M drawn points now
                # pay per camera move; `?depthSort=0` opts out.
                # `intensity` here is the Layers panel's DISPLAY RANGE control,
                # which is STORED as intensity/offset (intensity = 1/(max-min),
                # offset = -min/(max-min)) — so 4.52 is the window [0, 0.221].
                # On this direct-colour node the shader then applies it as a
                # plain colour gain; it only becomes a scalar-LUT window on
                # colormapped nodes. The previous [0, 2.361] was BOTH an
                # attenuation (a max above 1 cuts, here to 0.42x) and, even
                # undone, still ~4.5x short of a readable cloud — so of the
                # 10.7x total change only 2.36x undoes the cut (#1375).
                opacity=0.39,
                intensity=4.52,
                # Was 6.5. That figure was chosen as "deliberately heavy
                # because the screening is what gives the lobes depth", and it
                # overshot: at kappa=6.5 a single cell absorbs 0.92 of what is
                # behind it and its own self-screening S(tau)=0.36 eats most of
                # its emission, so the cloud reads as a shell with its interior
                # screened out rather than as depth-ordered structure. 2.12
                # keeps the near-absorbs-far cue that made volumetric worth
                # choosing while letting more of each cell's emission survive.
                absorption=2.12,
                blending_mode="volumetric",
                # Expose the single cells node in the viewer's Layers panel.
                # This rides onto the multi-LOD wrapper (not the per-rung
                # children), so the panel still shows one "cells" layer.
                layer=True,
                # Additive ladder only — no substitutive levels. Stacked over the
                # three colorings on a hidden axis, so 3,000,000 total is
                # ~1,000,000 RESIDENT against a 5,591,040 Points cap. The coarse
                # levels served a framing the screen-area selector never picks
                # (finest anchored at half-screen occupancy; this demo opens
                # auto-fitted), and cost four extra levels of nodes: 18 groups
                # -> 6.
                #
                # This also un-mixes the ladder. The absorption note above
                # explains that one kappa had to serve BOTH families because the
                # coarse levels were lifted gsplats; every node here is Points
                # now. The baked appearance is unchanged because it was tuned on
                # the finest level, which is what the opening pose showed then
                # and shows now.
                additive_lod=stream_ladder(
                    len(positions),
                    slices=hidden_axis_stops(positions, dims.non_displayed),
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
            add_demo_caption(
                scene,
                f"{n:,} human cells • scVI UMAP",
                DEMO_META.get("citation"),
            )
    aprint(f"\nScene saved to: {output_path}")
    return n


def main() -> None:
    """Resolve the cache, build the scene, and optionally launch the viewer."""
    flags = parse_demo_flags()
    override = os.environ.get("CENSUS_UMAP_CACHE")
    if override:
        cache = Path(override)
        if not cache.exists():
            _print_cache_guidance(f"coords cache not found at {cache}")
            return
    else:
        try:
            cache = ensure_dataset(DATASET)[0]
        except DatasetUnavailable as exc:
            _print_cache_guidance(f"default coords cache is unavailable: {exc}")
            raise SystemExit(1) from exc
    output_path = get_demos_output_dir() / "cellxgene_census_umap.luxar.zarr"
    # No CPU/GPU split any more: the build no longer coarsens, so it is a write,
    # not a compute. CENSUS_UMAP_MAX_CELLS also keeps the selected coloring
    # below the portable 5,591,040-Point node cap — see the module docstring.
    max_cells = int(os.environ.get("CENSUS_UMAP_MAX_CELLS", "1000000"))
    n = build_scene(cache, output_path, max_cells=max_cells)
    aprint(f"Built {n:,}-cell scene. To view: luxar serve --viewer {output_path}")
    if not flags.get("no_serve"):
        launch_viewer(output_path, open_browser=not flags.get("serve_only", False))


def _print_cache_guidance(detail: str) -> None:
    """Explain how to supply a regenerated Census UMAP cache."""
    aprint(
        f"⚠ {detail}\n"
        "  Generate it on a GPU box (see the module docstring / README):\n"
        "    python scripts/gen_census_umap.py --n 10000000 --out <cache>.npz\n"
        "  (needs cellxgene-census + cuml; ~96M primary human cells available),\n"
        "  then point this demo at it via CENSUS_UMAP_CACHE=<cache>.npz.\n"
        "  Or pre-built scenes can be served directly with `luxar serve --viewer`."
    )


if __name__ == "__main__":
    main()
