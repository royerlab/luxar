#!/usr/bin/env python3
"""GSplats Demo: LOD **recipe gallery** on a real Tribolium embryo (Light-Sheet)

Takes the ~300K-splat fit of the *Tribolium castaneum* embryo (the same
precomputed dataset as ``demo_gsplats_3d_tribolium_embryo.py``) and runs the
unified ``luxar gsplat lod --recipe`` pipeline to build the SIX scale-ordered
representation topologies side by side, so you can compare them directly:

    flat  →  stream  →  levels  →  tiles  →  overview  →  adaptive
   (small)  (medium)  (medium)   (large)    (huge)      (huge, adaptive)

================================================================================
WHAT THIS DEMONSTRATES — THE `lod --recipe` PIPELINE + THE NOVEL TOPOLOGIES
================================================================================

A fitted ``.gsplats.zarr`` is a flat bag of N splats. The canonical pipeline to
make it render well at any scale is::

    luxar gsplat cal volume.tiff cal.json          # pick a splat budget K*
    luxar gsplat fit volume.tiff fit.gsplats.zarr --seeds <K*>
    luxar gsplat lod fit.gsplats.zarr out.gsplats.zarr --recipe <RECIPE>
    luxar gsplat convert out.gsplats.zarr scene.luxar.zarr && luxar serve ...

This demo runs the ``lod --recipe`` step for each recipe on the SAME base fit and
grafts the six results into one scene as labelled, colour-coded columns. The three
novel topologies are the point:

- **levels** — a ``kind=lod`` of *substitutive* levels: each coarser level
  REPLACES the finer one with fewer, larger merged splats. The viewer shows
  exactly one level at a time and swaps by on-screen size, so a switch reads as a
  single shade jump (coarse = pale, fine = deep) across the whole object.
- **tiles** — a spatial BSP ``kind=partition`` where EACH part carries its
  own additive LOD ladder. Off-screen parts frustum-cull; visible parts stream
  detail progressively. Each BSP part gets a distinct HUE (see the spatial cells)
  and each ladder level a SHADE of that hue (see the coarse→fine accumulation).
- **overview** — an *unbalanced-by-design* ``kind=lod``: a single cheap coarse
  substitutive cap for the far/zoomed-out view, ABOVE a ``tiles`` fine
  branch for close-up. Detail structure exists only where you look closely. The
  coarse↔fine switch is unmistakable: zoom out → one uniform pale blob (the cap);
  zoom in → it bursts into the multicoloured fine partition parts.
- **adaptive** — a spatial BSP ``kind=partition`` where EACH part is its own
  *substitutive* lod group (coarse↔fine *replacement* per part). Unlike
  ``tiles`` (additive, accumulating) and ``overview`` (one global cap),
  every cell picks its own level by its own on-screen size — locally adaptive
  detail. Each part gets a distinct HUE; each substitutive level a SHADE of it,
  so a per-part swap reads as that part's shade jumping as you navigate.

For reference the two primitives bracket them: **flat** (one leaf, neutral grey)
and **stream** (one leaf + a prefix-sum ladder, each accumulating level painted
its own vivid hue so the coarse→fine streaming is unmistakable).

All colouring is applied AFTER the fact by a generic painter that walks whatever
tree the unmodified ``build_recipe`` engine produced (keyed on node type, never on
recipe identity) — so what renders is byte-for-byte the topology the CLI builds,
only recoloured.

Pipeline:
1. **Load** the locally cached ~300K-splat Tribolium fit (cold cache re-fits)
2. **Center** it so all six columns sit at the origin before placement
3. **Build** each recipe with ``build_recipe`` (the engine behind the CLI) and
   write each to a ``.gsplats.zarr`` — exactly what ``lod --recipe`` does
4. **Colour-code** the result after the fact (hue = part, shade = LOD level)
5. **Compose** one scene: six translated columns, each grafted via
   ``add_gsplats_from_file`` (exactly what ``gsplat convert`` does), with a legend
6. **Visualize** — pan across the row; zoom into a column to watch its LOD switch

DATA SOURCE & CITATIONS:
========================
Source:  Cell Tracking Challenge / Zenodo record 5270323 (GIANI paper)
Imaging: Zeiss LightSheet Z.1, Tribolium castaneum, 0.381 um isotropic
Cite:    Barry et al. (2022). GIANI. J. Cell Sci. 135, jcs259511.
         Maska et al. (2023). Cell Tracking Challenge. Nat. Methods 20, 1010-1020.

USAGE:
======
    python demo_gsplats_recipes_tribolium.py [--recompute] [--no-serve] [--serve-only]
                                             [--max-elements=N] [--factor=K]

Options:
    --recompute:       Force re-fitting the base splats from scratch (download + GPU)
    --no-serve:        Generate scene without launching viewer
    --serve-only:      Just serve a previously generated scene (skips rebuild)
    --max-elements=N:  Per-part BSP cap for tiles/overview/adaptive (default 50000)
    --factor=K:        Coarse-cap compression for overview/levels (default 4)
    Adaptive/levels use the fixed demo depth LEVELS=2.

Output:
    - Scene saved to:  datasets/demos/gsplats_recipes_tribolium.luxar.zarr
    - Automatically opens in browser
"""

DEMO_META = {
    "key": "gsplats_recipes_tribolium",
    "title": "LOD **recipe gallery** on a real Tribolium embryo (Light-Sheet)",
    "description": "Six LOD recipes side by side on a real Tribolium embryo light-sheet gsplat fit.",
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 3,
        "compute": "medium",
        # The precomputed fit is no longer shipped in-tree (the source data is
        # not redistributable), so a first run fetches the raw source and
        # refits — which needs a GPU. Nothing has to be placed by hand.
        "gpu": "required",
        "local_data": None,
    },
    "caches": ["gsplats_tribolium"],
    "outputs": ["gsplats_recipes_tribolium"],
    "citation": {
        "short": (
            "Barry 2021 (GIANI, Zenodo 5270323); "
            "Cell Tracking Challenge (Maška et al. 2023)"
        ),
        "ref": "Barry / Maška et al. 2023",
        "doi": "10.5281/zenodo.5270323",
    },
}

import colorsys
import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core import transforms
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    add_demo_caption,
    detect_device,
    launch_viewer,
    load_precomputed_gsplats,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData
from luxar.gsplats.io.save_gsplats import write_gsplats_tree
from luxar.gsplats.lod.recipes import RecipeParams, build_recipe
from luxar.gsplats.tree import (
    GSplatLeaf,
    GSplatLodGroup,
    GSplatNode,
    GSplatPartition,
    total_splats,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Per-part BSP cap for tiles / overview / adaptive. The base fit is ~300K splats;
# 50K → ~5 spatial parts, enough to see the cells and the frustum-culling story.
MAX_ELEMENTS = 50_000
# Coarse-cap compression for overview / levels (one substitutive level ≈ N/FACTOR splats).
#
# Was 8 with the default three levels, i.e. N/8, N/64, N/512. At the opening
# pose each of the six embryos occupies roughly 5% of the screen, below the
# ladder's first switch (1/8 of the screen area), so every column sat on its
# COARSEST level — ~600 splats for a 300K fit, individually visible blobs
# (2026-09-10 review: "way too coarse, you see the splats"). The gallery is six
# objects side by side, so no column can ever reach the half-screen finest
# anchor without zooming; the fix that stays within the recipes' own contract
# is a shallower, gentler ladder: K=4 and TWO levels gives N/4 and N/16, so the
# opening pose shows ~19K splats per embryo and one zoom step reaches N/4.
# The general "finer levels should trigger sooner" question is #2685.
FACTOR = 4
LEVELS = 2
# NB: the overview coarse↔fine switch uses viewport-relative coverage_fraction
# thresholds. Unlike a whole-object `levels` ladder, overview's pair is stamped by
# `partitioned_coverage_fractions`, so the coarse cap (fewer-but-larger splats) is
# what you see at the opening framing and the fine branch engages only once you
# zoom the embryo up to filling the viewport. That is the recipe's contract —
# "instant coarse overview, fine tiles on zoom" — and it is why overview does NOT
# take the whole-object half-screen-area anchor the other recipes use. No per-dataset
# threshold knob to tune either way.
# Additive-ladder depth for the stream recipe and the tiles / overview per-part
# (and per-level) streaming ladders.
N_LODS = 4
# Cheap O(N log N) additive ordering — keeps the demo fast on CPU.
ADDITIVE_METHOD = "self_energy"

RECIPES = ("flat", "stream", "levels", "tiles", "overview", "adaptive")

# Parse command-line flags.
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
for _arg in sys.argv:
    if _arg.startswith("--max-elements="):
        MAX_ELEMENTS = int(_arg.split("=")[1])
    elif _arg.startswith("--factor="):
        FACTOR = int(_arg.split("=")[1])

Arbol.max_depth = 5

# Colour palettes (RGB floats in [0, 1]).
#
# Debug-colouring scheme — every part AND every LOD level gets its own colour, so
# that BOTH facts are obvious at a glance: how many parts there are, and the
# instant a level switches. We encode them on two orthogonal axes:
#     hue   → which spatial partition part   (distinct parts ⇒ distinct hues)
#     shade → which LOD level within a part   (a switch ⇒ a visible shade jump)
# A standalone ladder (the ``stream`` recipe — a single "part") instead spreads
# its accumulating levels across the *full* hue wheel, so every streamed-in level
# is its own vivid colour. ``flat`` stays neutral grey (nothing to differentiate).
_GREY = (0.72, 0.74, 0.78)  # flat — a single undifferentiated leaf


def _hsv(h: float, s: float, v: float) -> tuple[float, float, float]:
    """HSV (hue wrapped to [0, 1)) to an RGB float triple."""
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (float(r), float(g), float(b))


def _level_shades(hue: float, n_levels: int) -> list[tuple[float, float, float]]:
    """``n_levels`` distinct shades of one ``hue``, ordered coarsest→finest:
    coarse = pale/bright, fine = vivid/deep. The large saturation+value swing
    makes a level switch unmistakable even within a single part's hue."""
    if n_levels <= 1:
        return [_hsv(hue, 0.70, 0.96)]
    return [
        _hsv(hue, 0.35 + 0.60 * (j / (n_levels - 1)), 1.0 - 0.45 * (j / (n_levels - 1)))
        for j in range(n_levels)
    ]


def _color_array(rgb: tuple[float, float, float], n: int) -> np.ndarray:
    """Tile one ``rgb`` into an ``(n, 3)`` float32 per-splat colour array."""
    return np.tile(np.asarray(rgb, dtype=np.float32), (n, 1))


def _recolor_sublod(
    sub: AdditiveSubLOD, rgb: tuple[float, float, float]
) -> AdditiveSubLOD:
    """A copy of ``sub`` with a flat (N, 3) debug colour (frozen → rebuild)."""
    return AdditiveSubLOD(
        centers=sub.centers,
        amplitudes=sub.amplitudes,
        cholesky_factors=sub.cholesky_factors,
        colors=_color_array(rgb, sub.n_splats),
        stats=dict(sub.stats),
        truncation_radius=sub.truncation_radius,
    )


def _recolor_leaf(leaf: GSplatLeaf, rgb: tuple[float, float, float]) -> GSplatLeaf:
    """A copy of ``leaf`` with every sub-LOD painted one flat ``rgb``."""
    return GSplatLeaf(
        additive_sublods=[_recolor_sublod(s, rgb) for s in leaf.additive_sublods],
        meta=dict(leaf.meta),
    )


def _recolor_leaf_by_shades(leaf: GSplatLeaf, hue: float) -> GSplatLeaf:
    """A copy of ``leaf`` whose additive sub-LODs (coarse→fine) get distinct
    shades of ``hue`` — so the prefix-sum accumulation reads as concentric shades
    and each streamed-in level is a visibly new colour."""
    subs = leaf.additive_sublods
    shades = _level_shades(hue, len(subs))
    return GSplatLeaf(
        additive_sublods=[_recolor_sublod(s, shades[j]) for j, s in enumerate(subs)],
        meta=dict(leaf.meta),
    )


# =============================================================================
# Generic, recipe-agnostic post-hoc painter
# =============================================================================
#
# We build every recipe with the *unmodified* ``build_recipe`` engine, then paint
# the result AFTER THE FACT by walking whatever tree it produced — keyed only on
# node *type* (the stable ``GSplatNode`` contract), never on which recipe made it
# or on assumed child positions. This guarantees we serialize byte-for-byte the
# same topology the CLI would (same constructors, ``meta`` preserved — including
# the ``coverage_fraction`` switch thresholds), differing only in the colour arrays.
# The two axes (see above): hue = spatial part, shade = LOD level.


def _arc_mid(arc: tuple[float, float]) -> float:
    """Midpoint hue of a ``[lo, hi)`` hue arc."""
    return 0.5 * (arc[0] + arc[1])


def _paint_data(data: GSplatData) -> GSplatData:
    """A flat ``GSplatData`` (the ``flat`` / ``stream`` matrix recipes).

    One sub-LOD → neutral grey (``flat``: nothing to differentiate). A multi-level
    ladder (``stream``) → each accumulating level its own vivid hue across the
    full wheel, so every streamed-in increment is unmistakably a new colour.
    """
    subs = data.additive_sublods
    n = len(subs)
    if n == 1:
        return GSplatData.from_additive_sublods([_recolor_sublod(subs[0], _GREY)])
    colored = [_recolor_sublod(s, _hsv(i / n, 0.80, 0.97)) for i, s in enumerate(subs)]
    return GSplatData.from_additive_sublods(colored)


def _paint_node(node: GSplatNode, arc: tuple[float, float]) -> GSplatNode:
    """Recursively paint a node tree within the hue arc ``[lo, hi)``.

    * ``GSplatPartition`` → split the arc evenly among the parts so every part is
      a distinct hue; recurse into each (each part owns its slice).
    * ``GSplatLodGroup``  → siblings are mutually-exclusive levels (only one shown
      at a time), so a switch is read as a colour change. Leaf levels get a
      distinct shade of the arc hue (coarse = pale, fine = deep); a non-leaf level
      (e.g. ``overview``'s tiles fine branch) self-colours its own parts —
      the switch then reads as one uniform blob ⇄ a burst of multicoloured parts.
    * ``GSplatLeaf``      → a partition part carrying an additive ladder: paint its
      sub-LODs as concentric shades (coarse→fine) of the arc hue.
    """
    if isinstance(node, GSplatPartition):
        n = node.n_children
        lo, hi = arc
        children: list[GSplatNode] = []
        for i, part in enumerate(node.children):
            sub_arc = (lo + (hi - lo) * i / n, lo + (hi - lo) * (i + 1) / n)
            children.append(_paint_node(part, sub_arc))
        return GSplatPartition(
            children=children, max_elements=node.max_elements, meta=dict(node.meta)
        )
    if isinstance(node, GSplatLodGroup):
        m = node.n_children
        shades = _level_shades(_arc_mid(arc), m)  # index 0 = coarsest … m-1 = finest
        children = []
        for k, level in enumerate(node.children):  # k: 0 = coarsest … m-1 = finest
            if isinstance(level, GSplatLeaf):
                children.append(_recolor_leaf(level, shades[k]))
            else:
                children.append(_paint_node(level, arc))
        return GSplatLodGroup(
            children=children,
            meta=dict(node.meta),
        )
    if isinstance(node, GSplatLeaf):
        return _recolor_leaf_by_shades(node, _arc_mid(arc))
    return node  # pragma: no cover - unknown node type: leave untouched


def paint_recipe(result: GSplatData | GSplatNode) -> GSplatData | GSplatNode:
    """Paint a built recipe result of either return shape (see ``build_recipe``)."""
    if isinstance(result, GSplatData):
        return _paint_data(result)
    return _paint_node(result, (0.0, 1.0))


# =============================================================================
# Recipe building (the engine behind `luxar gsplat lod --recipe`)
# =============================================================================


def _params() -> RecipeParams:
    """Shared ``RecipeParams`` (from the module knobs) for every recipe build."""
    return RecipeParams(
        n_lods=N_LODS,
        additive_method=ADDITIVE_METHOD,  # type: ignore[arg-type]
        max_elements=MAX_ELEMENTS,
        compression_factor=FACTOR,
        levels=LEVELS,
        device=detect_device(),
        seed=0,
    )


def _cli_for(recipe: str) -> str:
    """The equivalent `luxar gsplat lod` command for pedagogy."""
    base = (
        "luxar gsplat lod tribolium.gsplats.zarr {0}.gsplats.zarr --recipe {0}".format(
            recipe
        )
    )
    if recipe == "stream":
        return base + f" --n-lods {N_LODS} --method {ADDITIVE_METHOD}"
    if recipe == "levels":
        return base + f" --compression-factor {FACTOR} --levels {LEVELS}"
    if recipe == "tiles":
        return base + f" --max-elements {MAX_ELEMENTS} --n-lods {N_LODS}"
    if recipe == "overview":
        return base + f" --max-elements {MAX_ELEMENTS} --compression-factor {FACTOR}"
    if recipe == "adaptive":
        return (
            base
            + f" --max-elements {MAX_ELEMENTS} --compression-factor {FACTOR}"
            + f" --levels {LEVELS}"
        )
    return base


def build_and_write(base: GSplatData, recipe: str, out_path: Path) -> dict:
    """Build one recipe, colour-code its structure, write the .gsplats.zarr.

    Mirrors ``cli/lod.py`` exactly: matrix recipes (flat/stream) round-trip
    through ``GSplatData.save``; the substitutive ``levels`` reduction and the
    composed recipes (tiles/overview/adaptive) write a node tree via
    ``write_gsplats_tree``. Returns a small stats dict for the legend.
    """
    with asection(f"lod --recipe {recipe}"):
        aprint(f"$ {_cli_for(recipe)}")
        # Build with the UNMODIFIED engine, then paint the result after the fact
        # (generic, type-driven — see ``paint_recipe``).
        built = build_recipe(base, recipe, _params())  # type: ignore[arg-type]

        # ``levels`` is the substitutive reduction: a matrix GSplatData carrying
        # one leaf per level (coarsest→finest). Its ``.tree`` is a kind=lod group,
        # so paint it through the node painter (each level a shade) and write it as
        # a tree — the viewer treats it as a kind=lod group either way.
        if recipe == "levels":
            node = _paint_node(built.tree, (0.0, 1.0))
            n_levels = built.n_substitutive
            stats = {
                "splats": built.at_substitutive(0).n_splats,
                "structure": f"{n_levels}-level substitutive (coarse↔fine swap)",
            }
            write_gsplats_tree(
                out_path,
                node,
                ordering="hilbert",
                encoding_mode=EncodingMode.PRECISION,
            )
            aprint(f"→ {stats['splats']:,} splats · {stats['structure']}")
            return stats

        result = paint_recipe(built)

        # Structure stats for the legend.
        if isinstance(result, GSplatData):
            stats = {
                "splats": result.flattened().n_splats,
                "structure": (
                    f"{result.n_additive_sublods}-level ladder"
                    if recipe == "stream"
                    else "single leaf"
                ),
            }
            result.save(
                out_path, ordering="hilbert", encoding_mode=EncodingMode.PRECISION
            )
        else:
            if recipe == "tiles":
                n = result.n_children
                structure = f"{n} BSP part{'s' if n != 1 else ''}, ladder each"
            elif recipe == "adaptive":
                n = result.n_children
                structure = f"{n} BSP part{'s' if n != 1 else ''}, substitutive each"
            else:  # overview
                # children are coarsest→finest: [coarse cap, fine partition].
                fine = result.children[-1]
                n_parts = fine.n_children if isinstance(fine, GSplatPartition) else 1
                structure = f"coarse cap + {n_parts}-part fine branch"
            stats = {"splats": total_splats(result), "structure": structure}
            write_gsplats_tree(
                out_path,
                result,
                ordering="hilbert",
                encoding_mode=EncodingMode.PRECISION,
            )
        aprint(f"→ {stats['splats']:,} splats · {stats['structure']}")
        return stats


# =============================================================================
# Scene composition (six columns, exactly as `gsplat convert` would graft them)
# =============================================================================

# Per-recipe legend descriptions + a representative legend colour.
_RECIPE_DESC = {
    "flat": ("single leaf — no LOD, no partition", "rgb(184,189,199)"),
    "stream": (
        "one leaf + prefix-sum ladder (streams coarse→fine)",
        "rgb(80,180,242)",
    ),
    "levels": (
        "substitutive levels — coarse↔fine replacement (one shown at a time)",
        "rgb(150,120,240)",
    ),
    "tiles": ("BSP parts, each its own additive ladder", "rgb(102,217,153)"),
    "overview": (
        "coarse cap (far) + tiles fine branch (near)",
        "rgb(255,77,82)",
    ),
    "adaptive": (
        "BSP parts, each its own substitutive lod (per-part swap)",
        "rgb(255,179,71)",
    ),
}


def create_luxar_scene(
    recipe_paths: dict[str, Path], recipe_stats: dict[str, dict], output_path: Path
) -> Path:
    """Compose the six recipe .gsplats.zarr files into one side-by-side scene."""
    with asection("Composing recipe-gallery scene"):
        aprint(f"Output: {output_path.name}")

        dims = Dimensions(
            [
                Dimension("x", unit="px", display=True),
                Dimension("y", unit="px", display=True),
                Dimension("z", unit="px", display=True),
            ]
        )
        # Column stride: ~1.5× the embryo extent so columns don't overlap.
        stride = float(EMBRYO_EXTENT) * 1.5
        n = len(RECIPES)

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                citation=DEMO_META["citation"],
                dimensions=dims,
                # No tone mapping at all (#1459). Colour here IS the encoding
                # (hue = which part, shade = which LOD level) and the whole
                # scene stays inside [0, 1], so a passthrough is exact: ACES
                # lifts highlights and shifts hue, and Neutral would subtract
                # its offset and compress from peak 0.76 up — flattening
                # exactly the shade steps the gallery is making.
                viewer_config=ViewerConfig(cinematic_mode=True, tone_mapping="None"),
            )
            scene.attrs["title"] = "GSplats: lod --recipe gallery — Tribolium Embryo"
            scene.attrs["description"] = """
GSplats LOD recipe gallery — Tribolium castaneum embryo (Light-Sheet)
=====================================================================

The same ~300K-splat fit, laid out left→right as the six `luxar gsplat lod
--recipe` topologies: flat → stream → levels → tiles → overview → adaptive.

Colour is applied AFTER the fact: hue = which spatial part, shade = which LOD
level — so multiple parts read as different hues and any level switch reads as a
shade jump.

- levels: substitutive coarse↔fine replacement — one level shown at a time, the
  swap reads as a single shade jump across the whole object.
- tiles: BSP spatial parts (each a distinct hue), each with its own additive
  ladder (its levels are shades of that hue) — off-screen parts cull, visible
  parts stream.
- overview: an unbalanced lod tree — a cheap coarse cap above a tiles fine
  branch. Detail only where you look: zoom out → one uniform pale blob (the cap);
  zoom in → it bursts into the multicoloured fine parts.
- adaptive: BSP parts (each a distinct hue) where every part is its OWN substitutive
  lod group — each cell culls AND picks its own level by its own on-screen size
  (the per-part swap shows as that part's shade jumping).

Pan across the row; zoom into a column to watch its level switch. Data: Cell
Tracking Challenge / Zenodo 5270323. Cite: Barry et al. 2022; Maska et al. 2023.
            """

            for i, recipe in enumerate(RECIPES):
                col_x = (i - (n - 1) / 2.0) * stride
                xform = transforms.translate(col_x, 0.0, 0.0)
                group = scene.add_group(f"recipe_{recipe}", transform=xform)
                group.add_gsplats_from_file(
                    recipe,
                    str(recipe_paths[recipe]),
                    opacity=1.0,
                    absorption=1.0,
                    blending_mode="volumetric",
                    layer=True,
                )

            # ── Overlays ────────────────────────────────────────────────────
            scene.add_text(
                "GSplats — lod --recipe gallery (Tribolium embryo)",
                position=(0.02, 0.02),
                font_size=0.044,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                "Light-sheet microscopy • one fit, six LOD topologies",
                DEMO_META.get("citation"),
            )

            # Legend: one tinted line per recipe (left→right), with its structure.
            spacing = 0.045
            start_y = 0.12
            for i, recipe in enumerate(RECIPES):
                desc, css = _RECIPE_DESC[recipe]
                st = recipe_stats[recipe]
                scene.add_text(
                    f"{i + 1}. {recipe}  —  {st['splats']:,} splats · {st['structure']}\n"
                    f"   {desc}",
                    position=(0.02, start_y + spacing * i),
                    font_size=0.018,
                    font="mono",
                    color=css,
                    stroke_color="black",
                    stroke_width=0.0016,
                    line_height=1.35,
                )

            scene.add_text(
                "Left → right: the scale ladder. tiles, overview & adaptive\n"
                "are the new BSP/unbalanced topologies. Colour code: hue = which\n"
                "part, shade = which LOD level — so a switch reads as a shade jump\n"
                "and multiple parts as different hues. In #5 the coarse cap shows as\n"
                "one uniform blob far out, bursting into the fine parts up close.",
                position=(0.02, 0.86),
                font_size=0.018,
                font="mono",
                color="white",
                width=0.5,
                line_height=1.4,
                background="rgba(0,0,0,0.55)",
                padding=0.012,
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# Module-level so create_luxar_scene can size the column stride (set in main()).
EMBRYO_EXTENT: float = 1.0


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Load the base fit, build all six recipes, and compose/serve the gallery."""
    aprint("=" * 70)
    aprint("GSplats Demo: lod --recipe gallery — Tribolium castaneum Embryo")
    aprint("=" * 70)
    aprint("One ~300K-splat fit → flat | stream | levels | tiles | overview | adaptive")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_recipes_tribolium.luxar.zarr"

    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    # Load the precomputed base fit (or re-fit on --recompute).
    precomputed = load_precomputed_gsplats(
        "gsplats_tribolium",
        ["tribolium.gsplats.zarr.zip"],
        recompute=RECOMPUTE,
    )
    if precomputed is not None:
        from luxar.demos.demo_gsplats_3d_tribolium_embryo import (
            warn_if_cached_tribolium_fit_predates_floor,
        )

        warn_if_cached_tribolium_fit_predates_floor()
        base = precomputed[0]
    else:
        warn_if_no_cuda_gpu()
        from luxar.demos.demo_gsplats_3d_tribolium_embryo import (
            fit_tribolium,
            load_tribolium_volume,
        )

        base = fit_tribolium(load_tribolium_volume())

    # Center at the intensity-weighted centroid + dim amplitudes (matches the
    # other gsplat demos) so each column sits at the origin before placement.
    base = base.translate(-base.centers.T @ base.amplitudes / base.amplitudes.sum())
    # 0.03 matches the sibling `demo_gsplats_3d_tribolium_embryo`. In volumetric
    # mode the amplitude drives BOTH the emission and the optical depth
    # (tau = kappa*opacity*rayMass), so a hotter scale doesn't just brighten —
    # it saturates the whole support into an opaque slab.
    base = base.scale_intensity(0.03)

    global EMBRYO_EXTENT
    EMBRYO_EXTENT = float(
        np.linalg.norm(base.centers.max(axis=0) - base.centers.min(axis=0))
    ) / np.sqrt(3.0)

    aprint(f"Base fit: {base.n_splats:,} splats ({base.ndim}D)")

    # Build each recipe into a temp .gsplats.zarr, then graft into one scene.
    with tempfile.TemporaryDirectory(prefix="luxar_recipes_") as tmp:
        tmpdir = Path(tmp)
        recipe_paths: dict[str, Path] = {}
        recipe_stats: dict[str, dict] = {}
        for recipe in RECIPES:
            p = tmpdir / f"{recipe}.gsplats.zarr"
            recipe_stats[recipe] = build_and_write(base, recipe, p)
            recipe_paths[recipe] = p

        with asection("Summary"):
            for recipe in RECIPES:
                st = recipe_stats[recipe]
                aprint(f"{recipe:12s} {st['splats']:>8,} splats · {st['structure']}")

        scene_path = create_luxar_scene(recipe_paths, recipe_stats, output_path)

    if not NO_SERVE:
        aprint("\nLaunching viewer...")
        aprint("Pan across the row; zoom into a column to watch its LOD switch.")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
