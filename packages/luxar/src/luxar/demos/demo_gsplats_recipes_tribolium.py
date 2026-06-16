#!/usr/bin/env python3
"""GSplats Demo: LOD **recipe gallery** on a real Tribolium embryo (Light-Sheet)

Takes the ~256K-splat fit of the *Tribolium castaneum* embryo (the same
precomputed dataset as ``demo_gsplats_3d_tribolium_embryo.py``) and runs the
unified ``luxar gsplat lod --recipe`` pipeline to build the FOUR scale-ordered
representation topologies side by side, so you can compare them directly:

    flat   →   additive   →   partitioned   →   multiscale
   (small)    (medium)        (large)           (huge)

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
grafts the four results into one scene as labelled, colour-coded columns. The two
novel topologies are the point:

- **partitioned** — a spatial BSP ``kind=partition`` where EACH part carries its
  own additive LOD ladder. Off-screen parts frustum-cull; visible parts stream
  detail progressively. Here each BSP part is painted a distinct colour, so you
  can SEE the spatial cells.
- **multiscale** — an *unbalanced-by-design* ``kind=lod``: a single cheap coarse
  substitutive cap for the far/zoomed-out view, ABOVE a ``partitioned`` fine
  branch for close-up. Detail structure exists only where you look closely. Here
  the coarse cap is painted red ("far view") and the fine partition parts cool
  colours ("near detail").

For reference the two primitives bracket them: **flat** (one leaf, neutral grey)
and **additive** (one leaf + a prefix-sum ladder, painted on a blue→cyan ramp so
the coarse-first ordering is visible).

Pipeline:
1. **Load** the precomputed ~256K-splat Tribolium fit (Git LFS / local cache)
2. **Center** it so all four columns sit at the origin before placement
3. **Build** each recipe with ``build_recipe`` (the engine behind the CLI) and
   write each to a ``.gsplats.zarr`` — exactly what ``lod --recipe`` does
4. **Colour-code** the structure (per-part / cap-vs-fine / ladder ramp)
5. **Compose** one scene: four translated columns, each grafted via
   ``add_gsplats_from_file`` (exactly what ``gsplat convert`` does), with a legend
6. **Visualize** — pan across the row; zoom into a column to watch its LOD switch

DATA SOURCE & CITATIONS:
========================
Source:  Cell Tracking Challenge / Zenodo record 5270323 (GIANI paper)
Imaging: Zeiss LightSheet Z.1, Tribolium castaneum, 0.381 um isotropic
Cite:    Yin et al. (2022). GIANI. J. Cell Sci. 135(5), jcs259022.
         Maska et al. (2023). Cell Tracking Challenge. Nat. Methods 20, 1010-1020.

USAGE:
======
    python demo_gsplats_recipes_tribolium.py [--recompute] [--no-serve] [--serve-only]
                                             [--max-elements=N] [--factor=K]

Options:
    --recompute:       Force re-fitting the base splats from scratch (download + GPU)
    --no-serve:        Generate scene without launching viewer
    --serve-only:      Just serve a previously generated scene (skips rebuild)
    --max-elements=N:  Per-part BSP cap for partitioned/multiscale (default 50000)
    --factor=K:        Coarse-cap compression for multiscale (default 8)

Output:
    - Scene saved to:  datasets/demos/gsplats_recipes_tribolium.luxar.zarr
    - Automatically opens in browser
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core import transforms
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
from luxar.utils.demos import (
    detect_device,
    launch_viewer,
    load_precomputed_gsplats,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Per-part BSP cap for partitioned / multiscale. The base fit is ~256K splats;
# 50K → ~5 spatial parts, enough to see the cells and the frustum-culling story.
MAX_ELEMENTS = 50_000
# Coarse-cap compression for multiscale (one substitutive level ≈ N/FACTOR splats).
FACTOR = 8
# NB: the multiscale coarse↔fine switch uses the library-default selector anchor
# (count-derived ~10px), so the coarse cap engages as you zoom *out* (the embryo
# small on screen) and the fine, colour-coded parts show at the overview / up
# close. To make the coarse cap appear at a nearer zoom, pass
# RecipeParams(base_pixel_size=...) here (or `gsplat lod --base-pixel-size`).
# Additive ladder depth for additive / partitioned-part / multiscale-part ladders.
N_LODS = 4
# Cheap O(N log N) additive ordering — keeps the demo fast on CPU.
ADDITIVE_METHOD = "self_energy"

RECIPES = ("flat", "additive", "partitioned", "multiscale")

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
_GREY = (0.72, 0.74, 0.78)  # flat
_WARM_RED = (1.00, 0.30, 0.32)  # multiscale coarse cap (far view)
# Categorical palette for partition parts (distinct, readable).
_PART_PALETTE = [
    (0.20, 0.75, 0.95),  # cyan
    (0.40, 0.85, 0.45),  # green
    (1.00, 0.78, 0.20),  # amber
    (0.85, 0.45, 0.95),  # violet
    (0.95, 0.55, 0.30),  # orange
    (0.45, 0.60, 1.00),  # blue
    (0.95, 0.40, 0.65),  # pink
    (0.55, 0.90, 0.75),  # mint
]


def _color_array(rgb: tuple[float, float, float], n: int) -> np.ndarray:
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
    """A copy of ``leaf`` with every sub-LOD painted ``rgb``."""
    return GSplatLeaf(
        additive_sublods=[_recolor_sublod(s, rgb) for s in leaf.additive_sublods],
        meta=dict(leaf.meta),
    )


# =============================================================================
# Per-recipe structural colouring (makes each topology legible)
# =============================================================================


def recolor_flat(data: GSplatData) -> GSplatData:
    """Solid neutral grey — a single undifferentiated leaf."""
    flat = data.flattened()
    sub = flat.additive_sublods[0]
    return GSplatData.from_additive_sublods([_recolor_sublod(sub, _GREY)])


def recolor_additive(data: GSplatData) -> GSplatData:
    """Paint each additive sub-LOD on a blue→cyan ramp (coarse-first → fine)."""
    subs = data.additive_sublods
    n = len(subs)
    colored = []
    for i, sub in enumerate(subs):
        t = i / max(1, n - 1)
        rgb = (0.20 + 0.10 * t, 0.45 + 0.45 * t, 0.95)  # blue → cyan
        colored.append(_recolor_sublod(sub, rgb))
    return GSplatData.from_additive_sublods(colored)


def recolor_partition(node: GSplatPartition) -> GSplatPartition:
    """Paint each BSP part a distinct categorical colour (see the spatial cells)."""
    children: list[GSplatNode] = []
    for i, part in enumerate(node.children):
        rgb = _PART_PALETTE[i % len(_PART_PALETTE)]
        # to_spatial_partition yields leaf parts; recolor recursively to be safe.
        if isinstance(part, GSplatLeaf):
            children.append(_recolor_leaf(part, rgb))
        else:  # pragma: no cover - partitioned parts are leaves today
            children.append(part)
    return GSplatPartition(
        children=children, max_elements=node.max_elements, meta=dict(node.meta)
    )


def recolor_multiscale(node: GSplatLodGroup) -> GSplatLodGroup:
    """Coarse cap red ("far view"); fine partition parts cool colours ("near")."""
    fine, coarse = node.children  # finest→coarsest in memory
    fine_colored = (
        recolor_partition(fine) if isinstance(fine, GSplatPartition) else fine
    )
    coarse_colored = (
        _recolor_leaf(coarse, _WARM_RED) if isinstance(coarse, GSplatLeaf) else coarse
    )
    return GSplatLodGroup(
        children=[fine_colored, coarse_colored],
        default_level=node.default_level,
        meta=dict(node.meta),
    )


# =============================================================================
# Recipe building (the engine behind `luxar gsplat lod --recipe`)
# =============================================================================


def _params() -> RecipeParams:
    return RecipeParams(
        n_lods=N_LODS,
        additive_method=ADDITIVE_METHOD,  # type: ignore[arg-type]
        max_elements=MAX_ELEMENTS,
        compression_factor=FACTOR,
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
    if recipe == "additive":
        return base + f" --n-lods {N_LODS} --method {ADDITIVE_METHOD}"
    if recipe == "partitioned":
        return base + f" --max-elements {MAX_ELEMENTS} --n-lods {N_LODS}"
    if recipe == "multiscale":
        return base + f" --max-elements {MAX_ELEMENTS} --compression-factor {FACTOR}"
    return base


def build_and_write(base: GSplatData, recipe: str, out_path: Path) -> dict:
    """Build one recipe, colour-code its structure, write the .gsplats.zarr.

    Mirrors ``cli/lod.py`` exactly: matrix recipes (flat/additive) round-trip
    through ``GSplatData.save``; composed recipes (partitioned/multiscale) write
    the node tree via ``write_gsplats_tree``. Returns a small stats dict for the
    legend.
    """
    with asection(f"lod --recipe {recipe}"):
        aprint(f"$ {_cli_for(recipe)}")
        result = build_recipe(base, recipe, _params())  # type: ignore[arg-type]

        if recipe == "flat":
            result = recolor_flat(result)
        elif recipe == "additive":
            result = recolor_additive(result)
        elif recipe == "partitioned":
            result = recolor_partition(result)
        elif recipe == "multiscale":
            result = recolor_multiscale(result)

        # Structure stats for the legend.
        if isinstance(result, GSplatData):
            stats = {
                "splats": result.flattened().n_splats,
                "structure": (
                    f"{result.n_additive_sublods}-level ladder"
                    if recipe == "additive"
                    else "single leaf"
                ),
            }
            result.save(
                out_path, ordering="hilbert", encoding_mode=EncodingMode.PRECISION
            )
        else:
            if recipe == "partitioned":
                structure = f"{result.n_children} BSP parts, ladder each"
            else:  # multiscale
                fine = result.children[0]
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
# Scene composition (four columns, exactly as `gsplat convert` would graft them)
# =============================================================================

# Per-recipe legend descriptions + a representative legend colour.
_RECIPE_DESC = {
    "flat": ("single leaf — no LOD, no partition", "rgb(184,189,199)"),
    "additive": (
        "one leaf + prefix-sum ladder (streams coarse→fine)",
        "rgb(80,180,242)",
    ),
    "partitioned": ("BSP parts, each its own additive ladder", "rgb(102,217,153)"),
    "multiscale": (
        "coarse cap (far) + partitioned fine branch (near)",
        "rgb(255,77,82)",
    ),
}


def create_luxar_scene(
    recipe_paths: dict[str, Path], recipe_stats: dict[str, dict], output_path: Path
) -> Path:
    """Compose the four recipe .gsplats.zarr files into one side-by-side scene."""
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
            scene = compiler.create_scene(dimensions=dims)
            scene.attrs["title"] = "GSplats: lod --recipe gallery — Tribolium Embryo"
            scene.attrs["description"] = """
GSplats LOD recipe gallery — Tribolium castaneum embryo (Light-Sheet)
=====================================================================

The same ~256K-splat fit, laid out left→right as the four `luxar gsplat lod
--recipe` topologies: flat → additive → partitioned → multiscale.

- partitioned: BSP spatial parts (each a distinct colour), each with its own
  additive ladder — off-screen parts cull, visible parts stream.
- multiscale: an unbalanced lod tree — a cheap coarse cap (red, shown when far)
  above a partitioned fine branch (cool colours, shown up close). Detail only
  where you look.

Pan across the row; zoom into a column to watch its level switch. Data: Cell
Tracking Challenge / Zenodo 5270323. Cite: Yin et al. 2022; Maska et al. 2023.
            """

            for i, recipe in enumerate(RECIPES):
                col_x = (i - (n - 1) / 2.0) * stride
                xform = transforms.translate(col_x, 0.0, 0.0)
                group = scene.add_group(f"recipe_{recipe}", transform=xform)
                group.add_gsplats_from_file(
                    recipe,
                    str(recipe_paths[recipe]),
                    opacity=1.0,
                    blending_mode="additive",
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
            scene.add_text(
                "Light-sheet microscopy • one fit, four LOD topologies",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
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
                "Left → right: the scale ladder. partitioned & multiscale are the\n"
                "new BSP/unbalanced topologies — colours show the parts; the red\n"
                "cap in #4 is the cheap far-view level above the fine branch.",
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
    aprint("=" * 70)
    aprint("GSplats Demo: lod --recipe gallery — Tribolium castaneum Embryo")
    aprint("=" * 70)
    aprint("One ~256K-splat fit → flat | additive | partitioned | multiscale")
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
        ["tribolium_gsplats.gsplats.zarr.zip"],
        recompute=RECOMPUTE,
    )
    if precomputed is not None:
        base = precomputed[0]
    else:
        warn_if_no_cuda_gpu()
        import importlib.util

        sibling = Path(__file__).with_name("demo_gsplats_3d_tribolium_embryo.py")
        spec = importlib.util.spec_from_file_location("_tribolium_base", sibling)
        assert spec is not None and spec.loader is not None
        tri = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tri)
        base = tri.fit_tribolium(tri.load_tribolium_volume())

    # Center at the intensity-weighted centroid + dim amplitudes (matches the
    # other gsplat demos) so each column sits at the origin before placement.
    base = base.translate(-base.centers.T @ base.amplitudes / base.amplitudes.sum())
    base = base.scale_intensity(0.1)

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
