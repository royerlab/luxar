"""Shared scaffolding for the classical-splat interop demos.

The four ``demo_gsplats_interop_*`` demos all do the same thing — download a
classical Gaussian-splat file, import it to a ``GSplatData``, optionally add a
streaming/tiled LOD, cache the resulting ``.gsplats.zarr``, and build a
one-layer scene — differing only in the data source and LOD recipe. This
module holds that shared machinery so each demo file stays a thin, readable
manifest of *what* it downloads.

Not a demo itself (no ``demo_`` prefix), so the demo import smoke test skips it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig


def build_gsplats_cache(
    src: Path,
    cache_file: Path,
    *,
    recipe: Optional[str] = None,
    recompute: bool = False,
    **recipe_params: object,
) -> Path:
    """Import a classical splat file to a cached ``.gsplats.zarr``.

    With ``recipe`` set (``stream`` / ``tiles`` / …), the imported data is run
    through the LOD engine before caching so large scenes stream well. The
    cache is reused on subsequent runs unless ``recompute`` is set.

    Returns the path to the cached ``.gsplats.zarr``.
    """
    if cache_file.exists() and not recompute:
        # Existence-gated: recipe/knob changes in the demo (e.g. the #648
        # equal-count → stream:14000 ladder switch) do NOT retroactively
        # apply to an already-built cache — say so instead of silently
        # serving a stale recipe.
        aprint(
            f"✓ Cached import: {cache_file} "
            "(pass --recompute to rebuild with the current LOD recipe)"
        )
        return cache_file

    from luxar.gsplats.interop import import_gsplats

    with asection(f"Importing {src.name}"):
        data = import_gsplats(src)
        interop = data.stats.get("interop", {})
        aprint(
            f"✓ {data.n_splats:,} splats "
            f"({interop.get('source_format', '?')}, "
            f"SH degree {interop.get('source_sh_degree', 0)} → DC color)"
        )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if recipe is None:
        data.save(cache_file)
        return cache_file

    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.lod.recipes import RecipeParams, build_recipe

    with asection(f"Building '{recipe}' LOD"):
        built = build_recipe(data, recipe, RecipeParams(**recipe_params))  # type: ignore[arg-type]
    # Matrix recipes (flat/stream/levels) return a GSplatData; composed
    # recipes (tiles/overview/adaptive) return a node tree.
    from luxar.gsplats.gsplat_data import GSplatData

    if isinstance(built, GSplatData):
        built.save(cache_file)
    else:
        write_gsplats_tree(cache_file, built)
    return cache_file


def build_interop_scene(
    cache_file: Path,
    output_path: Path,
    *,
    title: str,
    layer_name: str,
    credit: str,
    colormap: Optional[str] = None,
    unit: str = "px",
    tone_mapping: str = "Neutral",
    intensity: float = 1.0,
    camera: Optional[CameraConfig] = None,
) -> Path:
    """Build a single-layer scene from a cached (possibly LOD'd) ``.gsplats.zarr``.

    Classical captures carry per-splat color, so ``colormap`` is normally
    ``None``. Imports render under ``blending_mode="normal"`` (alpha-over) —
    the surface-like, occluding look these photogrammetric scenes need, which
    composites correctly now that depth-sorted rendering has landed (R10). Both
    matrix and partition caches embed through ``add_gsplats_from_file``, which
    grafts whatever node shape the recipe produced.

    ``camera`` sets an initial viewer pose (overriding bounding-sphere
    auto-fit). Immersive 360° environment captures (e.g. Scaniverse room/yard
    scans) reconstruct the *whole surroundings* as a sphere, so auto-fit parks
    the camera outside looking at an opaque shell — pass a centered
    ``CameraConfig`` so the scene opens from inside, looking out.
    """
    with asection("Building scene"):
        dims = Dimensions([Dimension(a, unit=unit, display=True) for a in "xyz"])
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(tone_mapping=tone_mapping, camera=camera),
            )
            scene.attrs["title"] = title
            attrs: dict[str, object] = dict(
                blending_mode="normal", intensity=intensity, layer=True
            )
            if colormap is not None:
                attrs["colormap"] = colormap
            scene.add_gsplats_from_file(name=layer_name, path=str(cache_file), **attrs)
            scene.add_text(
                credit,
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.5)",
            )
        aprint(f"✓ Scene saved: {output_path}")
        return output_path
