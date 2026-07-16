#!/usr/bin/env python3
"""GSplats Interop Demo: Mip-NeRF 360 "garden" (antimatter15 .splat) → Luxar

Downloads the iconic Mip-NeRF 360 *garden* scene — a table with a potted plant
in a lush backyard, the de-facto "hello world" of 3D Gaussian Splatting — as an
antimatter15 ``.splat`` file and imports it into Luxar. At ~5 M splats it also
exercises the tiled-LOD path: the import is partitioned into spatially culled
tiles with streaming ladders so it loads smoothly in the web viewer.

================================================================================
CLASSICAL GAUSSIAN SPLATS (.splat) → LUXAR, TILED FOR STREAMING
================================================================================

The antimatter15 ``.splat`` format is flat 32-byte records (position, linear
scale, RGBA, quantized rotation). Luxar imports it, then `gsplat lod --recipe
tiles` (here via the Python engine) builds a spatial BSP of streaming tiles so
the viewer only fetches what the frustum needs.

DATA SOURCE & CITATION
----------------------
Mip-NeRF 360 dataset (Barron et al., CVPR 2022), trained to 3D Gaussian
Splatting (Kerbl et al., SIGGRAPH 2023) and hosted as ``.splat`` by the
antimatter15 web-splat project.
    https://huggingface.co/cakewalk/splat-data  (garden.splat, bicycle.splat)
Mip-NeRF 360 scenes are distributed by Google for RESEARCH USE. This demo
fetches them at runtime for research/education; Luxar redistributes nothing.

SELF-CONTAINED / CACHING
------------------------
Downloads the ~187 MB ``.splat`` to ``~/.cache/luxar/gsplats_interop_mipnerf/``
and caches the tiled ``.gsplats.zarr`` next to it. ``--recompute`` forces
re-download + re-import. ``--scene bicycle`` selects the bicycle scene instead.

USAGE
-----
    python demo_gsplats_interop_mipnerf_garden.py [--recompute] [--no-serve] [--scene garden|bicycle]

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan
"""

import sys
from pathlib import Path

from arbol import Arbol, aprint

from luxar.demos._interop_common import build_gsplats_cache, build_interop_scene
from luxar.utils.demos import (
    cached_download,
    launch_viewer,
    parse_demo_flags,
    print_data_provenance,
)
from luxar.utils.paths import get_demos_output_dir

DEMO_NAME = "gsplats_interop_mipnerf"
BASE_URL = "https://huggingface.co/cakewalk/splat-data/resolve/main"
SCENES = {
    "garden": {"file": "garden.splat", "size": 186_713_088},
    "bicycle": {"file": "bicycle.splat", "size": 196_222_528},
}
MAX_ELEMENTS_PER_TILE = 1_000_000  # ~5 M splats → a handful of streaming tiles

FLAGS = parse_demo_flags()
Arbol.max_depth = 5


def _scene_arg(default: str = "garden") -> str:
    for i, a in enumerate(sys.argv):
        if a == "--scene" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--scene="):
            return a.split("=", 1)[1]
    return default


def build_scene(scene_key: str = "garden") -> Path:
    """Download + import one Mip-NeRF .splat scene (tiled) and build its scene."""
    if scene_key not in SCENES:
        raise ValueError(f"Unknown scene {scene_key!r}; choose from {list(SCENES)}")
    spec = SCENES[scene_key]
    print_data_provenance(
        title=f"Mip-NeRF 360 — {scene_key} (3DGS .splat)",
        source="cakewalk/splat-data (antimatter15 web-splat)",
        license="Mip-NeRF 360 dataset — research use (Google)",
        url=f"{BASE_URL}/{spec['file']}",
        note="Downloaded at runtime for research/education; not redistributed.",
    )
    src = cached_download(
        f"{BASE_URL}/{spec['file']}",
        DEMO_NAME,
        spec["file"],
        expected_size=spec["size"],
    )
    cache_file = src.with_suffix(".gsplats.zarr")
    # ~5 M splats: tiles bound the viewer working set (a single stream ladder
    # would keep them all in one node).
    build_gsplats_cache(
        src,
        cache_file,
        recipe="tiles",
        recompute=FLAGS["recompute"],
        max_elements=MAX_ELEMENTS_PER_TILE,
        n_lods=4,
    )
    out = get_demos_output_dir() / f"gsplats_interop_mipnerf_{scene_key}.luxar.zarr"
    return build_interop_scene(
        cache_file,
        out,
        title=f"Mip-NeRF 360 {scene_key} — 3D Gaussian Splatting → Luxar",
        layer_name=scene_key,
        credit="Mip-NeRF 360 (Barron 2022) • 3DGS (Kerbl 2023) • research use",
    )


def main() -> None:
    aprint("=" * 70)
    aprint("GSplats Interop Demo: Mip-NeRF 360 (antimatter15 .splat)")
    aprint("=" * 70)

    scene_key = _scene_arg()
    out = get_demos_output_dir() / f"gsplats_interop_mipnerf_{scene_key}.luxar.zarr"
    if FLAGS["serve_only"]:
        if out.exists():
            launch_viewer(out)
        else:
            aprint(f"No scene at {out}. Run without --serve-only first.")
        return

    scene_path = build_scene(scene_key)
    if FLAGS["no_serve"]:
        aprint(f"Dataset generated at {scene_path}")
    else:
        aprint("Data credit: Mip-NeRF 360 (Google, research use) • 3DGS Kerbl 2023")
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
