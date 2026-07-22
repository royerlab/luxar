#!/usr/bin/env python3
"""GSplats Interop Demo: Astronomical observatories (SuperSplat compressed PLY) → Luxar

Downloads a Gaussian-splat capture of a real astronomical observatory — the
Vera C. Rubin Observatory by default (or Gemini South) — as a SuperSplat
*compressed* PLY and imports it into Luxar. These are CC BY 4.0 community
captures, so the demo is fully redistributable, and the subject is a natural
fit for a scientific-visualization project.

================================================================================
CLASSICAL GAUSSIAN SPLATS (SuperSplat compressed .ply) → LUXAR
================================================================================

SuperSplat's compressed PLY packs splats into 256-wide chunks with bit-packed
positions/rotations/scales/colors (11-10-11 positions, smallest-three
rotations, per-chunk min/max bounds). Luxar's importer dequantizes it against
the chunk bounds and lands oriented Gaussians directly.

DATA SOURCE & CITATION
----------------------
Observatory Gaussian-splat captures by khyron, released CC BY 4.0 on GitHub.
    https://github.com/khyron/Gaussian-Splatting  (releases: rubin, gemini-south)
Rubin Observatory / Gemini South are NSF–DOE / NSF NOIRLab facilities.

SELF-CONTAINED / CACHING
------------------------
Downloads the compressed PLY (~41 MB Rubin, ~83 MB Gemini South) to
``~/.cache/luxar/gsplats_interop_observatory/`` and caches the imported
``.gsplats.zarr``. ``--recompute`` forces re-download + re-import.
``--scene gemini-south`` selects the other observatory.

USAGE
-----
    python demo_gsplats_interop_observatory.py [--recompute] [--no-serve] [--scene rubin|gemini-south]

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan
"""

DEMO_META = {
    "key": "gsplats_interop_observatory",
    "title": "Astronomical observatories (SuperSplat compressed PLY) → Luxar",
    "description": "A Gaussian-splat capture of a real astronomical observatory (SuperSplat PLY) in Luxar.",
    "category": "photogrammetry",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 41,
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["gsplats_interop_observatory"],
    "outputs": [
        "gsplats_interop_observatory_rubin",
        "gsplats_interop_observatory_gemini-south",
    ],
}

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

DEMO_NAME = "gsplats_interop_observatory"
REPO = "https://github.com/khyron/Gaussian-Splatting/releases/download"
SCENES = {
    "rubin": {
        "url": f"{REPO}/1.0-rubin/rubin.compressed.ply",
        "file": "rubin.compressed.ply",
        "layer": "rubin_observatory",
        "title": "Vera C. Rubin Observatory",
    },
    "gemini-south": {
        "url": f"{REPO}/1.0-gemini-south/gemini_south.compressed.ply",
        "file": "gemini_south.compressed.ply",
        "layer": "gemini_south",
        "title": "Gemini South Telescope",
    },
}

FLAGS = parse_demo_flags()
Arbol.max_depth = 5


def _scene_arg(default: str = "rubin") -> str:
    for i, a in enumerate(sys.argv):
        if a == "--scene" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--scene="):
            return a.split("=", 1)[1]
    return default


def build_scene(scene_key: str = "rubin") -> Path:
    """Download + import one observatory compressed PLY and build its scene."""
    if scene_key not in SCENES:
        raise ValueError(f"Unknown scene {scene_key!r}; choose from {list(SCENES)}")
    spec = SCENES[scene_key]
    print_data_provenance(
        title=f"{spec['title']} (Gaussian splat)",
        source="khyron/Gaussian-Splatting (GitHub releases)",
        license="CC BY 4.0",
        url=spec["url"],
    )
    src = cached_download(spec["url"], DEMO_NAME, spec["file"])
    cache_file = src.with_name(f"{scene_key}.gsplats.zarr")
    # Moderate size → a stream ladder (fast first paint) is enough; no tiling.
    build_gsplats_cache(
        src, cache_file, recipe="stream", recompute=FLAGS["recompute"], n_lods=4
    )
    out = get_demos_output_dir() / f"gsplats_interop_observatory_{scene_key}.luxar.zarr"
    return build_interop_scene(
        cache_file,
        out,
        title=f"{spec['title']} — Gaussian splat → Luxar",
        layer_name=spec["layer"],
        credit=f"{spec['title']} • khyron • CC BY 4.0",
    )


def main() -> None:
    aprint("=" * 70)
    aprint("GSplats Interop Demo: Astronomical observatories (SuperSplat PLY)")
    aprint("=" * 70)

    scene_key = _scene_arg()
    out = get_demos_output_dir() / f"gsplats_interop_observatory_{scene_key}.luxar.zarr"
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
        aprint("Data credit: khyron observatory captures (CC BY 4.0)")
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
