#!/usr/bin/env python3
"""GSplats Interop Demo: Scaniverse SPZ captures (Niantic) → Luxar

Downloads the two official Niantic SPZ sample scans — a horned lizard and a
raccoon-family sculpture, both phone captures from the Scaniverse app — and
imports them straight into Luxar as native ``.gsplats.zarr`` layers. This is
the classical-splat *import* path end to end: SPZ (gzipped, quantized) →
GSplatData → scene, no fitting involved.

================================================================================
CLASSICAL GAUSSIAN SPLATS (SPZ) → LUXAR
================================================================================

SPZ is Niantic/Scaniverse's compact open splat format (24-bit fixed-point
positions, quantized scales/colors/rotations). Luxar's importer decodes it,
bakes the spherical-harmonics DC term to per-splat RGB, and lands oriented
Gaussians ready for the viewer — which, once depth-sorted rendering lands,
composite exactly like they do in Scaniverse.

DATA SOURCE & CITATION
----------------------
Niantic Labs ``spz`` reference repository sample scans (MIT license).
    https://github.com/nianticlabs/spz  (samples/hornedlizard.spz, racoonfamily.spz)
Format: Scaniverse SPZ. Announced at https://scaniverse.com/spz .

SELF-CONTAINED / CACHING
------------------------
Downloads the two ``.spz`` files (~18 + 24 MB) to
``~/.cache/luxar/gsplats_interop_spz/`` and caches the imported
``.gsplats.zarr`` next to them; subsequent runs are instant. ``--recompute``
forces re-download + re-import.

USAGE
-----
    python demo_gsplats_interop_spz_scaniverse.py [--recompute] [--no-serve] [--serve-only]

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan
"""

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

DEMO_NAME = "gsplats_interop_spz"
BASE_URL = "https://raw.githubusercontent.com/nianticlabs/spz/main/samples"
SCENES = {
    "hornedlizard": {"file": "hornedlizard.spz", "layer": "horned_lizard"},
    "racoonfamily": {"file": "racoonfamily.spz", "layer": "raccoon_family"},
}

FLAGS = parse_demo_flags()
Arbol.max_depth = 5


def build_scene(scene_key: str = "hornedlizard") -> Path:
    """Download + import one SPZ scan and build its scene."""
    spec = SCENES[scene_key]
    print_data_provenance(
        title=f"Scaniverse SPZ — {scene_key}",
        source="Niantic Labs spz reference repo",
        license="MIT",
        url=f"{BASE_URL}/{spec['file']}",
    )
    src = cached_download(f"{BASE_URL}/{spec['file']}", DEMO_NAME, spec["file"])
    cache_file = src.with_suffix(".gsplats.zarr")
    # Small scans → a stream ladder gives fast first paint without tiling.
    build_gsplats_cache(
        src, cache_file, recipe="stream", recompute=FLAGS["recompute"], n_lods=3
    )
    out = get_demos_output_dir() / f"gsplats_interop_spz_{scene_key}.luxar.zarr"
    return build_interop_scene(
        cache_file,
        out,
        title=f"Scaniverse SPZ capture — {scene_key} (Niantic)",
        layer_name=spec["layer"],
        credit="Niantic spz samples • MIT • SPZ → Luxar",
    )


def main() -> None:
    aprint("=" * 70)
    aprint("GSplats Interop Demo: Scaniverse SPZ captures (Niantic)")
    aprint("=" * 70)

    scene_key = "hornedlizard"
    out = get_demos_output_dir() / f"gsplats_interop_spz_{scene_key}.luxar.zarr"
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
        aprint("Data credit: Niantic Labs spz samples (MIT)")
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
