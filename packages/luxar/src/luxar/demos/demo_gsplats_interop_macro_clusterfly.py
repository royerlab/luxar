#!/usr/bin/env python3
"""GSplats Interop Demo: Dany Bittel macro-photogrammetry cluster fly → Luxar

Downloads Dany Bittel's macro Gaussian-splat capture of a cluster fly
(*Pollenia*) — a real insect photographed under a macro rig and reconstructed
into a 3D splat — and imports it straight into Luxar as a native
``.gsplats.zarr`` layer. Same classical-splat *import* path as the Scaniverse
SPZ demo, but the subject is a millimetre-scale insect rather than a phone
capture: a "fly trapped in digital amber" you can orbit from any angle.

================================================================================
MACRO PHOTOGRAMMETRY 3DGS (INRIA-DIALECT PLY) → LUXAR
================================================================================

The capture is a Postshot-trained 3D Gaussian Splatting model exported as an
INRIA-dialect ``point_cloud.ply`` (SH degree 3). Luxar's importer decodes it,
bakes the spherical-harmonics DC term to per-splat RGB, applies the Y-down →
Y-up reorientation classical PLYs need, and lands oriented Gaussians ready for
the viewer. A ``stream`` ladder gives fast first paint; the surface-like
``normal`` (alpha-over) compositing renders the fly as a solid, occluding
object (depth-sorted since R10), exactly as it looks in the source viewer.

DATA SOURCE & CITATION
----------------------
Cluster fly (*Pollenia*), macro Gaussian splat by Dany Bittel.
    https://danybittel.ch/macro
    Archive: https://github.com/danybittel/splats/releases/download/splat/cluster.fly.zip
License: Creative Commons Attribution 4.0 (CC BY 4.0) — free for commercial and
non-commercial use with attribution. Credited in-scene to "Dany Bittel
(danybittel.ch)". This demo fetches the model at runtime; Luxar redistributes
nothing.

SELF-CONTAINED / CACHING
------------------------
Range-extracts just the ``cluster fly L.ply`` member (~68 MB, ~300k splats) out
of the ~1 GB release archive via HTTP Range requests — never downloading the
other detail levels — to ``~/.cache/luxar/gsplats_interop_macro/`` and caches
the imported ``.gsplats.zarr`` next to it. ``--recompute`` forces re-extract +
re-import. Requires a server honoring byte ranges (GitHub's asset CDN does).

USAGE
-----
    python demo_gsplats_interop_macro_clusterfly.py [--recompute] [--no-serve] [--serve-only]

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan
"""

DEMO_META = {
    "key": "gsplats_interop_macro_clusterfly",
    "title": "Dany Bittel macro-photogrammetry cluster fly → Luxar",
    "description": "Dany Bittel's macro-photogrammetry Gaussian-splat capture of a cluster fly, imported into Luxar.",
    "category": "photogrammetry",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 68,
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["gsplats_interop_macro"],
    "outputs": ["gsplats_interop_macro_clusterfly"],
}

from pathlib import Path

from arbol import Arbol, aprint, asection

from luxar.demos._interop_common import build_gsplats_cache, build_interop_scene
from luxar.utils.demos import (
    launch_viewer,
    parse_demo_flags,
    print_data_provenance,
)
from luxar.utils.paths import get_demos_output_dir

DEMO_NAME = "gsplats_interop_macro"
ARCHIVE_URL = (
    "https://github.com/danybittel/splats/releases/download/splat/cluster.fly.zip"
)
# Member path + uncompressed size inside cluster.fly.zip (verified against the
# archive's central directory). The archive ships five detail levels
# (S/M/L/XL/XXL); L is the hero balance — ~300k splats, detailed but streams
# fast with a ladder. XL/XXL are archival (0.6M / 3.5M splats).
MEMBER = "cluster fly L.ply"
MEMBER_SIZE = 71_263_622  # bytes

# CC BY 4.0 REQUIRES visible attribution. This is the in-scene text overlay
# (rendered in the viewer AND baked into exported scenes) — the actual
# artifact-level credit, distinct from the console-only provenance print. Kept
# as a module constant so the license test can assert on it directly.
CREDIT = "Cluster fly • Dany Bittel (danybittel.ch) • CC BY 4.0 • macro 3DGS → Luxar"

CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME
CACHE_PLY = CACHE_DIR / "cluster_fly_L.ply"
CACHE_GSPLATS = CACHE_DIR / "cluster_fly.gsplats.zarr"

# NOTE on framing: unlike the Scaniverse 360° object scans (dense core at the
# origin + sparse environment shell out at radius ~200, which defeats
# bounding-sphere auto-fit), this macro capture is JUST the fly — a compact,
# floater-free model ~0.4 units across. Auto-fit frames it correctly, so no
# fixed subject camera is needed here.

FLAGS = parse_demo_flags()
Arbol.max_depth = 5


def fetch_member() -> Path:
    """Range-extract the L-level PLY member from the remote release archive."""
    from luxar.utils.download import download_zip_member

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with asection("Range-extracting cluster fly PLY from release archive"):
        aprint(f"Archive: {ARCHIVE_URL} (~1 GB)")
        aprint(f"Member:  {MEMBER} (~68 MB)")
        return download_zip_member(
            ARCHIVE_URL,
            MEMBER,
            CACHE_PLY,
            expected_size=MEMBER_SIZE,
            max_retries=5,
            timeout=600,
        )


def build_scene() -> Path:
    """Range-extract + import the cluster fly, build an auto-framed scene."""
    print_data_provenance(
        title="Cluster fly (Pollenia) — macro Gaussian splat",
        source="Dany Bittel (danybittel.ch)",
        license="CC BY 4.0",
        url=ARCHIVE_URL,
        note="Range-extracts ~68 MB (L level) of a ~1 GB archive; not redistributed.",
    )
    if not (CACHE_GSPLATS.exists() and not FLAGS["recompute"]):
        fetch_member()
    # Small single object → a streaming ladder (geometric ~14k → doubling)
    # gives fast first paint, no tiling. The first chunk carries most of the
    # energy, so the fly shows almost immediately, then refines.
    build_gsplats_cache(
        CACHE_PLY,
        CACHE_GSPLATS,
        recipe="stream",
        recompute=FLAGS["recompute"],
        breakpoints="stream:14000",
    )
    out = get_demos_output_dir() / "gsplats_interop_macro_clusterfly.luxar.zarr"
    return build_interop_scene(
        CACHE_GSPLATS,
        out,
        title="Macro cluster fly (Dany Bittel) → Luxar",
        layer_name="cluster_fly",
        credit=CREDIT,
    )


def main() -> None:
    """Build (or serve-only) the macro cluster fly scene."""
    aprint("=" * 70)
    aprint("GSplats Interop Demo: macro cluster fly (Dany Bittel)")
    aprint("=" * 70)

    out = get_demos_output_dir() / "gsplats_interop_macro_clusterfly.luxar.zarr"
    if FLAGS["serve_only"]:
        if out.exists():
            launch_viewer(out)
        else:
            aprint(f"No scene at {out}. Run without --serve-only first.")
        return

    scene_path = build_scene()
    if FLAGS["no_serve"]:
        aprint(f"Dataset generated at {scene_path}")
    else:
        aprint("Data credit: Dany Bittel macro cluster fly (CC BY 4.0)")
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
