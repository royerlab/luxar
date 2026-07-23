#!/usr/bin/env python3
"""GSplats Interop Demo: MatrixCity aerial (PlayCanvas SOG, ~13.6M) → Luxar LOD

Downloads the aerial "small city" split of the MatrixCity dataset — a
city-scale 3D Gaussian Splatting scene of **13,589,514 Gaussians**, published
on SuperSplat in PlayCanvas's compact **SOG** (Spatially Ordered Gaussians)
format — imports it through Luxar's new SOG reader, and builds a level-of-detail
topology so a 13.6M-splat city streams and renders in the web viewer. This is
the LARGEST scene in the interop set (≈45× the cluster fly) — a stress test for
Luxar's LOD at city scale.

================================================================================
SOG (WEBP-COMPRESSED 3DGS) → LUXAR, OVERVIEW LOD (COARSE SCAFFOLD + FINE TILES)
================================================================================

SOG stores a whole scene as a ``meta.json`` + a handful of lossless WebP images
(~15–20× smaller than PLY): 16-bit log-domain positions, codebook-indexed
scales/DC-color, and smallest-three quaternions. Luxar's importer decodes it
(baking SH-DC to per-splat RGB, dropping higher SH bands) into oriented
Gaussians.

**Reflecting the hierarchy the best possible way.** A city has no single
"right" detail level — you view the whole thing from above, then dive to a
street. That is exactly Luxar's ``overview`` recipe: one cheap **coarse global
level** (an instant, whole-city first paint — the analogue of hierarchical-GS's
*scaffold*) sitting above a **spatial BSP partition of fine tiles**, each tile
frustum-culled and carrying its own **streaming ladder** (progressive refinement
as you approach — the analogue of the per-*chunk* hierarchies). The viewer
swaps levels by viewport-relative coverage, so the 13.6M splats are never all
resident at once.

DATA SOURCE & CITATION
----------------------
MatrixCity (aerial, small city) — a synthetic city dataset for city-scale
neural rendering.
    Dataset: https://city-super.github.io/matrixcity/  (Li et al., ICCV 2023)
    3DGS reconstruction (FriendlySplat) published on SuperSplat:
    https://superspl.at/scene/ace6e5b0
The SOG bundle is fetched at runtime from SuperSplat's CDN for
research/education; Luxar redistributes nothing. Check the MatrixCity dataset
page for its license terms.

SELF-CONTAINED / CACHING
------------------------
Fetches the ~156 MB SOG bundle (``meta.json`` + 5 WebP images; the optional
higher-order-SH images are skipped — Luxar bakes DC only) to
``~/.cache/luxar/gsplats_interop_sog/`` and caches the LOD-built
``.gsplats.zarr`` next to it. ``--recompute`` forces re-import + re-LOD.

HEAVY: importing 13.6M splats and building the LOD needs several GB of RAM
and a few minutes; a workstation is recommended over a laptop.

USAGE
-----
    python demo_gsplats_interop_sog_matrixcity.py [--recompute] [--no-serve] [--serve-only]

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan
"""

DEMO_META = {
    "key": "gsplats_interop_sog_matrixcity",
    "title": "MatrixCity aerial (PlayCanvas SOG, ~13.6M) → Luxar LOD",
    "description": "MatrixCity aerial city: 13.6M-Gaussian PlayCanvas SOG scene imported with overview LOD.",
    "category": "photogrammetry",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 156,
        "compute": "heavy",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["gsplats_interop_sog_matrixcity"],
}

from pathlib import Path

from arbol import Arbol, aprint, asection

from luxar.demos._interop_common import build_gsplats_cache, build_interop_scene
from luxar.utils.demos import (
    cached_download,
    launch_viewer,
    parse_demo_flags,
    print_data_provenance,
)
from luxar.utils.paths import get_demos_output_dir

DEMO_NAME = "gsplats_interop_sog"
# SuperSplat/PlayCanvas CDN bundle for the MatrixCity aerial scene (id ace6e5b0).
BASE_URL = "https://d28zzqy0iyovbz.cloudfront.net/ace6e5b0/v1"
# Only the DC groups are needed — Luxar's SOG reader bakes SH-DC and drops the
# higher-order shN palette, so shN_centroids/shN_labels (~28 MB) aren't fetched.
SOG_FILES = (
    "meta.json",
    "means_l.webp",
    "means_u.webp",
    "scales.webp",
    "quats.webp",
    "sh0.webp",
)

CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME
BUNDLE_DIR = CACHE_DIR / "matrixcity_aerial"
CACHE_GSPLATS = CACHE_DIR / "matrixcity_aerial.gsplats.zarr"

# City-scale LOD: cap fine tiles at 500k splats (≈28 tiles for 13.6M) and put a
# 1/8-size coarse global level on top for the instant far view.
MAX_ELEMENTS_PER_TILE = 500_000
COARSE_COMPRESSION = 8

FLAGS = parse_demo_flags()
Arbol.max_depth = 5


def fetch_bundle() -> Path:
    """Download the SOG bundle files into a single cache directory."""
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    with asection("Fetching MatrixCity SOG bundle"):
        for name in SOG_FILES:
            cached_download(
                f"{BASE_URL}/{name}", f"{DEMO_NAME}/matrixcity_aerial", name
            )
    return BUNDLE_DIR


def build_scene() -> Path:
    """Fetch + import the MatrixCity SOG city and build its overview LOD."""
    print_data_provenance(
        title="MatrixCity aerial (small city) — 13.6M-splat 3DGS",
        source="MatrixCity (Li et al. 2023); 3DGS via FriendlySplat, hosted on SuperSplat",
        license="Research/education — see the MatrixCity dataset page",
        url="https://superspl.at/scene/ace6e5b0",
        note="SOG bundle fetched at runtime (~156 MB); not redistributed.",
    )
    if not (CACHE_GSPLATS.exists() and not FLAGS["recompute"]):
        fetch_bundle()
    # overview = coarse global scaffold + BSP-tiled fine detail, each tile
    # stream-laddered. The city-scale showcase for Luxar's LOD.
    build_gsplats_cache(
        BUNDLE_DIR,
        CACHE_GSPLATS,
        recipe="overview",
        recompute=FLAGS["recompute"],
        max_elements=MAX_ELEMENTS_PER_TILE,
        compression_factor=COARSE_COMPRESSION,
        # Streaming ladder → fast first paint: the COARSE CAP opens with a
        # ~14k-splat chunk (it keeps the raw user base — the first-paint
        # path); fine tiles under it get the sibling-raised base
        # max(14k, ceil(n/(2·K))) so an upgrade's committed prefix passes the
        # cap within 1-2 chunks. Refinement doubles geometrically from there.
        breakpoints="stream:14000",
    )
    out = get_demos_output_dir() / "gsplats_interop_sog_matrixcity.luxar.zarr"
    return build_interop_scene(
        CACHE_GSPLATS,
        out,
        title="MatrixCity aerial (13.6M splats, SOG) → Luxar overview LOD",
        layer_name="matrixcity",
        credit="MatrixCity • FriendlySplat/SuperSplat • SOG → Luxar • overview LOD",
    )


def main() -> None:
    aprint("=" * 70)
    aprint("GSplats Interop Demo: MatrixCity aerial (SOG, 13.6M) → Luxar LOD")
    aprint("=" * 70)

    out = get_demos_output_dir() / "gsplats_interop_sog_matrixcity.luxar.zarr"
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
        aprint("Data credit: MatrixCity (research/education) via SuperSplat SOG")
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
