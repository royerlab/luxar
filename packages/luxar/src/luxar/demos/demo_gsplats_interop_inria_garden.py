#!/usr/bin/env python3
"""GSplats Interop Demo: full-quality INRIA "garden" PLY (range-extracted) → Luxar

Downloads the FULL-QUALITY INRIA-trained *garden* scene — the reference 3D
Gaussian Splatting checkpoint at 30 000 iterations — and imports it into Luxar.
The twist: the official checkpoint lives inside a single 14.7 GB
``models.zip``, and this demo extracts just the ~1.45 GB ``point_cloud.ply``
member over HTTP Range requests, never downloading the other 13 GB.

================================================================================
RANGE-EXTRACTED INRIA PLY → LUXAR, SINGLE-LEAF FOR CORRECT DEPTH ORDER
================================================================================

``luxar.utils.download.download_zip_member`` reads the remote zip's central
directory (Zip64-aware — the archive is >4 GB) via Range requests, then streams
and inflates only the requested member. The INRIA PLY is then imported (SH DC
term baked to color) and built as a single ``stream`` leaf (one mesh +
progressive ladder), exactly like the ``.splat`` Mip-NeRF demo but at full
training fidelity. Single-leaf (not tiles) so the surface-like ``normal``
alpha-over compositing is one global per-splat depth sort — no tile-boundary
seams (see the Mip-NeRF demo's docstring for the full rationale).

DATA SOURCE & CITATION
----------------------
INRIA 3D Gaussian Splatting pre-trained models (Kerbl et al., SIGGRAPH 2023).
    https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/  (datasets/pretrained/models.zip)
Underlying capture: Mip-NeRF 360 (Barron et al., CVPR 2022). The 3DGS models
carry the INRIA Gaussian-Splatting license (RESEARCH / NON-COMMERCIAL); the
Mip-NeRF 360 scenes are Google research-use data. This demo fetches the member
at runtime for research/education; Luxar redistributes nothing.

SELF-CONTAINED / CACHING
------------------------
Range-extracts the ~1.45 GB ``point_cloud.ply`` (of a 14.7 GB archive) to
``~/.cache/luxar/gsplats_interop_inria/`` and caches the tiled
``.gsplats.zarr``. ``--recompute`` forces re-extract + re-import. Requires a
server honoring byte ranges (INRIA's does).

USAGE
-----
    python demo_gsplats_interop_inria_garden.py [--recompute] [--no-serve] [--serve-only]

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan
"""

from pathlib import Path

from arbol import Arbol, aprint, asection

from luxar.demos._interop_common import build_gsplats_cache, build_interop_scene
from luxar.utils.demos import (
    launch_viewer,
    parse_demo_flags,
    print_data_provenance,
)
from luxar.utils.paths import get_demos_output_dir

DEMO_NAME = "gsplats_interop_inria"
MODELS_ZIP = (
    "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
    "datasets/pretrained/models.zip"
)
# Member path + uncompressed size inside models.zip (verified against the
# archive's live central directory: the entry sits at byte offset ~5.6 GB, so
# extraction goes through the Zip64 path).
MEMBER = "garden/point_cloud/iteration_30000/point_cloud.ply"
MEMBER_SIZE = 1_447_027_964  # bytes

CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME
CACHE_PLY = CACHE_DIR / "garden_point_cloud.ply"
CACHE_GSPLATS = CACHE_DIR / "garden.gsplats.zarr"

FLAGS = parse_demo_flags()
Arbol.max_depth = 5


def fetch_member() -> Path:
    """Range-extract the garden PLY member from the remote models.zip."""
    from luxar.utils.download import download_zip_member

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with asection("Range-extracting garden PLY from models.zip"):
        aprint(f"Archive: {MODELS_ZIP} (14.7 GB)")
        aprint(f"Member:  {MEMBER} (~1.45 GB)")
        return download_zip_member(
            MODELS_ZIP,
            MEMBER,
            CACHE_PLY,
            expected_size=MEMBER_SIZE,
            max_retries=5,
            timeout=900,
        )


def build_scene() -> Path:
    """Range-extract + import the full-quality garden PLY (single-leaf stream)."""
    print_data_provenance(
        title="INRIA 3DGS garden (full-quality, 30k iters)",
        source="repo-sam.inria.fr pretrained models.zip",
        license="INRIA Gaussian-Splatting license — research/non-commercial "
        "(Mip-NeRF 360 scenes: Google research use)",
        url=MODELS_ZIP,
        note="Range-extracts ~1.45 GB of a 14.7 GB archive; not redistributed.",
    )
    if not (CACHE_GSPLATS.exists() and not FLAGS["recompute"]):
        fetch_member()
    build_gsplats_cache(
        CACHE_PLY,
        CACHE_GSPLATS,
        recipe="stream",
        recompute=FLAGS["recompute"],
        n_lods=6,
    )
    out = get_demos_output_dir() / "gsplats_interop_inria_garden.luxar.zarr"
    return build_interop_scene(
        CACHE_GSPLATS,
        out,
        title="INRIA 3DGS garden (full quality) → Luxar",
        layer_name="garden",
        credit="INRIA 3DGS (Kerbl 2023) • Mip-NeRF 360 • research use",
    )


def main() -> None:
    aprint("=" * 70)
    aprint("GSplats Interop Demo: full-quality INRIA garden (range-extracted PLY)")
    aprint("=" * 70)

    out = get_demos_output_dir() / "gsplats_interop_inria_garden.luxar.zarr"
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
        aprint("Data credit: INRIA 3DGS pretrained models (research/non-commercial)")
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
