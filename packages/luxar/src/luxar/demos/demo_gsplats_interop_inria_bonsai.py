#!/usr/bin/env python3
"""GSplats Interop Demo: full-quality INRIA "bonsai" PLY (range-extracted) → Luxar

Downloads the FULL-QUALITY INRIA-trained *bonsai* scene — the reference 3D
Gaussian Splatting checkpoint at 30 000 iterations — and imports it into Luxar.
The twist: the official checkpoint lives inside a single 14.7 GB
``models.zip``, and this demo extracts just the ~309 MB ``point_cloud.ply``
member over HTTP Range requests — 2% of the archive — never touching the
other 14.4 GB.

WHY BONSAI AND NOT GARDEN. This demo used to pull the *garden* member, which
made it a near-duplicate of ``demo_gsplats_interop_mipnerf_garden``: the
antimatter15 ``garden.splat`` that demo downloads was produced FROM this very
checkpoint, and the two Luxar stores came out at the same 5,834,784 splats,
the same bounding box to three decimals, bit-identical amplitudes and ~80%
bit-identical centres. Two demos, one picture. Pointing this one at a
different member of the same archive keeps everything that was unique about
it — the INRIA PLY dialect at full training fidelity, and the Zip64 central
directory path that only an archive above 4 GiB exercises — while showing a
different scene, and costs 309 MB instead of 1.45 GB.

================================================================================
RANGE-EXTRACTED INRIA PLY → LUXAR, TILED PARTITION (BSP-ORDERED)
================================================================================

``luxar.demos.download_zip_member`` reads the remote zip's central
directory (Zip64-aware — the archive is >4 GB) via Range requests, then streams
and inflates only the requested member. The INRIA PLY is then imported (SH DC
term baked to color) and built as a ``tiles`` spatial BSP partition (per-tile
frustum culling + per-tile streaming ladder), the same structure the ``.splat``
Mip-NeRF demo builds but from the reference float32 checkpoint. The surface-like ``normal``
alpha-over compositing draws the tiles back-to-front via the partition's stored
BSP split planes — an exact painter's order, camera-inside-safe (see the
Mip-NeRF demo's docstring for the full rationale).

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
Range-extracts the ~309 MB ``point_cloud.ply`` (of a 14.7 GB archive) to
``~/.cache/luxar/gsplats_interop_inria/`` and caches the tiled
``.gsplats.zarr``. ``--recompute`` forces re-extract + re-import. Requires a
server honoring byte ranges (INRIA's does).

USAGE
-----
    python demo_gsplats_interop_inria_bonsai.py [--recompute] [--no-serve] [--serve-only]

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan
"""

DEMO_META = {
    "key": "gsplats_interop_inria_bonsai",
    "title": 'full-quality INRIA "bonsai" PLY (range-extracted) → Luxar',
    "description": "Full-quality INRIA 'bonsai' 3DGS checkpoint, range-extracted from a 14.7 GB Zip64 archive.",
    "category": "photogrammetry",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 309,
        "compute": "heavy",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["gsplats_interop_inria"],
    "outputs": ["gsplats_interop_inria_bonsai"],
    "citation": {
        "short": "Kerbl et al. 2023 (3D Gaussian Splatting); Barron et al. 2022 (Mip-NeRF 360)",
        "ref": "Kerbl / Barron et al. 2022–2023",
        "doi": "10.1145/3592433",
        "license": "Research / non-commercial (INRIA 3DGS)",
    },
}

from pathlib import Path

from arbol import Arbol, aprint, asection

from luxar.core.viewer_config import CameraConfig
from luxar.demos import (
    launch_viewer,
    parse_demo_flags,
    print_data_provenance,
)
from luxar.demos._cinematic_camera import pull_in
from luxar.demos._interop_common import build_gsplats_cache, build_interop_scene
from luxar.utils.paths import get_demos_output_dir

DEMO_NAME = "gsplats_interop_inria"
MODELS_ZIP = (
    "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
    "datasets/pretrained/models.zip"
)
# Member path + uncompressed size inside models.zip, read off the archive's LIVE
# central directory (117 members; the 30k-iteration point clouds run from
# counter at 303 MB to bicycle at 1.52 GB). The archive is 14.66 GB, so locating
# any member at all goes through the Zip64 end-of-central-directory path —
# that is a property of the ARCHIVE, not of the member, which is why a small
# member still exercises everything this demo exists to exercise.
MEMBER = "bonsai/point_cloud/iteration_30000/point_cloud.ply"
MEMBER_SIZE = 308_716_644  # bytes

CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME
CACHE_PLY = CACHE_DIR / "bonsai_point_cloud.ply"
CACHE_GSPLATS = CACHE_DIR / "bonsai.gsplats.zarr"

# Per-tile splat cap for the `tiles` BSP partition (see the Mip-NeRF demo).
MAX_ELEMENTS_PER_TILE = 1_000_000

# Open framed on the BONSAI TREE and orbit around it.
#
# Like every Mip-NeRF 360 capture this is an inside-out scene: the cameras
# circled one object inside a cluttered room, and sparse floater splats blow the
# bounding box out to roughly +/-27 on every axis. Bounding-sphere auto-framing
# therefore parks the camera outside the whole room, where the capture reads as
# a ball of white NeRF spikes with a rug somewhere inside it.
#
# `target` is the bonsai itself, not the scene centre and not the bbox centre.
# It was located from the DATA rather than by eye: the blossoms are the only
# strongly pink thing in the room, so selecting splats with
# `r > 0.45*max and r > 1.45*g and g < b < r` isolates 8,223 of the 1.24 M
# splats, whose trimmed median sits at (0.50, -0.80, -1.32) with an extent of
# only 1.1 x 0.8 x 1.1 units. That is the tree. (For contrast, the median of ALL
# centres is (-1.01, -1.60, -3.71) — the middle of the room, a metre and a half
# away, which framed the rug instead.) The target doubles as the orbit pivot, so
# mouse-drag turns around the tree.
#
# The distance, 5 units at the viewer's 47 degree default, is about four times
# the tree's own size: close enough that the bonsai is unambiguously the
# subject, far enough that the desk, window and rug still place it in a room.
# The scene enables `cinematic_mode`, whose preset expands a 35 mm lens (63
# degrees) because no `fov` is pinned here, so the pose is pulled in to hold the
# same framing. The pull-in is about the TARGET, which is far from the origin —
# scaling about the origin instead would swing the camera off the tree entirely.
BONSAI_TARGET = (0.5, -0.8, -1.3)
BONSAI_CAMERA = CameraConfig(
    position=pull_in((4.0, 0.45, 2.05), BONSAI_TARGET),
    target=BONSAI_TARGET,
)

FLAGS = parse_demo_flags()
Arbol.max_depth = 5


def fetch_member() -> Path:
    """Range-extract the bonsai PLY member from the remote models.zip."""
    from luxar.demos import download_zip_member

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with asection("Range-extracting bonsai PLY from models.zip"):
        aprint(f"Archive: {MODELS_ZIP} (14.7 GB)")
        aprint(f"Member:  {MEMBER} (~309 MB)")
        return download_zip_member(
            MODELS_ZIP,
            MEMBER,
            CACHE_PLY,
            expected_size=MEMBER_SIZE,
            max_retries=5,
            timeout=900,
        )


def build_scene() -> Path:
    """Range-extract + import the full-quality bonsai PLY (single-leaf stream)."""
    print_data_provenance(
        title="INRIA 3DGS bonsai (full-quality, 30k iters)",
        source="repo-sam.inria.fr pretrained models.zip",
        license="INRIA Gaussian-Splatting license — research/non-commercial "
        "(Mip-NeRF 360 scenes: Google research use)",
        url=MODELS_ZIP,
        note="Range-extracts ~309 MB of a 14.7 GB Zip64 archive; not redistributed.",
    )
    if not (CACHE_GSPLATS.exists() and not FLAGS["recompute"]):
        fetch_member()
    build_gsplats_cache(
        CACHE_PLY,
        CACHE_GSPLATS,
        recipe="tiles",
        recompute=FLAGS["recompute"],
        max_elements=MAX_ELEMENTS_PER_TILE,
        # Streaming ladder for fast first paint (NOT equal-count `n_lods`).
        # Each tile's coarsest additive chunk is ~14k splats — vs the
        # equal-count 1/4 = ~155k for each of bonsai's two ~620k tiles. First
        # frame decodes ~14k × (tiles in view) instead of the full 1.24M, so the
        # scene shows something almost immediately, then refines by geometric
        # doubling. Energy is heavily front-loaded (the first ~14k already
        # carry the bulk of the opacity-weighted energy), so the fast first
        # frame still looks essentially complete.
        breakpoints="stream:14000",
    )
    out = get_demos_output_dir() / "gsplats_interop_inria_bonsai.luxar.zarr"
    return build_interop_scene(
        CACHE_GSPLATS,
        out,
        title="INRIA 3DGS bonsai (full quality) → Luxar",
        layer_name="bonsai",
        credit="INRIA 3DGS (Kerbl 2023) • Mip-NeRF 360 • research use",
        camera=BONSAI_CAMERA,
        citation=DEMO_META["citation"],
    )


def main() -> None:
    """Build (or serve-only) the range-extracted INRIA bonsai scene."""
    aprint("=" * 70)
    aprint("GSplats Interop Demo: full-quality INRIA bonsai (range-extracted PLY)")
    aprint("=" * 70)

    out = get_demos_output_dir() / "gsplats_interop_inria_bonsai.luxar.zarr"
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
