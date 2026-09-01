#!/usr/bin/env python3
"""GSplats Demo: 4D Cell Tracking Challenge — a matrix of zebrafish embryos

Six crops of a developing zebrafish embryo, laid out as a 3x2 matrix, each one a
full 100-timepoint light-sheet timelapse showing three things at once:

  1. **GSplats** — the image data itself, fitted per timepoint and stacked into a
     single 4D (ZYX + time) Gaussian-splat volume
  2. **Points**  — the annotated position of every tracked cell at the current
     timepoint, coloured by lineage
  3. **Lines**   — each cell's whole trajectory through space and time, so the
     lineage structure stays visible while the volume animates underneath

Six full timelapses is a lot of geometry (~43k splats per timepoint x 100
timepoints x 6 crops, so ~26M splats), which is the point: each crop carries its
own **substitutive LOD** ladder, so the whole matrix stays affordable and zooming
into one tile is what pays for its finest level.

================================================================================
DATA SOURCE & CITATIONS
================================================================================

Source:  Kaggle competition "Biohub - Cell Tracking During Development"
URL:     https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
Licence: CC0 (annotations), Chan Zuckerberg Biohub San Francisco
Imaging: Royer Group light-sheet microscopy of zebrafish embryos

Each of the 199 training crops is an OME-Zarr **0.5** (zarr v3) store —
``T=100, Z=64, Y=256, X=256`` uint16, voxel 1.625 x 0.40625 x 0.40625 um, so a
104 um cube — paired with a **GEFF** tracking graph (``<crop>.geff``) holding the
ground-truth cell positions and lineage edges. This demo ranks the crops by
annotation density and shows the top six (~1,500-1,950 annotated cells each);
`DATASETS` carries nine, and `--datasets 9` uses them all.

The competition data needs Kaggle credentials, so this demo cannot download it
unattended. See the "Requirements" section below.

WORKFLOW
========

1. **Fetch** the crops from Kaggle (image store + GEFF graph, ~450 MB each)
2. **Fit** GSplats to every timepoint of every crop (cached per timepoint)
3. **Combine** each crop's timepoints into one 4D GSplatData (time as a
   coarsening barrier, so no LOD level ever blends across time)
4. **Ladder** each crop with a substitutive LOD pyramid
5. **Read** each crop's GEFF graph into lineage polylines + per-timepoint points
6. **Lay out** the crops on the most compact grid that fits, one group per crop
7. **Visualise** — scrub time, watch cells move along their tracks

USAGE
=====
    python demo_gsplats_4d_cell_tracking_challenge.py [options]

Options:
    --datasets=N      How many crops to show, 1..9 (default: 6). They are laid out
                      on the most compact grid that holds them: 4 -> 2x2,
                      6 -> 3x2, 9 -> 3x3. The default is 6 because only seven
                      crops have cleared Kaggle's download quota and seven fills
                      no rectangle; six does.
    --timepoints=N    Timepoints per crop (default: 100, the whole timelapse)
    --seeds=K         Splats per timepoint fit (default: 60000, keeping ~43k)
    --recompute       Re-fit from scratch, ignoring the fit cache
    --no-serve        Build the scene without launching the viewer
    --serve-only      Serve a previously built scene

REQUIREMENTS
============
    - The Kaggle client: ``luxar demo deps --only kaggle --install`` (it is
      declared in ``luxar.demos.INSTALL_SPECS`` but in no extra, since only this
      demo needs it), PLUS an API token, which no install can supply: create one
      at https://www.kaggle.com/settings ("API tokens") and save it as
      ``~/.kaggle/access_token`` or export ``KAGGLE_API_TOKEN``.
      Neither is needed once the fit cache and downloaded crop metadata are warm.
    - CUDA GPU strongly recommended: ~24 s per timepoint fit at the default
      budget on an RTX PRO 6000 (~40 min per crop). Apple MPS is roughly 6x
      slower, so a full crop there is measured in hours.
    - PyTorch (included in ``luxar[gsplats]``)

Output:
    - Scene saved to: datasets/demos/gsplats_4d_cell_tracking_challenge.luxar.zarr
    - Per-timepoint fits cached under ~/.cache/luxar/gsplats_cell_tracking/
"""

DEMO_META = {
    "key": "gsplats_4d_cell_tracking_challenge",
    "title": "4D Cell Tracking Challenge (matrix)",
    "description": (
        "Six zebrafish embryo timelapses in a 3x2 matrix: 4D gsplat volumes "
        "with annotated cell positions and lineage tracks."
    ),
    "category": "microscopy",
    "geometry": "mixed",
    "requirements": {
        "download_mb": 4000,
        "compute": "heavy",
        "gpu": "optional",
        "local_data": "kaggle-auth",
    },
    "caches": ["gsplats_cell_tracking"],
    "outputs": ["gsplats_4d_cell_tracking_challenge"],
    "citation": {
        "short": "CZ Biohub San Francisco; imaging by the Royer Group",
        "ref": "CZ Biohub / Royer Group",
        "license": "CC0 1.0",
        "url": "https://www.kaggle.com/competitions/biohub-cell-tracking-during-development",
    },
}

import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler, transforms
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    add_demo_caption,
    detect_device,
    hsv_to_rgb,
    launch_viewer,
    parse_demo_flags,
    parse_int_arg,
    require_module,
    warn_if_no_cuda_gpu,
)
from luxar.demos.registry import DEMO_CACHE_ROOT
from luxar.encoding import EncodingMode
from luxar.gsplats import GSplatData
from luxar.gsplats.merged_quality import collect_part_provenance
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

COMPETITION = "biohub-cell-tracking-during-development"
COMPETITION_URL = f"https://www.kaggle.com/competitions/{COMPETITION}"

# Everything this demo caches lives under the ONE directory it declares in
# DEMO_META["caches"], i.e. `~/.cache/luxar/gsplats_cell_tracking/`. That is what
# makes `luxar demo cache list` account for the ~4 GB of downloaded crops and
# `luxar demo cache clear` actually reclaim it — a path outside the managed root
# (registry.DEMO_CACHE_ROOT) would be invisible to both, and would be reported as
# an unclaimed orphan if it happened to land inside.
PRECOMPUTED_DATASET = "gsplats_cell_tracking"
CACHE_DIR = DEMO_CACHE_ROOT / PRECOMPUTED_DATASET
DATA_DIR = CACHE_DIR / "kaggle"  # raw competition downloads
FITS_DIR = CACHE_DIR / "fits"  # per-timepoint gsplat fits

# The nine most densely annotated training crops, by GEFF node count. Eight come
# from embryo `6bba` and one from `44b6` — the two source acquisitions — so the
# matrix shows both. Counts (annotated cells) are noted for orientation; they are
# what makes these crops worth showing rather than the 50-node sparse ones.
DATASETS: tuple[str, ...] = (
    "6bba_09961292",  # 1950 nodes
    "6bba_bb9f20c3",  # 1925
    "6bba_784a78c9",  # 1788
    "6bba_57b7cc1e",  # 1659
    "6bba_cff5865f",  # 1571
    "6bba_5dfe9ad1",  # 1542
    "6bba_ae82a791",  # 1495
    "6bba_786893ac",  # 1465
    "44b6_d29c9ab2",  # 1353 — the other embryo
)

N_TIMEPOINTS = 100  # every crop is exactly 100 timepoints

# Splats per timepoint, requested; the fit's own culling keeps ~70%.
#
# `gsplat cal` put the held-out PSNR peak at K* = 12,126 and that is genuinely
# where the NOISE-masked metric turns over — but it is the wrong criterion for
# this data. Measured on one timepoint, against a scale-invariant PSNR over the
# nuclei mask (voxels > p90) rather than the whole mostly-dim stack:
#
#     K        kept splats   nuclei PSNR
#     12,126   9,429         37.41 dB
#     30,000   21,711        38.12 dB   (+0.71)
#     60,000   42,596        39.29 dB   (+1.88)
#     120,000  83,318        40.29 dB   (+2.87)
#
# About +1 dB per doubling with NO plateau, and it is plainly visible in a MIP:
# at 9.4k the nuclei are smooth blobs with the gaps between them filled in, and
# by 42k the boundaries and the dark gaps are back. A global average hides this
# because these crops are mostly dim tissue, so the easy background dominates it.
# 60,000 is the chosen balance — most of the visible gain for half the fit time
# and storage of the next rung up.
DEFAULT_SEEDS = 60000

# Grid pitch as a multiple of one crop's 104 um extent — a small gap so the tiles
# read as separate embryos rather than one slab.
GRID_GAP_FACTOR = 1.14

# Substitutive LOD. The ladder's DEPTH is this demo's single most important
# performance knob, and it is set from a measurement rather than a preference.
#
# The ladder is stamped `selector="screen-area"`: the finest level is anchored at
# half the screen and each step down is a halving, so a 4-level ladder gets
# thresholds [0, 0.125, 0.25, 0.5]. A tile of the matrix occupies a small
# fraction of that, so it draws — and therefore FETCHES — a coarse level, and the
# ladder's depth is exactly "how much of the fit the opening framing keeps".
#
# GRID SHAPE MOVES THIS. A 3x3 tile sits around 0.05 of the screen area, well
# under the lowest threshold; a 3x2 tile is roughly 0.12, which is close enough
# to the 0.125 boundary that it can select one level FINER depending on viewport
# aspect. That is the selector behaving correctly — fewer, bigger tiles deserve
# more detail — but it means changing the grid changes the per-frame cost, so
# check `__luxarDebug.getState().lodGroups[].activeLevel` after doing so rather
# than assuming the numbers below still apply.
#
# Measured on one timepoint by merging the fit down to each candidate count and
# rendering it back (see the MIP comparison that produced these numbers):
#
#   drawn/tp   x6 crops   verdict
#    1,105        6,630   smears; nuclei merge together (this was the old blur)
#    2,500       15,000   nucleus separation returning
#    5,000       30,000   clear boundaries and dark gaps  <-- knee
#   10,000       60,000   marginally crisper
#   20,000      120,000   barely distinguishable from 10k, twice the cost
#
# 3 coarser levels puts the drawn count at ~N/8 ~= 5.1k/timepoint: the knee, and
# a 4x cut in per-frame fetch+draw against the 1-level ladder that made scrubbing
# choppy. Measured in the viewer at the shipped 3x2: every group reports
# activeLevel 0 (coarsest), 29,774 splats drawn per frame across six crops. Note this is NOT a return to the old blur despite a similar drawn
# count — a level merged from the 60k fit is visibly better than 1.1k splats
# fitted directly, which is why the fit budget stays high even though most of it
# is only unpacked when you zoom into a tile.
LOD_COMPRESSION_FACTOR = 2
LOD_LEVELS = 3

# Additive (progressive streaming) sub-ladders INSIDE each substitutive level are
# on by default everywhere else, and are deliberately off here. In a 4D stacked
# ladder every sub-LOD spans all 100 timepoints, so the time slice keeps only
# ~1% of whichever prefix has arrived — a level's committed prefix therefore looks
# almost empty to the viewer no matter how much of it has downloaded. Bare levels
# swap cleanly instead. Flip to True to see the difference.
ADDITIVE_LADDERS = False

# Percentile of characteristic splat size above which splats are dropped as
# diffuse background (see combine_to_4d for why this matters so much here).
SCALE_MAX_PERCENTILE = 95.0

# Volume appearance, tuned in the viewer's Layers panel against the built matrix
# and applied to every crop.
#
# The two knobs work together, and the pairing is not the intuitive one. My first
# guess was to hold opacity at 1.0 and raise absorption for occlusion; sweeping
# the live uniforms showed that just trades a flat bright mass (absorption 1) for
# a near-invisible front shell (absorption 8). What actually reads is the
# opposite corner: a LOW per-splat opacity so ~50 nuclei along a ray accumulate
# instead of saturating, with only mild absorption for depth.
VOLUME_OPACITY = 0.11
VOLUME_ABSORPTION = 0.38

# Display-range window: which slice of the amplitude range the colormap spans.
# The panel exposes it as [min, max] and stores it as the shader pair
# ``intensity = 1 / (max - min)``, ``offset = -min / (max - min)``; the viewer
# recovers the window from them (ui/layers/layer-state.ts), so authoring the pair
# reproduces a tuned look instead of re-defaulting to the full range.
#
# DERIVED per crop rather than hard-coded, because the amplitude distribution
# moves with the splat budget: the same signal split across 4.5x more splats
# leaves each one dimmer, and the measured p99 fell from 0.550 (K=12,126) to
# 0.400 (K=60,000). A window frozen at the old numbers would spend its top third
# on empty amplitude and render everything darker.
#
# The tuned window's top (0.562) turned out to sit exactly on the finest level's
# 99th percentile (0.5617), so that is the rule: span from the bottom of the data
# up to p99. It reproduces the hand-tuned window to three decimals at the budget
# it was tuned on, and follows K and per-crop content on its own.
VOLUME_WINDOW_TOP_PERCENTILE = 99.0


def display_window(amplitudes: np.ndarray) -> tuple[float, float]:
    """The ``(intensity, offset)`` shader pair for this node's amplitude window.

    Spans ``[min, p99]``: flooring at the data's own minimum wastes no window
    below it, and clipping at p99 keeps the brightest few splats from taking the
    whole top of the colormap and leaving the nuclei dim.
    """
    lo = float(np.min(amplitudes))
    hi = float(np.percentile(amplitudes, VOLUME_WINDOW_TOP_PERCENTILE))
    if not hi > lo:  # degenerate (uniform amplitudes) — identity window
        return 1.0, 0.0
    span = hi - lo
    return 1.0 / span, -lo / span


# Nuclei in these crops are ~8 um across. A marker at ~1/6 of that flags "this
# cell is tracked" as a dot ON the nucleus rather than a ball covering it — at
# half the nucleus width the markers dominated the matrix framing and hid the very
# splats they annotate.
CELL_MARKER_RADIUS_UM = 1.3

# Tracks are context, not the subject: at 0.7 um and fully opaque they read as a
# solid cage over the embryo, and in a matrix (each tile a small share of the
# screen)
# that cage is most of what you see. Thin enough to sit between nuclei — median
# nearest-neighbour spacing is 1.84 um — and translucent enough that the splatted
# volume shows through where trajectories bundle.
TRACK_WIDTH_UM = 0.35
TRACK_OPACITY = 0.15

FLAGS = parse_demo_flags()

# Six, not nine: a COMPLETE 3x2 rectangle beats a 3x3 with holes in it. The last
# two of the nine ranked crops are stuck behind Kaggle's per-account download
# quota, and seven tiles cannot fill any rectangle either. Since DATASETS is
# ordered by annotation density, `[:6]` drops the sparsest available crop and
# keeps the six best. Raise it with `--datasets 9` once the other two are in hand.
DEFAULT_N_DATASETS = 6
N_DATASETS = parse_int_arg("datasets", DEFAULT_N_DATASETS)
TIMEPOINTS = parse_int_arg("timepoints", N_TIMEPOINTS)
SEEDS = parse_int_arg("seeds", DEFAULT_SEEDS)

_DEVICE: str | None = None


# =============================================================================
# Kaggle download
# =============================================================================


def _kaggle_cmd() -> list[str]:
    """The Kaggle client invocation, plus a token check.

    Invoked as ``<this interpreter> -m kaggle`` rather than by looking for a
    ``kaggle`` binary on PATH: the console script lands in whichever environment
    pip installed it into, which is routinely not the one running the demo.

    ``require_module`` is what makes ``luxar demo deps`` report and install the
    client. The token is checked separately because no install can supply it —
    importing kaggle without one succeeds, and the failure would otherwise
    surface as an opaque 401 on the first download.
    """
    require_module("kaggle")

    has_token = bool(os.environ.get("KAGGLE_API_TOKEN")) or any(
        (
            Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / name
        ).is_file()
        for name in ("access_token", "kaggle.json")
    )
    if not has_token:
        raise RuntimeError(
            "The Biohub cell-tracking competition data is behind an "
            "authenticated endpoint, so this demo needs a Kaggle API token.\n"
            "  1. create one at https://www.kaggle.com/settings ('API tokens')\n"
            "  2. save it as ~/.kaggle/access_token, or export KAGGLE_API_TOKEN\n"
            f"  3. accept the competition rules at {COMPETITION_URL}\n"
            "A warm fit cache needs none of this — see the module docstring."
        )
    return [sys.executable, "-m", "kaggle"]


# =============================================================================
# Precomputed crops (the hosted fast path)
# =============================================================================


def precomputed_file_names(dataset: str) -> tuple[str, str]:
    """The two hosted files for one crop: its LOD'd volume, and its tracks."""
    return f"{dataset}.gsplats.zarr.zip", f"{dataset}_tracks.npz"


def load_precomputed_crops(
    chosen: Sequence[str],
    *,
    manifest: Optional[dict] = None,
    cache_root: Optional[Path] = None,
) -> Optional[list[dict]]:
    """Build the crop list from the hosted derived product, or None to compute.

    This is the path that makes the demo runnable with **no Kaggle credentials
    and no GPU**: the manifest dataset holds, per crop, the finished 4D
    Gaussian-splat volume (LOD ladder included) plus the track geometry, so
    nothing has to be downloaded from the authenticated competition endpoint and
    nothing has to be fitted.

    Returns ``None`` — meaning "take the local fetch-and-fit path" — when:

    * ``--recompute`` was passed (:func:`ensure_dataset` raises
      :class:`LocalComputeDataset` for any dataset in that case), or
    * the dataset is not hosted yet. The manifest carries it as
      ``pending_upload``, so until the bytes are on Zenodo there is nothing to
      resolve; that is an expected state during the migration rather than a
      fault, and it is reported rather than swallowed.
    * the hosted record does not carry one of the CHOSEN crops. That is the same
      condition one crop at a time — ``--datasets N`` asks for the first N of a
      list the record may only partly cover — so it is routed the same way, and
      announced by name. It is not detectable as a fault: a partial record is
      exactly what a migration in progress looks like.

    Anything else — a checksum that will not verify, a missing packaged
    manifest, a ``PRECOMPUTED_DATASET`` the manifest does not know
    (:class:`DatasetNotFound`) — raises, because those are faults a demo must
    not route around. The fallback here is not cheap: it downloads from the
    authenticated Kaggle endpoint and fits every chosen crop on the GPU, so
    disguising a broken install as "the data is not published yet" costs a user
    many minutes and an account they may not have.

    ``manifest`` / ``cache_root`` exist for tests, mirroring
    :func:`~luxar.demos.ensure_dataset`.
    """
    from luxar.demos import DatasetUnavailable, LocalComputeDataset, ensure_dataset

    file_names = {
        name for dataset in chosen for name in precomputed_file_names(dataset)
    }
    try:
        paths = ensure_dataset(
            PRECOMPUTED_DATASET,
            file_names=file_names,
            recompute=FLAGS["recompute"],
            manifest=manifest,
            cache_root=cache_root,
        )
    except LocalComputeDataset:
        return None
    except DatasetUnavailable as exc:
        # The one routable condition: nothing anywhere holds these bytes yet.
        # `DatasetNotFound` is deliberately NOT caught — a dataset key the
        # manifest does not carry is a typo or a rename, i.e. a fault.
        aprint(
            f"Precomputed crops unavailable ({exc}); falling back to the Kaggle "
            "download and local fit (already-fitted timepoints are reused)."
        )
        return None

    by_name = {p.name: p for p in paths}
    crops: list[dict] = []
    for dataset in chosen:
        volume_name, tracks_name = precomputed_file_names(dataset)
        if volume_name not in by_name or tracks_name not in by_name:
            aprint(
                f"Hosted dataset has no entry for {dataset}; falling back to the "
                "Kaggle download and local fit (already-fitted timepoints are reused)."
            )
            return None
        crops.append(
            _crop_from_precomputed(dataset, by_name[volume_name], by_name[tracks_name])
        )
    return crops


def _crop_from_precomputed(dataset: str, volume: Path, tracks: Path) -> dict:
    """Rehydrate one crop dict from its two hosted files.

    Marker radius and track width are re-derived from the constants rather than
    read back from the payload. Both are uniform (``np.full`` of one constant),
    so they carry no measured information — but stored, they would make retuning
    the look require re-uploading the whole hosted set. Only the geometry that
    was actually derived from the data is loaded.
    """
    lod = GSplatData.load(volume, include_stats=False)
    with np.load(tracks) as npz:
        payload = {k: npz[k] for k in npz.files}

    n_timepoints = int(payload["n_timepoints"][0])
    intensity, offset = display_window(lod.amplitudes)
    # A crop with no usable annotation is stored as empty arrays (see
    # save_precomputed_crop), and must rehydrate as `None` rather than as a dict
    # of empty geometry — that is what the scene builder reads as "volume only".
    has_tracks = len(payload["line_vertices"]) and len(payload["line_indices"])
    return {
        "name": dataset,
        "lod": lod,
        "n_splats": int(lod.n_splats),
        "intensity": intensity,
        "offset": offset,
        "tracks": {
            "point_positions": payload["point_positions"],
            "point_colors": payload["point_colors"],
            "point_radii": np.full(
                len(payload["point_positions"]), CELL_MARKER_RADIUS_UM, np.float32
            ),
            "line_vertices": payload["line_vertices"],
            "line_colors": payload["line_colors"],
            "line_widths": np.full(
                len(payload["line_vertices"]), TRACK_WIDTH_UM, np.float32
            ),
            "line_indices": payload["line_indices"],
            "n_lineages": int(payload["n_lineages"][0]),
            "n_cells": int(payload["n_cells"][0]),
            "n_divisions": int(payload["n_divisions"][0]),
        }
        if has_tracks
        else None,
        "extent_um": float(payload["extent_um"][0]),
        "n_timepoints": n_timepoints,
    }


def save_precomputed_crop(crop: dict, out_dir: Path, n_timepoints: int) -> list[Path]:
    """Write one crop's hosted pair — used by ``scripts/build_cell_tracking_bundle.py``.

    Kept beside the loader so the two halves of the format cannot drift.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    volume_name, tracks_name = precomputed_file_names(crop["name"])
    volume_path = out_dir / volume_name
    crop["lod"].save(
        volume_path,
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )

    tracks = crop["tracks"] or {}
    payload = {
        "extent_um": np.asarray([crop["extent_um"]], dtype=np.float64),
        "n_timepoints": np.asarray([n_timepoints], dtype=np.int64),
    }
    for key in (
        "point_positions",
        "point_colors",
        "point_radii",
        "line_vertices",
        "line_colors",
        "line_widths",
        "line_indices",
    ):
        payload[key] = np.asarray(tracks.get(key, np.zeros((0,), np.float32)))
    for key in ("n_lineages", "n_cells", "n_divisions"):
        payload[key] = np.asarray([int(tracks.get(key, 0))], dtype=np.int64)

    tracks_path = out_dir / tracks_name
    np.savez_compressed(tracks_path, **payload)
    return [volume_path, tracks_path]


# =============================================================================
# Kaggle download
# =============================================================================


def _crop_metadata_files(dataset: str) -> list[str]:
    """The small files every run needs: the GEFF graph, and the image METADATA.

    Spelled out rather than listed from the API because the full file listing is
    ~25,000 rows (one per chunk across 199 crops) and paging it costs far more
    requests than fetching the ~120 files we actually want.

    The image ``zarr.json`` pair is in here rather than with the chunks because the
    scene needs the crop's shape and voxel size — to place the tile and to convert
    the GEFF's voxel coordinates to um — even when every timepoint is already
    fitted and no pixel is ever read.
    """
    geff, img = f"train/{dataset}.geff", f"train/{dataset}.zarr"
    paths = [
        f"{geff}/zarr.json",
        f"{geff}/nodes/zarr.json",
        f"{geff}/nodes/ids/zarr.json",
        f"{geff}/nodes/ids/c/0",
        f"{geff}/nodes/props/zarr.json",
        f"{geff}/edges/zarr.json",
        f"{geff}/edges/ids/zarr.json",
        f"{geff}/edges/ids/c/0/0",
        f"{geff}/edges/props/zarr.json",
    ]
    for axis in ("t", "z", "y", "x"):
        paths += [
            f"{geff}/nodes/props/{axis}/zarr.json",
            f"{geff}/nodes/props/{axis}/values/zarr.json",
            f"{geff}/nodes/props/{axis}/values/c/0",
        ]
    paths += [f"{img}/zarr.json", f"{img}/0/zarr.json"]
    return paths


def _crop_chunk_files(dataset: str, n_timepoints: int) -> list[str]:
    """The image chunks — one per timepoint, ~4.5 MB each. Only fitting reads these."""
    return [f"train/{dataset}.zarr/0/c/{t}/0/0/0" for t in range(n_timepoints)]


def _is_downloaded(path: Path) -> bool:
    """Whether a competition file is present AND non-empty.

    The size test is the point: an interrupted transfer can leave a 0-byte file,
    and treating that as present would skip it in the scan below and then fail
    when zarr tried to read it.
    """
    return path.is_file() and path.stat().st_size > 0


def _fetch_file(cmd: list[str], rel: str, dest_root: Path, max_tries: int = 6) -> bool:
    """Download one competition file to ``dest_root/rel``.

    ``kaggle competitions download -f`` flattens the download to the file's
    basename, so each file goes into its own destination directory to rebuild the
    zarr tree. Kaggle rate-limits bulk single-file fetches with HTTP 429, so a
    429 backs off exponentially rather than failing the run.
    """
    target = dest_root / rel
    if _is_downloaded(target):
        return True
    target.parent.mkdir(parents=True, exist_ok=True)

    wait = 8.0
    for attempt in range(1, max_tries + 1):
        proc = subprocess.run(
            [
                *cmd,
                "competitions",
                "download",
                COMPETITION,
                "-f",
                rel,
                "-p",
                str(target.parent),
                "-q",
                "-o",
            ],
            capture_output=True,
            text=True,
        )
        if _is_downloaded(target):
            return True
        blob = (proc.stdout or "") + (proc.stderr or "")
        if attempt == max_tries:
            aprint(f"  could not fetch {rel}: {blob.strip()[:200]}")
            return False
        throttled = "429" in blob or "Too Many Requests" in blob
        if throttled:
            aprint(f"  Kaggle throttled (429) — waiting {wait:.0f}s")
        time.sleep(wait if throttled else 2.0)
        wait = min(wait * 2, 300.0)
    return False


def fetch_dataset(
    dataset: str, n_timepoints: int, *, need_images: bool
) -> tuple[Path, Path]:
    """Ensure one crop's GEFF graph, image metadata and (if needed) chunks are local.

    Returns ``(image_store, geff_store)``. Already-present files are skipped, so an
    interrupted download resumes rather than restarting.

    ``need_images`` is False when every timepoint of this crop is already fitted.
    That skips ~450 MB of image chunks per crop, which is the difference between a
    warm-cache run working on a laptop and demanding 4 GB of raw data it will never
    read. Kaggle also rate-limits bulk single-file downloads hard enough that not
    asking is worth real time.
    """
    image = DATA_DIR / "train" / f"{dataset}.zarr"
    geff = DATA_DIR / "train" / f"{dataset}.geff"

    wanted = _crop_metadata_files(dataset)
    if need_images:
        wanted += _crop_chunk_files(dataset, n_timepoints)
    missing = [p for p in wanted if not _is_downloaded(DATA_DIR / p)]
    if not missing:
        return image, geff

    cmd = _kaggle_cmd()
    with asection(f"Downloading {dataset} ({len(missing)} of {len(wanted)} files)"):
        failed = 0
        for i, rel in enumerate(missing, 1):
            if not _fetch_file(cmd, rel, DATA_DIR):
                failed += 1
            if i % 25 == 0:
                aprint(f"  {i}/{len(missing)}")
        if failed:
            raise RuntimeError(
                f"{dataset}: {failed} of {len(missing)} files could not be "
                "downloaded. Kaggle rate-limits bulk single-file downloads; "
                "re-run the demo later and it will resume where it stopped."
            )
    return image, geff


# =============================================================================
# Fitting
# =============================================================================


def _device() -> str:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = detect_device()
    return _DEVICE


def voxel_size_of(image_store: Path) -> tuple[float, float, float]:
    """Read the crop's ZYX voxel size (um) from its OME-Zarr metadata.

    Both metadata layouts are accepted — see
    :func:`luxar.io.ome_zarr.resolve_ngff_attrs`, which owns that resolution. The
    scale vector is likewise looked up by :func:`luxar.io.ome_zarr.
    ngff_scale_transform`, which SEARCHES ``coordinateTransformations`` for the
    ``type == "scale"`` entry: indexing ``[0]`` here broke on any store whose
    first transform is a ``translation``.

    The two ways this can fail say different things, because they are fixed
    differently: no ``multiscales`` at all (the store is not the OME-Zarr crop
    this demo downloads) versus a ``multiscales`` that declares no ``scale``
    transform (it is, but it states no spacing).
    """
    import zarr

    from luxar.io.ome_zarr import ngff_scale_transform, resolve_ngff_attrs

    group = zarr.open_group(str(image_store), mode="r")
    attrs = dict(group.attrs)
    root = resolve_ngff_attrs(attrs)
    if not root.get("multiscales"):
        raise ValueError(
            f"{image_store} has no OME-Zarr `multiscales` metadata "
            f"(attributes present: {sorted(attrs)}); the crop's voxel size is "
            "read from it."
        )
    scale = ngff_scale_transform(
        root["multiscales"][0]["datasets"][0].get("coordinateTransformations")
    )
    if scale is None:
        raise ValueError(
            f"{image_store} declares OME-Zarr `multiscales` metadata but its "
            "first dataset states no `scale` coordinateTransformation; the "
            "crop's voxel size is read from it."
        )
    return tuple(float(s) for s in scale[1:])  # drop the time axis


def _fit_cache_path(dataset: str, t: int) -> Path:
    return FITS_DIR / dataset / f"t{t:04d}_k{SEEDS}.gsplats.zarr.zip"


def is_fit_cached(dataset: str, t: int) -> bool:
    """Whether timepoint ``t`` of ``dataset`` is already fitted.

    A cache entry counts only when its in-progress ``.tmp`` marker is absent: a
    process killed mid-save leaves a truncated store behind, and loading that
    would be worse than re-fitting.
    """
    path = _fit_cache_path(dataset, t)
    marker = path.with_suffix(path.suffix + ".tmp")
    return path.exists() and not marker.exists()


def needs_fitting(dataset: str, n_timepoints: int) -> bool:
    """Whether any timepoint of this crop still has to be fitted.

    Asked BEFORE downloading, so a fully-fitted crop never pulls its ~450 MB of
    image chunks (see :func:`fetch_dataset`), and torch is never demanded on a
    machine that only assembles a scene from a warm cache.
    """
    return FLAGS["recompute"] or any(
        not is_fit_cached(dataset, t) for t in range(n_timepoints)
    )


def fit_timelapse(
    dataset: str, image_store: Path, n_timepoints: int
) -> list[GSplatData]:
    """Fit GSplats to each timepoint of one crop, caching per timepoint.

    One 3D fit per timepoint (rather than one 4D fit over the whole movie) keeps
    each fit small and independently cacheable, and is what lets the timepoints be
    stacked afterwards with time as a hard coarsening barrier.
    """
    import zarr

    will_fit = needs_fitting(dataset, n_timepoints)
    voxel = voxel_size_of(image_store)
    cache = FITS_DIR / dataset
    cache.mkdir(parents=True, exist_ok=True)

    def cache_path(t: int) -> Path:
        return _fit_cache_path(dataset, t)

    def is_cached(t: int) -> bool:
        return is_fit_cached(dataset, t)

    # The image array is only opened when a fit will actually read pixels from it —
    # a warm cache must not need the chunks on disk at all.
    arr = zarr.open_group(str(image_store), mode="r")["0"] if will_fit else None
    if will_fit:
        require_module("torch", pip_name="luxar[gsplats]")
        warn_if_no_cuda_gpu()
    from luxar.gsplats import fit_gaussian_splats

    results: list[GSplatData] = []
    with asection(f"Fitting {dataset}: {n_timepoints} timepoints (seeds={SEEDS:,})"):
        for t in range(n_timepoints):
            cache_file = cache_path(t)
            marker = cache_file.with_suffix(cache_file.suffix + ".tmp")
            if not FLAGS["recompute"] and is_cached(t):
                try:
                    # include_stats=True: the cache carries the fit's
                    # normalization provenance (floor / image_min / image_max),
                    # and `concatenate` only propagates a background floor into
                    # the stacked scene when every part reports one (#1175).
                    results.append(GSplatData.load(cache_file, include_stats=True))
                    continue
                except Exception as exc:  # noqa: BLE001
                    aprint(f"  t={t} cache unreadable ({exc}); re-fitting")
                    cache_file.unlink(missing_ok=True)

            if arr is None:
                # A cache entry vanished (or turned unreadable) between the scan
                # above and now, so the images we decided not to fetch are exactly
                # what is missing. Say so instead of failing on a None.
                raise RuntimeError(
                    f"{dataset}: timepoint {t} is not in the fit cache, but the "
                    "image chunks were skipped because the cache looked complete. "
                    "Re-run the demo — the scan will now ask for them."
                )
            volume = np.asarray(arr[t]).astype(np.float32)
            fitted = fit_gaussian_splats(
                volume,
                seeds=SEEDS,
                device=_device(),
                voxel_size=voxel,
                # The acquisition is uint16; this timepoint was cast to float32
                # above, so without declaring the source dtype the stamped
                # `source_bytes` would price the float32 working copy and halve
                # the compression ratio the dataset publishes. The GRID is
                # untouched, so no source_shape is needed.
                source_dtype=str(arr.dtype),
                verbose=False,
            )
            # Marker file: if the process dies mid-save the leftover .tmp makes
            # the next run discard the partial cache rather than load a truncated
            # store.
            marker.touch()
            fitted.save(
                cache_file,
                encoding_mode=EncodingMode.MEMORY,
                include_fitting_info=True,
                compress="zip",
                zip_deflate=True,
            )
            marker.unlink(missing_ok=True)
            # Stack what was STORED, not the in-memory fit: the cache is written
            # under a lossy encoding, so appending `fitted` here would make a
            # cold run (unquantized) and a warm run (the cache-hit branch above,
            # which loads the quantized store) produce different scenes. Same
            # include_stats=True as that branch, so the two agree AND the fit's
            # normalization provenance survives into the stacked scene.
            results.append(GSplatData.load(cache_file, include_stats=True))
            if (t + 1) % 10 == 0 or t == n_timepoints - 1:
                aprint(f"  {t + 1}/{n_timepoints} fitted ({fitted.n_splats:,} splats)")
    return results


def crop_centre_um(image_store: Path) -> np.ndarray:
    """The crop's geometric centre in um, ZYX.

    Every crop is the same 104 um cube, so centring on the box centre — rather
    than on each crop's own amplitude-weighted centroid — makes all tiles
    occupy identical boxes and the grid line up exactly.
    """
    import zarr

    arr = zarr.open_group(str(image_store), mode="r")["0"]
    voxel = np.asarray(voxel_size_of(image_store), dtype=np.float64)
    return 0.5 * voxel * np.asarray(arr.shape[1:], dtype=np.float64)


def combine_to_4d(
    per_timepoint: list[GSplatData], centre_zyx: np.ndarray
) -> GSplatData:
    """Stack per-timepoint 3D fits into one 4D (ZYX + time) GSplatData.

    Per timepoint, in order:

    1. **Recentre** on the crop's box centre, so every tile shares one box.
    2. **Drop the diffuse large-scale tail.** A fit of a densely packed nuclei
       stack puts most splats at ~1 um (median characteristic size 1.1 um, median
       nearest-neighbour spacing 1.8 um) but leaves a tail of big low-frequency
       splats (99th percentile 3.5 um, up to 7 um along the coarse Z axis). Under
       volumetric blending optical depth grows with a splat's path length, so that
       few-percent tail renders as opaque discs that bury the nuclei it sits on
       top of. ``scale`` is the geometric-mean sigma over the SPATIAL axes only,
       which is why it survives the zero-variance time axis added below.
    3. **Normalise amplitude** to a fixed peak. The viewer scales a gsplats node
       by the single maximum stored for the whole node, so without this the
       brightness of the whole embryo would visibly jump whenever a timepoint with
       an unusually bright nucleus went past.

    Time is then added as a new dimension with ``sigma=0``: splats are
    instantaneous, they must not smear across frames.
    """
    prepared: list[GSplatData] = []
    for g in per_timepoint:
        g = g.translate(-centre_zyx)
        g = g.filter_by(scale_max=SCALE_MAX_PERCENTILE, scale_percentile=True)
        amp_max = float(g.amplitudes.max()) if g.n_splats else 0.0
        if amp_max > 0:
            g = g.scale_intensity(1.0 / amp_max)
        prepared.append(g)

    values = list(range(len(prepared)))
    # Preserve the as-fitted stamps before filtering and intensity normalization.
    part_provenance = collect_part_provenance(
        per_timepoint,
        values=values,
        fit_reference={
            "kind": "preprocessed",
            "note": "cropped acquisition; stacked splats are filtered and intensity-normalized",
        },
    )
    return GSplatData.combine_as_new_dimension(
        prepared,
        values=values,
        sigma=0.0,
        part_provenance=part_provenance,
    )


def build_lod(data: GSplatData) -> GSplatData:
    """Give a 4D crop a substitutive LOD ladder, with time as a hard barrier.

    ``coarsen_dims=(0, 1, 2)`` restricts merging to the ZYX centre columns, so a
    coarse level never merges splats from different timepoints into one — which
    would smear the whole timelapse into a single blur.
    """
    from luxar.gsplats.lod import RecipeParams, build_recipe

    return build_recipe(
        data,
        "levels",
        RecipeParams(
            compression_factor=LOD_COMPRESSION_FACTOR,
            levels=LOD_LEVELS,
            coarsen_dims=(0, 1, 2),
            additive_ladders=ADDITIVE_LADDERS,
        ),
    )


# =============================================================================
# Tracks
# =============================================================================


def lineage_colors(n_lineages: int) -> np.ndarray:
    """One saturated colour per lineage, hues spread by the golden ratio.

    Golden-ratio hue stepping keeps neighbouring lineage ids visually distinct
    (a plain linear ramp gives adjacent lineages near-identical hues). Colours are
    authored in sRGB and linearized, because the viewer works in linear light.
    """
    hues = (np.arange(n_lineages, dtype=np.float32) * 0.61803399) % 1.0
    return (hsv_to_rgb(hues, 0.85, 1.0) ** 2.2).astype(np.float32)


def track_geometry(
    geff_store: Path, centre_zyx: np.ndarray, n_timepoints: int
) -> dict | None:
    """Build point and polyline arrays for one crop's annotated cell tracks.

    Returns arrays in ``(z, y, x, time)`` column order — matching the gsplats'
    ``dim_order`` — or ``None`` when the crop has no usable annotation:

    * ``point_positions`` / ``point_colors`` / ``point_radii``: one marker per
      annotated cell per timepoint, coloured by lineage
    * ``line_vertices`` / ``line_colors`` / ``line_widths`` / ``line_indices``:
      the lineage forest as ``indexed`` lines — one vertex per annotated cell and
      one edge per tracking link. Sharing vertex rows between consecutive links is
      what keeps a track's joints continuous, and it makes a cell division render
      as a real fork instead of two detached strands.

    The whole graph is emitted as ONE indexed lines node rather than a polyline
    node per track: the crops carry up to ~1,900 tracked cells, and one node per
    track would mean hundreds of nodes per crop.
    """
    from luxar.gsplats.interop.geff import read_geff

    graph = read_geff(geff_store)
    if graph.n_nodes == 0:
        return None

    # GEFF stores voxel coordinates; scale to um and recentre exactly as the
    # gsplats were, so markers land on the nuclei they annotate.
    pos = graph.positions_um() - centre_zyx
    lineage = graph.lineage_ids()
    palette = lineage_colors(int(lineage.max()) + 1)
    vertices = np.column_stack([pos, graph.t.astype(np.float32)]).astype(np.float32)
    colors = palette[lineage]

    return window_tracks(
        {
            "point_positions": vertices,
            "point_colors": colors,
            "point_radii": np.full(len(vertices), CELL_MARKER_RADIUS_UM, np.float32),
            "line_vertices": vertices,
            "line_colors": colors,
            "line_widths": np.full(len(vertices), TRACK_WIDTH_UM, np.float32),
            "line_indices": graph.edge_indices().astype(np.uint32),
            "n_lineages": int(lineage.max()) + 1,
            "n_cells": len(vertices),
            "n_divisions": len(graph.divisions()),
        },
        n_timepoints,
    )


def window_tracks(tracks: dict | None, n_timepoints: int) -> dict | None:
    """Restrict track geometry to the first ``n_timepoints`` frames.

    Line vertices are NOT dropped — that would invalidate every edge index — so
    only edges with an endpoint past the window are removed. The per-timepoint
    markers carry no connectivity, so those ARE filtered, and the counts that
    describe the window (cells, divisions) are recomputed from what survived.
    ``None`` means nothing is left worth drawing, and the caller falls back to a
    volume-only tile.

    Both paths need this: the local one to honour ``--timepoints`` while fitting,
    and the hosted one because the published crops are whole 100-timepoint
    timelapses whatever window was asked for.
    """
    if tracks is None:
        return None

    vertex_t = np.asarray(tracks["line_vertices"])[:, -1]
    edges = np.asarray(tracks["line_indices"])
    if len(edges):
        edges = edges[
            (vertex_t[edges[:, 0]] < n_timepoints)
            & (vertex_t[edges[:, 1]] < n_timepoints)
        ]
    if not len(edges):
        return None

    visible = np.asarray(tracks["point_positions"])[:, -1] < n_timepoints
    out = dict(tracks)
    out["line_indices"] = edges.astype(np.uint32)
    for key in ("point_positions", "point_colors", "point_radii"):
        out[key] = np.asarray(tracks[key])[visible]
    out["n_cells"] = int(visible.sum())
    out["n_divisions"] = int(
        np.count_nonzero(np.bincount(edges[:, 0], minlength=len(vertex_t)) >= 2)
    )
    return out


# =============================================================================
# Scene
# =============================================================================


def grid_shape(n: int) -> tuple[int, int]:
    """Columns and rows of the most compact grid holding ``n`` tiles.

    Square when n is a perfect square (9 -> 3x3, 4 -> 2x2), otherwise the
    shortest rectangle that still fits: 6 -> 3x2, 8 -> 3x3 with one gap. A plain
    ``ceil(sqrt(n))`` square would leave 6 in a 3x3 with THREE holes, which is
    what the rectangle exists to avoid.
    """
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return cols, rows


def grid_transforms(n: int, pitch: float) -> list[np.ndarray]:
    """Row-major grid placement over :func:`grid_shape`.

    The grid is laid out in the scene's x/y plane and centred on the origin, so
    the default camera frames the whole matrix.
    """
    cols, rows = grid_shape(n)
    half_col = (cols - 1) / 2.0
    half_row = (rows - 1) / 2.0
    out = []
    for i in range(n):
        row, col = divmod(i, cols)
        out.append(
            transforms.translate(
                (col - half_col) * pitch, (row - half_row) * pitch, 0.0
            )
        )
    return out


def create_luxar_scene(
    crops: list[dict],
    n_timepoints: int,
    output_path: Path,
) -> Path:
    """Build the matrix scene: one group per crop, each with gsplats+points+lines."""
    n = len(crops)
    cols, rows = grid_shape(n)
    extent = float(crops[0]["extent_um"])
    xforms = grid_transforms(n, extent * GRID_GAP_FACTOR)

    total_splats = sum(int(c["n_splats"]) for c in crops)
    total_cells = sum(int(c["tracks"]["n_cells"]) for c in crops if c["tracks"])
    total_divisions = sum(int(c["tracks"]["n_divisions"]) for c in crops if c["tracks"])

    with asection(f"Creating scene: {cols}x{rows} matrix, {n} crops"):
        aprint(f"Output: {output_path.name}")
        aprint(f"Splats: {total_splats:,} across {n} crops (finest level)")
        aprint(f"Tracked cell positions: {total_cells:,}")
        aprint(f"Division events: {total_divisions:,}")

        dims = Dimensions(
            [
                Dimension("x", unit="um", display=True),
                Dimension("y", unit="um", display=True),
                Dimension("z", unit="um", display=True),
                Dimension(
                    "time",
                    unit="frame",
                    display=False,
                    discrete=True,
                    range=(0, n_timepoints - 1),
                    step=1.0,
                ),
            ]
        )

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                citation=DEMO_META["citation"],
                dimensions=dims,
                viewer_config=ViewerConfig(cinematic_mode=True),
            )

            scene.attrs["title"] = (
                f"GSplats: Cell Tracking Challenge — {n} zebrafish embryo timelapses"
            )
            scene.attrs["description"] = f"""
4D Cell Tracking — Biohub Challenge Data
========================================

{n} crops of a developing zebrafish embryo in a {cols}x{rows} matrix, each a full
{n_timepoints}-timepoint light-sheet timelapse.

What you are looking at:
  - GSplats: the image data, {total_splats:,} splats over all crops and timepoints
  - Points:  {total_cells:,} annotated cell positions, coloured by lineage
  - Lines:   every tracking link, so each cell's whole trajectory is visible;
             {total_divisions:,} of the joints are cell divisions, which fork

Level of detail:
  Each crop carries a substitutive LOD ladder ({LOD_LEVELS} coarser level(s),
  {LOD_COMPRESSION_FACTOR}x apart). A tile of the matrix occupies a small fraction
  of the screen and draws the coarser level; zoom into one and it refines to full
  detail.

Data Source:
  - Kaggle: Biohub - Cell Tracking During Development
  - {COMPETITION_URL}
  - Annotations CC0, Chan Zuckerberg Biohub San Francisco
  - Light-sheet imaging, Royer Group
  - Each crop: 100 x 64 x 256 x 256 (TZYX) uint16,
    voxel 1.625 x 0.40625 x 0.40625 um (a {extent:.0f} um cube)

Navigation:
  - Time slider (or the play button) scrubs through development
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
            """

            # One `layer=True` wrapper per crop, so the Layers panel offers one
            # row per embryo rather than three ("this embryo's splats"). The
            # wrapper restates `volumetric` because the panel falls back to a
            # group's default (additive) when no mode is authored, which would
            # both mislabel the row and hide the absorption slider.
            for crop, xform in zip(crops, xforms):
                group = scene.add_group(
                    crop["name"],
                    transform=xform,
                    layer=True,
                    blending_mode="volumetric",
                )

                # Volumetric + absorption for the image: emissive-only additive
                # rendering makes a dense nuclei stack read as a flat glow, while
                # volumetric occlusion keeps the embryo's depth structure. The
                # opacity / absorption / window trio is the tuned set — see the
                # VOLUME_* constants for what each one is doing and why.
                group.add_gsplats_from_data(
                    "volume",
                    crop["lod"],
                    lod_group=True,
                    dim_order=["z", "y", "x", "time"],
                    extend_to_all=[],
                    opacity=VOLUME_OPACITY,
                    absorption=VOLUME_ABSORPTION,
                    intensity=crop["intensity"],
                    offset=crop["offset"],
                    blending_mode="volumetric",
                    colormap="bop_blue",
                )

                tracks = crop["tracks"]
                if tracks is None:
                    continue

                # Tracks stay visible at every timepoint (`extend_to_all=["time"]`)
                # so the lineage structure reads even while the volume animates;
                # the markers do NOT, so only the cells present right now are
                # flagged.
                group.add_lines(
                    "tracks",
                    vertices=tracks["line_vertices"],
                    widths=tracks["line_widths"],
                    colors=tracks["line_colors"],
                    indices=tracks["line_indices"],
                    line_type="indexed",
                    dim_order=["z", "y", "x", "time"],
                    extend_to_all=["time"],
                    blending_mode="normal",
                    opacity=TRACK_OPACITY,
                )
                group.add_points(
                    "cells",
                    positions=tracks["point_positions"],
                    colors=tracks["point_colors"],
                    radii=tracks["point_radii"],
                    dim_order=["z", "y", "x", "time"],
                    extend_to_all=[],
                    blending_mode="normal",
                )

            # --- Overlays ---
            scene.add_text(
                "Cell Tracking During Development",
                position=(0.02, 0.02),
                font_size=0.045,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                "Light-sheet microscopy • zebrafish • tracked lineages",
                DEMO_META.get("citation"),
            )
            scene.add_text(
                f"{n} embryo crops • {n_timepoints} timepoints each\n"
                f"{total_splats:,} splats • {total_cells:,} tracked cell positions\n"
                "Colour = lineage. Lines are whole trajectories;\n"
                "dots mark the cells present at the current timepoint.",
                position=(0.02, 0.11),
                font_size=0.018,
                font="mono",
                color="white",
                width=0.46,
                line_height=1.45,
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Build (and serve) the cell-tracking matrix scene."""
    aprint("=" * 64)
    aprint("4D CELL TRACKING CHALLENGE — EMBRYO MATRIX")
    aprint("=" * 64)
    aprint(f"Dataset: {COMPETITION_URL}")

    output_path = (
        get_demos_output_dir() / "gsplats_4d_cell_tracking_challenge.luxar.zarr"
    )

    if FLAGS["serve_only"]:
        if output_path.exists():
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    n_datasets = max(1, min(N_DATASETS, len(DATASETS)))
    n_timepoints = max(2, min(TIMEPOINTS, N_TIMEPOINTS))
    chosen = DATASETS[:n_datasets]
    aprint(f"Crops: {n_datasets} • timepoints: {n_timepoints} • seeds: {SEEDS:,}")

    # Preferred path: the hosted derived product — no Kaggle credentials, no GPU.
    # Falls through to fetch-and-fit when it is unavailable or --recompute is set.
    crops = load_precomputed_crops(chosen)
    if crops is not None:
        hosted_tps = {int(c["n_timepoints"]) for c in crops}
        hosted_max = max(hosted_tps) if hosted_tps else n_timepoints
        aprint(
            f"Using precomputed crops from the '{PRECOMPUTED_DATASET}' dataset "
            f"({', '.join(str(t) for t in sorted(hosted_tps))} timepoints each)"
        )
        # `--timepoints` still has to be honoured off the hosted path. The volumes
        # are published whole, so the window is applied to the scene instead: the
        # time dimension stops at the requested frame (splats past it are then
        # unreachable, the slider cannot get there) and the tracks are cut to
        # match, so the tile is not annotated for frames it never shows.
        scene_timepoints = min(n_timepoints, hosted_max)
        if scene_timepoints < hosted_max:
            aprint(
                f"Showing the first {scene_timepoints} of {hosted_max} hosted "
                "timepoints (--timepoints); the hosted volumes are whole "
                "timelapses, so this trims the scene rather than the download."
            )
            for crop in crops:
                crop["tracks"] = window_tracks(crop["tracks"], scene_timepoints)
        scene_path = create_luxar_scene(crops, scene_timepoints, output_path)
        if not FLAGS["no_serve"]:
            aprint("\nLaunching viewer...")
            launch_viewer(scene_path)
        aprint("\nDone!")
        return

    crops = []
    for i, dataset in enumerate(chosen, 1):
        with asection(f"[{i}/{n_datasets}] {dataset}"):
            image_store, geff_store = fetch_dataset(
                dataset,
                n_timepoints,
                need_images=needs_fitting(dataset, n_timepoints),
            )
            centre = crop_centre_um(image_store)
            per_tp = fit_timelapse(dataset, image_store, n_timepoints)
            combined = combine_to_4d(per_tp, centre)
            aprint(f"Combined 4D: {combined.n_splats:,} splats")
            lod = build_lod(combined)
            counts = [int(lvl.n_splats_total) for lvl in lod.substitutive_levels]
            aprint(
                f"LOD levels (fine->coarse): {counts} "
                f"= {' / '.join(f'{c // n_timepoints:,}' for c in counts)} per timepoint"
            )
            # `lod.amplitudes` is the FINEST level, which is what the window is
            # judged on; the coarser levels ride the same authored pair.
            intensity, offset = display_window(lod.amplitudes)
            aprint(
                f"Display window: [{-offset / intensity:.3f}, "
                f"{(1 - offset) / intensity:.3f}] (min..p{VOLUME_WINDOW_TOP_PERCENTILE:g})"
            )
            tracks = track_geometry(geff_store, centre, n_timepoints)
            if tracks is None:
                aprint("No usable annotation for this crop — volume only")
            else:
                aprint(
                    f"Tracks: {tracks['n_cells']:,} cell positions in "
                    f"{tracks['n_lineages']:,} lineages, "
                    f"{len(tracks['line_indices']):,} links, "
                    f"{tracks['n_divisions']:,} divisions"
                )
            crops.append(
                {
                    "name": dataset,
                    "lod": lod,
                    "n_splats": counts[0] if counts else combined.n_splats,
                    "intensity": intensity,
                    "offset": offset,
                    "tracks": tracks,
                    "extent_um": float(2.0 * centre.max()),
                }
            )

    scene_path = create_luxar_scene(crops, n_timepoints, output_path)

    if not FLAGS["no_serve"]:
        aprint("\nLaunching viewer...")
        aprint("Scrub the time slider to watch the cells move along their tracks.")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
