#!/usr/bin/env python3
"""GSplats Demo: 4D Zebrafish Embryo Time-Lapse (Zenodo 1211599).

A 4D (3D + time) confocal recording of a living zebrafish embryo during
gastrulation, as **one** Gaussian-splat node with time as a real axis of the
data: 151 timepoints over five hours, while the fluorescently labelled
endodermal cells spread from a tight cluster over the whole yolk sphere.

The specimen occupies well under 2% of the imaged voxels, so a **wireframe cage
with a 100 um grid** is drawn around the acquisition volume as its own
toggleable layer. Without it there is nothing to judge scale, depth or drift
against — the cells float in an unmarked void, and the migration that is the
whole point of the recording is impossible to read.

WHAT MAKES THIS A 4D NODE (and not 64 nodes in a trench coat):
    Every timepoint is still fitted on its own — that is what keeps each fit
    small, cacheable and independently resumable — but the fits are then
    ``combine_as_new_dimension``-stacked into ONE 4D ``GSplatData`` whose fourth
    centre column is time (sigma = 0: a splat is instantaneous and must not
    smear across frames). That single node then gets a substitutive LOD ladder
    with time as a hard coarsening barrier, and the viewer's Time slider walks
    the axis. The previous version of this demo instead wrote one flat gsplats
    node per timepoint with ``fill={"time": t}`` — 64 sibling nodes, no LOD, no
    shared appearance, one Layers row each.

DATA SOURCE & CITATIONS:
    Zenodo record 1211599 — ``cxcr4aMO2_290112.lsm`` (Zeiss LSM, 2.1 GB),
    Danio rerio (zebrafish) cxcr4a morphant, endoderm label, confocal
    laser-scanning microscopy. DOI 10.5281/zenodo.1211599, CC BY-SA 4.0.

    Please cite the record if you use this data. ShareAlike binds whoever
    receives a scene built from it, which is why the licence is carried into
    the store rather than only mentioned here.

ACQUISITION (read from the LSM's own metadata, not assumed):
    151 timepoints x 44 x 512 x 512 uint8, voxel 7.1847 x 1.6774 x 1.6774 um
    (ZYX) — anisotropic by 4.28x along Z — one frame every 120.01 s, so the
    recording spans 5 h 00 m. Each fit runs on the voxel grid and is then scaled
    by that voxel size (an exact diagonal affine, carried into the Cholesky
    factors), so the shipped centres are in microns and the embryo has its true
    proportions; the imaged block is 316 x 859 x 859 um.

SPLAT BUDGET (measured on this data, 2026-08-22, RTX PRO 6000):
    The previous version asked for 2,000 seeds and shipped **533 splats** for a
    busy timepoint, because it also took the fitter's bare defaults: 1,000
    iterations (below the ``draft`` preset's 2,000) and ``cull_retention=0.95``,
    which discards 73% of the splats along with the last 5% of amplitude. At
    t=110, scored against the raw frame:

    | fit                              | splats | global | foreground | energy |
    |----------------------------------|--------|--------|------------|--------|
    | shipped (2k seeds, 1000 it, .95) |    549 |  31.24 |   6.71 dB  |  0.36  |
    | 16k seeds, 5000 it, cull .999    |  5,925 |  33.39 |   9.30 dB  |  0.46  |
    | 32k seeds, 5000 it, cull .9999   |  9,432 |  34.27 |  10.46 dB  |  0.50  |
    | 32k seeds, 5000 it, cull 1.0     | 32,000 |  34.27 |  10.45 dB  |  0.50  |
    | 64k seeds, 5000 it, cull 1.0     | 64,000 |  34.30 |  10.52 dB  |  0.50  |

    So ~9,400 splats is the plateau, not a compromise: keeping all 32,000
    changes the reconstruction by 0.01 dB, and doubling the seed budget on top
    buys 0.07 dB. Retaining 0.9999 of the amplitude instead of 0.95 is what
    actually moved the number — +3.7 dB on the foreground — because on a volume
    this sparse the discarded 5% of amplitude *is* the dim cells. Absolute
    foreground PSNR stays low because the labelled cells are clipped at 255 in
    an 8-bit acquisition and a Gaussian cannot reproduce a flat-topped plateau;
    read the column as a comparison between fits, not as an absolute grade.

    ``cal`` is deliberately not consulted here. Its blind-spot sweep put K* at
    1,852 for this dataset — the second point of its own grid, with a 0.17 dB
    confidence margin, measured on the XY-halved copy the demo used to fit — and
    the table above is what the archive is actually judged by.

USAGE:
    python demo_gsplats_4d_zebrafish_timelapse.py [--recompute] [--no-serve]
        [--serve-only] [--max-timepoints=N]

    --recompute:        Download the LSM and refit from scratch (needs a GPU).
    --no-serve:         Build the scene without launching the viewer.
    --serve-only:       Serve an already-built scene.
    --max-timepoints=N: Fit only N evenly spaced timepoints (refit path only).

OUTPUT:
    - Scene saved to:  datasets/demos/gsplats_4d_zebrafish_timelapse.luxar.zarr
    - Opens in the browser; press L for the Layers panel, play the Time slider.
"""

DEMO_META = {
    "key": "gsplats_4d_zebrafish_timelapse",
    "title": "4D Zebrafish Timelapse",
    "description": (
        "Five hours of zebrafish gastrulation as one 4D Gaussian-splat node, "
        "inside a gridded cage that gives the sparse embryo a scale."
    ),
    "category": "microscopy",
    "geometry": "mixed",
    "requirements": {
        "download_mb": 45,
        "compute": "medium",
        "gpu": "optional",
        "local_data": "git-lfs",
    },
    "caches": ["gsplats_zebrafish"],
    "outputs": ["gsplats_4d_zebrafish_timelapse"],
    "citation": {
        "short": "Aanstad 2018",
        "doi": "10.5281/zenodo.1211599",
        # ShareAlike: worth carrying into the store, since it binds whoever
        # receives a scene built from this data.
        "license": "CC BY-SA 4.0",
    },
}

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    add_demo_caption,
    detect_device,
    launch_viewer,
    load_dataset_gsplats,
    load_local_fit_gsplats_at,
    local_fit_path,
    parse_demo_flags,
    parse_int_arg,
    require_module,
    warn_if_no_cuda_gpu,
)
from luxar.demos.registry import DEMO_CACHE_ROOT
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.data_fetch import DatasetUnavailable
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEMO_NAME = "gsplats_zebrafish"
GSPLATS_FILE = "zebrafish_4d.gsplats.zarr.zip"
SCENE_NAME = "gsplats_4d_zebrafish_timelapse.luxar.zarr"

#: The raw acquisition, only needed on the ``--recompute`` path.
ZENODO_URL = "https://zenodo.org/api/records/1211599/files/cxcr4aMO2_290112.lsm/content"
LSM_BYTES = 2_080_484_264

#: The acquisition geometry, as the LSM records it. Held here rather than only
#: read from the file because the SCENE needs it and the scene is normally built
#: from the shipped archive, with no LSM in reach: the cage must be the imaged
#: BLOCK, and a box fitted to the splats instead would shrink around a specimen
#: that fills under 2% of it — a ruler cut to the size of the thing it measures.
#: :func:`open_lsm` checks the file against these on the refit path, so they
#: cannot drift silently.
ACQUISITION_SHAPE_ZYX = (44, 512, 512)
VOXEL_SIZE_ZYX_UM = (7.184696827249337, 1.6774389429296399, 1.6774389429296399)
FRAME_INTERVAL_S = 120.00882789993899

CACHE_DIR = DEMO_CACHE_ROOT / DEMO_NAME
LSM_PATH = CACHE_DIR / "cxcr4aMO2_290112.lsm"
#: Per-timepoint fits, keyed on the fit schedule so retuning cannot hit a stale
#: entry: the knobs below change splat SHAPES while leaving the count identical,
#: which a cache keyed on the frame index alone would silently serve back.
FITS_DIR = CACHE_DIR / "fits"
#: The stacked 4D archive this machine built, in the demo's own namespace. It is
#: a different artifact from the hosted file of the same purpose, so it must NOT
#: live at the manifest path — ``ensure_dataset`` hashes whatever it finds there
#: and quarantines a mismatch, which would refit on every launch (#1618).
LOCAL_FIT = local_fit_path(DEMO_NAME, GSPLATS_FILE)

#: Fit schedule. See the module docstring's SPLAT BUDGET table for the sweep
#: these came from. ``seeds`` proposes and ``cull_retention`` disposes: 32,000
#: seeds deliver ~9,400 splats on a busy frame and fewer on an early, near-empty
#: one, which is the point — the budget adapts to how much embryo there is.
SEEDS = 32_000
N_ITERS = 5_000
EARLY_STOP_PATIENCE = 500
CULL_RETENTION = 0.9999

#: Substitutive LOD: three coarse levels, each 4x lighter than the last, every
#: level carrying its own progressive ladder. Coarsening is restricted to the
#: three SPATIAL centre columns — time (column 3) is a hard barrier, because
#: merging across it would blend one timepoint's cells into the next and smear
#: the whole recording into a single haze.
#:
#: The archive is built with ``build_recipe`` rather than through
#: ``luxar.demos._lod_policy.save_with_lod``, for the reason its 4D sibling
#: ``demo_gsplats_4d_cell_tracking_challenge`` has: the helper writes exactly one
#: topology per demo and takes no barrier argument, while this demo writes two
#: kinds of store — the per-timepoint fits, which are scratch consumed by the
#: stack and want no ladder at all, and the one 4D archive it ships.
LOD_COMPRESSION_FACTOR = 4
LOD_LEVELS = 3
LOD_N_LODS = 4
LOD_COARSEN_DIMS = (0, 1, 2)

#: Appearance. Volumetric with a modest kappa: the specimen is a thin shell of
#: cells on a sphere, so a little occlusion separates near from far, while a
#: large one would hide the far side entirely. ``inferno`` starts at (0, 0, 4)
#: rather than viridis's dark purple, so the 98% of the volume that is empty
#: reads as true black instead of a haze.
COLORMAP = "inferno"
VOLUME_OPACITY = 1.0
VOLUME_ABSORPTION = 0.5

#: The acquisition cage: wireframe box + grid, in microns.
GRID_STEP_UM = 100.0
BOX_EDGE_WIDTH_UM = 2.0
GRID_LINE_WIDTH_UM = 1.0
BOX_EDGE_COLOR = (0.42, 0.62, 0.72)
GRID_LINE_COLOR = (0.16, 0.26, 0.32)

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

#: ``None`` = every timepoint in the recording. Only consulted when refitting.
MAX_TIMEPOINTS = parse_int_arg("max-timepoints", None, sys.argv)

Arbol.max_depth = 5

DEVICE: Optional[str] = None


# =============================================================================
# Acquisition
# =============================================================================
def open_lsm() -> Any:
    """Download the LSM if needed, check its geometry, and open it lazily.

    Returns a lazy ``(T, Z, Y, X)`` zarr view: the movie is 1.7 GB of voxels and
    only one timepoint is ever needed at a time, so it is never read whole.

    The LSM's own voxel size and frame interval are compared against the module
    constants above rather than merely read, because the scene uses the
    constants — a record re-uploaded with a different calibration would
    otherwise mislabel every axis of a rebuilt archive without a word.
    """
    tifffile = require_module("tifffile")
    zarr = require_module("zarr")

    from luxar.utils.download import robust_download

    with asection("Zebrafish LSM (Zenodo 1211599)"):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        robust_download(
            ZENODO_URL, LSM_PATH, max_retries=5, timeout=900, expected_size=LSM_BYTES
        )

        with tifffile.TiffFile(str(LSM_PATH)) as tif:
            meta = getattr(tif, "lsm_metadata", None) or {}
            voxel = tuple(
                float(meta.get(k, 0.0)) * 1e6
                for k in ("VoxelSizeZ", "VoxelSizeY", "VoxelSizeX")
            )
            interval = float(meta.get("TimeIntervall", 0.0))

        # series 0 is the image data; series 1 is the embedded RGB thumbnail.
        array = zarr.open(
            tifffile.imread(str(LSM_PATH), aszarr=True, series=0), mode="r"
        )
        aprint(f"Acquisition: {array.shape} {array.dtype}")
        aprint(f"Voxel (Z,Y,X): {voxel[0]:.4f} x {voxel[1]:.4f} x {voxel[2]:.4f} um")
        aprint(
            f"Frame interval: {interval:.2f} s "
            f"({array.shape[0] * interval / 3600:.2f} h total)"
        )

        drift = [
            f"{what}: file says {got}, this demo assumes {want}"
            for what, got, want, ok in (
                (
                    "grid",
                    tuple(array.shape[1:]),
                    ACQUISITION_SHAPE_ZYX,
                    tuple(array.shape[1:]) == ACQUISITION_SHAPE_ZYX,
                ),
                (
                    "voxel size (um)",
                    voxel,
                    VOXEL_SIZE_ZYX_UM,
                    np.allclose(voxel, VOXEL_SIZE_ZYX_UM, rtol=1e-4),
                ),
                (
                    "frame interval (s)",
                    interval,
                    FRAME_INTERVAL_S,
                    abs(interval - FRAME_INTERVAL_S) < 1e-3,
                ),
            )
            if not ok
        ]
        if drift:
            raise RuntimeError(
                "The downloaded LSM does not match the acquisition this demo is "
                "written against:\n  "
                + "\n  ".join(drift)
                + "\nUpdate ACQUISITION_SHAPE_ZYX / VOXEL_SIZE_ZYX_UM / "
                "FRAME_INTERVAL_S, and the numbers quoted in the docstring."
            )
        return array


def select_timepoints(n_total: int, limit: Optional[int]) -> list[int]:
    """Frame indices to fit — every frame, or a uniform subsample of ``limit``.

    The shipped archive takes every frame, which is the change that matters:
    the previous version asked for 64 of 151 and computed ``stride =
    n_total // n_use`` = 2, so it stopped at frame 126 and the last 50 minutes
    of gastrulation — the part where the endoderm has actually spread — never
    made it into the demo at all.

    A subsample is UNIFORM by construction, not merely evenly spread. Time
    becomes a discrete viewer dimension whose navigation snaps to multiples of
    one ``step``, so unevenly spaced frames would land between stops and those
    stops would silently render nothing. Being uniform costs the tail: a stride
    that does not divide the run stops short of the final frame. That is the
    right trade for a development flag and no trade at all for the archive.
    """
    if limit is None or limit >= n_total:
        return list(range(n_total))
    if limit < 2:
        raise ValueError(f"--max-timepoints must be at least 2, got {limit}")
    stride = max(1, (n_total - 1) // (limit - 1))
    return list(range(0, n_total, stride))[:limit]


# =============================================================================
# Fitting
# =============================================================================
def _fit_cache_path(frame: int) -> Path:
    return FITS_DIR / f"f{frame:04d}_k{SEEDS}_i{N_ITERS}.gsplats.zarr.zip"


def fit_timepoint(volume: np.ndarray, frame: int, acquisition: tuple):
    """Fit one timepoint, caching the result.

    The cached store is what gets stacked, not the in-memory fit: the cache is
    written under a lossy encoding, so returning the unquantized object would
    make a cold run and a warm run build different scenes. ``include_stats=True``
    on the read keeps the fit's normalization provenance (floor / image range),
    which ``combine_as_new_dimension`` only propagates into the stacked result
    when every part reports one (#1175).
    """
    global DEVICE
    cache_file = _fit_cache_path(frame)
    if cache_file.exists() and not RECOMPUTE:
        try:
            return GSplatData.load(cache_file, include_stats=True)
        except Exception as exc:  # noqa: BLE001
            aprint(f"  frame {frame}: cache unreadable ({exc}); refitting")
            cache_file.unlink(missing_ok=True)

    if DEVICE is None:
        DEVICE = detect_device()
    from luxar.gsplats import fit_gaussian_splats

    src_shape, src_dtype = acquisition
    result = fit_gaussian_splats(
        volume,
        seeds=SEEDS,
        n_iters=N_ITERS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        cull_retention=CULL_RETENTION,
        # Fit on the VOXEL grid, and apply the microns afterwards (see
        # `to_microns`). Handing the fitter `voxel_size` instead puts the
        # optimizer in physical space, which costs the same wall clock (A/B at
        # 32k seeds: 38.1 s voxel vs 32.6 s physical on frame 110) but scores
        # slightly worse on both frames tried — 34.26/10.44 dB against
        # 34.00/10.08 global/foreground at t=110, 46.37/16.29 against
        # 46.12/16.07 at t=0 — plausibly because a 4.28x-anisotropic grid makes
        # the fitter's isotropic seed shapes a worse starting guess in microns
        # than in voxels. It is also the space the SPLAT BUDGET table was
        # measured in. The scale itself is exact either way: a diagonal affine,
        # which `GSplatData.transform` carries into the Cholesky factors.
        output_space="voxel",
        # One timepoint of the stored uint8 stack is the source; `volume` is a
        # normalized float32 copy of it, so without this the compression ratio
        # would be quoted against a denominator 4x too large.
        source_shape=src_shape,
        source_dtype=src_dtype,
        device=DEVICE,
        verbose=False,
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    result.save(
        cache_file,
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )
    return GSplatData.load(cache_file, include_stats=True)


def fit_all_timepoints(array, frames: list[int]) -> list[GSplatData]:
    """Fit every selected timepoint, reading one frame of the movie at a time."""
    acquisition = (tuple(int(s) for s in array.shape[1:]), str(array.dtype))
    scale = (
        float(np.iinfo(array.dtype).max)
        if np.issubdtype(array.dtype, np.integer)
        else 1.0
    )
    fits: list[GSplatData] = []
    with asection(
        f"Fitting {len(frames)} timepoints (seeds={SEEDS:,}, {N_ITERS} iters)"
    ):
        for i, frame in enumerate(frames):
            volume = np.asarray(array[frame]).astype(np.float32) / scale
            fits.append(fit_timepoint(volume, frame, acquisition))
            if (i + 1) % 10 == 0 or i == len(frames) - 1:
                aprint(
                    f"  {i + 1}/{len(frames)} — frame {frame}: "
                    f"{fits[-1].n_splats:,} splats"
                )
        report_fit_quality(fits)
    return fits


def report_fit_quality(fits: list[GSplatData]) -> None:
    """Print what the fits actually achieved, from their own stamps.

    Each fit scores itself against the raw frame it was handed, so this costs
    nothing to print and is the number the archive should be judged by. Both
    columns are reported: on a volume this sparse, global PSNR is mostly the
    reward for predicting empty space correctly, and the foreground figure is
    the one that moves when the fit gets better or worse.
    """

    def column(key: str) -> Optional[np.ndarray]:
        vals = [g.stats.get(key) for g in fits]
        good = np.array([float(v) for v in vals if isinstance(v, (int, float))])
        return good if good.size else None

    counts = np.array([g.n_splats for g in fits])
    aprint(
        f"Splats per timepoint: median {int(np.median(counts)):,}, "
        f"range {counts.min():,}-{counts.max():,}, total {counts.sum():,}"
    )
    for label, key in (("global", "psnr_db"), ("foreground", "foreground_psnr_db")):
        col = column(key)
        if col is None:
            # A cache written before the fit stamped quality, or a fitter path
            # that skipped it — say so rather than print a silent blank.
            aprint(f"{label.capitalize()} PSNR: not stamped on these fits")
        else:
            aprint(
                f"{label.capitalize()} PSNR: median {np.median(col):.2f} dB, "
                f"range {col.min():.2f}-{col.max():.2f} dB"
            )


def acquisition_box_um() -> tuple[np.ndarray, np.ndarray]:
    """The imaged block in microns, centred on the origin: ``(bmin, bmax)``.

    Every timepoint shares one grid, so this is a constant of the recording —
    which is what lets the splats be recentred on it and the cage drawn around
    it without either needing the other in hand.
    """
    half = 0.5 * np.asarray(VOXEL_SIZE_ZYX_UM) * np.asarray(ACQUISITION_SHAPE_ZYX)
    return -half, half


def to_microns(fit: GSplatData) -> GSplatData:
    """Put one voxel-space fit into microns, centred on the acquisition box.

    The scale is the LSM's own voxel size — a diagonal affine, so
    ``GSplatData.transform`` carries it exactly into the Cholesky factors and
    the 4.28x axial anisotropy comes out corrected rather than merely stretched
    at display time.

    The recentring is on the *box* centre, not on the fit's own
    amplitude-weighted centroid. That matters more here than in a static demo:
    the labelled endoderm migrates across the yolk over five hours, so a
    per-frame centroid would chase it and the embryo would appear to swim on the
    spot while the cage slid past it.
    """
    scale = np.diag(np.asarray(VOXEL_SIZE_ZYX_UM, dtype=np.float64))
    centre = 0.5 * np.asarray(VOXEL_SIZE_ZYX_UM) * np.asarray(ACQUISITION_SHAPE_ZYX)
    return fit.transform(scale).translate(-centre)


def combine_to_4d(
    per_timepoint: list[GSplatData], times_min: list[float]
) -> GSplatData:
    """Stack per-timepoint 3D fits into one 4D (ZYX + time) dataset.

    Amplitudes are left alone. Every frame of this recording is an 8-bit stack
    whose brightest cells sit at 255, so the fits already share a scale, and
    normalising each frame to a fixed peak — the usual defence against a
    brightness pop — would flatten the real signal growth from 0.2% to 1.4%
    occupancy that is the developmental story.
    """
    stacked = GSplatData.combine_as_new_dimension(
        [to_microns(g) for g in per_timepoint],
        values=times_min,
        sigma=0.0,  # a splat is instantaneous; it must not smear across frames
    )
    aprint(f"Stacked {stacked.n_splats:,} splats over {len(per_timepoint)} timepoints")
    return stacked


def build_lod(stacked: GSplatData) -> GSplatData:
    """Give the 4D archive its substitutive ladder, with time a hard barrier."""
    from luxar.gsplats.lod import RecipeParams, build_recipe

    return build_recipe(
        stacked,
        "levels",
        RecipeParams(
            compression_factor=LOD_COMPRESSION_FACTOR,
            levels=LOD_LEVELS,
            n_lods=LOD_N_LODS,
            coarsen_dims=LOD_COARSEN_DIMS,
            device=DEVICE or "auto",
        ),
    )


# =============================================================================
# The acquisition cage
# =============================================================================
#: The 12 edges of a box, as index pairs into the 8 corners of :func:`_corners`
#: (bit i of a corner index selects max over min on axis i).
_BOX_EDGES = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),  # along axis 2
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),  # along axis 1
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # along axis 0
)


def _corners(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    """The 8 corners of an axis-aligned box, bit i selecting max on axis i."""
    return np.array(
        [
            [bmax[a] if (i >> (2 - a)) & 1 else bmin[a] for a in range(3)]
            for i in range(8)
        ],
        dtype=np.float32,
    )


def cage_lines(
    bmin: np.ndarray, bmax: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A wireframe box plus a grid ruled on its six faces.

    Returns ``(vertices, colors, widths)`` for ``line_type="segments"`` —
    consecutive vertex PAIRS are independent edges, so nothing ever joins the
    end of one grid line to the start of the next.

    The grid is anchored at 0 in scene coordinates (the box centre), not at the
    box minimum, so the rulings are symmetric about the specimen and a reading
    off them is a signed distance from the middle of the acquisition. Lines that
    land within a hair of a face are dropped, since the box edge already draws
    there and a doubled line reads as a brighter one.

    Colours are authored in sRGB and returned linearized, because the viewer
    composites in linear light.
    """
    segs: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    widths: list[float] = []

    corners = _corners(bmin, bmax)
    for a, b in _BOX_EDGES:
        segs.append(np.stack([corners[a], corners[b]]))
        cols.append(np.tile(np.float32(BOX_EDGE_COLOR) ** 2.2, (2, 1)))
        widths += [BOX_EDGE_WIDTH_UM] * 2

    eps = 1e-3 * float(np.max(bmax - bmin))
    for face_axis in range(3):
        u, v = [a for a in range(3) if a != face_axis]
        for face_at in (bmin[face_axis], bmax[face_axis]):
            # Rule the face along u (spanning v), then along v (spanning u).
            for ruled, spanned in ((u, v), (v, u)):
                lo = np.ceil(bmin[ruled] / step) * step
                for pos in np.arange(lo, bmax[ruled] + eps, step):
                    if min(abs(pos - bmin[ruled]), abs(pos - bmax[ruled])) < eps:
                        continue  # coincides with a box edge already drawn
                    a = np.empty(3, dtype=np.float32)
                    a[face_axis] = face_at
                    a[ruled] = pos
                    b = a.copy()
                    a[spanned], b[spanned] = bmin[spanned], bmax[spanned]
                    segs.append(np.stack([a, b]))
                    cols.append(np.tile(np.float32(GRID_LINE_COLOR) ** 2.2, (2, 1)))
                    widths += [GRID_LINE_WIDTH_UM] * 2

    return (
        np.concatenate(segs).astype(np.float32),
        np.concatenate(cols).astype(np.float32),
        np.asarray(widths, dtype=np.float32),
    )


# =============================================================================
# Data resolution
# =============================================================================
def load_or_build_gsplats() -> GSplatData:
    """Return the 4D stacked splats: hosted archive, local rebuild, or a refit."""
    if not RECOMPUTE:
        try:
            hosted = load_dataset_gsplats(DEMO_NAME, [GSPLATS_FILE])
            if hosted is not None:
                return hosted[0]
        except DatasetUnavailable:
            aprint("Precomputed 4D fit unavailable (Git LFS asset not pulled).")
        local = load_local_fit_gsplats_at([LOCAL_FIT], label=DEMO_NAME)
        if local is not None:
            return local[0]
        aprint(f"Falling back to download + fit (one-time; cached at {LOCAL_FIT}).")

    warn_if_no_cuda_gpu()
    array = open_lsm()
    frames = select_timepoints(int(array.shape[0]), MAX_TIMEPOINTS)
    fits = fit_all_timepoints(array, frames)
    times_min = [frame * FRAME_INTERVAL_S / 60.0 for frame in frames]
    stacked = combine_to_4d(fits, times_min)

    with asection("Building the LOD ladder"):
        laddered = build_lod(stacked)
        LOCAL_FIT.parent.mkdir(parents=True, exist_ok=True)
        laddered.save(
            LOCAL_FIT,
            encoding_mode=EncodingMode.MEMORY,
            include_fitting_info=True,
            compress="zip",
            zip_deflate=True,
        )
    loaded = load_local_fit_gsplats_at([LOCAL_FIT], label=DEMO_NAME)
    assert loaded is not None, LOCAL_FIT  # just written
    return loaded[0]


# =============================================================================
# Scene
# =============================================================================
def create_luxar_scene(stacked: GSplatData, output_path: Path) -> Path:
    """Build the 4D scene: one gsplats node, one cage layer, overlays."""
    times = np.unique(stacked.centers[:, 3])
    n_timepoints = int(times.size)
    t_min, t_max = float(times.min()), float(times.max())
    step_min = (t_max - t_min) / max(n_timepoints - 1, 1)

    # Time is a DISCRETE viewer dimension: navigation snaps to multiples of
    # `step` anchored at 0 and a chunk is only fetched within a quarter-step of
    # the snapped position, so a timepoint sitting off that grid produces a
    # slider stop that renders nothing at all — silently. Cheap to check here,
    # invisible if it ever stops holding.
    off_grid = np.abs(times - np.round(times / step_min) * step_min).max()
    if off_grid > 0.01 * step_min:
        raise RuntimeError(
            f"timepoints are not uniformly spaced (worst offset {off_grid:.4g} "
            f"min against a {step_min:.4g} min step); the viewer's discrete "
            "navigation would land on empty stops"
        )

    bmin, bmax = acquisition_box_um()

    with asection("Creating the 4D zebrafish scene"):
        aprint(f"Splats: {stacked.n_splats:,} over {n_timepoints} timepoints")
        aprint(f"Time: 0 to {t_max:.0f} min, step {step_min:.2f} min")
        aprint(f"Box: {' x '.join(f'{s:.0f}' for s in (bmax - bmin))} um")

        # Centre-column order is (Z, Y, X, time) — the order the fits were
        # stacked in — so the Dimensions list must follow it exactly.
        dims = Dimensions(
            [
                Dimension(
                    "Z", unit="um", display=True, range=(float(bmin[0]), float(bmax[0]))
                ),
                Dimension(
                    "Y", unit="um", display=True, range=(float(bmin[1]), float(bmax[1]))
                ),
                Dimension(
                    "X", unit="um", display=True, range=(float(bmin[2]), float(bmax[2]))
                ),
                # Real minutes, not a frame index: the LSM records a 120.01 s
                # interval, so the slider can read in the unit the biology
                # happens in. Values sit exactly on multiples of `step`, which
                # the viewer's discrete navigation snaps to.
                Dimension(
                    "Time",
                    unit="min",
                    display=False,
                    discrete=True,
                    step=step_min,
                    range=(t_min, t_max),
                ),
            ]
        )

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                citation=DEMO_META["citation"],
                viewer_config=ViewerConfig(cinematic_mode=True, tone_mapping="ACES"),
            )
            scene.attrs["title"] = "GSplats: Zebrafish Gastrulation, 4D Time-Lapse"
            scene.attrs["description"] = (
                f"A living zebrafish embryo (cxcr4a morphant) imaged by confocal "
                f"laser-scanning microscopy through gastrulation: {n_timepoints} "
                f"timepoints over {t_max / 60:.1f} hours, {stacked.n_splats:,} "
                f"Gaussian splats in one 4D node with time as its fourth centre "
                f"column. The labelled endodermal cells start as a tight cluster "
                f"and spread over the yolk; the wireframe cage is the imaged "
                f"volume, ruled every {GRID_STEP_UM:.0f} um. Play the Time slider "
                f"to run the recording; press L for per-layer controls. "
                f"Zenodo 1211599, DOI 10.5281/zenodo.1211599, CC BY-SA 4.0."
            )

            scene.add_gsplats_from_data(
                "endoderm",
                stacked,
                lod_group=True,
                dim_order=["Z", "Y", "X", "Time"],
                extend_to_all=[],
                colormap=COLORMAP,
                opacity=VOLUME_OPACITY,
                absorption=VOLUME_ABSORPTION,
                blending_mode="volumetric",
                layer=True,
            )

            verts, colors, widths = cage_lines(bmin, bmax, GRID_STEP_UM)
            scene.add_lines(
                "acquisition cage",
                vertices=verts,
                widths=widths,
                colors=colors,
                line_type="segments",
                dim_order=["Z", "Y", "X"],
                # The cage is the instrument, not the specimen: it is stored
                # once and shown at every timepoint.
                extend_to_all=["Time"],
                blending_mode="additive",
                opacity=0.9,
                layer=True,
            )

            scene.add_text(
                "Zebrafish gastrulation",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                f"{n_timepoints} timepoints over {t_max / 60:.1f} h\n"
                f"{stacked.n_splats:,} Gaussian splats\n"
                f"cage {bmax[2] - bmin[2]:.0f} x {bmax[1] - bmin[1]:.0f} x "
                f"{bmax[0] - bmin[0]:.0f} um, grid {GRID_STEP_UM:.0f} um",
                position=(0.02, 0.10),
                font_size=0.018,
                font="mono",
                color="white",
                line_height=1.45,
            )
            add_demo_caption(
                scene,
                "Confocal laser-scanning microscopy • endoderm label",
                DEMO_META.get("citation"),
            )

    aprint(f"Scene saved: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Resolve the 4D splats, build the scene, and optionally serve it."""
    aprint("=" * 70)
    aprint("GSplats Demo: 4D Zebrafish Embryo Time-Lapse")
    aprint("=" * 70)
    aprint("One 4D node • time as a real centre column • gridded acquisition cage")
    aprint("")

    output_path = get_demos_output_dir() / SCENE_NAME

    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: launching viewer…")
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    scene_path = create_luxar_scene(load_or_build_gsplats(), output_path)

    if not NO_SERVE:
        aprint("\nLaunching viewer… (press L for the Layers panel)")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
