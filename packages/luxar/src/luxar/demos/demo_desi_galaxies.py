#!/usr/bin/env python3
"""DESI DR1 — The Cosmic Web in 3D (the whole ~9.75M-object catalog)

Renders the large-scale structure of the Universe as a point cloud built from
the Dark Energy Spectroscopic Instrument's first data release (DESI DR1). Each
point is a real galaxy or quasar with a measured spectroscopic redshift; the
redshift is turned into a comoving distance so the sky positions (RA, Dec) plus
depth become true 3D Cartesian coordinates in megaparsecs. You are sitting at
the observer's origin, looking out at the two DESI footprint caps fanning into
the cosmic web — filaments, voids, and the shells of the baryon-acoustic scale.

================================================================================
(RA, Dec, redshift) → COMOVING Mpc → 3D POINT CLOUD
================================================================================

DESI measured spectroscopic redshifts for tens of millions of targets. The DR1
large-scale-structure (LSS) "clustering" catalogs are already reduced to
high-confidence, deduplicated extragalactic redshifts, split by tracer:
    BGS  — Bright Galaxy Survey   (nearby, z ≲ 0.4)
    LRG  — Luminous Red Galaxies  (z ~ 0.4–1.1)
    ELG  — Emission-Line Galaxies (z ~ 0.6–1.6)
    QSO  — Quasars                (z ~ 0.8–4, the most distant tracers)

Two colorings ship, toggled in the Layers panel:
  • By tracer — the four populations in distinct colors (naturally layered by
    distance, since each tracer occupies a redshift shell).
  • By redshift — a continuous colormap of lookback depth across the whole set.

DATA SOURCE & CITATION
----------------------
DESI DR1 LSS clustering catalogs (iron / v1.5):
    https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/
DESI Collaboration (2026), "Data Release 1 of the Dark Energy Spectroscopic
    Instrument", arXiv:2503.14745. Data released under CC BY 4.0.
    Acknowledgment: this product uses data obtained with the Dark Energy
    Spectroscopic Instrument (DESI). Derived 3D positions are a transform of the
    public RA/Dec/redshift columns.

SELF-CONTAINED / CACHING
------------------------
On a fresh machine this demo bootstraps itself with no manual steps:
  1. Fast path: a fully-built scene (both LOD colorings, ~73 MB) resolved through
     the checksum-verified dataset manifest from cache, Git LFS, or Zenodo. It is
     unzipped once into the demos output dir, then re-laddered once if its stored
     structure predates the current per-node safety ceiling.
  2. If that asset is unavailable, ``--recompute`` (or a missing asset)
     AUTOMATICALLY downloads the ~1 GB of DR1 LSS catalogs to
     ``~/.cache/luxar/desi_galaxies/`` (resumable), reads them with ``astropy``,
     converts (RA, Dec, z) → comoving Mpc, keeps every row, and builds the
     substitutive LOD (GPU-accelerated but slow on CPU-only machines — which is
     exactly why the built scene is hosted precomputed).
     If the DESI host is unavailable, rerun without ``--recompute`` to resolve
     the precomputed scene through the dataset manifest and published record.

USAGE
-----
    python demo_desi_galaxies.py [--recompute] [--no-serve] [--serve-only]

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan,  'C': fly controls
"""

DEMO_META = {
    "key": "desi_galaxies",
    "title": "DESI DR1 — The Cosmic Web in 3D (~9.75M-object catalog)",
    "description": "All ~9.75M real DESI DR1 galaxies and quasars placed in 3D by redshift.",
    "category": "astronomy",
    "geometry": "points",
    "requirements": {
        "download_mb": 73,
        "compute": "heavy",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["desi_galaxies"],
    "outputs": ["desi_galaxies"],
    "citation": {
        "short": "DESI Collaboration 2026 (DR1)",
        "ref": "DESI Collaboration 2026",
        "doi": "10.48550/arXiv.2503.14745",
        "license": "CC BY 4.0",
    },
}

from pathlib import Path
from typing import Any, Final, Optional

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import (
    CameraConfig,
    Dimension,
    Dimensions,
    LuxarZarrCompiler,
    ViewerConfig,
)
from luxar._zarr_compat import consolidate, open_group
from luxar.demos import (
    DatasetUnavailable,
    add_demo_caption,
    ensure_dataset,
    launch_viewer,
    parse_demo_flags,
    stamp_input_digests,
    substitutive_lod_or_flat,
)
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
from luxar.utils.lod_breakpoints import capped_stream_cuts
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = (
    "https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5"
)

# Tracer -> (NGC file, SGC file, display color RGB, tracer id).
# Colors: BGS warm/near, LRG red, ELG teal, QSO violet/far — a distance gradient.
TRACERS = {
    "BGS": {
        "files": (
            "BGS_BRIGHT_NGC_clustering.dat.fits",
            "BGS_BRIGHT_SGC_clustering.dat.fits",
        ),
        "color": (1.0, 0.72, 0.30),  # warm amber
        "id": 0,
    },
    "LRG": {
        "files": ("LRG_NGC_clustering.dat.fits", "LRG_SGC_clustering.dat.fits"),
        "color": (0.94, 0.28, 0.26),  # red
        "id": 1,
    },
    "ELG": {
        "files": (
            "ELG_LOPnotqso_NGC_clustering.dat.fits",
            "ELG_LOPnotqso_SGC_clustering.dat.fits",
        ),
        "color": (0.20, 0.80, 0.68),  # teal
        "id": 2,
    },
    "QSO": {
        "files": ("QSO_NGC_clustering.dat.fits", "QSO_SGC_clustering.dat.fits"),
        "color": (0.55, 0.45, 0.95),  # violet
        "id": 3,
    },
}
TRACER_ORDER = ["BGS", "LRG", "ELG", "QSO"]

# Exact byte sizes of the (versioned, immutable) v1.5 catalog files, so the
# resumable download can verify each file landed complete.
FILE_SIZES = {
    "BGS_BRIGHT_NGC_clustering.dat.fits": 340464960,
    "BGS_BRIGHT_SGC_clustering.dat.fits": 122624640,
    "LRG_NGC_clustering.dat.fits": 143196480,
    "LRG_SGC_clustering.dat.fits": 64272960,
    "ELG_LOPnotqso_NGC_clustering.dat.fits": 205819200,
    "ELG_LOPnotqso_SGC_clustering.dat.fits": 69024960,
    "QSO_NGC_clustering.dat.fits": 83298240,
    "QSO_SGC_clustering.dat.fits": 45178560,
}

# DESI fiducial cosmology (Planck-2018 base ΛCDM): H0=67.36, Om0≈0.3137.
COSMO_H0 = 67.36
COSMO_OM0 = 0.3137

DEMO_NAME = "desi_galaxies"
DERIVED_FILE = "desi_dr1_cosmic_web.npz"
SCENE_ZIP_FILE = "desi_dr1_cosmic_web.luxar.zarr.zip"

CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME
DERIVED_CACHE = CACHE_DIR / DERIVED_FILE

# Redshift window: keep good extragalactic redshifts; drop the tiny z≈0 blunders
# and the sparse very-high-z tail that just stretches the scene.
Z_MIN = 0.001
Z_MAX = 4.0

# Display / LOD parameters.
POINT_RADIUS = 1.2  # Mpc (visualization scale)
SCENE_INTENSITY = 0.05
# Stay below the 5,591,040-point element-texture cap on a conservative
# 4096-class GPU. The margin matches the globe demos and keeps every finest-LOD
# partition leaf drawable without viewer-side tail clamping.
SCENE_MAX_POINTS_PER_NODE: Final = 4_000_000
# No row cap: the scene carries the WHOLE DR1 catalog, ~9.75M objects.
#
# This was briefly capped at 1.25M (#1812) because the finest child downloaded
# all 9.75M rows in ONE commit and blocked the main thread. That diagnosis named
# the wrong culprit. The payload was unstreamable because the ladder under it was
# geometric: `stream:<c>` doubles until it reaches n, so its LAST increment is
# always n/2 whatever the base is — 4.88M points at full density, and still
# 625K at the 1.25M cap. Capping the catalog shrank that final commit without
# fixing its shape, and cost 87% of the survey to do it.
#
# `streaming_breakpoints` fixes the shape instead (see below), so the production
# demo leaves this at `None` (every row). The knob remains useful for focused
# local rebuilds and tests that need to exercise the sampling path explicitly.
SCENE_MAX_POINTS: int | None = None
# The shipped archive is the canonical sample. This seed makes recomputation
# repeatable within a NumPy release, not bit-stable across future NumPy releases.
SCENE_SAMPLE_SEED = 0
LOD = dict(compression_factor=8, levels=2, device="auto")

# First additive rung, in points. Small enough to land in a single zarr chunk,
# so time-to-first-pixel on the eager coarsest level is one range request.
SCENE_FIRST_CHUNK = 2_000

# Hard ceiling on ONE additive increment, in points. This is the number that
# makes the full catalog streamable: no single commit may block the main thread,
# whatever the level is worth in total. Set below the 1,000,000 that
# `scripts/check_demo_ladders.py` fails a leaf at, with margin.
SCENE_MAX_COMMIT = 900_000


def streaming_breakpoints(
    n: int,
    first_chunk: int = SCENE_FIRST_CHUNK,
    max_commit: int = SCENE_MAX_COMMIT,
) -> list[int]:
    """Cumulative additive cuts that double early, then step by a fixed cap.

    A pure ``stream:<c>`` ladder doubles all the way to ``n``, so its final
    increment is ``n/2`` — unstreamable once ``n`` is large, and the real reason
    the full catalog was thought to need a row cap. This keeps the geometric
    ramp, which is what makes first paint cheap, but stops before its next
    increment would exceed ``max_commit`` and finishes in equal steps of that
    size. Largest commit is therefore ``max_commit`` at ANY ``n``; with the
    defaults the geometric head totals 1,024,000 points.

    One list serves every substitutive level: ``_validate_counts`` clamps a
    cumulative list to the level's own ``n`` and stops there, so the 152K level
    simply takes the geometric head and the 9.75M level takes the whole thing.
    That also keeps the small levels laddered, which a sibling-aware
    ``stream:`` base would not have done at this scale.

    Reaching the coarser sibling's size — the point where swapping in this level
    is worth it — costs only the geometric head plus a step, ~20% of the
    finest level's payload, so the upgrade does not "wait until fully loaded".

    The schedule itself now lives in :func:`luxar.utils.lod_breakpoints.
    capped_stream_cuts`, shared with every other demo that ladders a
    multi-million-element leaf; this wrapper only pins this scene's two numbers.
    """
    return capped_stream_cuts(n, first_chunk, max_commit)


# The shipped scene must carry a real ladder on its finest level. Anyone whose
# `datasets/demos/` copy predates that gets a stale all-or-nothing scene and no
# diagnostic, because `main()` short-circuits on an existing output directory.
SCENE_MIN_SUBLODS = 3

# A reused scene from the temporary #1812 cap can have a well-shaped ladder but
# still contain only 1.25M of the fixed 9.75M-row DR1 catalog. Keep the threshold
# below the exact count to tolerate metadata/version variation while making that
# incomplete payload unmistakably stale.
SCENE_MIN_POINTS = 9_000_000

# The orbit pivot, in scene coordinates: the OBSERVER, i.e. the origin. See the
# comment in `create_scene` for why a bounding-box centre is wrong for this
# dataset. Kept as a constant because `ensure_origin_framing` has to recognise
# the same point in a scene it did not build.
SCENE_CAMERA_TARGET = (0.0, 0.0, 0.0)

# How far a reused scene's camera target may sit from the origin before it is
# treated as a stale bounding-box pivot. In Mpc, and generous: the wrong pivots
# this catches are hundreds to thousands of Mpc out (the pre-fix scene targeted
# z = 1,159 Mpc), while a correctly authored target is exactly (0, 0, 0).
SCENE_TARGET_TOLERANCE_MPC = 1.0

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

Arbol.max_depth = 5


# =============================================================================
# Pure helpers (unit-tested)
# =============================================================================


def sample_scene_catalog(
    positions: np.ndarray,
    redshift: np.ndarray,
    tracer_ids: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Take one deterministic, row-aligned sample for both scene colorings."""
    n_rows = len(positions)
    if len(redshift) != n_rows or len(tracer_ids) != n_rows:
        raise ValueError(
            "positions, redshift, and tracer_ids must have the same number of rows"
        )
    if max_points < 1:
        raise ValueError(f"max_points must be >= 1, got {max_points}")
    if n_rows <= max_points:
        return positions, redshift, tracer_ids

    rng = np.random.default_rng(SCENE_SAMPLE_SEED)
    indices = np.sort(rng.choice(n_rows, size=max_points, replace=False))
    return positions[indices], redshift[indices], tracer_ids[indices]


def radec_z_to_xyz(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    comoving_mpc: np.ndarray,
) -> np.ndarray:
    """Spherical sky coords + comoving distance → Cartesian Mpc (observer at 0).

    x = d·cos(dec)·cos(ra),  y = d·cos(dec)·sin(ra),  z = d·sin(dec).
    """
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    d = np.asarray(comoving_mpc, dtype=np.float64)
    cos_dec = np.cos(dec)
    x = d * cos_dec * np.cos(ra)
    y = d * cos_dec * np.sin(ra)
    z = d * np.sin(dec)
    return np.column_stack([x, y, z]).astype(np.float32)


def tracer_colors(tracer_ids: np.ndarray) -> np.ndarray:
    """Map per-point tracer ids → (N, 3) float32 RGB via the tracer palette."""
    palette = np.zeros((len(TRACER_ORDER), 3), dtype=np.float32)
    for name in TRACER_ORDER:
        palette[TRACERS[name]["id"]] = TRACERS[name]["color"]
    ids = np.asarray(tracer_ids, dtype=np.intp)
    return palette[ids]


def redshift_colors(redshift: np.ndarray) -> np.ndarray:
    """Map redshift → (N, 3) float32 RGB via the turbo colormap.

    The colormap is baked into per-point colors (rather than passed to the
    viewer as live ``scalars``) so that EVERY substitutive-LOD level shows the
    same depth gradient — a scalars layer only keeps the live colormap on its
    finest level, leaving coarse (zoomed-out) levels with baked colors that no
    longer track the map. Baking keeps the near→far turbo gradient consistent
    at all zooms. The range is robust (min → 98th percentile) so the populated
    z ≲ 1.6 bulk spans the full colormap and the sparse high-z tail clamps to
    the hot end instead of compressing everything into turbo's cold end.
    """
    from luxar.colormaps import scalars_to_colors

    z = np.asarray(redshift, dtype=np.float32)
    vmin = float(np.nanmin(z)) if z.size else 0.0
    vmax = float(np.nanpercentile(z, 98)) if z.size else 1.0
    if not vmax > vmin:
        vmax = vmin + 1.0
    return np.asarray(
        scalars_to_colors(z, "turbo", vmin=vmin, vmax=vmax), dtype=np.float32
    )


def quantize_positions(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize XYZ (Mpc) to int16 per-axis with an affine (offset, scale).

    Returns (q_int16, offset_f32[3], scale_f32[3]) such that
    ``positions ≈ q * scale + offset``. Halves on-disk size vs float32 with
    ~sub-Mpc precision over a several-Gpc box (visually lossless for a cloud).
    """
    pos = np.asarray(positions, dtype=np.float64)
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    scale = span / 65534.0
    q = np.round((pos - lo) / scale).astype(np.int32) - 32767
    q = np.clip(q, -32767, 32767).astype(np.int16)
    offset = (lo + 32767.0 * scale).astype(np.float32)
    return q, offset, scale.astype(np.float32)


def dequantize_positions(
    q: np.ndarray, offset: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Inverse of :func:`quantize_positions` → float32 XYZ (Mpc)."""
    return (
        np.asarray(q, dtype=np.float32) * np.asarray(scale, dtype=np.float32)
        + np.asarray(offset, dtype=np.float32)
    ).astype(np.float32)


def save_derived(
    path: Path, positions: np.ndarray, redshift: np.ndarray, tracer_ids: np.ndarray
) -> None:
    """Write the compact derived cloud (quantized XYZ + f16 z + uint8 tracer)."""
    q, offset, scale = quantize_positions(positions)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            pos_q=q,
            pos_offset=offset,
            pos_scale=scale,
            redshift=np.asarray(redshift, dtype=np.float16),
            tracer_id=np.asarray(tracer_ids, dtype=np.uint8),
        )
    tmp.rename(path)


def load_derived(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a derived cloud → (positions_f32, redshift_f32, tracer_id_uint8)."""
    with np.load(path) as data:
        positions = dequantize_positions(
            data["pos_q"], data["pos_offset"], data["pos_scale"]
        )
        redshift = data["redshift"].astype(np.float32)
        tracer_ids = data["tracer_id"].astype(np.uint8)
    return positions, redshift, tracer_ids


# =============================================================================
# Download + build (IO)
# =============================================================================


class DESICatalogDownloadError(RuntimeError):
    """A DESI source catalog could not be downloaded."""


def _catalog_download_error_message(url: str, exc: BaseException) -> str:
    """Return actionable diagnostics while preserving HTTP context."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    reason = getattr(response, "reason", None)
    if status is not None:
        detail = f"HTTP {status}"
        if reason:
            detail += f" {reason}"
    else:
        detail = f"{type(exc).__name__}: {exc}"

    return (
        "Could not download a DESI DR1 source catalog.\n"
        f"  URL: {url}\n"
        f"  Error: {detail}\n\n"
        "The DESI data host may be temporarily unavailable or under "
        "maintenance.\n"
        "  Check host availability: https://data.desi.lbl.gov/\n\n"
        "The precomputed scene does not need the source catalogs. Rerun this "
        "demo without --recompute to fetch it through the dataset manifest "
        "from the published record. If that manifest fetch also fails, it is "
        "a separate problem from the DESI host outage and is reported before "
        "the catalog fallback starts."
    )


def download_catalogs() -> list[tuple[str, Path]]:
    """Download the 8 DR1 LSS clustering FITS files (resumable). Returns
    (tracer_name, local_path) for each NGC/SGC file."""
    from requests.exceptions import RequestException

    from luxar.demos import robust_download

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: list[tuple[str, Path]] = []
    with asection("Downloading DESI DR1 LSS catalogs (~1 GB)"):
        for name in TRACER_ORDER:
            for fname in TRACERS[name]["files"]:
                dest = CACHE_DIR / fname
                url = f"{BASE_URL}/{fname}"
                aprint(f"{name}: {fname}")
                try:
                    robust_download(
                        url,
                        dest,
                        max_retries=5,
                        timeout=1800,
                        expected_size=FILE_SIZES.get(fname),
                    )
                except RequestException as exc:
                    raise DESICatalogDownloadError(
                        _catalog_download_error_message(url, exc)
                    ) from exc
                out.append((name, dest))
    return out


def build_pointcloud() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download + read the catalogs, convert to comoving XYZ, and cache the
    derived cloud. Returns (positions_f32, redshift_f32, tracer_id_uint8)."""
    from astropy.cosmology import FlatLambdaCDM
    from astropy.table import Table

    files = download_catalogs()
    cosmo = FlatLambdaCDM(H0=COSMO_H0, Om0=COSMO_OM0)

    ra_all, dec_all, z_all, tid_all = [], [], [], []
    with asection("Reading catalogs + selecting good redshifts"):
        for name, path in files:
            t = Table.read(path)
            ra = np.asarray(t["RA"], dtype=np.float64)
            dec = np.asarray(t["DEC"], dtype=np.float64)
            z = np.asarray(t["Z"], dtype=np.float64)
            keep = np.isfinite(z) & (z > Z_MIN) & (z < Z_MAX)
            ra_all.append(ra[keep])
            dec_all.append(dec[keep])
            z_all.append(z[keep])
            tid_all.append(
                np.full(int(keep.sum()), TRACERS[name]["id"], dtype=np.uint8)
            )
            aprint(f"{name} {path.name}: {int(keep.sum()):,} good")

    ra = np.concatenate(ra_all)
    dec = np.concatenate(dec_all)
    z = np.concatenate(z_all)
    tracer_ids = np.concatenate(tid_all)

    with asection(f"Converting {len(z):,} redshifts → comoving Mpc"):
        # Interpolate comoving_distance over a redshift grid (exact per-object is
        # slow for tens of millions); the grid is dense enough to be sub-Mpc.
        zgrid = np.linspace(Z_MIN, Z_MAX, 4000)
        dgrid = cosmo.comoving_distance(zgrid).to("Mpc").value
        comoving = np.interp(z, zgrid, dgrid)
        positions = radec_z_to_xyz(ra, dec, comoving)
        redshift = z.astype(np.float32)
        aprint(f"✓ {len(positions):,} galaxies/quasars in the cosmic web")

    save_derived(DERIVED_CACHE, positions, redshift, tracer_ids)
    return positions, redshift.astype(np.float32), tracer_ids


def load_or_build() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the cosmic-web cloud for the build path (reuses a cached derived
    npz if present, else downloads + converts)."""
    if not RECOMPUTE and DERIVED_CACHE.exists() and DERIVED_CACHE.stat().st_size > 1000:
        aprint(f"Loading cached derived cloud: {DERIVED_CACHE}")
        return load_derived(DERIVED_CACHE)
    return build_pointcloud()


def extract_shipped_scene(zip_path: Path, output_path: Path) -> None:
    """Unzip the precomputed built scene into ``output_path``.

    Extracts into a sibling ``.part`` dir and renames into place atomically, so
    an interrupted extraction never leaves a half-populated ``output_path`` that
    a later run would mistake for a complete cached scene.
    """
    import shutil
    import zipfile

    with asection("Unpacking precomputed scene"):
        aprint(f"Source: {zip_path.name} ({zip_path.stat().st_size / 1e6:.0f} MB)")
        staging = output_path.parent / (output_path.name + ".part")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(staging)
        staging.rename(output_path)
        aprint(f"Scene ready: {output_path}")


def ensure_origin_framing(scene_path: Path) -> bool:
    """Re-pin a reused scene's orbit pivot to the origin, and report whether it had to.

    ``main()`` prefers an existing ``datasets/demos/`` scene over the shipped
    asset, so a copy built before the pivot was moved to the observer keeps its
    bounding-box camera forever — the demo opens swinging the local universe
    around a point ~1.2 Gpc out in the ELG shell, and nothing says why. The
    ladder checks in :func:`warn_if_scene_is_stale` never saw this: the geometry
    of such a scene is fine, it is only framed wrong.

    Only the ``target`` is rewritten. Position, fov and the clipping planes are
    left as the older build computed them — they are consistent with each other
    and with the cloud, and the complaint a bounding-box pivot causes is the
    ORBIT CENTRE, not the distance. Rewriting one attribute also keeps this a
    metadata touch rather than a rebuild; ``--recompute`` remains the way to get
    the current framing in full.

    Returns True when the scene was already framed on the origin.
    """
    try:
        root = open_group(scene_path, mode="r+")
        viewer_config = dict(root.attrs.get("viewer_config") or {})
        camera = dict(viewer_config.get("camera") or {})
        target = camera.get("target")
    except Exception as exc:  # pragma: no cover - diagnostics only
        aprint(f"  ⚠ Could not inspect {scene_path} for its camera target: {exc}")
        return True

    # No authored camera at all: the viewer auto-frames on the bounding box,
    # which is the very thing this scene must not do. Say so rather than
    # inventing a distance we cannot derive without the catalog.
    if not camera or target is None:
        aprint(
            f"  ⚠ This scene carries no authored camera, so the viewer will "
            f"auto-frame it on its bounding box instead of the observer at the "
            f"origin. Rebuild it with:\n"
            f"      luxar demo run desi_galaxies -- --recompute\n"
            f"    or delete the scene and re-run — the fetch path re-ladders "
            f"the record under the current recipe:\n"
            f"      rm -rf {scene_path}"
        )
        return False

    offset = float(np.linalg.norm(np.asarray(target, dtype=np.float64)))
    if offset <= SCENE_TARGET_TOLERANCE_MPC:
        return True

    camera["target"] = list(SCENE_CAMERA_TARGET)
    viewer_config["camera"] = camera
    root.attrs["viewer_config"] = viewer_config
    consolidate(root)
    aprint(
        f"  🎥 Re-pinned this scene's orbit pivot to the observer at the origin "
        f"(it was {offset:,.0f} Mpc away, a bounding-box centre from an older "
        f"build). Run with --recompute for the current framing in full."
    )
    return False


def _warn_if_node_exceeds_capacity(
    scene_path: Path, layer_name: str, max_node_points: int
) -> None:
    if max_node_points <= SCENE_MAX_POINTS_PER_NODE:
        return

    aprint(
        f"  ⚠ This scene's '{layer_name}' finest level's largest single "
        f"node contains {max_node_points:,} points, above the current "
        f"{SCENE_MAX_POINTS_PER_NODE:,}-point demo ceiling. It lacks the "
        "current per-node safety margin, and larger nodes can silently "
        "lose their tail on a 4096-class GPU. Rebuild it with:\n"
        "      luxar demo run desi_galaxies -- --recompute\n"
        "    or delete the scene and re-run — the fetch path re-ladders "
        "the record under the current recipe:\n"
        f"      rm -rf {scene_path}"
    )


def _numbered_children(group: Any, prefix: str) -> list[str]:
    return sorted(
        (key for key in group.group_keys() if key.startswith(prefix)),
        key=lambda key: int(key.split("_", 1)[1]),
    )


def _finest_level(layer: Any) -> Any:
    """Return the finest substitutive level, or the layer itself when flat."""
    child_names = _numbered_children(layer, "child_")
    return layer[child_names[-1]] if child_names else layer


def _partition_nodes(level: Any) -> list[Any]:
    """Return a level's partition leaves, or the level itself when flat."""
    if level.attrs.get("kind") != "partition":
        return [level]
    return [level[name] for name in _numbered_children(level, "part_")]


def _finest_nodes(layer: Any) -> list[Any]:
    return _partition_nodes(_finest_level(layer))


def scene_exceeds_node_capacity(scene_path: Path) -> bool:
    """True when any layer's finest level holds a node above the demo ceiling.

    The decision half of :func:`_warn_if_node_exceeds_capacity`: that reports,
    this answers. Kept separate so the fetch path can ACT on the same condition
    the warning describes rather than duplicating the threshold.

    FAILS CLOSED. A scene this cannot inspect returns True — "repair it" — not
    False. Unknown is not the same as fine, and the asymmetry is cheap in one
    direction only: re-laddering a scene that was already compliant costs a few
    minutes of compute and changes nothing, while skipping a scene that was NOT
    compliant renders a node that can lose its tail with no further signal. The
    cost of being wrong is minutes one way and a silently truncated catalog the
    other, so the tie goes to repairing. This whole defect existed because a
    check that could not tell said nothing and carried on.
    """
    import zarr

    try:
        root = zarr.open(str(scene_path), mode="r")
    except Exception as exc:
        aprint(
            f"  ⚠ Could not open {scene_path} to check node capacity ({exc}); "
            "treating it as needing a re-ladder rather than assuming it is fine."
        )
        return True

    for layer_name in ("By tracer type", "By redshift"):
        try:
            for node in _finest_nodes(root[layer_name]):
                if int(node.attrs.get("n_points", 0)) > SCENE_MAX_POINTS_PER_NODE:
                    return True
        except Exception as exc:
            aprint(
                f"  ⚠ Could not inspect [{layer_name}] for node capacity ({exc}); "
                "treating it as needing a re-ladder rather than assuming it is fine."
            )
            return True
    return False


def warn_if_scene_is_stale(scene_path: Path) -> None:
    """Warn when a reused scene is incomplete or has an unsafe streaming ladder.

    ``main()`` reuses an existing ``datasets/demos/`` scene unconditionally, so a
    user who built an older version would otherwise keep either the temporary
    1.25M-row sample or an all-or-nothing 9.75M-point finest child forever. Warn
    loudly, name both remedies, and carry on: the old scene still renders, but
    incompletely or slowly. Both laddered layers are checked, since they share
    the same LOD.
    """
    import zarr

    try:
        root = zarr.open(str(scene_path), mode="r")
    except Exception as exc:  # pragma: no cover - diagnostics only
        aprint(f"  ⚠ Could not inspect {scene_path} for a streaming ladder: {exc}")
        return

    def streaming_stats(finest) -> tuple[int, int, int, list[int]]:
        sublod_counts = []
        n_points = 0
        max_node_points = 0
        increments = []
        for node in _partition_nodes(finest):
            n_sublods = int(node.attrs.get("n_additive_sublods", 1))
            sublod_counts.append(n_sublods)
            node_points = int(node.attrs.get("n_points", 0))
            n_points += node_points
            max_node_points = max(max_node_points, node_points)
            increments.extend(
                int(node[f"additive_{index}"].attrs.get("n_points", 0) or 0)
                for index in range(n_sublods)
                if f"additive_{index}" in node
            )
        return min(sublod_counts, default=1), n_points, max_node_points, increments

    for layer_name in ("By tracer type", "By redshift"):
        try:
            layer = root[layer_name]
            finest = _finest_level(layer)
            n_sublods, n_points, max_node_points, increments = streaming_stats(finest)
        except Exception as exc:
            aprint(
                f"  ⚠ Could not inspect {scene_path} [{layer_name}] for a "
                f"streaming ladder: {exc}"
            )
            continue

        if n_sublods < SCENE_MIN_SUBLODS:
            aprint(
                f"  ⚠ This scene's '{layer_name}' finest level has no streaming "
                f"ladder (n_additive_sublods={n_sublods}), so it will load "
                "all-at-once and may freeze the browser for a long time. Rebuild "
                "it with:\n"
                "      luxar demo run desi_galaxies -- --recompute\n"
                "    or delete the scene and re-run — the fetch path re-ladders "
                "the record under the current recipe:\n"
                f"      rm -rf {scene_path}"
            )

        if 0 < n_points < SCENE_MIN_POINTS:
            aprint(
                f"  ⚠ This scene's '{layer_name}' finest level contains only "
                f"{n_points:,} points; the current scene carries the full "
                f"~9.75M-object DR1 catalog. Rebuild it with:\n"
                "      luxar demo run desi_galaxies -- --recompute\n"
                "    or delete the scene and re-run — the fetch path re-ladders "
                "the record under the current recipe:\n"
                f"      rm -rf {scene_path}"
            )

        _warn_if_node_exceeds_capacity(scene_path, layer_name, max_node_points)

        # The size that matters is the biggest SINGLE commit, not the level
        # total: a geometric ladder's last increment is n/2, so an old scene can
        # carry five rungs and still hand the main thread millions of points at
        # once. Checking the total instead is what let the 1.25M cap look like a
        # fix (#1812) while the shape stayed broken.
        biggest = max(increments, default=0)
        if biggest > SCENE_MAX_COMMIT:
            aprint(
                f"  ⚠ This scene's '{layer_name}' finest level commits "
                f"{biggest:,} points in one rung, above the current "
                f"{SCENE_MAX_COMMIT:,}-point ceiling — it was built with the old "
                f"geometric ladder ({n_points:,} points, {n_sublods} rungs per part) "
                "and will stall the main thread on that rung. Rebuild it with:\n"
                "      luxar demo run desi_galaxies -- --recompute\n"
                "    or delete the scene and re-run — the fetch path re-ladders "
                "the record under the current recipe:\n"
                f"      rm -rf {scene_path}"
            )


# =============================================================================
# Scene
# =============================================================================


def _finest_level_cloud(root: Any, layer_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Decode one layer's finest-level positions and per-point RGB.

    Walks whatever structure the level happens to have — a bare leaf, an
    additive ladder, or a partition of ladders — and concatenates every
    sub-node, so this reads both the partitioned scene the demo authors and the
    unpartitioned one the record carries.

    NOT :func:`dequantize_positions`. That inverts this demo's own ``.npz``
    scheme (one global ``pos_offset``/``pos_scale``); a COMPILED scene stores
    positions as uint16 against per-chunk ``chunk_bounds``, which only
    ``luxar.encoding``'s decoder knows how to invert. Reaching for the local
    helper here yields plausible-looking garbage coordinates.
    """
    from luxar.encoding.decoder import ArrayDecoder

    decoder = ArrayDecoder()
    layer = root[layer_name]

    positions: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for node in _finest_nodes(layer):
        sub_names = _numbered_children(node, "additive_")
        # An additive ladder stores DISJOINT increments, so concatenating every
        # rung reconstructs the level exactly once — not a prefix, and not a
        # duplicate of the whole.
        sources = [node[name] for name in sub_names] or [node]
        for source in sources:
            if "positions" not in source:
                continue
            positions.append(decoder.decode(source["positions"], root))
            colors.append(decoder.decode(source["colors"], root))

    if not positions:
        raise ValueError(f"{layer_name!r} carries no decodable positions")
    return np.concatenate(positions), np.concatenate(colors)


def _finest_level_colors(root: Any, layer_name: str) -> tuple[int, np.ndarray]:
    """Decode RGB without materialising a second copy of shared positions."""
    from luxar.encoding.decoder import ArrayDecoder

    decoder = ArrayDecoder()
    nodes = _finest_nodes(root[layer_name])
    colors: list[np.ndarray] = []
    for node in nodes:
        sub_names = _numbered_children(node, "additive_")
        sources = [node[name] for name in sub_names] or [node]
        colors.extend(
            decoder.decode(source["colors"], root)
            for source in sources
            if "colors" in source
        )
    if not colors:
        raise ValueError(f"{layer_name!r} carries no decodable colors")
    return sum(int(node.attrs.get("n_points", 0)) for node in nodes), np.concatenate(
        colors
    )


def read_scene_clouds(scene_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Both layers' finest-level clouds, decoded out of a compiled scene."""
    import zarr

    root = zarr.open(str(scene_path), mode="r")
    return {
        layer_name: _finest_level_cloud(root, layer_name)
        for layer_name in ("By tracer type", "By redshift")
    }


def restructure_scene(scene_path: Path) -> Path:
    """Rebuild a fetched scene under the CURRENT authoring recipe.

    The Zenodo record carries a pre-built scene, and it is the only record that
    does — every other demo fetches raw fits/components and structures them at
    authoring time, so it cannot ship a stale STRUCTURE. This one can, and does:
    its finest level is a single unpartitioned node of ~9.75M points, 2.4x the
    :data:`SCENE_MAX_POINTS_PER_NODE` ceiling, which
    :func:`_warn_if_node_exceeds_capacity` has been warning about and carrying
    on from. A warning nobody reads is not a fix, and after the in-repo payload
    is removed (#2354) that record becomes the ONLY source, so the warning's
    own second remedy -- delete and re-unpack a shipped asset -- becomes
    circular.

    So re-ladder instead: decode the record's own points and colours and re-run
    the same ``add_points`` call ``--recompute`` uses, restoring the partition,
    the per-node ceiling and the coverage anchors. No new record version is
    needed, and the structure is derived by current code rather than trusted
    from a years-old build.

    Colours survive EXACTLY: they are stored ``lut_uint8`` with ``lut_mode=row``
    over a float32 LUT, so the decode is lossless (measured: max channel
    delta 0.0 against the in-repo scene). Positions are re-quantised onto the
    new per-part grids, which costs about what one quantisation round already
    costs -- measured against the in-repo scene, per-axis sorted marginals agree
    to <=0.16 Mpc, RMS <=0.05 Mpc, on a cloud spanning ~14,000 Mpc (1.2e-5
    relative).

    Stopgap: reads the scene through ``luxar.encoding``'s decoder directly
    because there is no general "extract a node's data from a compiled scene"
    API yet. Rewrite onto that surface when it lands -- see #2482.
    """
    import shutil

    import zarr

    root = zarr.open(str(scene_path), mode="r")
    has_existing_lod = any(
        _numbered_children(root[layer_name], "child_")
        for layer_name in ("By tracer type", "By redshift")
    )
    if has_existing_lod and substitutive_lod_or_flat(LOD) is None:
        aprint(
            "  ⚠ Keeping the fetched scene's existing substitutive LOD: this "
            "installation cannot rebuild those levels without torch and scipy. "
            "Install the gsplat dependencies, then delete the scene and re-run:\n"
            "      pip install 'luxar[gsplats]'\n"
            f"      rm -rf {scene_path}"
        )
        return scene_path

    tracer_positions, tracer_rgb = _finest_level_cloud(root, "By tracer type")
    redshift_count, redshift_rgb = _finest_level_colors(root, "By redshift")

    # The author shares ONE positions array between the layers (deduplicated by
    # the encoder's array_ref). The stored point counts let us verify that
    # contract without decoding another ~117 MB copy of those shared positions.
    if tracer_positions.shape[0] != redshift_count:
        raise ValueError(
            "the two layers carry different point counts "
            f"({tracer_positions.shape[0]:,} vs {redshift_count:,}); "
            "this scene is not the two-layer shape restructure_scene expects"
        )
    if tracer_rgb.shape[0] != tracer_positions.shape[0]:
        raise ValueError(
            f"'By tracer type' carries {tracer_positions.shape[0]:,} points but "
            f"{tracer_rgb.shape[0]:,} colors"
        )
    if redshift_rgb.shape[0] != redshift_count:
        raise ValueError(
            f"'By redshift' carries {redshift_count:,} points but "
            f"{redshift_rgb.shape[0]:,} colors"
        )

    aprint(
        f"  ♻ Re-laddering {tracer_positions.shape[0]:,} points under the current "
        f"recipe ({SCENE_MAX_POINTS_PER_NODE:,}/node ceiling)"
    )
    stem = scene_path.name.removesuffix(".luxar.zarr")
    staging = scene_path.with_name(f"{stem}.rebuild.luxar.zarr")
    backup = scene_path.with_name(f"{stem}.previous.luxar.zarr")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    try:
        create_scene(
            tracer_positions,
            None,
            None,
            staging,
            tracer_rgb=tracer_rgb,
            redshift_rgb=redshift_rgb,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        scene_path.rename(backup)
        staging.rename(scene_path)
    except BaseException:
        if backup.exists() and not scene_path.exists():
            backup.rename(scene_path)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    return scene_path


def create_scene(
    positions: np.ndarray,
    redshift: Optional[np.ndarray],
    tracer_ids: Optional[np.ndarray],
    output_path: Path,
    *,
    tracer_rgb: Optional[np.ndarray] = None,
    redshift_rgb: Optional[np.ndarray] = None,
) -> Path:
    """Build the DESI cosmic-web scene with tracer + redshift colorings.

    ``redshift`` and ``tracer_ids`` exist only to COLOUR the two layers. When a
    caller already holds the per-point RGB — :func:`restructure_scene` recovers
    it losslessly from a fetched scene — it passes ``tracer_rgb``/``redshift_rgb``
    instead and the scalars are unused. That avoids reconstructing scalars from
    colours, which is exact for the 4-entry tracer LUT but only bin-accurate for
    the redshift ramp; passing the colours through keeps the rebuild lossless.
    """
    precoloured = tracer_rgb is not None or redshift_rgb is not None
    if precoloured:
        if tracer_rgb is None or redshift_rgb is None:
            raise ValueError("pass both tracer_rgb and redshift_rgb, or neither")
        if SCENE_MAX_POINTS is not None:
            # Sampling selects rows from the scalars; with colours supplied
            # there are no scalars to select in lockstep, and sampling colours
            # separately would decouple them from the positions. Refuse loudly
            # rather than emit a scene whose colours belong to other points.
            raise ValueError(
                "SCENE_MAX_POINTS is set, which samples rows; that path needs "
                "the redshift/tracer scalars, not precomputed colours"
            )
    elif redshift is None or tracer_ids is None:
        raise ValueError(
            "redshift and tracer_ids are required when precomputed colours are not supplied"
        )
    with asection("Creating Luxar Scene"):
        dims = Dimensions(
            [
                Dimension("x", unit="Mpc", display=True),
                Dimension("y", unit="Mpc", display=True),
                Dimension("z", unit="Mpc", display=True),
            ]
        )

        # Orbit about the OBSERVER, i.e. the origin — the Milky Way and our solar
        # system, and the one point every DESI sightline radiates from. Framing a
        # 2-98 percentile bounding box instead put the pivot ~1.2 Gpc away down
        # +z (the caps are asymmetric in z), so orbiting swung the entire local
        # universe around a point out in the ELG shell.
        #
        # Distance comes from the RADIAL extent, not a box diagonal, because the
        # cloud surrounds the pivot rather than sitting in front of it: the p95
        # shell exactly fills the frame at r95/tan(fov/2). Opening at 0.75x that
        # keeps the deliberately close, immersive start — the populated bulk
        # slightly overfills the view and the sparse high-z tail runs off the
        # edges, which is the intended "inside the cosmic web" framing.
        radial = np.linalg.norm(positions.astype(np.float64), axis=1)
        r95 = float(np.percentile(radial, 95))
        r_max = float(radial.max())
        fov_deg = CINEMATIC_FOV_DEG
        cam_dist = 0.75 * r95 / np.tan(np.radians(fov_deg) / 2.0)
        camera = CameraConfig(
            position=(0.0, 0.0, cam_dist),
            target=SCENE_CAMERA_TARGET,
            up=(0.0, 1.0, 0.0),
            near=float(max(1.0, cam_dist * 0.005)),
            # Far must clear the whole cloud from the camera, which sits outside
            # it: worst case is the antipodal galaxy at cam_dist + r_max.
            far=float((cam_dist + r_max) * 1.5),
        )
        aprint(
            f"  🎥 Orbiting the observer at the origin; camera at "
            f"{cam_dist:,.0f} Mpc (r95={r95:,.0f}, r_max={r_max:,.0f})"
        )

        if SCENE_MAX_POINTS is None:
            scene_positions, scene_redshift, scene_tracer_ids = (
                positions,
                redshift,
                tracer_ids,
            )
            aprint(f"  🌌 Whole catalog: {len(positions):,} objects, no row cap")
        else:
            scene_positions, scene_redshift, scene_tracer_ids = sample_scene_catalog(
                positions,
                redshift,
                tracer_ids,
                max_points=SCENE_MAX_POINTS,
            )
            if len(scene_positions) < len(positions):
                aprint(
                    f"  📉 Sampling {len(scene_positions):,} of {len(positions):,} "
                    "catalog rows to bound the finest LOD payload"
                )

        # One ladder spec for both layers and every level, sized from the finest
        # payload; `_validate_counts` clamps it per level.
        stream_lod = dict(
            counts=streaming_breakpoints(len(scene_positions)),
            method="random",
            seed=0,
        )
        aprint(
            f"  🪜 Additive ladder: {len(stream_lod['counts'])} rungs, "
            f"largest commit <= {SCENE_MAX_COMMIT:,}"
        )
        # Additive flux is linear in element count, and every coarse level
        # preserves the sample's mass, so compensate every rung at the wrapper.
        scene_intensity = SCENE_INTENSITY * len(positions) / len(scene_positions)
        colors = tracer_rgb if precoloured else tracer_colors(scene_tracer_ids)

        # Resolved ONCE for both layers so a torch/scipy-free machine prints one
        # notice, not one per layer.
        lod = substitutive_lod_or_flat(LOD)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                citation=DEMO_META["citation"],
                dimensions=dims,
                viewer_config=ViewerConfig(cinematic_mode=True, camera=camera),
            )
            stamp_input_digests(scene)
            scene.attrs["title"] = "DESI DR1 — The Cosmic Web"

            # Layer 1: colored by tracer type (categorical populations).
            scene.add_points(
                "By tracer type",
                scene_positions,
                colors=colors,
                radii=POINT_RADIUS,
                opacity=0.9,
                blending_mode="additive",
                intensity=scene_intensity,
                layer=True,
                partition=dict(max_elements=SCENE_MAX_POINTS_PER_NODE),
                substitutive_lod=lod,
                additive_lod=stream_lod,
            )

            # Layer 2: colored by redshift (continuous depth), turbo baked into
            # per-point RGB so the gradient stays consistent across every LOD
            # level (see redshift_colors). Shares the same positions array →
            # deduplicated by the encoder's array_ref.
            scene.add_points(
                "By redshift",
                scene_positions,
                colors=(
                    redshift_rgb if precoloured else redshift_colors(scene_redshift)
                ),
                radii=POINT_RADIUS,
                opacity=0.9,
                blending_mode="additive",
                intensity=scene_intensity,
                layer=True,
                visible=False,
                partition=dict(max_elements=SCENE_MAX_POINTS_PER_NODE),
                substitutive_lod=lod,
                additive_lod=stream_lod,
            )

            scene.add_text(
                "DESI DR1 — The Cosmic Web",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.7)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                "~9.75M galaxies & quasars • redshift → comoving Mpc",
                DEMO_META.get("citation"),
            )
        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def _load_or_build_or_exit() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the cloud, converting known download failures to a clean CLI exit."""
    try:
        return load_or_build()
    except DESICatalogDownloadError as exc:
        aprint(f"❌ {exc}")
        raise SystemExit(1) from None


def _discard_incomplete_scene(scene_path: Path) -> bool:
    """Remove a compiler-marked partial store so the normal fetch path retries."""
    import shutil

    try:
        root = open_group(scene_path, mode="r")
    except Exception:
        return False
    if not root.attrs.get("incomplete"):
        return False
    aprint(f"  ⚠ Discarding incomplete cached scene: {scene_path}")
    shutil.rmtree(scene_path)
    return True


def _discard_rebuild_artifacts(scene_path: Path) -> None:
    """Remove stale staging/backup siblings left by an interrupted rebuild."""
    import shutil

    stem = scene_path.name.removesuffix(".luxar.zarr")
    for suffix in ("rebuild", "previous"):
        sibling = scene_path.with_name(f"{stem}.{suffix}.luxar.zarr")
        shutil.rmtree(sibling, ignore_errors=True)


def _restructure_fetched_scene_if_needed(scene_path: Path) -> None:
    if not scene_exceeds_node_capacity(scene_path):
        return
    try:
        restructure_scene(scene_path)
    except Exception as exc:
        aprint(
            f"  ⚠ Could not re-ladder the fetched scene ({exc}); "
            "keeping the original scene and continuing with diagnostics."
        )


def main() -> None:
    aprint("=" * 70)
    aprint("DESI DR1 Demo: The Cosmic Web in 3D")
    aprint("=" * 70)

    output_path = get_demos_output_dir() / "desi_galaxies.luxar.zarr"
    _discard_incomplete_scene(output_path)
    _discard_rebuild_artifacts(output_path)

    if SERVE_ONLY:
        if output_path.exists():
            ensure_origin_framing(output_path)
            warn_if_scene_is_stale(output_path)
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    if RECOMPUTE:
        # Full pipeline from source: download → convert → build LOD scene.
        positions, redshift, tracer_ids = _load_or_build_or_exit()
        aprint(f"Points: {len(positions):,}")
        create_scene(positions, redshift, tracer_ids, output_path)
    elif not output_path.exists():
        # Fast path: resolve and unzip the fully-built scene.
        try:
            scene_zip = ensure_dataset(DEMO_NAME)[0]
        except DatasetUnavailable as exc:
            scene_zip = None
            unavailable_reason = str(exc)
        if scene_zip is not None:
            extract_shipped_scene(scene_zip, output_path)
            stamp_input_digests(output_path)
            # The record's scene is a pre-built artifact, so its STRUCTURE is
            # whatever the build that produced it chose. Re-derive it under the
            # current recipe when it breaches the per-node ceiling, instead of
            # warning and rendering a node that can lose its tail. The other
            # stale-scene warnings also cover arbitrary user-built caches whose
            # provenance/shape is not established, so they remain diagnostic.
            _restructure_fetched_scene_if_needed(output_path)
            ensure_origin_framing(output_path)
            warn_if_scene_is_stale(output_path)
        else:
            aprint(
                f"Precomputed scene not available from the manifest: {unavailable_reason}\n"
                "Falling back to download + build (one-time; result is cached)."
            )
            positions, redshift, tracer_ids = _load_or_build_or_exit()
            aprint(f"Points: {len(positions):,}")
            create_scene(positions, redshift, tracer_ids, output_path)
    else:
        aprint(f"Using cached scene: {output_path}")
        ensure_origin_framing(output_path)
        warn_if_scene_is_stale(output_path)

    if NO_SERVE:
        aprint(f"Dataset generated at {output_path}")
    else:
        aprint("Data credit: DESI DR1 (DESI Collaboration 2026, arXiv:2503.14745)")
        launch_viewer(output_path)


if __name__ == "__main__":
    main()
