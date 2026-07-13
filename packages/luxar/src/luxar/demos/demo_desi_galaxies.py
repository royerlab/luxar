#!/usr/bin/env python3
"""DESI DR1 — The Cosmic Web in 3D (~9.75M galaxies & quasars)

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
DESI Collaboration (2025), "Data Release 1 of the Dark Energy Spectroscopic
    Instrument", arXiv:2503.14745. Data released under CC BY 4.0.
    Acknowledgment: this product uses data obtained with the Dark Energy
    Spectroscopic Instrument (DESI). Derived 3D positions are a transform of the
    public RA/Dec/redshift columns.

SELF-CONTAINED / CACHING
------------------------
On a fresh machine this demo bootstraps itself with no manual steps:
  1. Fast path: a fully-built scene (both LOD colorings, ~80 MB) shipped via
     Git LFS (``demos/data/desi_galaxies/``); it is unzipped once into the demos
     output dir and loads instantly — no per-launch LOD build.
  2. If that asset isn't pulled, ``--recompute`` (or a missing asset)
     AUTOMATICALLY downloads the ~1 GB of DR1 LSS catalogs to
     ``~/.cache/luxar/desi_galaxies/`` (resumable), reads them with ``astropy``,
     converts (RA, Dec, z) → comoving Mpc, and builds the scene (the substitutive
     LOD over ~10M points is GPU-accelerated but slow on CPU-only machines —
     which is exactly why the built scene ships precomputed).

USAGE
-----
    python demo_desi_galaxies.py [--recompute] [--no-serve] [--serve-only]

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan,  'C': fly controls
"""

from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import (
    CameraConfig,
    Dimension,
    Dimensions,
    LuxarZarrCompiler,
    ViewerConfig,
)
from luxar.utils.demos import is_lfs_pointer, launch_viewer, parse_demo_flags
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

# DESI fiducial cosmology (Planck-2018 base ΛCDM): H0=67.36, Om0≈0.3137.
COSMO_H0 = 67.36
COSMO_OM0 = 0.3137

DEMO_NAME = "desi_galaxies"
DERIVED_FILE = "desi_dr1_cosmic_web.npz"
SCENE_ZIP_FILE = "desi_dr1_cosmic_web.luxar.zarr.zip"

CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME
DERIVED_CACHE = CACHE_DIR / DERIVED_FILE

DATA_DIR = Path(__file__).parent / "data" / DEMO_NAME
SCENE_ZIP_SHIPPED = DATA_DIR / SCENE_ZIP_FILE

# Redshift window: keep good extragalactic redshifts; drop the tiny z≈0 blunders
# and the sparse very-high-z tail that just stretches the scene.
Z_MIN = 0.001
Z_MAX = 4.0

# Display / LOD parameters.
POINT_RADIUS = 1.2  # Mpc (visualization scale)
SCENE_INTENSITY = 0.05
LOD = dict(compression_factor=8, levels=3, device="auto")

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

Arbol.max_depth = 5


# =============================================================================
# Pure helpers (unit-tested)
# =============================================================================


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
# Download + build (IO; not unit-tested)
# =============================================================================


def download_catalogs() -> list[tuple[str, Path]]:
    """Download the 8 DR1 LSS clustering FITS files (resumable). Returns
    (tracer_name, local_path) for each NGC/SGC file."""
    from luxar.utils.download import robust_download

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: list[tuple[str, Path]] = []
    with asection("Downloading DESI DR1 LSS catalogs (~1 GB)"):
        for name in TRACER_ORDER:
            for fname in TRACERS[name]["files"]:
                dest = CACHE_DIR / fname
                aprint(f"{name}: {fname}")
                robust_download(
                    f"{BASE_URL}/{fname}", dest, max_retries=5, timeout=1800
                )
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

    with asection("Unpacking precomputed scene (Git LFS)"):
        aprint(f"Source: {zip_path.name} ({zip_path.stat().st_size / 1e6:.0f} MB)")
        staging = output_path.parent / (output_path.name + ".part")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(staging)
        staging.rename(output_path)
        aprint(f"Scene ready: {output_path}")


# =============================================================================
# Scene
# =============================================================================


def create_scene(
    positions: np.ndarray,
    redshift: np.ndarray,
    tracer_ids: np.ndarray,
    output_path: Path,
) -> Path:
    """Build the DESI cosmic-web scene with tracer + redshift colorings."""
    with asection("Creating Luxar Scene"):
        dims = Dimensions(
            [
                Dimension("x", unit="Mpc", display=True),
                Dimension("y", unit="Mpc", display=True),
                Dimension("z", unit="Mpc", display=True),
            ]
        )

        lo, hi = np.percentile(positions, [2, 98], axis=0)
        center = (lo + hi) / 2.0
        extent = float(np.max(hi - lo))
        fov_deg = 50.0
        fit_dist = (extent * 0.5) / np.tan(np.radians(fov_deg) / 2.0)
        cam_dist = fit_dist * 0.7
        camera = CameraConfig(
            position=(float(center[0]), float(center[1]), float(center[2] + cam_dist)),
            target=(float(center[0]), float(center[1]), float(center[2])),
            up=(0.0, 1.0, 0.0),
            fov=fov_deg,
            near=float(max(1.0, cam_dist * 0.005)),
            far=float(cam_dist * 20.0 + extent * 10.0),
        )

        colors = tracer_colors(tracer_ids)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims, viewer_config=ViewerConfig(camera=camera)
            )
            scene.attrs["title"] = "DESI DR1 — The Cosmic Web"

            # Layer 1: colored by tracer type (categorical populations).
            scene.add_points(
                "By tracer type",
                positions,
                colors=colors,
                radii=POINT_RADIUS,
                opacity=0.9,
                blending_mode="additive",
                intensity=SCENE_INTENSITY,
                layer=True,
                substitutive_lod=LOD,
            )

            # Layer 2: colored by redshift (continuous depth). Shares the same
            # positions array → deduplicated by the encoder's array_ref.
            scene.add_points(
                "By redshift",
                positions,
                scalars=redshift,
                colormap="turbo",
                radii=POINT_RADIUS,
                opacity=0.9,
                blending_mode="additive",
                intensity=SCENE_INTENSITY,
                layer=True,
                visible=False,
                substitutive_lod=LOD,
            )

            scene.add_text(
                "DESI DR1 — The Cosmic Web",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.7)",
                blend_mode="difference",
            )
            scene.add_text(
                "~9.75M galaxies & quasars • spectroscopic redshifts → comoving Mpc",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.5)",
            )
        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    aprint("=" * 70)
    aprint("DESI DR1 Demo: The Cosmic Web in 3D")
    aprint("=" * 70)

    output_path = get_demos_output_dir() / "desi_galaxies.luxar.zarr"

    if SERVE_ONLY:
        if output_path.exists():
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    if RECOMPUTE:
        # Full pipeline from source: download → convert → build LOD scene.
        positions, redshift, tracer_ids = load_or_build()
        aprint(f"Points: {len(positions):,}")
        create_scene(positions, redshift, tracer_ids, output_path)
    elif not output_path.exists():
        # Fast path: unzip the shipped, fully-built scene (instant, no LOD build).
        if SCENE_ZIP_SHIPPED.exists() and not is_lfs_pointer(SCENE_ZIP_SHIPPED):
            extract_shipped_scene(SCENE_ZIP_SHIPPED, output_path)
        else:
            aprint(
                "Precomputed scene not available (Git LFS asset not pulled). "
                "Falling back to download + build (one-time; result is cached)."
            )
            positions, redshift, tracer_ids = load_or_build()
            aprint(f"Points: {len(positions):,}")
            create_scene(positions, redshift, tracer_ids, output_path)
    else:
        aprint(f"Using cached scene: {output_path}")

    if NO_SERVE:
        aprint(f"Dataset generated at {output_path}")
    else:
        aprint("Data credit: DESI DR1 (DESI Collaboration 2025, arXiv:2503.14745)")
        launch_viewer(output_path)


if __name__ == "__main__":
    main()
