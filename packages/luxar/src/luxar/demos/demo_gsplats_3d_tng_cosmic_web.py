#!/usr/bin/env python3
"""GSplats Demo: IllustrisTNG Cosmic Web — Dark-Matter Density → Gaussian Splats

Gaussian-splats the whole 300 Mpc box of the **IllustrisTNG** TNG300-3-Dark
cosmological simulation: 244 million dark-matter particles collapsed onto a
1024³ density grid, then fit as oriented Gaussians. The result is the *cosmic
web* — the filaments, sheets, nodes, and voids that the dark matter of the
Universe condenses into under gravity, the same structure made famous by the
TNG300 publicity renders.

================================================================================
N-BODY PARTICLES → DENSITY FIELD → GAUSSIAN SPLATS (COSMOLOGY)
================================================================================

An N-body snapshot is a cloud of point masses, not a volume. We deposit the DM
particles onto a regular grid with **Cloud-In-Cell** (CIC) weighting — each
particle spreads trilinearly over its 8 surrounding voxels — then apply a very
light Gaussian smoothing (σ = 0.25 voxels) to knock down the coarsest particle
shot noise while keeping the fine filamentary detail. Log-normalizing the huge dynamic
range (voids vs. cluster cores span many decades) gives a fit-ready [0, 1] cube.
Fitting oriented Gaussians to it is the same ``fit → convert`` pipeline used for
confocal / light-sheet microscopy — here applied to the largest structures in
the Universe.

DATA SOURCE & CITATION
----------------------
IllustrisTNG public data release — TNG300-3-Dark, snapshot 99 (z = 0).
    205000 ckpc/h box (≈ 302.6 Mpc physical), 625³ = 244,140,625 DM particles.
    https://www.tng-project.org/data/
Nelson et al. (2019), "The IllustrisTNG simulations: public data release",
    Computational Astrophysics and Cosmology 6, 2. TNG data is free for public
    use with registration (an API key); see https://www.tng-project.org/users/register/

SELF-CONTAINED / CACHING
------------------------
On a fresh machine this demo bootstraps itself:
  1. Fast path: a small precomputed fit shipped via Git LFS
     (``demos/data/gsplats_tng_cosmic_web/``).
  2. If that asset isn't pulled, and ``--recompute`` is given, it downloads the
     ~15 GB TNG300-3-Dark snapshot (4 HDF5 chunks) to
     ``~/.cache/luxar/gsplats_tng_cosmic_web/``, builds the 1024³ CIC density
     grid, fits Gaussian splats on the GPU, and caches the fit.
``--recompute`` requires a TNG API key in the ``TNG_API_KEY`` environment
variable (free registration) and ~40 GB of scratch disk + a CUDA GPU.

USAGE
-----
    python demo_gsplats_3d_tng_cosmic_web.py [--recompute] [--no-serve] [--serve-only]
    TNG_API_KEY=... python demo_gsplats_3d_tng_cosmic_web.py --recompute

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan,  'C': fly controls
"""

DEMO_META = {
    "key": "gsplats_3d_tng_cosmic_web",
    "title": "IllustrisTNG Cosmic Web",
    "description": "IllustrisTNG TNG300 dark-matter density (244M particles) as Gaussian splats — the cosmic web.",
    "category": "astronomy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 50,  # approx
        "compute": "medium",
        "gpu": "optional",
        "local_data": "git-lfs",
    },
    "caches": ["gsplats_tng_cosmic_web"],
    "outputs": ["gsplats_3d_tng_cosmic_web"],
}

import os
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.demos import (
    detect_device,
    launch_viewer,
    load_precomputed_gsplats,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

TNG_API_BASE = "https://www.tng-project.org/api"
SIMULATION = "TNG300-3-Dark"
SNAPSHOT = 99  # z = 0

DEMO_NAME = "gsplats_tng_cosmic_web"
GSPLATS_FILE = "tng_cosmic_web.gsplats.zarr.zip"

CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME
CACHE_FILE = CACHE_DIR / GSPLATS_FILE

# Density-grid parameters (see module docstring). The raw CIC deposit is cached
# separately so σ can be re-tuned cheaply without re-reading 244M particles.
GRID = 1024
SIGMA = 0.25  # very soft (voxels): kill coarsest CIC grain, keep fine detail

# Fitting parameters (GPU). The whole box is fit as a single volume (no tiling
# — the cosmic web is a fairly uniform, space-filling texture). With only a
# very soft σ = 0.25 smoothing the field keeps its fine structure, so we spend
# a large budget (2M splats) to resolve the filaments and cluster nodes; a
# light post-fit cull then drops only the truly negligible tail.
MAX_SPLATS = 2_000_000
MAX_SPLATS_PER_PASS = 500_000
ITERS_PER_PASS = 4_000
PSNR_PATIENCE = 0.1
CULL_RETENTION = 0.99  # keep 99% of amplitude; drop only the negligible tail

# Additive display brightness. 2M splats over a 300 Mpc box accumulate strongly
# along each ray, so the exposure is dialed well down to avoid blow-out.
SCENE_INTENSITY = 0.15  # vivid look: gold cluster nodes, warm filaments, black voids

# Physical extent of the box (TNG300-3-Dark): 205000 ckpc/h at h = 0.6774, z = 0.
BOX_MPC = 205000.0 / 0.6774 / 1000.0  # ≈ 302.6 Mpc

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

Arbol.max_depth = 5
DEVICE = None


# =============================================================================
# Density-field construction (pure helpers are unit-tested; HDF5 IO is not)
# =============================================================================


def deposit_cic(coords: np.ndarray, box: float, grid: int) -> np.ndarray:
    """Deposit particles onto a periodic ``grid``³ density field with CIC weights.

    Each particle at fractional grid position spreads trilinearly over its 8
    surrounding voxels (Cloud-In-Cell), with periodic wrap-around at the box
    boundary. ``coords`` are particle positions in the same units as ``box``.
    Returns a float32 (grid, grid, grid) mass field.
    """
    c = np.asarray(coords, dtype=np.float64) / box * grid
    i0 = np.floor(c).astype(np.int64)
    fr = c - i0
    flat = np.zeros(grid**3, dtype=np.float64)
    for dx in (0, 1):
        wx = fr[:, 0] if dx else 1.0 - fr[:, 0]
        ix = (i0[:, 0] + dx) % grid
        for dy in (0, 1):
            wy = fr[:, 1] if dy else 1.0 - fr[:, 1]
            iy = (i0[:, 1] + dy) % grid
            for dz in (0, 1):
                wz = fr[:, 2] if dz else 1.0 - fr[:, 2]
                iz = (i0[:, 2] + dz) % grid
                idx = (ix * grid + iy) * grid + iz
                flat += np.bincount(idx, weights=wx * wy * wz, minlength=grid**3)
    return flat.reshape(grid, grid, grid).astype(np.float32)


def finalize_density(raw_grid: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth + log-normalize a raw CIC mass field into a fit-ready [0, 1] cube.

    A light Gaussian (``sigma`` voxels) turns discrete shot noise into a smooth
    field; ``log1p`` compresses the many-decade void↔cluster dynamic range; the
    result is scaled so its max is 1.
    """
    from scipy.ndimage import gaussian_filter

    sm = gaussian_filter(raw_grid, sigma=sigma) if sigma > 0 else raw_grid
    v = np.log1p(sm).astype(np.float32)
    peak = float(v.max())
    if peak > 0:
        v /= peak
    return v


def _download_snapshot() -> list[Path]:
    """Download the TNG300-3-Dark snapshot HDF5 chunks (resumable). Returns paths."""
    import h5py

    from luxar.utils.download import robust_download

    key = os.environ.get("TNG_API_KEY")
    if not key:
        raise RuntimeError(
            "TNG_API_KEY environment variable is not set. Register for a free "
            "IllustrisTNG API key at https://www.tng-project.org/users/register/ "
            "and export it as TNG_API_KEY to use --recompute."
        )
    hdr = {"api-key": key}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base = f"{TNG_API_BASE}/{SIMULATION}/snapshots/{SNAPSHOT}/files/snapshot-{SNAPSHOT}"

    with asection(f"Downloading {SIMULATION} snapshot {SNAPSHOT} (~15 GB)"):
        # Chunk 0 first, then read NumFilesPerSnapshot to learn the chunk count.
        first = CACHE_DIR / f"snap_{SNAPSHOT:03d}.0.hdf5"
        robust_download(
            f"{base}.0.hdf5", first, max_retries=5, timeout=1800, extra_headers=hdr
        )
        with h5py.File(first, "r") as f:
            n_chunks = int(f["Header"].attrs["NumFilesPerSnapshot"])
        aprint(f"Snapshot has {n_chunks} chunk(s)")
        files = [first]
        for i in range(1, n_chunks):
            fp = CACHE_DIR / f"snap_{SNAPSHOT:03d}.{i}.hdf5"
            robust_download(
                f"{base}.{i}.hdf5", fp, max_retries=5, timeout=1800, extra_headers=hdr
            )
            files.append(fp)
    return files


def build_density_grid() -> np.ndarray:
    """Download the snapshot, deposit DM with CIC, and return the σ-smoothed cube.

    The raw (unsmoothed) CIC field is cached so re-tuning ``SIGMA`` never re-reads
    the 244M particles.
    """
    import h5py

    raw_cache = CACHE_DIR / f"cic_raw_{GRID}.npy"
    if raw_cache.exists():
        with asection("Loading cached raw CIC field"):
            raw = np.load(raw_cache)
    else:
        files = _download_snapshot()
        with h5py.File(files[0], "r") as f:
            box = float(f["Header"].attrs["BoxSize"])
        with asection(f"CIC deposit → {GRID}³ ({len(files)} chunks)"):
            raw = np.zeros((GRID, GRID, GRID), dtype=np.float32)
            for fp in files:
                with h5py.File(fp, "r") as f:
                    coords = np.asarray(f["PartType1/Coordinates"][:])
                raw += deposit_cic(coords, box, GRID)
                aprint(f"  deposited {fp.name} ({len(coords):,} particles)")
            np.save(raw_cache, raw)
    with asection(f"Smooth (σ={SIGMA}) + log-normalize"):
        v = finalize_density(raw, SIGMA)
        aprint(
            f"✓ Fit-ready cube: {v.shape}, p50={np.percentile(v, 50):.3f}, "
            f"p99={np.percentile(v, 99):.3f}"
        )
    return v


# =============================================================================
# Fit
# =============================================================================


def fit_volume(volume: np.ndarray) -> GSplatData:
    """Fit Gaussian splats to the density cube and cache the result."""
    global DEVICE
    if DEVICE is None:
        DEVICE = detect_device()

    from luxar.gsplats import fit_progressive_gaussian_splats

    with asection(f"Fitting GSplats ({volume.shape}, max {MAX_SPLATS:,}, {DEVICE})"):
        result = fit_progressive_gaussian_splats(
            volume,
            max_splats=MAX_SPLATS,
            max_splats_per_pass=MAX_SPLATS_PER_PASS,
            iters_per_pass=ITERS_PER_PASS,
            psnr_patience=PSNR_PATIENCE,
            cull_retention=CULL_RETENTION,
            device=DEVICE,
            verbose=True,
        )
        aprint(f"Fitted {len(result.amplitudes):,} splats")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.save(
            CACHE_FILE,
            encoding_mode=EncodingMode.AUTO,
            include_fitting_info=False,
            compress="zip",
            zip_deflate=True,
        )
        return result


def load_or_build_gsplats() -> GSplatData:
    """Return fitted cosmic-web splats, self-contained on a fresh system."""
    if not RECOMPUTE:
        try:
            precomputed = load_precomputed_gsplats(DEMO_NAME, [GSPLATS_FILE])
            if precomputed is not None:
                return precomputed[0]
        except FileNotFoundError:
            aprint(
                "Precomputed fit not available (Git LFS asset not pulled). "
                "Re-run with --recompute (needs TNG_API_KEY + a CUDA GPU) to "
                "download the snapshot and refit."
            )
            raise

    warn_if_no_cuda_gpu()
    volume = build_density_grid()
    return fit_volume(volume)


# =============================================================================
# Scene
# =============================================================================


def create_luxar_scene(gsplats_data: GSplatData, output_path: Path) -> Path:
    """Build the IllustrisTNG cosmic-web scene."""
    with asection("Creating Luxar Scene"):
        # Fit centers live in voxel space; scale them (and their covariance)
        # to physical Mpc so the viewer axes read in real units, then center.
        mpc_per_voxel = BOX_MPC / GRID
        gsplats_data = (
            gsplats_data.transform(np.eye(3, dtype=np.float64) * mpc_per_voxel)
            .center_at_centroid()
            .scale_intensity(SCENE_INTENSITY)
        )
        dims = Dimensions(
            [
                Dimension("x", unit="Mpc", display=True),
                Dimension("y", unit="Mpc", display=True),
                Dimension("z", unit="Mpc", display=True),
            ]
        )
        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(tone_mapping="Neutral"),
            )
            scene.attrs["title"] = "GSplats: IllustrisTNG Cosmic Web (TNG300-3-Dark)"
            scene.add_gsplats_from_data(
                name="cosmic_web",
                result=gsplats_data,
                colormap="magma",
                opacity=1.0,
                blending_mode="additive",
                intensity=1.0,
                layer=True,
            )
            scene.add_text(
                "IllustrisTNG Cosmic Web",
                position=(0.02, 0.02),
                font_size=0.045,
                anchor="top-left",
                color="rgba(255,255,255,0.7)",
                blend_mode="difference",
            )
            scene.add_text(
                "TNG300-3-Dark z=0 • 244M dark-matter particles → Gaussian splats",
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
    aprint("GSplats Demo: IllustrisTNG Cosmic Web (TNG300-3-Dark)")
    aprint("=" * 70)
    aprint("244M dark-matter particles → density field → Gaussian splatting")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_3d_tng_cosmic_web.luxar.zarr"

    if SERVE_ONLY:
        if output_path.exists():
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    gsplats_data = load_or_build_gsplats()
    aprint(f"Splats: {len(gsplats_data.amplitudes):,}")
    scene_path = create_luxar_scene(gsplats_data, output_path)

    if NO_SERVE:
        aprint(f"Dataset generated at {scene_path}")
    else:
        aprint("Data credit: IllustrisTNG (Nelson et al. 2019, CompAC 6, 2)")
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
