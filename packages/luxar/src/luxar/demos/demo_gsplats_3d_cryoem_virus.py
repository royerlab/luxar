#!/usr/bin/env python3
"""GSplats Demo: Cryo-EM Giant Virus Capsid (EMDB) — Density Map → Gaussian Splats

Gaussian-splats a real cryo-electron-microscopy density map from the EMDB: the
capsid of Paramecium bursaria chlorella virus 1 (PBCV-1), a giant icosahedral
virus. The reconstructed 3D electron-density volume — a hollow ~1650 Å capsid
shell tiled with capsomers — is exactly what the Luxar splat fitter eats, so
this is the microscopy splat pipeline applied to structural biology.

================================================================================
VOLUMETRIC DENSITY → GAUSSIAN SPLATS (STRUCTURAL BIOLOGY)
================================================================================

An EMDB map is a dense regular voxel grid of electron density (MRC/CCP4 format).
Fitting oriented Gaussians to it gives a smooth, glowing, rotatable surface at
20–50× compression — no isosurface threshold needed — the same
``cal → fit → convert`` pipeline used for confocal/light-sheet microscopy.

DATA SOURCE & CITATION
----------------------
EMDB entry EMD-5384 — "The structure of PBCV-1 (five-fold averaged map)".
    700³ voxels, ~9.8 Å resolution. https://www.ebi.ac.uk/emdb/EMD-5384
Zhang, X. et al. (2011). "Three-dimensional structure and function of the
    Paramecium bursaria chlorella virus capsid." PNAS 108(36), 14837–14842.
EMDB is public domain / CC0. Map:
    https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-5384/map/emd_5384.map.gz

SELF-CONTAINED / CACHING
------------------------
On a fresh machine this demo bootstraps itself with no manual steps:
  1. Fast path: a small precomputed fit shipped via Git LFS
     (``demos/data/gsplats_cryoem_virus/``).
  2. If that asset isn't pulled, it AUTOMATICALLY downloads the 1.3 GB map to
     ``~/.cache/luxar/gsplats_cryoem_virus/``, fits Gaussian splats on the GPU,
     and caches the fit — so subsequent runs are instant.
``--recompute`` forces the download + fit path.

USAGE
-----
    python demo_gsplats_3d_cryoem_virus.py [--recompute] [--no-serve] [--serve-only]
    python demo_gsplats_3d_cryoem_virus.py --recompute --target-size 384 --max-splats 800000

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan,  'C': fly controls
"""

DEMO_META = {
    "key": "gsplats_3d_cryoem_virus",
    "title": "Cryo-EM Giant Virus Capsid (EMDB) — Density Map → Gaussian Splats",
    "description": "Real cryo-EM density map of the PBCV-1 giant virus capsid (EMD-5384) as Gaussian splats.",
    "category": "structural",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 50,  # approx
        "compute": "medium",
        "gpu": "optional",
        "local_data": "git-lfs",
    },
    "caches": ["gsplats_cryoem_virus"],
    "outputs": ["gsplats_3d_cryoem_virus"],
}

import sys
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import require_module
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

EMDB_ID = "EMD-5384"
MAP_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-5384/map/emd_5384.map.gz"
)
EXPECTED_MAP_SIZE = 1_274_143_910  # bytes

DEMO_NAME = "gsplats_cryoem_virus"
GSPLATS_FILE = "cryoem_virus.gsplats.zarr.zip"

CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME
CACHE_MAP = CACHE_DIR / "emd_5384.map.gz"
CACHE_FILE = CACHE_DIR / GSPLATS_FILE

# Fitting parameters (GPU). The 700³ map is downsampled before fitting; the
# capsid shell is smooth (9.8 Å) so a moderate budget captures the capsomers.
TARGET_SIZE = 512  # <=0 keeps native resolution (700³)
MAX_SPLATS = 1_500_000
MAX_SPLATS_PER_PASS = 300_000
ITERS_PER_PASS = 4_000
PSNR_PATIENCE = 0.1

# Display brightness (dialed down for a dense shell — see VH demo).
SCENE_INTENSITY = 0.03

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]


def _int_arg(argv: list[str], flag: str, default: int) -> int:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith(flag + "="):
            return int(a.split("=", 1)[1])
    return default


TARGET_SIZE = _int_arg(sys.argv, "--target-size", TARGET_SIZE)
MAX_SPLATS = _int_arg(sys.argv, "--max-splats", MAX_SPLATS)

Arbol.max_depth = 5
DEVICE = None


# =============================================================================
# Data loading (pure processing helper is unit-tested; download/mrc IO is not)
# =============================================================================


def normalize_map_volume(density: np.ndarray, target_size: int) -> np.ndarray:
    """Turn a raw EMDB density array into a fit-ready [0, 1] cube.

    Cryo-EM density has near-zero (often slightly negative) solvent; the
    structure is positive. Clip at zero, optionally downsample to ``target_size``
    per axis (``target_size`` <= 0 keeps native resolution), and normalize by a
    high percentile so the capsid shell fills the range without a few hot voxels
    crushing it.
    """
    from scipy.ndimage import zoom

    V = np.asarray(density, dtype=np.float32)
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    V = np.maximum(V, 0.0)  # drop solvent / negative density

    if target_size and target_size > 0:
        factors = [target_size / s for s in V.shape]
        if not all(abs(f - 1.0) < 1e-6 for f in factors):
            V = zoom(V, factors, order=1)

    hi = float(np.percentile(V, 99.9))
    V = np.clip(V / (hi + 1e-8), 0.0, 1.0)
    return V.astype(np.float32)


def load_map_volume(target_size: int = TARGET_SIZE) -> np.ndarray:
    """Download (resumable) + read the EMDB map, ready for fitting."""
    mrcfile = require_module("mrcfile")

    from luxar.utils.download import robust_download

    with asection(f"Downloading cryo-EM map ({EMDB_ID})"):
        aprint(f"Source: {MAP_URL}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        robust_download(
            MAP_URL,
            CACHE_MAP,
            max_retries=5,
            timeout=900,
            expected_size=EXPECTED_MAP_SIZE,
        )

    with asection("Reading MRC density map"):
        with mrcfile.open(CACHE_MAP, permissive=True) as mrc:
            density = np.asarray(mrc.data, dtype=np.float32)
        aprint(f"Raw map: shape={density.shape}, dtype={density.dtype}")
        V = normalize_map_volume(density, target_size)
        aprint(f"✓ Fit-ready cube: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")
        return V


# =============================================================================
# Fit
# =============================================================================


def fit_map(volume: np.ndarray) -> GSplatData:
    """Fit Gaussian splats to the density map and cache the result."""
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
            device=DEVICE,
            verbose=True,
        )
        aprint(f"Fitted {len(result.amplitudes):,} splats")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.save(
            CACHE_FILE,
            encoding_mode=EncodingMode.MEMORY,
            include_fitting_info=False,
            compress="zip",
            zip_deflate=True,
        )
        return result


def load_or_build_gsplats() -> GSplatData:
    """Return fitted virus-capsid splats, self-contained on a fresh system."""
    if not RECOMPUTE:
        try:
            precomputed = load_precomputed_gsplats(DEMO_NAME, [GSPLATS_FILE])
            if precomputed is not None:
                return precomputed[0]
        except FileNotFoundError:
            aprint(
                "Precomputed fit not available (Git LFS asset not pulled). "
                "Falling back to download + fit (one-time; result is cached)."
            )

    warn_if_no_cuda_gpu()
    volume = load_map_volume()
    return fit_map(volume)


# =============================================================================
# Scene
# =============================================================================


def create_luxar_scene(gsplats_data: GSplatData, output_path: Path) -> Path:
    """Build the cryo-EM virus-capsid scene."""
    with asection("Creating Luxar Scene"):
        gsplats_data = gsplats_data.center_at_centroid().scale_intensity(
            SCENE_INTENSITY
        )
        dims = Dimensions(
            [
                Dimension("x", unit="Å", display=True),
                Dimension("y", unit="Å", display=True),
                Dimension("z", unit="Å", display=True),
            ]
        )
        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(tone_mapping="ACES"),
            )
            scene.attrs["title"] = (
                "GSplats: Cryo-EM Giant Virus Capsid (PBCV-1, EMD-5384)"
            )
            scene.add_gsplats_from_data(
                name="virus_capsid",
                result=gsplats_data,
                # inferno, not viridis: viridis's first stop is (68, 1, 84)
                # dark purple, so empty and low-density space renders as a
                # visible haze on the black background. inferno starts at
                # (0, 0, 4), so zero density reads as true black.
                colormap="inferno",
                opacity=1.0,
                # Strong absorption (kappa 5) so the near side of the shell
                # occludes the far side — the capsid reads as a hollow
                # icosahedron instead of a translucent ball of density.
                absorption=5.0,
                blending_mode="volumetric",
                intensity=1.0,
                layer=True,
            )
            scene.add_text(
                "Cryo-EM Giant Virus Capsid (PBCV-1)",
                position=(0.02, 0.02),
                font_size=0.045,
                anchor="top-left",
                color="rgba(255,255,255,0.7)",
                blend_mode="difference",
            )
            scene.add_text(
                "EMDB EMD-5384 • electron-density map → Gaussian splats",
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
    aprint("GSplats Demo: Cryo-EM Giant Virus Capsid (PBCV-1, EMDB)")
    aprint("=" * 70)
    aprint("Real electron-density map → Gaussian splatting")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_3d_cryoem_virus.luxar.zarr"

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
        aprint("Data credit: EMDB EMD-5384 (Zhang et al. 2011, PNAS)")
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
