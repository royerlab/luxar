#!/usr/bin/env python3
"""GSplats Demo: 3D Interstellar Dust of the Solar Neighborhood (Leike & Enßlin 2020)

Gaussian-splats a real 3D reconstruction of the Milky Way's interstellar dust
around the Sun. Unlike the microscopy gsplat demos, the "volume" here is a
cube of *space*: a 740 × 740 × 540 pc reconstruction at 1 pc resolution of dust
extinction density inferred from Gaia + 2MASS/PANSTARRS stellar extinctions.
Fitting oriented Gaussians turns the cube into glowing 3D fog you can fly
through — the Local Bubble around the Sun, and the Orion, Taurus, Perseus and
Cepheus molecular clouds.

================================================================================
VOLUMETRIC SCIENCE → GAUSSIAN SPLATS (BEYOND MICROSCOPY)
================================================================================

The dust reconstruction is a dense regular voxel grid of extinction density —
exactly the input the Luxar splat fitter expects. This is the same
``cal → fit → convert`` pipeline used for confocal/light-sheet microscopy,
applied to astrophysics: real volumetric data, 20–50× compression, smooth
oriented-ellipsoid rendering.

DATA SOURCE & CITATION
----------------------
Leike, R. H., Glatzle, M., & Enßlin, T. A. (2020).
    "Resolving nearby dust clouds." Astronomy & Astrophysics, 639, A138.
    DOI: 10.1051/0004-6361/202038169  (arXiv:2004.06732)
Data (Zenodo record 3993082): ``mean_std.h5`` — mean + std of the dust
    extinction density on a 740 × 740 × 540 grid at 1 pc/voxel, Sun-centered.
    https://doi.org/10.5281/zenodo.3993082  (CC BY 4.0)

SELF-CONTAINED / CACHING
------------------------
On a fresh machine this demo bootstraps itself with no manual steps:
  1. Fast path: a precomputed FULL-RESOLUTION fit shipped via Git LFS
     (``demos/data/gsplats_milkyway_dust/``, ~8 MB) — the native 740×740×540
     cube fit to ~675k Gaussian splats (PSNR ~35 dB).
  2. If that asset isn't pulled, it AUTOMATICALLY downloads the 2.4 GB cube to
     ``~/.cache/luxar/gsplats_milkyway_dust/`` (resumable), fits Gaussian splats
     on the GPU, and caches the fit there — so subsequent runs are instant.
``--recompute`` forces the download + fit path.

The ``--recompute`` default reproduces the shipped asset (native resolution,
~1M splats), which needs a large-VRAM GPU (~40 GB+). On a smaller card, pass a
downscale + lighter budget, e.g. ``--target-size 256 --max-splats 200000``.

USAGE
-----
    python demo_gsplats_3d_milky_way_dust.py [--recompute] [--no-serve] [--serve-only]
    python demo_gsplats_3d_milky_way_dust.py --recompute --target-size 256 --max-splats 200000

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan,  'C': fly controls
"""

DEMO_META = {
    "key": "gsplats_3d_milky_way_dust",
    "title": "Milky Way Dust (Gaussian splats)",
    "description": "Real 3D dust reconstruction of the solar neighborhood (Leike & Ensslin 2020) as Gaussian splats.",
    "category": "astronomy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 8,
        "compute": "medium",
        "gpu": "optional",
        "local_data": "git-lfs",
    },
    "caches": ["gsplats_milkyway_dust"],
    "outputs": ["gsplats_3d_milky_way_dust"],
}

import sys
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

ZENODO_URL = "https://zenodo.org/records/3993082/files/mean_std.h5"
EXPECTED_H5_SIZE = 2_365_636_096  # bytes (drives robust_download resume/verify)

DEMO_NAME = "gsplats_milkyway_dust"
GSPLATS_FILE = "milkyway_dust.gsplats.zarr.zip"

CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME
CACHE_H5 = CACHE_DIR / "mean_std.h5"
CACHE_FILE = CACHE_DIR / GSPLATS_FILE

VOXEL_SIZE_PC = 1.0  # native resolution of the reconstruction

# Fitting parameters (GPU). Defaults reproduce the shipped full-resolution asset:
# the native 740×740×540 cube (TARGET_SIZE=0 ⇒ no downscale) fit to ~1M splats.
# This needs a large-VRAM GPU (~40 GB+); on a smaller card pass e.g.
# `--target-size 256 --max-splats 200000` for a lighter, downscaled refit.
TARGET_SIZE = 0  # 0 (or negative) ⇒ fit at native resolution (no downscale)
MAX_SPLATS = 1_000_000
MAX_SPLATS_PER_PASS = 200_000
ITERS_PER_PASS = 5_000
PSNR_PATIENCE = 0.1

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]


def _int_arg(argv: list[str], flag: str, default: int) -> int:
    """Parse an int CLI flag in either ``--flag value`` or ``--flag=value`` form."""
    for i, _arg in enumerate(argv):
        if _arg == flag and i + 1 < len(argv):
            return int(argv[i + 1])
        if _arg.startswith(flag + "="):
            return int(_arg.split("=", 1)[1])
    return default


TARGET_SIZE = _int_arg(sys.argv, "--target-size", TARGET_SIZE)
MAX_SPLATS = _int_arg(sys.argv, "--max-splats", MAX_SPLATS)

Arbol.max_depth = 5
DEVICE = None


# =============================================================================
# Data Loading  (pure processing helpers are unit-tested; I/O is not)
# =============================================================================


def normalize_dust_volume(mean: np.ndarray, target_size: int) -> np.ndarray:
    """Turn the raw reconstruction array into a fit-ready [0, 1] cube.

    Handles both linear-density and log-density storage (negatives ⇒ log ⇒
    ``exp``), optionally downsamples to ``target_size`` per axis (``target_size``
    <= 0 keeps native resolution), and robustly normalizes with a high-percentile
    clip so a few dense cloud cores don't crush the diffuse structure.
    """
    from scipy.ndimage import zoom

    V = np.asarray(mean, dtype=np.float32)
    # If the stored quantity is log-density (has negatives), exponentiate.
    if np.nanmin(V) < 0.0:
        V = np.exp(V)
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

    if target_size and target_size > 0:
        factors = [target_size / s for s in V.shape]
        if not all(abs(f - 1.0) < 1e-6 for f in factors):
            V = zoom(V, factors, order=1)

    lo = float(np.percentile(V, 1.0))
    hi = float(np.percentile(V, 99.5))
    V = np.clip((V - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return V.astype(np.float32)


def load_dust_volume(target_size: int = TARGET_SIZE) -> np.ndarray:
    """Download (resumable) + load the dust cube, ready for fitting."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "h5py is required for this demo. Install with: pip install h5py "
            "(or `pip install luxar[demos]`)."
        ) from exc

    from luxar.utils.download import robust_download

    with asection("Downloading 3D dust reconstruction (Leike & Enßlin 2020)"):
        aprint("Source: https://doi.org/10.5281/zenodo.3993082  (mean_std.h5, 2.4 GB)")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        robust_download(
            ZENODO_URL,
            CACHE_H5,
            max_retries=5,
            timeout=900,
            expected_size=EXPECTED_H5_SIZE,
        )

    with asection("Loading dust cube"):
        with h5py.File(CACHE_H5, "r") as f:
            key = _find_mean_dataset(f)
            aprint(f"Reading dataset '{key}' {f[key].shape}")
            mean = f[key][:]
        aprint(f"Raw cube: shape={mean.shape}, dtype={mean.dtype}")
        V = normalize_dust_volume(mean, target_size)
        aprint(f"✓ Fit-ready cube: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")
        return V


def _find_mean_dataset(h5file) -> str:
    """Locate the 'mean' 3D dataset (top-level or nested), else the largest 3D."""
    import h5py

    if "mean" in h5file and isinstance(h5file["mean"], h5py.Dataset):
        return "mean"
    best_key = None
    best_size = -1
    stack = [(k, h5file[k]) for k in h5file.keys()]
    while stack:
        name, obj = stack.pop()
        if isinstance(obj, h5py.Group):
            stack.extend((f"{name}/{k}", obj[k]) for k in obj.keys())
        elif isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            if name.split("/")[-1] == "mean":
                return name
            if obj.size > best_size:
                best_size, best_key = obj.size, name
    if best_key is None:
        raise ValueError("No 3D dataset found in the dust HDF5 file.")
    return best_key


# =============================================================================
# GSplats Fitting
# =============================================================================


def fit_dust(volume: np.ndarray) -> GSplatData:
    """Fit Gaussian splats to the dust cube and cache the result."""
    global DEVICE
    if DEVICE is None:
        DEVICE = detect_device()

    from luxar.gsplats import fit_progressive_gaussian_splats

    with asection(
        f"Fitting GSplats (max {MAX_SPLATS:,} splats, {ITERS_PER_PASS} iters/pass)"
    ):
        aprint(f"Volume: {volume.shape}   Device: {DEVICE}")
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
        aprint(f"Caching fit to {CACHE_FILE.name}")
        result.save(
            CACHE_FILE,
            encoding_mode=EncodingMode.MEMORY,
            include_fitting_info=True,
            compress="zip",
            zip_deflate=True,
        )
        return result


def load_or_build_gsplats() -> GSplatData:
    """Return fitted dust splats, self-contained on a fresh system.

    Fast path: precomputed fit (Git LFS / local cache). If that asset is not
    available, automatically download the cube and fit — no manual ``git lfs
    pull`` required.
    """
    if not RECOMPUTE:
        try:
            precomputed = load_precomputed_gsplats(DEMO_NAME, [GSPLATS_FILE])
            if precomputed is not None:
                return precomputed[0]
        except FileNotFoundError:
            aprint("")
            aprint(
                "Precomputed fit not available (Git LFS asset not pulled). "
                "Falling back to download + fit (one-time; result is cached)."
            )

    warn_if_no_cuda_gpu()
    volume = load_dust_volume()
    return fit_dust(volume)


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(gsplats_data: GSplatData, output_path: Path) -> Path:
    """Create the 3D dust scene."""
    with asection("Creating Luxar Scene"):
        gsplats_data = gsplats_data.center_at_centroid()
        gsplats_data = gsplats_data.scale_intensity(0.15)

        dims = Dimensions(
            [
                Dimension("x", unit="pc", display=True),
                Dimension("y", unit="pc", display=True),
                Dimension("z", unit="pc", display=True),
            ]
        )
        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            # ACES on purpose, not by omission: its highlight rolloff is what
            # keeps the dense cloud cores from clipping flat. That trades away
            # exact `inferno` hue fidelity, so the compiler's LUT/tone-mapping
            # warning is EXPECTED here (it fires for any non-Neutral scene) —
            # this is dust, not a scientific colour encoding.
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(tone_mapping="ACES", exposure=-0.17),
            )
            scene.attrs["title"] = (
                "GSplats: Milky Way Interstellar Dust (Leike & Enßlin 2020)"
            )

            scene.add_gsplats_from_data(
                name="interstellar_dust",
                result=gsplats_data,
                colormap="inferno",
                opacity=1.0,
                # Light volumetric compositing: near dust softly occludes far
                # dust, giving the clouds depth without crushing the diffuse
                # structure the way full kappa=1 absorption would.
                blending_mode="volumetric",
                absorption=0.3,
                # Display window [0, 0.095]. The shipped fit's robust range
                # (p99.9) tops out near 0.081, so this holds the faint diffuse
                # filaments just below clipping — brighter and the dense cores
                # flatten into featureless white.
                intensity=1.0 / 0.095,
                layer=True,
            )

            scene.add_text(
                "Interstellar Dust — Solar Neighborhood",
                position=(0.02, 0.02),
                font_size=0.045,
                anchor="top-left",
                color="rgba(255,255,255,0.65)",
                blend_mode="difference",
            )
            scene.add_text(
                "Leike & Enßlin 2020 • 3D dust density • ~1 pc/voxel",
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
    aprint("GSplats Demo: Milky Way Interstellar Dust (Leike & Enßlin 2020)")
    aprint("=" * 70)
    aprint("Real 3D dust reconstruction → Gaussian splatting → glowing fog")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_3d_milky_way_dust.luxar.zarr"

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
        aprint("Data credit: Leike, Glatzle & Enßlin (2020), A&A 639, A138")
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
