#!/usr/bin/env python3
"""GSplats Demo: Visible Human Head — Real-Color Anatomy (NLM Cryosections)

Gaussian-splats the human head in TRUE COLOR from the National Library of
Medicine's Visible Human Project cryosection photographs — actual photographs of
a frozen cadaver sliced at 1 mm, so the brain, skull, muscle, and vasculature
appear in their natural anatomical colors (not a false-color transfer function).
This is the splat pipeline applied to real photographic volumetric anatomy.

================================================================================
COLOR VOLUME → GAUSSIAN SPLATS
================================================================================

Unlike the fluorescence/CT gsplat demos (scalar intensity + a colormap), the
Visible Human cryosections are genuine RGB photographs. We keep that color:

  1. Fit Gaussian splats to the volume's *luminance* (one fit → the geometry:
     centers, covariances, amplitudes).
  2. Sample the original RGB volume at each splat center → a real per-splat
     color.
  3. Render additively with those per-splat colors → photographic-color anatomy.

The blue frozen-block background and the ruler strip are masked out before
fitting (tissue is warm-toned, R > B; the gel background is blue), so splats
land on anatomy only.

DATA SOURCE & CITATION
----------------------
U.S. National Library of Medicine — The Visible Human Project® (Male).
    Axial color cryosection photographs, head subset (slices a_vm1001–a_vm1377),
    2048×1216 24-bit RGB at 1 mm spacing.
    https://www.nlm.nih.gov/research/visible/visible_human.html
    Public domain (NLM terms, 2019). Data:
    https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Male-Images/PNG_format/head/

SELF-CONTAINED / CACHING
------------------------
On a fresh machine this demo bootstraps itself with no manual steps:
  1. Fast path: a precomputed fit + per-splat colors shipped via Git LFS
     (``demos/data/gsplats_visible_human_head/``).
  2. If those assets aren't pulled, it AUTOMATICALLY downloads the 377 color
     slices (~1.1 GB) to ``~/.cache/luxar/gsplats_visible_human_head/``, builds
     the masked RGB volume, fits luminance on the GPU, samples colors, and caches
     the result — so subsequent runs are instant.
``--recompute`` forces the download + build + fit path.

USAGE
-----
    python demo_gsplats_3d_visible_human_head.py [--recompute] [--no-serve] [--serve-only]

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan,  'C': fly controls
"""

from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.demos import (
    detect_device,
    is_lfs_pointer,
    launch_viewer,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

HEAD_BASE_URL = (
    "https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Male-Images/PNG_format/head"
)
SLICE_FIRST, SLICE_LAST = 1001, 1377  # inclusive; 377 axial head slices

DEMO_NAME = "gsplats_visible_human_head"
FIT_FILE = "vh_head.gsplats.zarr.zip"
COLORS_FILE = "vh_head_colors.npz"

CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME
PNG_DIR = CACHE_DIR / "head_png"
CACHE_FIT = CACHE_DIR / FIT_FILE
CACHE_COLORS = CACHE_DIR / COLORS_FILE

DATA_DIR = Path(__file__).resolve().parent / "data" / DEMO_NAME
LFS_FIT = DATA_DIR / FIT_FILE
LFS_COLORS = DATA_DIR / COLORS_FILE

# Preprocessing / fit parameters.
RULER_CROP_FRAC = 0.14  # drop the bottom rows (color-scale ruler + slice label)
TARGET_MAX_DIM = 896  # resample so the largest (physical) spatial axis is this
MAX_SPLATS = 4_000_000
MAX_SPLATS_PER_PASS = 600_000
ITERS_PER_PASS = 4_000
PSNR_PATIENCE = 0.1

# Display brightness (additive): a dense head over-accumulates, so scale the
# fitted amplitudes far down to keep the core from blowing out to white.
SCENE_INTENSITY = 0.008

# Physical voxel spacing of the NLM VHM color cryosections: 1.0 mm axial (slice
# spacing) vs ~0.33 mm in-plane. The assembly resamples to physically-cubic
# voxels using this ratio, so the fit volume has the correct anatomical aspect
# (no render-time stretch) and the fitter yields well-shaped, non-elongated splats.
VOXEL_Z_MM = 1.0
VOXEL_XY_MM = 0.33

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

Arbol.max_depth = 5
DEVICE = None


# =============================================================================
# Image processing (pure helpers — unit-tested; downloads/fit/IO are not)
# =============================================================================


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma of an RGB array (..., 3) in [0, 1] → (...) float32."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of anatomical tissue vs the blue gel / black background.

    Cryosection tissue is warm-toned (red ≳ blue); the frozen embedding gel is
    blue (blue > red) and the surround is near-black. Keeps warm, non-dark
    pixels. ``rgb`` is (..., 3) in [0, 1]; returns a (...) bool array.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    brightness = r + g + b
    warm = r >= b * 0.95
    return warm & (brightness > 0.18)


def mask_background(rgb: np.ndarray) -> np.ndarray:
    """Zero out non-tissue voxels so splats are fit to anatomy only."""
    mask = tissue_mask(rgb)
    return (rgb * mask[..., None]).astype(np.float32)


def crop_to_content(
    rgb: np.ndarray, pad: int = 4, rel: float = 0.05
) -> tuple[np.ndarray, tuple]:
    """Crop a masked (Z, Y, X, 3) volume to the tissue bounding box (+pad).

    A slice along each axis counts as content when its tissue-voxel count
    exceeds ``rel`` times that axis's peak count. This peak-relative rule is
    robust to persistent frame-edge artifacts (thin scan borders / specks that
    survive masking across many slices): the anatomy planes are near the peak,
    the artifacts are a tiny fraction of it. Returns the cropped volume and the
    (z0, z1, y0, y1, x0, x1) box used.
    """
    occ = rgb.sum(axis=-1) > 0.0
    if not occ.any():
        z, y, x = rgb.shape[:3]
        return rgb, (0, z, 0, y, 0, x)
    Z, Y, X = occ.shape
    zc, yc, xc = occ.sum(axis=(1, 2)), occ.sum(axis=(0, 2)), occ.sum(axis=(0, 1))

    def _extent(counts: np.ndarray) -> np.ndarray:
        idx = np.where(counts > rel * float(counts.max()))[0]
        return idx if len(idx) else np.where(counts > 0)[0]

    zi, yi, xi = _extent(zc), _extent(yc), _extent(xc)
    z0, z1 = int(zi[0]), int(zi[-1]) + 1
    y0, y1 = max(0, int(yi[0]) - pad), min(Y, int(yi[-1]) + 1 + pad)
    x0, x1 = max(0, int(xi[0]) - pad), min(X, int(xi[-1]) + 1 + pad)
    return rgb[z0:z1, y0:y1, x0:x1], (z0, z1, y0, y1, x0, x1)


def sample_colors(rgb_vol: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest-voxel RGB color for each splat center.

    ``centers`` are in (z, y, x) voxel coordinates of ``rgb_vol`` (Z, Y, X, 3).
    Returns an (N, 3) float32 array in [0, 1], clamped to the volume bounds.
    """
    zi = np.clip(np.round(centers[:, 0]).astype(int), 0, rgb_vol.shape[0] - 1)
    yi = np.clip(np.round(centers[:, 1]).astype(int), 0, rgb_vol.shape[1] - 1)
    xi = np.clip(np.round(centers[:, 2]).astype(int), 0, rgb_vol.shape[2] - 1)
    return rgb_vol[zi, yi, xi].astype(np.float32)


# =============================================================================
# Data loading (network / IO — not unit-tested)
# =============================================================================


def download_head_slices() -> Path:
    """Download the 377 head cryosection PNGs to the cache (download-once)."""
    import requests

    PNG_DIR.mkdir(parents=True, exist_ok=True)
    n_have = len(list(PNG_DIR.glob("a_vm*.png")))
    expected = SLICE_LAST - SLICE_FIRST + 1
    if n_have >= expected:
        aprint(f"  Using {n_have} cached head slices")
        return PNG_DIR

    with asection("Downloading Visible Human head cryosections (NLM, ~1.1 GB)"):
        aprint(f"Source: {HEAD_BASE_URL}")
        session = requests.Session()
        for n in range(SLICE_FIRST, SLICE_LAST + 1):
            dest = PNG_DIR / f"a_vm{n}.png"
            if dest.exists() and dest.stat().st_size > 1024:
                continue
            tmp = dest.with_suffix(".png.part")
            with session.get(f"{HEAD_BASE_URL}/a_vm{n}.png", timeout=120) as r:
                r.raise_for_status()
                tmp.write_bytes(r.content)
            tmp.rename(dest)
            if (n - SLICE_FIRST) % 40 == 0:
                aprint(f"  {n - SLICE_FIRST + 1}/{expected} slices")
    return PNG_DIR


def assemble_volume(png_dir: Path, target_max_dim: int = TARGET_MAX_DIM) -> np.ndarray:
    """Load the head PNGs into a masked, cropped, downsampled RGB volume.

    Returns a (Z, Y, X, 3) float32 array in [0, 1] with the blue gel / ruler
    background zeroed out.
    """
    from PIL import Image
    from scipy import ndimage
    from scipy.ndimage import binary_opening, zoom

    with asection("Assembling RGB head volume"):
        paths = sorted(
            png_dir.glob("a_vm*.png"), key=lambda p: int(p.stem.replace("a_vm", ""))
        )
        if not paths:
            raise FileNotFoundError(f"No head slices in {png_dir}")
        first = np.asarray(Image.open(paths[0]).convert("RGB"))
        cut = int(first.shape[0] * (1.0 - RULER_CROP_FRAC))  # drop bottom ruler rows
        vol = np.empty((len(paths), cut, first.shape[1], 3), dtype=np.float32)
        for i, p in enumerate(paths):
            a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            vol[i] = a[:cut]
        aprint(f"  Stacked {len(paths)} slices → {vol.shape}")

        vol = mask_background(vol)
        # The black background carries sparse/streaky warm scan noise that
        # survives masking. Open to drop isolated voxels, then keep only the
        # largest connected component (the head) so disconnected edge noise
        # strips don't inflate the crop box.
        occ = binary_opening(vol.sum(axis=-1) > 0.0)
        lbl, n = ndimage.label(occ)
        if n > 1:
            counts = np.bincount(lbl.ravel())
            counts[0] = 0
            occ = lbl == int(counts.argmax())
        vol *= occ[..., None].astype(np.float32)
        vol, box = crop_to_content(vol)
        aprint(f"  Masked + cleaned + cropped to {vol.shape[:3]} (box {box})")

        # Resample to physically-cubic voxels: the slices are 1.0 mm apart but
        # in-plane pixels are ~0.33 mm, so downsample each axis in proportion to
        # its physical extent (largest → target_max_dim). This bakes the correct
        # anatomical aspect into the fit volume (no render-time stretch needed)
        # and lets the fitter produce well-shaped, non-elongated splats.
        nz, ny, nx = vol.shape[:3]
        phys = np.array(
            [nz * VOXEL_Z_MM, ny * VOXEL_XY_MM, nx * VOXEL_XY_MM], dtype=np.float64
        )
        out = np.maximum(1, np.round(phys / phys.max() * target_max_dim)).astype(int)
        factors = (out[0] / nz, out[1] / ny, out[2] / nx, 1.0)
        vol = zoom(vol, factors, order=1).clip(0.0, 1.0).astype(np.float32)
        aprint(f"  Resampled to physically-isotropic {vol.shape[:3]}")
        return vol


# =============================================================================
# Fit
# =============================================================================


def fit_head(rgb_vol: np.ndarray) -> tuple[GSplatData, np.ndarray]:
    """Fit luminance, sample per-splat colors, cache both. Returns (fit, colors)."""
    global DEVICE
    if DEVICE is None:
        DEVICE = detect_device()

    from luxar.gsplats import fit_progressive_gaussian_splats

    lum = luminance(rgb_vol)
    with asection(f"Fitting GSplats to luminance ({lum.shape}, device={DEVICE})"):
        result = fit_progressive_gaussian_splats(
            lum,
            max_splats=MAX_SPLATS,
            max_splats_per_pass=MAX_SPLATS_PER_PASS,
            iters_per_pass=ITERS_PER_PASS,
            psnr_patience=PSNR_PATIENCE,
            device=DEVICE,
            verbose=True,
        )
        aprint(f"Fitted {len(result.amplitudes):,} splats")

    with asection("Sampling per-splat colors from the RGB volume"):
        colors = sample_colors(rgb_vol, result.centers)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.save(
        CACHE_FIT,
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )
    tmp = CACHE_COLORS.parent / (CACHE_COLORS.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, colors=colors)
    tmp.rename(CACHE_COLORS)
    return result, colors


def load_or_build() -> tuple[GSplatData, np.ndarray]:
    """Return (fit, per-splat colors), self-contained on a fresh system."""
    if not RECOMPUTE:
        # local processed cache first
        if CACHE_FIT.exists() and CACHE_COLORS.exists():
            aprint("  Using cached fit + colors")
            fit = GSplatData.load(CACHE_FIT, include_stats=False)
            with np.load(CACHE_COLORS) as d:
                return fit, d["colors"]
        # shipped LFS assets
        if (
            LFS_FIT.exists()
            and LFS_COLORS.exists()
            and not is_lfs_pointer(LFS_FIT)
            and not is_lfs_pointer(LFS_COLORS)
        ):
            aprint("  Copying shipped fit + colors from package data to cache")
            import shutil

            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(LFS_FIT, CACHE_FIT)
            shutil.copy2(LFS_COLORS, CACHE_COLORS)
            fit = GSplatData.load(CACHE_FIT, include_stats=False)
            with np.load(CACHE_COLORS) as d:
                return fit, d["colors"]
        aprint(
            "Precomputed fit not available (Git LFS assets not pulled). "
            "Falling back to download + fit (one-time; result is cached)."
        )

    warn_if_no_cuda_gpu()
    png_dir = download_head_slices()
    vol = assemble_volume(png_dir)
    return fit_head(vol)


# =============================================================================
# Scene
# =============================================================================


def create_luxar_scene(fit: GSplatData, colors: np.ndarray, output_path: Path) -> Path:
    """Build the true-color Visible Human head scene."""
    with asection("Creating Luxar Scene"):
        # Aspect is already correct (the volume was resampled to cubic voxels),
        # so just center and dim. Additive blending (the gsplat norm): a dense
        # head over-accumulates, so amplitudes are scaled WAY down to avoid a
        # blown-out white core — reduce brightness, not blend mode.
        centered = fit.center_at_centroid().scale_intensity(SCENE_INTENSITY)
        dims = Dimensions(
            [
                Dimension("x", unit="mm", display=True),
                Dimension("y", unit="mm", display=True),
                Dimension("z", unit="mm", display=True),
            ]
        )
        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(tone_mapping="Neutral"),
            )
            scene.attrs["title"] = "GSplats: Visible Human Head (NLM cryosections)"
            scene.add_gsplats(
                name="visible_human_head",
                centers=centered.centers,
                amplitudes=centered.amplitudes,
                cholesky_factors=centered.cholesky_factors,
                colors=colors.astype(np.float32),
                opacity=1.0,
                blending_mode="additive",
                layer=True,
            )
            scene.add_text(
                "Visible Human Head — real-color anatomy",
                position=(0.02, 0.02),
                font_size=0.045,
                anchor="top-left",
                color="rgba(255,255,255,0.7)",
                blend_mode="difference",
            )
            scene.add_text(
                "NLM Visible Human Project • color cryosections → Gaussian splats",
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
    aprint("GSplats Demo: Visible Human Head — Real-Color Anatomy (NLM)")
    aprint("=" * 70)
    aprint("Photographic color cryosections → Gaussian splatting")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_3d_visible_human_head.luxar.zarr"

    if SERVE_ONLY:
        if output_path.exists():
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    fit, colors = load_or_build()
    aprint(f"Splats: {len(fit.amplitudes):,}")

    scene_path = create_luxar_scene(fit, colors, output_path)

    if NO_SERVE:
        aprint(f"Dataset generated at {scene_path}")
    else:
        aprint("Data credit: U.S. National Library of Medicine — Visible Human Project")
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
