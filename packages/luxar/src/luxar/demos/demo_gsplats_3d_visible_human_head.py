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
  3. Render volumetrically with those per-splat colors → photographic-color anatomy.

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
The colors sidecar is indexed positionally against the fit, and the cache goes
through ``save_with_lod`` (a streaming ladder whose rungs are each written in
hilbert order), which reorders splats — so the colors are sampled from the SAVED
store's own order (save → reload → sample), never from the in-memory fit. On load
the pair is verified against that invariant (splats sharing a voxel must share a
color); a mismatched pair is reported and refitted rather than rendered.

On a fresh machine this demo bootstraps itself with no manual steps:
  1. Fast path: the manifest resolves a precomputed fit and its matching colors
     sidecar through a checksum-verified cache, the in-repo Git LFS copies, or
     the hosted record. The pair is then verified against the invariant above.
  2. If no precomputed pair is available, it downloads the 377 color slices
     (~1.1 GB) to ``~/.cache/luxar/gsplats_visible_human_head/``, builds the
     masked RGB volume, fits luminance on the GPU, caches the fit, then reloads
     it and samples the colors from the stored splat order — so subsequent runs
     load that (verified) local pair instantly.
``--recompute`` forces the download + build + fit path.

Manifest integrity faults are not treated as ordinary absence: a stale in-repo
object matching neither pin, or a cache write that cannot be repaired, is
reported by the resolver instead of being hidden behind an automatic refit.

The fast path was broken for a while (#1670): the shipped sidecar had been
sampled in the pre-save splat order, so it did not correspond to the shipped
store (measured same-voxel agreement 0.00097 over 1,911,192 splats) and the
guard rejected it on every run. Recovering it needed no refit — the fit itself
was never wrong, only the color ORDER — so the volume was rebuilt and resampled
at the shipped store's own centers. The sidecar carries no positions, so a
mis-ordered one can never be repaired in place: resampling is the only route.

That resample is exactly the operation ``load_or_build`` refuses to perform
automatically, for the reason given at its rejection branch: agreement 1.0 does
NOT prove the resample used the right coordinate frame, because splats sharing a
voxel share an index in any frame whatsoever. It was therefore verified out of
band, on three pieces of evidence this guard cannot produce:

  * the rebuilt volume's shape, ``(636, 451, 896)``, matches the stored centers
    spanning ``[0, 0, 0]``–``[635, 450, 895]`` exactly, so neither the crop box
    nor the resample factor drifted;
  * 99.96% of the stored centers land on non-zero (tissue) voxels, against a
    32.93% tissue fraction for the volume as a whole — and every deliberately
    misaligned frame scores lower (a 10-voxel shift 98.9%, 25 voxels 90.7%, a
    y/x axis swap 22.0%, below the base rate);
  * the regenerated colors preserve the previous sidecar's colour distribution
    (total-variation distance 0.0065), which pins the SOURCE — the same volume,
    masked the same way — independently of the ordering.

The three are complementary, and none suffices alone: the distribution check
would survive a small translation, the tissue-hit rate would survive a subtle
resample change, and the shape check alone says nothing about content.

Anyone regenerating this sidecar should reproduce all three rather than trusting
the agreement number alone. ``--recompute`` is the supported route and writes a
fresh fit AND a matching sidecar via :func:`save_and_sample_colors`. The cheaper
repair, when the fit is fine and only the sidecar is lost, is not wired into the
demo (see the refusal in :func:`load_or_build`): set ``data_dir`` to
``Path(__file__).parent / "data" / DEMO_NAME``, call
``vol, _ = assemble_volume(PNG_DIR)``, :func:`sample_colors` at
``GSplatData.load(data_dir / FIT_FILE).centers``, then :func:`_save_colors_u8` to
``data_dir / COLORS_FILE`` and run ``make gen-data-manifest``. This preserves the
shipped fit and the 20 MB of Git LFS history that goes with it.

USAGE
-----
    python demo_gsplats_3d_visible_human_head.py [--recompute] [--no-serve] [--serve-only]

Controls:
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan,  'C': fly controls
"""

DEMO_META = {
    "key": "gsplats_3d_visible_human_head",
    "title": "Visible Human Head (cryosection anatomy)",
    "description": "Human head in true color from NLM Visible Human cryosections as Gaussian splats.",
    "category": "medical",
    "geometry": "gsplats",
    "requirements": {
        # Back to 25 (#1670 resolved): the shipped sidecar was regenerated
        # against the shipped fit and the pair now passes `_colors_match_fit`
        # at agreement 1.0, so the DEFAULT path is the two Git-LFS assets —
        # 20.6 MB fit + 5.0 MB colors — not the 1.1 GB cryosection download.
        # Read by `luxar demo run-all`, whose `--max-download-mb` default of 200
        # therefore stops skipping this demo.
        "download_mb": 25,
        # "medium" again for the same reason: the default path is a cached load
        # of a 1.9M-splat store, not a progressive fit over a ~10 GB RGB volume.
        # Only `--recompute` still pays that.
        "compute": "medium",
        # Still "optional": the fit genuinely runs on CPU (slowly).
        "gpu": "optional",
        "local_data": "git-lfs",
    },
    "caches": ["gsplats_visible_human_head"],
    "outputs": ["gsplats_3d_visible_human_head"],
    "citation": {
        "short": "NLM Visible Human Project (Spitzer et al. 1996)",
        "ref": "Spitzer et al. 1996",
        "doi": "10.1136/jamia.1996.96236280",
        "license": "Public domain (NLM)",
    },
}

from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import (
    DatasetUnavailable,
    add_demo_caption,
    detect_device,
    ensure_dataset,
    launch_viewer,
    load_local_fit_gsplats_at,
    local_fit_path,
    parse_demo_flags,
    voxel_sampled_payload_agreement,
    warn_if_no_cuda_gpu,
)
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
from luxar.demos._lod_policy import save_with_lod
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
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
# A local refit is OUR pair, not a copy of the hosted one, so both halves go to
# the demo's local-fit namespace (#1618), outside the manifest checksum gate.
LOCAL_FIT = local_fit_path(DEMO_NAME, FIT_FILE)
LOCAL_COLORS = local_fit_path(DEMO_NAME, COLORS_FILE)

# Preprocessing / fit parameters.
RULER_CROP_FRAC = 0.14  # drop the bottom rows (color-scale ruler + slice label)
TARGET_MAX_DIM = 896  # resample so the largest (physical) spatial axis is this
MAX_SPLATS = 4_000_000
MAX_SPLATS_PER_PASS = 600_000
ITERS_PER_PASS = 4_000
PSNR_PATIENCE = 0.1

# Display brightness: volumetric compositing bounds the sum, but this dense
# head still reads hot, so scale amplitudes down to keep the core from clipping.
SCENE_INTENSITY = 0.008

# Minimum same-voxel color agreement for a cached/shipped (fit, colors) pair to be
# trusted. Aligned data scores exactly 1.000. DO NOT LOOSEN THIS — the gate is
# deliberately tight, because the interesting failures are NEAR MISSES rather
# than full shuffles: a different space-filling curve, or a changed within-voxel
# tie-break, lands at 0.90-0.97 (measured on the CT demo's shipped pair, whose
# aligned sidecar can be permuted the way each mistake would have written it:
# the writer's own morton order instead of hilbert 0.901, roll-by-one 0.955,
# adjacent-pair swap 0.962 — the worst case the constant has to stay above).
# Only a FULL shuffle falls to the payload's own chance level Σp², which is
# payload-dependent and is NOT what the threshold is set against: 1.4e-05 for
# this demo's sampled uint8 RGB (measured over the 1,911,192 rows of the shipped
# `vh_head_colors.npz`; an actual full shuffle of it scores 2.85e-05), against
# 0.027 for the CT demo's 117 organ labels.
MIN_COLOR_AGREEMENT = 0.99

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


def assemble_volume(
    png_dir: Path, target_max_dim: int = TARGET_MAX_DIM
) -> tuple[np.ndarray, tuple]:
    """Load the head PNGs into a masked, cropped, downsampled RGB volume.

    Returns ``(vol, acquisition)``: a (Z, Y, X, 3) float32 array in [0, 1] with
    the blue gel / ruler background zeroed out, and the ``(shape, dtype)`` of the
    PNG stack as downloaded, to be declared to the fit — ``vol`` is a cropped,
    masked and resampled copy of it.
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
        # The acquisition is the PNG stack as downloaded: 8-bit RGB at full
        # slice resolution, before the ruler crop, the masking and the
        # physically-isotropic resample below. `first` is read above without the
        # float cast the loop applies, so its dtype is the stored one.
        acquisition = (
            (len(paths), first.shape[0], first.shape[1], 3),
            str(first.dtype),
        )
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
        return vol, acquisition


# =============================================================================
# Fit
# =============================================================================


def _save_colors_u8(colors: np.ndarray, path: Path) -> None:
    """Save per-splat colors as uint8 (0–255) — ~4× smaller than float32."""
    u8 = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, colors=u8)
    tmp.rename(path)


def _load_colors_f32(path: Path) -> np.ndarray:
    """Load per-splat colors as float32 in [0, 1] (uint8 assets are rescaled)."""
    with np.load(path) as d:
        c = d["colors"]
    return c.astype(np.float32) / 255.0 if c.dtype == np.uint8 else c.astype(np.float32)


def _colors_match_fit(fit: GSplatData, colors: np.ndarray, source: str) -> bool:
    """Is this (fit, colors) pair positionally aligned? Reports why if not.

    The colors are indexed positionally against the fit; the manifest declares
    this generation contract with ``positional_pair``, while this check catches
    a sidecar written in a different splat order — every splat renders some
    other splat's color. Both were sampled nearest-voxel, so splats sharing a
    voxel must share a color — see
    ``voxel_sampled_payload_agreement``.
    """
    if len(colors) != len(fit.centers):
        aprint(
            f"{source}: {len(colors):,} colors for {len(fit.centers):,} splats "
            "— the sidecar does not belong to this fit."
        )
        return False
    agreement = voxel_sampled_payload_agreement(fit.centers, colors)
    if agreement is None:
        # Too little evidence to judge — accepted (rejecting would force a
        # multi-GB refit on every sparse fit), but never silently.
        aprint(
            f"{source}: too few same-voxel splats to check the color order "
            "— the pair is accepted UNVERIFIED."
        )
        return True
    if agreement < MIN_COLOR_AGREEMENT:
        aprint(
            f"{source}: same-voxel color agreement {agreement:.3f} < "
            f"{MIN_COLOR_AGREEMENT} — the colors are not in the fit's splat order."
        )
        return False
    return True


def save_and_sample_colors(
    fit: GSplatData, rgb_vol: np.ndarray
) -> tuple[GSplatData, np.ndarray]:
    """Cache the fit, then sample the per-splat colors in the STORE's own order.

    The writer reorders splats — ``save_with_lod`` splits them into a streaming
    ladder and each rung is written spatially (``ordering="hilbert"``) — so
    sampling the RGB volume at the in-memory fit's centers would produce a sidecar
    that no longer lines up with what ``load`` hands back. Saving first and
    sampling the RELOADED centers makes the pair aligned by construction under any
    writer ordering, and makes this path return exactly what the cached path will
    load next run — including the uint8 quantization of the colors.
    Returns ``(stored_fit, colors)``.
    """
    LOCAL_FIT.parent.mkdir(parents=True, exist_ok=True)
    save_with_lod(
        fit,
        LOCAL_FIT,
        # `stream`, not `levels`: the head IS a large orbited single object, but
        # the scene is built from explicit `centers=`/`amplitudes=` arrays plus
        # the per-splat colours sidecar, and `add_gsplats` writes a flat leaf —
        # so a substitutive ladder would cost the ~38% extra bytes recorded in
        # `_lod_policy` and be discarded before the viewer ever saw it. Carrying
        # the levels into the scene needs the colours to live on the
        # `GSplatData` so the whole fit can go through `add_gsplats_from_data`;
        # until then `stream` is the honest choice.
        recipe="stream",
        encoding_mode=EncodingMode.MEMORY,  # uint8 Cholesky — smallest on-disk
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )
    stored = GSplatData.load(LOCAL_FIT, include_stats=False)
    with asection("Sampling per-splat colors from the RGB volume"):
        colors = sample_colors(rgb_vol, stored.centers)
    _save_colors_u8(colors, LOCAL_COLORS)
    aprint(f"Cached the refit pair under {LOCAL_FIT.parent}")
    # Read the sidecar back so the recompute path matches the shipped/cached path
    # exactly (both render the quantized colors).
    return stored, _load_colors_f32(LOCAL_COLORS)


def fit_head(rgb_vol: np.ndarray, acquisition=None) -> tuple[GSplatData, np.ndarray]:
    """Fit luminance, cache, reload, sample colors; returns ``(stored_fit, colors)``."""
    global DEVICE
    if DEVICE is None:
        DEVICE = detect_device()

    from luxar.gsplats import fit_progressive_gaussian_splats

    lum = luminance(rgb_vol)
    with asection(f"Fitting GSplats to luminance ({lum.shape}, device={DEVICE})"):
        src_shape, src_dtype = acquisition or (None, None)
        result = fit_progressive_gaussian_splats(
            lum,
            # The acquisition is the 8-bit RGB slice stack as downloaded; the
            # fit is of its luminance, resampled and cropped. Declaring the
            # stack keeps the ratio about the data rather than about `lum`.
            # The two grids differ in rank, which `gsplat info` shows: source
            # 4D (slices, H, W, 3), fitted 3D.
            source_shape=src_shape,
            source_dtype=src_dtype,
            max_splats=MAX_SPLATS,
            max_splats_per_pass=MAX_SPLATS_PER_PASS,
            iters_per_pass=ITERS_PER_PASS,
            psnr_patience=PSNR_PATIENCE,
            device=DEVICE,
            verbose=True,
        )
        aprint(f"Fitted {len(result.amplitudes):,} splats")

    return save_and_sample_colors(result, rgb_vol)


def local_refit_pair() -> tuple[GSplatData, np.ndarray] | None:
    """A pair THIS machine refitted earlier, or None if there is nothing usable.

    Consulted after the manifest-resolved pair and BEFORE refitting, which is
    what makes the refit one-time (#1618).

    Guarded, unlike the manifest door above it: these bytes have no checksum, no
    remote and no second copy, so
    a truncated zip here (a Ctrl-C mid-save) would otherwise raise ``BadZipFile``
    out of :func:`load_or_build` on EVERY launch with a manual delete as the only
    recovery. ``load_local_fit_gsplats_at`` reports the path and the error and
    returns None; the refit then overwrites the rubble.

    Read through the ``LOCAL_*`` constants, which is also what the refit WRITES
    through — a door that re-derived its path from the cache root would be a
    second source of truth for it (#1618 review, A). The sidecar is checked first
    because it is a ``stat()`` and the fit is a large zip decode.
    """
    if not LOCAL_COLORS.exists():
        return None
    local = load_local_fit_gsplats_at([LOCAL_FIT], label=DEMO_NAME)
    if local is None:
        return None
    try:
        colors = _load_colors_f32(LOCAL_COLORS)
    except Exception as exc:  # noqa: BLE001 — a refit is the recovery
        aprint(f"⚠️  Local colors {LOCAL_COLORS} unreadable ({exc!r}).")
        return None
    if not _colors_match_fit(local[0], colors, f"{LOCAL_FIT} + {LOCAL_COLORS}"):
        return None
    aprint("  Using this machine's own earlier refit")
    return local[0], colors


def load_or_build() -> tuple[GSplatData, np.ndarray]:
    """Return (fit, per-splat colors), self-contained on a fresh system."""
    if not RECOMPUTE:
        # Deliberately not gated on an existing cache: the resolver returns the
        # whole positional pair through cache -> in-repo LFS -> hosted data and
        # verifies both manifest digests before either path is consumed.
        try:
            resolved = {path.name: path for path in ensure_dataset(DEMO_NAME)}
        except DatasetUnavailable as exc:
            aprint(f"Manifest fetch unavailable ({exc}).")
            resolved = {}
        if resolved:
            fit_path = resolved[FIT_FILE]
            colors_path = resolved[COLORS_FILE]
            precomputed = GSplatData.load(fit_path, include_stats=False)
            colors = _load_colors_f32(colors_path)
            if _colors_match_fit(precomputed, colors, f"{fit_path} + {colors_path}"):
                return precomputed, colors
        # A pair this machine refitted earlier, in its own namespace — checked
        # BEFORE refitting, which is what makes the refit below one-time.
        pair = local_refit_pair()
        if pair is not None:
            return pair
        # A rejected pair triggers a FULL refit, not a cheap re-sample of the
        # assembled volume at the stored centers, even though that would be far
        # cheaper (no fit, just the ~1.1 GB assembly). The reason is that a
        # re-sample cannot be VERIFIED by this guard: splats sharing a voxel share
        # an index in any coordinate frame whatsoever, so a re-sample taken in the
        # wrong frame (a different crop box, a different resample factor — exactly
        # the parameters that drift between the shipped artifact and today's code)
        # still scores agreement 1.0. It would need its own, separate guard; until
        # one exists, refitting is the only outcome this file can vouch for.
        aprint(
            "Precomputed fit not available (the manifest pair is unavailable, or "
            "the fit and its colors sidecar disagree). Falling back to "
            f"download + fit (one-time; cached under {LOCAL_FIT.parent})."
        )

    warn_if_no_cuda_gpu()
    png_dir = download_head_slices()
    vol, acquisition = assemble_volume(png_dir)
    return fit_head(vol, acquisition)


# =============================================================================
# Scene
# =============================================================================


def create_luxar_scene(fit: GSplatData, colors: np.ndarray, output_path: Path) -> Path:
    """Build the true-color Visible Human head scene."""
    with asection("Creating Luxar Scene"):
        # Aspect is already correct (the volume was resampled to cubic voxels),
        # so just center and dim. Volumetric compositing bounds the accumulated
        # radiance, but a dense head still reads hot, so amplitudes stay scaled
        # WAY down — reduce brightness, not blend mode.
        centered = fit.center_at_centroid().scale_intensity(SCENE_INTENSITY)
        dims = Dimensions(
            [
                Dimension("x", unit="mm", display=True),
                Dimension("y", unit="mm", display=True),
                Dimension("z", unit="mm", display=True),
            ]
        )

        # Face the viewer, head up. Without an authored camera the viewer
        # auto-frames on the bounding box with world +Y up, which for this
        # volume lands the body on its side — the subject is not axis-aligned
        # to the viewer's defaults in any orientation that reads as "a person".
        #
        # The fit's centers are in the source volume's index order, so the
        # anatomy maps onto the columns rather than onto the x/y/z the scene
        # declares:
        #   col 0 = axial slice index. NLM cuts the VHM from the head DOWN, so
        #           the index grows inferiorly -> SUPERIOR is -x.
        #   col 1 = image row. Axial cryosection photographs put anterior at the
        #           top of the frame (row 0) -> ANTERIOR is -y.
        #   col 2 = image column -> LEFT-RIGHT, the widest axis (shoulders).
        # Both readings are confirmed by the built cloud: the head protrudes
        # along -col0, and looking down col0 shows a head with the shoulders
        # spread along col2.
        #
        # So: stand off along -y (in front of the face), look back at the
        # centroid, and point "up" along -x.
        half = np.maximum(
            np.abs(centered.centers.max(axis=0)),
            np.abs(centered.centers.min(axis=0)),
        )
        # Fit the taller of (height, width) into the frame, with a little air.
        need = float(max(half[0], half[2])) * 1.15
        cam_dist = need / np.tan(np.radians(CINEMATIC_FOV_DEG) / 2.0)
        radius = float(np.linalg.norm(half))
        camera = CameraConfig(
            position=(0.0, -cam_dist, 0.0),
            target=(0.0, 0.0, 0.0),
            up=(-1.0, 0.0, 0.0),
            near=float(max(1.0, (cam_dist - radius) * 0.5)),
            far=float((cam_dist + radius) * 2.0),
        )
        aprint(
            f"  🎥 Facing the subject: camera {cam_dist:,.0f} in front "
            f"(-y), up = -x (superior)"
        )
        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                citation=DEMO_META["citation"],
                dimensions=dims,
                viewer_config=ViewerConfig(
                    cinematic_mode=True, tone_mapping="ACES", camera=camera
                ),
            )
            scene.attrs["title"] = "GSplats: Visible Human Head (NLM cryosections)"
            scene.add_gsplats(
                name="visible_human_head",
                centers=centered.centers,
                amplitudes=centered.amplitudes,
                cholesky_factors=centered.cholesky_factors,
                colors=colors.astype(np.float32),
                opacity=1.0,
                absorption=1.0,
                blending_mode="volumetric",
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
            add_demo_caption(
                scene,
                "NLM Visible Human Project • color cryosections → Gaussian splats",
                DEMO_META.get("citation"),
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
