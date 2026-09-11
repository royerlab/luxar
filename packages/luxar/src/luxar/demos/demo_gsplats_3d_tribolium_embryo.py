#!/usr/bin/env python3
"""GSplats Demo: 3D Tribolium castaneum Embryo (Zenodo / Cell Tracking Challenge)

Visualises a large, isotropic 3D light-sheet microscopy volume of a developing
beetle embryo using Gaussian splatting.

================================================================================
REAL MICROSCOPY — LIGHT-SHEET IMAGING OF A TRIBOLIUM EMBRYO
================================================================================

The dataset is a single time-point from a long-term light-sheet fluorescence
microscopy recording of *Tribolium castaneum* (red flour beetle) embryonic
development.  The volume is near-isotropic at 0.381 µm per voxel and covers
965 × 1871 × 991 voxels — roughly 368 × 713 × 378 µm.

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source:  Cell Tracking Challenge (celltrackingchallenge.net)
Zenodo:  https://zenodo.org/records/5270323
Record:  GIANI Paper — Supplemental File 2
Size:    2.6 GB download  /  3.3 GB uncompressed
Volume:  965 × 1871 × 991 voxels, isotropic 0.381 µm

Imaging:
--------
Microscope:  Zeiss LightSheet Z.1
Organism:    Tribolium castaneum (red flour beetle)
Label:       Fluorescent reporter (nuclear / histone)
Resolution:  0.381 × 0.381 × 0.381 µm (isotropic)

How to Cite:
------------
Barry et al. (2022).  GIANI — open-source software for automated analysis
of 3D microscopy images.  *Journal of Cell Science*, 135, jcs259511.
DOI: 10.1242/jcs.259511

Cell Tracking Challenge — Maska, M. et al. (2023).  The Cell Tracking
Challenge: 10 years of objective benchmarking.  *Nature Methods*, 20, 1010–1020.

WORKFLOW:
=========

1. **Download** ZIP from Zenodo (2.6 GB, with resume support)
2. **Extract** multi-page TIFF stack from archive
3. **Load** as 3D volume in raw counts, optionally downsample
4. **Fit** Gaussian splats with GPU acceleration + caching
5. **Create 3D scene** in voxel coordinates
6. **Visualise** — rotate, zoom, explore the embryo

USAGE:
======
    python demo_gsplats_3d_tribolium_embryo.py [--recompute] [--no-serve] [--serve-only] [--downsample=N]

Options:
    --recompute:      Force GPU re-fitting (reuses cached download/extraction)
    --no-serve:       Generate scene without launching viewer
    --serve-only:     Just serve a previously generated scene
    --show-roundtrip: Show matplotlib comparison of original vs reconstructed volume
    --downsample=N:   Downsample factor for fitting (default: 1)

By default, precomputed GSplats are loaded from the local cache under
~/.cache/luxar/gsplats_tribolium. A cold cache re-fits from the raw source
(requires a GPU and, if uncached, network).
If the local Tribolium cache predates the 675-count floor, run once with
--recompute, which reuses the downloaded archive and extracted TIFF.
A stale fit is detected and warned about on load: the cache cannot be compared
against the packaged LFS source (which does not exist for this
non-redistributable dataset), so the recorded floor is checked instead.

Output:
    - Scene saved to:  datasets/demos/gsplats_3d_tribolium_embryo.luxar.zarr
    - Automatically opens in browser
"""

DEMO_META = {
    "key": "gsplats_3d_tribolium_embryo",
    "title": "3D Tribolium Embryo",
    "description": "A 3D light-sheet volume of a developing Tribolium beetle embryo as Gaussian splats.",
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 3,
        "compute": "medium",
        # The precomputed fit is no longer shipped in-tree (the source data is
        # not redistributable), so a first run fetches the raw source and
        # refits — which needs a GPU. Nothing has to be placed by hand.
        "gpu": "required",
        "local_data": None,
    },
    "caches": ["gsplats_tribolium"],
    "outputs": ["gsplats_3d_tribolium_embryo"],
    "citation": {
        "short": (
            "Barry 2021 (GIANI, Zenodo 5270323); "
            "Cell Tracking Challenge (Maška et al. 2023)"
        ),
        "ref": "Barry / Maška et al. 2023",
        "doi": "10.5281/zenodo.5270323",
    },
}

import sys
import zipfile
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    MissingDependencyError,
    add_demo_caption,
    launch_viewer,
    load_precomputed_gsplats,
    parse_demo_flags,
    require_module,
    warn_if_no_cuda_gpu,
)
from luxar.demos._lod_policy import save_with_lod
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Data source
ZENODO_URL = (
    "https://zenodo.org/api/records/5270323/files/Supplemental_File_2.zip/content"
)

# Volume specs
VOXEL_SIZE_UM = 0.381  # Isotropic voxel size in micrometres

# Fit parameters (fixed-K, seeds=K*)
MAX_SPLATS = 510000

# Specimen background level, in RAW CAMERA COUNTS, measured on this volume.
#
# This stack has TWO background levels, not one, and the difference is what
# makes the default floor the wrong tool here:
#
#   ~204 counts   the medium OUTSIDE the embryo (detector offset). Very tight:
#                 pooled corner cubes give p50 203, p99 224.
#   ~675 counts   the specimen's OWN background INSIDE the embryo
#                 (autofluorescence + scattered light). A gap between nuclei at
#                 the volume centre reads p10 614 / p50 644 / p90 674, and the
#                 interior histogram peaks flatly over 650-700.
#
# `floor="auto"` takes the histogram MODE over the whole box. MORE THAN HALF of
# this FOV is empty medium (the embryo envelope covers ~44% of the box, so the
# medium is ~56%), so the narrow 204-count medium peak is the tallest bin and
# `auto` resolves to ~205 — it strips the detector offset and stops there. The
# median cap cannot rescue it either, since the mode (204) is already BELOW the
# median (333). The specimen's own 675-count haze is never seen, leaving 75.6% of
# the interior MASS as background for the splats to spend themselves on; that
# haze is what buried the nuclei.
#
# The estimator is not wrong, it is just being shown the wrong population: run
# `auto` on a specimen-only crop and it lands on ~675 by itself. Hence an
# explicit value here rather than a different spec. Measured on a 96x512x512
# interior crop at 30k seeds, scoring reconstruction contrast between nuclei
# (>1200 counts) and background (500-750), with the dim-nuclei band (800-1200)
# as the guardrail against over-flooring:
#
#   floor   splats   nuclei/bg contrast   dim/nuclei   nuclei PSNR
#   none    24,921            13.6x          0.286       32.74 dB
#   500     24,499            16.1x          0.274       32.13 dB
#   600     20,453             426x          0.202       29.90 dB
#   675     18,467            3092x          0.147       28.33 dB   <- chosen
#   750     18,110          41,319x          0.100       26.64 dB
#   p90     17,039             inf           0.001       19.19 dB   <- destroys it
#
# 675 is the measured specimen background, and the guardrail is healthy there:
# the dim band keeps 15% of the nuclei level and nuclei PSNR gives up 4.4 dB
# against an unfloored fit. `p90` is the failure mode the repo warns about —
# contrast looks infinite because the dim band is simply gone (0.001) and nuclei
# PSNR collapses by 13 dB. Do not raise this without re-running that table: a
# higher floor always looks cleaner in a MIP, which is exactly the trap.
SPECIMEN_BACKGROUND_COUNTS = 675.0
EXPECTED_VOLUME_MAX_COUNTS = 15902.0

# Cache location
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_tribolium"

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
SHOW_ROUNDTRIP = "--show-roundtrip" in sys.argv

DOWNSAMPLE_FACTOR = 1
for _arg in sys.argv:
    if _arg.startswith("--downsample="):
        DOWNSAMPLE_FACTOR = int(_arg.split("=")[1])

# Arbol logging depth
Arbol.max_depth = 5

# Auto-detected on first fit
DEVICE = None


# =============================================================================
# Data Loading
# =============================================================================


def download_tribolium_data() -> Path:
    """Download the Tribolium embryo ZIP from Zenodo (2.6 GB).

    Returns:
        Path to downloaded ZIP file.
    """
    from luxar.demos import robust_download

    zip_path = CACHE_DIR / "Supplemental_File_2.zip"

    # robust_download handles caching (via expected_size), resume, and verification
    with asection("Downloading Tribolium embryo dataset from Zenodo"):
        aprint("Source: https://zenodo.org/records/5270323")
        aprint("Size:   ~2.5 GB")

        robust_download(
            ZENODO_URL,
            zip_path,
            max_retries=5,
            timeout=600,
            expected_size=2_641_547_030,
        )

    return zip_path


def extract_and_load_volume(zip_path: Path) -> np.ndarray:
    """Extract TIFF from ZIP and load as 3D volume.

    Args:
        zip_path: Path to downloaded ZIP archive.

    Returns:
        3D float32 volume in raw camera counts, shape (Z, Y, X).
        Deliberately NOT normalised: the fitter normalises internally, and
        raw counts are what make ``SPECIMEN_BACKGROUND_COUNTS`` meaningful.
    """
    tifffile = require_module("tifffile")

    extract_dir = CACHE_DIR / "extracted"

    with asection("Extracting TIFF from ZIP"):
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            aprint(f"Extracting to {extract_dir}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Only extract TIFF files
                tiff_members = [
                    m for m in zf.namelist() if m.lower().endswith((".tif", ".tiff"))
                ]
                if not tiff_members:
                    # Extract everything if no TIFFs found at top level
                    zf.extractall(extract_dir)
                else:
                    for member in tiff_members:
                        zf.extract(member, extract_dir)
                    aprint(f"  Extracted {len(tiff_members)} TIFF file(s)")
        else:
            aprint("Using cached extraction")

    with asection("Loading 3D volume"):
        # Find all TIFF files recursively (case-insensitive)
        tiff_files = sorted(
            f for f in extract_dir.rglob("*") if f.suffix.lower() in (".tif", ".tiff")
        )

        if not tiff_files:
            raise FileNotFoundError(
                f"No TIFF files found in {extract_dir}. Check the ZIP contents."
            )

        aprint(f"Found {len(tiff_files)} TIFF file(s)")

        if len(tiff_files) == 1:
            # Single multi-page TIFF
            volume = tifffile.imread(str(tiff_files[0]))
        else:
            # Multiple TIFF slices — stack as Z slices
            volume = tifffile.imread([str(f) for f in tiff_files])

        aprint(f"Raw volume: shape={volume.shape}, dtype={volume.dtype}")

        # Handle multi-channel: if 4D with small second axis, pick channel 0
        if volume.ndim == 4 and volume.shape[1] <= 4:
            aprint(f"  Multi-channel detected ({volume.shape[1]} ch), using channel 0")
            volume = volume[:, 0, :, :]
        elif volume.ndim == 4 and volume.shape[0] <= 4:
            aprint(f"  Multi-channel detected ({volume.shape[0]} ch), using channel 0")
            volume = volume[0, :, :, :]

        if volume.ndim != 3:
            raise ValueError(
                f"Expected 3D volume, got {volume.ndim}D with shape {volume.shape}"
            )

        # Handed on in RAW CAMERA COUNTS. The scaling to [0, 1] still happens —
        # the fit requires it — but it now lives in `fit_tribolium` next to the
        # floor it has to agree with, rather than here where the two were a
        # function call apart.
        #
        # The point is that a floor is only checkable in counts. In this stack the
        # medium sits at ~204 and the specimen background at ~675, both readable
        # straight off a histogram; expressed against the normalised volume the
        # same level is 0.0424, a number nobody can sanity-check and which goes
        # quietly wrong if the source maximum ever moves. So the measurement stays
        # in counts and is converted at the point of use.
        volume = volume.astype(np.float32)
        aprint(
            f"Loaded (raw counts): shape={volume.shape}, "
            f"range=[{volume.min():.0f}, {volume.max():.0f}], "
            f"size={volume.nbytes / (1024**3):.1f} GB"
        )

    return volume


def load_tribolium_volume() -> np.ndarray:
    """Full pipeline: download, extract, load, downsample.

    Note: The full volume is ~7 GB as float32 (965 x 1871 x 991).
    Fitting temporarily holds both raw and normalised copies before the fitter's
    own working memory. Downsampling reduces each copy to ~0.9 GB at 2x.

    Returns:
        3D float32 volume in raw camera counts, ready for GSplat fitting.
    """
    zip_path = download_tribolium_data()
    volume = extract_and_load_volume(zip_path)

    if DOWNSAMPLE_FACTOR > 1:
        with asection(f"Downsampling by {DOWNSAMPLE_FACTOR}x"):
            from scipy.ndimage import zoom

            factor = 1.0 / DOWNSAMPLE_FACTOR
            original_shape = volume.shape
            volume = zoom(volume, factor, order=1)
            aprint(f"  {original_shape} -> {volume.shape}")

    return volume


# =============================================================================
# GSplats Fitting
# =============================================================================


def _fit_intensity_scale(volume: np.ndarray) -> tuple[float, float, float]:
    """Return max, normalised floor, and surviving fit range.

    Raises:
        ValueError: if the volume's maximum does not exceed the specimen floor.

    The min-max normalisation this replaced divided by ``(vmax - vmin + 1e-8)``,
    and that epsilon was load-bearing: it kept an all-zero volume from dividing
    by zero. Scaling by ``vmax`` alone reintroduces that division, so it is
    guarded here — the one place the division happens, which covers both the fit
    and the round-trip caller — and reported for what it means, since an empty
    volume at this point says the download or TIFF extraction produced nothing.
    """
    vmax = float(volume.max())
    if not vmax > SPECIMEN_BACKGROUND_COUNTS:
        raise ValueError(
            f"Tribolium volume maximum ({vmax}) does not exceed the specimen "
            f"background floor ({SPECIMEN_BACKGROUND_COUNTS}) — the download or "
            f"TIFF extraction under {CACHE_DIR} is empty or corrupt. Clear it with "
            f"`luxar demo cache clear gsplats_3d_tribolium_embryo` and re-run with "
            f"--recompute."
        )
    floor_normalised = SPECIMEN_BACKGROUND_COUNTS / vmax
    return vmax, floor_normalised, 1.0 - floor_normalised


def fit_tribolium(volume: np.ndarray) -> GSplatData:
    """Fit Gaussian splats to the Tribolium volume (no cache check — caller handles that).

    Args:
        volume: 3D float32 volume (Z, Y, X) in raw camera counts.

    Returns:
        Fitted GSplatData.
    """
    cache_file = CACHE_DIR / "tribolium.gsplats.zarr.zip"

    # Auto-detect device
    global DEVICE
    if DEVICE is None:
        from luxar.demos import detect_device

        DEVICE = detect_device()

    from luxar.gsplats import fit_gaussian_splats

    with asection(f"Fitting GSplats (fixed-K joint fit: seeds={MAX_SPLATS})"):
        aprint(f"Volume shape: {volume.shape}")
        aprint(f"Device: {DEVICE}")

        # Scale to [0, 1] HERE rather than in the loader, so that the floor and
        # the normalisation it rides on stay in one place and visibly agree.
        #
        # The fitter normalises internally, so the optimisation itself is
        # scale-free. The result is not: `fit_gaussian_splats` rescales returned
        # amplitudes into the units it was handed, so raw counts would emit
        # amplitudes larger by the expected ~15900-count volume maximum. That
        # silently breaks the scene's `scale_intensity` and every display setting
        # downstream of it.
        #
        # So the floor is converted into the same normalised space instead of
        # being hardcoded there: dividing the MEASURED count level by this
        # volume's own maximum keeps `SPECIMEN_BACKGROUND_COUNTS` a checkable
        # camera value and still tracks the data if the source ever changes.
        vmax, floor_normalised, _fit_range = _fit_intensity_scale(volume)
        volume = volume / vmax
        aprint(
            f"Normalised by max {vmax:.0f} counts; floor "
            f"{SPECIMEN_BACKGROUND_COUNTS:.0f} counts -> {floor_normalised:.6f}"
        )

        result = fit_gaussian_splats(
            volume,
            seeds=MAX_SPLATS,
            device=DEVICE,
            # Explicit specimen background rather than the default `auto`, which
            # resolves to the ~205-count DETECTOR offset here and leaves the
            # embryo's own ~675-count haze in place for the splats to fit. See
            # SPECIMEN_BACKGROUND_COUNTS for the measurement and the floor sweep.
            floor=floor_normalised,
            verbose=True,
        )

        n_splats = len(result.amplitudes)
        aprint(f"Fitted {n_splats:,} splats")

        # Cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        aprint(f"Caching to {cache_file.name}")
        save_with_lod(
            result,
            cache_file,
            recipe="stream",
            encoding_mode=EncodingMode.MEMORY,
            include_fitting_info=True,
            compress="zip",
            zip_deflate=True,
        )

    return GSplatData.load(cache_file, include_stats=False)


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(
    gsplats_data: GSplatData,
    output_path: Path | None = None,
) -> Path:
    """Create 3D Luxar scene from fitted GSplats.

    Args:
        gsplats_data: Fitted Gaussian splats.
        output_path: Output .zarr path (default: demos output dir).

    Returns:
        Path to saved scene.
    """
    if output_path is None:
        output_path = get_demos_output_dir() / "gsplats_3d_tribolium_embryo.luxar.zarr"

    with asection("Creating 3D Luxar Scene"):
        aprint(f"Output: {output_path.name}")

        # Dimensions in voxel coordinates (fitting produces voxel-space centers)
        dims = Dimensions(
            [
                Dimension("x", unit="px", display=True),
                Dimension("y", unit="px", display=True),
                Dimension("z", unit="px", display=True),
            ]
        )

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                # Neutral tone-mapping, not the viewer's default ACES (#1459):
                # what reaches the top of the range is the embryo's dense core,
                # and Neutral rolls those peaks off instead of clipping them
                # flat. Exposure sits at the 0-stop identity: the +1.97 stops
                # the pre-floor fit needed went away with the 2026-09-10
                # re-tune below, which brings the light up through the layer's
                # own opacity and colour window instead. All four numbers were
                # dialled together in the hosted viewer, so moving one needs a
                # live A/B, not a blind flip.
                viewer_config=ViewerConfig(cinematic_mode=True, tone_mapping="Neutral"),
                citation=DEMO_META["citation"],
            )

            scene.attrs["title"] = "GSplats: Tribolium castaneum Embryo (Light-Sheet)"
            scene.attrs["description"] = """
3D Gaussian Splatting — Tribolium castaneum Embryo
====================================================

A near-isotropic 3D light-sheet fluorescence microscopy volume of a
developing red flour beetle embryo, represented as Gaussian splats.

Data Source:
  - Cell Tracking Challenge / Zenodo record 5270323
  - Zeiss LightSheet Z.1
  - Volume: 965 x 1871 x 991 voxels at 0.381 um isotropic

How to Cite:
  Barry et al. (2022). GIANI. J. Cell Sci. 135, jcs259511.
  Maska et al. (2023). Cell Tracking Challenge. Nat. Methods 20, 1010-1020.

Navigation:
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
            """

            with asection("Adding GSplats"):
                # Centre at centroid and scale intensity
                gsplats_data = gsplats_data.translate(
                    -gsplats_data.centers.T
                    @ gsplats_data.amplitudes
                    / gsplats_data.amplitudes.sum()
                )
                gsplats_data = gsplats_data.scale_intensity(0.03)

                n_splats = len(gsplats_data.amplitudes)

                # Warm amber colour for the embryo fluorescence
                colors = np.tile(
                    np.array([1.0, 0.85, 0.4], dtype=np.float32), (n_splats, 1)
                )

                scene.add_gsplats(
                    name="tribolium_embryo",
                    centers=gsplats_data.centers,
                    amplitudes=gsplats_data.amplitudes,
                    cholesky_factors=gsplats_data.cholesky_factors,
                    colors=colors,
                    # `volumetric` — emission-absorption (Max 1995) rather than
                    # alpha-over: the embryo then reads as dense tissue with
                    # real front-to-back depth cueing instead of a shell of
                    # composited surface peaks. The three numbers below were
                    # re-dialled together in the hosted viewer's Layers panel
                    # against the 675-count-floor fit (2026-09-10), replacing
                    # the pre-floor baseline (kappa 3.13 / opacity 0.06 /
                    # window 0-1.085 under +1.97 stops of exposure).
                    blending_mode="volumetric",
                    # Absorption carries the depth: kappa 2.53 is well above the
                    # 1.0 identity, so the far side of the embryo attenuates
                    # visibly through the near side. (kappa=0 would render
                    # exactly like additive.)
                    absorption=2.53,
                    # Full per-splat emission. The floor-refitted store has lost
                    # the haze the old 0.06 was holding down, so the light now
                    # comes from the splats themselves rather than from a
                    # +2-stop exposure boost on a dimmed layer.
                    opacity=1.0,
                    # COLOUR RANGE 0-0.533 as the Layers panel shows it. This is
                    # a DIRECT-COLOUR node (explicit RGB, no colormap), so the
                    # window is on the authored colour and `intensity` is a
                    # plain gain: intensity = 1/(hi-lo) = 1/0.533, with offset
                    # left at the 0 identity because the window starts at zero.
                    # A window top BELOW 1.0 is a ~1.9x brightening of the
                    # amber, which is what replaces the old exposure stops.
                    intensity=1.0 / 0.533,
                    layer=True,
                )
                aprint(f"Added {n_splats:,} splats")

            # Overlay annotations
            scene.add_text(
                "Tribolium Embryo",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(scene, "Light-sheet microscopy", DEMO_META.get("citation"))

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Round-Trip Visualisation
# =============================================================================


def show_roundtrip_comparison(
    volume: np.ndarray,
    gsplats_data: GSplatData,
) -> None:
    """Show original vs round-trip reconstructed volume side by side."""
    try:
        plt = require_module("matplotlib.pyplot")
    except MissingDependencyError as exc:
        aprint(f"Skipping --show-roundtrip: {exc}")
        return

    with asection("Round-trip reconstruction comparison"):
        with asection("Rendering reconstruction"):
            recon = gsplats_data.render_to_volume(shape=volume.shape, device=DEVICE)
            reference, recon = _prepare_roundtrip_comparison(volume, recon)
            mse = float(np.mean((reference - recon) ** 2))
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
            aprint(f"  PSNR: {psnr:.2f} dB, MSE: {mse:.6g}")
            aprint(
                f"  (reference is floor-suppressed at "
                f"{SPECIMEN_BACKGROUND_COUNTS:.0f} counts, as fitted)"
            )
        volume = reference

        mid_z = volume.shape[0] // 2
        orig_slice = volume[mid_z]
        recon_slice = recon[mid_z]
        diff_slice = np.abs(orig_slice - recon_slice)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        axes[0].imshow(orig_slice, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original (floor-suppressed)")
        axes[0].axis("off")

        axes[1].imshow(recon_slice, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Reconstructed (PSNR {psnr:.1f} dB)")
        axes[1].axis("off")

        im = axes[2].imshow(diff_slice, cmap="inferno", vmin=0, vmax=0.3)
        axes[2].set_title("|Difference|")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Tribolium Embryo — Round-Trip Comparison — z-slice {mid_z}  "
            f"({len(gsplats_data.amplitudes):,} splats)",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()


def _prepare_roundtrip_comparison(
    volume: np.ndarray, recon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Put raw counts and a fitter render on one scale, rescaling recon in place."""
    vmax, floor_normalised, fit_range = _fit_intensity_scale(volume)
    reference = volume - SPECIMEN_BACKGROUND_COUNTS
    reference /= vmax * fit_range
    np.clip(reference, 0.0, 1.0, out=reference)

    # finalize_results restores the surviving range to amplitudes, so renders
    # land on (V - floor) / vmax. Divide it back out for the [0, 1] reference.
    recon /= fit_range
    return reference, recon


# =============================================================================
# Main
# =============================================================================


def warn_if_cached_tribolium_fit_predates_floor() -> None:
    """Say so when the loaded cache was fitted before the specimen floor existed.

    Without this the fix applies only to whoever happens to have a cold cache: a
    fit made earlier still carries the embryo's haze, and nothing else notices.
    ``load_precomputed_gsplats`` compares the cache against the packaged LFS
    source, which does not exist for this non-redistributable dataset, so there
    is no staleness signal to piggyback on.

    A fit records the floor it used under ``stats["floor"]`` (normalised units),
    which is read here through ``include_stats=True`` rather than by naming a
    metadata document — the store may be zarr format 2 or 3 and the document
    names differ. Note the sibling readers do NOT expose it:
    ``load_precomputed_gsplats`` hardcodes ``include_stats=False``, and
    ``inspect_gsplats_zarr`` reads only the ``fitting/`` group while the floor
    lands in ``pipeline/``. Hence the extra load; the file is ~2 MB.

    Two signals, both one-sided so a good cache is never flagged:

    * no ``floor`` recorded at all — either a pre-floor fit, or a reduction pass
      that rewrote ``pipeline/`` (which is what happened to the old shipped fit);
    * a floor far below the specimen level, i.e. the ~205-count detector offset
      ``auto`` used to resolve to. Comparing normalised values is safe in this
      direction because ``--downsample`` lowers the volume maximum and therefore
      RAISES the normalised floor — it cannot push a good fit under the bar.
    """
    cache_file = CACHE_DIR / "tribolium.gsplats.zarr.zip"
    if not cache_file.exists():
        return
    try:
        stats = GSplatData.load(cache_file, include_stats=True).stats or {}
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never break the demo
        aprint(f"(could not read the cached fit's floor: {exc})")
        return

    floor = stats.get("floor")
    # Half the expected level: comfortably clear of 675 counts, comfortably above
    # the ~205-count offset it must catch.
    suspicious = 0.5 * (SPECIMEN_BACKGROUND_COUNTS / EXPECTED_VOLUME_MAX_COUNTS)
    if floor is None or float(floor) < suspicious:
        recorded = "none recorded" if floor is None else f"{float(floor):.6f}"
        aprint(
            f"WARNING: this cached fit predates the specimen background floor "
            f"(floor: {recorded}; expected about "
            f"{SPECIMEN_BACKGROUND_COUNTS / EXPECTED_VOLUME_MAX_COUNTS:.6f}). It "
            f"still contains the "
            f"embryo's ~{SPECIMEN_BACKGROUND_COUNTS:.0f}-count autofluorescence "
            f"haze, which obscures the nuclei. Re-run with --recompute."
        )


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: 3D Tribolium castaneum Embryo (Light-Sheet)")
    aprint("=" * 70)
    aprint("Large isotropic 3D volume -> Gaussian splatting visualisation")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_3d_tribolium_embryo.luxar.zarr"

    # Serve-only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    # Try loading the local cache; a cold cache re-fits from the raw source.
    precomputed = load_precomputed_gsplats(
        "gsplats_tribolium",
        ["tribolium.gsplats.zarr.zip"],
        recompute=RECOMPUTE,
    )

    volume = None

    if precomputed is not None:
        gsplats_data = precomputed[0]
        warn_if_cached_tribolium_fit_predates_floor()
    else:
        # --recompute path: download raw data, fit from scratch
        warn_if_no_cuda_gpu()
        volume = load_tribolium_volume()
        gsplats_data = fit_tribolium(volume)

    # Optional round-trip visualisation
    if SHOW_ROUNDTRIP:
        if volume is not None:
            show_roundtrip_comparison(volume, gsplats_data)
        else:
            aprint(
                "Cannot show round-trip: original volume not available "
                "(loaded from precomputed cache). Re-run with --recompute."
            )

    # Report
    with asection("Summary"):
        aprint(f"Splats:  {len(gsplats_data.amplitudes):,}")
        aprint(f"Dim:     {gsplats_data.centers.shape[1]}D")

    # Create scene
    scene_path = create_luxar_scene(gsplats_data)

    # Launch viewer
    if not NO_SERVE:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
