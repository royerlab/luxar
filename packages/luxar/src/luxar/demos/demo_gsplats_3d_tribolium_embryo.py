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
Yin, Z. et al. (2022).  GIANI — open-source software for automated analysis
of 3D microscopy images.  *Journal of Cell Science*, 135(5), jcs259022.
DOI: 10.1242/jcs.259022

Cell Tracking Challenge — Maska, M. et al. (2023).  The Cell Tracking
Challenge: 10 years of objective benchmarking.  *Nature Methods*, 20, 1010–1020.

WORKFLOW:
=========

1. **Download** ZIP from Zenodo (2.6 GB, with resume support)
2. **Extract** multi-page TIFF stack from archive
3. **Load** as 3D volume, normalise to [0, 1], optionally downsample
4. **Fit** Gaussian splats with GPU acceleration + caching
5. **Create 3D scene** in voxel coordinates
6. **Visualise** — rotate, zoom, explore the embryo

USAGE:
======
    python demo_gsplats_3d_tribolium_embryo.py [--recompute] [--no-serve] [--serve-only] [--downsample=N]

Options:
    --recompute:      Force re-fitting from scratch (download + GPU fitting)
    --no-serve:       Generate scene without launching viewer
    --serve-only:     Just serve a previously generated scene
    --show-roundtrip: Show matplotlib comparison of original vs reconstructed volume
    --downsample=N:   Downsample factor for fitting (default: 1)

By default, precomputed GSplats are loaded from package data (Git LFS).
Use --recompute to re-fit from scratch (requires network + GPU).

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
        "gpu": "optional",
        "local_data": "git-lfs",
    },
    "caches": ["gsplats_tribolium"],
    "outputs": ["gsplats_3d_tribolium_embryo"],
}

import sys
import zipfile
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import require_module
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.demos import (
    launch_viewer,
    load_precomputed_gsplats,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
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
    from luxar.utils.download import robust_download

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
        3D float32 volume normalised to [0, 1], shape (Z, Y, X).
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

        # Normalise to float32 [0, 1]
        volume = volume.astype(np.float32)
        vmin, vmax = volume.min(), volume.max()
        volume = (volume - vmin) / (vmax - vmin + 1e-8)
        aprint(
            f"Normalised: shape={volume.shape}, "
            f"range=[{volume.min():.3f}, {volume.max():.3f}], "
            f"size={volume.nbytes / (1024**3):.1f} GB"
        )

    return volume


def load_tribolium_volume() -> np.ndarray:
    """Full pipeline: download, extract, load, downsample.

    Note: The full volume is ~7 GB as float32 (965 x 1871 x 991).
    Downsampling reduces this to ~0.9 GB at 2x.  Ensure sufficient RAM.

    Returns:
        3D float32 volume ready for GSplat fitting.
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


def fit_tribolium(volume: np.ndarray) -> GSplatData:
    """Fit Gaussian splats to the Tribolium volume (no cache check — caller handles that).

    Args:
        volume: 3D float32 volume (Z, Y, X), normalised to [0, 1].

    Returns:
        Fitted GSplatData.
    """
    cache_file = CACHE_DIR / "tribolium.gsplats.zarr.zip"

    # Auto-detect device
    global DEVICE
    if DEVICE is None:
        from luxar.utils.demos import detect_device

        DEVICE = detect_device()

    from luxar.gsplats import fit_gaussian_splats

    with asection(f"Fitting GSplats (fixed-K joint fit: seeds={MAX_SPLATS})"):
        aprint(f"Volume shape: {volume.shape}")
        aprint(f"Device: {DEVICE}")

        result = fit_gaussian_splats(
            volume,
            seeds=MAX_SPLATS,
            device=DEVICE,
            verbose=True,
        )

        n_splats = len(result.amplitudes)
        aprint(f"Fitted {n_splats:,} splats")

        # Cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        aprint(f"Caching to {cache_file.name}")
        result.save(
            cache_file,
            encoding_mode=EncodingMode.MEMORY,
            include_fitting_info=True,
            compress="zip",
            zip_deflate=True,
        )

    return result


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
                # Neutral tone-mapping (not the viewer's default ACES, which lifts
                # highlights and shifts hue) — a deliberate exception to the house
                # ACES recommendation, verified against this volume.
                # The blending mode below projects each splat's peak instead of
                # integrating along the view ray, so nothing accumulates and the
                # embryo needs ~2 stops of exposure to sit at a normal level.
                viewer_config=ViewerConfig(tone_mapping="Neutral", exposure=1.97),
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
  Yin et al. (2022). GIANI. J. Cell Sci. 135(5), jcs259022.
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
                    opacity=1.0,
                    # `normal` rather than an accumulating mode: this light-sheet
                    # volume carries a heavy diffuse background, and integrating
                    # it along every ray buries the embryo in haze. `normal`
                    # composites the projected 2D-Gaussian peak (surface
                    # density) with alpha-over instead, so the background stops
                    # summing and the surface nuclei stay crisp.
                    blending_mode="normal",
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
            scene.add_text(
                "Light-sheet microscopy",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

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
        import matplotlib.pyplot as plt
    except ImportError:
        aprint(
            "matplotlib is required for --show-roundtrip. Install with: pip install 'luxar[demos]'"
        )
        return

    with asection("Round-trip reconstruction comparison"):
        with asection("Rendering reconstruction"):
            recon = gsplats_data.render_to_volume(shape=volume.shape, device=DEVICE)
            mse = float(np.mean((volume - recon) ** 2))
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
            aprint(f"  PSNR: {psnr:.2f} dB, MSE: {mse:.6g}")

        mid_z = volume.shape[0] // 2
        orig_slice = volume[mid_z]
        recon_slice = recon[mid_z]
        diff_slice = np.abs(orig_slice - recon_slice)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        axes[0].imshow(orig_slice, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original")
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


# =============================================================================
# Main
# =============================================================================


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

    # Try loading precomputed data (from Git LFS / local cache)
    precomputed = load_precomputed_gsplats(
        "gsplats_tribolium",
        ["tribolium.gsplats.zarr.zip"],
        recompute=RECOMPUTE,
    )

    volume = None

    if precomputed is not None:
        gsplats_data = precomputed[0]
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
