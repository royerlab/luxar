#!/usr/bin/env python3
"""GSplats Demo: 4D Zebrafish Embryo Time-Lapse (Zenodo)

Visualises a 4D (3D + time) confocal laser-scanning microscopy recording of a
living zebrafish embryo during gastrulation, using Gaussian splatting with a
time dimension slider.

================================================================================
4D TIME-LAPSE — ZEBRAFISH GASTRULATION
================================================================================

The dataset is a time-lapse volumetric fluorescence microscopy image sequence
of a living zebrafish embryo (cxcr4aMO).  Endodermal cells are fluorescently
labelled and imaged during gastrulation — the critical stage when the embryo
reorganises from a ball of cells into a layered body plan.

Each time-point is fitted independently as 3D Gaussian splats, then embedded
into a 4D scene using ``dim_order`` + ``fill`` to assign each fit to its
corresponding time coordinate.

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source:  Zenodo record 1211599
URL:    https://zenodo.org/records/1211599
File:   cxcr4aMO2_290112.lsm (Zeiss LSM format)
Size:   ~2.1 GB
DOI:    10.5281/zenodo.1211599

Imaging:
--------
Microscope:  Confocal laser-scanning microscope (Zeiss LSM)
Organism:    Danio rerio (zebrafish), cxcr4aMO morphant
Label:       Fluorescent endoderm label
Stage:       Gastrulation

How to Cite:
------------
If you use this dataset, please cite the original Zenodo record:
DOI: 10.5281/zenodo.1211599

WORKFLOW:
=========

1. **Download** LSM file from Zenodo (2.1 GB, with resume support)
2. **Load** with tifffile (native LSM support) -> (T, [C,] Z, Y, X)
3. **Fit** each time-point as 3D GSplats (with per-timepoint caching)
4. **Create 4D scene** [X, Y, Z, Time] using dim_order + fill
5. **Visualise** — scrub through time with the Time slider

USAGE:
======
    python demo_gsplats_4d_zebrafish_timelapse.py [--recompute] [--no-serve] [--serve-only] [--max-timepoints=N]

Options:
    --recompute:         Force re-fitting from scratch (ignore precomputed/cached results)
    --no-serve:          Generate scene without launching viewer
    --serve-only:        Just serve a previously generated scene
    --show-roundtrip:    Show matplotlib comparison of original vs reconstructed volumes
    --max-timepoints=N:  Max number of timepoints to process (default: 64)
    --downsample-xy=N:   XY downsample factor (default: 2)

Output:
    - Scene saved to:  datasets/demos/gsplats_4d_zebrafish_timelapse.luxar.zarr
    - Automatically opens in browser
"""

DEMO_META = {
    "key": "gsplats_4d_zebrafish_timelapse",
    "title": "4D Zebrafish Timelapse",
    "description": "A 4D confocal timelapse of zebrafish gastrulation as per-timepoint Gaussian splats.",
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 2,
        "compute": "medium",
        "gpu": "optional",
        "local_data": "git-lfs",
    },
    "caches": ["gsplats_zebrafish"],
    "outputs": ["gsplats_4d_zebrafish_timelapse"],
}

import sys
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.demos import (
    launch_viewer,
    load_precomputed_bundle,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Data source
ZENODO_URL = "https://zenodo.org/api/records/1211599/files/cxcr4aMO2_290112.lsm/content"

# Fit parameters (fixed-K, seeds=K*)
MAX_SPLATS = 2000

# Cache location
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_zebrafish"

# Precomputed data configuration
_PRECOMPUTED_DEMO_NAME = "gsplats_zebrafish"
_PRECOMPUTED_BUNDLE_NAME = "zebrafish.gsplats.zarr.zip"
# Precomputed bundle contains 64 frames at step 2: 0, 2, 4, ..., 126
_PRECOMPUTED_FRAME_INDICES = list(range(0, 128, 2))  # 64 frames
_PRECOMPUTED_VOXEL_SIZE_ZYX = (3.99, 0.91, 0.91)  # µm

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
SHOW_ROUNDTRIP = "--show-roundtrip" in sys.argv

MAX_TIMEPOINTS = 64
DOWNSAMPLE_XY = 2

for _arg in sys.argv:
    if _arg.startswith("--max-timepoints="):
        MAX_TIMEPOINTS = int(_arg.split("=")[1])
    elif _arg.startswith("--downsample-xy="):
        DOWNSAMPLE_XY = int(_arg.split("=")[1])

# Arbol logging depth
Arbol.max_depth = 5

# Auto-detected on first fit
DEVICE = None


# =============================================================================
# Data Loading
# =============================================================================


def download_zebrafish_data() -> Path:
    """Download the zebrafish LSM from Zenodo (~2.1 GB).

    Returns:
        Path to downloaded LSM file.
    """
    from luxar.utils.download import robust_download

    lsm_path = CACHE_DIR / "cxcr4aMO2_290112.lsm"

    with asection("Downloading zebrafish embryo dataset from Zenodo"):
        aprint("Source: https://zenodo.org/records/1211599")
        aprint("Size:   ~1.9 GB")

        robust_download(
            ZENODO_URL,
            lsm_path,
            max_retries=5,
            timeout=600,
            expected_size=2_080_484_264,
        )

    return lsm_path


def load_zebrafish_volumes() -> tuple:
    """Download and load the zebrafish LSM as a list of 3D volumes.

    Returns:
        Tuple of (volumes, voxel_size_zyx, time_indices):
        - volumes: List of 3D float32 volumes, one per timepoint, each normalised to [0, 1].
        - voxel_size_zyx: Tuple of (Z, Y, X) voxel spacing in micrometres, or None.
        - time_indices: List of source frame indices used (for cache key stability).
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError(
            "tifffile is required for this demo.\nInstall with: pip install tifffile"
        )

    lsm_path = download_zebrafish_data()

    with asection("Loading zebrafish LSM"):
        # Extract voxel spacing from LSM metadata
        voxel_size_zyx = None
        try:
            with tifffile.TiffFile(str(lsm_path)) as tif:
                if hasattr(tif, "lsm_metadata") and tif.lsm_metadata:
                    meta = tif.lsm_metadata
                    vz = meta.get("VoxelSizeZ", 0) * 1e6  # m → µm
                    vy = meta.get("VoxelSizeY", 0) * 1e6
                    vx = meta.get("VoxelSizeX", 0) * 1e6
                    if vz > 0 and vy > 0 and vx > 0:
                        voxel_size_zyx = (vz, vy, vx)
                        aprint(
                            f"LSM voxel spacing (Z,Y,X): "
                            f"({vz:.4f}, {vy:.4f}, {vx:.4f}) µm"
                        )
        except Exception as e:
            aprint(f"Could not extract voxel spacing from LSM: {e}")

        raw = tifffile.imread(str(lsm_path))
        aprint(f"LSM shape: {raw.shape}, dtype: {raw.dtype}")

        # tifffile returns LSM as (T, C, Z, Y, X) or (T, Z, Y, X) or other combos
        # Handle various shapes:
        if raw.ndim == 5:
            # (T, C, Z, Y, X) — select channel 0
            aprint(
                f"  5D detected: (T={raw.shape[0]}, C={raw.shape[1]}, "
                f"Z={raw.shape[2]}, Y={raw.shape[3]}, X={raw.shape[4]})"
            )
            aprint("  Using channel 0")
            raw = raw[:, 0, :, :, :]
        elif raw.ndim == 4:
            # (T, Z, Y, X) — already good
            aprint(
                f"  4D detected: (T={raw.shape[0]}, Z={raw.shape[1]}, "
                f"Y={raw.shape[2]}, X={raw.shape[3]})"
            )
        elif raw.ndim == 3:
            # Single volume — treat as single timepoint
            aprint(f"  3D detected: single volume {raw.shape}")
            raw = raw[np.newaxis, ...]
        else:
            raise ValueError(f"Unexpected LSM shape: {raw.shape} ({raw.ndim}D)")

        n_total = raw.shape[0]
        n_use = min(n_total, MAX_TIMEPOINTS)

        # Sample timepoints evenly across the full recording so we capture
        # the developmental progression, not just the first few frames.
        stride = max(1, n_total // n_use)
        time_indices = list(range(0, n_total, stride))[:n_use]
        aprint(
            f"Using {len(time_indices)} of {n_total} timepoints "
            f"(stride={stride}, indices={time_indices[0]}..{time_indices[-1]})"
        )

        volumes = []
        for t in time_indices:
            V = raw[t].astype(np.float32)
            # Normalise to [0, 1]
            vmin, vmax = V.min(), V.max()
            V = (V - vmin) / (vmax - vmin + 1e-8)

            if DOWNSAMPLE_XY > 1:
                from scipy.ndimage import zoom

                factors = (1.0, 1.0 / DOWNSAMPLE_XY, 1.0 / DOWNSAMPLE_XY)
                V = zoom(V, factors, order=1)

            volumes.append(V)

        # Adjust voxel spacing for XY downsampling
        if voxel_size_zyx is not None and DOWNSAMPLE_XY > 1:
            vz, vy, vx = voxel_size_zyx
            voxel_size_zyx = (vz, vy * DOWNSAMPLE_XY, vx * DOWNSAMPLE_XY)
            aprint(
                f"Adjusted voxel spacing for {DOWNSAMPLE_XY}x XY downsample: "
                f"({voxel_size_zyx[0]:.4f}, {voxel_size_zyx[1]:.4f}, {voxel_size_zyx[2]:.4f}) µm"
            )

        aprint(f"Loaded {len(volumes)} volumes, shape per volume: {volumes[0].shape}")

    return volumes, voxel_size_zyx, time_indices


# =============================================================================
# Colour Utilities
# =============================================================================


def _time_color(t_frac: float) -> tuple:
    """Map a normalised time fraction [0, 1] to an RGB colour.

    Produces a visually appealing cyan -> green -> yellow -> orange progression.
    """
    # Simple three-stop gradient: cyan -> green -> warm yellow
    if t_frac < 0.5:
        f = t_frac * 2.0
        r = 0.0 + f * 0.3
        g = 0.8 + f * 0.2
        b = 1.0 - f * 0.7
    else:
        f = (t_frac - 0.5) * 2.0
        r = 0.3 + f * 0.7
        g = 1.0 - f * 0.2
        b = 0.3 - f * 0.2
    return (max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b)))


# =============================================================================
# GSplats Fitting
# =============================================================================


def fit_timepoint(
    volume: np.ndarray,
    label: str,
    cache_file: Path,
    voxel_size=None,
) -> GSplatData:
    """Fit GSplats to a single timepoint volume with caching.

    Args:
        volume: 3D float32 volume (Z, Y, X), normalised to [0, 1].
        label: Human-readable label for logging.
        cache_file: Path to .gsplats.zarr.zip cache file.
        voxel_size: Optional tuple of (Z, Y, X) voxel spacing.

    Returns:
        Fitted GSplatData.
    """
    # Check cache
    if cache_file.exists() and not RECOMPUTE:
        try:
            result = GSplatData.load(cache_file, include_stats=False)
            aprint(f"  Loaded {len(result.amplitudes):,} cached splats ({label})")
            return result
        except Exception as e:
            aprint(f"  Cache load failed: {e}, re-fitting...")

    # Auto-detect device
    global DEVICE
    if DEVICE is None:
        from luxar.utils.demos import detect_device

        DEVICE = detect_device()

    from luxar.gsplats import fit_gaussian_splats

    aprint(f"Fitting {label} (fixed-K joint fit: seeds={MAX_SPLATS})...")

    result = fit_gaussian_splats(
        volume,
        seeds=MAX_SPLATS,
        device=DEVICE,
        verbose=True,
        voxel_size=voxel_size,
    )

    aprint(f"  Fitted {len(result.amplitudes):,} splats")

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.save(
        cache_file,
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )

    return result


def fit_all_timepoints(volumes: list, voxel_size=None, time_indices=None) -> list:
    """Fit GSplats to every timepoint.

    Args:
        volumes: List of 3D float32 volumes.
        voxel_size: Optional tuple of (Z, Y, X) voxel spacing.
        time_indices: Source frame indices (for cache key stability).
            If None, uses sequential indices 0, 1, 2, ...

    Returns:
        List of GSplatData, one per timepoint.
    """
    if time_indices is None:
        time_indices = list(range(len(volumes)))

    with asection(f"Fitting GSplats ({len(volumes)} timepoints)"):
        gsplats_list = []
        for t, (volume, src_idx) in enumerate(zip(volumes, time_indices)):
            # Use source frame index in cache filename so changing
            # MAX_TIMEPOINTS doesn't cause stale cache hits.
            cache_file = CACHE_DIR / f"zebrafish_frame{src_idx:04d}.gsplats.zarr.zip"
            with asection(f"Timepoint {t}/{len(volumes) - 1} (frame {src_idx})"):
                gsplats = fit_timepoint(
                    volume,
                    f"T={t} (frame {src_idx})",
                    cache_file,
                    voxel_size=voxel_size,
                )
                gsplats_list.append(gsplats)
        return gsplats_list


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(
    gsplats_list: list,
    output_path: Path | None = None,
) -> Path:
    """Create 4D Luxar scene with Time dimension.

    Each timepoint's 3D GSplats are embedded into the 4D scene using
    ``dim_order=["z", "y", "x"]`` with ``fill={"time": t}``.

    Args:
        gsplats_list: List of GSplatData, one per timepoint.
        output_path: Output .zarr path.

    Returns:
        Path to saved scene.
    """
    if output_path is None:
        output_path = (
            get_demos_output_dir() / "gsplats_4d_zebrafish_timelapse.luxar.zarr"
        )

    n_timepoints = len(gsplats_list)

    if n_timepoints < 2:
        raise ValueError(
            f"Need at least 2 timepoints for a 4D scene, got {n_timepoints}. "
            f"Use --max-timepoints=N with N >= 2."
        )

    with asection("Creating 4D Luxar Scene"):
        aprint(f"Output: {output_path.name}")
        aprint(f"Timepoints: {n_timepoints}")

        # When voxel_size is provided to the fitting function, output centers
        # are in physical coordinates (µm) by default (output_space="real").
        dims = Dimensions(
            [
                Dimension("x", unit="um", display=True),
                Dimension("y", unit="um", display=True),
                Dimension("z", unit="um", display=True),
                Dimension(
                    "time",
                    unit="frame",
                    display=False,
                    discrete=True,
                    range=(0, n_timepoints - 1),
                    step=1.0,
                ),
            ]
        )

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
            )

            scene.attrs["title"] = "GSplats: Zebrafish Embryo 4D Time-Lapse (Confocal)"
            scene.attrs["description"] = f"""
4D Gaussian Splatting — Zebrafish Embryo Gastrulation
======================================================

A time-lapse (4D) volumetric fluorescence microscopy recording of a
living zebrafish embryo during gastrulation.  Fluorescently labelled
endodermal cells are captured with confocal laser-scanning microscopy.

Each of the {n_timepoints} time-points is independently fitted as 3D
Gaussian splats and embedded into this 4D scene.

Data Source:
  - Zenodo record 1211599
  - DOI: 10.5281/zenodo.1211599
  - Organism: Danio rerio (zebrafish), cxcr4aMO morphant

Navigation:
  - Use the Time slider to scrub through development
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
  - Colours progress from cyan (early) to warm yellow (late)
            """

            # Compute shared centroid across ALL timepoints for alignment
            with asection("Computing shared centroid"):
                all_centers = [g.centers for g in gsplats_list]
                all_amps = [g.amplitudes for g in gsplats_list]
                total_amp = sum(a.sum() for a in all_amps)
                if total_amp > 0:
                    shared_centroid = (
                        sum(c.T @ a for c, a in zip(all_centers, all_amps)) / total_amp
                    )
                else:
                    shared_centroid = np.mean(
                        np.concatenate(all_centers, axis=0), axis=0
                    )
                aprint(f"Shared centroid: {shared_centroid}")

            # Add each timepoint
            for t, gsplats in enumerate(gsplats_list):
                with asection(f"Adding timepoint {t}"):
                    gsplats = gsplats.translate(-shared_centroid)

                    # Normalise per-timepoint so max amplitude = 0.1.
                    # Without this, early frames (very sparse signal) are
                    # invisible while late frames dominate.
                    amp_max = gsplats.amplitudes.max()
                    if amp_max > 0:
                        gsplats = gsplats.scale_intensity(0.1 / amp_max)
                    else:
                        gsplats = gsplats.scale_intensity(0.1)

                    n_splats = len(gsplats.amplitudes)

                    # Time-based colour
                    t_frac = t / max(n_timepoints - 1, 1)
                    color = _time_color(t_frac)
                    colors = np.tile(np.array(color, dtype=np.float32), (n_splats, 1))

                    scene.add_gsplats(
                        name=f"gsplats_t{t:04d}",
                        centers=gsplats.centers,
                        amplitudes=gsplats.amplitudes,
                        cholesky_factors=gsplats.cholesky_factors,
                        colors=colors,
                        dim_order=["z", "y", "x"],
                        fill={"time": float(t)},
                        fill_sigma={"time": 0},
                        extend_to_all=[],
                        opacity=1.0,
                        absorption=1.0,
                        blending_mode="volumetric",
                        layer=True,
                    )
                    aprint(f"  Added {n_splats:,} splats at time={t}")

            # --- Overlays ---
            scene.add_text(
                "Zebrafish Gastrulation Timelapse",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Per-timepoint labels
            for t in range(n_timepoints):
                scene.add_text(
                    f"t = {t} / {n_timepoints - 1}",
                    position=(0.02, 0.97),
                    font_size=0.015,
                    anchor="bottom-left",
                    color="#ffcc44",
                    visible_range={"time": t},
                    transition="fade",
                    transition_duration=0.15,
                )

            scene.add_text(
                f"{n_timepoints} timepoints \u2022 Confocal laser-scanning microscopy",
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

# Show first, middle, and last timepoints in the round-trip comparison
_ROUNDTRIP_SAMPLE_COUNT = 3


def show_roundtrip_comparison(
    volumes: list[np.ndarray],
    gsplats_list: list[GSplatData],
) -> None:
    """Show original vs round-trip reconstructed volumes for sample timepoints."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        aprint(
            "matplotlib is required for --show-roundtrip. Install with: pip install matplotlib"
        )
        return

    n_total = len(volumes)
    # Pick first, middle, last
    if n_total <= _ROUNDTRIP_SAMPLE_COUNT:
        sample_indices = list(range(n_total))
    else:
        sample_indices = [0, n_total // 2, n_total - 1]
    n_show = len(sample_indices)

    with asection(
        f"Round-trip reconstruction comparison ({n_show} of {n_total} timepoints)"
    ):
        reconstructions = []
        for t in sample_indices:
            with asection(f"Rendering timepoint {t}"):
                recon = gsplats_list[t].render_to_volume(
                    shape=volumes[t].shape, device=DEVICE
                )
                reconstructions.append(recon)
                mse = float(np.mean((volumes[t] - recon) ** 2))
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
                aprint(f"  T={t}: PSNR: {psnr:.2f} dB, MSE: {mse:.6g}")

        fig, axes = plt.subplots(n_show, 3, figsize=(14, 4.5 * n_show), squeeze=False)

        for row, (t, recon) in enumerate(zip(sample_indices, reconstructions)):
            volume = volumes[t]
            mid_z = volume.shape[0] // 2
            orig_slice = volume[mid_z]
            recon_slice = recon[mid_z]
            diff_slice = np.abs(orig_slice - recon_slice)

            mse = float(np.mean((volume - recon) ** 2))
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

            axes[row, 0].imshow(orig_slice, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(f"Original — T={t}")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(recon_slice, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title(f"Reconstructed (PSNR {psnr:.1f} dB)")
            axes[row, 1].axis("off")

            im = axes[row, 2].imshow(diff_slice, cmap="inferno", vmin=0, vmax=0.3)
            axes[row, 2].set_title("|Difference|")
            axes[row, 2].axis("off")
            fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Zebrafish Time-Lapse — Round-Trip Comparison — z-slice {mid_z}  "
            f"({sum(len(g.amplitudes) for g in gsplats_list):,} total splats)",
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
    aprint("GSplats Demo: 4D Zebrafish Embryo Time-Lapse (Confocal)")
    aprint("=" * 70)
    aprint("Per-timepoint 3D fitting -> 4D scene with dim_order + fill")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_4d_zebrafish_timelapse.luxar.zarr"

    # Serve-only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    # Try loading precomputed data from Git LFS bundle
    # The bundle contains 64 frames (indices 0, 2, 4, ..., 126)
    # If MAX_TIMEPOINTS < 64, subsample from the precomputed set
    precomputed_indices = _PRECOMPUTED_FRAME_INDICES
    if MAX_TIMEPOINTS < len(precomputed_indices):
        # Subsample evenly from the precomputed set
        stride = max(1, len(precomputed_indices) // MAX_TIMEPOINTS)
        precomputed_indices = precomputed_indices[::stride][:MAX_TIMEPOINTS]

    precomputed_file_names = [
        f"zebrafish_frame{idx:04d}.gsplats.zarr.zip" for idx in precomputed_indices
    ]

    gsplats_list = load_precomputed_bundle(
        _PRECOMPUTED_DEMO_NAME,
        _PRECOMPUTED_BUNDLE_NAME,
        precomputed_file_names,
        recompute=RECOMPUTE,
    )

    volumes = None

    if gsplats_list is None:
        # Recompute path: warn about GPU requirements, load data, fit
        warn_if_no_cuda_gpu()

        # Load data — per-timepoint cache checks happen inside fit_all_timepoints().
        # We can't skip the load here because the time_indices (which frames to use)
        # depend on the stride computed from the data's total frame count.
        volumes, voxel_size_zyx, time_indices = load_zebrafish_volumes()

        # Fit GSplats per timepoint (with per-frame caching)
        gsplats_list = fit_all_timepoints(
            volumes, voxel_size=voxel_size_zyx, time_indices=time_indices
        )

    # Optional round-trip visualisation
    if SHOW_ROUNDTRIP:
        if volumes is not None:
            show_roundtrip_comparison(volumes, gsplats_list)
        else:
            aprint(
                "Cannot show round-trip: original volumes not available "
                "(loaded from precomputed cache). Re-run with --recompute."
            )

    # Report
    with asection("Fitting Summary"):
        total_splats = sum(len(g.amplitudes) for g in gsplats_list)
        aprint(f"Total splats: {total_splats:,} across {len(gsplats_list)} timepoints")
        for t, g in enumerate(gsplats_list):
            aprint(f"  T={t}: {len(g.amplitudes):,} splats")

    # Create 4D scene
    scene_path = create_luxar_scene(gsplats_list)

    # Launch viewer
    if not NO_SERVE:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
