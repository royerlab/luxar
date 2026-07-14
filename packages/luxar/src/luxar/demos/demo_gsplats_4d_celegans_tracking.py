#!/usr/bin/env python3
"""GSplats Demo: 4D C. elegans Nuclei Tracking with Lines (Zenodo)

Visualises a 4D (3D + time) confocal microscopy time-series of a developing
C. elegans embryo using Gaussian splatting for the volumes AND polylines for
the tracked cell nuclei trajectories.

================================================================================
4D TRACKING — VOLUME RENDERING + CELL LINEAGE TRACKS
================================================================================

The dataset contains fully tracked confocal time-series of C. elegans embryos,
with nuclei segmented and tracked by StarryNite + manual curation.  The demo
combines:

  1. **GSplats** — 3D Gaussian splats fitted per timepoint for volume rendering
  2. **Lines**  — 4D polylines showing each cell's trajectory through time

Each cell track is a polyline in 4D (X, Y, Z, Time) space, coloured by
lineage.  The tracks use ``extend_to_all=["time"]`` so the full trajectory
context is always visible regardless of the current time-slider position.

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source:  Zenodo record 6460303
URL:    https://zenodo.org/records/6460303
File:   mskcc_confocal.zip (26.1 GB)
DOI:    10.5281/zenodo.6460303

Imaging:
--------
Microscope:  Zeiss Axio Observer.Z1
Organism:    C. elegans embryo
Resolution:  0.75 x 0.15 x 0.15 um (ZYX)
Volume:      41 x 512 x 512 per timepoint
Temporal:    75s interval, 400 timepoints
Tracking:    StarryNite + manual curation (full lineage)

How to Cite:
------------
Hirsch, P. et al. (2022).  3D+time nuclei tracking dataset of confocal
fluorescence microscopy time series of C. elegans embryos.
DOI: 10.5281/zenodo.6460303

WORKFLOW:
=========

1. **Download** ZIP from Zenodo (26 GB — large, with resume support)
2. **Extract** sample s1 data (TIFF images + tracking CSV)
3. **Parse** tracking CSV for nuclei positions and lineage IDs
4. **Preprocess** each volume: N2S-NLM denoise (3D) + CLAHE (cached per-timepoint)
5. **Fit** GSplats per timepoint on preprocessed volumes (with per-timepoint caching)
6. **Create 4D scene** [X, Y, Z, Time]
7. **Add GSplats** per timepoint using dim_order + fill
8. **Add Lines** for cell tracks as 4D polylines
9. **Visualise** — scrub through time, see tracks + volumes

USAGE:
======
    python demo_gsplats_4d_celegans_tracking.py [options]

Options:
    --recompute:      Force re-processing from scratch (download + preprocess + GPU fitting)
    --no-serve:       Generate scene without launching viewer
    --serve-only:     Just serve a previously generated scene
    --show-roundtrip: Show matplotlib comparison of original vs reconstructed volumes
    --timepoints=N:   Number of timepoints to process (default: 400, max: 400)
    --sample=N:       Which sample to use: 1, 2, or 3 (default: 1)

By default, precomputed per-timepoint GSplats are loaded from package data (Git LFS).
Tracking lines require a Zenodo download even in default mode.
Use --recompute to re-fit from scratch (requires network + CUDA GPU).

Requirements:
    - CUDA GPU strongly recommended (fitting on CPU can be orders of magnitude slower, depending on hardware)
    - PyTorch for NLM denoising (included in luxar[gsplats])

Output:
    - Scene saved to:  datasets/demos/gsplats_4d_celegans_tracking.luxar.zarr
    - Automatically opens in browser
"""

import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.gsplats.clahe import apply_clahe
from luxar.gsplats import fit_gaussian_splats
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
ZENODO_URL = "https://zenodo.org/api/records/6460303/files/mskcc_confocal.zip/content"

# Volume specs
VOXEL_SIZE_ZYX = (0.75, 0.15, 0.15)  # Micrometres
IMAGE_SHAPE = (41, 512, 512)  # Z, Y, X per timepoint

# Progressive fitting parameters
# Note: peak GPU memory scales with accumulated splats * volume_size.
# For 41x512x512 volumes on a 24 GB GPU, ~4000 accumulated splats is the
# safe ceiling (~18 GB peak during quality-evaluation rendering).
MAX_SPLATS = 22000  # Total splats per timepoint
MAX_SPLATS_PER_PASS = 4500  # Splats added per progressive pass
ITERS_PER_PASS = 3000  # Iterations per pass
PSNR_PATIENCE = 0.3  # Stop if ΔPSNR < this (dB) — tighter to save a pass

# Cache location
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_celegans"

# Fading trail visualisation (per-timepoint rolling trail of the last
# TRAIL_HISTORY segments leading up to the current time, each dimmed by
# TRAIL_FADE[age] where age 0 is the newest segment).
TRAIL_HISTORY = 80  # frames of history per comet tail (~100 min at 75 s/frame)
# Brightness fade per segment age (index 0 = newest/brightest). Length must
# equal TRAIL_HISTORY. Linear ramp from 1.0 down to 0.1 gives a smooth
# photographic long-exposure look over the TRAIL_HISTORY-frame window.
TRAIL_FADE = tuple(round(v, 3) for v in np.linspace(1.0, 0.1, TRAIL_HISTORY))
TRAIL_LINE_WIDTH = 0.083  # µm — very thin (≈ 1/3 of a cell-displacement)
# so that consecutive segments flow together as a line (at wider widths,
# individual segments render as short rectangles that look like radial spikes
# rather than a coherent tail)
TRAIL_OPACITY = 0.2  # alpha for fading trails so GSplats underneath show through
CURRENT_POINT_RADIUS = 0.3  # µm — small marker (well under ~3–5 µm nucleus
# diameter) so current-position points don't obscure the GSplat cells
# Moving-average window (in timepoints) used to smooth track positions
# before drawing trails. StarryNite positions are voxel-quantised (0.15 µm
# lateral, 0.75 µm axial); without smoothing, per-frame jitter of ±1 voxel
# dominates the real per-frame biological motion (~0.3-1 µm), producing
# cross/X-shaped artefacts instead of smooth comet tails. A 5-frame
# window (~6 minutes) preserves true trajectories while denoising
# discretisation. Set to 1 to disable smoothing.
TRAIL_SMOOTH_WINDOW = 5  # denoises voxel-quantisation jitter without eating
# real motion signal. A ~5-frame window (~6 min) is sized to the noise
# timescale, not to TRAIL_HISTORY — heavier smoothing over long histories
# smears out genuine biological displacement and shortens visible tails.
# Trail-segment stride: since we smooth positions over TRAIL_SMOOTH_WINDOW
# frames, consecutive per-frame segments are nearly collinear. Striding by
# ~half the smoothing window keeps each segment an independent motion step
# and halves the line count with no visible loss. Must be >= 1.
TRAIL_SEGMENT_STRIDE = max(1, TRAIL_SMOOTH_WINDOW // 2)
# Reject implausibly long per-frame displacements. C. elegans nuclei move
# roughly 1–5 µm/min during morphogenesis (much less in early cleavage). At
# 75 s per frame, real jumps are well under ~5 µm; a 5 µm threshold rejects
# mis-linked segments caused by the StarryNite parser falling back to
# per-file indices for unnamed nuclei (these "cells" teleport across the
# embryo between timepoints and produce visual starbursts).
TRAIL_MAX_DISPLACEMENT_UM = 5.0

# Preprocessing parameters (CLAHE + Noise2Self-calibrated NLM)
PREPROCESS_CLAHE_TILE = 16
PREPROCESS_CLAHE_CLIP = 2.0
PREPROCESS_NLM_PATCH_SIZE = 5
PREPROCESS_NLM_PATCH_DISTANCE = 7

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
SHOW_ROUNDTRIP = "--show-roundtrip" in sys.argv

DEFAULT_TIMEPOINTS = 400
MAX_TIMEPOINTS = 400
TIMEPOINTS = DEFAULT_TIMEPOINTS
SAMPLE_INDEX = 1

for _arg in sys.argv:
    if _arg.startswith("--timepoints="):
        TIMEPOINTS = min(int(_arg.split("=")[1]), MAX_TIMEPOINTS)
    elif _arg.startswith("--sample="):
        SAMPLE_INDEX = int(_arg.split("=")[1])

SAMPLE_NAME = f"mskcc_confocal_s{SAMPLE_INDEX}"

# Arbol logging depth
Arbol.max_depth = 5

# Auto-detected on first fit
DEVICE = None


# =============================================================================
# Colour Utilities
# =============================================================================


def _hue_to_rgb(hue: float) -> tuple:
    """Convert a hue value [0, 1] to an RGB tuple using HSV with S=0.85, V=0.95."""
    h = hue * 6.0
    c = 0.95 * 0.85  # V * S
    x = c * (1.0 - abs(h % 2 - 1.0))
    m = 0.95 - c

    if h < 1:
        r, g, b = c, x, 0
    elif h < 2:
        r, g, b = x, c, 0
    elif h < 3:
        r, g, b = 0, c, x
    elif h < 4:
        r, g, b = 0, x, c
    elif h < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return (r + m, g + m, b + m)


# =============================================================================
# Data Loading
# =============================================================================


def download_celegans_data() -> Path:
    """Download the C. elegans confocal ZIP from Zenodo (~26 GB).

    Returns:
        Path to downloaded ZIP file.
    """
    from luxar.utils.download import robust_download

    zip_path = CACHE_DIR / "mskcc_confocal.zip"

    with asection("Downloading C. elegans dataset from Zenodo"):
        aprint("Source:  https://zenodo.org/records/6460303")
        aprint("Size:    ~24 GB (this may take 30-60 minutes)")
        aprint("Sample:  " + SAMPLE_NAME)

        robust_download(
            ZENODO_URL,
            zip_path,
            max_retries=5,
            timeout=600,
            expected_size=26_141_247_410,
        )

    return zip_path


def extract_sample_data(zip_path: Path) -> tuple:
    """Extract only the needed sample from the ZIP (lazy extraction).

    Only extracts TIFF images and tracking files for the selected sample,
    avoiding full extraction of the 26 GB archive.

    Args:
        zip_path: Path to the downloaded ZIP archive.

    Returns:
        (tiff_files, csv_files, nuclei_dirs): Sorted TIFF paths, CSV tracking
        file paths, and directories containing StarryNite nuclei files.
    """
    extract_dir = CACHE_DIR / "extracted"

    with asection(f"Extracting sample {SAMPLE_NAME}"):
        with zipfile.ZipFile(zip_path, "r") as zf:
            # List all members belonging to our sample
            all_members = zf.namelist()
            sample_members = [m for m in all_members if SAMPLE_NAME in m]

            if not sample_members:
                available = {m.split("/")[0] for m in all_members if "/" in m}
                raise FileNotFoundError(
                    f"Sample {SAMPLE_NAME} not found in ZIP. "
                    f"Available: {sorted(available)}"
                )

            aprint(f"Found {len(sample_members)} files for {SAMPLE_NAME}")

            # Extract only files we haven't already extracted
            extracted_count = 0
            for member in sample_members:
                target = extract_dir / member
                if not target.exists():
                    zf.extract(member, extract_dir)
                    extracted_count += 1

            if extracted_count > 0:
                aprint(f"  Extracted {extracted_count} new files")
            else:
                aprint("  Using cached extraction")

        # Find TIFF images and tracking files
        # The sample directory might be at different nesting levels depending
        # on the ZIP structure, so search for it:
        sample_dir = extract_dir / SAMPLE_NAME
        if not sample_dir.is_dir():
            # Search recursively for the sample directory
            candidates = list(extract_dir.rglob(SAMPLE_NAME))
            candidates = [c for c in candidates if c.is_dir()]
            if candidates:
                sample_dir = candidates[0]
            else:
                raise FileNotFoundError(
                    f"Sample directory {SAMPLE_NAME} not found after extraction"
                )
        aprint(f"  Sample dir: {sample_dir}")

        # TIFF images — may be in root or a subdirectory
        tiff_dirs = [sample_dir]
        for subdir in sample_dir.iterdir():
            if subdir.is_dir():
                tiff_dirs.append(subdir)

        tiff_files = []
        for d in tiff_dirs:
            tiff_files.extend(
                f
                for f in d.iterdir()
                if f.is_file() and f.suffix.lower() in (".tif", ".tiff")
            )

        # Remove duplicates and sort
        tiff_files = sorted(set(tiff_files))

        # Tracking files — look for CSV first, then StarryNite nuclei files
        csv_files = sorted(sample_dir.rglob("*.csv"))
        nuclei_dirs = sorted(d for d in sample_dir.rglob("nuclei") if d.is_dir())

        aprint(f"  TIFF images: {len(tiff_files)}")
        aprint(f"  CSV files: {len(csv_files)}")
        aprint(f"  Nuclei dirs: {len(nuclei_dirs)}")

    return tiff_files, csv_files, nuclei_dirs


def load_timepoint_volume(tiff_path: Path) -> np.ndarray:
    """Load a single timepoint as a 3D volume.

    Args:
        tiff_path: Path to TIFF file.

    Returns:
        3D float32 volume normalised to [0, 1].
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError(
            "tifffile is required for this demo.\nInstall with: pip install tifffile"
        )

    volume = tifffile.imread(str(tiff_path)).astype(np.float32)

    # Handle potential extra dimensions
    if volume.ndim == 4 and volume.shape[0] <= 4:
        volume = volume[0]  # channel dimension
    elif volume.ndim == 4 and volume.shape[-1] <= 4:
        volume = volume[..., 0]

    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume from {tiff_path.name}, "
            f"got {volume.ndim}D with shape {volume.shape}"
        )

    # Robust normalisation: subtract camera background and clip hot pixels.
    # This confocal data has a high dark-current offset (~2070 counts) that
    # dominates the signal.  A single saturated pixel (65535 at t=100, t=130)
    # would compress the real signal into <2% of [0,1] under naive min-max.
    # P1 floor removes camera dark current; P99.999 ceiling clips only ~108
    # hot/dead pixels per volume while preserving >99.9% of nuclei signal.
    # (P99.9 is too aggressive — it saturates the top 10% of nuclei peaks.)
    vmin = np.percentile(volume, 1.0)
    vmax = np.percentile(volume, 99.999)
    if vmax > vmin:
        volume = np.clip((volume - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        volume = np.zeros_like(volume)

    return volume


# =============================================================================
# Tracking Data Loading
# =============================================================================


def load_tracks_from_csv(csv_files: list, n_timepoints: int) -> dict:
    """Load tracking data from CSV file(s).

    The CSV format from this dataset contains nuclei positions per timepoint.
    We auto-detect column names since formats vary.

    Args:
        csv_files: List of CSV file paths.
        n_timepoints: Maximum number of timepoints to include.

    Returns:
        Dict with 'tracks' (track_id -> list of (t, z, y, x)) and
        'colors' (track_id -> (r, g, b)).
    """
    if not csv_files:
        return None

    tracks = {}  # track_id -> [(t, z, y, x), ...]

    for csv_path in csv_files:
        with asection(f"Parsing {csv_path.name}"):
            with open(csv_path) as f:
                # Try to detect format
                first_line = f.readline().strip()
                f.seek(0)

                if "," in first_line:
                    reader = csv.DictReader(f)
                    headers = reader.fieldnames
                    aprint(f"  Columns: {headers}")

                    # Map common column name variants
                    col_map = {}
                    for h in headers:
                        hl = h.strip().lower()
                        if hl in ("frame", "t", "time", "timepoint"):
                            col_map["t"] = h
                        elif hl in ("x", "pos_x", "position_x"):
                            col_map["x"] = h
                        elif hl in ("y", "pos_y", "position_y"):
                            col_map["y"] = h
                        elif hl in ("z", "pos_z", "position_z"):
                            col_map["z"] = h
                        elif hl in (
                            "track_id",
                            "trackid",
                            "track",
                            "lineage_id",
                            "lineageid",
                            "id",
                            "label",
                            "cell",
                            "cell_name",
                            "name",
                            "identity",
                        ):
                            if "track" not in col_map:
                                col_map["track"] = h

                    required = {"t", "x", "y", "z"}
                    if not required.issubset(col_map.keys()):
                        missing = required - col_map.keys()
                        aprint(f"  Missing columns: {missing}, skipping")
                        continue

                    track_col = col_map.get("track")
                    aprint(
                        f"  Mapped: t={col_map['t']}, x={col_map['x']}, "
                        f"y={col_map['y']}, z={col_map['z']}, "
                        f"track={track_col}"
                    )

                    row_count = 0
                    for row in reader:
                        try:
                            t = int(float(row[col_map["t"]].strip()))
                        except (ValueError, KeyError):
                            continue

                        if t >= n_timepoints:
                            continue

                        try:
                            x = float(row[col_map["x"]].strip())
                            y = float(row[col_map["y"]].strip())
                            z = float(row[col_map["z"]].strip())
                        except (ValueError, KeyError):
                            continue

                        if track_col and row.get(track_col, "").strip():
                            tid = row[track_col].strip()
                        else:
                            tid = f"cell_{row_count}"

                        if tid not in tracks:
                            tracks[tid] = []
                        tracks[tid].append((t, z, y, x))
                        row_count += 1

                    aprint(
                        f"  Parsed {row_count} positions, {len(tracks)} unique tracks"
                    )

    if not tracks:
        return None

    # Sort each track by time
    for tid in tracks:
        tracks[tid].sort(key=lambda p: p[0])

    # Filter: keep only tracks with >= 2 timepoints (need a line)
    tracks = {tid: pts for tid, pts in tracks.items() if len(pts) >= 2}

    # Assign colours by lineage (unique hue per track)
    n_tracks = len(tracks)
    colors = {}
    for i, tid in enumerate(sorted(tracks.keys())):
        hue = i / max(n_tracks, 1)
        colors[tid] = _hue_to_rgb(hue)

    aprint(f"  {n_tracks} tracks with >= 2 positions")
    return {"tracks": tracks, "colors": colors}


def load_tracks_from_nuclei_files(nuclei_dirs: list, n_timepoints: int) -> dict:
    """Load tracking data from StarryNite nuclei files.

    Each file is named t001-nuclei, t002-nuclei, etc. and contains
    comma-delimited rows with nucleus positions and identities.

    Fields (per WormGUIDES / StarryNite convention):
        index, predecessor, cell_num, alt_id, unknown,
        x, y, z, size, cell_name, ...

    Args:
        nuclei_dirs: List of directories containing nuclei files.
        n_timepoints: Maximum timepoints to include.

    Returns:
        Dict with 'tracks' and 'colors', or None if no data found.
    """
    if not nuclei_dirs:
        return None

    tracks = {}  # cell_name -> [(t, z, y, x), ...]

    for nuclei_dir in nuclei_dirs:
        nuclei_files = sorted(nuclei_dir.glob("t*-nuclei"))
        if not nuclei_files:
            nuclei_files = sorted(nuclei_dir.glob("t*nuclei*"))
        if not nuclei_files:
            continue

        with asection(f"Parsing {len(nuclei_files)} nuclei files"):
            for nf in nuclei_files:
                # Extract timepoint from filename (e.g., t001-nuclei -> 0)
                fname = nf.stem
                try:
                    t_str = "".join(c for c in fname if c.isdigit())
                    t = int(t_str) - 1  # StarryNite is 1-indexed
                except ValueError:
                    continue

                if t < 0 or t >= n_timepoints:
                    continue

                with open(nf) as f:
                    for line in f:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) < 10:
                            continue

                        try:
                            x = float(parts[5])
                            y = float(parts[6])
                            z = float(parts[7])
                        except (ValueError, IndexError):
                            continue

                        cell_name = parts[9].strip() if len(parts) > 9 else ""
                        if not cell_name:
                            # Skip unnamed nuclei: the per-file `index`
                            # (parts[0]) is NOT a consistent identity across
                            # timepoints, so pooling them under
                            # ``cell_{index}`` teleports a synthetic cell
                            # across the embryo every frame. These spurious
                            # tracks produce the "starburst" artefact in
                            # downstream fading-trail visualisations.
                            continue

                        if cell_name not in tracks:
                            tracks[cell_name] = []
                        tracks[cell_name].append((t, z, y, x))

            aprint(f"  Parsed {len(tracks)} unique cells")

    if not tracks:
        return None

    # Sort and filter
    for tid in tracks:
        tracks[tid].sort(key=lambda p: p[0])

    tracks = {tid: pts for tid, pts in tracks.items() if len(pts) >= 2}

    n_tracks = len(tracks)
    colors = {}
    for i, tid in enumerate(sorted(tracks.keys())):
        hue = i / max(n_tracks, 1)
        colors[tid] = _hue_to_rgb(hue)

    aprint(f"  {n_tracks} tracks with >= 2 positions")
    return {"tracks": tracks, "colors": colors}


def load_tracking_data(csv_files: list, nuclei_dirs: list, n_timepoints: int) -> dict:
    """Load tracking data, trying CSV first then StarryNite nuclei files.

    Args:
        csv_files: List of CSV file paths.
        nuclei_dirs: List of directories with nuclei files.
        n_timepoints: Max timepoints.

    Returns:
        Dict with 'tracks' and 'colors', or None.
    """
    with asection("Loading tracking data"):
        # Try CSV first
        result = load_tracks_from_csv(csv_files, n_timepoints)
        if result:
            return result

        # Fall back to StarryNite nuclei files
        aprint("No CSV tracking data found, trying StarryNite nuclei files...")
        result = load_tracks_from_nuclei_files(nuclei_dirs, n_timepoints)
        if result:
            return result

        aprint("No tracking data found — scene will have volumes only (no tracks)")
        return None


# =============================================================================
# Cache Utilities
# =============================================================================


def _is_cached(cache_file: Path) -> bool:
    """Check whether a valid cache file exists.

    Returns True only when the file exists AND is not a leftover partial
    write (indicated by a corresponding .tmp file still present).
    """
    if RECOMPUTE:
        return False
    if not cache_file.exists():
        return False
    # If a .tmp sibling exists, the previous save was interrupted
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    if tmp_file.exists():
        # Clean up the corrupt cache and the tmp marker
        cache_file.unlink(missing_ok=True)
        tmp_file.unlink(missing_ok=True)
        return False
    return True


# =============================================================================
# Volume Preprocessing (CLAHE + N2S-NLM)
# =============================================================================


def calibrate_nlm_once(first_volume: np.ndarray | None) -> float:
    """Calibrate Non-Local Means denoising using the Noise2Self (J-invariant) method.

    Uses ``luxar.gsplats.preprocessing.calibrate_nlm_h`` (GPU-accelerated) on a
    single representative 2D slice to find the optimal ``h`` parameter for NLM
    denoising.  The result is cached in
    ``CACHE_DIR / "nlm_calibration_s{SAMPLE_INDEX}.json"`` so subsequent runs
    skip the calibration step entirely.

    Args:
        first_volume: 3D float32 volume (Z, Y, X) from the first timepoint,
            normalised to [0, 1].  May be None if the calibration is expected
            to be loaded from cache.

    Returns:
        Optimal ``h`` parameter for NLM denoising.
    """
    import torch

    from luxar.gsplats.preprocessing import calibrate_nlm_h

    cal_file = CACHE_DIR / f"nlm_calibration_s{SAMPLE_INDEX}.json"

    if _is_cached(cal_file):
        try:
            with open(cal_file) as f:
                cal = json.load(f)
            h = cal["h"]
            aprint(f"  Loaded cached NLM calibration: h={h:.6f}")
            return h
        except Exception as e:
            aprint(f"  Calibration cache load failed: {e}, re-calibrating")
            cal_file.unlink(missing_ok=True)

    if first_volume is None:
        raise RuntimeError(
            "NLM calibration cache is missing or corrupt and no volume was "
            "provided for re-calibration.  Re-run with --no-cache."
        )

    with asection("Calibrating NLM denoiser (Noise2Self / J-invariant)"):
        from luxar.gsplats.utils.device import resolve_torch_device

        vol_tensor = torch.from_numpy(first_volume)
        device = str(resolve_torch_device())

        h = calibrate_nlm_h(
            vol_tensor,
            patch_size=PREPROCESS_NLM_PATCH_SIZE,
            search_distance=PREPROCESS_NLM_PATCH_DISTANCE,
            use_2d_slice=True,
            device=device,
        )
        aprint(f"  Optimal h={h:.6f}")

        # Cache the result with marker-file safety
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_file = cal_file.with_suffix(cal_file.suffix + ".tmp")
        tmp_file.touch()
        with open(cal_file, "w") as f:
            json.dump({"h": float(h), "sample": SAMPLE_INDEX}, f)
        tmp_file.unlink(missing_ok=True)

    return h


def preprocess_volume(volume: np.ndarray, nlm_h: float) -> np.ndarray:
    """Preprocess a 3D volume with NLM denoising followed by CLAHE.

    Pipeline:
      1. Full 3D Non-Local Means denoising (GPU-accelerated via luxar)
      2. CLAHE contrast enhancement (GPU-accelerated via luxar)

    Args:
        volume: 3D float32 volume (Z, Y, X), normalised to [0, 1].
        nlm_h: Calibrated ``h`` parameter for NLM denoising.

    Returns:
        Preprocessed float32 volume, normalised to [0, 1].
    """
    import torch

    from luxar.gsplats.preprocessing import denoise_nlm
    from luxar.gsplats.utils.device import resolve_torch_device

    device = str(resolve_torch_device())

    # Step 1: Full 3D Non-Local Means denoising (GPU-accelerated)
    vol_tensor = torch.from_numpy(volume).to(device)
    denoised = denoise_nlm(
        vol_tensor,
        h=nlm_h,
        patch_size=PREPROCESS_NLM_PATCH_SIZE,
        search_distance=PREPROCESS_NLM_PATCH_DISTANCE,
    )

    # Step 2: CLAHE (GPU-accelerated)
    result = apply_clahe(
        denoised,
        tile_size=PREPROCESS_CLAHE_TILE,
        clip_limit=PREPROCESS_CLAHE_CLIP,
    )
    result = result.cpu().numpy()

    # Free GPU tensors from preprocessing
    del vol_tensor, denoised
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Re-normalise to [0, 1]
    rmin, rmax = result.min(), result.max()
    if rmax > rmin:
        result = (result - rmin) / (rmax - rmin)

    return result


def preprocess_timepoint(
    tiff_path: Path,
    label: str,
    cache_file: Path,
    nlm_h: float,
) -> np.ndarray:
    """Load, preprocess, and cache a single timepoint volume.

    If a valid cache exists, the preprocessed volume is loaded directly
    (skipping TIFF reading and denoising).  Uses the same marker-file
    pattern as GSplat caching for crash safety.

    Args:
        tiff_path: Path to raw TIFF file.
        label: Human-readable label for logging.
        cache_file: Path to cache file (.npy).
        nlm_h: Calibrated ``h`` for NLM denoising.

    Returns:
        Preprocessed float32 volume, normalised to [0, 1].
    """
    if _is_cached(cache_file):
        try:
            volume = np.load(cache_file)
            aprint(f"  Loaded cached preprocessed volume ({label})")
            return volume
        except Exception as e:
            aprint(f"  Preprocessed cache load failed: {e}, reprocessing")
            cache_file.unlink(missing_ok=True)

    aprint(f"  Preprocessing {label} (NLM denoise + CLAHE)...")
    raw_volume = load_timepoint_volume(tiff_path)
    volume = preprocess_volume(raw_volume, nlm_h)

    # Cache with marker-file safety
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    tmp_file.touch()
    np.save(cache_file, volume)
    tmp_file.unlink(missing_ok=True)

    return volume


# =============================================================================
# GSplats Fitting
# =============================================================================


def _load_cached(cache_file: Path, label: str) -> GSplatData | None:
    """Try loading a cached GSplatData.  Returns None on failure."""
    try:
        result = GSplatData.load(cache_file, include_stats=False)
        aprint(f"  Loaded {len(result.amplitudes):,} cached splats ({label})")
        return result
    except Exception as e:
        aprint(f"  Cache load failed: {e}, will re-fit")
        cache_file.unlink(missing_ok=True)
        return None


def fit_timepoint(
    volume: np.ndarray,
    label: str,
    cache_file: Path,
) -> GSplatData:
    """Fit GSplats to a single timepoint with caching.

    Uses a marker-file mechanism to detect interrupted saves: a ``.tmp``
    sentinel is created before writing and removed after.  If the process
    is killed mid-save, ``_is_cached()`` will detect the leftover
    sentinel on the next run and discard the partial cache file.

    Args:
        volume: 3D float32 volume (Z, Y, X), normalised to [0, 1].
        label: Human-readable label for logging.
        cache_file: Path to cache file.

    Returns:
        Fitted GSplatData.
    """
    if _is_cached(cache_file):
        result = _load_cached(cache_file, label)
        if result is not None:
            return result

    global DEVICE
    if DEVICE is None:
        from luxar.utils.demos import detect_device

        DEVICE = detect_device()

    aprint(
        f"  Progressive fitting {label} "
        f"(max {MAX_SPLATS} splats, {MAX_SPLATS_PER_PASS}/pass, "
        f"patience {PSNR_PATIENCE} dB)..."
    )

    # Progressive fitting: iteratively fits residuals in multiple passes,
    # building a multi-LOD representation from coarse to fine detail.
    # Pass voxel_size so GSplats account for the strong Z-anisotropy
    # (0.75 µm Z vs 0.15 µm XY = 5x).  output_space defaults to "real",
    # so centers come back in physical µm coordinates.
    result = fit_gaussian_splats(
        volume,
        seeds=MAX_SPLATS,
        device=DEVICE,
        verbose=True,
        lr=0.01,
        voxel_size=VOXEL_SIZE_ZYX,
    )

    aprint(f"  Fitted {len(result.amplitudes):,} splats")

    # Marker-file save: create .tmp sentinel before writing, remove after.
    # If interrupted mid-save, _is_cached() will detect the leftover .tmp
    # on the next run and discard the partial cache file.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    tmp_file.touch()  # marker: save in progress
    result.save(
        cache_file,
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )
    tmp_file.unlink(missing_ok=True)  # save complete — remove marker

    return result


def preprocess_and_fit_all_timepoints(tiff_files: list) -> list:
    """Preprocess and fit GSplats for each timepoint.

    Two-pass pipeline:
      1. **Calibrate** NLM denoiser (once, on first timepoint's middle slice)
      2. **For each timepoint**: preprocess (NLM + CLAHE) → fit GSplats

    Both preprocessing and fitting results are cached independently, so
    interrupted runs resume from where they left off.

    Args:
        tiff_files: List of TIFF file paths (one per timepoint).

    Returns:
        List of GSplatData.
    """
    n = min(len(tiff_files), TIMEPOINTS)

    # --- Pass 1: Calibrate NLM denoiser -----------------------------------

    with asection("NLM Calibration"):
        cal_file = CACHE_DIR / f"nlm_calibration_s{SAMPLE_INDEX}.json"
        if _is_cached(cal_file):
            # Fast path: calibration already cached, skip TIFF loading
            nlm_h = calibrate_nlm_once(None)
        else:
            first_vol = load_timepoint_volume(tiff_files[0])
            nlm_h = calibrate_nlm_once(first_vol)
            del first_vol  # Free memory

    # --- Pre-scan caches --------------------------------------------------

    preprocess_cache_files = [
        CACHE_DIR / f"celegans_s{SAMPLE_INDEX}_t{t:04d}_preprocessed.npy"
        for t in range(n)
    ]
    gsplat_cache_files = [
        CACHE_DIR / f"celegans_s{SAMPLE_INDEX}_t{t:04d}.gsplats.zarr.zip"
        for t in range(n)
    ]
    preprocess_cached = [_is_cached(f) for f in preprocess_cache_files]
    gsplat_cached = [_is_cached(f) for f in gsplat_cache_files]

    n_pp_cached = sum(preprocess_cached)
    n_gs_cached = sum(gsplat_cached)

    with asection(f"Processing {n} timepoints"):
        aprint(f"  Preprocessed cached: {n_pp_cached}/{n}")
        aprint(f"  GSplats cached: {n_gs_cached}/{n}")
        aprint(f"  Remaining: {n - n_gs_cached} to fit")

        gsplats_list = []
        for t in range(n):
            with asection(f"Timepoint {t}/{n - 1}"):
                # If GSplats are already cached, skip everything
                if gsplat_cached[t]:
                    gsplats = _load_cached(gsplat_cache_files[t], f"T={t}")
                    if gsplats is not None:
                        gsplats_list.append(gsplats)
                        continue

                # Preprocess (load from cache or compute)
                volume = preprocess_timepoint(
                    tiff_files[t],
                    f"T={t}",
                    preprocess_cache_files[t],
                    nlm_h,
                )

                # Fit GSplats on preprocessed volume
                gsplats = fit_timepoint(volume, f"T={t}", gsplat_cache_files[t])
                gsplats_list.append(gsplats)
                del volume  # Free numpy array early

                # Aggressively free GPU memory between timepoints to
                # prevent OOM from CUDA allocator fragmentation
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return gsplats_list


# =============================================================================
# Round-Trip Visualisation
# =============================================================================

_ROUNDTRIP_SAMPLE_COUNT = 3


def show_roundtrip_comparison(
    gsplats_list: list[GSplatData],
    n_timepoints: int,
) -> None:
    """Show original vs round-trip reconstructed volumes for sample timepoints.

    Loads preprocessed volumes from cache for comparison.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        aprint(
            "matplotlib is required for --show-roundtrip. Install with: pip install matplotlib"
        )
        return

    # Pick first, middle, last
    n_total = min(n_timepoints, len(gsplats_list))
    if n_total <= _ROUNDTRIP_SAMPLE_COUNT:
        sample_indices = list(range(n_total))
    else:
        sample_indices = [0, n_total // 2, n_total - 1]
    n_show = len(sample_indices)

    with asection(
        f"Round-trip reconstruction comparison ({n_show} of {n_total} timepoints)"
    ):
        # Load preprocessed volumes from cache
        volumes = []
        valid_indices = []
        for t in sample_indices:
            cache_file = (
                CACHE_DIR / f"celegans_s{SAMPLE_INDEX}_t{t:04d}_preprocessed.npy"
            )
            if cache_file.exists():
                vol = np.load(cache_file)
                volumes.append(vol)
                valid_indices.append(t)
            else:
                aprint(f"  T={t}: preprocessed cache not found, skipping")

        if not volumes:
            aprint("No preprocessed volumes found. Run with --recompute first.")
            return

        reconstructions = []
        for t, vol in zip(valid_indices, volumes):
            with asection(f"Rendering timepoint {t}"):
                recon = gsplats_list[t].render_to_volume(shape=vol.shape, device=DEVICE)
                reconstructions.append(recon)
                mse = float(np.mean((vol - recon) ** 2))
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
                aprint(f"  T={t}: PSNR: {psnr:.2f} dB, MSE: {mse:.6g}")

        n_show = len(valid_indices)
        fig, axes = plt.subplots(n_show, 3, figsize=(14, 4.5 * n_show), squeeze=False)

        for row, (t, vol, recon) in enumerate(
            zip(valid_indices, volumes, reconstructions)
        ):
            mid_z = vol.shape[0] // 2
            orig_slice = vol[mid_z]
            recon_slice = recon[mid_z]
            diff_slice = np.abs(orig_slice - recon_slice)

            mse = float(np.mean((vol - recon) ** 2))
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
            f"C. elegans — Round-Trip Comparison — z-slice {mid_z}  "
            f"({sum(len(g.amplitudes) for g in gsplats_list):,} total splats)",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()


# =============================================================================
# Scene Creation
# =============================================================================


def add_cell_tracks(
    scene,
    tracking_data: dict,
    shared_centroid: np.ndarray,
) -> None:
    """Add cell tracking lines to the scene.

    All tracks are batched into a single "segments" line node.  Each consecutive
    pair of positions in a track becomes one line segment.  Vertices are 4D:
    [x, y, z, time] in physical µm coordinates.

    Track positions from CSV/nuclei files are in voxel coordinates and are
    converted to physical µm using VOXEL_SIZE_ZYX before centering.

    Args:
        scene: Luxar Scene object.
        tracking_data: Dict with 'tracks' and 'colors'.
        shared_centroid: 3D centroid in physical µm (Z, Y, X order),
            matching the GSplats output space.
    """
    tracks = tracking_data["tracks"]
    track_colors = tracking_data["colors"]

    # Voxel-to-physical conversion factors (Z, Y, X)
    vz, vy, vx = VOXEL_SIZE_ZYX

    with asection(f"Adding {len(tracks)} cell track lines"):
        # Build segment vertices: pairs of consecutive points per track
        all_verts = []
        all_colors = []

        # shared_centroid is in µm, order [Z, Y, X] (from GSplat fitting).
        # Scene dims are [x, y, z, time], so remap:
        cx, cy, cz = shared_centroid[2], shared_centroid[1], shared_centroid[0]

        total_segments = 0
        for tid, positions in tracks.items():
            color = track_colors[tid]
            color_arr = np.array(color, dtype=np.float32)

            for i in range(len(positions) - 1):
                t0, z0, y0, x0 = positions[i]
                t1, z1, y1, x1 = positions[i + 1]

                # Convert voxel coords to physical µm (same as GSplats:
                # physical = voxel_index * voxel_size, no half-voxel offset)
                # then subtract the shared centroid.
                # 4D vertex: [x_um, y_um, z_um, time] — matching scene dims
                v0 = np.array(
                    [
                        x0 * vx - cx,
                        y0 * vy - cy,
                        z0 * vz - cz,
                        float(t0),
                    ],
                    dtype=np.float32,
                )
                v1 = np.array(
                    [
                        x1 * vx - cx,
                        y1 * vy - cy,
                        z1 * vz - cz,
                        float(t1),
                    ],
                    dtype=np.float32,
                )

                all_verts.append(v0)
                all_verts.append(v1)
                all_colors.append(color_arr)
                all_colors.append(color_arr)
                total_segments += 1

        if total_segments == 0:
            aprint("  No valid track segments to add")
            return

        vertices = np.array(all_verts, dtype=np.float32)
        colors = np.array(all_colors, dtype=np.float32)

        aprint(f"  {total_segments:,} segments from {len(tracks)} tracks")
        aprint(f"  Vertices shape: {vertices.shape}")

        scene.add_lines(
            name="cell_tracks",
            vertices=vertices,
            widths=0.3,
            colors=colors,
            line_type="segments",
            extend_to_all=["time"],
            layer=True,
        )

        aprint(f"  Added cell_tracks node ({total_segments:,} segments)")


def add_fading_trail_tracks(
    scene,
    tracking_data: dict,
    shared_centroid: np.ndarray,
    n_timepoints: int,
) -> None:
    """Add per-timepoint fading-trail tracks + current-position points.

    At each current time ``T``, emits line segments covering the last
    ``TRAIL_HISTORY`` hops ending at ``T`` — i.e. segments
    ``(T - k - 1) -> (T - k)`` for ``k = 0 .. TRAIL_HISTORY - 1``.
    Each segment's RGB is scaled by ``TRAIL_FADE[k]`` (newest = brightest),
    so older hops fade toward black.

    Every current cell position at ``T`` is also emitted as a bright point
    so the live state is clearly visible above the trail.

    All emitted geometry uses ``extend_to_all=[]`` so it only appears at
    its tagged time — scrubbing the time slider produces a rolling
    comet-tail effect behind each tracked cell.

    Args:
        scene: Luxar Scene object.
        tracking_data: Dict with ``tracks`` (tid -> list[(t, z, y, x)]) and
            ``colors`` (tid -> (r, g, b) in [0, 1]).
        shared_centroid: 3D centroid in physical µm (Z, Y, X order),
            matching the GSplats output space.
        n_timepoints: Number of timepoints in the scene (time dim length).
    """
    tracks = tracking_data["tracks"]
    track_colors = tracking_data["colors"]
    vz, vy, vx = VOXEL_SIZE_ZYX
    # shared_centroid is in µm order [Z, Y, X]; scene dims are [x, y, z, time]
    cx, cy, cz = shared_centroid[2], shared_centroid[1], shared_centroid[0]

    # Build per-track {t: (x, y, z)} lookup in physical µm, centered.
    #
    # IMPORTANT: the StarryNite nuclei x-coordinate is flipped relative to
    # the voxel-index convention used by the Gaussian-splat fitter for this
    # MSKCC confocal dataset — the fitter sees the volume with its x-axis
    # reversed. We reflect x about the image width (IMAGE_SHAPE[2] - 1) so
    # that fitter-space and track-space align. Empirically, this brings
    # mean nearest-gsplat distance across named cells down to 0.6-0.9 µm
    # at every timepoint tested (vs 1.5-2.1 µm for a naive
    # centroid-reflection and 2.3-2.9 µm for no flip at all).
    x_width = IMAGE_SHAPE[2]  # 512 along x
    positions_by_time: dict[int, dict[int, np.ndarray]] = {}
    for tid, positions in tracks.items():
        pbt: dict[int, np.ndarray] = {}
        for t, z, y, x in positions:
            x_flipped = (x_width - 1) - x
            pbt[int(t)] = np.array(
                [x_flipped * vx - cx, y * vy - cy, z * vz - cz],
                dtype=np.float32,
            )
        positions_by_time[tid] = pbt

    # Temporal smoothing is applied ONLY to trail-line vertices, not to the
    # current-timepoint point positions. Smoothing the point positions would
    # offset the splat markers from the underlying nuclei image by up to a
    # voxel. Trails, by contrast, benefit from smoothing — it dampens the
    # sub-voxel jitter that dominates short-horizon biological motion.
    # Endpoints are left as-is (windows simply shrink near boundaries).
    positions_smoothed: dict[int, dict[int, np.ndarray]] = positions_by_time
    if TRAIL_SMOOTH_WINDOW and TRAIL_SMOOTH_WINDOW > 1:
        half = TRAIL_SMOOTH_WINDOW // 2
        positions_smoothed = {}
        for tid, pbt in positions_by_time.items():
            out: dict[int, np.ndarray] = {}
            for t, p in pbt.items():
                # Gather centred window of neighbours that actually exist
                neighbours = [
                    pbt[tt] for tt in range(t - half, t + half + 1) if tt in pbt
                ]
                out[t] = (
                    np.mean(np.stack(neighbours, axis=0), axis=0).astype(np.float32)
                    if neighbours
                    else p
                )
            positions_smoothed[tid] = out

    trail_verts: list[np.ndarray] = []
    trail_colors: list[np.ndarray] = []
    point_positions: list[np.ndarray] = []
    point_colors: list[np.ndarray] = []
    n_rejected = 0

    with asection(
        f"Building fading trails (history={TRAIL_HISTORY}) over "
        f"{n_timepoints} timepoints"
    ):
        for current_t in range(n_timepoints):
            for tid, pbt in positions_by_time.items():
                base = np.asarray(track_colors[tid], dtype=np.float32)
                pbt_smooth = positions_smoothed[tid]

                # Current-position point (bright). Use RAW positions so the
                # splat markers align with the nuclei in the image volume.
                if current_t in pbt:
                    p = pbt[current_t]
                    point_positions.append(
                        np.array([p[0], p[1], p[2], float(current_t)], dtype=np.float32)
                    )
                    point_colors.append(base)

                # Fading trail segments ending at current_t.
                # Use SMOOTHED positions so trails read as clean motion paths,
                # and stride by TRAIL_SEGMENT_STRIDE to avoid emitting nearly
                # collinear segments within the smoothing window.
                # ANCHOR: the tip of the newest segment (seg_idx=0) uses the
                # RAW position at current_t so the trail visibly connects to
                # the cell marker (which is also drawn at the raw position).
                # Only past positions (t < current_t) are smoothed.
                stride = TRAIL_SEGMENT_STRIDE
                n_segments = TRAIL_HISTORY // stride
                for seg_idx in range(n_segments):
                    t_new = current_t - seg_idx * stride
                    t_old = current_t - (seg_idx + 1) * stride
                    if t_new not in pbt_smooth or t_old not in pbt_smooth:
                        continue
                    p_old = pbt_smooth[t_old]
                    p_new = pbt[current_t] if seg_idx == 0 else pbt_smooth[t_new]
                    # Reject implausibly long jumps — threshold scales with
                    # the segment's temporal span.
                    if (
                        float(np.linalg.norm(p_new - p_old))
                        > TRAIL_MAX_DISPLACEMENT_UM * stride
                    ):
                        n_rejected += 1
                        continue
                    faded = base * float(TRAIL_FADE[seg_idx * stride])

                    trail_verts.append(
                        np.array(
                            [p_old[0], p_old[1], p_old[2], float(current_t)],
                            dtype=np.float32,
                        )
                    )
                    trail_verts.append(
                        np.array(
                            [p_new[0], p_new[1], p_new[2], float(current_t)],
                            dtype=np.float32,
                        )
                    )
                    trail_colors.append(faded)
                    trail_colors.append(faded)

        n_segs = len(trail_verts) // 2
        n_pts = len(point_positions)
        aprint(f"  Fading trails: {n_segs:,} segments")
        aprint(f"  Current positions: {n_pts:,} points")
        if n_rejected > 0:
            aprint(
                f"  Rejected {n_rejected:,} segments exceeding "
                f"{TRAIL_MAX_DISPLACEMENT_UM} µm/frame × "
                f"{TRAIL_SEGMENT_STRIDE}-frame stride (mis-linked tracks)"
            )

    if trail_verts:
        scene.add_lines(
            name="cell_tracks_trail",
            vertices=np.asarray(trail_verts, dtype=np.float32),
            widths=TRAIL_LINE_WIDTH,
            colors=np.asarray(trail_colors, dtype=np.float32),
            line_type="segments",
            extend_to_all=[],
            layer=True,
            opacity=TRAIL_OPACITY,
        )
        aprint(
            f"  Added cell_tracks_trail node ({n_segs:,} segments, "
            f"opacity={TRAIL_OPACITY})"
        )

    if point_positions:
        scene.add_points(
            name="current_positions",
            positions=np.asarray(point_positions, dtype=np.float32),
            colors=np.asarray(point_colors, dtype=np.float32),
            radii=CURRENT_POINT_RADIUS,
            extend_to_all=[],
            layer=True,
            opacity=0.2,
        )
        aprint(f"  Added current_positions node ({n_pts:,} points, opacity=0.2)")


def _compute_max_sigma(gsplats: GSplatData) -> np.ndarray:
    """Compute the maximum standard deviation across spatial dimensions per splat."""
    ndim = gsplats.centers.shape[1]
    chol = gsplats.cholesky_factors
    max_sigma = np.zeros(len(chol))
    for d in range(ndim):
        start = d * (d + 1) // 2
        var = np.zeros(len(chol))
        for i in range(d + 1):
            var += chol[:, start + i] ** 2
        max_sigma = np.maximum(max_sigma, np.sqrt(var))
    return max_sigma


def combine_timepoints_to_4d(gsplats_list: list[GSplatData]) -> GSplatData:
    """Process and combine per-timepoint 3D GSplats into a single 4D dataset.

    For each timepoint:
      1. Translate to amplitude-weighted shared centroid (all timepoints)
      2. Filter oversized background splats (sigma > 3 um)
      3. Normalise amplitudes per-timepoint (max = 0.1)
      4. Skip per-splat color — the scene uses the ``bop_blue`` colormap

    Then combines all timepoints into one 4D dataset using
    ``GSplatData.combine_as_new_dimension`` with ``sigma=0`` (splats do not
    extend in the time dimension).

    The result is cached so repeated runs skip the combine step.

    Args:
        gsplats_list: List of 3D GSplatData, one per timepoint.

    Returns:
        Single 4D GSplatData with all timepoints combined.
    """
    n_timepoints = len(gsplats_list)

    if n_timepoints < 2:
        raise ValueError(
            f"Need at least 2 timepoints for 4D, got {n_timepoints}. "
            f"Use --timepoints=N with N >= 2."
        )

    cache_file = CACHE_DIR / (
        f"celegans_s{SAMPLE_INDEX}_combined_4d_{n_timepoints}tp.gsplats.zarr.zip"
    )

    if _is_cached(cache_file):
        result = _load_cached(cache_file, "combined 4D")
        if result is not None:
            return result

    with asection(f"Combining {n_timepoints} timepoints into single 4D dataset"):
        # Compute amplitude-weighted shared centroid across ALL timepoints
        with asection("Computing shared centroid"):
            all_centers = [g.centers for g in gsplats_list]
            all_amps = [g.amplitudes for g in gsplats_list]
            total_amp = sum(a.sum() for a in all_amps)
            if total_amp > 0:
                shared_centroid = (
                    sum(c.T @ a for c, a in zip(all_centers, all_amps)) / total_amp
                )
            else:
                shared_centroid = np.mean(np.concatenate(all_centers, axis=0), axis=0)
            aprint(f"Shared centroid: {shared_centroid}")

        # Process each timepoint: translate, filter, normalise, colour
        processed = []
        for t, gsplats in enumerate(gsplats_list):
            with asection(f"Processing timepoint {t}"):
                gsplats = gsplats.translate(-shared_centroid)

                # Filter oversized "background" splats (sigma > 3 um).
                # C. elegans nuclei are ~3-5 um diameter, so anything larger
                # is diffuse background that drowns out detail splats.
                sigma_threshold = 3.0  # um
                max_sigma = _compute_max_sigma(gsplats)
                keep = max_sigma < sigma_threshold
                n_before = gsplats.n_splats
                gsplats = gsplats.filter(keep)
                n_after = gsplats.n_splats

                if n_after < n_before:
                    aprint(
                        f"  Filtered {n_before - n_after} oversized splats "
                        f"(sigma > {sigma_threshold} um), {n_after} remain"
                    )

                # Normalise per-timepoint so max amplitude = 0.1. Leave
                # per-splat colors unset — the scene uses the ``bop_blue``
                # colormap on the gsplats layer (see add_gsplats_from_data
                # below) to shade splats by amplitude.
                if n_after > 0:
                    amp_max = gsplats.amplitudes.max()
                    if amp_max > 0:
                        gsplats = gsplats.scale_intensity(0.1 / amp_max)

                processed.append(gsplats)
                aprint(f"  {n_after:,} splats ready")

        # Combine into a single 4D dataset: time is the new dimension (sigma=0)
        combined = GSplatData.combine_as_new_dimension(
            processed,
            values=[float(t) for t in range(n_timepoints)],
            sigma=0.0,
        )
        aprint(
            f"Combined: {combined.n_splats:,} splats, "
            f"{combined.ndim}D (3D spatial + time)"
        )

        # Cache the combined result
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
        tmp_file.touch()
        combined.save(
            cache_file,
            encoding_mode=EncodingMode.MEMORY,
            include_fitting_info=True,
            compress="zip",
            zip_deflate=True,
        )
        tmp_file.unlink(missing_ok=True)
        aprint(f"Cached combined 4D dataset: {cache_file.name}")

    return combined


# -- Specific-brightness threshold for background rejection ----------------
# Specific brightness = amplitude / spatial_volume (3D only, ignoring time).
# After per-timepoint normalisation (max amplitude → 0.1), nuclei have
# sb ~0.05-1.5 while diffuse background splats have sb ~0.001-0.004.
# A threshold of 0.005 sits at the knee between the two populations,
# removing ~14% of splats that contribute only diffuse haze.
SPEC_BRIGHTNESS_THRESHOLD = 0.005


def filter_background_splats(
    combined: GSplatData,
    n_spatial_dims: int = 3,
) -> GSplatData:
    """Remove diffuse background splats from the combined 4D dataset.

    Uses *specific brightness* (amplitude / characteristic volume) to
    distinguish compact, bright nuclei from diffuse, dim background.
    Only the spatial dimensions are used for the volume computation
    (the time dimension has sigma=0, which would collapse the
    determinant).  The result is cached so repeated runs with the same
    threshold skip the filtering step.

    Args:
        combined: Combined 4D GSplatData (output of ``combine_timepoints_to_4d``).
        n_spatial_dims: Number of leading spatial dimensions (default 3).

    Returns:
        Filtered GSplatData with background splats removed.
    """
    if combined.ndim <= n_spatial_dims:
        raise ValueError(
            f"Expected >{n_spatial_dims}D data (spatial + extra dims), "
            f"got {combined.ndim}D"
        )

    # Cache key encodes splat count (ties to specific combine output)
    # and the brightness threshold.
    sb_str = f"{SPEC_BRIGHTNESS_THRESHOLD:.4f}".replace(".", "p")
    cache_file = CACHE_DIR / (
        f"celegans_s{SAMPLE_INDEX}_filtered_4d_{combined.n_splats}n_sb{sb_str}"
        f".gsplats.zarr.zip"
    )

    if _is_cached(cache_file):
        result = _load_cached(cache_file, "filtered 4D")
        if result is not None:
            return result

    with asection("Filtering background splats (specific brightness)"):
        n_before = combined.n_splats

        # Compute spatial-only volumes — the full nD volumes() method
        # includes the time dimension (sigma=0) which collapses det(Σ)
        # to near-zero.  The first k packed Cholesky elements are the
        # spatial block (time is appended as the last dimension by
        # combine_as_new_dimension).
        n_chol_spatial = n_spatial_dims * (n_spatial_dims + 1) // 2
        spatial_chol = combined.cholesky_factors[:, :n_chol_spatial]
        diag_indices = np.cumsum(np.arange(1, n_spatial_dims + 1)) - 1
        det_L = np.prod(spatial_chol[:, diag_indices], axis=1)
        spatial_vols = np.abs(det_L**2) ** (1.0 / n_spatial_dims)

        spec_brightness = combined.amplitudes / np.clip(spatial_vols, 1e-8, None)
        keep = spec_brightness > SPEC_BRIGHTNESS_THRESHOLD
        filtered = combined.filter(keep)
        n_after = filtered.n_splats
        aprint(
            f"Removed {n_before - n_after:,} background splats "
            f"(sb <= {SPEC_BRIGHTNESS_THRESHOLD}), "
            f"{n_after:,} remain ({100 * n_after / n_before:.1f}%)"
        )

        # Cache the filtered result
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
        tmp_file.touch()
        filtered.save(
            cache_file,
            encoding_mode=EncodingMode.MEMORY,
            include_fitting_info=True,
            compress="zip",
            zip_deflate=True,
        )
        tmp_file.unlink(missing_ok=True)
        aprint(f"Cached filtered 4D dataset: {cache_file.name}")

    return filtered


def create_luxar_scene(
    combined_4d: GSplatData,
    tracking_data: dict | None = None,
    shared_centroid: np.ndarray | None = None,
    output_path: Path | None = None,
) -> Path:
    """Create 4D Luxar scene from a single combined 4D GSplat dataset.

    Args:
        combined_4d: Single 4D GSplatData (3D spatial + time).
        tracking_data: Optional tracking data with 'tracks' and 'colors'.
        shared_centroid: 3D (Z, Y, X) centroid in physical µm that was
            subtracted from the GSplat centers during 4D combine. Required
            to align tracks with GSplats.
        output_path: Output .zarr path.

    Returns:
        Path to saved scene.
    """
    if output_path is None:
        output_path = get_demos_output_dir() / "gsplats_4d_celegans_tracking.luxar.zarr"

    # Infer number of timepoints from the time coordinate (last column)
    time_coords = combined_4d.centers[:, -1]
    n_timepoints = int(time_coords.max()) + 1

    with asection("Creating 4D Luxar Scene"):
        aprint(f"Output: {output_path.name}")
        aprint(f"Splats: {combined_4d.n_splats:,} ({combined_4d.ndim}D)")
        aprint(f"Timepoints: {n_timepoints}")

        # GSplats are fitted with voxel_size -> output in um (output_space="real").
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

            scene.attrs["title"] = "GSplats: C. elegans Embryo — Nuclei Tracking"
            scene.attrs["description"] = f"""
4D Gaussian Splatting — C. elegans Embryo
==========================================

A confocal microscopy time-series of a developing C. elegans embryo,
fully tracked with StarryNite and manually curated.

Visualisation:
  - Single 4D GSplat dataset ({combined_4d.n_splats:,} splats)
  - 3D Gaussian splats per timepoint, discrete in time (no temporal extent)

Data Source:
  - Zenodo record 6460303
  - DOI: 10.5281/zenodo.6460303
  - Sample: {SAMPLE_NAME}
  - {n_timepoints} of 400 timepoints shown

Imaging:
  - Zeiss Axio Observer.Z1
  - 41 x 512 x 512 voxels per timepoint
  - 0.75 x 0.15 x 0.15 um (ZYX)
  - 75s temporal resolution

Navigation:
  - Use the Time slider to scrub through development
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
            """

            # Add the single combined 4D dataset.
            # Centers are [Z, Y, X, Time] from embed_dimension; dim_order maps
            # them to the scene's [x, y, z, time] dimensions.
            scene.add_gsplats_from_data(
                name="gsplats_4d",
                result=combined_4d,
                dim_order=["z", "y", "x", "time"],
                extend_to_all=[],
                opacity=1.0,
                blending_mode="additive",
                colormap="bop_blue",
                layer=True,
            )
            aprint(f"Added single 4D gsplats node: {combined_4d.n_splats:,} splats")

            # Add cell tracks as per-timepoint rolling fading trails
            # (short windowed trails fix the "too cluttered" issue of showing
            # every full lineage all the time). Also emits a bright current-
            # position point per tracked cell, giving a three-geometry-type
            # visualisation (GSplats + Lines + Points) in a single scene.
            if tracking_data and shared_centroid is not None:
                add_fading_trail_tracks(
                    scene,
                    tracking_data,
                    shared_centroid,
                    n_timepoints=n_timepoints,
                )

            # --- Overlays ---
            # Title
            scene.add_text(
                "C. elegans Nuclei Tracking",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Info
            scene.add_text(
                "Confocal \u2022 Cell tracking",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: 4D C. elegans Embryo — Nuclei Tracking")
    aprint("=" * 70)
    aprint(f"Sample: {SAMPLE_NAME}  |  Timepoints: {TIMEPOINTS}")
    aprint("Volume rendering (GSplats) + cell lineage tracks (Lines)")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_4d_celegans_tracking.luxar.zarr"

    # Serve-only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    # Try loading precomputed per-timepoint GSplats from Git LFS bundle
    n_use = TIMEPOINTS
    file_names = [
        f"celegans_s{SAMPLE_INDEX}_t{t:04d}.gsplats.zarr.zip" for t in range(n_use)
    ]
    precomputed = load_precomputed_bundle(
        "gsplats_celegans",
        "celegans_s1_gsplats.zip",
        file_names,
        recompute=RECOMPUTE,
    )

    tracking_data = None

    if precomputed is not None:
        gsplats_list = precomputed
        aprint(f"Loaded {len(gsplats_list)} precomputed timepoints")

        # Tracks: the GSplats come from Git LFS (no Zenodo download needed),
        # but tracking lineages require the StarryNite nuclei files from the
        # Zenodo archive. Fast path: if the nuclei directory has already been
        # extracted (e.g. via HTTP-range extraction of just the tracks/), use
        # it directly and skip the 24 GB zip download/resume entirely.
        extracted_nuclei_dir = (
            CACHE_DIR
            / "extracted"
            / "mskcc-confocal"
            / f"mskcc_confocal_s{SAMPLE_INDEX}"
            / "tracks"
            / "nuclei"
        )
        if extracted_nuclei_dir.is_dir() and any(
            extracted_nuclei_dir.glob("t*-nuclei*")
        ):
            aprint(f"Using cached StarryNite tracks at {extracted_nuclei_dir}")
            tracking_data = load_tracks_from_nuclei_files(
                [extracted_nuclei_dir], len(gsplats_list)
            )
        else:
            aprint("Fetching StarryNite tracking lineage (Zenodo)...")
            zip_path = download_celegans_data()
            _, csv_files, nuclei_dirs = extract_sample_data(zip_path)
            tracking_data = load_tracking_data(
                csv_files, nuclei_dirs, len(gsplats_list)
            )
    else:
        # --recompute path: download, preprocess, fit from scratch
        warn_if_no_cuda_gpu()

        # Download
        zip_path = download_celegans_data()

        # Extract sample data
        tiff_files, csv_files, nuclei_dirs = extract_sample_data(zip_path)

        if not tiff_files:
            aprint("ERROR: No TIFF files found for sample. Aborting.")
            return

        n_available = len(tiff_files)
        n_use = min(n_available, TIMEPOINTS)
        aprint(f"Available timepoints: {n_available}, using: {n_use}")

        # Load tracking data
        tracking_data = load_tracking_data(csv_files, nuclei_dirs, n_use)

        # Preprocess + fit GSplats per timepoint
        gsplats_list = preprocess_and_fit_all_timepoints(tiff_files[:n_use])

    # Optional round-trip visualisation
    if SHOW_ROUNDTRIP:
        show_roundtrip_comparison(gsplats_list, len(gsplats_list))

    # Report per-timepoint fitting
    with asection("Fitting Summary"):
        total_splats = sum(len(g.amplitudes) for g in gsplats_list)
        aprint(f"Total splats: {total_splats:,} across {len(gsplats_list)} timepoints")
        if tracking_data:
            aprint(f"Tracks: {len(tracking_data['tracks'])}")
            total_pts = sum(len(pts) for pts in tracking_data["tracks"].values())
            aprint(f"Track points: {total_pts:,}")

    # Precompute the same amplitude-weighted shared centroid that
    # combine_timepoints_to_4d applies internally — we need it downstream
    # so cell tracks can be aligned to the centered GSplat coordinate frame.
    all_centers = [g.centers for g in gsplats_list]
    all_amps = [g.amplitudes for g in gsplats_list]
    _total_amp = sum(a.sum() for a in all_amps)
    if _total_amp > 0:
        shared_centroid = (
            sum(c.T @ a for c, a in zip(all_centers, all_amps)) / _total_amp
        )
    else:
        shared_centroid = np.mean(np.concatenate(all_centers, axis=0), axis=0)
    aprint(f"Shared centroid (ZYX µm): {shared_centroid}")

    # Combine all timepoints into a single 4D GSplat dataset (cached)
    combined_4d = combine_timepoints_to_4d(gsplats_list)

    with asection("Combined 4D Summary"):
        aprint(f"Total 4D splats: {combined_4d.n_splats:,}")
        aprint(f"Dimensions: {combined_4d.ndim}D")

    # Filter diffuse background splats by specific brightness (cached)
    combined_4d = filter_background_splats(combined_4d)

    with asection("Filtered 4D Summary"):
        aprint(f"Splats after filtering: {combined_4d.n_splats:,}")

    # Create 4D scene from the single combined dataset
    scene_path = create_luxar_scene(combined_4d, tracking_data, shared_centroid)

    # Launch viewer
    if not NO_SERVE:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
