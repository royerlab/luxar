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
    --no-cache:      Force re-fitting (ignore cached GSplats)
    --no-serve:      Generate scene without launching viewer
    --serve-only:    Just serve a previously generated scene
    --timepoints=N:  Number of timepoints to process (default: 400, max: 400)
    --sample=N:      Which sample to use: 1, 2, or 3 (default: 1)

Requirements:
    - CUDA GPU strongly recommended (fitting is ~100x slower on CPU)
    - scikit-image for N2S-NLM denoising:  pip install scikit-image

Output:
    - Scene saved to:  datasets/demos/gsplats_4d_celegans_tracking.zarr
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
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.demos import launch_viewer, warn_if_no_cuda_gpu
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Data source
ZENODO_URL = "https://zenodo.org/api/records/6460303/files/mskcc_confocal.zip/content"

# Volume specs
VOXEL_SIZE_ZYX = (0.75, 0.15, 0.15)  # Micrometres
IMAGE_SHAPE = (41, 512, 512)  # Z, Y, X per timepoint

# Fitting parameters
N_SEEDS = 5000  # Per timepoint (small volumes)
N_ITERS = 6000

# Cache location
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_celegans"

# Preprocessing parameters (CLAHE + Noise2Self-calibrated NLM)
PREPROCESS_CLAHE_TILE = 16
PREPROCESS_CLAHE_CLIP = 2.0
PREPROCESS_NLM_PATCH_SIZE = 3
PREPROCESS_NLM_PATCH_DISTANCE = 5

# Parse command-line flags
NO_CACHE = "--no-cache" in sys.argv
NO_SERVE = "--no-serve" in sys.argv
SERVE_ONLY = "--serve-only" in sys.argv

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

    # Normalise
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
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
                            cell_name = f"cell_{parts[0].strip()}"

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
    if NO_CACHE:
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

    Uses skimage's ``calibrate_denoiser`` on a single representative 2D slice
    to find the optimal ``h`` parameter for ``denoise_nl_means``.  The result
    is cached in ``CACHE_DIR / "nlm_calibration_s{SAMPLE_INDEX}.json"`` so
    subsequent runs skip the calibration step entirely.

    Args:
        first_volume: 3D float32 volume (Z, Y, X) from the first timepoint,
            normalised to [0, 1].  May be None if the calibration is expected
            to be loaded from cache.

    Returns:
        Optimal ``h`` parameter for ``denoise_nl_means``.
    """
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

    try:
        from skimage.restoration import calibrate_denoiser, denoise_nl_means
    except ImportError as e:
        raise ImportError(
            "scikit-image is required for preprocessing.\n"
            "Install with: pip install scikit-image"
        ) from e

    with asection("Calibrating NLM denoiser (Noise2Self / J-invariant)"):
        # Use middle z-slice — representative since imaging is consistent
        mid_z = first_volume.shape[0] // 2
        cal_slice = first_volume[mid_z]
        aprint(f"  Calibrating on z={mid_z} slice ({cal_slice.shape})")

        calibrated = calibrate_denoiser(
            cal_slice,
            denoise_nl_means,
            denoise_parameters=dict(
                h=np.arange(0.005, 0.08, 0.005),
                fast_mode=[True],
                patch_size=[PREPROCESS_NLM_PATCH_SIZE],
                patch_distance=[PREPROCESS_NLM_PATCH_DISTANCE],
            ),
            stride=2,
        )

        # Extract the calibrated h from the partial-function keywords.
        # calibrate_denoiser returns functools.partial(denoise_invariant, ...)
        # with the best params nested in 'denoiser_kwargs'.
        best_kwargs = calibrated.keywords.get("denoiser_kwargs", {})
        h = best_kwargs.get("h", 0.02)
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
    """Preprocess a 3D volume with N2S-NLM denoising followed by CLAHE.

    Pipeline:
      1. Full 3D Non-Local Means denoising (using calibrated ``h``)
      2. CLAHE contrast enhancement (GPU-accelerated via luxar)

    Args:
        volume: 3D float32 volume (Z, Y, X), normalised to [0, 1].
        nlm_h: Calibrated ``h`` parameter for ``denoise_nl_means``.

    Returns:
        Preprocessed float32 volume, normalised to [0, 1].
    """
    import torch
    from skimage.restoration import denoise_nl_means

    # Step 1: Full 3D Non-Local Means denoising
    denoised = denoise_nl_means(
        volume,
        h=nlm_h,
        patch_size=PREPROCESS_NLM_PATCH_SIZE,
        patch_distance=PREPROCESS_NLM_PATCH_DISTANCE,
        fast_mode=True,
    ).astype(np.float32)

    # Step 2: CLAHE (GPU-accelerated)
    vol_torch = torch.from_numpy(denoised)
    if torch.cuda.is_available():
        vol_torch = vol_torch.cuda()
    result = apply_clahe(
        vol_torch,
        tile_size=PREPROCESS_CLAHE_TILE,
        clip_limit=PREPROCESS_CLAHE_CLIP,
    )
    result = result.cpu().numpy()

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
        import torch

        if torch.cuda.is_available():
            DEVICE = "cuda"
            aprint("Using CUDA device")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE = "mps"
            aprint("Using MPS device (Metal acceleration)")
        else:
            DEVICE = "cpu"
            aprint("Using CPU device")

    from luxar.gsplats import fit_gaussian_splats

    aprint(f"  Fitting {label} ({N_ITERS} iters, {N_SEEDS} seeds)...")

    # Pass voxel_size so GSplats account for the strong Z-anisotropy
    # (0.75 µm Z vs 0.15 µm XY = 5x).  output_space defaults to "real",
    # so centers come back in physical µm coordinates.
    result = fit_gaussian_splats(
        volume,
        lr=0.01,
        seeds=N_SEEDS,
        n_iters=N_ITERS,
        device=DEVICE,
        verbose=True,
        enable_dynamic_ops=True,
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

        return gsplats_list


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
        )

        aprint(f"  Added cell_tracks node ({total_segments:,} segments)")


def create_luxar_scene(
    gsplats_list: list,
    tracking_data: dict = None,
    output_path: Path = None,
) -> Path:
    """Create 4D Luxar scene with GSplats per timepoint and optional track lines.

    Args:
        gsplats_list: List of GSplatData, one per timepoint.
        tracking_data: Optional tracking data with 'tracks' and 'colors'.
        output_path: Output .zarr path.

    Returns:
        Path to saved scene.
    """
    if output_path is None:
        output_path = get_demos_output_dir() / "gsplats_4d_celegans_tracking.zarr"

    n_timepoints = len(gsplats_list)

    if n_timepoints < 2:
        raise ValueError(
            f"Need at least 2 timepoints for a 4D scene, got {n_timepoints}. "
            f"Use --timepoints=N with N >= 2."
        )

    with asection("Creating 4D Luxar Scene"):
        aprint(f"Output: {output_path.name}")
        aprint(f"Timepoints: {n_timepoints}")
        aprint(f"Tracking: {'yes' if tracking_data else 'no'}")

        # GSplats are fitted with voxel_size → output in µm (output_space="real").
        # Track line vertices are also converted to µm in add_cell_tracks().
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
            scene = compiler.create_scene(dimensions=dims)

            has_tracks = tracking_data is not None
            n_tracks = len(tracking_data["tracks"]) if has_tracks else 0

            scene.attrs["title"] = "GSplats: C. elegans Embryo — Nuclei Tracking"
            scene.attrs["description"] = f"""
4D Gaussian Splatting + Cell Lineage Tracks — C. elegans Embryo
================================================================

A confocal microscopy time-series of a developing C. elegans embryo,
fully tracked with StarryNite and manually curated.

Visualisation:
  - GSplats: 3D Gaussian splats per timepoint (volume rendering)
  - Lines:   4D polylines for {n_tracks} tracked cell nuclei trajectories

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
  - Cell tracks are always visible as coloured lines
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
            """

            # Compute shared centroid across ALL timepoints
            with asection("Computing shared centroid"):
                all_centers = [g.centers for g in gsplats_list]
                all_amps = [g.amplitudes for g in gsplats_list]
                total_amp = sum(a.sum() for a in all_amps)
                shared_centroid = (
                    sum(c.T @ a for c, a in zip(all_centers, all_amps)) / total_amp
                )
                aprint(f"Shared centroid: {shared_centroid}")

            # Add GSplats per timepoint
            for t, gsplats in enumerate(gsplats_list):
                with asection(f"Adding GSplats timepoint {t}"):
                    gsplats = gsplats.translate(-shared_centroid)

                    # Filter out oversized "background" splats.
                    # The fitting produces a few huge splats (sigma > 5 µm)
                    # that cover the entire volume as a diffuse glow,
                    # drowning out the ~2000 small detail splats (sigma
                    # ~0.1 µm) that represent actual nuclei structure.
                    # C. elegans nuclei are ~3-5 µm, so sigma > 3 µm is
                    # clearly background.
                    ndim_spatial = gsplats.centers.shape[1]
                    chol = gsplats.cholesky_factors
                    max_sigma = np.zeros(len(chol))
                    for d in range(ndim_spatial):
                        start = d * (d + 1) // 2
                        var = np.zeros(len(chol))
                        for i in range(d + 1):
                            var += chol[:, start + i] ** 2
                        max_sigma = np.maximum(max_sigma, np.sqrt(var))

                    sigma_threshold = 3.0  # µm — nuclei are ~3-5 µm diameter
                    keep = max_sigma < sigma_threshold
                    n_before = len(gsplats.amplitudes)
                    centers_f = gsplats.centers[keep]
                    amps_f = gsplats.amplitudes[keep]
                    chol_f = gsplats.cholesky_factors[keep]
                    sharp_f = gsplats.sharpnesses[keep]
                    n_after = len(amps_f)

                    if n_after < n_before:
                        aprint(
                            f"  Filtered {n_before - n_after} oversized splats "
                            f"(sigma > {sigma_threshold} µm), {n_after} remain"
                        )

                    # Normalise per-timepoint so max amplitude = 0.1.
                    if n_after > 0:
                        amp_max = amps_f.max()
                        if amp_max > 0:
                            amps_f = amps_f * (0.1 / amp_max)

                    # Soft green colour for the fluorescence
                    colors = np.tile(
                        np.array([0.4, 1.0, 0.5], dtype=np.float32),
                        (n_after, 1),
                    )

                    scene.add_gsplats(
                        name=f"gsplats_t{t:04d}",
                        centers=centers_f,
                        amplitudes=amps_f,
                        cholesky_factors=chol_f,
                        colors=colors,
                        sharpness=sharp_f,
                        dim_order=["z", "y", "x"],
                        fill={"time": float(t)},
                        fill_sigma={"time": 0},
                        extend_to_all=[],
                        opacity=0.7,
                        blending_mode="additive",
                    )
                    aprint(f"  Added {n_after:,} splats at time={t}")

            # Add cell track lines
            if tracking_data:
                add_cell_tracks(scene, tracking_data, shared_centroid)

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    warn_if_no_cuda_gpu()
    aprint("=" * 70)
    aprint("GSplats Demo: 4D C. elegans Embryo — Nuclei Tracking")
    aprint("=" * 70)
    aprint(f"Sample: {SAMPLE_NAME}  |  Timepoints: {TIMEPOINTS}")
    aprint("Volume rendering (GSplats) + cell lineage tracks (Lines)")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_4d_celegans_tracking.zarr"

    # Serve-only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

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

    # Report
    with asection("Fitting Summary"):
        total_splats = sum(len(g.amplitudes) for g in gsplats_list)
        aprint(f"Total splats: {total_splats:,} across {len(gsplats_list)} timepoints")
        if tracking_data:
            aprint(f"Tracks: {len(tracking_data['tracks'])}")
            total_pts = sum(len(pts) for pts in tracking_data["tracks"].values())
            aprint(f"Track points: {total_pts:,}")

    # Create 4D scene
    scene_path = create_luxar_scene(gsplats_list, tracking_data)

    # Launch viewer
    if not NO_SERVE:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
