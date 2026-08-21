#!/usr/bin/env python3
"""Self-Contained Demo: 3D STORM Super-Resolution Microscopy

Visualize microtubule cytoskeleton at nanometer resolution using real STORM
super-resolution microscopy localizations as Gaussian splats.

================================================================================
SUPER-RESOLUTION MICROSCOPY: STORM/PALM
================================================================================

BREAKING THE DIFFRACTION LIMIT
-------------------------------
Traditional light microscopy is limited by diffraction to ~200-300 nm resolution.
This means cellular structures smaller than this appear blurred together.

STORM (Stochastic Optical Reconstruction Microscopy) and PALM (Photo-Activated
Localization Microscopy) achieve ~20 nm resolution - 10x better than the
diffraction limit!

HOW STORM WORKS:
----------------
1. **Photoactivatable fluorophores**: Special dyes that can be switched on/off
2. **Sparse activation**: Only a few molecules emit at once (well-separated)
3. **Precise localization**: Fit PSF to find center with ~10-20 nm precision
4. **Repeat thousands of times**: Build up complete image from localizations
5. **Reconstruct**: Combine all localizations into super-resolved image

Each detected molecule gives:
- **Position (x, y, z)**: Center coordinates
- **Precision (σx, σy, σz)**: Localization uncertainty (10-30 nm typically)
- **Photons**: Number of photons detected
- **Frame**: Which image frame the molecule appeared in

3D STORM METHODS:
-----------------
- **Astigmatism**: Cylindrical lens makes PSF elliptical, shape encodes z
- **Biplane**: Two focal planes separated by ~500 nm
- **Double-helix**: PSF rotates as a function of z position

MICROTUBULES:
-------------
Microtubules are hollow cylinders (~25 nm outer diameter) that form the cell's
structural skeleton and highway system. They're perfect for STORM because:
- Long, linear structures (μm scale)
- Important for cell division, transport, shape
- ~25 nm diameter is BELOW the diffraction limit!
- With STORM, individual microtubule filaments are resolved!

GAUSSIAN SPLATS FOR STORM:
---------------------------
The two views deliberately use Gaussian splats with different provenance:

Widefield splats are a BASIS fitted to a continuous diffraction-blurred field:
- Rasterize localization events, weighted by detected photons
- Convolve with the lateral/axial widefield PSF
- Fit a compact set of anisotropic Gaussian splats to that volume

Super-resolution splats are EPISTEMIC uncertainty ellipsoids, placed directly
from the detector output — there is nothing to fit:
- **Center**: Detected molecule position (x, y, z)
- **Covariance**: CRLB combined with antibody-label linkage uncertainty
- **Amplitude**: Photon count or intensity
- **Variation**: Dimmer, uncertain localizations remain visibly fuzzier

This demo shows:
1. **Super-resolved view**: Each localization as a Gaussian splat
2. **Widefield comparison**: Photon-weighted raster, PSF blur, and fitted splats
3. **Categorical dimension**: Toggle between widefield ↔ super-resolution

DATA SOURCE:
============
Zenodo: 3D STORM Dataset - COS7 Cells, Alpha-Tubulin
https://zenodo.org/records/3547521

- **Sample**: COS7 cells (monkey kidney fibroblasts)
- **Target**: Alpha-tubulin (microtubule protein)
- **Stain**: Alexa Fluor 647 immunofluorescence
- **Method**: 3D STORM with astigmatism
- **Resolution**: ~20 nm lateral, ~50 nm axial
- **Localizations**: Millions per field of view
- **Format**: CSV with columns (x, y, z, precision, photons, etc.)

WHAT YOU'LL SEE:
================
- Intricate 3D microtubule network
- Individual filaments clearly resolved (~25 nm diameter)
- Crossings and bundling visible
- Toggle between blurry widefield and sharp super-resolution
- Each splat = one detected fluorophore molecule!

Usage:
    python demo_storm_3d_microtubules.py [--max-localizations=N]

    Options:
    --max-localizations=N   Limit number of localizations (default: 5M)
    --no-serve              Generate dataset without launching viewer
    --field=N               Select field of view (default: 4)
    --recompute             Ignore the cached widefield fit

Controls:
    - Press '1' to select VIEW dimension
    - Press '['/']' to toggle: Widefield ↔ Super-Resolution
    - Rotate to explore 3D microtubule network
    - Zoom in to see individual molecules!
    - Ctrl+C to stop and cleanup
"""

from __future__ import annotations

DEMO_META = {
    "key": "storm_3d_microtubules",
    "title": "STORM 3D Microtubules",
    "description": "Real 3D STORM super-resolution localizations of COS7 microtubules rendered as Gaussian splats.",
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 1800,
        "compute": "heavy",
        "gpu": "required",
        "local_data": None,
    },
    "caches": ["storm_data"],
    "outputs": ["storm_3d_microtubules"],
}

import hashlib
import sys
from pathlib import Path

import numpy as np
import requests
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer, require_module
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Zenodo dataset
ZENODO_RECORD = "3547521"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# Default parameters
DEFAULT_FIELD = 4  # Field of view number
DEFAULT_MAX_LOCALIZATIONS = 5_000_000  # Limit for demo performance

# Visualization parameters
WIDEFIELD_PSF_SIGMA_NM = 100.0  # lateral conventional microscopy PSF width
WIDEFIELD_AXIAL_PSF_SIGMA_NM = 200.0
SUPERRES_PSF_SIGMA = 20.0  # nm - super-resolution PSF width
LABEL_LINKAGE_SIGMA_NM = 17.0  # primary + secondary IgG linkage error
PIXEL_SIZE = 106.0  # nm - from dataset metadata
WIDEFIELD_VOXEL_SIZE_UM = 0.075
WIDEFIELD_TILE_SIZE = (64, 256, 256)
WIDEFIELD_OVERLAP = (8, 32, 32)
WIDEFIELD_SEEDS_PER_TILE = 6000
WIDEFIELD_N_ITERS = 5000
WIDEFIELD_CULL_RETENTION = 0.999
WIDEFIELD_FLOOR = "none"
WIDEFIELD_OUTPUT_SPACE = "real"
WIDEFIELD_ENABLE_DYNAMIC_OPS = True

# Visualization scale factor applied to every splat sigma (widefield and
# super-res alike), preserving their relative sizes. 1.0 = physically faithful
# widths (super-res splats really are ~tens of nm in a ~60 μm scene, so they
# read as fine points); raise it only if you want to exaggerate splat size for
# a zoomed-out overview.
VIS_SCALE = 1.0


# Cache paths
CACHE_DIR = Path.home() / ".cache" / "luxar" / "storm_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _widefield_cache_path(
    csv_path: Path,
    max_localizations: int | None,
) -> Path:
    """Return a cache path keyed by source identity and every fit parameter."""
    source_stat = csv_path.stat()
    cache_spec = (
        csv_path.name,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        max_localizations,
        WIDEFIELD_PSF_SIGMA_NM,
        WIDEFIELD_AXIAL_PSF_SIGMA_NM,
        WIDEFIELD_VOXEL_SIZE_UM,
        WIDEFIELD_TILE_SIZE,
        WIDEFIELD_OVERLAP,
        WIDEFIELD_SEEDS_PER_TILE,
        WIDEFIELD_N_ITERS,
        WIDEFIELD_CULL_RETENTION,
        WIDEFIELD_FLOOR,
        WIDEFIELD_OUTPUT_SPACE,
        WIDEFIELD_ENABLE_DYNAMIC_OPS,
        "stream",
    )
    digest = hashlib.sha256(repr(cache_spec).encode()).hexdigest()[:12]
    return CACHE_DIR / f"{csv_path.stem}_widefield_{digest}.gsplats.zarr.zip"


# =============================================================================
# Data Download
# =============================================================================


def download_storm_localizations(
    field: int = 4,
    cache_dir: Path = CACHE_DIR,
) -> Path:
    """Download STORM localization CSV from Zenodo.

    Args:
        field: Field of view number (4 or 5)
        cache_dir: Where to cache the download

    Returns:
        Path to downloaded CSV file
    """
    filename = f"Cos7_MT_A647_FOV_{field}_Localizations.csv"
    cache_file = cache_dir / filename
    url = f"{ZENODO_BASE_URL}/{filename}"

    with asection(f"Downloading STORM localizations (FOV {field})"):
        if cache_file.exists():
            size_mb = cache_file.stat().st_size / (1024**2)
            aprint(f"✓ Using cached file: {cache_file.name}")
            aprint(f"  Size: {size_mb:.1f} MB")
            return cache_file

        aprint("Dataset: 3D STORM - COS7 Microtubules")
        aprint(f"Source: Zenodo record {ZENODO_RECORD}")
        aprint(f"URL: {url}")
        aprint("")
        aprint("⏱️  Downloading ~1.8 GB (may take 2-5 minutes)...")
        aprint("")

        # Try curl first (more robust), fallback to requests
        import shutil
        import subprocess
        import time

        if shutil.which("curl"):
            # Use curl for robust download with automatic resume
            aprint("Using curl for download (automatic resume support)...")
            try:
                cmd = [
                    "curl",
                    "-L",  # Follow redirects
                    "-C",
                    "-",  # Resume from partial
                    "--retry",
                    "10",
                    "--retry-delay",
                    "5",
                    "--max-time",
                    "3600",
                    "-o",
                    str(cache_file),
                    "-#",  # Progress bar
                    url,
                ]

                subprocess.run(cmd, check=True)
                aprint(f"✓ Downloaded to {cache_file}")
                aprint(f"  Size: {cache_file.stat().st_size / (1024**2):.1f} MB")
                return cache_file

            except subprocess.CalledProcessError:
                aprint("⚠️  curl failed, trying Python requests...")

        # Fallback: Python requests with manual resume
        try:
            temp_file = cache_file.parent / f"{cache_file.name}.partial"
            resume_pos = 0
            if temp_file.exists():
                resume_pos = temp_file.stat().st_size
                aprint(
                    f"Found partial download ({resume_pos / (1024**2):.1f} MB), resuming..."
                )

            # Robust download with resume support
            max_retries = 5
            retry_delay = 10

            for attempt in range(max_retries):
                try:
                    # Request with resume support + browser headers
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                        "Accept-Encoding": "gzip, deflate, br",
                        "DNT": "1",
                        "Connection": "keep-alive",
                        "Upgrade-Insecure-Requests": "1",
                    }
                    if resume_pos > 0:
                        headers["Range"] = f"bytes={resume_pos}-"

                    response = requests.get(
                        url, headers=headers, stream=True, timeout=300
                    )

                    # Handle rate limiting
                    if response.status_code == 429:
                        if attempt < max_retries - 1:
                            aprint(
                                f"⚠️  Rate limited, waiting {retry_delay}s (attempt {attempt + 1}/{max_retries})..."
                            )
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                        else:
                            raise Exception("Rate limit exceeded after retries")

                    # Handle resume response codes
                    if response.status_code == 206:  # Partial content (resume)
                        aprint(f"✓ Resuming from {resume_pos / (1024**2):.1f} MB")
                        mode = "ab"  # Append mode
                    elif response.status_code == 200:  # Full download
                        mode = "wb"  # Write mode
                        resume_pos = 0
                    else:
                        response.raise_for_status()
                        continue

                    # Get total size
                    content_range = response.headers.get("Content-Range")
                    if content_range:
                        total_size = int(content_range.split("/")[-1])
                    else:
                        total_size = (
                            int(response.headers.get("content-length", 0)) + resume_pos
                        )

                    aprint(f"File size: {total_size / (1024**2):.1f} MB")

                    downloaded = resume_pos
                    chunk_size = 1024 * 1024  # 1 MB chunks

                    with open(temp_file, mode) as f:
                        last_progress = downloaded
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)

                                # Progress every 100 MB
                                if downloaded - last_progress >= 100 * 1024 * 1024:
                                    percent = (
                                        (downloaded / total_size * 100)
                                        if total_size > 0
                                        else 0
                                    )
                                    aprint(
                                        f"  Progress: {downloaded / (1024**2):.0f} / {total_size / (1024**2):.0f} MB ({percent:.0f}%)"
                                    )
                                    last_progress = downloaded

                    # Download complete - move temp to final
                    temp_file.rename(cache_file)
                    aprint(f"✓ Downloaded to {cache_file}")
                    aprint(
                        f"  Final size: {cache_file.stat().st_size / (1024**2):.1f} MB"
                    )
                    break  # Success!

                except (
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                ) as e:
                    if attempt < max_retries - 1:
                        aprint(f"⚠️  Download interrupted ({e})")
                        aprint(
                            f"   Retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 120)  # Max 2 min
                        continue
                    else:
                        raise

        except Exception as e:
            aprint(f"❌ Download failed: {e}")
            if cache_file.exists():
                cache_file.unlink()
            raise

    return cache_file


# =============================================================================
# Localization Processing
# =============================================================================


def parse_storm_localizations(
    csv_path: Path,
    max_localizations: int | None = None,
) -> dict:
    """Parse STORM localizations from CSV file.

    Args:
        csv_path: Path to localization CSV
        max_localizations: Optional limit on number of localizations

    Returns:
        Dictionary with keys: x, y, z, precision_x, precision_y, precision_z,
        photons, frame, etc.
    """
    # Gated here, not in main(): the localization CSV is parsed with pandas.
    pd = require_module("pandas")

    with asection("Parsing STORM localizations"):
        aprint(f"CSV file: {csv_path.name}")
        aprint("Reading CSV (may take 30-60 seconds for large files)...")

        # Read CSV (may have millions of rows!)
        if max_localizations:
            df = pd.read_csv(csv_path, nrows=max_localizations)
            aprint(
                f"✓ Loaded {len(df):,} localizations (limited to {max_localizations:,})"
            )
        else:
            df = pd.read_csv(csv_path)
            aprint(f"✓ Loaded {len(df):,} total localizations")

        df.columns = [column.strip() for column in df.columns]
        aprint(f"  Columns: {list(df.columns)[:10]}")

        # Extract key columns (column names may vary)
        # Common formats: x, y, z or xnm, ynm, znm
        result = {}

        coordinate_scales = {}

        # Extract coordinates (prioritize nm columns!)
        for key_base in ["x", "y", "z"]:
            # Try nm columns first, then pixel columns
            for variant in [
                f"{key_base}_nm",
                f"{key_base}nm",
                f"{key_base} [nm]",
                f"{key_base}_pix",
                key_base,
            ]:
                if variant in df.columns:
                    values = df[variant].values
                    scale = PIXEL_SIZE if "_pix" in variant else 1.0
                    values = values * scale
                    result[key_base] = values
                    coordinate_scales[key_base] = scale
                    break

        # Precision/uncertainty (CRLB = Cramér-Rao Lower Bound)
        for key_base in ["x", "y", "z"]:
            # Try various precision column names
            for variant in [
                f"crlb_{key_base}nm",
                f"crlb_{key_base}",
                f"precision_{key_base}",
                f"sigma_{key_base}",
            ]:
                if variant in df.columns:
                    values = df[variant].values
                    if variant.startswith("crlb_") and "nm" not in variant:
                        values = values * coordinate_scales.get(key_base, 1.0)
                    result[f"precision_{key_base}"] = values
                    break

        # Additional attributes
        if "photons" in df.columns:
            result["photons"] = df["photons"].values
        if "intensity" in df.columns:
            result["intensity"] = df["intensity"].values
        if "frame" in df.columns:
            result["frame"] = df["frame"].values

        # Filter out NaN values (some localizations have bad data)
        if "x" in result and "y" in result and "z" in result:
            valid_mask = ~(
                np.isnan(result["x"]) | np.isnan(result["y"]) | np.isnan(result["z"])
            )
            # Also filter NaN precision values
            for prec_key in ["precision_x", "precision_y", "precision_z"]:
                if prec_key in result:
                    valid_mask &= ~np.isnan(result[prec_key])
            n_invalid = (~valid_mask).sum()
            if n_invalid > 0:
                aprint(f"  Filtering {n_invalid:,} invalid (NaN) localizations")
                for key in result:
                    result[key] = result[key][valid_mask]
                aprint(f"  Kept {len(result['x']):,} valid localizations")

        aprint("✓ Extracted columns:")
        for key in ["x", "y", "z"]:
            if key in result:
                aprint(
                    f"  {key}: [{result[key].min():.1f}, {result[key].max():.1f}] nm"
                )
        for key in ["precision_x", "precision_y", "precision_z"]:
            if key in result:
                aprint(f"  {key}: median = {np.median(result[key]):.1f} nm")

    return result


def extract_centers_and_amplitudes(
    localizations: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract centers, amplitudes, and per-localization precision.

    Args:
        localizations: Dictionary with x, y, z, photons, and (optionally) the
            per-axis localization precision ``precision_x/y/z`` (CRLB, in nm).

    Returns:
        Tuple ``(centers_um, amplitudes, precision_um)`` where centers are in
        micrometers and ``precision_um`` is an ``(N, 3)`` array of per-axis
        uncertainty (σx, σy, σz) in micrometers, combining the localization CRLB
        and label-linkage error in quadrature. ``precision_um`` is ``None`` when
        the dataset carries no CRLB columns (caller falls back to a fixed sigma).
    """
    with asection("Extracting splat data from localizations"):
        n_loc = len(localizations["x"])
        aprint(f"Processing {n_loc:,} localizations...")

        # Centers in micrometers (nm -> μm)
        centers_um = np.column_stack(
            [
                localizations["x"] / 1000,
                localizations["y"] / 1000,
                localizations["z"] / 1000,
            ]
        ).astype(np.float32)

        # Amplitudes from photon counts (or uniform if not available)
        if "photons" in localizations:
            amplitudes = localizations["photons"].astype(np.float32)
            # Normalize to reasonable range
            amplitudes = amplitudes / np.percentile(amplitudes, 99) * 0.5
        elif "intensity" in localizations:
            amplitudes = localizations["intensity"].astype(np.float32)
            amplitudes = amplitudes / np.percentile(amplitudes, 99) * 0.5
        else:
            amplitudes = np.ones(n_loc, dtype=np.float32) * 0.3

        amplitudes = np.clip(amplitudes, 0.01, 1.0)

        # CRLB is only the localization-estimator uncertainty. The fluorophore
        # is displaced from alpha-tubulin by the primary + secondary antibody
        # sandwich, so the physical uncertainty is their independent error
        # budget in quadrature. The 17 nm linkage term removes the arbitrary
        # lower floor while preserving genuine per-localization variation.
        precision_um: np.ndarray | None = None
        if all(f"precision_{a}" in localizations for a in "xyz"):
            crlb_nm = np.column_stack(
                [localizations[f"precision_{a}"] for a in "xyz"]
            ).astype(np.float32)
            crlb_nm = np.maximum(crlb_nm, 0.0)
            precision_nm = np.hypot(crlb_nm, LABEL_LINKAGE_SIGMA_NM)
            precision_nm = np.minimum(precision_nm, 150.0)
            precision_um = precision_nm / 1000.0
            aprint(
                "  Using CRLB + antibody-linkage uncertainty: "
                f"median σ = [{np.median(precision_nm, axis=0).round(1)}] nm"
            )
        else:
            aprint(
                "  No CRLB precision columns — super-res will use a fixed "
                f"{SUPERRES_PSF_SIGMA} nm sigma"
            )

        aprint(f"✓ Extracted {n_loc:,} localizations")
        aprint(f"  Centers: {centers_um.shape}")
        aprint(
            f"  Spatial range: [{centers_um.min(axis=0)}] to [{centers_um.max(axis=0)}] μm"
        )

    return centers_um, amplitudes, precision_um


def rasterize_widefield_volume(
    centers_um: np.ndarray,
    photon_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize photon-weighted localizations and convolve with the PSF.

    Args:
        centers_um: Centered localization coordinates in ``(x, y, z)`` order.
        photon_weights: Raw detected photons per localization.

    Returns:
        ``(volume, origin_zyx)`` where the volume is in ``(z, y, x)`` order and
        ``origin_zyx`` maps voxel index zero back into centered scene space.
    """
    centers_um = np.asarray(centers_um, dtype=np.float32)
    photon_weights = np.asarray(photon_weights, dtype=np.float32)
    if centers_um.ndim != 2 or centers_um.shape[1] != 3:
        raise ValueError(f"centers_um must have shape (N, 3), got {centers_um.shape}")
    if photon_weights.shape != (len(centers_um),):
        raise ValueError(
            "photon_weights must have one value per center, got "
            f"{photon_weights.shape} for {len(centers_um)} centers"
        )
    if len(centers_um) == 0:
        raise ValueError("cannot rasterize an empty localization set")
    if not np.all(np.isfinite(centers_um)) or not np.all(np.isfinite(photon_weights)):
        raise ValueError("centers_um and photon_weights must be finite")
    if np.any(photon_weights < 0):
        raise ValueError("photon_weights must be non-negative")

    gaussian_filter = require_module("scipy.ndimage").gaussian_filter
    centers_zyx = centers_um[:, ::-1]
    sigma_um = np.array(
        [
            WIDEFIELD_AXIAL_PSF_SIGMA_NM / 1000.0,
            WIDEFIELD_PSF_SIGMA_NM / 1000.0,
            WIDEFIELD_PSF_SIGMA_NM / 1000.0,
        ],
        dtype=np.float64,
    )
    padding_um = 4.0 * sigma_um
    origin_zyx = (
        np.floor((centers_zyx.min(axis=0) - padding_um) / WIDEFIELD_VOXEL_SIZE_UM)
        * WIDEFIELD_VOXEL_SIZE_UM
    )
    upper_zyx = centers_zyx.max(axis=0) + padding_um
    shape = np.ceil((upper_zyx - origin_zyx) / WIDEFIELD_VOXEL_SIZE_UM).astype(int) + 1
    indices = np.rint((centers_zyx - origin_zyx) / WIDEFIELD_VOXEL_SIZE_UM).astype(
        np.int64
    )
    indices = np.clip(indices, 0, shape - 1)

    volume = np.zeros(tuple(shape), dtype=np.float32)
    np.add.at(volume, tuple(indices.T), photon_weights)
    volume = gaussian_filter(
        volume,
        sigma=sigma_um / WIDEFIELD_VOXEL_SIZE_UM,
        mode="reflect",
    ).astype(np.float32, copy=False)
    return volume, origin_zyx.astype(np.float32)


def fit_widefield_gsplats(
    centers_um: np.ndarray,
    photon_weights: np.ndarray,
    cache_file: Path,
    *,
    recompute: bool = False,
):
    """Fit and cache a compact gsplat basis for the synthetic widefield image."""
    from luxar.gsplats.gsplat_data import GSplatData

    if cache_file.exists() and not recompute:
        with asection("Loading cached widefield fit"):
            result = GSplatData.load(cache_file, include_stats=False)
            aprint(f"✓ Loaded {len(result.amplitudes):,} fitted splats")
            return result

    from luxar.demos import detect_device
    from luxar.demos._lod_policy import save_with_lod
    from luxar.encoding import EncodingMode
    from luxar.gsplats import fit_tiled

    volume, origin_zyx = rasterize_widefield_volume(centers_um, photon_weights)
    peak = float(volume.max())
    if peak <= 0:
        raise ValueError("widefield photon raster contains no positive signal")
    volume /= peak
    device = detect_device()

    with asection("Fitting photon-weighted widefield volume"):
        aprint(f"  Volume (z,y,x): {volume.shape}")
        aprint(f"  Voxel size: {WIDEFIELD_VOXEL_SIZE_UM} μm")
        aprint(f"  Device: {device}")
        result = fit_tiled(
            volume,
            tile_size=WIDEFIELD_TILE_SIZE,
            overlap=WIDEFIELD_OVERLAP,
            seeds=WIDEFIELD_SEEDS_PER_TILE,
            n_iters=WIDEFIELD_N_ITERS,
            cull_retention=WIDEFIELD_CULL_RETENTION,
            device=device,
            voxel_size=WIDEFIELD_VOXEL_SIZE_UM,
            output_space=WIDEFIELD_OUTPUT_SPACE,
            floor=WIDEFIELD_FLOOR,
            source_shape=volume.shape,
            source_dtype=str(volume.dtype),
            verbose=True,
            enable_dynamic_ops=WIDEFIELD_ENABLE_DYNAMIC_OPS,
        ).translate(origin_zyx)
        aprint(f"✓ Fitted {len(result.amplitudes):,} widefield splats")
        save_with_lod(
            result,
            cache_file,
            recipe="stream",
            encoding_mode=EncodingMode.MEMORY,
            include_fitting_info=True,
            compress="zip",
            zip_deflate=True,
        )
        aprint(f"✓ Cached {cache_file.name}")
    return GSplatData.load(cache_file, include_stats=False)


def _make_superresolution_gsplats(
    centers_um: np.ndarray,
    amplitudes: np.ndarray,
    precision_um: np.ndarray | None,
):
    """Build epistemic per-localization splats in three spatial dimensions."""
    from luxar.gsplats.gsplat_data import GSplatData

    n_splats = len(centers_um)
    cholesky = np.zeros((n_splats, 6), dtype=np.float32)
    if precision_um is not None:
        sigma_um = (precision_um * VIS_SCALE).astype(np.float32)
        cholesky[:, 0] = sigma_um[:, 0]
        cholesky[:, 2] = sigma_um[:, 1]
        cholesky[:, 5] = sigma_um[:, 2]
    else:
        sigma_um = np.empty((n_splats, 3), dtype=np.float32)
        sigma_um[:, :2] = (SUPERRES_PSF_SIGMA * VIS_SCALE) / 1000.0
        sigma_um[:, 2] = (SUPERRES_PSF_SIGMA * VIS_SCALE * 2) / 1000.0
        cholesky[:, 0] = sigma_um[:, 0]
        cholesky[:, 2] = sigma_um[:, 1]
        cholesky[:, 5] = sigma_um[:, 2]
    colors = np.tile(np.array([0.2, 1.0, 1.0], dtype=np.float32), (n_splats, 1))
    return GSplatData(
        centers=np.asarray(centers_um, dtype=np.float32),
        cholesky_factors=cholesky,
        amplitudes=np.asarray(amplitudes, dtype=np.float32),
        colors=colors,
    )


# =============================================================================
# Scene Creation
# =============================================================================


def create_storm_scene(
    centered_centers_um: np.ndarray,
    amplitudes: np.ndarray,
    widefield_gsplats,
    precision_um: np.ndarray | None = None,
    output_path: Path | None = None,
) -> Path:
    """Create Luxar scene with STORM data comparing widefield vs super-resolution.

    Two views on a categorical ``view`` dimension:
    - Widefield: a compact fitted basis for the photon-weighted localization
      raster after convolution with the diffraction-limited PSF.
    - Super-resolution: each splat's covariance is the localization's OWN
      anisotropic error budget (CRLB plus label linkage in ``precision_um``).

    Args:
        centered_centers_um: Localization centers in micrometers (N, 3), already
            shifted into the same centered coordinate frame as ``widefield_gsplats``.
        amplitudes: Splat amplitudes (N,)
        widefield_gsplats: Fitted 3D ``GSplatData`` in ``(z, y, x)`` order.
        precision_um: Per-localization precision (N, 3) in μm, or None for a
            fixed super-resolution sigma.
        output_path: Optional path to save the scene (default: demos directory)

    Returns:
        Path to output scene
    """
    # Use provided path or save to demos directory
    if output_path is None:
        output_path = get_demos_output_dir() / "storm_3d_microtubules.luxar.zarr"

    with asection("Creating Luxar scene"):
        # 4D scene: VIEW (categorical) + X, Y, Z (spatial)
        # VIEW dimension: 0 = widefield (blurry), 1 = super-resolution (sharp)
        dims = Dimensions(
            [
                Dimension(
                    "view",
                    unit="",
                    display=False,
                    categories=["Widefield", "Super-resolution"],  # Named categories!
                ),
                Dimension("x", unit="μm", display=True),
                Dimension("y", unit="μm", display=True),
                Dimension("z", unit="μm", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Add metadata (keep simple for JSON compatibility)
            scene.attrs["title"] = "3D STORM Microtubule Network"

            with asection("Adding fitted and measured microscopy views"):
                widefield_max = float(widefield_gsplats.amplitudes.max())
                if widefield_max > 0:
                    widefield_gsplats = widefield_gsplats.scale_intensity(
                        0.3 / widefield_max
                    )
                superresolution_gsplats = _make_superresolution_gsplats(
                    centered_centers_um,
                    amplitudes,
                    precision_um,
                )

                scene.add_gsplats_from_data(
                    name="widefield_fit",
                    result=widefield_gsplats,
                    dim_order=["z", "y", "x"],
                    fill={"view": 0.0},
                    fill_sigma={"view": 0.1},
                    extend_to_all=[],
                    opacity=0.8,
                    absorption=1.0,
                    blending_mode="volumetric",
                    colormap="gray",
                    layer=True,
                )
                scene.add_gsplats_from_data(
                    name="superresolution_localizations",
                    result=superresolution_gsplats,
                    dim_order=["x", "y", "z"],
                    fill={"view": 1.0},
                    fill_sigma={"view": 0.1},
                    extend_to_all=[],
                    opacity=0.8,
                    absorption=1.0,
                    blending_mode="volumetric",
                    layer=True,
                )
                aprint(
                    "✓ Added independently sized views: "
                    f"{len(widefield_gsplats.amplitudes):,} fitted widefield / "
                    f"{len(superresolution_gsplats.amplitudes):,} measured super-res"
                )

            # --- Overlays ---
            # Title
            scene.add_text(
                "3D STORM Microtubules",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Dimension-aware view mode labels
            scene.add_html(
                '<div style="font-size:1.5vh;font-weight:bold;color:#88bbff">Widefield</div>'
                f'<div style="font-size:1.3vh;color:#aaa">fitted after {WIDEFIELD_PSF_SIGMA_NM:.0f}/{WIDEFIELD_AXIAL_PSF_SIGMA_NM:.0f} nm PSF blur</div>',
                position=(0.02, 0.97),
                anchor="bottom-left",
                visible_range={"view": 0},
                transition="fade",
                transition_duration=0.3,
            )
            scene.add_html(
                '<div style="font-size:1.5vh;font-weight:bold;color:#44ff88">Super-Resolution</div>'
                f'<div style="font-size:1.3vh;color:#aaa">\u03c3 = CRLB \u2295 {LABEL_LINKAGE_SIGMA_NM:.0f} nm label linkage</div>',
                position=(0.02, 0.97),
                anchor="bottom-left",
                visible_range={"view": 1},
                transition="fade",
                transition_duration=0.3,
            )

            # Sample info (always visible)
            scene.add_text(
                "COS7 \u03b1-tubulin \u2022 Zenodo 3547521",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint("✓ Scene saved")

    return output_path


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    field = DEFAULT_FIELD
    max_loc = DEFAULT_MAX_LOCALIZATIONS
    recompute = "--recompute" in sys.argv

    for arg in sys.argv[1:]:
        if arg.startswith("--field="):
            field = int(arg.split("=")[1])
        elif arg.startswith("--max-localizations="):
            max_loc = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("3D STORM SUPER-RESOLUTION MICROSCOPY")
    aprint("=" * 70)
    aprint("")
    aprint("Microtubule network at nanometer resolution!")
    aprint("")
    aprint("Dataset:")
    aprint("  • Source: Zenodo 3547521 (3D STORM COS7 Alpha-Tubulin)")
    aprint("  • Sample: COS7 cells (monkey kidney fibroblasts)")
    aprint("  • Target: Alpha-tubulin (microtubule protein)")
    aprint("  • Method: 3D STORM with astigmatism")
    aprint("")
    aprint("Resolution:")
    aprint("  • Conventional microscopy: ~250 nm (diffraction limit)")
    aprint("  • STORM super-resolution: ~20 nm lateral, ~50 nm axial")
    aprint("  • Improvement: 10-12x better!")
    aprint("")
    aprint("Parameters:")
    aprint(f"  • Field of view: {field}")
    aprint(f"  • Max localizations: {max_loc:,}")
    aprint("")

    try:
        # Download localizations
        csv_file = download_storm_localizations(field=field)

        # Parse localizations
        localizations = parse_storm_localizations(csv_file, max_localizations=max_loc)

        # The measured localization amplitudes stay separately authored for the
        # epistemic super-resolution view. The widefield image is integrated
        # from raw photon counts before fitting its compact Gaussian basis.
        centers_um, amplitudes, precision_um = extract_centers_and_amplitudes(
            localizations
        )
        if "photons" in localizations:
            photon_weights = localizations["photons"].astype(np.float32)
        elif "intensity" in localizations:
            photon_weights = localizations["intensity"].astype(np.float32)
        else:
            photon_weights = np.ones(len(centers_um), dtype=np.float32)

        total_photons = float(photon_weights.sum())
        if total_photons > 0:
            centroid_um = (centers_um.T @ photon_weights) / total_photons
        else:
            centroid_um = centers_um.mean(axis=0)
        centered_centers_um = centers_um - centroid_um
        aprint(f"Centered both views at photon-weighted COM: {centroid_um.round(3)} μm")

        widefield_gsplats = fit_widefield_gsplats(
            centered_centers_um,
            photon_weights,
            _widefield_cache_path(csv_file, max_loc),
            recompute=recompute,
        )

        # If --no-serve, generate and exit without launching viewer
        if "--no-serve" in sys.argv:
            output_path = get_demos_output_dir() / "storm_3d_microtubules.luxar.zarr"
            scene_path = create_storm_scene(
                centered_centers_um,
                amplitudes,
                widefield_gsplats,
                precision_um,
                output_path=output_path,
            )
            aprint(f"Dataset generated at {scene_path}")
            aprint(f"Localizations: {len(centers_um):,}")
            return

        # Create scene in demos directory for serving
        scene_path = create_storm_scene(
            centered_centers_um,
            amplitudes,
            widefield_gsplats,
            precision_um,
        )

        # Stats
        aprint("")
        aprint("=" * 70)
        aprint("STORM VISUALIZATION COMPLETE")
        aprint("=" * 70)
        aprint(f"Localizations: {len(centers_um):,}")
        aprint(
            f"Microtubule network visible in {len(centers_um):,} molecular detections!"
        )
        aprint("")
        aprint("In the viewer:")
        aprint("  - Press '1' to select VIEW dimension")
        aprint("  - Press '['/']' to toggle Widefield / Super-Resolution")
        aprint("  - Widefield: Blurry ~250 nm resolution (gray)")
        aprint("  - Super-res: Sharp ~20 nm resolution (cyan splats)")
        aprint("  - Zoom in to see individual microtubules!")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Explore the nanoscale world!")
        aprint("Press Ctrl+C when done.")
        aprint("")

        launch_viewer(scene_path)

    except Exception as e:
        aprint(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
