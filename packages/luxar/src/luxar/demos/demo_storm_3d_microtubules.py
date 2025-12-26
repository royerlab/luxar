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
Each localization is NATURALLY a Gaussian splat!
- **Center**: Detected molecule position (x, y, z)
- **Covariance**: Localization precision (σx, σy, σz)
- **Amplitude**: Photon count or intensity
- **Sharpness**: Can represent localization quality

This demo shows:
1. **Super-resolved view**: Each localization as a Gaussian splat
2. **Widefield comparison**: Synthetic conventional microscopy
3. **Categorical dimension**: Toggle between widefield ↔ super-resolution

DATA SOURCE:
============
Zenodo: 3D STORM Dataset - COS7 Cells, Alpha-Tubulin
https://zenodo.org/record/3547521

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
    --max-localizations=N   Limit number of localizations (default: 1M)
    --no-cache              Force re-download and re-processing
    --no-serve              Generate dataset without launching viewer
    --field=N               Select field of view (default: 4)

Controls:
    - Press '1' to select VIEW dimension
    - Press '['/']' to toggle: Widefield ↔ Super-Resolution
    - Rotate to explore 3D microtubule network
    - Zoom in to see individual molecules!
    - Ctrl+C to stop and cleanup
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import requests
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler

# =============================================================================
# Configuration
# =============================================================================

# Zenodo dataset
ZENODO_RECORD = "3547521"
ZENODO_BASE_URL = f"https://zenodo.org/record/{ZENODO_RECORD}/files"

# Default parameters
DEFAULT_FIELD = 4  # Field of view number
DEFAULT_MAX_LOCALIZATIONS = 1_000_000  # Limit for demo performance

# Visualization parameters
WIDEFIELD_PSF_SIGMA = 150.0  # nm - conventional microscopy PSF width
PIXEL_SIZE = 106.0  # nm - from dataset metadata
SUPER_RES_PRECISION_SCALE = 1.5  # Scale factor for localization precision

# Cache paths
CACHE_DIR = Path.home() / ".cache" / "luxar" / "storm_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
                    "-C", "-",  # Resume from partial
                    "--retry", "10",
                    "--retry-delay", "5",
                    "--max-time", "3600",
                    "-o", str(cache_file),
                    "-#",  # Progress bar
                    url
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
                aprint(f"Found partial download ({resume_pos / (1024**2):.1f} MB), resuming...")

            # Robust download with resume support
            max_retries = 5
            retry_delay = 10

            for attempt in range(max_retries):
                try:
                    # Request with resume support + browser headers
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                    }
                    if resume_pos > 0:
                        headers['Range'] = f'bytes={resume_pos}-'

                    response = requests.get(url, headers=headers, stream=True, timeout=300)

                    # Handle rate limiting
                    if response.status_code == 429:
                        if attempt < max_retries - 1:
                            aprint(f"⚠️  Rate limited, waiting {retry_delay}s (attempt {attempt + 1}/{max_retries})...")
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
                    content_range = response.headers.get('Content-Range')
                    if content_range:
                        total_size = int(content_range.split('/')[-1])
                    else:
                        total_size = int(response.headers.get("content-length", 0)) + resume_pos

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
                                    percent = (downloaded / total_size * 100) if total_size > 0 else 0
                                    aprint(f"  Progress: {downloaded / (1024**2):.0f} / {total_size / (1024**2):.0f} MB ({percent:.0f}%)")
                                    last_progress = downloaded

                    # Download complete - move temp to final
                    temp_file.rename(cache_file)
                    aprint(f"✓ Downloaded to {cache_file}")
                    aprint(f"  Final size: {cache_file.stat().st_size / (1024**2):.1f} MB")
                    break  # Success!

                except (requests.exceptions.ChunkedEncodingError,
                        requests.exceptions.ConnectionError) as e:
                    if attempt < max_retries - 1:
                        aprint(f"⚠️  Download interrupted ({e})")
                        aprint(f"   Retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})...")
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
    import pandas as pd

    with asection("Parsing STORM localizations"):
        aprint(f"CSV file: {csv_path.name}")
        aprint("Reading CSV (may take 30-60 seconds for large files)...")

        # Read CSV (may have millions of rows!)
        if max_localizations:
            df = pd.read_csv(csv_path, nrows=max_localizations)
            aprint(f"✓ Loaded {len(df):,} localizations (limited to {max_localizations:,})")
        else:
            df = pd.read_csv(csv_path)
            aprint(f"✓ Loaded {len(df):,} total localizations")

        aprint(f"  Columns: {list(df.columns)[:10]}")

        # Extract key columns (column names may vary)
        # Common formats: x, y, z or xnm, ynm, znm
        result = {}

        # Try different column name conventions
        for key_base in ['x', 'y', 'z']:
            # Try: x, xnm, x [nm], x_nm, etc.
            for variant in [key_base, f'{key_base}nm', f'{key_base} [nm]', f'{key_base}_nm']:
                if variant in df.columns:
                    result[key_base] = df[variant].values
                    break

        # Precision/uncertainty
        for key_base in ['x', 'y', 'z']:
            for variant in [f'precision_{key_base}', f'{key_base}_precision', f'sigma_{key_base}', f'{key_base}_std']:
                if variant in df.columns:
                    result[f'precision_{key_base}'] = df[variant].values
                    break

        # Additional attributes
        if 'photons' in df.columns:
            result['photons'] = df['photons'].values
        if 'intensity' in df.columns:
            result['intensity'] = df['intensity'].values
        if 'frame' in df.columns:
            result['frame'] = df['frame'].values

        aprint("✓ Extracted columns:")
        for key in ['x', 'y', 'z']:
            if key in result:
                aprint(f"  {key}: [{result[key].min():.1f}, {result[key].max():.1f}] nm")
        for key in ['precision_x', 'precision_y', 'precision_z']:
            if key in result:
                aprint(f"  {key}: median = {np.median(result[key]):.1f} nm")

    return result


def create_gsplats_from_localizations(
    localizations: dict,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create Gaussian splats from STORM localizations.

    Args:
        localizations: Dictionary with x, y, z, precision, etc.
        cache_path: Optional path to cache processed splats

    Returns:
        Tuple of (centers, covariances, amplitudes, sharpnesses)
    """
    # Check cache first
    if cache_path and cache_path.exists():
        with asection("Loading cached splat data"):
            aprint(f"Cache: {cache_path}")
            cached = np.load(cache_path)
            centers = cached["centers"]
            covariances = cached["covariances"]
            amplitudes = cached["amplitudes"]
            sharpnesses = cached["sharpnesses"]
            aprint(f"✓ Loaded {len(centers):,} cached splats")
            return centers, covariances, amplitudes, sharpnesses

    with asection("Creating Gaussian splats from localizations"):
        n_loc = len(localizations['x'])
        aprint(f"Processing {n_loc:,} localizations...")

        # Centers (convert to voxel coordinates if in nm)
        centers = np.column_stack([
            localizations['x'] / PIXEL_SIZE,
            localizations['y'] / PIXEL_SIZE,
            localizations['z'] / PIXEL_SIZE,
        ]).astype(np.float32)

        # Covariance matrices from localization precision
        # Each localization has uncertainty → diagonal covariance
        covariances = []
        for i in range(n_loc):
            # Get precision (uncertainty) in each dimension
            sigma_x = localizations.get('precision_x', np.full(n_loc, 20.0))[i] / PIXEL_SIZE
            sigma_y = localizations.get('precision_y', np.full(n_loc, 20.0))[i] / PIXEL_SIZE
            sigma_z = localizations.get('precision_z', np.full(n_loc, 50.0))[i] / PIXEL_SIZE

            # Scale up for visibility
            sigma_x *= SUPER_RES_PRECISION_SCALE
            sigma_y *= SUPER_RES_PRECISION_SCALE
            sigma_z *= SUPER_RES_PRECISION_SCALE

            # Create diagonal covariance matrix (3x3)
            cov = np.diag([sigma_x**2, sigma_y**2, sigma_z**2])
            covariances.append(cov)

        covariances = np.array(covariances, dtype=np.float32)

        # Amplitudes from photon counts (or uniform if not available)
        if 'photons' in localizations:
            amplitudes = localizations['photons'].astype(np.float32)
            # Normalize to reasonable range
            amplitudes = amplitudes / np.percentile(amplitudes, 99) * 0.5
        elif 'intensity' in localizations:
            amplitudes = localizations['intensity'].astype(np.float32)
            amplitudes = amplitudes / np.percentile(amplitudes, 99) * 0.5
        else:
            amplitudes = np.ones(n_loc, dtype=np.float32) * 0.3

        amplitudes = np.clip(amplitudes, 0.01, 1.0)

        # Sharpness (higher = sharper, represents good localization)
        sharpnesses = np.full(n_loc, 3.0, dtype=np.float32)  # Sharp splats

        aprint(f"✓ Created {n_loc:,} Gaussian splats")
        aprint(f"  Centers: {centers.shape}")
        aprint(f"  Covariances: {covariances.shape}")
        aprint(f"  Spatial range: {centers.min(axis=0)} to {centers.max(axis=0)}")

        # Cache for future runs
        if cache_path:
            np.savez(
                cache_path,
                centers=centers,
                covariances=covariances,
                amplitudes=amplitudes,
                sharpnesses=sharpnesses,
            )
            aprint(f"✓ Cached splat data to {cache_path}")

    return centers, covariances, amplitudes, sharpnesses


def generate_synthetic_widefield(
    localizations: dict,
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    """Generate synthetic widefield image from localizations.

    Simulates conventional (diffraction-limited) microscopy by blurring
    localizations with a large PSF.

    Args:
        localizations: STORM localizations
        volume_shape: Output volume shape

    Returns:
        3D widefield volume
    """
    with asection("Generating synthetic widefield comparison"):
        aprint("Simulating conventional microscopy (diffraction-limited)...")
        aprint(f"  PSF sigma: {WIDEFIELD_PSF_SIGMA} nm")

        # Create volume
        volume = np.zeros(volume_shape, dtype=np.float32)

        # Convert to voxel coordinates
        x_voxel = (localizations['x'] / PIXEL_SIZE).astype(int)
        y_voxel = (localizations['y'] / PIXEL_SIZE).astype(int)
        z_voxel = (localizations['z'] / PIXEL_SIZE).astype(int)

        # Clip to volume bounds
        valid = (
            (x_voxel >= 0) & (x_voxel < volume_shape[0]) &
            (y_voxel >= 0) & (y_voxel < volume_shape[1]) &
            (z_voxel >= 0) & (z_voxel < volume_shape[2])
        )

        x_voxel = x_voxel[valid]
        y_voxel = y_voxel[valid]
        z_voxel = z_voxel[valid]

        # Accumulate localizations
        for x, y, z in zip(x_voxel, y_voxel, z_voxel):
            volume[x, y, z] += 1

        # Blur with large Gaussian (conventional microscopy PSF)
        from scipy.ndimage import gaussian_filter
        sigma_voxels = WIDEFIELD_PSF_SIGMA / PIXEL_SIZE
        volume = gaussian_filter(volume, sigma=[sigma_voxels, sigma_voxels, sigma_voxels * 2])

        # Normalize
        if volume.max() > 0:
            volume = volume / volume.max()

        aprint(f"✓ Generated {volume_shape} widefield volume")
        aprint(f"  Range: [{volume.min():.3f}, {volume.max():.3f}]")

    return volume


# =============================================================================
# Scene Creation
# =============================================================================


def create_storm_scene(
    centers: np.ndarray,
    covariances: np.ndarray,
    amplitudes: np.ndarray,
    sharpnesses: np.ndarray,
    widefield_volume: np.ndarray | None = None,
) -> Path:
    """Create Luxar scene with STORM data and optional widefield comparison.

    Args:
        centers: Splat centers
        covariances: Splat covariance matrices
        amplitudes: Splat amplitudes
        sharpnesses: Splat sharpnesses
        widefield_volume: Optional synthetic widefield volume

    Returns:
        Path to output scene
    """
    # Save to examples directory (permanent location)
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    output_path = examples_dir / "storm_3d_microtubules_example.zarr"

    with asection("Creating Luxar scene"):
        # Create dimensions with categorical view toggle
        dims = Dimensions([
            Dimension(
                "view",
                unit="",
                categories=["Conventional (Widefield)", "Super-Resolution (STORM)"],
                display=False,
                description="Microscopy mode: toggle between diffraction-limited and super-resolution",
            ),
            Dimension("x", unit="μm", display=True),
            Dimension("y", unit="μm", display=True),
            Dimension("z", unit="μm", display=True),
        ])

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Add metadata (keep simple for JSON compatibility)
            scene.attrs["title"] = "3D STORM Microtubule Network"

            # Create BOTH views as gsplats with different PSF sizes
            with asection("Adding both microscopy views as gsplats"):
                from luxar.gsplats.fit_result import GSplatData

                # Convert centers to μm
                centers_um = centers * PIXEL_SIZE / 1000

                # Create TWO sets of gsplats - same positions, different covariances!
                n_splats = len(centers)

                # Prepare storage for both views
                all_centers = []
                all_cholesky = []
                all_amplitudes = []
                all_sharpnesses = []
                all_colors = []

                aprint(f"Creating {n_splats:,} splats × 2 views...")

                # VIEW 0: Widefield (large PSF ~150 nm)
                widefield_psf_um = WIDEFIELD_PSF_SIGMA / 1000  # nm to μm
                widefield_cov = np.diag([widefield_psf_um**2, widefield_psf_um**2, widefield_psf_um**2 * 4])

                for i in range(n_splats):
                    # Add view dimension coordinate (0 = widefield)
                    center_4d = np.array([0.0, centers_um[i, 0], centers_um[i, 1], centers_um[i, 2]], dtype=np.float32)
                    all_centers.append(center_4d)

                    # Large covariance for widefield
                    # 4D: add zero variance in view dimension
                    cov_4d = np.zeros((4, 4), dtype=np.float32)
                    cov_4d[1:, 1:] = widefield_cov  # Spatial part
                    cov_4d[0, 0] = 1e-6  # Tiny variance in view dimension

                    L = np.linalg.cholesky(cov_4d)
                    # Pack 4D Cholesky: [L00, L10, L11, L20, L21, L22, L30, L31, L32, L33]
                    chol = [L[0,0], L[1,0], L[1,1], L[2,0], L[2,1], L[2,2], L[3,0], L[3,1], L[3,2], L[3,3]]
                    all_cholesky.append(chol)

                    all_amplitudes.append(amplitudes[i] * 0.5)  # Dim for widefield
                    all_sharpnesses.append(1.0)  # Soft
                    all_colors.append([128, 128, 128])  # Gray

                # VIEW 1: Super-resolution (small PSF ~20 nm from precision)
                for i in range(n_splats):
                    # Add view dimension coordinate (1 = super-res)
                    center_4d = np.array([1.0, centers_um[i, 0], centers_um[i, 1], centers_um[i, 2]], dtype=np.float32)
                    all_centers.append(center_4d)

                    # Small covariance from localization precision
                    cov_3d = covariances[i] * (PIXEL_SIZE / 1000)**2
                    cov_4d = np.zeros((4, 4), dtype=np.float32)
                    cov_4d[1:, 1:] = cov_3d  # Spatial part
                    cov_4d[0, 0] = 1e-6  # Tiny variance in view dimension

                    L = np.linalg.cholesky(cov_4d)
                    chol = [L[0,0], L[1,0], L[1,1], L[2,0], L[2,1], L[2,2], L[3,0], L[3,1], L[3,2], L[3,3]]
                    all_cholesky.append(chol)

                    all_amplitudes.append(amplitudes[i])
                    all_sharpnesses.append(3.0)  # Sharp
                    all_colors.append([76, 230, 230])  # Cyan

                # Create combined GSplatData
                gsplat_data = GSplatData(
                    centers=np.array(all_centers, dtype=np.float32),
                    cholesky_factors=np.array(all_cholesky, dtype=np.float32),
                    amplitudes=np.array(all_amplitudes, dtype=np.float32),
                    sharpnesses=np.array(all_sharpnesses, dtype=np.float32),
                    colors=np.array(all_colors, dtype=np.uint8),
                )

                scene.add_gsplats_from_data(
                    name="microtubules",
                    result=gsplat_data,
                    opacity=0.8,
                    blending_mode="additive",
                )

                aprint(f"✓ Added {n_splats * 2:,} splats (2 views × {n_splats:,})")

        aprint("✓ Scene saved")

    return output_path


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    field = DEFAULT_FIELD
    max_loc = DEFAULT_MAX_LOCALIZATIONS
    no_cache = "--no-cache" in sys.argv

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

    # Check dependencies
    try:
        import pandas  # noqa: F401
        from scipy.ndimage import gaussian_filter  # noqa: F401
    except ImportError as e:
        aprint(f"❌ Missing dependency: {e}")
        aprint("")
        aprint("Install with:")
        aprint("  pip install pandas scipy")
        sys.exit(1)

    # Setup caching
    splats_cache = CACHE_DIR / f"FOV_{field}_splats_{max_loc}.npz"

    if no_cache:
        if splats_cache.exists():
            splats_cache.unlink()

    try:
        # Download localizations
        csv_file = download_storm_localizations(field=field)

        # Parse localizations
        localizations = parse_storm_localizations(csv_file, max_localizations=max_loc)

        # Create splats
        centers, covariances, amplitudes, sharpnesses = create_gsplats_from_localizations(
            localizations,
            cache_path=splats_cache,
        )

        # Generate synthetic widefield for comparison
        # Estimate volume bounds
        x_max = int(np.ceil(centers[:, 0].max())) + 10
        y_max = int(np.ceil(centers[:, 1].max())) + 10
        z_max = int(np.ceil(centers[:, 2].max())) + 10
        volume_shape = (x_max, y_max, z_max)

        widefield = generate_synthetic_widefield(localizations, volume_shape)

        # Create scene
        scene_path = create_storm_scene(
            centers, covariances, amplitudes, sharpnesses,
            widefield_volume=widefield,
        )

        # Stats
        aprint("")
        aprint("=" * 70)
        aprint("STORM VISUALIZATION COMPLETE")
        aprint("=" * 70)
        aprint(f"Localizations: {len(centers):,}")
        aprint(f"Volume bounds: {volume_shape}")
        aprint(f"Microtubule network visible in {len(centers):,} molecular detections!")
        aprint("")
        aprint("In the viewer:")
        aprint("  • Press '1' to select VIEW dimension")
        aprint("  • Press '['/']' to toggle Widefield ↔ Super-Resolution")
        aprint("  • Widefield: Blurry ~250 nm resolution (gray)")
        aprint("  • Super-res: Sharp ~20 nm resolution (cyan splats)")
        aprint("  • Zoom in to see individual microtubules!")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Explore the nanoscale world!")
        aprint("Press Ctrl+C when done.")
        aprint("")

        if "--no-serve" in sys.argv:
            aprint("✓ Dataset generated successfully (--no-serve mode)")
            return

        subprocess.run(
            ["luxar", "serve", str(scene_path), "--viewer", "--open"],
            check=True,
        )

    except KeyboardInterrupt:
        aprint("\n🛑 Stopping demo...")
    except Exception as e:
        aprint(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    aprint("✓ Cleanup complete")


if __name__ == "__main__":
    main()
