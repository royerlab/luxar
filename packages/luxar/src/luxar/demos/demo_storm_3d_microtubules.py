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
    --max-localizations=N   Limit number of localizations (default: 5M)
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

import sys
from pathlib import Path

import numpy as np
import requests
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Zenodo dataset
ZENODO_RECORD = "3547521"
ZENODO_BASE_URL = f"https://zenodo.org/record/{ZENODO_RECORD}/files"

# Default parameters
DEFAULT_FIELD = 4  # Field of view number
DEFAULT_MAX_LOCALIZATIONS = 5_000_000  # Limit for demo performance

# Visualization parameters
WIDEFIELD_PSF_SIGMA = 100.0  # nm - conventional microscopy PSF width
SUPERRES_PSF_SIGMA = 20.0  # nm - super-resolution PSF width
PIXEL_SIZE = 106.0  # nm - from dataset metadata

# Scale factor for visualization (physical PSF is too small to see in the viewer)
# The scene spans ~60 μm, so 20 nm splats would be invisible. Scale up for visibility.
# With VIS_SCALE=40: super-res=800nm≈0.8μm, widefield=4μm - matches original committed scale
# (Original used SUPER_RES_PRECISION_SCALE_XY=100 with ~8nm precision → 0.8μm sigma)
VIS_SCALE = (
    1.0  # Makes splats visible while maintaining relative widefield/superres ratio
)


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
    import pandas as pd

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

        aprint(f"  Columns: {list(df.columns)[:10]}")

        # Extract key columns (column names may vary)
        # Common formats: x, y, z or xnm, ynm, znm
        result = {}

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
                    # Convert pixels to nm if needed
                    if "_pix" in variant:
                        values = values * PIXEL_SIZE
                    result[key_base] = values
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
                    # CRLB is already in nm (or pixels)
                    if (
                        "crlb_" in variant
                        and "_pix" not in variant
                        and "nm" not in variant
                    ):
                        # Plain 'crlb_x' might be in pixels
                        values = values * PIXEL_SIZE
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
) -> tuple[np.ndarray, np.ndarray]:
    """Extract centers and amplitudes from STORM localizations.

    Args:
        localizations: Dictionary with x, y, z, photons, etc.

    Returns:
        Tuple of (centers_um, amplitudes) where centers are in micrometers
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

        aprint(f"✓ Extracted {n_loc:,} localizations")
        aprint(f"  Centers: {centers_um.shape}")
        aprint(
            f"  Spatial range: [{centers_um.min(axis=0)}] to [{centers_um.max(axis=0)}] μm"
        )

    return centers_um, amplitudes


# =============================================================================
# Scene Creation
# =============================================================================


def create_storm_scene(
    centers_um: np.ndarray,
    amplitudes: np.ndarray,
    output_path: Path | None = None,
) -> Path:
    """Create Luxar scene with STORM data comparing widefield vs super-resolution.

    Uses simple isotropic PSF sigmas for both views:
    - Widefield: WIDEFIELD_PSF_SIGMA (100 nm) - diffraction-limited
    - Super-resolution: SUPERRES_PSF_SIGMA (20 nm) - STORM precision

    Args:
        centers_um: Splat centers in micrometers (N, 3)
        amplitudes: Splat amplitudes (N,)
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

            # Create BOTH views as gsplats with different PSF sizes
            with asection("Adding both microscopy views as gsplats"):
                from luxar.gsplats.gsplat_data import GSplatData

                # Center at center-of-mass (amplitude-weighted)
                total_amplitude = amplitudes.sum()
                if total_amplitude > 0:
                    centroid = (centers_um.T @ amplitudes) / total_amplitude
                else:
                    centroid = centers_um.mean(axis=0)

                centered = centers_um - centroid
                aprint(
                    f"Centered at COM (was at [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}] μm)"
                )
                aprint(
                    f"  New range: [{centered.min(axis=0)}] to [{centered.max(axis=0)}]"
                )

                n_splats = len(centers_um)
                aprint(f"Creating {n_splats:,} splats × 2 views...")
                aprint(
                    f"  Physical PSF: widefield={WIDEFIELD_PSF_SIGMA} nm, super-res={SUPERRES_PSF_SIGMA} nm"
                )
                aprint(
                    f"  Vis scale: {VIS_SCALE}x (effective: {WIDEFIELD_PSF_SIGMA * VIS_SCALE} nm / {SUPERRES_PSF_SIGMA * VIS_SCALE} nm)"
                )

                # Precompute covariance matrices (simple isotropic)
                # Apply visualization scale to make splats visible
                widefield_sigma_um = (
                    WIDEFIELD_PSF_SIGMA * VIS_SCALE
                ) / 1000  # nm -> μm
                superres_sigma_um = (SUPERRES_PSF_SIGMA * VIS_SCALE) / 1000  # nm -> μm

                # Z has 2x worse resolution (typical for 3D STORM)
                widefield_cov = np.diag(
                    [
                        widefield_sigma_um**2,
                        widefield_sigma_um**2,
                        (widefield_sigma_um * 2) ** 2,
                    ]
                )
                superres_cov = np.diag(
                    [
                        superres_sigma_um**2,
                        superres_sigma_um**2,
                        (superres_sigma_um * 2) ** 2,
                    ]
                )

                # Precompute 4D covariances and their Cholesky factors
                def make_4d_cholesky(cov_3d: np.ndarray) -> np.ndarray:
                    """Create 4D Cholesky from 3D covariance."""
                    cov_4d = np.zeros((4, 4), dtype=np.float32)
                    cov_4d[1:, 1:] = cov_3d  # Spatial part
                    cov_4d[0, 0] = 1e-6  # Tiny variance in view dimension
                    chol = np.linalg.cholesky(cov_4d)
                    # Pack lower triangular: [L00, L10, L11, L20, L21, L22, L30, L31, L32, L33]
                    return np.array(
                        [
                            chol[0, 0],
                            chol[1, 0],
                            chol[1, 1],
                            chol[2, 0],
                            chol[2, 1],
                            chol[2, 2],
                            chol[3, 0],
                            chol[3, 1],
                            chol[3, 2],
                            chol[3, 3],
                        ],
                        dtype=np.float32,
                    )

                widefield_chol = make_4d_cholesky(widefield_cov)
                superres_chol = make_4d_cholesky(superres_cov)

                aprint(f"  Widefield sigma: {widefield_sigma_um:.3f} μm")
                aprint(f"  Super-res sigma: {superres_sigma_um:.3f} μm")
                aprint(f"  Widefield cholesky: {widefield_chol}")
                aprint(f"  Super-res cholesky: {superres_chol}")

                # Build arrays for both views (vectorized where possible)
                # VIEW 0: Widefield (gray, dimmer)
                # VIEW 1: Super-resolution (cyan, brighter)
                all_centers = np.zeros((n_splats * 2, 4), dtype=np.float32)
                all_cholesky = np.zeros((n_splats * 2, 10), dtype=np.float32)
                all_amplitudes = np.zeros(n_splats * 2, dtype=np.float32)
                all_colors = np.zeros((n_splats * 2, 3), dtype=np.float32)

                # Widefield view (indices 0 to n_splats-1)
                all_centers[:n_splats, 0] = 0.0  # view = 0
                all_centers[:n_splats, 1:] = centered
                all_cholesky[:n_splats] = widefield_chol
                all_amplitudes[:n_splats] = amplitudes * 0.3  # Dimmer for widefield
                all_colors[:n_splats] = [0.7, 0.7, 0.7]  # Gray

                # Super-resolution view (indices n_splats to 2*n_splats-1)
                all_centers[n_splats:, 0] = 1.0  # view = 1
                all_centers[n_splats:, 1:] = centered
                all_cholesky[n_splats:] = superres_chol
                all_amplitudes[n_splats:] = amplitudes  # Full brightness
                all_colors[n_splats:] = [0.2, 1.0, 1.0]  # Cyan

                # Create GSplatData
                gsplat_data = GSplatData(
                    centers=all_centers,
                    cholesky_factors=all_cholesky,
                    amplitudes=all_amplitudes,
                    colors=all_colors,
                )

                scene.add_gsplats_from_data(
                    name="microtubules",
                    result=gsplat_data,
                    opacity=0.8,
                    blending_mode="additive",
                )

                aprint(f"✓ Added {n_splats * 2:,} splats (2 views × {n_splats:,})")

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
                f'<div style="font-size:1.3vh;color:#aaa">\u03c3 \u2248 {WIDEFIELD_PSF_SIGMA} nm (diffraction-limited)</div>',
                position=(0.02, 0.97),
                anchor="bottom-left",
                visible_range={"view": 0},
                transition="fade",
                transition_duration=0.3,
            )
            scene.add_html(
                '<div style="font-size:1.5vh;font-weight:bold;color:#44ff88">Super-Resolution</div>'
                f'<div style="font-size:1.3vh;color:#aaa">\u03c3 \u2248 {SUPERRES_PSF_SIGMA} nm (STORM)</div>',
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
    except ImportError as e:
        aprint(f"Missing dependency: {e}")
        aprint("")
        aprint("Install with:")
        aprint("  pip install pandas")
        sys.exit(1)

    try:
        # Download localizations
        csv_file = download_storm_localizations(field=field)

        # Parse localizations
        localizations = parse_storm_localizations(csv_file, max_localizations=max_loc)

        # Extract centers and amplitudes (simple, no per-localization covariances)
        centers_um, amplitudes = extract_centers_and_amplitudes(localizations)

        # If --no-serve, generate and exit without launching viewer
        if "--no-serve" in sys.argv:
            output_path = get_demos_output_dir() / "storm_3d_microtubules.luxar.zarr"
            scene_path = create_storm_scene(
                centers_um, amplitudes, output_path=output_path
            )
            aprint(f"Dataset generated at {scene_path}")
            aprint(f"Localizations: {len(centers_um):,}")
            return

        # Create scene in demos directory for serving
        scene_path = create_storm_scene(centers_um, amplitudes)

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
