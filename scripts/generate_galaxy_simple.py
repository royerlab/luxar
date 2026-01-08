#!/usr/bin/env python3
"""
Gaia DR3 Galaxy Data Generator - Fetch brightest stars from ESA Gaia Archive

This script fetches real star data from the European Space Agency's Gaia Data
Release 3 and prepares it for visualization in Luxar.

Data Source:
    ESA/Gaia/DPAC - Gaia Data Release 3 (2022)
    Archive: https://gea.esac.esa.int/archive/

    Citation:
    Gaia Collaboration, Vallenari et al. (2023)
    "Gaia Data Release 3: Summary of the content and survey properties"
    Astronomy & Astrophysics, 674, A1
    DOI: 10.1051/0004-6361/202243940

Method:
    1. Query ESA Gaia Archive via TAP (astroquery)
    2. Select brightest stars with high-quality parallax measurements
    3. Transform to Galactocentric coordinates (Astropy)
    4. Save as raw zarr table for demo consumption

DEPENDENCIES:
    pip install astroquery astropy

USAGE:
    python scripts/generate_galaxy_simple.py --count 10000     # 10k stars (fast)
    python scripts/generate_galaxy_simple.py --count 100000    # 100k stars (medium)
    python scripts/generate_galaxy_simple.py --count 3000000   # 3M stars (slow)

OUTPUT:
    Raw zarr table with arrays: x_kpc, y_kpc, z_kpc, phot_g_mean_mag, bp_rp
    (NOT Luxar format - demo converts to Luxar)
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from arbol import aprint

try:
    import astropy.units as u
    from astropy.coordinates import Galactocentric, SkyCoord
    from astroquery.gaia import Gaia
except ImportError:
    aprint("ERROR: Missing dependencies.")
    aprint("Please install: pip install astroquery astropy")
    sys.exit(1)

R0_KPC = 8.122  # Sun-GC distance


def fetch_stars(count: int) -> pd.DataFrame:
    """Fetch top N brightest stars from Gaia DR3."""
    aprint(f"\nFetching {count:,} brightest stars from Gaia DR3...")

    query = f"""
SELECT TOP {count}
  source_id, ra, dec,
  parallax, parallax_over_error,
  phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp
FROM gaiadr3.gaia_source
WHERE parallax > 0.1
  AND parallax_over_error > 5
  AND phot_bp_mean_mag IS NOT NULL
  AND phot_rp_mean_mag IS NOT NULL
ORDER BY phot_g_mean_mag ASC
"""

    aprint("  Querying ESA Gaia Archive...")
    job = Gaia.launch_job_async(query)
    results = job.get_results()

    aprint(f"  ✓ Retrieved {len(results):,} stars")

    df = results.to_pandas()
    aprint(
        f"  Magnitude range: {df['phot_g_mean_mag'].min():.2f} to {df['phot_g_mean_mag'].max():.2f}"
    )

    return df


def compute_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distances from parallax."""
    aprint("\nComputing distances...")
    df["dist_pc"] = 1000.0 / df["parallax"]
    aprint(
        f"  Distance range: {df['dist_pc'].min():.1f} to {df['dist_pc'].max():.1f} pc"
    )
    return df


def transform_to_galactocentric(df: pd.DataFrame, rmax_kpc: float) -> pd.DataFrame:
    """Transform to Galactocentric coordinates."""
    aprint(f"\nTransforming to Galactocentric (R0={R0_KPC} kpc)...")

    dist_kpc = (df["dist_pc"].values * u.pc).to(u.kpc)
    c_icrs = SkyCoord(
        ra=df["ra"].values * u.deg,
        dec=df["dec"].values * u.deg,
        distance=dist_kpc,
    )

    gc = c_icrs.transform_to(Galactocentric(galcen_distance=R0_KPC * u.kpc))

    df["x_kpc"] = gc.x.to(u.kpc).value
    df["y_kpc"] = gc.y.to(u.kpc).value
    df["z_kpc"] = gc.z.to(u.kpc).value
    df["r_gc_kpc"] = np.sqrt(df["x_kpc"] ** 2 + df["y_kpc"] ** 2 + df["z_kpc"] ** 2)

    aprint(f"  Applying cut: r_gc <= {rmax_kpc} kpc")
    df_cut = df[df["r_gc_kpc"] <= rmax_kpc].copy()

    aprint(f"  Stars within {rmax_kpc} kpc: {len(df_cut):,} / {len(df):,}")

    return df_cut


def create_zarr(df: pd.DataFrame, output_path: Path):
    """Save raw Gaia data as zarr table (NOT Luxar format)."""
    import zarr

    aprint(f"\nSaving raw data to zarr: {output_path}")

    if output_path.exists():
        shutil.rmtree(output_path)

    # Create zarr store with raw table data
    store = zarr.open(str(output_path), mode="w")

    # Save positions
    store.create_dataset(
        "x_kpc", data=df["x_kpc"].values.astype(np.float32), chunks=(100000,)
    )
    store.create_dataset(
        "y_kpc", data=df["y_kpc"].values.astype(np.float32), chunks=(100000,)
    )
    store.create_dataset(
        "z_kpc", data=df["z_kpc"].values.astype(np.float32), chunks=(100000,)
    )

    # Save Gaia photometry
    store.create_dataset(
        "phot_g_mean_mag",
        data=df["phot_g_mean_mag"].values.astype(np.float32),
        chunks=(100000,),
    )
    store.create_dataset(
        "bp_rp", data=df["bp_rp"].values.astype(np.float32), chunks=(100000,)
    )

    # Save metadata
    store.attrs["num_stars"] = len(df)
    store.attrs["magnitude_range"] = [
        float(df["phot_g_mean_mag"].min()),
        float(df["phot_g_mean_mag"].max()),
    ]
    store.attrs["description"] = "Gaia DR3 stars in Galactocentric coordinates"
    store.attrs["data_source"] = "ESA Gaia DR3"

    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e6
    aprint(f"✓ Saved raw data: {output_path} ({size_mb:.2f} MB)")
    aprint("  Arrays: x_kpc, y_kpc, z_kpc, phot_g_mean_mag, bp_rp")


def create_zip(zarr_path: Path):
    """Create zip archive."""
    zip_path = zarr_path.with_suffix(".zarr.zip")
    aprint(f"\nCreating zip: {zip_path}")

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in zarr_path.rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(zarr_path.parent))

    aprint(f"✓ Created: {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Gaia stars and create galaxy zarr"
    )
    parser.add_argument(
        "--count", type=int, default=10000, help="Number of stars to fetch"
    )
    parser.add_argument(
        "--rmax-kpc", type=float, default=30.0, help="Max distance from GC (kpc)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("galaxy.zarr"), help="Output path"
    )
    args = parser.parse_args()

    aprint("=" * 60)
    aprint(f"Simple Gaia Galaxy Generator - {args.count:,} stars")
    aprint("=" * 60)

    df = fetch_stars(args.count)
    df = compute_distances(df)
    df = transform_to_galactocentric(df, args.rmax_kpc)
    create_zarr(df, args.output)
    create_zip(args.output)

    aprint("\n" + "=" * 60)
    aprint("SUCCESS!")
    aprint(f"  Stars: {len(df):,}")
    aprint(f"  Zarr: {args.output}")
    aprint(f"  Zip: {args.output.with_suffix('.zarr.zip')}")
    aprint("=" * 60)
    aprint("\nNote: This is a RAW data table (not Luxar format).")
    aprint("Use demo_gaia_milky_way_3m.py to convert and visualize.")


if __name__ == "__main__":
    main()
