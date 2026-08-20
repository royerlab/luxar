"""Build the non-redistributable Gaia DR3 catalog used by the Milky Way demo."""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from arbol import aprint

from luxar._zarr_compat import create_array, open_group
from luxar.demos._dependencies import MissingDependencyError, require_module
from luxar.encoding.compression import WIDTH_AWARE_DEFAULT, resolve_compressor

if TYPE_CHECKING:
    import pandas as pd

CATALOG_STEM = "milky_way_gaia_3m"
RAW_ZARR_NAME = f"{CATALOG_STEM}.zarr"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "luxar" / CATALOG_STEM
DEFAULT_CATALOG_FILE = DEFAULT_CACHE_DIR / f"{RAW_ZARR_NAME}.zip"
DEFAULT_COUNT = 3_000_000
DEFAULT_RMAX_KPC = 30.0
R0_KPC = 8.122

GAIA_ACKNOWLEDGEMENT = (
    "This work has made use of data from the European Space Agency (ESA) mission "
    "Gaia, processed by the Gaia Data Processing and Analysis Consortium (DPAC)."
)


def fetch_stars(count: int) -> pd.DataFrame:
    """Fetch the brightest high-quality-parallax stars from Gaia DR3."""
    gaia = require_module("astroquery.gaia").Gaia

    aprint("")
    aprint("ESA/Gaia/DPAC acknowledgement:")
    aprint(f"  {GAIA_ACKNOWLEDGEMENT}")
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
    results = gaia.launch_job_async(query).get_results()
    aprint(f"  ✓ Retrieved {len(results):,} stars")

    frame = results.to_pandas()
    aprint(
        "  Magnitude range: "
        f"{frame['phot_g_mean_mag'].min():.2f} to "
        f"{frame['phot_g_mean_mag'].max():.2f}"
    )
    return frame


def compute_distances(frame: pd.DataFrame) -> pd.DataFrame:
    """Add heliocentric distances computed from parallax."""
    aprint("\nComputing distances...")
    frame["dist_pc"] = 1000.0 / frame["parallax"]
    aprint(
        f"  Distance range: {frame['dist_pc'].min():.1f} to "
        f"{frame['dist_pc'].max():.1f} pc"
    )
    return frame


def transform_to_galactocentric(frame: pd.DataFrame, rmax_kpc: float) -> pd.DataFrame:
    """Transform ICRS positions into Galactocentric coordinates."""
    units = require_module("astropy.units")
    coordinates = require_module("astropy.coordinates")

    aprint(f"\nTransforming to Galactocentric (R0={R0_KPC} kpc)...")
    dist_kpc = (frame["dist_pc"].values * units.pc).to(units.kpc)
    icrs = coordinates.SkyCoord(
        ra=frame["ra"].values * units.deg,
        dec=frame["dec"].values * units.deg,
        distance=dist_kpc,
    )
    galactocentric = icrs.transform_to(
        coordinates.Galactocentric(galcen_distance=R0_KPC * units.kpc)
    )

    frame["x_kpc"] = galactocentric.x.to(units.kpc).value
    frame["y_kpc"] = galactocentric.y.to(units.kpc).value
    frame["z_kpc"] = galactocentric.z.to(units.kpc).value
    frame["r_gc_kpc"] = np.sqrt(
        frame["x_kpc"] ** 2 + frame["y_kpc"] ** 2 + frame["z_kpc"] ** 2
    )

    aprint(f"  Applying cut: r_gc <= {rmax_kpc} kpc")
    selected = frame[frame["r_gc_kpc"] <= rmax_kpc].copy()
    aprint(f"  Stars within {rmax_kpc} kpc: {len(selected):,} / {len(frame):,}")
    return selected


def create_zarr(frame: pd.DataFrame, output_path: Path) -> None:
    """Save the transformed Gaia table in the demo's raw zarr format."""
    aprint(f"\nSaving raw data to zarr: {output_path}")
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    store = open_group(output_path, mode="w")
    # Named, never defaulted: this module now lives in the package, where an
    # omitted compressor silently means "auto" (Blosc/lz4/5 at format 2, zstd at
    # format 3) rather than Luxar's measured width-aware policy.
    compressor = resolve_compressor(WIDTH_AWARE_DEFAULT, np.float32)
    for name in ("x_kpc", "y_kpc", "z_kpc", "phot_g_mean_mag", "bp_rp"):
        create_array(
            store,
            name,
            data=frame[name].values.astype(np.float32),
            chunks=(100_000,),
            compressor=compressor,
        )

    store.attrs["num_stars"] = len(frame)
    store.attrs["magnitude_range"] = [
        float(frame["phot_g_mean_mag"].min()),
        float(frame["phot_g_mean_mag"].max()),
    ]
    store.attrs["description"] = "Gaia DR3 stars in Galactocentric coordinates"
    store.attrs["data_source"] = "ESA Gaia DR3"

    size_mb = sum(p.stat().st_size for p in output_path.rglob("*") if p.is_file()) / 1e6
    aprint(f"✓ Saved raw data: {output_path} ({size_mb:.2f} MB)")
    aprint("  Arrays: x_kpc, y_kpc, z_kpc, phot_g_mean_mag, bp_rp")


def create_zip(zarr_path: Path, zip_path: Path | None = None) -> Path:
    """Archive ``zarr_path`` with its directory name as the top-level member."""
    destination = zip_path or zarr_path.with_suffix(".zarr.zip")
    aprint(f"\nCreating zip: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(zarr_path.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(zarr_path.parent))

    aprint(f"✓ Created: {destination} ({destination.stat().st_size / 1e6:.2f} MB)")
    return destination


def generate_raw_catalog(
    output_path: Path, *, count: int = DEFAULT_COUNT, rmax_kpc: float = DEFAULT_RMAX_KPC
) -> int:
    """Query, transform, and write one raw Gaia catalog."""
    frame = fetch_stars(count)
    frame = compute_distances(frame)
    frame = transform_to_galactocentric(frame, rmax_kpc)
    create_zarr(frame, output_path)
    return len(frame)


def generate_catalog(
    output_path: Path, *, count: int = DEFAULT_COUNT, rmax_kpc: float = DEFAULT_RMAX_KPC
) -> tuple[Path, int]:
    """Generate a raw catalog and its adjacent zip archive."""
    stars = generate_raw_catalog(output_path, count=count, rmax_kpc=rmax_kpc)
    return create_zip(output_path), stars


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def build_catalog(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    count: int = DEFAULT_COUNT,
    rmax_kpc: float = DEFAULT_RMAX_KPC,
    recompute: bool = False,
) -> Path:
    """Build and cache the demo catalog, preserving a usable zip until promotion.

    A completed raw zarr is retained beside the zip. If a run is interrupted
    after that directory is promoted, the next invocation only has to recreate
    the archive rather than repeat the multi-million-row TAP query.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / RAW_ZARR_NAME
    catalog_path = cache_dir / f"{RAW_ZARR_NAME}.zip"

    if catalog_path.is_file() and not recompute:
        return catalog_path

    build_id = uuid.uuid4().hex
    staged_raw = cache_dir / f".{RAW_ZARR_NAME}.{build_id}.tmp"
    staged_zip = cache_dir / f".{RAW_ZARR_NAME}.{build_id}.zip.tmp"
    try:
        if recompute or not raw_path.is_dir():
            aprint(f"Building the Gaia DR3 catalog in {cache_dir}")
            generate_raw_catalog(
                staged_raw,
                count=count,
                rmax_kpc=rmax_kpc,
            )
            _remove_path(raw_path)
            staged_raw.replace(raw_path)
        else:
            aprint(f"Resuming from the completed raw catalog at {raw_path}")

        create_zip(raw_path, staged_zip)
        staged_zip.replace(catalog_path)
    finally:
        _remove_path(staged_raw)
        _remove_path(staged_zip)

    return catalog_path


def main(argv: list[str] | None = None) -> int:
    """CLI used by ``scripts/generate_galaxy_simple.py``."""
    parser = argparse.ArgumentParser(
        description="Fetch Gaia stars and create a raw galaxy zarr table"
    )
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--rmax-kpc", type=float, default=DEFAULT_RMAX_KPC)
    parser.add_argument("--output", type=Path, default=Path("galaxy.zarr"))
    args = parser.parse_args(argv)

    aprint("=" * 60)
    aprint(f"Simple Gaia Galaxy Generator - {args.count:,} stars")
    aprint("=" * 60)
    try:
        zip_path, stars = generate_catalog(
            args.output, count=args.count, rmax_kpc=args.rmax_kpc
        )
    except MissingDependencyError as exc:
        aprint(f"ERROR: {exc}")
        return 1

    aprint("\n" + "=" * 60)
    aprint("SUCCESS!")
    aprint(f"  Stars: {stars:,}")
    aprint(f"  Zarr: {args.output}")
    aprint(f"  Zip: {zip_path}")
    aprint("=" * 60)
    aprint("\nNote: This is a RAW data table (not Luxar format).")
    aprint("Use demo_gaia_milky_way_3m.py to convert and visualize.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
