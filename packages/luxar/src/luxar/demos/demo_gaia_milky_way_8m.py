#!/usr/bin/env python3
# mypy: ignore-errors
"""Milky Way Stars - 8 Million Star Dataset

DATASET SPECIFICATIONS:
=======================

Star Count & Source:
- 8,149,112 stars
- Source: Unknown (pre-computed dataset from CSV)
- Data file: milky_way_gaia_8m.zarr.zip (143 MB compressed)

Physical Scale & Extent:
- Coordinate system: Unknown (x, y, z in unspecified units)
- Spatial extent: ~776 units across
  * X: -305 to +308 units (span: 613)
  * Y: -519 to +257 units (span: 776)
  * Z: -210 to +144 units (span: 354)
- No explicit cutoff radius specified
- Origin: Unknown reference point

Photometry & Colors:
- Apparent magnitude range: -1.09 to 2843 (extreme outliers present!)
  * Most stars: mag 9 to 15 (99th percentile)
  * Brightest: mag -1.09 (very bright, Sirius-like)
  * Outliers: up to mag 2843 (extremely faint, likely data artifacts)
- Colors: Generated from magnitude (original 'color' column has unusable encoded values)
  * Uses percentile-based normalization (1st-99th) to handle outliers
  * Blue (bright) → White (medium) → Red (faint)

Visualization Parameters:
- No coordinate scaling (uses original units)
- Point radii: Pre-computed sizes × 10 for visibility
- Filters: Stars with size > 0 only
- No reference markers (coordinate system unknown)

Comparison to demo_gaia_milky_way_3m.py:
- 8M vs 3M stars
- Unknown vs Galactocentric coordinates
- Magnitude-based vs real BP-RP colors
- No markers vs Sun/Betelgeuse/Rigel markers
- Unknown provenance vs ESA Gaia DR3 with full citation
- Larger scale (776 units vs 20 kpc)

Usage:
    python demo_gaia_small.py

Controls:
    - Mouse drag: Rotate view
    - Mouse wheel: Zoom in/out
    - Ctrl+C: Stop and cleanup
"""

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import zarr
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_demos_output_dir

# Find the data file relative to this script
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "data" / "milky_way_gaia_8m.zarr.zip"


def load_and_convert_gaia_small(
    data_zip_path: Path, temp_dir: Path, output_path: Path
) -> int:
    """Load gaia_small zarr table and convert to Luxar format.

    Args:
        data_zip_path: Path to gaia_small.zarr.zip
        temp_dir: Temporary directory for extraction
        output_path: Path for Luxar-formatted output

    Returns:
        Number of stars processed
    """
    with asection("Loading Gaia Small Dataset"):
        aprint(f"Data file: {data_zip_path.name}")

        # Extract zarr from zip
        aprint("Extracting zarr from zip...")
        with zipfile.ZipFile(data_zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        raw_zarr_path = temp_dir / "milky_way_gaia_8m.zarr"
        aprint(f"✓ Extracted to: {raw_zarr_path}")

        # Open extracted zarr
        store = zarr.open(str(raw_zarr_path), mode="r")

        # Read arrays
        x = store["x"][:]
        y = store["y"][:]
        z = store["z"][:]
        appmag = store["appmag"][:]
        size = store["size"][:]
        # Note: 'color' column exists but contains encoded values (large negative numbers)
        # We generate colors from magnitude instead

        n_stars = len(x)
        aprint(f"✓ Loaded {n_stars:,} stars")
        aprint(
            f"  Extent: ~{max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()):.0f} units"
        )
        aprint(f"  Magnitude range: {appmag.min():.2f} to {appmag.max():.2f}")

    with asection("Converting to Luxar Format"):
        # Combine positions
        aprint("Creating position array...")
        positions = np.column_stack([x, y, z]).astype(np.float32)

        # Generate colors from magnitude (color column has unusable encoded values)
        aprint("Generating colors from magnitude...")

        # Use percentile-based normalization to handle outliers
        # Most stars are mag 9-15, but outliers go to 2843
        mag_p1 = np.percentile(appmag, 1)  # 1st percentile
        mag_p99 = np.percentile(appmag, 99)  # 99th percentile
        aprint(
            f"  Magnitude range (1st-99th percentile): {mag_p1:.2f} to {mag_p99:.2f}"
        )

        # Normalize using percentiles (brighter = bluer, fainter = redder)
        mag_norm = np.clip((appmag - mag_p1) / (mag_p99 - mag_p1), 0, 1)

        # Color scheme: blue (bright) -> white (medium) -> red (faint)
        r = 0.5 + 0.5 * mag_norm
        g = 0.7 - 0.3 * np.abs(mag_norm - 0.5)
        b = 1.0 - 0.7 * mag_norm

        colors = np.stack([r, g, b], axis=1).astype(np.float32)
        aprint("✓ Generated color range from blue (bright) to red (faint)")

        # Use provided sizes (already in appropriate units)
        aprint("Using pre-computed sizes...")
        radii = size.astype(np.float32)

        # Filter out zero-size stars
        valid = radii > 0
        aprint(f"Filtering stars with size > 0: {valid.sum():,} / {n_stars:,}")

        positions = positions[valid]
        colors = colors[valid]
        radii = radii[valid] * 10
        n_stars = len(positions)

        aprint(f"✓ Prepared {n_stars:,} stars for Luxar")

    with asection("Creating Luxar Scene"):
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Add the stars
            scene.add_points(
                "Stars",
                positions,
                colors=colors,
                radii=radii,
                opacity=0.8,
                blending_mode="additive",
                intensity=0.016,
            )

            # Overlay annotations
            scene.add_text(
                "Milky Way Stars",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                "8.1M stars \u2022 Gaia catalog",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        total_size = sum(
            f.stat().st_size for f in output_path.rglob("*") if f.is_file()
        )
        aprint(f"✓ Created Luxar scene: {output_path}")
        aprint(f"  Size: {total_size / 1e6:.1f} MB")

    return n_stars


def main() -> None:
    """Main demo entry point."""
    aprint("=" * 70)
    aprint("GAIA SMALL - 8 MILLION STARS")
    aprint("=" * 70)
    aprint("")
    aprint("Visualizing 8.1 million stars with pre-computed positions!")
    aprint("")
    aprint("Dataset Details:")
    aprint("  • Stars: 8,149,112")
    aprint("  • Scale: ~776 units across")
    aprint("  • Pre-computed coordinates and sizes")
    aprint("  • Colors generated from magnitude")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        if not DATA_FILE.exists():
            aprint(f"❌ Error: Data file not found: {DATA_FILE}")
            aprint("Expected: packages/luxar/src/luxar/demos/data/gaia_small.zarr.zip")
            sys.exit(1)
        output_path = get_demos_output_dir() / "gaia_small.luxar.zarr"
        with tempfile.TemporaryDirectory(prefix="luxar_demo_gaia_small_") as tmpdir:
            tmp_path = Path(tmpdir)
            load_and_convert_gaia_small(DATA_FILE, tmp_path, output_path)
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_gaia_small_") as tmpdir:
        tmp_path = Path(tmpdir)

        try:
            if not DATA_FILE.exists():
                aprint(f"❌ Error: Data file not found: {DATA_FILE}")
                aprint(
                    "Expected: packages/luxar/src/luxar/demos/data/gaia_small.zarr.zip"
                )
                sys.exit(1)

            # Convert to Luxar format
            luxar_zarr_path = tmp_path / "gaia_small.luxar.zarr"
            n_stars = load_and_convert_gaia_small(DATA_FILE, tmp_path, luxar_zarr_path)

            aprint("")
            aprint("=" * 70)
            aprint("VIEWING TIPS")
            aprint("=" * 70)
            aprint("")
            aprint("Navigation:")
            aprint(f"  • {n_stars:,} stars across ~776 units")
            aprint("  • Start zoomed OUT to see the full structure")
            aprint("  • Zoom IN to see individual stars")
            aprint("  • Colors: Blue (bright) → White → Red (faint)")
            aprint("")
            aprint("=" * 70)
            aprint("LAUNCHING VIEWER")
            aprint("=" * 70)
            aprint("Browser will open automatically...")
            aprint("Press Ctrl+C when done.")
            aprint("")

            # Launch viewer
            subprocess.run(
                ["luxar", "serve", str(luxar_zarr_path), "--viewer", "--open"],
                check=True,
            )

        except KeyboardInterrupt:
            aprint("\n🛑 Stopping demo...")
        except subprocess.CalledProcessError as e:
            aprint(f"\n❌ Error launching viewer: {e}")
            aprint("Make sure viewer is built:")
            aprint("   cd packages/luxar-viewer && pnpm build")
            sys.exit(1)
        except FileNotFoundError as e:
            if "luxar" in str(e):
                aprint("\n❌ Error: 'luxar' command not found")
                aprint("Install luxar: pip install -e .")
            else:
                aprint(f"\n❌ Error: {e}")
            sys.exit(1)

    aprint("")
    aprint("Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
