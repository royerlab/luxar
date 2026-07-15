#!/usr/bin/env python3
"""Real Milky Way Stars from Gaia DR3 - 3 Million Star Dataset

DATASET SPECIFICATIONS:
=======================

Star Count & Selection:
- 3,000,000 stars (top 3M brightest from Gaia DR3)
- Selection query criteria:
  * parallax > 0.1 mas  → distances up to ~10 kpc from Sun
  * parallax_over_error > 5  → high-precision measurements only
  * Has BP and RP photometry → color information available
  * ORDER BY phot_g_mean_mag ASC → sorted by brightness

Physical Scale & Extent:
- Distance from Sun: 1.3 to ~10,000 pc (0.0013 to 10 kpc)
- Galactocentric extent: ~20 kpc across (X: -18 to +2 kpc)
- Cutoff radius: 30 kpc from Galactic Center
- Coordinate system: Galactocentric (origin at GC, Sun at -8.122 kpc X)

Photometry & Colors:
- Magnitude: G = 1.94 to 11.99
  * Brightest: mag 1.94 (Sirius-like, naked-eye visible)
  * Faintest: mag 11.99 (requires telescope)
- Colors: Real BP-RP color index from Gaia photometry
  * BP-RP → RGB conversion for stellar temperature visualization
  * Blue: Hot stars, Red: Cool stars

Visualization Parameters:
- Coordinate scaling: 10x multiplier (easier navigation)
- Point radii: 0.01 to 0.11 (scaled units), magnitude-dependent
- Reference markers:
  * Sun: Yellow, radius 0.35 (10x typical star)
  * Betelgeuse: Red, radius 0.35, at 168 pc
  * Rigel: Blue, radius 0.35, at 265 pc

Data Source & Attribution:
    ESA/Gaia/DPAC - Gaia Data Release 3 (2022)

    Mission: https://www.cosmos.esa.int/gaia
    Archive: https://gea.esac.esa.int/archive/

    Citation:
    Gaia Collaboration, Vallenari et al. (2023)
    "Gaia Data Release 3: Summary of the content and survey properties"
    Astronomy & Astrophysics, 674, A1
    DOI: 10.1051/0004-6361/202243940
    https://doi.org/10.1051/0004-6361/202243940

Data Generation Method:
    The raw data was fetched using the script:
    scripts/generate_galaxy_simple.py

    Query executed via ESA Gaia Archive TAP service:
    ```sql
    SELECT TOP 3000000
      source_id, ra, dec, parallax, parallax_over_error,
      phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp
    FROM gaiadr3.gaia_source
    WHERE parallax > 0.1
      AND parallax_over_error > 5
      AND phot_bp_mean_mag IS NOT NULL
      AND phot_rp_mean_mag IS NOT NULL
    ORDER BY phot_g_mean_mag ASC
    ```

    Post-processing:
    1. Distance computed from parallax: d[pc] = 1000 / parallax[mas]
    2. Transformed to Galactocentric coordinates using Astropy
       with galcen_distance = 8.122 kpc (GRAVITY Collaboration 2018)
    3. Filtered to stars within 30 kpc of Galactic Center
    4. Stored as raw zarr table (x_kpc, y_kpc, z_kpc, phot_g_mean_mag, bp_rp)

    References:
    - Galactocentric distance: GRAVITY Collaboration (2018), A&A 615, L15
      https://doi.org/10.1051/0004-6361/201833718

Scientific Context:
    The Gaia space telescope has measured positions, motions, and colors of
    ~1.8 billion stars in our galaxy. This demo shows the 3M brightest stars
    with high-quality distance measurements, transformed to a Galactocentric
    reference frame centered on the Galactic Center.

    Colors represent stellar temperature:
    - Blue: Hot, young stars (spectral type O, B, A)
    - White/Yellow: Sun-like stars (spectral type F, G)
    - Red/Orange: Cool, old stars (spectral type K, M)

    The coordinate system places you at approximately (-8.122, 0, 0) kpc from
    the Galactic Center - the location of our Sun!

Reference Markers:
    Three famous stars are marked for orientation:
    - Sun: Our home star at (-8.122, 0, 0) kpc
    - Betelgeuse: Red supergiant in Orion, ~168 pc from Sun
    - Rigel: Blue supergiant in Orion, ~265 pc from Sun

Usage:
    python demo_gaia_milky_way.py

Controls:
    - Mouse drag: Rotate view
    - Mouse wheel: Zoom in/out
    - Ctrl+C: Stop and cleanup

Viewing Tips:
    - Start zoomed out to see the overall structure
    - Notice the thin disk of the Milky Way
    - Dense concentration toward the Galactic Center
    - Zoom in to see individual stars
    - Colors reflect real stellar temperatures!
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import zarr
from arbol import aprint, asection

from luxar import (
    CameraConfig,
    Dimension,
    Dimensions,
    LuxarZarrCompiler,
    ViewerConfig,
)
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir

# Find the data file relative to this script
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "data" / "milky_way_gaia_3m.zarr.zip"


def compute_colors(bp_rp: np.ndarray, phot_g_mean_mag: np.ndarray) -> np.ndarray:
    """Convert BP-RP color index and magnitude to RGB colors.

    Args:
        bp_rp: BP-RP color index from Gaia
        phot_g_mean_mag: G-band magnitude

    Returns:
        RGB colors (N, 3) as float32
    """
    bp_rp_norm = np.clip((bp_rp + 0.5) / 5.0, 0, 1)
    r = bp_rp_norm
    g = 1.0 - 2.0 * np.abs(bp_rp_norm - 0.5)
    b = 1.0 - bp_rp_norm

    brightness = np.clip((21 - phot_g_mean_mag) / 18.0, 0.1, 1.0)
    colors = np.stack([r * brightness, g * brightness, b * brightness], axis=1).astype(
        np.float32
    )
    return colors  # type: ignore[no-any-return]


def compute_radii(phot_g_mean_mag: np.ndarray) -> np.ndarray:
    """Compute point radii from magnitude.

    Uses visualization-friendly radii (not physically accurate).
    Points are 10-50x larger than actual stars for visibility.

    Args:
        phot_g_mean_mag: G-band magnitude

    Returns:
        Radii as float32 (in kpc)
    """
    mag_norm = np.clip((21 - phot_g_mean_mag) / 18.0, 0, 1)
    # Visualization scale: 0.001 to 0.011 kpc (1 to 11 pc)
    # Larger than physical stars but visible at galactic scales
    return (0.001 + 0.01 * mag_norm**2).astype(np.float32)  # type: ignore[no-any-return]


def load_and_convert_gaia_data(data_zarr_path: Path, output_path: Path) -> int:
    """Load raw Gaia zarr table and convert to Luxar format.

    This function demonstrates the pipeline:
    Raw Gaia data (zarr table) → Luxar scene (zarr)

    Args:
        data_zarr_path: Path to extracted raw galaxy.zarr
        output_path: Path for Luxar-formatted output

    Returns:
        Number of stars processed
    """
    # Scale multiplier for visualization
    # Coordinates are scaled up to make the scene easier to navigate
    SCALE = 10.0  # 10x larger for better visualization

    with asection("Loading Raw Gaia Data"):
        # Open raw zarr store (supports reading from zip directly!)
        if str(data_zarr_path).endswith(".zip"):
            # Read directly from zip
            store = zarr.open(
                f"zip://{data_zarr_path}::milky_way_gaia_3m.zarr", mode="r"
            )
        else:
            store = zarr.open(str(data_zarr_path), mode="r")

        # Read arrays
        x_kpc = store["x_kpc"][:]
        y_kpc = store["y_kpc"][:]
        z_kpc = store["z_kpc"][:]
        phot_g_mean_mag = store["phot_g_mean_mag"][:]
        bp_rp = store["bp_rp"][:]

        n_stars = len(x_kpc)
        aprint(f"✓ Loaded {n_stars:,} stars")
        aprint(
            f"  Magnitude range: {phot_g_mean_mag.min():.2f} to {phot_g_mean_mag.max():.2f}"
        )

    with asection("Converting to Luxar Format"):
        # Combine positions and scale up for better visualization
        aprint(f"Creating position array (scaling by {SCALE}x for visibility)...")
        positions = np.column_stack(
            [x_kpc * SCALE, y_kpc * SCALE, z_kpc * SCALE]
        ).astype(np.float32)

        # Compute colors from Gaia photometry
        aprint("Computing stellar colors from BP-RP...")
        colors = compute_colors(bp_rp, phot_g_mean_mag)

        # Compute radii from brightness and scale
        aprint("Computing point radii from magnitude...")
        radii = compute_radii(phot_g_mean_mag) * SCALE

        aprint(f"✓ Prepared {n_stars:,} stars for Luxar")

    with asection("Creating Luxar Scene"):
        dims = Dimensions(
            [
                Dimension("x", unit="kpc", display=True),
                Dimension("y", unit="kpc", display=True),
                Dimension("z", unit="kpc", display=True),
            ]
        )

        # Start the camera pulled IN, framing the bright stellar bulk (the 3M
        # brightest stars cluster near the Sun, not the galactic centre). Robust
        # 2–98th percentile bounds ignore sparse-halo outliers that would
        # otherwise make the auto-fit zoom way out and leave the galaxy a tiny
        # dot. Closer start = galaxy fills the view AND the coverage-fraction LOD
        # immediately shows a finer level.
        lo, hi = np.percentile(positions, [2, 98], axis=0)
        center = (lo + hi) / 2.0
        extent = float(np.max(hi - lo))
        fov_deg = 47.0
        fit_dist = (extent * 0.5) / np.tan(np.radians(fov_deg) / 2.0)
        cam_dist = fit_dist * 0.65  # pull in ~35% tighter than a plain fit
        camera = CameraConfig(
            position=(
                float(center[0]),
                float(center[1] + extent * 0.15),
                float(center[2] + cam_dist),
            ),
            target=(float(center[0]), float(center[1]), float(center[2])),
            up=(0.0, 1.0, 0.0),
            fov=fov_deg,
            near=float(max(0.5, cam_dist * 0.005)),
            far=float(cam_dist * 20.0 + extent * 10.0),
        )

        with LuxarZarrCompiler(output_path) as compiler:
            # Bake a dark-sky appearance: a moderate exposure keeps the
            # background black (a high exposure floods the faint-star haze into a
            # grey wash), a raised bloom threshold blooms only the brightest
            # stars, and Neutral tone-mapping preserves true stellar colours
            # (ACES would shift blue/red star hues).
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    camera=camera,
                    exposure=0.5,
                    tone_mapping="Neutral",
                    bloom_enabled=True,
                    bloom_strength=0.15,
                    bloom_threshold=0.85,
                ),
            )

            # Add the stars with substitutive Points LOD: coarse levels replace
            # the 3M-star cloud with fewer, larger mass-preserving Gaussian splats
            # when the galaxy is small on screen, so the viewer only pays for the
            # detail it can resolve (the census demo uses the same wiring). The
            # `layer=True` flag rides onto the wrapper kind=lod group → one "Stars"
            # layer in the Layers panel.
            scene.add_points(
                "Stars",
                positions,
                colors=colors,
                radii=radii,
                opacity=0.9,
                blending_mode="additive",
                intensity=0.031,
                layer=True,
                substitutive_lod=dict(compression_factor=8, levels=3, device="auto"),
            )

            # Add reference markers for famous stars
            aprint("Adding reference markers...")

            # Calculate marker radius: 10x typical scaled star radius
            # Typical star: ~0.0035 kpc × SCALE = 0.035
            # Marker: 10x typical = 0.35
            typical_star_radius = (0.001 + 0.01 * 0.5**2) * SCALE  # Mid-brightness star
            marker_radius = typical_star_radius * 10

            # Sun marker at the Sun's Galactocentric position
            r0_kpc = 8.122  # Sun-GC distance
            sun_position = np.array([[-r0_kpc * SCALE, 0.0, 0.0]], dtype=np.float32)
            sun_color = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)  # Yellow

            scene.add_points(
                "Sun",
                sun_position,
                colors=sun_color,
                radii=marker_radius,
                opacity=1.0,
                blending_mode="normal",
                layer=True,
            )
            aprint(f"  ✓ Sun at ({-r0_kpc * SCALE:.1f}, 0, 0)")

            # Betelgeuse (red supergiant in Orion, ~168 pc from Sun)
            betelgeuse_pos = np.array(
                [[-8.278192 * SCALE, -0.056179 * SCALE, -0.004962 * SCALE]],
                dtype=np.float32,
            )
            betelgeuse_color = np.array([[1.0, 0.3, 0.0]], dtype=np.float32)  # Red

            scene.add_points(
                "Betelgeuse",
                betelgeuse_pos,
                colors=betelgeuse_color,
                radii=marker_radius,
                opacity=1.0,
                blending_mode="normal",
                layer=True,
            )
            aprint("  ✓ Betelgeuse (red supergiant, 168 pc)")

            # Rigel (blue supergiant in Orion, ~265 pc from Sun)
            rigel_pos = np.array(
                [[-8.331409 * SCALE, -0.117085 * SCALE, -0.091686 * SCALE]],
                dtype=np.float32,
            )
            rigel_color = np.array([[0.5, 0.7, 1.0]], dtype=np.float32)  # Blue

            scene.add_points(
                "Rigel",
                rigel_pos,
                colors=rigel_color,
                radii=marker_radius,
                opacity=1.0,
                blending_mode="normal",
                layer=True,
            )
            aprint("  ✓ Rigel (blue supergiant, 265 pc)")

            # Overlay annotations
            scene.add_text(
                "Milky Way (Gaia DR3)",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                "3M stars \u2022 Galactocentric coords",
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


def load_and_convert_from_zip(data_zip_path: Path, temp_dir: Path) -> Path:
    """Load raw Gaia data from zip and convert to Luxar format.

    Args:
        data_zip_path: Path to galaxy.zarr.zip
        temp_dir: Temporary directory for extraction

    Returns:
        Path to Luxar-formatted zarr
    """
    with asection("Loading Gaia DR3 Dataset"):
        if not data_zip_path.exists():
            aprint(f"❌ Error: Data file not found: {data_zip_path}")
            aprint("")
            aprint("The galaxy.zarr.zip file should be in:")
            aprint(f"  {data_zip_path.parent}/")
            aprint("")
            aprint("To generate the dataset:")
            aprint("  cd scripts")
            aprint("  hatch run python generate_galaxy_simple.py --count 3000000")
            raise FileNotFoundError(f"Data file not found: {data_zip_path}")

        aprint(f"Data file: {data_zip_path}")
        aprint(f"Size: {data_zip_path.stat().st_size / 1e6:.1f} MB")

        # Extract zarr from zip to temp directory
        aprint("\nExtracting Gaia data from zip...")
        import zipfile

        with zipfile.ZipFile(data_zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        raw_zarr_path = temp_dir / "milky_way_gaia_3m.zarr"
        aprint(f"✓ Extracted to: {raw_zarr_path}")

    # Convert to Luxar format
    luxar_zarr_path = temp_dir / "galaxy.luxar.zarr"
    load_and_convert_gaia_data(raw_zarr_path, luxar_zarr_path)

    return luxar_zarr_path


def main() -> None:
    """Main demo entry point."""
    aprint("=" * 70)
    aprint("GAIA DR3 MILKY WAY - 3 MILLION REAL STARS")
    aprint("=" * 70)
    aprint("")
    aprint("Visualizing 3,000,000 real stars from ESA's Gaia Data Release 3!")
    aprint("")
    aprint("Dataset Details:")
    aprint("  • Source: Gaia DR3 (European Space Agency)")
    aprint("  • Stars: 3 million brightest (mag 1.94 to 12.62)")
    aprint("  • Coordinates: Galactocentric (x, y, z in kpc)")
    aprint("  • Colors: Real stellar temperatures from BP-RP photometry")
    aprint("  • Quality: High-precision (parallax_over_error > 5)")
    aprint("")
    aprint("What You'll See:")
    aprint("  • The thin disk of the Milky Way")
    aprint("  • Dense central bulge (Galactic Center)")
    aprint("  • Real stellar distribution and colors")
    aprint("  • You are viewing from the Sun's position!")
    aprint("")
    aprint("Color Guide:")
    aprint("  🔵 Blue/White:  Hot young stars (O, B, A types)")
    aprint("  🟡 Yellow:      Sun-like stars (F, G types)")
    aprint("  🔴 Red/Orange:  Cool old stars (K, M types)")
    aprint("")
    aprint("Scientific Context:")
    aprint("  This is REAL astronomical data! Each point is an actual star")
    aprint("  measured by the Gaia space telescope. The colors represent real")
    aprint("  stellar temperatures, and the positions are transformed to a")
    aprint("  Galactocentric reference frame.")
    aprint("")
    aprint("  The Sun (and Earth) is at approximately (-8.122, 0, 0) kpc")
    aprint("  from the Galactic Center. You're viewing our galaxy from home!")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "galaxy.luxar.zarr"
        try:
            # Extract the raw .zarr from the zip to a temp dir, then convert to the
            # persistent output_path (same extraction the serve path uses — reading
            # the zip in place via a zip:// store is unreliable across zarr versions).
            import zipfile

            with tempfile.TemporaryDirectory(prefix="luxar_demo_gaia_") as tmpdir:
                with zipfile.ZipFile(DATA_FILE, "r") as zf:
                    zf.extractall(tmpdir)
                load_and_convert_gaia_data(
                    Path(tmpdir) / "milky_way_gaia_3m.zarr", output_path
                )
        except FileNotFoundError as e:
            aprint(f"\n❌ Error: {e}")
            sys.exit(1)
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_gaia_") as tmpdir:
        tmp_path = Path(tmpdir)

        # Load from zip and convert to Luxar format (extracts to temp_dir)
        zarr_path = load_and_convert_from_zip(DATA_FILE, tmp_path)

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS")
        aprint("=" * 70)
        aprint("")
        aprint("Navigation:")
        aprint("  • Start zoomed OUT to see the full galactic structure")
        aprint("  • Look for the thin disk and central bulge")
        aprint("  • The Milky Way is a flat disk ~30 kpc across")
        aprint("  • Zoom IN to see individual stars with colors")
        aprint("")
        aprint("What to Explore:")
        aprint("  • Yellow marker: Our Sun (you are here!)")
        aprint("  • Red marker: Betelgeuse (red supergiant, 168 pc)")
        aprint("  • Blue marker: Rigel (blue supergiant, 265 pc)")
        aprint("  • Origin (0,0,0): The Galactic Center (8 kpc away)")
        aprint("  • Top-down view: See the disk structure")
        aprint("  • Edge-on view: See how thin the disk is")
        aprint("  • Navigate toward origin to approach Galactic Center")
        aprint("")
        aprint("This is Real Science:")
        aprint("  • Every point is a real star with measured position")
        aprint("  • Colors reflect actual stellar surface temperatures")
        aprint("  • Distances determined from parallax measurements")
        aprint("  • Published in: Gaia Collaboration (2022), A&A")
        aprint("")

        # Launch viewer
        launch_viewer(zarr_path)

    aprint("")
    aprint("Cleanup complete - temporary files removed")
    aprint("")
    aprint("=" * 70)
    aprint("DATA ATTRIBUTION & SOURCES")
    aprint("=" * 70)
    aprint("")
    aprint("Data Credit: ESA/Gaia/DPAC")
    aprint("  Mission: https://www.cosmos.esa.int/gaia")
    aprint("  Archive: https://gea.esac.esa.int/archive/")
    aprint("")
    aprint("Citation:")
    aprint("  Gaia Collaboration, Vallenari et al. (2023)")
    aprint('  "Gaia Data Release 3: Summary of the content and survey properties"')
    aprint("  Astronomy & Astrophysics, 674, A1")
    aprint("  https://doi.org/10.1051/0004-6361/202243940")
    aprint("")
    aprint("Data Generation:")
    aprint("  See: scripts/generate_galaxy_simple.py for data generation")
    aprint("  Source: ESA Gaia DR3 (https://gea.esac.esa.int/archive/)")
    aprint("")
    aprint("Coordinate System:")
    aprint("  Galactocentric frame with R₀ = 8.122 kpc")
    aprint("  Reference: GRAVITY Collaboration (2018), A&A 615, L15")
    aprint("  https://doi.org/10.1051/0004-6361/201833718")
    aprint("")
    aprint("Thank you for exploring our galaxy!")
    aprint("")


if __name__ == "__main__":
    main()
