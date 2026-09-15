#!/usr/bin/env python3
"""Self-Contained Demo: Realistic Multi-Armed Spiral Galaxy

This demo demonstrates:
- Beautiful multi-armed logarithmic spiral structure
- Realistic star distribution with density falloff
- Color variation (blue young stars in arms, red/yellow old stars in bulge)
- Central galactic bulge with different stellar population
- Stellar halo with sparse old stars
- Realistic astronomical scales and proportions
- Complete workflow: generate → serve → view → cleanup

The demo is completely self-contained - all galaxy generation code is in this file.

Mathematical Background:
    Spiral galaxies follow logarithmic spirals described by:
        r(θ) = a × exp(b × θ)

    Where:
    - r is distance from center
    - θ is angle
    - a is initial radius
    - b controls tightness (b=0.3 is typical)

    Multiple arms are created by rotating the spiral by 2π/n_arms. Stars are
    distributed along arms with random scatter and density that decreases with
    radius following an exponential profile.

Usage:
    python demo_spiral_galaxy.py [--stars=N] [--arms=N]

Controls:
    - Mouse to rotate and explore
    - Zoom in to see individual stars
    - Notice blue young stars in spiral arms
    - Red/yellow old stars in central bulge
    - Ctrl+C to stop and cleanup
"""

DEMO_META = {
    "key": "spiral_galaxy",
    "title": "Spiral Galaxy",
    "description": "A procedural multi-armed logarithmic-spiral galaxy with realistic stellar populations.",
    "category": "synthetic",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["spiral_galaxy"],
    # Procedurally generated: no external dataset, nothing to credit.
    "citation": None,
}

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import add_demo_caption, launch_viewer
from luxar.demos._lod_policy import stream_ladder
from luxar.utils.paths import get_demos_output_dir


def generate_spiral_arm(
    n_stars: int,
    arm_offset: float,
    rng: np.random.Generator,
    inner_radius: float = 1.0,
    outer_radius: float = 20.0,
    pitch_angle: float = 0.3,
    scatter: float = 0.8,
    thickness: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one spiral arm with realistic star distribution.

    Args:
        n_stars: Number of stars in this arm
        arm_offset: Angular offset for this arm (radians)
        rng: Seeded random generator for reproducibility
        inner_radius: Start of spiral
        outer_radius: End of spiral
        pitch_angle: Spiral tightness (0.2-0.4)
        scatter: Random scatter perpendicular to arm
        thickness: Vertical thickness of galaxy disk

    Returns:
        Tuple of (positions, ages) where ages indicate star age (0=young, 1=old)
    """
    # Sample radii with Gaussian density profile for smooth natural falloff
    # Peak density at mid-radius, smooth decay toward edges
    mid_radius = (inner_radius + outer_radius) / 2
    sigma_radius = (outer_radius - inner_radius) / 3

    # Gaussian distribution centered at mid_radius
    radii = rng.normal(mid_radius, sigma_radius, n_stars)

    # Only clip negative values, allow natural extension beyond outer_radius
    # This creates perfectly smooth density falloff with no sharp edges
    radii = np.abs(radii)  # Mirror negative values for symmetry

    # Calculate spiral angle for each radius
    # Inverse of logarithmic spiral: θ = log(r/a) / b
    theta_spiral = np.log(radii / inner_radius) / pitch_angle + arm_offset

    # Add scatter perpendicular to spiral
    # Tighter scatter for more defined arms (better contrast)
    scatter_angle = rng.normal(0, scatter * 0.7, n_stars)
    theta = theta_spiral + scatter_angle

    # Convert to Cartesian (2D in disk plane)
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)

    # Add vertical thickness (thin disk)
    z = rng.normal(0, thickness * (1 + radii / outer_radius * 0.5), n_stars)

    positions = np.column_stack([x, y, z])

    # Star age: younger stars in outer arms (blue), older toward center (red)
    # Age inversely proportional to radius
    ages = 1.0 - (radii - inner_radius) / (outer_radius - inner_radius)
    ages = np.clip(ages + rng.normal(0, 0.1, n_stars), 0, 1)

    return positions.astype(np.float32), ages.astype(np.float32)


def generate_galactic_bulge(
    n_stars: int,
    rng: np.random.Generator,
    bulge_radius: float = 3.0,
    bulge_height: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate central galactic bulge with smooth Gaussian distribution.

    Args:
        n_stars: Number of stars in bulge
        rng: Seeded random generator for reproducibility
        bulge_radius: Scale radius (sigma) of bulge
        bulge_height: Vertical scale (sigma)

    Returns:
        Tuple of (positions, ages)
    """
    # Pure Gaussian distribution for smooth, natural bulge
    # No hard cutoff - density fades naturally with distance
    positions = rng.standard_normal((n_stars, 3))

    # Scale to ellipsoidal Gaussian (flattened in z)
    positions[:, 0] *= bulge_radius  # x direction
    positions[:, 1] *= bulge_radius  # y direction
    positions[:, 2] *= bulge_height  # z (flattened)

    # No hard cutoff - keep all stars for smooth Gaussian profile
    # This creates natural density falloff without sharp edges

    # All bulge stars are old (red/yellow)
    # Slight age variation for realism
    ages = rng.uniform(0.7, 1.0, n_stars).astype(np.float32)

    return positions.astype(np.float32), ages


def generate_stellar_halo(
    n_stars: int,
    rng: np.random.Generator,
    halo_radius: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sparse stellar halo with ancient stars.

    Args:
        n_stars: Number of halo stars
        rng: Seeded random generator for reproducibility
        halo_radius: Extent of halo

    Returns:
        Tuple of (positions, ages)
    """
    # Spherical distribution, very sparse
    positions = rng.standard_normal((n_stars, 3))
    positions = positions / np.linalg.norm(positions, axis=1, keepdims=True)

    # Radial distribution (1/r² falloff, confined to halo_radius)
    u = rng.random(n_stars)
    r = halo_radius * (1 - u) ** 0.5  # Power law distribution
    positions *= r[:, np.newaxis]

    # Very old stars
    ages = np.ones(n_stars, dtype=np.float32)

    return positions.astype(np.float32), ages


def age_to_color(ages: np.ndarray) -> np.ndarray:
    """Convert stellar age to realistic color.

    Args:
        ages: Age array (0=young/blue, 1=old/red)

    Returns:
        RGB colors (n_stars, 3)
    """
    colors = np.zeros((len(ages), 3), dtype=np.float32)

    # Young stars (age < 0.3): Blue-white (hot)
    young = ages < 0.3
    colors[young, 0] = 0.7 + 0.3 * (1 - ages[young] / 0.3)  # Some red
    colors[young, 1] = 0.8 + 0.2 * (1 - ages[young] / 0.3)  # More green
    colors[young, 2] = 1.0  # Full blue

    # Middle age (0.3 - 0.7): Yellow-white
    middle = (ages >= 0.3) & (ages < 0.7)
    t = (ages[middle] - 0.3) / 0.4
    colors[middle, 0] = 1.0  # Full red
    colors[middle, 1] = 1.0 - 0.3 * t  # Decreasing green
    colors[middle, 2] = 0.8 - 0.6 * t  # Decreasing blue

    # Old stars (age >= 0.7): Red-orange (cool)
    old = ages >= 0.7
    t = (ages[old] - 0.7) / 0.3
    colors[old, 0] = 1.0  # Full red
    colors[old, 1] = 0.6 - 0.3 * t  # Some orange
    colors[old, 2] = 0.2 - 0.2 * t  # Little blue

    return colors  # type: ignore[no-any-return]


def generate_spiral_galaxy(
    output_path: Path,
    n_stars: int = 500000,
    n_arms: int = 4,
    arm_stars_ratio: float = 0.7,
    bulge_stars_ratio: float = 0.25,
    halo_stars_ratio: float = 0.05,
    seed: int = 0,
) -> int:
    """Generate a realistic multi-armed spiral galaxy.

    This function contains ALL the generation logic - completely self-contained.

    Args:
        output_path: Where to write zarr store
        n_stars: Total number of stars
        n_arms: Number of spiral arms (2-6 typical)
        arm_stars_ratio: Fraction of stars in spiral arms
        bulge_stars_ratio: Fraction in central bulge
        halo_stars_ratio: Fraction in stellar halo
        seed: Seed for the star-distribution RNG (reproducible output)

    Returns:
        Total number of stars generated
    """
    rng = np.random.default_rng(seed)

    with asection(f"Generating Spiral Galaxy ({n_stars:,} stars, {n_arms} arms)"):
        # Calculate star counts
        n_arm_stars = int(n_stars * arm_stars_ratio)
        n_bulge_stars = int(n_stars * bulge_stars_ratio)
        n_halo_stars = int(n_stars * halo_stars_ratio)

        stars_per_arm = n_arm_stars // n_arms

        aprint(f"Spiral arms: {n_arms} arms with {stars_per_arm:,} stars each")
        aprint(f"Central bulge: {n_bulge_stars:,} stars")
        aprint(f"Stellar halo: {n_halo_stars:,} stars")

        all_positions = []
        all_ages = []

        # Generate spiral arms
        with asection("Generating Spiral Arms"):
            for i in range(n_arms):
                arm_angle = i * 2 * np.pi / n_arms
                aprint(
                    f"  Arm {i + 1}/{n_arms} (offset: {np.degrees(arm_angle):.1f}°)..."
                )

                positions, ages = generate_spiral_arm(
                    stars_per_arm,
                    arm_offset=arm_angle,
                    rng=rng,
                    inner_radius=2.0,
                    outer_radius=25.0,
                    pitch_angle=0.3,
                    scatter=0.4,  # Reduced scatter for better arm contrast
                    thickness=0.4,
                )

                all_positions.append(positions)
                all_ages.append(ages)

            aprint(f"✓ Generated {n_arms} spiral arms with {n_arm_stars:,} total stars")

        # Generate central bulge
        with asection("Generating Galactic Bulge"):
            bulge_pos, bulge_ages = generate_galactic_bulge(
                n_bulge_stars,
                rng=rng,
                bulge_radius=4.0,
                bulge_height=3.0,
            )
            all_positions.append(bulge_pos)
            all_ages.append(bulge_ages)
            aprint(f"✓ Generated central bulge with {len(bulge_pos):,} stars")

        # Generate stellar halo
        with asection("Generating Stellar Halo"):
            halo_pos, halo_ages = generate_stellar_halo(
                n_halo_stars,
                rng=rng,
                halo_radius=35.0,
            )
            all_positions.append(halo_pos)
            all_ages.append(halo_ages)
            aprint(f"✓ Generated stellar halo with {n_halo_stars:,} stars")

        # Combine all components
        aprint("\nCombining galactic components...")
        positions = np.vstack(all_positions)
        ages = np.concatenate(all_ages)

        aprint(f"✓ Total stars: {len(positions):,}")

        # Generate realistic stellar colors
        aprint("Generating realistic stellar colors...")
        colors = age_to_color(ages)

        aprint("✓ Colors: Blue (young) → Yellow (middle) → Red (old)")

        # Generate radii based on stellar type
        # Younger stars slightly larger (visual weight in arms)
        # Older stars smaller
        aprint("Calculating stellar sizes...")
        radii = 0.04 + (1 - ages) * 0.03  # Young stars bigger
        radii = radii.astype(np.float32)

        # Sharpness varies with age (young stars sharper); normalized [0, 1] knob
        sharpnesses = (0.6 + ages * 0.25).astype(np.float32)

        aprint(f"✓ Radii range: [{radii.min():.3f}, {radii.max():.3f}]")

    # Write to Zarr
    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension("x", unit="kpc", display=True),
                Dimension("y", unit="kpc", display=True),
                Dimension("z", unit="kpc", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims, viewer_config=ViewerConfig(cinematic_mode=True)
            )

            scene.add_points(
                "SpiralGalaxy",
                positions,
                colors=colors,
                radii=radii,
                sharpness=sharpnesses,
                opacity=0.9,
                blending_mode="additive",
                intensity=0.278,
                layer=True,
                additive_lod=stream_ladder(len(positions)),
            )

            # Overlay annotations
            scene.add_text(
                "Spiral Galaxy",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                f"{n_arms} arms \u2022 {len(positions) / 1000:.0f}K stars",
                DEMO_META.get("citation"),
            )

        aprint(f"✓ Written to {output_path}")
        aprint(f"✓ Dataset size: ~{len(positions) * 40 / 1024 / 1024:.1f} MB")

    return len(positions)


def main() -> None:
    """Main demo entry point."""
    # Parse arguments
    n_stars = 500000  # Half million stars for dense, realistic galaxy
    n_arms = 4  # Classic 4-armed spiral

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--stars="):
                n_stars = int(arg.split("=")[1])
            elif arg.startswith("--arms="):
                n_arms = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("SPIRAL GALAXY DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Generating a realistic multi-armed spiral galaxy!")
    aprint(f"Total stars: {n_stars:,}")
    aprint(f"Spiral arms: {n_arms}")
    aprint("")
    aprint("Galactic structure:")
    aprint("  • Spiral arms (70%): Young blue stars in logarithmic spirals")
    aprint("  • Central bulge (25%): Old red/yellow stars in ellipsoid")
    aprint("  • Stellar halo (5%): Ancient stars in sparse spherical distribution")
    aprint("")
    aprint("Realistic features:")
    aprint("  • Exponential density falloff with radius")
    aprint("  • Color-magnitude relationship (blue=hot, red=cool)")
    aprint("  • Size variation with stellar age")
    aprint("  • Logarithmic spiral structure")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "spiral_galaxy.luxar.zarr"
        generate_spiral_galaxy(
            output_path,
            n_stars=n_stars,
            n_arms=n_arms,
            arm_stars_ratio=0.70,
            bulge_stars_ratio=0.25,
            halo_stars_ratio=0.05,
        )
        aprint(f"✓ Dataset generated at {output_path}")
        return

    # Use temporary directory (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_galaxy_") as tmpdir:
        output_path = Path(tmpdir) / "spiral_galaxy.luxar.zarr"

        # Generate galaxy
        generate_spiral_galaxy(
            output_path,
            n_stars=n_stars,
            n_arms=n_arms,
            arm_stars_ratio=0.70,
            bulge_stars_ratio=0.25,
            halo_stars_ratio=0.05,
        )

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS")
        aprint("=" * 70)
        aprint("Once the viewer opens:")
        aprint("")
        aprint("  • Rotate to see the galaxy from different angles")
        aprint("  • Top-down view: See the beautiful spiral structure")
        aprint("  • Edge-on view: See the thin disk and central bulge")
        aprint("  • Zoom in: Individual stars with realistic colors")
        aprint("  • Zoom out: Overall galactic structure")
        aprint("")
        aprint("Color coding:")
        aprint("  🔵 Blue: Young, hot stars in spiral arms")
        aprint("  🟡 Yellow: Middle-aged stars")
        aprint("  🔴 Red: Old, cool stars in bulge and halo")
        aprint("")
        aprint("Try different configurations:")
        aprint("  --arms=2  → Barred spiral")
        aprint("  --arms=3  → Triangular spiral")
        aprint("  --arms=5  → Dense multi-arm spiral")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Browser will open automatically. Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    aprint("✓ Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
