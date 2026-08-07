#!/usr/bin/env python3
"""Self-Contained Demo: 5D Spiral Galaxy with Time Evolution

This demo demonstrates:
- Creating large-scale 5D data with millions of points
- Multiple interleaved spiral arms evolving over time
- Continuous and discrete dimension navigation
- Time-based animation showing spiral arm rotation
- Channel-based coloring (different emission wavelengths)
- Complete workflow: generate → serve → view → cleanup

The demo is completely self-contained - all generation code is in this file.

Mathematical Background:
    Multiple logarithmic spiral arms are generated with:
    - Radial expansion following r = a * exp(b * theta)
    - Vertical oscillation creating 3D wave patterns
    - Time evolution rotating and expanding the arms
    - Channel variation representing different wavelengths/populations

Usage:
    python demo_5d_spiral_galaxy.py [--points=N]

Controls:
    - Press '4' to select T dimension (time), then use [ ] to navigate
    - Press '5' to select Channel dimension, then use [ ] to navigate
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
"""

DEMO_META = {
    "key": "spiral_galaxy_5d",
    "title": "Spiral Galaxy 5D",
    "description": "Procedurally generated 5D spiral galaxy: logarithmic arms evolving over time and channels.",
    "category": "synthetic",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["spiral_galaxy_5d"],
}

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir


def generate_5d_spiral_galaxy(
    output_path: Path,
    n_points_per_arm: int = 50000,
    n_arms: int = 4,
    n_time_steps: int = 20,
    n_channels: int = 3,
) -> int:
    """Generate a 5D spiral galaxy with time evolution and channels.

    This function contains ALL the generation logic - completely self-contained.

    Creates a spiral galaxy structure where:
    - Each arm is a logarithmic spiral
    - Points evolve over time (rotation + expansion)
    - Different channels represent different stellar populations

    Args:
        output_path: Where to write the zarr store
        n_points_per_arm: Points per spiral arm per time step per channel
        n_arms: Number of spiral arms
        n_time_steps: Number of time evolution steps
        n_channels: Number of wavelength channels

    Returns:
        Total number of points generated
    """
    total_points = n_points_per_arm * n_arms * n_time_steps * n_channels

    with asection(f"Generating 5D Spiral Galaxy ({total_points:,} points)"):
        aprint(f"Points per arm: {n_points_per_arm:,}")
        aprint(f"Spiral arms: {n_arms}")
        aprint(f"Time steps: {n_time_steps}")
        aprint(f"Channels: {n_channels}")
        aprint(f"Total points: {total_points:,}")

        # Channel colors representing different stellar populations
        # Young stars (blue), intermediate (green/yellow), old stars (red)
        channel_base_colors = np.array(
            [
                [0.3, 0.5, 1.0],  # Channel 0: Young hot stars (blue)
                [0.8, 0.9, 0.3],  # Channel 1: Intermediate (yellow-green)
                [1.0, 0.4, 0.2],  # Channel 2: Old cool stars (red-orange)
            ],
            dtype=np.float32,
        )

        # Pre-allocate arrays
        all_positions = []
        all_colors = []
        all_radii = []

        # Galaxy parameters
        galaxy_radius = 80.0  # Maximum radius
        galaxy_thickness = 15.0  # Vertical spread
        spiral_tightness = 0.3  # How tightly wound (smaller = tighter)

        with asection("Generating spiral arms"):
            for t in range(n_time_steps):
                time_fraction = t / max(n_time_steps - 1, 1)
                time_rotation = time_fraction * 0.5 * np.pi  # Rotate over time

                for c in range(n_channels):
                    # Each channel has slightly different distribution
                    # Young stars more concentrated in arms, old stars more spread
                    concentration = 1.0 - c * 0.2

                    for arm_idx in range(n_arms):
                        # Arm starting angle (evenly distributed)
                        arm_offset = arm_idx * 2 * np.pi / n_arms

                        # Random parameter along spiral (not uniform in angle)
                        rng = np.random.default_rng(seed=t * 1000 + c * 100 + arm_idx)
                        u = rng.uniform(0, 1, n_points_per_arm)

                        # Logarithmic spiral: r = a * exp(b * theta)
                        # Use square root for more points near center
                        r_param = np.sqrt(u) * galaxy_radius

                        # Theta from radius (inverse of log spiral)
                        theta = np.log(r_param / 5.0 + 1) / spiral_tightness
                        theta += arm_offset + time_rotation

                        # Add spread perpendicular to arm (more for old stars)
                        arm_width = 5.0 + (1 - concentration) * 10.0
                        theta += rng.normal(0, arm_width / r_param.clip(10, None))

                        # Convert to Cartesian
                        x = r_param * np.cos(theta)
                        y = r_param * np.sin(theta)

                        # Z varies with radius (thicker disk at center)
                        z_scale = galaxy_thickness * (1 - r_param / galaxy_radius * 0.5)
                        z = rng.normal(0, z_scale * (1 - concentration * 0.5))

                        # Create 5D positions [X, Y, Z, W(time), Channel]
                        positions = np.zeros((n_points_per_arm, 5), dtype=np.float32)
                        positions[:, 0] = x
                        positions[:, 1] = y
                        positions[:, 2] = z
                        positions[:, 3] = t  # Time coordinate
                        positions[:, 4] = c  # Channel coordinate

                        all_positions.append(positions)

                        # Colors: base color with brightness variation
                        brightness = 0.5 + 0.5 * (1 - r_param / galaxy_radius)
                        brightness *= rng.uniform(0.7, 1.3, n_points_per_arm)
                        colors = (
                            np.tile(channel_base_colors[c], (n_points_per_arm, 1))
                            * brightness[:, np.newaxis]
                        )
                        all_colors.append(colors.astype(np.float32))

                        # Radii: vary with position (larger near center)
                        base_radius = 1.5 - c * 0.3  # Smaller for older populations
                        radii = base_radius * (
                            0.5 + 0.5 * (1 - r_param / galaxy_radius)
                        )
                        radii *= rng.uniform(0.8, 1.2, n_points_per_arm)
                        all_radii.append(radii.astype(np.float32))

                aprint(f"  Time step {t + 1}/{n_time_steps} complete")

        # Combine all data
        with asection("Combining data"):
            positions = np.vstack(all_positions)
            colors = np.vstack(all_colors)
            radii = np.concatenate(all_radii)
            aprint(f"Final shape: {positions.shape}")

    # Write to zarr
    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension(name="X", unit="kpc", range=(-100, 100), display=True),
                Dimension(name="Y", unit="kpc", range=(-100, 100), display=True),
                Dimension(name="Z", unit="kpc", range=(-30, 30), display=True),
                Dimension(
                    name="T",
                    unit="Myr",
                    range=(0, n_time_steps - 1),
                    display=False,
                    spatial=True,
                    step=1.0,
                ),
                Dimension(
                    name="Channel",
                    unit="",
                    categories=["Young Stars", "Intermediate", "Old Stars"],
                    display=False,
                    description="Stellar population age - young (blue), intermediate (yellow), old (red)",
                ),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "SpiralGalaxy",
                positions,
                colors=colors,
                radii=radii,
                sharpness=0.55,
                opacity=0.85,
                gamma=1.0,
                blending_mode="additive",
                intensity=5.58,
                layer=True,
            )

            # --- Overlays ---
            # Title
            scene.add_text(
                "5D Spiral Galaxy",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Dimension-aware labels for Channel
            channel_labels = ["Young Stars", "Intermediate", "Old Stars"]
            for i, label in enumerate(channel_labels):
                scene.add_text(
                    label,
                    position=(0.02, 0.97),
                    font_size=0.015,
                    anchor="bottom-left",
                    color="#ffcc44",
                    visible_range={"Channel": float(i)},
                    transition="fade",
                    transition_duration=0.15,
                )

            # Info
            scene.add_text(
                "1.2M stars \u2022 3 populations",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"Written to {output_path}")

    return total_points


def main() -> None:
    """Main demo entry point."""
    # Default parameters for visualization (~1.2M points total)
    n_points = 5000  # Per arm per time step per channel

    if len(sys.argv) > 1 and sys.argv[1].startswith("--points="):
        n_points = int(sys.argv[1].split("=")[1])

    n_arms = 4
    n_time_steps = 20
    n_channels = 3

    total = n_points * n_arms * n_time_steps * n_channels

    aprint("=" * 70)
    aprint("5D SPIRAL GALAXY DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Generating a multi-dimensional spiral galaxy visualization")
    aprint(f"Points per arm/time/channel: {n_points:,}")
    aprint(f"Spiral arms: {n_arms}")
    aprint(f"Time steps: {n_time_steps}")
    aprint(f"Channels: {n_channels}")
    aprint(f"Total points: {total:,}")
    aprint("")
    aprint("Dimensions:")
    aprint("  X, Y, Z: Spatial coordinates (kpc)")
    aprint("  T: Time evolution (Myr)")
    aprint("  Channel: Stellar population (0=young, 1=intermediate, 2=old)")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "spiral_galaxy_5d.luxar.zarr"
        generate_5d_spiral_galaxy(
            output_path,
            n_points_per_arm=n_points,
            n_arms=n_arms,
            n_time_steps=n_time_steps,
            n_channels=n_channels,
        )
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_5d_galaxy_") as tmpdir:
        output_path = Path(tmpdir) / "spiral_galaxy_5d.luxar.zarr"

        # Generate the dataset
        generate_5d_spiral_galaxy(
            output_path,
            n_points_per_arm=n_points,
            n_arms=n_arms,
            n_time_steps=n_time_steps,
            n_channels=n_channels,
        )

        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("The viewer will open in your browser automatically.")
        aprint("Press Ctrl+C when done to stop and cleanup.")
        aprint("")
        aprint("Navigation controls:")
        aprint("  - Press '4' to select T (time), then [ ] to animate")
        aprint("  - Press '5' to select Channel, then [ ] to switch populations")
        aprint("  - Use mouse to orbit/zoom")
        aprint("  - Blue = young stars, Yellow = intermediate, Red = old stars")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
