#!/usr/bin/env python3
"""Self-Contained Demo: Point Sharpness Showcase

This demo demonstrates:
- Comprehensive showcase of the point sharpness feature in Luxar
- Sharpness gradient showing smooth transition from peaky (0.0) to hard-edged (1.0)
- Fixed sharpness comparison with labeled rows (Peaky to Hard-edged)
- Mixed sharpness cloud with color-coded sharpness values
- Sinusoidal sharpness wave pattern
- Complete workflow: generate → serve → view → cleanup

The demo is completely self-contained - all generation code is in this file.

Sharpness Parameter:
    The sharpness parameter is a normalized [0, 1] knob controlling the edge
    falloff of points. The viewer maps it to a super-Gaussian falloff exponent
    beta = 2^(6s - 2):
    - s = 0.0: beta = 0.25, a peaky/cuspy profile.
    - s = 0.5: beta = 2.0, a true Gaussian (the default).
    - s = 1.0: beta = 16.0, a hard, disc-like edge.
    - Falloff: intensity = exp(-(r²)^(beta/2)) (shifted-truncated super-Gaussian).

Usage:
    python demo_sharpness_showcase.py [--points N]

Controls:
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir


def create_sharpness_gradient_example(scene, n_points: int = 5000) -> None:
    """Create a grid showing gradual sharpness transition.

    Generates a grid of points where sharpness increases from left to right,
    creating a smooth gradient from soft glowing points to sharp disc-like points.

    Args:
        scene: LuxarZarrCompiler scene to add points to
        n_points: Number of points in the grid (default: 5000)
    """
    with asection("Creating sharpness gradient"):
        # Create a grid of points
        grid_size = int(np.sqrt(n_points))
        x = np.linspace(-10, 10, grid_size)
        y = np.linspace(-5, 5, grid_size)
        xx, yy = np.meshgrid(x, y)

        positions = np.column_stack(
            [xx.flatten(), yy.flatten(), np.zeros(xx.size)]
        ).astype(np.float32)

        # Sharpness increases from left to right (normalized [0, 1] knob)
        normalized_x = (positions[:, 0] + 10) / 20  # 0 to 1
        sharpness = normalized_x.astype(np.float32)  # 0.0 to 1.0

        # All points same size for fair comparison
        radii = np.full(positions.shape[0], 0.3, dtype=np.float32)

        # Color gradient to visualize sharpness (blue → red)
        colors = np.zeros((positions.shape[0], 3), dtype=np.float32)
        colors[:, 0] = normalized_x  # Red increases
        colors[:, 2] = 1 - normalized_x  # Blue decreases

        # Offset vertically
        positions[:, 1] += 10

        scene.add_points(
            "SharpnessGradient",
            positions,
            colors,
            radii=radii,
            sharpness=sharpness,
            layer=True,
            intensity=0.125,
        )
        aprint(f"✓ Added {len(positions):,} points (gradient: soft blue → sharp red)")


def create_sharpness_comparison_example(scene) -> None:
    """Create rows of points with different fixed sharpness values.

    Each row demonstrates a specific sharpness value with labeled examples,
    from peaky (0.0) to hard-edged (1.0) on the normalized knob.

    Args:
        scene: LuxarZarrCompiler scene to add points to
    """
    with asection("Creating sharpness comparison rows"):
        n_points_per_row = 20
        sharpness_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = [
            "Peaky (0.0)",
            "Soft (0.2)",
            "Near-Gaussian (0.4)",
            "Crisp (0.6)",
            "Hard (0.8)",
            "Maximum (1.0)",
        ]

        for i, (sharp_val, label) in enumerate(zip(sharpness_values, labels)):
            # Create a row of points
            x = np.linspace(-8, 8, n_points_per_row)
            y = np.full(n_points_per_row, i * 2.5 - 5)
            z = np.zeros(n_points_per_row)

            positions = np.column_stack([x, y, z]).astype(np.float32)

            # All points have the same sharpness value
            sharpness = np.full(n_points_per_row, sharp_val, dtype=np.float32)
            radii = np.full(n_points_per_row, 0.4, dtype=np.float32)

            # Different color for each row (rainbow)
            hue = i / len(sharpness_values)
            if hue < 1 / 6:
                r, g, b = 1.0, hue * 6, 0
            elif hue < 2 / 6:
                r, g, b = 2 - hue * 6, 1.0, 0
            elif hue < 3 / 6:
                r, g, b = 0, 1.0, hue * 6 - 2
            elif hue < 4 / 6:
                r, g, b = 0, 4 - hue * 6, 1.0
            else:
                r, g, b = hue * 6 - 4, 0, 1.0

            colors = np.tile([r, g, b], (n_points_per_row, 1)).astype(np.float32)

            # Offset to the right
            positions[:, 0] += 15

            scene.add_points(
                f"Sharpness_{label}",
                positions,
                colors,
                radii=radii,
                sharpness=sharpness,
                layer=True,
                intensity=0.125,
            )

        aprint(
            f"✓ Added {len(sharpness_values) * n_points_per_row:,} points (6 comparison rows)"
        )


def create_mixed_sharpness_example(scene, n_points: int = 10000) -> None:
    """Create a sphere with mixed sharpness values.

    Generates a spherical cloud with randomly distributed sharpness values,
    creating clusters of soft (blue), medium (green), and sharp (red) points.

    Args:
        scene: LuxarZarrCompiler scene to add points to
        n_points: Number of points in the sphere (default: 10000)
    """
    with asection("Creating mixed sharpness cloud"):
        rng = np.random.default_rng(789)

        # Generate points in a sphere
        theta = rng.uniform(0, 2 * np.pi, n_points)
        phi = np.arccos(rng.uniform(-1, 1, n_points))
        r = rng.uniform(3, 5, n_points)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        positions = np.column_stack([x, y, z]).astype(np.float32)

        # Mix of sharpness values - create clusters (normalized [0, 1] knob)
        sharpness = np.zeros(n_points, dtype=np.float32)
        # 1/3 soft (peaky) points
        sharpness[: n_points // 3] = rng.uniform(0.1, 0.3, n_points // 3)
        # 1/3 medium (near-Gaussian) points
        sharpness[n_points // 3 : 2 * n_points // 3] = rng.uniform(
            0.4, 0.6, n_points // 3
        )
        # 1/3 sharp (hard-edged) points
        sharpness[2 * n_points // 3 :] = rng.uniform(
            0.75, 1.0, n_points - 2 * n_points // 3
        )

        # Shuffle to mix them
        rng.shuffle(sharpness)

        # Size varies with sharpness (sharp points are smaller)
        radii = 0.4 - sharpness * 0.2  # Larger soft points, smaller sharp points
        radii = np.clip(radii, 0.1, 0.4).astype(np.float32)

        # Color based on sharpness (blue=soft, green=medium, red=sharp)
        normalized_sharp = sharpness
        colors = np.zeros((n_points, 3), dtype=np.float32)
        colors[:, 0] = normalized_sharp  # Red for sharp
        colors[:, 1] = 0.5 * (
            1 - np.abs(normalized_sharp - 0.5) * 2
        )  # Green for medium
        colors[:, 2] = 1 - normalized_sharp  # Blue for soft

        # Offset down and left
        positions[:, 1] -= 10
        positions[:, 0] -= 15

        scene.add_points(
            "MixedSharpnessCloud",
            positions,
            colors,
            radii=radii,
            sharpness=sharpness,
            layer=True,
            intensity=0.125,
        )
        aprint(
            f"✓ Added {len(positions):,} points (mixed cloud: blue=soft, green=medium, red=sharp)"
        )


def create_sharpness_wave_example(scene, n_points: int = 4000) -> None:
    """Create a wave pattern where sharpness varies sinusoidally.

    Generates a 3D wave surface where both height and sharpness vary together,
    creating visual depth through the sharpness variations.

    Args:
        scene: LuxarZarrCompiler scene to add points to
        n_points: Number of points in the wave grid (default: 4000)
    """
    with asection("Creating sharpness wave pattern"):
        # Create a wave grid
        grid_size = int(np.sqrt(n_points))
        x = np.linspace(-10, 10, grid_size)
        z = np.linspace(-5, 5, grid_size)
        xx, zz = np.meshgrid(x, z)

        # Wave height based on distance from origin
        distance = np.sqrt(xx**2 + zz**2)
        y = 2 * np.sin(distance * 0.5)

        positions = np.column_stack([xx.flatten(), y.flatten(), zz.flatten()]).astype(
            np.float32
        )

        # Sharpness varies with the wave (normalized [0, 1] knob)
        sharpness = 0.5 + 0.45 * np.sin(distance.flatten() * 0.5)
        sharpness = sharpness.astype(np.float32)

        # Radii also vary slightly
        radii = 0.2 + 0.1 * np.cos(distance.flatten() * 0.5)
        radii = radii.astype(np.float32)

        # Colors based on height (red=high, blue=low)
        normalized_y = (y.flatten() + 2) / 4
        colors = np.zeros((positions.shape[0], 3), dtype=np.float32)
        colors[:, 0] = normalized_y
        colors[:, 1] = 1 - np.abs(normalized_y - 0.5) * 2
        colors[:, 2] = 1 - normalized_y

        # Offset down
        positions[:, 1] -= 20

        scene.add_points(
            "SharpnessWave",
            positions,
            colors,
            radii=radii,
            sharpness=sharpness,
            layer=True,
            intensity=0.125,
        )
        aprint(f"✓ Added {len(positions):,} points (sinusoidal wave pattern)")


def generate_sharpness_showcase(
    output_path: Path,
    gradient_points: int = 5000,
    mixed_points: int = 10000,
    wave_points: int = 4000,
) -> None:
    """Generate complete sharpness showcase scene.

    This function contains ALL the generation logic - completely self-contained.
    Creates four distinct demonstrations of the sharpness feature.

    Args:
        output_path: Where to write the zarr store
        gradient_points: Number of points in gradient grid
        mixed_points: Number of points in mixed cloud
        wave_points: Number of points in wave pattern
    """
    with asection("Generating Sharpness Showcase"):
        aprint("Four demonstrations:")
        aprint("  1. Gradient: Smooth transition soft → sharp")
        aprint("  2. Comparison: Six labeled sharpness values")
        aprint("  3. Mixed Cloud: Random sharpness distribution")
        aprint("  4. Wave: Sinusoidal sharpness variation")
        aprint("")

        # Define 3D dimensions
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Add all four examples
            create_sharpness_gradient_example(scene, n_points=gradient_points)
            create_sharpness_comparison_example(scene)
            create_mixed_sharpness_example(scene, n_points=mixed_points)
            create_sharpness_wave_example(scene, n_points=wave_points)

            # --- Overlays ---
            # Title
            scene.add_text(
                "Point Sharpness Showcase",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Info
            scene.add_text(
                "Sharpness range 0\u20131 (super-Gaussian \u03b2 = 2^(6s-2))",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        total_points = gradient_points + 120 + mixed_points + wave_points
        aprint(f"\n✓ Scene complete: {total_points:,} points total")


def main() -> None:
    """Main demo entry point."""
    # Parse simple command line args (optional)
    gradient_points = 5000
    mixed_points = 10000
    wave_points = 4000

    if len(sys.argv) > 1 and sys.argv[1].startswith("--points="):
        # Scale all point counts proportionally
        scale = int(sys.argv[1].split("=")[1]) / (
            gradient_points + mixed_points + wave_points
        )
        gradient_points = int(gradient_points * scale)
        mixed_points = int(mixed_points * scale)
        wave_points = int(wave_points * scale)

    aprint("=" * 70)
    aprint("SHARPNESS SHOWCASE DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Comprehensive demonstration of the point sharpness feature")
    aprint("")
    aprint("What you'll see:")
    aprint("  • Top: Gradient from soft glowing (left) to sharp (right)")
    aprint("  • Right: Six rows showing different fixed sharpness values")
    aprint("  • Bottom Left: Spherical cloud with mixed sharpness")
    aprint("  • Bottom: Wave pattern with varying sharpness")
    aprint("")
    total = gradient_points + 120 + mixed_points + wave_points
    aprint(f"Total points: {total:,}")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "sharpness_showcase.zarr"
        generate_sharpness_showcase(
            output_path,
            gradient_points=gradient_points,
            mixed_points=mixed_points,
            wave_points=wave_points,
        )
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_sharpness_") as tmpdir:
        output_path = Path(tmpdir) / "sharpness_showcase.zarr"

        # Generate the dataset (all code in this file!)
        generate_sharpness_showcase(
            output_path,
            gradient_points=gradient_points,
            mixed_points=mixed_points,
            wave_points=wave_points,
        )

        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("The viewer will open in your browser automatically.")
        aprint("Press Ctrl+C when done to stop and cleanup.")
        aprint("")
        aprint("TIP: Zoom in to see sharpness differences clearly")
        aprint("   - Soft points glow with gradual falloff")
        aprint("   - Sharp points have crisp, disc-like edges")
        aprint("")

        launch_viewer(output_path)

    # Cleanup happens automatically
    aprint("Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
