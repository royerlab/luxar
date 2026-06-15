#!/usr/bin/env python3
"""Multiple Objects Example - Demonstrates multiple points objects in one scene.

This educational example demonstrates:
- Creating multiple distinct points objects
- Using different rendering properties per object
- Spatial arrangement and organization
- Color coding and visual differentiation
- Mixed object sizes and densities
- Educational scene composition

Educational value:
- Learn to compose scenes with multiple distinct point clouds
- Understand per-object rendering property configuration
- See spatial arrangement strategies for multi-object scenes
- Good reference for real-world multi-dataset visualization
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_spiral_galaxy(n_points: int, radius: float, height: float) -> np.ndarray:
    """Create points in a spiral galaxy pattern.

    Args:
        n_points: Number of points
        radius: Maximum radius of the spiral
        height: Thickness of the galaxy disk

    Returns:
        Array of 3D positions forming a spiral galaxy
    """
    # Generate spiral parameters
    t = np.linspace(0, 4 * np.pi, n_points)
    r = np.linspace(0.1, radius, n_points)

    # Create spiral arms
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = np.random.normal(0, height, n_points)

    return np.column_stack([x, y, z]).astype(np.float32)


def create_globular_cluster(n_points: int, center: tuple, radius: float) -> np.ndarray:
    """Create points in a dense globular cluster.

    Args:
        n_points: Number of points
        center: Center position (x, y, z)
        radius: Radius of the cluster

    Returns:
        Array of 3D positions forming a globular cluster
    """
    # Generate points with higher density toward center
    # Use exponential distribution for radius
    radii = np.random.exponential(radius / 3, n_points)
    radii = np.clip(radii, 0, radius)

    # Random directions
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)

    # Convert to Cartesian
    x = radii * np.sin(phi) * np.cos(theta) + center[0]
    y = radii * np.sin(phi) * np.sin(theta) + center[1]
    z = radii * np.cos(phi) + center[2]

    return np.column_stack([x, y, z]).astype(np.float32)


def create_nebula_cloud(n_points: int, center: tuple, size: tuple) -> np.ndarray:
    """Create points in a nebula-like cloud.

    Args:
        n_points: Number of points
        center: Center position (x, y, z)
        size: Size in each dimension (sx, sy, sz)

    Returns:
        Array of 3D positions forming a nebula cloud
    """
    # Generate cloud with varying density
    positions = []

    for _ in range(n_points):
        # Use multiple random samples for more natural clustering
        x = np.random.normal(center[0], size[0] / 3)
        y = np.random.normal(center[1], size[1] / 3)
        z = np.random.normal(center[2], size[2] / 3)
        positions.append([x, y, z])

    return np.array(positions, dtype=np.float32)


def create_ring_system(
    n_rings: int, center: tuple, inner_radius: float, outer_radius: float
) -> np.ndarray:
    """Create points forming a ring system.

    Args:
        n_rings: Number of concentric rings
        center: Center position
        inner_radius: Inner radius
        outer_radius: Outer radius

    Returns:
        Array of 3D positions forming ring system
    """
    positions = []

    # Create concentric rings
    radii = np.linspace(inner_radius, outer_radius, n_rings)

    for radius in radii:
        # Points per ring depends on circumference
        points_in_ring = int(radius * 50)  # Density scaling
        angles = np.linspace(0, 2 * np.pi, points_in_ring, endpoint=False)

        for angle in angles:
            # Add some randomness to ring positions
            r_noise = radius + np.random.normal(0, radius * 0.05)
            x = r_noise * np.cos(angle) + center[0]
            y = r_noise * np.sin(angle) + center[1]
            z = center[2] + np.random.normal(0, 0.1)  # Thin ring
            positions.append([x, y, z])

    return np.array(positions, dtype=np.float32)


def main():
    """Create a scene with multiple different points objects."""
    output_path = get_examples_output_dir() / "multiple_objects_example.luxar.zarr"

    with asection("Multiple Objects Example Setup"):
        aprint(f"Creating multiple objects demonstration at {output_path}")
        aprint("This example shows:")
        aprint("- Multiple distinct points objects")
        aprint("- Different shapes, sizes, and densities")
        aprint("- Varied rendering properties per object")
        aprint("- Spatial organization and composition")
        aprint("- Color coding for visual differentiation")

    # Create scene

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # Running tally so the summary reports the true point count (the ring
        # systems use radius-dependent density, so it cannot be known up front).
        total_points = 0

        with asection("Central Spiral Galaxy"):
            # 1. Central spiral galaxy
            aprint("Creating central spiral galaxy...")
            galaxy_positions = create_spiral_galaxy(15000, 8.0, 0.5)
            scene.add_points(
                "SpiralGalaxy",
                galaxy_positions,
                colors=[0.78, 0.78, 1.0],  # Light blue
                radii=0.03,
                sharpness=0.6,
                opacity=0.8,
                gamma=1.1,
                blending_mode="additive",
            )
            total_points += len(galaxy_positions)
            aprint(f"  Added {len(galaxy_positions):,} points in spiral pattern")

        with asection("Globular Clusters"):
            # 2. Globular clusters around the galaxy
            aprint("Creating globular clusters...")
            cluster_positions = [(-12, 8, 3), (10, -6, -2), (-8, -10, 4), (15, 5, -3)]

            cluster_colors = [
                [1.0, 0.78, 0.39],  # Golden
                [1.0, 0.59, 0.59],  # Pink-red
                [0.59, 1.0, 0.59],  # Light green
                [1.0, 1.0, 0.59],  # Light yellow
            ]

            for i, (pos, color) in enumerate(zip(cluster_positions, cluster_colors)):
                cluster_points = create_globular_cluster(2000, pos, 1.5)
                scene.add_points(
                    f"GlobularCluster{i + 1}",
                    cluster_points,
                    colors=color,
                    radii=0.04,
                    sharpness=0.5,
                    opacity=0.9,
                    gamma=1.2,
                    blending_mode="additive",
                )
                total_points += len(cluster_points)
            aprint(
                f"  Added {len(cluster_positions)} clusters with {len(cluster_points):,} points each"
            )

        with asection("Colorful Nebulae"):
            # 3. Nebula clouds
            aprint("Creating colorful nebulae...")
            nebula_data = [
                {
                    "center": (-20, 0, 0),
                    "size": (3, 4, 2),
                    "color": [1.0, 0.39, 0.59],
                    "name": "RedNebula",
                },
                {
                    "center": (0, 15, 0),
                    "size": (2, 3, 3),
                    "color": [0.39, 1.0, 0.59],
                    "name": "GreenNebula",
                },
                {
                    "center": (0, -15, 0),
                    "size": (4, 2, 2),
                    "color": [0.59, 0.39, 1.0],
                    "name": "PurpleNebula",
                },
            ]

            for nebula in nebula_data:
                nebula_points = create_nebula_cloud(
                    3000, nebula["center"], nebula["size"]
                )
                scene.add_points(
                    nebula["name"],
                    nebula_points,
                    colors=nebula["color"],
                    radii=0.06,
                    sharpness=0.3,
                    opacity=0.6,
                    gamma=1.0,
                    blending_mode="normal",
                )
                total_points += len(nebula_points)
            aprint(f"  Added {len(nebula_data)} nebulae with 3,000 points each")

        with asection("Planetary Ring Systems"):
            # 4. Planetary ring systems
            aprint("Creating planetary ring systems...")
            ring_systems = [
                {
                    "center": (0, 0, 10),
                    "inner": 2,
                    "outer": 4,
                    "color": [0.78, 0.59, 0.39],
                },
                {
                    "center": (0, 0, -8),
                    "inner": 1.5,
                    "outer": 3,
                    "color": [0.59, 0.78, 1.0],
                },
            ]

            for i, ring_data in enumerate(ring_systems):
                ring_points = create_ring_system(
                    8, ring_data["center"], ring_data["inner"], ring_data["outer"]
                )
                scene.add_points(
                    f"RingSystem{i + 1}",
                    ring_points,
                    colors=ring_data["color"],
                    radii=0.025,
                    sharpness=0.65,
                    opacity=0.7,
                    gamma=0.9,
                    blending_mode="normal",
                )
                total_points += len(ring_points)
            aprint(
                f"  Added {len(ring_systems)} ring systems with variable point densities"
            )

        with asection("Background Star Field"):
            # 5. Scattered star field (background)
            aprint("Creating background star field...")
            n_stars = 5000
            star_positions = np.random.uniform(-25, 25, (n_stars, 3)).astype(np.float32)

            # Variable star colors and sizes
            star_colors = []
            star_radii = []
            for _ in range(n_stars):
                # Random star colors (white to yellow to red)
                temp = np.random.random()
                if temp < 0.3:
                    color = [1.0, 1.0, 1.0]  # White
                elif temp < 0.6:
                    color = [1.0, 1.0, 0.78]  # Yellow-white
                elif temp < 0.8:
                    color = [1.0, 0.86, 0.59]  # Yellow
                else:
                    color = [1.0, 0.71, 0.47]  # Orange-red

                star_colors.append(color)
                star_radii.append(np.random.uniform(0.01, 0.02))

            scene.add_points(
                "BackgroundStars",
                star_positions,
                colors=np.array(star_colors, dtype=np.float32),
                radii=np.array(star_radii, dtype=np.float32),
                sharpness=0.8,
                opacity=0.4,
                gamma=1.0,
                blending_mode="normal",
            )
            total_points += n_stars
            aprint(f"  Added {n_stars:,} background stars with varied colors and sizes")

        with asection("Particle Stream"):
            # 6. Dense particle stream
            aprint("Creating particle stream...")
            stream_positions = []
            t = np.linspace(0, 10, 1000)
            for i, time in enumerate(t):
                x = 20 + time * 0.5 + 0.2 * np.sin(time * 2)
                y = np.sin(time) * 2
                z = np.cos(time) * 2 + 0.1 * time
                stream_positions.append([x, y, z])

            scene.add_points(
                "ParticleStream",
                np.array(stream_positions, dtype=np.float32),
                colors=[0.39, 1.0, 1.0],  # Cyan
                radii=0.05,
                sharpness=0.85,
                opacity=0.8,
                gamma=1.3,
                blending_mode="additive",
            )
            total_points += len(stream_positions)
            aprint(f"  Added {len(stream_positions):,} points in helical trajectory")

        add_explainer(
            scene,
            title="Multiple objects in one scene",
            body="Six kinds of point object composed into a single "
            "scene, each with its own color, <code>radii</code>, <code>sharpness</code>, "
            "<code>opacity</code>, and <strong>blending mode</strong> to build visual "
            "hierarchy and depth.",
            observe=[
                "A light-blue spiral galaxy sits at the center.",
                "Four globular clusters and two ring systems orbit it.",
                "Red, green, and purple nebulae glow with soft edges.",
                "A faint star field fills the background; a particle stream trails off.",
            ],
            observe_label="Look for",
        )

        with asection("Educational Summary"):
            # Print educational summary
            aprint("=" * 70)
            aprint("MULTIPLE OBJECTS DEMONSTRATION")
            aprint("=" * 70)
            aprint("Objects in Scene:")
            aprint("  1. Central Spiral Galaxy (15k points):")
            aprint("     - Light blue, additive blending, small radius")
            aprint("     - Mathematical spiral pattern")

            aprint("  2. Four Globular Clusters (2k points each):")
            aprint("     - Golden, pink, green, yellow colors")
            aprint("     - Dense spherical distributions")
            aprint("     - Positioned around the galaxy")

            aprint("  3. Three Nebula Clouds (3k points each):")
            aprint("     - Red, green, purple diffuse clouds")
            aprint("     - Normal blending, larger radius")
            aprint("     - Gaussian distribution shapes")

            aprint("  4. Two Ring Systems:")
            aprint("     - Brown and blue concentric rings")
            aprint("     - Above and below galaxy plane")
            aprint("     - Variable point density by radius")

            aprint("  5. Background Star Field (5k points):")
            aprint("     - Random positions throughout scene")
            aprint("     - Variable colors (white to orange)")
            aprint("     - Small radius, low opacity")

            aprint("  6. Particle Stream (1k points):")
            aprint("     - Cyan helical trajectory")
            aprint("     - Additive blending for glow")
            aprint("     - Mathematical parametric curve")

            aprint("\nTotal Objects: 6 distinct kinds of point object")
            aprint(f"Total Points: {total_points:,} points")
            aprint("Rendering Modes: Normal and additive blending")
            aprint("Point Sizes: 0.01 to 0.06 radius range")
            aprint("Colors: Full spectrum with varied opacity")

            aprint("\nScene Composition Techniques:")
            aprint("- Layered depth with foreground/background")
            aprint("- Color harmony with complementary schemes")
            aprint("- Size variation for visual hierarchy")
            aprint("- Opacity variation for depth perception")
            aprint("- Mixed blending modes for effects")

            aprint(f"\nTo view: luxar serve {output_path}")
            aprint("Explore the rich variety of objects and their properties!")
            aprint("=" * 70)


if __name__ == "__main__":
    main()
