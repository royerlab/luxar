#!/usr/bin/env python3
"""Hierarchy Example - Demonstrates parent-child relationships in Luxar scenes.

This educational example demonstrates:
- Creating hierarchical node structures
- Property inheritance from parent to child
- Transform inheritance and composition
- Rendering attribute inheritance (opacity, gamma, blending)
- Nested group hierarchies
- Educational visualization of hierarchical structures

Educational value:
- Understand how parent-child relationships affect rendering
- Learn property inheritance (opacity, gamma, blending propagate down)
- See how transforms compose through the hierarchy
- Master group organization for complex multi-object scenes
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint, asection

from luxar import Dimensions, LuxarZarrCompiler, transforms
from luxar.utils.paths import get_examples_output_dir


def create_constellation_points(n_points: int, radius: float) -> np.ndarray:
    """Create points arranged in a spherical constellation.

    Args:
        n_points: Number of points to generate
        radius: Radius of the sphere

    Returns:
        Array of 3D positions on sphere surface
    """
    # Use spherical coordinates for even distribution
    indices = np.arange(0, n_points, dtype=float) + 0.5
    theta = np.arccos(1 - 2 * indices / n_points)  # Polar angle
    phi = np.pi * (1 + 5**0.5) * indices  # Azimuthal angle (golden ratio)

    # Convert to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.column_stack([x, y, z]).astype(np.float32)


def create_ring_points(n_points: int, radius: float, height: float = 0.0) -> np.ndarray:
    """Create points arranged in a ring.

    Args:
        n_points: Number of points in the ring
        radius: Radius of the ring
        height: Z-height of the ring

    Returns:
        Array of 3D positions forming a ring
    """
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.full(n_points, height)

    return np.column_stack([x, y, z]).astype(np.float32)


def main():
    """Create a scene demonstrating hierarchical relationships."""
    output_path = get_examples_output_dir() / "hierarchy_example.zarr"

    with asection("Hierarchy Example Setup"):
        aprint(f"Creating hierarchy demonstration at {output_path}")
        aprint("This example shows:")
        aprint("- Parent-child node relationships")
        aprint("- Property inheritance (opacity, gamma, blending)")
        aprint("- Transform inheritance and composition")
        aprint("- Nested group hierarchies")
        aprint("- Visual organization of complex scenes")

    # Create scene
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        with asection("Root Level Objects"):
            # 1. Root level objects (no parent)
            aprint("Creating root-level reference objects...")

            # Central reference constellation
            central_positions = create_constellation_points(100, 1.0)
            scene.add_points(
                "CentralConstellation",
                central_positions,
                colors=[1.0, 1.0, 1.0],  # White
                radii=0.05,
                sharpness=0.5,
                opacity=1.0,
                gamma=1.0,
            )

        with asection("Solar System Hierarchy (First Level)"):
            # 2. First level hierarchy - Solar System analogy
            aprint("Creating solar system hierarchy...")

            # Sun (parent group)
            sun_transform = transforms.translate(0, 0, 0)  # At origin
            sun_group = scene.add_group(
                "SolarSystem",
                transform=sun_transform,
                opacity=0.9,  # Inherited by all children
                gamma=1.2,  # Slightly brighter
                blending_mode="additive",  # Glowing effect
            )

            # Sun object (child of sun_group)
            sun_positions = create_constellation_points(50, 0.3)
            scene.add_points(
                "Sun",
                sun_positions,
                colors=[1.0, 0.78, 0.39],  # Yellow-orange
                radii=0.08,
                parent=sun_group,
                # Inherits: opacity=0.9, gamma=1.2, blending_mode="additive"
            )

        with asection("Planetary Systems (Second Level)"):
            # 3. Second level hierarchy - Planets
            aprint("Creating planetary systems...")

            # Earth system (child of solar system)
            earth_transform = transforms.translate(4, 0, 0)  # Orbit position
            earth_group = sun_group.add_group(
                "EarthSystem",
                transform=earth_transform,
                opacity=0.8,  # Override parent's opacity
                # Inherits: gamma=1.2, blending_mode="additive" from sun_group
            )

            # Earth (child of earth system)
            earth_positions = create_constellation_points(30, 0.2)
            scene.add_points(
                "Earth",
                earth_positions,
                colors=[0.39, 0.59, 1.0],  # Blue
                radii=0.06,
                parent=earth_group,
                # Inherits: opacity=0.8, gamma=1.2, blending_mode="additive"
            )

            # Moon (child of earth system, sibling of Earth)
            moon_transform = transforms.translate(0.5, 0, 0)  # Relative to Earth system
            moon_positions = create_constellation_points(15, 0.08)
            scene.add_points(
                "Moon",
                moon_positions,
                colors=[0.78, 0.78, 0.78],  # Gray
                radii=0.04,
                transform=moon_transform,
                parent=earth_group,
                # Inherits: opacity=0.8, gamma=1.2, blending_mode="additive"
            )

            # Mars system (another child of solar system)
            mars_transform = transforms.translate(6, 0, 0)
            mars_group = sun_group.add_group(
                "MarsSystem",
                transform=mars_transform,
                gamma=1.0,  # Override parent's gamma
                # Inherits: opacity=0.9, blending_mode="additive" from sun_group
            )

            # Mars (child of mars system)
            mars_positions = create_constellation_points(20, 0.15)
            scene.add_points(
                "Mars",
                mars_positions,
                colors=[1.0, 0.39, 0.39],  # Red
                radii=0.05,
                parent=mars_group,
                # Inherits: opacity=0.9, gamma=1.0, blending_mode="additive"
            )

        with asection("Space Station Complex (Separate Hierarchy)"):
            # 4. Separate hierarchy - Space Station Complex
            aprint("Creating space station hierarchy...")

            # Space station group (separate from solar system)
            station_transform = transforms.compose(
                transforms.translate(0, 0, 5), transforms.rotate_z(np.pi / 6)
            )
            station_group = scene.add_group(
                "SpaceStationComplex",
                transform=station_transform,
                opacity=0.7,
                gamma=1.1,
                blending_mode="normal",  # Different from solar system
            )

            # Central hub
            hub_positions = create_ring_points(20, 0.3)
            scene.add_points(
                "CentralHub",
                hub_positions,
                colors=[0.59, 1.0, 0.59],  # Light green
                radii=0.07,
                parent=station_group,
            )

            # Docking rings (children of station)
            for i in range(3):
                angle = i * 2 * np.pi / 3
                ring_transform = transforms.compose(
                    transforms.rotate_z(angle), transforms.translate(0.8, 0, 0)
                )

                ring_positions = create_ring_points(12, 0.15)
                scene.add_points(
                    f"DockingRing{i + 1}",
                    ring_positions,
                    colors=[1.0, 0.59, 1.0],  # Light magenta
                    radii=0.05,
                    transform=ring_transform,
                    parent=station_group,
                    # Inherits: opacity=0.7, gamma=1.1, blending_mode="normal"
                )

        with asection("Deep Nested Hierarchy (4 Levels)"):
            # 5. Deep hierarchy example
            aprint("Creating deep nested hierarchy...")

            # Level 1: Galaxy
            galaxy_transform = transforms.translate(0, 8, 0)
            galaxy_group = scene.add_group(
                "Galaxy", transform=galaxy_transform, opacity=0.6
            )

            # Level 2: Star cluster
            cluster_group = galaxy_group.add_group("StarCluster", gamma=1.3)

            # Level 3: Binary star system
            binary_transform = transforms.translate(1, 0, 0)
            binary_group = cluster_group.add_group(
                "BinarySystem", transform=binary_transform, blending_mode="additive"
            )

            # Level 4: Individual stars
            star1_positions = create_constellation_points(15, 0.1)
            scene.add_points(
                "Star1",
                star1_positions,
                colors=[1.0, 1.0, 0.78],  # Bright white-yellow
                radii=0.04,
                transform=transforms.translate(-0.2, 0, 0),
                parent=binary_group,
            )

            star2_positions = create_constellation_points(12, 0.08)
            scene.add_points(
                "Star2",
                star2_positions,
                colors=[1.0, 0.59, 0.59],  # Light red
                radii=0.035,
                transform=transforms.translate(0.2, 0, 0),
                parent=binary_group,
            )

        add_explainer(
            scene,
            title="Hierarchy & Inheritance",
            body=(
                "Groups nest into a tree where each child's transform is "
                "relative to its parent, so <strong>positions accumulate</strong> "
                "down the hierarchy. Rendering attributes (<code>opacity</code>, "
                "<code>gamma</code>, <code>blending_mode</code>) also inherit "
                "unless a child overrides them."
            ),
            observe=[
                "Solar System glows (<code>additive</code>); Sun, Earth+Moon, "
                "and Mars sit at increasing <code>+X</code> orbits.",
                "Space Station (rotated, <code>normal</code> blend) sits above "
                "at <code>+Z</code> with 3 docking rings.",
                "Galaxy stack is 4 levels deep, far up at <code>+Y</code>.",
                "Overridden opacity/gamma make some subgroups dimmer/brighter.",
            ],
            observe_label="Notice",
        )

        with asection("Educational Summary"):
            # Print educational summary
            aprint("=" * 60)
            aprint("HIERARCHY DEMONSTRATION")
            aprint("=" * 60)
            aprint("Scene Structure:")
            aprint("  Center: Reference constellation (white, no parent)")
            aprint("  Solar System Group (glowing, additive blending):")
            aprint("    └─ Sun (yellow-orange)")
            aprint("    └─ Earth System:")
            aprint("        ├─ Earth (blue)")
            aprint("        └─ Moon (gray)")
            aprint("    └─ Mars System:")
            aprint("        └─ Mars (red)")
            aprint("  Space Station Complex (normal blending):")
            aprint("    ├─ Central Hub (green)")
            aprint("    └─ 3x Docking Rings (magenta)")
            aprint("  Galaxy → Star Cluster → Binary System:")
            aprint("    ├─ Star 1 (bright yellow)")
            aprint("    └─ Star 2 (light red)")

            aprint("\nProperty Inheritance:")
            aprint("- Solar System: opacity=0.9, gamma=1.2, additive blending")
            aprint("- Earth System: overrides opacity=0.8, inherits rest")
            aprint("- Mars System: overrides gamma=1.0, inherits rest")
            aprint("- Space Station: opacity=0.7, gamma=1.1, normal blending")
            aprint("- Galaxy hierarchy: 4 levels deep with accumulated properties")

            aprint("\nTransform Inheritance:")
            aprint("- Child transforms are relative to parent coordinate systems")
            aprint("- Final position = parent_transform * child_transform")
            aprint("- Deep hierarchies accumulate all ancestor transforms")

            aprint("\nScene Organization Benefits:")
            aprint("- Logical grouping of related objects")
            aprint("- Consistent styling through inheritance")
            aprint("- Easy manipulation of object groups")
            aprint("- Scalable scene management")

            aprint(f"\nTo view: luxar serve {output_path}")
            aprint("Notice how properties flow down the hierarchy!")
            aprint("=" * 60)


if __name__ == "__main__":
    main()
