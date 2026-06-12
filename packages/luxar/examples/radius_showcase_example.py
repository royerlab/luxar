#!/usr/bin/env python3
"""Radius Showcase Example - Comprehensive demonstration of point radius features.

This example demonstrates:
- Size gradients: points that grow progressively along a path
- Distance-based sizing: radii calculated from distance to center
- Random sizing: natural variation in point sizes
- Layered spheres: concentric structures with different radii
- How radius affects visual hierarchy and data emphasis

Educational value:
- Learn the full capabilities of the radius parameter
- Understand size-based visual organization techniques
- See both structured and natural size patterns
- Master radius as a data dimension (not just aesthetic)
- Learn when each sizing approach is appropriate

When to use these techniques:
- Size gradients: Show progression, importance, or ordering
- Distance-based: Radial symmetry, centrality, or proximity
- Random sizing: Natural, organic appearance
- Layered: Hierarchical structures, nested organization
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_size_gradient_example(scene, n_points: int = 10000) -> None:
    """Create a spiral with points that gradually increase in size.

    Args:
        scene: The scene to add points to
        n_points: Number of points in the spiral
    """
    aprint("Creating size gradient demo...")

    # Generate a 3D spiral
    t = np.linspace(0, 8 * np.pi, n_points)
    x = np.cos(t) * t / (2 * np.pi)
    y = np.sin(t) * t / (2 * np.pi)
    z = t / (2 * np.pi)

    positions = np.column_stack([x, y, z]).astype(np.float32)

    # Radii grow from 0.02 to 0.5 along the spiral
    radii = np.linspace(0.02, 0.5, n_points).astype(np.float32)

    # Colors transition from blue to red
    colors = np.zeros((n_points, 3), dtype=np.float32)
    colors[:, 0] = np.linspace(0, 1.0, n_points)  # Red channel
    colors[:, 2] = np.linspace(1.0, 0, n_points)  # Blue channel

    scene.add_points("SizeGradientSpiral", positions, colors, radii=radii)


def create_distance_based_example(scene, n_points: int = 5000) -> None:
    """Create a sphere where point size depends on distance from center.

    Args:
        scene: The scene to add points to
        n_points: Number of points on the sphere
    """
    aprint("Creating distance-based sizing demo...")

    # Generate points on a sphere using spherical coordinates
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    phi = np.arccos(rng.uniform(-1, 1, n_points))

    # Convert to Cartesian coordinates
    r = 5.0  # Sphere radius
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    positions = np.column_stack([x, y, z]).astype(np.float32)

    # Add some noise to create a fuzzy sphere
    positions += rng.normal(0, 0.2, positions.shape).astype(np.float32)

    # Radii based on distance from origin (inverted - closer = larger)
    distances = np.linalg.norm(positions, axis=1)
    radii = (0.4 / (distances / r)).astype(np.float32)
    radii = np.clip(radii, 0.05, 0.8)  # Clamp to reasonable range

    # Colors based on position (creates a rainbow effect)
    colors = np.zeros((n_points, 3), dtype=np.float32)
    colors[:, 0] = (x + r) / (2 * r)
    colors[:, 1] = (y + r) / (2 * r)
    colors[:, 2] = (z + r) / (2 * r)

    # Offset to avoid overlap with other demos
    positions[:, 0] += 15

    scene.add_points("DistanceBasedSphere", positions, colors, radii=radii)


def create_random_sizing_example(scene, n_points: int = 8000) -> None:
    """Create a cube with randomly sized points.

    Args:
        scene: The scene to add points to
        n_points: Number of points in the cube
    """
    aprint("Creating random sizing demo...")

    rng = np.random.default_rng(123)

    # Generate points in a cube
    positions = rng.uniform(-5, 5, (n_points, 3)).astype(np.float32)

    # Random radii with a bias towards smaller sizes
    radii = rng.exponential(0.15, n_points).astype(np.float32)
    radii = np.clip(radii, 0.01, 0.6)

    # Colors based on radius (heat map: small=blue, large=red)
    normalized_radii = (radii - radii.min()) / (radii.max() - radii.min())
    colors = np.zeros((n_points, 3), dtype=np.float32)
    colors[:, 0] = normalized_radii  # Red
    colors[:, 2] = 1 - normalized_radii  # Blue

    # Offset to avoid overlap
    positions[:, 0] -= 15

    scene.add_points("RandomSizedCube", positions, colors, radii=radii)


def create_layered_spheres_example(
    scene, n_layers: int = 5, points_per_layer: int = 1000
) -> None:
    """Create concentric spheres with different point sizes per layer.

    Args:
        scene: The scene to add points to
        n_layers: Number of concentric sphere layers
        points_per_layer: Number of points per sphere layer
    """
    aprint("Creating layered spheres demo...")

    rng = np.random.default_rng(456)
    all_positions = []
    all_colors = []
    all_radii = []

    for i in range(n_layers):
        # Each layer has a different radius
        layer_radius = (i + 1) * 1.5

        # Generate points on sphere
        theta = rng.uniform(0, 2 * np.pi, points_per_layer)
        phi = np.arccos(rng.uniform(-1, 1, points_per_layer))

        x = layer_radius * np.sin(phi) * np.cos(theta)
        y = layer_radius * np.sin(phi) * np.sin(theta)
        z = layer_radius * np.cos(phi)

        positions = np.column_stack([x, y, z]).astype(np.float32)

        # Offset vertically
        positions[:, 2] += 15

        # Point size decreases with layer number (outer layers have smaller points)
        radii = np.full(points_per_layer, 0.3 - i * 0.05, dtype=np.float32)

        # Each layer has a different color
        hue = i / n_layers
        # Simple HSV to RGB conversion (S=1, V=1)
        if hue < 1 / 6:
            r, g, b = 1, hue * 6, 0
        elif hue < 2 / 6:
            r, g, b = 2 - hue * 6, 1, 0
        elif hue < 3 / 6:
            r, g, b = 0, 1, hue * 6 - 2
        elif hue < 4 / 6:
            r, g, b = 0, 4 - hue * 6, 1
        elif hue < 5 / 6:
            r, g, b = hue * 6 - 4, 0, 1
        else:
            r, g, b = 1, 0, 6 - hue * 6

        colors = np.tile([r, g, b], (points_per_layer, 1)).astype(np.float32)

        all_positions.append(positions)
        all_colors.append(colors)
        all_radii.append(radii)

    # Combine all layers
    positions = np.vstack(all_positions)
    colors = np.vstack(all_colors)
    radii = np.hstack(all_radii)

    scene.add_points("LayeredSpheres", positions, colors, radii=radii)


def main():
    """Run the radius showcase example."""
    output_path = get_examples_output_dir() / "radius_showcase_example.luxar.zarr"

    aprint(f"Creating radius showcase example at {output_path}")
    aprint(
        "This example showcases the per-point radius feature with various visualizations:"
    )
    aprint("- Size gradient spiral: Points that grow along a spiral")
    aprint("- Distance-based sphere: Point size based on distance from center")
    aprint("- Random sized cube: Points with random radii")
    aprint("- Layered spheres: Concentric spheres with different point sizes")

    # Create scene

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # Add all examples
        create_size_gradient_example(scene)
        create_distance_based_example(scene)
        create_random_sizing_example(scene)
        create_layered_spheres_example(scene)

        # Explainer overlay describing what to look for in the viewer.
        add_explainer(
            scene,
            title="Per-Point Radii",
            body=(
                "The per-point <code>radii</code> array drives on-screen size, "
                "turning scale into a <strong>data dimension</strong>. Four "
                "showcases use a size gradient, distance-based sizing, random "
                "variation, and layered concentric spheres."
            ),
            observe=[
                "Spiral points grow smoothly from tiny to large along its length.",
                "Sphere points enlarge toward the center (inverse distance).",
                "Cube shows organic random sizes colored small-blue to large-red.",
                "Nested spheres use a distinct point size per layer.",
            ],
            observe_label="Look for",
        )

        # Finalize

        aprint(f"\n✓ Example scene created successfully at {output_path}")

        aprint("\n" + "=" * 60)
        aprint("VIEWING INSTRUCTIONS:")
        aprint(f"1. Run: luxar serve {output_path}")
        aprint("2. Rotate the view to see all four showcases")
        aprint("")
        aprint("Look for:")
        aprint("- Spiral with gradually increasing point sizes")
        aprint("- Sphere with larger points near the center")
        aprint("- Cube with randomly sized colorful points")
        aprint("- Nested spheres with different point sizes per layer")
        aprint("=" * 60)


if __name__ == "__main__":
    main()
