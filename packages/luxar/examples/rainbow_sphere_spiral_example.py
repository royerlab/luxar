#!/usr/bin/env python3
"""Rainbow Sphere Spiral Example - Dense sphere with spiraling rainbow colors.

This example demonstrates:
- Creating high-density points (200,000) on a sphere surface
- Using spherical spiral (Fibonacci-like) for even distribution
- Calculating proper point spacing to avoid gaps
- Soft-edged spheres (sharpness 1.0)
- Smooth rainbow gradient flowing along the spiral

Educational value:
- Learn spherical point distribution techniques
- Understand golden ratio for even sphere coverage
- Master high-density visualization (stress testing)
- See how point spacing prevents gaps in dense clouds
- Excellent performance benchmark for rendering

Key principle:
- Golden ratio spiral provides most uniform sphere coverage
- Point spacing = sphere_radius / sqrt(n_points) prevents gaps
- High density (200k points) tests rendering performance
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_spherical_spiral(n_points: int = 200000, radius: float = 10.0) -> np.ndarray:
    """Create points distributed in a spherical spiral pattern.

    Uses a technique similar to Fibonacci spiral on a sphere for even distribution.

    Args:
        n_points: Number of points to generate
        radius: Radius of the sphere

    Returns:
        Array of 3D positions
    """
    indices = np.arange(0, n_points, dtype=float) + 0.5

    # Golden angle in radians
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    # Generate spherical coordinates
    theta = indices * golden_angle  # Azimuthal angle
    y = 1 - (indices / float(n_points - 1)) * 2  # y goes from 1 to -1
    # Clamp y to avoid numerical issues at poles
    y = np.clip(y, -1.0, 1.0)
    radius_at_y = np.sqrt(1 - y * y)  # Radius at y

    # Convert to Cartesian coordinates
    x = np.cos(theta) * radius_at_y * radius
    z = np.sin(theta) * radius_at_y * radius
    y = y * radius

    return np.column_stack([x, y, z]).astype(np.float32)


def create_rainbow_colors(n_points: int) -> np.ndarray:
    """Generate smooth rainbow colors along the spiral.

    Args:
        n_points: Number of color values to generate

    Returns:
        Array of RGB colors
    """
    # Parameter t goes from 0 to 1 along the spiral
    t = np.linspace(0, 1, n_points)

    # Create smooth rainbow using sine waves
    # Offset by 2π/3 for each channel to create RGB sequence
    r = np.sin(2 * np.pi * t) * 0.5 + 0.5
    g = np.sin(2 * np.pi * t + 2 * np.pi / 3) * 0.5 + 0.5
    b = np.sin(2 * np.pi * t + 4 * np.pi / 3) * 0.5 + 0.5

    colors = np.column_stack([r, g, b]).astype(np.float32)

    return colors


def calculate_point_radius(n_points: int, sphere_radius: float) -> float:
    """Calculate appropriate point radius based on sphere size and point count.

    For evenly distributed points on a sphere, calculates radius such that
    there's approximately one radius of space between points.

    Args:
        n_points: Number of points on the sphere
        sphere_radius: Radius of the sphere

    Returns:
        Appropriate point radius
    """
    # Average area per point on sphere surface
    sphere_area = 4 * np.pi * sphere_radius**2
    area_per_point = sphere_area / n_points

    # Average distance between neighbors (approximation)
    avg_distance = np.sqrt(area_per_point)

    # Point radius should be about half the distance for spacing
    point_radius = avg_distance / 2

    return point_radius * 0.5


def main():
    """Create a dense rainbow sphere with spiral point distribution."""
    output_path = get_examples_output_dir() / "rainbow_sphere_spiral_example.zarr"

    aprint(f"Creating dense rainbow sphere spiral at {output_path}")
    aprint("This example creates a high-density sphere with:")
    aprint("- 200,000 points in a spherical spiral pattern")
    aprint("- Points spaced approximately one radius apart")
    aprint("- Soft-edged spheres (sharpness 1.0)")
    aprint("- Smooth rainbow gradient along the spiral")

    # Parameters
    n_points = 200000  # High density
    sphere_radius = 10.0

    # Create scene

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # Generate the spherical spiral
        aprint(f"\nGenerating spherical spiral with {n_points:,} points...")
        positions = create_spherical_spiral(n_points, sphere_radius)

        # Generate rainbow colors
        aprint("Creating rainbow color gradient...")
        colors = create_rainbow_colors(n_points)

        # Calculate appropriate radius for spacing
        point_radius = calculate_point_radius(n_points, sphere_radius)
        aprint(f"Calculated point radius: {point_radius:.4f} units")

        # Create uniform radii array
        radii = np.full(n_points, point_radius, dtype=np.float32)

        # Soft-edged spheres (low sharpness)
        sharpness = np.full(n_points, 1.0, dtype=np.float32)

        # Add points to scene
        scene.add_points(
            "DenseRainbowSphere",
            positions,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
        )

        # Print summary
        aprint(f"\n✓ Created dense rainbow sphere with {n_points:,} points")
        aprint(f"  Sphere radius: {sphere_radius} units")
        aprint(f"  Point radius: {point_radius:.4f} units")
        aprint(f"  Average spacing: ~{2 * point_radius:.4f} units")
        aprint("  Sharpness: 1.0 (soft-edged spheres)")
        aprint("  Colors: Full rainbow spectrum")

        # Explainer card describing the dense-sphere stress test.
        add_explainer(
            scene,
            title="Rainbow Sphere Spiral",
            body=(
                "200,000 points placed on a sphere via a golden-angle "
                "(Fibonacci) spiral for near-uniform coverage, colored by a "
                "smooth rainbow gradient running along the spiral."
            ),
            observe=[
                "Coverage is even, with no clustering at the poles.",
                "The rainbow flows continuously along the spiral.",
                "Per-point radii keep neighbors roughly one radius apart.",
            ],
            observe_label="Notice",
        )

        aprint("\n" + "=" * 60)
        aprint("VIEWING INSTRUCTIONS")
        aprint("=" * 60)
        aprint("1. Start the server:")
        aprint(f"   luxar serve {output_path}")
        aprint("\n2. Performance tips:")
        aprint("   - This is a high-density dataset (200k points)")
        aprint("   - Initial loading may take a moment")
        aprint("   - Zoom in to see individual points")
        aprint("   - Points are spaced ~1 radius apart")
        aprint("\n3. Visual features:")
        aprint("   - Spherical spiral ensures perfect distribution")
        aprint("   - Rainbow flows continuously along the spiral")
        aprint("   - Sharpness 1.0 yields soft-edged spheres")
        aprint("   - No overlapping due to calculated spacing")
        aprint("=" * 60)


if __name__ == "__main__":
    main()
