#!/usr/bin/env python3
"""Rainbow Sphere Spiral Example - Creates a dense sphere with spiraling rainbow colors.

This example demonstrates:
- Creating a very dense point cloud (200,000 points) on a sphere
- Using spherical spiral for even distribution
- Calculating appropriate point spacing
- High sharpness for crisp point rendering
- Smooth rainbow gradient along the spiral
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import Scene


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

    # Convert to uint8
    colors = np.column_stack([r * 255, g * 255, b * 255]).astype(np.uint8)

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

    return point_radius * 0.66


def main():
    """Create a dense rainbow sphere with spiral point distribution."""
    output_path = Path(__file__).parent / "rainbow_sphere_spiral_example.zarr"

    aprint(f"Creating dense rainbow sphere spiral at {output_path}")
    aprint("This example creates a high-density sphere with:")
    aprint("- 200,000 points in a spherical spiral pattern")
    aprint("- Points spaced approximately one radius apart")
    aprint("- High sharpness for crisp rendering")
    aprint("- Smooth rainbow gradient along the spiral")

    # Parameters
    n_points = 200000  # High density
    sphere_radius = 10.0

    # Create scene
    scene = Scene(output_path)

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

    # Use high sharpness for crisp points
    sharpness = np.full(n_points, 1.0, dtype=np.float32)

    # Add points to scene
    scene.add_points(
        "DenseRainbowSphere",
        positions,
        colors=colors,
        radii=radii,
        sharpness=sharpness,
    )

    scene.finalize()

    # Print summary
    aprint(f"\n✓ Created dense rainbow sphere with {n_points:,} points")
    aprint(f"  Sphere radius: {sphere_radius} units")
    aprint(f"  Point radius: {point_radius:.4f} units")
    aprint(f"  Average spacing: ~{2 * point_radius:.4f} units")
    aprint("  Sharpness: 8.0 (high - crisp edges)")
    aprint("  Colors: Full rainbow spectrum")

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
    aprint("   - High sharpness creates crisp, well-defined points")
    aprint("   - No overlapping due to calculated spacing")
    aprint("=" * 60)


if __name__ == "__main__":
    main()
