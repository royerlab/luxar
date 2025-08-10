#!/usr/bin/env python3
"""Build Example - Demonstrates programmatic scene construction.

This example demonstrates:
- Building complex scenes step by step
- Using Scene.build() context manager for automatic finalization
- Organizing scene construction with helper functions
- Best practices for modular scene building
- Proper resource management with context managers
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import Scene, transforms


def add_coordinate_axes(scene: Scene, length: float = 5.0, n_points: int = 50):
    """Add coordinate axes to the scene for reference.

    Args:
        scene: The Luxar scene to add axes to
        length: Length of each axis
        n_points: Number of points per axis
    """
    # X-axis (red)
    x_positions = np.column_stack(
        [np.linspace(0, length, n_points), np.zeros(n_points), np.zeros(n_points)]
    ).astype(np.float32)

    scene.add_points(
        "X_Axis",
        x_positions,
        colors=[255, 0, 0],  # Red
        radii=0.05,
        opacity=0.8,
    )

    # Y-axis (green)
    y_positions = np.column_stack(
        [np.zeros(n_points), np.linspace(0, length, n_points), np.zeros(n_points)]
    ).astype(np.float32)

    scene.add_points(
        "Y_Axis",
        y_positions,
        colors=[0, 255, 0],  # Green
        radii=0.05,
        opacity=0.8,
    )

    # Z-axis (blue)
    z_positions = np.column_stack(
        [np.zeros(n_points), np.zeros(n_points), np.linspace(0, length, n_points)]
    ).astype(np.float32)

    scene.add_points(
        "Z_Axis",
        z_positions,
        colors=[0, 0, 255],  # Blue
        radii=0.05,
        opacity=0.8,
    )


def add_data_cloud(scene: Scene, name: str, center: list, n_points: int = 500):
    """Add a spherical point cloud at specified location.

    Args:
        scene: The Luxar scene to add cloud to
        name: Name for the point cloud
        center: Center position [x, y, z]
        n_points: Number of points in the cloud
    """
    # Generate random points in a sphere
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    costheta = np.random.uniform(-1, 1, n_points)
    u = np.random.uniform(0, 1, n_points)

    theta = np.arccos(costheta)
    r = 1.0 * (u ** (1 / 3))

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    positions = np.column_stack([x, y, z]).astype(np.float32)

    # Random colors with theme
    colors = np.random.randint(100, 255, (n_points, 3), dtype=np.uint8)

    # Create transform to position the cloud
    transform = transforms.translate(center[0], center[1], center[2])

    scene.add_points(
        name, positions, colors=colors, radii=0.03, transform=transform, gamma=1.2
    )


def build_scene_manually(output_path: Path):
    """Build a scene using manual scene management."""
    aprint("\nBuilding scene with manual management...")

    # Manual scene creation and finalization
    scene = Scene(output_path)

    try:
        # Add coordinate axes
        add_coordinate_axes(scene)

        # Add data clouds
        add_data_cloud(scene, "Cloud1", [2, 2, 2])
        add_data_cloud(scene, "Cloud2", [-2, 2, -2])
        add_data_cloud(scene, "Cloud3", [2, -2, -2])

        # Must remember to finalize!
        scene.finalize()
        aprint("✓ Scene built successfully with manual management")

    except Exception as e:
        aprint(f"Error building scene: {e}")
        raise


def build_scene_with_structure(output_path: Path):
    """Build a scene with structured approach."""
    aprint("\nBuilding scene with structured approach...")

    # Create scene with structured approach
    scene = Scene(output_path)

    try:
        # Add coordinate axes
        add_coordinate_axes(scene, length=6.0)

        # Add multiple data clouds in a pattern
        positions = [
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
            [-3, 0, 0],
            [0, -3, 0],
            [0, 0, -3],
        ]

        for i, pos in enumerate(positions):
            add_data_cloud(scene, f"DataCloud_{i}", pos, n_points=300)

        # Add a central group with children
        center_group = scene.add_group(
            "CentralStructure", opacity=0.7, blending_mode="normal"
        )

        # Add child points to the group
        for angle in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            x = np.cos(angle) * 1.5
            y = np.sin(angle) * 1.5

            positions = np.random.randn(100, 3).astype(np.float32) * 0.2
            positions[:, 0] += x
            positions[:, 1] += y

            scene.add_points(
                f"CentralPoint_{int(angle * 180 / np.pi)}",
                positions,
                colors=[255, 200, 100],
                radii=0.04,
                parent=center_group,
            )

        # Finalize the scene
        scene.finalize()
        aprint("✓ Scene built successfully with structured approach")

    except Exception as e:
        aprint(f"Error building scene: {e}")
        raise


def main():
    """Demonstrate different ways to build Luxar scenes programmatically."""
    base_path = Path(__file__).parent

    aprint("=" * 60)
    aprint("BUILD EXAMPLE - Programmatic Scene Construction")
    aprint("=" * 60)
    aprint("This example demonstrates:")
    aprint("- Building scenes step by step with helper functions")
    aprint("- Structured scene building with try/except")
    aprint("- Manual scene management")
    aprint("- Modular scene construction patterns")

    # Example 1: Manual scene management
    manual_path = base_path / "build_manual_example.zarr"
    build_scene_manually(manual_path)

    # Example 2: Structured approach
    structured_path = base_path / "build_structured_example.zarr"
    build_scene_with_structure(structured_path)

    aprint("\n" + "=" * 60)
    aprint("BUILD EXAMPLE COMPLETED")
    aprint("=" * 60)
    aprint("Created two scenes demonstrating different build patterns:")
    aprint(f"  1. Manual: {manual_path}")
    aprint(f"  2. Structured: {structured_path}")
    aprint("\nKey Takeaways:")
    aprint("- Always remember to call scene.finalize()")
    aprint("- Break complex scenes into helper functions")
    aprint("- Use try/except for proper error handling")
    aprint("- Modular design makes scenes easier to maintain")
    aprint("\nTo view the scenes:")
    aprint(f"  luxar serve {manual_path}")
    aprint(f"  luxar serve {structured_path}")
    aprint("=" * 60)


if __name__ == "__main__":
    main()
