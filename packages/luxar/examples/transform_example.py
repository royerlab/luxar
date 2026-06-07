#!/usr/bin/env python3
"""Transform Example - Demonstrates the Luxar transform system.

This educational example demonstrates:
- Creating and applying 3D transformations (translate, rotate, scale)
- Composing multiple transformations
- Transform inheritance in hierarchies
- Using the luxar.transforms module
- Visualizing coordinate systems and transformations

Educational value:
- Learn how translate, rotate, and scale transforms work
- Understand composition order (right-multiply convention)
- See how child nodes inherit parent transforms
- Master the luxar.transforms API for scene construction
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler, transforms
from luxar.utils.paths import get_examples_output_dir


def create_coordinate_axes(length: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Create coordinate axis lines for visualization.

    Args:
        length: Length of each axis

    Returns:
        Tuple of (positions, colors) for axis visualization
    """
    # Create axis points: X (red), Y (green), Z (blue)
    positions = np.array(
        [
            # X-axis points (red)
            [0, 0, 0],
            [length, 0, 0],
            # Y-axis points (green)
            [0, 0, 0],
            [0, length, 0],
            # Z-axis points (blue)
            [0, 0, 0],
            [0, 0, length],
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            # X-axis (red)
            [1.0, 0.39, 0.39],
            [1.0, 0.39, 0.39],
            # Y-axis (green)
            [0.39, 1.0, 0.39],
            [0.39, 1.0, 0.39],
            # Z-axis (blue)
            [0.39, 0.39, 1.0],
            [0.39, 0.39, 1.0],
        ],
        dtype=np.float32,
    )

    return positions, colors


def create_cube_points(size: float = 1.0, density: int = 5) -> np.ndarray:
    """Create points arranged in a cube shape.

    Args:
        size: Size of the cube
        density: Number of points along each edge

    Returns:
        Array of 3D positions forming a cube
    """
    # Create grid of points
    coords = np.linspace(-size / 2, size / 2, density)
    positions = []

    # Add points on each face of the cube
    for i in range(density):
        for j in range(density):
            # Front and back faces
            positions.append([-size / 2, coords[i], coords[j]])
            positions.append([size / 2, coords[i], coords[j]])
            # Left and right faces
            positions.append([coords[i], -size / 2, coords[j]])
            positions.append([coords[i], size / 2, coords[j]])
            # Top and bottom faces
            positions.append([coords[i], coords[j], -size / 2])
            positions.append([coords[i], coords[j], size / 2])

    return np.array(positions, dtype=np.float32)


def main():
    """Create a scene demonstrating various transformations."""
    output_path = get_examples_output_dir() / "transform_example.zarr"

    aprint(f"Creating transform system demonstration at {output_path}")
    aprint("This example shows:")
    aprint("- Basic transformations: translate, rotate, scale")
    aprint("- Transform composition and chaining")
    aprint("- Coordinate system visualization")
    aprint("- Transform inheritance in hierarchies")

    # Create scene

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # 1. Origin coordinate system (reference)
        aprint("\nCreating reference coordinate system at origin...")
        origin_pos, origin_colors = create_coordinate_axes(2.0)
        scene.add_points(
            "OriginAxes", origin_pos, colors=origin_colors, radii=0.1, sharpness=4.0
        )

        # 2. Basic translation
        aprint("Demonstrating translation transform...")
        cube_positions = create_cube_points(1.0, 4)
        translation = transforms.translate(3, 0, 0)

        scene.add_points(
            "TranslatedCube",
            cube_positions,
            colors=[1.0, 0.78, 0.39],  # Orange
            radii=0.08,
            transform=translation,
        )

        # Add coordinate system for translated object
        translated_axes_pos, translated_axes_colors = create_coordinate_axes(1.5)
        scene.add_points(
            "TranslatedAxes",
            translated_axes_pos,
            colors=translated_axes_colors,
            radii=0.08,
            transform=translation,
            sharpness=4.0,
        )

        # 3. Basic rotation
        aprint("Demonstrating rotation transform...")
        rotation = transforms.rotate_z(45)  # 45 degrees around Z
        rotated_translation = transforms.translate(0, 3, 0)
        rotated_transform = transforms.compose(rotation, rotated_translation)

        scene.add_points(
            "RotatedCube",
            cube_positions,
            colors=[0.39, 1.0, 0.78],  # Cyan
            radii=0.08,
            transform=rotated_transform,
        )

        # Add coordinate system for rotated object
        scene.add_points(
            "RotatedAxes",
            translated_axes_pos,
            colors=translated_axes_colors,
            radii=0.08,
            transform=rotated_transform,
            sharpness=4.0,
        )

        # 4. Scaling
        aprint("Demonstrating scaling transform...")
        scale = transforms.scale(0.5, 0.5, 2.0)  # Thin and tall
        scaled_translation = transforms.translate(-3, 0, 0)
        scaled_transform = transforms.compose(scale, scaled_translation)

        scene.add_points(
            "ScaledCube",
            cube_positions,
            colors=[1.0, 0.39, 0.78],  # Magenta
            radii=0.08,
            transform=scaled_transform,
        )

        # Add coordinate system for scaled object
        scene.add_points(
            "ScaledAxes",
            translated_axes_pos,
            colors=translated_axes_colors,
            radii=0.08,
            transform=scaled_transform,
            sharpness=4.0,
        )

        # 5. Complex composition
        aprint("Demonstrating complex transform composition...")
        # Rotate around Y, then scale, then translate
        complex_transform = transforms.compose(
            transforms.rotate_y(30),  # 30 degrees
            transforms.scale(1.5, 0.8, 1.2),  # Non-uniform scale
            transforms.translate(0, -3, 0),  # Move down
        )

        scene.add_points(
            "ComplexTransformCube",
            cube_positions,
            colors=[0.78, 0.39, 1.0],  # Purple
            radii=0.08,
            transform=complex_transform,
        )

        scene.add_points(
            "ComplexAxes",
            translated_axes_pos,
            colors=translated_axes_colors,
            radii=0.08,
            transform=complex_transform,
            sharpness=4.0,
        )

        # 6. Hierarchical transforms (parent-child)
        aprint("Demonstrating hierarchical transforms...")

        # Create parent group with transform
        parent_transform = transforms.compose(
            transforms.translate(0, 0, 3), transforms.rotate_z(22.5)
        )
        parent_group = scene.add_group("ParentGroup", transform=parent_transform)

        # Child objects inherit parent's transform
        small_cube_positions = create_cube_points(0.5, 3)

        # Child 1: Only has local translation
        child1_transform = transforms.translate(1, 1, 0)
        scene.add_points(
            "Child1",
            small_cube_positions,
            colors=[1.0, 1.0, 0.39],  # Yellow
            radii=0.06,
            transform=child1_transform,
            parent=parent_group,
        )

        # Child 2: Local rotation and translation
        child2_transform = transforms.compose(
            transforms.rotate_x(45), transforms.translate(-1, 1, 0)
        )
        scene.add_points(
            "Child2",
            small_cube_positions,
            colors=[0.39, 1.0, 1.0],  # Light blue
            radii=0.06,
            transform=child2_transform,
            parent=parent_group,
        )

        # Parent coordinate system
        scene.add_points(
            "ParentAxes",
            translated_axes_pos,
            colors=translated_axes_colors,
            radii=0.06,
            transform=transforms.identity(),  # No additional transform
            parent=parent_group,
            sharpness=4.0,
        )

        add_explainer(
            scene,
            title="Transform System",
            body=(
                "Each cube is the same geometry placed by a different "
                "transform built with <code>transforms.translate/rotate/"
                "scale</code> and combined via <code>compose</code>. "
                "Coloured <strong>RGB axes</strong> mark each object's local "
                "frame; the back group shows children inheriting a parent "
                "transform."
            ),
            observe=[
                "Orange cube sits at <code>+X</code>, magenta at "
                "<code>-X</code> (thin and tall from non-uniform scale).",
                "Cyan cube is rotated 45 deg about Z at <code>+Y</code>.",
                "Each cube carries its own tilted red/green/blue axes.",
                "Back group's yellow/blue children move with the parent.",
            ],
            observe_label="Look for",
        )

        # Print educational summary
        aprint("\n" + "=" * 60)
        aprint("TRANSFORM SYSTEM DEMONSTRATION")
        aprint("=" * 60)
        aprint("Scene Layout:")
        aprint("  Center: Reference coordinate system (origin)")
        aprint("  Right: Translated cube (orange) + axes")
        aprint("  Top: Rotated cube (cyan) + axes")
        aprint("  Left: Scaled cube (magenta) + axes")
        aprint("  Bottom: Complex transform (purple) + axes")
        aprint("  Back: Hierarchical group (yellow/blue children)")

        aprint("\nTransformation Types:")
        aprint("- Translation: Moves objects in space")
        aprint("- Rotation: Rotates around axes (X, Y, Z)")
        aprint("- Scaling: Changes size (uniform or non-uniform)")
        aprint("- Composition: Combines multiple transforms")
        aprint("- Hierarchical: Child objects inherit parent transforms")

        aprint("\nTransform Functions Used:")
        aprint("- transforms.translate(x, y, z)")
        aprint("- transforms.rotate_x/y/z(angle_degrees)")
        aprint("- transforms.scale(sx, sy, sz)")
        aprint("- transforms.compose(transform1, transform2, ...)")
        aprint("- transforms.identity() - no transformation")

        aprint("\nCoordinate Systems:")
        aprint("- Red axis: X direction")
        aprint("- Green axis: Y direction")
        aprint("- Blue axis: Z direction")
        aprint("- Each transformed object shows its local coordinate system")

        aprint(f"\nTo view: luxar serve {output_path}")
        aprint("Rotate the view to see all transformations clearly!")
        aprint("=" * 60)


if __name__ == "__main__":
    main()
