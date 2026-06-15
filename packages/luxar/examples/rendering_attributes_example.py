#!/usr/bin/env python3
"""Rendering Attributes Example - Educational demonstration of point rendering properties.

This educational example demonstrates:
- Setting and modifying rendering attributes (opacity, gamma, blending_mode)
- Property validation and error handling
- Method chaining for fluent API usage
- Runtime modification of point properties
- Visual comparison of different attribute settings

Educational value:
- Learn how to set and modify rendering properties on points
- Understand validation rules for opacity, gamma, and blending modes
- See method chaining for fluent API design patterns
- Understand how to compare different attribute combinations visually
- Learn best practices for property modification and error handling
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_grid_positions(n_points: int, spacing: float = 1.0) -> np.ndarray:
    """Create a grid of point positions for organized display.

    Args:
        n_points: Number of points (will use sqrt for grid dimensions)
        spacing: Distance between points

    Returns:
        Array of 3D positions in a grid
    """
    grid_size = int(np.sqrt(n_points))
    x = np.linspace(0, (grid_size - 1) * spacing, grid_size)
    y = np.linspace(0, (grid_size - 1) * spacing, grid_size)

    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)

    positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return positions[:n_points].astype(np.float32)


def main():
    """Create a scene demonstrating rendering attributes and their modifications."""
    output_path = get_examples_output_dir() / "rendering_attributes_example.luxar.zarr"

    with asection("Scene Setup and Initialization"):
        aprint(f"Creating rendering attributes demonstration at {output_path}")
        aprint("This educational example shows:")
        aprint("- Default rendering properties")
        aprint("- Custom opacity, gamma, and blending modes")
        aprint("- Runtime property modification")
        aprint("- Method chaining for fluent API")
        aprint("- Validation and error handling")

    # Create scene
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # Parameters
        n_points = 100
        base_positions = create_grid_positions(n_points, spacing=0.5)
        point_radius = 0.08
        aprint(f"Using {n_points} points in organized grid layout")

        with asection("Creating Points with Different Attributes"):
            # 1. Default rendering properties
            aprint("Creating points with default rendering attributes...")
            points1 = scene.add_points(
                "DefaultAttributes",
                base_positions + np.array([0, 0, 0]),
                colors=[1.0, 0.39, 0.39],  # Red
                radii=point_radius,
            )
            aprint(
                f"  Default - opacity: {points1.opacity}, gamma: {points1.gamma}, blending: {points1.blending_mode}"
            )

            # 2. Custom opacity (transparency)
            aprint("Creating semi-transparent points...")
            points2 = scene.add_points(
                "TransparentPoints",
                base_positions + np.array([0, 3, 0]),
                colors=[0.39, 1.0, 0.39],  # Green
                radii=point_radius,
                opacity=0.6,
            )
            aprint(
                f"  Transparent - opacity: {points2.opacity}, gamma: {points2.gamma}, blending: {points2.blending_mode}"
            )

            # 3. Custom gamma (brightness)
            aprint("Creating bright points with high gamma...")
            points3 = scene.add_points(
                "BrightPoints",
                base_positions + np.array([6, 0, 0]),
                colors=[0.39, 0.39, 1.0],  # Blue
                radii=point_radius,
                gamma=1.8,
            )
            aprint(
                f"  Bright - opacity: {points3.opacity}, gamma: {points3.gamma}, blending: {points3.blending_mode}"
            )

            # 4. Normal blending mode with custom opacity
            aprint("Creating normal blending points...")
            points4 = scene.add_points(
                "NormalBlending",
                base_positions + np.array([6, 3, 0]),
                colors=[1.0, 1.0, 0.39],  # Yellow
                radii=point_radius,
                blending_mode="normal",
                opacity=0.8,
            )
            aprint(
                f"  Normal blend - opacity: {points4.opacity}, gamma: {points4.gamma}, blending: {points4.blending_mode}"
            )

        with asection("Runtime Property Modification"):
            # 5. Demonstrate runtime property modification
            aprint("Modifying first points's attributes...")
            original_opacity = points1.opacity
            original_gamma = points1.gamma
            original_blending = points1.blending_mode

            points1.opacity = 0.7
            points1.gamma = 0.8
            points1.blending_mode = "normal"
            aprint(f"  Changed from opacity={original_opacity} to {points1.opacity}")
            aprint(f"  Changed from gamma={original_gamma} to {points1.gamma}")
            aprint(
                f"  Changed from blending='{original_blending}' to '{points1.blending_mode}'"
            )

        with asection("Method Chaining API"):
            # 6. Demonstrate method chaining
            aprint("Chaining multiple attribute changes...")
            points2.set_opacity(0.4).set_gamma(1.3).set_blending_mode("additive")
            aprint(
                f"  Chained result - opacity: {points2.opacity}, gamma: {points2.gamma}, blending: {points2.blending_mode}"
            )

        with asection("Validation and Error Handling"):
            # 7. Demonstrate validation and error handling
            aprint("Testing property validation...")

            # Test opacity validation
            try:
                points1.opacity = 1.5  # Should fail - opacity must be 0.0 to 1.0
                aprint("  ERROR: Opacity validation failed!")
            except ValueError as e:
                aprint(f"  ✓ Opacity validation working: {e}")

            # Test gamma validation
            try:
                points1.gamma = 0.05  # Should fail - gamma must be >= 0.1
                aprint("  ERROR: Gamma validation failed!")
            except ValueError as e:
                aprint(f"  ✓ Gamma validation working: {e}")

            # Test blending mode validation
            try:
                points1.blending_mode = "invalid_mode"  # Should fail
                aprint("  ERROR: Blending mode validation failed!")
            except ValueError as e:
                aprint(f"  ✓ Blending mode validation working: {e}")

        # Explainer overlay describing what to look for in the viewer.
        add_explainer(
            scene,
            title="Point Rendering Attributes",
            body=(
                "Four point grids contrast <code>opacity</code>, "
                "<code>gamma</code>, and <code>blending_mode</code>. The script "
                "also mutates attributes at runtime and exercises validation, "
                "showing these are <strong>live, checked</strong> properties."
            ),
            observe=[
                "Bottom-left red uses default attributes as the baseline.",
                "Top-left green is visibly semi-transparent at 60% opacity.",
                "Bottom-right blue is brighter from its higher <code>gamma</code>.",
                "Console shows validation rejecting out-of-range values.",
            ],
            observe_label="Verify",
        )

        with asection("Educational Summary"):
            # Print educational summary
            aprint("=" * 60)
            aprint("RENDERING ATTRIBUTES DEMONSTRATION")
            aprint("=" * 60)
            aprint("Scene Layout:")
            aprint("  Bottom-left: Red points with default attributes")
            aprint("  Top-left: Green points with 60% opacity")
            aprint("  Bottom-right: Blue points with gamma=1.8 (bright)")
            aprint("  Top-right: Yellow points with normal blending")

            aprint("\nRuntime Modifications:")
            aprint("- Red points: Modified to normal blending, 70% opacity, gamma=0.8")
            aprint(
                "- Green points: Chained to additive blending, 40% opacity, gamma=1.3"
            )

            aprint("\nRendering Properties:")
            aprint("- Opacity: Controls transparency (0.0 = transparent, 1.0 = opaque)")
            aprint("- Gamma: Brightness correction (0.1 to 10.0, 1.0 = neutral)")
            aprint("- Blending: How colors combine (normal, additive)")

            aprint("\nValidation Rules:")
            aprint("- Opacity: Must be between 0.0 and 1.0")
            aprint("- Gamma: Must be between 0.1 and 10.0")
            aprint("- Blending: Must be valid mode (normal, additive)")

            aprint(f"\nTo view: luxar serve {output_path}")
            aprint("Python-side attribute handling working correctly!")
            aprint("=" * 60)


if __name__ == "__main__":
    main()
