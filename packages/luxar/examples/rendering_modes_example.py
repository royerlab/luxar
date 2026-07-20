#!/usr/bin/env python3
"""Rendering Modes Example - Demonstrates different blending modes and opacity effects.

This educational example demonstrates:
- Different blending modes (normal and additive; the viewer also supports
  max, opaque, luminous, and volumetric)
- Opacity effects and transparency
- Gamma correction for brightness control
- Parent-child inheritance of rendering properties
- Creating educational visualizations comparing rendering techniques

Educational value:
- Learn how blending modes affect visual appearance
- Understand opacity and gamma correction parameters
- See side-by-side comparison of rendering property combinations
- Know when to use additive vs normal blending
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_sphere_positions(n_points: int = 5000, radius: float = 1.0) -> np.ndarray:
    """Create evenly distributed points on a sphere surface.

    Args:
        n_points: Number of points to generate
        radius: Radius of the sphere

    Returns:
        Array of 3D positions
    """
    # Generate points using spherical coordinates
    indices = np.arange(0, n_points, dtype=float) + 0.5
    theta = np.arccos(1 - 2 * indices / n_points)  # Polar angle
    phi = np.pi * (1 + 5**0.5) * indices  # Azimuthal angle (golden ratio)

    # Convert to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.column_stack([x, y, z]).astype(np.float32)


def main():
    """Create a scene demonstrating different rendering modes and effects."""
    output_path = get_examples_output_dir() / "rendering_modes_example.luxar.zarr"

    aprint(f"Creating rendering modes demonstration at {output_path}")
    aprint("This example shows:")
    aprint("- Normal blending: Standard transparency")
    aprint("- Additive blending: Glowing/HDR effects")
    aprint("- Opacity control: Transparency levels")
    aprint("- Gamma correction: Brightness adjustment")
    aprint("- Property inheritance: Parent-child relationships")

    # Create scene

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # Parameters
        n_points = 5000
        radius = 1.0
        point_radius = 0.05

        # Create base sphere positions
        base_positions = create_sphere_positions(n_points, radius)

        # 1. Normal blending - opaque red sphere (reference)
        aprint("\nCreating normal blending examples...")
        positions1 = base_positions + np.array([-3, 2, 0])
        colors1 = [1.0, 0.31, 0.31]  # Red
        scene.add_points(
            "NormalOpaque",
            positions1,
            colors=colors1,
            radii=point_radius,
            blending_mode="normal",
            opacity=1.0,
            gamma=1.0,
        )

        # 2. Normal blending - transparent green sphere
        positions2 = base_positions + np.array([0, 2, 0])
        # NOTE: integer triple is detected as HDR (max>1.0); the visual
        # E2E baseline (blending-modes-rendering.png) captures this. Do
        # NOT change to floats — see packages/luxar-viewer/.../blending-modes.spec.ts.
        colors2 = [80, 255, 80]  # Green (HDR-detected by intent — see note above)
        scene.add_points(
            "NormalTransparent",
            positions2,
            colors=colors2,
            radii=point_radius,
            blending_mode="normal",
            opacity=0.6,
            gamma=1.0,
        )

        # 3. Additive blending - glowing blue sphere
        aprint("Creating additive blending (glow effect)...")
        positions3 = base_positions + np.array([3, 2, 0])
        colors3 = [0.39, 0.59, 1.0]  # Light blue
        scene.add_points(
            "AdditiveGlow",
            positions3,
            colors=colors3,
            radii=point_radius * 1.2,  # Slightly larger for glow
            blending_mode="additive",
            opacity=0.8,
            gamma=1.3,  # Brighter
        )

        # 4. Demonstration of property inheritance
        aprint("Creating parent-child inheritance examples...")

        # Create parent group with shared properties
        parent_group = scene.add_group(
            "InheritanceGroup", opacity=0.4, blending_mode="additive", gamma=1.1
        )

        # Child 1: Inherits all parent properties
        positions5 = base_positions + np.array([0, -2, 0])
        colors5 = [1.0, 1.0, 0.39]  # Yellow
        scene.add_points(
            "InheritedYellow",
            positions5,
            colors=colors5,
            radii=point_radius,
            parent=parent_group,
            # Inherits: opacity=0.4, blending_mode="additive", gamma=1.1
        )

        # Child 2: Overrides parent's gamma while inheriting other properties
        positions6 = base_positions + np.array([3, -2, 0])
        colors6 = [0.39, 1.0, 1.0]  # Cyan
        scene.add_points(
            "OverrideCyan",
            positions6,
            colors=colors6,
            radii=point_radius,
            parent=parent_group,
            gamma=1.8,  # Override parent's gamma
            # Inherits: opacity=0.4, blending_mode="additive"
        )

        # Explainer overlay describing what to look for in the viewer.
        add_explainer(
            scene,
            title="Blending Modes & Inheritance",
            body=(
                "Five spheres compare <code>blending_mode</code>, "
                "<code>opacity</code>, and <code>gamma</code>. A parent group "
                "shares additive blending with its children, one of which "
                "<strong>overrides</strong> the inherited gamma."
            ),
            observe=[
                "Top-left red is fully opaque; top-center green is 60% transparent.",
                "Top-right blue glows where <code>additive</code> samples overlap.",
                "Bottom yellow inherits the group's settings; cyan overrides gamma.",
                "Additive cores brighten toward white as points stack.",
            ],
            observe_label="Look for",
        )

        # Print educational summary
        aprint("\n" + "=" * 60)
        aprint("RENDERING MODES DEMONSTRATION")
        aprint("=" * 60)
        aprint("Scene Layout (viewed from front):")
        aprint("  Top row (y=2):")
        aprint("    Left:   Red sphere - Normal blending, opaque")
        aprint("    Center: Green sphere - Normal blending, 60% transparent")
        aprint("    Right:  Blue sphere - Additive blending, glowing effect")
        aprint("  Bottom row (y=-2):")
        aprint("    Center: Yellow sphere - Inherits group properties")
        aprint("    Right:  Cyan sphere - Inherits + overrides gamma")

        aprint("\nRendering Properties Explained:")
        aprint("- Normal blending: Standard alpha compositing")
        aprint("- Additive blending: Colors add together (HDR/glow)")
        aprint("- Opacity: Controls transparency (0.0 to 1.0)")
        aprint("- Gamma: Brightness correction (0.2 to 5.0)")

        aprint("\nInheritance Example:")
        aprint("- Group has opacity=0.4, additive blending, gamma=1.1")
        aprint("- Yellow sphere inherits all properties")
        aprint("- Cyan sphere inherits opacity + blending, overrides gamma=1.8")

        aprint(f"\nTo view: luxar serve {output_path}")
        aprint("=" * 60)


if __name__ == "__main__":
    main()
