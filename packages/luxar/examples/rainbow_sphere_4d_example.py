#!/usr/bin/env python3
"""4D Spatial Rainbow Sphere Example - True 4D hypersphere with all spatial dimensions.

This example demonstrates:
- True 4D spatial geometry (X, Y, Z, W)
- Points distributed on a 4D hypersphere
- 3D slices through W dimension show spheres of varying radius
- Rainbow colors based on 4D position
- Proper scene dimension definitions for 4D spatial data

Educational value:
- Understand true 4D spatial geometry (not time-based 4D)
- Learn how 3D slices through a hypersphere change with W position
- See the relationship between slice position and visible sphere radius
- Master 4D dimension configuration for spatial data
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_4d_hypersphere_points(
    n_points: int = 500000, radius: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Create points uniformly distributed on a 4D hypersphere.

    Uses Gaussian method to get uniform distribution on the 4D sphere surface.

    Args:
        n_points: Number of points to generate
        radius: Radius of the hypersphere

    Returns:
        Tuple of (positions, radii) where positions is (n_points, 4) and radii is (n_points,)
    """
    aprint(f"├ Generating {n_points} points on 4D hypersphere...")

    # Generate points on unit 4-sphere using normal distribution method
    # This gives uniform distribution on the sphere surface
    # Generate 4D Gaussian random points
    points = np.random.randn(n_points, 4)
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    # Scale to desired radius
    points *= radius

    # Calculate point radii based on density
    # For a 4D hypersphere, surface area is proportional to r^3
    # So point spacing is approximately proportional to (r^3 / n_points)^(1/3)
    point_spacing = radius * (1.0 / n_points) ** (1.0 / 3.0)
    radii = np.full(
        n_points, point_spacing * 2.5, dtype=np.float32
    )  # 2.5x for visibility

    return points.astype(np.float32), radii


def create_4d_rainbow_colors(positions: np.ndarray) -> np.ndarray:
    """Create rainbow colors based on 4D position.

    Uses a combination of angular positions in 4D space to create smooth color gradients.

    Args:
        positions: 4D positions array (n_points, 4)

    Returns:
        RGB color array (n_points, 3) as float32
    """
    n_points = len(positions)

    # Calculate angles in different 2D planes
    # XY plane angle
    angle_xy = np.arctan2(positions[:, 1], positions[:, 0])
    # ZW plane angle
    angle_zw = np.arctan2(positions[:, 3], positions[:, 2])

    # Combine angles to create hue
    # Normalize to [0, 1]
    hue = (angle_xy / (2 * np.pi) + 0.5) % 1.0

    # Use ZW angle to modulate saturation and value
    saturation = 0.7 + 0.3 * np.sin(angle_zw)
    value = 0.8 + 0.2 * np.cos(angle_zw)

    # Convert HSV to RGB using vectorized operations
    h = hue * 6.0
    c = value * saturation
    x = c * (1 - np.abs(h % 2 - 1))
    m = value - c

    # Create RGB arrays
    r = np.zeros(n_points, dtype=np.float32)
    g = np.zeros(n_points, dtype=np.float32)
    b = np.zeros(n_points, dtype=np.float32)

    # Assign colors based on hue sector
    mask1 = (h >= 0) & (h < 1)
    r[mask1] = c[mask1]
    g[mask1] = x[mask1]

    mask2 = (h >= 1) & (h < 2)
    r[mask2] = x[mask2]
    g[mask2] = c[mask2]

    mask3 = (h >= 2) & (h < 3)
    g[mask3] = c[mask3]
    b[mask3] = x[mask3]

    mask4 = (h >= 3) & (h < 4)
    g[mask4] = x[mask4]
    b[mask4] = c[mask4]

    mask5 = (h >= 4) & (h < 5)
    r[mask5] = x[mask5]
    b[mask5] = c[mask5]

    mask6 = h >= 5
    r[mask6] = c[mask6]
    b[mask6] = x[mask6]

    # Add minimum value and combine
    colors = np.stack([r + m, g + m, b + m], axis=1)

    # Return as float32
    return colors.astype(np.float32)


def main():
    """Create and save a 4D spatial rainbow sphere."""
    output_path = get_examples_output_dir() / "rainbow_sphere_4d_example.zarr"

    aprint(f"├ Creating 4D spatial rainbow sphere at {output_path}")
    aprint("├ This example creates a true 4D hypersphere where:")
    aprint("├ - All 4 dimensions (X, Y, Z, W) are spatial")
    aprint("├ - Points are distributed on the 4D sphere surface")
    aprint("├ - Slicing through W shows 3D spheres of varying radius")
    aprint("├ - Colors create a 4D rainbow pattern")

    # Create scene with 4D dimensions
    dimensions = Dimensions(
        [
            Dimension(name="X", unit="", range=(-15, 15), display=True),
            Dimension(name="Y", unit="", range=(-15, 15), display=True),
            Dimension(name="Z", unit="", range=(-15, 15), display=True),
            Dimension(
                name="W",
                unit="",
                range=(-15, 15),
                display=False,
                spatial=True,
                step=0.5,
            ),
        ]
    )

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=dimensions)
        aprint("│")

        # Generate 4D hypersphere points
        positions, radii = create_4d_hypersphere_points(n_points=50000, radius=10.0)

        # Create rainbow colors based on 4D position
        aprint("├ Creating 4D rainbow colors...")
        colors = create_4d_rainbow_colors(positions)

        # Create sharpness array (high sharpness for crisp points)
        sharpness = np.full(len(positions), 0.9, dtype=np.float32)

        # Add points to scene
        aprint(f"├ Adding points node with {len(positions)} points in 4D space")
        scene.add_points(
            "Rainbow4DHypersphere",
            positions=positions,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
        )

        add_explainer(
            scene,
            title="4D Spatial Hypersphere",
            body=(
                "Points lie on a true 4D sphere (X, Y, Z, W all spatial); the "
                "non-displayed <code>W</code> axis is sliced to reveal 3D "
                "cross-sections. Press <code>4</code> then <code>[</code>/"
                "<code>]</code> to step through W."
            ),
            observe=[
                "The visible 3D sphere grows toward W=0 and shrinks toward the edges.",
                "No points appear near W=-10 or W=+10 (outside the hypersphere).",
                "Colors form a smooth 4D rainbow that shifts with W.",
            ],
            observe_label="Look for",
        )

        # Finalize scene

        aprint("│")
        aprint(f"│ ✓ Created 4D spatial rainbow sphere with {len(positions)} points")
        aprint("├   Hypersphere radius: 10.0 units")
        aprint(f"├   Point radii: {radii[0]:.3f} units")
        aprint("├   Colors: 4D rainbow pattern")
        aprint("│")
        aprint("├" + "=" * 70)
        aprint("├ 4D VIEWING INSTRUCTIONS")
        aprint("├" + "=" * 70)
        aprint("├ 1. Start the server:")
        aprint(f"├    luxar serve {output_path}")
        aprint("│")
        aprint("├ 2. Navigate through the W dimension:")
        aprint("├    - Press '1' to control the W dimension")
        aprint("├    - Use '[' and ']' keys to slice through W")
        aprint("├    - Watch the 3D sphere grow and shrink!")
        aprint("│")
        aprint("├ 3. What you'll see at different W values:")
        aprint("├    - W = -10: No points (outside hypersphere)")
        aprint("├    - W = -5: Small sphere (edge of hypersphere)")
        aprint("├    - W = 0: Large sphere (center slice)")
        aprint("├    - W = +5: Small sphere (other edge)")
        aprint("├    - W = +10: No points (outside again)")
        aprint("│")
        aprint("├ 4. This demonstrates:")
        aprint("├    - True 4D geometry (not 3D + time)")
        aprint("├    - How 3D slices of 4D objects change shape")
        aprint("├    - The geometry of a 4D hypersphere")
        aprint("├" + "=" * 70)


if __name__ == "__main__":
    main()
