#!/usr/bin/env python3
"""4D Temporal Spiral Sphere Example - Creates animated spiral spheres across time.

This example demonstrates:
- Creating a 4D dataset with X, Y, Z spatial dimensions and T temporal dimension
- Rotating spiral sphere animation over time
- Undulating rainbow colors across temporal frames
- Pulsating point sizes with spatial patterns
- Dynamic sharpness variations
- High-density points (200,000 points per frame × 512 frames)

Educational value:
- Learn to create large-scale temporal animations
- Understand time as a discrete navigation dimension
- See how per-frame transformations create smooth animation
- Good stress test for high-density temporal rendering
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_spherical_spiral(
    n_points: int = 200000, radius: float = 10.0, rotation: float = 0.0
) -> np.ndarray:
    """Create points distributed in a spherical spiral pattern with rotation.

    Uses a technique similar to Fibonacci spiral on a sphere for even distribution.

    Args:
        n_points: Number of points to generate
        radius: Radius of the sphere
        rotation: Additional rotation angle in radians

    Returns:
        Array of 3D positions
    """
    indices = np.arange(0, n_points, dtype=float) + 0.5

    # Golden angle in radians
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    # Generate spherical coordinates with rotation
    theta = indices * golden_angle + rotation  # Add rotation to azimuthal angle
    y = 1 - (indices / float(n_points - 1)) * 2  # y goes from 1 to -1
    # Clamp y to avoid numerical issues at poles
    y = np.clip(y, -1.0, 1.0)
    radius_at_y = np.sqrt(1 - y * y)  # Radius at y

    # Convert to Cartesian coordinates
    x = np.cos(theta) * radius_at_y * radius
    z = np.sin(theta) * radius_at_y * radius
    y = y * radius

    return np.column_stack([x, y, z]).astype(np.float32)


def create_undulating_rainbow_colors(
    n_points: int, t_frame: int, n_frames: int
) -> np.ndarray:
    """Generate rainbow colors that undulate across time.

    Args:
        n_points: Number of color values to generate
        t_frame: Current time frame
        n_frames: Total number of frames

    Returns:
        Array of RGB colors
    """
    # Parameter along the spiral
    spiral_t = np.linspace(0, 1, n_points)

    # Time-based phase shift for undulation
    time_phase = 2 * np.pi * t_frame / n_frames

    # Create undulating rainbow using sine waves with time-based phase
    # The colors "flow" along the spiral over time
    color_freq = 2 * np.pi  # Frequency of color changes along spiral
    wave_speed = 2.0  # Speed of color wave propagation

    r = np.sin(color_freq * spiral_t + time_phase * wave_speed) * 0.5 + 0.5
    g = (
        np.sin(color_freq * spiral_t + time_phase * wave_speed + 2 * np.pi / 3) * 0.5
        + 0.5
    )
    b = (
        np.sin(color_freq * spiral_t + time_phase * wave_speed + 4 * np.pi / 3) * 0.5
        + 0.5
    )

    return np.column_stack([r, g, b]).astype(np.float32)


def create_pulsating_radii(
    positions: np.ndarray, base_radius: float, t_frame: int, n_frames: int
) -> np.ndarray:
    """Generate pulsating point radii with spatial patterns.

    Args:
        positions: 3D positions of points
        base_radius: Base radius for points
        t_frame: Current time frame
        n_frames: Total number of frames

    Returns:
        Array of point radii
    """
    # Time-based pulsation
    time_phase = 2 * np.pi * t_frame / n_frames

    # Create spatial pattern based on position
    # Use distance from origin and angular position for interesting patterns
    distances = np.linalg.norm(positions, axis=1)
    theta = np.arctan2(positions[:, 2], positions[:, 0])
    phi = np.arcsin(np.clip(positions[:, 1] / (distances + 1e-6), -1, 1))

    # Combine multiple pulsation patterns
    # Pattern 1: Global pulsation
    global_pulse = np.sin(time_phase * 3) * 0.1

    # Pattern 2: Latitudinal waves
    lat_pulse = np.sin(phi * 4 + time_phase * 2) * 0.1

    # Pattern 3: Longitudinal waves
    long_pulse = np.cos(theta * 6 - time_phase * 1.5) * 0.05

    # Combine patterns (max ±20% change)
    radius_multiplier = 1.0 + global_pulse + lat_pulse + long_pulse
    radius_multiplier = np.clip(radius_multiplier, 0.8, 1.2)  # Ensure ±20% max

    return (base_radius * radius_multiplier).astype(np.float32)


def create_dynamic_sharpness(
    positions: np.ndarray, t_frame: int, n_frames: int
) -> np.ndarray:
    """Generate dynamic sharpness values with spatial and temporal variation.

    Args:
        positions: 3D positions of points
        t_frame: Current time frame
        n_frames: Total number of frames

    Returns:
        Array of sharpness values
    """
    # Time-based variation
    time_phase = 2 * np.pi * t_frame / n_frames

    # Create spatial pattern
    y_normalized = (positions[:, 1] + 10) / 20  # Normalize y from [0, 1]

    # Sharpness waves from poles to equator
    # Creates bands of varying sharpness that move over time
    sharpness_pattern = np.sin(y_normalized * 8 * np.pi + time_phase * 4)

    # Map to the normalized [0, 1] sharpness knob, range [0.25, 0.75]
    # When sharpness_pattern is -1: sharpness = 0.25 (softer/peakier)
    # When sharpness_pattern is +1: sharpness = 0.75 (crisper)
    sharpness = 0.5 + sharpness_pattern * 0.25

    return sharpness.astype(np.float32)


def calculate_point_spacing(n_points: int, radius: float) -> float:
    """Calculate appropriate point spacing for the given sphere and point count.

    Args:
        n_points: Number of points
        radius: Radius of the sphere

    Returns:
        Estimated spacing between neighboring points
    """
    # Surface area of sphere
    surface_area = 4 * np.pi * radius**2
    # Area per point
    area_per_point = surface_area / n_points
    # Approximate spacing (assuming uniform distribution)
    spacing = np.sqrt(area_per_point)
    return spacing


def main():
    """Generate the 4D temporal spiral sphere dataset."""
    aprint("Creating 4D Temporal Spiral Sphere Example")
    aprint("=" * 50)

    # Parameters - increased for testing lazy loading
    n_points_per_frame = 200000
    n_frames = 512
    sphere_radius = 10.0

    # Calculate total rotation over all frames (10 points worth)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    total_rotation = 10 * golden_angle  # 10 points worth of rotation

    # Calculate point spacing for radius
    spacing = calculate_point_spacing(n_points_per_frame, sphere_radius)
    point_radius = spacing * 0.4  # 40% of spacing for good coverage

    aprint(f"Generating {n_points_per_frame:,} points per frame")
    aprint(f"Total frames: {n_frames}")
    aprint(f"Total points: {n_points_per_frame * n_frames:,}")
    aprint(f"Sphere radius: {sphere_radius}")
    aprint(f"Point spacing: {spacing:.4f}")
    aprint(f"Point radius: {point_radius:.4f}")
    aprint(f"Total rotation: {total_rotation:.4f} radians ({10} points)")

    # Output path
    output_path = get_examples_output_dir() / "temporal_spiral_sphere_4d_example.zarr"

    # Create scene with 4D dimensions (only first 3 displayed)
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions(
                [
                    Dimension(
                        name="x", unit="μm", range=(-15, 15), step=0.5, display=True
                    ),
                    Dimension(
                        name="y", unit="μm", range=(-15, 15), step=0.5, display=True
                    ),
                    Dimension(
                        name="z", unit="μm", range=(-15, 15), step=0.5, display=True
                    ),
                    Dimension(
                        name="t",
                        unit="frame",
                        range=(0, n_frames - 1),
                        step=1,
                        display=False,
                        discrete=True,
                    ),
                ]
            )
        )

        # Prepare 4D arrays for all frames
        all_positions = []
        all_colors = []
        all_radii = []
        all_sharpness = []

        aprint(f"\nGenerating {n_frames} temporal frames...")

        for t in range(n_frames):
            if t % 32 == 0:  # Progress indicator every 32 frames
                aprint(f"  Frame {t}/{n_frames}...")

            # Calculate rotation for this frame
            rotation = (t / n_frames) * total_rotation

            # Generate 3D positions with rotation
            positions_3d = create_spherical_spiral(
                n_points_per_frame, sphere_radius, rotation
            )

            # Add time dimension to create 4D positions
            t_values = np.full((n_points_per_frame, 1), t, dtype=np.float32)
            positions_4d = np.hstack([positions_3d, t_values])

            # Generate time-varying properties
            colors = create_undulating_rainbow_colors(n_points_per_frame, t, n_frames)
            radii = create_pulsating_radii(positions_3d, point_radius, t, n_frames)
            sharpness = create_dynamic_sharpness(positions_3d, t, n_frames)

            all_positions.append(positions_4d)
            all_colors.append(colors)
            all_radii.append(radii)
            all_sharpness.append(sharpness)

        # Concatenate all frames
        aprint("\nCombining all frames...")
        all_positions = np.vstack(all_positions)
        all_colors = np.vstack(all_colors)
        all_radii = np.concatenate(all_radii)
        all_sharpness = np.concatenate(all_sharpness)

        # Add points to scene
        aprint("Adding points to scene...")
        scene.add_points(
            "temporal_spiral_sphere",
            positions=all_positions,
            colors=all_colors,
            radii=all_radii,
            sharpness=all_sharpness,
        )

        add_explainer(
            scene,
            title="Temporal Spiral Sphere",
            body=(
                "A dense Fibonacci-spiral sphere animated over a discrete "
                "<code>t</code> dimension, with per-frame undulating colors, "
                "pulsating <code>radii</code>, and dynamic <code>sharpness</code>. "
                "Press <code>4</code> then <code>[</code>/<code>]</code> to step "
                "through time."
            ),
            observe=[
                "The spiral pattern rotates smoothly across time frames.",
                "Rainbow color bands flow along the spiral as time advances.",
                "Point sizes pulse and sharpness bands sweep pole to equator.",
            ],
            observe_label="Look for",
        )

    # Context manager will call finalize automatically
    aprint("\n✅ 4D Temporal Spiral Sphere dataset created successfully!")
    aprint(f"📁 Output: {output_path}")
    aprint(f"📊 Dataset shape: {all_positions.shape}")
    aprint(f"🎨 Color range: [{all_colors.min():.2f}, {all_colors.max():.2f}]")
    aprint(f"📏 Radius range: [{all_radii.min():.4f}, {all_radii.max():.4f}]")
    aprint(
        f"✨ Sharpness range: [{all_sharpness.min():.4f}, {all_sharpness.max():.4f}]"
    )
    aprint("\nTo visualize: luxar serve temporal_spiral_sphere_4d_example.zarr")
    aprint(
        "Then navigate through time with the 't' dimension controls (press 4, then [/])"
    )


if __name__ == "__main__":
    main()
