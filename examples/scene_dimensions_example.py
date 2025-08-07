#!/usr/bin/env python
"""
Example demonstrating scene-level dimensions with explicit stepping.

This shows how to define dimensions at the scene level with:
- Custom step sizes for navigation
- Units and ranges
- Display configuration
"""

import numpy as np

from luxar import Dimension, Dimensions, Scene


def main():
    """Create a scene with explicit dimension definitions."""
    aprint("Creating scene with explicit dimensions...")

    # Define scene dimensions with custom stepping
    dims = Dimensions(
        [
            # Time dimension with 0.5s steps
            Dimension("time", unit="s", range=(0, 10), step=0.5, display=False),
            # Z dimension with fine 0.1um steps
            Dimension("z", unit="um", range=(-50, 50), step=0.1, display=False),
            # X,Y dimensions displayed with default stepping
            Dimension("y", unit="um", range=(-100, 100), display=True),
            Dimension("x", unit="um", range=(-100, 100), display=True),
            # Discrete channel dimension
            Dimension(
                "channel",
                unit="ch",
                range=(0, 2),
                step=1.0,
                display=True,
                discrete=True,
            ),
        ]
    )

    # Create scene with dimensions
    scene = Scene("scene_dims_test.zarr", dimensions=dims)

    # Create test data that spans the dimension ranges
    n_points_per_combo = 50
    positions = []
    colors = []
    radii = []

    # Sample points at specific time/z combinations
    time_samples = [0, 2.5, 5, 7.5, 10]  # 5 time points
    z_samples = [-40, -20, 0, 20, 40]  # 5 z positions
    channels = [0, 1, 2]  # 3 channels

    for t in time_samples:
        for z in z_samples:
            for ch in channels:
                # Create a small cluster at this time/z/channel
                for _ in range(n_points_per_combo):
                    # Random positions in x,y
                    x = np.random.normal(0, 20)
                    y = np.random.normal(0, 20)

                    positions.append([t, z, y, x, ch])

                    # Color based on channel
                    if ch == 0:
                        colors.append([255, 50, 50])  # Red
                    elif ch == 1:
                        colors.append([50, 255, 50])  # Green
                    else:
                        colors.append([50, 50, 255])  # Blue

                    # Radius varies with z depth (smaller when deeper)
                    radius = 0.3 + 0.2 * (z + 50) / 100
                    radii.append(radius)

    # Convert to arrays
    positions = np.array(positions, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    radii = np.array(radii, dtype=np.float32)

    aprint(f"Created {len(positions)} points across dimensions:")
    aprint(f"  - Time: {len(time_samples)} samples at {time_samples}")
    aprint(f"  - Z: {len(z_samples)} depths at {z_samples}")
    aprint(f"  - Channels: {len(channels)} channels")

    # Add points - dimensions are validated automatically
    scene.add_points("TestPoints", positions, colors=colors, radii=radii)

    # The scene dimensions are stored and will be used by the viewer
    scene.finalize()

    aprint(f"\n✓ Scene created at {scene.get_store_path()}")
    aprint("\n" + "=" * 70)
    aprint("SCENE DIMENSIONS TEST")
    aprint("=" * 70)
    aprint("\nDimension Configuration:")
    for i, dim in enumerate(dims.dimensions):
        status = "displayed" if dim.display else "hidden"
        aprint(
            f"  {i + 1}. {dim.name} ({dim.unit}): "
            f"range={dim.range}, step={dim.get_step():.2f}, {status}"
        )

    aprint("\nNavigation Instructions:")
    aprint("1. Start server: luxar serve scene_dims_test.zarr")
    aprint("2. Open viewer in browser")
    aprint("\nKeyboard controls:")
    aprint("  - Press '1' to control time (steps of 0.5s)")
    aprint("  - Press '2' to control z depth (steps of 0.1um)")
    aprint("  - Use '[' and ']' to navigate")
    aprint("\nExpected behavior:")
    aprint("  - Time navigation jumps by 0.5s increments")
    aprint("  - Z navigation moves smoothly by 0.1um")
    aprint("  - Points get smaller at deeper z values")
    aprint("  - Three color channels visible (R, G, B)")
    aprint("=" * 70)


if __name__ == "__main__":
    main()
