#!/usr/bin/env python3
"""Simple nD Example - Basic 5D grid dataset for learning dimension navigation.

This example demonstrates:
- Creating a 5D dataset with clear visual structure
- Navigating through non-displayed dimensions (time and depth)
- Using color and intensity to indicate position in nD space
- Proper dimension metadata configuration
- Grid-based layouts for easy orientation

Educational value:
- Good first example for learning nD dimension navigation
- Understand how non-displayed dimensions work as sliders
- See visual patterns that confirm correct slice positions
- Foundation for more complex nD examples
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_5d_grid(
    n_times: int = 5, n_depths: int = 4, n_channels: int = 3, grid_size: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Create a 5D grid dataset with distinct visual patterns.

    Args:
        n_times: Number of time points
        n_depths: Number of depth layers
        n_channels: Number of channels (displayed as separate grids)
        grid_size: Size of the x,y grid

    Returns:
        Tuple of (positions, colors) arrays
    """
    positions = []
    colors = []

    # Create distinct patterns for each time/depth combination
    for t in range(n_times):
        for d in range(n_depths):
            for c in range(n_channels):
                # Create a different pattern for each time point
                for y in range(grid_size):
                    for x in range(grid_size):
                        # Apply pattern based on time
                        if t == 0:
                            # Full grid
                            include = True
                        elif t == 1:
                            # Checkerboard pattern
                            include = (x + y) % 2 == 0
                        elif t == 2:
                            # Diagonal lines
                            include = abs(x - y) <= 1
                        elif t == 3:
                            # Border only
                            include = (
                                x == 0
                                or x == grid_size - 1
                                or y == 0
                                or y == grid_size - 1
                            )
                        else:
                            # Center cross
                            include = x == grid_size // 2 or y == grid_size // 2

                        if not include:
                            continue

                        # Position in 5D space [time, depth, y, x, channel]
                        pos = [
                            t,  # time dimension
                            d,  # depth dimension
                            y - grid_size / 2,  # y (centered)
                            x - grid_size / 2,  # x (centered)
                            c * 6 - 6,  # channel (spread out: -6, 0, 6)
                        ]
                        positions.append(pos)

                        # Base color for each channel
                        base_colors = {
                            0: [1.0, 0.2, 0.2],  # Red
                            1: [0.2, 1.0, 0.2],  # Green
                            2: [0.2, 0.2, 1.0],  # Blue
                        }
                        color = base_colors[c].copy()

                        # Modulate intensity based on time and depth
                        # Earlier times are brighter, deeper slices are dimmer
                        time_fade = 1.0 - t * 0.15
                        depth_fade = 1.0 - d * 0.2
                        intensity = time_fade * depth_fade

                        color = [c * intensity for c in color]
                        colors.append(color)

    return np.array(positions, dtype=np.float32), np.array(colors, dtype=np.float32)


def main():
    """Create a simple nD example for learning dimension navigation."""
    output_path = get_examples_output_dir() / "simple_nd_example.zarr"
    aprint(f"Creating simple nD example at {output_path}")

    # Dataset parameters
    n_times = 5
    n_depths = 4
    n_channels = 3
    grid_size = 10

    # Define scene dimensions
    dimensions = Dimensions(
        [
            Dimension(
                "time",
                unit="frame",
                range=(0, n_times - 1),
                step=1,
                display=False,
                discrete=True,
                description="Time progression",
            ),
            Dimension(
                "depth",
                unit="layer",
                range=(0, n_depths - 1),
                step=1,
                display=False,
                discrete=True,
                description="Depth layers",
            ),
            Dimension("y", unit="units", range=(-6, 6)),
            Dimension("x", unit="units", range=(-6, 6)),
            Dimension("channel", unit="ch", range=(-8, 8), display=True),
        ]
    )

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=dimensions)

        # Create the 5D grid data
        aprint("Generating 5D grid dataset...")
        positions, colors = create_5d_grid(n_times, n_depths, n_channels, grid_size)

        # Add consistent radius for all points
        radii = np.full(len(positions), 0.25, dtype=np.float32)

        # Add points to scene
        scene.add_points(
            "Grid5D",
            positions,
            colors=colors,
            radii=radii,
            sharpness=np.full(len(positions), 2.0, dtype=np.float32),
        )

        add_explainer(
            scene,
            title="nD Navigation Basics",
            body=(
                "A 5D grid where the hidden <code>time</code> and "
                "<code>depth</code> dimensions act as sliders; the displayed "
                "slice is a flat grid of coloured points."
            ),
            observe=[
                "Press <code>1</code> for time, then <code>[</code> / "
                "<code>]</code> to step through patterns.",
                "Time steps: full grid, checkerboard, diagonals, border, centre cross.",
                "Press <code>2</code> for depth: deeper layers are dimmer.",
                "Three coloured groups left-to-right: red, green, blue channels.",
            ],
        )

        # Print summary and instructions
        aprint(f"\n✓ Created 5D grid with {len(positions):,} points")
        aprint(
            f"  Dimensions: {n_times} times × {n_depths} depths × {n_channels} channels"
        )
        aprint(f"  Grid size: {grid_size}×{grid_size}")

        aprint("\n" + "=" * 60)
        aprint("NAVIGATION INSTRUCTIONS")
        aprint("=" * 60)
        aprint("1. Start the server:")
        aprint(f"   luxar serve {output_path}")

        aprint("\n2. Dimension controls:")
        aprint("   - Press '1' to select TIME dimension")
        aprint("   - Press '2' to select DEPTH dimension")
        aprint("   - Press '[' to go backward, ']' to go forward")

        aprint("\n3. Visual patterns by time:")
        aprint("   - Time 0: Full grid (all points)")
        aprint("   - Time 1: Checkerboard pattern")
        aprint("   - Time 2: Diagonal lines")
        aprint("   - Time 3: Border outline only")
        aprint("   - Time 4: Center cross")

        aprint("\n4. Visual changes by depth:")
        aprint("   - Depth 0: Brightest colors")
        aprint("   - Depth 1-3: Progressively dimmer")

        aprint("\n5. Channel layout:")
        aprint("   - Left: Red channel")
        aprint("   - Center: Green channel")
        aprint("   - Right: Blue channel")
        aprint("=" * 60)


if __name__ == "__main__":
    main()
