#!/usr/bin/env python3
"""Scene Dimensions Example - Comprehensive dimension configuration.

This example demonstrates:
- Defining scene-level dimensions with custom properties
- Setting step sizes for keyboard navigation
- Configuring units and physical ranges
- Controlling which dimensions are displayed vs navigable
- Dimension validation for all objects in the scene

Educational value:
- Learn proper dimension configuration for nD scenes
- Understand displayed vs non-displayed dimensions
- Master step size configuration for navigation
- See how dimensions provide automatic validation

Key principle:
- Scene dimensions define the coordinate system for ALL objects
- Step sizes control keyboard navigation granularity
- Non-displayed dimensions must be discrete
- Dimensions enable automatic range validation
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def main():
    """Create a scene with explicit dimension definitions."""
    aprint("Creating scene with explicit dimensions...")

    # Define scene dimensions with custom stepping
    # IMPORTANT: Ranges must match where data actually exists
    dims = Dimensions(
        [
            # Time dimension with 0.5s steps
            Dimension("time", unit="s", range=(0, 10), step=0.5, display=False),
            # Z dimension with fine 0.1um steps - range matches actual data
            Dimension("z", unit="um", range=(-40, 40), step=0.1, display=False),
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
    output_path = get_examples_output_dir() / "scene_dimensions_example.zarr"
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=dims)

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
                            colors.append([1.0, 0.2, 0.2])  # Red
                        elif ch == 1:
                            colors.append([0.2, 1.0, 0.2])  # Green
                        else:
                            colors.append([0.2, 0.2, 1.0])  # Blue

                        # Radius varies with z depth (smaller when deeper)
                        radius = 0.3 + 0.2 * (z + 50) / 100
                        radii.append(radius)

        # Convert to arrays
        positions = np.array(positions, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        radii = np.array(radii, dtype=np.float32)

        aprint(f"Created {len(positions)} points across dimensions:")
        aprint(f"  - Time: {len(time_samples)} samples at {time_samples}")
        aprint(f"  - Z: {len(z_samples)} depths at {z_samples}")
        aprint(f"  - Channels: {len(channels)} channels")

        # Add points - dimensions are validated automatically
        scene.add_points("TestPoints", positions, colors=colors, radii=radii)

        # The scene dimensions are stored and will be used by the viewer

        add_explainer(
            scene,
            title="Scene Dimensions",
            body=(
                "Scene-level <code>Dimension</code>s define the coordinate "
                "system, units, ranges, and per-dimension <code>step</code> "
                "used for keyboard navigation. Here <strong>time</strong> and "
                "<strong>z</strong> are non-displayed (navigable) while "
                "<strong>x/y</strong> and <strong>channel</strong> are shown."
            ),
            observe=[
                "Press <code>1</code> then <code>[</code>/<code>]</code>: time "
                "steps by 0.5 s.",
                "Press <code>2</code>: z navigates smoothly in 0.1 um steps.",
                "Points shrink at deeper z values.",
                "Three colour channels are visible (R, G, B).",
            ],
            observe_label="Verify",
        )

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
        aprint("1. Start server: luxar serve scene_dimensions_example.zarr")
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
