#!/usr/bin/env python3
"""Broadcasting API Example - Demonstrates different broadcasting options."""

from pathlib import Path

import numpy as np
from arbol import aprint

import luxar
from luxar import LuxarZarrCompiler

# Create a 5D scene (X, Y, Z, Time, Channel)
scene_path = Path(__file__).parent / "broadcast_api_example.zarr"

dimensions = luxar.Dimensions(
    [
        luxar.Dimension(name="X", unit="μm", range=(-10, 10), display=True),
        luxar.Dimension(name="Y", unit="μm", range=(-10, 10), display=True),
        luxar.Dimension(name="Z", unit="μm", range=(-10, 10), display=True),
        luxar.Dimension(
            name="Time",
            unit="frame",
            range=(0, 4),
            display=False,
            step=1.0,
            discrete=True,
        ),
        luxar.Dimension(
            name="Channel",
            unit="",
            range=(0, 2),
            display=False,
            discrete=True,
            step=1.0,
        ),
    ]
)

with LuxarZarrCompiler(scene_path) as compiler:
    scene = compiler.create_scene(dimensions=dimensions)

    # Example 1: No broadcasting (default)
    # These points only appear at time=0, channel=0
    points_no_broadcast = np.array(
        [
            [-5, 0, 0, 0, 0],  # X, Y, Z, Time=0, Channel=0
            [-3, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    scene.add_points(
        "NoBroadcast",
        points_no_broadcast,
        colors=[1.0, 0.0, 0.0],  # Red
        radii=0.5,
        # broadcast_dims=None is the default - no broadcasting
    )

    # Example 2: Explicit broadcasting to Time only
    # These points appear at all times but only channel=1
    points_time_broadcast = np.array(
        [
            [0, -5, 0, 0, 1],  # Time=0, Channel=1 (but broadcasts to all times)
            [0, -3, 0, 0, 1],
            [0, -1, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    scene.add_points(
        "TimeBroadcast",
        points_time_broadcast,
        colors=[0.0, 1.0, 0.0],  # Green
        radii=0.5,
        broadcast_dims=["Time"],  # Explicit: broadcast to all times
    )

    # Example 3: Broadcast to all non-displayed dimensions
    # These points appear at all times AND all channels
    points_all_broadcast = np.array(
        [
            [5, 0, 0, 0, 0],  # Only defined once, but appears everywhere
            [3, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    scene.add_points(
        "AllBroadcast",
        points_all_broadcast,
        colors=[0.0, 0.0, 1.0],  # Blue
        radii=0.5,
        broadcast_dims="all",  # Broadcast to all non-displayed dimensions
    )

    # Example 4: Full coverage (no broadcasting needed)
    # Create points for specific time/channel combinations
    full_coverage_points = []
    for t in range(5):  # All 5 time points
        for c in range(3):  # All 3 channels
            # Create a point that moves over time
            x = -8 + t * 2
            y = 5
            z = -5 + c * 2
            full_coverage_points.append([x, y, z, t, c])

    scene.add_points(
        "FullCoverage",
        np.array(full_coverage_points, dtype=np.float32),
        colors=[1.0, 1.0, 0.0],  # Yellow
        radii=0.3,
        # No broadcasting needed - we have full coverage
    )

aprint(f"✓ Broadcasting example created at {scene_path}")
aprint("\nExpected behavior when viewing:")
aprint("- Red points (NoBroadcast): Only visible at Time=0, Channel=0")
aprint("- Green points (TimeBroadcast): Visible at all times but only Channel=1")
aprint("- Blue points (AllBroadcast): Visible at all times and all channels")
aprint("- Yellow points (FullCoverage): Different positions at each time/channel")
aprint("\nTest by navigating Time (press 4 then [/]) and Channel (press 5 then [/])")
