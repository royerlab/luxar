#!/usr/bin/env python3
"""Broadcasting Clarification Example - Shows exactly how broadcasting works."""

from pathlib import Path

import numpy as np
from arbol import aprint

import luxar
from luxar import LuxarZarrCompiler

scene_path = Path(__file__).parent / "broadcast_clarification_example.zarr"

dimensions = luxar.Dimensions(
    [
        luxar.Dimension(name="X", unit="μm", range=(-10, 10), display=True),
        luxar.Dimension(name="Y", unit="μm", range=(-10, 10), display=True),
        luxar.Dimension(name="Z", unit="μm", range=(-10, 10), display=True),
        luxar.Dimension(
            name="Time",
            unit="frame",
            range=(0, 2),
            display=False,
            step=1.0,
            discrete=True,
        ),
    ]
)

with LuxarZarrCompiler(scene_path) as compiler:
    scene = compiler.create_scene(dimensions=dimensions)

    # IMPORTANT: Position arrays ALWAYS include ALL dimensions!
    # Broadcasting doesn't mean "skip dimensions" - it means "replicate across dimensions"

    # Example 1: WITHOUT broadcasting
    # We manually create points for each time
    aprint("Creating points WITHOUT broadcasting (manual replication):")
    points_manual = []
    for t in range(3):  # Time 0, 1, 2
        points_manual.append([0, 0, 0, t])  # Center point at each time
        points_manual.append([2, 0, 0, t])  # Right point at each time
        points_manual.append([-2, 0, 0, t])  # Left point at each time

    points_manual = np.array(points_manual, dtype=np.float32)
    aprint(f"  - Created {len(points_manual)} points (3 points × 3 times)")
    aprint(f"  - Position array shape: {points_manual.shape}")
    aprint(f"  - Memory used: {points_manual.nbytes} bytes")

    scene.add_points(
        "ManualReplication",
        points_manual,
        colors=[1.0, 0.0, 0.0],  # Red
        radii=0.5,
        # No broadcasting - we manually created all points
    )

    # Example 2: WITH broadcasting
    # We create points only once, at Time=0
    aprint("\nCreating points WITH broadcasting (memory efficient):")
    points_broadcast = np.array(
        [
            [0, 5, 0, 0],  # Center point at Time=0 ONLY
            [2, 5, 0, 0],  # Right point at Time=0 ONLY
            [-2, 5, 0, 0],  # Left point at Time=0 ONLY
        ],
        dtype=np.float32,
    )

    aprint(f"  - Created {len(points_broadcast)} points (only at Time=0)")
    aprint(f"  - Position array shape: {points_broadcast.shape}")
    aprint(f"  - Memory used: {points_broadcast.nbytes} bytes")
    aprint("  - But will appear at Time=0, 1, and 2 due to broadcasting!")

    scene.add_points(
        "BroadcastReplication",
        points_broadcast,
        colors=[0.0, 1.0, 0.0],  # Green
        radii=0.5,
        broadcast_dims=["Time"],  # These 3 points will appear at ALL times
    )

    # Example 3: Partial broadcasting
    # Some points at specific times
    aprint("\nCreating points with partial coverage:")
    points_partial = np.array(
        [
            [0, -5, 0, 0],  # Only at Time=0
            [2, -5, 0, 1],  # Only at Time=1
            # Note: No points at Time=2
        ],
        dtype=np.float32,
    )

    aprint(f"  - Created {len(points_partial)} points")
    aprint("  - Time=0: 1 point, Time=1: 1 point, Time=2: 0 points")

    scene.add_points(
        "PartialCoverage",
        points_partial,
        colors=[0.0, 0.0, 1.0],  # Blue
        radii=0.5,
        # No broadcasting - points only appear at their defined times
    )

aprint(f"\n✓ Clarification example created at {scene_path}")
aprint("\nKey Points:")
aprint("1. Position arrays ALWAYS have ALL dimensions (X, Y, Z, Time, etc.)")
aprint(
    "2. Broadcasting means 'show these points at all values of specified dimensions'"
)
aprint("3. WITHOUT broadcasting: Points only appear at their defined dimension values")
aprint("4. WITH broadcasting: Points appear at ALL values of broadcast dimensions")
aprint("\nMemory savings:")
aprint(f"- Manual replication: {9 * 4 * 4} bytes (9 points × 4 coords × 4 bytes)")
aprint(f"- With broadcasting: {3 * 4 * 4} bytes (3 points × 4 coords × 4 bytes)")
aprint(f"- Saved: {(9 - 3) * 4 * 4} bytes (66% reduction!)")
aprint("\nViewing instructions:")
aprint("1. luxar serve broadcast_clarification_example.zarr")
aprint("2. Press '4' to select Time dimension")
aprint("3. Press '[' and ']' to navigate through time")
aprint("4. Red points: Appear at all times (manually replicated)")
aprint("5. Green points: Appear at all times (broadcast from Time=0)")
aprint("6. Blue points: Only at Time=0 and Time=1")
