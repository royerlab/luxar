#!/usr/bin/env python3
"""Test script to verify the new data loader architecture works correctly."""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import Dimension, Dimensions, LuxarZarrCompiler

# Create a simple test scene with spatial index
output_path = "test_new_loader_example.zarr"

# Clean up existing file if present

if Path(output_path).exists():
    import shutil

    shutil.rmtree(output_path)

# Create some 5D points (x, y, z, time, channel)
n_points = 10000
n_dims = 5

# Create structured data that should be easy to verify
positions = np.zeros((n_points, n_dims), dtype=np.float32)

# Fill spatial dimensions (0, 1, 2) with random positions
positions[:, :3] = np.random.randn(n_points, 3) * 10

# Fill time dimension (3) with discrete time steps
time_steps = 5
for i in range(n_points):
    positions[i, 3] = (i // (n_points // time_steps)) * 2.0  # 0, 2, 4, 6, 8

# Fill channel dimension (4) with discrete channels
channels = 3
for i in range(n_points):
    positions[i, 4] = i % channels  # 0, 1, 2, 0, 1, 2, ...

# Create colors that depend on time and channel
# This will help verify attribute alignment
colors = np.zeros((n_points, 3), dtype=np.float32)
for i in range(n_points):
    time_idx = int(positions[i, 3] / 2)
    channel_idx = int(positions[i, 4])

    # Color based on time (red component) and channel (green/blue)
    colors[i, 0] = (time_idx / (time_steps - 1)) * 255  # Red increases with time
    colors[i, 1] = (
        (channel_idx / (channels - 1)) * 255 if channels > 1 else 0
    )  # Green for channel
    colors[i, 2] = (1.0 - time_idx / (time_steps - 1)) * 255  # Blue decreases with time

# Add radii that vary with position
radii = np.ones(n_points, dtype=np.float32) * 0.5
# Make some points larger based on their time coordinate
radii[positions[:, 3] > 4] = 1.5

# Define dimensions
dims = Dimensions(
    [
        Dimension("x", unit="μm", display=True, range=(-30, 30), step=1.0),
        Dimension("y", unit="μm", display=True, range=(-30, 30), step=1.0),
        Dimension("z", unit="μm", display=True, range=(-30, 30), step=1.0),
        Dimension(
            "time", unit="s", display=False, range=(0, 8), step=2.0, discrete=True
        ),
        Dimension(
            "channel", unit="", display=False, range=(0, 2), step=1.0, discrete=True
        ),
    ]
)

aprint("Creating scene with spatial index...")

# Create scene with spatial index
with LuxarZarrCompiler(
    output_path,
    enable_spatial_index=True,  # Enable spatial indexing
) as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Add points with spatial index
    scene.add_points(
        "test_points",
        positions,
        colors=colors,
        radii=radii,
        grid_shape=(5, 5, 5, 3, 3),  # Grid resolution for spatial index
        opacity=0.9,
    )

aprint(f"\nTest dataset created: {output_path}")
aprint(f"Points: {n_points}")
aprint(f"Dimensions: {n_dims} (x, y, z, time, channel)")
aprint(f"Time steps: {time_steps} (at t=0, 2, 4, 6, 8)")
aprint(f"Channels: {channels} (0, 1, 2)")
aprint("\nColors are coded by time (red) and channel (green)")
aprint("Radii are larger for t>4")
aprint("\nTo test:")
aprint(f"1. Run: luxar serve {output_path}")
aprint("2. Navigate through time dimension (should see color change)")
aprint("3. Navigate through channel dimension (should see color change)")
aprint("4. Verify that colors match points (alignment test)")
aprint("\nNavigation:")
aprint("  - Use [ ] keys to navigate through time")
aprint("  - Press 4 then [ ] to navigate channels")
