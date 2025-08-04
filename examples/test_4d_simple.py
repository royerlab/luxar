#!/usr/bin/env python
"""Simple 4D test case - time series of 3D points."""

import numpy as np
from luxar import Scene
from luxar.types import DimensionMetadata

# Create simple 4D data (time + xyz)
n_times = 5
n_points = 100

# Create 4D positions: [t, x, y, z]
positions = np.zeros((n_times * n_points, 4), dtype=np.float32)

for t in range(n_times):
    for i in range(n_points):
        idx = t * n_points + i
        positions[idx] = [
            t,                                    # time
            np.cos(i * 0.1 + t) * 10,           # x - rotating
            np.sin(i * 0.1 + t) * 10,           # y - rotating
            i * 0.5 - 25                         # z - vertical spread
        ]

# Create scene
scene = Scene("test_4d.zarr")

# Add dimension metadata
metadata = [
    DimensionMetadata(name="t", unit="s", scale=1.0),
    DimensionMetadata(name="x", unit="um", scale=1.0),
    DimensionMetadata(name="y", unit="um", scale=1.0),
    DimensionMetadata(name="z", unit="um", scale=1.0),
]

# Simple colors
colors = np.full((n_times * n_points, 3), 255, dtype=np.uint8)

scene.add_points("TimeSeries4D", positions, colors=colors, dimension_metadata=metadata)
scene.finalize()

print(f"Created 4D test at test_4d.zarr")
print(f"Total points: {n_times * n_points}")
print(f"Dimensions: time + xyz")