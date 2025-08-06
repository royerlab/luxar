#!/usr/bin/env python3
"""Test dimension sliders with dense 5D data grid."""

import numpy as np
from pathlib import Path
import luxar
from luxar import Scene, transforms

# Create output directory
output_dir = Path("zarr_scenes")
output_dir.mkdir(exist_ok=True)

# Create a 5D scene (X, Y, Z, Time, Channel)
scene_path = output_dir / "test_5d_dense_grid.zarr"
scene = Scene(
    scene_path,
    dimensions=luxar.Dimensions([
        luxar.Dimension(name="X", unit="μm", range=(-30, 30), display=True),
        luxar.Dimension(name="Y", unit="μm", range=(-30, 30), display=True),
        luxar.Dimension(name="Z", unit="μm", range=(-30, 30), display=True),
        luxar.Dimension(name="Time", unit="frame", range=(0, 9), display=False, step=1.0, discrete=True),
        luxar.Dimension(name="Channel", unit="", range=(0, 2), display=False, discrete=True, step=1.0),
    ])
)

# Create a dense 3D grid that changes over time and channels
# Grid spacing
grid_size = 10  # 10x10x10 grid
spacing = 4.0   # 4 μm between points

# Time and channel parameters
n_time_points = 10  # 0 to 9
n_channels = 3      # 0, 1, 2

# Channel colors
channel_colors = [
    [1.0, 0.3, 0.3],  # Red
    [0.3, 1.0, 0.3],  # Green  
    [0.3, 0.3, 1.0],  # Blue
]

all_positions = []
all_colors = []
all_radii = []

# Create grid for each time point and channel
for t in range(n_time_points):
    for c in range(n_channels):
        # Create 3D grid
        x = np.linspace(-spacing * (grid_size-1)/2, spacing * (grid_size-1)/2, grid_size)
        y = np.linspace(-spacing * (grid_size-1)/2, spacing * (grid_size-1)/2, grid_size)
        z = np.linspace(-spacing * (grid_size-1)/2, spacing * (grid_size-1)/2, grid_size)
        
        # Create meshgrid
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        # Flatten to get point positions
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        z_flat = zz.flatten()
        
        # Apply time-based transformation
        # Rotate around center based on time
        angle = t * 10  # degrees
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotate around Z axis
        x_rot = x_flat * cos_a - y_flat * sin_a
        y_rot = x_flat * sin_a + y_flat * cos_a
        z_rot = z_flat
        
        # Apply channel-based offset
        channel_offset = (c - 1) * 1.0  # Shift channels slightly in Z
        z_rot += channel_offset
        
        # Create 5D positions
        n_points = len(x_flat)
        positions = np.zeros((n_points, 5), dtype=np.float32)
        positions[:, 0] = x_rot
        positions[:, 1] = y_rot
        positions[:, 2] = z_rot
        positions[:, 3] = t  # Time coordinate
        positions[:, 4] = c  # Channel coordinate
        
        all_positions.append(positions)
        
        # Colors based on channel with gradient based on position
        colors = np.tile(channel_colors[c], (n_points, 1))
        # Add gradient based on height (z position)
        brightness = (z_flat - z_flat.min()) / (z_flat.max() - z_flat.min() + 1e-6)
        colors = colors * brightness[:, np.newaxis]
        all_colors.append(colors)
        
        # Radii - vary by time (growing/shrinking)
        base_radius = 2.5  # Increased from 1.5
        time_factor = 1.0 + 0.3 * np.sin(t * np.pi / 5)  # Oscillate
        radii = np.full(n_points, base_radius * time_factor, dtype=np.float32)
        all_radii.append(radii)

# Combine all data
positions = np.vstack(all_positions)
colors = np.vstack(all_colors)
radii = np.concatenate(all_radii)

# Add central marker points at each time/channel to help with orientation
marker_positions = []
marker_colors = []
marker_radii = []

for t in range(n_time_points):
    for c in range(n_channels):
        # Add a larger central marker
        marker_pos = np.array([[0, 0, (c-1)*1.0, t, c]], dtype=np.float32)
        marker_positions.append(marker_pos)
        
        # Make markers bright white
        marker_colors.append([[1.0, 1.0, 1.0]])
        
        # Larger radius for markers
        marker_radii.append([3.0])

# Add markers to main arrays
positions = np.vstack([positions] + marker_positions)
colors = np.vstack([colors] + marker_colors)
radii = np.concatenate([radii] + marker_radii)

# Add points to scene
points = scene.add_points(
    "dense_grid",
    positions=positions,
    colors=colors,
    radii=radii,
    sharpness=np.full(len(positions), 2.0, dtype=np.float32),
    parent=None,
    transform=transforms.identity(),
    opacity=1.0,
    gamma=1.0,
    blending_mode="additive"
)

# Add info text
info_text = f"""
5D Dense Grid Test
==================
Total dimensions: 5 (X, Y, Z, Time, Channel)
Displayed: X, Y, Z
Sliders: Time (0-9 frames), Channel (0-2)

Grid: {grid_size}x{grid_size}x{grid_size} = {grid_size**3} points per frame
Total points: {len(positions)} 
- {n_time_points} time frames
- {n_channels} channels (R/G/B)
- Grid rotates over time
- Point size oscillates
- White markers at center

Controls:
- Drag sliders to navigate Time and Channel
- Use [ / ] keys for fine control
- Press 1/2 to select dimension
"""

# Add metadata
scene.attrs["info"] = info_text

# Finalize
scene.finalize()

print(f"Created dense 5D test scene: {scene_path}")
print(f"Total points: {len(positions)}")
print("\nDimensions:")
for i, dim in enumerate(scene.dimensions.dimensions):
    displayed = "displayed" if dim.display else "slider"
    print(f"  {i}: {dim.name} ({dim.unit}) - {displayed}")
print(f"\nOpen in viewer with proper zarr path")