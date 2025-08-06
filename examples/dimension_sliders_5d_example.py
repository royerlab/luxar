#!/usr/bin/env python3
"""Test dimension sliders with 5D data."""

import numpy as np
from pathlib import Path
import luxar
from luxar import Scene, transforms

# Create output directory
output_dir = Path("zarr_scenes")
output_dir.mkdir(exist_ok=True)

# Create a 5D scene (X, Y, Z, Time, Channel)
scene_path = output_dir / "test_5d_sliders.zarr"
scene = Scene(
    scene_path,
    dimensions=luxar.Dimensions([
        luxar.Dimension(name="X", unit="μm", range=(-50, 50), display=True),
        luxar.Dimension(name="Y", unit="μm", range=(-50, 50), display=True),
        luxar.Dimension(name="Z", unit="μm", range=(-50, 50), display=True),
        luxar.Dimension(name="Time", unit="s", range=(0, 10), display=False, step=1.0),
        luxar.Dimension(name="Channel", unit="", range=(0, 2), display=False, discrete=True, step=1.0),
    ])
)

# Create some test data - moving points over time and channels
n_time_points = 11  # 0 to 10 seconds
n_channels = 3  # 0, 1, 2
n_points_per_frame = 1000

# Generate base positions (spiral)
theta = np.linspace(0, 4 * np.pi, n_points_per_frame)
base_radius = np.linspace(10, 40, n_points_per_frame)

all_positions = []
all_colors = []
all_radii = []

# Channel colors
channel_colors = [
    [1.0, 0.2, 0.2],  # Red
    [0.2, 1.0, 0.2],  # Green  
    [0.2, 0.2, 1.0],  # Blue
]

for t in range(n_time_points):
    for c in range(n_channels):
        # Animate the spiral over time
        time_offset = t * 0.2 * np.pi
        channel_offset = c * 2 * np.pi / 3
        
        x = base_radius * np.cos(theta + time_offset + channel_offset)
        y = base_radius * np.sin(theta + time_offset + channel_offset)
        z = np.linspace(-30, 30, n_points_per_frame) + 5 * np.sin(time_offset)
        
        # Create 5D positions (X, Y, Z, Time, Channel)
        positions = np.zeros((n_points_per_frame, 5), dtype=np.float32)
        positions[:, 0] = x
        positions[:, 1] = y
        positions[:, 2] = z
        positions[:, 3] = t  # Time coordinate
        positions[:, 4] = c  # Channel coordinate
        
        all_positions.append(positions)
        
        # Colors based on channel
        colors = np.tile(channel_colors[c], (n_points_per_frame, 1))
        all_colors.append(colors)
        
        # Radii - smaller for higher channels
        radii = np.full(n_points_per_frame, 2.0 - c * 0.5, dtype=np.float32)
        all_radii.append(radii)

# Combine all data
positions = np.vstack(all_positions)
colors = np.vstack(all_colors)
radii = np.concatenate(all_radii)

# Add points to scene
points = scene.add_points(
    "animated_spiral",
    positions=positions,
    colors=colors,
    radii=radii,
    sharpness=np.full(len(positions), 2.0, dtype=np.float32),
    parent=None,
    transform=transforms.identity(),
    opacity=0.9,
    gamma=1.0,
    blending_mode="normal"
)

# Add info text
info_text = f"""
5D Dimension Slider Test
========================
Total dimensions: 5 (X, Y, Z, Time, Channel)
Displayed: X, Y, Z
Sliders: Time (0-10s), Channel (0-2)

Points: {len(positions)} total
- {n_time_points} time points
- {n_channels} channels  
- {n_points_per_frame} points per frame

Use the sliders or keyboard controls:
- Press 1 to control Time dimension
- Press 2 to control Channel dimension
- Press [ / ] to navigate selected dimension
- Drag sliders for smooth navigation
"""

# Add metadata  
scene.attrs["info"] = info_text

# Finalize
scene.finalize()

print(f"Created 5D test scene: {scene_path}")
print(f"Total points: {len(positions)}")
print("\nDimensions:")
for i, dim in enumerate(scene.dimensions.dimensions):
    displayed = "displayed" if dim.display else "slider"
    print(f"  {i}: {dim.name} ({dim.unit}) - {displayed}")
print(f"\nOpen in viewer: python -m http.server 5801")
print(f"Then navigate to: http://localhost:5801/?src=zarr_scenes/test_5d_sliders.zarr")