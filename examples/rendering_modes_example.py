"""Test full rendering pipeline with different attributes."""

import numpy as np
from luxar import Scene

# Create a scene with multiple layers showing different rendering modes
scene = Scene("delme/test_full_rendering.zarr")

# Create positions for different point clouds
n_points = 10000
base_positions = np.random.randn(n_points, 3) * 2

# 1. Normal blending - opaque red sphere
positions1 = base_positions + np.array([0, 0, 0])
colors1 = np.full((n_points, 3), [255, 50, 50], dtype=np.uint8)  # Red
scene.add_points(
    "normal_opaque", 
    positions1, 
    colors=colors1,
    blending_mode="normal",
    opacity=1.0,
    gamma=1.0
)

# 2. Normal blending - transparent green sphere
positions2 = base_positions + np.array([3, 0, 0])
colors2 = np.full((n_points, 3), [50, 255, 50], dtype=np.uint8)  # Green
scene.add_points(
    "normal_transparent", 
    positions2, 
    colors=colors2,
    blending_mode="normal",
    opacity=0.5,
    gamma=1.0
)

# 3. Additive blending - blue glowing sphere
positions3 = base_positions + np.array([0, 3, 0])
colors3 = np.full((n_points, 3), [50, 50, 255], dtype=np.uint8)  # Blue
scene.add_points(
    "additive_glow", 
    positions3, 
    colors=colors3,
    blending_mode="additive",
    opacity=0.8,
    gamma=1.2  # Slightly brighter
)

# 4. Multiply blending - dark purple sphere
positions4 = base_positions + np.array([-3, 0, 0])
colors4 = np.full((n_points, 3), [255, 50, 255], dtype=np.uint8)  # Purple
scene.add_points(
    "multiply_dark", 
    positions4, 
    colors=colors4,
    blending_mode="multiply",
    opacity=0.7,
    gamma=0.8  # Darker
)

# 5. Group with inherited properties
group = scene.add_group("transparent_group", opacity=0.3, blending_mode="additive")

# Child inherits parent's opacity and blending
positions5 = base_positions + np.array([0, -3, 0])
colors5 = np.full((n_points, 3), [255, 255, 50], dtype=np.uint8)  # Yellow
scene.add_points(
    "inherited_yellow", 
    positions5, 
    colors=colors5,
    parent=group
    # Should inherit opacity=0.3 and blending_mode="additive"
)

# Child overrides parent's gamma
positions6 = base_positions + np.array([0, -3, 3])
colors6 = np.full((n_points, 3), [50, 255, 255], dtype=np.uint8)  # Cyan
scene.add_points(
    "override_cyan", 
    positions6, 
    colors=colors6,
    parent=group,
    gamma=1.5  # Override parent's gamma
    # Should inherit opacity=0.3 and blending_mode="additive"
)

print("\nCreated test scene with multiple rendering modes:")
print("- Red: normal blending, opaque")
print("- Green: normal blending, 50% transparent")
print("- Blue: additive blending, glowing")
print("- Purple: multiply blending, darkening")
print("- Yellow/Cyan: inherited from transparent group")
print("\nRun 'luxar serve delme/test_full_rendering.zarr' to view")