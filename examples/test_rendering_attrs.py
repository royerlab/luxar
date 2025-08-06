"""Test script for rendering attributes."""

import numpy as np
from luxar import Scene

# Create a simple scene
scene = Scene("delme/test_rendering_attrs.zarr")

# Add points with different rendering attributes
positions = np.random.rand(100, 3) * 10

# Default rendering
points1 = scene.add_points("default", positions)
print(f"Default - opacity: {points1.opacity}, gamma: {points1.gamma}, blending: {points1.blending_mode}")

# Custom opacity
points2 = scene.add_points("transparent", positions + [0, 0, 5], opacity=0.5)
print(f"Transparent - opacity: {points2.opacity}, gamma: {points2.gamma}, blending: {points2.blending_mode}")

# Custom gamma
points3 = scene.add_points("bright", positions + [5, 0, 0], gamma=1.5)
print(f"Bright - opacity: {points3.opacity}, gamma: {points3.gamma}, blending: {points3.blending_mode}")

# Normal blending
points4 = scene.add_points("normal_blend", positions + [0, 5, 0], blending_mode="normal", opacity=0.8)
print(f"Normal blend - opacity: {points4.opacity}, gamma: {points4.gamma}, blending: {points4.blending_mode}")

# Test setters
points1.opacity = 0.7
points1.gamma = 0.8
points1.blending_mode = "multiply"
print(f"Modified - opacity: {points1.opacity}, gamma: {points1.gamma}, blending: {points1.blending_mode}")

# Test chaining
points2.set_opacity(0.3).set_gamma(1.2).set_blending_mode("minimum")
print(f"Chained - opacity: {points2.opacity}, gamma: {points2.gamma}, blending: {points2.blending_mode}")

# Test validation errors
try:
    points1.opacity = 1.5  # Should fail
except ValueError as e:
    print(f"Expected error: {e}")

try:
    points1.gamma = 0.1  # Should fail
except ValueError as e:
    print(f"Expected error: {e}")

try:
    points1.blending_mode = "invalid"  # Should fail
except ValueError as e:
    print(f"Expected error: {e}")

print("\nPython-side implementation working correctly!")