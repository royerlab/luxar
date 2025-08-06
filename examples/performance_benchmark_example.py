"""Performance test with many nodes and unique material combinations."""

import numpy as np
from luxar import Scene
import time

# Create scene
scene = Scene("delme/test_performance.zarr")

# Test parameters
num_nodes = 100
points_per_node = 1000

print(f"Creating scene with {num_nodes} nodes, {points_per_node} points each")
print(f"Total points: {num_nodes * points_per_node:,}")

start_time = time.time()

# Create nodes with various material combinations
blending_modes = ["normal", "additive", "multiply"]
opacities = [0.3, 0.5, 0.7, 0.9, 1.0]
gammas = [0.8, 1.0, 1.2, 1.5]

material_combinations = set()

for i in range(num_nodes):
    # Create varied positions
    positions = np.random.randn(points_per_node, 3) * 0.5
    positions += np.array([
        (i % 10) * 3,
        ((i // 10) % 10) * 3,
        (i // 100) * 3
    ])
    
    # Cycle through different rendering properties
    blending = blending_modes[i % len(blending_modes)]
    opacity = opacities[i % len(opacities)]
    gamma = gammas[i % len(gammas)]
    
    # Track unique material combinations
    material_combinations.add((blending, round(opacity, 2), round(gamma, 2)))
    
    # Random colors
    colors = np.random.randint(50, 255, (points_per_node, 3), dtype=np.uint8)
    
    scene.add_points(
        f"node_{i:03d}",
        positions,
        colors=colors,
        blending_mode=blending,
        opacity=opacity,
        gamma=gamma
    )

end_time = time.time()

print(f"\nScene creation time: {end_time - start_time:.2f} seconds")
print(f"Unique material combinations: {len(material_combinations)}")
print("\nMaterial cache should contain approximately this many entries")
print("(actual cache may be smaller due to rounding)")
print("\nRun 'luxar serve delme/test_performance.zarr' to test rendering performance")