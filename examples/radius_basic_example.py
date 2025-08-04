#!/usr/bin/env python3
"""
Basic example demonstrating the radius feature in Luxar.

This simple example creates three rows of points with different radii to clearly
show how the radius parameter affects point size visualization.
"""

import numpy as np
from pathlib import Path
from arbol import aprint
from luxar import Scene


def main():
    """Create a simple test scene with three groups of different-sized points."""
    output_path = Path("radius_basic_example.zarr")
    
    aprint(f"Creating radius test scene at {output_path}")
    
    scene = Scene(output_path)
    
    # Create three rows of points with different sizes
    n_points = 10
    y_positions = [-2, 0, 2]
    radii_values = [0.05, 0.2, 0.5]
    colors_rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    labels = ["Small", "Medium", "Large"]
    
    for i, (y, radius, color, label) in enumerate(zip(y_positions, radii_values, colors_rgb, labels)):
        # Create a line of points
        x = np.linspace(-5, 5, n_points)
        y = np.full(n_points, y)
        z = np.zeros(n_points)
        
        positions = np.column_stack([x, y, z]).astype(np.float32)
        radii = np.full(n_points, radius, dtype=np.float32)
        colors = np.tile(color, (n_points, 1)).astype(np.uint8)
        
        scene.add_points(f"{label}Points", positions, colors, radii=radii)
        aprint(f"Added {label} points with radius {radius}")
    
    scene.finalize()
    
    aprint(f"\n✓ Test scene created successfully!")
    aprint(f"\nExpected result when viewing:")
    aprint("- Top row: Large blue points (radius=0.5)")
    aprint("- Middle row: Medium green points (radius=0.2)")
    aprint("- Bottom row: Small red points (radius=0.05)")
    aprint("\nTo view: luxar serve {output_path}")


if __name__ == "__main__":
    main()