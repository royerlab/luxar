#!/usr/bin/env python
"""Example: DOF Test Grid - Creates a grid of points at different depths for testing depth of field effects.

This example creates a large 2D grid of fine points positioned at different Z depths,
perfect for testing and tuning the depth of field (DOF) post-processing effect.
The points are arranged in a pattern that makes it easy to see which parts are in focus.
"""

import numpy as np
import zarr
from arbol import aprint

import luxar
from luxar import transforms as tf


def populate_dof_test_grid(scene):
    """Populate scene with a grid of points at different depths for DOF testing."""
    aprint("Creating DOF test grid scene...")
    
    # Set scene attributes for better initial view
    scene.attrs["camera_position"] = [0, 0, 50]  # Position camera back to see the grid
    scene.attrs["camera_target"] = [0, 0, 0]  # Look at origin
    scene.attrs["camera_fov"] = 60  # Field of view
    
    # Define dimensions for the point cloud
    scene.dimensions = luxar.Dimensions([
        luxar.Dimension("x", unit="px", range=(-60, 60)),
        luxar.Dimension("y", unit="px", range=(-60, 60)),
        luxar.Dimension("z", unit="px", range=(-25, 25))
    ])
    
    # Create multiple layers at different depths
    layers = []
    
    # Parameters for the grid
    grid_size = 80  # Points per side in each layer (reduced for better performance)
    spacing = 0.8    # Spacing between points
    num_layers = 8  # Number of depth layers
    depth_range = 30.0  # Total depth range (reduced to be more visible)
    
    # Generate points for each layer
    for layer_idx in range(num_layers):
        # Calculate Z position for this layer
        z_pos = -depth_range/2 + (layer_idx * depth_range / (num_layers - 1))
        
        # Create 2D grid for this layer
        x = np.linspace(-grid_size/2 * spacing, grid_size/2 * spacing, grid_size)
        y = np.linspace(-grid_size/2 * spacing, grid_size/2 * spacing, grid_size)
        xx, yy = np.meshgrid(x, y)
        
        # Flatten to create point positions
        positions = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            np.full(xx.size, z_pos)
        ])
        
        # Create BRIGHT colors that vary by depth (rainbow gradient)
        hue = layer_idx / (num_layers - 1)  # 0 to 1
        
        # Convert HSV to RGB (simplified rainbow) - using full brightness
        if hue < 1/6:
            r, g, b = 1, hue * 6, 0
        elif hue < 2/6:
            r, g, b = (2/6 - hue) * 6, 1, 0
        elif hue < 3/6:
            r, g, b = 0, 1, (hue - 2/6) * 6
        elif hue < 4/6:
            r, g, b = 0, (4/6 - hue) * 6, 1
        elif hue < 5/6:
            r, g, b = (hue - 4/6) * 6, 0, 1
        else:
            r, g, b = 1, 0, (1 - hue) * 6
        
        # HDR COLORS - pump up the values way beyond 1.0!
        # We're in HDR space, so we can use values like 10.0 for super bright points
        hdr_multiplier = 10.0  # 10x brightness in HDR space
        colors = np.tile([r * hdr_multiplier, g * hdr_multiplier, b * hdr_multiplier], (positions.shape[0], 1))
        
        # Don't clip! We want the full HDR brightness
        # HDR values can and should exceed 1.0 for bright emission
        
        # Create smaller, sharper points for DOF testing
        radii = np.full(positions.shape[0], 0.4)  # Smaller radius for sharper points
        
        # Uniform size across depths for consistent DOF testing
        # No perspective scaling - we want uniform points
        
        layers.append({
            'positions': positions,
            'colors': colors,
            'radii': radii,
            'z_pos': z_pos,
            'layer_idx': layer_idx
        })
    
    # Combine all layers
    all_positions = np.vstack([layer['positions'] for layer in layers])
    all_colors = np.vstack([layer['colors'] for layer in layers])
    all_radii = np.hstack([layer['radii'] for layer in layers])
    
    aprint(f"Total points: {len(all_positions):,}")
    aprint(f"Depth layers: {num_layers}")
    aprint(f"Points per layer: {grid_size * grid_size:,}")
    
    # Add the main point cloud to scene
    scene.add_points(
        "test_grid",
        positions=all_positions,
        colors=all_colors,
        radii=all_radii,
        sharpness=5.0  # Higher sharpness = sharper, more defined points for DOF testing
    )
    
    # Add some reference markers at specific depths
    marker_positions = []
    marker_colors = []
    marker_radii = []
    
    # Create larger marker points at key depths
    for i, z in enumerate([-12, -8, -4, 0, 4, 8, 12]):
        # Create a cross pattern at each depth
        for dx, dy in [(0, 0), (4, 0), (-4, 0), (0, 4), (0, -4)]:
            marker_positions.append([dx, dy + 35, z])
            marker_colors.append([1, 1, 1])  # White markers
            marker_radii.append(1.0)  # Smaller, sharper markers
    
    if marker_positions:
        scene.add_points(
            "depth_markers",
            positions=np.array(marker_positions),
            colors=np.ones_like(marker_colors) * 15.0,  # SUPER bright white in HDR
            radii=np.array(marker_radii),
            sharpness=5.0  # Sharp markers for clear visibility
        )
    
    # Add text labels as separate point clouds forming numbers
    def create_digit_points(digit, x_offset, y_offset, z_pos):
        """Create points forming a digit at specified position."""
        # Simple 5x7 dot matrix patterns for digits
        patterns = {
            '0': "01110100011000110001100011000101110",
            '1': "00100011000010000100001000010001110",
            '2': "01110100010000100010001000100011111",
            '3': "01110100010000100110000011000101110",
            '4': "00010001100101010010111110001000010",
            '5': "11111100001111000001000011000101110",
            '6': "00110010001000011110100011000101110",
            '7': "11111000010001000100010001000100010",
            '8': "01110100011000101110100011000101110",
            '9': "01110100011000101111000010001001100",
            '-': "00000000000000011111000000000000000",
        }
        
        points = []
        pattern = patterns.get(str(digit), patterns['0'])
        
        for i, char in enumerate(pattern):
            if char == '1':
                row = i // 5
                col = i % 5
                points.append([
                    x_offset + col * 0.8,
                    y_offset - row * 1.0,
                    z_pos
                ])
        
        return points
    
    # Add depth labels
    label_positions = []
    label_colors = []
    label_radii = []
    
    for z in [-12, -8, -4, 0, 4, 8, 12]:
        z_str = str(int(z))
        x_start = -40
        
        for i, digit in enumerate(z_str):
            digit_points = create_digit_points(digit, x_start + i * 6, 40, z)
            label_positions.extend(digit_points)
            label_colors.extend([[1, 1, 0]] * len(digit_points))  # Bright yellow
            label_radii.extend([0.4] * len(digit_points))  # Smaller, sharper labels
    
    if label_positions:
        scene.add_points(
            "depth_labels",
            positions=np.array(label_positions),
            colors=np.ones((len(label_positions), 3)) * 15.0,  # SUPER bright HDR white
            radii=np.array(label_radii),
            sharpness=5.0  # Sharp labels
        )


def main():
    """Main function to generate and save the DOF test scene."""
    # The scene will be saved at this path
    output_path = "test_dof_grid_example.zarr"
    
    # Create scene at the specified path
    scene = luxar.Scene(store_path=output_path)
    
    # Populate the scene
    populate_dof_test_grid(scene)
    
    # Consolidate metadata for efficient loading
    zarr.consolidate_metadata(output_path)
    aprint("✓ Metadata consolidated for efficient loading")
    
    aprint(f"\n✅ DOF test grid saved to: {output_path}")
    aprint("\n" + "="*60)
    aprint("📝 Testing Instructions:")
    aprint("="*60)
    aprint("1. Open the viewer with: luxar serve test_dof_grid_example.zarr")
    aprint("2. Press 'C' to center the camera on the point cloud")
    aprint("3. Press 'R' to open Rendering Controls")
    aprint("4. Navigate to 'Post-Processing Effects' → 'Depth of Field'")
    aprint("5. Enable DOF and adjust parameters:")
    aprint("   - Focus Distance: Controls which depth layer is sharp (try 10-20)")
    aprint("   - Blur Strength: Controls how blurry out-of-focus areas are")
    aprint("\n6. Look for:")
    aprint("   - Sharp focus at the selected distance")
    aprint("   - Gradual blur increase away from focal plane")
    aprint("   - No tiling or checkerboard artifacts")
    aprint("   - Smooth transitions between focus zones")
    aprint("\n7. The colored layers help identify different depths:")
    aprint("   - Red/Orange: Far background (z = -15 to -7.5)")
    aprint("   - Green/Cyan: Middle ground (z = -7.5 to 0)")
    aprint("   - Blue/Purple: Foreground (z = 0 to 15)")
    aprint("   - White crosses: Depth markers at -12, -8, -4, 0, 4, 8, 12")
    aprint("   - Yellow numbers: Depth labels")
    aprint("\nTip: If you can't see the grid, press 'C' to center the camera!")
    aprint("="*60)


if __name__ == "__main__":
    main()