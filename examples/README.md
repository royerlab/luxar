# Luxar Examples

This directory contains example scripts demonstrating various features of Luxar, the nD scene compiler and GPU-accelerated renderer. Each example is designed to be educational and showcases specific capabilities of the system.

## Quick Start

1. **Install Luxar** (if not already installed):
   ```bash
   pip install -e packages/luxar
   ```

2. **Run an example**:
   ```bash
   cd examples
   python <example_name>.py
   ```

3. **View the result**:
   ```bash
   luxar serve <output_file>.zarr
   ```
   Then open http://localhost:5173 in your browser.

## Examples Overview

### Basic Examples

#### 1. **radius_basic_example.py** - Introduction to Point Radii
A minimal example showing how to control point sizes:
- Three rows of points with different radii (small, medium, large)
- Color-coded for easy identification
- Perfect for verifying the radius feature works correctly

```bash
python radius_basic_example.py
luxar serve radius_basic_example.zarr
```

#### 2. **simple_nd_example.py** - Learning nD Navigation
A 5D grid dataset designed for learning dimension navigation:
- Clear visual patterns that change with time (full grid → checkerboard → diagonal → border → cross)
- Color intensity changes with depth
- Three color channels displayed spatially
- Ideal for understanding how non-displayed dimensions work

```bash
python simple_nd_example.py
luxar serve simple_nd_example.zarr
```

### Dimension Navigation Examples

#### 3. **dimension_navigation_example.py** - Interactive Shape Sequences
Demonstrates clear visual feedback for dimension navigation:
- Different geometric shapes at each frame (circle, square, triangle, star, cross)
- Distinct colors for each shape
- Perfect for testing keyboard navigation controls

```bash
python dimension_navigation_example.py
luxar serve dimension_navigation_example.zarr
```

#### 4. **dimension_sliders_5d_example.py** - Advanced 5D Navigation
Animated spiral with dimension sliders UI:
- 5D dataset with rotating spiral animation
- Time and channel as non-displayed dimensions
- Demonstrates dimension slider interface
- Shows radius-based slicing in action

```bash
python dimension_sliders_5d_example.py
luxar serve dimension_sliders_5d_example.zarr
```

#### 5. **dense_grid_5d_example.py** - Dense 5D Grid
Tests performance and slicing with dense data:
- 5D grid with 30,000+ points
- Multiple time frames and channels
- Good for testing performance with dimension navigation
- Shows how dense data behaves with slicing

```bash
python dense_grid_5d_example.py
luxar serve dense_grid_5d_example.zarr
```

#### 6. **rainbow_sphere_4d_example.py** - True 4D Spatial Geometry
Beautiful demonstration of 4D hypersphere:
- All 4 dimensions are spatial (X, Y, Z, W)
- Slicing through W dimension shows spheres of varying radius
- Rainbow colors based on 4D position
- Perfect for understanding higher-dimensional geometry

```bash
python rainbow_sphere_4d_example.py
luxar serve test_4d_rainbow_sphere_example.zarr
```

#### 7. **time_series_4d_example.py** - Animated 4D Data
Shows how to create time-varying 3D visualizations:
- Rotating spiral that evolves over time
- Color gradient from blue to red across time steps
- Demonstrates smooth animation through the time dimension

```bash
python time_series_4d_example.py
luxar serve time_series_4d_example.zarr
```

#### 8. **scene_dimensions_example.py** - Advanced Dimension Configuration
Comprehensive example of scene-level dimension definitions:
- Custom units, ranges, and step sizes
- Mixed discrete and continuous dimensions
- Multichannel time series with proper metadata

```bash
python scene_dimensions_example.py
luxar serve scene_dimensions_example.zarr
```

### Point Attribute Examples

#### 9. **radius_showcase_example.py** - Comprehensive Radius Features
Full showcase of per-point radius functionality:
- **Size Gradient Spiral**: Points that grow along a spiral path
- **Distance-Based Sphere**: Points sized by distance from center
- **Random Sized Cube**: Randomly distributed radii
- **Layered Spheres**: Concentric spheres with different sizes

```bash
python radius_showcase_example.py
luxar serve radius_showcase_example.zarr
```

#### 10. **sharpness_showcase_example.py** - Point Edge Control
Demonstrates the sharpness parameter for edge falloff:
- **Sharpness Gradient**: Smooth transition from soft to sharp
- **Fixed Comparison**: Side-by-side sharpness values
- **Mixed Cloud**: Varying sharpness in one cloud
- **Sharpness Wave**: Sinusoidal patterns

```bash
python sharpness_showcase_example.py
luxar serve sharpness_showcase_example.zarr
```

#### 11. **sharpness_compensation_example.py** - Size Compensation
Shows how sharpness affects apparent size:
- Points with identical radii but different sharpness
- Demonstrates shader compensation for consistent sizing
- Color gradient from soft (blue) to sharp (red)

```bash
python sharpness_compensation_example.py
luxar serve sharpness_compensation_example.zarr
```

### Advanced Examples

#### 12. **nd_points_example.py** - Mixed Dimensionality Scene
Complex example with multiple point clouds of different dimensions:
- 5D time series data
- 2D projection planes
- 3D reference geometry
- Shows how to mix different dimensional data in one scene

```bash
python nd_points_example.py
luxar serve nd_points_example.zarr
```

#### 13. **radius_slicing_example.py** - nD Visibility Concepts
Educational example about radius-based visibility:
- Points as nD hyperspheres intersecting viewing hyperplanes
- Larger radii visible across more dimension slices
- Gradient demonstration of radius effects
- Useful for understanding uncertainty visualization

```bash
python radius_slicing_example.py
luxar serve radius_slicing_example.zarr
```

#### 14. **rainbow_sphere_spiral_example.py** - Beautiful Visualization
Creates an aesthetically pleasing sphere with rainbow colors:
- Points arranged in a spherical spiral (Fibonacci-like distribution)
- Smooth rainbow gradient flowing along the spiral
- Varying point sizes creating a wave pattern
- Larger accent points as visual landmarks

```bash
python rainbow_sphere_spiral_example.py
luxar serve rainbow_sphere_spiral_example.zarr
```

### Transform and Hierarchy Examples

#### 15. **transform_example.py** - Transform System Demonstration  
Comprehensive showcase of the transform system:
- Translation, rotation, and scaling operations
- Transform composition and matrix multiplication
- Coordinate system visualization with RGB axes
- Hierarchical transform inheritance
- Complex transform combinations

```bash
python transform_example.py
luxar serve transform_example.zarr
```

#### 16. **hierarchy_example.py** - Scene Hierarchy and Inheritance
Demonstrates parent-child relationships and property inheritance:
- 4-level deep nested hierarchy
- Property inheritance (opacity, gamma, blending modes)
- Transform propagation through hierarchy
- Space-themed visualization (solar system, space station, galaxy)
- Clear demonstration of scene graph benefits

```bash
python hierarchy_example.py
luxar serve hierarchy_example.zarr
```

#### 17. **multiple_objects_example.py** - Multiple Point Clouds
Showcases complex scenes with multiple distinct objects:
- Six different point cloud objects (spiral galaxy, clusters, nebula, rings, stars, particles)
- Varied rendering properties per object
- Visual composition techniques
- ~31,000 total points across objects
- Color harmony and visual hierarchy

```bash
python multiple_objects_example.py
luxar serve multiple_objects_example.zarr
```

### Rendering and Performance Examples

#### 18. **rendering_modes_example.py** - Blending Modes Comparison
Educational demonstration of different blending modes:
- Normal, additive, and multiply blending modes
- Side-by-side comparison of effects
- Parent-child property inheritance
- Visual understanding of compositing techniques

```bash
python rendering_modes_example.py
luxar serve rendering_modes_example.zarr
```

#### 19. **rendering_attributes_example.py** - Rendering Attributes API
Comprehensive demonstration of rendering attributes:
- Setting and modifying opacity, gamma, and blending modes
- Method chaining API usage
- Error handling and validation
- Visual layout for easy comparison of effects

```bash
python rendering_attributes_example.py
luxar serve rendering_attributes_example.zarr
```

#### 20. **performance_benchmark_example.py** - Performance Testing
Educational benchmark demonstrating performance analysis:
- Scene creation with 100 nodes and varying materials
- Systematic testing of material combinations
- Detailed timing and performance metrics
- Performance analysis and optimization insights
- Progress tracking with educational output

```bash
python performance_benchmark_example.py
luxar serve performance_benchmark_example.zarr
```

## Keyboard Navigation Controls

When viewing nD data (>3D), use these controls:

### Dimension Navigation
- **Number keys (1-9)**: Select which non-displayed dimension to navigate
- **`[` and `]`**: Move backward/forward in the selected dimension
- **Mouse wheel**: Zoom in/out
- **Shift + Mouse wheel**: Adjust field of view

### Camera Controls
- **Left mouse drag**: Rotate view (arcball rotation)
- **Right mouse drag**: Pan view
- **Space**: Toggle fullscreen
- **H**: Show help overlay

### Advanced Controls
- **Ctrl/Cmd + P**: Toggle performance stats
- **Ctrl/Cmd + A**: Toggle advanced rendering controls

## Key Concepts

### Dimensions
- **Displayed dimensions**: The 3D subset shown in the viewer (max 3)
- **Non-displayed dimensions**: Additional dimensions navigated via sliders/keyboard
- **Discrete dimensions**: Integer steps (e.g., time frames, channels)
- **Continuous dimensions**: Smooth navigation (e.g., depth, wavelength)

### Point Attributes
- **positions**: nD coordinates (required)
- **colors**: RGB values 0-255 (optional, defaults to white)
- **radii**: Per-point size control (optional, defaults to 0.1)
- **sharpness**: Edge falloff 0.5-10.0 (optional, defaults to 2.0)

### nD Visualization Features
- **Dimension sliders**: Visual UI for navigating non-displayed dimensions
- **Status bar**: Shows current position in nD space
- **Radius-based slicing**: Points visible based on hypersphere intersection
- **Scene-level dimensions**: Consistent coordinate system across all objects

### Performance Tips
- Keep point counts reasonable for interactivity (<10M points)
- Use appropriate chunk sizes for your data patterns
- Consider using different sharpness values for visual hierarchy
- Combine radius and sharpness for rich visual effects

## Creating Your Own Examples

Template for new examples:

```python
#!/usr/bin/env python3
"""Example Name - Brief description of what this demonstrates.

This example demonstrates:
- Key feature 1
- Key feature 2
- Key feature 3
"""

import numpy as np
from pathlib import Path
from arbol import aprint
from luxar import Scene, Dimensions, Dimension

def main():
    """Create an example demonstrating [feature]."""
    output_path = Path("my_example.zarr")
    aprint(f"Creating example at {output_path}")
    
    # Define dimensions if using nD data
    dimensions = Dimensions([
        Dimension("x", unit="μm", range=(-50, 50), display=True),
        Dimension("y", unit="μm", range=(-50, 50), display=True),
        Dimension("z", unit="μm", range=(-50, 50), display=True),
        Dimension("time", unit="s", range=(0, 10), display=False, discrete=True)
    ])
    
    # Create scene
    scene = Scene(output_path, dimensions=dimensions)
    
    # Generate your data
    n_points = 1000
    positions = np.random.randn(n_points, 4).astype(np.float32) * 10
    colors = np.random.randint(0, 255, (n_points, 3), dtype=np.uint8)
    radii = np.random.uniform(0.05, 0.2, n_points).astype(np.float32)
    sharpness = np.full(n_points, 2.0, dtype=np.float32)
    
    # Add to scene
    scene.add_points(
        "MyPointCloud",
        positions,
        colors=colors,
        radii=radii,
        sharpness=sharpness
    )
    
    scene.finalize()
    
    # Print instructions
    aprint(f"✓ Example created with {n_points:,} points")
    aprint(f"\nTo view: luxar serve {output_path}")

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

1. **Points not visible**
   - Check that positions are within camera view
   - Verify dimension ranges if using scene-level dimensions
   - Ensure colors are in valid range [0, 255]
   - For nD data, check current slice position

2. **Performance problems**
   - Reduce point count
   - Increase chunk size in scene creation
   - Close other browser tabs
   - Disable advanced rendering effects

3. **Navigation not working**
   - Make sure you have non-displayed dimensions
   - Press number key first, then use brackets
   - Check browser console for error messages
   - Ensure canvas has focus (click on it)

4. **Colors look wrong**
   - Ensure uint8 dtype for colors
   - Check value range is 0-255, not 0-1
   - Verify RGB order (not BGR)

## Advanced Topics

### Radius-Based Slicing
Points in nD are treated as hyperspheres. When viewing a 3D slice:
- Point visibility depends on distance from slice position
- Larger radius = visible across more slices
- Effective radius shrinks as: `r_eff = sqrt(r² - d²)`
- Points fade out smoothly near slice boundaries

### Sharpness Effects
- **0.5-1.0**: Very soft, glowing appearance
- **2.0**: Default balanced falloff
- **5.0-10.0**: Sharp edges with minimal falloff
- Higher sharpness can make points appear smaller

### Performance Optimization
- Use `np.float32` for positions and radii
- Use `np.uint8` for colors
- Consider chunking large datasets
- Pre-compute expensive operations
- Use discrete dimensions for frame-based data

## Contributing

When adding new examples:
1. Follow the naming convention: `feature_description_example.py`
2. Include comprehensive docstrings
3. Use `arbol.aprint` for formatted output
4. Add clear instructions for users
5. Update this README with your example
6. Ensure output files follow `*_example.zarr` naming

## Generated Datasets

After running examples, you'll find these .zarr datasets:
- Each example generates a corresponding `.zarr` file
- These are self-contained and can be shared
- Use `luxar serve <dataset>.zarr` to view any dataset
- Clean up old datasets with `rm -rf *.zarr` when needed

## License

These examples are part of the Luxar project and are licensed under the same terms as the main project.