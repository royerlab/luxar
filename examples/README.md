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

3. **Start the viewer** (in a separate terminal):
   ```bash
   cd packages/luxar-player
   pnpm install  # First time only
   pnpm run dev
   ```

4. **Serve the data**:
   ```bash
   luxar serve <output_file>.zarr
   ```

5. **Open in browser**: Navigate to http://localhost:5173

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

#### 4. **time_series_4d_example.py** - Animated 4D Data
Shows how to create time-varying 3D visualizations:
- Rotating spiral that evolves over time
- Color gradient from blue to red across time steps
- Demonstrates smooth animation through the time dimension

```bash
python time_series_4d_example.py
luxar serve time_series_4d_example.zarr
```

#### 5. **scene_dimensions_example.py** - Advanced Dimension Configuration
Comprehensive example of scene-level dimension definitions:
- Custom units, ranges, and step sizes
- Mixed discrete and continuous dimensions
- Multichannel time series with proper metadata

```bash
python scene_dimensions_example.py
luxar serve scene_dimensions_example.zarr
```

### Point Attribute Examples

#### 6. **radius_showcase_example.py** - Comprehensive Radius Features
Full showcase of per-point radius functionality:
- **Size Gradient Spiral**: Points that grow along a spiral path
- **Distance-Based Sphere**: Points sized by distance from center
- **Random Sized Cube**: Randomly distributed radii
- **Layered Spheres**: Concentric spheres with different sizes

```bash
python radius_showcase_example.py
luxar serve radius_showcase_example.zarr
```

#### 7. **sharpness_showcase_example.py** - Point Edge Control
Demonstrates the sharpness parameter for edge falloff:
- **Sharpness Gradient**: Smooth transition from soft to sharp
- **Fixed Comparison**: Side-by-side sharpness values
- **Mixed Cloud**: Varying sharpness in one cloud
- **Sharpness Wave**: Sinusoidal patterns

```bash
python sharpness_showcase_example.py
luxar serve sharpness_showcase_example.zarr
```

#### 8. **sharpness_compensation_example.py** - Size Compensation
Shows how sharpness affects apparent size:
- Points with identical radii but different sharpness
- Demonstrates shader compensation for consistent sizing
- Color gradient from soft (blue) to sharp (red)

```bash
python sharpness_compensation_example.py
luxar serve sharpness_compensation_example.zarr
```

### Advanced Examples

#### 9. **nd_points_example.py** - Mixed Dimensionality Scene
Complex example with multiple point clouds of different dimensions:
- 5D time series data
- 2D projection planes
- 3D reference geometry
- Shows how to mix different dimensional data in one scene

```bash
python nd_points_example.py
luxar serve nd_points_example.zarr
```

#### 10. **radius_slicing_example.py** - nD Visibility Concepts
Educational example about radius-based visibility:
- Points as nD hyperspheres intersecting viewing hyperplanes
- Larger radii visible across more dimension slices
- Gradient demonstration of radius effects
- Useful for understanding uncertainty visualization

```bash
python radius_slicing_example.py
luxar serve radius_slicing_example.zarr
```

## Keyboard Navigation Controls

When viewing nD data (>3D), use these controls:

- **Number keys (1-9)**: Select which non-displayed dimension to navigate
- **`[` and `]`**: Move backward/forward in the selected dimension
- **Mouse**: Rotate/pan/zoom the 3D view

## Key Concepts

### Dimensions
- **Displayed dimensions**: The 3D subset shown in the viewer (max 3)
- **Non-displayed dimensions**: Additional dimensions navigated via keyboard
- **Discrete dimensions**: Integer steps (e.g., time frames, channels)
- **Continuous dimensions**: Smooth navigation (e.g., depth, wavelength)

### Point Attributes
- **positions**: nD coordinates (required)
- **colors**: RGB values 0-255 (optional, defaults to white)
- **radii**: Per-point size control (optional, defaults to 0.1)
- **sharpness**: Edge falloff 0.5-10.0 (optional, defaults to 2.0)

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
        Dimension("x", unit="units"),
        Dimension("y", unit="units"),
        Dimension("z", unit="units")
    ])
    
    # Create scene
    scene = Scene(output_path, dimensions=dimensions)
    
    # Generate your data
    n_points = 1000
    positions = np.random.randn(n_points, 3).astype(np.float32) * 10
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

2. **Performance problems**
   - Reduce point count
   - Increase chunk size in scene creation
   - Close other browser tabs

3. **Navigation not working**
   - Make sure you have non-displayed dimensions
   - Press number key first, then use brackets
   - Check browser console for error messages

4. **Colors look wrong**
   - Ensure uint8 dtype for colors
   - Check value range is 0-255, not 0-1
   - Verify RGB order (not BGR)

## Advanced Topics

### Radius-Based Slicing
Points in nD are treated as hyperspheres. When viewing a 3D slice:
- Point visibility depends on distance from slice position
- Larger radius = visible across more slices
- Points shrink as they move away from slice center

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

## Contributing

When adding new examples:
1. Follow the naming convention: `feature_description_example.py`
2. Include comprehensive docstrings
3. Use `arbol.aprint` for formatted output
4. Add clear instructions for users
5. Update this README with your example

## License

These examples are part of the Luxar project and are licensed under the same terms as the main project.