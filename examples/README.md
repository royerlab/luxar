# Luxar Examples

This directory contains example scripts demonstrating various features of Luxar. All examples use the `arbol` library for formatted console output.

## Examples Overview

### 1. **radius_showcase_example.py** - Comprehensive Radius Feature Showcase
A full showcase of the per-point radius functionality with multiple visualization techniques:
- **Size Gradient Spiral**: Points that grow from small to large along a spiral path
- **Distance-Based Sphere**: Points sized inversely to their distance from center
- **Random Sized Cube**: Points with randomly distributed radii
- **Layered Spheres**: Concentric spheres with different point sizes per layer

```bash
python radius_showcase_example.py
luxar serve radius_showcase_example.zarr
```

### 2. **sharpness_showcase_example.py** - Comprehensive Sharpness Feature Showcase
Demonstrates how the sharpness parameter controls point edge falloff:
- **Sharpness Gradient**: Smooth transition from soft (0.5) to sharp (10.0)
- **Fixed Sharpness Comparison**: Side-by-side comparison of different sharpness values
- **Mixed Sharpness Cloud**: Single cloud with varying sharpness per point
- **Sharpness Wave**: Sinusoidal sharpness variation creating visual patterns

```bash
python sharpness_showcase_example.py
luxar serve sharpness_showcase_example.zarr
```

### 3. **radius_basic_example.py** - Simple Radius Example
A minimal example showing three rows of points with different radii:
- Top row: Large points (radius=0.5)
- Middle row: Medium points (radius=0.2)
- Bottom row: Small points (radius=0.05)

Perfect for quickly verifying the radius feature is working.

```bash
python radius_basic_example.py
luxar serve radius_basic_example.zarr
```

### 4. **sharpness_compensation_example.py** - Sharpness Size Compensation
Demonstrates how the shader compensates for apparent size changes due to sharpness:
- Shows points with identical radii but different sharpness values
- All points should appear roughly the same size
- Colors transition from blue (soft) to red (sharp)

```bash
python sharpness_compensation_example.py
luxar serve sharpness_compensation_example.zarr
```

## Running the Examples

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
   npm install  # First time only
   npm run dev
   ```

4. **Serve the data**:
   ```bash
   luxar serve <output_file>.zarr
   ```

5. **Open in browser**: Navigate to http://localhost:5173

## Output Files

Each example creates a `.zarr` directory containing the compiled scene data. These directories can be:
- Served using `luxar serve`
- Shared with others (they're self-contained)
- Loaded in custom applications using the Zarr library

## Features Demonstrated

### Point Attributes
- **positions**: 3D coordinates (required)
- **colors**: RGB values (optional, defaults to white)
- **radii**: Per-point size control (optional, defaults to 0.1)
- **sharpness**: Edge falloff control (optional, defaults to 2.0)

### Sharpness Values
- `0.5-1.0`: Very soft, glowing appearance
- `2.0`: Default, balanced falloff
- `5.0-10.0`: Sharp edges with minimal falloff

### Best Practices
- Keep point counts reasonable for interactive performance (<10M points)
- Use appropriate chunk sizes for your data patterns
- Consider using different sharpness values for visual hierarchy
- Combine radius and sharpness for rich visual effects

## Creating Your Own Examples

To create a new example:

```python
#!/usr/bin/env python3
"""Brief description of your example."""

import numpy as np
from pathlib import Path
from arbol import aprint
from luxar import Scene

def main():
    output_path = Path("my_example.zarr")
    aprint(f"Creating example at {output_path}")
    
    # Create scene
    scene = Scene(output_path)
    
    # Add your points
    positions = np.random.randn(1000, 3).astype(np.float32)
    colors = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)
    radii = np.random.uniform(0.05, 0.2, 1000).astype(np.float32)
    sharpness = np.random.uniform(0.5, 5.0, 1000).astype(np.float32)
    
    scene.add_points("MyPoints", positions, colors, radii=radii, sharpness=sharpness)
    scene.finalize()
    
    aprint("✓ Example created successfully!")

if __name__ == "__main__":
    main()
```

## Troubleshooting

- **Points not visible**: Check that positions are within the camera view
- **Performance issues**: Reduce point count or increase chunk size
- **Colors look wrong**: Ensure colors are uint8 in range [0, 255]
- **Points too large/small**: Adjust the radius values or camera distance