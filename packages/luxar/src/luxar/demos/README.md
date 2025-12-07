# Luxar Demos

This folder contains self-contained demo scripts that showcase Luxar's capabilities. Each demo is a complete, runnable Python script that generates data, launches the viewer, and cleans up automatically.

## Purpose

These demos are:
- **Didactic**: Easy to understand and learn from
- **Self-contained**: All generation code in one file
- **Complete**: Generate → Serve → View → Cleanup workflow
- **Copy-pasteable**: Can be used as templates for your own visualizations

## Available Demos

### demo_network_performance.py - Network Performance Testing ⭐ NEW
Large multi-cluster dataset (1M points) for testing viewer performance under network constraints.

**Run**:
```bash
# Default: 1M points with slow broadband simulation
python packages/luxar/src/luxar/demos/demo_network_performance.py

# Test with 3G mobile connection
python packages/luxar/src/luxar/demos/demo_network_performance.py --profile 3g

# Large dataset with satellite latency
python packages/luxar/src/luxar/demos/demo_network_performance.py --points=2000000 --profile satellite

# Compare full speed vs throttled
python packages/luxar/src/luxar/demos/demo_network_performance.py --no-simulation
python packages/luxar/src/luxar/demos/demo_network_performance.py --profile slow-broadband
```

**Demonstrates**:
- Network simulation feature (bandwidth throttling, latency, jitter, packet loss)
- Testing viewer performance under realistic network conditions
- Progressive loading behavior with limited bandwidth
- Cache effectiveness under bandwidth constraints
- Multi-cluster particle systems with varying densities
- Vectorized HSV→RGB color conversion
- Large dataset handling (1M+ points)

**Use Cases**:
- Test viewer performance on slow connections
- Validate caching and progressive loading
- Compare loading behavior across network profiles
- Performance regression testing
- UX research for minimum viable network requirements

**Watch browser DevTools Network tab to see throttling in action!**

---

### demo_lorenz.py - Lorenz Attractor
Beautiful chaotic attractor with rainbow color gradient.

**Run**:
```bash
python packages/luxar/src/luxar/demos/demo_lorenz.py
# Or with custom point count:
python packages/luxar/src/luxar/demos/demo_lorenz.py --points=100000
```

**Demonstrates**:
- Generating trajectory by integrating differential equations
- Vectorized HSV→RGB color conversion
- Time-based color gradients
- Progressive writing
- Chaotic systems visualization

### demo_rainbow_sphere.py - Fibonacci Spiral Sphere
Dense sphere (200k points) with perfect distribution and rainbow colors.

**Run**:
```bash
python packages/luxar/src/luxar/demos/demo_rainbow_sphere.py
# Or with custom point count:
python packages/luxar/src/luxar/demos/demo_rainbow_sphere.py --points=100000
```

**Demonstrates**:
- Fibonacci (golden angle) spiral for optimal sphere coverage
- Smooth rainbow gradient using phase-shifted sine waves
- Automatic point spacing calculation (sphere area / n_points)
- High density visualization (200k points)
- Mathematical point distribution

### demo_volumetric_cloud.py - Fractal Cloud Structure
Realistic cloud using multi-octave fractal noise and varying point sizes.

**Run**:
```bash
python packages/luxar/src/luxar/demos/demo_volumetric_cloud.py
# Or with custom candidate count:
python packages/luxar/src/luxar/demos/demo_volumetric_cloud.py --points=1000000
```

**Demonstrates**:
- Fractal noise generation (self-contained Perlin-like implementation)
- Multi-octave noise for natural detail at multiple scales
- Volumetric density filtering (creates wisps and gaps)
- Varying point sizes based on local density
- Soft, cloud-like appearance (low sharpness 0.5-2.0)
- 3D Gaussian falloff for puff shape
- Realistic atmospheric effects

### demo_cubic_array.py - 3D Cubic Grid with Star Background
Dense 100³ grid (1M points) with 500k background stars.

**Run**:
```bash
python packages/luxar/src/luxar/demos/demo_cubic_array.py
```

**Demonstrates**:
- Creating regular grids using meshgrid
- Multi-layer scenes (foreground + background)
- Depth-of-field visualization techniques
- Very high point density (1.5M points total)
- Sharp disc-like points (main grid)
- Vectorized star color generation
- Semi-transparent background layers
- Different blending modes (additive vs normal)

### demo_mandelbulb.py - 3D Mandelbulb Fractal ⭐ NEW
Stunning volumetric representation of the famous Mandelbulb 3D fractal.

**Run**:
```bash
python packages/luxar/src/luxar/demos/demo_mandelbulb.py
# Or with custom resolution:
python packages/luxar/src/luxar/demos/demo_mandelbulb.py --resolution=128 --power=8
```

**Demonstrates**:
- 3D fractal mathematics (extension of Mandelbrot set)
- Distance estimation for surface detection
- Iteration-based coloring for visual depth
- Adaptive point sizing based on detail level
- Spherical coordinate transformation
- Escape-time algorithm in 3D
- Self-similar structure at multiple scales

### demo_spiral_galaxy.py - Realistic Multi-Armed Spiral Galaxy ⭐ NEW
Beautiful astronomical simulation of a barred spiral galaxy.

**Run**:
```bash
python packages/luxar/src/luxar/demos/demo_spiral_galaxy.py
# Or with custom parameters:
python packages/luxar/src/luxar/demos/demo_spiral_galaxy.py --stars=500000 --arms=4
```

**Demonstrates**:
- Logarithmic spiral arm generation
- Realistic stellar population distributions
- Color variation (blue young stars in arms, red/yellow old stars in bulge)
- Central galactic bulge modeling
- Stellar halo with sparse old stars
- Exponential density falloff
- Astronomical scales and proportions
- Realistic astrophysical effects

### demo_4d_fractals.py - 4D Geometric Fractal Explorer ⭐ NEW
Interactive exploration of 6 different 4D geometric fractals with categorical dimension.

**Run**:
```bash
python packages/luxar/src/luxar/demos/demo_4d_fractals.py
# Or with custom grid resolution:
python packages/luxar/src/luxar/demos/demo_4d_fractals.py --grid=64
```

**Demonstrates**:
- 4D spatial navigation (XYZ + W dimension)
- Categorical dimension (select between 6 fractal types)
- Large dataset with spatial indexing (~100M+ points, ~1M visible)
- XOR Fractal (bitwise XOR self-similarity)
- Menger Sponge 4D (recursive subdivision)
- Sierpinski 4D (modular arithmetic patterns)
- Cantor Dust 4D (product of 1D sets)
- Checkerboard and Diamond patterns
- Instant generation with vectorized operations
- nD slicing and visualization

## Demo Pattern

Each demo follows this self-contained pattern:

```python
#!/usr/bin/env python3
"""Demo: Description

What this demonstrates...
"""

import subprocess
import tempfile
from pathlib import Path
import numpy as np
from arbol import aprint, asection
from luxar import LuxarZarrCompiler, Dimension, Dimensions

def generate_my_data(output_path: Path, **params) -> None:
    """Generate the dataset - ALL CODE IN THIS FUNCTION.

    Args:
        output_path: Where to write zarr
        **params: Generation parameters
    """
    with asection("Generating Data"):
        # 1. Generate positions, colors, radii, etc.
        # ALL generation logic here - no external functions!
        positions = ...
        colors = ...

        # 2. Write to zarr
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points('name', positions, colors=colors, ...)

def main():
    """Entry point - generate and serve."""
    with tempfile.TemporaryDirectory(prefix="luxar_demo_") as tmpdir:
        output_path = Path(tmpdir) / "demo.zarr"

        # Generate
        generate_my_data(output_path)

        # Serve using CLI (handles server lifecycle)
        try:
            subprocess.run([
                "luxar", "serve", str(output_path),
                "--viewer", "--open"
            ], check=True)
        except KeyboardInterrupt:
            aprint("\\n🛑 Stopping...")

    # Auto-cleanup
    aprint("✓ Cleanup complete")

if __name__ == "__main__":
    main()
```

## Key Principles

### 1. Self-Contained
**All generation code must be in the demo file itself.**

✅ **GOOD**:
```python
def generate_my_data(output_path):
    # Generate positions here
    x = np.linspace(0, 10, 1000)
    positions = ...
    # Write to zarr here
    with LuxarZarrCompiler(output_path) as compiler:
        ...
```

❌ **BAD**:
```python
def generate_my_data(output_path):
    # Calls external function - not self-contained!
    from my_utils import create_positions
    positions = create_positions()  # ← External dependency!
```

### 2. Use Temporary Directory
Always use `tempfile.TemporaryDirectory()` to ensure cleanup:

```python
with tempfile.TemporaryDirectory(prefix="luxar_demo_myname_") as tmpdir:
    output = Path(tmpdir) / "data.zarr"
    generate_data(output)
    serve_and_view(output)
# Automatic cleanup when context exits
```

### 3. Use CLI for Serving
Don't reimplement server logic - use the luxar CLI:

```python
subprocess.run([
    "luxar", "serve", str(output_path),
    "--viewer",  # Also serve viewer
    "--open"     # Open browser automatically
], check=True)
```

Benefits:
- Reuses tested server code
- Handles CORS, directory listing, etc.
- Stops cleanly on Ctrl+C
- No complex server lifecycle code

### 4. Handle Errors Gracefully

```python
try:
    subprocess.run(["luxar", "serve", ...], check=True)
except KeyboardInterrupt:
    aprint("\\n🛑 Stopping...")  # Normal user stop
except subprocess.CalledProcessError:
    aprint("❌ Error - is viewer built?")  # Helpful message
except FileNotFoundError:
    aprint("❌ luxar command not found")  # Installation issue
```

### 5. Use arbol for Output
Structure console output with arbol for clarity:

```python
from arbol import aprint, asection

with asection("Generating Data"):
    aprint("Creating grid...")
    # ... generation code ...
    aprint(f"✓ Created {n_points:,} points")

with asection("Writing to Zarr"):
    # ... writing code ...
    aprint(f"✓ Written to {path}")
```

## Creating New Demos

1. **Copy a template** (demo_lorenz.py or demo_cubic_array.py)
2. **Rename** to demo_yourname.py
3. **Update docstring** with what it demonstrates
4. **Implement generation** in the generate_* function (keep everything in that function!)
5. **Test** by running: `python demo_yourname.py`
6. **Ctrl+C** to stop and verify cleanup works

## Tips

### For Large Datasets
Use progressive writing to avoid memory issues:

```python
# DON'T load all data at once if very large
# DO generate and write in chunks

with LuxarZarrCompiler(output) as compiler:
    scene = compiler.create_scene()

    # Write in batches
    for i in range(num_batches):
        batch = generate_batch(i)
        scene.add_points(f'batch_{i}', batch)
```

### For nD Demos
Specify dimensions with proper display flags:

```python
dims = Dimensions([
    Dimension('x', unit='um', display=True),
    Dimension('y', unit='um', display=True),
    Dimension('z', unit='um', display=True),
    Dimension('time', unit='s', display=False, discrete=True, range=(0, 99))
])
```

### For Complex Math
Add comments explaining the mathematics:

```python
# Generate sphere using spherical coordinates
# φ ∈ [0, 2π], θ ∈ [0, π]
# x = r sin(θ) cos(φ)
# y = r sin(θ) sin(φ)
# z = r cos(θ)
phi = np.random.uniform(0, 2*np.pi, n)
theta = np.arccos(np.random.uniform(-1, 1, n))
...
```

## Running All Demos

```bash
# From project root:
python packages/luxar/src/luxar/demos/demo_lorenz.py
python packages/luxar/src/luxar/demos/demo_rainbow_sphere.py
python packages/luxar/src/luxar/demos/demo_volumetric_cloud.py
python packages/luxar/src/luxar/demos/demo_cubic_array.py
python packages/luxar/src/luxar/demos/demo_mandelbulb.py
python packages/luxar/src/luxar/demos/demo_spiral_galaxy.py
python packages/luxar/src/luxar/demos/demo_4d_fractals.py

# Or since they're executable:
./packages/luxar/src/luxar/demos/demo_lorenz.py
./packages/luxar/src/luxar/demos/demo_rainbow_sphere.py
./packages/luxar/src/luxar/demos/demo_volumetric_cloud.py
./packages/luxar/src/luxar/demos/demo_cubic_array.py
./packages/luxar/src/luxar/demos/demo_mandelbulb.py
./packages/luxar/src/luxar/demos/demo_spiral_galaxy.py
./packages/luxar/src/luxar/demos/demo_4d_fractals.py
```

## Troubleshooting

**"luxar command not found"**:
```bash
pip install -e .  # Install luxar in development mode
```

**"Viewer not built"**:
```bash
cd packages/luxar-viewer
pnpm install
pnpm build
```

**Port already in use**:
The CLI automatically finds available ports, but if issues persist, try specifying a different port:
```bash
# Won't work with subprocess.run in demos currently
# For manual testing: luxar serve data.zarr --viewer --port 8001 --viewer-port 5174
```

## Philosophy

These demos are **teaching tools**. They should be:
- Simple enough to understand in 5 minutes
- Complete enough to show real capability
- Clean enough to use as templates
- Fun enough to inspire creativity!

Keep them self-contained so anyone can read ONE file and understand the complete workflow.