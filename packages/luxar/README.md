# 🐍 Luxar Core - Python nD Scene Compiler

A Python library for compiling n-dimensional scientific datasets into optimized Zarr archives for high-performance visualization. Luxar Core provides the data compilation layer of the Luxar ecosystem, transforming your scientific data into a format optimized for GPU-accelerated rendering.

## 🎯 Design Philosophy

Luxar Core is built on the principle that **data compilation should never be the bottleneck**. Whether you're working with thousands or billions of data points, in 3D or higher dimensions, Luxar Core scales with your ambitions:

- **Dimension-agnostic**: Built to handle 3D, 4D, and beyond
- **Geometry-flexible**: Extensible architecture for points, lines, surfaces, volumes
- **Scale-unlimited**: Performance bounded by storage, not architecture
- **Stream-ready**: Chunked Zarr format enables progressive loading

## 📦 Installation

```bash
# From PyPI (when available)
pip install luxar

# Development installation
git clone https://github.com/royerlab/luxar.git
cd luxar
pip install -e ".[dev]"
```

## 🚀 Quick Start

```python
import numpy as np
from luxar import LuxarZarrCompiler, Dimensions, Dimension

# Create a scene with dimensions for nD visualization
dimensions = Dimensions([
    Dimension("x", unit="μm", range=(-100, 100), display=True),
    Dimension("y", unit="μm", range=(-100, 100), display=True),  
    Dimension("z", unit="μm", range=(-50, 50), display=True),
    Dimension("time", unit="s", range=(0, 10), step=0.1, display=False)
])

with LuxarZarrCompiler("my_dataset.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dimensions)
    
    # Add 4D points data (time + xyz)
    positions = np.random.randn(1_000_000, 4).astype(np.float32)
    colors = np.random.rand(1_000_000, 3).astype(np.float32)  # HDR colors supported
    radii = np.random.uniform(0.1, 0.5, 1_000_000).astype(np.float32)
    scene.add_points("TimeSeriesData", positions, colors=colors, radii=radii)
    
    # Add hierarchical organization  
    group = scene.add_group("Experiment1")
    scene.add_points("Measurement", positions2, colors2, parent=group)

# Finalize (consolidates metadata for fast loading)
# Context manager handles finalization automatically

# Serve for visualization
# luxar serve my_dataset.zarr
```

## 🏗️ Architecture

### Scene Graph

Luxar uses a hierarchical scene graph that mirrors Zarr's group structure:

```
Scene (root)
├── Group
│   ├── Points
│   ├── Lines (future)
│   └── Surface (future)
└── Points
```

Each node can have:
- **Transform**: 4x4 transformation matrix
- **Attributes**: Arbitrary metadata
- **Datasets**: Geometry-specific data arrays

### Node Types

#### Currently Implemented
- **Scene**: Root node managing the Zarr store
- **Group**: Organizational nodes with transforms
- **Points**: Point cloud geometry with positions and colors

#### Planned Geometry Types
- **Lines**: Connected line segments with per-vertex attributes
- **Surfaces**: Triangulated meshes with normals and textures
- **Volumes**: Volumetric data with transfer functions
- **Tensors**: Higher-dimensional data with projections

### Data Format

Luxar uses Zarr v2 format for maximum compatibility:

```
dataset.zarr/
├── .zattrs                    # Scene metadata
├── .zgroup                    # Zarr group marker
├── node_name/
│   ├── .zattrs               # Node attributes
│   ├── .zgroup               # Group marker
│   ├── positions/            # Geometry data
│   │   ├── .zarray          # Array metadata
│   │   └── 0.0.0            # Chunks
│   └── colors/
└── .zmetadata                # Consolidated metadata
```

## 🌟 nD Visualization Features

### Scene-Level Dimensions

Luxar uses scene-level dimension definitions to ensure consistency across all objects:

```python
from luxar import LuxarZarrCompiler, Dimensions, Dimension

# Define your nD coordinate system
dimensions = Dimensions([
    # Spatial dimensions (displayed in 3D viewer)
    Dimension("x", unit="μm", range=(-100, 100), display=True),
    Dimension("y", unit="μm", range=(-100, 100), display=True),
    Dimension("z", unit="μm", range=(-50, 50), display=True),
    
    # Non-displayed dimensions (navigated via sliders)
    Dimension("time", unit="s", range=(0, 60), step=0.5, display=False),
    Dimension("channel", unit="", range=(0, 3), discrete=True, display=False),
    Dimension("depth", unit="μm", range=(-20, 20), display=False)
])

scene = Scene("multidimensional.zarr", dimensions=dimensions)
```

### Dimension Types

- **Displayed Dimensions**: The 3D subset shown in the viewer (max 3)
- **Non-Displayed Dimensions**: Additional dimensions navigated via UI sliders
- **Discrete Dimensions**: Integer steps for frame-based data
- **Continuous Dimensions**: Smooth navigation for continuous variables

### Point Attributes

Enhanced points visualization with per-point attributes:

```python
# Generate 5D data (time, z, x, y, channel) 
n_points = 50_000
positions = np.random.randn(n_points, 5).astype(np.float32)

# Scale to fit dimension ranges
positions[:, 0] *= 5      # time: -5 to 5
positions[:, 1] *= 25     # z: -25 to 25  
positions[:, 2] *= 50     # x: -50 to 50
positions[:, 3] *= 50     # y: -50 to 50
positions[:, 4] = np.random.choice([0, 1, 2], n_points)  # discrete channels

# Per-point attributes
colors = np.random.randint(0, 255, (n_points, 3), dtype=np.uint8)
radii = np.random.uniform(0.1, 2.0, n_points).astype(np.float32)
sharpness = np.random.uniform(0.5, 10.0, n_points).astype(np.float32)

scene.add_points(
    "Points5D",
    positions,
    colors=colors,
    radii=radii,
    sharpness=sharpness
)
```

### Radius-Based Slicing

Points in nD space are treated as hyperspheres. When viewing a 3D slice:
- Point visibility depends on hypersphere intersection with viewing hyperplane
- Larger radius = visible across more dimension slices
- Natural representation of uncertainty or spread in higher dimensions

## 📖 API Reference

### Scene Class

```python
class Scene:
    """Root scene node managing a Zarr store."""
    
    def __init__(
        self, 
        store_path: Optional[PathLike] = None,
        units: str = "metre",
        version: str = "0.2",
        compressor: Optional[Compressor] = DEFAULT_COMP,
        dimensions: Optional[Dimensions] = None
    ) -> None:
        """
        Create a new scene.
        
        Args:
            store_path: Path to Zarr store (None for temporary)
            units: Physical units for coordinates
            version: Luxar format version
            compressor: Zarr compressor for datasets
            dimensions: Scene-level dimension definitions for nD data
        """
    
    def add_group(self, name: str, **attrs) -> Node:
        """Add a group node for organization.
        
        Args:
            name: Group node name
            **attrs: Additional attributes including:
                opacity: float (0.0-1.0, default 1.0) - Node opacity
                gamma: float (0.2-2.0, default 1.0) - Gamma correction
                blending_mode: str ("normal", "additive", default "additive")
                transform: list[float] - 16-element 4x4 transformation matrix (use transforms.to_list())
        """
    
    def add_points(
        self, 
        name: str,
        positions: np.ndarray,  # shape: (N, D), dtype: float32
        colors: Optional[np.ndarray] = None,  # shape: (N, 3), dtype: uint8
        radii: Optional[np.ndarray] = None,  # shape: (N,), dtype: float32
        sharpness: Optional[np.ndarray] = None,  # shape: (N,), dtype: float32
        parent: Optional[Node] = None,
        **attrs
    ) -> Points:
        """Add a points to the scene.
        
        Args:
            name: Node name
            positions: nD coordinates where D matches scene dimensions
            colors: RGB colors (0-255)
            radii: Per-point radii for size control
            sharpness: Edge falloff (0.5-10.0)
            parent: Parent node in hierarchy
            **attrs: Additional attributes including:
                opacity: float (0.0-1.0, default 1.0) - Node opacity
                gamma: float (0.2-2.0, default 1.0) - Gamma correction  
                blending_mode: str ("normal", "additive", default "additive")
                transform: list[float] - 16-element 4x4 transformation matrix (use transforms.to_list())
        """
    
    def finalize(self) -> None:
        """Consolidate metadata for optimal loading performance."""
```

### Node Class

```python
class Node:
    """Base class for all scene graph nodes."""
    
    def add_group(self, name: str, **attrs) -> Node:
        """Add a child group."""
    
    @property
    def transform(self) -> np.ndarray:
        """4x4 transformation matrix."""
```

### Points Class

```python
class Points(Node):
    """Point cloud geometry node."""
    
    # Created automatically via Scene.add_points()
    # Manages positions, colors, radii, and sharpness datasets in Zarr
```

### Dimensions Class

```python
class Dimensions:
    """Container for scene-level dimension definitions."""
    
    def __init__(self, dimensions: List[Dimension]) -> None:
        """Create dimensions from list of Dimension objects."""
    
    def validate_positions(self, positions: np.ndarray) -> None:
        """Validate that positions match dimension count."""
```

### Dimension Class

```python
class Dimension:
    """Single dimension definition with metadata."""
    
    def __init__(
        self,
        name: str,
        unit: str = "",
        scale: float = 1.0,
        range: Optional[Tuple[float, float]] = None,
        display: bool = False,
        discrete: bool = False,
        step: float = 1.0
    ) -> None:
        """
        Define a dimension.
        
        Args:
            name: Dimension name (e.g., "x", "time", "channel")
            unit: Physical unit (e.g., "μm", "s", "nm")
            scale: Scale factor for unit conversion
            range: Min/max values as (min, max)
            display: Whether shown in 3D viewer (max 3)
            discrete: Whether dimension has discrete steps
            step: Step size for navigation
        """
```

## 🖥️ Command Line Interface

Luxar provides a CLI for common operations:

```bash
# Serve a Zarr dataset for visualization
luxar serve dataset.zarr

# Create a demo dataset without serving
luxar demo --no-serve --output demo.zarr --points 100000

# Build a scene from a Python script
luxar build scene_script.py

# Get information about a dataset
luxar info dataset.zarr
```

**Serving datasets:**
- `luxar serve <path.zarr>` - Start HTTP server for dataset  
- Use `--port` to specify port (default: 8000)
- Use `--host` to bind to specific interface

**Creating demo data:**
- `luxar demo` - Create and serve demo with viewer (opens browser)
- `luxar demo --no-serve --output <path.zarr>` - Create demo without serving
- Use `--points` to specify number of points (default: 10,000)
- Use `--seed` for reproducible results

## 🎨 Rendering Attributes

Control the visual appearance of nodes with rendering attributes:

```python
from luxar import LuxarZarrCompiler

scene = Scene("styled_scene.zarr")

# Add points with custom rendering
points = scene.add_points(
    "StyledPoints",
    positions,
    colors=colors,
    opacity=0.8,           # Semi-transparent (0.0-1.0)
    gamma=1.5,             # Brighter gamma correction (0.2-2.0)
    blending_mode="normal" # Use normal blending instead of additive
)

# Modify rendering after creation
points.opacity = 0.5
points.gamma = 0.8
points.blending_mode = "normal"

# Chain modifications
points.set_opacity(0.7).set_gamma(1.2).set_blending_mode("additive")

# Apply to groups as well
group = scene.add_group(
    "TransparentGroup",
    opacity=0.6,
    blending_mode="normal"
)

# Context manager handles finalization automatically
```

**Blending Modes:**
- `"additive"` (default): HDR additive blending, good for glowing effects
- `"normal"`: Standard alpha blending
- `"multiply"`: Multiplicative blending, creates darkening effects
- `"minimum"`: Takes minimum values, creates intersection effects
- `"maximum"`: Takes maximum values, creates union effects

## 🔬 Advanced Usage

### Custom Chunking Strategy

```python
from luxar import LuxarZarrCompiler
import zarr

# Custom chunking for streaming large datasets
scene = Scene("large_dataset.zarr")

# Configure chunk size based on expected access patterns
# Smaller chunks = better streaming, more overhead
# Larger chunks = better sequential access, worse random access
chunk_size = 65536  # Points per chunk

positions = np.random.randn(10_000_000, 3).astype(np.float32)
colors = np.random.rand(10_000_000, 3).astype(np.uint8)

# Luxar automatically handles chunking, but you can tune it
# via the underlying Zarr arrays if needed
scene.add_points("LargeCloud", positions, colors)
# Context manager handles finalization automatically
```

### Hierarchical Data Organization

```python
# Organize multi-scale experimental data
scene = Scene("experiment.zarr")

# Time series organization
for t in range(num_timepoints):
    time_group = scene.add_group(f"t_{t:04d}")
    
    # Multiple measurements per timepoint
    for sensor_id in range(num_sensors):
        data = load_sensor_data(t, sensor_id)
        scene.add_points(
            f"sensor_{sensor_id}", 
            data.positions,
            data.colors,
            parent=time_group
        )

# Context manager handles finalization automatically
```

### Transforms and Coordinate Systems

Luxar provides comprehensive transform utilities for 3D scene manipulation:

```python
from luxar import LuxarZarrCompiler, transforms

scene = Scene("transformed_scene.zarr")

# Basic transforms
translation = transforms.translate(10, 5, 0)      # Move 10 units in X, 5 in Y
rotation = transforms.rotate(45, 'z')             # Rotate 45° around Z axis
scaling = transforms.scale(2, 2, 2)               # Scale 2x in all dimensions
uniform_scale = transforms.scale(uniform=0.5)     # Scale uniformly by 0.5

# Compose multiple transforms (applied left-to-right)
combined = transforms.compose(translation, rotation, scaling)

# Apply transforms to groups (transforms must be converted to list format)
group = scene.add_group("MyGroup")
group.transform = combined  # Can set directly as matrix

# OR during creation
group = scene.add_group("MyGroup", transform=transforms.to_list(combined))

# Or use the transform property
group.transform = transforms.rotate_x(30)  # Rotate 30° around X

# Hierarchical transforms (child inherits parent transform)
parent = scene.add_group("Robot")
parent.transform = transforms.translate(100, 0, 0)

child = parent.add_group("Sensor")  
child.transform = transforms.rotate_y(90)  # Relative to parent

# Advanced transforms
look_at = transforms.look_at(
    eye=(10, 10, 10),     # Camera position
    target=(0, 0, 0),     # Look at origin
    up=(0, 1, 0)          # Y-up
)

# Inverse transforms
t = transforms.translate(5, 0, 0)
t_inv = transforms.inverse(t)  # Translates -5, 0, 0

# Context manager handles finalization automatically
```

### Extending with New Geometry Types

```python
from luxar.node import Node
from luxar._io import create_dataset

class Lines(Node):
    """Example: Adding line geometry support."""
    
    def __init__(
        self,
        name: str,
        vertices: np.ndarray,  # shape: (N, 3)
        edges: np.ndarray,     # shape: (M, 2), indices
        colors: Optional[np.ndarray] = None,
        parent: Optional[Node] = None,
        **attrs
    ):
        # Initialize node
        group = parent._group.create_group(name)
        super().__init__(name, group)
        
        # Store geometry
        create_dataset(group, "vertices", vertices)
        create_dataset(group, "edges", edges)
        if colors is not None:
            create_dataset(group, "colors", colors)
        
        # Set type attribute
        group.attrs["type"] = "lines"
        group.attrs.update(attrs)
```

## 🎯 Performance Optimization

### Chunk Size Selection

| Data Pattern | Recommended Chunk Size | Rationale |
|-------------|------------------------|-----------|
| Sequential scan | 1-10 MB | Minimize read operations |
| Random access | 64-256 KB | Balance latency vs throughput |
| Streaming | 256-512 KB | Optimize for network transfer |
| Local processing | 10-50 MB | Maximize CPU cache usage |

### Compression Trade-offs

```python
from numcodecs import Blosc

# For maximum speed (local visualization)
fast_compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)

# For network streaming (bandwidth-limited)
balanced_compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)

# For archival storage (space-limited)
compact_compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE)

scene = Scene("data.zarr", compressor=balanced_compressor)
```

### Memory-Efficient Processing

```python
# Process massive datasets without loading into memory
def process_in_chunks(scene_path: str, chunk_size: int = 1_000_000):
    """Example: Apply colormap to huge points."""
    
    root = zarr.open_group(scene_path, mode='r+')
    positions = root['points/positions']
    colors = root['points/colors']
    
    # Process in chunks
    for i in range(0, len(positions), chunk_size):
        chunk_positions = positions[i:i + chunk_size]
        
        # Compute colors based on height
        heights = chunk_positions[:, 2]
        chunk_colors = height_to_color(heights)
        
        # Write back
        colors[i:i + chunk_size] = chunk_colors
```

## 🧪 Testing

```bash
# Run test suite with Hatch
hatch run test

# With coverage
hatch run test-cov

# Run specific test pattern
hatch run test -- -k test_hierarchical_scene
```

## 🛠️ Development

### Code Style

The project enforces strict code quality standards:

```bash
# Format code (use Hatch environment)
hatch run ruff format packages/luxar/src/

# Type checking  
hatch run type-check

# Linting
hatch run lint

# All checks with Hatch
hatch run lint && hatch run type-check
```

### Adding New Features

1. **Design**: Consider nD compatibility and streaming requirements
2. **Implement**: Follow existing patterns in `node.py` and `points.py`
3. **Test**: Add comprehensive tests with >80% coverage
4. **Document**: Update this README and add docstrings
5. **Type**: Ensure full type annotations for mypy strict mode

## 📚 Scientific Integration

### NumPy Compatibility

```python
# Luxar works seamlessly with NumPy
positions = np.random.randn(1000, 3).astype(np.float32)
colors = ((positions - positions.min()) / 
          (positions.max() - positions.min()) * 255).astype(np.uint8)

scene.add_points("normalized", positions, colors)
```

### Pandas DataFrames

```python
import pandas as pd

# Convert DataFrame to points
df = pd.read_csv("measurements.csv")
positions = df[['x', 'y', 'z']].values.astype(np.float32)
colors = df[['r', 'g', 'b']].values.astype(np.uint8)

scene.add_points("measurements", positions, colors)
```

### SciPy Spatial Data

```python
from scipy.spatial import Delaunay

# Future: Triangulated surfaces from points
points = np.random.randn(1000, 3)
tri = Delaunay(points[:, :2])  # 2D triangulation

# When surface support is added:
# scene.add_surface("delaunay", points, tri.simplices)
```

## 🚀 Roadmap

### Near Term
- [ ] Line geometry support
- [ ] Surface mesh support
- [ ] Custom attributes per vertex
- [ ] Time-varying data support

### Medium Term
- [ ] Volume rendering support
- [ ] nD tensor visualization
- [ ] Level-of-detail generation
- [ ] Parallel scene compilation

### Long Term
- [ ] Distributed scene compilation
- [ ] Cloud-optimized formats
- [ ] Real-time data streaming
- [ ] ML-assisted visualization

## 📄 License

Luxar Core is part of the Luxar project. See the main [LICENSE](../../LICENSE) file.

## 🔗 See Also

- [Luxar Viewer README](../luxar-viewer/README.md) - WebGL renderer documentation
- [Main README](../../README.md) - Ecosystem overview
- [Zarr Documentation](https://zarr.readthedocs.io/) - Storage format details
- [Examples](../../examples/) - Code examples and tutorials