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
from luxar import Scene

# Create a scene
scene = Scene("my_dataset.zarr")

# Add point cloud data
positions = np.random.randn(1_000_000, 3).astype(np.float32)
colors = (np.random.rand(1_000_000, 3) * 255).astype(np.uint8)
scene.add_points("PointCloud", positions, colors)

# Add hierarchical organization
group = scene.add_group("Experiment1")
scene.add_points("Measurement", positions2, colors2, parent=group)

# Finalize (consolidates metadata for fast loading)
scene.finalize()
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
        compressor: Optional[Compressor] = DEFAULT_COMP
    ) -> None:
        """
        Create a new scene.
        
        Args:
            store_path: Path to Zarr store (None for temporary)
            units: Physical units for coordinates
            version: Luxar format version
            compressor: Zarr compressor for datasets
        """
    
    def add_group(self, name: str, **attrs) -> Node:
        """Add a group node for organization."""
    
    def add_points(
        self, 
        name: str,
        positions: np.ndarray,  # shape: (N, 3), dtype: float32
        colors: Optional[np.ndarray] = None,  # shape: (N, 3), dtype: uint8
        parent: Optional[Node] = None,
        **attrs
    ) -> Points:
        """Add a point cloud to the scene."""
    
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
    # Manages positions and colors datasets in Zarr
```

## 🔬 Advanced Usage

### Custom Chunking Strategy

```python
from luxar import Scene
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
scene.finalize()
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

scene.finalize()
```

### Transforms and Coordinate Systems

Luxar provides comprehensive transform utilities for 3D scene manipulation:

```python
from luxar import Scene, transforms

scene = Scene("transformed_scene.zarr")

# Basic transforms
translation = transforms.translate(10, 5, 0)      # Move 10 units in X, 5 in Y
rotation = transforms.rotate(45, 'z')             # Rotate 45° around Z axis
scaling = transforms.scale(2, 2, 2)               # Scale 2x in all dimensions
uniform_scale = transforms.scale(uniform=0.5)     # Scale uniformly by 0.5

# Compose multiple transforms (applied left-to-right)
combined = transforms.compose(translation, rotation, scaling)

# Apply transforms to groups
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

scene.finalize()
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
    """Example: Apply colormap to huge point cloud."""
    
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
# Run test suite
pytest packages/luxar/src/luxar/tests/

# With coverage
pytest --cov=luxar --cov-report=html

# Run specific test
pytest -xvs tests/test_scene_structure.py::test_hierarchical_scene
```

## 🛠️ Development

### Code Style

The project enforces strict code quality standards:

```bash
# Format code
black packages/luxar/src/
isort packages/luxar/src/

# Type checking
mypy packages/luxar/src/luxar/

# Linting
flake8 packages/luxar/src/

# All checks
make check
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

# Convert DataFrame to point cloud
df = pd.read_csv("measurements.csv")
positions = df[['x', 'y', 'z']].values.astype(np.float32)
colors = df[['r', 'g', 'b']].values.astype(np.uint8)

scene.add_points("measurements", positions, colors)
```

### SciPy Spatial Data

```python
from scipy.spatial import Delaunay

# Future: Triangulated surfaces from point clouds
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

- [Luxar Player README](../luxar-player/README.md) - WebGL renderer documentation
- [Main README](../../README.md) - Ecosystem overview
- [Zarr Documentation](https://zarr.readthedocs.io/) - Storage format details
- [Examples](examples/) - Code examples and tutorials