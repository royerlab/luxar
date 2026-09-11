# 🐍 Luxar Core - Python nD Scene Compiler

A Python library for compiling n-dimensional scientific datasets into optimized Zarr archives for high-performance visualization. Luxar Core provides the data compilation layer of the Luxar ecosystem, transforming your scientific data into a format optimized for GPU-accelerated rendering.

**[▶ See what it compiles to](https://demos.luxarviewer.dev)** — 90 demos rendered as live, interactive scenes in the browser.

## 🎯 Design Philosophy

Luxar Core is built on the principle that **data compilation should never be the bottleneck**. Whether you're working with thousands or billions of data points, in 3D or higher dimensions, Luxar Core scales with your ambitions:

- **Dimension-agnostic**: Built to handle 3D, 4D, and beyond
- **Geometry-flexible**: Extensible architecture for points, lines, Gaussian splats, and more
- **Scale-unlimited**: Performance bounded by storage, not architecture
- **Stream-ready**: Chunked Zarr format enables progressive loading

## 📦 Installation

```bash
# From PyPI (when available)
pip install luxar

# Optional Gaussian splatting support
pip install "luxar[gsplats]"

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

with LuxarZarrCompiler("my_dataset.luxar.zarr") as compiler:
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
# luxar serve my_dataset.luxar.zarr
```

## 🏗️ Architecture

### Scene Graph

Luxar uses a hierarchical scene graph that mirrors Zarr's group structure:

```
Scene (root)
├── Group
│   ├── Points
│   ├── Lines
│   ├── GSplats
│   ├── Mesh
│   └── Group (nested)
└── Points
```

Each node can have:
- **Transform**: 4x4 transformation matrix (displayed spatial dimensions)
- **nD Transform**: Per-dimension affine or permutation transforms (non-displayed dimensions)
- **Attributes**: Arbitrary metadata
- **Datasets**: Geometry-specific data arrays

### Node Types

#### Currently Implemented
- **Scene**: Root node managing the Zarr store
- **Group**: Organizational nodes with transforms
- **Points**: Point geometry with positions, colors, radii, sharpness
- **Lines**: Line/curve geometry with vertices, widths, colors, sharpness
- **GSplats**: Gaussian splat geometry with centers, amplitudes, cholesky factors, colors
- **Mesh**: Triangle surfaces with faces, optional per-vertex normals (plus the required `normal_dims` companion), colors, scalars
- **Overlay**: 2D overlay geometry for annotations and labels

#### Planned Geometry Types
- **Volumes**: Volumetric data with transfer functions
- **Tensors**: Higher-dimensional data with projections

### Data Format

Luxar uses Zarr v2 format for maximum compatibility:

```
dataset.luxar.zarr/
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

with LuxarZarrCompiler("multidimensional.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dimensions)
    # ... add data to scene ...
```

### Dimension Types

- **Displayed Dimensions**: The 3D subset shown in the viewer (max 3)
- **Non-Displayed Dimensions**: Additional dimensions navigated via UI sliders
- **Discrete Dimensions**: Integer steps for frame-based data
- **Continuous Dimensions**: Smooth navigation for continuous variables

### Point Attributes

Enhanced point rendering with per-point attributes:

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
sharpness = np.random.uniform(0.0, 1.0, n_points).astype(np.float32)

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
- Larger radius = visible across more slices only for dimensions declared `spatial=True` (non-displayed dimensions default to non-spatial)
- Natural representation of uncertainty or spread in higher dimensions

### nD Transforms

Non-displayed dimensions (Time, Channel, etc.) support per-dimension transforms for dataset alignment:

```python
# Align a dataset captured in milliseconds to a scene using seconds
group = scene.add_group(
    "DatasetB",
    transform=transforms.translate(10, 0, 0),     # spatial alignment
    nd_transform={
        "Time": {"scale": 0.001, "offset": 50.0}, # ms to seconds, shifted
        "Channel": {"permutation": [2, 1, 0]},     # remap categories
    },
)
```

- **Continuous/discrete dimensions**: affine transforms (`scale` + `offset`)
- **Categorical dimensions**: permutation maps (index remapping)
- Compose hierarchically through the scene graph, just like spatial transforms
- Viewer uses inverse-query approach (O(1) per dimension, not O(N) per point)

### Gaussian Splatting

Luxar includes a complete Gaussian splatting pipeline for volumetric data:

- **Tiled fitting**: Split large volumes into overlapping tiles, fit independently, merge results
- **Quality metrics**: PSNR, SSIM, and normalized cross-correlation for comparing fitted splats against source volumes
- **CLI tools**: `luxar gsplat fit`, `luxar gsplat render`, `luxar gsplat merge`, `luxar gsplat filter`, `luxar gsplat slice`

```python
import torch
from luxar.gsplats import fit_gaussian_splats
from luxar.gsplats.metrics import compute_psnr, compute_ssim

result = fit_gaussian_splats(volume, device='cuda')
rendered = result.render_to_volume(volume.shape)
# Metrics require torch tensors
vol_t = torch.as_tensor(volume, dtype=torch.float32)
ren_t = torch.as_tensor(rendered, dtype=torch.float32)
print(f"PSNR: {compute_psnr(ren_t, vol_t):.1f} dB")
print(f"SSIM: {compute_ssim(ren_t, vol_t):.4f}")
```

## 📖 API Reference

### LuxarZarrCompiler

The compiler is the entry point for creating Luxar scenes with progressive writing.

```python
class LuxarZarrCompiler:
    """Progressive Zarr compiler with context manager support."""

    def __init__(
        self,
        store_path: Optional[PathLike] = None,
        compressor: Optional[Compressor] = DEFAULT_COMP,
        version: str = LUXAR_VERSION_CURRENT,
        enable_spatial_index: bool = True,
        encoding_mode: EncodingMode = EncodingMode.AUTO,
        ordering_method: Literal["morton", "hilbert"] = "hilbert",
        float16_allowed: bool = False,
        auto_partition_max_elements: Optional[int] = None,
    ) -> None:
        """
        Create a new Zarr compiler for progressive writing.

        Args:
            store_path: Path to Zarr store (None for temporary directory)
            compressor: Zarr compressor for datasets
            version: Luxar format version
            enable_spatial_index: Whether to build spatial indices (default: True)
            encoding_mode: Encoding mode (AUTO/PRECISION/MEMORY)
            ordering_method: Spatial ordering ("morton" or "hilbert", default: "hilbert")
            float16_allowed: Whether to allow float16 encoding (default: False)
            auto_partition_max_elements: If set, add_points/add_gsplats auto-apply
                partition=dict(max_elements=N) when the element count exceeds N
                (explicit partition= at the call site always wins; default None)
        """

    def create_scene(self, dimensions: Dimensions) -> Scene:
        """Create a scene with this compiler as writer.

        Args:
            dimensions: Scene-level dimension definitions (REQUIRED)

        Returns:
            Scene object for building the scene graph
        """
```

### Scene Class

```python
class Scene:
    """Root scene node. Created via LuxarZarrCompiler.create_scene()."""

    def add_group(self, name: str, **attrs) -> Node:
        """Add a group node for organization.

        Args:
            name: Group node name
            **attrs: Additional attributes including:
                opacity: float (0.0-1.0, default 1.0) - Node opacity
                gamma: float (0.1-10.0, default 1.0) - Gamma correction
                blending_mode: str ("normal", "additive", "max", "opaque", "luminous", default "additive")
                transform: np.ndarray or list[float] - 4x4 transformation matrix (row-major)
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
            sharpness: Edge falloff, normalized [0, 1] (0.5 = Gaussian)
            parent: Parent node in hierarchy
            **attrs: Additional attributes including:
                opacity: float (0.0-1.0, default 1.0) - Node opacity
                gamma: float (0.1-10.0, default 1.0) - Gamma correction
                blending_mode: str ("normal", "additive", "max", "opaque", "luminous", default "additive")
                transform: np.ndarray or list[float] - 4x4 transformation matrix (row-major)
        """

    # Finalization is handled automatically by the LuxarZarrCompiler context manager
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
    """Point geometry node."""

    # Created automatically via Scene.add_points()
    # Manages positions, colors, radii, and sharpness datasets in Zarr
```

### Dimensions Class

```python
class Dimensions:
    """Container for scene-level dimension definitions."""

    def __init__(self, dimensions: List[Dimension]) -> None:
        """Create dimensions from list of Dimension objects."""

    @classmethod
    def default_3d(cls) -> Dimensions:
        """Create default 3D dimensions (x, y, z in generic units)."""

    @classmethod
    def default_timeseries(cls, n_timepoints: int = 100, time_unit: str = "s") -> Dimensions:
        """Create default time series dimensions (t, x, y, z)."""

    def validate_positions(self, positions: np.ndarray) -> None:
        """Validate that positions match dimension count."""
```

### Dimension Class

```python
@dataclass
class Dimension:
    """Single dimension definition with metadata.

    Attributes:
        name: Dimension name (e.g., "x", "time", "channel")
        unit: Physical unit (e.g., "μm", "s", "nm")
        range: Min/max values as (min, max)
        step: Step size for navigation (None = auto-calculate)
        display: Whether shown in 3D viewer (max 3)
        discrete: Whether dimension has discrete steps
        cyclic: Whether dimension wraps around (for angles, periodic states)
        scale: Scale factor for unit conversion (default 1.0)
        spatial: Whether points extend through this dimension (None = auto-determine)
        categories: Optional category labels for categorical dimensions
        description: Optional human-readable description
    """

    name: str
    unit: str = ""
    range: Optional[Tuple[float, float]] = None
    step: Optional[float] = None
    display: bool = True
    discrete: bool = False
    cyclic: bool = False
    scale: float = 1.0
    spatial: Optional[bool] = None
    categories: CategoryList = None
    description: str = ""
```

## 🖥️ Command Line Interface

Luxar provides a CLI for common operations:

```bash
# Serve a Luxar scene for visualization
luxar serve dataset.luxar.zarr

# Create a demo dataset without serving
luxar demo run lorenz -- --no-serve --points=100000

# Get information about a dataset
luxar info dataset.luxar.zarr
```

**Serving datasets:**
- `luxar serve <path.luxar.zarr>` - Start HTTP server for dataset
- Use `--port` to specify port (default: 8000)
- Use `--host` to bind to specific interface

**Creating demo data:**
- `luxar demo` - Create and serve demo with viewer (opens browser)
- `luxar demo run lorenz -- --no-serve` - Generate a demo scene without serving
- Use `--points` to specify number of points (default: 10,000)
- Use `--seed` for reproducible results

## 🎨 Rendering Attributes

Control the visual appearance of nodes with rendering attributes:

```python
from luxar import LuxarZarrCompiler, Dimensions

dims = Dimensions.default_3d()
with LuxarZarrCompiler("styled_scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Add points with custom rendering attributes
    scene.add_points(
        "StyledPoints",
        positions,
        colors=colors,
        opacity=0.8,           # Semi-transparent (0.0-1.0)
        gamma=1.5,             # Brighter gamma correction (0.1-10.0)
        blending_mode="normal" # Use normal blending instead of additive
    )

    # Apply to groups as well
    group = scene.add_group(
        "TransparentGroup",
        opacity=0.6,
        blending_mode="normal"
    )
```

**Blending Modes:**
- `"additive"` (default): HDR additive blending, ignores depth (renders on top of everything)
- `"normal"`: Standard alpha blending (semi-transparent)
- `"max"`: Maximum blending, brightest values win
- `"opaque"`: Solid rendering with depth write (closest object wins)
- `"luminous"`: Same as additive visually, but respects depth occlusion

## 🔬 Advanced Usage

### Custom Chunking Strategy

```python
from luxar import LuxarZarrCompiler, Dimensions

dims = Dimensions.default_3d()

positions = np.random.randn(10_000_000, 3).astype(np.float32)
colors = np.random.rand(10_000_000, 3).astype(np.uint8)

# Luxar automatically handles chunking and spatial ordering
with LuxarZarrCompiler("large_dataset.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_points("LargeCloud", positions, colors)
```

### Hierarchical Data Organization

```python
from luxar import LuxarZarrCompiler, Dimensions

dims = Dimensions.default_3d()
with LuxarZarrCompiler("experiment.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)

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
```

### Transforms and Coordinate Systems

Luxar provides comprehensive transform utilities for 3D scene manipulation:

```python
from luxar import LuxarZarrCompiler, Dimensions, transforms

dims = Dimensions.default_3d()
with LuxarZarrCompiler("transformed_scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Basic transforms
    translation = transforms.translate(10, 5, 0)      # Move 10 units in X, 5 in Y
    rotation = transforms.rotate(45, 'z')             # Rotate 45° around Z axis
    scaling = transforms.scale(2, 2, 2)               # Scale 2x in all dimensions
    uniform_scale = transforms.scale(uniform=0.5)     # Scale uniformly by 0.5

    # Compose multiple transforms (applied left-to-right)
    combined = transforms.compose(translation, rotation, scaling)

    # Apply transforms to groups (via transform attribute)
    group = scene.add_group("MyGroup", transform=combined)

    # Hierarchical transforms (child inherits parent transform)
    parent = scene.add_group("Robot", transform=transforms.translate(100, 0, 0))
    scene.add_group("Sensor", parent=parent, transform=transforms.rotate_y(90))

# Transform utilities (standalone, no scene needed)
look_at = transforms.look_at(
    eye=(10, 10, 10),     # Camera position
    target=(0, 0, 0),     # Look at origin
    up=(0, 1, 0)          # Y-up
)

t = transforms.translate(5, 0, 0)
t_inv = transforms.inverse(t)  # Translates -5, 0, 0
```

### Using Lines and GSplats

Lines and Gaussian splats are first-class geometry types alongside points:

```python
from luxar import LuxarZarrCompiler, Dimensions

dims = Dimensions.default_3d()
with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Add line geometry
    vertices = np.array([[0,0,0],[1,1,0],[2,0,0]], dtype=np.float32)
    widths = np.array([0.1, 0.05, 0.1], dtype=np.float32)
    scene.add_lines("MyLine", vertices, widths=widths)

    # Add Gaussian splats (from a volume)
    scene.add_gsplats_from_volume("MySplats", volume_data)
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
from luxar import LuxarZarrCompiler, Dimensions

# For maximum speed (local visualization)
fast_compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)

# For network streaming (bandwidth-limited)
balanced_compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)

# For archival storage (space-limited)
compact_compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE)

dims = Dimensions.default_3d()
with LuxarZarrCompiler("data.luxar.zarr", compressor=balanced_compressor) as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_points("cloud", positions, colors)
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
3. **Test**: Add comprehensive tests that preserve the enforced
   `[tool.coverage.report] fail_under` threshold in `pyproject.toml`
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

# Compute Delaunay triangulation
points = np.random.randn(1000, 3)
tri = Delaunay(points[:, :2])  # 2D triangulation

# Visualize the triangulation vertices as points
scene.add_points("delaunay_vertices", points.astype(np.float32))
```

## Roadmap

### Near Term
- [x] Line geometry support
- [x] Gaussian splat geometry support
- [x] Surface mesh support
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
- [Examples](examples/) - Code examples and tutorials
