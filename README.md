# 🌌 Luxar

A high-performance system for compiling and visualizing arbitrary-sized n-dimensional scenes containing points, lines, surfaces, volumes, and more. Luxar delivers visualization performance limited only by your graphics card, display resolution, and network bandwidth—not by software constraints.

## 🎯 Vision

Luxar reimagines scientific visualization as a hardware-limited problem, not a software-limited one. By separating data compilation from rendering, and using GPU-optimized formats throughout, Luxar ensures that your visualization performance scales with your hardware, not against your software.

## ✨ The Luxar Ecosystem

```
┌─────────────────────┐         ┌──────────────────────┐
│   Python/NumPy      │         │   Web Browser        │
│   Scientific Data   │         │   GPU Rendering      │
│                     │         │                      │
│ ┌─────────────────┐ │         │ ┌──────────────────┐ │
│ │  Luxar Core     │ │  Zarr   │ │  Luxar Viewer    │ │
│ │  (Compiler)     ├─┼────────▶┼─┤  (Renderer)      │ │
│ └─────────────────┘ │         │ └──────────────────┘ │
└─────────────────────┘         └──────────────────────┘
```

### 🐍 [Luxar Core](packages/luxar/README.md)
Python library for compiling n-dimensional scientific datasets into optimized Zarr archives. Handles hierarchical scene organization, coordinate transforms, and efficient chunking strategies.

### 🌐 [Luxar Viewer](packages/luxar-viewer/README.md)  
GPU-accelerated WebGL renderer with HDR pipeline, real-time effects, and streaming capabilities. Delivers maximum performance through custom shaders and progressive loading.

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.9 or higher (usually pre-installed on Linux/macOS)
- **Node.js**: 20.19+ (auto-installed by `make dev-setup`)
- **Modern browser**: WebGL 2.0 support required
- **Ubuntu/Debian only**: Install pipx first: `sudo apt-get install -y pipx`

### One-Command Setup + Demo

```bash
# Clone and set up development environment
git clone https://github.com/royerlab/luxar.git
cd luxar
make dev-setup      # Auto-installs Node.js, pnpm, Hatch (no sudo needed)
make demo-and-serve # Creates demo and starts viewer

# Opens browser with a beautiful Lorenz attractor visualization
```

### Basic Workflow

```bash
# 1. Set up environment (first time only)
make dev-setup

# 2. Create a scene in Python
hatch run luxar demo --no-serve --output my_scene.zarr --points 1000000

# 3. Visualize in browser (run in separate terminals)
make viewer                            # Starts viewer at http://localhost:5173
make serve-data DATASET=my_scene.zarr  # Serves your data at http://localhost:8000
```

### Python Example

```python
import numpy as np
from luxar import LuxarZarrCompiler, Dimensions, Dimension, transforms

# Create scene with explicit dimensions
dims = Dimensions([
    Dimension("time", unit="s", range=(0, 10), step=0.5, display=False),
    Dimension("x", unit="um", range=(-100, 100), display=True),
    Dimension("y", unit="um", range=(-100, 100), display=True),
    Dimension("z", unit="um", range=(-50, 50), display=True),
])

# Use LuxarZarrCompiler for progressive writing
with LuxarZarrCompiler("output.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    
    # Add 4D points (time + xyz)
    positions = np.random.randn(100_000, 4).astype(np.float32)
    colors = np.random.rand(100_000, 3).astype(np.float32)  # HDR colors supported
    radii = np.random.uniform(0.1, 0.5, 100_000).astype(np.float32)
    scene.add_points("TimeSeriesPoints", positions, colors=colors, radii=radii)
    
    # Add transformed point groups
    transform = transforms.compose(
        transforms.translate(5, 0, 0),  # Move 5 units along X
        transforms.rotate_z(np.pi/4),    # Rotate 45° around Z
        transforms.scale(0.5, 0.5, 0.5)  # Scale down by half
    )
    group = scene.add_group("TransformedData", transform=transform)
    
    # Context manager handles finalization automatically

# View with nD navigation: luxar serve output.zarr
# Use keyboard: Press '1' to select time dimension, '[' and ']' to navigate
```

📚 **For detailed usage, see package-specific documentation:**
- [Python API Guide](packages/luxar/README.md#-quick-start)
- [Viewer Setup Guide](packages/luxar-viewer/README.md#-quick-start)

## 🏗️ Architecture

### Why Two Components?

1. **Language Optimization**: Python for data science, JavaScript for GPU
2. **Scalability**: Compile once, visualize anywhere
3. **Flexibility**: Mix and match tools for your workflow
4. **Performance**: Each component optimized for its domain

### Data Flow

```
Your Data → Luxar Core → Zarr Archive → Luxar Viewer → GPU → Display
   │            │             │              │
   │            │             │              └─ WebGL 2.0 shaders
   │            │             └─ Chunked, compressed, streamable
   │            └─ Scene graph, transforms, metadata
   └─ NumPy arrays, DataFrames, any Python data
```

### Project Structure

```
luxar/
├── packages/
│   ├── luxar/                 # Python scene compiler
│   │   ├── README.md         # Python package docs
│   │   └── src/luxar/        # Source code
│   └── luxar-viewer/         # WebGL renderer  
│       ├── README.md         # Viewer docs
│       └── src/              # TypeScript source
├── Makefile                  # Convenient commands
├── pyproject.toml            # Python configuration
└── README.md                 # This file
```

## 📋 Key Features

### 🐍 Luxar Core
- **Universal Scene Graph** - Supports points, lines, surfaces, volumes, and nD data
- **Scene-Level Dimensions** - Define coordinate systems with units, ranges, and navigation steps
- **Zarr Backend** - Chunked storage for streaming massive datasets
- **Python Native** - Integrates with NumPy, Pandas, and scientific Python
- **CLI Tools** - Command-line interface for quick operations

📚 [Full Python Documentation →](packages/luxar/README.md)

### 🌐 Luxar Viewer
- **GPU Acceleration** - WebGL 2.0 with custom shaders
- **WASM Acceleration** - Rust-on-WASM hot function acceleration
- **nD Navigation** - Browse through multiple dimensions with keyboard controls
- **Radius-Based Slicing** - Natural point visibility based on hypersphere intersections
- **HDR Rendering** - 16-bit precision with bloom effects
- **Streaming Ready** - Progressive loading of large datasets
- **Cross-Platform** - Runs in any modern web browser

📚 [Full Viewer Documentation →](packages/luxar-viewer/README.md)


## 🔧 Development

### Prerequisites

The build system is designed to work on **fresh Linux and macOS machines** with minimal pre-installed tools. All other dependencies are installed automatically.

**Minimal requirements:**
- Python 3.9+ (usually pre-installed)
- Git and curl

**Ubuntu/Debian only** (due to PEP 668):
```bash
sudo apt-get install -y pipx && pipx ensurepath && source ~/.bashrc
```

### Quick Setup

```bash
# Complete setup - auto-installs everything (no sudo needed for most tools)
make dev-setup

# Check what's installed
make check-deps

# Run all quality checks
make check

# See all available commands
make help
```

`make dev-setup` automatically installs:
- **Python**: Hatch (via pipx) for environment management
- **Node.js**: v22+ via nvm (Linux) or Homebrew (macOS) - no sudo needed
- **TypeScript**: pnpm and all npm dependencies
- **Pre-commit hooks**: Automatic code quality checks

### Development Resources

- 📚 [Build System Guide](docs/guides/developer/BUILD_SYSTEM_SPEC.md) - Complete Makefile documentation
- 📚 [Python Development Guide](packages/luxar/README.md#-development) - Testing, code style, extending
- 📚 [Viewer Development Guide](packages/luxar-viewer/README.md#-development) - TypeScript, WebGL, shaders
- 📚 [Contributing Guidelines](CONTRIBUTING.md) - How to contribute

### Key Commands

| Task | Command |
|------|---------|
| **Setup** | |
| Full dev setup | `make dev-setup` |
| Check dependencies | `make check-deps` |
| Setup Rust/WASM | `make setup-rust` |
| **Quality** | |
| All checks | `make check` |
| Format code | `make format-all` |
| Run all tests | `make test-all` |
| **Viewer** | |
| Start dev server | `make viewer` |
| Build for production | `make viewer-build` |
| **Data** | |
| Create demo | `make demo-and-serve` |
| Run examples | `make run-examples` |

### Rust/WASM Support (Optional)

WASM acceleration is optional. The viewer works without it (uses TypeScript fallback):

```bash
make setup-rust    # Install Rust + wasm-pack
make wasm-build    # Build WASM module
make wasm-test     # Run Rust tests
```

## 📋 Data Format

Luxar uses Zarr for efficient, chunked storage of large points datasets with support for arbitrary dimensionality.

### Zarr Structure

```
scene.zarr/
├── .zattrs                 # Scene-level metadata (version, dimensions, units)
├── .zgroup                 # Zarr group marker
├── .zmetadata             # Consolidated metadata (created by finalize())
└── <node_name>/           # Scene nodes (groups or points)
    ├── .zattrs            # Node metadata (type, transform, rendering)
    ├── .zgroup            # Zarr group marker
    ├── positions/         # nD coordinates (Float32, shape: [N, D])
    │   ├── .zarray
    │   └── [chunks...]
    ├── colors/            # RGB colors (Uint8, shape: [N, 3]) - optional
    │   ├── .zarray
    │   └── [chunks...]
    ├── radii/             # Point radii (Float32, shape: [N]) - optional
    │   ├── .zarray
    │   └── [chunks...]
    ├── sharpness/         # Point edge sharpness (Float32, shape: [N]) - optional
    │   ├── .zarray
    │   └── [chunks...]
    └── <child_nodes>/     # Nested child nodes (recursive structure)
```

### Attributes Schema

```json
{
  "luxar_version": "0.3",
  "type": "scene|group|points",
  "units": "um",  // Physical units
  "scene_dimensions": {  // Scene-level coordinate system
    "dimensions": [
      {
        "name": "x",
        "unit": "um",
        "range": [-100.0, 100.0],
        "step": 1.0,
        "display": true
      }
      // ... more dimensions
    ]
  },
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],  // 4x4 matrix
  "opacity": 1.0,  // 0.0-1.0
  "gamma": 1.0,    // 0.2-2.0
  "blending_mode": "additive"  // normal|additive|max
}
```

📚 **For complete format specification, see [LUXAR_ZARR_FORMAT.md](docs/guides/user/LUXAR_ZARR_FORMAT.md)**

### Performance Recommendations

- **Chunk Size**: Default 32KB elements, optimal range 64KB-1MB per chunk
- **Compression**: Blosc with zstd level 3 and bit-shuffle for scientific data
- **Point Count**: 100K-10M points per scene for smooth interaction
- **Data Types**: Float32 for positions/radii/sharpness, Uint8 for colors
- **Dimensionality**: Supports arbitrary nD points, viewer displays 3D slices

## 🎯 Use Cases

### Scientific Visualization

```python
from luxar import LuxarZarrCompiler, Dimensions

# Visualize experimental data points
positions = load_experimental_coordinates()  # Your data loading
colors = map_values_to_colors(experimental_values)

with LuxarZarrCompiler("experiment_results.zarr") as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.from_positions(positions))
    scene.add_points("ExperimentData", positions, colors=colors)
```

### Geographic Data

```python
from luxar import LuxarZarrCompiler, Dimensions

# Visualize GPS coordinates with elevation
lat_lon_alt = load_gps_data()
positions = convert_to_cartesian(lat_lon_alt)
colors = elevation_to_color_map(lat_lon_alt[:, 2])

with LuxarZarrCompiler("geographic_data.zarr") as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.from_positions(positions))
    scene.add_points("GPSData", positions, colors=colors)
```

### Synthetic Datasets

```python
from luxar import LuxarZarrCompiler, Dimensions

# Generate procedural points
def generate_fractal_points(iterations=5, scale=2.0):
    # Your fractal algorithm here
    return positions, colors

positions, colors = generate_fractal_points()

with LuxarZarrCompiler("fractal.zarr") as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.from_positions(positions))
    scene.add_points("FractalPoints", positions, colors=colors)
```

## 🚨 Troubleshooting

### Common Issues

**ImportError: No module named 'luxar'**
```bash
make dev-setup    # Sets up complete environment including luxar
# Or if you have hatch installed:
hatch shell       # Activates the development environment
```

**Viewer shows white screen**
- Check browser console for errors
- Verify Zarr dataset is accessible
- Ensure CORS headers if serving from different domain

**Poor rendering performance**
- Reduce point count (<1M points recommended)
- Lower bloom quality settings
- Verify GPU acceleration is enabled

**Zarr loading errors**
- Validate dataset structure with `luxar info dataset.zarr`
- Check positions/colors array shapes and data types
- Ensure proper Zarr metadata files exist

**Development Tool Issues**
```bash
# Pre-commit hooks failing
make pre-commit-run  # Run manually to see specific errors

# Type checking errors
make type-check     # Run mypy to see detailed type issues

# Test failures
make test-cov      # Run tests with detailed output and coverage

# Code formatting issues
make format        # Auto-fix most formatting problems

# Clean slate
make clean         # Remove all temporary files and caches
```

### Browser Compatibility

| Browser | Version | WebGL 2.0 | Status |
|---------|---------|-----------|--------|
| Chrome | 90+ | ✅ | Fully supported |
| Firefox | 88+ | ✅ | Fully supported |
| Safari | 14+ | ✅ | Supported |
| Edge | 90+ | ✅ | Fully supported |

## 📊 Performance Benchmarks

### Dataset Size Guidelines

| Points | Memory Usage | Load Time | Interaction |
|--------|-------------|-----------|-------------|
| 100K | ~5MB | <1s | Smooth (60 FPS) |
| 1M | ~50MB | ~3s | Good (30-60 FPS) |
| 10M | ~500MB | ~15s | Acceptable (15-30 FPS) |
| 100M+ | ~5GB | >60s | Requires optimization |

### Optimization Strategies

1. **Chunk Strategy**: Use roughly cubic chunks (1000³ points)
2. **Level of Detail**: Implement LOD for very large datasets
3. **Compression**: Enable blosc compression in Zarr
4. **Culling**: Consider frustum culling for complex scenes
5. **Streaming**: Implement progressive loading for massive datasets

## 🌐 Network Simulation Testing

Test your viewer's performance under realistic network conditions using built-in network simulation:

### Quick Start

```bash
# List available network profiles
luxar profiles

# Test with 3G mobile connection
luxar serve data.zarr --profile 3g --viewer --open

# Test with custom parameters
luxar serve data.zarr --bandwidth 500kbps --latency 200ms --jitter 10%

# Demo with network simulation
luxar demo --profile satellite --open
```

### Available Profiles

| Profile | Bandwidth | Latency | Jitter | Packet Loss | Use Case |
|---------|-----------|---------|--------|-------------|----------|
| `3g` | 384 kbps | 300 ms | 10% | 1% | Slow mobile |
| `4g` | 10 mbps | 100 ms | 10% | 0.5% | Typical mobile |
| `5g` | 100 mbps | 30 ms | 5% | 0.1% | Modern mobile |
| `broadband` | 50 mbps | 20 ms | 5% | 0.1% | Home internet |
| `satellite` | 25 mbps | 600 ms | 15% | 1% | High latency |
| `rural` | 1 mbps | 100 ms | 20% | 2% | Poor connection |
| `congested` | 2 mbps | 200 ms | 25% | 3% | Network overload |

### Simulation Parameters

- **`--profile <name>`** - Use a preset connection profile
- **`--bandwidth <value>`** - Limit bandwidth (e.g., '1mbps', '500kbps', '10mbps')
- **`--latency <value>`** - Add network latency (e.g., '100ms', '500ms', '1s')
- **`--jitter <value>`** - Add latency variation (e.g., '10%', '0.1')
- **`--packet-loss <value>`** - Simulate dropped requests (e.g., '1%', '0.01')

Individual parameters override profile defaults:

```bash
# Use 4G profile but with higher latency
luxar serve data.zarr --profile 4g --latency 300ms --viewer
```

### Use Cases

**Performance Testing**: How does the viewer handle slow connections?
```bash
luxar serve large_dataset.zarr --profile 3g --viewer
```

**Cache Validation**: Does caching reduce redundant requests?
```bash
luxar serve data.zarr --bandwidth 100kbps --viewer
# Monitor browser DevTools Network tab
```

**UX Research**: What's the minimum viable bandwidth?
```bash
# Test progressively slower connections
luxar serve data.zarr --bandwidth 2mbps --viewer
luxar serve data.zarr --bandwidth 1mbps --viewer
luxar serve data.zarr --bandwidth 500kbps --viewer
```

**Regression Testing**: Did changes affect loading performance?
```bash
# Before and after comparisons
luxar serve data.zarr --profile broadband --viewer
```

⚠️ **Note**: Network simulation is for **development and testing only**. Never use in production.

## 🤝 Contributing

We welcome contributions! The project uses automated development tooling to maintain high code quality.

### Quick Start for Contributors

```bash
# Complete development setup
make dev-setup

# Make your changes, then run quality checks
make check && make test-cov

# Commit (pre-commit hooks ensure quality)
git commit -m "Your contribution"
```

For detailed contributing guidelines, development setup, coding standards, and more, see **[CONTRIBUTING.md](CONTRIBUTING.md)**.

### Development Standards
- **Code Quality**: Automatic formatting, linting, and type checking
- **Testing**: 80%+ coverage requirement with comprehensive test suite
- **Security**: Automated vulnerability scanning
- **Documentation**: Google-style docstrings for all public APIs

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- **Three.js** - WebGL 3D rendering engine
- **Zarr** - Chunked, compressed array storage
- **Zarrita** - JavaScript Zarr implementation
- **NumPy** - Numerical computing in Python
- **FastAPI** - Modern Python web framework
- **Vite** - Fast frontend development tooling

## 📞 Support & Resources

- **Documentation**: [Detailed API docs and tutorials]
- **Examples**: See `packages/luxar/examples/` directory
- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [Community forum](link-to-discussions)
- **Performance Tips**: [Optimization guide](link-to-performance-guide)

## 🎓 Learning Resources

### Getting Started
1. Run the basic demo: `hatch run luxar demo --no-serve --output demo.zarr --points 50000`
2. Start the viewer: `make viewer`
3. Serve and load data: `make serve-data DATASET=demo.zarr` then open `http://localhost:5173/?src=http://localhost:8000`

### Advanced Topics
- Custom shader development for specialized rendering
- Implementing level-of-detail systems
- Building interactive data exploration tools
- Integrating with scientific computing workflows

---

Built with ❤️ for the scientific visualization and data exploration community

**Ready to visualize your data in 3D? Start with `luxar demo` and explore! 🚀**