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
│ │  Luxar Core     │ │  Zarr   │ │  Luxar Player    │ │
│ │  (Compiler)     ├─┼────────▶┼─┤  (Renderer)      │ │
│ └─────────────────┘ │         │ └──────────────────┘ │
└─────────────────────┘         └──────────────────────┘
```

### 🐍 [Luxar Core](packages/luxar/README.md)
Python library for compiling n-dimensional scientific datasets into optimized Zarr archives. Handles hierarchical scene organization, coordinate transforms, and efficient chunking strategies.

### 🌐 [Luxar Player](packages/luxar-player/README.md)  
GPU-accelerated WebGL renderer with HDR pipeline, real-time effects, and streaming capabilities. Delivers maximum performance through custom shaders and progressive loading.

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.9 or higher
- **Node.js**: 18 or higher
- **Modern browser**: WebGL 2.0 support required

### One-Command Demo

```bash
# Clone and see Luxar in action
git clone https://github.com/royerlab/luxar.git
cd luxar
make demo-and-serve

# Opens browser with a beautiful Lorenz attractor visualization
```

### Basic Workflow

```bash
# 1. Install Luxar
pip install -e .

# 2. Create a scene in Python
luxar random --out my_scene.zarr --n 1000000

# 3. Visualize in browser
make viewer  # Starts viewer at http://localhost:5173
make serve-data DATASET=my_scene.zarr  # Serves your data
```

### Python Example

```python
import numpy as np
from luxar import Scene

# Create and populate a scene
scene = Scene("output.zarr")
scene.add_points(
    "MyData",
    positions=np.random.randn(1_000_000, 3).astype(np.float32),
    colors=np.random.randint(0, 255, (1_000_000, 3), dtype=np.uint8)
)
scene.finalize()

# View: luxar serve output.zarr
```

📚 **For detailed usage, see package-specific documentation:**
- [Python API Guide](packages/luxar/README.md#-quick-start)
- [Viewer Setup Guide](packages/luxar-player/README.md#-quick-start)

## 🏗️ Architecture

### Why Two Components?

1. **Language Optimization**: Python for data science, JavaScript for GPU
2. **Scalability**: Compile once, visualize anywhere
3. **Flexibility**: Mix and match tools for your workflow
4. **Performance**: Each component optimized for its domain

### Data Flow

```
Your Data → Luxar Core → Zarr Archive → Luxar Player → GPU → Display
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
│   └── luxar-player/         # WebGL renderer  
│       ├── README.md         # Viewer docs
│       └── src/              # TypeScript source
├── Makefile                  # Convenient commands
├── pyproject.toml            # Python configuration
└── README.md                 # This file
```

## 📋 Key Features

### 🐍 Luxar Core
- **Universal Scene Graph** - Supports points, lines, surfaces, volumes, and nD data
- **Zarr Backend** - Chunked storage for streaming massive datasets
- **Python Native** - Integrates with NumPy, Pandas, and scientific Python
- **CLI Tools** - Command-line interface for quick operations

📚 [Full Python Documentation →](packages/luxar/README.md)

### 🌐 Luxar Player
- **GPU Acceleration** - WebGL 2.0 with custom shaders
- **HDR Rendering** - 16-bit precision with bloom effects
- **Streaming Ready** - Progressive loading of large datasets
- **Cross-Platform** - Runs in any modern web browser

📚 [Full Viewer Documentation →](packages/luxar-player/README.md)

## 🔧 Development

### Quick Setup

```bash
# Complete setup with all tools
make dev-setup

# Run all quality checks
make check

# See all available commands
make help
```

### Development Resources

- 📚 [Python Development Guide](packages/luxar/README.md#-development) - Testing, code style, extending
- 📚 [Viewer Development Guide](packages/luxar-player/README.md#-development) - TypeScript, WebGL, shaders
- 📚 [Contributing Guidelines](CONTRIBUTING.md) - How to contribute

### Key Commands

| Task | Command |
|------|---------|
| Format code | `make format` |
| Run tests | `make test` |
| Type check | `make type-check` |
| Start viewer | `make viewer` |
| Create demo | `make demo-and-serve` |

## 📋 Data Format

Luxar uses Zarr for efficient, chunked storage of large point cloud datasets.

### Zarr Structure

```
dataset.zarr/
├── .zmetadata              # Consolidated metadata (optional)
├── .zattrs                 # Root attributes
├── positions/              # 3D coordinates (Float32, shape: [N, 3])
│   ├── .zarray
│   └── [chunks...]
├── colors/                 # RGB colors (Uint8, shape: [N, 3])
│   ├── .zarray
│   └── [chunks...]
└── subgroup_name/          # Nested scene nodes
    ├── .zattrs
    ├── positions/
    └── colors/
```

### Attributes Schema

```json
{
  "type": "points",
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]  // Optional 4x4 matrix
}
```

### Performance Recommendations

- **Chunk Size**: 64KB-1MB per chunk for optimal I/O
- **Compression**: Use blosc compression for network efficiency
- **Point Count**: 100K-10M points per scene for smooth interaction
- **Data Types**: Float32 for positions, Uint8 for colors

## 🎯 Use Cases

### Scientific Visualization

```python
# Visualize experimental data points
positions = load_experimental_coordinates()  # Your data loading
colors = map_values_to_colors(experimental_values)
scene = Scene("experiment_results.zarr")
scene.add_points("ExperimentData", positions, colors)
scene.finalize()
```

### Geographic Data

```python
# Visualize GPS coordinates with elevation
lat_lon_alt = load_gps_data()
positions = convert_to_cartesian(lat_lon_alt)
colors = elevation_to_color_map(lat_lon_alt[:, 2])
scene = Scene("geographic_data.zarr")
scene.add_points("GPSData", positions, colors)
scene.finalize()
```

### Synthetic Datasets

```python
# Generate procedural point clouds
def generate_fractal_points(iterations=5, scale=2.0):
    # Your fractal algorithm here
    return positions, colors

positions, colors = generate_fractal_points()
scene = Scene("fractal.zarr")
scene.add_points("FractalPoints", positions, colors)
scene.finalize()
```

## 🚨 Troubleshooting

### Common Issues

**ImportError: No module named 'luxar'**
```bash
pip install -e .  # Install in development mode
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
1. Run the basic demo: `luxar random --out demo.zarr --n 50000`
2. Start the viewer: `cd packages/luxar-player && npm run dev`
3. Load your data: `http://localhost:5173/?src=/path/to/demo.zarr`

### Advanced Topics
- Custom shader development for specialized rendering
- Implementing level-of-detail systems
- Building interactive data exploration tools
- Integrating with scientific computing workflows

---

Built with ❤️ for the scientific visualization and data exploration community

**Ready to visualize your data in 3D? Start with `luxar random --out demo.zarr` and explore! 🚀**