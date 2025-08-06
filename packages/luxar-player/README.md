# 🌌 Luxar Player

A GPU-accelerated WebGL renderer for arbitrarily large n-dimensional scientific datasets stored in Zarr format. Delivers maximum visualization performance limited only by your graphics hardware, display resolution, and network bandwidth—not by software constraints. Features advanced HDR rendering, real-time effects, and intuitive navigation controls.

## ✨ Features

- **🎨 Advanced HDR Rendering**: 16-bit floating-point precision with ACES filmic tone mapping
- **✨ Real-time Bloom Effects**: Professional-quality UnrealBloomPass with customizable parameters  
- **🖱️ Intuitive Navigation**: Smooth camera controls optimized for scientific data exploration
- **📱 Responsive Design**: Seamless fullscreen support and dynamic viewport management
- **⚡ Unlimited Performance**: GPU-accelerated pipeline designed to scale with hardware capabilities
- **🎯 Geometry Rendering**: Extensible architecture supporting points, lines, surfaces, volumes (currently points)
- **📊 Performance Monitoring**: Built-in FPS and timing metrics for optimization
- **🌊 Streaming Ready**: Chunked Zarr format enables progressive loading of massive datasets
- **🔌 Extensible Architecture**: Modular design ready for additional geometry types and rendering modes
- **🎛️ nD Navigation**: Beautiful dimension sliders UI for exploring higher-dimensional data
- **🔍 Radius-Based Slicing**: Natural visualization of nD points as hyperspheres
- **⌨️ Keyboard Controls**: Intuitive keyboard navigation for dimension selection and stepping

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Modern web browser with WebGL 2.0 support
- Zarr dataset (see [Data Format](#data-format) section)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd luxar/packages/luxar-player

# Install dependencies
npm install

# Start development server
npm run dev
```

The viewer will be available at `http://localhost:5173`

### Usage

```bash
# View a specific Zarr dataset
http://localhost:5173/?src=/path/to/your/dataset.zarr

# View demo dataset (if available)
http://localhost:5173
```

## 🎮 Controls

### Camera Controls
| Input | Action |
|-------|--------|
| **Mouse Drag** | Rotate camera around scene |
| **Mouse Wheel** | Zoom in/out |
| **Right Click + Drag** | Pan camera |
| **Shift + Mouse Wheel** | Change field of view |
| **Space** | Toggle fullscreen mode |
| **H** | Show/hide help overlay |
| **Shift + P** | Toggle performance statistics |
| **Esc** | Exit fullscreen / Dismiss overlays |

### nD Navigation (for datasets with >3 dimensions)
| Input | Action |
|-------|--------|
| **Number keys (1-9)** | Select which non-displayed dimension to navigate |
| **`[` and `]`** | Step backward/forward in the selected dimension |
| **Dimension Sliders** | Click and drag to navigate through dimensions |

## 🗂️ Data Format

Luxar Player expects Zarr datasets with the following structure:

```
dataset.zarr/
├── .zmetadata                 # Consolidated metadata (optional)
├── .zattrs                    # Scene attributes including dimensions
├── positions/                 # nD coordinates (Float32, shape: [N, D])
│   ├── .zarray
│   └── [chunks...]
├── colors/                    # RGB colors (Uint8, shape: [N, 3]) - optional
│   ├── .zarray
│   └── [chunks...]
├── radii/                     # Point radii (Float32, shape: [N]) - optional
│   ├── .zarray
│   └── [chunks...]
└── sharpness/                 # Edge falloff (Float32, shape: [N]) - optional
    ├── .zarray
    └── [chunks...]
```

### Group Attributes (.zattrs)

```json
{
  "type": "points",
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],  // 4x4 transform matrix (optional)
  "sceneDimensions": {  // Required for nD data
    "dimensions": [
      {"name": "x", "unit": "μm", "range": [-100, 100], "display": true},
      {"name": "y", "unit": "μm", "range": [-100, 100], "display": true},
      {"name": "z", "unit": "μm", "range": [-50, 50], "display": true},
      {"name": "time", "unit": "s", "range": [0, 10], "display": false, "step": 0.1}
    ]
  }
}
```

### Supported Data Types

- **Positions**: Float32 arrays with shape `[N, D]` where D matches dimension count
- **Colors**: Uint8 arrays with shape `[N, 3]` (RGB values 0-255)
- **Radii**: Float32 arrays with shape `[N]` (per-point radius)
- **Sharpness**: Float32 arrays with shape `[N]` (edge falloff 0.5-10.0)
- **Transform**: Optional 4x4 transformation matrix for positioning/scaling

## 🌟 nD Visualization

Luxar Player supports visualization of n-dimensional data beyond traditional 3D:

### Scene-Level Dimensions
Every Zarr dataset defines its dimensions at the scene level:
- **Displayed dimensions**: The 3D subset shown in the viewer (max 3)
- **Non-displayed dimensions**: Additional dimensions navigated via sliders/keyboard
- **Dimension metadata**: Names, units, ranges, and navigation step sizes

### Radius-Based Slicing
Points in nD space are treated as hyperspheres. When viewing a 3D slice:
- Point visibility depends on hypersphere intersection with viewing hyperplane
- Larger radius = visible across more dimension slices
- Effective radius shrinks as: `r_eff = sqrt(r² - d²)` where d is distance from slice
- Natural representation of uncertainty or spread in higher dimensions

### Dimension Navigation UI
- **Beautiful sliders**: Napari-inspired design for intuitive navigation
- **Status bar**: Shows current position in nD space with units
- **Keyboard shortcuts**: Quick dimension selection and stepping
- **Smart initialization**: Non-displayed dimensions start at their minimum values

### Example: 5D Time Series
```javascript
// Dataset with x, y, z (displayed) and time, channel (non-displayed)
// Press '1' to select time dimension
// Use '[' and ']' to step through time
// Press '2' to select channel dimension
// Sliders update automatically
```

## 🛠️ Development

### Project Structure

```
src/
├── main.ts                    # Application entry point
├── app.ts                     # Main application class
├── scene-manager.ts           # 3D scene and renderer setup
├── scene-dims-manager.ts      # Scene-level dimension state management
├── post-processing.ts         # HDR pipeline and bloom effects
├── animation-controller.ts    # Render loop and performance
├── input-handler.ts           # User interaction handling
├── dimension-sliders.ts       # nD navigation UI components
├── shader-manager.ts          # Custom GLSL shaders
├── zarr_loader.ts            # Zarr dataset loading with nD support
├── ui.ts                     # User interface components
├── performance-monitor.ts     # FPS and timing metrics
├── config.ts                 # Configuration constants
├── types/                     # TypeScript type definitions
│   └── dims.ts               # Dimension type definitions
└── utils/                     # Utility functions
    ├── slicing.ts            # nD slicing algorithms
    └── dims-navigation.ts    # Dimension navigation helpers
```

### Available Scripts

```bash
# Development
npm run dev          # Start development server with hot reload
npm run build        # Build for production
npm run preview      # Preview production build

# Code Quality
npm run lint         # Run ESLint
npm run typecheck    # Run TypeScript type checking
npm run format       # Format code with Prettier

# Testing
npm run test         # Run unit tests (if configured)
```

### Configuration

Edit `src/config.ts` to customize:

```typescript
export const CONFIG = {
  CAMERA: {
    FOV: 60,                    // Field of view (degrees)
    INITIAL_POSITION: { x: 0, y: 0, z: 8 },
    FOV_MIN: 10,
    FOV_MAX: 200,
  },
  SCENE: {
    BACKGROUND_COLOR: 0x111111,  // Dark gray background
  },
  ANIMATION: {
    IDLE_TIMEOUT_MS: 2000,      // Auto-pause after 2 seconds
  },
  // ... more options
};
```

## 🔧 Advanced Usage

### Custom Shader Parameters

Modify `src/shader-manager.ts` to adjust point rendering:

```typescript
export const SHADER_CONFIG = {
  POINTS: {
    SIZE: 8.0,                  // Default point size in pixels
    HDR_MULTIPLIER: 13.0,       // Bloom intensity
    BASE_ALPHA: 0.01,           // Base transparency
    FALLOFF_STEEPNESS: 20.0,    // Edge softness
    DEFAULT_RADIUS: 1.0,        // Default point radius
    DEFAULT_SHARPNESS: 2.0,     // Default edge falloff
  },
};
```

Point rendering now supports per-point attributes:
- **radius**: Individual point sizes for visual hierarchy
- **sharpness**: Control edge falloff (0.5 = soft glow, 10.0 = sharp edges)
- **Automatic compensation**: Shader adjusts intensity based on sharpness

### HDR Post-Processing

Customize bloom effects in `src/post-processing.ts`:

```typescript
export const POST_PROCESSING_CONFIG = {
  BLOOM: {
    THRESHOLD: 0.0,             // Bloom threshold (0.0 = everything glows)
    STRENGTH: 0.1,              // Bloom intensity
    RADIUS: 0.5,                // Bloom spread
    RESOLUTION_SCALE: 4,        // Performance vs quality
  },
};
```

### Performance Optimization

- **Point Count**: Optimize for datasets with millions of points
- **Chunk Size**: Zarr chunk sizes of 64KB-1MB work well
- **LOD**: Consider implementing level-of-detail for very large datasets
- **Compression**: Use Zarr compression (e.g., blosc) to reduce network transfer

## 🎯 Performance Tips

1. **Ideal Dataset Size**: 100K-10M points for smooth interaction
2. **Chunk Strategy**: Use roughly square chunks (e.g., 1000x1000 points)
3. **Network**: Serve Zarr data from same domain to avoid CORS issues
4. **Browser**: Chrome and Firefox offer best WebGL performance
5. **Hardware**: Dedicated GPU recommended for large datasets

## 🚨 Troubleshooting

### Common Issues

**White screen on load**
- Check browser console for errors
- Verify Zarr dataset URL is accessible
- Ensure CORS headers are set if serving from different domain

**Poor performance**
- Check dataset size (>10M points may be slow)
- Reduce bloom quality in config
- Verify GPU acceleration is enabled in browser

**Zarr loading errors**
- Verify dataset structure matches expected format
- Check that positions and colors arrays exist
- Ensure proper Zarr metadata (.zarray files)

### Browser Compatibility

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 90+ | ✅ Fully supported |
| Firefox | 88+ | ✅ Fully supported |
| Safari | 14+ | ✅ Supported |
| Edge | 90+ | ✅ Fully supported |

### WebGL Requirements

- WebGL 2.0 support required
- Float texture support (for HDR rendering)
- Minimum 2GB GPU memory recommended for large datasets

## 📝 API Reference

### URL Parameters

- `?src=<path>` - Path to Zarr dataset
- `?debug=true` - Enable debug logging
- `?fps=true` - Show FPS counter on startup

### Programmatic Usage

```javascript
import { LuxarApp } from './src/app.js';

const app = new LuxarApp();
await app.init('/path/to/dataset.zarr');

// Access components
const { sceneManager, animationController } = app.components;

// Update bloom settings
sceneManager.postProcessing.updateBloomSettings(0.2, 0.8, 0.1);
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow TypeScript strict mode
- Use ESLint and Prettier for code formatting
- Add JSDoc comments for public APIs
- Test with various dataset sizes
- Ensure WebGL resource cleanup

## 📄 License

Copyright (c) 2024 The Luxar Authors

[Add your license information here]

## 🙏 Acknowledgments

- **Three.js** - 3D rendering engine
- **Zarrita** - Zarr format support
- **Vite** - Development tooling
- **Contributors** - Thanks to all who helped improve this project

## 📞 Support

- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Documentation**: [Project Wiki](link-to-wiki)

---

Built with ❤️ for the scientific visualization community