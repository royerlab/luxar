# 🌌 Luxar Viewer

A GPU-accelerated WebGL renderer for arbitrarily large n-dimensional scientific datasets stored in Zarr format. Delivers maximum visualization performance limited only by your graphics hardware, display resolution, and network bandwidth—not by software constraints. Features advanced HDR rendering, real-time effects, and intuitive navigation controls.

## ✨ Features

- **🎨 Advanced HDR Rendering**: 16-bit floating-point precision with ACES filmic tone mapping
- **✨ Real-time Bloom Effects**: Professional-quality UnrealBloomPass with customizable parameters
- **🖱️ Intuitive Navigation**: Smooth camera controls optimized for scientific data exploration
- **📱 Responsive Design**: Seamless fullscreen support and dynamic viewport management
- **⚡ Unlimited Performance**: GPU-accelerated pipeline designed to scale with hardware capabilities
- **🎯 Geometry Rendering**: Extensible architecture supporting points, lines, Gaussian splats, and more
- **📊 Performance Monitoring**: Built-in FPS and timing metrics for optimization
- **🌊 Streaming Ready**: Chunked Zarr format enables progressive loading of massive datasets
- **🔌 Extensible Architecture**: Modular design ready for additional geometry types and rendering modes
- **🎛️ nD Navigation**: Beautiful dimension sliders UI for exploring higher-dimensional data
- **🔍 Radius-Based Slicing**: Natural visualization of nD data using hypersphere intersection
- **⌨️ Keyboard Controls**: Intuitive keyboard navigation for dimension selection and stepping
- **⚙️ Advanced Anti-Aliasing**: Multiple AA techniques (FXAA, SMAA, MSAA, SSAA) with known compatibility notes
- **🧩 Unified Configuration**: Centralized config system in `src/config/` with TypeScript types
- **📸 Recording Panel**: Screenshot (PNG/WebP/JPEG) and video capture (WebM) with turntable mode
- **📏 Scale Bar**: Physical scale bar overlay using dimension unit metadata
- **🔄 nD Transforms**: Inverse-query transforms for non-displayed dimensions (affine and categorical)
- **🎯 Material Caching**: Optimized material management with intelligent caching strategy

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and pnpm (preferred package manager)
- Modern web browser with WebGL 2.0 support
- Zarr dataset (see [Data Format](#data-format) section)

**Browser Recommendations**:
- **Firefox** (recommended for large datasets): 2× faster WebAssembly decompression (1 GB/s vs 500 MB/s)
- **Chrome/Edge**: Excellent compatibility, good performance
- **Safari**: Good support, slightly lower WASM performance

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd luxar/packages/luxar-viewer

# Install dependencies
pnpm install

# Start development server
pnpm dev
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

### Control Modes
Luxar Player supports two navigation modes:
- **Orbit Mode** (default): Traditional 3D viewer controls - rotate around a target point
- **Fly Mode**: First-person navigation with WASD movement and inertial physics

| Key | Action |
|-----|--------|
| **V** | Toggle between Orbit and Fly control modes |
| **I** | Toggle inertial mode (Fly mode only) |
| **F** | Recenter camera on scene |
| **C** | Toggle between native center and bounding box center |

### Orbit Mode Controls
| Input | Action |
|-------|--------|
| **Mouse Drag** | Rotate camera around scene |
| **Mouse Wheel** | Zoom in/out |
| **Right Click + Drag** | Pan camera |
| **Shift + Mouse Wheel** | Change field of view |

### Fly Mode Controls
| Input | Action |
|-------|--------|
| **W/S** | Move forward/backward |
| **A/D** | Strafe left/right |
| **Alt+W / Alt+S** | Move up/down |
| **Arrow Keys** | Look up/down/left/right |
| **Mouse Drag** | Free look (rotate camera) |
| **I** | Toggle inertial physics (drift/momentum) |

### General Controls
| Input | Action |
|-------|--------|
| **Space** | Toggle fullscreen mode |
| **H** | Show/hide help overlay |
| **R** | Toggle advanced rendering controls panel |
| **P** | Toggle performance statistics |
| **N** | Toggle nD dimension panel |
| **O** | Open dataset browser |
| **Ctrl+L** | Toggle debug console |
| **Esc** | Exit fullscreen / Close panels |

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

## 🎛️ Advanced Rendering Controls

The advanced rendering controls panel (located on the left side) provides real-time adjustment of:

### Visual Effects
- **Bloom Settings**: Threshold, strength, radius, and resolution scale
- **HDR Controls**: Exposure and tone mapping parameters
- **Point Rendering**: HDR multiplier and falloff parameters

### Anti-Aliasing Options
- **FXAA**: Fast approximate anti-aliasing (recommended for additive blending)
- **MSAA**: Multi-sample anti-aliasing with sample count selection (2x, 4x, 8x)
- **SMAA**: Subpixel morphological anti-aliasing with preset quality levels
- **SSAA**: Super-sample anti-aliasing with resolution multipliers (1.5x, 2x, 4x)

### Performance Features
- **Real-time preview**: Changes are applied immediately with smooth animations
- **Settings persistence**: User preferences are saved between sessions
- **Performance impact indicators**: Visual feedback on rendering cost
- **Preset management**: Quick access to optimized configurations

## 🛠️ Development

### Project Structure

```
src/
├── core/
│   ├── main.ts                    # Application entry point
│   └── app.ts                     # Main application class
├── config/
│   ├── index.ts                   # Unified configuration system
│   └── types.ts                   # Configuration type definitions
├── controls/
│   ├── controls-manager.ts        # Control mode switching and management
│   ├── luxar-fly-controls.ts      # Custom fly controls with inertial physics
│   ├── control-config.ts          # Control system configuration
│   └── types.ts                   # Control system type definitions
├── scene/
│   ├── scene-manager.ts           # 3D scene and renderer setup
│   ├── scene-dims-manager.ts      # Scene-level dimension state management
│   ├── animation-controller.ts    # Render loop and performance
│   └── camera-utils.ts            # Camera type union, type guards, and projection helpers
├── rendering/
│   ├── post-processing.ts         # HDR pipeline and bloom effects
│   ├── shader-manager.ts          # Custom GLSL shaders
│   └── material-manager.ts        # Material caching and optimization
├── data/
│   ├── zarr-loader.ts            # Zarr dataset loading with nD support
│   └── nd-transform.ts           # nD transform inverse-query for non-displayed dimensions
├── input/
│   ├── input-handler.ts           # User interaction handling
│   └── input-context-manager.ts   # Keyboard conflict resolution
├── ui/
│   ├── dimension-sliders.ts       # nD navigation UI components
│   ├── performance-monitor.ts     # FPS and timing metrics
│   ├── rendering-controls.ts      # Advanced rendering controls panel
│   ├── debug-console.ts           # In-app debug console (Ctrl+L)
│   ├── recording-panel.ts         # Screenshot and video capture panel
│   └── components/
│       └── scale-bar.ts           # Physical scale bar overlay
├── tests/
│   ├── controls-manager.test.ts   # Control system unit tests
│   ├── luxar-fly-controls.test.ts # Fly controls unit tests
│   └── input-context-manager.test.ts # Input context tests
├── types/
│   └── dims.ts                   # Dimension type definitions
└── utils/
    ├── slicing.ts                # nD slicing algorithms
    ├── console-interceptor.ts    # Console output capture
    ├── hdr-detection.ts          # HDR display detection
    ├── memory-detector.ts        # Memory availability detection
    └── log.ts                    # Structured logging utility
```

### Available Scripts

```bash
# Development
pnpm dev             # Start development server with hot reload
pnpm build           # Build for production
pnpm preview         # Preview production build

# Code Quality
pnpm lint            # Run ESLint
pnpm typecheck       # Run TypeScript type checking
pnpm format          # Format code with Prettier
pnpm check           # Run all quality checks (typecheck + lint + test)

# Testing
pnpm test            # Run unit tests with Vitest
pnpm test:coverage   # Run tests with coverage report
pnpm test:ui         # Run tests with interactive UI
pnpm test:watch      # Run tests in watch mode
```

### Configuration

Luxar Player uses a unified configuration system in `src/config/`. Edit `src/config/index.ts` to customize:

```typescript
export const config: AppConfig = {
  camera: {
    fov: 60,                    // Field of view (degrees)
    initialPosition: { x: 0, y: 0, z: 8 },
    fovMin: 10,
    fovMax: 200,
  },
  scene: {
    backgroundColor: 0x111111,  // Dark gray background
  },
  animation: {
    idleTimeoutMs: 2000,       // Auto-pause after 2 seconds
  },
  renderingControls: {
    defaults: {
      fxaaEnabled: true,        // FXAA anti-aliasing (recommended)
      msaaEnabled: false,       // MSAA (hardware-accelerated, fast and sharp)
      ssaaEnabled: false,       // SSAA (supersampling, highest quality, heavy cost)
      // ... more rendering options
    },
  },
  // ... more options
};
```

## 🔧 Advanced Usage

### Material Caching System

The material manager in `src/rendering/material-manager.ts` provides optimized material handling:

```typescript
// Supported blending modes
type BlendingMode = 'normal' | 'additive' | 'max';

// Materials are automatically cached based on properties
const material = materialManager.getMaterial({
  blendingMode: 'additive',
  opacity: 1.0,
  gamma: 2.2,
});
```

### Custom Shader Parameters

Modify shader configuration in `src/config/index.ts`:

```typescript
shader: {
  points: {
    size: 8.0,                  // Default point size in pixels
    hdrMultiplier: 13.0,        // Bloom intensity
    baseAlpha: 0.01,            // Base transparency
    falloffSteepness: 20.0,     // Edge softness
  },
},
```

Point rendering now supports per-point attributes:
- **radius**: Individual point sizes for visual hierarchy
- **sharpness**: Control edge falloff (0.5 = soft glow, 10.0 = sharp edges)
- **Automatic compensation**: Shader adjusts intensity based on sharpness

### HDR Post-Processing

Customize bloom effects in `src/config/index.ts`:

```typescript
postProcessing: {
  bloom: {
    threshold: 0.01,            // Bloom threshold (0.0 = everything glows)
    strength: 0.1,              // Bloom intensity
    radius: 0.5,                // Bloom spread
    resolutionScale: 4,         // Performance vs quality
  },
  toneMapping: {
    final: {
      outputColorSpace: THREE.SRGBColorSpace,
      toneMapping: THREE.ACESFilmicToneMapping,
    },
  },
},
```

### Anti-Aliasing Configuration

Luxar Player supports multiple anti-aliasing techniques with important compatibility notes:

```typescript
renderingControls: {
  defaults: {
    fxaaEnabled: true,          // FXAA: Works well with additive blending
    msaaEnabled: false,         // MSAA: Hardware-accelerated, fast and sharp
    ssaaEnabled: false,         // SSAA: Supersampling, highest quality, heavy cost
    smaaEnabled: false,         // SMAA: Advanced edge-detection AA
  },
},
```

**Anti-Aliasing Notes:**
- **MSAA**: Hardware-accelerated, fast and sharp — great default for most scenes
- **FXAA**: Fastest post-process AA, may slightly blur the image
- **SMAA**: Advanced edge detection with preset quality levels (LOW/MEDIUM/HIGH/ULTRA)
- **SSAA**: Highest quality (supersampling), significant performance cost

### Performance Optimization

- **Element Count**: Optimize for datasets with millions of elements
- **Chunk Size**: Zarr chunk sizes of 64KB-1MB work well
- **LOD**: Consider implementing level-of-detail for very large datasets
- **Compression**: Use Zarr compression (e.g., blosc) to reduce network transfer

### Performance Monitoring

Built-in performance monitoring in `src/ui/performance-monitor.ts` provides:

```typescript
// Access performance metrics
const monitor = new PerformanceMonitor();
monitor.startFrame();
// ... rendering work ...
monitor.endFrame();

// Get metrics
const fps = monitor.getFPS();
const frameTime = monitor.getAverageFrameTime();
```

- **Real-time FPS**: Continuously updated frame rate display
- **Frame timing**: Average and instantaneous frame time measurements
- **GPU performance**: WebGL timing queries when available
- **Memory usage**: WebGL resource monitoring
- **Automatic idle detection**: Pauses monitoring during idle periods

## 🎯 Performance Tips

1. **Ideal Dataset Size**: 100K-10M elements for smooth interaction
2. **Chunk Strategy**: Use roughly square chunks (e.g., 1000x1000 elements)
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
- Check dataset size (>10M elements may be slow)
- Reduce bloom quality in config
- Verify GPU acceleration is enabled in browser
- Try disabling MSAA/SSAA and using FXAA instead

**Zarr loading errors**
- Verify dataset structure matches expected format
- Check that positions and colors arrays exist
- Ensure proper Zarr metadata (.zarray files)
- Verify scene dimensions are defined for nD datasets

**Anti-aliasing selection**
- Try MSAA first (fast, sharp, hardware-accelerated)
- Use FXAA or SMAA for lightweight post-process smoothing
- Use SSAA only for final renders (heavy performance cost)

**nD navigation not working**
- Verify `sceneDimensions` are defined in dataset `.zattrs`
- Check that non-displayed dimensions have proper `range` and `step` values
- Ensure dimension count matches position data shape

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
import { LuxarApp } from './src/core/app.js';
import { config } from './src/config/index.js';

const app = new LuxarApp();
await app.init('/path/to/dataset.zarr');

// Access components
const { sceneManager, animationController } = app.components;

// Update bloom settings through configuration
config.postProcessing.bloom.strength = 0.2;
config.postProcessing.bloom.radius = 0.8;
config.postProcessing.bloom.threshold = 0.1;

// Update rendering controls
const renderingControls = app.components.renderingControls;
renderingControls.updateSettings({
  fxaaEnabled: true,
  bloomStrength: 0.15,
  exposure: 1.2,
});
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
