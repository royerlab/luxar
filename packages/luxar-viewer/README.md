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

- Node.js 22+ and pnpm (preferred package manager)
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
Luxar Viewer supports three navigation modes:
- **Orbit Mode** (default): Quaternion-based rotation around a target point (no gimbal lock)
- **Fly Mode**: First-person navigation with WASD movement and inertial physics
- **Ortho Mode**: Orthographic pan + zoom for 2D viewing

| Key | Action |
|-----|--------|
| **V** | Cycle control modes: Orbit -> Fly -> Ortho -> Orbit |
| **I** | Toggle inertial mode (Fly mode only) |
| **F** | Recenter camera on scene |
| **C** | Toggle cinematic mode |

### Orbit Mode Controls
| Input | Action |
|-------|--------|
| **Left Mouse Drag** | Pan camera |
| **Right Click + Drag** | Rotate camera around scene |
| **Mouse Wheel** | Zoom in/out |
| **Shift + Mouse Wheel** | Roll (view-axis rotation) |

### Fly Mode Controls
| Input | Action |
|-------|--------|
| **W/S** | Move forward/backward |
| **A/D** | Strafe left/right |
| **Alt+W / Alt+S** | Move up/down |
| **Arrow Keys** | Look up/down/left/right |
| **Left Mouse Drag** | Strafe (screen-space translation) |
| **Right Mouse Drag** | Free look (rotate camera) |
| **Mouse Wheel** | Forward/backward velocity impulse |
| **Shift + Mouse Wheel** | Roll (view-axis rotation) |
| **I** | Toggle inertial physics (drift/momentum) |

### Ortho Mode Controls
| Input | Action |
|-------|--------|
| **Left Mouse Drag** | Pan (Napari/Google Maps convention) |
| **Mouse Wheel** | Zoom in/out |
| **Shift + Mouse Wheel** | Roll (view-axis rotation) |

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

Luxar Viewer expects Zarr datasets with the following structure:

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

Luxar Viewer supports visualization of n-dimensional data beyond traditional 3D:

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
├── cache/
│   ├── index.ts                   # Cache module exports
│   ├── cached-zarr-array.ts       # Cached zarr array access
│   ├── chunk-prefetcher.ts        # Chunk prefetching logic
│   ├── decompressed-chunk-cache.ts # Decompressed chunk caching
│   ├── lru-cache.ts               # LRU cache implementation
│   ├── opfs-store.ts              # Origin Private File System store
│   ├── segmented-lru-cache.ts     # Segmented LRU cache
│   ├── two-level-caching-store.ts # Two-level caching store
│   └── types.ts                   # Cache type definitions
├── config/
│   ├── index.ts                   # Unified configuration system
│   ├── types.ts                   # Configuration type definitions
│   ├── validation.ts              # Configuration validation
│   ├── viewer-config-utils.ts     # Viewer config utilities
│   └── viewer-state-capture.ts    # Viewer state capture
├── controls/
│   ├── controls-manager.ts        # Control mode switching and management
│   ├── luxar-fly-controls.ts      # Custom fly controls with inertial physics
│   ├── luxar-orbit-controls.ts    # Quaternion-based orbit controls
│   └── types.ts                   # Control system type definitions
├── data/
│   ├── array-decoder.ts           # Array decoding from zarr
│   ├── chunk-spatial-index.ts     # Point chunk spatial indexing
│   ├── data-accumulator.ts        # Data accumulation for progressive loading
│   ├── data-loader-types.ts       # Data loader type definitions
│   ├── data-monitor-manager.ts    # Data monitoring
│   ├── directory-navigator.ts     # Directory navigation
│   ├── effective-radius-calculator.ts # nD effective radius computation
│   ├── gsplats-chunk-spatial-index.ts # GSplats chunk spatial indexing
│   ├── gsplats-processor.ts       # Gaussian splat processing
│   ├── gsplats-progressive-loader.ts # Progressive GSplats loading
│   ├── gsplats-spatial-index-loader.ts # GSplats spatial index loading
│   ├── index.ts                   # Data module exports
│   ├── lines-chunk-spatial-index.ts # Lines chunk spatial indexing
│   ├── lines-spatial-index-loader.ts # Lines spatial index loading
│   ├── loader-registry.ts         # Loader registry
│   ├── nd-transform.ts            # nD transform inverse-query for non-displayed dimensions
│   ├── point-spatial-index-loader.ts # Point spatial index loading
│   ├── scene-graph-builder.ts     # Scene graph construction from zarr
│   ├── scene-loader.ts            # Scene loading orchestration
│   ├── scene-loader-manager.ts    # Scene loader management
│   ├── stats-aggregator.ts        # Statistics aggregation
│   ├── tolerance-computer.ts      # Tolerance computation
│   ├── view-state-manager.ts      # View state management
│   ├── zarr-loader.ts             # Zarr dataset loading with nD support
│   └── loaders/                   # Modular loader subsystem
│       ├── base-types.ts          # Base loader type definitions
│       ├── index.ts               # Loader module exports
│       ├── integration-example.ts # Loader integration example
│       ├── range-loader.ts        # Range-based loading
│       ├── spatial-query-builder.ts # Spatial query construction
│       └── transferable-accumulator.ts # Transferable data accumulation
├── input/
│   ├── input-handler.ts           # User interaction handling
│   ├── input-handler-utils.ts     # Input handler utilities
│   └── input-context-manager.ts   # Keyboard conflict resolution
├── profiling/
│   └── update-profiler.ts         # Update profiling
├── rendering/
│   ├── adaptive-dpr-manager.ts    # Adaptive device pixel ratio management
│   ├── chromatic-lens-distortion-effect.ts # Chromatic lens distortion effect
│   ├── colormap-data.ts           # Colormap data definitions
│   ├── colormap-textures.ts       # Colormap texture generation
│   ├── detector-noise-effect.ts   # Detector noise effect
│   ├── gpu-buffer-pool.ts         # GPU buffer pooling and reuse
│   ├── gsplat-material.ts         # Gaussian splat material
│   ├── line-material.ts           # Line material
│   ├── luxar-tone-mapping-effect.ts # Custom tone mapping effect
│   ├── material-manager.ts        # Material caching and optimization
│   ├── point-material.ts          # Point material with custom shaders
│   ├── post-processing-manager.ts # HDR pipeline and bloom effects
│   ├── postprocessing-types.ts    # Post-processing type definitions
│   └── robust-vignette-effect.ts  # Vignette effect
├── scene/
│   ├── animation-controller.ts    # Render loop and performance
│   ├── camera-utils.ts            # Camera type union, type guards, and projection helpers
│   ├── dimension-animation-manager.ts # Dimension animation management
│   ├── scene-dims-manager.ts      # Scene-level dimension state management
│   ├── scene-manager.ts           # 3D scene and renderer setup
│   └── scene-manager-utils.ts     # Scene manager utilities
├── styles/                        # CSS styles
│   ├── index.css                  # Main stylesheet
│   ├── reset.css                  # CSS reset
│   ├── base/                      # Base styles
│   ├── components/                # Component styles
│   └── themes/                    # Theme stylesheets
├── themes/
│   ├── index.ts                   # Theme module exports
│   ├── glass-filters.ts           # Glass filter effects
│   ├── theme-manager.ts           # Theme management
│   ├── types.ts                   # Theme type definitions
│   └── themes/
│       ├── dark.theme.ts          # Dark theme
│       ├── frosted-glass.theme.ts # Frosted glass theme
│       ├── light.theme.ts         # Light theme
│       └── liquid-glass.theme.ts  # Liquid glass theme
├── types/
│   ├── animation.ts               # Animation type definitions
│   ├── dims.ts                    # Dimension type definitions
│   ├── float16array.d.ts          # Float16Array type declaration
│   ├── gsplats.ts                 # Gaussian splats type definitions
│   ├── index.ts                   # Type module exports
│   ├── lines.ts                   # Lines type definitions
│   ├── points.ts                  # Points type definitions
│   └── zarr.ts                    # Zarr type definitions
├── ui/
│   ├── data-loading-monitor.ts    # Data loading progress monitor
│   ├── data-monitor-templates.ts  # Data monitor HTML templates
│   ├── data-monitor-types.ts      # Data monitor type definitions
│   ├── dataset-browser.ts         # Dataset browser panel
│   ├── debug-console.ts           # In-app debug console (Ctrl+L)
│   ├── dimension-sliders.ts       # nD navigation UI components
│   ├── helpers.ts                 # UI helper utilities
│   ├── performance-monitor.ts     # FPS and timing metrics
│   ├── recording-panel.ts         # Screenshot and video capture panel
│   ├── rendering-controls.ts      # Advanced rendering controls panel
│   ├── rendering-controls-utils.ts # Rendering controls utilities
│   ├── components/
│   │   ├── base/
│   │   │   └── ui-component.ts    # Base UI component class
│   │   ├── colormap-legend.ts     # Colormap legend overlay
│   │   ├── event-queue.ts         # Event queue for UI updates
│   │   ├── hierarchical-timing-panel.ts # Hierarchical timing panel
│   │   ├── loading-advisor.ts     # Loading advisor overlay
│   │   ├── polling-loop.ts        # Polling loop for UI updates
│   │   ├── resolution-indicator.ts # Resolution indicator overlay
│   │   └── scale-bar.ts           # Physical scale bar overlay
│   ├── layers/                    # Layer management UI
│   │   ├── index.ts               # Layer module exports
│   │   ├── layers-panel.ts        # Layers panel
│   │   ├── layer-state.ts         # Layer state management
│   │   └── range-slider.ts        # Range slider component
│   ├── rendering-controls/        # Rendering controls sub-modules
│   │   ├── anti-aliasing-setup.ts # Anti-aliasing setup
│   │   ├── camera-setup.ts        # Camera setup
│   │   ├── hdr-setup.ts           # HDR setup
│   │   ├── navigation-setup.ts    # Navigation setup
│   │   ├── post-processing-setup.ts # Post-processing setup
│   │   └── types.ts               # Rendering controls type definitions
│   └── gui/                       # Custom GUI framework
│       ├── index.ts               # GUI module exports
│       ├── controllers/
│       │   ├── boolean-controller.ts  # Boolean controller
│       │   ├── function-controller.ts # Function/button controller
│       │   ├── number-controller.ts   # Number controller
│       │   ├── option-controller.ts   # Option/select controller
│       │   └── string-controller.ts   # String controller
│       ├── core/
│       │   ├── controller.ts      # Base controller class
│       │   ├── folder.ts          # Folder/section container
│       │   ├── gui.ts             # GUI root class
│       │   └── types.ts           # GUI type definitions
│       ├── dom/
│       │   └── event-manager.ts   # DOM event management
│       ├── styles/                # GUI CSS styles
│       │   ├── controller.css
│       │   ├── folder.css
│       │   └── gui.css
│       └── utils/
│           ├── auto-blur.ts       # Auto-blur utility
│           └── value-formatting.ts # Value formatting utility
├── utils/
│   ├── console-interceptor.ts     # Console output capture
│   ├── escape-html.ts             # HTML escaping utility
│   ├── hdr-color-conversion.ts    # HDR color conversion
│   ├── hdr-detection.ts           # HDR display detection
│   ├── hdr-video-encoder.ts       # HDR video encoding
│   ├── log.ts                     # Structured logging utility
│   └── memory-detector.ts         # Memory availability detection
├── wasm/
│   ├── index.ts                   # WASM module loader
│   ├── types.ts                   # WASM type definitions
│   ├── rust/                      # Rust WASM source
│   │   └── src/                   # Rust source files
│   └── typescript/                # TypeScript fallback implementations
│       ├── decode.ts              # Array decoding
│       ├── effective_radii.ts     # Effective radius computation
│       ├── gsplats.ts             # Gaussian splat processing
│       ├── gsplats_processing.ts  # GSplat processing utilities
│       ├── index.ts               # TypeScript fallback exports
│       ├── lines.ts               # Lines processing
│       ├── lines_clipping.ts      # Lines clipping
│       ├── points.ts              # Points processing
│       ├── projection.ts          # Projection utilities
│       └── spatial.ts             # Spatial utilities
└── workers/
    ├── data-worker.ts             # Background data processing worker
    └── worker-pool.ts             # Worker pool management
```

### Available Scripts

```bash
# Development
pnpm dev             # Start development server with hot reload
pnpm build           # Build for production (includes WASM build)
pnpm preview         # Preview production build

# Code Quality
pnpm lint            # Run ESLint
pnpm typecheck       # Run TypeScript type checking
pnpm format          # Format code with Prettier
pnpm check           # Run all quality checks (typecheck + lint + test)

# Unit Testing
pnpm test            # Run unit tests with Vitest
pnpm test:coverage   # Run tests with coverage report
pnpm test:ui         # Run tests with interactive UI
pnpm test:watch      # Run tests in watch mode
pnpm test:with-fixtures  # Generate test fixtures, then run tests

# E2E Testing (Playwright)
pnpm test:e2e        # Run all E2E tests
pnpm test:e2e:ui     # Run E2E tests with interactive UI
pnpm test:e2e:debug  # Run E2E tests in debug mode
pnpm test:e2e:report # Show E2E test report

# WASM
pnpm build:wasm      # Build Rust WASM module
pnpm build:wasm:dev  # Build WASM in development mode
pnpm test:wasm       # Run Rust unit tests (cargo test)
pnpm bench:wasm      # Run WASM vs TypeScript benchmarks

# Fixtures & Media
pnpm test:generate-fixtures  # Generate test fixtures from Python
pnpm readme-images   # Generate README screenshot images
pnpm readme-videos   # Generate README video recordings

# AI Debugging
pnpm agent:debug     # Run Playwright agent driver (headless)
pnpm agent:debug:visible  # Run agent driver with visible browser
```

### Configuration

Luxar Viewer uses a unified configuration system in `src/config/`. Edit `src/config/index.ts` to customize:

```typescript
export const config: AppConfig = {
  camera: {
    initialPosition: { x: 0, y: 0, z: 8 },
    fovMin: 10,
    fovMax: 170,
  },
  scene: {
    backgroundColor: 0x111111,  // Dark gray background
  },
  animation: {
    idleTimeoutMs: 2000,       // Auto-pause after 2 seconds
  },
  renderingControls: {
    defaults: {
      fov: 47,                  // Field of view in degrees (50mm Normal)
      bloomEnabled: false,      // Bloom effect (opt-in via zarr viewer_config)
      bloomStrength: 0.25,      // Bloom intensity multiplier
      bloomRadius: 1.0,         // Blur radius for bloom spread
      bloomThreshold: 0.01,     // Luminance threshold for bloom
      fxaaEnabled: false,       // FXAA anti-aliasing
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

### Per-Point Attributes

Point rendering supports per-point attributes:
- **radius**: Individual point sizes for visual hierarchy
- **sharpness**: Control edge falloff (0.5 = soft glow, 10.0 = sharp edges)
- **Automatic compensation**: Shader adjusts intensity based on sharpness

### HDR Post-Processing

Customize bloom and tone mapping in `src/config/index.ts` under `renderingControls.defaults`:

```typescript
renderingControls: {
  defaults: {
    bloomEnabled: false,        // Enable/disable bloom effect
    bloomThreshold: 0.01,       // Luminance threshold (0.0 = everything glows)
    bloomStrength: 0.25,        // Bloom intensity multiplier
    bloomRadius: 1.0,           // Bloom spread
    bloomLevels: 8,             // Mipmap levels (1-12, lower = faster)
    toneMapping: 'Neutral',     // Options: None, Linear, Reinhard, Cineon, ACES, AgX, Neutral
    exposure: 0.0,              // Log2 stops (0 = neutral, +1 = 2x brighter)
  },
},
```

### Anti-Aliasing Configuration

Luxar Viewer supports multiple anti-aliasing techniques with important compatibility notes:

```typescript
renderingControls: {
  defaults: {
    fxaaEnabled: false,         // FXAA: Fast post-process AA (disabled by default)
    msaaEnabled: false,         // MSAA: Hardware-accelerated, fast and sharp
    msaaSamples: 4,             // MSAA sample count (2, 4, 8)
    smaaEnabled: false,         // SMAA: Advanced edge-detection AA
    ssaaEnabled: false,         // SSAA: Supersampling, highest quality, heavy cost
    ssaaMultiplier: 2.0,        // SSAA resolution multiplier (1.5x, 2x, 4x)
  },
},
```

**Anti-Aliasing Notes:**
- **MSAA**: Hardware-accelerated, fast and sharp — great default for most scenes. Note: MSAA has limitations with additive blending (used by GSplats); consider FXAA or SMAA for scenes with Gaussian splats
- **FXAA**: Fastest post-process AA, may slightly blur the image
- **SMAA**: Advanced edge detection with preset quality levels (LOW/MEDIUM/HIGH/ULTRA)
- **SSAA**: Highest quality (supersampling), significant performance cost

### Performance Optimization

- **Element Count**: Optimize for datasets with millions of elements
- **Chunk Size**: Zarr chunk sizes of 64KB-1MB work well
- **LOD**: Consider implementing level-of-detail for very large datasets
- **Compression**: Use Zarr compression (e.g., blosc) to reduce network transfer

### Performance Monitoring

Built-in performance monitoring in `src/ui/performance-monitor.ts` wraps [stats.js](https://github.com/mrdoob/stats.js/) and provides:

```typescript
// PerformanceMonitor wraps stats.js for FPS, frame time, and memory tracking
const monitor = new PerformanceMonitor();
monitor.begin();    // Call at the start of each frame
// ... rendering work ...
monitor.end();      // Call at the end of each frame

// Visibility control
monitor.show();     // Show the stats panel
monitor.hide();     // Hide the stats panel
monitor.toggle();   // Toggle visibility
```

- **Real-time FPS**: Continuously updated frame rate display (panel 0)
- **Frame timing**: Milliseconds per frame (panel 1)
- **Memory usage**: JavaScript heap size monitoring (panel 2)
- **Panel cycling**: `cyclePanels()` rotates through FPS, frame time, and memory views
- **Idle optimization**: Only measures when visible to avoid overhead

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
| Safari | 15+ | ✅ Supported |
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

// Access components (available after init)
const { sceneManager, animationController, renderingControls } = app.components;

// Modify bloom defaults before initialization (or for next scene load)
config.renderingControls.defaults.bloomEnabled = true;
config.renderingControls.defaults.bloomStrength = 0.2;
config.renderingControls.defaults.bloomRadius = 0.8;
config.renderingControls.defaults.bloomThreshold = 0.1;

// Available components:
//   sceneManager          - 3D scene, renderer, camera, controls
//   animationController   - Render loop, per-frame callbacks
//   inputHandler          - Keyboard/mouse input
//   renderingControls     - UI panel for rendering settings
//   adaptiveDPRManager    - Dynamic resolution scaling
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

Copyright (c) 2025-2026 The Luxar Authors

This project is licensed under the BSD-3-Clause License. See the [LICENSE](../../LICENSE) file for details.

## 🙏 Acknowledgments

- **Three.js** - 3D rendering engine
- **Zarrita** - Zarr format support
- **Vite** - Development tooling
- **Contributors** - Thanks to all who helped improve this project

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/royerlab/luxar/issues)
- **Discussions**: [GitHub Discussions](https://github.com/royerlab/luxar/discussions)
- **Documentation**: [Project Repository](https://github.com/royerlab/luxar)

---

Built with ❤️ for the scientific visualization community
