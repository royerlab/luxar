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
- **⚙️ Anti-Aliasing**: FXAA / MSAA / SSAA with known compatibility notes
- **🧩 Unified Configuration**: Centralized config system in `src/config/` with TypeScript types
- **📸 Recording Panel**: Screenshots (PNG/WebP/JPEG/EXR), image-sequence ZIPs, EXR-sequence ZIPs, and video capture (WebM/MP4/MKV via mediabunny) with turntable mode
- **📏 Scale Bar**: Physical scale bar overlay using dimension unit metadata
- **🔄 nD Transforms**: Inverse-query transforms for non-displayed dimensions (affine and categorical)
- **🎯 Material Caching**: Optimized material management with intelligent caching strategy

## 📦 Embedding (single viewer per page)

`luxar-viewer` ships as a side-effect-free ES module. Importing the
package does not patch your `console`, inject CSS into your `body`, or
mutate `:root` — the viewer only touches DOM you give it via the `canvas`
option, plus the UI overlays it mounts into the `container` you provide
(defaulting to `document.body`).

```bash
npm install @royerlab/luxar-viewer three   # three is a peer dep
```

```ts
import { LuxarApp } from '@royerlab/luxar-viewer';
import '@royerlab/luxar-viewer/styles.css'; // component styles, prefixed under .luxar-*

const canvas = document.querySelector<HTMLCanvasElement>('#viewer-canvas')!;
const app = new LuxarApp();
await app.init({
  canvas,
  src: 'https://example.com/data.zarr',
  updateBrowserUrl: false, // default: don't rewrite host URL on dataset change
});

// Later (e.g. when the host route unmounts):
app.dispose(); // removes all listeners, GPU resources, UI
```

### `LuxarAppOptions`

| Option             | Type                  | Default         | Notes                                                                                                                                                                                                        |
| ------------------ | --------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `canvas`           | `HTMLCanvasElement`   | —               | The canvas the viewer renders into. Required.                                                                                                                                                                |
| `container`        | `HTMLElement`         | `document.body` | Host element the viewer mounts all overlays/panels/toasts/dialogs into. A non-`body` container is promoted to a containing block (`contain: layout`) so fixed overlays scope to it; restored on `dispose()`. |
| `src`              | `string`              | config          | Initial Zarr URL. Empty/missing shows the dataset browser.                                                                                                                                                   |
| `debug`            | `boolean`             | `false`         | Exposes `window.__luxarDebug` for Playwright / dev console.                                                                                                                                                  |
| `loaderConfig`     | `LoaderConfig`        | —               | Cache and prefetch flags (`noCache`, `cacheDebug`, `clearCache`, `noPrefetch`, `prefetchDebug`).                                                                                                             |
| `updateBrowserUrl` | `boolean`             | `false`         | Opt in to mirroring picked datasets into the browser URL. `bootstrapStandalone()` sets this to `true`.                                                                                                       |
| `wasmPath`         | `string`              | —               | Override for bundlers that don't resolve `import.meta.url` for WASM (webpack 4, Parcel 1, etc.).                                                                                                             |
| `workerPath`       | `string`              | —               | Same, for the data worker.                                                                                                                                                                                   |
| `renderer`         | `'webgl' \| 'webgpu'` | `'webgl'`       | Force the rendering backend. `'webgpu'` uses `WebGPURenderer` + TSL `NodeMaterial`, falling back to WebGL2 when no adapter.                                                                                  |
| `webgpuForceWebGL` | `boolean`             | `false`         | Diagnostic: with `renderer: 'webgpu'`, route through Three.js's internal WebGL2 backend while keeping the WebGPU/TSL API surface.                                                                            |
| `perfTimestamp`    | `boolean`             | `false`         | Opt in to WebGPU `timestamp-query` GPU profiling. Tiny runtime cost; ignored under WebGL.                                                                                                                    |
| `openCacheStats`   | `boolean`             | `false`         | Open the data-loading monitor (Cache tab, expanded) once the scene is wired up — useful for profiling cache behaviour.                                                                                       |
| `factories`        | `AppFactories`        | —               | Construction overrides for the heavy components built by `init()` (scene manager, recording panel, …). For tests and advanced embedders; omit for the production path.                                       |

### Programmatic API

Beyond `init()`/`dispose()`, `LuxarApp` exposes flat methods so a host page can
drive the viewer without the built-in UI. All throw if called before `init()`.

```ts
// Dataset
await app.switchDataset('https://example.com/other.zarr'); // reload in place

// nD dimensions
const dims = app.getDimensions(); // { ndim, displayed, currentStep, metadata, ranges } (cloned)
app.setDimensionValue(/* index */ 3, /* value */ 12);
await app.awaitDimensionUpdate(); // resolve once the slice data has loaded

// Camera
app.recenterCamera(); // fit/recenter on the scene (the 'F' key)
const pose = app.getCameraPose();
app.setCameraPose(pose); // e.g. restore a saved view

// Viewport — auto-resizes to the canvas via a ResizeObserver; call manually
// after a synchronous layout change you know the observer won't catch in time.
app.resize();

// Screenshot (async — WebGPU readback is async)
const blob = await app.screenshot({ format: 'png' }); // 'png' | 'webp' | 'jpeg'
```

**Events** — subscribe with `on(event, listener)`, which returns an unsubscribe:

```ts
const off = app.on('dataset-loaded', ({ src }) => console.log('loaded', src));
app.on('dataset-error', ({ src, error }) => console.error(src, error));
app.on('dimensions-changed', (dims) => updateMyUI(dims));
app.on('selection', (sel) => console.log(sel)); // { nodeName, elementIndex } | null
// off();
```

> **Note on `selection`:** fires with the element under the cursor (or `null`
> when the hover clears) on any dataset. Subscribe **before** the dataset loads
> (i.e. before `init()` / `switchDataset()`) — the GPU picking pipeline is
> provisioned at load time only when a listener exists, so picking stays
> zero-cost for pages that never consume it. Hover-driven; click-to-select is
> a planned follow-up.

### What's NOT supported in v1

- **Multiple viewers on the same page.** `ThemeManager`, the worker pool, and several UI components are still page-singletons. Mounting two `LuxarApp` instances at once will share state.
- **Shadow DOM isolation.** The viewer uses regular DOM. The CSS is prefixed under `.luxar-*` classnames, but a host page that already styles `.luxar-foo` will collide.
- **SSR / non-browser rendering.** `LuxarApp.init()` throws a friendly error if `window`/`document` are unavailable.

A runnable example with a non-trivial host page lives in
[`examples/embed/`](./examples/embed/).

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

| Key   | Action                                              |
| ----- | --------------------------------------------------- |
| **V** | Cycle control modes: Orbit -> Fly -> Ortho -> Orbit |
| **I** | Toggle inertial mode (Fly mode only)                |
| **F** | Recenter camera on scene                            |
| **C** | Toggle cinematic mode                               |

### Orbit Mode Controls

| Input                   | Action                     |
| ----------------------- | -------------------------- |
| **Left Mouse Drag**     | Pan camera                 |
| **Right Click + Drag**  | Rotate camera around scene |
| **Mouse Wheel**         | Zoom in/out                |
| **Shift + Mouse Wheel** | Roll (view-axis rotation)  |

### Fly Mode Controls

| Input                   | Action                                   |
| ----------------------- | ---------------------------------------- |
| **W/S**                 | Move forward/backward                    |
| **A/D**                 | Strafe left/right                        |
| **Alt+W / Alt+S**       | Move up/down                             |
| **Arrow Keys**          | Look up/down/left/right                  |
| **Left Mouse Drag**     | Strafe (screen-space translation)        |
| **Right Mouse Drag**    | Free look (rotate camera)                |
| **Mouse Wheel**         | Forward/backward velocity impulse        |
| **Shift + Mouse Wheel** | Roll (view-axis rotation)                |
| **I**                   | Toggle inertial physics (drift/momentum) |

### Ortho Mode Controls

| Input                   | Action                              |
| ----------------------- | ----------------------------------- |
| **Left Mouse Drag**     | Pan (Napari/Google Maps convention) |
| **Mouse Wheel**         | Zoom in/out                         |
| **Shift + Mouse Wheel** | Roll (view-axis rotation)           |

### General Controls

| Input      | Action                                   |
| ---------- | ---------------------------------------- |
| **Space**  | Toggle fullscreen mode                   |
| **H**      | Show/hide help overlay                   |
| **R**      | Toggle advanced rendering controls panel |
| **P**      | Toggle performance statistics            |
| **N**      | Toggle nD dimension panel                |
| **O**      | Open dataset browser                     |
| **Ctrl+L** | Toggle debug console                     |
| **Esc**    | Exit fullscreen / Close panels           |

### nD Navigation (for datasets with >3 dimensions)

| Input                 | Action                                           |
| --------------------- | ------------------------------------------------ |
| **Number keys (1-9)** | Select which non-displayed dimension to navigate |
| **`[` and `]`**       | Step backward/forward in the selected dimension  |
| **Dimension Sliders** | Click and drag to navigate through dimensions    |

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
  "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], // 4x4 transform matrix (optional)
  "sceneDimensions": {
    // Required for nD data
    "dimensions": [
      { "name": "x", "unit": "μm", "range": [-100, 100], "display": true },
      { "name": "y", "unit": "μm", "range": [-100, 100], "display": true },
      { "name": "z", "unit": "μm", "range": [-50, 50], "display": true },
      { "name": "time", "unit": "s", "range": [0, 10], "display": false, "step": 0.1 }
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
- **SSAA**: Super-sample anti-aliasing with resolution multipliers (1.5x, 2x, 4x)

### Performance Features

- **Real-time preview**: Changes are applied immediately with smooth animations
- **Settings persistence**: User preferences are saved between sessions
- **Performance impact indicators**: Visual feedback on rendering cost
- **Preset management**: Quick access to optimized configurations

## 🛠️ Development

### Project Structure

The viewer source tree is organized into 17 subpackages, each with its own
`README.md` documenting its files, public surface, invariants, and
dependencies in detail. The top-level shape:

```
src/
├── index.ts                  # Public-API barrel (side-effect-free)
├── lib-styles-entry.ts       # CSS-only build entry
│
├── core/                     # Application bootstrap, lifecycle, debug interface
├── config/                   # Unified configuration system (sections/ + zarr-bridge/)
├── cache/                    # 3-tier cache: L0 decompressed, L1 memory, L2 OPFS
├── data/                     # Zarr loading, nD slicing, per-geometry loaders
├── rendering/                # GLSL/TSL materials, post-processing, picking, GPU buffer pool
├── scene/                    # SceneManager + animation + scene-manager helpers
├── controls/                 # Orbit / Fly / Ortho camera controls
├── input/                    # Keyboard / mouse handlers, context routing
├── ui/                       # GUI library, panels, monitors, recording panel
├── styles/                   # CSS (base, components, themes, embed vs standalone)
├── themes/                   # Theme manager + dark/light/glass theme definitions
├── types/                    # Type definitions and ambient declarations
├── utils/                    # Cross-cutting utilities (log, event bus, HDR, platform)
├── wasm/                     # Rust kernels + TypeScript fallback (parity-tested)
├── workers/                  # Worker pool, data worker, validation
├── profiling/                # UpdateProfiler for hierarchical timing
└── tests/                    # Unit (vitest), e2e harnesses, mocks, builders, benchmarks
```

For per-subpackage details — file tables, public exports, invariants — read
the `README.md` inside the subfolder. The major subpackages also document
their own subpackages (e.g. `rendering/README.md` links to `materials/`,
`picking/`, `post-processing/`, `material-manager/`, `node-factory/`,
`gpu-buffer-pool/`).

See also: [`src/README.md`](./src/README.md) for the navigational hub and
the enforced layer order, [`CONVENTIONS.md`](./CONVENTIONS.md) for project-
wide conventions, and [`ARCHITECTURE-DIAGRAMS.md`](./ARCHITECTURE-DIAGRAMS.md)
for high-level diagrams.

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
# Prerequisite: examples + fixtures must exist. Run once locally:
#   make run-examples
#   pnpm test:generate-fixtures
# Or use `make test-e2e` from the repo root which orchestrates this.
# E2E is currently disabled in GitHub CI (browser/GPU reliability);
# `pnpm test:e2e:smoke` is the subset the workflow re-enable would
# run (also useful locally for quick verification).
pnpm test:e2e        # Run all E2E tests
pnpm test:e2e:smoke  # Run the non-GPU smoke subset
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

### Native launcher environment variables

The native launcher binaries (produced by `make build-launchers` in the
repo root and used by `luxar export --native ...`) honor:

- `LUXAR_LAUNCHER_NO_WEBVIEW=1` — Skip the embedded WebView and open the
  exported scene in the system default browser instead. Useful on
  headless / minimal Linux installs (missing libwebkit2gtk) and for
  smoke-testing the launcher itself without a graphical session.

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
    backgroundColor: 0x111111, // Dark gray background
  },
  animation: {
    idleTimeoutMs: 2000, // Auto-pause after 2 seconds
  },
  renderingControls: {
    defaults: {
      fov: 47, // Field of view in degrees (50mm Normal)
      bloomEnabled: false, // Bloom effect (opt-in via zarr viewer_config)
      bloomStrength: 0.25, // Bloom intensity multiplier
      bloomRadius: 1.0, // Blur radius for bloom spread
      bloomThreshold: 0.01, // Luminance threshold for bloom
      fxaaEnabled: false, // FXAA anti-aliasing
      msaaEnabled: false, // MSAA (hardware-accelerated, fast and sharp)
      ssaaEnabled: false, // SSAA (supersampling, highest quality, heavy cost)
      // ... more rendering options
    },
  },
  // ... more options
};
```

## 🔧 Advanced Usage

### Material Caching System

The material manager in `src/rendering/material-manager.ts` provides optimized material handling. Per-geometry getters return cached materials keyed by property bucketing:

```typescript
// Supported blending modes
type BlendingMode = 'normal' | 'additive' | 'max' | 'opaque' | 'luminous';

// Per-geometry getters — see src/rendering/material-manager.ts for
// PointMaterialProperties / LineMaterialProperties / GSplatMaterialProperties
const pointMat = materialManager.getPointMaterial({
  blendingMode: 'additive',
  opacity: 1.0,
  gamma: 2.2,
});
const lineMat = materialManager.getLineMaterial({
  /* ... */
});
const gsplatMat = materialManager.getGSplatMaterial({
  /* ... */
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
    toneMapping: 'ACES',        // Default. Options: None, Linear, Reinhard, Cineon, ACES, AgX, Neutral
                                // (use 'Neutral' for exact colormap-LUT fidelity)
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
    ssaaEnabled: false,         // SSAA: Supersampling, highest quality, heavy cost
    ssaaMultiplier: 2.0,        // SSAA resolution multiplier (1.5x, 2x, 4x)
  },
},
```

**Anti-Aliasing Notes:**

- **MSAA**: Hardware-accelerated, fast and sharp — great default for most scenes. Note: MSAA has limitations with additive blending (used by GSplats); consider FXAA for scenes with Gaussian splats
- **FXAA**: Fastest post-process AA, may slightly blur the image
- **SSAA**: Highest quality (supersampling), significant performance cost

### Performance Optimization

- **Element Count**: Optimize for datasets with millions of elements
- **Chunk Size**: Zarr chunk sizes of 16KB-256KB (target 64KB) — matches `TARGET_CHUNK_BYTES` in `luxar.typing_utils.constants`
- **LOD**: Consider implementing level-of-detail for very large datasets
- **Compression**: Use Zarr compression (e.g., blosc) to reduce network transfer

### Performance Monitoring

Built-in performance monitoring in `src/ui/performance-monitor.ts` wraps [stats.js](https://github.com/mrdoob/stats.js/) and provides:

```typescript
// PerformanceMonitor wraps stats.js for FPS, frame time, and memory tracking
const monitor = new PerformanceMonitor(); // no args; injects its own DOM panel

// Visibility control — measurement is driven by the animation loop's
// `frame-start` / `frame-end` events on the event bus. The monitor only
// subscribes while visible, so stats.js incurs no cost when hidden.
monitor.show(); // Show the stats panel (subscribes to frame timing)
monitor.hide(); // Hide the stats panel (unsubscribes)
monitor.toggle(); // Toggle visibility
monitor.cyclePanels(); // Rotate FPS -> frame time -> memory
monitor.visible; // boolean getter for current visibility
```

- **Real-time FPS**: Continuously updated frame rate display (panel 0)
- **Frame timing**: Milliseconds per frame (panel 1)
- **Memory usage**: JavaScript heap size monitoring (panel 2)
- **Panel cycling**: `cyclePanels()` rotates through FPS, frame time, and memory views
- **Idle optimization**: Only subscribes to frame timing while visible to avoid overhead

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
- Use FXAA for lightweight post-process smoothing
- Use SSAA only for final renders (heavy performance cost)

**nD navigation not working**

- Verify `sceneDimensions` are defined in dataset `.zattrs`
- Check that non-displayed dimensions have proper `range` and `step` values
- Ensure dimension count matches position data shape

### Browser Compatibility

| Browser | Version | Status             |
| ------- | ------- | ------------------ |
| Chrome  | 90+     | ✅ Fully supported |
| Firefox | 88+     | ✅ Fully supported |
| Safari  | 15+     | ✅ Supported       |
| Edge    | 90+     | ✅ Fully supported |

### WebGL Requirements

- WebGL 2.0 support required
- Float texture support (for HDR rendering)
- Minimum 2GB GPU memory recommended for large datasets

## 📝 API Reference

### URL Parameters

- `?src=<path>` — Path to Zarr dataset
- `?debug` — Expose `window.__luxarDebug` for Playwright / dev console
- `?no-cache` — Disable all cache tiers (L0 + L1 + L2) for this session
- `?cache-debug` — Verbose cache logging
- `?clear-cache` — Clear all caches before loading
- `?no-prefetch` — Disable adjacent-chunk prefetching (caches still active)
- `?prefetch-debug` — Verbose prefetch logging
- `?renderer=webgpu` — Use `WebGPURenderer` (TSL `NodeMaterial`) instead of the default `WebGLRenderer`
- `?renderer=webgpu&webgpu-force-webgl` — TSL/WebGPU API surface but Three.js routes through its internal WebGL2 backend (diagnostic)

### Programmatic Usage

```javascript
import { LuxarApp } from './src/core/app.js';
import { config } from './src/config/index.js';

const app = new LuxarApp();
await app.init({
  canvas: document.getElementById('app'),
  src: '/path/to/dataset.zarr',
});

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
