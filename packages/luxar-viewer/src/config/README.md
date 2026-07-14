# Config Package

Unified configuration system for the Luxar player application. This package centralizes all configuration values to ensure consistency and make the application easy to customize.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Configuration Sections](#configuration-sections)
- [Usage Examples](#usage-examples)
- [Type Safety](#type-safety)
- [Best Practices](#best-practices)
- [Subpackages](#subpackages)
- [Public API / Exports](#public-api--exports)

## Overview

The config package provides a centralized configuration system that manages all application settings from rendering parameters to UI behavior. It uses TypeScript's strict typing system to ensure configuration consistency across all application components.

**Key Philosophy**: All configuration values should be defined in one place with clear documentation, proper typing, and sensible defaults.

## Key Features

- **Unified Configuration**: Single source of truth for all application settings
- **Type Safety**: Complete TypeScript interfaces for all configuration sections
- **Hierarchical Organization**: Logical grouping of related configuration values
- **Detailed Documentation**: Extensive comments explaining each setting's purpose
- **Default Values**: Carefully chosen defaults for optimal user experience
- **Debug Console**: Dedicated configuration for debugging features

## Architecture

```text
config/
├── index.ts                       # Composes section literals into the `config` AppConfig
├── types.ts                       # Barrel re-exports of section types + AppConfig interface
├── validation.ts                  # Dispatcher; imports per-section validators
├── url-params.ts                  # URL ?param=value parsing (self-contained)
├── user-settings.ts               # Persisted global prefs (localStorage luxar.settings) —
│                                  #   Settings-popover model; live values mutate config, startup
│                                  #   values thread through bootstrap (URL params always win)
├── constants.ts                   # WASM ABI constants
├── sections/
│   ├── camera/             {data,types,validate}.ts
│   ├── animation/          {data,types}.ts
│   ├── adaptive-dpr/       {data,types,validate}.ts
│   ├── scene/              {data,types,validate}.ts   # includes ShaderConfig
│   ├── ui/                 {data,types}.ts            # includes DebugConsoleConfig, UIComponentsConfig
│   ├── rendering-controls/ {data,types,validate}.ts   # includes RenderingSettings, validateBloomConsistency
│   ├── controls/           {data,types,validate}.ts   # includes Fly/Orbit/ScaleMultipliers/ConfigRange
│   ├── input/              {data,types,validate}.ts
│   ├── data-loading/       {data,types,validate}.ts   # composes the 5 sub-sections below
│   │   ├── spatial/        {data,types}.ts
│   │   ├── network/        {data,types,validate}.ts
│   │   ├── memory/         {data,types,validate}.ts
│   │   ├── monitor/        {data,types}.ts
│   │   └── performance/    {data,types,validate}.ts
│   ├── webgl/              {data,types,validate}.ts
│   ├── cache/              {data,types,validate}.ts
│   └── dimension-animation/{data,types}.ts
└── zarr-bridge/
    ├── viewer-config-utils.ts     # snake_case ↔ camelCase conversion for zarr viewer_config
    └── viewer-state-capture.ts    # Captures complete viewer state for export (Ctrl+Shift+S)
```

The architecture groups each configuration section's data + types + validator into a `sections/<name>/` triplet:

- `index.ts` is a thin orchestrator (~60 lines) that imports each section's `Config` literal and composes them into the `AppConfig` object.
- `types.ts` is a re-export barrel: every section type is imported then re-exported, plus the `AppConfig` interface that ties them together.
- `validation.ts` is a dispatcher: it imports each section's `validate<Section>` function and calls them in `validateConfig`.
- `zarr-bridge/viewer-config-utils.ts` converts between zarr `viewer_config` (snake_case) and TypeScript `RenderingSettings` (camelCase), extracts camera overrides and background color.
- `zarr-bridge/viewer-state-capture.ts` captures the live viewer state (camera, rendering settings, theme, dimensions, animation) as a `ZarrViewerConfig` JSON object for clipboard export.

Each `sections/<name>/data.ts` exports a single `<name>Config` constant; the public type symbols (e.g. `CameraConfig`, `RenderingSettings`) are re-exported through the root `types.ts` barrel, so external consumers continue to `import type { CameraConfig } from 'src/config/types'`.

## Configuration Sections

### Camera Configuration

Controls 3D perspective, navigation, and viewing parameters:

```typescript
camera: {
  // Note: fov, near, far live in renderingControls.defaults as the single source of truth
  fovMin: 10,                 // Zoom limits
  fovMax: 170,                // Must be <180°
  fovSensitivity: 0.05        // Ctrl+wheel zoom sensitivity
}
// Default FOV (47, 50mm Normal) is in renderingControls.defaults.fov
```

### Shader Configuration

Point rendering uses `falloff * opacity` for alpha, matching line material behavior. Per-node color adjustment (intensity, offset, gamma) is configured per-material. Global exposure/offset/gamma are in `renderingControls.defaults` and applied inside the mega-shader post-processing pass.

### User Interface Configuration

UI element behavior, timing, and styling:

```typescript
ui: {
  zIndex: {
    loading: 1000,
    error: 1000,
    help: 1001
  },
  timings: {
    errorAutoDismissMs: 10000,    // Auto-hide errors
    helpClickDelayMs: 100         // Help interaction delay
  },
  spinner: {
    size: 24,                     // Loading spinner size
    borderWidth: 3
  }
}
```

### Adaptive DPR (Device Pixel Ratio)

Dynamic resolution scaling to maintain smooth frame rates:

```typescript
adaptiveDPR: {
  enabled: true,               // Enable adaptive DPR system
  minDPR: 0.5,                 // Minimum allowed DPR (quality floor)
  scaleDownFactor: 0.9,        // Factor when scaling down (10% reduction)
  scaleUpFactor: 1.05,         // Factor when scaling up (5% increase)
  hysteresisSeconds: 3,        // Sustained high-FPS time required before scaling up
  evaluationIntervalMs: 500,   // How often to evaluate FPS (ms)

  // Refresh-rate-relative thresholds (replace fixed minFPS/maxFPS)
  scaleDownFpsRatio: 0.75,     // Scale down below 75% of the display's rAF cap
  scaleUpFpsRatio: 0.90,       // Count toward scale-up above 90% of the cap
  refreshRateFallback: 60,     // Cap assumed before the estimator warms up
  midbandGraceSamples: 1,      // Mid-band samples tolerated before the streak resets

  // U-shape probe (every scale-down is verified before it stands)
  probeWindowMs: 1500,         // Wait before judging a scale-down's effect
  probeImprovement: 1.05,      // Required FPS gain to keep the move
  probeMinSamples: 8,          // Min frame samples to settle a probe

  // Learned floor + exponential backoff on repeated rejections
  floorTtlMs: 30_000,          // First rung: floor stays 30s
  backoffMultiplier: 2,        // 30s → 60s → 2min → ...
  backoffMaxTtlMs: 300_000,    // ... capped at 5min

  // Evidence-based ceiling (HiDPI → 1.0 demotion under sustained load)
  ceilingTtlMs: 60_000,
  punishedAscentWindowMs: 3000,
  punishedAscentThreshold: 2,

  // Session hygiene
  contentChangeRecheckMs: 5000, // Content-change coalescing / early re-probe
  gapResetMs: 350               // Frame gap that resets the FPS window
}
```

**How It Works**:

- FPS thresholds are relative to the display's achievable rAF rate, so
  60/120/144Hz monitors and 30Hz low-power throttling all behave sanely
- When FPS drops below `scaleDownFpsRatio × cap`, DPR is reduced by
  `scaleDownFactor` and a probe verifies the move actually helped
  (rejected moves revert and set a floor; repeated rejections back off)
- When FPS stays above `scaleUpFpsRatio × cap` for `hysteresisSeconds`,
  DPR increases by `scaleUpFactor`
- Asymmetric scaling (slower up, faster down) prevents quality oscillation
- `minDPR` prevents image from becoming too pixelated
- Validation for the cross-field invariants lives in
  `sections/adaptive-dpr/validate.ts`

### Rendering Controls

User-adjustable settings with persistence:

```typescript
renderingControls: {
  defaults: {
    // Camera clipping
    dynamicClippingEnabled: true,   // Auto-adjust clipping planes per frame
    // Bloom and HDR
    bloomEnabled: false,
    bloomThreshold: 0.01,
    bloomStrength: 0.25,
    bloomRadius: 1.0,
    bloomLevels: 8,
    // Global EOG (Exposure-Offset-Gamma) applied in the mega-shader
    exposure: 0.0,
    globalOffset: 0.0,
    globalGamma: 1.0,
    // Anti-aliasing
    fxaaEnabled: false,
    msaaEnabled: false,         // Incompatible with additive blending
    msaaSamples: 4,
    ssaaEnabled: false,
    toneMapping: 'ACES',        // Default; use 'Neutral' for exact colormap-LUT fidelity
    // Detector noise (physics-based: Poisson + Gaussian + FPN)
    detectorNoiseEnabled: false,
    detectorNoiseReadoutSigma: 0.002,  // Temporal readout noise
    detectorNoisePhotonGain: 0.002,    // Shot noise visibility
    detectorNoiseFpnSigma: 0.001,     // Fixed pattern noise
    vignetteEnabled: false,
    controlType: 'orbit',       // vs 'fly' or 'ortho'
    autoRotate: false
  }
}
```

### WebGL Configuration

WebGL context and renderer settings for optimal 3D rendering:

```typescript
webgl: {
  context: {
    alpha: false,                       // No canvas transparency
    antialias: false,                  // Backbuffer MSAA off (scene renders
                                       // to the HDR target; renderTarget.samples
                                       // controls real MSAA)
    depth: true,                       // Enable depth buffer
    stencil: false,                    // No stencil (saves memory)
    powerPreference: 'high-performance', // GPU preference
    preserveDrawingBuffer: false,      // Better performance
    desynchronized: true,              // Async updates
  },
  renderer: {
    precision: 'highp',                // Shader precision
    logarithmicDepthBuffer: false,    // Standard depth (faster)
  },
  renderTarget: {
    depthBuffer: true,                 // Depth testing
    stencilBuffer: false,              // No stencil
    samples: 0,                        // MSAA samples
  },
}
```

### Cache Configuration

OPFS-based zarr caching with a three-level hierarchy:

```typescript
cache: {
  enabled: true,              // Enable OPFS caching
  l0Enabled: true,            // L0: decompressed chunk cache (fastest)
  l0MaxSizeMB: 200,           // L0 memory budget — CEILING; heap-aware sizing may scale it down (heap-budget.ts)
  sliceCacheEnabled: true,    // S-cache: per-(node,view) decoded-slice cache (instant slice revisits)
  sliceCacheMaxSizeMB: 128,   // S-cache budget — fixed FALLBACK (no performance.memory) + shrink floor; heap-aware sizing scales it up/down
  l1MaxSizeMB: 100,           // L1 in-memory LRU — CEILING; heap-aware sizing may scale it down
  l2MaxSizeMB: 2048,          // L2: persistent OPFS cache (disk, fixed — not heap-sized)
  opfsOperationTimeoutMs: 10000, // Per-OPFS-operation deadline (ms)
  externalDatasetTtlMs: null, // Optional TTL (ms) for non-local datasets; null = no expiry
  debug: false                // Enable cache debug logging
}
```

**Cache Levels**:

- **L0** (Decompressed chunks): Fastest access, holds decompressed zarr chunks in memory
- **L1** (Memory LRU): In-memory cache with LRU eviction
- **L2** (OPFS): Browser-native persistent storage, survives page reloads

### Dimension Animation Configuration

Controls FPS-based animation through dimension ranges (e.g., animating through time slices):

```typescript
dimensionAnimation: {
  defaults: {
    targetFPS: 10,                    // Default animation speed
    loop: 'loop',                     // 'once' | 'loop' | 'bounce'
    direction: 'forward'              // 'forward' | 'backward'
  },
  presets: {
    fps: [1, 2, 5, 10, 15, 30, 60],   // Quick FPS presets
    customMin: 0.1,                   // Minimum custom FPS
    customMax: 120                    // Maximum custom FPS
  },
  timing: {
    minFrameTimeMs: 16,               // Minimum frame duration
    continuousTraverseSeconds: 10     // Duration for continuous traverse mode
  },
  ui: {
    showFPSFeedback: true,            // Show FPS feedback in UI
    feedbackThreshold: 0.8            // Warning when actual FPS < target * threshold
  },
  playback: {
    budgetFraction: 0.6,              // Fraction of the frame window handed to progressive loaders per tick
    minBudgetMs: 8,                   // Budget floor at high target FPS
    overheadReserveMs: 50             // Slow FPS: budget = frame window − reserve (projection/commit/render)
  }
}
```

**Loop Modes**:

- **once**: Play through the range once and stop
- **loop**: Restart from beginning when reaching the end
- **bounce**: Reverse direction at each end (ping-pong)

### Data Loading Performance Configuration

Performance optimization pipeline (object pooling, web workers, WASM,
GPU buffer pool):

```typescript
dataLoading: {
  performance: {
    // Object pooling (reuse buffers across updates)
    useAccumulators: true,
    initialAccumulatorCapacity: 8192,
    accumulatorGrowthFactor: 1.5,

    // Web Workers (offload CPU work)
    useWebWorkers: true,
    workerCount: 0,                  // 0 = auto-detect based on navigator.hardwareConcurrency


    // GPU buffer pool (reuse WebGL buffers)
    useGPUBufferPool: true,
    gpuPoolMaxSize: 20,
    gpuPoolEvictionFrames: 300,

    // Debugging
    enablePerformanceMonitoring: false
  }
}
```

**Components**:

- **Accumulators**: Pre-allocate and reuse typed arrays to avoid GC pressure
- **Workers**: Offload nD projection and visibility computation to Web Workers
- **WASM**: Rust-compiled WebAssembly for spatial queries, visibility, and decoding
- **GPU Buffer Pool**: Reuse WebGL buffer objects to avoid GPU allocation overhead

### Debug Console Configuration

Development and debugging features. The debug console settings live under
`config.ui.debugConsole` (see the `ui/` section), not at the top level:

```typescript
// Access via config.ui.debugConsole
config.ui.debugConsole = {
  panel: {
    defaultWidth: 600,
    defaultHeight: 400,
    minWidth: 400,
    maxWidth: 1200,
    // minHeight, maxHeight, bottomOffset, leftOffset also defined
  },
  interceptor: {
    maxBufferSize: 10000, // Ring buffer for console messages
  },
  // resize + style sub-objects also defined
};
```

## Usage Examples

### Importing Configuration

```typescript
import { config } from '../config';
import type { AppConfig, RenderingSettings } from '../config';

// Access specific configuration sections
const cameraSettings = config.camera;
const bloomSettings = config.renderingControls.defaults;
```

### Using Configuration in Components

```typescript
// In SceneManager
const camera = new THREE.PerspectiveCamera(
  config.renderingControls.defaults.fov,
  aspectRatio,
  config.renderingControls.defaults.near,
  config.renderingControls.defaults.far
);

// In PostProcessing
postProcessing.setBloomEnabled(
  config.renderingControls.defaults.bloomEnabled,
  config.renderingControls.defaults.bloomStrength,
  config.renderingControls.defaults.bloomRadius,
  config.renderingControls.defaults.bloomThreshold
);
postProcessing.setBloomLevels(config.renderingControls.defaults.bloomLevels);
```

### Customizing Configuration

```typescript
// For development or testing, create modified config
const testConfig = {
  ...config,
  renderingControls: {
    ...config.renderingControls,
    defaults: {
      ...config.renderingControls.defaults,
      bloomStrength: 0.1, // Reduce bloom for testing
    },
  },
};
```

## Type Safety

The config package provides complete TypeScript interfaces for all configuration sections:

```typescript
interface AppConfig {
  camera: CameraConfig;
  animation: AnimationConfig;
  adaptiveDPR: AdaptiveDPRConfig;
  scene: SceneConfig;
  shader: ShaderConfig;
  ui: UIConfig;
  renderingControls: RenderingControlsConfig;
  controls: ControlsConfig;
  input: InputConfig;
  dataLoading: DataLoadingConfig;
  webgl: WebGLConfig;
  cache: CacheConfig;
  dimensionAnimation: DimensionAnimationConfig;
  defaultZarrPath: string;
}

interface RenderingSettings {
  // Dynamic clipping planes
  dynamicClippingEnabled: boolean;
  // Bloom and HDR
  bloomThreshold: number;
  bloomStrength: number;
  // ... all user-adjustable settings
  toneMapping: 'None' | 'Linear' | 'Reinhard' | 'Cineon' | 'ACES' | 'AgX' | 'Neutral';
  controlType: 'orbit' | 'fly' | 'ortho';
}
```

**Benefits of Strong Typing**:

- Compile-time validation of configuration values
- IntelliSense support in IDEs
- Safer changes when configuration structure evolves
- Self-documenting configuration interface

## Best Practices

### Configuration Organization

1. **Logical Grouping**: Related settings should be grouped together
2. **Clear Naming**: Use descriptive names that indicate purpose and units
3. **Documentation**: Every setting should have a comment explaining its purpose
4. **Sensible Defaults**: Defaults should provide good user experience out of the box

### Performance Considerations

```typescript
// Use 'as const' for literal type inference
export const config: AppConfig = {
  // ... configuration
} as const;

// This ensures TypeScript treats values as literals, not generic types
// Enables better type checking and optimization
```

### Accessing Configuration

```typescript
// ✅ Good: Import specific sections when possible
import { config } from '../config';
const { fov, near, far } = config.renderingControls.defaults;

// ✅ Good: Use type imports for interfaces
import type { CameraConfig } from '../config';

// ❌ Avoid: Don't modify configuration at runtime
// config.renderingControls.defaults.fov = 90; // This would break immutability
```

### Adding New Configuration

1. **Add to Types**: Define the interface in `types.ts`
2. **Add to Config**: Include the actual values in `index.ts`
3. **Document**: Provide clear comments explaining the purpose
4. **Test**: Ensure the configuration works in relevant components

### Debug Configuration

```typescript
// Debug console configuration is now part of main config
import { config } from '../config';

// Access debug settings
const panelWidth = config.ui.debugConsole.panel.defaultWidth;
const bufferSize = config.ui.debugConsole.interceptor.maxBufferSize;
```

### Versioning

When updating configuration:

1. **Compatibility**: Consider existing users
2. **Deprecation Strategy**: Stage breaking changes clearly when needed
3. **Validation**: Ensure new values are within acceptable ranges
4. **Documentation**: Update comments and examples

The config package is the foundation that ensures consistent behavior across all Luxar components while providing flexibility for customization and future enhancements.

---

## Subpackages

- **[sections/](sections/README.md)** — per-slice `data.ts` / `types.ts` /
  (optional) `validate.ts` triples; one folder per `AppConfig` section.
  Sections without cross-field invariants (`animation/`,
  `dimension-animation/`, `ui/`) omit `validate.ts`.
- **[zarr-bridge/](zarr-bridge/README.md)** — snake_case ↔ camelCase
  conversion between zarr `viewer_config` and `RenderingSettings`, plus the
  live-viewer state capture used by the `Ctrl+Shift+S` export.

## See Also

- `./index.ts` — composes section literals into the `config: AppConfig` export.
- `./types.ts` — barrel re-export of all section types + the `AppConfig` interface.
- `./validation.ts` — central dispatcher that invokes each section's
  `validate*` function.
- `./url-params.ts` — single point where `window.location.search` is read.
- `./constants.ts` — `MAX_SUPPORTED_DIMS` (mirrored in Rust/WASM).

---

## Public API / Exports

From `./index.ts`:

- `config: AppConfig` — the composed application configuration object.
- `AppConfig`, `RenderingSettings` — type re-exports (full set lives in `./types.ts`).

From `./validation.ts`:

- `validateConfig(config)` — dispatcher returning `{ valid, errors, warnings }`.
- `logValidationResults(result)`, `validateAndLog(config)`.

From `./url-params.ts`:

- `readUrlParams(search?)` — typed snapshot of recognized `?param=value` pairs.
- `normalizeDataSourceUrl(rawSrc)`, `buildDataSourceBrowserUrl(src, location)`,
  `replaceBrowserDataSourceUrl(src, target?)`.

From `./constants.ts`:

- `MAX_SUPPORTED_DIMS = 16` — mirrored in Rust as the WASM ABI cap on nD.
