# luxar-viewer.config - Technical Specification

**Version**: 1.2.0
**Last Updated**: 2025-12-09

## Purpose

The `luxar-viewer.config` package provides a unified, type-safe configuration system for all application settings. It serves as the single source of truth for default values, limits, and behavior parameters across all components.

**Core Responsibility**: Centralize all configuration values with proper TypeScript typing, clear documentation, and sensible defaults to ensure consistency and ease of customization.

---

## Table of Contents

1. [Configuration Architecture](#1-configuration-architecture)
2. [Camera Configuration](#2-camera-configuration)
3. [Controls Configuration](#3-controls-configuration)
4. [Animation Configuration](#4-animation-configuration)
5. [Scene Configuration](#5-scene-configuration)
6. [Shader Configuration](#6-shader-configuration)
7. [PostProcessing Configuration](#7-postprocessing-configuration)
8. [UI Configuration](#8-ui-configuration)
9. [Input Configuration](#9-input-configuration)
10. [Data Loading Configuration](#10-data-loading-configuration)
11. [WebGL Configuration](#11-webgl-configuration)
12. [Cache Configuration](#12-cache-configuration)
13. [Rendering Controls Configuration](#13-rendering-controls-configuration)
14. [Type Safety & Validation](#14-type-safety--validation)

---

## 1. Configuration Architecture

### 1.1 Structure

```typescript
// config/index.ts - Exports the main config object
export const config: AppConfig = {
    camera: { ... },
    animation: { ... },
    scene: { ... },
    shader: { ... },
    postProcessing: { ... },
    ui: { ... },
    renderingControls: { ... },
    controls: { ... },
    input: { ... },
    dataLoading: { ... },
    webgl: { ... },
    cache: { ... },
    defaultZarrPath: '',
    canvasId: 'app'
}

// config/types.ts - Type definitions
export interface AppConfig {
    camera: CameraConfig
    animation: AnimationConfig
    scene: SceneConfig
    shader: ShaderConfig
    postProcessing: PostProcessingConfig
    ui: UIConfig
    renderingControls: RenderingControlsConfig
    controls: ControlsConfig
    input: InputConfig
    dataLoading: DataLoadingConfig
    webgl: WebGLConfig
    cache: CacheConfig
    defaultZarrPath: string
    canvasId: string
}
```

**Design Principle**: Configuration is **read-only** at runtime (`as const`). Modifications create new config objects rather than mutating existing ones.

---

## 2. Camera Configuration

### 2.1 Purpose

Controls 3D perspective, field of view, clipping planes, and professional lens presets including realistic lens distortion.

### 2.2 Structure

```typescript
interface CameraConfig {
  fov: number;
  near: number;
  far: number;
  initialPosition: { x: number; y: number; z: number };
  fovMin: number;
  fovMax: number;
  fovSensitivity: number;
  fovPresets: Record<string, number>;
  lensDistortionPresets: Record<
    string,
    {
      distortionX: number;
      distortionY: number;
      principalPointX: number;
      principalPointY: number;
      focalLengthX: number;
      focalLengthY: number;
      skew: number;
    }
  >;
}
```

### 2.3 Defaults

- **fov**: 47° (50mm Normal lens equivalent, natural human vision)
- **near**: 0.1 (objects closer are not rendered)
- **far**: 1000 (objects further are not rendered)
- **initialPosition**: `{ x: 0, y: 0, z: 8 }`
- **fovMin**: 10° (prevents excessive zoom-in)
- **fovMax**: 200° (prevents excessive zoom-out)
- **fovSensitivity**: 0.05 (Shift+wheel FOV change rate)
- **fovPresets**: Professional lens equivalents
  - `'28mm Wide'`: 75° horizontal FOV
  - `'35mm'`: 63° horizontal FOV
  - `'50mm Normal'`: 47° horizontal FOV (default)
  - `'85mm Portrait'`: 29° horizontal FOV
  - `'135mm Tele'`: 18° horizontal FOV
  - `'Custom'`: -1 (preserves slider value)
- **lensDistortionPresets**: Realistic lens characteristics per focal length
  - Wide angle (28mm): Barrel distortion (-0.07)
  - Normal (50mm): No distortion (0.0)
  - Telephoto (135mm): Barrel distortion (0.07)

### 2.4 Usage Example

```typescript
import { config } from '../config';

// Initialize camera with defaults
const camera = new PerspectiveCamera(
  config.camera.fov,
  aspectRatio,
  config.camera.near,
  config.camera.far
);
camera.position.set(
  config.camera.initialPosition.x,
  config.camera.initialPosition.y,
  config.camera.initialPosition.z
);

// Apply FOV preset
const presetFov = config.camera.fovPresets['85mm Portrait']; // 29°
camera.fov = presetFov;
camera.updateProjectionMatrix();

// Apply matching lens distortion
const distortion = config.camera.lensDistortionPresets['85mm Portrait'];
lensDistortionPass.uniforms.distortion.value.set(distortion.distortionX, distortion.distortionY);
```

### 2.5 Validation Rules

- **fov**: Must be 1-180°
- **near**: Must be > 0
- **far**: Must be > near
- **fovMin**: Must be < fovMax
- **fovSensitivity**: Typical range 0.01-0.2

---

## 3. Controls Configuration

### 3.1 Purpose

Defines behavior for fly and orbit control systems, including movement physics, damping, rotation, and zoom parameters.

### 3.2 Structure

```typescript
interface ControlsConfig {
  fly: FlyControlsConfig;
  orbit: OrbitControlsConfig;
}

interface FlyControlsConfig {
  inertialMode: { default: boolean };
  movement: {
    speed: ConfigRange;
    acceleration: ConfigRange;
    damping: ConfigRange;
  };
  rotation: {
    speed: ConfigRange;
    damping: ConfigRange;
  };
  look: {
    mouseSpeed: ConfigValue;
  };
  physics: {
    velocityThreshold: number;
    dampingPower: number;
    angularVelocityThreshold: number;
  };
}

interface OrbitControlsConfig {
  autoRotate: { speed: ConfigRange };
  zoom: {
    minDistance: number;
    maxDistance: number;
    speed: ConfigRange;
  };
  damping: {
    enabled: boolean;
    factor: ConfigRange;
  };
}

interface ConfigRange {
  min: number;
  max: number;
  default: number;
  step?: number;
}
```

### 3.3 Defaults

**Fly Controls**:

- **inertialMode.default**: `true` (smooth physics-based movement)
- **movement.speed**: min: 0.5, max: 50.0, default: 5.0, step: 0.1
- **movement.acceleration**: min: 0.1, max: 2.0, default: 0.5, step: 0.1
- **movement.damping**: min: 0.9, max: 0.99999, default: 0.999, step: 0.0001
- **rotation.speed**: min: 0.1, max: 5.0, default: 1.5, step: 0.1
- **rotation.damping**: min: 0.9, max: 0.9999, default: 0.99, step: 0.0001
- **look.mouseSpeed.default**: 0.002
- **physics.velocityThreshold**: 1e-4 (stop threshold)
- **physics.dampingPower**: 60 (frame-rate independent damping)
- **physics.angularVelocityThreshold**: 1e-4 (rotation stop threshold)

**Orbit Controls**:

- **autoRotate.speed**: min: 0.1, max: 5.0, default: 0.25, step: 0.1
- **zoom.minDistance**: 0.1
- **zoom.maxDistance**: 1000
- **zoom.speed**: min: 0.5, max: 2.0, default: 1.0, step: 0.1
- **damping.enabled**: `true`
- **damping.factor**: min: 0.01, max: 0.3, default: 0.05, step: 0.01

### 3.4 Usage Example

```typescript
import { config } from '../config';

// Initialize fly controls with config defaults
const flyControls = new FlyControls(camera, domElement);
flyControls.movementSpeed = config.controls.fly.movement.speed.default; // 5.0
flyControls.damping = config.controls.fly.movement.damping.default; // 0.999
flyControls.inertialMode = config.controls.fly.inertialMode.default; // true

// Create UI slider from config range
const speedRange = config.controls.fly.movement.speed;
createSlider('Movement Speed', speedRange.min, speedRange.max, speedRange.default, speedRange.step);
```

### 3.5 Validation Rules

- **movement.damping**: Must be 0.9-0.99999 (higher = less damping)
- **physics.velocityThreshold**: Must be > 0
- **zoom.minDistance**: Must be > 0
- **zoom.maxDistance**: Must be > minDistance

---

## 4. Animation Configuration

### 4.1 Purpose

Controls animation loop timing, idle behavior, and performance targets for power efficiency and smooth rendering.

### 4.2 Structure

```typescript
interface AnimationConfig {
  idleTimeoutMs: number;
  targetFPS: number;
  minFPS: number;
}
```

### 4.3 Defaults

- **idleTimeoutMs**: 2000 (pause animation after 2s of no interaction to save power)
- **targetFPS**: 60 (ideal frame rate)
- **minFPS**: 30 (minimum acceptable FPS before quality reduction)

### 4.4 Usage Example

```typescript
import { config } from '../config';

class AnimationManager {
  private lastInteraction = Date.now();
  private idleTimeout = config.animation.idleTimeoutMs;

  update() {
    const now = Date.now();
    const timeSinceInteraction = now - this.lastInteraction;

    if (timeSinceInteraction > this.idleTimeout) {
      // Pause rendering to save power
      this.pauseAnimation();
    }
  }

  onUserInteraction() {
    this.lastInteraction = Date.now();
    this.resumeAnimation();
  }
}
```

### 4.5 Validation Rules

- **idleTimeoutMs**: Must be > 0 (typical: 1000-5000ms)
- **targetFPS**: Typical: 30-144
- **minFPS**: Must be > 0 and < targetFPS

---

## 5. Scene Configuration

### 5.1 Purpose

Defines 3D scene visual properties including background color and default camera framing behavior.

### 5.2 Structure

```typescript
interface SceneConfig {
  backgroundColor: number;
  defaultFitRatio: number;
}
```

### 5.3 Defaults

- **backgroundColor**: `0x111111` (dark gray for good contrast with points)
- **defaultFitRatio**: 0.75 (fit bounds to 75% of view when centering camera)

### 5.4 Usage Example

```typescript
import { config } from '../config';

const scene = new THREE.Scene();
scene.background = new THREE.Color(config.scene.backgroundColor);

// Fit camera to bounds
function fitCameraToBounds(bounds: Box3) {
  const center = bounds.getCenter(new Vector3());
  const size = bounds.getSize(new Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = camera.fov * (Math.PI / 180);
  const distance = maxDim / (2 * Math.tan(fov / 2) * config.scene.defaultFitRatio);
  camera.position.copy(center).add(new Vector3(0, 0, distance));
}
```

### 5.5 Validation Rules

- **backgroundColor**: Must be valid hex color (0x000000-0xFFFFFF)
- **defaultFitRatio**: Must be 0-1 (typical: 0.5-0.9)

---

## 6. Shader Configuration

### 6.1 Purpose

Controls shader-specific rendering parameters for point rendering, including HDR intensity and alpha blending.

### 6.2 Structure

```typescript
interface ShaderConfig {
  points: {
    hdrMultiplier: number;
    baseAlpha: number;
  };
}
```

### 6.3 Defaults

- **points.hdrMultiplier**: 16.0 (HDR color multiplier for bloom effects)
- **points.baseAlpha**: 0.01 (base alpha intensity for additive blending)

### 6.4 Usage Example

```typescript
import { config } from '../config';

const pointsMaterial = new THREE.ShaderMaterial({
  uniforms: {
    uHdrMultiplier: { value: config.shader.points.hdrMultiplier },
    uBaseAlpha: { value: config.shader.points.baseAlpha },
  },
  vertexShader: `...`,
  fragmentShader: `
    void main() {
      vec3 hdrColor = color * ${config.shader.points.hdrMultiplier};
      gl_FragColor = vec4(hdrColor, ${config.shader.points.baseAlpha});
    }
  `,
  blending: THREE.AdditiveBlending,
  transparent: true,
});
```

### 6.5 Validation Rules

- **hdrMultiplier**: Must be >= 0 (typical: 1.0-100.0)
- **baseAlpha**: Must be 0-1 (typical: 0.001-0.1 for additive)

---

## 7. PostProcessing Configuration

### 7.1 Purpose

Controls HDR rendering, tone mapping, and post-processing effects including bloom, depth of field, ambient occlusion, vignette, chromatic aberration, detector noise, and lens distortion.

### 7.2 Structure

```typescript
interface PostProcessingConfig {
  hdr: {
    renderTargetType: typeof HalfFloatType;
  };
  toneMapping: {
    initial: {
      outputColorSpace: typeof LinearSRGBColorSpace;
      toneMapping: typeof NoToneMapping;
    };
    final: {
      outputColorSpace: typeof SRGBColorSpace;
      toneMapping: typeof ACESFilmicToneMapping;
    };
  };
}
```

**Note**: Bloom and other effect settings are in `renderingControls.defaults` for centralization (see section 13).

### 7.3 Defaults

- **hdr.renderTargetType**: `THREE.HalfFloatType` (16-bit float for HDR precision without banding)
- **toneMapping.initial.outputColorSpace**: `THREE.LinearSRGBColorSpace`
- **toneMapping.initial.toneMapping**: `THREE.NoToneMapping` (preserve HDR values)
- **toneMapping.final.outputColorSpace**: `THREE.SRGBColorSpace`
- **toneMapping.final.toneMapping**: `THREE.ACESFilmicToneMapping` (cinematic look)

### 7.4 Usage Example

```typescript
import { config } from '../config';
import * as THREE from 'three';

// Create HDR render target
const renderTarget = new THREE.WebGLRenderTarget(width, height, {
  type: config.postProcessing.hdr.renderTargetType,
  colorSpace: config.postProcessing.toneMapping.initial.outputColorSpace,
});

// Initial renderer (no tone mapping for HDR pipeline)
renderer.outputColorSpace = config.postProcessing.toneMapping.initial.outputColorSpace;
renderer.toneMapping = config.postProcessing.toneMapping.initial.toneMapping;

// Final output pass (apply tone mapping)
const outputPass = new OutputPass();
outputPass.outputColorSpace = config.postProcessing.toneMapping.final.outputColorSpace;
outputPass.toneMapping = config.postProcessing.toneMapping.final.toneMapping;
```

### 7.5 Post-Processing Effects

All effect settings are in `renderingControls.defaults` (section 13):

**Bloom Effects**:

- **bloomThreshold**: 0.01 (luminance threshold, lower = more bloom)
- **bloomStrength**: 0.25 (intensity multiplier)
- **bloomRadius**: 1.0 (blur spread)
- **bloomLevels**: 8 (mipmap levels, 1-12)

**Depth of Field**:

- **dofEnabled**: false
- **dofFocus**: 10 (focus distance)
- **dofStrength**: 0.5 (blur strength, 0-1)

**Ambient Occlusion**:

- **aoEnabled**: false
- **aoQuality**: 'medium' ('low' | 'medium' | 'high' | 'ultra')

**Vignette**:

- **vignetteEnabled**: false
- **vignetteDarkness**: 0.5 (0-1)
- **vignetteOffset**: 0.5 (distance from center, 0-1)

**Chromatic Aberration**:

- **chromaticAberrationEnabled**: false
- **chromaticAberrationStrength**: 0.15

**Detector Noise** (physics-based: Poisson + Gaussian + FPN):

- **detectorNoiseEnabled**: false
- **detectorNoiseReadoutSigma**: 0.01 (temporal readout noise, 0-0.1)
- **detectorNoisePhotonGain**: 0.01 (shot noise visibility, 0.0001-0.1)
- **detectorNoiseFpnSigma**: 0.005 (fixed pattern noise, 0-0.05)

**Lens Distortion**:

- **lensDistortionEnabled**: false
- **lensDistortionX**: -0.04 (radial distortion X, barrel < 0, pincushion > 0)
- **lensDistortionY**: -0.04 (radial distortion Y)
- **lensPrincipalPointX**: 0 (optical center offset X)
- **lensPrincipalPointY**: 0 (optical center offset Y)
- **lensFocalLengthX**: 1.045 (focal length X)
- **lensFocalLengthY**: 1.045 (focal length Y)
- **lensSkew**: 0 (skew in radians)

**Lens Distortion Presets** (from `camera.lensDistortionPresets`):

- `'28mm Wide'`: Barrel distortion (-0.07, -0.07)
- `'35mm'`: Moderate barrel (-0.05, -0.05)
- `'50mm Normal'`: No distortion (0, 0)
- `'85mm Portrait'`: Pincushion (0.05, 0.05)
- `'135mm Tele'`: Barrel (0.07, 0.07)

### 7.6 Validation Rules

- **bloomThreshold**: 0-1
- **bloomStrength**: Typical 0-2 (can go higher)
- **bloomRadius**: Typical 0-2
- **bloomLevels**: 1-12
- **dofStrength**: 0-1
- **vignetteDarkness**: 0-1
- **vignetteOffset**: 0-1
- **detectorNoise** parameters: must be within specified ranges

---

## 8. UI Configuration

### 8.1 Purpose

Controls all UI styling, layout, z-index layers, timings, component configurations, and the debug console. Provides a unified design system for consistent styling.

### 8.2 Structure

```typescript
interface UIConfig {
  zIndex: {
    dimensionSliders: number;
    performanceMonitor: number;
    debugConsole: number;
    datasetBrowser: number;
    loading: number;
    error: number;
    help: number;
    renderingControls: number;
    statsMonitor: number;
  };
  timings: {
    errorAutoDismissMs: number;
    helpClickDelayMs: number;
  };
  spinner: {
    size: number;
    borderWidth: number;
  };
  styles: UIStyles;
  debugConsole: DebugConsoleConfig;
  components: UIComponentsConfig;
}
```

### 8.3 Defaults

**Z-Index Layers** (semantic layering):

- **Base layer** (100-199):
  - dimensionSliders: 100
  - performanceMonitor: 100
  - debugConsole: 150
- **Mid-layer overlays** (1000-1999):
  - datasetBrowser: 1000
  - loading: 1000
  - error: 1000
  - help: 1001
  - renderingControls: 1999
- **Top layer** (2000+):
  - statsMonitor: 2000

**Timings**:

- **errorAutoDismissMs**: 10000 (10s)
- **helpClickDelayMs**: 100 (prevent accidental close)

**Spinner**:

- **size**: 24 pixels
- **borderWidth**: 3 pixels

**Styles** (unified design system):

- **Colors**: Semantic colors (success, warning, error, info), text colors, backgrounds, cache visualization
- **Typography**: Font families, sizes (title, body, small, tiny), line heights
- **Spacing**: Panel, section, and element spacing with semantic names
- **Effects**: Backdrop blur, box shadows, border radius, transitions

**Debug Console**:

- **panel**: defaultWidth: 600, defaultHeight: 400, min/max bounds, offsets
- **interceptor.maxBufferSize**: 10000 messages
- **resize.borderWidth**: 4 pixels
- **style**: backgroundColor, borderColor, borderRadius, backdropBlur, boxShadow

**Components** (per-component configs):

- **datasetBrowser**: zIndex, borderRadius, padding
- **debugConsole**: zIndex, borderRadius
- **renderingControls**: borderRadius
- **dataMonitor**: borderRadius, padding

### 8.4 Usage Example

```typescript
import { config } from '../config';

// Apply z-index
const helpOverlay = document.createElement('div');
helpOverlay.style.zIndex = config.ui.zIndex.help.toString(); // 1001

// Use style system colors
const errorPanel = document.createElement('div');
errorPanel.style.backgroundColor = config.ui.styles.colors.panelBg;
errorPanel.style.color = config.ui.styles.colors.primaryText;
errorPanel.style.borderRadius = `${config.ui.styles.effects.borderRadius}px`;

// Use typography
const title = document.createElement('h1');
Object.assign(title.style, {
  fontFamily: config.ui.styles.typography.fontFamily,
  ...config.ui.styles.typography.title,
});

// Use spacing
const section = document.createElement('div');
section.style.padding = `${config.ui.styles.spacing.sectionPadding}px`;
section.style.gap = `${config.ui.styles.spacing.elementGap}px`;

// Configure debug console
const debugConsole = new DebugConsole();
debugConsole.setSize(
  config.ui.debugConsole.panel.defaultWidth,
  config.ui.debugConsole.panel.defaultHeight
);
```

### 8.5 Validation Rules

- **zIndex** values: Must be unique or intentionally shared
- **timings**: Must be > 0
- **spinner.size**: Typical 16-48 pixels
- **debugConsole.panel.minWidth**: Must be < defaultWidth < maxWidth
- **colors**: Should be valid CSS color strings

---

## 9. Input Configuration

### 9.1 Purpose

Defines keyboard shortcuts, mouse behavior, fly mode keys, and dimension navigation keys for user interaction.

### 9.2 Structure

```typescript
interface InputConfig {
  defaultSensitivity: number;
  keyboard: {
    shortcuts: {
      toggleFullscreen: string;
      toggleHelp: string;
      toggleDimensions: string;
      toggleDatasetBrowser: string;
      togglePerformance: string;
      toggleRendering: string;
      toggleDebugConsole: string;
      recenterCamera: string;
      toggleControlMode: string;
      toggleInertialMode: string;
      toggleCinematicMode: string;
    };
    flyModeKeys: string[];
    dimensionKeys: string[];
  };
  mouse: {
    doubleClickDelay: number;
  };
}
```

### 9.3 Defaults

**Sensitivity**:

- **defaultSensitivity**: 0.1

**Keyboard Shortcuts**:

- **toggleFullscreen**: ' ' (space)
- **toggleHelp**: 'h'
- **toggleDimensions**: 'n'
- **toggleDatasetBrowser**: 'o'
- **togglePerformance**: 'p'
- **toggleRendering**: 'r'
- **toggleDebugConsole**: 'ctrl+l'
- **recenterCamera**: 'f'
- **toggleControlMode**: 'v'
- **toggleInertialMode**: 'i'
- **toggleCinematicMode**: 'c'

**Fly Mode Keys**: `['w', 'a', 's', 'd', 'q', 'e', 'W', 'A', 'S', 'D', 'Q', 'E', 'Shift', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']`

**Dimension Keys**: `['[', ']', '1', '2', '3', '4', '5', '6', '7', '8', '9']`

**Mouse**:

- **doubleClickDelay**: 300ms

### 9.4 Usage Example

```typescript
import { config } from '../config';

// Register keyboard shortcuts
document.addEventListener('keydown', (e) => {
  const key = e.ctrlKey ? `ctrl+${e.key}` : e.key;

  switch (key) {
    case config.input.keyboard.shortcuts.toggleFullscreen:
      toggleFullscreen();
      break;
    case config.input.keyboard.shortcuts.toggleHelp:
      toggleHelp();
      break;
    case config.input.keyboard.shortcuts.recenterCamera:
      fitCameraToScene();
      break;
  }
});

// Check if key is a fly mode key
function isFlyModeKey(key: string): boolean {
  return config.input.keyboard.flyModeKeys.includes(key);
}

// Check if key is a dimension key
function isDimensionKey(key: string): boolean {
  return config.input.keyboard.dimensionKeys.includes(key);
}

// Handle double-click
let lastClickTime = 0;
canvas.addEventListener('click', (e) => {
  const now = Date.now();
  if (now - lastClickTime < config.input.mouse.doubleClickDelay) {
    onDoubleClick(e);
  }
  lastClickTime = now;
});
```

### 9.5 Validation Rules

- **defaultSensitivity**: Typical 0.01-0.5
- **keyboard.shortcuts**: Keys should be valid KeyboardEvent.key values
- **doubleClickDelay**: Typical 200-500ms

---

## 10. Data Loading Configuration

### 10.1 Purpose

Controls spatial query parameters, network behavior, memory management, and performance monitoring for efficient data streaming.

### 10.2 Structure

```typescript
interface DataLoadingConfig {
  spatial: {
    defaultTolerance: number;
    defaultMaxRadius: number;
  };
  network: {
    timeoutMs: number;
    maxConcurrent: number;
    retryAttempts: number;
  };
  memory: {
    targetHeapUsage: number;
    minCacheMB: number;
    checkIntervalMs: number;
    adjustmentThresholds: {
      critical: number;
      high: number;
    };
  };
  monitor: {
    timings: MonitorTimings;
    thresholds: MonitorThresholds;
    limits: MonitorLimits;
  };
}
```

### 10.3 Defaults

**Spatial**:

- **defaultTolerance**: 0.1 (nD slicing tolerance)
- **defaultMaxRadius**: 0.1 (spatial query max radius)

**Network**:

- **timeoutMs**: 30000 (30s)
- **maxConcurrent**: 6 (parallel chunk requests)
- **retryAttempts**: 3

**Memory**:

- **targetHeapUsage**: 0.8 (80% of available heap)
- **minCacheMB**: 128
- **checkIntervalMs**: 10000 (10s)
- **adjustmentThresholds.critical**: 0.85 (85%)
- **adjustmentThresholds.high**: 0.7 (70%)

**Monitor Timings**:

- **eventCleanupInterval**: 30000ms
- **maxEventAge**: 300000ms (5 min)
- **ratesCacheTimeout**: 1000ms
- **defaultUpdateInterval**: 100ms
- **minRenderInterval**: 100ms
- **timelinePointInterval**: 200ms
- **defaultTimeRange**: 60s
- **queryCleanupCheckInterval**: 10ms
- **maxQueryAge**: 60000ms (1 min)

**Monitor Thresholds**:

- **lowCacheHitRate**: 30%
- **highQueryTime**: 100ms
- **highLoadTime**: 500ms
- **highMemoryUsage**: 0.8 (80%)
- **highErrorRate**: 0.05 (5%)
- **lowQueryEfficiency**: 0.5 (50%)

**Monitor Limits**:

- **maxEvents**: 1000
- **maxTimelinePoints**: 300
- **maxAdvisorHistory**: 100
- **defaultMemoryLimit**: 1GB
- **rateCalculationWindow**: 5000ms
- **bandwidthCalculationWindow**: 1000ms

### 10.4 Usage Example

```typescript
import { config } from '../config';

// Configure data loader
const loader = new ZarrLoader(dataUrl, {
  network: {
    timeout: config.dataLoading.network.timeoutMs,
    maxConcurrent: config.dataLoading.network.maxConcurrent,
    retryAttempts: config.dataLoading.network.retryAttempts,
  },
  memory: {
    targetHeapUsage: config.dataLoading.memory.targetHeapUsage,
    minCacheMB: config.dataLoading.memory.minCacheMB,
  },
});

// Spatial query with tolerance
const tolerance = config.dataLoading.spatial.defaultTolerance;
const points = await loader.queryPoints({
  bounds: queryBounds,
  tolerance,
});

// Memory monitoring
setInterval(() => {
  const heapUsage = performance.memory.usedJSHeapSize / performance.memory.jsHeapSizeLimit;
  if (heapUsage > config.dataLoading.memory.adjustmentThresholds.critical) {
    loader.reduceCache();
  }
}, config.dataLoading.memory.checkIntervalMs);
```

### 10.5 Validation Rules

- **spatial.defaultTolerance**: Must be > 0
- **spatial.defaultMaxRadius**: Must be > 0
- **network.timeoutMs**: Must be > 0
- **network.maxConcurrent**: Must be > 0
- **network.retryAttempts**: Must be >= 0
- **memory.targetHeapUsage**: 0-1 (typical: 0.7-0.9)
- **memory.minCacheMB**: Must be > 0

---

## 11. WebGL Configuration

### 11.1 Purpose

Controls WebGL context creation, renderer settings, render target configuration, and performance profiles for optimal GPU utilization.

### 11.2 Structure

```typescript
interface WebGLConfig {
  context: WebGLContextAttributes;
  renderer: WebGLRendererConfig;
  renderTarget: WebGLRenderTargetConfig;
  profiles: {
    quality: WebGLPerformanceProfile;
    balanced: WebGLPerformanceProfile;
    performance: WebGLPerformanceProfile;
  };
}
```

### 11.3 Defaults

**Context** (WebGL2 context attributes):

- **alpha**: false (no transparency in canvas)
- **antialias**: true (smoother edges)
- **depth**: true (enable depth buffer)
- **stencil**: false (not needed, saves memory)
- **powerPreference**: 'high-performance' (request high-perf GPU)
- **colorSpace**: 'display-p3' (wide color gamut)
- **preserveDrawingBuffer**: false (better performance)
- **desynchronized**: true (async updates for better perf)
- **premultipliedAlpha**: true (standard alpha blending)
- **failIfMajorPerformanceCaveat**: false (don't fail on slow GPUs)

**Renderer** (THREE.WebGLRenderer settings):

- **antialias**: true
- **powerPreference**: 'high-performance'
- **preserveDrawingBuffer**: false
- **logarithmicDepthBuffer**: false (standard depth, faster)
- **precision**: 'highp' (high precision for quality)
- **premultipliedAlpha**: true
- **shadowMap.enabled**: false (no shadows for points)
- **shadowMap.type**: `THREE.PCFSoftShadowMap`

**RenderTarget**:

- **depthBuffer**: true (needed for depth testing)
- **stencilBuffer**: false (saves memory)
- **samples**: 0 (MSAA disabled for additive blending compatibility)

**Performance Profiles**:

- **quality**: high-performance, antialias: true, precision: 'highp'
- **balanced**: default, antialias: true, precision: 'mediump'
- **performance**: low-power, antialias: false, precision: 'lowp'

### 11.4 Usage Example

```typescript
import { config } from '../config';
import * as THREE from 'three';

// Create WebGL context with config
const canvas = document.getElementById('app') as HTMLCanvasElement;
const gl = canvas.getContext('webgl2', config.webgl.context);

// Create THREE.js renderer with config
const renderer = new THREE.WebGLRenderer({
  canvas,
  context: gl,
  ...config.webgl.renderer,
});

// Create render target for post-processing
const renderTarget = new THREE.WebGLRenderTarget(width, height, {
  ...config.webgl.renderTarget,
  type: config.postProcessing.hdr.renderTargetType,
});

// Switch to performance profile on low-end device
if (isMobileDevice()) {
  const profile = config.webgl.profiles.performance;
  renderer.setPixelRatio(1.0); // No devicePixelRatio scaling
  renderer.capabilities.precision = profile.precision;
}
```

### 11.5 Validation Rules

- **context.powerPreference**: Must be 'high-performance' | 'low-power' | 'default'
- **renderer.powerPreference**: Must match context.powerPreference
- **renderer.precision**: Must be 'highp' | 'mediump' | 'lowp'
- **renderTarget.samples**: 0, 2, 4, or 8
- **colorSpace**: Typical 'srgb', 'display-p3', 'rec2020'
- Consistency checks:
  - context.antialias should match renderer.antialias
  - context.powerPreference should match renderer.powerPreference
  - context.preserveDrawingBuffer should match renderer.preserveDrawingBuffer

---

## 12. Cache Configuration

### 12.1 Purpose

Controls OPFS-based zarr caching with two-tier architecture: L1 memory cache and L2 persistent OPFS cache for efficient data reuse.

### 12.2 Structure

```typescript
interface CacheConfig {
  enabled: boolean;
  l1MaxSizeMB: number;
  l2MaxSizeMB: number;
  debug: boolean;
}
```

### 12.3 Defaults

- **enabled**: true (OPFS caching enabled)
- **l1MaxSizeMB**: 100 (memory cache)
- **l2MaxSizeMB**: 2048 (OPFS cache)
- **debug**: false (disable cache debug logging)

### 12.4 Usage Example

```typescript
import { config } from '../config';

// Initialize cache system
const cache = new TwoTierCache({
  enabled: config.cache.enabled,
  l1MaxSizeMB: config.cache.l1MaxSizeMB,
  l2MaxSizeMB: config.cache.l2MaxSizeMB,
  debug: config.cache.debug,
});

// Check cache before fetching
const cacheKey = `${dataUrl}/${chunkPath}`;
let data = await cache.get(cacheKey);

if (!data) {
  // Cache miss - fetch from network
  data = await fetch(`${dataUrl}/${chunkPath}`).then((r) => r.arrayBuffer());
  await cache.set(cacheKey, data);
}

return data;
```

### 12.5 Validation Rules

- **l1MaxSizeMB**: Must be > 0 (typical: 50-500MB)
- **l2MaxSizeMB**: Must be > 0 (typical: 1024-8192MB)
- **l2MaxSizeMB** should be >> l1MaxSizeMB for effective two-tier caching

---

## 13. Rendering Controls Configuration

### 13.1 Purpose

Provides user-adjustable rendering settings with defaults for camera, bloom, anti-aliasing, tone mapping, post-processing effects, and navigation controls. Single source of truth for all visual settings.

### 13.2 Structure

```typescript
interface RenderingControlsConfig {
  defaults: RenderingSettings;
}

interface RenderingSettings {
  // Camera
  fov: number;
  fovPreset: '28mm Wide' | '35mm' | '50mm Normal' | '85mm Portrait' | '135mm Tele' | 'Custom';
  near: number;
  far: number;
  // Dynamic clipping planes
  dynamicClippingEnabled: boolean;
  clippingAdaptSpeed: number;
  // Bloom (single source of truth)
  bloomThreshold: number;
  bloomStrength: number;
  bloomRadius: number;
  bloomLevels: number;
  hdrMultiplier: number;
  // Anti-aliasing
  fxaaEnabled: boolean;
  msaaEnabled: boolean;
  msaaSamples: number;
  smaaEnabled: boolean;
  smaaThreshold: number;
  smaaSearchSteps: number;
  ssaaEnabled: boolean;
  ssaaMultiplier: number;
  // Tone mapping
  toneMapping: 'None' | 'Linear' | 'Reinhard' | 'Cineon' | 'ACES' | 'AgX' | 'Neutral';
  // Depth of field
  dofEnabled: boolean;
  dofFocus: number;
  dofStrength: number;
  // Chromatic aberration
  chromaticAberrationEnabled: boolean;
  chromaticAberrationStrength: number;
  // Ambient occlusion
  aoEnabled: boolean;
  aoQuality: 'low' | 'medium' | 'high' | 'ultra';
  // Vignette
  vignetteEnabled: boolean;
  vignetteDarkness: number;
  vignetteOffset: number;
  // Detector noise
  detectorNoiseEnabled: boolean;
  detectorNoiseReadoutSigma: number;
  detectorNoisePhotonGain: number;
  detectorNoiseFpnSigma: number;
  // Lens distortion
  lensDistortionEnabled: boolean;
  lensDistortionX: number;
  lensDistortionY: number;
  lensPrincipalPointX: number;
  lensPrincipalPointY: number;
  lensFocalLengthX: number;
  lensFocalLengthY: number;
  lensSkew: number;
  // Navigation
  controlType: 'orbit' | 'arcball' | 'fly';
  autoRotate: boolean;
  autoRotateSpeed: number;
  // Fly controls (optional, added at runtime)
  flyMovementSpeed?: number;
  flyRotationSpeed?: number;
  flyInertialMode?: boolean;
  flyDamping?: number;
  flyRotationDamping?: number;
}
```

### 13.3 Defaults

**Camera**:

- **fov**: 47 (50mm Normal)
- **fovPreset**: '50mm Normal'
- **near**: 0.1
- **far**: 1000

**Dynamic Clipping Planes**:

- **dynamicClippingEnabled**: true (auto-adjust clipping planes based on camera position)
- **clippingAdaptSpeed**: 0.1 (exponential smoothing factor, range 0.01-0.5)

**Bloom** (single source of truth):

- **bloomThreshold**: 0.01
- **bloomStrength**: 0.25
- **bloomRadius**: 1.0
- **bloomLevels**: 8
- **hdrMultiplier**: 16.0

**Anti-aliasing**:

- **fxaaEnabled**: false
- **msaaEnabled**: false (incompatible with additive blending)
- **msaaSamples**: 4
- **smaaEnabled**: false
- **smaaThreshold**: 0.1
- **smaaSearchSteps**: 8
- **ssaaEnabled**: false (brightness issues with additive)
- **ssaaMultiplier**: 2.0

**Tone Mapping**:

- **toneMapping**: 'ACES' (cinematic look)

**Depth of Field**:

- **dofEnabled**: false
- **dofFocus**: 10
- **dofStrength**: 0.5

**Chromatic Aberration**:

- **chromaticAberrationEnabled**: false
- **chromaticAberrationStrength**: 0.15

**Ambient Occlusion**:

- **aoEnabled**: false
- **aoQuality**: 'medium'

**Vignette**:

- **vignetteEnabled**: false
- **vignetteDarkness**: 0.5
- **vignetteOffset**: 0.5

**Detector Noise**:

- **detectorNoiseEnabled**: false
- **detectorNoiseReadoutSigma**: 0.01
- **detectorNoisePhotonGain**: 0.01
- **detectorNoiseFpnSigma**: 0.005

**Lens Distortion**:

- **lensDistortionEnabled**: false
- **lensDistortionX**: -0.04
- **lensDistortionY**: -0.04
- **lensPrincipalPointX**: 0
- **lensPrincipalPointY**: 0
- **lensFocalLengthX**: 1.045
- **lensFocalLengthY**: 1.045
- **lensSkew**: 0

**Navigation**:

- **controlType**: 'orbit'
- **autoRotate**: false
- **autoRotateSpeed**: 0.25

### 13.4 Usage Example

```typescript
import { config } from '../config';

// Initialize rendering settings from defaults
let settings = { ...config.renderingControls.defaults };

// Apply bloom settings
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(width, height),
  settings.bloomStrength,
  settings.bloomRadius,
  settings.bloomThreshold
);
bloomPass.nMips = settings.bloomLevels;

// Apply tone mapping
renderer.toneMapping = getToneMappingConstant(settings.toneMapping);

// Toggle depth of field
if (settings.dofEnabled) {
  const dofPass = new DepthOfFieldPass(camera, {
    focus: settings.dofFocus,
    strength: settings.dofStrength,
  });
  composer.addPass(dofPass);
}

// Update settings from UI
function onBloomStrengthChange(value: number) {
  settings.bloomStrength = value;
  bloomPass.strength = value;
  saveSettings(settings);
}
```

### 13.5 Validation Rules

- **clippingAdaptSpeed**: 0.01-0.5 (lower = smoother, higher = faster response)
- **bloomThreshold**: 0-1
- **bloomStrength**: Typical 0-2
- **bloomRadius**: Typical 0-2
- **bloomLevels**: 1-12
- **hdrMultiplier**: >= 0
- **dofStrength**: 0-1
- **vignetteDarkness**: 0-1
- **vignetteOffset**: 0-1
- **detectorNoiseReadoutSigma**: 0-0.1
- **detectorNoisePhotonGain**: 0.0001-0.1
- **detectorNoiseFpnSigma**: 0-0.05

---

## 14. Type Safety & Validation

### 14.1 Strict Typing

All configuration sections have corresponding TypeScript interfaces ensuring:

- Compile-time validation
- IDE autocomplete
- Refactoring safety
- Self-documenting structure
- Literal types with `as const` for precise type inference

### 14.2 Runtime Validation

The `config/validation.ts` module provides comprehensive runtime validation:

```typescript
import { validateAndLog } from './config/validation';

// Validate configuration at startup
const isValid = validateAndLog(config);
if (!isValid) {
  throw new Error('Invalid configuration');
}
```

**Validation Functions**:

- `validateConfig(config)`: Returns `ValidationResult` with errors and warnings
- `validateAndLog(config)`: Validates and logs results automatically
- `logValidationResults(result)`: Pretty-prints validation results

**What Gets Validated**:

- Camera: FOV bounds, near/far planes, sensitivity ranges
- Rendering: HDR multiplier, alpha ranges, bloom settings
- Controls: Damping ranges, speed limits, thresholds
- Data Loading: Network timeouts, memory thresholds, spatial parameters
- Scene: Background color, fit ratio
- Input: Sensitivity ranges
- WebGL: Power preference, precision, consistency checks

### 14.3 Configuration Best Practices

1. **Never mutate config directly** - create new objects with spread operator
2. **Use type imports** - `import type { AppConfig } from './config/types'`
3. **Validate at boundaries** - check user input before applying to config
4. **Document units** - include units in comments (ms, pixels, ratio, etc.)
5. **Provide ranges** - use `ConfigRange` interface for adjustable values
6. **Semantic naming** - use descriptive names that explain purpose
7. **Centralize defaults** - avoid duplicating defaults across modules
8. **Test validation** - ensure validation catches invalid configurations

---

## Changelog

- **v1.2.0** (2025-12-09): Dynamic clipping planes configuration
  - **ADDED**: `dynamicClippingEnabled` setting (default: true)
  - **ADDED**: `clippingAdaptSpeed` setting (default: 0.1, range: 0.01-0.5)
  - **UPDATED**: RenderingSettings interface with dynamic clipping fields
  - **UPDATED**: Validation rules for clippingAdaptSpeed
  - Enables smooth automatic clipping plane adjustment as camera moves

- **v1.1.0** (2025-12-09): Complete specification with all 12 configuration sections
  - **NEW**: Animation configuration (idle timeout, FPS targets)
  - **NEW**: Scene configuration (background color, fit ratio)
  - **NEW**: Shader configuration (HDR multiplier, base alpha)
  - **NEW**: PostProcessing configuration (HDR, tone mapping, all effects)
  - **NEW**: UI configuration (z-index, timings, styles, debug console, components)
  - **NEW**: Input configuration (keyboard shortcuts, fly keys, dimension keys, mouse)
  - **NEW**: Data Loading configuration (spatial, network, memory, monitoring)
  - **NEW**: WebGL configuration (context, renderer, render target, profiles)
  - **NEW**: Cache configuration (L1/L2 sizes, debug flags)
  - **NEW**: Rendering Controls configuration (bloom, DOF, AO, vignette, detector noise, lens distortion)
  - **UPDATED**: Expanded camera configuration with lens distortion presets
  - **UPDATED**: Expanded controls configuration with detailed fly and orbit settings
  - **UPDATED**: Added comprehensive validation rules for all sections
  - **IMPROVED**: Better structure with clear section numbering and table of contents

- **v1.0.0** (2025-01-30): Initial specification
  - Unified configuration system with type-safe interfaces
  - Hierarchical organization of settings
  - Camera, controls, rendering configuration (partial)
  - Min/max/default/step values for UI integration
  - Professional FOV presets (28mm-135mm equivalents)
