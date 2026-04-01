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

```typescript
config/
├── index.ts                 # Main configuration object and exports
├── types.ts                 # TypeScript interfaces for all config sections
├── validation.ts            # Runtime validation of configuration values
├── viewer-config-utils.ts   # Viewer config utility functions
└── viewer-state-capture.ts  # Viewer state capture for export
```

The architecture follows a separation of concerns approach:

- `index.ts` contains the actual configuration values and imports
- `types.ts` defines all TypeScript interfaces
- `validation.ts` provides runtime validation of configuration values

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

Point rendering shader constants:

```typescript
shader: {
  points: {
    baseAlpha: 0.01,         // Base alpha intensity
  }
}
```

**Note**: Per-node color adjustment (intensity, offset, gamma) is configured per-material. Global exposure/offset/gamma are in `renderingControls.defaults` and applied in the `LuxarToneMappingEffect` post-processing pass. Post-processing pipeline internals (HalfFloatType, tone mapping modes) are managed directly by the PostProcessingManager.

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
  targetFPS: 55,               // Target FPS (slightly below 60 to prevent toggling)
  minFPS: 50,                  // FPS threshold for scaling down resolution
  maxFPS: 58,                  // FPS threshold for scaling up resolution
  minDPR: 0.5,                 // Minimum allowed DPR (quality floor)
  scaleDownFactor: 0.9,        // Factor when scaling down (10% reduction)
  scaleUpFactor: 1.05,         // Factor when scaling up (5% increase)
  hysteresisSeconds: 3,        // Seconds FPS must stay above maxFPS before scaling up
  evaluationIntervalMs: 500    // How often to evaluate FPS (ms)
}
```

**How It Works**:

- When FPS drops below `minFPS`, DPR is reduced by `scaleDownFactor`
- When FPS stays above `maxFPS` for `hysteresisSeconds`, DPR increases by `scaleUpFactor`
- Asymmetric scaling (slower up, faster down) prevents quality oscillation
- Hysteresis prevents rapid toggling between quality levels
- `minDPR` prevents image from becoming too pixelated

### Rendering Controls

User-adjustable settings with persistence:

```typescript
renderingControls: {
  defaults: {
    // Camera clipping
    dynamicClippingEnabled: true,   // Auto-adjust clipping planes per frame
    // Bloom and HDR
    bloomThreshold: 0.01,
    bloomStrength: 0.5,
    bloomRadius: 0.6,
    bloomLevels: 8,
    // Global EOG (Exposure-Offset-Gamma) in LuxarToneMappingEffect
    exposure: 1.0,
    globalOffset: 0.0,
    globalGamma: 1.0,
    // Anti-aliasing
    fxaaEnabled: false,
    msaaEnabled: false,         // Incompatible with additive blending
    msaaSamples: 4,
    smaaEnabled: false,
    ssaaEnabled: false,
    toneMapping: 'Neutral',
    dofEnabled: false,          // Depth of field
    // Detector noise (physics-based: Poisson + Gaussian + FPN)
    detectorNoiseEnabled: false,
    detectorNoiseReadoutSigma: 0.01,  // Temporal readout noise
    detectorNoisePhotonGain: 0.01,    // Shot noise visibility
    detectorNoiseFpnSigma: 0.005,     // Fixed pattern noise
    vignetteEnabled: false,
    aoEnabled: false,           // Ambient occlusion
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
    antialias: true,                   // Enable edge smoothing
    depth: true,                       // Enable depth buffer
    stencil: false,                    // No stencil (saves memory)
    powerPreference: 'high-performance', // GPU preference
    colorSpace: 'display-p3',          // Wide color gamut
    preserveDrawingBuffer: false,      // Better performance
    desynchronized: true,              // Async updates
  },
  renderer: {
    antialias: true,                   // MSAA antialiasing
    precision: 'highp',                // Shader precision
    logarithmicDepthBuffer: false,    // Standard depth (faster)
  },
  renderTarget: {
    depthBuffer: true,                 // Depth testing
    stencilBuffer: false,              // No stencil
    samples: 0,                        // MSAA samples
  },
  profiles: {
    quality: { /* high-performance settings */ },
    balanced: { /* default settings */ },
    performance: { /* low-power settings */ }
  }
}
```

### Debug Console Configuration

Development and debugging features:

```typescript
// Debug console configuration is now in config.ui.debugConsole
debugConsole: {
  panel: {
    defaultWidth: 600,
    defaultHeight: 400,
    minWidth: 400,
    maxWidth: 1200,
  },
  interceptor: {
    maxBufferSize: 10000, // Ring buffer for console messages
  },
  // ... other settings
}
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
const bloomEffect = new BloomEffect({
  intensity: config.renderingControls.defaults.bloomStrength,
  luminanceThreshold: config.renderingControls.defaults.bloomThreshold,
  levels: config.renderingControls.defaults.bloomLevels,
  // radius is set on mipmapBlurPass after creation
});
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
  scene: SceneConfig;
  shader: ShaderConfig;
  ui: UIConfig;
  renderingControls: RenderingControlsConfig;
  controls: ControlsConfig;
  input: InputConfig;
  dataLoading: DataLoadingConfig;
  webgl: WebGLConfig;
  defaultZarrPath: string;
  canvasId: string;
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
- Refactoring safety when changing configuration structure
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
const { fov, near, far } = config.camera;

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

### Migration and Versioning

When updating configuration:

1. **Backwards Compatibility**: Consider existing users
2. **Deprecation Strategy**: Gradual migration for breaking changes
3. **Validation**: Ensure new values are within acceptable ranges
4. **Documentation**: Update comments and examples

The config package is the foundation that ensures consistent behavior across all Luxar components while providing flexibility for customization and future enhancements.
