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
├── index.ts           # Main configuration object and exports
├── types.ts           # TypeScript interfaces for all config sections
└── debug-console.ts   # Debug console specific configuration
```

The architecture follows a separation of concerns approach:
- `index.ts` contains the actual configuration values and imports
- `types.ts` defines all TypeScript interfaces
- `debug-console.ts` isolates debug-specific configuration

## Configuration Sections

### Camera Configuration
Controls 3D perspective, navigation, and viewing parameters:
```typescript
camera: {
  fov: 60,                    // Field of view (human-like viewing)
  near: 0.1,                  // Near clipping plane
  far: 1000,                  // Far clipping plane
  initialPosition: { x: 0, y: 0, z: 8 },
  fovMin: 10,                 // Zoom limits
  fovMax: 200,
  fovSensitivity: 0.05        // Shift+wheel zoom sensitivity
}
```

### Rendering Configuration
HDR pipeline, bloom effects, and point rendering:
```typescript
rendering: {
  hdrEnabled: true,
  bloom: {
    strength: 0.25,           // Glow intensity
    radius: 1.0,              // Glow spread
    threshold: 0.0,           // Brightness trigger (0.0 = everything glows)
    resolutionScale: 4        // Quality vs performance
  },
  points: {
    size: 4.0,               // Base point size (pixels)
    hdrMultiplier: 16.0,     // HDR bloom multiplier
    baseAlpha: 0.01,         // Transparency
    falloffSteepness: 40.0   // Edge softness
  }
}
```

### Post-Processing Pipeline
Advanced rendering effects configuration:
```typescript
postProcessing: {
  hdr: {
    renderTargetType: THREE.HalfFloatType  // 16-bit float precision
  },
  toneMapping: {
    initial: {
      outputColorSpace: THREE.LinearSRGBColorSpace,
      toneMapping: THREE.NoToneMapping
    },
    final: {
      outputColorSpace: THREE.SRGBColorSpace,
      toneMapping: THREE.ACESFilmicToneMapping
    }
  }
}
```

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

### Rendering Controls
User-adjustable settings with persistence:
```typescript
renderingControls: {
  defaults: {
    bloomThreshold: 0.01,
    bloomStrength: 0.25,
    bloomRadius: 1.0,
    exposure: 1.0,
    hdrMultiplier: 16.0,
    fxaaEnabled: false,
    msaaEnabled: false,         // Incompatible with additive blending
    toneMapping: 'ACES',
    dofEnabled: false,          // Depth of field
    controlType: 'orbit',       // vs 'fly'
    autoRotate: false
  }
}
```

### Debug Console Configuration
Development and debugging features:
```typescript
// debug-console.ts
DEBUG_CONSOLE_CONFIG = {
  panel: {
    defaultWidth: 600,
    defaultHeight: 400,
    minWidth: 400,
    maxWidth: 1200
  },
  interceptor: {
    maxBufferSize: 10000      // Ring buffer for console messages
  },
  style: {
    backgroundColor: 'rgba(20, 20, 20, 0.95)',
    borderRadius: 8,
    backdropBlur: 10
  }
}
```

## Usage Examples

### Importing Configuration
```typescript
import { config } from '../config';
import type { AppConfig, RenderingSettings } from '../config';

// Access specific configuration sections
const cameraSettings = config.camera;
const bloomSettings = config.rendering.bloom;
```

### Using Configuration in Components
```typescript
// In SceneManager
const camera = new THREE.PerspectiveCamera(
  config.camera.fov,
  aspectRatio,
  config.camera.near,
  config.camera.far
);

// In PostProcessing
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(width, height),
  config.postProcessing.bloom.strength,
  config.postProcessing.bloom.radius,
  config.postProcessing.bloom.threshold
);
```

### Customizing Configuration
```typescript
// For development or testing, create modified config
const testConfig = {
  ...config,
  rendering: {
    ...config.rendering,
    hdrEnabled: false  // Disable HDR for testing
  }
};
```

## Type Safety

The config package provides complete TypeScript interfaces for all configuration sections:

```typescript
interface AppConfig {
  camera: CameraConfig;
  animation: AnimationConfig;
  scene: SceneConfig;
  rendering: RenderingConfig;
  shader: ShaderConfig;
  postProcessing: PostProcessingConfig;
  ui: UIConfig;
  renderingControls: RenderingControlsConfig;
  defaultZarrPath: string;
  canvasId: string;
}

interface RenderingSettings {
  bloomThreshold: number;
  bloomStrength: number;
  // ... all user-adjustable settings
  toneMapping: 'None' | 'Linear' | 'Reinhard' | 'Cineon' | 'ACES' | 'AgX' | 'Neutral';
  controlType: 'orbit' | 'fly';
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
// config.camera.fov = 90; // This would break immutability
```

### Adding New Configuration
1. **Add to Types**: Define the interface in `types.ts`
2. **Add to Config**: Include the actual values in `index.ts`
3. **Document**: Provide clear comments explaining the purpose
4. **Test**: Ensure the configuration works in relevant components

### Debug Configuration
```typescript
// Debug console configuration is isolated for modularity
import { DEBUG_CONSOLE_CONFIG } from './debug-console';

// Access debug settings
const panelWidth = DEBUG_CONSOLE_CONFIG.panel.defaultWidth;
const bufferSize = DEBUG_CONSOLE_CONFIG.interceptor.maxBufferSize;
```

### Migration and Versioning
When updating configuration:
1. **Backwards Compatibility**: Consider existing users
2. **Deprecation Strategy**: Gradual migration for breaking changes
3. **Validation**: Ensure new values are within acceptable ranges
4. **Documentation**: Update comments and examples

The config package is the foundation that ensures consistent behavior across all Luxar components while providing flexibility for customization and future enhancements.