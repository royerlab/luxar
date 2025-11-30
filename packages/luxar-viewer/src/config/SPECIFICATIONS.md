# luxar-viewer.config - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2025-01-30

## Purpose

The `luxar-viewer.config` package provides a unified, type-safe configuration system for all application settings. It serves as the single source of truth for default values, limits, and behavior parameters across all components.

**Core Responsibility**: Centralize all configuration values with proper TypeScript typing, clear documentation, and sensible defaults to ensure consistency and ease of customization.

---

## Table of Contents

1. [Configuration Architecture](#configuration-architecture)
2. [Key Configuration Sections](#key-configuration-sections)
3. [Type Safety](#type-safety)

---

## 1. Configuration Architecture

### 1.1 Structure

```typescript
// config/index.ts - Exports the main config object
export const config: AppConfig = {
    camera: { ... },
    scene: { ... },
    shader: { ... },
    postProcessing: { ... },
    ui: { ... },
    controls: { ... },
    input: { ... },
    dataLoading: { ... },
    webgl: { ... },
    renderingControls: { ... }
}

// config/types.ts - Type definitions
export interface AppConfig {
    camera: CameraConfig
    scene: SceneConfig
    // ... all sections
}
```

**Design Principle**: Configuration is **read-only** at runtime. Modifications create new config objects rather than mutating.

---

## 2. Key Configuration Sections

### 2.1 Camera Configuration

```typescript
interface CameraConfig {
  fov: number; // Default FOV (47° = 50mm equivalent)
  near: number; // Near clipping plane (0.1)
  far: number; // Far clipping plane (1000)
  initialPosition: { x: number; y: number; z: number };
  fovMin: number; // Min FOV (10°)
  fovMax: number; // Max FOV (200°)
  fovSensitivity: number; // Shift+wheel sensitivity (0.05)
}
```

### 2.2 Controls Configuration

```typescript
interface ControlsConfig {
  fly: {
    inertialMode: { default: boolean };
    movement: {
      speed: { min: number; max: number; default: number; step: number };
      damping: { min: number; max: number; default: number; step: number };
      acceleration: { min: number; max: number; default: number; step: number };
    };
    rotation: {
      speed: { min: number; max: number; default: number; step: number };
      damping: { min: number; max: number; default: number; step: number };
    };
    physics: {
      velocityThreshold: number;
      angularVelocityThreshold: number;
    };
  };
  orbit: {
    autoRotate: {
      speed: { min: number; max: number; default: number };
    };
    zoom: {
      minDistance: number;
      maxDistance: number;
    };
  };
}
```

**Usage Pattern**:

```typescript
import { config } from '../config';

const speed = config.controls.fly.movement.speed.default; // 5.0
const minSpeed = config.controls.fly.movement.speed.min; // 0.5
const maxSpeed = config.controls.fly.movement.speed.max; // 50.0
```

### 2.3 Rendering Defaults

```typescript
interface RenderingControlsConfig {
  defaults: {
    bloomThreshold: number;
    bloomStrength: number;
    bloomRadius: number;
    bloomLevels: number;
    hdrMultiplier: number;
    fxaaEnabled: boolean;
    smaaEnabled: boolean;
    msaaEnabled: boolean; // false (incompatible with additive)
    toneMapping: 'None' | 'ACES' | 'AgX' | 'Reinhard' | 'Linear' | 'Neutral';
    controlType: 'orbit' | 'arcball' | 'fly';
    autoRotate: boolean;
    // ... other rendering settings
  };
}
```

---

## 3. Type Safety

### 3.1 Strict Typing

All configuration sections have corresponding TypeScript interfaces ensuring:

- Compile-time validation
- IDE autocomplete
- Refactoring safety
- Self-documenting structure

### 3.2 Configuration Validation

**Runtime Validation** (optional):

```typescript
function validateConfig(config: AppConfig): boolean {
  // Check camera FOV bounds
  if (config.camera.fov < config.camera.fovMin || config.camera.fov > config.camera.fovMax) {
    throw new Error('Invalid camera FOV');
  }

  // Check control damping ranges
  const flyDamping = config.controls.fly.movement.damping;
  if (flyDamping.default < flyDamping.min || flyDamping.default > flyDamping.max) {
    throw new Error('Invalid fly damping default');
  }

  // ... other validations

  return true;
}
```

---

## Changelog

- **v1.0.0** (2025-01-30): Initial specification
  - Unified configuration system with type-safe interfaces
  - Hierarchical organization of settings
  - Camera, controls, rendering, UI, and WebGL configuration
  - Min/max/default/step values for UI integration
  - Debug console configuration
  - Professional FOV presets (28mm-135mm equivalents)
