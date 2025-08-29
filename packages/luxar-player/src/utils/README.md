# Utils Package

Utility functions and helpers for advanced features in the Luxar player. This package contains specialized algorithms for console debugging, HDR detection, memory management, and structured logging.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Modules](#modules)
- [Console Interception](#console-interception)
- [HDR Detection](#hdr-detection)
- [Memory Detection](#memory-detection)
- [Structured Logging](#structured-logging)
- [Usage Examples](#usage-examples)
- [Performance Considerations](#performance-considerations)
- [Best Practices](#best-practices)

## Overview

The utils package provides specialized functionality that extends beyond basic 3D visualization to support advanced features like debugging capabilities, HDR display optimization, and memory management. These utilities are the foundation for Luxar's advanced visualization capabilities.

**Core Philosophy**: Provide robust, well-tested utility functions that handle complex system interactions with clear interfaces and comprehensive error handling.

## Key Features

- **Console Interception**: Ring buffer system for capturing all browser console output
- **HDR Detection**: Comprehensive display capability detection and configuration
- **Memory Detection**: Intelligent memory availability detection for cache sizing
- **Structured Logging**: Consistent logging format with module identification

## Architecture

```typescript
utils/
├── console-interceptor.ts # Console output capture and buffering system
├── hdr-detection.ts      # HDR display capability detection
├── memory-detector.ts    # System memory detection for cache management
├── log.ts               # Structured logging utility
└── slicing.ts           # nD slicing algorithms (moved from dims-navigation)
```

Each module is focused on a specific domain with minimal dependencies, promoting reusability and maintainability.

## Modules

### console-interceptor.ts - Console Debugging

Ring buffer system for capturing and managing console output:

**Core Features**:

- **Singleton Pattern**: Global console interception
- **Ring Buffer**: Efficient 10,000 message circular buffer
- **Early Capture**: Starts before application initialization
- **Real-time Listeners**: Callback system for live console updates

### hdr-detection.ts - HDR Display Detection

Comprehensive system for detecting and configuring HDR display capabilities:

**Core Functions**:

- `detectHDRCapabilities()` - Complete capability detection
- `configureHDRRenderer()` - Optimal Three.js renderer setup
- `logHDRCapabilities()` - Detailed capability reporting
- `isHDRDisplay()` - Simple boolean HDR check

## Console Interception

### Ring Buffer Implementation

```typescript
class ConsoleInterceptor {
  private messageBuffer: BufferedMessage[] = [];
  private bufferIndex = 0;
  private readonly maxBufferSize = 10000;
  private hasWrapped = false;
}
```

**Key Features**:

- **Memory Efficient**: Fixed-size circular buffer prevents memory leaks
- **Early Capture**: Starts before any other code executes
- **Original Preservation**: Maintains original console.\* functionality
- **Stack Traces**: Automatic stack trace extraction for errors

### Message Structure

```typescript
interface BufferedMessage {
  type: 'log' | 'warn' | 'error' | 'info' | 'debug';
  timestamp: Date;
  args: any[];
  stack?: string; // For errors
}
```

### Usage Patterns

```typescript
// Import at the very top of main.ts to ensure early capture
import { consoleInterceptor } from '../utils/console-interceptor';

// Access buffered messages
const messages = consoleInterceptor.getBufferedMessages();

// Add real-time listener
consoleInterceptor.addListener((message) => {
  // Handle new console output in real-time
});
```

## HDR Detection

### Comprehensive Capability Detection

```typescript
interface HDRCapabilities {
  p3Gamut: boolean; // Display P3 wide color gamut
  rec2020Gamut: boolean; // Rec2020 gamut support
  hdr: boolean; // High dynamic range
  deepColor: boolean; // 10-bit+ color depth
  floatTextures: boolean; // WebGL float texture support
  colorDepth: {
    // Actual color buffer depth
    red: number;
    green: number;
    blue: number;
  };
  recommendedColorSpace: 'srgb' | 'display-p3' | 'rec2020';
}
```

### Detection Methods

```typescript
// CSS media query detection
const hdr = window.matchMedia('(dynamic-range: high)').matches;
const p3Gamut = window.matchMedia('(color-gamut: p3)').matches;

// WebGL extension detection
const floatTextures = !!(
  gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float')
);

// Hardware color depth
const colorDepth = {
  red: gl.getParameter(gl.RED_BITS),
  green: gl.getParameter(gl.GREEN_BITS),
  blue: gl.getParameter(gl.BLUE_BITS),
};
```

### Renderer Configuration

```typescript
export function configureHDRRenderer(
  renderer: THREE.WebGLRenderer,
  capabilities: HDRCapabilities
): void {
  // Set optimal color space based on display capabilities
  if (capabilities.rec2020Gamut && capabilities.hdr) {
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  } else if (capabilities.p3Gamut) {
    renderer.outputColorSpace = THREE.SRGBColorSpace;
  }

  // Note: Tone mapping is now handled by PostProcessingManager
  // to avoid conflicts with the pmndrs library
}
```

## Dimension Navigation

### Intelligent Stepping

```typescript
export function stepDimension(
  dims: SimpleDims,
  dimIndex: number,
  direction: 1 | -1,
  ranges: Array<[number, number]>,
  options: NavigationOptions = {}
): boolean;
```

**Features**:

- **Adaptive Step Size**: 10% of dimension range by default
- **Boundary Handling**: Wrapping or clamping at edges
- **Absolute Steps**: Override with fixed step sizes
- **Safety Checks**: Only non-displayed dimensions can be stepped

### Navigation Options

```typescript
interface NavigationOptions {
  stepSize?: number; // Fraction of range (0.1 = 10%)
  wrap?: boolean; // Wrap at boundaries (periodic data)
  absoluteStep?: number; // Fixed step size in data units
}
```

### Real-time Updates

The navigation system provides smooth, real-time updates:

```typescript
// Keyboard navigation typically uses first two non-displayed dimensions
const [primaryDim, secondaryDim] = getNavigableDimensions(dims);

// Step through dimension and update visualization if changed
if (stepDimension(dims, primaryDim, direction, ranges)) {
  updatePointCloudSlice(points, positions, colors, radii, sharpness, dims, numPoints);
}
```

## Usage Examples

### Console Debugging Setup

```typescript
// At the very top of main.ts
import { consoleInterceptor } from '../utils/console-interceptor';

// Later, in debug console component
const messages = consoleInterceptor.getBufferedMessages();
consoleInterceptor.addListener((message) => {
  displayMessage(message);
});
```

### HDR Display Optimization

```typescript
import {
  detectHDRCapabilities,
  configureHDRRenderer,
  logHDRCapabilities,
} from '../utils/hdr-detection';

// Detect capabilities and configure renderer
const hdrCapabilities = detectHDRCapabilities(renderer);
logHDRCapabilities(hdrCapabilities);
configureHDRRenderer(renderer, hdrCapabilities);

// Check for true HDR support
if (isHDRDisplay(hdrCapabilities)) {
  // Enable advanced HDR features
  enableAdvancedHDRFeatures();
}
```

## Performance Considerations

### Slicing Optimization

- **Early Termination**: Break loops as soon as point is excluded
- **Memory Reuse**: Reuse buffers when possible to minimize allocations

### Navigation Efficiency

- **Minimal Updates**: Only update GPU when dimensions actually change
- **Batch Operations**: Group multiple attribute updates together
- **Bounds Caching**: Cache bounding sphere calculations when possible

### Console Buffer Management

- **Fixed Size**: Ring buffer prevents memory growth
- **Efficient Rotation**: Modulo arithmetic for circular indexing
- **Listener Optimization**: Use Set for O(1) listener management

## Best Practices

### Algorithm Design

```typescript
// ✅ Good: Clear mathematical documentation
// Mathematical foundation: R_effective = √(R² - D²)
const effectiveRadius = Math.sqrt(radiusSquared - distanceSquared);

// ✅ Good: Handle edge cases gracefully
const safeRadius = effectiveRadius > 0 ? effectiveRadius : 0;
```

### Error Handling

```typescript
// ✅ Good: Validate inputs early
if (dimIndex < 0 || dimIndex >= dims.ndim) {
  return false; // Early return for invalid inputs
}

// ✅ Good: Graceful degradation
if (!colors) {
  // Provide sensible fallback instead of crashing
  return createDefaultColors(numPoints);
}
```

### Memory Management

```typescript
// ✅ Good: Reuse typed arrays when possible
const reusableBuffer = new Float32Array(maxPoints * 3);

// ✅ Good: Use appropriate data types
const indices = new Uint32Array(visibleCount); // vs Array<number>
```

### Performance Monitoring

```typescript
// ✅ Good: Log performance-critical operations
console.log(`Sliced ${numPoints} points to ${visibleCount} in ${elapsed}ms`);

// ✅ Good: Warn about performance issues
if (visibleCount === 0) {
  console.warn('No points visible at current slice position!');
}
```

The utils package provides the mathematical and system-level foundation that enables Luxar's advanced visualization capabilities while maintaining performance and reliability.
