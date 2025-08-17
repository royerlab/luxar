# Utils Package

Utility functions and helpers for advanced features in the Luxar player. This package contains specialized algorithms for nD data slicing, console debugging, HDR detection, and dimensional navigation.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Modules](#modules)
- [nD Slicing Algorithms](#nd-slicing-algorithms)
- [Console Interception](#console-interception)
- [HDR Detection](#hdr-detection)
- [Dimension Navigation](#dimension-navigation)
- [Usage Examples](#usage-examples)
- [Performance Considerations](#performance-considerations)
- [Best Practices](#best-practices)

## Overview

The utils package provides specialized functionality that extends beyond basic 3D visualization to support advanced features like high-dimensional data navigation, debugging capabilities, and HDR display optimization. These utilities are the mathematical and algorithmic foundation for Luxar's advanced visualization capabilities.

**Core Philosophy**: Provide robust, well-tested utility functions that handle complex mathematical operations and system interactions with clear interfaces and comprehensive error handling.

## Key Features

- **nD Slicing**: Advanced algorithms for navigating high-dimensional point clouds
- **Console Interception**: Ring buffer system for capturing all browser console output
- **HDR Detection**: Comprehensive display capability detection and configuration
- **Dimension Navigation**: Smooth navigation through multi-dimensional datasets
- **Hypersphere Mathematics**: Geometrically accurate radius calculations for nD spheres

## Architecture

```typescript
utils/
├── slicing.ts            # nD slicing algorithms and hypersphere mathematics
├── dims-navigation.ts    # Navigation utilities for multi-dimensional data
├── console-interceptor.ts # Console output capture and buffering system
└── hdr-detection.ts      # HDR display capability detection
```

Each module is focused on a specific domain with minimal dependencies, promoting reusability and maintainability.

## Modules

### slicing.ts - nD Slicing Algorithms
Advanced mathematical algorithms for slicing high-dimensional point clouds:

**Core Functions**:
- `slicePoints()` - Radius-based hypersphere intersection slicing
- `extractDisplayDimensions()` - Project nD points to 3D visualization space
- `computeEffectiveRadii()` - Calculate cross-sectional radii of sliced nD spheres
- `sliceColorsFloat32()` - HDR color slicing with float32 precision
- `sliceScalarAttribute()` - Generic attribute slicing for point properties

### dims-navigation.ts - Dimension Navigation
User interaction utilities for navigating high-dimensional space:

**Core Functions**:
- `stepDimension()` - Intelligent stepping with boundary handling
- `jumpToDimension()` - Direct positioning via UI sliders
- `updatePointCloudSlice()` - Real-time GPU geometry updates
- `getNavigableDimensions()` - Identify keyboard-controllable dimensions

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

## nD Slicing Algorithms

### Hypersphere Intersection Mathematics

The core slicing algorithm implements radius-based hypersphere intersection:

```typescript
// Mathematical foundation: For a point with nD position P and radius R,
// the point is visible if its nD hypersphere intersects the slice hyperplane
// Distance = √(Σ(Pi - Ci)²) where C is current position in non-displayed dims
// Point is visible if distance ≤ radius

export function slicePoints(
  positions: Float32Array,
  dims: SimpleDims,
  numPoints: number,
  radii?: Float32Array,
  fallbackTolerance = 0.1
): Uint32Array
```

**Algorithm Details**:
1. **Discrete Dimensions**: Exact matching for categorical data (time frames, channels)
2. **Continuous Dimensions**: Radius-based inclusion for smooth navigation
3. **Early Termination**: Efficient point rejection for performance
4. **Fallback Tolerance**: Default radius when per-point radii unavailable

### Effective Radius Calculation

When an nD hypersphere is sliced by hyperplanes, the cross-section radius follows:

```typescript
// R_effective = √(R² - D²) where D is distance to hyperplane
// This preserves accurate visual representation of point sizes after slicing
```

### Point Cloud Update Pipeline

```typescript
// Complete pipeline from nD slicing to GPU rendering:
export function updatePointCloudSlice(
  points: THREE.Points,
  originalPositions: Float32Array,
  originalColors: Float32Array | undefined,
  originalRadii: Float32Array | undefined,
  originalSharpness: Float32Array | undefined,
  dims: SimpleDims,
  numPoints: number
): void
```

**Pipeline Stages**:
1. **nD Slicing**: Radius-based hypersphere intersection
2. **3D Projection**: Extract display dimensions
3. **Effective Radii**: Compute cross-sectional radii
4. **GPU Updates**: Update vertex attributes and buffers
5. **Bounds Calculation**: Recalculate bounding volumes

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
- **Original Preservation**: Maintains original console.* functionality
- **Stack Traces**: Automatic stack trace extraction for errors

### Message Structure

```typescript
interface BufferedMessage {
  type: 'log' | 'warn' | 'error' | 'info' | 'debug';
  timestamp: Date;
  args: any[];
  stack?: string;  // For errors
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
  p3Gamut: boolean;          // Display P3 wide color gamut
  rec2020Gamut: boolean;     // Rec2020 gamut support
  hdr: boolean;              // High dynamic range
  deepColor: boolean;        // 10-bit+ color depth
  floatTextures: boolean;    // WebGL float texture support
  colorDepth: {              // Actual color buffer depth
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
  gl.getExtension('EXT_color_buffer_float') ||
  gl.getExtension('EXT_color_buffer_half_float')
);

// Hardware color depth
const colorDepth = {
  red: gl.getParameter(gl.RED_BITS),
  green: gl.getParameter(gl.GREEN_BITS),
  blue: gl.getParameter(gl.BLUE_BITS)
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
  
  // Configure tone mapping for HDR
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = capabilities.hdr ? 1.4 : 1.0;
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
): boolean
```

**Features**:
- **Adaptive Step Size**: 10% of dimension range by default
- **Boundary Handling**: Wrapping or clamping at edges
- **Absolute Steps**: Override with fixed step sizes
- **Safety Checks**: Only non-displayed dimensions can be stepped

### Navigation Options

```typescript
interface NavigationOptions {
  stepSize?: number;        // Fraction of range (0.1 = 10%)
  wrap?: boolean;          // Wrap at boundaries (periodic data)
  absoluteStep?: number;   // Fixed step size in data units
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

### nD Point Cloud Slicing

```typescript
import { slicePoints, extractDisplayDimensions, updatePointCloudSlice } from '../utils/slicing';
import { stepDimension } from '../utils/dims-navigation';

// Slice nD point cloud at current position
const visibleIndices = slicePoints(positions, dims, numPoints, radii);
const positions3D = extractDisplayDimensions(positions, visibleIndices, dims);

// Navigate to next slice
if (stepDimension(dims, dimIndex, 1, ranges)) {
  // Update GPU geometry with new slice
  updatePointCloudSlice(points, positions, colors, radii, sharpness, dims, numPoints);
}
```

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
  logHDRCapabilities 
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
- **Set Lookup**: O(1) displayed dimension checks using Set
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