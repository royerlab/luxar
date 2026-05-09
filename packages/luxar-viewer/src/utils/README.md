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
- **HDR Video Encoding**: 10-bit HDR video encoding via WebCodecs (AV1/VP9)
- **HDR Color Conversion**: Linear sRGB to BT.2020 PQ I420P10 pipeline
- **Memory Detection**: Intelligent memory availability detection for cache sizing
- **Structured Logging**: Consistent logging format with module identification
- **HTML Escaping**: XSS prevention for safe HTML rendering

## Architecture

```typescript
utils/
├── console-interceptor.ts  # Console output capture and buffering system
├── escape-html.ts          # HTML entity escaping for safe rendering
├── hdr-color-conversion.ts # Linear sRGB to BT.2020 PQ I420P10 conversion
├── hdr-detection.ts        # HDR display capability detection
└── log.ts                  # Structured logging utility
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

### log.ts - Structured Logging

Consistent logging with emoji prefixes and module identification:

**Core Exports**:

- `log` - Quick logging object (`log.info()`, `log.error()`, `log.warning()`, `log.success()`, `log.load()`, `log.update()`, `log.query()`, `log.data()`, `log.custom()`, `log.raw()`)
- `Modules` - Standard module name constants (LUXAR, APP, WORKER_POOL, WASM, etc.)
- `LogEmoji` - Standard emoji constants for log categories
- `formatLog()` - Format a log message with consistent style
- `createModuleLogger()` - Create a module-scoped logger instance

### hdr-detection.ts - HDR Display Detection

Comprehensive system for detecting and configuring HDR display capabilities:

**Core Functions**:

- `detectHDRCapabilities()` - Complete capability detection
- `configureHDRRenderer()` - Log detected HDR capabilities (legacy name, doesn't actually configure renderer)
- `logHDRCapabilities()` - Detailed capability reporting
- `isHDRDisplay()` - Simple boolean HDR check
- `getOptimalRenderTargetType()` - Get optimal THREE.TextureDataType for display

### hdr-color-conversion.ts - HDR Color Conversion

Converts linear sRGB float data to BT.2020 PQ I420P10 for WebCodecs:

**Core Function**:

- `rgbaFloatToI420P10()` - Linear sRGB RGBA float to BT.2020 PQ YCbCr I420P10

### escape-html.ts - HTML Escaping

Simple XSS prevention utility:

- `escapeHtml()` - Escape HTML special characters (&, <, >, ")

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

### HDR Capability Logging

```typescript
export function configureHDRRenderer(
  renderer: THREE.WebGLRenderer, // Unused - for backward compatibility
  capabilities: HDRCapabilities
): void {
  // Log detected capabilities (informational only)
  // Does NOT configure renderer - PostProcessingManager handles that
}
```

**Note**: Despite the legacy name, this function only logs HDR capabilities. It does NOT configure the renderer. All actual HDR configuration (outputColorSpace, toneMapping) is handled by `PostProcessingManager` to avoid conflicts with the pmndrs/postprocessing library.

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
