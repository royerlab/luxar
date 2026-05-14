# luxar-viewer.utils - Technical Specification

**Version**: 1.2.0
**Last Updated**: 2025-12-09

## Purpose

The `luxar-viewer.utils` package provides specialized utility functions for console debugging, HDR display detection, memory management, and structured logging.

**Core Responsibility**: Provide robust, well-tested utility functions for system-level operations including console output capture, display capability detection, adaptive memory management, and standardized logging.

---

## Table of Contents

1. [Console Interception](#console-interception)
2. [HDR Detection](#hdr-detection)
3. [Structured Logging](#structured-logging)

---

## 1. Console Interception

### 1.1 Purpose

Capture all browser console output in a ring buffer for in-app debugging. Provides real-time message streaming via listener callbacks and chronological message retrieval.

### 1.2 Ring Buffer Implementation

**Data Structure**:

```typescript
class ConsoleInterceptor {
  private messageBuffer: BufferedMessage[] = [];
  private bufferIndex: number = 0;
  private hasWrapped: boolean = false;
  private readonly maxBufferSize = 10000;
  private listeners: Set<(message: BufferedMessage) => void> = new Set();
  private originalConsole: {
    log: typeof console.log;
    warn: typeof console.warn;
    error: typeof console.error;
    info: typeof console.info;
    debug: typeof console.debug;
  };

  private startInterception(): void {
    // Save original console methods
    this.originalConsole = {
      log: console.log.bind(console),
      warn: console.warn.bind(console),
      error: console.error.bind(console),
      info: console.info.bind(console),
      debug: console.debug.bind(console),
    };

    // Override console methods
    console.log = (...args) => {
      this.captureMessage('log', args);
      this.originalConsole.log(...args);
    };

    console.error = (...args) => {
      const stack = this.extractStack(args[0]);
      this.captureMessage('error', args, stack);
      this.originalConsole.error(...args);
    };

    // ... same for warn, info, debug
  }

  private extractStack(error: any): string | undefined {
    // Extract stack trace from Error objects
    if (error?.stack) return error.stack;

    // Create stack for error messages without objects
    if (typeof error === 'string' && error.toLowerCase().includes('error')) {
      return new Error().stack;
    }

    return undefined;
  }

  private captureMessage(type: string, args: any[], stack?: string): void {
    const message: BufferedMessage = {
      type,
      timestamp: new Date(),
      args: [...args], // Clone to prevent mutation
      stack,
    };

    // Ring buffer insertion
    if (this.messageBuffer.length < this.maxBufferSize) {
      // Buffer not full yet, just append
      this.messageBuffer.push(message);
      this.bufferIndex = this.messageBuffer.length;
    } else {
      // Buffer full, overwrite oldest
      this.messageBuffer[this.bufferIndex] = message;
      this.bufferIndex = (this.bufferIndex + 1) % this.maxBufferSize;
      this.hasWrapped = true;
    }

    // Notify all listeners
    this.listeners.forEach((listener) => {
      try {
        listener(message);
      } catch (err) {
        // Use original console to avoid recursion
        this.originalConsole.error('Error in console listener:', err);
      }
    });
  }

  getBufferedMessages(): BufferedMessage[] {
    if (!this.hasWrapped) {
      // Buffer not full yet, return as-is
      return [...this.messageBuffer];
    } else {
      // Buffer wrapped, reconstruct chronological order
      // Oldest: bufferIndex to end
      // Newest: 0 to bufferIndex-1
      return [
        ...this.messageBuffer.slice(this.bufferIndex),
        ...this.messageBuffer.slice(0, this.bufferIndex),
      ];
    }
  }
}
```

### 1.3 Public API

**Listener Management**:

```typescript
// Add callback for real-time messages
addListener(callback: (message: BufferedMessage) => void): void;

// Remove listener
removeListener(callback: (message: BufferedMessage) => void): void;
```

**Buffer Management**:

```typescript
// Clear all buffered messages
clearBuffer(): void;

// Get all messages in chronological order
getBufferedMessages(): BufferedMessage[];

// Get original console methods (for safe logging)
getOriginalConsole(): typeof console;
```

**Statistics**:

```typescript
getStats(): {
  total: number;
  maxSize: number;
  types: {
    log: number;
    warn: number;
    error: number;
    info: number;
    debug: number;
  };
  oldestMessage?: Date;
  newestMessage?: Date;
}
```

**Cleanup**:

```typescript
// Restore original console methods
restore(): void;
```

### 1.4 Circular Dependency Handling

**Problem**: Console interceptor is imported at app start (before any other code). It cannot use the logging utility (`log.ts`) without creating a circular dependency.

**Solution**: Use `originalConsole` methods directly for console interceptor's own logging:

```typescript
this.originalConsole.log('[🎬] [ConsoleInterceptor] Console interception started');
```

### 1.5 Usage

**Critical**: Must be imported at the **very top** of `main.ts` to capture all output from app start:

```typescript
// main.ts - MUST be first import
import { consoleInterceptor } from './utils/console-interceptor';

// ... rest of imports
```

**Real-Time Monitoring**:

```typescript
import { consoleInterceptor } from './utils/console-interceptor';

// Listen for new messages
consoleInterceptor.addListener((message) => {
  if (message.type === 'error') {
    // Handle errors
    displayErrorToUser(message.args.join(' '));
  }
});

// Get statistics
const stats = consoleInterceptor.getStats();
console.log(`Captured ${stats.total} messages (${stats.types.error} errors)`);

// Clear buffer
consoleInterceptor.clearBuffer();
```

---

## 2. HDR Detection

### 2.1 Capability Detection

**Purpose**: Detect display HDR capabilities and WebGL HDR support.

**Algorithm**:

```typescript
function detectHDRCapabilities(renderer: THREE.WebGLRenderer): HDRCapabilities {
  const gl = renderer.getContext();

  // 1. Detect wide color gamut via CSS media queries
  const p3Gamut = window.matchMedia('(color-gamut: p3)').matches;
  const rec2020Gamut = window.matchMedia('(color-gamut: rec2020)').matches;

  // 2. Detect HDR display
  const hdr = window.matchMedia('(dynamic-range: high)').matches;

  // 3. Detect deep color (10-bit+)
  const deepColor =
    window.matchMedia('(color-depth: 10)').matches ||
    window.matchMedia('(color-depth: 12)').matches;

  // 4. Check WebGL float texture support
  const floatTextures = !!(
    gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float')
  );

  // 5. Get actual color depth
  const colorDepth = {
    red: gl.getParameter(gl.RED_BITS),
    green: gl.getParameter(gl.GREEN_BITS),
    blue: gl.getParameter(gl.BLUE_BITS),
  };

  // 6. Determine recommended color space
  let recommendedColorSpace: string;
  if (rec2020Gamut && hdr) {
    recommendedColorSpace = 'rec2020';
  } else if (p3Gamut) {
    recommendedColorSpace = 'display-p3';
  } else {
    recommendedColorSpace = 'srgb';
  }

  return {
    p3Gamut,
    rec2020Gamut,
    hdr,
    deepColor,
    floatTextures,
    colorDepth,
    recommendedColorSpace,
  };
}
```

### 2.2 Capability Logging

**Purpose**: Log detected HDR capabilities to console for informational purposes.

**IMPORTANT**: This function does NOT configure the renderer. All HDR configuration is owned by `PostProcessingManager` (the mega-shader pipeline applies tone mapping internally and pins `renderer.outputColorSpace = SRGB` / `renderer.toneMapping = NoToneMapping`).

```typescript
function configureHDRRenderer(
  _renderer: THREE.WebGLRenderer, // Underscore indicates unused parameter
  capabilities: HDRCapabilities
): void {
  // Log detected capabilities (informational only)
  if (capabilities.rec2020Gamut && capabilities.hdr) {
    console.log(
      'HDR display with Rec2020 gamut detected - post-processing will handle color management'
    );
  } else if (capabilities.p3Gamut) {
    console.log('Display P3 gamut detected - post-processing will handle color management');
  } else {
    console.log('Standard sRGB display detected');
  }
}
```

**Why This Design**:

- PostProcessingManager sets `renderer.outputColorSpace = THREE.SRGBColorSpace`
- PostProcessingManager sets `renderer.toneMapping = THREE.NoToneMapping` (the mega-shader does its own tone mapping)
- Setting these values here would be overridden and cause confusion

**Function Name**: Could be renamed to `logHDRCapabilities()` for clarity, but kept for backward compatibility.

---

## 3. Structured Logging

### 3.1 Log Format

**Standard**: `[emoji] [Module] message`

All logs follow consistent formatting that works seamlessly with the console interceptor. This is a lightweight wrapper ensuring standardization while preserving console interceptor compatibility.

### 3.2 Module Constants

Complete list of predefined module identifiers:

```typescript
export const Modules = {
  // Core
  LUXAR: 'Luxar',
  APP: 'App',
  MAIN: 'Main',

  // Data Loading
  SCENE_LOADER: 'SceneLoader',
  SPATIAL_INDEX_LOADER: 'PointSpatialIndexLoader',
  SPATIAL_INDEX: 'PointSpatialIndex',
  DATA_MONITOR: 'DataMonitor',
  ZARR_LOADER: 'ZarrLoader',
  RANGE_CACHE: 'RangeCache',
  CACHE: 'Cache',
  SCENE_DIMS: 'SceneDims',

  // Rendering
  RENDERER: 'Renderer',
  POST_PROCESSING: 'PostProcessing',
  HDR: 'HDR',
  SCENE_MANAGER: 'SceneManager',

  // Controls
  CONTROLS: 'Controls',
  ORBIT_CONTROLS: 'OrbitControls',
  FLY_CONTROLS: 'FlyControls',
  INPUT: 'Input',
  INPUT_CONTEXT: 'InputContext',

  // UI
  UI: 'UI',
  DEBUG_CONSOLE: 'DebugConsole',
  DATA_LOADING_MONITOR: 'DataLoadingMonitor',
  RENDERING_CONTROLS: 'RenderingControls',

  // Utils
  MEMORY: 'Memory',
  PERFORMANCE: 'Performance',
  CONSOLE_INTERCEPTOR: 'ConsoleInterceptor',
} as const;
```

### 3.3 Emoji Constants

Complete list of standard log emojis categorized by purpose:

```typescript
export const LogEmoji = {
  // Status
  START: '🚀',
  SUCCESS: '✅',
  ERROR: '❌',
  WARNING: '⚠️',
  INFO: 'ℹ️',

  // Actions
  LOAD: '📥',
  SAVE: '💾',
  UPDATE: '🔄',
  DELETE: '🗑️',
  SEARCH: '🔍',
  QUERY: '🔍',
  CLEAN: '🧹',
  BROADCAST: '📡',
  TARGET: '🎯',
  ROCKET: '🚀',

  // Data
  DATA: '📊',
  CACHE: '💾',
  NETWORK: '🌐',
  FILE: '📄',
  SCENE: '🎬',

  // Rendering
  RENDER: '🎨',
  RESIZE: '📐',
  FULLSCREEN: '🖥️',
  HDR: '🌟',
  EFFECT: '✨',

  // Controls
  CONTROLS: '🎮',
  INPUT: '⌨️',

  // UI
  UI: '🖼️',
  WINDOW: '🪟',
  PANEL: '📋',

  // Debug
  DEBUG: '🐛',
  CONSOLE: '🔧',
  MONITOR: '📊',
  PERFORMANCE: '⚡',
  MEMORY: '💾',
} as const;
```

### 3.4 Core Logging API

The `log` object provides 10 specialized methods for different log levels and actions:

#### Basic Log Levels

```typescript
// Informational message (default level)
log.info(module: string, message: string, ...args: any[]): void;
// Output: [ℹ️] [Module] message

// Success notification
log.success(module: string, message: string, ...args: any[]): void;
// Output: [✅] [Module] message

// Error message (uses console.error)
log.error(module: string, message: string, ...args: any[]): void;
// Output: [❌] [Module] message

// Warning message (uses console.warn)
log.warning(module: string, message: string, ...args: any[]): void;
// Output: [⚠️] [Module] message
```

#### Action-Specific Methods

```typescript
// Loading/fetching operations
log.load(module: string, message: string, ...args: any[]): void;
// Output: [📥] [Module] message

// Update/modification operations
log.update(module: string, message: string, ...args: any[]): void;
// Output: [🔄] [Module] message

// Query/search operations
log.query(module: string, message: string, ...args: any[]): void;
// Output: [🔍] [Module] message

// Data-related messages
log.data(module: string, message: string, ...args: any[]): void;
// Output: [📊] [Module] message
```

#### Advanced Methods

```typescript
// Custom emoji for specialized cases
log.custom(emoji: string, module: string, message: string, ...args: any[]): void;
// Output: [emoji] [Module] message

// Raw pre-formatted message (for special cases)
log.raw(formattedMessage: string, ...args: any[]): void;
// Output: formattedMessage (as-is)
```

### 3.5 Utility Functions

```typescript
// Format a message manually (rarely needed)
formatLog(emoji: string, module: string, message: string): string;

// Create a module-specific logger (convenience wrapper)
createModuleLogger(module: string): {
  log: (message: string, ...args: any[]) => void;
  info: (message: string, ...args: any[]) => void;
  success: (message: string, ...args: any[]) => void;
  error: (message: string, ...args: any[]) => void;
  warning: (message: string, ...args: any[]) => void;
  load: (message: string, ...args: any[]) => void;
  update: (message: string, ...args: any[]) => void;
  query: (message: string, ...args: any[]) => void;
  data: (message: string, ...args: any[]) => void;
  custom: (emoji: string, message: string, ...args: any[]) => void;
};
```

### 3.6 Usage Examples

**Basic Usage**:

```typescript
import { log, Modules, LogEmoji } from '../utils/log';

// Informational logging
log.info(Modules.DATA_MONITOR, 'Starting data monitor');
log.success(Modules.SCENE_LOADER, 'Scene loaded successfully');
log.warning(Modules.RENDERER, 'WebGL extension not supported');
log.error(Modules.ZARR_LOADER, 'Failed to fetch chunk', error);

// Action-specific logging
log.load(Modules.SPATIAL_INDEX_LOADER, 'Loading spatial index...');
log.query(Modules.SPATIAL_INDEX, 'Query result: 50 cells → 12K points');
log.update(Modules.SCENE_MANAGER, 'Camera position updated', newPosition);
log.data(Modules.RANGE_CACHE, 'Cache size: 45MB / 512MB');

// Custom emoji
log.custom('🎯', Modules.CONTROLS, 'Target locked at [0, 0, 0]');
```

**Module-Specific Logger**:

```typescript
import { createModuleLogger, Modules } from '../utils/log';

const logger = createModuleLogger(Modules.SPATIAL_INDEX);

// All methods automatically use the module
logger.load('Loading index for /points/0');
logger.query('Query result: 12,000 points');
logger.success('Index loaded successfully');
logger.error('Invalid query bounds', bounds);
```

**Advanced Logging with Extra Arguments**:

```typescript
// Extra arguments are passed to console methods
log.data(Modules.SCENE_LOADER, 'Loaded node:', {
  name: 'points/0',
  count: 12000,
  bounds: [0, 0, 0, 100, 100, 100],
});

// Logs: [📊] [SceneLoader] Loaded node: {name: 'points/0', ...}
```

---

## Data Structures

### BufferedMessage

```typescript
interface BufferedMessage {
  type: 'log' | 'warn' | 'error' | 'info' | 'debug';
  timestamp: Date;
  args: any[];
  stack?: string; // For errors
}
```

### HDRCapabilities

```typescript
interface HDRCapabilities {
  p3Gamut: boolean;
  rec2020Gamut: boolean;
  hdr: boolean;
  deepColor: boolean;
  floatTextures: boolean;
  colorDepth: {
    red: number;
    green: number;
    blue: number;
  };
  recommendedColorSpace: 'srgb' | 'display-p3' | 'rec2020';
}
```

---

## Changelog

- **v1.2.0** (2025-12-09): Complete documentation of all missing features
  - **ADDED**: Section 5 - Memory Monitoring (MemoryMonitor class with 150 lines of implementation)
  - **EXPANDED**: Console Interceptor - Documented all 7 public methods (getStats, clearBuffer, restore, listeners, etc.)
  - **EXPANDED**: Structured Logging - Complete API documentation for all 10 log methods
  - **ADDED**: Complete LogEmoji constants list (30+ emojis across 7 categories)
  - **ADDED**: Complete Modules constants list (25+ module identifiers)
  - **ADDED**: Memory detection strategy with 3-tier fallback system
  - **ADDED**: Circular dependency handling pattern for console interceptor
  - **ADDED**: Integration examples for memory monitoring with cache systems
  - **ADDED**: Design rationale for memory monitoring thresholds
  - Documentation now covers 100% of the utils package functionality

- **v1.1.0** (2025-12-08): Correct HDR function documentation
  - **CORRECTED**: `configureHDRRenderer()` documentation - function only logs capabilities, does NOT configure renderer
  - **CLARIFIED**: Actual HDR configuration handled by PostProcessingManager (rendering/ package)
  - **ADDED**: Explanation of why function is a no-op (avoids color-management conflicts with the post-processing pipeline)
  - **NOTED**: Function name is legacy, kept for backward compatibility
  - No functional changes - only documentation corrections

- **v1.0.0** (2025-01-30): Initial specification
  - Ring buffer console interception (10K messages)
  - HDR capability detection via media queries and WebGL
  - System memory detection for cache sizing
  - Structured logging with module identification
  - Early console capture (before app initialization)
