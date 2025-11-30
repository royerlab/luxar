# luxar-viewer.utils - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2025-01-30

## Purpose

The `luxar-viewer.utils` package provides specialized utility functions for console debugging, HDR display detection, memory management, and structured logging.

**Core Responsibility**: Provide robust, well-tested utility functions for system-level operations including console output capture, display capability detection, and standardized logging.

---

## Table of Contents

1. [Console Interception](#console-interception)
2. [HDR Detection](#hdr-detection)
3. [Memory Detection](#memory-detection)
4. [Structured Logging](#structured-logging)

---

## 1. Console Interception

### 1.1 Purpose

Capture all browser console output in a ring buffer for in-app debugging.

### 1.2 Ring Buffer Implementation

**Data Structure**:

```typescript
class ConsoleInterceptor {
    private messageBuffer: BufferedMessage[] = new Array(maxBufferSize)
    private bufferIndex: number = 0
    private hasWrapped: boolean = false
    private readonly maxBufferSize = 10000

    intercept(): void {
        // Save original console methods
        const originalLog = console.log
        const originalWarn = console.warn
        const originalError = console.error

        // Override console methods
        console.log = (...args) => {
            this.addMessage('log', args)
            originalLog.apply(console, args)
        }

        // ... same for warn, error, info, debug
    }

    private addMessage(type: string, args: any[]): void {
        const message: BufferedMessage = {
            type,
            timestamp: new Date(),
            args: args,
            stack: type === 'error' ? new Error().stack : undefined
        }

        // Ring buffer insertion
        this.messageBuffer[this.bufferIndex] = message
        this.bufferIndex = (this.bufferIndex + 1) % this.maxBufferSize

        if (this.bufferIndex === 0) {
            this.hasWrapped = true
        }

        // Notify listeners
        this.notifyListeners(message)
    }

    getBufferedMessages(): BufferedMessage[] {
        if (!this.hasWrapped) {
            // Buffer not full yet, return filled portion
            return this.messageBuffer.slice(0, this.bufferIndex)
        } else {
            // Buffer wrapped, return in correct order
            return [
                ...this.messageBuffer.slice(this.bufferIndex),
                ...this.messageBuffer.slice(0, this.bufferIndex)
            ]
        }
    }
}
```

**Critical**: Must be imported at the **very top** of `main.ts` to capture all output from app start.

---

## 2. HDR Detection

### 2.1 Capability Detection

**Purpose**: Detect display HDR capabilities and WebGL HDR support.

**Algorithm**:

```typescript
function detectHDRCapabilities(renderer: THREE.WebGLRenderer): HDRCapabilities {
    const gl = renderer.getContext()

    // 1. Detect wide color gamut via CSS media queries
    const p3Gamut = window.matchMedia('(color-gamut: p3)').matches
    const rec2020Gamut = window.matchMedia('(color-gamut: rec2020)').matches

    // 2. Detect HDR display
    const hdr = window.matchMedia('(dynamic-range: high)').matches

    // 3. Detect deep color (10-bit+)
    const deepColor = window.matchMedia('(color-depth: 10)').matches ||
                      window.matchMedia('(color-depth: 12)').matches

    // 4. Check WebGL float texture support
    const floatTextures = !!(
        gl.getExtension('EXT_color_buffer_float') ||
        gl.getExtension('EXT_color_buffer_half_float')
    )

    // 5. Get actual color depth
    const colorDepth = {
        red: gl.getParameter(gl.RED_BITS),
        green: gl.getParameter(gl.GREEN_BITS),
        blue: gl.getParameter(gl.BLUE_BITS)
    }

    // 6. Determine recommended color space
    let recommendedColorSpace: string
    if (rec2020Gamut && hdr) {
        recommendedColorSpace = 'rec2020'
    } else if (p3Gamut) {
        recommendedColorSpace = 'display-p3'
    } else {
        recommendedColorSpace = 'srgb'
    }

    return {
        p3Gamut,
        rec2020Gamut,
        hdr,
        deepColor,
        floatTextures,
        colorDepth,
        recommendedColorSpace
    }
}
```

### 2.2 Renderer Configuration

**Purpose**: Configure THREE.js renderer based on detected capabilities.

```typescript
function configureHDRRenderer(
    renderer: THREE.WebGLRenderer,
    capabilities: HDRCapabilities
): void {
    // Set output color space based on display
    if (capabilities.rec2020Gamut && capabilities.hdr) {
        renderer.outputColorSpace = THREE.LinearSRGBColorSpace
    } else if (capabilities.p3Gamut) {
        renderer.outputColorSpace = THREE.SRGBColorSpace
    } else {
        renderer.outputColorSpace = THREE.SRGBColorSpace
    }

    // Note: Tone mapping handled by PostProcessingManager
}
```

---

## 3. Memory Detection

### 3.1 System Memory Estimation

**Purpose**: Detect available system memory for intelligent cache sizing.

**Algorithm**:

```typescript
function detectAvailableMemory(): number {
    // Try to get actual memory from performance API
    if ('memory' in performance) {
        const perfMemory = (performance as any).memory
        if (perfMemory?.jsHeapSizeLimit) {
            return perfMemory.jsHeapSizeLimit / (1024 * 1024)  // MB
        }
    }

    // Fallback: Estimate from device class
    const userAgent = navigator.userAgent.toLowerCase()

    // Mobile devices
    if (/mobile|android|iphone|ipad/.test(userAgent)) {
        return 512  // Conservative 512 MB for mobile
    }

    // Desktop/laptop
    return 2048  // 2 GB default for desktop
}
```

**Usage**: Size caches appropriately (e.g., 25% of available memory).

---

## 4. Structured Logging

### 4.1 Log Format

**Standard**: `[emoji] [Module] message`

**Module Enum**:

```typescript
enum Modules {
    Luxar = 'Luxar',
    Data = 'Data',
    Scene = 'Scene',
    Rendering = 'Rendering',
    Controls = 'Controls',
    Input = 'Input'
}
```

**Emoji Convention**:

```typescript
enum LogEmoji {
    Info = 'ℹ️',
    Success = '✅',
    Warning = '⚠️',
    Error = '❌',
    Loading = '⏳',
    Debug = '🔧'
}
```

### 4.2 Logging Function

```typescript
function log(
    module: Modules,
    message: string,
    emoji: LogEmoji = LogEmoji.Info
): void {
    console.log(`[${emoji}] [${module}] ${message}`)
}
```

**Usage**:

```typescript
import { log, Modules, LogEmoji } from '../utils/log'

log(Modules.Data, 'Loading spatial index...', LogEmoji.Loading)
log(Modules.Data, 'Loaded 50 cells → 12K points', LogEmoji.Success)
log(Modules.Rendering, 'MSAA incompatible with additive blending', LogEmoji.Warning)
```

---

## Data Structures

### BufferedMessage

```typescript
interface BufferedMessage {
    type: 'log' | 'warn' | 'error' | 'info' | 'debug'
    timestamp: Date
    args: any[]
    stack?: string  // For errors
}
```

### HDRCapabilities

```typescript
interface HDRCapabilities {
    p3Gamut: boolean
    rec2020Gamut: boolean
    hdr: boolean
    deepColor: boolean
    floatTextures: boolean
    colorDepth: {
        red: number
        green: number
        blue: number
    }
    recommendedColorSpace: 'srgb' | 'display-p3' | 'rec2020'
}
```

---

## Changelog

- **v1.0.0** (2025-01-30): Initial specification
  - Ring buffer console interception (10K messages)
  - HDR capability detection via media queries and WebGL
  - System memory detection for cache sizing
  - Structured logging with module identification
  - Early console capture (before app initialization)
