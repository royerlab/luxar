> **⚠️ Archived — historical document, not maintained.** Kept for design history; it reflects the project state as of its original date and may not match current code. Do not treat it as current guidance. See [the archive README](README.md) for status labels and retention policy.

# Luxar Performance Optimization Specification v3.6.0

> **v3.6.0 Changes (2025-12-23)**:
> - 🟡 FIXED: Config uses `dataLoading.performance` (not separate `dataLoader`)
> - 🟡 FIXED: ArrayDecoder location clarified (MAIN THREAD, not worker) in all places
> - 🟡 FIXED: Worker init comment corrected (WASM only, not ArrayDecoder)
> - 🟡 FIXED: Phase 2 success criteria corrected (WASM queries, not ArrayDecoder in worker)
> - 🟡 FIXED: Architecture diagram shows ArrayDecoder on MAIN THREAD
>
> **v3.5.0 Changes (2025-12-23)**:
> - 🔴 FIXED: processGSplats() now correctly documents Mahalanobis distance algorithm
> - 🔴 FIXED: acquireGSplatsGeometry() no longer takes ndim (always 3D after processing)
> - 🔴 FIXED: SceneManager.updateGSplats() now shows correct data flow (process → pack → update)
> - 🔴 ADDED: Note about actual vs proposed GSplats patterns (direct mesh vs GPUBufferPool)
> - 🔴 ADDED: mahalanobisDistance() helper function documentation
> - 🔴 ADDED: Two-pass visibility filtering documentation
>
> **v3.4.0 Changes (2025-12-23)**:
> - 🔴 FIXED: createGSplatsGeometry uses 3D centers and packed Cholesky (aCholesky01/23/45)
> - 🔴 FIXED: updateGSplatsGeometry takes PackedGSplatsData (3D), not LoadedGSplatsData (nD)
> - 🔴 ADDED: processGSplats() transformation function (nD → 3D with amplitude attenuation)
> - 🔴 ADDED: packCholeskyForShader() function (6-element → 3×vec2 packing)
> - 🔴 ADDED: ProcessedGSplatsData and PackedGSplatsData interfaces
> - 🔴 ADDED: Complete GSplats data flow diagram (Loader → processGSplats → packCholesky → GPU)
>
> **v3.3.0 Changes (2025-12-23)**:
> - 🟡 FIXED: createPointsGeometry uses Float32Array (RGB, 3 components) for HDR color support
> - 🟡 FIXED: updatePointsGeometry uses Float32Array and count*3 for colors (matching scene-loader.ts)
>
> **v3.2.0 Changes (2025-12-23)**:
> - 🟡 FIXED: PointsDataAccumulator now supports HDR colors (Float32Array)
> - 🟡 FIXED: Architecture diagram accurately shows ArrayDecoder on main thread
> - 🟡 FIXED: createLinesGeometry uses Float32Array for clipped flags (matches line-material.ts)
> - 🟡 FIXED: Points colors use RGB (3 components), not RGBA (4 components)
> - 🟡 FIXED: Color fill() handles Uint8Array/Uint16Array → Float32Array conversion
>
> **v3.1.0 Changes (2025-12-23)**:
> - 🔴 FIXED: widths in LoadedLinesData are PER-VERTEX, not per-segment
> - 🔴 FIXED: GPU Buffer Pool uses ProcessedLinesData (per-segment), not LoadedLinesData (per-vertex)
> - 🔴 FIXED: ArrayDecoder stays on main thread (needs zarr.Array objects)
> - 🔴 ADDED: buildInstanceBuffers() transformation function for Lines
> - 🔴 FIXED: createLinesGeometry() creates InstancedBufferGeometry with per-segment attributes

**Version**: 3.6.0 (Config Integration & ArrayDecoder Location Consistency)
**Date**: 2025-12-23
**Status**: Ready for Implementation
**Revision Notes**: ALL node types (Points, Lines, GSplats) now have complete, accurate data flow documentation

## Executive Summary

This specification details performance optimizations for Luxar's data loading, caching, and rendering pipeline for **all three primitive types: Points, Lines, and Gaussian Splats (GSplats)**. The design is informed by production-proven patterns from SparkJS and adapted for Luxar's unique requirements: nD visualization, multiple primitive types, and spatial index-based loading.

**Core Goals:**
1. **Eliminate GC Pressure**: Reduce garbage collection pauses from ~10-20ms to <2ms per frame
2. **Offload Main Thread**: Move CPU-intensive work to Web Workers (60fps sustained)
3. **Accelerate Hot Paths**: Use WASM for nD distance computation and spatial queries (3-5x speedup)
4. **Minimize GPU Uploads**: Implement buffer pooling and partial updates (50-60% reduction)

**Supported Primitive Types:**
- **Points**: positions, colors, radii, sharpness
- **Lines**: vertices, segments (indices), widths, colors, sharpness
- **GSplats**: centers, amplitudes, choleskyFactors, colors

**Browser Requirements** (Modern Only - No Backwards Compatibility):
- WebAssembly support (Chrome 57+, Firefox 52+, Safari 11+)
- Web Workers (universal)
- SharedArrayBuffer (Chrome 68+, Firefox 79+, Safari 15.2+)
- WebGL2 (all target browsers)

**If any requirement missing: FAIL with error message. No fallbacks, no degraded mode.**

**Relationship to Existing Cache:**
- **Existing**: L1/L2 zarr chunk cache (network → memory/disk) - **UNCHANGED**
- **New Layer 1**: CPU buffer pooling (TypedArrays) - **Phase 1**
- **New Layer 2**: GPU buffer pooling (THREE.BufferGeometry) - **Phase 4**

These are **complementary layers**, not replacements.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component 1: Generic Data Accumulators](#2-component-1-generic-data-accumulators)
3. [Component 2: Web Workers with ArrayDecoder](#3-component-2-web-workers-with-arraydecoder)
4. [Component 3: WASM Acceleration (All Types)](#4-component-3-wasm-acceleration-all-types)
5. [Component 4: GPU Buffer Pool (All Geometry Types)](#5-component-4-gpu-buffer-pool-all-geometry-types)
6. [Integration Strategy](#6-integration-strategy)
7. [Testing Strategy](#7-testing-strategy)
8. [Performance Targets & Metrics](#8-performance-targets--metrics)
9. [Implementation Phases](#9-implementation-phases)
10. [Edge Cases & Caveats](#10-edge-cases--caveats)

---

## 1. Architecture Overview

### 1.1 Current State - Three Loader Types

Luxar has **three independent spatial index loaders**:

```
PointSpatialIndexLoader
  ├─ Data: positions, colors, radii, sharpness
  ├─ Loading: Single-phase (query chunks → load data)
  └─ Rendering: THREE.Points

LinesSpatialIndexLoader
  ├─ Data: vertices, segments (indices), widths, colors, sharpness
  ├─ Loading: Two-phase (query segments → derive vertices → remap indices)
  └─ Rendering: THREE.Line

GSplatsSpatialIndexLoader
  ├─ Data: centers, amplitudes, choleskyFactors, colors
  ├─ Loading: Single-phase (query chunks → load data)
  └─ Rendering: Custom instanced mesh
```

**Current Pain Points (All Three Types):**
1. **Allocation Storm**: New TypedArrays every frame (~84 MB/sec for 100K points)
2. **Main Thread Blocking**: Decoding + spatial queries block rendering (40-80ms)
3. **GPU Thrashing**: Creating/destroying geometry every update
4. **No SIMD**: TypeScript loops miss vectorization

### 1.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Main Thread (Rendering 60fps)                      │
│                                                                               │
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐  │
│  │  SceneManager    │◄─────┤  MaterialManager │◄─────┤ GPU Buffer Pool  │  │
│  │  - THREE.Scene   │      │  - Material cache│      │ - Points geom    │  │
│  │  - Render loop   │      │  - Uniform update│      │ - Lines geom     │  │
│  └────────┬─────────┘      └──────────────────┘      │ - GSplats geom   │  │
│           │                                           └──────────────────┘  │
│           │ Request load (ViewState)                                        │
│           ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │         Loader Coordinator (Polymorphic by Type)                      │   │
│  │  PointSpatialIndexLoader   | Lines...  | GSplats...                  │   │
│  │  - Owns persistent Accumulator (per type)                            │   │
│  │  - Fetches zarr chunks (async IO - non-blocking)                     │   │
│  │  - Decodes via ArrayDecoder (on main thread!)                        │   │
│  │  - Dispatches spatial queries to worker                              │   │
│  └───────────────────────────┬──────────────────────────────────────────┘   │
│                               │                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  ArrayDecoder (MAIN THREAD - needs zarr.Array objects)                │  │
│  │  Handles: broadcasting, LUT, quantization, array_ref, delta, Blosc   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                               │ postMessage(spatial query params)            │
└───────────────────────────────┼───────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Worker Thread (WASM Processing)                     │
│                                                                               │
│  ┌──────────────────┐                             ┌──────────────────┐       │
│  │ SpatialQueryTask │                             │ VisibilityTask   │       │
│  │ - WASM query     │                             │ - WASM nD dist   │       │
│  │ - Chunk bounds   │                             │ - Filter points  │       │
│  └────────┬─────────┘                             └────────┬─────────┘       │
│           │                                                 │                 │
│           └─────────────────────┬───────────────────────────┘                 │
│                                 ▼                                             │
│                  ┌───────────────────────────────────────────┐               │
│                  │         WASM Module (Rust)                │               │
│                  │  - query_chunks_for_view()               │               │
│                  │  - compute_nd_visibility_points()        │               │
│                  │  - compute_nd_visibility_lines()         │               │
│                  │  - compute_nd_visibility_gsplats()       │               │
│                  │  - interleave_attributes() [optional]    │               │
│                  └───────────────────────────────────────────┘               │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Worker-Zarr Pattern Clarification

**CRITICAL**: Worker receives **PRE-FETCHED encoded chunks** (not zarr URLs):

```typescript
// Main thread: Fetch encoded chunk from zarr
const zarrArray = await zarr.open(location, { kind: 'array' });
const encodedChunk = await zarrArray.get([chunkIndex]);

// Worker: Decode only (via ArrayDecoder)
const decoded = await worker.decodeChunk({
  encodedData: encodedChunk,  // Already fetched!
  attrs: arrayMetadata,
});
```

This avoids complex zarr store handling in workers.

---

## 2. Component 1: Generic Data Accumulators

### 2.1 Problem Statement

All three loaders allocate new TypedArrays every view update:

```typescript
// Current: ~84 MB/sec garbage for 100K points at 30fps
// In PointSpatialIndexLoader:
const positions = new Float32Array(count * 3);  // 1.2 MB
const colors = new Uint8Array(count * 4);       // 400 KB
const radii = new Float32Array(count);          // 400 KB
const sharpness = new Float32Array(count);      // 400 KB

// Similar for Lines (vertices + indices + widths) and GSplats (centers + cholesky + ...)
```

### 2.2 Solution: Type-Specific Accumulator Pattern

**Design Principle**: Generic accumulator interface, specialized implementations per type.

**File**: `packages/luxar-viewer/src/data/data-accumulator.ts` (NEW)

```typescript
/**
 * Generic data accumulator interface for all primitive types.
 *
 * Key Features:
 * - Allocates buffers once with 1.5x headroom
 * - Returns views (subarrays) of persistent buffers
 * - Only reallocates when capacity exceeded
 * - Eliminates per-frame GC
 *
 * Inspired by: SparkJS PackedSplats.ensureSplats()
 */

import * as THREE from 'three';
import type {
  PointsData,
  ViewState,
  LoaderConfig,
  SceneNode,
} from './data-loader-types';

import type {
  LoadedLinesData,
  LinesViewState,
  LinesDataLoader,
} from '../types/lines';

import type {
  LoadedGSplatsData,
  GSplatsViewState,
  GSplatsDataLoader,
} from '../types/gsplats';

import { log, Modules } from '../utils/log';

/**
 * Base accumulator interface
 *
 * NOTE: Lines accumulator has different signatures (needs 2 counts/offsets)
 * so it doesn't strictly implement this interface - that's OK, it's a guide not a contract
 */
export interface DataAccumulator<TData> {
  /**
   * Ensure capacity for at least `needed` elements
   * @returns true if buffer grew (reallocation occurred)
   */
  ensureCapacity(needed: number): boolean;

  /**
   * Get data view with specified count(s)
   * WARNING: Returned arrays are views into persistent buffers
   *
   * Signature varies by type:
   * - Points: getData(count: number)
   * - Lines: getData(segmentCount: number, vertexCount: number)
   * - GSplats: getData(count: number)
   */
  getData(...args: number[]): TData;

  /**
   * Fill accumulator at offset(s)
   *
   * Signature varies by type:
   * - Points: fill(offset: number, data: Partial<TData>)
   * - Lines: fill(segmentOffset: number, vertexOffset: number, data: Partial<TData>)
   * - GSplats: fill(offset: number, data: Partial<TData>)
   */
  fill(...args: any[]): void;

  /**
   * Get statistics
   */
  getStats(): AccumulatorStats;

  /**
   * Dispose (release memory)
   */
  dispose(): void;
}

export interface AccumulatorStats {
  capacity: number;
  allocations: number;
  growthEvents: number;
  memoryMB: number;
}

/**
 * Points data accumulator
 *
 * NOTE: Supports both SDR (Uint8Array) and HDR (Float32Array) colors.
 * The actual codebase handles Float16Array, Float32Array, Uint8Array, Uint16Array.
 * This accumulator uses Float32Array internally and can convert on output.
 */
export class PointsDataAccumulator implements DataAccumulator<PointsData> {
  // Persistent buffers (never disposed until loader destroyed)
  private positionBuffer: Float32Array;
  private colorBuffer: Float32Array;  // Use Float32 to support HDR!
  private radiiBuffer: Float32Array;
  private sharpnessBuffer: Float32Array;

  // Current capacity (number of points)
  private capacity: number;

  // Color mode tracking
  private isHDR = false;  // Track whether colors are HDR (need full Float32) or SDR

  // Statistics
  private allocations = 0;
  private totalGrowths = 0;

  // Metadata tracking
  private bounds = new THREE.Box3();
  private ndim: number;
  private totalPoints: number;
  private usedSpatialIndex = false;

  constructor(initialCapacity = 1024, ndim = 3, totalPoints = 0) {
    this.capacity = initialCapacity;
    this.ndim = ndim;
    this.totalPoints = totalPoints;

    this.positionBuffer = new Float32Array(initialCapacity * 3);
    this.colorBuffer = new Float32Array(initialCapacity * 3);  // RGB Float32 for HDR support
    this.radiiBuffer = new Float32Array(initialCapacity);
    this.sharpnessBuffer = new Float32Array(initialCapacity);

    this.allocations++;
  }

  /**
   * Ensure capacity with 1.5x growth strategy
   */
  ensureCapacity(needed: number): boolean {
    if (needed <= this.capacity) return false;

    // Calculate new capacity with 1.5x growth factor
    let newCapacity = this.capacity;
    while (newCapacity < needed) {
      newCapacity = Math.ceil(newCapacity * 1.5);
    }

    log.info(
      Modules.DATA_LOADER,
      `Growing PointsDataAccumulator: ${this.capacity} → ${newCapacity} points`
    );

    // Allocate new buffers
    const newPositions = new Float32Array(newCapacity * 3);
    const newColors = new Float32Array(newCapacity * 3);  // RGB Float32
    const newRadii = new Float32Array(newCapacity);
    const newSharpness = new Float32Array(newCapacity);

    // Copy existing data
    newPositions.set(this.positionBuffer);
    newColors.set(this.colorBuffer);
    newRadii.set(this.radiiBuffer);
    newSharpness.set(this.sharpnessBuffer);

    // Replace buffers
    this.positionBuffer = newPositions;
    this.colorBuffer = newColors;
    this.radiiBuffer = newRadii;
    this.sharpnessBuffer = newSharpness;
    this.capacity = newCapacity;

    this.allocations++;
    this.totalGrowths++;

    return true;
  }

  /**
   * Set whether colors are HDR (for proper output type)
   */
  setHDRMode(isHDR: boolean): void {
    this.isHDR = isHDR;
  }

  /**
   * Get PointsData with metadata (CORRECT interface matching data-loader-types.ts!)
   *
   * Colors are returned as Float32Array (RGB). The actual scene-loader handles
   * both SDR and HDR colors - Float32Array works for both.
   */
  getData(count: number): PointsData {
    if (count > this.capacity) {
      throw new Error(
        `Cannot get ${count} points from accumulator with capacity ${this.capacity}`
      );
    }

    // Compute bounds from loaded positions
    this.bounds.makeEmpty();
    for (let i = 0; i < count; i++) {
      const x = this.positionBuffer[i * 3 + 0];
      const y = this.positionBuffer[i * 3 + 1];
      const z = this.positionBuffer[i * 3 + 2];
      this.bounds.expandByPoint(new THREE.Vector3(x, y, z));
    }

    // Return PointsData with CORRECT metadata structure
    // Colors are RGB Float32 (3 components, not 4)
    return {
      positions: this.positionBuffer.subarray(0, count * 3),
      colors: this.colorBuffer.subarray(0, count * 3),  // RGB, not RGBA!
      radii: this.radiiBuffer.subarray(0, count),
      sharpness: this.sharpnessBuffer.subarray(0, count),
      metadata: {
        totalPoints: this.totalPoints,
        loadedPoints: count,
        bounds: this.bounds.clone(),
        ndim: this.ndim,
        usedSpatialIndex: this.usedSpatialIndex,
        dtypes: {
          positions: 'float32',
          colors: this.isHDR ? 'float32' : 'uint8',  // Track actual source dtype
          radii: 'float32',
          sharpness: 'float32',
        },
      },
    };
  }

  /**
   * Fill accumulator at offset
   *
   * NOTE: Colors are stored as RGB (3 components), not RGBA.
   * Input can be Float32Array (HDR) or Uint8Array/Uint16Array (SDR).
   * SDR colors are converted to [0,1] float range on fill.
   */
  fill(offset: number, data: Partial<PointsData>): void {
    if (data.positions) {
      this.positionBuffer.set(data.positions, offset * 3);
    }
    if (data.colors) {
      // Handle different color array types
      if (data.colors instanceof Float32Array) {
        this.colorBuffer.set(data.colors, offset * 3);
      } else if (data.colors instanceof Uint8Array) {
        // Convert Uint8 [0,255] to Float32 [0,1]
        const floatColors = new Float32Array(data.colors.length);
        for (let i = 0; i < data.colors.length; i++) {
          floatColors[i] = data.colors[i] / 255;
        }
        this.colorBuffer.set(floatColors, offset * 3);
      } else if (data.colors instanceof Uint16Array) {
        // Convert Uint16 [0,65535] to Float32 [0,1]
        const floatColors = new Float32Array(data.colors.length);
        for (let i = 0; i < data.colors.length; i++) {
          floatColors[i] = data.colors[i] / 65535;
        }
        this.colorBuffer.set(floatColors, offset * 3);
      }
    }
    if (data.radii) {
      this.radiiBuffer.set(data.radii as Float32Array, offset);
    }
    if (data.sharpness) {
      this.sharpnessBuffer.set(data.sharpness as Float32Array, offset);
    }
  }

  /**
   * Update metadata (called by loader)
   */
  updateMetadata(metadata: {
    ndim?: number;
    totalPoints?: number;
    usedSpatialIndex?: boolean;
  }): void {
    if (metadata.ndim !== undefined) this.ndim = metadata.ndim;
    if (metadata.totalPoints !== undefined) this.totalPoints = metadata.totalPoints;
    if (metadata.usedSpatialIndex !== undefined)
      this.usedSpatialIndex = metadata.usedSpatialIndex;
  }

  getStats(): AccumulatorStats {
    // Memory: pos(12) + color(12) + radii(4) + sharpness(4) = 32 bytes per point
    return {
      capacity: this.capacity,
      allocations: this.allocations,
      growthEvents: this.totalGrowths,
      memoryMB: (this.capacity * 32) / 1024 / 1024,
    };
  }

  dispose(): void {
    this.positionBuffer = new Float32Array(0);
    this.colorBuffer = new Float32Array(0);
    this.radiiBuffer = new Float32Array(0);
    this.sharpnessBuffer = new Float32Array(0);
    this.capacity = 0;
    this.isHDR = false;
  }
}

/**
 * Lines data accumulator (CORRECTED STRUCTURE!)
 *
 * CRITICAL: Flat buffers matching types/lines.ts:170-194
 * CRITICAL: widths is PER-VERTEX (N,), NOT per-segment!
 */
export class LinesDataAccumulator implements DataAccumulator<LoadedLinesData> {
  // FLAT buffers (not nested!)
  private vertexBuffer: Float32Array;        // ndim-dimensional vertices
  private segmentBuffer: Uint32Array;        // index pairs
  private widthBuffer: Float32Array;         // PER-VERTEX widths (N,) - NOT per-segment!
  private colorBuffer: Float32Array;         // RGB Float32 per-vertex (always allocated)
  private sharpnessBuffer: Float32Array;     // per-vertex (always allocated)

  // Track whether data actually has colors/sharpness
  private hasColors = false;
  private hasSharpness = false;

  private vertexCapacity: number;
  private segmentCapacity: number;
  private ndim: number;

  private allocations = 0;
  private totalGrowths = 0;

  constructor(
    initialVertexCapacity = 1024,
    initialSegmentCapacity = 512,
    ndim = 3
  ) {
    this.vertexCapacity = initialVertexCapacity;
    this.segmentCapacity = initialSegmentCapacity;
    this.ndim = ndim;

    // Always allocate buffers (persistent, never null)
    // NOTE: widths is per-VERTEX, allocated with vertexCapacity!
    this.vertexBuffer = new Float32Array(initialVertexCapacity * ndim);
    this.segmentBuffer = new Uint32Array(initialSegmentCapacity * 2);
    this.widthBuffer = new Float32Array(initialVertexCapacity);  // PER-VERTEX!
    this.colorBuffer = new Float32Array(initialVertexCapacity * 3); // RGB
    this.sharpnessBuffer = new Float32Array(initialVertexCapacity);

    this.allocations++;
  }

  ensureCapacity(needed: number): boolean {
    // Lines need capacity for both vertices and segments
    // Estimate: vertices ≈ segments * 1.5 (segments share vertices)
    const neededVertices = Math.ceil(needed * 1.5);
    const neededSegments = needed;

    let grew = false;

    // Grow vertices if needed (widths grow with vertices!)
    if (neededVertices > this.vertexCapacity) {
      let newVertexCap = this.vertexCapacity;
      while (newVertexCap < neededVertices) {
        newVertexCap = Math.ceil(newVertexCap * 1.5);
      }

      const newVertexBuf = new Float32Array(newVertexCap * this.ndim);
      const newWidthBuf = new Float32Array(newVertexCap);  // PER-VERTEX!
      const newColorBuf = new Float32Array(newVertexCap * 3); // RGB
      const newSharpnessBuf = new Float32Array(newVertexCap);

      newVertexBuf.set(this.vertexBuffer);
      newWidthBuf.set(this.widthBuffer);  // widths grow with vertices
      newColorBuf.set(this.colorBuffer);
      newSharpnessBuf.set(this.sharpnessBuffer);

      this.vertexBuffer = newVertexBuf;
      this.widthBuffer = newWidthBuf;
      this.colorBuffer = newColorBuf;
      this.sharpnessBuffer = newSharpnessBuf;
      this.vertexCapacity = newVertexCap;

      grew = true;
    }

    // Grow segments if needed (only segment indices, not widths!)
    if (neededSegments > this.segmentCapacity) {
      let newSegmentCap = this.segmentCapacity;
      while (newSegmentCap < neededSegments) {
        newSegmentCap = Math.ceil(newSegmentCap * 1.5);
      }

      const newSegmentBuf = new Uint32Array(newSegmentCap * 2);
      newSegmentBuf.set(this.segmentBuffer);

      this.segmentBuffer = newSegmentBuf;
      this.segmentCapacity = newSegmentCap;

      grew = true;
    }

    if (grew) {
      this.allocations++;
      this.totalGrowths++;
    }

    return grew;
  }

  /**
   * CORRECTED: getData returns LoadedLinesData with proper nullable handling
   * NOTE: widths is per-VERTEX (vertexCount), NOT per-segment!
   */
  getData(segmentCount: number, vertexCount: number): LoadedLinesData {
    return {
      // FLAT structure (not nested)
      vertices: this.vertexBuffer.subarray(0, vertexCount * this.ndim),
      segments: this.segmentBuffer.subarray(0, segmentCount * 2),
      widths: this.widthBuffer.subarray(0, vertexCount),  // PER-VERTEX! Not segmentCount!
      colors: this.hasColors
        ? this.colorBuffer.subarray(0, vertexCount * 3)
        : null,
      sharpness: this.hasSharpness
        ? this.sharpnessBuffer.subarray(0, vertexCount)
        : null,
      segmentCount,
      vertexCount,
      ndim: this.ndim,
    };
  }

  /**
   * CORRECTED: fill with proper tracking of data presence
   * NOTE: widths uses vertexOffset (per-vertex), NOT segmentOffset!
   */
  fill(segmentOffset: number, vertexOffset: number, data: Partial<LoadedLinesData>): void {
    if (data.vertices) {
      this.vertexBuffer.set(data.vertices, vertexOffset * this.ndim);
    }
    if (data.segments) {
      this.segmentBuffer.set(data.segments, segmentOffset * 2);
    }
    if (data.widths) {
      // widths is PER-VERTEX, use vertexOffset!
      this.widthBuffer.set(data.widths, vertexOffset);
    }
    if (data.colors) {
      this.hasColors = true;
      this.colorBuffer.set(data.colors, vertexOffset * 3);
    }
    if (data.sharpness) {
      this.hasSharpness = true;
      this.sharpnessBuffer.set(data.sharpness, vertexOffset);
    }
  }

  getStats(): AccumulatorStats {
    return {
      capacity: this.segmentCapacity,
      allocations: this.allocations,
      growthEvents: this.totalGrowths,
      memoryMB:
        (this.vertexCapacity * (this.ndim * 4 + 4 + 3 * 4 + 4) + // vertices + widths + colors + sharpness
          this.segmentCapacity * (2 * 4)) / // segments only (no widths here!)
        1024 /
        1024,
    };
  }

  dispose(): void {
    this.vertexBuffer = new Float32Array(0);
    this.segmentBuffer = new Uint32Array(0);
    this.widthBuffer = new Float32Array(0);
    this.colorBuffer = new Float32Array(0);
    this.sharpnessBuffer = new Float32Array(0);
    this.vertexCapacity = 0;
    this.segmentCapacity = 0;
    this.hasColors = false;
    this.hasSharpness = false;
  }
}

/**
 * GSplats data accumulator (CORRECTED FIELD NAMES!)
 *
 * CRITICAL FIX: camelCase choleskyFactors (not snake_case cholesky_factors)
 */
export class GSplatsDataAccumulator implements DataAccumulator<LoadedGSplatsData> {
  private centerBuffer: Float32Array;
  private amplitudeBuffer: Float32Array;
  private choleskyBuffer: Float32Array; // CORRECT: for choleskyFactors field
  private colorBuffer: Float32Array;    // RGB Float32 (always allocated)

  // Track whether data has colors
  private hasColors = false;

  private capacity: number;
  private ndim: number;
  private choleskySize: number; // Elements per splat

  private allocations = 0;
  private totalGrowths = 0;

  constructor(initialCapacity = 1024, ndim = 3) {
    this.capacity = initialCapacity;
    this.ndim = ndim;
    this.choleskySize = (ndim * (ndim + 1)) / 2;

    this.centerBuffer = new Float32Array(initialCapacity * ndim);
    this.amplitudeBuffer = new Float32Array(initialCapacity);
    this.choleskyBuffer = new Float32Array(initialCapacity * this.choleskySize);
    this.colorBuffer = new Float32Array(initialCapacity * 3); // RGB

    this.allocations++;
  }

  ensureCapacity(needed: number): boolean {
    if (needed <= this.capacity) return false;

    let newCapacity = this.capacity;
    while (newCapacity < needed) {
      newCapacity = Math.ceil(newCapacity * 1.5);
    }

    log.info(
      Modules.DATA_LOADER,
      `Growing GSplatsDataAccumulator: ${this.capacity} → ${newCapacity} splats`
    );

    const newCenters = new Float32Array(newCapacity * this.ndim);
    const newAmplitudes = new Float32Array(newCapacity);
    const newCholesky = new Float32Array(newCapacity * this.choleskySize);
    const newColors = new Float32Array(newCapacity * 3); // RGB

    newCenters.set(this.centerBuffer);
    newAmplitudes.set(this.amplitudeBuffer);
    newCholesky.set(this.choleskyBuffer);
    newColors.set(this.colorBuffer); // Always allocated, never null

    this.centerBuffer = newCenters;
    this.amplitudeBuffer = newAmplitudes;
    this.choleskyBuffer = newCholesky;
    this.colorBuffer = newColors;
    this.capacity = newCapacity;

    this.allocations++;
    this.totalGrowths++;

    return true;
  }

  /**
   * CORRECTED: Returns LoadedGSplatsData with camelCase choleskyFactors
   */
  getData(count: number): LoadedGSplatsData {
    return {
      centers: this.centerBuffer.subarray(0, count * this.ndim),
      amplitudes: this.amplitudeBuffer.subarray(0, count),
      choleskyFactors: this.choleskyBuffer.subarray(0, count * this.choleskySize), // CORRECT: camelCase!
      colors: this.hasColors
        ? this.colorBuffer.subarray(0, count * 3)
        : null,  // Nullable based on data presence
      splatCount: count,
      ndim: this.ndim,
    };
  }

  fill(offset: number, data: Partial<LoadedGSplatsData>): void {
    if (data.centers) {
      this.centerBuffer.set(data.centers, offset * this.ndim);
    }
    if (data.amplitudes) {
      this.amplitudeBuffer.set(data.amplitudes, offset);
    }
    if (data.choleskyFactors) { // CORRECT: camelCase!
      this.choleskyBuffer.set(data.choleskyFactors, offset * this.choleskySize);
    }
    if (data.colors) {
      this.hasColors = true;  // Mark as present
      this.colorBuffer.set(data.colors, offset * 3);
    }
  }

  getStats(): AccumulatorStats {
    const bytesPerSplat =
      this.ndim * 4 + // centers
      4 + // amplitude
      this.choleskySize * 4 + // cholesky
      3 * 4; // color (RGB)

    return {
      capacity: this.capacity,
      allocations: this.allocations,
      growthEvents: this.totalGrowths,
      memoryMB: (this.capacity * bytesPerSplat) / 1024 / 1024,
    };
  }

  dispose(): void {
    this.centerBuffer = new Float32Array(0);
    this.amplitudeBuffer = new Float32Array(0);
    this.choleskyBuffer = new Float32Array(0);
    this.colorBuffer = new Float32Array(0);
    this.capacity = 0;
    this.hasColors = false;
  }
}
```

### 2.3 Lines Processing: buildInstanceBuffers()

**CRITICAL**: Lines require transformation from per-vertex LoadedLinesData to per-segment ProcessedLinesData.

**File**: `packages/luxar-viewer/src/data/lines-processor.ts` (NEW)

```typescript
/**
 * Transform LoadedLinesData (per-vertex) to ProcessedLinesData (per-segment).
 *
 * This function:
 * 1. Clips segments to nD slice
 * 2. Converts per-vertex → per-segment attributes
 * 3. Extracts 3D positions from nD space
 * 4. Interpolates attributes for clipped endpoints
 * 5. Calculates segment lengths and tracks clipping state
 */

import type { LoadedLinesData, ProcessedLinesData, ClippedSegment } from '../types/lines';

/**
 * Linear interpolation
 */
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Linear interpolation for 3D vectors
 */
function lerpVec3(a: number[], b: number[], t: number): number[] {
  return [
    lerp(a[0], b[0], t),
    lerp(a[1], b[1], t),
    lerp(a[2], b[2], t),
  ];
}

/**
 * 3D distance calculation
 */
function distance3D(a: number[], b: number[]): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const dz = b[2] - a[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Clip a segment to the nD slice (Cohen-Sutherland style).
 * Returns clipped 3D endpoints and interpolation parameters.
 */
function clipSegmentToSlice(
  p1: number[],
  p2: number[],
  slicePosition: number[],
  tolerance: number[],
  displayDims: number[]
): ClippedSegment {
  let t1 = 0;
  let t2 = 1;
  let visible = true;

  // Check each non-display dimension
  for (let dim = 0; dim < slicePosition.length; dim++) {
    if (displayDims.includes(dim)) continue;

    const pos1 = p1[dim];
    const pos2 = p2[dim];
    const sliceVal = slicePosition[dim];
    const tol = tolerance[dim];

    const min = sliceVal - tol;
    const max = sliceVal + tol;

    // Both outside same side → invisible
    if ((pos1 < min && pos2 < min) || (pos1 > max && pos2 > max)) {
      visible = false;
      break;
    }

    // Clip to bounds
    const delta = pos2 - pos1;
    if (Math.abs(delta) > 1e-10) {
      if (pos1 < min) t1 = Math.max(t1, (min - pos1) / delta);
      if (pos1 > max) t1 = Math.max(t1, (max - pos1) / delta);
      if (pos2 < min) t2 = Math.min(t2, (min - pos1) / delta);
      if (pos2 > max) t2 = Math.min(t2, (max - pos1) / delta);
    }

    if (t1 > t2) {
      visible = false;
      break;
    }
  }

  if (!visible) {
    return { p1: [0, 0, 0], p2: [0, 0, 0], t1: 0, t2: 1, visible: false };
  }

  // Extract 3D positions at clipped t values
  const clippedP1: number[] = [];
  const clippedP2: number[] = [];
  for (const dim of displayDims) {
    clippedP1.push(lerp(p1[dim], p2[dim], t1));
    clippedP2.push(lerp(p1[dim], p2[dim], t2));
  }

  return { p1: clippedP1, p2: clippedP2, t1, t2, visible: true };
}

/**
 * Build GPU instance buffers from loaded lines data.
 *
 * Transforms per-vertex data into per-segment instance attributes.
 *
 * @param loadedData - Raw lines data from loader (per-vertex)
 * @param slicePosition - Current position in nD space
 * @param tolerance - Per-dimension tolerance
 * @param displayDims - Which dimensions to display (e.g., [0, 1, 2])
 * @returns Processed data ready for GPU (per-segment)
 */
export function buildInstanceBuffers(
  loadedData: LoadedLinesData,
  slicePosition: number[],
  tolerance: number[],
  displayDims: number[]
): ProcessedLinesData {
  const { vertices, segments, widths, colors, sharpness, ndim, segmentCount } = loadedData;

  // Pre-allocate output arrays (may be smaller after clipping)
  const maxSegments = segmentCount;
  const startPositions = new Float32Array(maxSegments * 3);
  const endPositions = new Float32Array(maxSegments * 3);
  const startColors = new Float32Array(maxSegments * 3);
  const endColors = new Float32Array(maxSegments * 3);
  const startWidths = new Float32Array(maxSegments);
  const endWidths = new Float32Array(maxSegments);
  const startSharpness = new Float32Array(maxSegments);
  const endSharpness = new Float32Array(maxSegments);
  const segmentLengths = new Float32Array(maxSegments);
  const startClipped = new Uint8Array(maxSegments);
  const endClipped = new Uint8Array(maxSegments);

  let outIdx = 0;

  for (let i = 0; i < segmentCount; i++) {
    // Get vertex indices (local space)
    const v0 = segments[i * 2];
    const v1 = segments[i * 2 + 1];

    // Extract nD positions
    const p1 = Array.from(vertices.slice(v0 * ndim, (v0 + 1) * ndim));
    const p2 = Array.from(vertices.slice(v1 * ndim, (v1 + 1) * ndim));

    // Clip to slice
    const clipped = clipSegmentToSlice(p1, p2, slicePosition, tolerance, displayDims);
    if (!clipped.visible) continue;

    // Write 3D positions
    startPositions.set(clipped.p1, outIdx * 3);
    endPositions.set(clipped.p2, outIdx * 3);

    // Interpolate and write colors
    const c0 = colors ? Array.from(colors.slice(v0 * 3, (v0 + 1) * 3)) : [1, 1, 1];
    const c1 = colors ? Array.from(colors.slice(v1 * 3, (v1 + 1) * 3)) : [1, 1, 1];
    const startC = lerpVec3(c0, c1, clipped.t1);
    const endC = lerpVec3(c0, c1, clipped.t2);
    startColors.set(startC, outIdx * 3);
    endColors.set(endC, outIdx * 3);

    // Interpolate widths (PER-VERTEX in LoadedLinesData!)
    const w0 = widths[v0];  // v0 is VERTEX index
    const w1 = widths[v1];  // v1 is VERTEX index
    startWidths[outIdx] = lerp(w0, w1, clipped.t1);
    endWidths[outIdx] = lerp(w0, w1, clipped.t2);

    // Interpolate sharpness (default 1.0 if not present)
    const s0 = sharpness ? sharpness[v0] : 1.0;
    const s1 = sharpness ? sharpness[v1] : 1.0;
    startSharpness[outIdx] = lerp(s0, s1, clipped.t1);
    endSharpness[outIdx] = lerp(s0, s1, clipped.t2);

    // Calculate 3D segment length
    segmentLengths[outIdx] = distance3D(clipped.p1, clipped.p2);

    // Track clipping for cap factor adjustment
    startClipped[outIdx] = clipped.t1 > 0 ? 1 : 0;
    endClipped[outIdx] = clipped.t2 < 1 ? 1 : 0;

    outIdx++;
  }

  // Trim arrays to actual size (after clipping, may have fewer segments)
  return {
    startPositions: startPositions.slice(0, outIdx * 3),
    endPositions: endPositions.slice(0, outIdx * 3),
    startColors: startColors.slice(0, outIdx * 3),
    endColors: endColors.slice(0, outIdx * 3),
    startWidths: startWidths.slice(0, outIdx),
    endWidths: endWidths.slice(0, outIdx),
    startSharpness: startSharpness.slice(0, outIdx),
    endSharpness: endSharpness.slice(0, outIdx),
    segmentLengths: segmentLengths.slice(0, outIdx),
    startClipped: startClipped.slice(0, outIdx),
    endClipped: endClipped.slice(0, outIdx),
    segmentCount: outIdx,
  };
}
```

### 2.4 Integration into Loaders

**File**: `packages/luxar-viewer/src/data/point-spatial-index-loader.ts` (MODIFY)

```typescript
import { PointsDataAccumulator } from './data-accumulator';
import type { DataLoader, PointsData, ViewState } from './data-loader-types';

export class PointSpatialIndexLoader implements DataLoader {
  private accumulator: PointsDataAccumulator;

  constructor(...) {
    // Initialize accumulator with metadata from spatial index
    const initialCapacity = Math.min(
      this.node.attrs.n_points ? this.node.attrs.n_points * 0.1 : 8192,
      8192
    );
    this.accumulator = new PointsDataAccumulator(
      initialCapacity,
      this.chunkIndex?.metadata.ndim || 3,
      this.node.attrs.n_points || 0
    );

    // Update metadata
    this.accumulator.updateMetadata({
      usedSpatialIndex: !!this.chunkIndex,
    });
  }

  /**
   * CORRECT METHOD NAME (matches DataLoader interface)
   */
  async loadPoints(viewState: ViewState): Promise<PointsData> {
    // ... spatial query ...
    const totalPoints = ranges.reduce((sum, r) => sum + r.count, 0);

    // Ensure capacity (rarely triggers reallocation)
    this.accumulator.ensureCapacity(totalPoints);

    // Fill accumulator
    let offset = 0;
    for (const range of ranges) {
      const chunkData = await this.loadChunkRange(range);
      this.accumulator.fill(offset, chunkData);
      offset += range.count;
    }

    // Return with CORRECT metadata structure
    return this.accumulator.getData(totalPoints);
  }

  /**
   * CORRECT METHOD NAME (matches DataLoader interface)
   */
  async updateView(viewState: ViewState): Promise<PointsData> {
    // For now, delegate to loadPoints
    // Future: Implement incremental updates
    return this.loadPoints(viewState);
  }

  dispose(): void {
    this.accumulator.dispose();
    // ... other cleanup ...
  }
}
```

**Similar pattern for LinesSpatialIndexLoader and GSplatsSpatialIndexLoader with CORRECTED method names:**
- `loadLines()` (not `loadPoints`)
- `loadGSplats()` (not `loadPoints`)

### 2.5 GSplats Processing: processGSplats() and packCholeskyForShader()

**CRITICAL**: GSplats require a multi-stage transformation similar to Lines:
1. `LoadedGSplatsData` (nD raw data from loader)
2. `ProcessedGSplatsData` (3D data via `processGSplats()`)
3. `PackedGSplatsData` (packed for GPU via `packCholeskyForShader()`)

**File**: `packages/luxar-viewer/src/data/gsplats-processor.ts` (EXISTS)

#### ProcessedGSplatsData Interface (from types/gsplats.ts)

```typescript
/**
 * Processed gsplats data ready for GPU rendering.
 * After nD → 3D slicing:
 * - Centers are in 3D display space
 * - Cholesky factors are 3D (6 elements per splat)
 * - Amplitudes are attenuated based on distance to hyperplane
 */
export interface ProcessedGSplatsData {
  /** Splat centers in 3D display space (M * 3) */
  centers3D: Float32Array;

  /** Attenuated amplitudes (M,) */
  amplitudes: Float32Array;

  /** 3D Cholesky factors (M * 6), packed as [L00, L10, L11, L20, L21, L22] */
  choleskyFactors3D: Float32Array;

  /** Splat colors RGB (M * 3) */
  colors: Float32Array;

  /** Splat sharpness values (M,) */
  sharpness: Float32Array;

  /** Number of visible splats after nD clipping */
  splatCount: number;
}
```

#### PackedGSplatsData Interface (for GPU)

```typescript
/**
 * Packed GSplats data ready for GPU upload.
 * Cholesky factors are split into 3 vec2 attributes for efficient shader access.
 */
export interface PackedGSplatsData {
  /** Splat centers in 3D (M * 3) */
  centers3D: Float32Array;

  /** Attenuated amplitudes (M,) */
  amplitudes: Float32Array;

  /** Packed Cholesky [L00, L10] (M * 2) */
  cholesky01: Float32Array;

  /** Packed Cholesky [L11, L20] (M * 2) */
  cholesky23: Float32Array;

  /** Packed Cholesky [L21, L22] (M * 2) */
  cholesky45: Float32Array;

  /** Splat colors RGB (M * 3) */
  colors: Float32Array;

  /** Number of splats */
  splatCount: number;
}
```

#### processGSplats() Function

Transforms nD LoadedGSplatsData to 3D ProcessedGSplatsData using **Mahalanobis distance**
for attenuation (matching actual implementation in `gsplats-processor.ts`).

**Key Algorithm Features:**
1. **Two-pass approach**: First counts visible splats, then extracts data (efficient memory allocation)
2. **Mahalanobis distance**: Uses Cholesky factors in hidden dimensions for ellipsoid-aware attenuation
3. **Visibility filtering**: Splats below `minAmplitude` threshold are excluded
4. **Standard Gaussian falloff**: Attenuation uses `exp(-0.5 * mahal^2)`

```typescript
/**
 * Compute packed index for Cholesky element L[row, col].
 * Packed lower-triangular: [L00, L10, L11, L20, L21, L22, ...]
 */
function packedIndex(row: number, col: number): number {
  return (row * (row + 1)) / 2 + col;
}

/**
 * Extract a submatrix from packed lower-triangular Cholesky factors.
 */
function extractCholeskySubmatrix(
  packed: Float32Array,
  offset: number,
  ndim: number,
  keepDims: number[],  // Sorted ascending!
  output: Float32Array,
  outputOffset: number
): void {
  const subNdim = keepDims.length;
  let outIdx = outputOffset;

  for (let subRow = 0; subRow < subNdim; subRow++) {
    const origRow = keepDims[subRow];
    for (let subCol = 0; subCol <= subRow; subCol++) {
      const origCol = keepDims[subCol];
      output[outIdx++] = packed[offset + packedIndex(origRow, origCol)];
    }
  }
}

/**
 * Compute Mahalanobis distance using forward substitution.
 * Given L (lower-triangular Cholesky), solve L·y = d, then ||y|| is Mahalanobis distance.
 */
function mahalanobisDistance(
  diff: number[],
  packedL: Float32Array,
  offset: number,
  ndim: number
): number {
  // Forward substitution: solve L · y = diff
  const y = new Array(ndim);

  for (let i = 0; i < ndim; i++) {
    let val = diff[i];
    for (let j = 0; j < i; j++) {
      val -= packedL[offset + packedIndex(i, j)] * y[j];
    }
    const diag = packedL[offset + packedIndex(i, i)];
    y[i] = diag > 1e-10 ? val / diag : 0;
  }

  // Compute ||y||
  let sumSq = 0;
  for (let i = 0; i < ndim; i++) {
    sumSq += y[i] * y[i];
  }
  return Math.sqrt(sumSq);
}

/**
 * Process nD GSplats data to 3D for rendering.
 *
 * Uses Mahalanobis distance in hidden dimensions for proper ellipsoid-aware
 * attenuation. This matches the actual implementation in gsplats-processor.ts.
 *
 * @param loaded - Raw nD data from GSplatsSpatialIndexLoader
 * @param viewState - Current view state with displayDims and slicePosition
 * @returns Processed 3D data ready for Cholesky packing
 */
export function processGSplats(
  loaded: LoadedGSplatsData,
  viewState: GSplatsViewState
): ProcessedGSplatsData {
  const { displayDims, slicePosition } = viewState;
  const ndim = loaded.ndim;
  const splatCount = loaded.splatCount;

  // Compute hidden dimensions (all dims not in displayDims)
  const hiddenDims = [];
  for (let d = 0; d < ndim; d++) {
    if (!displayDims.includes(d)) {
      hiddenDims.push(d);
    }
  }

  // Sort dimensions for consistent submatrix extraction (CRITICAL!)
  const sortedDisplayDims = [...displayDims].sort((a, b) => a - b);
  const sortedHiddenDims = [...hiddenDims].sort((a, b) => a - b);

  // Compute packed sizes
  const fullPackedSize = (ndim * (ndim + 1)) / 2;
  const display3DPackedSize = 6;  // 3D Cholesky has 6 elements
  const hiddenPackedSize = (sortedHiddenDims.length * (sortedHiddenDims.length + 1)) / 2;

  // Minimum amplitude threshold (splats below this are invisible)
  const minAmplitude = 1e-6;

  // FIRST PASS: Count visible splats
  let visibleCount = 0;
  const visibleIndices: number[] = [];

  for (let i = 0; i < splatCount; i++) {
    // Compute attenuation in hidden dimensions using Mahalanobis distance
    let attenuation = 1.0;
    if (sortedHiddenDims.length > 0) {
      const centerOffset = i * ndim;
      const diff = sortedHiddenDims.map(
        (d) => slicePosition[d] - loaded.centers[centerOffset + d]
      );

      // Extract hidden Cholesky submatrix
      const hiddenCholesky = new Float32Array(hiddenPackedSize);
      extractCholeskySubmatrix(
        loaded.choleskyFactors,
        i * fullPackedSize,
        ndim,
        sortedHiddenDims,
        hiddenCholesky,
        0
      );

      // Compute Mahalanobis distance in hidden dims
      const mahalDist = mahalanobisDistance(diff, hiddenCholesky, 0, sortedHiddenDims.length);

      // Standard Gaussian attenuation: exp(-½ · mahal²)
      attenuation = Math.exp(-0.5 * mahalDist * mahalDist);
    }

    const attenuatedAmplitude = loaded.amplitudes[i] * attenuation;

    if (attenuatedAmplitude >= minAmplitude) {
      visibleIndices.push(i);
      visibleCount++;
    }
  }

  // Allocate output arrays for visible splats only
  const centers3D = new Float32Array(visibleCount * 3);
  const choleskyFactors3D = new Float32Array(visibleCount * display3DPackedSize);
  const amplitudes = new Float32Array(visibleCount);
  const colors = new Float32Array(visibleCount * 3);

  // SECOND PASS: Extract visible splat data
  for (let outIdx = 0; outIdx < visibleCount; outIdx++) {
    const srcIdx = visibleIndices[outIdx];
    const srcCenterOffset = srcIdx * ndim;
    const srcCholeskyOffset = srcIdx * fullPackedSize;

    // Extract 3D center using SORTED display dimensions (must match Cholesky order!)
    const dstCenterOffset = outIdx * 3;
    for (let d = 0; d < 3 && d < sortedDisplayDims.length; d++) {
      centers3D[dstCenterOffset + d] = loaded.centers[srcCenterOffset + sortedDisplayDims[d]];
    }

    // Extract 3D Cholesky submatrix
    extractCholeskySubmatrix(
      loaded.choleskyFactors,
      srcCholeskyOffset,
      ndim,
      sortedDisplayDims,
      choleskyFactors3D,
      outIdx * display3DPackedSize
    );

    // Compute attenuated amplitude (recalculate for visible splats)
    let attenuation = 1.0;
    if (sortedHiddenDims.length > 0) {
      const diff = sortedHiddenDims.map(
        (d) => slicePosition[d] - loaded.centers[srcCenterOffset + d]
      );
      const hiddenCholesky = new Float32Array(hiddenPackedSize);
      extractCholeskySubmatrix(
        loaded.choleskyFactors,
        srcCholeskyOffset,
        ndim,
        sortedHiddenDims,
        hiddenCholesky,
        0
      );
      const mahalDist = mahalanobisDistance(diff, hiddenCholesky, 0, sortedHiddenDims.length);
      attenuation = Math.exp(-0.5 * mahalDist * mahalDist);
    }

    amplitudes[outIdx] = loaded.amplitudes[srcIdx] * attenuation;

    // Copy colors (default to white if not present)
    const srcColorOffset = srcIdx * 3;
    const dstColorOffset = outIdx * 3;
    if (loaded.colors) {
      colors[dstColorOffset] = loaded.colors[srcColorOffset];
      colors[dstColorOffset + 1] = loaded.colors[srcColorOffset + 1];
      colors[dstColorOffset + 2] = loaded.colors[srcColorOffset + 2];
    } else {
      colors[dstColorOffset] = 1.0;
      colors[dstColorOffset + 1] = 1.0;
      colors[dstColorOffset + 2] = 1.0;
    }
  }

  return {
    centers3D,
    amplitudes,
    choleskyFactors3D,
    colors,
    splatCount: visibleCount,  // NOTE: May be less than input splatCount!
  };
}
```

#### packCholeskyForShader() Function

Packs 3D Cholesky factors into 3 vec2 attributes for efficient shader access:

```typescript
/**
 * Pack 3D Cholesky factors into 3 vec2 attributes for GPU.
 *
 * Input: choleskyFactors3D[M * 6] = [L00, L10, L11, L20, L21, L22] per splat
 * Output:
 *   - cholesky01[M * 2] = [L00, L10] per splat
 *   - cholesky23[M * 2] = [L11, L20] per splat
 *   - cholesky45[M * 2] = [L21, L22] per splat
 *
 * In the shader, these reconstruct the 3x3 lower triangular matrix:
 *   L = [[L00, 0,   0  ],
 *        [L10, L11, 0  ],
 *        [L20, L21, L22]]
 */
export function packCholeskyForShader(
  choleskyFactors3D: Float32Array,
  splatCount: number
): { cholesky01: Float32Array; cholesky23: Float32Array; cholesky45: Float32Array } {
  const cholesky01 = new Float32Array(splatCount * 2);
  const cholesky23 = new Float32Array(splatCount * 2);
  const cholesky45 = new Float32Array(splatCount * 2);

  for (let i = 0; i < splatCount; i++) {
    const srcOffset = i * 6;
    const dstOffset = i * 2;

    // [L00, L10]
    cholesky01[dstOffset + 0] = choleskyFactors3D[srcOffset + 0];  // L00
    cholesky01[dstOffset + 1] = choleskyFactors3D[srcOffset + 1];  // L10

    // [L11, L20]
    cholesky23[dstOffset + 0] = choleskyFactors3D[srcOffset + 2];  // L11
    cholesky23[dstOffset + 1] = choleskyFactors3D[srcOffset + 3];  // L20

    // [L21, L22]
    cholesky45[dstOffset + 0] = choleskyFactors3D[srcOffset + 4];  // L21
    cholesky45[dstOffset + 1] = choleskyFactors3D[srcOffset + 5];  // L22
  }

  return { cholesky01, cholesky23, cholesky45 };
}
```

#### Complete GSplats Data Flow

```
GSplatsSpatialIndexLoader
         │
         ▼ loadGSplats()
   LoadedGSplatsData
   (nD: centers, amplitudes, choleskyFactors)
         │
         ▼ processGSplats()
  ProcessedGSplatsData
  (3D: centers3D, choleskyFactors3D, attenuated amplitudes)
         │
         ▼ packCholeskyForShader()
    PackedGSplatsData
    (GPU-ready: centers3D, cholesky01/23/45)
         │
         ▼ GPUBufferPool.updateGSplatsGeometry()
       GPU Buffers
```

---

## 3. Component 2: Web Workers with ArrayDecoder

### 3.1 Worker-Main Thread Division of Labor

**CRITICAL DESIGN DECISION**: ArrayDecoder stays on main thread!

> ⚠️ **KNOWN OPTIMIZATION OPPORTUNITY**
>
> Having ArrayDecoder on the main thread means decoding (Blosc decompression,
> dequantization, LUT lookups) happens on the main thread. This could cause
> frame drops during heavy loading.
>
> **Why it's here for now:**
> - `zarrita.js` doesn't have built-in web worker support
> - ArrayDecoder needs `zarr.Array` objects which aren't easily serializable
> - Network I/O is usually the bottleneck, not decoding
>
> **Future optimization (if profiling shows it's needed):**
> 1. Fetch raw chunk bytes on main thread
> 2. Transfer bytes to worker via `postMessage` (transferable)
> 3. Use separate decompression library in worker (e.g., `numcodecs.js` for Blosc)
> 4. Handle dequantization/LUT in worker with serialized metadata
>
> **Profile first** after Phase 1 to see if this is actually a bottleneck.

The `ArrayDecoder.decode()` method signature is:
```typescript
async decode(
  array: zarr.Array<...>,     // Takes zarr.Array, NOT raw bytes!
  attrs: ArrayMetadata,
  expectedElements: number,
  zarrRootLoc?: zarr.Location<...>  // Needed for array_ref resolution
): Promise<Float32Array>
```

This means ArrayDecoder needs:
1. Access to the zarr store (for fetching chunks)
2. Access to zarrRootLoc (for resolving `array_ref` deduplication)

**Therefore, the worker handles ONLY:**
- WASM spatial index queries (chunk bounding box tests)
- WASM nD visibility computation (hyperbolic distance filtering)
- WASM attribute interleaving (optional)

**Main thread handles:**
- Zarr chunk fetching (async IO, non-blocking)
- ArrayDecoder decoding (handles all 7+ encoding types)
- Accumulator buffer management

This is actually optimal because:
- Zarr fetching is async IO (doesn't block main thread)
- Spatial queries are the CPU-intensive bottleneck (perfect for WASM worker)
- ArrayDecoder needs zarr context that's complex to serialize

### 3.2 Worker Architecture

**File**: `packages/luxar-viewer/src/workers/data-worker.ts` (NEW)

```typescript
/**
 * Data processing worker for CPU-intensive WASM tasks.
 *
 * NOTE: ArrayDecoder stays on main thread (needs zarr.Array objects).
 * Worker handles ONLY spatial queries and nD visibility computation.
 *
 * Responsibilities:
 * - WASM spatial index queries (chunk bounding box tests)
 * - WASM nD visibility computation (hyperbolic distance filtering)
 * - WASM attribute interleaving (optional optimization)
 *
 * NOT handled here (stays on main thread):
 * - Zarr chunk fetching (async IO)
 * - ArrayDecoder decoding (needs zarr context)
 */

import { expose } from 'comlink';
import { initWasm, type WasmModule } from './wasm-bindings';

// Worker-side persistent state
let wasmModule: WasmModule | null = null;

// Persistent buffers (avoid per-task allocations)
let chunkBoundsCache: Float32Array | null = null;
let visibilityMaskBuffer: Uint8Array | null = null;

/**
 * Initialize worker (called once at startup)
 * FAILS if WASM unavailable (no fallback - modern browsers only)
 */
async function initialize(): Promise<void> {
  console.log('[DataWorker] Initializing...');

  // Load WASM module (REQUIRED - no fallback)
  try {
    wasmModule = await initWasm();
  } catch (error) {
    console.error('[DataWorker] WASM initialization FAILED:', error);
    throw new Error(
      'WASM unavailable. Luxar requires WebAssembly support. ' +
      'Please use a modern browser (Chrome 57+, Firefox 52+, Safari 11+).'
    );
  }

  // Pre-allocate buffers
  chunkBoundsCache = new Float32Array(1000 * 10 * 2); // 1000 chunks, 10D
  visibilityMaskBuffer = new Uint8Array(100000); // 100K points

  console.log('[DataWorker] Ready (WASM enabled)');
}

/**
 * Task 1: Query spatial index (generic for all types)
 */
async function querySpatialIndex(params: {
  chunkBounds: Float32Array;
  slicePosition: Float32Array;
  tolerance: Float32Array;
  numChunks: number;
  ndim: number;
}): Promise<Uint32Array> {
  const { chunkBounds, slicePosition, tolerance, numChunks, ndim } = params;

  const matchingChunks = new Uint32Array(numChunks); // Max size

  // Call WASM (REQUIRED - no fallback)
  const count = wasmModule!.query_chunks_for_view(
    chunkBounds,
    slicePosition,
    tolerance,
    ndim,
    numChunks,
    matchingChunks
  );

  return matchingChunks.subarray(0, count);
}

/**
 * Task 2: Compute nD visibility for Points
 *
 * NOTE: Decoding happens on main thread via ArrayDecoder.
 * This task receives ALREADY DECODED Float32Array data.
 */
async function computeNDVisibilityPoints(params: {
  positions: Float32Array;
  radii: Float32Array;
  slicePosition: Float32Array;
  tolerance: Float32Array;
  ndim: number;
  numPoints: number;
}): Promise<{ visibilityMask: Uint8Array; visibleCount: number }> {
  const { positions, radii, slicePosition, tolerance, ndim, numPoints } = params;

  // Ensure buffer capacity
  if (visibilityMaskBuffer!.length < numPoints) {
    visibilityMaskBuffer = new Uint8Array(Math.ceil(numPoints * 1.5));
  }

  // Call WASM (REQUIRED - no fallback)
  const visibleCount = wasmModule!.compute_nd_visibility_points(
    positions,
    radii,
    slicePosition,
    tolerance,
    ndim,
    numPoints,
    visibilityMaskBuffer!
  );

  return {
    visibilityMask: visibilityMaskBuffer!.subarray(0, numPoints),
    visibleCount,
  };
}

/**
 * Task 4: Compute nD visibility for Lines (check segment endpoints)
 */
async function computeNDVisibilityLines(params: {
  vertices: Float32Array;
  segments: Uint32Array; // Pairs of vertex indices
  radii: Float32Array;
  slicePosition: Float32Array;
  tolerance: Float32Array;
  ndim: number;
  numSegments: number;
}): Promise<{ visibilityMask: Uint8Array; visibleCount: number }> {
  // WASM implementation checks if EITHER endpoint is visible
  const visibleCount = wasmModule!.compute_nd_visibility_lines(
    params.vertices,
    params.segments,
    params.radii,
    params.slicePosition,
    params.tolerance,
    params.ndim,
    params.numSegments,
    visibilityMaskBuffer!
  );

  return {
    visibilityMask: visibilityMaskBuffer!.subarray(0, params.numSegments),
    visibleCount,
  };
}

/**
 * Task 5: Compute nD visibility for GSplats (check center + ellipsoid extent)
 */
async function computeNDVisibilityGSplats(params: {
  centers: Float32Array;
  choleskyFactors: Float32Array; // CORRECT: camelCase!
  slicePosition: Float32Array;
  tolerance: Float32Array;
  ndim: number;
  numSplats: number;
}): Promise<{ visibilityMask: Uint8Array; visibleCount: number }> {
  // WASM computes ellipsoid extent from Cholesky factors
  const visibleCount = wasmModule!.compute_nd_visibility_gsplats(
    params.centers,
    params.choleskyFactors, // CORRECT: camelCase!
    params.slicePosition,
    params.tolerance,
    params.ndim,
    params.numSplats,
    visibilityMaskBuffer!
  );

  return {
    visibilityMask: visibilityMaskBuffer!.subarray(0, params.numSplats),
    visibleCount,
  };
}

// Expose worker API (NO decoding - ArrayDecoder stays on main thread)
const workerAPI = {
  initialize,
  querySpatialIndex,
  computeNDVisibilityPoints,
  computeNDVisibilityLines,
  computeNDVisibilityGSplats,
};

expose(workerAPI);

export type DataWorkerAPI = typeof workerAPI;
```

### 3.3 Worker Pool Manager

**File**: `packages/luxar-viewer/src/workers/worker-pool.ts` (NEW)

```typescript
/**
 * Worker pool manager - single worker for now, designed for future multi-worker
 */

import { wrap, Remote } from 'comlink';
import type { DataWorkerAPI } from './data-worker';

class WorkerPool {
  private worker: Worker | null = null;
  private workerAPI: Remote<DataWorkerAPI> | null = null;
  private initPromise: Promise<void> | null = null;

  async initialize(): Promise<void> {
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      // Create worker
      this.worker = new Worker(
        new URL('./data-worker.ts', import.meta.url),
        { type: 'module' }
      );

      // Wrap with Comlink
      this.workerAPI = wrap<DataWorkerAPI>(this.worker);

      // Initialize worker (loads WASM only - ArrayDecoder stays on main thread)
      // FAILS if WASM unavailable (no fallback)
      try {
        await this.workerAPI.initialize();
      } catch (error) {
        console.error('[WorkerPool] Worker initialization FAILED:', error);
        throw new Error(
          'Failed to initialize data worker. ' +
          'Luxar requires WebAssembly and Web Workers support.'
        );
      }

      console.log('[WorkerPool] Worker ready');
    })();

    return this.initPromise;
  }

  async getWorker(): Promise<Remote<DataWorkerAPI>> {
    await this.initialize();
    return this.workerAPI!;
  }

  dispose(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
      this.workerAPI = null;
      this.initPromise = null;
    }
  }
}

export const workerPool = new WorkerPool();
```

---

## 4. Component 3: WASM Acceleration (All Types)

### 4.1 Rust WASM Module

**File**: `packages/luxar-viewer/src/workers/wasm/src/lib.rs` (NEW)

```rust
use wasm_bindgen::prelude::*;

/// Query chunks for any primitive type (generic)
#[wasm_bindgen]
pub fn query_chunks_for_view(
    chunk_bounds: &[f32],
    slice_position: &[f32],
    tolerance: &[f32],
    ndim: usize,
    num_chunks: usize,
    output: &mut [u32],
) -> u32 {
    let mut match_count = 0;

    for chunk_idx in 0..num_chunks {
        let mut intersects = true;

        for dim in 0..ndim {
            let bounds_offset = (chunk_idx * ndim * 2) + (dim * 2);
            let chunk_min = chunk_bounds[bounds_offset];
            let chunk_max = chunk_bounds[bounds_offset + 1];

            let query_min = slice_position[dim] - tolerance[dim];
            let query_max = slice_position[dim] + tolerance[dim];

            if chunk_max < query_min || chunk_min > query_max {
                intersects = false;
                break;
            }
        }

        if intersects {
            output[match_count as usize] = chunk_idx as u32;
            match_count += 1;
        }
    }

    match_count
}

/// Compute nD visibility for Points
#[wasm_bindgen]
pub fn compute_nd_visibility_points(
    positions: &[f32],
    radii: &[f32],
    slice_position: &[f32],
    tolerance: &[f32],
    ndim: usize,
    num_points: usize,
    output_mask: &mut [u8],
) -> u32 {
    let mut visible_count = 0;

    for pt_idx in 0..num_points {
        let pt_offset = pt_idx * ndim;
        let radius = radii[pt_idx];

        let mut dist_sq = 0.0_f32;
        for dim in 0..ndim {
            let delta = positions[pt_offset + dim] - slice_position[dim];
            let effective_tolerance = tolerance[dim] + radius;
            let normalized = delta / effective_tolerance;
            dist_sq += normalized * normalized;
        }

        let visible = dist_sq <= 1.0;
        output_mask[pt_idx] = if visible { 1 } else { 0 };
        if visible {
            visible_count += 1;
        }
    }

    visible_count
}

/// Compute nD visibility for Lines (check endpoints)
#[wasm_bindgen]
pub fn compute_nd_visibility_lines(
    vertices: &[f32],
    segments: &[u32], // Pairs: [v0, v1, v0, v1, ...]
    radii: &[f32], // Per-segment width
    slice_position: &[f32],
    tolerance: &[f32],
    ndim: usize,
    num_segments: usize,
    output_mask: &mut [u8],
) -> u32 {
    let mut visible_count = 0;

    for seg_idx in 0..num_segments {
        let v0_idx = segments[seg_idx * 2] as usize;
        let v1_idx = segments[seg_idx * 2 + 1] as usize;
        let width = radii[seg_idx];

        // Check if EITHER endpoint is visible
        let mut v0_visible = true;
        let mut v1_visible = true;

        // Check vertex 0
        let mut dist_sq_v0 = 0.0_f32;
        for dim in 0..ndim {
            let delta = vertices[v0_idx * ndim + dim] - slice_position[dim];
            let effective_tolerance = tolerance[dim] + width;
            let normalized = delta / effective_tolerance;
            dist_sq_v0 += normalized * normalized;
        }
        v0_visible = dist_sq_v0 <= 1.0;

        // Check vertex 1
        let mut dist_sq_v1 = 0.0_f32;
        for dim in 0..ndim {
            let delta = vertices[v1_idx * ndim + dim] - slice_position[dim];
            let effective_tolerance = tolerance[dim] + width;
            let normalized = delta / effective_tolerance;
            dist_sq_v1 += normalized * normalized;
        }
        v1_visible = dist_sq_v1 <= 1.0;

        // Segment visible if EITHER endpoint visible
        let visible = v0_visible || v1_visible;
        output_mask[seg_idx] = if visible { 1 } else { 0 };
        if visible {
            visible_count += 1;
        }
    }

    visible_count
}

/// Compute nD visibility for GSplats (center + ellipsoid extent)
#[wasm_bindgen]
pub fn compute_nd_visibility_gsplats(
    centers: &[f32],
    cholesky_factors: &[f32],
    slice_position: &[f32],
    tolerance: &[f32],
    ndim: usize,
    num_splats: usize,
    output_mask: &mut [u8],
) -> u32 {
    let cholesky_size = (ndim * (ndim + 1)) / 2;
    let mut visible_count = 0;

    for splat_idx in 0..num_splats {
        let center_offset = splat_idx * ndim;
        let cholesky_offset = splat_idx * cholesky_size;

        // Compute maximum ellipsoid extent from Cholesky factors
        // (diagonal elements give axis scales)
        let mut max_extent = 0.0_f32;
        let mut chol_idx = 0;
        for dim in 0..ndim {
            // Diagonal element at position: dim + sum(0..dim)
            let diag_pos = cholesky_offset + chol_idx;
            let scale = cholesky_factors[diag_pos].abs();
            max_extent = max_extent.max(scale);
            chol_idx += ndim - dim; // Skip to next diagonal
        }

        // Check if center + max extent is within tolerance
        let mut dist_sq = 0.0_f32;
        for dim in 0..ndim {
            let delta = centers[center_offset + dim] - slice_position[dim];
            let effective_tolerance = tolerance[dim] + max_extent;
            let normalized = delta / effective_tolerance;
            dist_sq += normalized * normalized;
        }

        let visible = dist_sq <= 1.0;
        output_mask[splat_idx] = if visible { 1 } else { 0 };
        if visible {
            visible_count += 1;
        }
    }

    visible_count
}
```

### 4.2 Build Configuration

**File**: `packages/luxar-viewer/scripts/build-wasm.sh` (NEW)

```bash
#!/bin/bash
set -e

echo "Building Luxar WASM module..."

cd "$(dirname "$0")/../src/workers/wasm"

# Build with wasm-pack
wasm-pack build \
  --target web \
  --out-dir ../../../dist/wasm \
  --release

echo "WASM module built successfully!"
```

**Add to `package.json`:**
```json
{
  "scripts": {
    "build:wasm": "bash scripts/build-wasm.sh",
    "prebuild": "pnpm build:wasm",
    "dev": "vite"
  }
}
```

---

## 5. Component 4: GPU Buffer Pool (All Geometry Types)

### 5.1 Data Flow: LoadedLinesData → ProcessedLinesData → GPU

**CRITICAL**: Lines require a transformation step before GPU upload:

```
LoadedLinesData (from loader)     ProcessedLinesData (for GPU)
├─ vertices: nD per-vertex    →   ├─ startPositions: 3D per-segment
├─ segments: index pairs      →   ├─ endPositions: 3D per-segment
├─ widths: PER-VERTEX (N,)    →   ├─ startWidths: per-segment
├─ colors: PER-VERTEX         →   ├─ endWidths: per-segment
└─ sharpness: PER-VERTEX      →   ├─ startColors/endColors: per-segment
                                   ├─ startSharpness/endSharpness: per-segment
                                   └─ segmentLengths, startClipped, endClipped
```

The transformation (`buildInstanceBuffers()`) performs:
1. nD clipping to slice
2. Per-vertex → per-segment attribute conversion
3. 3D extraction from nD space
4. Attribute interpolation for clipped endpoints

### 5.2 Multi-Type GPU Buffer Pool

**File**: `packages/luxar-viewer/src/rendering/gpu-buffer-pool.ts` (NEW)

```typescript
/**
 * GPU buffer pool for all three geometry types: Points, Lines, GSplats
 *
 * IMPORTANT: Lines use ProcessedLinesData (per-segment), NOT LoadedLinesData (per-vertex)!
 */

import * as THREE from 'three';
import { log, Modules } from '../utils/log';
import type { PointsData } from '../data/data-loader-types';
import type { ProcessedLinesData } from '../types/lines';  // NOT LoadedLinesData!
import type { ProcessedGSplatsData } from '../types/gsplats';  // After nD→3D processing

/**
 * Packed GSplats data ready for GPU upload.
 * Cholesky factors are split into 3 vec2 attributes for efficient shader access.
 *
 * NOTE: This is generated by calling packCholeskyForShader() on ProcessedGSplatsData.
 */
export interface PackedGSplatsData {
  centers3D: Float32Array;      // M * 3
  amplitudes: Float32Array;     // M
  cholesky01: Float32Array;     // M * 2 [L00, L10]
  cholesky23: Float32Array;     // M * 2 [L11, L20]
  cholesky45: Float32Array;     // M * 2 [L21, L22]
  colors: Float32Array;         // M * 3 (RGB)
  splatCount: number;
}

interface PooledBuffer {
  geometry: THREE.BufferGeometry;
  capacity: number;
  type: 'points' | 'lines' | 'gsplats';
  inUse: boolean;
  lastUsedFrame: number;
}

export class GPUBufferPool {
  private pointBuffers = new Map<string, PooledBuffer[]>();
  private lineBuffers = new Map<string, PooledBuffer[]>();
  private gsplatBuffers = new Map<string, PooledBuffer[]>();

  private activeBuffers = new Map<string, PooledBuffer>();
  private frameCount = 0;

  private maxPoolSize = 20;
  private stats = {
    allocations: 0,
    reuses: 0,
    evictions: 0,
    capacityGrowths: 0,
  };

  /**
   * Acquire geometry for Points
   */
  acquirePointsGeometry(nodeId: string, pointCount: number): THREE.BufferGeometry {
    this.frameCount++;

    let pooled = this.activeBuffers.get(nodeId);

    if (pooled && pooled.type === 'points') {
      if (pooled.capacity >= pointCount) {
        pooled.lastUsedFrame = this.frameCount;
        this.stats.reuses++;
        return pooled.geometry;
      } else {
        this.growPointsBuffer(pooled, pointCount);
        this.stats.capacityGrowths++;
        return pooled.geometry;
      }
    }

    // Try to find in pool
    for (const [_bucket, buffers] of this.pointBuffers.entries()) {
      for (let i = 0; i < buffers.length; i++) {
        const candidate = buffers[i];
        if (!candidate.inUse && candidate.capacity >= pointCount) {
          buffers.splice(i, 1);
          candidate.inUse = true;
          candidate.lastUsedFrame = this.frameCount;
          this.activeBuffers.set(nodeId, candidate);
          this.stats.reuses++;
          return candidate.geometry;
        }
      }
    }

    // Allocate new
    const newCapacity = Math.ceil(pointCount * 1.5);
    const geometry = this.createPointsGeometry(newCapacity);

    pooled = {
      geometry,
      capacity: newCapacity,
      type: 'points',
      inUse: true,
      lastUsedFrame: this.frameCount,
    };

    this.activeBuffers.set(nodeId, pooled);
    this.stats.allocations++;

    return geometry;
  }

  /**
   * Acquire geometry for Lines (ProcessedLinesData - per-segment attributes)
   *
   * NOTE: Lines geometry only needs segmentCount since all attributes are per-segment
   */
  acquireLinesGeometry(
    nodeId: string,
    segmentCount: number
  ): THREE.InstancedBufferGeometry {
    this.frameCount++;

    let pooled = this.activeBuffers.get(nodeId);

    if (pooled && pooled.type === 'lines') {
      if (pooled.capacity >= segmentCount) {
        pooled.lastUsedFrame = this.frameCount;
        this.stats.reuses++;
        return pooled.geometry as THREE.InstancedBufferGeometry;
      } else {
        this.growLinesBuffer(pooled, segmentCount);
        this.stats.capacityGrowths++;
        return pooled.geometry as THREE.InstancedBufferGeometry;
      }
    }

    // Try pool
    for (const [_bucket, buffers] of this.lineBuffers.entries()) {
      for (let i = 0; i < buffers.length; i++) {
        const candidate = buffers[i];
        if (!candidate.inUse && candidate.capacity >= segmentCount) {
          buffers.splice(i, 1);
          candidate.inUse = true;
          candidate.lastUsedFrame = this.frameCount;
          this.activeBuffers.set(nodeId, candidate);
          this.stats.reuses++;
          return candidate.geometry as THREE.InstancedBufferGeometry;
        }
      }
    }

    // Allocate new (only segment capacity needed - all attributes are per-segment)
    const newSegmentCap = Math.ceil(segmentCount * 1.5);
    const geometry = this.createLinesGeometry(newSegmentCap);

    pooled = {
      geometry,
      capacity: newSegmentCap,
      type: 'lines',
      inUse: true,
      lastUsedFrame: this.frameCount,
    };

    this.activeBuffers.set(nodeId, pooled);
    this.stats.allocations++;

    return geometry;
  }

  /**
   * Acquire geometry for GSplats
   *
   * NOTE: GSplats geometry is ALWAYS 3D after processing - ndim is NOT needed!
   * The nD → 3D conversion happens in processGSplats() BEFORE reaching GPUBufferPool.
   */
  acquireGSplatsGeometry(nodeId: string, splatCount: number): THREE.BufferGeometry {
    this.frameCount++;

    // Similar pattern to points...
    // (omitted for brevity)

    const newCapacity = Math.ceil(splatCount * 1.5);
    const geometry = this.createGSplatsGeometry(newCapacity);  // No ndim!

    const pooled = {
      geometry,
      capacity: newCapacity,
      type: 'gsplats' as const,
      inUse: true,
      lastUsedFrame: this.frameCount,
    };

    this.activeBuffers.set(nodeId, pooled);
    this.stats.allocations++;

    return geometry;
  }

  /**
   * Create Points geometry
   */
  private createPointsGeometry(capacity: number): THREE.BufferGeometry {
    const geometry = new THREE.BufferGeometry();

    geometry.setAttribute('position',
      new THREE.Float32BufferAttribute(new Float32Array(capacity * 3), 3));
    // Use Float32Array with 3 components (RGB) for HDR support
    // Matches scene-loader.ts which handles normalization based on actual data type
    geometry.setAttribute('color',
      new THREE.Float32BufferAttribute(new Float32Array(capacity * 3), 3));
    geometry.setAttribute('radius',
      new THREE.Float32BufferAttribute(new Float32Array(capacity), 1));
    geometry.setAttribute('sharpness',
      new THREE.Float32BufferAttribute(new Float32Array(capacity), 1));

    // Set all to dynamic usage
    for (const key in geometry.attributes) {
      geometry.attributes[key].setUsage(THREE.DynamicDrawUsage);
    }

    return geometry;
  }

  /**
   * Create Lines geometry for ProcessedLinesData (per-segment instanced attributes)
   *
   * Lines use InstancedBufferGeometry with per-segment attributes:
   * - aStartPos, aEndPos: 3D segment endpoints
   * - aStartColor, aEndColor: per-segment colors
   * - aStartWidth, aEndWidth: per-segment widths
   * - aStartSharpness, aEndSharpness: per-segment sharpness
   */
  private createLinesGeometry(segmentCapacity: number): THREE.InstancedBufferGeometry {
    const geometry = new THREE.InstancedBufferGeometry();

    // Base quad geometry (4 vertices, 2 triangles)
    const quadPositions = new Float32Array([
      -1, -1,  // bottom-left
       1, -1,  // bottom-right
      -1,  1,  // top-left
       1,  1,  // top-right
    ]);
    geometry.setAttribute('aQuadCorner',
      new THREE.Float32BufferAttribute(quadPositions, 2));
    geometry.setIndex([0, 1, 2, 2, 1, 3]);

    // Per-segment instanced attributes (ProcessedLinesData format)
    geometry.setAttribute('aStartPos',
      new THREE.InstancedBufferAttribute(new Float32Array(segmentCapacity * 3), 3));
    geometry.setAttribute('aEndPos',
      new THREE.InstancedBufferAttribute(new Float32Array(segmentCapacity * 3), 3));
    geometry.setAttribute('aStartColor',
      new THREE.InstancedBufferAttribute(new Float32Array(segmentCapacity * 3), 3));
    geometry.setAttribute('aEndColor',
      new THREE.InstancedBufferAttribute(new Float32Array(segmentCapacity * 3), 3));
    geometry.setAttribute('aStartWidth',
      new THREE.InstancedBufferAttribute(new Float32Array(segmentCapacity), 1));
    geometry.setAttribute('aEndWidth',
      new THREE.InstancedBufferAttribute(new Float32Array(segmentCapacity), 1));
    geometry.setAttribute('aStartSharpness',
      new THREE.InstancedBufferAttribute(new Float32Array(segmentCapacity), 1));
    geometry.setAttribute('aEndSharpness',
      new THREE.InstancedBufferAttribute(new Float32Array(segmentCapacity), 1));
    geometry.setAttribute('aSegmentLength',
      new THREE.InstancedBufferAttribute(new Float32Array(segmentCapacity), 1));
    // Use Float32Array for clipped flags (shader expects float, matching line-material.ts)
    geometry.setAttribute('aStartClipped',
      new THREE.InstancedBufferAttribute(new Float32Array(segmentCapacity), 1));
    geometry.setAttribute('aEndClipped',
      new THREE.InstancedBufferAttribute(new Float32Array(segmentCapacity), 1));

    // Set all instanced attributes to dynamic usage
    for (const key of ['aStartPos', 'aEndPos', 'aStartColor', 'aEndColor',
                       'aStartWidth', 'aEndWidth', 'aStartSharpness', 'aEndSharpness',
                       'aSegmentLength', 'aStartClipped', 'aEndClipped']) {
      geometry.getAttribute(key).setUsage(THREE.DynamicDrawUsage);
    }

    return geometry;
  }

  /**
   * Create GSplats geometry (instanced)
   *
   * CRITICAL: GSplats GPU geometry uses PROCESSED 3D data, not raw nD data!
   * - Centers are 3D (from ProcessedGSplatsData.centers3D)
   * - Cholesky factors are packed into 3 vec2 attributes (from packCholeskyForShader)
   *
   * The shader expects:
   * - aCenter: vec3 (3D splat center)
   * - aCholesky01: vec2 (L00, L10)
   * - aCholesky23: vec2 (L11, L20)
   * - aCholesky45: vec2 (L21, L22)
   *
   * ⚠️ ACTUAL VS PROPOSED PATTERN ⚠️
   * The ACTUAL implementation in scene-loader.ts uses:
   *   - createInstancedGSplatsMesh() / updateInstancedGSplatsMesh() from gsplat-material.ts
   *   - These functions manage their own InstancedBufferGeometry internally
   *
   * The GPUBufferPool pattern proposed here is an OPTIMIZATION that would:
   *   - Reuse geometry buffers across view updates
   *   - Reduce GPU memory allocations
   *   - Enable partial buffer updates
   *
   * During implementation, either:
   *   a) Modify gsplat-material.ts to use GPUBufferPool, OR
   *   b) Keep current pattern and skip GPUBufferPool for GSplats
   */
  private createGSplatsGeometry(capacity: number): THREE.BufferGeometry {
    const geometry = new THREE.BufferGeometry();

    // Per-splat instance attributes (3D processed data!)
    // Centers are always 3D after nD → 3D projection
    geometry.setAttribute('aCenter',
      new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3));

    // Amplitudes (attenuated based on hyperplane distance)
    geometry.setAttribute('aAmplitude',
      new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1));

    // Packed 3D Cholesky factors (from packCholeskyForShader)
    // L = [[L00, 0, 0], [L10, L11, 0], [L20, L21, L22]]
    geometry.setAttribute('aCholesky01',
      new THREE.InstancedBufferAttribute(new Float32Array(capacity * 2), 2));  // [L00, L10]
    geometry.setAttribute('aCholesky23',
      new THREE.InstancedBufferAttribute(new Float32Array(capacity * 2), 2));  // [L11, L20]
    geometry.setAttribute('aCholesky45',
      new THREE.InstancedBufferAttribute(new Float32Array(capacity * 2), 2));  // [L21, L22]

    // Colors (RGB Float32)
    geometry.setAttribute('aColor',
      new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3));

    // Sharpness
    geometry.setAttribute('aSharpness',
      new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1));

    // Set dynamic usage for all attributes
    for (const key in geometry.attributes) {
      geometry.attributes[key].setUsage(THREE.DynamicDrawUsage);
    }

    return geometry;
  }

  /**
   * Grow buffers (similar to accumulator pattern)
   */
  private growPointsBuffer(pooled: PooledBuffer, neededCapacity: number): void {
    let newCapacity = pooled.capacity;
    while (newCapacity < neededCapacity) {
      newCapacity = Math.ceil(newCapacity * 1.5);
    }

    log.info(Modules.RENDERER,
      `Growing Points GPU buffer: ${pooled.capacity} → ${newCapacity} points`);

    pooled.geometry.dispose();
    pooled.geometry = this.createPointsGeometry(newCapacity);
    pooled.capacity = newCapacity;
  }

  private growLinesBuffer(pooled: PooledBuffer, segmentCount: number): void {
    const newSegmentCap = Math.ceil(segmentCount * 1.5);

    log.info(Modules.RENDERER,
      `Growing Lines GPU buffer: ${pooled.capacity} → ${newSegmentCap} segments`);

    pooled.geometry.dispose();
    pooled.geometry = this.createLinesGeometry(newSegmentCap);
    pooled.capacity = newSegmentCap;
  }

  /**
   * Release buffer back to pool
   */
  release(nodeId: string): void {
    const pooled = this.activeBuffers.get(nodeId);
    if (!pooled) return;

    this.activeBuffers.delete(nodeId);
    pooled.inUse = false;

    // Add to appropriate pool
    const bucket = this.getCapacityBucket(pooled.capacity);
    if (pooled.type === 'points') {
      if (!this.pointBuffers.has(bucket)) {
        this.pointBuffers.set(bucket, []);
      }
      this.pointBuffers.get(bucket)!.push(pooled);
    } else if (pooled.type === 'lines') {
      if (!this.lineBuffers.has(bucket)) {
        this.lineBuffers.set(bucket, []);
      }
      this.lineBuffers.get(bucket)!.push(pooled);
    } else if (pooled.type === 'gsplats') {
      if (!this.gsplatBuffers.has(bucket)) {
        this.gsplatBuffers.set(bucket, []);
      }
      this.gsplatBuffers.get(bucket)!.push(pooled);
    }

    this.maybeEvict();
  }

  /**
   * Update Points geometry with new data (partial updates)
   */
  updatePointsGeometry(
    geometry: THREE.BufferGeometry,
    pointsData: PointsData,
    previousCount?: number
  ): void {
    const { positions, colors, radii, sharpness, metadata } = pointsData;
    const count = metadata.loadedPoints;

    // Update position
    const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute;
    (posAttr.array as Float32Array).set(positions);
    if (previousCount !== undefined && count < previousCount) {
      posAttr.addUpdateRange(0, count * 3);
    } else {
      posAttr.needsUpdate = true;
    }

    // Update color (RGB Float32 for HDR support)
    const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute;
    if (colors) {
      // Handle both Float32Array (HDR) and Uint8/Uint16 (SDR) by converting to float
      // scene-loader.ts handles normalization for Uint types
      (colorAttr.array as Float32Array).set(colors);
      if (previousCount !== undefined && count < previousCount) {
        colorAttr.addUpdateRange(0, count * 3);  // RGB = 3 components
      } else {
        colorAttr.needsUpdate = true;
      }
    }

    // Similar for radius, sharpness...

    geometry.setDrawRange(0, count);
  }

  /**
   * Update Lines geometry with ProcessedLinesData (per-segment attributes)
   *
   * IMPORTANT: Takes ProcessedLinesData (from buildInstanceBuffers), NOT LoadedLinesData!
   * All attributes are per-segment (M segments), not per-vertex.
   */
  updateLinesGeometry(
    geometry: THREE.InstancedBufferGeometry,
    processedData: ProcessedLinesData,
    previousSegmentCount?: number
  ): void {
    const {
      startPositions, endPositions,
      startColors, endColors,
      startWidths, endWidths,
      startSharpness, endSharpness,
      segmentLengths,
      startClipped, endClipped,
      segmentCount
    } = processedData;

    // Update start positions
    const startPosAttr = geometry.getAttribute('aStartPos') as THREE.InstancedBufferAttribute;
    (startPosAttr.array as Float32Array).set(startPositions);
    startPosAttr.needsUpdate = true;

    // Update end positions
    const endPosAttr = geometry.getAttribute('aEndPos') as THREE.InstancedBufferAttribute;
    (endPosAttr.array as Float32Array).set(endPositions);
    endPosAttr.needsUpdate = true;

    // Update start colors
    const startColorAttr = geometry.getAttribute('aStartColor') as THREE.InstancedBufferAttribute;
    (startColorAttr.array as Float32Array).set(startColors);
    startColorAttr.needsUpdate = true;

    // Update end colors
    const endColorAttr = geometry.getAttribute('aEndColor') as THREE.InstancedBufferAttribute;
    (endColorAttr.array as Float32Array).set(endColors);
    endColorAttr.needsUpdate = true;

    // Update widths (per-segment start/end)
    const startWidthAttr = geometry.getAttribute('aStartWidth') as THREE.InstancedBufferAttribute;
    (startWidthAttr.array as Float32Array).set(startWidths);
    startWidthAttr.needsUpdate = true;

    const endWidthAttr = geometry.getAttribute('aEndWidth') as THREE.InstancedBufferAttribute;
    (endWidthAttr.array as Float32Array).set(endWidths);
    endWidthAttr.needsUpdate = true;

    // Update sharpness (per-segment start/end)
    const startSharpAttr = geometry.getAttribute('aStartSharpness') as THREE.InstancedBufferAttribute;
    (startSharpAttr.array as Float32Array).set(startSharpness);
    startSharpAttr.needsUpdate = true;

    const endSharpAttr = geometry.getAttribute('aEndSharpness') as THREE.InstancedBufferAttribute;
    (endSharpAttr.array as Float32Array).set(endSharpness);
    endSharpAttr.needsUpdate = true;

    // Update segment lengths
    const lengthAttr = geometry.getAttribute('aSegmentLength') as THREE.InstancedBufferAttribute;
    (lengthAttr.array as Float32Array).set(segmentLengths);
    lengthAttr.needsUpdate = true;

    // Update clipped flags (convert Uint8 to Float32 for shader)
    const startClipAttr = geometry.getAttribute('aStartClipped') as THREE.InstancedBufferAttribute;
    const startClipFloat = new Float32Array(startClipped);  // Convert Uint8 → Float32
    (startClipAttr.array as Float32Array).set(startClipFloat);
    startClipAttr.needsUpdate = true;

    const endClipAttr = geometry.getAttribute('aEndClipped') as THREE.InstancedBufferAttribute;
    const endClipFloat = new Float32Array(endClipped);  // Convert Uint8 → Float32
    (endClipAttr.array as Float32Array).set(endClipFloat);
    endClipAttr.needsUpdate = true;

    // Set instance count (for instanced rendering)
    geometry.instanceCount = segmentCount;
  }

  /**
   * Update GSplats geometry (FULL IMPLEMENTATION)
   *
   * CRITICAL: Takes PACKED 3D data (after processGSplats + packCholeskyForShader),
   * NOT raw LoadedGSplatsData!
   *
   * Data flow:
   * 1. LoadedGSplatsData (nD) → processGSplats() → ProcessedGSplatsData (3D)
   * 2. ProcessedGSplatsData → packCholeskyForShader() → PackedGSplatsData
   * 3. PackedGSplatsData → updateGSplatsGeometry() → GPU
   */
  updateGSplatsGeometry(
    geometry: THREE.BufferGeometry,
    packedData: PackedGSplatsData,
    previousCount?: number
  ): void {
    const {
      centers3D,
      amplitudes,
      cholesky01,
      cholesky23,
      cholesky45,
      colors,
      sharpness,
      splatCount,
    } = packedData;

    // Update 3D centers
    const centerAttr = geometry.getAttribute('aCenter') as THREE.InstancedBufferAttribute;
    (centerAttr.array as Float32Array).set(centers3D);
    if (previousCount !== undefined && splatCount < previousCount) {
      centerAttr.addUpdateRange(0, splatCount * 3);  // Always 3D!
    } else {
      centerAttr.needsUpdate = true;
    }

    // Update attenuated amplitudes
    const ampAttr = geometry.getAttribute('aAmplitude') as THREE.InstancedBufferAttribute;
    (ampAttr.array as Float32Array).set(amplitudes);
    ampAttr.needsUpdate = true;

    // Update packed Cholesky factors (3 vec2 attributes)
    const chol01Attr = geometry.getAttribute('aCholesky01') as THREE.InstancedBufferAttribute;
    (chol01Attr.array as Float32Array).set(cholesky01);
    chol01Attr.needsUpdate = true;

    const chol23Attr = geometry.getAttribute('aCholesky23') as THREE.InstancedBufferAttribute;
    (chol23Attr.array as Float32Array).set(cholesky23);
    chol23Attr.needsUpdate = true;

    const chol45Attr = geometry.getAttribute('aCholesky45') as THREE.InstancedBufferAttribute;
    (chol45Attr.array as Float32Array).set(cholesky45);
    chol45Attr.needsUpdate = true;

    // Update colors (RGB Float32)
    const colorAttr = geometry.getAttribute('aColor') as THREE.InstancedBufferAttribute;
    (colorAttr.array as Float32Array).set(colors);
    colorAttr.needsUpdate = true;

    // Update sharpness
    const sharpnessAttr = geometry.getAttribute('aSharpness') as THREE.InstancedBufferAttribute;
    (sharpnessAttr.array as Float32Array).set(sharpness);
    sharpnessAttr.needsUpdate = true;

    // Set instance count (for instanced rendering)
    geometry.instanceCount = splatCount;
  }

  private getCapacityBucket(capacity: number): string {
    if (capacity < 1000) return '1k';
    if (capacity < 5000) return '5k';
    if (capacity < 10000) return '10k';
    if (capacity < 50000) return '50k';
    if (capacity < 100000) return '100k';
    return '500k';
  }

  private maybeEvict(): void {
    // LRU eviction (similar to accumulator)
    // ... (omitted for brevity)
  }

  getStats() {
    return {
      ...this.stats,
      totalActive: this.activeBuffers.size,
      totalPooled:
        [...this.pointBuffers.values()].reduce((sum, arr) => sum + arr.length, 0) +
        [...this.lineBuffers.values()].reduce((sum, arr) => sum + arr.length, 0) +
        [...this.gsplatBuffers.values()].reduce((sum, arr) => sum + arr.length, 0),
    };
  }

  dispose(): void {
    // Dispose all geometries
    for (const pooled of this.activeBuffers.values()) {
      pooled.geometry.dispose();
    }
    for (const buffers of this.pointBuffers.values()) {
      buffers.forEach(p => p.geometry.dispose());
    }
    for (const buffers of this.lineBuffers.values()) {
      buffers.forEach(p => p.geometry.dispose());
    }
    for (const buffers of this.gsplatBuffers.values()) {
      buffers.forEach(p => p.geometry.dispose());
    }

    this.activeBuffers.clear();
    this.pointBuffers.clear();
    this.lineBuffers.clear();
    this.gsplatBuffers.clear();
  }
}

export const gpuBufferPool = new GPUBufferPool();
```

---

## 6. Integration Strategy

### 6.1 Vite Configuration for Workers and WASM

**File**: `packages/luxar-viewer/vite.config.ts` (ADD SECTION)

```typescript
import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';

export default defineConfig({
  // ... existing config ...

  plugins: [
    wasm(), // Enable WASM support
  ],

  worker: {
    format: 'es', // Use ES modules in workers
    plugins: [wasm()],
  },

  build: {
    target: 'esnext', // Required for import.meta.url

    rollupOptions: {
      output: {
        // Ensure workers are properly bundled
        manualChunks(id) {
          if (id.includes('data-worker.ts')) {
            return 'data-worker';
          }
        },
      },
    },
  },

  optimizeDeps: {
    exclude: ['@luxar/wasm'], // Don't pre-bundle WASM
  },
});
```

### 6.2 Config Integration

**File**: `packages/luxar-viewer/src/config/index.ts` (ADD TO EXISTING `dataLoading` SECTION)

⚠️ **IMPORTANT**: The actual config uses `dataLoading` (not `dataLoader`). Add a new
`performance` sub-section within the existing `dataLoading` config block.

```typescript
export const config = {
  // ... existing sections ...

  // Data loading configuration (EXISTING - add 'performance' sub-section)
  dataLoading: {
    spatial: { /* existing */ },
    network: { /* existing */ },
    memory: { /* existing */ },
    monitor: { /* existing */ },

    // NEW: Performance optimization settings
    performance: {
      // Object pooling (Phase 1)
      useAccumulators: true,
      initialAccumulatorCapacity: 8192,
      accumulatorGrowthFactor: 1.5,

      // Web Workers (Phase 2)
      useWebWorkers: true,
      workerCount: 1, // Single worker for now

      // WASM acceleration (Phase 3)
      useWASM: true,
      wasmModulePath: '/dist/wasm/luxar_wasm_bg.wasm',

      // GPU buffer pool (Phase 4)
      useGPUBufferPool: true,
      gpuPoolMaxSize: 20,
      gpuPoolEvictionFrames: 300,

      // Performance monitoring
      enablePerformanceMonitoring: false,
    },
  },
};
```

**Access pattern**: `config.dataLoading.performance.useAccumulators` etc.

### 6.3 SceneManager Integration

**File**: `packages/luxar-viewer/src/scene/scene-manager.ts` (MODIFY)

```typescript
import { gpuBufferPool } from '../rendering/gpu-buffer-pool';
import { buildInstanceBuffers } from '../data/lines-processor';  // Lines transformation
import { processGSplats, packCholeskyForShader } from '../data/gsplats-processor';  // GSplats transformation
import { materialManager } from '../rendering/material-manager';
import type { PointsData } from '../data/data-loader-types';
import type { LoadedLinesData, ProcessedLinesData } from '../types/lines';
import type { LoadedGSplatsData, GSplatsViewState, ProcessedGSplatsData } from '../types/gsplats';
import type { PackedGSplatsData } from '../rendering/gpu-buffer-pool';  // GPU-ready format

export class SceneManager {
  // Track previous counts for partial updates
  private nodePointCounts = new Map<string, number>();
  private nodeSegmentCounts = new Map<string, number>();
  private nodeSplatCounts = new Map<string, number>();

  /**
   * Update points node
   */
  updatePoints(nodeId: string, pointsData: PointsData): void {
    const geometry = gpuBufferPool.acquirePointsGeometry(
      nodeId,
      pointsData.metadata.loadedPoints
    );

    const previousCount = this.nodePointCounts.get(nodeId);
    gpuBufferPool.updatePointsGeometry(geometry, pointsData, previousCount);
    this.nodePointCounts.set(nodeId, pointsData.metadata.loadedPoints);

    // Create or update mesh
    let mesh = this.pointsMeshes.get(nodeId);
    if (!mesh) {
      const material = materialManager.getPointMaterial({ /* ... */ });
      mesh = new THREE.Points(geometry, material);
      this.scene.add(mesh);
      this.pointsMeshes.set(nodeId, mesh);
    } else {
      mesh.geometry = geometry;
    }
  }

  /**
   * Update lines node
   *
   * IMPORTANT: Lines require TWO steps:
   * 1. buildInstanceBuffers() transforms LoadedLinesData → ProcessedLinesData
   * 2. GPU pool updates geometry with per-segment ProcessedLinesData
   */
  updateLines(
    nodeId: string,
    loadedData: LoadedLinesData,
    viewState: { slicePosition: number[]; tolerance: number[]; displayDims: number[] }
  ): void {
    // STEP 1: Transform per-vertex LoadedLinesData → per-segment ProcessedLinesData
    const processedData: ProcessedLinesData = buildInstanceBuffers(
      loadedData,
      viewState.slicePosition,
      viewState.tolerance,
      viewState.displayDims
    );

    // STEP 2: Acquire geometry with ProcessedLinesData segment count
    const geometry = gpuBufferPool.acquireLinesGeometry(
      nodeId,
      processedData.segmentCount  // Use PROCESSED count (after clipping)
    );

    // STEP 3: Update geometry with ProcessedLinesData (per-segment)
    const previousCount = this.nodeSegmentCounts.get(nodeId);
    gpuBufferPool.updateLinesGeometry(geometry, processedData, previousCount);
    this.nodeSegmentCounts.set(nodeId, processedData.segmentCount);

    // Create or update mesh (InstancedMesh for lines)
    let mesh = this.linesMeshes.get(nodeId);
    if (!mesh) {
      const material = materialManager.getLineMaterial({ /* ... */ });
      mesh = new THREE.Mesh(geometry, material);  // Mesh, not Line!
      this.scene.add(mesh);
      this.linesMeshes.set(nodeId, mesh);
    } else {
      mesh.geometry = geometry;
    }
  }

  /**
   * Update gsplats node
   *
   * CRITICAL DATA FLOW (matches actual scene-loader.ts):
   * 1. LoadedGSplatsData (nD raw from zarr)
   * 2. → processGSplats() → ProcessedGSplatsData (3D sliced + attenuated)
   * 3. → packCholeskyForShader() → Packed Cholesky arrays
   * 4. → GPUBufferPool / mesh update → GPU
   *
   * NOTE: The actual implementation uses createInstancedGSplatsMesh/updateInstancedGSplatsMesh
   * from gsplat-material.ts. The GPUBufferPool pattern here is a PROPOSED optimization.
   */
  updateGSplats(nodeId: string, gsplatsData: LoadedGSplatsData, viewState: GSplatsViewState): void {
    // Step 1: Process nD → 3D (visibility filtering + attenuation)
    const processed = processGSplats(gsplatsData, viewState);

    // Step 2: Pack Cholesky factors for shader attributes
    const packed = packCholeskyForShader(processed.choleskyFactors3D, processed.splatCount);

    // Step 3: Acquire GPU geometry (always 3D - no ndim parameter!)
    const geometry = gpuBufferPool.acquireGSplatsGeometry(
      nodeId,
      processed.splatCount  // Use processed count (may be less due to visibility filtering!)
    );

    // Step 4: Create PackedGSplatsData for GPU upload
    const packedData: PackedGSplatsData = {
      centers3D: processed.centers3D,
      amplitudes: processed.amplitudes,
      cholesky01: packed.cholesky01,
      cholesky23: packed.cholesky23,
      cholesky45: packed.cholesky45,
      colors: processed.colors,
      splatCount: processed.splatCount,
    };

    // Step 5: Update GPU buffers
    const previousCount = this.nodeSplatCounts.get(nodeId);
    gpuBufferPool.updateGSplatsGeometry(geometry, packedData, previousCount);
    this.nodeSplatCounts.set(nodeId, processed.splatCount);

    // Step 6: Create or update mesh
    let mesh = this.gsplatsMeshes.get(nodeId);
    if (!mesh) {
      const material = materialManager.getGSplatMaterial({ /* ... */ });
      mesh = new THREE.Mesh(geometry, material);
      this.scene.add(mesh);
      this.gsplatsMeshes.set(nodeId, mesh);
    } else {
      mesh.geometry = geometry;
    }
  }

  /**
   * Remove node
   */
  removeNode(nodeId: string, type: 'points' | 'lines' | 'gsplats'): void {
    // Remove from scene
    if (type === 'points') {
      const mesh = this.pointsMeshes.get(nodeId);
      if (mesh) {
        this.scene.remove(mesh);
        this.pointsMeshes.delete(nodeId);
      }
      this.nodePointCounts.delete(nodeId);
    } else if (type === 'lines') {
      const mesh = this.linesMeshes.get(nodeId);
      if (mesh) {
        this.scene.remove(mesh);
        this.linesMeshes.delete(nodeId);
      }
      this.nodeSegmentCounts.delete(nodeId);
    } else if (type === 'gsplats') {
      const mesh = this.gsplatsMeshes.get(nodeId);
      if (mesh) {
        this.scene.remove(mesh);
        this.gsplatsMeshes.delete(nodeId);
      }
      this.nodeSplatCounts.delete(nodeId);
    }

    // Release GPU buffer
    gpuBufferPool.release(nodeId);
  }
}
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

**Test Points Accumulator:**
```typescript
// src/data/data-accumulator.test.ts
import { describe, it, expect } from 'vitest';
import { PointsDataAccumulator } from './data-accumulator';
import * as THREE from 'three';

describe('PointsDataAccumulator', () => {
  it('should grow capacity by 1.5x', () => {
    const acc = new PointsDataAccumulator(1000, 3, 10000);
    const grew = acc.ensureCapacity(1500);
    expect(grew).toBe(true);
    expect(acc.getStats().capacity).toBe(2250); // ceil(1500 * 1.5)
  });

  it('should return PointsData with correct metadata structure', () => {
    const acc = new PointsDataAccumulator(1000, 3, 10000);
    acc.fill(0, {
      positions: new Float32Array([1, 2, 3]),
      colors: new Uint8Array([255, 0, 0, 255]),
      radii: new Float32Array([0.5]),
      sharpness: new Float32Array([2.0]),
    });

    const data = acc.getData(1);

    // Verify metadata structure
    expect(data.metadata).toBeDefined();
    expect(data.metadata.loadedPoints).toBe(1);
    expect(data.metadata.totalPoints).toBe(10000);
    expect(data.metadata.ndim).toBe(3);
    expect(data.metadata.bounds).toBeInstanceOf(THREE.Box3);
  });
});
```

**Test Lines Accumulator (CORRECTED Structure):**
```typescript
describe('LinesDataAccumulator', () => {
  it('should handle flat buffers correctly', () => {
    const acc = new LinesDataAccumulator(1000, 500, 3);

    acc.fill(0, 0, {
      vertices: new Float32Array([0, 0, 0, 1, 1, 1]), // 2 vertices, 3D
      segments: new Uint32Array([0, 1]), // 1 segment
      widths: new Float32Array([0.1]),
      colors: new Float32Array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), // RGB Float32
      sharpness: new Float32Array([2.0, 2.0]),
    });

    const data = acc.getData(1, 2); // 1 segment, 2 vertices
    expect(data.segmentCount).toBe(1);
    expect(data.vertexCount).toBe(2);
    expect(data.colors).not.toBeNull(); // RGB Float32
    expect(data.colors!.length).toBe(6); // 2 vertices * 3 (RGB)
  });
});
```

**Test GSplats Accumulator (CORRECTED Field Names):**
```typescript
describe('GSplatsDataAccumulator', () => {
  it('should use camelCase choleskyFactors', () => {
    const acc = new GSplatsDataAccumulator(1000, 3);

    acc.fill(0, {
      centers: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]), // 3D: 6 elements
      colors: new Float32Array([1.0, 0.0, 0.0]), // RGB Float32
    });

    const data = acc.getData(1);
    expect(data.choleskyFactors).toBeDefined(); // CORRECT: camelCase!
    expect(data.choleskyFactors.length).toBe(6);
    expect(data.colors).not.toBeNull();
    expect(data.colors!.length).toBe(3); // RGB
  });
});
```

---

## 8. Performance Targets & Metrics

| Metric | Current | Target | Stretch Goal |
|--------|---------|--------|--------------|
| GC pause frequency | 10-20ms/2-3s | <2ms/10s | <1ms |
| Main thread block (Points) | 40-80ms | <5ms | <1ms |
| Main thread block (Lines) | 80-120ms | <10ms | <5ms |
| Main thread block (GSplats) | 60-100ms | <8ms | <3ms |
| Spatial query (1K chunks) | 8.2ms | 2ms | <1ms |
| nD visibility (100K pts) | 45ms | 10ms | 5ms |
| GPU buffer allocation | 2-3ms | 0ms (reuse) | 0ms |
| Frame rate (during load) | 15-25 FPS | 55-60 FPS | 60 FPS |
| Memory overhead | Baseline | +15% | +10% |

---

## 9. Implementation Phases

### Phase 1: Object Pooling (Week 1)

**Deliverables:**
- `data-accumulator.ts` with all three types (CORRECTED structures)
- Integration into all three loaders (CORRECTED method names)
- Unit tests (>95% coverage)
- Benchmarks

**Success Criteria:**
- GC pause avg: <2ms
- Memory overhead: <15%
- All tests pass

---

### Phase 2: Web Workers (Week 2)

**Deliverables:**
- `data-worker.ts` with ArrayDecoder integration
- `worker-pool.ts`
- Tests for all three primitive types
- Worker communication verified
- Vite configuration for worker bundling

**Success Criteria:**
- Main thread block: <5ms (Points), <10ms (Lines), <8ms (GSplats)
- Frame rate: 55-60 FPS
- WASM spatial queries work in worker (ArrayDecoder stays on main thread)

---

### Phase 3: WASM Acceleration (Week 3)

**Deliverables:**
- Rust WASM module with all functions
- Build integration (wasm-pack + vite)
- Benchmarks showing speedup
- Error handling (fails if WASM unavailable)

**Success Criteria:**
- Spatial query: <2ms
- nD visibility: <10ms
- SIMD verification
- NO fallback (error message if WASM missing)

---

### Phase 4: GPU Buffer Pool (Week 4)

**Deliverables:**
- `gpu-buffer-pool.ts` with all three geometry types
- SceneManager integration
- WebGL context loss handling
- Performance validation

**Success Criteria:**
- GPU allocation: 0ms (reuse)
- Partial updates working
- Memory: <500MB VRAM for 1M elements
- All three types supported

---

## 10. Edge Cases & Caveats

### 10.1 No Backwards Compatibility (Option C)

**Policy: No fallbacks, no old formats**

1. **No Browser Fallbacks:**
   - If WASM unavailable → FAIL with clear error message
   - If Workers unavailable → FAIL
   - If SharedArrayBuffer unavailable → FAIL
   - User sees: "Luxar requires modern browser (Chrome 57+, Firefox 52+, Safari 11+)"

2. **No Old Data Formats:**
   - Remove support for pre-1.0 zarr formats
   - No migration tools
   - Users must regenerate datasets

**Rationale**: Early-stage project, clean break acceptable.

### 10.2 Lines Two-Phase Complexity

Lines require special handling:
- Phase 1: Query segment chunks
- Phase 2: Derive vertex chunks from indices
- Phase 3: Remap indices (global → local)
- Phase 4: Clip segments at boundaries

**Worker must support all phases.**

### 10.3 Buffer View Invalidation

```typescript
// ❌ WRONG: Storing reference to subarray
const data = accumulator.getData(1000);
this.cachedData = data; // ⚠️ DANGER!
// If accumulator grows, this.cachedData is invalid

// ✅ CORRECT: Copy if you need to persist
const data = accumulator.getData(1000);
const positionsCopy = new Float32Array(data.positions);
this.cachedPositions = positionsCopy;
```

### 10.4 Existing Cache is Complementary

```
Network Request
    ↓
[L1/L2 Zarr Cache] ← EXISTING (unchanged)
    ↓
ArrayDecoder (MAIN THREAD)  ← Needs zarr.Array objects
    ↓
[CPU Accumulators] ← NEW (Phase 1)
    ↓
[GPU Buffer Pool] ← NEW (Phase 4)
    ↓
THREE.Scene → Render

[WASM Worker handles: Spatial index queries, nD visibility computation]
```

**No conflict - separate layers!**

### 10.5 RGB Colors (3 Components, Multiple Formats)

**CRITICAL**: All types use **RGB** (3 components), NOT RGBA (4 components):

```typescript
// Lines and GSplats (after processing)
colors: Float32Array | null;  // RGB, 3 components per vertex/splat (nullable)

// Points (supports multiple formats for HDR/SDR)
colors?: Uint8Array | Uint16Array | Float32Array;  // RGB, 3 components
// - Uint8Array: SDR mode [0-255], normalized by THREE.js
// - Float32Array: HDR mode [0-∞), no normalization
```

**Note**: The accumulator and GPU buffer pool use Float32Array internally
for maximum flexibility. Conversion happens as needed.

### 10.6 nD Vertices → 3D Extraction

Lines store vertices in nD space. The processor must extract 3D positions based on displayDims:

```typescript
// Extract 3D from nD vertices
const positions3D = new Float32Array(vertexCount * 3);
for (let i = 0; i < vertexCount; i++) {
  positions3D[i * 3 + 0] = vertices[i * ndim + displayDims[0]];
  positions3D[i * 3 + 1] = vertices[i * ndim + displayDims[1]];
  positions3D[i * 3 + 2] = vertices[i * ndim + displayDims[2]];
}
```

---

## Summary

This fully corrected specification (v3.6.0):

**v3.6.0 Corrections:**
- ✅ Config integration uses **`dataLoading.performance`** (matches existing pattern)
- ✅ ArrayDecoder location **consistently documented as MAIN THREAD** throughout
- ✅ Worker responsibilities clarified (**WASM spatial queries only**)
- ✅ Architecture diagram **correctly shows ArrayDecoder on main thread**

**v3.5.0 Corrections:**
- ✅ processGSplats() now uses **Mahalanobis distance** (not simple Gaussian)
- ✅ Two-pass visibility filtering with **minAmplitude threshold**
- ✅ acquireGSplatsGeometry() **no longer takes ndim** (always 3D after processing)
- ✅ SceneManager.updateGSplats() shows **correct data flow** (process → pack → update)
- ✅ Added **PackedGSplatsData interface** in GPUBufferPool module
- ✅ Documents **actual vs proposed** GSplats pattern (direct mesh vs GPUBufferPool)

**Previous Corrections (v3.0-v3.4):**
- ✅ Fixes ALL interface mismatches (LoadedLinesData, LoadedGSplatsData)
- ✅ Uses correct field names (choleskyFactors camelCase, not snake_case)
- ✅ Uses correct method names (loadLines, loadGSplats, not all loadPoints)
- ✅ Implements complete GPU buffer pool methods (updateLinesGeometry, updateGSplatsGeometry)
- ✅ Uses correct color types (RGB Float32 for Lines/GSplats, HDR for Points)
- ✅ Handles nD vertices extraction to 3D correctly
- ✅ Includes proper Vite configuration for workers/WASM
- ✅ Clarifies worker-zarr pattern (pre-fetched chunks)
- ✅ Corrects accumulator signatures (getData, fill)
- ✅ Defines Option C backwards compatibility (no fallbacks)
- ✅ Provides complete, implementation-ready code examples

**Key GSplats Data Flow:**
```
LoadedGSplatsData (nD)
    ↓ processGSplats() [Mahalanobis distance, visibility filtering]
ProcessedGSplatsData (3D)
    ↓ packCholeskyForShader()
PackedGSplatsData (GPU-ready)
    ↓ updateGSplatsGeometry() / updateInstancedGSplatsMesh()
GPU Buffers
```

**Status: READY FOR IMPLEMENTATION**

---

**Document Version**: 3.6.0 (Config Integration & ArrayDecoder Location Consistency)
**Date**: 2025-12-23
**Approval**: Ready for Phase 1 start
