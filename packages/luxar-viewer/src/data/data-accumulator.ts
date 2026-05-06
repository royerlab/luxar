/**
 * Data Accumulators - Multi-type object pooling for Points, Lines, and GSplats
 *
 * Implements persistent TypedArray buffers that grow by 1.5x when needed,
 * eliminating per-frame allocations and reducing GC pressure.
 *
 * Supports Float32Array, Uint8Array, and Uint16Array natively (no conversion!)
 * for optimal memory efficiency consistent with multi-type GPU buffer pool.
 *
 * Based on Performance Optimization Specification v3.6.0
 *
 * ✅ IMPLEMENTATION STATUS (Phase 1 - ALL LOADING PHASES COMPLETE):
 * - Infrastructure: COMPLETE ✅
 * - Multi-type support: COMPLETE ✅ (Float32/Uint8/Uint16)
 * - Unit tests: COMPLETE ✅ (26 tests, >95% coverage)
 * - Deep integration (Points): COMPLETE ✅ (zero-allocation loading)
 * - Deep integration (Lines): COMPLETE ✅ (zero-allocation loading, bug fixed 2025-12-27)
 * - Deep integration (GSplats): COMPLETE ✅ (zero-allocation loading)
 *
 * All three loaders now use deep integration for LOADING phase:
 * - Write directly to accumulator buffers during data loading
 * - Return zero-copy subarrays from accumulator.getData()
 * - Achieve complete zero-allocation operation in loading hot path
 *
 * Processing phases (buildInstanceBuffers, processGSplats) analyzed and determined
 * to be optimal with current two-pass algorithms - accumulator NOT recommended.
 */

import * as THREE from 'three';
import type { LoadedPointsData, PositionArray, ColorArray, ScalarArray } from './data-loader-types';
import type { LoadedLinesData } from '../types/lines';
import type { LoadedGSplatsData } from '../types/gsplats';
import { log, Modules } from '../utils/log';

/**
 * Generic accumulator interface
 */
export interface DataAccumulator<
  TData,
  TGetArgs extends unknown[] = unknown[],
  TFillArgs extends unknown[] = unknown[],
> {
  /**
   * Ensure capacity (grow if needed)
   * @returns true if capacity was grown
   */
  ensureCapacity(needed: number): boolean;

  /**
   * Get data view (subarray of internal buffers)
   *
   * Signature varies by type:
   * - Points: getData(count: number): LoadedPointsData
   * - Lines: getData(segmentCount: number, vertexCount: number): LoadedLinesData
   * - GSplats: getData(count: number): LoadedGSplatsData
   */
  getData(...args: TGetArgs): TData;

  /**
   * Fill accumulator at offset(s)
   *
   * Signature varies by type:
   * - Points: fill(offset: number, data: Partial<TData>)
   * - Lines: fill(segmentOffset: number, vertexOffset: number, data: Partial<TData>)
   * - GSplats: fill(offset: number, data: Partial<TData>)
   */
  fill(...args: TFillArgs): void;

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
 * Attribute types for Points accumulator (matches GPU buffer pool)
 */
export interface PointsAccumulatorTypes {
  position: 'Float32Array';
  color: 'Float32Array' | 'Uint8Array' | 'Uint16Array';
  radius: 'Float32Array' | 'Uint8Array';
  sharpness: 'Float32Array' | 'Uint8Array';
}

/**
 * Points data accumulator with FULL multi-type support
 *
 * Handles Float32Array, Uint8Array, and Uint16Array natively (no conversion)
 * to maintain memory efficiency and consistency with multi-type GPU buffer pool.
 *
 * Type is detected on first fill() and remains fixed for the accumulator's lifetime.
 */
export class LoadedPointsDataAccumulator implements DataAccumulator<
  LoadedPointsData,
  [number],
  [number, Partial<LoadedPointsData>]
> {
  // Persistent buffers (typed based on data)
  private positionBuffer: Float32Array; // Always Float32
  private colorBuffer: Float32Array | Uint8Array | Uint16Array;
  private radiiBuffer: Float32Array | Uint8Array;
  private sharpnessBuffer: Float32Array | Uint8Array;

  // Type tracking
  private types: PointsAccumulatorTypes | null = null;

  // Track which attributes have been filled
  private hasColors = false;
  private hasRadii = false;
  private hasSharpness = false;

  // Current capacity (number of points)
  private capacity: number;

  // Statistics
  private allocations = 0;
  private totalGrowths = 0;

  // Metadata tracking
  private ndim: number;
  private totalPoints: number;
  private usedSpatialIndex = false;

  /**
   * Create a new Points data accumulator with multi-type support
   *
   * @param initialCapacity - Initial capacity in number of points (default: 1024)
   * @param ndim - Number of dimensions in source data (default: 3)
   * @param totalPoints - Total points in dataset for metadata tracking (default: 0)
   *
   * @example
   * ```typescript
   * const accumulator = new LoadedPointsDataAccumulator(8192, 4, 10000);
   * accumulator.ensureCapacity(5000);
   * accumulator.fill(0, { positions, colors, radii, sharpness });
   * const data = accumulator.getData(5000);
   * ```
   */
  constructor(initialCapacity = 1024, ndim = 3, totalPoints = 0) {
    this.capacity = initialCapacity;
    this.ndim = ndim;
    this.totalPoints = totalPoints;

    // Start with Float32Array (will be replaced on first fill with actual types)
    this.positionBuffer = new Float32Array(initialCapacity * 3);
    this.colorBuffer = new Float32Array(initialCapacity * 3);
    this.radiiBuffer = new Float32Array(initialCapacity);
    this.sharpnessBuffer = new Float32Array(initialCapacity);

    this.allocations++;
  }

  /**
   * Ensure accumulator has sufficient capacity, growing if needed
   *
   * Uses 1.5x growth strategy to minimize reallocation events while avoiding
   * excessive memory overhead. Growth preserves attribute types (Uint8/Uint16/Float32).
   *
   * @param needed - Minimum required capacity (number of points)
   * @returns true if capacity was grown, false if already sufficient
   *
   * @example
   * ```typescript
   * // Ensure space for 10000 points
   * const didGrow = accumulator.ensureCapacity(10000);
   * if (didGrow) {
   *   console.log('Accumulator grew to', accumulator.getStats().capacity);
   * }
   * ```
   */
  ensureCapacity(needed: number): boolean {
    if (needed <= this.capacity) return false;

    // Calculate new capacity with 1.5x growth factor
    let newCapacity = this.capacity;
    while (newCapacity < needed) {
      newCapacity = Math.ceil(newCapacity * 1.5);
    }

    log.info(
      Modules.DATA_ACCUMULATOR,
      `Growing LoadedPointsDataAccumulator: ${this.capacity} → ${newCapacity} points`
    );

    // Allocate new buffers with SAME types as current (type-preserving growth)
    const newPositions = new Float32Array(newCapacity * 3);
    newPositions.set(this.positionBuffer);
    this.positionBuffer = newPositions;

    // Color: Type-preserving
    if (this.colorBuffer instanceof Uint8Array) {
      const newColors = new Uint8Array(newCapacity * 3);
      newColors.set(this.colorBuffer);
      this.colorBuffer = newColors;
    } else if (this.colorBuffer instanceof Uint16Array) {
      const newColors = new Uint16Array(newCapacity * 3);
      newColors.set(this.colorBuffer);
      this.colorBuffer = newColors;
    } else {
      const newColors = new Float32Array(newCapacity * 3);
      newColors.set(this.colorBuffer);
      this.colorBuffer = newColors;
    }

    // Radius: Type-preserving
    if (this.radiiBuffer instanceof Uint8Array) {
      const newRadii = new Uint8Array(newCapacity);
      newRadii.set(this.radiiBuffer);
      this.radiiBuffer = newRadii;
    } else {
      const newRadii = new Float32Array(newCapacity);
      newRadii.set(this.radiiBuffer);
      this.radiiBuffer = newRadii;
    }

    // Sharpness: Type-preserving
    if (this.sharpnessBuffer instanceof Uint8Array) {
      const newSharpness = new Uint8Array(newCapacity);
      newSharpness.set(this.sharpnessBuffer);
      this.sharpnessBuffer = newSharpness;
    } else {
      const newSharpness = new Float32Array(newCapacity);
      newSharpness.set(this.sharpnessBuffer);
      this.sharpnessBuffer = newSharpness;
    }

    this.capacity = newCapacity;
    this.allocations++;
    this.totalGrowths++;

    return true;
  }

  /**
   * Get LoadedPointsData with NATIVE types (zero-copy subarrays)
   *
   * Returns views (subarrays) into accumulator buffers with native types preserved.
   * No conversion or copying occurs. Computes fresh bounds from positions.
   *
   * @param count - Number of points to return (must be ≤ capacity)
   * @returns LoadedPointsData with subarrays pointing to accumulator buffers
   * @throws Error if count > capacity
   *
   * @example
   * ```typescript
   * const data = accumulator.getData(5000);
   * // data.positions is a subarray of accumulator.positionBuffer
   * // data.colors might be Uint8Array, Uint16Array, or Float32Array
   * console.log('Actual type:', data.colors?.constructor.name);
   * console.log('Dtype:', data.metadata.dtypes.colors); // 'uint8', 'uint16', or 'float32'
   * ```
   */
  getData(count: number): LoadedPointsData {
    if (count > this.capacity) {
      throw new Error(`Cannot get ${count} points from accumulator with capacity ${this.capacity}`);
    }

    // Compute bounds from loaded positions (zero allocations — no Vector3 per point)
    const bounds = new THREE.Box3();
    if (count > 0) {
      let minX = Infinity,
        minY = Infinity,
        minZ = Infinity;
      let maxX = -Infinity,
        maxY = -Infinity,
        maxZ = -Infinity;
      for (let i = 0; i < count; i++) {
        const x = this.positionBuffer[i * 3];
        const y = this.positionBuffer[i * 3 + 1];
        const z = this.positionBuffer[i * 3 + 2];
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (z < minZ) minZ = z;
        if (z > maxZ) maxZ = z;
      }
      bounds.min.set(minX, minY, minZ);
      bounds.max.set(maxX, maxY, maxZ);
    }

    // Return LoadedPointsData with NATIVE types (matches what was filled!)
    return {
      positions: this.positionBuffer.subarray(0, count * 3) as PositionArray,
      colors: this.hasColors ? (this.colorBuffer.subarray(0, count * 3) as ColorArray) : undefined,
      radii: this.hasRadii ? (this.radiiBuffer.subarray(0, count) as ScalarArray) : undefined,
      sharpness: this.hasSharpness
        ? (this.sharpnessBuffer.subarray(0, count) as ScalarArray)
        : undefined,
      pointCount: count,
      ndim: this.ndim,
      metadata: {
        totalPoints: this.totalPoints,
        loadedPoints: count,
        bounds, // Use freshly computed bounds
        usedSpatialIndex: this.usedSpatialIndex,
        dtypes: {
          positions: 'float32',
          colors:
            this.types?.color === 'Uint8Array'
              ? 'uint8'
              : this.types?.color === 'Uint16Array'
                ? 'uint16'
                : 'float32',
          radii: this.types?.radius === 'Uint8Array' ? 'uint8' : 'float32',
          sharpness: this.types?.sharpness === 'Uint8Array' ? 'uint8' : 'float32',
        },
      },
    };
  }

  /**
   * Detect and initialize buffer types on first fill
   */
  private initializeTypes(data: Partial<LoadedPointsData>): void {
    if (this.types) return; // Already initialized

    const types: PointsAccumulatorTypes = {
      position: 'Float32Array',
      color:
        data.colors instanceof Uint8Array
          ? 'Uint8Array'
          : data.colors instanceof Uint16Array
            ? 'Uint16Array'
            : 'Float32Array',
      radius: data.radii instanceof Uint8Array ? 'Uint8Array' : 'Float32Array',
      sharpness: data.sharpness instanceof Uint8Array ? 'Uint8Array' : 'Float32Array',
    };

    this.types = types;

    // Recreate buffers with correct types (only if types differ from initial Float32)
    if (types.color === 'Uint8Array') {
      this.colorBuffer = new Uint8Array(this.capacity * 3);
    } else if (types.color === 'Uint16Array') {
      this.colorBuffer = new Uint16Array(this.capacity * 3);
    }

    if (types.radius === 'Uint8Array') {
      this.radiiBuffer = new Uint8Array(this.capacity);
    }

    if (types.sharpness === 'Uint8Array') {
      this.sharpnessBuffer = new Uint8Array(this.capacity);
    }
  }

  /**
   * Fill accumulator buffers at offset with point data
   *
   * Detects attribute types on first fill and creates appropriately-typed buffers.
   * Subsequent fills must use matching types. Preserves native types (no conversion).
   *
   * @param offset - Starting point index to fill from
   * @param data - Partial point data to fill (any subset of attributes)
   * @param data.positions - 3D positions (Float32Array, size: count*3)
   * @param data.colors - Optional colors (Float32Array|Uint8Array|Uint16Array, size: count*3)
   * @param data.radii - Optional radii (Float32Array|Uint8Array, size: count)
   * @param data.sharpness - Optional sharpness (Float32Array|Uint8Array, size: count)
   *
   * @example
   * ```typescript
   * // First fill: detects types (Uint8 colors)
   * accumulator.fill(0, {
   *   positions: new Float32Array([1,2,3]),
   *   colors: new Uint8Array([255, 128, 0])
   * });
   * // Accumulator now has Uint8Array colorBuffer
   *
   * // Subsequent fills: must use same types
   * accumulator.fill(1, {
   *   positions: new Float32Array([4,5,6]),
   *   colors: new Uint8Array([0, 255, 128])  // Must be Uint8Array
   * });
   * ```
   */
  fill(offset: number, data: Partial<LoadedPointsData>): void {
    // Initialize types on first fill
    this.initializeTypes(data);

    // Fill buffers with native types (no conversion!)
    if (data.positions) {
      (this.positionBuffer as Float32Array).set(data.positions as Float32Array, offset * 3);
    }
    if (data.colors) {
      this.hasColors = true;
      if (data.colors instanceof Uint8Array) {
        (this.colorBuffer as Uint8Array).set(data.colors, offset * 3);
      } else if (data.colors instanceof Uint16Array) {
        (this.colorBuffer as Uint16Array).set(data.colors, offset * 3);
      } else {
        (this.colorBuffer as Float32Array).set(data.colors, offset * 3);
      }
    }
    if (data.radii) {
      this.hasRadii = true;
      if (data.radii instanceof Uint8Array) {
        (this.radiiBuffer as Uint8Array).set(data.radii, offset);
      } else {
        (this.radiiBuffer as Float32Array).set(data.radii as Float32Array, offset);
      }
    }
    if (data.sharpness) {
      this.hasSharpness = true;
      if (data.sharpness instanceof Uint8Array) {
        (this.sharpnessBuffer as Uint8Array).set(data.sharpness, offset);
      } else {
        (this.sharpnessBuffer as Float32Array).set(data.sharpness as Float32Array, offset);
      }
    }
  }

  /**
   * Update accumulator metadata
   *
   * Updates metadata fields for the accumulated points. Note: bounds parameter
   * is ignored - getData() always computes fresh bounds from positions.
   *
   * @param metadata - Partial metadata update
   * @param metadata.ndim - Number of dimensions
   * @param metadata.totalPoints - Total points in full dataset
   * @param metadata.usedSpatialIndex - Whether spatial index was used for loading
   * @param metadata.bounds - Ignored (computed in getData())
   * @param metadata.usedEffectiveRadius - Tracked but not used in return
   */
  updateMetadata(metadata: {
    ndim?: number;
    totalPoints?: number;
    usedSpatialIndex?: boolean;
    bounds?: THREE.Box3;
    usedEffectiveRadius?: boolean;
  }): void {
    if (metadata.ndim !== undefined) this.ndim = metadata.ndim;
    if (metadata.totalPoints !== undefined) this.totalPoints = metadata.totalPoints;
    if (metadata.usedSpatialIndex !== undefined) this.usedSpatialIndex = metadata.usedSpatialIndex;
    // Don't update bounds via updateMetadata - getData() computes it from positions
    // This avoids potential corruption from external bounds objects
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

  // --- Public accessors for direct buffer access (used by loaders) ---

  /** Whether attribute types have been initialized via fill() */
  hasTypes(): boolean {
    return this.types !== null;
  }

  getPositionBuffer(): Float32Array {
    return this.positionBuffer;
  }

  getColorBuffer(): Float32Array | Uint8Array | Uint16Array {
    return this.colorBuffer;
  }

  getRadiiBuffer(): Float32Array | Uint8Array {
    return this.radiiBuffer;
  }

  getSharpnessBuffer(): Float32Array | Uint8Array {
    return this.sharpnessBuffer;
  }

  dispose(): void {
    this.positionBuffer = new Float32Array(0);
    this.colorBuffer = new Float32Array(0);
    this.radiiBuffer = new Float32Array(0);
    this.sharpnessBuffer = new Float32Array(0);
    this.capacity = 0;
    this.types = null;
    this.hasColors = false;
    this.hasRadii = false;
    this.hasSharpness = false;
  }
}

/**
 * Attribute types for Lines accumulator (matches GPU buffer pool pattern)
 */
export interface LinesAccumulatorTypes {
  position: 'Float32Array';
  color: 'Float32Array' | 'Uint8Array' | 'Uint16Array';
  width: 'Float32Array';
  sharpness: 'Float32Array';
}

/**
 * Lines data accumulator with FULL multi-type support
 *
 * CRITICAL: Flat buffers matching types/lines.ts:170-194
 * CRITICAL: widths is PER-VERTEX (N,), NOT per-segment!
 *
 * Handles Float32Array, Uint8Array, and Uint16Array for colors natively (no conversion)
 * to maintain memory efficiency and consistency with multi-type GPU buffer pool.
 *
 * Type is detected on first fill() and remains fixed for the accumulator's lifetime.
 */
export class LinesDataAccumulator implements DataAccumulator<
  LoadedLinesData,
  [number, number],
  [number, number, Partial<LoadedLinesData>]
> {
  // FLAT buffers (not nested!)
  private vertexBuffer: Float32Array; // ndim-dimensional vertices
  private segmentBuffer: Uint32Array; // index pairs
  private widthBuffer: Float32Array; // PER-VERTEX widths (N,) - NOT per-segment!
  private colorBuffer: Float32Array | Uint8Array | Uint16Array; // RGB per-vertex (multi-type!)
  private sharpnessBuffer: Float32Array; // per-vertex (always allocated)

  // Type tracking (like Points accumulator)
  private types: LinesAccumulatorTypes | null = null;

  // Track whether data actually has colors/sharpness
  private hasColors = false;
  private hasSharpness = false;

  private vertexCapacity: number;
  private segmentCapacity: number;
  private ndim: number;

  private allocations = 0;
  private totalGrowths = 0;

  constructor(initialVertexCapacity = 1024, initialSegmentCapacity = 512, ndim = 3) {
    this.vertexCapacity = initialVertexCapacity;
    this.segmentCapacity = initialSegmentCapacity;
    this.ndim = ndim;

    // Always allocate buffers (persistent, never null)
    // NOTE: widths is per-VERTEX, allocated with vertexCapacity!
    // Start with Float32Array (will be replaced on first fill with actual types)
    this.vertexBuffer = new Float32Array(initialVertexCapacity * ndim);
    this.segmentBuffer = new Uint32Array(initialSegmentCapacity * 2);
    this.widthBuffer = new Float32Array(initialVertexCapacity); // PER-VERTEX!
    this.colorBuffer = new Float32Array(initialVertexCapacity * 3); // RGB
    this.sharpnessBuffer = new Float32Array(initialVertexCapacity);

    this.allocations++;
  }

  /**
   * Initialize types from first data fill (like Points accumulator)
   */
  private initializeTypes(data: Partial<LoadedLinesData>): void {
    if (this.types) return; // Already initialized

    const types: LinesAccumulatorTypes = {
      position: 'Float32Array',
      color:
        data.colors instanceof Uint8Array
          ? 'Uint8Array'
          : data.colors instanceof Uint16Array
            ? 'Uint16Array'
            : 'Float32Array',
      width: 'Float32Array',
      sharpness: 'Float32Array',
    };

    this.types = types;

    // Recreate color buffer with correct type (only if types differ from initial Float32)
    if (types.color === 'Uint8Array') {
      this.colorBuffer = new Uint8Array(this.vertexCapacity * 3);
    } else if (types.color === 'Uint16Array') {
      this.colorBuffer = new Uint16Array(this.vertexCapacity * 3);
    }
  }

  /**
   * Ensure capacity for both vertices and segments.
   *
   * @param vertexCount - Actual vertex count needed
   * @param segmentCount - Optional actual segment count (if not provided, estimates from vertex count)
   * @returns true if buffers grew
   *
   * IMPORTANT: For particle tracks (N vertices, N-1 segments), the ratio is ~1:1 not 1.5:1.
   * Always pass actual segment count when known to avoid silent buffer truncation!
   */
  ensureCapacity(vertexCount: number, segmentCount?: number): boolean {
    const neededVertices = vertexCount;
    // Use actual segment count if provided, otherwise estimate (may be too small!)
    const neededSegments = segmentCount ?? Math.ceil(vertexCount / 1.5);

    let grew = false;

    // Grow vertices if needed (widths grow with vertices!)
    if (neededVertices > this.vertexCapacity) {
      let newVertexCap = this.vertexCapacity;
      while (newVertexCap < neededVertices) {
        newVertexCap = Math.ceil(newVertexCap * 1.5);
      }

      log.info(
        Modules.DATA_ACCUMULATOR,
        `Growing LinesDataAccumulator: ${this.vertexCapacity} → ${newVertexCap} vertices`
      );

      const newVertexBuf = new Float32Array(newVertexCap * this.ndim);
      const newWidthBuf = new Float32Array(newVertexCap); // PER-VERTEX!
      const newSharpnessBuf = new Float32Array(newVertexCap);

      newVertexBuf.set(this.vertexBuffer);
      newWidthBuf.set(this.widthBuffer); // widths grow with vertices
      newSharpnessBuf.set(this.sharpnessBuffer);

      this.vertexBuffer = newVertexBuf;
      this.widthBuffer = newWidthBuf;
      this.sharpnessBuffer = newSharpnessBuf;

      // Color: Type-preserving growth (like Points accumulator)
      if (this.colorBuffer instanceof Uint8Array) {
        const newColorBuf = new Uint8Array(newVertexCap * 3);
        newColorBuf.set(this.colorBuffer);
        this.colorBuffer = newColorBuf;
      } else if (this.colorBuffer instanceof Uint16Array) {
        const newColorBuf = new Uint16Array(newVertexCap * 3);
        newColorBuf.set(this.colorBuffer);
        this.colorBuffer = newColorBuf;
      } else {
        const newColorBuf = new Float32Array(newVertexCap * 3);
        newColorBuf.set(this.colorBuffer);
        this.colorBuffer = newColorBuf;
      }

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
      positions: this.vertexBuffer.subarray(0, vertexCount * this.ndim),
      segments: this.segmentBuffer.subarray(0, segmentCount * 2),
      widths: this.widthBuffer.subarray(0, vertexCount), // PER-VERTEX! Not segmentCount!
      colors: this.hasColors ? this.colorBuffer.subarray(0, vertexCount * 3) : null,
      sharpness: this.hasSharpness ? this.sharpnessBuffer.subarray(0, vertexCount) : null,
      segmentCount,
      vertexCount,
      ndim: this.ndim,
    };
  }

  /**
   * Fill accumulator buffers with multi-type support
   *
   * Detects attribute types on first fill and creates appropriately-typed buffers.
   * Subsequent fills must use matching types. Preserves native types (no conversion).
   *
   * NOTE: widths uses vertexOffset (per-vertex), NOT segmentOffset!
   */
  fill(segmentOffset: number, vertexOffset: number, data: Partial<LoadedLinesData>): void {
    // Initialize types on first fill with colors
    if (data.colors) {
      this.initializeTypes(data);
    }

    if (data.positions) {
      this.vertexBuffer.set(data.positions, vertexOffset * this.ndim);
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
      // Multi-type support: set with correct type (no conversion!)
      if (data.colors instanceof Uint8Array) {
        (this.colorBuffer as Uint8Array).set(data.colors, vertexOffset * 3);
      } else if (data.colors instanceof Uint16Array) {
        (this.colorBuffer as Uint16Array).set(data.colors, vertexOffset * 3);
      } else {
        (this.colorBuffer as Float32Array).set(data.colors, vertexOffset * 3);
      }
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

  // --- Public accessors for direct buffer access (used by loaders) ---

  getVertexBuffer(): Float32Array {
    return this.vertexBuffer;
  }

  getSegmentBuffer(): Uint32Array {
    return this.segmentBuffer;
  }

  getWidthBuffer(): Float32Array {
    return this.widthBuffer;
  }

  getColorBuffer(): Float32Array | Uint8Array | Uint16Array {
    return this.colorBuffer;
  }

  getSharpnessBuffer(): Float32Array {
    return this.sharpnessBuffer;
  }

  dispose(): void {
    this.vertexBuffer = new Float32Array(0);
    this.segmentBuffer = new Uint32Array(0);
    this.widthBuffer = new Float32Array(0);
    this.colorBuffer = new Float32Array(0);
    this.sharpnessBuffer = new Float32Array(0);
    this.vertexCapacity = 0;
    this.segmentCapacity = 0;
    this.types = null;
    this.hasColors = false;
    this.hasSharpness = false;
  }
}

/**
 * Attribute types for GSplats accumulator (matches GPU buffer pool pattern)
 */
export interface GSplatsAccumulatorTypes {
  position: 'Float32Array';
  amplitude: 'Float32Array';
  cholesky: 'Float32Array';
  color: 'Float32Array' | 'Uint8Array' | 'Uint16Array';
}

/**
 * GSplats data accumulator with FULL multi-type support
 *
 * CRITICAL FIX: camelCase choleskyFactors (not snake_case cholesky_factors)
 *
 * Handles Float32Array, Uint8Array, and Uint16Array for colors natively (no conversion)
 * to maintain memory efficiency and consistency with multi-type GPU buffer pool.
 *
 * Type is detected on first fill() and remains fixed for the accumulator's lifetime.
 */
export class GSplatsDataAccumulator implements DataAccumulator<
  LoadedGSplatsData,
  [number],
  [number, Partial<LoadedGSplatsData>]
> {
  private centerBuffer: Float32Array;
  private amplitudeBuffer: Float32Array;
  private choleskyBuffer: Float32Array; // CORRECT: for choleskyFactors field
  private colorBuffer: Float32Array | Uint8Array | Uint16Array; // RGB (multi-type!)

  // Type tracking (like Points accumulator)
  private types: GSplatsAccumulatorTypes | null = null;

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

    // Start with Float32Array (will be replaced on first fill with actual types)
    this.centerBuffer = new Float32Array(initialCapacity * ndim);
    this.amplitudeBuffer = new Float32Array(initialCapacity);
    this.choleskyBuffer = new Float32Array(initialCapacity * this.choleskySize);
    this.colorBuffer = new Float32Array(initialCapacity * 3); // RGB

    this.allocations++;
  }

  /**
   * Initialize types from first data fill (like Points accumulator)
   */
  private initializeTypes(data: Partial<LoadedGSplatsData>): void {
    if (this.types) return; // Already initialized

    const types: GSplatsAccumulatorTypes = {
      position: 'Float32Array',
      amplitude: 'Float32Array',
      cholesky: 'Float32Array',
      color:
        data.colors instanceof Uint8Array
          ? 'Uint8Array'
          : data.colors instanceof Uint16Array
            ? 'Uint16Array'
            : 'Float32Array',
    };

    this.types = types;

    // Recreate color buffer with correct type (only if types differ from initial Float32)
    if (types.color === 'Uint8Array') {
      this.colorBuffer = new Uint8Array(this.capacity * 3);
    } else if (types.color === 'Uint16Array') {
      this.colorBuffer = new Uint16Array(this.capacity * 3);
    }
  }

  ensureCapacity(needed: number): boolean {
    if (needed <= this.capacity) return false;

    let newCapacity = this.capacity;
    while (newCapacity < needed) {
      newCapacity = Math.ceil(newCapacity * 1.5);
    }

    log.info(
      Modules.DATA_ACCUMULATOR,
      `Growing GSplatsDataAccumulator: ${this.capacity} → ${newCapacity} splats`
    );

    const newCenters = new Float32Array(newCapacity * this.ndim);
    const newAmplitudes = new Float32Array(newCapacity);
    const newCholesky = new Float32Array(newCapacity * this.choleskySize);

    newCenters.set(this.centerBuffer);
    newAmplitudes.set(this.amplitudeBuffer);
    newCholesky.set(this.choleskyBuffer);

    this.centerBuffer = newCenters;
    this.amplitudeBuffer = newAmplitudes;
    this.choleskyBuffer = newCholesky;

    // Color: Type-preserving growth (like Points accumulator)
    if (this.colorBuffer instanceof Uint8Array) {
      const newColors = new Uint8Array(newCapacity * 3);
      newColors.set(this.colorBuffer);
      this.colorBuffer = newColors;
    } else if (this.colorBuffer instanceof Uint16Array) {
      const newColors = new Uint16Array(newCapacity * 3);
      newColors.set(this.colorBuffer);
      this.colorBuffer = newColors;
    } else {
      const newColors = new Float32Array(newCapacity * 3);
      newColors.set(this.colorBuffer);
      this.colorBuffer = newColors;
    }

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
      positions: this.centerBuffer.subarray(0, count * this.ndim),
      amplitudes: this.amplitudeBuffer.subarray(0, count),
      choleskyFactors: this.choleskyBuffer.subarray(0, count * this.choleskySize), // CORRECT: camelCase!
      colors: this.hasColors ? this.colorBuffer.subarray(0, count * 3) : null, // Nullable based on data presence
      splatCount: count,
      ndim: this.ndim,
    };
  }

  /**
   * Fill accumulator buffers with multi-type support
   *
   * Detects attribute types on first fill and creates appropriately-typed buffers.
   * Subsequent fills must use matching types. Preserves native types (no conversion).
   */
  fill(offset: number, data: Partial<LoadedGSplatsData>): void {
    // Initialize types on first fill with colors
    if (data.colors) {
      this.initializeTypes(data);
    }

    if (data.positions) {
      this.centerBuffer.set(data.positions, offset * this.ndim);
    }
    if (data.amplitudes) {
      this.amplitudeBuffer.set(data.amplitudes, offset);
    }
    if (data.choleskyFactors) {
      // CORRECT: camelCase!
      this.choleskyBuffer.set(data.choleskyFactors, offset * this.choleskySize);
    }
    if (data.colors) {
      this.hasColors = true; // Mark as present
      // Multi-type support: set with correct type (no conversion!)
      if (data.colors instanceof Uint8Array) {
        (this.colorBuffer as Uint8Array).set(data.colors, offset * 3);
      } else if (data.colors instanceof Uint16Array) {
        (this.colorBuffer as Uint16Array).set(data.colors, offset * 3);
      } else {
        (this.colorBuffer as Float32Array).set(data.colors, offset * 3);
      }
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

  // --- Public accessors for direct buffer access (used by loaders) ---

  getCenterBuffer(): Float32Array {
    return this.centerBuffer;
  }

  getAmplitudeBuffer(): Float32Array {
    return this.amplitudeBuffer;
  }

  getCholeskyBuffer(): Float32Array {
    return this.choleskyBuffer;
  }

  getColorBuffer(): Float32Array | Uint8Array | Uint16Array {
    return this.colorBuffer;
  }

  setColorBuffer(buffer: Float32Array | Uint8Array | Uint16Array): void {
    this.colorBuffer = buffer;
  }

  dispose(): void {
    this.centerBuffer = new Float32Array(0);
    this.amplitudeBuffer = new Float32Array(0);
    this.choleskyBuffer = new Float32Array(0);
    this.colorBuffer = new Float32Array(0);
    this.capacity = 0;
    this.types = null;
    this.hasColors = false;
  }
}
