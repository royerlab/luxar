/**
 * Data Accumulators — multi-type object pooling for Points, Lines, and
 * GSplats.
 *
 * Implements persistent TypedArray buffers that grow by 1.5x when
 * needed, eliminating per-frame allocations and reducing GC pressure.
 * Supports Float32Array, Uint8Array, and Uint16Array natively for
 * memory efficiency consistent with the multi-type GPU buffer pool.
 *
 * All three loaders use deep integration for the LOADING path:
 *   - write directly to accumulator buffers during data loading,
 *   - return zero-copy subarrays from `accumulator.getData()`,
 *   - achieve zero-allocation operation in the loading hot path.
 *
 * Processing helpers (projectLinesTo3D, projectGSplats) are kept on
 * their two-pass algorithms — accumulator integration there did not
 * pay off in benchmarks.
 */

import * as THREE from 'three';
import type {
  LoadedPointsData,
  PositionArray,
  ColorArray,
  ScalarArray,
} from '../data-loader-types';
import type { LoadedLinesData } from '../../types/lines';
import type { LoadedGSplatsData } from '../../types/gsplats';
import { log, Modules } from '../../utils/log';

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
  /**
   * per-point scalar values for colormap lookup. Optional —
   * unset when the first fill carried no `scalars` (most datasets).
   * Float16Array is tracked separately so the accumulator can keep
   * native dtype until it's widened at the GPU upload site.
   */
  scalar?: 'Float32Array' | 'Float16Array' | 'Uint8Array';
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
  /** per-point scalar buffer for colormap lookup. */
  private scalarBuffer: Float32Array | Uint8Array;

  // Type tracking
  private types: PointsAccumulatorTypes | null = null;

  // Track which attributes have been filled
  private hasColors = false;
  private hasRadii = false;
  private hasSharpness = false;
  /** tracks whether scalars were ever filled this session. */
  private hasScalars = false;
  /**
   * B.6: flips to true on `dispose()`. Getters/fill/getData throw a
   * descriptive error if called after disposal so caller bugs don't
   * silently operate on the empty-buffer state.
   */
  private _disposed = false;
  /**
   * C.2: highest point index touched by any `fill()` call. Lets
   * `ensureCapacity()` copy only the live prefix into the new buffers
   * instead of the full capacity — when growing 1024 → 1536 after
   * filling 800 points, the position copy goes from 12288 floats down
   * to 2400 floats, a 5× reduction on the hot loading path.
   */
  private usedCount = 0;

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
    // C.4: scalars are an optional attribute on most datasets. Start
    // with an empty sentinel buffer so a non-scalar load doesn't reserve
    // `initialCapacity * 4 B` up front. `fill()` allocates on first use.
    this.scalarBuffer = new Float32Array(0);

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
    this.assertNotDisposed('ensureCapacity');
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

    // C.2: copy only the live prefix (usedCount * stride). When the
    // accumulator is fresh or sparsely filled this is dramatically
    // cheaper than copying the full old buffer; for full accumulators
    // it's identical work since usedCount === capacity.
    const liveCount = Math.min(this.usedCount, this.capacity);
    const livePos = liveCount * 3;
    const liveColor = liveCount * 3;
    const liveScalar = liveCount;

    // Allocate new buffers with SAME types as current (type-preserving growth)
    const newPositions = new Float32Array(newCapacity * 3);
    newPositions.set(this.positionBuffer.subarray(0, livePos));
    this.positionBuffer = newPositions;

    // Color: Type-preserving
    if (this.colorBuffer instanceof Uint8Array) {
      const newColors = new Uint8Array(newCapacity * 3);
      newColors.set(this.colorBuffer.subarray(0, liveColor));
      this.colorBuffer = newColors;
    } else if (this.colorBuffer instanceof Uint16Array) {
      const newColors = new Uint16Array(newCapacity * 3);
      newColors.set(this.colorBuffer.subarray(0, liveColor));
      this.colorBuffer = newColors;
    } else {
      const newColors = new Float32Array(newCapacity * 3);
      newColors.set(this.colorBuffer.subarray(0, liveColor));
      this.colorBuffer = newColors;
    }

    // Radius: Type-preserving
    if (this.radiiBuffer instanceof Uint8Array) {
      const newRadii = new Uint8Array(newCapacity);
      newRadii.set(this.radiiBuffer.subarray(0, liveCount));
      this.radiiBuffer = newRadii;
    } else {
      const newRadii = new Float32Array(newCapacity);
      newRadii.set(this.radiiBuffer.subarray(0, liveCount));
      this.radiiBuffer = newRadii;
    }

    // Sharpness: Type-preserving
    if (this.sharpnessBuffer instanceof Uint8Array) {
      const newSharpness = new Uint8Array(newCapacity);
      newSharpness.set(this.sharpnessBuffer.subarray(0, liveCount));
      this.sharpnessBuffer = newSharpness;
    } else {
      const newSharpness = new Float32Array(newCapacity);
      newSharpness.set(this.sharpnessBuffer.subarray(0, liveCount));
      this.sharpnessBuffer = newSharpness;
    }

    // Scalar — type-preserving growth. C.4: skip growth when the
    // accumulator never saw scalars (buffer stayed at sentinel size 0);
    // initializeTypes will allocate at the new capacity on first
    // scalar fill.
    if (this.scalarBuffer.length > 0) {
      if (this.scalarBuffer instanceof Uint8Array) {
        const newScalars = new Uint8Array(newCapacity);
        newScalars.set(this.scalarBuffer.subarray(0, liveScalar));
        this.scalarBuffer = newScalars;
      } else {
        const newScalars = new Float32Array(newCapacity);
        newScalars.set(this.scalarBuffer.subarray(0, liveScalar));
        this.scalarBuffer = newScalars;
      }
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
      // scalars optional — populated when the source data carried them.
      scalars: this.hasScalars ? (this.scalarBuffer.subarray(0, count) as ScalarArray) : undefined,
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
          scalars: this.types?.scalar === 'Uint8Array' ? 'uint8' : 'float32',
        },
      },
    };
  }

  /**
   * Detect and initialize buffer types on first fill
   */
  private initializeTypes(data: Partial<LoadedPointsData>): void {
    // C.4: even after the initial color/radius/sharpness types have
    // been pinned by a prior fill, a later fill might be the first to
    // carry scalars — in which case we lazily allocate the scalar
    // buffer here without disturbing the other type pins.
    if (this.types && this.types.scalar === undefined && data.scalars) {
      let scalarType: 'Float32Array' | 'Float16Array' | 'Uint8Array';
      if (data.scalars instanceof Uint8Array) {
        scalarType = 'Uint8Array';
      } else if (
        typeof globalThis.Float16Array !== 'undefined' &&
        data.scalars instanceof globalThis.Float16Array
      ) {
        scalarType = 'Float16Array';
      } else {
        scalarType = 'Float32Array';
      }
      this.types.scalar = scalarType;
      this.scalarBuffer =
        scalarType === 'Uint8Array'
          ? new Uint8Array(this.capacity)
          : new Float32Array(this.capacity);
      return;
    }
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
    if (data.scalars) {
      if (data.scalars instanceof Uint8Array) {
        types.scalar = 'Uint8Array';
      } else if (
        typeof globalThis.Float16Array !== 'undefined' &&
        data.scalars instanceof globalThis.Float16Array
      ) {
        types.scalar = 'Float16Array';
      } else {
        types.scalar = 'Float32Array';
      }
    }

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

    // C.4: lazily allocate the scalar buffer on first sight of scalars
    // (sized to current capacity). When `types.scalar` is undefined the
    // buffer stays at length 0.
    if (types.scalar === 'Uint8Array') {
      this.scalarBuffer = new Uint8Array(this.capacity);
    } else if (types.scalar) {
      this.scalarBuffer = new Float32Array(this.capacity);
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
    this.assertNotDisposed('fill');
    // Initialize types on first fill
    this.initializeTypes(data);

    // C.2: track the highest filled index for cheap ensureCapacity copies.
    const filledCount = data.positions
      ? data.positions.length / 3
      : data.colors
        ? data.colors.length / 3
        : (data.radii?.length ?? data.sharpness?.length ?? data.scalars?.length ?? 0);
    if (filledCount > 0) {
      this.usedCount = Math.max(this.usedCount, offset + filledCount);
    }

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
    // scalars copied with native type (no conversion).
    if (data.scalars) {
      this.hasScalars = true;
      if (data.scalars instanceof Uint8Array) {
        (this.scalarBuffer as Uint8Array).set(data.scalars, offset);
      } else {
        (this.scalarBuffer as Float32Array).set(data.scalars as Float32Array, offset);
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

  /** B.6: introspection — true if dispose() has been called. */
  isDisposed(): boolean {
    return this._disposed;
  }

  /**
   * B.6: throws if a mutating call lands on a disposed accumulator.
   * Read-only getters return the (empty) buffers so existing post-
   * dispose "did we wipe?" assertions keep working without raising.
   */
  private assertNotDisposed(method: string): void {
    if (this._disposed) {
      throw new Error(
        `LoadedPointsDataAccumulator.${method}() called after dispose() — accumulator is no longer usable.`
      );
    }
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

  /**
   * Direct accessor for the scalar buffer (used by loaders writing through).
   *
   * C.4: the accumulator starts with a zero-length scalar buffer to
   * avoid reserving 4 B/point on the common no-scalars path. The first
   * `getScalarBuffer()` call allocates to current capacity so the
   * spatial-index loader's direct-write path (`loadVertexRanges` →
   * `scalarBuffer`) writes into a real buffer.
   */
  getScalarBuffer(): Float32Array | Uint8Array {
    if (this.scalarBuffer.length === 0 && this.capacity > 0 && !this._disposed) {
      this.scalarBuffer = new Float32Array(this.capacity);
    }
    return this.scalarBuffer;
  }

  dispose(): void {
    // B.6: zero-length sentinels keep the field types non-nullable
    // (avoiding ?-checks at every internal read), and the small backing
    // ArrayBuffers are GC-eligible once the accumulator itself drops.
    // The `_disposed` flag is the real signal — getters/fill throw.
    this.positionBuffer = new Float32Array(0);
    this.colorBuffer = new Float32Array(0);
    this.radiiBuffer = new Float32Array(0);
    this.sharpnessBuffer = new Float32Array(0);
    this.scalarBuffer = new Float32Array(0);
    this.capacity = 0;
    this.usedCount = 0;
    this.types = null;
    this.hasColors = false;
    this.hasRadii = false;
    this.hasSharpness = false;
    this.hasScalars = false;
    this._disposed = true;
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
  /**
   * A.2: optional scalar dtype. Mirrors `PointsAccumulatorTypes.scalar`
   * so Uint8 scalars stay in Uint8 storage until projection widens
   * them. Omitted when the first fill carried no scalars (most
   * datasets).
   */
  scalar?: 'Float32Array' | 'Float16Array' | 'Uint8Array';
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
  /**
   * A.2: per-vertex scalar buffer for colormap lookup. Allocated as
   * Float32Array initially; replaced with Uint8Array on first fill if
   * the input scalars are Uint8. Float16 input stays in the default
   * Float32 buffer (widened at fill time via the typed-array `.set()`
   * numeric conversion).
   */
  private scalarBuffer: Float32Array | Uint8Array;

  // Type tracking (like Points accumulator)
  private types: LinesAccumulatorTypes | null = null;

  // Track whether data actually has colors/sharpness
  private hasColors = false;
  private hasSharpness = false;
  /** tracks whether scalars were ever filled this session. */
  private hasScalars = false;
  /** B.6: flips to true on dispose(); fill/ensureCapacity then throw. */
  private _disposed = false;
  /**
   * C.2: highest vertex / segment indices touched by any `fill()` call.
   * Lets `ensureCapacity()` copy only the live prefix into the new
   * buffers instead of the full capacity.
   */
  private usedVertexCount = 0;
  private usedSegmentCount = 0;

  private vertexCapacity: number;
  private segmentCapacity: number;
  private ndim: number;

  private allocations = 0;
  private totalGrowths = 0;

  private assertNotDisposed(method: string): void {
    if (this._disposed) {
      throw new Error(
        `LinesDataAccumulator.${method}() called after dispose() — accumulator is no longer usable.`
      );
    }
  }

  /** B.6: introspection — true if dispose() has been called. */
  isDisposed(): boolean {
    return this._disposed;
  }

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
    // C.4: scalars are an optional per-vertex attribute. Start with an
    // empty sentinel buffer so non-scalar line datasets don't reserve
    // `initialVertexCapacity * 4 B` up front. `initializeTypes` allocates
    // it on first scalar fill.
    this.scalarBuffer = new Float32Array(0);

    this.allocations++;
  }

  /**
   * Initialize types from first data fill (like Points accumulator)
   */
  private initializeTypes(data: Partial<LoadedLinesData>): void {
    // C.4: a later fill might be the first to carry scalars; allocate
    // the scalar buffer at the current vertex capacity without re-pinning
    // the color/width/sharpness types.
    if (this.types && this.types.scalar === undefined && data.scalars) {
      let scalarType: 'Float32Array' | 'Float16Array' | 'Uint8Array';
      if (data.scalars instanceof Uint8Array) {
        scalarType = 'Uint8Array';
      } else if (
        typeof globalThis.Float16Array !== 'undefined' &&
        data.scalars instanceof globalThis.Float16Array
      ) {
        scalarType = 'Float16Array';
      } else {
        scalarType = 'Float32Array';
      }
      this.types.scalar = scalarType;
      this.scalarBuffer =
        scalarType === 'Uint8Array'
          ? new Uint8Array(this.vertexCapacity)
          : new Float32Array(this.vertexCapacity);
      return;
    }
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
    if (data.scalars) {
      if (data.scalars instanceof Uint8Array) {
        types.scalar = 'Uint8Array';
      } else if (
        typeof globalThis.Float16Array !== 'undefined' &&
        data.scalars instanceof globalThis.Float16Array
      ) {
        types.scalar = 'Float16Array';
      } else {
        types.scalar = 'Float32Array';
      }
    }

    this.types = types;

    // Recreate color buffer with correct type (only if types differ from initial Float32)
    if (types.color === 'Uint8Array') {
      this.colorBuffer = new Uint8Array(this.vertexCapacity * 3);
    } else if (types.color === 'Uint16Array') {
      this.colorBuffer = new Uint16Array(this.vertexCapacity * 3);
    }

    // C.4 / A.2: lazily allocate scalar buffer when first scalar fill is
    // observed. Sized to current vertex capacity. Skipped when there
    // are no scalars (`types.scalar === undefined`).
    if (types.scalar === 'Uint8Array') {
      this.scalarBuffer = new Uint8Array(this.vertexCapacity);
    } else if (types.scalar) {
      this.scalarBuffer = new Float32Array(this.vertexCapacity);
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
    this.assertNotDisposed('ensureCapacity');
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

      // C.2: copy only the live prefix.
      const liveVerts = Math.min(this.usedVertexCount, this.vertexCapacity);
      const liveVertexFloats = liveVerts * this.ndim;
      const liveColorFloats = liveVerts * 3;

      const newVertexBuf = new Float32Array(newVertexCap * this.ndim);
      const newWidthBuf = new Float32Array(newVertexCap); // PER-VERTEX!
      const newSharpnessBuf = new Float32Array(newVertexCap);

      newVertexBuf.set(this.vertexBuffer.subarray(0, liveVertexFloats));
      newWidthBuf.set(this.widthBuffer.subarray(0, liveVerts)); // widths grow with vertices
      newSharpnessBuf.set(this.sharpnessBuffer.subarray(0, liveVerts));

      this.vertexBuffer = newVertexBuf;
      this.widthBuffer = newWidthBuf;
      this.sharpnessBuffer = newSharpnessBuf;

      // A.2 / C.4: scalar buffer grows only when allocated; the
      // sentinel-empty (length 0) buffer stays empty until the first
      // scalar fill, at which point initializeTypes sizes it.
      if (this.scalarBuffer.length > 0) {
        if (this.scalarBuffer instanceof Uint8Array) {
          const newScalarBuf = new Uint8Array(newVertexCap);
          newScalarBuf.set(this.scalarBuffer.subarray(0, liveVerts));
          this.scalarBuffer = newScalarBuf;
        } else {
          const newScalarBuf = new Float32Array(newVertexCap);
          newScalarBuf.set(this.scalarBuffer.subarray(0, liveVerts));
          this.scalarBuffer = newScalarBuf;
        }
      }

      // Color: Type-preserving growth (like Points accumulator)
      if (this.colorBuffer instanceof Uint8Array) {
        const newColorBuf = new Uint8Array(newVertexCap * 3);
        newColorBuf.set(this.colorBuffer.subarray(0, liveColorFloats));
        this.colorBuffer = newColorBuf;
      } else if (this.colorBuffer instanceof Uint16Array) {
        const newColorBuf = new Uint16Array(newVertexCap * 3);
        newColorBuf.set(this.colorBuffer.subarray(0, liveColorFloats));
        this.colorBuffer = newColorBuf;
      } else {
        const newColorBuf = new Float32Array(newVertexCap * 3);
        newColorBuf.set(this.colorBuffer.subarray(0, liveColorFloats));
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

      // C.2: copy only the live prefix of segment-index pairs.
      const liveSegments = Math.min(this.usedSegmentCount, this.segmentCapacity);
      const newSegmentBuf = new Uint32Array(newSegmentCap * 2);
      newSegmentBuf.set(this.segmentBuffer.subarray(0, liveSegments * 2));

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
      // per-vertex scalars now flow through the accumulator path.
      // Omitted (undefined) when not present to match the new
      // optional `LoadedLinesData.scalars?` shape.
      ...(this.hasScalars ? { scalars: this.scalarBuffer.subarray(0, vertexCount) } : {}),
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
    this.assertNotDisposed('fill');
    // Initialize types on first fill with colors or scalars
    if (data.colors || data.scalars) {
      this.initializeTypes(data);
    }

    // C.2: track live prefixes for cheap ensureCapacity copies.
    if (data.positions) {
      const filledVerts = data.positions.length / this.ndim;
      this.usedVertexCount = Math.max(this.usedVertexCount, vertexOffset + filledVerts);
    } else if (data.widths) {
      this.usedVertexCount = Math.max(this.usedVertexCount, vertexOffset + data.widths.length);
    }
    if (data.segments) {
      const filledSegs = data.segments.length / 2;
      this.usedSegmentCount = Math.max(this.usedSegmentCount, segmentOffset + filledSegs);
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
    // A.2: copy scalars into per-vertex buffer preserving native dtype
    // (Uint8 stays Uint8, Float16 widens to Float32 via numeric .set()).
    if (data.scalars) {
      this.hasScalars = true;
      if (data.scalars instanceof Uint8Array) {
        (this.scalarBuffer as Uint8Array).set(data.scalars, vertexOffset);
      } else {
        // Float32Array and Float16Array both target the Float32 buffer.
        // TypedArray.set widens Float16 → Float32 numerically.
        (this.scalarBuffer as Float32Array).set(
          data.scalars as unknown as ArrayLike<number>,
          vertexOffset
        );
      }
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

  /**
   * Direct accessor for the per-vertex scalar buffer.
   *
   * C.4: lazy-allocated on first access. Non-scalar line datasets keep
   * `scalarBuffer.length === 0` indefinitely, saving 4 B/vertex.
   */
  getScalarBuffer(): Float32Array | Uint8Array {
    if (this.scalarBuffer.length === 0 && this.vertexCapacity > 0 && !this._disposed) {
      this.scalarBuffer = new Float32Array(this.vertexCapacity);
    }
    return this.scalarBuffer;
  }

  /**
   * mark scalars as present. Used by loaders that write directly
   * into `scalarBuffer` via `getScalarBuffer()` (zero-copy path) and
   * therefore need to flip `hasScalars` without going through `fill()`.
   */
  markScalarsLoaded(): void {
    this.hasScalars = true;
  }

  dispose(): void {
    this.vertexBuffer = new Float32Array(0);
    this.segmentBuffer = new Uint32Array(0);
    this.widthBuffer = new Float32Array(0);
    this.colorBuffer = new Float32Array(0);
    this.sharpnessBuffer = new Float32Array(0);
    this.scalarBuffer = new Float32Array(0);
    this.vertexCapacity = 0;
    this.segmentCapacity = 0;
    this.usedVertexCount = 0;
    this.usedSegmentCount = 0;
    this.types = null;
    this.hasColors = false;
    this.hasSharpness = false;
    this.hasScalars = false;
    this._disposed = true;
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
  /** B.6: flips to true on dispose(); fill/ensureCapacity then throw. */
  private _disposed = false;
  /** C.2: highest splat index touched by any `fill()` call. */
  private usedCount = 0;

  private capacity: number;
  private ndim: number;
  private choleskySize: number; // Elements per splat

  private allocations = 0;
  private totalGrowths = 0;

  private assertNotDisposed(method: string): void {
    if (this._disposed) {
      throw new Error(
        `LoadedGSplatsDataAccumulator.${method}() called after dispose() — accumulator is no longer usable.`
      );
    }
  }

  /** B.6: introspection — true if dispose() has been called. */
  isDisposed(): boolean {
    return this._disposed;
  }

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
    this.assertNotDisposed('ensureCapacity');
    if (needed <= this.capacity) return false;

    let newCapacity = this.capacity;
    while (newCapacity < needed) {
      newCapacity = Math.ceil(newCapacity * 1.5);
    }

    log.info(
      Modules.DATA_ACCUMULATOR,
      `Growing GSplatsDataAccumulator: ${this.capacity} → ${newCapacity} splats`
    );

    // C.2: copy only the live prefix.
    const live = Math.min(this.usedCount, this.capacity);
    const liveCenterFloats = live * this.ndim;
    const liveColorFloats = live * 3;
    const liveCholeskyFloats = live * this.choleskySize;

    const newCenters = new Float32Array(newCapacity * this.ndim);
    const newAmplitudes = new Float32Array(newCapacity);
    const newCholesky = new Float32Array(newCapacity * this.choleskySize);

    newCenters.set(this.centerBuffer.subarray(0, liveCenterFloats));
    newAmplitudes.set(this.amplitudeBuffer.subarray(0, live));
    newCholesky.set(this.choleskyBuffer.subarray(0, liveCholeskyFloats));

    this.centerBuffer = newCenters;
    this.amplitudeBuffer = newAmplitudes;
    this.choleskyBuffer = newCholesky;

    // Color: Type-preserving growth (like Points accumulator)
    if (this.colorBuffer instanceof Uint8Array) {
      const newColors = new Uint8Array(newCapacity * 3);
      newColors.set(this.colorBuffer.subarray(0, liveColorFloats));
      this.colorBuffer = newColors;
    } else if (this.colorBuffer instanceof Uint16Array) {
      const newColors = new Uint16Array(newCapacity * 3);
      newColors.set(this.colorBuffer.subarray(0, liveColorFloats));
      this.colorBuffer = newColors;
    } else {
      const newColors = new Float32Array(newCapacity * 3);
      newColors.set(this.colorBuffer.subarray(0, liveColorFloats));
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
    this.assertNotDisposed('fill');
    // Initialize types on first fill with colors
    if (data.colors) {
      this.initializeTypes(data);
    }

    // C.2: track live prefix for cheap ensureCapacity copies.
    const filledCount = data.positions
      ? data.positions.length / this.ndim
      : (data.amplitudes?.length ?? 0);
    if (filledCount > 0) {
      this.usedCount = Math.max(this.usedCount, offset + filledCount);
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
    this.usedCount = 0;
    this.types = null;
    this.hasColors = false;
    this._disposed = true;
  }
}
