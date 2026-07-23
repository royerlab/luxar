/**
 * Points data accumulator — multi-type object pooling for Points geometry.
 *
 * Implements persistent TypedArray buffers that grow by 1.5x when
 * needed, eliminating per-frame allocations and reducing GC pressure.
 * Supports Float32Array, Uint8Array, and Uint16Array natively for
 * memory efficiency consistent with the multi-type GPU buffer pool.
 *
 * The loader's deep-integration LOADING path writes directly to these
 * buffers and returns zero-copy subarrays from `getData()`.
 */

import * as THREE from 'three';
import type {
  LoadedPointsData,
  PositionArray,
  ColorArray,
  ScalarArray,
} from '../data-loader-types';
import { log, Modules } from '../../utils/log';
import type { DataAccumulator, AccumulatorStats } from './types';

export type { AccumulatorStats } from './types';

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
   * Flips to true on `dispose()`. Getters/fill/getData throw a
   * descriptive error if called after disposal so caller bugs don't
   * silently operate on the empty-buffer state.
   */
  private _disposed = false;
  /**
   * Highest point index touched by any `fill()` call. Lets
   * `ensureCapacity()` copy only the live prefix into the new buffers
   * instead of the full capacity — when growing 1024 → 1536 after
   * filling 800 points, the position copy goes from 12288 floats down
   * to 2400 floats, a 5× reduction on the hot loading path.
   */
  private usedCount = 0;

  /** Components per color item: 3 (RGB) or 4 (RGBA — alpha = per-point opacity). */
  private colorComponents: 3 | 4 = 3;

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
    this.colorBuffer = new Float32Array(initialCapacity * 3); // RGB(A) — see configureColorComponents
    this.radiiBuffer = new Float32Array(initialCapacity);
    this.sharpnessBuffer = new Float32Array(initialCapacity);
    // Scalars are optional on most datasets. Start with an empty sentinel
    // buffer so a non-scalar load doesn't reserve `initialCapacity * 4 B`
    // up front. `fill()` allocates on first use.
    this.scalarBuffer = new Float32Array(0);

    this.allocations++;
  }

  /**
   * Declare the color layout (3 = RGB, 4 = RGBA) BEFORE the first color
   * fill. The loader knows the layout from the zarr array shape at open
   * time; the accumulator needs it to size every color buffer. Throws if
   * colors were already written with a different layout — the layout is a
   * property of the dataset, not of an individual load. Mirrors
   * `LoadedGSplatsDataAccumulator.configureColorComponents`.
   */
  configureColorComponents(components: 3 | 4): void {
    this.assertNotDisposed('configureColorComponents');
    if (components === this.colorComponents) return;
    if (this.hasColors) {
      throw new Error(
        `LoadedPointsDataAccumulator: color layout changed to ${components} components after ` +
          `colors were already written with ${this.colorComponents}`
      );
    }
    this.colorComponents = components;
    // Re-size the (still empty) color buffer, preserving its element type.
    const n = this.capacity * components;
    this.colorBuffer =
      this.colorBuffer instanceof Uint8Array
        ? new Uint8Array(n)
        : this.colorBuffer instanceof Uint16Array
          ? new Uint16Array(n)
          : new Float32Array(n);
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

    // Copy only the live prefix (usedCount * stride). When the
    // accumulator is fresh or sparsely filled this is dramatically
    // cheaper than copying the full previous buffer; for full accumulators
    // it's identical work since usedCount === capacity.
    const liveCount = Math.min(this.usedCount, this.capacity);
    const livePos = liveCount * 3;
    const liveColor = liveCount * this.colorComponents;
    const liveScalar = liveCount;

    // Allocate new buffers with SAME types as current (type-preserving growth)
    const newPositions = new Float32Array(newCapacity * 3);
    newPositions.set(this.positionBuffer.subarray(0, livePos));
    this.positionBuffer = newPositions;

    // Color: Type-preserving (strided by the declared RGB(A) layout)
    if (this.colorBuffer instanceof Uint8Array) {
      const newColors = new Uint8Array(newCapacity * this.colorComponents);
      newColors.set(this.colorBuffer.subarray(0, liveColor));
      this.colorBuffer = newColors;
    } else if (this.colorBuffer instanceof Uint16Array) {
      const newColors = new Uint16Array(newCapacity * this.colorComponents);
      newColors.set(this.colorBuffer.subarray(0, liveColor));
      this.colorBuffer = newColors;
    } else {
      const newColors = new Float32Array(newCapacity * this.colorComponents);
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

    // Scalar — type-preserving growth. Skip growth when the accumulator
    // never saw scalars (buffer stayed at sentinel size 0); initializeTypes
    // will allocate at the new capacity on first scalar fill.
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
      colors: this.hasColors
        ? (this.colorBuffer.subarray(0, count * this.colorComponents) as ColorArray)
        : undefined,
      colorComponents: this.colorComponents,
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
          // [integration.md OOS4] Float16 scalars must self-report as
          // 'float16' so consumers can branch on the actual on-disk dtype.
          // Pre-fix, every non-Uint8 scalar tagged as 'float32', silently
          // hiding Float16 data even though `PointsAccumulatorTypes.scalar`
          // explicitly tracks 'Float16Array' separately.
          scalars:
            this.types?.scalar === 'Uint8Array'
              ? 'uint8'
              : this.types?.scalar === 'Float16Array'
                ? 'float16'
                : 'float32',
        },
      },
    };
  }

  /**
   * Detect and initialize buffer types on first fill
   */
  private initializeTypes(data: Partial<LoadedPointsData>): void {
    // Even after the initial color/radius/sharpness types have been pinned
    // by a prior fill, a later fill might be the first to carry scalars —
    // in which case we lazily allocate the scalar buffer here without
    // disturbing the other type pins.
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
      this.colorBuffer = new Uint8Array(this.capacity * this.colorComponents);
    } else if (types.color === 'Uint16Array') {
      this.colorBuffer = new Uint16Array(this.capacity * this.colorComponents);
    }

    if (types.radius === 'Uint8Array') {
      this.radiiBuffer = new Uint8Array(this.capacity);
    }

    if (types.sharpness === 'Uint8Array') {
      this.sharpnessBuffer = new Uint8Array(this.capacity);
    }

    // Lazily allocate the scalar buffer on first sight of scalars (sized
    // to current capacity). When `types.scalar` is undefined the buffer
    // stays at length 0.
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

    // Track the highest filled index for cheap ensureCapacity copies.
    // Prefer 1-per-point sources (radii, sharpness, scalars) over the
    // multi-channel color buffer so the count is correct regardless of
    // the RGB vs RGBA layout. Positions are still checked first because
    // they're the canonical 3-floats-per-point attribute on this code
    // path. Only fall through to colors / colorComponents when no
    // 1-per-point source is available.
    const filledCount = data.positions
      ? data.positions.length / 3
      : (data.radii?.length ??
        data.sharpness?.length ??
        data.scalars?.length ??
        (data.colors ? data.colors.length / this.colorComponents : 0));
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
        (this.colorBuffer as Uint8Array).set(data.colors, offset * this.colorComponents);
      } else if (data.colors instanceof Uint16Array) {
        (this.colorBuffer as Uint16Array).set(data.colors, offset * this.colorComponents);
      } else {
        (this.colorBuffer as Float32Array).set(data.colors, offset * this.colorComponents);
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
    // Bounds are intentionally NOT updated via updateMetadata — getData()
    // computes them fresh from positions. Warn loudly when a caller
    // passes `bounds` so they don't silently believe their update took
    // effect (e.g., during scene refresh after a coordinate transform).
    if (metadata.bounds !== undefined) {
      log.warning(
        Modules.DATA_ACCUMULATOR,
        'LoadedPointsDataAccumulator.updateMetadata: `bounds` argument ignored — ' +
          'getData() recomputes bounds from positions. Drop the bounds field from ' +
          'the call site or compute bounds before fill().'
      );
    }
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

  /** Introspection — true if dispose() has been called. */
  isDisposed(): boolean {
    return this._disposed;
  }

  /**
   * Throws if a mutating call lands on a disposed accumulator. Read-only
   * getters return the empty buffers so disposal assertions can inspect
   * the cleared state without raising.
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
   * The accumulator starts with a zero-length scalar buffer to avoid
   * reserving 4 B/point on the common no-scalars path. The first
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
    // Zero-length sentinels keep the field types non-nullable (avoiding
    // ?-checks at every internal read), and the small backing ArrayBuffers
    // are GC-eligible once the accumulator itself drops. The `_disposed`
    // flag is the real signal — getters/fill throw.
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
