/**
 * GSplats data accumulator — multi-type object pooling for Gaussian splats.
 *
 * Persistent typed-array buffers for splat centers, amplitudes, packed
 * Cholesky factors (camelCase `choleskyFactors`), and RGB color. Grows
 * by 1.5×, never past the count needed (`./growth.ts`);
 * Uint8/Uint16/Float32 colors handled natively.
 *
 * The loader's deep-integration LOADING path writes directly to these
 * buffers and returns zero-copy subarrays from `getData()`.
 */

import type { LoadedGSplatsData } from '../../types/gsplats';
import { log, Modules } from '../../utils/log';
import type { DataAccumulator, AccumulatorStats } from './types';
import { nextCapacity } from './growth';

export type { AccumulatorStats } from './types';

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
  /** Flips to true on dispose(); fill/ensureCapacity then throw. */
  private _disposed = false;
  /** Highest splat index touched by any `fill()` call. */
  private usedCount = 0;

  private capacity: number;
  private ndim: number;
  private choleskySize: number; // Elements per splat
  /** Components per color item: 3 (RGB) or 4 (RGBA — alpha = per-splat opacity). */
  private colorComponents: 3 | 4 = 3;

  private allocations = 0;
  private totalGrowths = 0;

  private assertNotDisposed(method: string): void {
    if (this._disposed) {
      throw new Error(
        `LoadedGSplatsDataAccumulator.${method}() called after dispose() — accumulator is no longer usable.`
      );
    }
  }

  /** Introspection — true if dispose() has been called. */
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
    this.colorBuffer = new Float32Array(initialCapacity * 3); // RGB(A) — see configureColorComponents

    this.allocations++;
  }

  /**
   * Declare the color layout (3 = RGB, 4 = RGBA) BEFORE the first color
   * fill. The loader knows the layout from the zarr array shape at open
   * time; the accumulator needs it to size every color buffer. Throws if
   * colors were already written with a different layout — the layout is a
   * property of the dataset, not of an individual load.
   */
  configureColorComponents(components: 3 | 4): void {
    this.assertNotDisposed('configureColorComponents');
    if (components === this.colorComponents) return;
    if (this.hasColors) {
      throw new Error(
        `GSplatsDataAccumulator: color layout changed to ${components} components after ` +
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
      this.colorBuffer = new Uint8Array(this.capacity * this.colorComponents);
    } else if (types.color === 'Uint16Array') {
      this.colorBuffer = new Uint16Array(this.capacity * this.colorComponents);
    }
  }

  ensureCapacity(needed: number): boolean {
    this.assertNotDisposed('ensureCapacity');
    if (needed <= this.capacity) return false;

    // 1.5x amortised growth, clamped to the count actually needed — see
    // `nextCapacity` for why the repeated-multiply loop overshot.
    const newCapacity = nextCapacity(this.capacity, needed);

    log.info(
      Modules.DATA_ACCUMULATOR,
      `Growing GSplatsDataAccumulator: ${this.capacity} → ${newCapacity} splats`
    );

    // Copy only the live prefix.
    const live = Math.min(this.usedCount, this.capacity);
    const liveCenterFloats = live * this.ndim;
    const liveColorFloats = live * this.colorComponents;
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
      const newColors = new Uint8Array(newCapacity * this.colorComponents);
      newColors.set(this.colorBuffer.subarray(0, liveColorFloats));
      this.colorBuffer = newColors;
    } else if (this.colorBuffer instanceof Uint16Array) {
      const newColors = new Uint16Array(newCapacity * this.colorComponents);
      newColors.set(this.colorBuffer.subarray(0, liveColorFloats));
      this.colorBuffer = newColors;
    } else {
      const newColors = new Float32Array(newCapacity * this.colorComponents);
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
      colors: this.hasColors ? this.colorBuffer.subarray(0, count * this.colorComponents) : null, // Nullable based on data presence
      colorComponents: this.colorComponents,
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

    // Track live prefix for cheap ensureCapacity copies.
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
        (this.colorBuffer as Uint8Array).set(data.colors, offset * this.colorComponents);
      } else if (data.colors instanceof Uint16Array) {
        (this.colorBuffer as Uint16Array).set(data.colors, offset * this.colorComponents);
      } else {
        (this.colorBuffer as Float32Array).set(data.colors, offset * this.colorComponents);
      }
    }
  }

  getStats(): AccumulatorStats {
    const bytesPerSplat =
      this.ndim * 4 + // centers
      4 + // amplitude
      this.choleskySize * 4 + // cholesky
      this.colorComponents * 4; // color (RGB or RGBA)

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
