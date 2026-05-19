/**
 * GSplats data accumulator — multi-type object pooling for Gaussian splats.
 *
 * Persistent typed-array buffers for splat centers, amplitudes, packed
 * Cholesky factors (camelCase `choleskyFactors`), and RGB color. Grows
 * by 1.5×; Uint8/Uint16/Float32 colors handled natively.
 *
 * The loader's deep-integration LOADING path writes directly to these
 * buffers and returns zero-copy subarrays from `getData()`.
 */

import type { LoadedGSplatsData } from '../../types/gsplats';
import { log, Modules } from '../../utils/log';
import type { DataAccumulator, AccumulatorStats } from './types';

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
