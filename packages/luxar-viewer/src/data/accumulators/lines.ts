/**
 * Lines data accumulator — multi-type object pooling for Lines geometry.
 *
 * Implements persistent flat TypedArray buffers (NOT nested) matching
 * the `types/lines.ts` data shape. Widths are PER-VERTEX, not per-segment.
 * Grows by 1.5x; Uint8/Uint16/Float32 colors handled natively.
 *
 * The loader's deep-integration LOADING path writes directly to these
 * buffers and returns zero-copy subarrays from `getData()`.
 */

import type { LoadedLinesData } from '../../types/lines';
import { log, Modules } from '../../utils/log';
import type { DataAccumulator, AccumulatorStats } from './types';

export type { AccumulatorStats } from './types';

/**
 * Attribute types for Lines accumulator (matches GPU buffer pool pattern)
 */
export interface LinesAccumulatorTypes {
  position: 'Float32Array';
  color: 'Float32Array' | 'Uint8Array' | 'Uint16Array';
  width: 'Float32Array';
  sharpness: 'Float32Array';
  /**
   * Optional scalar dtype. Mirrors `PointsAccumulatorTypes.scalar` so
   * Uint8 scalars stay in Uint8 storage until projection widens them.
   * Omitted when the first fill carried no scalars (most datasets).
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
   * Per-vertex scalar buffer for colormap lookup. Allocated as
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
  /** Flips to true on dispose(); fill/ensureCapacity then throw. */
  private _disposed = false;
  /**
   * Highest vertex / segment indices touched by any `fill()` call.
   * Lets `ensureCapacity()` copy only the live prefix into the new
   * buffers instead of the full capacity.
   */
  private usedVertexCount = 0;
  private usedSegmentCount = 0;

  /** Components per color entry: 3 (RGB) or 4 (RGBA — alpha = per-vertex opacity). */
  private colorComponents: 3 | 4 = 3;

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

  /** Introspection — true if dispose() has been called. */
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
    this.colorBuffer = new Float32Array(initialVertexCapacity * 3); // RGB(A) — see configureColorComponents
    this.sharpnessBuffer = new Float32Array(initialVertexCapacity);
    // Scalars are an optional per-vertex attribute. Start with an empty
    // sentinel buffer so non-scalar line datasets don't reserve
    // `initialVertexCapacity * 4 B` up front. `initializeTypes` allocates
    // it on first scalar fill.
    this.scalarBuffer = new Float32Array(0);

    this.allocations++;
  }

  /**
   * Declare the color layout (3 = RGB, 4 = RGBA) BEFORE the first color
   * fill. The loader knows the layout from the zarr array shape at open
   * time; the accumulator needs it to size every color buffer. Throws if
   * colors were already written with a different layout — the layout is a
   * property of the dataset, not of an individual load. Mirrors
   * `LoadedPointsDataAccumulator.configureColorComponents`.
   */
  configureColorComponents(components: 3 | 4): void {
    this.assertNotDisposed('configureColorComponents');
    if (components === this.colorComponents) return;
    if (this.hasColors) {
      throw new Error(
        `LinesDataAccumulator: color layout changed to ${components} components after ` +
          `colors were already written with ${this.colorComponents}`
      );
    }
    this.colorComponents = components;
    // Re-size the (still empty) color buffer, preserving its element type.
    const n = this.vertexCapacity * components;
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
  private initializeTypes(data: Partial<LoadedLinesData>): void {
    // A later fill might be the first to carry scalars; allocate the
    // scalar buffer at the current vertex capacity without re-pinning the
    // color/width/sharpness types.
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
      this.colorBuffer = new Uint8Array(this.vertexCapacity * this.colorComponents);
    } else if (types.color === 'Uint16Array') {
      this.colorBuffer = new Uint16Array(this.vertexCapacity * this.colorComponents);
    }

    // Lazily allocate scalar buffer when first scalar fill is observed.
    // Sized to current vertex capacity. Skipped when there are no scalars
    // (`types.scalar === undefined`).
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
   * @param segmentCount - Optional actual segment count. When omitted,
   *   defaults to `vertexCount`. This is exact for particle tracks
   *   (N vertices → N-1 segments, ratio 1:1) and a safe over-estimate
   *   for typical line meshes (ratio 1.5:1 vertices-to-segments).
   * @returns true if buffers grew
   *
   * Always pass the actual segment count when known. The fallback is
   * sized for safety, not for efficiency — under-sizing here causes
   * silent buffer truncation when the loader later writes segments
   * past the estimated boundary.
   */
  ensureCapacity(vertexCount: number, segmentCount?: number): boolean {
    this.assertNotDisposed('ensureCapacity');
    const neededVertices = vertexCount;
    // [integration.md OOS1] Fallback default: vertexCount. Previously the
    // estimate was `Math.ceil(vertexCount / 1.5)` (~0.67 × vertexCount),
    // which UNDER-estimates for particle tracks where N vertices yields
    // N-1 segments (ratio ~1:1). Under-sizing silently truncated writes
    // past the boundary in any caller that omitted segmentCount. The
    // new default over-estimates for typical 1.5:1 meshes and is exact
    // for particle tracks — pay a small memory cost for correctness.
    const neededSegments = segmentCount ?? vertexCount;

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

      // Copy only the live prefix.
      const liveVerts = Math.min(this.usedVertexCount, this.vertexCapacity);
      const liveVertexFloats = liveVerts * this.ndim;
      const liveColorFloats = liveVerts * this.colorComponents;

      const newVertexBuf = new Float32Array(newVertexCap * this.ndim);
      const newWidthBuf = new Float32Array(newVertexCap); // PER-VERTEX!
      const newSharpnessBuf = new Float32Array(newVertexCap);

      newVertexBuf.set(this.vertexBuffer.subarray(0, liveVertexFloats));
      newWidthBuf.set(this.widthBuffer.subarray(0, liveVerts)); // widths grow with vertices
      newSharpnessBuf.set(this.sharpnessBuffer.subarray(0, liveVerts));

      this.vertexBuffer = newVertexBuf;
      this.widthBuffer = newWidthBuf;
      this.sharpnessBuffer = newSharpnessBuf;

      // Scalar buffer grows only when allocated; the sentinel-empty
      // (length 0) buffer stays empty until the first scalar fill, at
      // which point initializeTypes sizes it.
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
        const newColorBuf = new Uint8Array(newVertexCap * this.colorComponents);
        newColorBuf.set(this.colorBuffer.subarray(0, liveColorFloats));
        this.colorBuffer = newColorBuf;
      } else if (this.colorBuffer instanceof Uint16Array) {
        const newColorBuf = new Uint16Array(newVertexCap * this.colorComponents);
        newColorBuf.set(this.colorBuffer.subarray(0, liveColorFloats));
        this.colorBuffer = newColorBuf;
      } else {
        const newColorBuf = new Float32Array(newVertexCap * this.colorComponents);
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

      // Copy only the live prefix of segment-index pairs.
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
      colors: this.hasColors
        ? this.colorBuffer.subarray(0, vertexCount * this.colorComponents)
        : null,
      ...(this.hasColors ? { colorComponents: this.colorComponents } : {}),
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

    // Track live prefixes for cheap ensureCapacity copies.
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
        (this.colorBuffer as Uint8Array).set(data.colors, vertexOffset * this.colorComponents);
      } else if (data.colors instanceof Uint16Array) {
        (this.colorBuffer as Uint16Array).set(data.colors, vertexOffset * this.colorComponents);
      } else {
        (this.colorBuffer as Float32Array).set(data.colors, vertexOffset * this.colorComponents);
      }
    }
    if (data.sharpness) {
      this.hasSharpness = true;
      this.sharpnessBuffer.set(data.sharpness, vertexOffset);
    }
    // Copy scalars into per-vertex buffer preserving native dtype
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
        (this.vertexCapacity * (this.ndim * 4 + 4 + this.colorComponents * 4 + 4) + // vertices + widths + colors + sharpness
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
   * Lazy-allocated on first access. Non-scalar line datasets keep
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
