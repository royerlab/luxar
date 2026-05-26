/**
 * Test Data Builders for Luxar Tests
 *
 * These builders provide a fluent API for creating test data,
 * making tests more readable and maintainable.
 *
 * ## Benefits:
 * - **Readability**: Tests clearly express their intent
 * - **Maintainability**: Changes to data structures are centralized
 * - **Reusability**: Common test scenarios can be shared
 * - **Type Safety**: TypeScript ensures valid test data
 *
 * ## Usage:
 * ```typescript
 * const pointCloud = new PointsBuilder()
 *   .withPoints(1000)
 *   .withDimensions(4)
 *   .withColors()
 *   .withRadii(0.5)
 *   .build();
 * ```
 */

import { SimpleDims, DimensionMetadata } from '../../types/dims';
import * as THREE from 'three';
import { vi } from 'vitest';

/**
 * Builder for points test data
 */
export class PointsBuilder {
  private numPoints: number = 100;
  private dimensions: number = 3;
  private positions: Float32Array | null = null;
  private colors: Float32Array | null = null;
  private radii: Float32Array | null = null;
  private sharpness: Float32Array | null = null;

  /**
   * Set the number of points
   */
  withPoints(num: number): this {
    this.numPoints = num;
    return this;
  }

  /**
   * Set the number of dimensions
   */
  withDimensions(dims: number): this {
    this.dimensions = dims;
    return this;
  }

  /**
   * Add specific positions
   */
  withPositions(positions: Float32Array | number[]): this {
    this.positions = positions instanceof Float32Array ? positions : new Float32Array(positions);
    return this;
  }

  /**
   * Generate random positions
   */
  withRandomPositions(range: [number, number] = [-1, 1]): this {
    const positions = new Float32Array(this.numPoints * this.dimensions);
    const [min, max] = range;
    for (let i = 0; i < positions.length; i++) {
      positions[i] = min + Math.random() * (max - min);
    }
    this.positions = positions;
    return this;
  }

  /**
   * Add colors (random if not specified)
   */
  withColors(colors?: Float32Array | number[]): this {
    if (colors) {
      this.colors = colors instanceof Float32Array ? colors : new Float32Array(colors);
    } else {
      // Generate random colors
      this.colors = new Float32Array(this.numPoints * 3);
      for (let i = 0; i < this.colors.length; i++) {
        this.colors[i] = Math.random();
      }
    }
    return this;
  }

  /**
   * Set radii.
   *
   * Accepts either:
   *  - a scalar (`number`) — broadcast to every point, OR
   *  - an explicit `Float32Array` / `number[]` of length `numPoints` —
   *    cloned into a fresh Float32Array so the builder owns its own
   *    buffer (a later mutation by the caller cannot leak in).
   *
   * Mismatched-length arrays throw a clear error rather than silently
   * truncating or zero-extending.
   */
  withRadii(radius: number | Float32Array | number[]): this {
    if (typeof radius === 'number') {
      this.radii = new Float32Array(this.numPoints).fill(radius);
    } else {
      if (radius.length !== this.numPoints) {
        throw new Error(
          `withRadii: array length ${radius.length} does not match numPoints ${this.numPoints}`
        );
      }
      this.radii = new Float32Array(radius);
    }
    return this;
  }

  /**
   * Add varying radii
   */
  withVaryingRadii(min: number = 0.1, max: number = 1.0): this {
    this.radii = new Float32Array(this.numPoints);
    for (let i = 0; i < this.numPoints; i++) {
      this.radii[i] = min + Math.random() * (max - min);
    }
    return this;
  }

  /**
   * Add sharpness values
   */
  withSharpness(value: number = 2.0): this {
    this.sharpness = new Float32Array(this.numPoints).fill(value);
    return this;
  }

  /**
   * Build the points data
   */
  build(): {
    positions: Float32Array;
    colors: Float32Array | null;
    radii: Float32Array | null;
    sharpness: Float32Array | null;
    numPoints: number;
    dimensions: number;
  } {
    // Generate positions if not set
    if (!this.positions) {
      this.withRandomPositions();
    }

    // Return FRESH copies of every typed array. Without cloning, two
    // build() calls return the same buffer instance; a test that mutates
    // the result of the first call would leak into every subsequent
    // build() (and into any other test holding a reference). Float32Array
    // construction with another typed array argument copies the bytes.
    return {
      positions: new Float32Array(this.positions!),
      colors: this.colors ? new Float32Array(this.colors) : null,
      radii: this.radii ? new Float32Array(this.radii) : null,
      sharpness: this.sharpness ? new Float32Array(this.sharpness) : null,
      numPoints: this.numPoints,
      dimensions: this.dimensions,
    };
  }
}

/**
 * Builder for lines test data.
 *
 * Mirrors the PointsBuilder API: chained `withX()` setters and a final
 * `build()` that returns plain typed arrays + counts. Use this in
 * spatial-index loader, projection, and clipping tests so the test
 * fixtures stay consistent across geometry types.
 *
 * @example
 * ```ts
 * const lines = new LinesBuilder()
 *   .withSegments(50)
 *   .withDimensions(3)
 *   .withRandomVertices()
 *   .withWidths(1.0)
 *   .build();
 * ```
 */
export class LinesBuilder {
  private numSegments: number = 100;
  private dimensions: number = 3;
  // Vertices are stored as flat (numSegments * 2) * dimensions floats.
  private vertices: Float32Array | null = null;
  private widths: Float32Array | null = null;
  private segments: Uint32Array | null = null;
  private colors: Float32Array | null = null;
  private sharpness: Float32Array | null = null;

  /** Set the number of line segments. */
  withSegments(num: number): this {
    this.numSegments = num;
    return this;
  }

  /** Set the number of spatial dimensions. */
  withDimensions(dims: number): this {
    this.dimensions = dims;
    return this;
  }

  /** Provide explicit vertex data ((numSegments * 2) * dimensions floats). */
  withVertices(vertices: Float32Array | number[]): this {
    this.vertices = vertices instanceof Float32Array ? vertices : new Float32Array(vertices);
    return this;
  }

  /** Generate random vertex data within `range`. */
  withRandomVertices(range: [number, number] = [-1, 1]): this {
    const verts = new Float32Array(this.numSegments * 2 * this.dimensions);
    const [min, max] = range;
    for (let i = 0; i < verts.length; i++) {
      verts[i] = min + Math.random() * (max - min);
    }
    this.vertices = verts;
    return this;
  }

  /** Set uniform widths. */
  withWidths(width: number): this {
    this.widths = new Float32Array(this.numSegments).fill(width);
    return this;
  }

  /** Set varying widths. */
  withVaryingWidths(min: number = 0.1, max: number = 1.0): this {
    this.widths = new Float32Array(this.numSegments);
    for (let i = 0; i < this.numSegments; i++) {
      this.widths[i] = min + Math.random() * (max - min);
    }
    return this;
  }

  /**
   * Provide explicit segment indices. If omitted, auto-generates the canonical
   * `[0, 1, 2, 3, ..., 2N-1]` packing where each segment takes two consecutive
   * vertex indices.
   */
  withSegmentIndices(segments: Uint32Array | number[]): this {
    this.segments = segments instanceof Uint32Array ? segments : new Uint32Array(segments);
    return this;
  }

  /** Add per-vertex colors as RGB floats. Random when no values are passed. */
  withColors(colors?: Float32Array | number[]): this {
    if (colors) {
      this.colors = colors instanceof Float32Array ? colors : new Float32Array(colors);
    } else {
      this.colors = new Float32Array(this.numSegments * 2 * 3);
      for (let i = 0; i < this.colors.length; i++) {
        this.colors[i] = Math.random();
      }
    }
    return this;
  }

  /** Per-vertex sharpness scalars. */
  withSharpness(value: number = 2.0): this {
    this.sharpness = new Float32Array(this.numSegments * 2).fill(value);
    return this;
  }

  /**
   * Build the test fixture. `vertices` is auto-populated with random values
   * if no `withVertices`/`withRandomVertices` call preceded `build()`.
   * `widths` defaults to a uniform 1.0 if unset (required attribute).
   * `segments` defaults to the canonical packing if unset.
   */
  build(): {
    vertices: Float32Array;
    widths: Float32Array;
    segments: Uint32Array;
    colors: Float32Array | null;
    sharpness: Float32Array | null;
    numSegments: number;
    dimensions: number;
  } {
    if (!this.vertices) {
      this.withRandomVertices();
    }
    if (!this.widths) {
      this.withWidths(1.0);
    }
    if (!this.segments) {
      const seg = new Uint32Array(this.numSegments * 2);
      for (let i = 0; i < seg.length; i++) seg[i] = i;
      this.segments = seg;
    }
    // Clone every output buffer — see PointsBuilder.build() for rationale.
    return {
      vertices: new Float32Array(this.vertices!),
      widths: new Float32Array(this.widths!),
      segments: new Uint32Array(this.segments!),
      colors: this.colors ? new Float32Array(this.colors) : null,
      sharpness: this.sharpness ? new Float32Array(this.sharpness) : null,
      numSegments: this.numSegments,
      dimensions: this.dimensions,
    };
  }
}

/**
 * Builder for gsplats test data.
 *
 * Mirrors PointsBuilder. Required attributes are `centers`, `amplitudes`,
 * and `choleskyFactors`. The Cholesky packing follows the lower-triangular
 * layout used by the rest of the viewer: ndim*(ndim+1)/2 floats per splat.
 *
 * @example
 * ```ts
 * const gs = new GSplatsBuilder()
 *   .withSplats(200)
 *   .withDimensions(3)
 *   .withRandomCenters()
 *   .withAmplitudes(1.0)
 *   .withIsotropicCovariance(0.05)
 *   .build();
 * ```
 */
export class GSplatsBuilder {
  private numSplats: number = 100;
  private dimensions: number = 3;
  private centers: Float32Array | null = null;
  private amplitudes: Float32Array | null = null;
  private choleskyFactors: Float32Array | null = null;
  private colors: Float32Array | null = null;

  /** Set the number of splats. */
  withSplats(num: number): this {
    this.numSplats = num;
    return this;
  }

  /** Set the number of spatial dimensions. */
  withDimensions(dims: number): this {
    this.dimensions = dims;
    return this;
  }

  /** Provide explicit centers. Layout: numSplats * dimensions floats. */
  withCenters(centers: Float32Array | number[]): this {
    this.centers = centers instanceof Float32Array ? centers : new Float32Array(centers);
    return this;
  }

  /** Generate random centers within `range`. */
  withRandomCenters(range: [number, number] = [-1, 1]): this {
    const c = new Float32Array(this.numSplats * this.dimensions);
    const [min, max] = range;
    for (let i = 0; i < c.length; i++) {
      c[i] = min + Math.random() * (max - min);
    }
    this.centers = c;
    return this;
  }

  /** Uniform amplitudes (required attribute). */
  withAmplitudes(value: number): this {
    this.amplitudes = new Float32Array(this.numSplats).fill(value);
    return this;
  }

  /** Varying amplitudes. */
  withVaryingAmplitudes(min: number = 0.1, max: number = 1.0): this {
    this.amplitudes = new Float32Array(this.numSplats);
    for (let i = 0; i < this.numSplats; i++) {
      this.amplitudes[i] = min + Math.random() * (max - min);
    }
    return this;
  }

  /**
   * Set isotropic Gaussian covariance via a single sigma. The Cholesky factor
   * for an isotropic Gaussian is sigma * I, so the lower-triangular packing
   * is sigma on the diagonals and zero elsewhere.
   *
   * Packing (ndim=3): `[L00, L10, L11, L20, L21, L22]` →
   * `[sigma, 0, sigma, 0, 0, sigma]`.
   */
  withIsotropicCovariance(sigma: number = 0.1): this {
    const ndim = this.dimensions;
    const packed = (ndim * (ndim + 1)) / 2;
    this.choleskyFactors = new Float32Array(this.numSplats * packed);

    for (let s = 0; s < this.numSplats; s++) {
      let idx = s * packed;
      for (let row = 0; row < ndim; row++) {
        for (let col = 0; col <= row; col++) {
          this.choleskyFactors[idx++] = row === col ? sigma : 0;
        }
      }
    }
    return this;
  }

  /** Provide explicit Cholesky factors (numSplats * ndim*(ndim+1)/2 floats). */
  withCholeskyFactors(factors: Float32Array | number[]): this {
    this.choleskyFactors = factors instanceof Float32Array ? factors : new Float32Array(factors);
    return this;
  }

  /** Per-splat RGB colors. Random when no values are passed. */
  withColors(colors?: Float32Array | number[]): this {
    if (colors) {
      this.colors = colors instanceof Float32Array ? colors : new Float32Array(colors);
    } else {
      this.colors = new Float32Array(this.numSplats * 3);
      for (let i = 0; i < this.colors.length; i++) {
        this.colors[i] = Math.random();
      }
    }
    return this;
  }

  /**
   * Build the test fixture. `centers` is auto-populated with random values
   * if unset; `amplitudes` defaults to 1.0; `choleskyFactors` defaults to
   * isotropic sigma=0.1 if unset (all three are required attributes).
   */
  build(): {
    centers: Float32Array;
    amplitudes: Float32Array;
    choleskyFactors: Float32Array;
    colors: Float32Array | null;
    numSplats: number;
    dimensions: number;
  } {
    if (!this.centers) {
      this.withRandomCenters();
    }
    if (!this.amplitudes) {
      this.withAmplitudes(1.0);
    }
    if (!this.choleskyFactors) {
      this.withIsotropicCovariance(0.1);
    }
    // Clone every output buffer — see PointsBuilder.build() for rationale.
    return {
      centers: new Float32Array(this.centers!),
      amplitudes: new Float32Array(this.amplitudes!),
      choleskyFactors: new Float32Array(this.choleskyFactors!),
      colors: this.colors ? new Float32Array(this.colors) : null,
      numSplats: this.numSplats,
      dimensions: this.dimensions,
    };
  }
}

/**
 * Builder for dimension configuration
 */
export class DimensionsBuilder {
  private ndim: number = 3;
  private displayed: number[] = [0, 1, 2];
  private currentStep: number[] = [];
  private metadata: DimensionMetadata[] = [];

  /**
   * Set total number of dimensions
   */
  withNDimensions(n: number): this {
    this.ndim = n;
    this.currentStep = new Array(n).fill(0);
    return this;
  }

  /**
   * Set which dimensions are displayed. The Luxar scene contract caps
   * displayed dimensions at 3 (the visual XYZ axes); a caller passing
   * more than 3 indices is asserting a stronger contract than the
   * viewer can satisfy. Silently slicing to 3 hid this misuse.
   *
   * Now the builder throws on >3 indices so tests are forced to be
   * explicit about which dims they want visible. Passing exactly 0–3
   * is unchanged.
   */
  withDisplayed(...indices: number[]): this {
    if (indices.length > 3) {
      throw new Error(
        'DimensionsBuilder.withDisplayed: at most 3 dimensions may be displayed; ' +
          `got ${indices.length} indices ${JSON.stringify(indices)}`
      );
    }
    this.displayed = indices.slice();
    return this;
  }

  /**
   * Add a dimension with metadata
   */
  withDimension(
    index: number,
    name: string,
    unit: string = '',
    range: [number, number] = [0, 1],
    options: {
      display?: boolean;
      discrete?: boolean;
      step?: number;
    } = {}
  ): this {
    // Extend metadata array if necessary
    while (this.metadata.length <= index) {
      this.metadata.push({
        name: `dim${this.metadata.length}`,
        unit: '',
        scale: 1.0,
        range: [0, 1],
        display: false,
      });
    }

    this.metadata[index] = {
      name,
      unit,
      scale: 1.0,
      range,
      display: options.display ?? index < 3,
      discrete: options.discrete,
      step: options.step,
    };
    return this;
  }

  /**
   * Add standard spatial dimensions
   */
  withSpatialDimensions(unit: string = 'μm', range: [number, number] = [0, 100]): this {
    this.withDimension(0, 'x', unit, range, { display: true });
    this.withDimension(1, 'y', unit, range, { display: true });
    this.withDimension(2, 'z', unit, range, { display: true });
    return this;
  }

  /**
   * Add a time dimension
   */
  withTimeDimension(range: [number, number] = [0, 10], step: number = 0.1): this {
    const index = 3; // Time is typically dimension 3
    if (this.ndim < 4) {
      this.ndim = 4;
      this.currentStep = new Array(this.ndim).fill(0);
    }
    this.withDimension(index, 'time', 's', range, {
      display: false,
      discrete: false,
      step,
    });
    return this;
  }

  /**
   * Add a channel dimension
   */
  withChannelDimension(numChannels: number = 3): this {
    const index = 4; // Channel is typically dimension 4
    if (this.ndim < 5) {
      this.ndim = 5;
      this.currentStep = new Array(this.ndim).fill(0);
    }
    this.withDimension(index, 'channel', '', [0, numChannels - 1], {
      display: false,
      discrete: true,
      step: 1,
    });
    return this;
  }

  /**
   * Set current position in non-displayed dimensions
   */
  withCurrentPosition(...values: number[]): this {
    for (let i = 0; i < values.length && i < this.ndim; i++) {
      this.currentStep[i] = values[i];
    }
    return this;
  }

  /**
   * Build the dimensions configuration
   */
  build(): SimpleDims {
    // Fill in any missing metadata
    while (this.metadata.length < this.ndim) {
      this.metadata.push({
        name: `dim${this.metadata.length}`,
        unit: '',
        scale: 1.0,
        range: [0, 1],
        display: this.displayed.includes(this.metadata.length),
      });
    }

    return {
      ndim: this.ndim,
      displayed: this.displayed,
      currentStep: this.currentStep,
      metadata: this.metadata.slice(0, this.ndim), // Ensure array matches ndim
    };
  }
}

/**
 * Builder for Three.js scene test data
 */
export class SceneBuilder {
  private scene: THREE.Scene;
  private objects: THREE.Object3D[] = [];

  constructor() {
    this.scene = new THREE.Scene();
  }

  /**
   * Add a points to the scene
   */
  withPoints(
    positions: Float32Array,
    options: {
      colors?: Float32Array;
      dims?: SimpleDims;
    } = {}
  ): this {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    if (options.colors) {
      geometry.setAttribute('color', new THREE.BufferAttribute(options.colors, 3));
    }

    const material = new THREE.PointsMaterial({
      size: 0.1,
      vertexColors: !!options.colors,
    });

    const points = new THREE.Points(geometry, material);

    // Add dims metadata if provided
    if (options.dims) {
      (points as any).dims = options.dims;
    }

    this.scene.add(points);
    this.objects.push(points);
    return this;
  }

  /**
   * Add a group to the scene
   */
  withGroup(name: string = 'group'): this {
    const group = new THREE.Group();
    group.name = name;
    this.scene.add(group);
    this.objects.push(group);
    return this;
  }

  /**
   * Add nested structure
   */
  withNestedStructure(levels: number = 2): this {
    let parent: THREE.Object3D = this.scene;
    for (let i = 0; i < levels; i++) {
      const group = new THREE.Group();
      group.name = `level_${i}`;
      parent.add(group);
      parent = group;
      this.objects.push(group);
    }
    return this;
  }

  /**
   * Build the scene
   */
  build(): THREE.Scene {
    return this.scene;
  }

  /**
   * Get all objects added to the scene
   */
  getObjects(): THREE.Object3D[] {
    return this.objects;
  }
}

/**
 * Builder for chunk test data
 */
export class ChunkBuilder {
  private shape: number[] = [100, 10];
  private chunkShape: number[] = [20, 10];
  private dtype: string = '<f4';
  private data: Float32Array | Uint8Array | null = null;

  /**
   * Set array shape
   */
  withShape(...dims: number[]): this {
    this.shape = dims;
    return this;
  }

  /**
   * Set chunk shape
   */
  withChunkShape(...dims: number[]): this {
    this.chunkShape = dims;
    return this;
  }

  /**
   * Set data type
   */
  withDtype(dtype: string): this {
    this.dtype = dtype;
    return this;
  }

  /**
   * Generate random data.
   *
   * [builders OOS3] Pre-fix dtype detection was
   * `dtype.includes('u1') || dtype.includes('uint8')` — bare substring
   * matching. This misclassified:
   *   - `'<u16'` (uint16) → was routed to the Uint8 branch (contains 'u1')
   *   - `'complex_uint8_v2'` (any future dtype with 'uint8' embedded)
   *     → also routed to the Uint8 branch
   *
   * Now we match the exact set of supported uint8 dtype spellings.
   * Zarr typed-dtype prefixes `<`, `>`, `|` denote byte order; `'uint8'`
   * is the numpy long form. Anything else falls through to the Float32
   * branch (the previous default).
   */
  withRandomData(): this {
    const size = this.chunkShape.reduce((a, b) => a * b, 1);
    if (ChunkBuilder.UINT8_DTYPES.has(this.dtype)) {
      this.data = new Uint8Array(size);
      for (let i = 0; i < size; i++) {
        this.data[i] = Math.floor(Math.random() * 256);
      }
    } else {
      this.data = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        this.data[i] = Math.random();
      }
    }
    return this;
  }

  /**
   * Supported uint8 dtype spellings used by `withRandomData()` and any
   * future dtype-branching builders. Kept as `static readonly` so tests
   * can reference the exact contract.
   */
  static readonly UINT8_DTYPES: ReadonlySet<string> = new Set([
    'u1', // bare numpy abbreviation
    '|u1', // byte-order-agnostic
    '<u1', // little-endian
    '>u1', // big-endian
    'uint8', // numpy long form
  ]);

  /**
   * Build the chunk metadata
   */
  build(): {
    shape: number[];
    chunkShape: number[];
    dtype: string;
    data: Float32Array | Uint8Array;
    chunkGrid: number[];
  } {
    if (!this.data) {
      this.withRandomData();
    }

    const chunkGrid = this.shape.map((s, i) => Math.ceil(s / this.chunkShape[i]));

    return {
      shape: this.shape,
      chunkShape: this.chunkShape,
      dtype: this.dtype,
      data: this.data!,
      chunkGrid,
    };
  }
}

/**
 * Builder for mock zarr arrays
 */
export class MockZarrArrayBuilder {
  private shape: number[] = [1000, 3];
  private chunks: number[] = [100, 3];
  private dtype: string = '<f4';
  private metadata: any = {};

  withShape(...dims: number[]): this {
    this.shape = dims;
    return this;
  }

  withChunks(...dims: number[]): this {
    this.chunks = dims;
    return this;
  }

  withDtype(dtype: string): this {
    this.dtype = dtype;
    return this;
  }

  withMetadata(metadata: any): this {
    this.metadata = metadata;
    return this;
  }

  build(): any {
    const chunkLen = this.chunks.reduce((a, b) => a * b, 1);
    return {
      shape: this.shape,
      chunks: this.chunks,
      dtype: this.dtype,
      metadata: this.metadata,
      // Use mockImplementation (not mockResolvedValue) so EACH call
      // returns a fresh Float32Array. mockResolvedValue captures one
      // payload and reuses it forever — a test that mutated the
      // returned `.data` would corrupt every subsequent get() call.
      get: vi.fn().mockImplementation(async () => ({
        data: new Float32Array(chunkLen),
      })),
    };
  }
}
