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
   * Add uniform radii
   */
  withRadii(radius: number): this {
    this.radii = new Float32Array(this.numPoints).fill(radius);
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

    return {
      positions: this.positions!,
      colors: this.colors,
      radii: this.radii,
      sharpness: this.sharpness,
      numPoints: this.numPoints,
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
   * Set which dimensions are displayed
   */
  withDisplayed(...indices: number[]): this {
    this.displayed = indices.slice(0, 3); // Max 3 displayed
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
   * Generate random data
   */
  withRandomData(): this {
    const size = this.chunkShape.reduce((a, b) => a * b, 1);
    if (this.dtype.includes('u1') || this.dtype.includes('uint8')) {
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
    return {
      shape: this.shape,
      chunks: this.chunks,
      dtype: this.dtype,
      metadata: this.metadata,
      get: vi.fn().mockResolvedValue({
        data: new Float32Array(this.chunks.reduce((a, b) => a * b, 1)),
      }),
    };
  }
}
