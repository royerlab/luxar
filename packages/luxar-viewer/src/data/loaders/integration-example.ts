/**
 * Integration Example: Using TransferableAccumulator with Workers
 *
 * This example demonstrates how to use the unified loader infrastructure
 * to achieve BOTH zero-allocation AND worker CPU offload.
 *
 * NOTE: This is a reference implementation showing the pattern.
 * It can be used as a template for integrating into the actual loaders.
 *
 * @module data/loaders/integration-example
 */

import { log, Modules } from '../../utils/log';
import { getWorkerPool } from '../../workers/worker-pool';
import {
  TransferableAccumulator,
  createPointsAccumulator,
  type PointsBuffers,
} from './transferable-accumulator';
import { SpatialQueryBuilder } from './spatial-query-builder';
import type { BaseViewState, LoadRange } from './base-types';

// ============================================================================
// Example: Points Loader Integration
// ============================================================================

/**
 * Example parameters for worker projection.
 */
interface ProjectionParams {
  /** Source positions (nD) */
  positions: Float32Array;

  /** Display dimension indices */
  displayDims: [number, number, number];

  /** Total number of points */
  pointCount: number;

  /** Source colors (optional) */
  colors?: Float32Array | null;

  /** Source radii (optional) */
  radii?: Float32Array | null;
}

/**
 * Example class showing how to integrate TransferableAccumulator with workers.
 *
 * Key pattern:
 * 1. Create accumulator once (owns reusable buffers)
 * 2. For each update:
 *    a. Detach buffers from accumulator
 *    b. Transfer to worker (zero-copy)
 *    c. Worker fills buffers
 *    d. Transfer back (zero-copy)
 *    e. Adopt buffers back into accumulator
 */
export class PointsLoaderIntegrationExample {
  private accumulator: TransferableAccumulator<PointsBuffers>;
  private lastPointCount = 0;

  constructor(initialCapacity = 10000) {
    this.accumulator = createPointsAccumulator(initialCapacity);
    log.info(
      Modules.DATA_ACCUMULATOR,
      `[Example] Created accumulator with capacity ${initialCapacity}`
    );
  }

  /**
   * Load and project points using the zero-allocation worker pattern.
   *
   * @param sourcePositions - Source nD positions
   * @param viewState - Current view state
   * @param options - Loading options
   * @returns Projected 3D points data
   */
  async loadAndProject(
    sourcePositions: Float32Array,
    viewState: BaseViewState,
    options: {
      colors?: Float32Array;
      radii?: Float32Array;
    } = {}
  ): Promise<{
    positions3D: Float32Array;
    colors?: Float32Array;
    radii?: Float32Array;
    pointCount: number;
  }> {
    const pointCount = sourcePositions.length / viewState.slicePosition.length;

    // Step 1: Ensure capacity (may allocate if growing)
    this.accumulator.ensureCapacity(pointCount);

    // Step 2: Enable optional buffers if needed
    if (options.colors) {
      this.accumulator.enableBuffer('colors');
    }
    if (options.radii) {
      this.accumulator.enableBuffer('radii');
    }

    // Step 3: Detach buffers for transfer to worker
    const buffers = this.accumulator.detach();

    // Step 4: Get ArrayBuffer list for Comlink.transfer()
    const transferables = this.accumulator.getTransferables(buffers);

    try {
      // Step 5: Execute projection on worker with pre-allocated buffers
      const result = await this.projectOnWorker(
        {
          positions: sourcePositions,
          displayDims: viewState.displayDims as [number, number, number],
          pointCount,
          colors: options.colors ?? null,
          radii: options.radii ?? null,
        },
        buffers,
        transferables
      );

      // Step 6: Adopt buffers back (zero-copy)
      this.accumulator.adopt(result.buffers);
      this.lastPointCount = result.pointCount;

      // Return views into the accumulator's buffers
      const accBuffers = this.accumulator.getBuffers()!;
      return {
        positions3D: accBuffers.positions.subarray(0, result.pointCount * 3),
        colors: (accBuffers.colors as Float32Array | undefined)?.subarray(0, result.pointCount * 3),
        radii: accBuffers.radii?.subarray(0, result.pointCount),
        pointCount: result.pointCount,
      };
    } catch {
      // On error, we need to reallocate buffers since they were transferred
      log.warning(Modules.DATA_ACCUMULATOR, '[Example] Worker failed, reallocating buffers');
      this.accumulator = createPointsAccumulator(pointCount);

      // Fall back to main thread processing
      return this.projectOnMainThread(
        sourcePositions,
        viewState.displayDims as [number, number, number],
        pointCount,
        options.colors,
        options.radii
      );
    }
  }

  /**
   * Execute projection on worker thread.
   */
  private async projectOnWorker(
    params: ProjectionParams,
    outputBuffers: PointsBuffers,
    transferables: ArrayBuffer[]
  ): Promise<{
    buffers: PointsBuffers;
    pointCount: number;
  }> {
    // Get worker (demonstrates pool usage)
    const pool = getWorkerPool();
    await pool.getWorker(); // Demonstrates acquiring a worker

    // For this example, we simulate the worker behavior (actual implementation
    // would use Comlink.transfer() to send outputBuffers to worker)
    log.info(
      Modules.DATA_ACCUMULATOR,
      `[Example] Transferring ${transferables.length} buffers to worker`
    );

    // Simulated worker processing (fills the pre-allocated buffers)
    const { positions, displayDims, pointCount, colors, radii } = params;
    const ndim = positions.length / pointCount;

    // Extract display dimensions
    const [dx, dy, dz] = displayDims;

    for (let i = 0; i < pointCount; i++) {
      const srcOffset = i * ndim;
      const dstOffset = i * 3;

      outputBuffers.positions[dstOffset] = positions[srcOffset + dx];
      outputBuffers.positions[dstOffset + 1] = positions[srcOffset + dy];
      outputBuffers.positions[dstOffset + 2] = positions[srcOffset + dz];

      if (colors && outputBuffers.colors) {
        const colorOffset = i * 3;
        outputBuffers.colors[dstOffset] = colors[colorOffset];
        outputBuffers.colors[dstOffset + 1] = colors[colorOffset + 1];
        outputBuffers.colors[dstOffset + 2] = colors[colorOffset + 2];
      }

      if (radii && outputBuffers.radii) {
        outputBuffers.radii[i] = radii[i];
      }
    }

    return {
      buffers: outputBuffers,
      pointCount,
    };
  }

  /**
   * Fallback: Main thread projection.
   */
  private projectOnMainThread(
    positions: Float32Array,
    displayDims: [number, number, number],
    pointCount: number,
    colors?: Float32Array,
    radii?: Float32Array
  ): {
    positions3D: Float32Array;
    colors?: Float32Array;
    radii?: Float32Array;
    pointCount: number;
  } {
    const ndim = positions.length / pointCount;
    const [dx, dy, dz] = displayDims;

    const positions3D = new Float32Array(pointCount * 3);
    let projectedColors: Float32Array | undefined;
    let projectedRadii: Float32Array | undefined;

    for (let i = 0; i < pointCount; i++) {
      const srcOffset = i * ndim;
      const dstOffset = i * 3;

      positions3D[dstOffset] = positions[srcOffset + dx];
      positions3D[dstOffset + 1] = positions[srcOffset + dy];
      positions3D[dstOffset + 2] = positions[srcOffset + dz];
    }

    if (colors) {
      // This example only handles Float32Array colors
      projectedColors = new Float32Array(colors);
    }

    if (radii) {
      projectedRadii = new Float32Array(radii);
    }

    return {
      positions3D,
      colors: projectedColors,
      radii: projectedRadii,
      pointCount,
    };
  }

  /**
   * Get accumulator statistics.
   */
  getStats(): {
    capacity: number;
    lastPointCount: number;
    allocations: number;
    reuseCount: number;
    peakMemoryBytes: number;
  } {
    const stats = this.accumulator.getStats();
    return {
      capacity: this.accumulator.getCapacity(),
      lastPointCount: this.lastPointCount,
      allocations: stats.allocations,
      reuseCount: stats.reuseCount,
      peakMemoryBytes: stats.peakMemoryBytes,
    };
  }

  /**
   * Dispose resources.
   */
  dispose(): void {
    this.accumulator.dispose();
    log.info(Modules.DATA_ACCUMULATOR, '[Example] Disposed accumulator');
  }
}

// ============================================================================
// Example: Using SpatialQueryBuilder for Visible Range Calculation
// ============================================================================

/**
 * Example showing how to use SpatialQueryBuilder for chunk-based loading.
 */
export async function queryVisibleRangesExample(
  chunkIndex: {
    chunkBounds: Float32Array;
    chunkCount: number;
    metadata: { ndim: number; chunk_size: number };
  },
  viewState: BaseViewState,
  totalElements: number,
  options: {
    extendDims?: string[];
    maxRadius?: number;
  } = {}
): Promise<LoadRange[]> {
  const builder = new SpatialQueryBuilder(
    chunkIndex,
    viewState,
    totalElements,
    chunkIndex.metadata.chunk_size
  );

  // Configure query with optional settings
  if (options.extendDims) {
    builder.withExtendToAll(options.extendDims);
  }

  if (options.maxRadius) {
    builder.withMaxRadius(options.maxRadius);
  }

  // Execute and get merged ranges
  const ranges = await builder.execute();

  log.info(
    Modules.SPATIAL_INDEX,
    `[Example] Query returned ${ranges.length} ranges covering ` +
      `${ranges.reduce((sum, r) => sum + (r.end - r.start), 0)} elements`
  );

  return ranges;
}

// ============================================================================
// Full Integration Example
// ============================================================================

/**
 * Complete example showing the full loading pipeline.
 *
 * This demonstrates:
 * 1. Spatial query to find visible chunks
 * 2. Loading data from visible ranges
 * 3. Projecting to 3D using TransferableAccumulator pattern
 */
export async function fullLoadingPipelineExample(
  chunkIndex: {
    chunkBounds: Float32Array;
    chunkCount: number;
    metadata: { ndim: number; chunk_size: number };
  },
  viewState: BaseViewState,
  dataLoader: {
    loadPositions(ranges: LoadRange[]): Promise<Float32Array>;
    loadColors?(ranges: LoadRange[]): Promise<Float32Array>;
    loadRadii?(ranges: LoadRange[]): Promise<Float32Array>;
    getTotalElements(): number;
  },
  options: {
    extendDims?: string[];
    maxRadius?: number;
  } = {}
): Promise<{
  positions3D: Float32Array;
  colors?: Float32Array;
  radii?: Float32Array;
  pointCount: number;
  stats: ReturnType<PointsLoaderIntegrationExample['getStats']>;
}> {
  // Step 1: Query visible ranges
  const ranges = await queryVisibleRangesExample(
    chunkIndex,
    viewState,
    dataLoader.getTotalElements(),
    options
  );

  if (ranges.length === 0) {
    return {
      positions3D: new Float32Array(0),
      pointCount: 0,
      stats: {
        capacity: 0,
        lastPointCount: 0,
        allocations: 0,
        reuseCount: 0,
        peakMemoryBytes: 0,
      },
    };
  }

  // Step 2: Load data from ranges
  const positions = await dataLoader.loadPositions(ranges);
  const colors = dataLoader.loadColors ? await dataLoader.loadColors(ranges) : undefined;
  const radii = dataLoader.loadRadii ? await dataLoader.loadRadii(ranges) : undefined;

  // Step 3: Project to 3D using TransferableAccumulator
  const loader = new PointsLoaderIntegrationExample();

  try {
    const result = await loader.loadAndProject(positions, viewState, {
      colors,
      radii,
    });

    return {
      ...result,
      stats: loader.getStats(),
    };
  } finally {
    loader.dispose();
  }
}
