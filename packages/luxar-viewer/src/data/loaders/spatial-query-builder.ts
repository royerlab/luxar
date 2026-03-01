/**
 * SpatialQueryBuilder - Unified spatial query logic for all loader types.
 *
 * This module extracts the duplicated spatial query building logic from all three
 * spatial index loaders (Points, Lines, GSplats) into a single reusable component.
 *
 * Features:
 * - Unified tolerance calculation (displayed dims → infinite, hidden dims → step-based)
 * - Worker dispatch for CPU-intensive spatial queries
 * - Main thread fallback on worker failure
 * - extend_to_all dimension handling
 *
 * @module data/loaders/spatial-query-builder
 */

import { log, Modules } from '../../utils/log';
import type { BaseViewState, LoadRange } from './base-types';
import type { DimensionMetadata } from '../../types/dims';

// ============================================================================
// Tolerance Calculation
// ============================================================================

/**
 * Default tolerance for hidden dimensions when no step is specified.
 */
export const DEFAULT_HIDDEN_DIM_TOLERANCE = 3.0;

/**
 * Infinite tolerance for displayed dimensions (all points visible in displayed space).
 */
export const DISPLAYED_DIM_TOLERANCE = 1e10;

/**
 * Configuration for tolerance calculation.
 */
export interface ToleranceConfig {
  /** Default tolerance for hidden dimensions without step info */
  defaultTolerance?: number;

  /** Multiplier for step-based tolerance (tolerance = step * multiplier) */
  stepMultiplier?: number;

  /** Maximum radius for points (used as fallback) */
  maxRadius?: number;
}

/**
 * Calculate query tolerance for each dimension.
 *
 * This unified function replaces the duplicated tolerance calculation in:
 * - point-spatial-index-loader.ts (calculateSpatialQueryTolerance)
 * - lines-chunk-spatial-index.ts (computeLinesTolerance)
 * - gsplats-chunk-spatial-index.ts (computeGSplatsTolerance)
 *
 * Logic:
 * - Displayed dimensions (in displayDims): INFINITE tolerance (see all points)
 * - Hidden dimensions:
 *   - If step available: step * multiplier
 *   - If viewState tolerance provided: use that
 *   - Otherwise: defaultTolerance
 *
 * @param viewState - Current view state with displayDims, tolerance, dimensions
 * @param ndim - Full dimensionality of the dataset
 * @param config - Optional tolerance configuration
 * @returns Tolerance array of length ndim
 */
export function computeQueryTolerance(
  viewState: BaseViewState,
  ndim: number,
  config: ToleranceConfig = {}
): number[] {
  const {
    defaultTolerance = DEFAULT_HIDDEN_DIM_TOLERANCE,
    stepMultiplier = 1.0,
    maxRadius = defaultTolerance,
  } = config;

  const tolerance = new Array<number>(ndim).fill(0);
  const { displayDims, dimensions } = viewState;

  for (let d = 0; d < ndim; d++) {
    if (displayDims.includes(d)) {
      // Displayed dimensions: infinite tolerance (see all points in displayed space)
      tolerance[d] = DISPLAYED_DIM_TOLERANCE;
    } else {
      // Hidden dimensions: calculate appropriate tolerance
      if (dimensions && dimensions[d]?.step !== undefined) {
        // Use step-based tolerance if dimension metadata available
        tolerance[d] = dimensions[d]!.step! * stepMultiplier;
      } else if (viewState.tolerance[d] !== undefined && viewState.tolerance[d] > 0) {
        // Use explicit tolerance from view state
        tolerance[d] = viewState.tolerance[d];
      } else {
        // Fallback to maxRadius or default
        tolerance[d] = maxRadius;
      }
    }
  }

  return tolerance;
}

/**
 * Build query position from view state.
 *
 * Ensures the position array has the correct length for the dataset's ndim.
 *
 * @param viewState - Current view state
 * @param ndim - Full dimensionality
 * @returns Position array of length ndim
 */
export function buildQueryPosition(viewState: BaseViewState, ndim: number): number[] {
  const position = new Array<number>(ndim).fill(0);

  for (let d = 0; d < ndim && d < viewState.slicePosition.length; d++) {
    position[d] = viewState.slicePosition[d] ?? 0;
  }

  return position;
}

// ============================================================================
// Spatial Query Execution
// ============================================================================

/**
 * Parameters for spatial index query.
 */
export interface SpatialQueryParams {
  /** Chunk bounding boxes (flattened Float32Array) */
  chunkBounds: Float32Array;

  /** Query position (one value per dimension) */
  queryPosition: number[];

  /** Query tolerance (one value per dimension) */
  queryTolerance: number[];

  /** Number of chunks in index */
  numChunks: number;

  /** Dimensionality */
  ndim: number;
}

/**
 * Execute spatial query on main thread.
 *
 * Always runs on main thread — AABB scan is O(chunks × ndim) and completes in
 * microseconds. Worker roundtrips add ~3ms each (structured clone, postMessage,
 * deserialization), which dominates when many nodes query concurrently.
 *
 * @param params - Query parameters
 * @returns Array of chunk indices that match the query
 */
export function executeSpatialQuery(params: SpatialQueryParams): number[] {
  const { chunkBounds, queryPosition, queryTolerance, numChunks, ndim } = params;
  return queryChunksMainThread(chunkBounds, queryPosition, queryTolerance, numChunks, ndim);
}

/**
 * Main thread implementation of chunk spatial query.
 *
 * Uses AABB (axis-aligned bounding box) intersection test in nD space.
 * A chunk intersects if its bounds overlap the query box in ALL dimensions.
 *
 * @param chunkBounds - Flattened chunk bounding boxes
 * @param position - Query position
 * @param tolerance - Query tolerance
 * @param numChunks - Number of chunks
 * @param ndim - Dimensionality
 * @returns Matching chunk indices
 */
function queryChunksMainThread(
  chunkBounds: Float32Array,
  position: number[],
  tolerance: number[],
  numChunks: number,
  ndim: number
): number[] {
  const matchingChunks: number[] = [];

  for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    let intersects = true;

    for (let d = 0; d < ndim; d++) {
      // Chunk bounds layout: chunkBounds[chunkIdx, d, min/max] flattened row-major
      const offset = chunkIdx * ndim * 2 + d * 2;
      const chunkMin = chunkBounds[offset];
      const chunkMax = chunkBounds[offset + 1];

      // Query box for this dimension
      const queryMin = position[d] - tolerance[d];
      const queryMax = position[d] + tolerance[d];

      // No intersection if boxes don't overlap
      if (chunkMax < queryMin || chunkMin > queryMax) {
        intersects = false;
        break;
      }
    }

    if (intersects) {
      matchingChunks.push(chunkIdx);
    }
  }

  log.query(Modules.SPATIAL_INDEX, `Query: ${matchingChunks.length}/${numChunks} chunks match`);

  return matchingChunks;
}

// ============================================================================
// Range Conversion
// ============================================================================

/**
 * Convert chunk indices to load ranges.
 *
 * @param chunkIndices - Matching chunk indices
 * @param chunkSize - Elements per chunk
 * @param totalElements - Total elements in dataset
 * @returns Array of load ranges
 */
export function chunkIndicesToRanges(
  chunkIndices: number[],
  chunkSize: number,
  totalElements: number
): LoadRange[] {
  return chunkIndices.map((chunkIdx) => ({
    start: chunkIdx * chunkSize,
    end: Math.min((chunkIdx + 1) * chunkSize, totalElements),
  }));
}

/**
 * Merge overlapping or adjacent ranges.
 *
 * Reduces the number of zarr chunk loads by combining contiguous ranges.
 *
 * @param ranges - Unmerged ranges
 * @returns Merged ranges (sorted by start)
 */
export function mergeRanges(ranges: LoadRange[]): LoadRange[] {
  if (ranges.length === 0) return [];

  // Sort by start
  const sorted = [...ranges].sort((a, b) => a.start - b.start);

  const merged: LoadRange[] = [];
  let current = sorted[0];

  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].start <= current.end) {
      // Overlapping or adjacent - merge
      current = { start: current.start, end: Math.max(current.end, sorted[i].end) };
    } else {
      // Gap - push current and start new
      merged.push(current);
      current = sorted[i];
    }
  }

  merged.push(current);
  return merged;
}

// ============================================================================
// Extend-to-all Handling
// ============================================================================

/**
 * Check if visibility should be extended across all values in current non-displayed dimensions.
 *
 * @param extendDims - Dimension names to extend across
 * @param viewState - Current view state
 * @param dimMetadata - Optional dimension metadata for name resolution
 * @returns True if currently extending visibility
 */
export function shouldExtendVisibility(
  extendDims: string[] | undefined,
  viewState: BaseViewState,
  dimMetadata?: DimensionMetadata[]
): boolean {
  if (!extendDims || extendDims.length === 0) {
    return false;
  }

  if (!dimMetadata || dimMetadata.length === 0) {
    return false;
  }

  // Get names of non-displayed dimensions
  const hiddenDimNames = dimMetadata
    .filter((_, idx) => !viewState.displayDims.includes(idx))
    .map((meta) => meta.name)
    .filter((name) => name !== undefined);

  // Check if any extend_to_all dimension is currently hidden
  return extendDims.some((edim) => hiddenDimNames.includes(edim));
}

/**
 * Create a "load all" range for extend_to_all scenarios.
 *
 * @param totalElements - Total elements in dataset
 * @returns Single range covering all elements
 */
export function createLoadAllRange(totalElements: number): LoadRange[] {
  return [{ start: 0, end: totalElements }];
}

// ============================================================================
// SpatialQueryBuilder Class
// ============================================================================

/**
 * Builder class for constructing and executing spatial queries.
 *
 * Provides a fluent API for building spatial queries with proper tolerance calculation.
 *
 * @example
 * ```typescript
 * const builder = new SpatialQueryBuilder(chunkIndex, viewState);
 * const ranges = await builder
 *   .withExtendToAll(extendDims)
 *   .withMaxRadius(maxRadius)
 *   .execute();
 * ```
 */
export class SpatialQueryBuilder {
  private chunkBounds: Float32Array;
  private numChunks: number;
  private ndim: number;
  private chunkSize: number;
  private totalElements: number;
  private viewState: BaseViewState;

  // Optional configuration
  private extendDims?: string[];
  private toleranceConfig: ToleranceConfig = {};

  constructor(
    index: {
      chunkBounds: Float32Array;
      chunkCount: number;
      metadata: { ndim: number; chunk_size?: number };
    },
    viewState: BaseViewState,
    totalElements: number,
    chunkSize?: number
  ) {
    this.chunkBounds = index.chunkBounds;
    this.numChunks = index.chunkCount;
    this.ndim = index.metadata.ndim;
    this.chunkSize = chunkSize ?? index.metadata.chunk_size ?? 1000;
    this.totalElements = totalElements;
    this.viewState = viewState;
  }

  /**
   * Configure extend_to_all dimensions.
   */
  withExtendToAll(extendDims: string[] | undefined): this {
    this.extendDims = extendDims;
    return this;
  }

  /**
   * Configure max radius for tolerance fallback.
   */
  withMaxRadius(maxRadius: number): this {
    this.toleranceConfig.maxRadius = maxRadius;
    return this;
  }

  /**
   * Configure default tolerance for hidden dimensions.
   */
  withDefaultTolerance(tolerance: number): this {
    this.toleranceConfig.defaultTolerance = tolerance;
    return this;
  }

  /**
   * Configure step multiplier for tolerance calculation.
   */
  withStepMultiplier(multiplier: number): this {
    this.toleranceConfig.stepMultiplier = multiplier;
    return this;
  }

  /**
   * Execute the spatial query.
   *
   * @returns Merged load ranges for visible elements
   */
  async execute(): Promise<LoadRange[]> {
    // Check for extend_to_all - if applicable, return all elements
    if (shouldExtendVisibility(this.extendDims, this.viewState, this.viewState.dimensions)) {
      log.query(
        Modules.SPATIAL_INDEX,
        `Extending visibility across: ${this.extendDims?.join(', ')}`
      );
      return createLoadAllRange(this.totalElements);
    }

    // Build query position and tolerance
    const queryPosition = buildQueryPosition(this.viewState, this.ndim);
    const queryTolerance = computeQueryTolerance(this.viewState, this.ndim, this.toleranceConfig);

    log.query(
      Modules.SPATIAL_INDEX,
      `Query: pos=[${queryPosition.map((p) => p.toFixed(2)).join(', ')}]`
    );
    log.info(
      Modules.SPATIAL_INDEX,
      `Query: tol=[${queryTolerance.map((t) => (t > 1e9 ? '∞' : t.toFixed(2))).join(', ')}]`
    );

    // Execute query
    const chunkIndices = executeSpatialQuery({
      chunkBounds: this.chunkBounds,
      queryPosition,
      queryTolerance,
      numChunks: this.numChunks,
      ndim: this.ndim,
    });

    // Convert to ranges and merge
    const ranges = chunkIndicesToRanges(chunkIndices, this.chunkSize, this.totalElements);
    return mergeRanges(ranges);
  }
}
