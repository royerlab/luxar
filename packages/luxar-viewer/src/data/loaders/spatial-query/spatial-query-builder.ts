/**
 * SpatialQueryBuilder - Canonical spatial query API for Points, Lines, GSplats.
 *
 * Provides one unified entry point for chunk-bounds-based AABB queries:
 * builds the query position from view state, computes (or accepts) per-dimension
 * tolerance, runs the AABB scan, converts matching chunks to load ranges, and
 * coalesces overlapping/adjacent ranges.
 *
 * Tolerance source is selected at construction:
 * - `geometryType: 'points' | 'lines' | 'gsplats'` — delegates to
 *   `tolerance-computer.computeTolerance` for geometry-aware semantics.
 * - `tolerance: number[]` — caller-supplied array (used when the caller has
 *   geometry-specific logic that the unified computer does not model, e.g. the
 *   points loader's `EffectiveRadiusConfig`-driven tolerance).
 *
 * Always runs on the main thread — AABB scans are O(numChunks × ndim) and
 * complete in microseconds. Worker dispatch would add ~3 ms structured-clone
 * overhead per call, which dominates when many nodes query concurrently.
 *
 * @module data/loaders/spatial-query-builder
 */

import { log, Modules } from '../../../utils/log';
import { computeTolerance, type ToleranceOptions } from './tolerance-computer';
import type { GeometryKind } from '../../data-loader-types';
import type { BaseViewState, LoadRange } from '../base-types';

// ============================================================================
// Canonical Chunk Spatial Index Type
// ============================================================================

/**
 * Canonical chunk spatial index shape consumed by `SpatialQueryBuilder`.
 *
 * All three geometry types (Points, Lines segments, GSplats) produce indices
 * of this shape: a flattened `(numChunks, ndim, 2)` Float32Array of [min, max]
 * bounds per chunk per dimension, plus the chunk count and dimensionality.
 *
 * Lines have a dual index (vertex + segment); only the segment side is used
 * for the chunk query. The vertex side is loaded separately and consumed via
 * a different code path (sorted-indices → contiguous ranges).
 */
export interface ChunkSpatialIndex {
  /** Chunk bounding boxes: shape `(numChunks, ndim, 2)` flattened row-major. */
  chunkBounds: Float32Array;

  /** Number of chunks in the index. */
  chunkCount: number;

  /** Common metadata fields used by the query path. */
  metadata: {
    /** Full dimensionality of the dataset. */
    ndim: number;

    /** Elements per chunk (points/segments/splats). */
    chunk_size?: number;
  };
}

// ============================================================================
// Helpers (also exported for direct use by callers that don't need the builder)
// ============================================================================

/**
 * Build the query position array, padded/truncated to `ndim`.
 */
export function buildQueryPosition(viewState: BaseViewState, ndim: number): number[] {
  const position = new Array<number>(ndim).fill(0);
  for (let d = 0; d < ndim && d < viewState.slicePosition.length; d++) {
    position[d] = viewState.slicePosition[d] ?? 0;
  }
  return position;
}

/** Parameters for `executeSpatialQuery`. */
export interface SpatialQueryParams {
  /** Chunk bounding boxes (flattened Float32Array, layout `[numChunks][ndim][2]`). */
  chunkBounds: Float32Array;
  /** Query position (one value per dimension). */
  queryPosition: number[];
  /** Query tolerance (one value per dimension). */
  queryTolerance: number[];
  /** Number of chunks in the index. */
  numChunks: number;
  /** Dimensionality. */
  ndim: number;
}

/**
 * Run the AABB scan and return matching chunk indices.
 *
 * A chunk matches if its bounds overlap the query box in every dimension.
 * Early-exit on the first non-overlapping dimension keeps the inner loop tight.
 *
 * `logModule` is the label for the diagnostic log line emitted on completion;
 * pass the geometry-specific module so log filtering / triage works.
 */
export function executeSpatialQuery(
  params: SpatialQueryParams,
  logModule: string = Modules.SPATIAL_INDEX
): number[] {
  const { chunkBounds, queryPosition, queryTolerance, numChunks, ndim } = params;
  const matchingChunks: number[] = [];

  for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    let intersects = true;

    for (let d = 0; d < ndim; d++) {
      const offset = chunkIdx * ndim * 2 + d * 2;
      const chunkMin = chunkBounds[offset];
      const chunkMax = chunkBounds[offset + 1];

      const queryMin = queryPosition[d] - queryTolerance[d];
      const queryMax = queryPosition[d] + queryTolerance[d];

      if (chunkMax < queryMin || chunkMin > queryMax) {
        intersects = false;
        break;
      }
    }

    if (intersects) {
      matchingChunks.push(chunkIdx);
    }
  }

  log.query(logModule, `Query: ${matchingChunks.length}/${numChunks} chunks match`);
  return matchingChunks;
}

/** Convert chunk indices to load ranges (clipped at `totalElements`). */
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

/** Coalesce overlapping or adjacent ranges. Result is sorted by `start`. */
export function mergeRanges(ranges: LoadRange[]): LoadRange[] {
  if (ranges.length === 0) return [];

  const sorted = [...ranges].sort((a, b) => a.start - b.start);
  const merged: LoadRange[] = [];
  let current = sorted[0];

  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].start <= current.end) {
      current = { start: current.start, end: Math.max(current.end, sorted[i].end) };
    } else {
      merged.push(current);
      current = sorted[i];
    }
  }
  merged.push(current);
  return merged;
}

/**
 * Decide whether `extendDims` triggers a "load all elements" short-circuit.
 *
 * Returns true iff at least one extend-to-all dimension is currently hidden
 * (not in `viewState.displayDims`). When that happens, the caller should skip
 * the spatial query and load every element so the user can navigate freely
 * along the extended dimension.
 */
export function shouldExtendVisibility(
  extendDims: string[] | undefined,
  viewState: BaseViewState
): boolean {
  if (!extendDims || extendDims.length === 0) return false;
  const dims = viewState.dimensions;
  if (!dims || dims.length === 0) return false;

  const hiddenDimNames = dims
    .filter((_, idx) => !viewState.displayDims.includes(idx))
    .map((meta) => meta.name)
    .filter((name): name is string => name !== undefined);

  return extendDims.some((edim) => hiddenDimNames.includes(edim));
}

/** Single range covering all elements in the dataset. */
export function createLoadAllRange(totalElements: number): LoadRange[] {
  return [{ start: 0, end: totalElements }];
}

// ============================================================================
// SpatialQueryBuilder
// ============================================================================

/**
 * Constructor-options for `SpatialQueryBuilder`.
 *
 * Discriminated union: callers must supply EITHER `geometryType` (tolerance is
 * computed via `tolerance-computer.computeTolerance`) OR `tolerance` (a
 * pre-computed per-dimension array). Both are mutually exclusive.
 */
export type SpatialQueryOptions = {
  /** Total elements in the dataset (for clipping the last range). */
  totalElements: number;

  /** Override `index.metadata.chunk_size` if absent there. */
  chunkSize?: number;

  /** Names of `extend_to_all` dimensions; triggers load-all when any is hidden. */
  extendDims?: string[];

  /**
   * Log module label for diagnostic messages emitted by this query
   * (`Modules.SPATIAL_INDEX_LOADER`, `Modules.LINES_LOADER`,
   * `Modules.GSPLATS_SPATIAL_INDEX_LOADER`, …). Defaults to
   * `Modules.SPATIAL_INDEX` when omitted, but callers should pass their
   * geometry-specific module so log filtering / triage works.
   */
  logModule?: string;
} & (
  | {
      /** Geometry-aware tolerance via `computeTolerance(geometryType, …)`. */
      geometryType: GeometryKind;
      /** Optional tuning passed to `computeTolerance`. */
      toleranceOptions?: ToleranceOptions;
      tolerance?: never;
    }
  | {
      /** Pre-computed per-dimension tolerance (used for points). */
      tolerance: number[];
      geometryType?: never;
      toleranceOptions?: never;
    }
);

/**
 * Builder that runs one chunk-bounds spatial query end-to-end.
 *
 * @example geometry-aware (gsplats / lines)
 * ```ts
 * const ranges = await new SpatialQueryBuilder(index, viewState, {
 *   geometryType: 'gsplats',
 *   totalElements: attrs.n_splats,
 *   chunkSize: attrs.chunk_size,
 *   extendDims: attrs.extend_to_all,
 * }).execute();
 * ```
 *
 * @example pre-computed tolerance (points with EffectiveRadiusConfig)
 * ```ts
 * const tolerance = calculateSpatialQueryTolerance(viewState, config, ndim);
 * const ranges = await new SpatialQueryBuilder(index, viewState, {
 *   tolerance,
 *   totalElements: attrs.n_points,
 *   chunkSize: attrs.chunk_size,
 *   extendDims: attrs.extend_to_all,
 * }).execute();
 * ```
 */
export class SpatialQueryBuilder {
  private readonly chunkBounds: Float32Array;
  private readonly numChunks: number;
  private readonly ndim: number;
  private readonly chunkSize: number;
  private readonly totalElements: number;
  private readonly viewState: BaseViewState;
  private readonly extendDims?: string[];
  private readonly options: SpatialQueryOptions;
  private readonly logModule: string;

  constructor(index: ChunkSpatialIndex, viewState: BaseViewState, options: SpatialQueryOptions) {
    this.chunkBounds = index.chunkBounds;
    this.numChunks = index.chunkCount;
    this.ndim = index.metadata.ndim;
    this.chunkSize = options.chunkSize ?? index.metadata.chunk_size ?? 1000;
    this.totalElements = options.totalElements;
    this.viewState = viewState;
    this.extendDims = options.extendDims;
    this.options = options;
    this.logModule = options.logModule ?? Modules.SPATIAL_INDEX;
  }

  /**
   * Run the query. Returns merged load ranges for visible elements.
   *
   * Short-circuits to "load all" when `extendDims` is active for any currently
   * hidden dimension (the user is navigating across an extended axis).
   */
  async execute(): Promise<LoadRange[]> {
    if (shouldExtendVisibility(this.extendDims, this.viewState)) {
      log.query(this.logModule, `Extending visibility across: ${this.extendDims?.join(', ')}`);
      return createLoadAllRange(this.totalElements);
    }

    const queryPosition = buildQueryPosition(this.viewState, this.ndim);
    const queryTolerance = this.resolveTolerance();

    log.query(this.logModule, `Query: pos=[${queryPosition.map((p) => p.toFixed(2)).join(', ')}]`);
    log.info(
      this.logModule,
      `Query: tol=[${queryTolerance.map((t) => (t > 1e9 ? '∞' : t.toFixed(2))).join(', ')}]`
    );

    const chunkIndices = executeSpatialQuery(
      {
        chunkBounds: this.chunkBounds,
        queryPosition,
        queryTolerance,
        numChunks: this.numChunks,
        ndim: this.ndim,
      },
      this.logModule
    );

    const ranges = chunkIndicesToRanges(chunkIndices, this.chunkSize, this.totalElements);
    return mergeRanges(ranges);
  }

  /** Pick tolerance source per the discriminated-union options. */
  private resolveTolerance(): number[] {
    if (this.options.tolerance !== undefined) {
      return this.options.tolerance;
    }
    return computeTolerance(
      this.options.geometryType,
      this.viewState.displayDims,
      this.ndim,
      this.viewState.dimensions,
      this.options.toleranceOptions
    );
  }
}
