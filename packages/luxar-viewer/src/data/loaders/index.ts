/**
 * Unified Loader Architecture - Shared Components
 *
 * This module provides the foundation for the unified spatial index loader architecture,
 * enabling code reuse across Points, Lines, and GSplats loaders.
 *
 * ## Components
 *
 * ### base-types.ts
 * Common type definitions shared by all loaders:
 * - BaseViewState: Common view state interface
 * - LoadRange: Generic range for array loading
 * - BaseChunkSpatialIndex: Common spatial index interface
 * - SpatialDataLoader: Generic loader interface
 *
 * ### range-loader.ts
 * Unified encoding dispatch for range-based array loading:
 * - RangeLoader: Handles broadcasted, quantized, LUT, array_ref, and direct encodings
 * - Worker dispatch with main thread fallback
 * - Replaces ~600 lines of duplicated code
 *
 * ### spatial-query-builder.ts
 * Unified spatial query logic:
 * - SpatialQueryBuilder: Fluent API for building and executing spatial queries
 * - computeQueryTolerance: Unified tolerance calculation
 * - executeSpatialQuery: Worker dispatch for spatial queries
 * - Replaces ~300 lines of duplicated code
 *
 * ## Usage
 *
 * ```typescript
 * import {
 *   RangeLoader,
 *   SpatialQueryBuilder,
 *   type BaseViewState,
 *   type LoadRange,
 * } from './loaders';
 *
 * // Build spatial query
 * const builder = new SpatialQueryBuilder(chunkIndex, viewState, totalElements);
 * const ranges = await builder.withMaxRadius(5.0).execute();
 *
 * // Load data with encoding dispatch
 * const rangeLoader = new RangeLoader(refRegistry);
 * const elementsWritten = await rangeLoader.loadRanges(
 *   array, attrs, ranges, outputBuffer, totalElements
 * );
 * ```
 *
 * ## Architecture
 *
 * ```
 * ┌─────────────────────────────────────────────────────────────┐
 * │                    UNIFIED LOADERS                          │
 * │  (Points, Lines, GSplats extend common patterns)            │
 * │                          │                                  │
 * │          ┌───────────────┼───────────────┐                  │
 * │          ▼               ▼               ▼                  │
 * │   SpatialQueryBuilder  RangeLoader   Accumulator            │
 * │   (query logic)     (encoding)    (buffer reuse)            │
 * │          │               │               │                  │
 * │          └───────────────┼───────────────┘                  │
 * │                          ▼                                  │
 * │                    WorkerPool                               │
 * │              (CPU offload + WASM)                           │
 * └─────────────────────────────────────────────────────────────┘
 * ```
 *
 * @module data/loaders
 */

// Base types
export {
  type BaseViewState,
  type LoadRange,
  type BaseChunkSpatialIndex,
  type LoaderOptions,
  type LoaderDependencies,
  type BaseLoader,
  type SpatialDataLoader,
  type AccumulatorBuffers,
  type AccumulatorStats,
  type LoadedDataMetadata,
  hasDimensionMetadata,
  getDisplayDimCount,
  isHiddenDimension,
} from './base-types';

// Range loader (encoding dispatch)
export {
  RangeLoader,
  type RangeLoaderConfig,
  type EncodingType,
  getSharedRangeLoader,
  getSharedRefRegistry,
  resetSharedRangeLoader,
} from './range-loader';

// Spatial query builder
export {
  SpatialQueryBuilder,
  computeQueryTolerance,
  buildQueryPosition,
  executeSpatialQuery,
  chunkIndicesToRanges,
  mergeRanges,
  shouldExtendVisibility,
  createLoadAllRange,
  type ToleranceConfig,
  type SpatialQueryParams,
  DEFAULT_HIDDEN_DIM_TOLERANCE,
  DISPLAYED_DIM_TOLERANCE,
} from './spatial-query-builder';

// Transferable accumulator (zero-allocation + worker offload)
export {
  TransferableAccumulator,
  createPointsAccumulator,
  createLinesAccumulator,
  createGSplatsAccumulator,
  type BufferShape,
  type TransferableBuffers,
  type PointsBuffers,
  type LinesBuffers,
  type GSplatsBuffers,
  type WorkerProjectionRequest,
  type WorkerProjectionResponse,
} from './transferable-accumulator';

// Integration example (reference implementation)
export {
  PointsLoaderIntegrationExample,
  queryVisibleRangesExample,
  fullLoadingPipelineExample,
} from './integration-example';
