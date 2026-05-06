/**
 * Unified Loader Architecture - Shared Components
 *
 * Foundation for the unified spatial-index loader architecture, shared by the
 * Points, Lines, and GSplats loaders.
 *
 * - **base-types.ts** — common type definitions (`BaseViewState`, `LoadRange`,
 *   `BaseChunkSpatialIndex`, `SpatialDataLoader`).
 * - **range-loader.ts** — unified encoding dispatch (broadcasted, quantized,
 *   LUT, array_ref, direct), with worker offload + main-thread fallback.
 * - **spatial-query-builder.ts** — canonical chunk-bounds query API.
 *   `SpatialQueryBuilder` accepts either a `geometryType` (delegates tolerance
 *   to `tolerance-computer.computeTolerance`) or a pre-computed `tolerance`
 *   (used by points, which has bespoke `EffectiveRadiusConfig` semantics).
 * - **transferable-accumulator.ts** — zero-allocation buffers transferable to
 *   workers for CPU-offloaded projection.
 *
 * @example
 * ```ts
 * import {
 *   RangeLoader,
 *   SpatialQueryBuilder,
 *   type BaseViewState,
 *   type LoadRange,
 * } from './loaders';
 *
 * const ranges = await new SpatialQueryBuilder(chunkIndex, viewState, {
 *   geometryType: 'gsplats',
 *   totalElements: attrs.n_splats,
 *   chunkSize: attrs.chunk_size,
 *   extendDims: attrs.extend_to_all,
 * }).execute();
 *
 * const rangeLoader = new RangeLoader(refRegistry);
 * await rangeLoader.loadRanges(array, attrs, ranges, outputBuffer, totalElements);
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
  type SpatialQueryOptions,
  type ChunkSpatialIndex,
  type SpatialQueryParams,
  buildQueryPosition,
  executeSpatialQuery,
  chunkIndicesToRanges,
  mergeRanges,
  shouldExtendVisibility,
  createLoadAllRange,
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
