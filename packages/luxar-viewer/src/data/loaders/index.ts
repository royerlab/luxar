/**
 * Unified Loader Architecture - Shared Components
 *
 * Foundation for the unified spatial-index loader architecture, shared by the
 * Points, Lines, and GSplats loaders, plus label-picking and overlay loaders.
 *
 * Subpackages:
 * - **spatial-query/** — chunk-bounds probe, tolerance computer, AABB query
 *   builder, and the encoding-dispatching range loader.
 * - **picking/** — label and image-label loaders consumed by core/app/picking.
 * - **overlays/** — overlay config loader consumed by the overlay manager.
 *
 * Root-level files:
 * - **base-types.ts** — canonical shared types (`BaseViewState`, `LoadRange`, …).
 * - **color-loader.ts** — color-attribute encode/decode shared by the three
 *   spatial-index loaders and the mesh whole-node loader.
 * - **chunk-bounds-loader.ts** — zarr chunk-bounds probe (layer below spatial-query).
 * - **loader-metrics.ts**, **monitor-events.ts**, **once-init.ts**,
 *   **extend-to-all-preflight.ts** — small cross-cutting helpers used by every
 *   spatial loader.
 *
 * External callers should import from this barrel rather than reaching into
 * deep paths so subpackage internals can move without callsite churn.
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
  type LoadedDataMetadata,
  hasDimensionMetadata,
  getDisplayDimCount,
  isHiddenDimension,
} from './base-types';

// Spatial-query pipeline
export {
  RangeLoader,
  type RangeLoaderConfig,
  type EncodingType,
} from './spatial-query/range-loader';
export { prefetchRangesIntoCache } from './spatial-query/prefetch-ranges';
export { isAbortError } from './abort-error';
export {
  getSharedRangeLoader,
  getSharedRefRegistry,
  resetSharedRangeLoader,
} from './spatial-query/range-loader/shared-instance';

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
} from './spatial-query/spatial-query-builder';

export {
  computeTolerance,
  type DimensionInfo,
  type ToleranceOptions,
} from './spatial-query/tolerance-computer';

export { fetchChunkBoundsArray, type ChunkBoundsArray } from './chunk-bounds-loader';

// Color attributes
export {
  allocateColorBuffer,
  getExpectedColorType,
  colorBufferTypeMatches,
  restoreOriginalDtype,
  loadColorRanges,
  assertColorLayout,
  colorComponentsOf,
  type ColorRange,
  type ColorBuffer,
  type ColorBufferKind,
} from './color-loader';

// Cross-cutting utilities
export {
  recordLoadEvent,
  computeLoadLatency,
  finishQueryTracking,
  makeInitialLoaderMetrics,
  buildSpatialIndexMetrics,
  type LoaderMetricsCounters,
  type QueryMetricsCounters,
} from './loader-metrics';

export {
  loadSliceWithCache,
  recordLoadMetrics,
  runWithActiveSignal,
  runWithResidencyProbe,
  type SpatialFacadeCtx,
  type FacadeViewState,
  type FacadeMetrics,
} from './spatial-facade';

export { LoaderEventEmitter } from './monitor-events';

export { OnceInit } from './once-init';

export { warnExtendToAllNoDimensions, announceExtendToAllOnce } from './extend-to-all-preflight';

// Picking
export { LabelLoader } from './picking/label-loader';
export { ImageLabelLoader } from './picking/image-label-loader';

// Overlays
export {
  loadOverlayConfigs,
  MAX_OVERLAY_HTML_CHARS,
  type OverlayConfig,
} from './overlays/overlay-loader';
