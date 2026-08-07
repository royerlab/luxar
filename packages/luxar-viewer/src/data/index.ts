/**
 * Data loading module exports.
 *
 * Clean architecture for loading nD points data from Zarr stores
 * with proper spatial index support and aligned attribute loading.
 */

// Main exports
export {
  loadScene,
  updateView,
  updateSceneForDimensions,
  prefetchSceneForDimensions,
  releasePrefetchResources,
  dispose,
} from './zarr-loader';

// Core components
export { SceneLoader } from './scene-loader';
export { PointsSpatialIndexLoader } from './points/points-spatial-index-loader';
export { ViewStateManager, type SceneDimensions } from './view-state-manager';

// State management
export { SceneLoaderManager, getSceneLoader } from './scene-loader-manager';
// Note: DataMonitorManager (and the cycleDataMonitor accessor) lives
// in `../ui/data-monitor-manager` — it's a UI panel
// manager, not a data-loading concern. Import from there directly.
// The show/hide/toggle accessors were removed; production code uses
// the 'panel-cycle' / 'panel-hide' event bus instead.

/**
 * Core data-loading contract types: the points {@link DataLoader} interface
 * (the Points-specific loader contract, paralleled by separate
 * `LinesDataLoader`/`GSplatsDataLoader`/`MeshDataLoader` for the other three
 * geometry types), the {@link ViewState} query passed to loaders,
 * loaded-payload shapes ({@link LoadedPointsData}), loader configuration/stats,
 * and the scene-graph ({@link SceneNode}) and spatial-query result types.
 *
 * These are re-exports of convenience for the barrel's own consumers, not the
 * canonical import path. `data-loader-types` forwards the points types
 * (`DataLoader`, `ViewState`, `LoadedPointsData`, `PointRange`) from
 * `types/points` — where new code should import those from — and declares the
 * loader-side shapes (`LoaderConfig`, `SceneNode`, `SpatialQueryResult`,
 * `LoaderStats`) itself.
 */
export type {
  DataLoader,
  ViewState,
  LoadedPointsData,
  LoaderConfig,
  PointRange,
  SceneNode,
  SpatialQueryResult,
  LoaderStats,
} from './data-loader-types';

// Spatial query (canonical chunk-bounds API used by Points/Lines/GSplats).
// Re-exports from the loaders barrel so the geometry-specific loaders only
// import via `./loaders`.
export {
  SpatialQueryBuilder,
  type ChunkSpatialIndex,
  type SpatialQueryOptions,
  chunkIndicesToRanges,
  mergeRanges,
} from './loaders';

// Directory navigation
export { DirectoryNavigator } from './nav/directory-navigator';
/**
 * Result types for {@link DirectoryNavigator}: a single {@link DirectoryEntry}
 * listing row and the {@link NavigationResult} returned when browsing a store's
 * directory tree.
 */
export type { DirectoryEntry, NavigationResult } from './nav/directory-navigator';

// Array decoding (for Python luxar.encoding compatibility)
export {
  ArrayDecoder,
  ArrayRefRegistry,
  loadAndDecodeOptionalArray,
} from './array-decoder/decoder';
/**
 * An encoded array's on-disk `.zattrs` metadata (shape, dtype, and nested
 * `encoding` block) for Python `luxar.encoding` arrays. This is the stored
 * input that {@link ArrayDecoder}`.decode` consumes to reconstruct the array,
 * not something the decoder produces.
 */
export type { ArrayMetadata } from './array-decoder/decoder';

// Lines data loading
export { LinesSpatialIndexLoader } from './lines/lines-spatial-index-loader';
// nD → 3D line projection now lives solely in the worker dispatcher
// (`workers/data-worker/projection/lines.ts`); only the empty-payload
// factory remains in `lines/projection.ts` and is consumed directly by
// the loader, so nothing is re-exported from the barrel here.

// Extracted modules (decomposed from SceneLoader)
export {
  getAggregatedPointsAccumulatorStats,
  getAggregatedLinesAccumulatorStats,
  getAggregatedGSplatsAccumulatorStats,
} from './stats/aggregator';
export { LoaderRegistry, type FailedLoaderInfo } from './scene-loader/loaders/loader-registry';
export { computeTolerance, type ToleranceOptions } from './loaders';
