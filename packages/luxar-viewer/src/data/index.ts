/**
 * Data loading module exports.
 *
 * Clean architecture for loading nD points data from Zarr stores
 * with proper spatial index support and aligned attribute loading.
 */

// Main exports
export { loadScene, updateView, updateSceneForDimensions, dispose } from './zarr-loader';

// Core components
export { SceneLoader } from './scene-loader';
export { PointsSpatialIndexLoader } from './points/points-spatial-index-loader';
export { ViewStateManager, type SceneDimensions } from './view-state-manager';

// State management
export { SceneLoaderManager, getSceneLoader } from './scene-loader-manager';
// Note: DataMonitorManager (and the show/hide/toggle/cycleDataMonitor
// accessors) lived in this barrel historically but the file moved to
// ui/monitors/data-monitor-manager.ts in Phase 8.6 (it's a UI panel
// manager, not a data-loading concern). Import from
// `../ui/monitors/data-monitor-manager` directly.

// Types
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
export { DirectoryNavigator } from './utils/directory-navigator';
export type { DirectoryEntry, NavigationResult } from './utils/directory-navigator';

// Array decoding (for Python luxar.encoding compatibility)
export { ArrayDecoder, ArrayRefRegistry, loadAndDecodeOptionalArray } from './utils/array-decoder';
export type { ArrayMetadata } from './utils/array-decoder';

// Lines data loading
export { LinesSpatialIndexLoader } from './lines/lines-spatial-index-loader';
export {
  clipSegmentToSlice,
  buildInstanceBuffers,
  lerp,
  lerpVec3,
  distance3D,
} from './lines/projection';

// Scene graph builder (extracted from SceneLoader for modularity)
export { SceneGraphBuilder } from './utils/scene-graph-builder';
export type { StoreEntry } from './utils/scene-graph-builder';

// Extracted modules (decomposed from SceneLoader)
export {
  getAggregatedPointsAccumulatorStats,
  getAggregatedLinesAccumulatorStats,
  getAggregatedGSplatsAccumulatorStats,
} from './utils/stats-aggregator';
export { LoaderRegistry, type FailedLoaderInfo } from './loaders/loader-registry';
export { computeTolerance, type GeometryType, type ToleranceOptions } from './utils/tolerance-computer';
