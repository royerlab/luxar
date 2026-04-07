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
export { PointSpatialIndexLoader } from './point-spatial-index-loader';
export { ViewStateManager, type SceneDimensions } from './view-state-manager';

// State management
export { SceneLoaderManager, getSceneLoader } from './scene-loader-manager';
export {
  DataMonitorManager,
  getDataMonitor,
  showDataMonitor,
  hideDataMonitor,
  toggleDataMonitor,
  cycleDataMonitor,
} from './data-monitor-manager';

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

// Chunk-based spatial index (NEW)
export {
  loadChunkSpatialIndex,
  queryChunksForView,
  chunkIndicesToRanges,
  mergePointRanges,
  type ChunkSpatialIndex,
} from './chunk-spatial-index';

// Directory navigation
export { DirectoryNavigator } from './directory-navigator';
export type { DirectoryEntry, NavigationResult } from './directory-navigator';

// Array decoding (for Python luxar.encoding compatibility)
export { ArrayDecoder, ArrayRefRegistry, loadAndDecodeOptionalArray } from './array-decoder';
export type { ArrayMetadata } from './array-decoder';

// Lines data loading
export { LinesSpatialIndexLoader } from './lines-spatial-index-loader';
export {
  clipSegmentToSlice,
  buildInstanceBuffers,
  lerp,
  lerpVec3,
  distance3D,
} from './lines-spatial-index-loader';
export {
  loadLinesChunkSpatialIndex,
  querySegmentChunksForView,
  queryVertexChunksForView,
  computeLinesTolerance,
  segmentChunkIndicesToRanges,
  vertexChunkIndicesToRanges,
  mergeRanges,
  computeVertexChunksForIndices,
  computeVertexRangesFromIndices,
} from './lines-chunk-spatial-index';

// Scene graph builder (extracted from SceneLoader for modularity)
export { SceneGraphBuilder } from './scene-graph-builder';
export type { StoreEntry } from './scene-graph-builder';

// Extracted modules (decomposed from SceneLoader)
export {
  getAggregatedPointsAccumulatorStats,
  getAggregatedLinesAccumulatorStats,
  getAggregatedGSplatsAccumulatorStats,
} from './stats-aggregator';
export { LoaderRegistry, type FailedLoaderInfo } from './loader-registry';
export { computeTolerance, type GeometryType, type ToleranceOptions } from './tolerance-computer';
