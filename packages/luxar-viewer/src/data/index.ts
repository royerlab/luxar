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
  getCacheStats,
  clearCaches,
  dispose,
} from './zarr-loader';

// Core components
export { SceneLoader } from './scene-loader';
export { SpatialIndexLoader } from './spatial-index-loader';
export { RangeCache, RangeCacheKey } from './range-cache';

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
  PointsData,
  LoaderConfig,
  CacheStats,
  PointRange,
  SceneNode,
  SpatialQueryResult,
  LoaderStats,
} from './data-loader-types';

// Spatial index utilities
export {
  loadSpatialIndex,
  querySpatialIndex,
  mergePointRanges,
  calculateChunksToLoad,
  estimateMemoryUsage,
  debugSpatialIndex,
  type SpatialIndex,
  type SpatialIndexMetadata,
} from './spatial-index';

// Directory navigation
export { DirectoryNavigator } from './directory-navigator';
export type { DirectoryEntry, NavigationResult } from './directory-navigator';
