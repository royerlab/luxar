/**
 * Type definitions for the new Data Loading Monitor architecture.
 *
 * This module defines the interfaces and types for the event-driven
 * monitoring system that tracks spatial index queries, cache performance,
 * and loading statistics in real-time.
 */

import type { PointRange } from '../data';

/**
 * Event types emitted by data loaders
 */
export type MonitorEventType =
  | 'query' // Spatial index or range query initiated
  | 'load' // Data loaded from source
  | 'cache-hit' // Data found in cache
  | 'cache-miss' // Data not in cache, needs loading
  | 'evict' // Data evicted from cache
  | 'error' // Loading or query error
  | 'prefetch'; // Prefetch operation

/**
 * Loader types in the system
 */
export type LoaderType = 'point-spatial-index';

/**
 * Event emitted by data loaders for monitoring
 */
export interface MonitorEvent {
  type: MonitorEventType;
  loader: LoaderType;
  timestamp: number;
  data: {
    path?: string;
    arrayName?: string;
    ranges?: PointRange[];
    points?: number;
    cells?: number;
    latency?: number;
    memory?: number;
    cacheKey?: string;
    error?: string;
    // Spatial-specific data
    queryPosition?: number[];
    queryTolerance?: number[];
    gridBounds?: { min: number[]; max: number[] };
  };
}

/**
 * Listener function for monitor events
 */
export type MonitorEventListener = (event: MonitorEvent) => void;

/**
 * Interface for objects that can be monitored
 */
export interface LoaderMonitor {
  addEventListener(listener: MonitorEventListener): void;
  removeEventListener(listener: MonitorEventListener): void;
  getMetrics(): LoaderMetrics;
  getActiveQueries(): QueryInfo[];
}

/**
 * Metrics for a specific loader
 */
export interface LoaderMetrics {
  type: LoaderType;
  path: string;
  // Basic counters
  queries: number;
  loads: number;
  evictions: number;
  errors: number;
  // Performance metrics
  pointsLoaded: number; // Cumulative (for throughput calculation)
  bytesLoaded: number;
  // Dataset info (NEW)
  datasetSize: number; // Total points in dataset (from zarr metadata)
  visiblePoints: number; // Currently visible/rendered points (non-cumulative)
  avgQueryTime: number;
  avgLoadTime: number;
  // Memory usage
  memoryUsed: number;
  memoryLimit: number;
  // Spatial index specific metrics
  spatialIndex?: PointSpatialIndexMetrics;
}

/**
 * Spatial index specific metrics
 */
export interface PointSpatialIndexMetrics {
  gridShape: number[];
  gridOrigin: number[];
  cellSize: number[];
  occupiedCells: number;
  totalCells: number;
  avgCellsPerQuery: number;
  avgPointsPerCell: number;
  queryEfficiency: number; // Points loaded / points in query region
  lastQueryBounds?: { min: number[]; max: number[] };
  rangesInCache: number; // Number of cached range queries
}

/**
 * Information about an active or recent query
 */
export interface QueryInfo {
  id: string;
  loader: LoaderType;
  path: string;
  startTime: number;
  endTime?: number;
  status: 'pending' | 'loading' | 'complete' | 'error';
  cells?: number;
  points?: number;
  ranges?: PointRange[];
  fromCache?: boolean;
  error?: string;
}

/**
 * Recommendation for performance improvement
 */
export interface Recommendation {
  id: string;
  severity: 'info' | 'warning' | 'error';
  category: 'performance' | 'memory' | 'configuration';
  title: string;
  message: string;
  suggestion?: string;
  metric?: string;
  value?: number;
  threshold?: number;
}

/**
 * Monitor display configuration
 */
export interface MonitorConfig {
  // Display settings
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  theme: 'dark' | 'light' | 'auto';
  defaultView: 'compact' | 'detailed' | 'debug';

  // Update settings
  updateInterval: number; // ms between UI updates
  maxEvents: number; // Maximum events to keep in history

  // Feature flags
  showSpatialGrid: boolean;
  showTimeline: boolean;
  showRecommendations: boolean;
  autoExpand: boolean; // Auto-expand on warnings

  // Performance
  enableProfiling: boolean;
  sampleRate: number; // Sample 1 in N events for profiling
}

/**
 * Aggregated statistics across all loaders
 */
export interface GlobalStats {
  totalLoaders: number;
  activeSpatialLoaders: number;
  activeFallbackLoaders: number; // Kept for compatibility but always 0
  totalPoints: number; // Cumulative points loaded (for throughput)
  totalMemory: number;
  // Dataset metrics - Points
  datasetSize: number; // Total points in all datasets
  visiblePoints: number; // Currently visible/rendered points
  // Dataset metrics - Lines
  datasetSegments: number; // Total segments in all line datasets
  visibleSegments: number; // Currently visible/rendered segments (for lines, typically equals total)
  // Additional properties expected by tests
  totalQueries: number;
  totalLoads: number;
  totalCacheHits: number;
  totalPointsLoaded: number;
  totalMemoryUsed: number;
  globalCacheHitRate: number;
  avgQueryTime: number;
  queriesPerSecond: number;
  recommendations: Recommendation[];
}

/**
 * UI Component state
 */
export interface MonitorUIState {
  isVisible: boolean;
  isExpanded: boolean;
  activeTab: 'overview' | 'cache' | 'performance' | 'insights'; // Removed 'spatial' as it's not implemented
  selectedLoader?: string;
  timeRange: number; // Seconds of history to show
  spatialViewDimensions?: [number, number]; // Which 2D slice to show (for future use)
}

/**
 * Cache metrics for detailed analytics
 */
export interface CacheMetrics {
  totalCacheMemory: number;
  memoryLimit: number;
  memoryPercent: number;
  totalEntries: number;
  totalAccesses: number;
  recentHitRate: number;
  evictionsPerMin: number;
  avgEntrySize: number;
  reuseRatio: number;
  hitsPerSecond: number;
  missesPerSecond: number;
  avgAccessTime: number;
  queriesPerSec: number;
  loadsPerSec: number;
  bandwidth: number;
  /** L1 memory cache breakdown (optional, only when CacheStatsProvider connected) */
  l1?: {
    size: number;
    count: number;
    hits: number;
    misses: number;
    evictions: number;
  };
  /** L2 OPFS cache breakdown (optional, only when CacheStatsProvider connected) */
  l2?: {
    size: number;
    count: number;
    reads: number;
    writes: number;
  };
  /** Whether caching is enabled */
  enabled?: boolean;
  /** Network I/O stats (optional, only when CacheStatsProvider connected) */
  network?: {
    bytesTransferred: number;
    requestCount: number;
    bandwidth: number;
  };
}

/**
 * Performance timeline data point
 */
export interface TimelinePoint {
  timestamp: number;
  queryTime?: number;
  loadTime?: number;
  cacheHitRate?: number;
  memoryUsed?: number;
  pointsLoaded?: number;
  loaderType?: LoaderType;
  event?: MonitorEventType;
}

/**
 * Spatial grid cell state for visualization
 */
export interface GridCellState {
  x: number;
  y: number;
  z?: number;
  isOccupied: boolean;
  isCached: boolean;
  isLoading: boolean;
  isQueried: boolean;
  points: number;
  lastAccess?: number;
}

/**
 * Node type for scene graph display
 */
export type SceneGraphNodeType = 'scene' | 'group' | 'points' | 'lines' | 'mesh';

/**
 * Scene graph node for UI display.
 * Simplified version of SceneNode from data-loader-types.ts.
 */
export interface SceneGraphNode {
  /** Path in the zarr store */
  path: string;
  /** Display name (last component of path or 'Scene') */
  name: string;
  /** Node type */
  type: SceneGraphNodeType;
  /** Number of points (for points nodes) */
  pointCount?: number;
  /** Number of segments (for lines nodes) */
  segmentCount?: number;
  /** Number of vertices (for lines nodes) */
  vertexCount?: number;
  /** Whether this node is currently loading */
  isLoading?: boolean;
  /** Whether this node has a spatial index */
  hasSpatialIndex?: boolean;
  /** Child nodes */
  children: SceneGraphNode[];
  /** UI state: whether node is expanded in tree view */
  isExpanded?: boolean;
}

/**
 * Scene graph state for monitor
 */
export interface SceneGraphState {
  /** Root node of the scene graph */
  root: SceneGraphNode | null;
  /** Total number of nodes */
  totalNodes: number;
  /** Number of points nodes */
  pointsNodes: number;
  /** Number of lines nodes */
  linesNodes: number;
  /** Total points across all nodes */
  totalPoints: number;
  /** Total segments across all lines */
  totalSegments: number;
}

/**
 * Interface for objects that provide cache statistics.
 * Used for loose coupling between TwoLevelCachingStore and DataLoadingMonitor.
 */
export interface CacheStatsProvider {
  /** Get current cache statistics */
  getStats(): {
    l1: {
      metadataSize: number;
      chunksSize: number;
      metadataCount: number;
      chunksCount: number;
      hits: number;
      misses: number;
      evictions: number;
    };
    l2: {
      size: number;
      count: number;
      reads: number;
      writes: number;
    };
    network: {
      bytesTransferred: number;
      requestCount: number;
      bandwidth: number;
    };
  };
  /** Clear L1 memory cache */
  clearL1(): void;
  /** Clear L2 OPFS cache */
  clearL2(): Promise<void>;
  /** Clear all caches (L1 + L2) */
  clearAll(): Promise<void>;
  /** Check if caching is enabled */
  isEnabled(): boolean;
}
