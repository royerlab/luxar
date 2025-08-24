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
export type LoaderType = 'spatial-index';

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
  cacheHits: number;
  cacheMisses: number;
  evictions: number;
  errors: number;
  // Performance metrics
  pointsLoaded: number;
  bytesLoaded: number;
  avgQueryTime: number;
  avgLoadTime: number;
  cacheHitRate: number;
  // Memory usage
  memoryUsed: number;
  memoryLimit: number;
  // Spatial index specific metrics
  spatialIndex?: SpatialIndexMetrics;
}

/**
 * Spatial index specific metrics
 */
export interface SpatialIndexMetrics {
  gridShape: number[];
  gridOrigin: number[];
  cellSize: number[];
  occupiedCells: number;
  totalCells: number;
  avgCellsPerQuery: number;
  avgPointsPerCell: number;
  queryEfficiency: number; // Points loaded / points in query region
  lastQueryBounds?: { min: number[]; max: number[] };
  cellsInCache: number;
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
  totalPoints: number;
  totalMemory: number;
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
  activeTab: 'overview' | 'spatial' | 'performance' | 'insights';
  selectedLoader?: string;
  timeRange: number; // Seconds of history to show
  spatialViewDimensions?: [number, number]; // Which 2D slice to show
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
