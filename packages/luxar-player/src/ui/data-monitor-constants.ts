/**
 * Data Loading Monitor Constants
 * 
 * Centralized configuration constants for the Data Loading Monitor.
 * These values control timing, performance, and behavior of the monitoring system.
 */

/**
 * Timing constants for monitor operations
 */
export const MonitorTimings = {
  /**
   * Interval between automatic event cleanup cycles
   * Prevents memory buildup from old events
   */
  EVENT_CLEANUP_INTERVAL: 30000, // 30 seconds
  
  /**
   * Maximum age for events before they are removed
   * Older events are purged to prevent memory leaks
   */
  MAX_EVENT_AGE: 300000, // 5 minutes
  
  /**
   * Cache timeout for rate calculations
   * Prevents recalculating rates more than once per second
   */
  RATES_CACHE_TIMEOUT: 1000, // 1 second
  
  /**
   * Default UI update interval
   * Controls how often the monitor UI refreshes
   */
  DEFAULT_UPDATE_INTERVAL: 100, // 100ms = 10Hz
  
  /**
   * Minimum time between timeline renders
   * Prevents excessive canvas redraws
   */
  MIN_RENDER_INTERVAL: 100, // 100ms = 10 FPS max
  
  /**
   * Maximum interval between timeline data points
   * Controls timeline data aggregation
   */
  TIMELINE_POINT_INTERVAL: 200, // 200ms = max 5 points per second
  
  /**
   * Default time range for timeline display
   */
  DEFAULT_TIME_RANGE: 60, // 60 seconds
  
  /**
   * Query cleanup check interval
   * Check for old queries every N queries
   */
  QUERY_CLEANUP_CHECK_INTERVAL: 10, // Check every 10 queries
  
  /**
   * Maximum age for queries before cleanup
   */
  MAX_QUERY_AGE: 60000, // 1 minute
} as const;

/**
 * Performance thresholds for recommendations
 */
export const PerformanceThresholds = {
  /**
   * Cache hit rate below this is concerning
   */
  LOW_CACHE_HIT_RATE: 30, // 30%
  
  /**
   * Query time above this is considered slow
   */
  HIGH_QUERY_TIME: 100, // 100ms
  
  /**
   * Load time above this is considered slow
   */
  HIGH_LOAD_TIME: 500, // 500ms
  
  /**
   * Memory usage above this percentage triggers warnings
   */
  HIGH_MEMORY_USAGE: 0.8, // 80%
  
  /**
   * Error rate above this percentage is concerning
   */
  HIGH_ERROR_RATE: 0.05, // 5%
  
  /**
   * Query efficiency below this indicates poor spatial indexing
   */
  LOW_QUERY_EFFICIENCY: 0.5, // 50%
} as const;

/**
 * Limits and defaults for monitor behavior
 */
export const MonitorLimits = {
  /**
   * Maximum number of events to keep in history
   */
  MAX_EVENTS: 1000,
  
  /**
   * Maximum timeline points to keep
   */
  MAX_TIMELINE_POINTS: 300, // 5 minutes at 1Hz
  
  /**
   * Maximum event history for advisor
   */
  MAX_ADVISOR_HISTORY: 100,
  
  /**
   * Default memory limit if not provided
   */
  DEFAULT_MEMORY_LIMIT: 1024 * 1024 * 1024, // 1GB
  
  /**
   * Time window for rate calculations
   */
  RATE_CALCULATION_WINDOW: 5000, // 5 seconds
  
  /**
   * Time window for bandwidth calculation
   */
  BANDWIDTH_CALCULATION_WINDOW: 1000, // 1 second
} as const;

/**
 * Type guard to check if a string is a valid tab
 */
export const VALID_TABS = ['overview', 'cache', 'performance', 'insights'] as const;
export type ValidTab = typeof VALID_TABS[number];

export function isValidTab(tab: string): tab is ValidTab {
  return VALID_TABS.includes(tab as ValidTab);
}

/**
 * Type guard to check if a loader has cache stats
 */
export interface LoaderWithCacheStats {
  getCacheStats(): {
    hits: number;
    misses: number;
    avgAccessTime: number;
  };
}

export function hasCacheStats(loader: unknown): loader is LoaderWithCacheStats {
  return (
    typeof loader === 'object' &&
    loader !== null &&
    'getCacheStats' in loader &&
    typeof (loader as Record<string, unknown>).getCacheStats === 'function'
  );
}