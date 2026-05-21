/**
 * Data loading monitor timings
 */
export interface MonitorTimings {
  eventCleanupInterval: number;
  maxEventAge: number;
  ratesCacheTimeout: number;
  defaultUpdateInterval: number;
  minRenderInterval: number;
  timelinePointInterval: number;
  defaultTimeRange: number;
  queryCleanupCheckInterval: number;
  maxQueryAge: number;
}

/**
 * Data loading monitor thresholds
 */
export interface MonitorThresholds {
  lowCacheHitRate: number;
  highQueryTime: number;
  highLoadTime: number;
  highMemoryUsage: number;
  highErrorRate: number;
  lowQueryEfficiency: number;
}

/**
 * Data loading monitor limits
 */
export interface MonitorLimits {
  maxEvents: number;
  maxTimelinePoints: number;
  maxAdvisorHistory: number;
  defaultMemoryLimit: number;
  rateCalculationWindow: number;
  bandwidthCalculationWindow: number;
}

/**
 * Data loading monitor configuration
 */
export interface DataLoadingMonitorConfig {
  timings: MonitorTimings;
  thresholds: MonitorThresholds;
  limits: MonitorLimits;
}
