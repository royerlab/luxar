import type { DataLoadingMonitorConfig } from './types';

/**
 * Data loading monitor configuration
 */
export const dataLoadingMonitorConfig: DataLoadingMonitorConfig = {
  timings: {
    eventCleanupInterval: 30000,
    maxEventAge: 300000,
    ratesCacheTimeout: 1000,
    defaultUpdateInterval: 100,
    minRenderInterval: 100,
    timelinePointInterval: 200,
    defaultTimeRange: 60,
    queryCleanupCheckInterval: 10,
    maxQueryAge: 60000,
  },
  thresholds: {
    lowCacheHitRate: 30,
    highQueryTime: 100,
    highLoadTime: 500,
    highMemoryUsage: 0.8,
    highErrorRate: 0.05,
    lowQueryEfficiency: 0.5,
  },
  limits: {
    maxEvents: 1000,
    maxTimelinePoints: 300,
    maxAdvisorHistory: 100,
    defaultMemoryLimit: 1024 * 1024 * 1024,
    rateCalculationWindow: 5000,
    bandwidthCalculationWindow: 1000,
  },
};
