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
    queryCleanupCheckInterval: 10,
    maxQueryAge: 60000,
  },
  thresholds: {
    highQueryTime: 100,
    highLoadTime: 500,
    highErrorRate: 0.05,
    lowQueryEfficiency: 0.5,
  },
  limits: {
    maxEvents: 1000,
    maxAdvisorHistory: 100,
    rateCalculationWindow: 5000,
    bandwidthCalculationWindow: 1000,
  },
};
