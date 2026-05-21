import type { DataLoadingMemoryConfig } from './types';

/**
 * Data loading memory configuration
 */
export const dataLoadingMemoryConfig: DataLoadingMemoryConfig = {
  targetHeapUsage: 0.8,
  minCacheMB: 128,
  checkIntervalMs: 10000,
  adjustmentThresholds: {
    critical: 0.85,
    high: 0.7,
  },
};
