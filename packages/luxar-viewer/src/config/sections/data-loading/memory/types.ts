/**
 * Data loading memory configuration
 */
export interface DataLoadingMemoryConfig {
  targetHeapUsage: number;
  minCacheMB: number;
  checkIntervalMs: number;
  adjustmentThresholds: {
    critical: number;
    high: number;
  };
}
