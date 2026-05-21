import type { DataLoadingNetworkConfig } from './network/types';
export type { DataLoadingNetworkConfig };

import type { DataLoadingMemoryConfig } from './memory/types';
export type { DataLoadingMemoryConfig };

import type {
  DataLoadingMonitorConfig,
  MonitorLimits,
  MonitorThresholds,
  MonitorTimings,
} from './monitor/types';
export type { DataLoadingMonitorConfig, MonitorLimits, MonitorThresholds, MonitorTimings };

import type { DataLoadingSpatialConfig } from './spatial/types';
export type { DataLoadingSpatialConfig };

import type { DataLoadingPerformanceConfig } from './performance/types';
export type { DataLoadingPerformanceConfig };

/**
 * Data loading configuration
 */
export interface DataLoadingConfig {
  spatial: DataLoadingSpatialConfig;
  network: DataLoadingNetworkConfig;
  memory: DataLoadingMemoryConfig;
  monitor: DataLoadingMonitorConfig;
  performance: DataLoadingPerformanceConfig;
}
