import type { DataLoadingConfig } from './types';
import { dataLoadingSpatialConfig } from './spatial/data';
import { dataLoadingNetworkConfig } from './network/data';
import { dataLoadingMemoryConfig } from './memory/data';
import { dataLoadingMonitorConfig } from './monitor/data';
import { dataLoadingPerformanceConfig } from './performance/data';

/**
 * Data loading configuration
 */
export const dataLoadingConfig: DataLoadingConfig = {
  spatial: dataLoadingSpatialConfig,
  network: dataLoadingNetworkConfig,
  memory: dataLoadingMemoryConfig,
  monitor: dataLoadingMonitorConfig,
  performance: dataLoadingPerformanceConfig,
};
