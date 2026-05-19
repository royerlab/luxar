import type { DataLoadingNetworkConfig } from './types';

/**
 * Data loading network configuration
 */
export const dataLoadingNetworkConfig: DataLoadingNetworkConfig = {
  timeoutMs: 30000,
  // Dedicated short budget for the L2 cache-validation HEAD probe
  // (MultiLevelCachingStore.getRemoteContentHash). On flaky networks this
  // path must NOT block scene loading for the full timeoutMs — failing
  // fast is better since we can render from cached data.
  validationTimeoutMs: 5000,
  maxConcurrent: 6,
  retryAttempts: 3,
};
