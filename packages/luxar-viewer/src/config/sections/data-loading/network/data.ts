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
  // Concurrent prefetch fetches (ChunkPrefetcher). Default matches the value
  // the prefetcher always used in practice (this knob was unwired until 2026-07).
  maxConcurrent: 4,
  retryAttempts: 3,
  // Fetch-gate lane widths (utils/fetch-concurrency.ts); see FetchGateConfig.
  fetchGate: {
    maxChunkFetches: 24,
    maxMultiplexedChunkFetches: 96,
    maxMetadataFetches: 4,
    http1MaxChunkFetches: 4,
    http1MaxMetadataFetches: 2,
    speculativeShare: 0.25,
  },
};
