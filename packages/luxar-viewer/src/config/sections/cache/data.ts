import type { CacheConfig } from './types';

/**
 * OPFS-based zarr cache configuration
 */
export const cacheConfig: CacheConfig = {
  enabled: true,
  l0Enabled: true, // L0 decompressed chunk cache - eliminates ~2ms Blosc decompression per chunk
  l0MaxSizeMB: 200, // 200MB for decompressed chunks (5x larger than compressed, but instant access)
  sliceCacheEnabled: true, // S-cache: per-(node,view) decoded-slice cache — instant slice revisits
  sliceCacheMaxSizeMB: 128, // 128MB for decoded per-slice geometry (byte-LRU, shared across nodes)
  l1MaxSizeMB: 100,
  l2MaxSizeMB: 2048,
  opfsOperationTimeoutMs: 10_000,
  opfsTimeoutTripThreshold: 3, // consecutive timeouts before the L2 circuit breaker trips (sticky)
  opfsWriteConcurrency: 4, // background L2 writes run at most 4-wide (bounds OPFS contention)
  // Pending L2 write depth cap; oldest dropped past this (best-effort tier).
  // Retained chunk bytes are separately capped at max(resolved L1, 64MB), so
  // depth binds only below roughly 4-6KB per pending chunk with these defaults.
  opfsWriteQueueMax: 16_384,
  externalDatasetTtlMs: null,
  debug: false,
};
