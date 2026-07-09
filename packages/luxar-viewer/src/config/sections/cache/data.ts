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
  externalDatasetTtlMs: null,
  debug: false,
};
