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
  // Pending L2 writes cap; oldest dropped past this (best-effort tier). A queued
  // task holds only a closure over an L1-resident buffer, so depth costs task
  // objects, not bytes. 1 024 dropped ~9 800 of ~11 000 chunk writes on a
  // 14.8 M-splat load (2026-09 audit, finding 8), so the warm second visit
  // re-fetched almost everything.
  opfsWriteQueueMax: 16_384,
  externalDatasetTtlMs: null,
  debug: false,
};
