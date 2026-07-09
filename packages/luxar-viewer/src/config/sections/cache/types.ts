/**
 * OPFS-based zarr cache configuration
 */
export interface CacheConfig {
  /** Enable OPFS caching (default: true) */
  enabled: boolean;
  /** Enable L0 decompressed chunk cache (default: true) */
  l0Enabled: boolean;
  /** L0 decompressed chunk cache size in MB (default: 200) */
  l0MaxSizeMB: number;
  /**
   * Enable the SliceCache ("S-cache") — a per-(node, view) LRU of the decoded
   * per-slice geometry that lets revisiting a slice (e.g. scrubbing back to a
   * timepoint) skip the load+decode entirely (default: true).
   */
  sliceCacheEnabled: boolean;
  /** SliceCache size in MB (default: 128). Byte-budget LRU shared across nodes. */
  sliceCacheMaxSizeMB: number;
  /** L1 memory cache size in MB (default: 100) */
  l1MaxSizeMB: number;
  /** L2 OPFS cache size in MB (default: 2048) */
  l2MaxSizeMB: number;
  /**
   * Per-operation timeout for OPFS file I/O (read/write/delete) in ms.
   * A hung browser OPFS handle would otherwise stall cache operations
   * indefinitely; this bound degrades them to a cache miss / write skip.
   * Default: 10_000 ms.
   */
  opfsOperationTimeoutMs: number;
  /**
   * TTL in ms for external datasets that lack Luxar's `content_hash`.
   * When set, a cached external dataset older than this is invalidated
   * on next init. `null` (default) means no TTL — the cache may be
   * stale indefinitely until manually cleared.
   */
  externalDatasetTtlMs: number | null;
  /** Enable cache debug logging (default: false) */
  debug: boolean;
}
