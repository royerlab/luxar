/**
 * Type definitions for the OPFS-based zarr cache system.
 */

/**
 * Statistics for segmented LRU cache.
 */
export interface CacheStats {
  /** Total bytes in metadata segment */
  metadataSize: number;
  /** Total bytes in chunks segment */
  chunksSize: number;
  /** Number of entries in metadata segment */
  metadataCount: number;
  /** Number of entries in chunks segment */
  chunksCount: number;
  /** Total cache hits across all segments */
  hits: number;
  /** Total cache misses across all segments */
  misses: number;
  /** Total evictions across all segments */
  evictions: number;
}

/**
 * How the cached dataset is validated against the remote source.
 *
 * - `content-hash`: dataset has Luxar's `content_hash` attr; mismatch
 *   triggers a full clear. Strongest guarantee.
 * - `ttl`: external dataset without `content_hash`; we trust the
 *   cache for `cache.externalDatasetTtlMs` and revalidate after.
 * - `none`: external dataset, no TTL configured — cache may be stale
 *   indefinitely until manually cleared. Surfaced in the UI as a
 *   warning badge so the user knows what they're getting.
 */
export type CacheValidationMode = 'content-hash' | 'ttl' | 'none';

/**
 * Metadata structure persisted to OPFS for L2 cache management.
 * Stored in _cache_meta.json within each dataset's OPFS directory.
 */
export interface OPFSMetadata {
  /** Original dataset URL (for listDatasets() debugging) */
  baseUrl: string;
  /** L2 index entries: [key, {size, order}] */
  entries: [string, { size: number; order: number }][];
  /** Total size of all cached entries in bytes */
  totalSize: number;
  /** LRU order counter (monotonically increasing) */
  orderCounter: number;
  /** Content hash of root .zattrs for cache invalidation */
  contentHash: string | null;
  /**
   * Filename-encoding version. Bumped when keyToFileName() output
   * changes so a loadMetadata() with a stale version invalidates the
   * directory rather than reading old-format files. Absent (undefined)
   * means version 1 (legacy `btoa(key)` Latin-1 only).
   */
  encodingVersion?: number;
  /**
   * Validation mode used at last init. Persisted so a session that
   * loaded with TTL semantics can re-evaluate the TTL window on the
   * next visit; persisted alongside `lastValidatedAt`.
   */
  validationMode?: CacheValidationMode;
  /** Wall-clock millis at last successful validation. */
  lastValidatedAt?: number;
}

/** Current OPFS filename-encoding version. Bumped only when keyToFileName changes. */
export const OPFS_ENCODING_VERSION = 2;

/**
 * Snapshot returned by `MultiLevelCachingStore.getStats()`. Aggregates
 * the L1 segmented-LRU stats, the L2 OPFS stats, network counters,
 * per-tier demand-hit counters, cache health (validation mode + OPFS
 * availability), and the `?clear-cache` invocation counter.
 *
 * Consumed by the data-loading monitor, the debug overlay, and the
 * cache E2E suite.
 */
export interface MultiLevelCacheStats {
  l1: CacheStats;
  l2: {
    size: number;
    count: number;
    reads: number;
    writes: number;
    misses: number;
    oversizedWriteSkipped?: number;
    quotaWriteSkipped?: number;
    evictions?: number;
    writeFailures?: number;
    corruptedEntries?: number;
    metadataParseFailures?: number;
    orphanedFilesRemoved?: number;
  };
  network: {
    /** Bytes fetched over the network (L3) — excludes cache-served bytes. */
    bytesTransferred: number;
    /** Count of actual network fetches (L3). */
    requestCount: number;
    /** Current network bandwidth (bytes/sec, ~10s sliding window). */
    bandwidth: number;
    /**
     * Cumulative bytes delivered to demand callers across ALL tiers
     * (L1 + L2 + network). Unlike `bytesTransferred`, this stays
     * non-zero on a warm/cache-served reload, so the monitor's
     * "data loaded" figure reflects real I/O even with zero network.
     */
    totalBytesServed: number;
    /** Count of demand reads served across all tiers. */
    totalRequestsServed: number;
  };
  /**
   * Per-tier demand-hit counters (user demand only — prefetch traffic
   * is excluded). Each user-demand `getResult` call increments exactly
   * one of `l1Hits`, `l2Hits`, or `networkRequests`. Combined with the
   * L0 provider's stats, this lets the monitor surface an effective
   * demand hit-rate rather than the L1-only ratio.
   */
  demand: { l1Hits: number; l2Hits: number; networkRequests: number };
  /**
   * Cache health snapshot. Surfaced by the data monitor status badges
   * and debug diagnostics.
   */
  health: {
    /** Validation mode the dataset is using (or 'none' if external + no TTL). */
    validationMode: CacheValidationMode;
    /** Wall-clock millis at last successful validation, or null. */
    lastValidatedAt: number | null;
    /**
     * `true` when the dataset has no `content_hash` AND no TTL is
     * configured — surfaced as a UI warning since the cache may be
     * stale indefinitely.
     */
    unvalidatedExternalDataset: boolean;
    /**
     * S2: `true` when OPFS is available and L2 is operational, or
     * when caching is disabled (no L2 expected). `false` only when
     * caching is enabled but OPFS could not be acquired — drives the
     * `opfs-unavailable` status badge.
     */
    opfsAvailable: boolean;
  };
  /**
   * S4: number of times `?clear-cache` triggered a clearAll on init
   * for this store. Increments at most once per store lifetime today
   * but typed as a counter so future re-init paths stay observable.
   */
  clearOnInitCount: number;
}
