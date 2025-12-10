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
 * Extended statistics for the full two-level cache system.
 * Used by CacheStatsProvider for DataLoadingMonitor integration.
 */
export interface ExtendedCacheStats {
  /** L1 memory cache statistics */
  l1: CacheStats;
  /** L2 OPFS persistent cache statistics */
  l2: {
    /** Total bytes in L2 cache */
    size: number;
    /** Number of entries in L2 cache */
    count: number;
    /** Total reads from L2 */
    reads: number;
    /** Total writes to L2 */
    writes: number;
  };
  /** Whether caching is enabled */
  enabled: boolean;
}

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
}
