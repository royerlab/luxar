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
