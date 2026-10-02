import { LRUCache } from '../lru-cache';
import type { CacheStats } from '../types';

/**
 * Segmented LRU with name-based routing: Metadata files go to a dedicated segment,
 * data chunks go to the chunks segment. No frequency tracking needed.
 */
export class SegmentedLRUCache {
  private static readonly MIN_METADATA_SIZE = 10 * 1024 * 1024; // 10MB floor
  /**
   * The floor gives way on a small budget: never more than this share of the
   * total. A heap- or `?cacheBudgetMB`-scaled L1 can be far below 10 MB (40 MB
   * total gives ~8 MB), and a fixed floor there left the chunk segment at zero —
   * every L1 chunk rejected — with the segments together over the budget.
   */
  private static readonly MAX_METADATA_FLOOR_SHARE = 0.5;

  // Metadata file patterns - these go to metadata segment
  private static readonly METADATA_PATTERNS = [
    '.zmetadata', // Consolidated metadata (zarr v2)
    '.zarray', // Array metadata
    '.zattrs', // Attribute metadata
    'zarr.json', // zarr v3 metadata
  ];

  // Metadata segment: zarr metadata files (20% of cache, min 10MB or half the
  // cache, whichever is smaller)
  private metadata: LRUCache<Uint8Array>;

  // Chunks segment: data chunks (80% of cache)
  private chunks: LRUCache<Uint8Array>;

  /** Configured total byte budget (metadata + chunks segments). */
  private readonly totalSize: number;

  constructor(totalSize: number) {
    this.totalSize = totalSize;
    const getSize = (v: Uint8Array) => v.byteLength;
    const floor = Math.min(
      SegmentedLRUCache.MIN_METADATA_SIZE,
      totalSize * SegmentedLRUCache.MAX_METADATA_FLOOR_SHARE
    );
    const metadataSize = Math.max(totalSize * 0.2, floor);
    // Guard: a non-positive totalSize must not give a negative segment.
    const chunksSize = Math.max(0, totalSize - metadataSize);

    this.metadata = new LRUCache(metadataSize, getSize);
    this.chunks = new LRUCache(chunksSize, getSize);
  }

  /**
   * Check if a key is a metadata file based on name pattern.
   */
  private static isMetadataFile(key: string): boolean {
    return SegmentedLRUCache.METADATA_PATTERNS.some(
      (pattern) => key.endsWith(pattern) || key === pattern
    );
  }

  get(key: string): Uint8Array | undefined {
    // Route to correct segment to avoid double-counting misses.
    // Previously, checking metadata first then chunks would record a spurious
    // miss on the metadata segment for every chunk lookup, inflating miss stats.
    if (SegmentedLRUCache.isMetadataFile(key)) {
      return this.metadata.get(key);
    }
    return this.chunks.get(key);
  }

  set(key: string, data: Uint8Array): void {
    // Route by name pattern: metadata files → metadata segment, else → chunks segment
    if (SegmentedLRUCache.isMetadataFile(key)) {
      this.metadata.set(key, data);
    } else {
      this.chunks.set(key, data);
    }
  }

  has(key: string): boolean {
    if (SegmentedLRUCache.isMetadataFile(key)) {
      return this.metadata.has(key);
    }
    return this.chunks.has(key);
  }

  clear(): void {
    this.metadata.clear();
    this.chunks.clear();
  }

  getStats(): CacheStats {
    return {
      metadataSize: this.metadata.size,
      chunksSize: this.chunks.size,
      metadataCount: this.metadata.count,
      chunksCount: this.chunks.count,
      hits: this.metadata.hitCount + this.chunks.hitCount,
      misses: this.metadata.missCount + this.chunks.missCount,
      evictions: this.metadata.evictionCount + this.chunks.evictionCount,
      maxSize: this.totalSize,
    };
  }
}
