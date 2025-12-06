import { LRUCache } from './lru-cache';
import type { CacheStats } from './types';

/**
 * Segmented LRU with name-based routing: Metadata files go to a dedicated segment,
 * data chunks go to the chunks segment. No frequency tracking needed.
 */
export class SegmentedLRUCache {
  private static readonly MIN_METADATA_SIZE = 10 * 1024 * 1024; // 10MB floor

  // Metadata file patterns - these go to metadata segment
  private static readonly METADATA_PATTERNS = [
    '.zmetadata', // Consolidated metadata (zarr v2)
    '.zarray', // Array metadata
    '.zattrs', // Attribute metadata
    'zarr.json', // zarr v3 metadata
  ];

  // Metadata segment: zarr metadata files (20% of cache, min 10MB)
  private metadata: LRUCache<Uint8Array>;

  // Chunks segment: data chunks (80% of cache)
  private chunks: LRUCache<Uint8Array>;

  constructor(totalSize: number) {
    const getSize = (v: Uint8Array) => v.byteLength;
    const metadataSize = Math.max(totalSize * 0.2, SegmentedLRUCache.MIN_METADATA_SIZE);
    const chunksSize = totalSize - metadataSize;

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
    // Check metadata segment first
    const metadataHit = this.metadata.get(key);
    if (metadataHit) {
      return metadataHit;
    }

    // Check chunks segment
    const chunksHit = this.chunks.get(key);
    if (chunksHit) {
      return chunksHit;
    }

    return undefined;
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
    return this.metadata.has(key) || this.chunks.has(key);
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
    };
  }
}
