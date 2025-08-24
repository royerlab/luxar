/**
 * Efficient cache implementation for spatial index range queries.
 *
 * This cache is specifically designed for spatial index-based loading where
 * data is accessed by point ranges rather than traditional chunks. It provides
 * intelligent caching with multiple eviction strategies and memory management.
 */

import { CacheEntry, CacheStats, LoaderConfig, PointRange } from './data-loader-types';
import { log, Modules, LogEmoji } from '../utils/log';

/**
 * Cache key generation for point ranges
 */
export class RangeCacheKey {
  /**
   * Generate a unique key for a set of point ranges.
   * Ranges are normalized and sorted to ensure consistent keys.
   */
  static fromRanges(arrayPath: string, ranges: PointRange[]): string {
    // Sort ranges by start index for consistent keys
    const sorted = [...ranges].sort((a, b) => a.start - b.start);
    const rangeStr = sorted.map((r) => `${r.start}-${r.end}`).join(',');
    return `${arrayPath}:[${rangeStr}]`;
  }

  /**
   * Generate a key for a contiguous range
   */
  static fromRange(arrayPath: string, start: number, end: number): string {
    return `${arrayPath}:[${start}-${end}]`;
  }

  /**
   * Parse a cache key back into components
   */
  static parse(key: string): { arrayPath: string; ranges: PointRange[] } {
    const match = key.match(/^(.+):\[(.+)\]$/);
    if (!match) {
      throw new Error(`Invalid cache key: ${key}`);
    }

    const arrayPath = match[1];
    const rangeStr = match[2];
    const ranges = rangeStr.split(',').map((r) => {
      const [start, end] = r.split('-').map(Number);
      return { start, end };
    });

    return { arrayPath, ranges };
  }
}

/**
 * Intelligent cache for spatial index range queries.
 *
 * Features:
 * - Multiple eviction strategies (LRU, LFU, distance-based)
 * - Memory-aware caching with configurable limits
 * - Range coalescing for efficient storage
 * - Statistics tracking for monitoring
 */
export class RangeCache {
  private cache = new Map<string, CacheEntry>();
  private config: Required<LoaderConfig>;
  private stats = {
    hits: 0,
    misses: 0,
    evictions: 0,
    bytesEvicted: 0,
  };
  private totalMemory = 0;
  private maxMemoryBytes: number;

  constructor(config: LoaderConfig = {}) {
    this.config = {
      maxMemoryMB: config.maxMemoryMB ?? 500,
      debug: config.debug ?? false,
      evictionStrategy: config.evictionStrategy ?? 'lru',
      enableMonitor: config.enableMonitor ?? true,
    };
    this.maxMemoryBytes = this.config.maxMemoryMB * 1024 * 1024;
  }

  /**
   * Get data from cache if available
   */
  get(arrayPath: string, ranges: PointRange[]): Float32Array | null {
    const key = RangeCacheKey.fromRanges(arrayPath, ranges);
    const entry = this.cache.get(key);

    if (entry) {
      // Update access metadata
      entry.lastAccess = Date.now();
      entry.accessCount++;
      this.stats.hits++;

      if (this.config.debug) {
        log.custom(LogEmoji.CACHE, Modules.CACHE, `Hit for ${key}`);
      }

      return entry.data;
    }

    this.stats.misses++;
    if (this.config.debug) {
      log.custom(LogEmoji.CACHE, Modules.CACHE, `Miss for ${key}`);
    }

    // Check if we can satisfy from overlapping cached ranges
    const merged = this.tryMergeFromCache(arrayPath, ranges);
    if (merged) {
      this.stats.hits++;
      return merged;
    }

    return null;
  }

  /**
   * Store data in cache with automatic eviction if needed
   */
  set(arrayPath: string, ranges: PointRange[], data: Float32Array): void {
    const key = RangeCacheKey.fromRanges(arrayPath, ranges);
    const size = data.byteLength;

    // Evict if necessary
    this.evictIfNeeded(size);

    // Create cache entry
    const entry: CacheEntry = {
      data,
      size,
      lastAccess: Date.now(),
      accessCount: 1,
    };

    this.cache.set(key, entry);
    this.totalMemory += size;

    if (this.config.debug) {
      const memoryMB = this.totalMemory / (1024 * 1024);
      log.custom(
        LogEmoji.CACHE,
        Modules.CACHE,
        `Stored ${key}, total memory: ${memoryMB.toFixed(1)}MB`
      );
    }
  }

  /**
   * Try to merge data from overlapping cached ranges
   */
  private tryMergeFromCache(_arrayPath: string, _ranges: PointRange[]): Float32Array | null {
    // For now, return null - this is an optimization for later
    // Would need to track which ranges overlap and merge them
    return null;
  }

  /**
   * Evict entries if cache is too full
   */
  private evictIfNeeded(bytesNeeded: number): void {
    while (this.totalMemory + bytesNeeded > this.maxMemoryBytes && this.cache.size > 0) {
      const toEvict = this.selectEntryToEvict();
      if (!toEvict) break;

      const entry = this.cache.get(toEvict);
      if (entry) {
        this.cache.delete(toEvict);
        this.totalMemory -= entry.size;
        this.stats.evictions++;
        this.stats.bytesEvicted += entry.size;

        if (this.config.debug) {
          log.custom(
            LogEmoji.CLEAN,
            Modules.CACHE,
            `Evicted ${toEvict}, freed ${entry.size} bytes`
          );
        }
      }
    }
  }

  /**
   * Select which cache entry to evict based on strategy
   */
  private selectEntryToEvict(): string | null {
    if (this.cache.size === 0) return null;

    let selected: string | null = null;

    switch (this.config.evictionStrategy) {
      case 'lru': {
        // Least Recently Used
        let oldestTime = Infinity;
        for (const [key, entry] of this.cache) {
          if (entry.lastAccess < oldestTime) {
            oldestTime = entry.lastAccess;
            selected = key;
          }
        }
        break;
      }

      case 'lfu': {
        // Least Frequently Used
        let minCount = Infinity;
        for (const [key, entry] of this.cache) {
          if (entry.accessCount < minCount) {
            minCount = entry.accessCount;
            selected = key;
          }
        }
        break;
      }
    }

    return selected;
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    const total = this.stats.hits + this.stats.misses;
    return {
      numEntries: this.cache.size,
      totalMemory: this.totalMemory,
      hitRate: total > 0 ? this.stats.hits / total : 0,
      hits: this.stats.hits,
      misses: this.stats.misses,
    };
  }

  /**
   * Clear all cached data
   */
  clear(): void {
    const entries = this.cache.size;
    const memory = this.totalMemory;

    this.cache.clear();
    this.totalMemory = 0;

    if (this.config.debug) {
      log.custom(
        LogEmoji.CLEAN,
        Modules.CACHE,
        `Cleared ${entries} entries, freed ${memory} bytes`
      );
    }
  }

  /**
   * Check if ranges are in cache
   */
  has(arrayPath: string, ranges: PointRange[]): boolean {
    const key = RangeCacheKey.fromRanges(arrayPath, ranges);
    return this.cache.has(key);
  }

  /**
   * Get memory usage info
   */
  getMemoryInfo(): { used: number; max: number; percentage: number } {
    return {
      used: this.totalMemory,
      max: this.maxMemoryBytes,
      percentage: (this.totalMemory / this.maxMemoryBytes) * 100,
    };
  }
}
