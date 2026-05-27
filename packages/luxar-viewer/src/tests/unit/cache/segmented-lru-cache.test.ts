import { describe, it, expect, beforeEach } from 'vitest';
import { SegmentedLRUCache } from '../../../cache/multi-level-caching-store/segmented-lru-cache';

describe('SegmentedLRUCache', () => {
  let cache: SegmentedLRUCache;

  beforeEach(() => {
    cache = new SegmentedLRUCache(100 * 1024 * 1024); // 100MB
  });

  describe('Metadata File Detection', () => {
    it('should recognize .zmetadata as metadata', () => {
      cache.set('.zmetadata', new Uint8Array(1000));
      const stats = cache.getStats();
      expect(stats.metadataCount).toBe(1);
      expect(stats.chunksCount).toBe(0);
    });

    it('should recognize .zarray files as metadata', () => {
      cache.set('positions/.zarray', new Uint8Array(500));
      cache.set('colors/.zarray', new Uint8Array(500));
      const stats = cache.getStats();
      expect(stats.metadataCount).toBe(2);
      expect(stats.chunksCount).toBe(0);
    });

    it('should recognize .zattrs files as metadata', () => {
      cache.set('.zattrs', new Uint8Array(200));
      cache.set('points/.zattrs', new Uint8Array(200));
      const stats = cache.getStats();
      expect(stats.metadataCount).toBe(2);
    });

    it('should recognize zarr.json as metadata', () => {
      cache.set('zarr.json', new Uint8Array(1000));
      const stats = cache.getStats();
      expect(stats.metadataCount).toBe(1);
    });

    it('should recognize metadata patterns at any path depth', () => {
      cache.set('a/b/c/.zarray', new Uint8Array(100));
      cache.set('deep/nested/path/.zattrs', new Uint8Array(100));
      const stats = cache.getStats();
      expect(stats.metadataCount).toBe(2);
      expect(stats.chunksCount).toBe(0);
    });

    it('should route data chunks to chunks segment', () => {
      cache.set('positions/0.0.0', new Uint8Array(32768));
      cache.set('colors/1.2.3', new Uint8Array(16384));
      const stats = cache.getStats();
      expect(stats.metadataCount).toBe(0);
      expect(stats.chunksCount).toBe(2);
    });

    it('should handle mixed metadata and chunks', () => {
      cache.set('.zmetadata', new Uint8Array(2048));
      cache.set('positions/.zarray', new Uint8Array(500));
      cache.set('positions/0.0.0', new Uint8Array(32768));
      cache.set('positions/0.0.1', new Uint8Array(32768));

      const stats = cache.getStats();
      expect(stats.metadataCount).toBe(2);
      expect(stats.chunksCount).toBe(2);
    });
  });

  describe('Segment Size Calculation', () => {
    it('should route metadata to separate segment', () => {
      // Simple test: verify metadata goes to metadata segment
      cache.set('.zmetadata', new Uint8Array(1000));
      cache.set('.zarray', new Uint8Array(500));

      const stats = cache.getStats();
      expect(stats.metadataCount).toBe(2);
      expect(stats.metadataSize).toBe(1500);
      expect(stats.chunksCount).toBe(0);
      expect(stats.chunksSize).toBe(0);
    });

    it('should route chunks to separate segment', () => {
      cache.set('positions/0.0.0', new Uint8Array(32768));
      cache.set('colors/0.0.0', new Uint8Array(16384));

      const stats = cache.getStats();
      expect(stats.chunksCount).toBe(2);
      expect(stats.chunksSize).toBe(49152);
      expect(stats.metadataCount).toBe(0);
      expect(stats.metadataSize).toBe(0);
    });

    it('should maintain separate accounting', () => {
      cache.set('.zmetadata', new Uint8Array(2048));
      cache.set('chunk', new Uint8Array(32768));
      cache.set('.zattrs', new Uint8Array(512));

      const stats = cache.getStats();
      expect(stats.metadataSize).toBe(2560); // 2048 + 512
      expect(stats.chunksSize).toBe(32768);
      expect(stats.metadataCount).toBe(2);
      expect(stats.chunksCount).toBe(1);
    });
  });

  describe('Segment Isolation', () => {
    it('should not evict metadata when chunks segment is full', () => {
      // Use 50MB cache: metadata=10MB (min), chunks=40MB
      const cache = new SegmentedLRUCache(50 * 1024 * 1024);

      // Add metadata (will be protected)
      cache.set('.zmetadata', new Uint8Array(1024 * 1024)); // 1MB

      // Fill chunks segment with 40MB
      cache.set('chunk1', new Uint8Array(20 * 1024 * 1024)); // 20MB
      cache.set('chunk2', new Uint8Array(20 * 1024 * 1024)); // 20MB

      // Add another 10MB chunk - should evict chunk1, not metadata
      cache.set('chunk3', new Uint8Array(25 * 1024 * 1024)); // 25MB

      expect(cache.has('.zmetadata')).toBe(true); // Metadata protected
      expect(cache.has('chunk1')).toBe(false); // Evicted
      expect(cache.has('chunk3')).toBe(true);
    });

    it('should evict within metadata segment when metadata full', () => {
      // Use 50MB cache: metadata=10MB (min), chunks=40MB
      const cache = new SegmentedLRUCache(50 * 1024 * 1024);

      cache.set('.zmetadata', new Uint8Array(5 * 1024 * 1024)); // 5MB
      cache.set('.zarray', new Uint8Array(5 * 1024 * 1024)); // 10MB total (at limit)
      cache.set('.zattrs', new Uint8Array(6 * 1024 * 1024)); // Exceeds 10MB!

      // Both .zmetadata AND .zarray should be evicted to make room
      // 10 + 6 = 16, need to evict 6+MB
      // Evict .zmetadata (5MB) → 5MB remains, still need 1MB more
      // Evict .zarray (5MB) → 0MB, now can add .zattrs (6MB)
      expect(cache.has('.zmetadata')).toBe(false); // Evicted first
      expect(cache.has('.zarray')).toBe(false); // Evicted second
      expect(cache.has('.zattrs')).toBe(true); // Only one remaining
    });
  });

  describe('Basic Operations', () => {
    it('should retrieve from both segments', () => {
      cache.set('.zmetadata', new Uint8Array([1, 2, 3]));
      cache.set('chunk', new Uint8Array([4, 5, 6]));

      expect(cache.get('.zmetadata')).toEqual(new Uint8Array([1, 2, 3]));
      expect(cache.get('chunk')).toEqual(new Uint8Array([4, 5, 6]));
    });

    it('should check existence in both segments', () => {
      cache.set('.zarray', new Uint8Array(10));
      cache.set('data', new Uint8Array(10));

      expect(cache.has('.zarray')).toBe(true);
      expect(cache.has('data')).toBe(true);
      expect(cache.has('nonexistent')).toBe(false);
    });

    it('should clear both segments', () => {
      cache.set('.zmetadata', new Uint8Array(1000));
      cache.set('chunk1', new Uint8Array(1000));
      cache.set('chunk2', new Uint8Array(1000));

      cache.clear();

      const stats = cache.getStats();
      expect(stats.metadataSize).toBe(0);
      expect(stats.chunksSize).toBe(0);
      expect(stats.metadataCount).toBe(0);
      expect(stats.chunksCount).toBe(0);
    });
  });

  describe('Statistics', () => {
    it('should track sizes accurately across segments', () => {
      cache.set('.zmetadata', new Uint8Array(2048));
      cache.set('.zarray', new Uint8Array(512));
      cache.set('chunk1', new Uint8Array(32768));
      cache.set('chunk2', new Uint8Array(16384));

      const stats = cache.getStats();
      expect(stats.metadataSize).toBe(2560); // 2048 + 512
      expect(stats.chunksSize).toBe(49152); // 32768 + 16384
      expect(stats.metadataCount).toBe(2);
      expect(stats.chunksCount).toBe(2);
    });

    it('should update stats after deletions via eviction', () => {
      const small = new SegmentedLRUCache(50 * 1024 * 1024); // 50MB
      small.set('.zmetadata', new Uint8Array(1024 * 1024)); // 1MB
      small.set('chunk1', new Uint8Array(20 * 1024 * 1024)); // 20MB
      small.set('chunk2', new Uint8Array(20 * 1024 * 1024)); // 20MB

      // This should evict chunk1 and chunk2
      small.set('chunk3', new Uint8Array(35 * 1024 * 1024)); // 35MB

      const stats = small.getStats();
      expect(stats.chunksCount).toBe(1); // Only chunk3
      expect(stats.metadataCount).toBe(1); // .zmetadata protected
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty cache', () => {
      const stats = cache.getStats();
      expect(stats.metadataSize).toBe(0);
      expect(stats.chunksSize).toBe(0);
      expect(cache.get('.zmetadata')).toBeUndefined();
      expect(cache.get('chunk')).toBeUndefined();
    });

    it('should handle keys that partially match patterns', () => {
      // These should NOT be considered metadata
      cache.set('my.zmetadata.backup', new Uint8Array(10));
      cache.set('zarray', new Uint8Array(10)); // Missing dot
      cache.set('zattrs.old', new Uint8Array(10)); // Wrong position

      const stats = cache.getStats();
      expect(stats.chunksCount).toBe(3); // All routed to chunks
      expect(stats.metadataCount).toBe(0);
    });

    it('should handle exact pattern matches', () => {
      cache.set('.zmetadata', new Uint8Array(10));
      cache.set('.zarray', new Uint8Array(10));
      cache.set('.zattrs', new Uint8Array(10));
      cache.set('zarr.json', new Uint8Array(10));

      const stats = cache.getStats();
      expect(stats.metadataCount).toBe(4); // All exact matches
      expect(stats.chunksCount).toBe(0);
    });

    it('should handle concurrent access to both segments', () => {
      // Interleave metadata and chunk operations
      cache.set('.zmetadata', new Uint8Array(1000));
      cache.set('chunk1', new Uint8Array(1000));
      expect(cache.get('.zmetadata')).toBeDefined();
      cache.set('.zarray', new Uint8Array(500));
      expect(cache.get('chunk1')).toBeDefined();
      cache.set('chunk2', new Uint8Array(1000));

      const stats = cache.getStats();
      expect(stats.metadataCount).toBe(2);
      expect(stats.chunksCount).toBe(2);
      expect(stats.metadataSize).toBe(1500);
      expect(stats.chunksSize).toBe(2000);
    });

    // workers.md O3 / cache.md G6 [P5]: audit-id moved to comment per Phase E54.
    it('clamps chunksSize to 0 when totalSize < MIN_METADATA_SIZE (10MB)', () => {
      // [cache.md/G6][P5] Pre-audit boundary: a regression that flipped the
      // `Math.max(0, ...)` in the source to allow negative chunksSize would
      // corrupt eviction but pass current tests. Pin: with totalSize=1MB
      // (well below the 10MB metadata floor), the chunks segment holds
      // nothing (every chunk write must immediately be rejected/evicted).
      const tiny = new SegmentedLRUCache(1024 * 1024);

      // A 100-byte chunk goes to the chunks segment. With chunksSize===0
      // budget, the entry must immediately be rejected (oversized) or
      // evicted; the chunks-count and chunks-size remain at 0.
      tiny.set('chunk', new Uint8Array(100));
      const stats = tiny.getStats();
      expect(stats.chunksSize).toBe(0);
      expect(stats.chunksCount).toBe(0);

      // Metadata, on the other hand, fits in the 10MB floor.
      tiny.set('.zmetadata', new Uint8Array(1000));
      const stats2 = tiny.getStats();
      expect(stats2.metadataCount).toBe(1);
      expect(stats2.metadataSize).toBe(1000);
    });
  });

  describe('Real-World Zarr Patterns', () => {
    it('should handle typical zarr dataset structure', () => {
      // Metadata files
      cache.set('.zmetadata', new Uint8Array(5000)); // Consolidated metadata
      cache.set('points/.zattrs', new Uint8Array(200));
      cache.set('points/positions/.zarray', new Uint8Array(300));
      cache.set('points/positions/.zattrs', new Uint8Array(100));
      cache.set('points/colors/.zarray', new Uint8Array(300));

      // Data chunks
      cache.set('points/positions/0.0.0', new Uint8Array(32768));
      cache.set('points/positions/0.0.1', new Uint8Array(32768));
      cache.set('points/colors/0.0.0', new Uint8Array(16384));

      const stats = cache.getStats();
      expect(stats.metadataCount).toBe(5);
      expect(stats.chunksCount).toBe(3);
      expect(stats.metadataSize).toBe(5900);
      expect(stats.chunksSize).toBe(81920);
    });
  });

  describe('Hit/Miss Counter Aggregation', () => {
    it('should aggregate hits from both segments', () => {
      cache.set('.zmetadata', new Uint8Array(100));
      cache.set('chunk', new Uint8Array(100));

      // Hit metadata
      cache.get('.zmetadata');
      // Hit chunk
      cache.get('chunk');
      cache.get('chunk');

      const stats = cache.getStats();
      expect(stats.hits).toBe(3); // 1 metadata + 2 chunks
    });

    it('should aggregate misses from both segments', () => {
      cache.set('.zmetadata', new Uint8Array(100));
      cache.set('chunk', new Uint8Array(100));

      // Miss in chunks segment (routed by key pattern)
      cache.get('missing_chunk'); // 1 miss in chunks segment
      cache.get('another_missing'); // 1 miss in chunks segment
      // Miss in metadata segment
      cache.get('.zattrs'); // 1 miss in metadata segment (not set)

      const stats = cache.getStats();
      // Each get() is routed to exactly one segment, so one miss per call
      expect(stats.misses).toBe(3);
    });

    it('should track hits and misses accurately for metadata files (exact counts)', () => {
      // cache.md W22 fix: previous version asserted `misses >= 1` which
      // allowed silent miss-count drift. Each get() routes to exactly one
      // segment, so 2 hits + 1 miss → exactly 2 hits and exactly 1 miss.
      cache.set('.zmetadata', new Uint8Array(100));
      cache.set('.zarray', new Uint8Array(100));

      cache.get('.zmetadata'); // Hit
      cache.get('.zarray'); // Hit
      cache.get('.zattrs'); // Miss (not set)

      const stats = cache.getStats();
      expect(stats.hits).toBe(2);
      expect(stats.misses).toBe(1);
    });

    it('should reset hit/miss counters on clear()', () => {
      cache.set('.zmetadata', new Uint8Array(100));
      cache.set('chunk', new Uint8Array(100));

      cache.get('.zmetadata'); // Hit
      cache.get('missing'); // Miss

      let stats = cache.getStats();
      expect(stats.hits).toBeGreaterThan(0);

      cache.clear();

      stats = cache.getStats();
      expect(stats.hits).toBe(0);
      expect(stats.misses).toBe(0);
    });

    it('counts a single metadata get() as exactly 1 hit + 0 misses (no cross-segment double-count)', () => {
      // cache.md W23 fix: renamed from 'should track hits for promoted
      // items from chunks to metadata lookup' which was misleading — the
      // body tests a single set+get of metadata, not any "promotion".
      cache.set('.zmetadata', new Uint8Array(100));
      cache.get('.zmetadata');

      const stats = cache.getStats();
      expect(stats.hits).toBe(1);
      expect(stats.misses).toBe(0);
    });
  });

  describe('Eviction Counter Aggregation', () => {
    it('should aggregate evictions from both segments', () => {
      // Use 50MB cache: metadata=10MB (min), chunks=40MB
      const smallCache = new SegmentedLRUCache(50 * 1024 * 1024);

      // Fill metadata segment (10MB)
      smallCache.set('.zmetadata', new Uint8Array(5 * 1024 * 1024)); // 5MB
      smallCache.set('.zarray', new Uint8Array(5 * 1024 * 1024)); // 10MB total

      // Add more metadata to trigger eviction
      smallCache.set('.zattrs', new Uint8Array(6 * 1024 * 1024)); // 16MB > 10MB, evicts 2

      let stats = smallCache.getStats();
      expect(stats.evictions).toBeGreaterThan(0); // Metadata evictions

      // Fill chunks segment
      smallCache.set('chunk1', new Uint8Array(20 * 1024 * 1024)); // 20MB
      smallCache.set('chunk2', new Uint8Array(20 * 1024 * 1024)); // 40MB total
      smallCache.set('chunk3', new Uint8Array(25 * 1024 * 1024)); // 65MB > 40MB, evicts chunk1

      stats = smallCache.getStats();
      // Should have evictions from both segments
      expect(stats.evictions).toBeGreaterThan(1);
    });

    it('should reset eviction counter on clear()', () => {
      // Use 50MB cache
      const smallCache = new SegmentedLRUCache(50 * 1024 * 1024);

      // Trigger some evictions
      smallCache.set('.zmetadata', new Uint8Array(10 * 1024 * 1024)); // 10MB (fills metadata)
      smallCache.set('.zattrs', new Uint8Array(5 * 1024 * 1024)); // Evicts from metadata

      let stats = smallCache.getStats();
      expect(stats.evictions).toBeGreaterThan(0);

      smallCache.clear();

      stats = smallCache.getStats();
      expect(stats.evictions).toBe(0);
    });

    it('should track evictions independently in each segment', () => {
      // Use 50MB cache: metadata=10MB, chunks=40MB
      const smallCache = new SegmentedLRUCache(50 * 1024 * 1024);

      // Fill and evict from metadata only
      smallCache.set('.zmetadata', new Uint8Array(10 * 1024 * 1024)); // Full
      smallCache.set('.zarray', new Uint8Array(5 * 1024 * 1024)); // Evicts .zmetadata

      let stats = smallCache.getStats();
      const metadataEvictions = stats.evictions;
      expect(metadataEvictions).toBeGreaterThan(0);

      // Fill chunks without eviction
      smallCache.set('chunk1', new Uint8Array(10 * 1024 * 1024));

      stats = smallCache.getStats();
      expect(stats.evictions).toBe(metadataEvictions); // No new evictions

      // Now trigger chunk eviction
      smallCache.set('chunk2', new Uint8Array(35 * 1024 * 1024)); // Needs to evict chunk1

      stats = smallCache.getStats();
      expect(stats.evictions).toBeGreaterThan(metadataEvictions); // Added chunk eviction
    });
  });
});
