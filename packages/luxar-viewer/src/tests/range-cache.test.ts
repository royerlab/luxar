/**
 * Tests for the new data loading architecture.
 *
 * These tests validate that the new loaders correctly:
 * - Load data with spatial indices
 * - Align attributes properly
 * - Handle caching efficiently
 * - Require spatial indices for all data
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { RangeCache, RangeCacheKey, type PointRange } from '../data';

describe('RangeCache', () => {
  let cache: RangeCache;

  beforeEach(() => {
    cache = new RangeCache({ maxMemoryMB: 10 });
  });

  afterEach(() => {
    cache.clear();
  });

  describe('Cache Key Generation', () => {
    it('should generate consistent keys for same ranges', () => {
      const ranges1: PointRange[] = [
        { start: 0, end: 100 },
        { start: 200, end: 300 },
      ];
      const ranges2: PointRange[] = [
        { start: 0, end: 100 },
        { start: 200, end: 300 },
      ];

      const key1 = RangeCacheKey.fromRanges('positions', ranges1);
      const key2 = RangeCacheKey.fromRanges('positions', ranges2);

      expect(key1).toBe(key2);
    });

    it('should generate different keys for different ranges', () => {
      const ranges1: PointRange[] = [{ start: 0, end: 100 }];
      const ranges2: PointRange[] = [{ start: 100, end: 200 }];

      const key1 = RangeCacheKey.fromRanges('positions', ranges1);
      const key2 = RangeCacheKey.fromRanges('positions', ranges2);

      expect(key1).not.toBe(key2);
    });

    it('should parse keys back to components', () => {
      const ranges: PointRange[] = [
        { start: 10, end: 20 },
        { start: 30, end: 40 },
      ];
      const key = RangeCacheKey.fromRanges('colors', ranges);
      const parsed = RangeCacheKey.parse(key);

      expect(parsed.arrayPath).toBe('colors');
      expect(parsed.ranges).toEqual(ranges);
    });
  });

  describe('Cache Operations', () => {
    it('should store and retrieve data', () => {
      const ranges: PointRange[] = [{ start: 0, end: 100 }];
      const data = new Float32Array([1, 2, 3, 4, 5]);

      cache.set('positions', ranges, data);
      const retrieved = cache.get('positions', ranges);

      expect(retrieved).toEqual(data);
    });

    it('should return null for missing data', () => {
      const ranges: PointRange[] = [{ start: 0, end: 100 }];
      const retrieved = cache.get('positions', ranges);

      expect(retrieved).toBeNull();
    });

    it('should track cache statistics', () => {
      const ranges: PointRange[] = [{ start: 0, end: 100 }];
      const data = new Float32Array([1, 2, 3]);

      // Miss
      cache.get('positions', ranges);

      // Store
      cache.set('positions', ranges, data);

      // Hit
      cache.get('positions', ranges);

      const stats = cache.getStats();
      expect(stats.hits).toBe(1);
      expect(stats.misses).toBe(1);
      expect(stats.hitRate).toBe(0.5);
    });

    it('should track average access time', () => {
      const cache = new RangeCache({ maxMemoryMB: 10 });
      const data = new Float32Array(100).fill(1);

      // Store data first
      cache.set('positions', [{ start: 0, end: 100 }], data);

      // Access the data multiple times
      cache.get('positions', [{ start: 0, end: 100 }]); // hit
      cache.get('positions', [{ start: 100, end: 200 }]); // miss
      cache.get('positions', [{ start: 0, end: 100 }]); // hit

      const stats = cache.getStats();

      // Should have avgAccessTime property
      expect(stats.avgAccessTime).toBeDefined();
      expect(typeof stats.avgAccessTime).toBe('number');

      // Access time should be positive but very small (< 10ms for in-memory operations)
      expect(stats.avgAccessTime).toBeGreaterThanOrEqual(0);
      expect(stats.avgAccessTime).toBeLessThan(10); // Should be less than 10ms for memory access

      // Verify we tracked 3 accesses
      expect(stats.hits).toBe(2);
      expect(stats.misses).toBe(1);
    });

    it('should reset access time stats on clear', () => {
      const cache = new RangeCache({ maxMemoryMB: 10 });
      const data = new Float32Array(100).fill(1);

      // Store and access data
      cache.set('positions', [{ start: 0, end: 100 }], data);
      cache.get('positions', [{ start: 0, end: 100 }]);

      let stats = cache.getStats();
      expect(stats.avgAccessTime).toBeGreaterThanOrEqual(0);
      expect(stats.hits).toBe(1);

      // Clear cache
      cache.clear();

      // Stats should be reset
      stats = cache.getStats();
      expect(stats.avgAccessTime).toBe(0);
      expect(stats.hits).toBe(0);
      expect(stats.misses).toBe(0);
    });

    it('should evict old entries when memory limit exceeded', () => {
      // Set very small limit
      cache = new RangeCache({ maxMemoryMB: 0.001 }); // 1KB

      const ranges1: PointRange[] = [{ start: 0, end: 100 }];
      const ranges2: PointRange[] = [{ start: 100, end: 200 }];

      // Each Float32Array of 1000 elements = 4KB
      const data1 = new Float32Array(1000);
      const data2 = new Float32Array(1000);

      cache.set('positions', ranges1, data1);
      cache.set('positions', ranges2, data2);

      // First should be evicted
      expect(cache.get('positions', ranges1)).toBeNull();
      // Second should still be there
      expect(cache.get('positions', ranges2)).toEqual(data2);
    });
  });

  describe('Memory Management', () => {
    it('should track memory usage', () => {
      const ranges: PointRange[] = [{ start: 0, end: 100 }];
      const data = new Float32Array(1000); // 4KB

      cache.set('positions', ranges, data);

      const memInfo = cache.getMemoryInfo();
      expect(memInfo.used).toBe(4000);
      expect(memInfo.percentage).toBeGreaterThan(0);
    });

    it('should clear all data', () => {
      const ranges1: PointRange[] = [{ start: 0, end: 100 }];
      const ranges2: PointRange[] = [{ start: 100, end: 200 }];

      cache.set('positions', ranges1, new Float32Array(100));
      cache.set('colors', ranges2, new Float32Array(300));

      cache.clear();

      expect(cache.get('positions', ranges1)).toBeNull();
      expect(cache.get('colors', ranges2)).toBeNull();
      expect(cache.getMemoryInfo().used).toBe(0);
    });
  });
});

import { mergePointRanges } from '../data';

describe('Point Range Operations', () => {
  it('should merge adjacent ranges', () => {
    const ranges: PointRange[] = [
      { start: 0, end: 100 },
      { start: 100, end: 200 },
      { start: 300, end: 400 },
      { start: 200, end: 300 },
    ];

    const merged = mergePointRanges(ranges);

    // Should merge into one contiguous range
    expect(merged).toEqual([{ start: 0, end: 400 }]);
  });

  it('should not merge non-adjacent ranges', () => {
    const ranges: PointRange[] = [
      { start: 0, end: 100 },
      { start: 200, end: 300 },
      { start: 400, end: 500 },
    ];

    const merged = mergePointRanges(ranges);

    // Should remain separate
    expect(merged).toEqual(ranges);
  });
});
