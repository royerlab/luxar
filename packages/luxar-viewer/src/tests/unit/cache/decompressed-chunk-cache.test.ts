/**
 * Unit tests for DecompressedChunkCache (L0 cache layer)
 *
 * Tests the LRU-based cache for decompressed zarr chunks that eliminates
 * Blosc decompression overhead on cache hits.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  DecompressedChunkCache,
  type DecompressedChunk,
} from '../../../cache/decompressed-chunk-cache';

describe('DecompressedChunkCache', () => {
  let cache: DecompressedChunkCache;

  beforeEach(() => {
    // Create a fresh cache for each test
    cache = new DecompressedChunkCache({ maxSize: 1024 * 1024 }); // 1MB for tests
  });

  describe('Basic Operations', () => {
    it('should store and retrieve chunks', () => {
      const key = DecompressedChunkCache.makeKey('/points/positions', [0, 1, 2]);
      const chunk: DecompressedChunk = {
        data: new Float32Array([1, 2, 3, 4, 5, 6]),
        shape: [2, 3],
        stride: [3, 1],
      };

      cache.set(key, chunk);
      const retrieved = cache.get(key);

      expect(retrieved).toBeDefined();
      expect(retrieved?.data).toEqual(chunk.data);
      expect(retrieved?.shape).toEqual(chunk.shape);
      expect(retrieved?.stride).toEqual(chunk.stride);
    });

    it('should return undefined for missing keys', () => {
      const key = DecompressedChunkCache.makeKey('/nonexistent', [0]);
      expect(cache.get(key)).toBeUndefined();
    });

    it('should check if key exists with has()', () => {
      const key = DecompressedChunkCache.makeKey('/points/positions', [0]);
      const chunk: DecompressedChunk = {
        data: new Float32Array([1, 2, 3]),
        shape: [3],
        stride: [1],
      };

      expect(cache.has(key)).toBe(false);
      cache.set(key, chunk);
      expect(cache.has(key)).toBe(true);
    });

    it('should clear all entries', () => {
      const key1 = DecompressedChunkCache.makeKey('/a', [0]);
      const key2 = DecompressedChunkCache.makeKey('/b', [0]);
      const chunk: DecompressedChunk = {
        data: new Float32Array([1, 2, 3]),
        shape: [3],
        stride: [1],
      };

      cache.set(key1, chunk);
      cache.set(key2, chunk);
      expect(cache.has(key1)).toBe(true);
      expect(cache.has(key2)).toBe(true);

      cache.clear();
      expect(cache.has(key1)).toBe(false);
      expect(cache.has(key2)).toBe(false);
    });
  });

  describe('LRU Eviction', () => {
    it('should evict least recently used entries when size limit exceeded', () => {
      // Each chunk is ~160 bytes (24 floats = 96 bytes + 64 overhead)
      // Create cache with size limit that can hold ~2 chunks
      const smallCache = new DecompressedChunkCache({ maxSize: 320 });

      const chunk1: DecompressedChunk = {
        data: new Float32Array(24),
        shape: [24],
        stride: [1],
      };
      const chunk2: DecompressedChunk = {
        data: new Float32Array(24),
        shape: [24],
        stride: [1],
      };
      const chunk3: DecompressedChunk = {
        data: new Float32Array(24),
        shape: [24],
        stride: [1],
      };

      const key1 = DecompressedChunkCache.makeKey('/a', [0]);
      const key2 = DecompressedChunkCache.makeKey('/b', [0]);
      const key3 = DecompressedChunkCache.makeKey('/c', [0]);

      smallCache.set(key1, chunk1);
      smallCache.set(key2, chunk2);

      // Verify both are in cache
      expect(smallCache.has(key1)).toBe(true);
      expect(smallCache.has(key2)).toBe(true);

      // Adding third should evict first (LRU)
      smallCache.set(key3, chunk3);

      // cache.md W26 fix: pin the full LRU contract — key1 evicted
      // (oldest), key2 AND key3 both still present (most-recently used).
      // Previous version only asserted key3 present, leaving a regression
      // that evicts both key1 AND key2 invisible.
      expect(smallCache.has(key1)).toBe(false);
      expect(smallCache.has(key2)).toBe(true);
      expect(smallCache.has(key3)).toBe(true);
    });

    it('should update LRU order on get()', () => {
      // Size limit that can hold ~2 chunks
      const smallCache = new DecompressedChunkCache({ maxSize: 320 });

      const chunk: DecompressedChunk = {
        data: new Float32Array(24),
        shape: [24],
        stride: [1],
      };

      const key1 = DecompressedChunkCache.makeKey('/a', [0]);
      const key2 = DecompressedChunkCache.makeKey('/b', [0]);
      const key3 = DecompressedChunkCache.makeKey('/c', [0]);

      smallCache.set(key1, chunk);
      smallCache.set(key2, chunk);

      // Access key1 to make it recently used
      smallCache.get(key1);

      // Add key3 - should evict key2 (now LRU since key1 was accessed)
      smallCache.set(key3, chunk);

      expect(smallCache.has(key1)).toBe(true); // Recently accessed
      expect(smallCache.has(key2)).toBe(false); // LRU, evicted
      expect(smallCache.has(key3)).toBe(true); // Just added
    });
  });

  describe('Statistics', () => {
    it('should track hits and misses', () => {
      const key = DecompressedChunkCache.makeKey('/points/positions', [0]);
      const chunk: DecompressedChunk = {
        data: new Float32Array([1, 2, 3]),
        shape: [3],
        stride: [1],
      };

      // Initial stats
      let stats = cache.getStats();
      expect(stats.hits).toBe(0);
      expect(stats.misses).toBe(0);

      // Miss
      cache.get(key);
      stats = cache.getStats();
      expect(stats.misses).toBe(1);

      // Add entry
      cache.set(key, chunk);

      // Hit
      cache.get(key);
      stats = cache.getStats();
      expect(stats.hits).toBe(1);
      expect(stats.hitRate).toBeCloseTo(0.5, 5); // 1 hit, 1 miss
    });

    it('should track size and count', () => {
      const chunk: DecompressedChunk = {
        data: new Float32Array(100), // 400 bytes
        shape: [100],
        stride: [1],
      };

      cache.set(DecompressedChunkCache.makeKey('/a', [0]), chunk);
      cache.set(DecompressedChunkCache.makeKey('/b', [0]), chunk);

      const stats = cache.getStats();
      expect(stats.count).toBe(2);
      // Size should be data bytes + metadata overhead per entry
      expect(stats.size).toBeGreaterThan(800); // 2 * 400 bytes + overhead
    });

    it('tracks evictions: evictions == (sets - count) holds exactly', () => {
      // cache.md W25 fix: previous version asserted `evictions > 0`.
      // Strengthen using the algebraic invariant: for a cache that never
      // grows beyond `maxSize`, evictions equals the number of
      // never-readded entries, i.e. (sets - currentCount). Three sets
      // into a small cache yields a deterministic count + evictions
      // pair whose sum is 3.
      const smallCache = new DecompressedChunkCache({ maxSize: 200 });

      const chunk: DecompressedChunk = {
        data: new Float32Array(24),
        shape: [24],
        stride: [1],
      };

      smallCache.set(DecompressedChunkCache.makeKey('/a', [0]), chunk);
      smallCache.set(DecompressedChunkCache.makeKey('/b', [0]), chunk);
      smallCache.set(DecompressedChunkCache.makeKey('/c', [0]), chunk);

      const stats = smallCache.getStats();
      // Conservation: count + evictions == sets (3).
      expect(stats.count + stats.evictions).toBe(3);
      // Sanity: at least one eviction must have occurred for a 200-byte
      // cap holding 3 × 96-byte payloads.
      expect(stats.evictions).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Key Generation', () => {
    it('should generate correct keys with makeKey()', () => {
      const key = DecompressedChunkCache.makeKey('/scene/points/positions', [0, 1, 2]);
      expect(key).toBe('/scene/points/positions:0,1,2');
    });

    it('should handle single dimension', () => {
      const key = DecompressedChunkCache.makeKey('/radii', [5]);
      expect(key).toBe('/radii:5');
    });

    it('should handle many dimensions', () => {
      const key = DecompressedChunkCache.makeKey('/data', [1, 2, 3, 4, 5]);
      expect(key).toBe('/data:1,2,3,4,5');
    });

    it('should parse keys back correctly with parseKey()', () => {
      const original = { arrayPath: '/scene/points/positions', chunkCoords: [0, 1, 2] };
      const key = DecompressedChunkCache.makeKey(original.arrayPath, original.chunkCoords);
      const parsed = DecompressedChunkCache.parseKey(key);

      expect(parsed).toEqual(original);
    });

    it('should return null for invalid keys', () => {
      expect(DecompressedChunkCache.parseKey('invalid-no-colon')).toBeNull();
      expect(DecompressedChunkCache.parseKey('/path:not,numbers,here')).toBeNull();
    });
  });

  describe('Data Type Support', () => {
    it('should handle Float32Array data', () => {
      const key = DecompressedChunkCache.makeKey('/positions', [0]);
      const chunk: DecompressedChunk = {
        data: new Float32Array([1.5, 2.5, 3.5]),
        shape: [3],
        stride: [1],
      };

      cache.set(key, chunk);
      const retrieved = cache.get(key);

      expect(retrieved?.data).toBeInstanceOf(Float32Array);
      expect((retrieved?.data as Float32Array)[0]).toBe(1.5);
    });

    it('should handle Uint8Array data', () => {
      const key = DecompressedChunkCache.makeKey('/colors', [0]);
      const chunk: DecompressedChunk = {
        data: new Uint8Array([255, 128, 64]),
        shape: [3],
        stride: [1],
      };

      cache.set(key, chunk);
      const retrieved = cache.get(key);

      expect(retrieved?.data).toBeInstanceOf(Uint8Array);
      expect((retrieved?.data as Uint8Array)[0]).toBe(255);
    });

    it('should handle Uint32Array data (for indices)', () => {
      const key = DecompressedChunkCache.makeKey('/segments', [0]);
      const chunk: DecompressedChunk = {
        data: new Uint32Array([0, 1, 1, 2, 2, 3]),
        shape: [3, 2],
        stride: [2, 1],
      };

      cache.set(key, chunk);
      const retrieved = cache.get(key);

      expect(retrieved?.data).toBeInstanceOf(Uint32Array);
    });
  });
});
