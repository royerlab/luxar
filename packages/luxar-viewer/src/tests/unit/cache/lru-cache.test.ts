import { describe, it, expect, beforeEach } from 'vitest';
import { LRUCache } from '../../../cache/lru-cache';

describe('LRUCache', () => {
  let cache: LRUCache<Uint8Array>;
  const getSize = (v: Uint8Array) => v.byteLength;

  beforeEach(() => {
    cache = new LRUCache(100, getSize); // 100 bytes max
  });

  describe('Basic Operations', () => {
    it('should store and retrieve values', () => {
      const data = new Uint8Array([1, 2, 3]);
      cache.set('key1', data);
      expect(cache.get('key1')).toEqual(data);
    });

    it('should return undefined for missing keys', () => {
      expect(cache.get('nonexistent')).toBeUndefined();
    });

    it('should check key existence with has()', () => {
      cache.set('key1', new Uint8Array([1, 2, 3]));
      expect(cache.has('key1')).toBe(true);
      expect(cache.has('key2')).toBe(false);
    });

    it('should delete existing keys', () => {
      cache.set('key1', new Uint8Array([1, 2, 3]));
      expect(cache.delete('key1')).toBe(true);
      expect(cache.has('key1')).toBe(false);
      expect(cache.size).toBe(0);
    });

    it('should return false when deleting non-existent keys', () => {
      expect(cache.delete('nonexistent')).toBe(false);
    });

    it('should clear all entries', () => {
      cache.set('key1', new Uint8Array(10));
      cache.set('key2', new Uint8Array(20));
      cache.clear();
      expect(cache.size).toBe(0);
      expect(cache.count).toBe(0);
      expect(cache.has('key1')).toBe(false);
    });
  });

  describe('Size Tracking', () => {
    it('should track total size accurately', () => {
      cache.set('key1', new Uint8Array(10));
      expect(cache.size).toBe(10);

      cache.set('key2', new Uint8Array(20));
      expect(cache.size).toBe(30);

      cache.delete('key1');
      expect(cache.size).toBe(20);
    });

    it('should track entry count', () => {
      cache.set('key1', new Uint8Array(10));
      cache.set('key2', new Uint8Array(20));
      expect(cache.count).toBe(2);

      cache.delete('key1');
      expect(cache.count).toBe(1);
    });

    it('should update size when replacing existing key', () => {
      cache.set('key1', new Uint8Array(10));
      expect(cache.size).toBe(10);

      cache.set('key1', new Uint8Array(30)); // Replace with larger
      expect(cache.size).toBe(30);
      expect(cache.count).toBe(1); // Still just one entry
    });

    it('should handle zero-size items', () => {
      cache.set('key1', new Uint8Array(0));
      expect(cache.size).toBe(0);
      expect(cache.count).toBe(1);
      expect(cache.has('key1')).toBe(true);
    });
  });

  describe('LRU Eviction', () => {
    it('should evict least recently used when full', () => {
      cache.set('key1', new Uint8Array(40)); // 40 bytes
      cache.set('key2', new Uint8Array(40)); // 80 bytes
      cache.set('key3', new Uint8Array(40)); // 120 bytes - exceeds 100!

      // key1 should be evicted (least recently used)
      expect(cache.has('key1')).toBe(false);
      expect(cache.has('key2')).toBe(true);
      expect(cache.has('key3')).toBe(true);
      expect(cache.size).toBe(80);
    });

    it('should evict multiple items if needed', () => {
      cache.set('key1', new Uint8Array(30));
      cache.set('key2', new Uint8Array(30));
      cache.set('key3', new Uint8Array(30));
      // Total: 90 bytes

      cache.set('key4', new Uint8Array(50)); // Needs to evict key1 and key2

      expect(cache.has('key1')).toBe(false); // Evicted
      expect(cache.has('key2')).toBe(false); // Evicted
      expect(cache.has('key3')).toBe(true);
      expect(cache.has('key4')).toBe(true);
      expect(cache.size).toBe(80); // 30 + 50
    });

    it('should handle item larger than cache size', () => {
      cache.set('key1', new Uint8Array(50));
      cache.set('key2', new Uint8Array(150)); // Larger than 100 byte max!

      // All previous items evicted, new item added even though oversized
      expect(cache.has('key1')).toBe(false);
      expect(cache.has('key2')).toBe(true);
      expect(cache.size).toBe(150); // Exceeds max!
    });

    it('should not evict when adding zero-size item to full cache', () => {
      cache.set('key1', new Uint8Array(100)); // Exactly full
      cache.set('key2', new Uint8Array(0)); // Zero size

      expect(cache.has('key1')).toBe(true); // Not evicted
      expect(cache.has('key2')).toBe(true);
      expect(cache.size).toBe(100);
    });

    it('should break eviction loop safely if cache becomes empty', () => {
      cache.set('key1', new Uint8Array(50));
      cache.set('key2', new Uint8Array(200)); // Much larger than cache

      // Should evict key1, then add key2 even though oversized
      expect(cache.has('key1')).toBe(false);
      expect(cache.has('key2')).toBe(true);
    });
  });

  describe('LRU Ordering (Move to End)', () => {
    it('should move accessed items to end (most recently used)', () => {
      cache.set('key1', new Uint8Array(30));
      cache.set('key2', new Uint8Array(30));
      cache.set('key3', new Uint8Array(30));
      // Total: 90 bytes

      // Access key1 (moves it to end)
      // Order now: key2 (oldest), key3, key1 (newest)
      cache.get('key1');

      // Add key4 (50 bytes) - needs 90 + 50 = 140 bytes, max is 100
      // Must evict key2 (30) AND key3 (30) to make room for key4
      cache.set('key4', new Uint8Array(50));

      expect(cache.has('key1')).toBe(true); // Saved by access!
      expect(cache.has('key2')).toBe(false); // Evicted first
      expect(cache.has('key3')).toBe(false); // Evicted second
      expect(cache.has('key4')).toBe(true);
      expect(cache.size).toBe(80); // 30 + 50
    });

    it('should update order on every get()', () => {
      cache.set('key1', new Uint8Array(30));
      cache.set('key2', new Uint8Array(30));

      // Access key1 repeatedly
      cache.get('key1');
      cache.get('key1');
      cache.get('key1');

      // Add oversized item
      cache.set('key3', new Uint8Array(50));

      // key1 should still be there (accessed most recently)
      expect(cache.has('key1')).toBe(true);
      expect(cache.has('key2')).toBe(false); // Evicted
    });

    it('should not move non-existent keys', () => {
      cache.set('key1', new Uint8Array(50));
      const result = cache.get('nonexistent'); // Shouldn't crash

      expect(result).toBeUndefined();
      expect(cache.count).toBe(1); // Unchanged
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty cache operations', () => {
      expect(cache.get('key1')).toBeUndefined();
      expect(cache.delete('key1')).toBe(false);
      expect(cache.has('key1')).toBe(false);
      expect(cache.size).toBe(0);
      expect(cache.count).toBe(0);

      cache.clear(); // Should not crash
      expect(cache.size).toBe(0);
    });

    it('should handle single-item cache correctly', () => {
      cache.set('key1', new Uint8Array(50));

      expect(cache.size).toBe(50);
      expect(cache.count).toBe(1);
      expect(cache.get('key1')).toBeDefined();

      cache.clear();
      expect(cache.size).toBe(0);
      expect(cache.count).toBe(0);
    });

    it('should handle exact capacity', () => {
      cache.set('key1', new Uint8Array(100)); // Exactly 100 bytes
      expect(cache.size).toBe(100);
      expect(cache.count).toBe(1);

      // Adding one more byte should evict key1
      cache.set('key2', new Uint8Array(1));
      expect(cache.has('key1')).toBe(false);
      expect(cache.has('key2')).toBe(true);
    });

    it('should handle rapid updates to same key', () => {
      cache.set('key1', new Uint8Array(10));
      cache.set('key1', new Uint8Array(20));
      cache.set('key1', new Uint8Array(15));

      expect(cache.size).toBe(15); // Last value
      expect(cache.count).toBe(1); // Still one entry
    });

    it('should maintain consistency after many operations', () => {
      for (let i = 0; i < 100; i++) {
        cache.set(`key${i}`, new Uint8Array(10));
      }

      // Should have evicted early keys
      expect(cache.count).toBeLessThanOrEqual(10); // 100 bytes / 10 bytes = 10 max
      expect(cache.size).toBeLessThanOrEqual(100);

      // Most recent keys should be present
      expect(cache.has('key99')).toBe(true);
      expect(cache.has('key98')).toBe(true);

      // Earliest keys should be evicted
      expect(cache.has('key0')).toBe(false);
      expect(cache.has('key1')).toBe(false);
    });
  });

  describe('Size Function Edge Cases', () => {
    it('should work with custom size functions', () => {
      interface CustomValue {
        data: string;
        weight: number;
      }

      const customCache = new LRUCache<CustomValue>(100, (v) => v.weight);

      customCache.set('key1', { data: 'hello', weight: 50 });
      customCache.set('key2', { data: 'world', weight: 30 });

      expect(customCache.size).toBe(80);
      expect(customCache.count).toBe(2);

      customCache.set('key3', { data: 'test', weight: 40 }); // Evicts key1

      expect(customCache.has('key1')).toBe(false);
      expect(customCache.size).toBe(70); // 30 + 40
    });
  });

  describe('Stress Test', () => {
    it('should handle many sequential operations efficiently', () => {
      const iterations = 1000;
      const startTime = performance.now();

      for (let i = 0; i < iterations; i++) {
        cache.set(`key${i}`, new Uint8Array(5));
        if (i % 2 === 0) {
          cache.get(`key${Math.max(0, i - 5)}`); // Access recent keys
        }
      }

      const elapsed = performance.now() - startTime;

      // Should complete in reasonable time (O(1) per operation)
      expect(elapsed).toBeLessThan(100); // 100ms for 1000 ops

      // Size should not exceed max
      expect(cache.size).toBeLessThanOrEqual(100);

      // Should have approximately correct number of entries
      expect(cache.count).toBeGreaterThan(0);
      expect(cache.count).toBeLessThanOrEqual(20); // 100 / 5
    });
  });
});
