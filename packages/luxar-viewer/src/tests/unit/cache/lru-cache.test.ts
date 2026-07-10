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

    it('delete() of missing key does not perturb currentSize bookkeeping', () => {
      // MED-1 regression: prior code path computed `this.cache.get(key)`
      // and then guarded with `if (value !== undefined)`. That worked in
      // practice because Map.get returns undefined for missing keys, but
      // the bookkeeping is now guarded by `cache.has(key)` so a future
      // tweak (e.g. allowing stored undefined sentinels) cannot
      // accidentally decrement currentSize on a no-op delete.
      cache.set('present', new Uint8Array(40));
      expect(cache.size).toBe(40);
      // Repeatedly delete missing keys — size must stay at 40.
      expect(cache.delete('ghost-a')).toBe(false);
      expect(cache.delete('ghost-b')).toBe(false);
      expect(cache.delete('ghost-c')).toBe(false);
      expect(cache.size).toBe(40);
      expect(cache.count).toBe(1);
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

    it('should reject item larger than cache size', () => {
      cache.set('key1', new Uint8Array(50));
      cache.set('key2', new Uint8Array(150)); // Larger than 100 byte max!

      // Oversized item silently rejected, existing entries preserved
      expect(cache.has('key1')).toBe(true);
      expect(cache.has('key2')).toBe(false);
      expect(cache.size).toBe(50);
    });

    it('should not evict when adding zero-size item to full cache', () => {
      cache.set('key1', new Uint8Array(100)); // Exactly full
      cache.set('key2', new Uint8Array(0)); // Zero size

      expect(cache.has('key1')).toBe(true); // Not evicted
      expect(cache.has('key2')).toBe(true);
      expect(cache.size).toBe(100);
    });

    it('should reject oversized items without evicting existing entries', () => {
      cache.set('key1', new Uint8Array(50));
      cache.set('key2', new Uint8Array(200)); // Much larger than cache

      // Oversized item rejected, existing entries preserved
      expect(cache.has('key1')).toBe(true);
      expect(cache.has('key2')).toBe(false);
      expect(cache.size).toBe(50);
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
      // [cache.md/Wn][P2] Previously asserted `get('key1')).toBeDefined()`.
      // The value is a Uint8Array(50) of zeros, so a regression returning
      // a wrong-sized buffer would still pass `toBeDefined()`. Pin exact
      // length and a representative byte.
      cache.set('key1', new Uint8Array(50));

      expect(cache.size).toBe(50);
      expect(cache.count).toBe(1);
      const got = cache.get('key1');
      expect(got?.byteLength).toBe(50);
      expect(got?.[0]).toBe(0);

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

    // [cache.md/G3][P5] Boundary: maxSize === 0 → degenerate cache that
    // rejects any non-empty write (size > maxSize triggers the early return
    // in set() at lru-cache.ts:106). Zero-byte values still slot in.
    it('maxSize=0 degenerate cache rejects all non-empty writes', () => {
      const zeroCache = new LRUCache<Uint8Array>(0, getSize);
      zeroCache.set('key1', new Uint8Array(10));
      expect(zeroCache.has('key1')).toBe(false);
      expect(zeroCache.count).toBe(0);
      expect(zeroCache.size).toBe(0);

      // Zero-byte payload: size === maxSize === 0, so the size-overflow
      // guard does NOT reject it. It is accepted but adds nothing to size.
      zeroCache.set('empty', new Uint8Array(0));
      expect(zeroCache.has('empty')).toBe(true);
      expect(zeroCache.size).toBe(0);
      expect(zeroCache.count).toBe(1);
    });

    // [cache.md/G3][P5] Boundary: eviction at exact-capacity boundary.
    // Adding one byte beyond capacity must evict exactly enough to fit.
    it('eviction triggers at size === maxSize + 1 (one byte over capacity)', () => {
      // cache.maxSize is 100 (per beforeEach).
      cache.set('a', new Uint8Array(100)); // fills cache exactly
      expect(cache.size).toBe(100);
      expect(cache.count).toBe(1);

      // One additional byte → must evict a.
      cache.set('b', new Uint8Array(1));
      expect(cache.has('a')).toBe(false);
      expect(cache.has('b')).toBe(true);
      expect(cache.size).toBe(1);
      expect(cache.count).toBe(1);
      // The single eviction must have been counted.
      expect(cache.evictionCount).toBe(1);
    });

    // [cache.md/G3][P5] Cache key boundary: empty-string key is accepted.
    // (Empty key is a legitimate value in a Map; the LRU treats it as any
    // other key.)
    it('empty-string key roundtrips through set/get/has/delete', () => {
      const payload = new Uint8Array([42]);
      cache.set('', payload);
      expect(cache.has('')).toBe(true);
      expect(cache.get('')).toEqual(payload);
      expect(cache.delete('')).toBe(true);
      expect(cache.has('')).toBe(false);
    });

    // [cache.md/G3][P5] Cache key boundary: a key >1KB roundtrips intact.
    // The LRU is a Map<string,V>, so long keys are pure hash overhead;
    // pin the behavior in case a future change introduces key truncation
    // or hashing that loses uniqueness at scale.
    it('very long key (>1KB) roundtrips and is distinct from any prefix', () => {
      const longKey = 'k'.repeat(1500);
      const prefixKey = 'k'.repeat(750);
      cache.set(longKey, new Uint8Array([1]));
      cache.set(prefixKey, new Uint8Array([2]));
      // Both distinct entries stored.
      expect(cache.count).toBe(2);
      expect(cache.get(longKey)).toEqual(new Uint8Array([1]));
      expect(cache.get(prefixKey)).toEqual(new Uint8Array([2]));
    });

    // [cache.md/G3][P5] Cache key boundary: unicode key (multi-byte UTF-8
    // sequences). LRU stores keys as JS strings, so this MUST work — but
    // pin the contract so a regression (e.g. one that base64-encodes keys
    // and uses btoa) cannot silently break it.
    it('unicode key (multi-byte UTF-8 / emoji) roundtrips intact', () => {
      const unicodeKey = '通道/データ/positions/🎯/0.0';
      cache.set(unicodeKey, new Uint8Array([7, 8, 9]));
      expect(cache.get(unicodeKey)).toEqual(new Uint8Array([7, 8, 9]));
      expect(cache.has(unicodeKey)).toBe(true);
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

    // Audit G3 (viewer-data-cache-workers-wasm): the LRU cache uses
    // `value.byteLength` as its default size — and that field can be
    // NaN if a caller assembles a malformed object. Pin the current
    // (forgiving) behaviour so future refactors know what the contract is.
    //
    // Negative byteLength is structurally impossible for Uint8Array so
    // we test the NaN-size case via a custom-size cache.
    it('NaN-valued custom size leaves cache.size as NaN (documentary)', () => {
      // Currently the cache does not validate the size return — passing
      // a function that returns NaN poisons cache.size. If the
      // production code starts rejecting NaN sizes, this test should
      // be flipped to assert the rejection contract.
      const customCache = new LRUCache<{ weight: number }>(100, (v) => v.weight);
      customCache.set('bad', { weight: NaN });
      expect(Number.isNaN(customCache.size)).toBe(true);
      // The cache still records the entry — count reflects the entry.
      expect(customCache.count).toBe(1);
    });

    it('zero-size item adds to count but not size', () => {
      // Audit G3: zero-size boundary is already covered for byteLength=0
      // above; pin the analogous custom-size case so the contract
      // ("size 0 is valid; entry is tracked") is symmetric.
      const customCache = new LRUCache<{ weight: number }>(100, (v) => v.weight);
      customCache.set('zero', { weight: 0 });
      expect(customCache.size).toBe(0);
      expect(customCache.count).toBe(1);
      expect(customCache.has('zero')).toBe(true);
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

  describe('Hit/Miss Counter Tracking', () => {
    it('should track hits correctly', () => {
      cache.set('key1', new Uint8Array([1, 2, 3]));

      expect(cache.hitCount).toBe(0);
      cache.get('key1'); // Hit
      expect(cache.hitCount).toBe(1);
      cache.get('key1'); // Hit
      cache.get('key1'); // Hit
      expect(cache.hitCount).toBe(3);
    });

    it('should track misses correctly', () => {
      expect(cache.missCount).toBe(0);
      cache.get('nonexistent'); // Miss
      expect(cache.missCount).toBe(1);
      cache.get('another'); // Miss
      cache.get('missing'); // Miss
      expect(cache.missCount).toBe(3);
    });

    it('should track both hits and misses independently', () => {
      cache.set('key1', new Uint8Array([1]));

      cache.get('key1'); // Hit
      cache.get('nonexistent'); // Miss
      cache.get('key1'); // Hit
      cache.get('missing'); // Miss

      expect(cache.hitCount).toBe(2);
      expect(cache.missCount).toBe(2);
    });

    it('should reset counters on clear()', () => {
      cache.set('key1', new Uint8Array([1]));
      cache.get('key1'); // Hit
      cache.get('missing'); // Miss

      expect(cache.hitCount).toBe(1);
      expect(cache.missCount).toBe(1);

      cache.clear();

      expect(cache.hitCount).toBe(0);
      expect(cache.missCount).toBe(0);
    });

    it('should count hit when accessing existing key that was just set', () => {
      cache.set('key1', new Uint8Array([1]));
      expect(cache.hitCount).toBe(0); // Set doesn't count as hit

      cache.get('key1');
      expect(cache.hitCount).toBe(1);
    });

    it('should not count as miss when has() returns false', () => {
      // has() doesn't increment counters
      cache.has('nonexistent');
      expect(cache.missCount).toBe(0);
      expect(cache.hitCount).toBe(0);
    });
  });

  describe('Scan-Mode Eviction (evictMostRecent)', () => {
    // Scan-resistant policy for sequential scans (dimension playback): the
    // MOST-recently-used entry is needed farthest in the future during a
    // cyclic scan, so it is the victim — the oldest prefix stays resident.
    let cache: LRUCache<string>;

    beforeEach(() => {
      cache = new LRUCache<string>(30, (v) => v.length);
    });

    it('evicts the most-recently-used entry instead of the oldest', () => {
      cache.set('a', '0123456789'); // 10 bytes, oldest
      cache.set('b', '0123456789');
      cache.set('c', '0123456789'); // MRU
      cache.set('d', '0123456789', { evictMostRecent: true }); // overflow

      expect(cache.has('a')).toBe(true); // oldest SURVIVES under scan
      expect(cache.has('b')).toBe(true);
      expect(cache.has('c')).toBe(false); // MRU evicted
      expect(cache.has('d')).toBe(true);
    });

    it('never evicts the key being inserted', () => {
      cache.set('a', '0123456789');
      cache.set('b', '0123456789');
      cache.set('c', '0123456789');
      // Re-store an EXISTING key with a larger value that forces eviction:
      // the incoming key must not be its own victim.
      cache.set('c', '01234567890123456789', { evictMostRecent: true }); // 20 bytes
      expect(cache.has('c')).toBe(true);
      expect(cache.size).toBeLessThanOrEqual(30);
    });

    it('a get-promoted entry becomes the scan victim (correct for cyclic scans)', () => {
      cache.set('a', '0123456789');
      cache.set('b', '0123456789');
      cache.set('c', '0123456789');
      cache.get('a'); // promote 'a' to MRU — just used = needed farthest ahead
      cache.set('d', '0123456789', { evictMostRecent: true });

      expect(cache.has('a')).toBe(false); // promoted entry evicted
      expect(cache.has('b')).toBe(true);
      expect(cache.has('c')).toBe(true);
    });

    it('single-entry cache: evicts it and terminates (no infinite loop)', () => {
      const tiny = new LRUCache<string>(10, (v) => v.length);
      tiny.set('a', '0123456789');
      tiny.set('b', '0123456789', { evictMostRecent: true });
      expect(tiny.has('a')).toBe(false);
      expect(tiny.has('b')).toBe(true);
      expect(tiny.count).toBe(1);
    });

    it('default mode is byte-identical to plain LRU (opt absent or false)', () => {
      cache.set('a', '0123456789');
      cache.set('b', '0123456789');
      cache.set('c', '0123456789');
      cache.set('d', '0123456789', { evictMostRecent: false });
      expect(cache.has('a')).toBe(false); // oldest evicted, as always
      expect(cache.has('d')).toBe(true);
    });

    it('cyclic-scan pathology: scan mode turns 0% loop-2 hits into loop-head hits', () => {
      // 10-key cycle over a 3-entry budget. Plain LRU: loop 2 hits NOTHING
      // (each key was evicted before its revisit). Scan mode: the oldest
      // prefix (k0,k1) stays resident and hits right after the wrap.
      const run = (scan: boolean) => {
        const c = new LRUCache<string>(30, (v) => v.length);
        // Loop 1: store the full cycle.
        for (let i = 0; i < 10; i++) c.set(`k${i}`, '0123456789', { evictMostRecent: scan });
        // Loop 2: revisit — count hits.
        let hits = 0;
        for (let i = 0; i < 10; i++) {
          if (c.get(`k${i}`) !== undefined) hits++;
          else c.set(`k${i}`, '0123456789', { evictMostRecent: scan });
        }
        return hits;
      };
      expect(run(false)).toBe(0); // the classic sequential-scan LRU pathology
      expect(run(true)).toBeGreaterThanOrEqual(2); // loop-head prefix survives
    });
  });

  describe('Eviction Counter Tracking', () => {
    it('should track evictions correctly', () => {
      expect(cache.evictionCount).toBe(0);

      // Fill cache: 40 + 40 = 80 bytes
      cache.set('key1', new Uint8Array(40));
      cache.set('key2', new Uint8Array(40));
      expect(cache.evictionCount).toBe(0);

      // Add 40 more bytes - exceeds 100, must evict key1
      cache.set('key3', new Uint8Array(40));
      expect(cache.evictionCount).toBe(1);
      expect(cache.has('key1')).toBe(false);
    });

    it('should count multiple evictions in single set()', () => {
      // Add 3 items: 30 + 30 + 30 = 90 bytes
      cache.set('key1', new Uint8Array(30));
      cache.set('key2', new Uint8Array(30));
      cache.set('key3', new Uint8Array(30));
      expect(cache.evictionCount).toBe(0);

      // Add 50 bytes - needs to evict key1 (30) AND key2 (30) = 2 evictions
      cache.set('key4', new Uint8Array(50));
      expect(cache.evictionCount).toBe(2);
      expect(cache.has('key1')).toBe(false);
      expect(cache.has('key2')).toBe(false);
    });

    it('should reset eviction counter on clear()', () => {
      cache.set('key1', new Uint8Array(40));
      cache.set('key2', new Uint8Array(40));
      cache.set('key3', new Uint8Array(40)); // Evicts key1

      expect(cache.evictionCount).toBe(1);

      cache.clear();
      expect(cache.evictionCount).toBe(0);
    });

    it('should accumulate evictions over multiple operations', () => {
      // Each set evicts the previous item
      cache.set('key1', new Uint8Array(100)); // Full
      expect(cache.evictionCount).toBe(0);

      cache.set('key2', new Uint8Array(100)); // Evicts key1
      expect(cache.evictionCount).toBe(1);

      cache.set('key3', new Uint8Array(100)); // Evicts key2
      expect(cache.evictionCount).toBe(2);

      cache.set('key4', new Uint8Array(100)); // Evicts key3
      expect(cache.evictionCount).toBe(3);
    });

    it('should not count eviction when replacing same key', () => {
      cache.set('key1', new Uint8Array(50));
      cache.set('key1', new Uint8Array(60)); // Replace, not evict

      expect(cache.evictionCount).toBe(0);
      expect(cache.size).toBe(60);
    });
  });

  describe('onEvict Callback', () => {
    it('should call onEvict when items are evicted', () => {
      const evicted: Array<{ key: string; size: number }> = [];
      const evictCache = new LRUCache<Uint8Array>(100, getSize, (key, value) => {
        evicted.push({ key, size: value.byteLength });
      });

      evictCache.set('key1', new Uint8Array(60));
      evictCache.set('key2', new Uint8Array(60)); // Evicts key1
      expect(evicted).toHaveLength(1);
      expect(evicted[0]).toEqual({ key: 'key1', size: 60 });
    });

    it('should call onEvict for all items on clear()', () => {
      const evicted: string[] = [];
      const evictCache = new LRUCache<Uint8Array>(100, getSize, (key) => {
        evicted.push(key);
      });

      evictCache.set('a', new Uint8Array(10));
      evictCache.set('b', new Uint8Array(20));
      evictCache.set('c', new Uint8Array(30));
      evictCache.clear();

      expect(evicted.sort()).toEqual(['a', 'b', 'c']);
    });

    it('should not call onEvict when no callback provided', () => {
      // No callback — should not throw
      const noCallbackCache = new LRUCache<Uint8Array>(50, getSize);
      noCallbackCache.set('key1', new Uint8Array(40));
      noCallbackCache.set('key2', new Uint8Array(40)); // Evicts key1, no error
      expect(noCallbackCache.count).toBe(1);
    });

    // workers.md O3 / cache.md G2 [P5]: previous test name embedded the
    // audit-id in the `it` string; moved to this comment per Phase
    // E53 to decouple the suite from the audit-numbering lifecycle.
    it('fires onEvict when an item is removed via delete()', () => {
      // [cache.md/G2][P5] Previous onEvict tests only exercised eviction via
      // size-overflow and clear(); never via delete(). Mutating the
      // `onEvict?.(key, value)` line inside delete() to a no-op would not
      // fail any existing test. Pin the delete() path explicitly.
      const evicted: Array<{ key: string; size: number }> = [];
      const cache = new LRUCache<Uint8Array>(1000, getSize, (key, value) => {
        evicted.push({ key, size: value.byteLength });
      });

      cache.set('to-delete', new Uint8Array(42));
      cache.set('to-keep', new Uint8Array(8));

      const ok = cache.delete('to-delete');
      expect(ok).toBe(true);
      expect(evicted).toEqual([{ key: 'to-delete', size: 42 }]);

      // delete() on a missing key returns false and does NOT fire onEvict.
      const evictedBefore = evicted.length;
      const ok2 = cache.delete('never-existed');
      expect(ok2).toBe(false);
      expect(evicted.length).toBe(evictedBefore);
    });

    // workers.md O3 / cache.md G2 [P5]: audit-id moved from test name
    // to this comment per Phase E53.
    it('delete() decrements currentSize by exactly the deleted entry size', () => {
      // Complement to the onEvict-on-delete test: pin the size bookkeeping
      // so mutating the `this.currentSize -= ...` line in delete() to a no-op
      // (or a wrong sign) is detected.
      const cache = new LRUCache<Uint8Array>(1000, getSize);
      cache.set('a', new Uint8Array(100));
      cache.set('b', new Uint8Array(50));
      expect(cache.size).toBe(150);

      cache.delete('a');
      expect(cache.size).toBe(50);
      expect(cache.count).toBe(1);
      // Deleting the same key again is a no-op.
      cache.delete('a');
      expect(cache.size).toBe(50);
    });
  });

  describe('Boundary cases [cache.md/G3][P5]', () => {
    it('maxSize === 0 rejects every write (degenerate cache)', () => {
      // [cache.md/G3][P5] Previously no boundary test for maxSize===0.
      // Implementation: `if (size > maxSize) return` — any non-zero-size
      // value (every Uint8Array of length >=1) is rejected pre-insertion,
      // so the cache stays empty and onEvict is never fired.
      const evicted: string[] = [];
      const cache = new LRUCache<Uint8Array>(0, getSize, (key) => {
        evicted.push(key);
      });

      cache.set('k1', new Uint8Array(1));
      expect(cache.count).toBe(0);
      expect(cache.size).toBe(0);
      expect(cache.has('k1')).toBe(false);
      // Pre-insertion rejection: onEvict is NOT called (the entry never
      // entered the cache, so there is nothing to "evict").
      expect(evicted).toEqual([]);
    });

    // workers.md O3 / cache.md G3 [P5]: audit-id moved from test name
    // to this comment per Phase E53.
    it('rejects oversized values silently without disturbing existing entries', () => {
      // Companion to the maxSize===0 test: a value larger than maxSize is
      // rejected, and the cache's existing contents are unchanged. Mutating
      // the `if (size > this.maxSize) return` early-return would erase the
      // pre-existing entry — this test pins against that regression.
      const cache = new LRUCache<Uint8Array>(100, getSize);
      cache.set('keeper', new Uint8Array(50));
      expect(cache.has('keeper')).toBe(true);
      expect(cache.size).toBe(50);

      cache.set('too-big', new Uint8Array(200));
      // The oversized value is rejected; the pre-existing entry survives.
      expect(cache.has('too-big')).toBe(false);
      expect(cache.has('keeper')).toBe(true);
      expect(cache.size).toBe(50);
    });
  });

  describe('Pinned keys (eviction protection)', () => {
    it('skips a pinned key when choosing an LRU victim', () => {
      const c = new LRUCache<Uint8Array>(100, getSize);
      c.set('a', new Uint8Array(40)); // oldest
      c.set('b', new Uint8Array(40));
      c.pin('a'); // protect the oldest (would normally be the LRU victim)
      c.set('d', new Uint8Array(40)); // 120 > 100 → must evict; 'a' is pinned → 'b' goes
      expect(c.has('a')).toBe(true);
      expect(c.has('b')).toBe(false);
      expect(c.has('d')).toBe(true);
    });

    it('skips a pinned key under scan (MRU-victim) eviction too', () => {
      const c = new LRUCache<Uint8Array>(100, getSize);
      c.set('a', new Uint8Array(40));
      c.set('b', new Uint8Array(40)); // newest → the scan victim
      c.pin('b'); // but it's pinned (e.g. a just-prefetched t+1)
      c.set('d', new Uint8Array(40), { evictMostRecent: true }); // evicts 'a', not pinned 'b'
      expect(c.has('b')).toBe(true);
      expect(c.has('a')).toBe(false);
      expect(c.has('d')).toBe(true);
    });

    it('evicts a pinned key as a LAST RESORT rather than violate the byte budget', () => {
      const c = new LRUCache<Uint8Array>(100, getSize);
      c.set('a', new Uint8Array(60));
      c.pin('a');
      // A 60-byte insert can't fit alongside the pinned 60; nothing else to
      // evict → the pin is sacrificed so the invariant (size ≤ max) holds.
      c.set('b', new Uint8Array(60));
      expect(c.has('b')).toBe(true);
      expect(c.has('a')).toBe(false);
      expect(c.size).toBeLessThanOrEqual(100);
    });

    it('unpin() re-exposes a key to eviction', () => {
      const c = new LRUCache<Uint8Array>(100, getSize);
      c.set('a', new Uint8Array(40));
      c.set('b', new Uint8Array(40));
      c.pin('a');
      c.unpin('a');
      c.set('d', new Uint8Array(40)); // 'a' now unprotected and oldest → evicted
      expect(c.has('a')).toBe(false);
      expect(c.has('b')).toBe(true);
      expect(c.has('d')).toBe(true);
    });

    it('clear() drops all pins', () => {
      const c = new LRUCache<Uint8Array>(100, getSize);
      c.set('a', new Uint8Array(40));
      c.pin('a');
      c.clear();
      // Re-populate; 'a' must be a normal, evictable entry again.
      c.set('a', new Uint8Array(40));
      c.set('b', new Uint8Array(40));
      c.set('d', new Uint8Array(40)); // evicts LRU 'a' (pin did not survive clear)
      expect(c.has('a')).toBe(false);
    });
  });
});
