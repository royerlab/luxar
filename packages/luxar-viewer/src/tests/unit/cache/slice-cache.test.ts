import { describe, it, expect } from 'vitest';
import { SliceCache } from '../../../cache/slice-cache';

/** A payload of `bytes` bytes (a single typed array) for size-accounting tests. */
function entry(bytes: number, tag = 'x') {
  return { payload: { tag, buf: new Uint8Array(bytes) }, bytes };
}

describe('SliceCache', () => {
  it('stores and retrieves an opaque payload by key', () => {
    const c = new SliceCache({ maxSize: 1024 });
    const key = SliceCache.makeKey('/neuromast', 'view-8');
    const e = entry(64, 't8');
    c.set(key, e);
    expect(c.has(key)).toBe(true);
    const got = c.get(key);
    expect(got).toBe(e);
    expect((got!.payload as { tag: string }).tag).toBe('t8');
  });

  it('returns undefined and counts a miss for an unknown key', () => {
    const c = new SliceCache({ maxSize: 1024 });
    expect(c.get(SliceCache.makeKey('/n', 'v'))).toBeUndefined();
    expect(c.getStats().misses).toBe(1);
    expect(c.getStats().hits).toBe(0);
  });

  it('tracks hit/miss/hitRate in stats', () => {
    const c = new SliceCache({ maxSize: 1024 });
    const k = SliceCache.makeKey('/n', 'v');
    c.set(k, entry(10));
    c.get(k); // hit
    c.get(k); // hit
    c.get(SliceCache.makeKey('/n', 'other')); // miss
    const s = c.getStats();
    expect(s.hits).toBe(2);
    expect(s.misses).toBe(1);
    expect(s.count).toBe(1);
    expect(s.hitRate).toBeCloseTo(2 / 3, 5);
  });

  it('evicts least-recently-used entries to stay within the byte budget', () => {
    // Budget fits ~2 entries of 400 bytes; a 3rd forces an eviction.
    const c = new SliceCache({ maxSize: 1000 });
    const a = SliceCache.makeKey('/n', 'a');
    const b = SliceCache.makeKey('/n', 'b');
    const d = SliceCache.makeKey('/n', 'd');
    c.set(a, entry(400, 'a'));
    c.set(b, entry(400, 'b'));
    c.get(a); // touch 'a' so 'b' becomes LRU
    c.set(d, entry(400, 'd')); // 1200 > 1000 → evict LRU ('b')
    expect(c.has(a)).toBe(true);
    expect(c.has(d)).toBe(true);
    expect(c.has(b)).toBe(false);
    expect(c.getStats().evictions).toBe(1);
    expect(c.getStats().size).toBeLessThanOrEqual(1000);
  });

  it('rejects an entry larger than the whole budget without corrupting state', () => {
    const c = new SliceCache({ maxSize: 100 });
    c.set(SliceCache.makeKey('/n', 'big'), entry(200));
    expect(c.has(SliceCache.makeKey('/n', 'big'))).toBe(false);
    expect(c.getStats().count).toBe(0);
    expect(c.getStats().size).toBe(0);
  });

  it('clear() empties the cache and resets counters', () => {
    const c = new SliceCache({ maxSize: 1024 });
    const k = SliceCache.makeKey('/n', 'v');
    c.set(k, entry(10));
    c.get(k);
    c.clear();
    expect(c.has(k)).toBe(false);
    const s = c.getStats();
    expect(s.count).toBe(0);
    expect(s.size).toBe(0);
    expect(s.hits).toBe(0);
    expect(s.misses).toBe(0);
  });

  it('makeKey namespaces by node path so identical views on different nodes never collide', () => {
    const c = new SliceCache({ maxSize: 1024 });
    const k1 = SliceCache.makeKey('/a', 'view');
    const k2 = SliceCache.makeKey('/b', 'view');
    expect(k1).not.toBe(k2);
    c.set(k1, entry(10, 'a'));
    c.set(k2, entry(10, 'b'));
    expect((c.get(k1)!.payload as { tag: string }).tag).toBe('a');
    expect((c.get(k2)!.payload as { tag: string }).tag).toBe('b');
  });
});
