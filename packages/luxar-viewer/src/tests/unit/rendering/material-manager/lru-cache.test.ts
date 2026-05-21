/**
 * Direct unit tests for the generic material-manager LRU helpers.
 *
 * The orchestrator-level tests (`material-cache-lru.test.ts`,
 * `material-manager.test.ts`) exercise these through real Point/Line/
 * GSplat materials; this file pins them as pure Map-helpers with no
 * THREE coupling.
 */

import { describe, it, expect, vi } from 'vitest';
import { lruGet, lruSet } from '../../../../rendering/material-manager/lru-cache';

interface Box {
  readonly id: string;
}
const box = (id: string): Box => ({ id });

describe('lruGet', () => {
  it('returns undefined on a miss', () => {
    const cache = new Map<string, Box>();
    expect(lruGet(cache, 'absent')).toBeUndefined();
  });

  it('returns the cached value on a hit', () => {
    const cache = new Map<string, Box>();
    const value = box('a');
    cache.set('a', value);
    expect(lruGet(cache, 'a')).toBe(value);
  });

  it('promotes the hit entry to most-recently-used by delete+reinsert', () => {
    // Map preserves insertion order — after `lruGet('a')`, 'a' should be
    // at the *end* of the iteration order (most recent), not the start.
    const cache = new Map<string, Box>([
      ['a', box('a')],
      ['b', box('b')],
      ['c', box('c')],
    ]);

    lruGet(cache, 'a');

    expect([...cache.keys()]).toEqual(['b', 'c', 'a']);
  });

  it('miss does NOT mutate insertion order', () => {
    const cache = new Map<string, Box>([
      ['a', box('a')],
      ['b', box('b')],
    ]);

    lruGet(cache, 'missing');

    expect([...cache.keys()]).toEqual(['a', 'b']);
  });
});

describe('lruSet', () => {
  it('inserts a new value when under maxSize without invoking onEvict', () => {
    const cache = new Map<string, Box>();
    const onEvict = vi.fn();

    lruSet(cache, 'a', box('a'), 3, onEvict);
    lruSet(cache, 'b', box('b'), 3, onEvict);

    expect(cache.size).toBe(2);
    expect(cache.has('a')).toBe(true);
    expect(cache.has('b')).toBe(true);
    expect(onEvict).not.toHaveBeenCalled();
  });

  it('evicts the LRU entry (first inserted) when at the bound', () => {
    const cache = new Map<string, Box>();
    const evicted: Array<[string, Box]> = [];
    const onEvict = (k: string, v: Box): void => {
      evicted.push([k, v]);
    };

    lruSet(cache, 'a', box('a'), 2, onEvict);
    lruSet(cache, 'b', box('b'), 2, onEvict);
    // Cache now at bound. Next insert evicts 'a' (LRU).
    lruSet(cache, 'c', box('c'), 2, onEvict);

    expect([...cache.keys()]).toEqual(['b', 'c']);
    expect(evicted.length).toBe(1);
    expect(evicted[0][0]).toBe('a');
    expect(evicted[0][1].id).toBe('a');
  });

  it('chain-evicts when the cache is over the bound by more than one entry', () => {
    // Pre-fill with 4 entries, then drop maxSize to 2 and insert a 5th.
    // The while-loop should chain-evict 3 entries (4 + 1 -> evict 3 -> 2).
    const cache = new Map<string, Box>([
      ['a', box('a')],
      ['b', box('b')],
      ['c', box('c')],
      ['d', box('d')],
    ]);
    const evicted: string[] = [];

    lruSet(cache, 'e', box('e'), 2, (k) => evicted.push(k));

    expect([...cache.keys()]).toEqual(['d', 'e']);
    expect(evicted).toEqual(['a', 'b', 'c']);
  });

  it('maxSize === 0 disables eviction (unbounded growth)', () => {
    const cache = new Map<string, Box>();
    const onEvict = vi.fn();

    for (let i = 0; i < 100; i++) {
      lruSet(cache, `k${i}`, box(`k${i}`), 0, onEvict);
    }

    expect(cache.size).toBe(100);
    expect(onEvict).not.toHaveBeenCalled();
  });

  it('re-setting an existing key overwrites the value but does NOT promote position', () => {
    // The orchestrator's call pattern is lruGet-then-lruSet-on-miss, so
    // lruSet on an existing key is rare. JS Map.set on an existing key
    // keeps insertion order — lruSet inherits that behaviour. Document
    // it explicitly so a future "let's also promote on overwrite" tweak
    // updates this test deliberately.
    const cache = new Map<string, Box>([
      ['a', box('a-old')],
      ['b', box('b')],
    ]);
    const onEvict = vi.fn();

    lruSet(cache, 'a', box('a-new'), 10, onEvict);

    expect([...cache.keys()]).toEqual(['a', 'b']);
    expect(cache.get('a')!.id).toBe('a-new');
    expect(onEvict).not.toHaveBeenCalled();
  });

  it('passes the evicted (key, value) pair correctly to onEvict', () => {
    const cache = new Map<string, Box>();
    const a = box('a');
    cache.set('a', a);
    const onEvict = vi.fn();

    lruSet(cache, 'b', box('b'), 1, onEvict);

    expect(onEvict).toHaveBeenCalledTimes(1);
    expect(onEvict).toHaveBeenCalledWith('a', a);
  });

  it('handles the degenerate maxSize === 1 case correctly', () => {
    const cache = new Map<string, Box>();
    const onEvict = vi.fn();

    lruSet(cache, 'a', box('a'), 1, onEvict);
    expect(cache.size).toBe(1);
    expect(onEvict).not.toHaveBeenCalled();

    lruSet(cache, 'b', box('b'), 1, onEvict);
    expect([...cache.keys()]).toEqual(['b']);
    expect(onEvict).toHaveBeenCalledWith('a', expect.objectContaining({ id: 'a' }));
  });

  it('does not call onEvict for an evicted key whose value is undefined (safety guard)', () => {
    // Synthesize a Map that returns undefined for a tracked key (the
    // helper guards `if (lruValue !== undefined)`). Use sparse insert
    // to simulate: pre-set a key then delete its value through a custom
    // Map subclass that returns undefined despite `has`.
    //
    // Easier: just check that the standard onEvict path is gated by the
    // value-undefined branch by inserting `undefined`-typed entries.
    const cache = new Map<string, Box | undefined>();
    cache.set('a', undefined);
    const onEvict = vi.fn();

    lruSet(cache, 'b', box('b'), 1, onEvict);

    // 'a' is evicted from the Map but its value was undefined → no callback
    expect([...cache.keys()]).toEqual(['b']);
    expect(onEvict).not.toHaveBeenCalled();
  });
});
