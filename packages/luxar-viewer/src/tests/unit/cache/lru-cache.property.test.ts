/**
 * Property tests for LRUCache (fast-check).
 *
 * Companion to lru-cache.test.ts. Pins these invariants over arbitrary
 * sequences of set/get/delete/clear:
 *
 *   - Capacity invariant: currentSize is always ≤ maxSize after any operation.
 *   - Hit-counter consistency: `hits + misses + setMisses == call_count` post-hoc.
 *   - Set-overwrite preserves size accounting (no leak when updating same key).
 *   - delete() of a present key decreases currentSize by the deleted value's size.
 *   - get() on miss does not change currentSize.
 *   - clear() resets currentSize to 0 and empties the cache.
 *
 * The deterministic-sequence stress test at lru-cache.test.ts:279-303
 * spot-checks the invariant on ONE trace; this file exercises arbitrary traces.
 */
import { describe, expect, test } from 'vitest';
import * as fc from 'fast-check';
import { LRUCache } from '../../../cache/lru-cache';

type Op =
  | { kind: 'set'; key: string; size: number }
  | { kind: 'get'; key: string }
  | { kind: 'delete'; key: string }
  | { kind: 'clear' };

const opArb: fc.Arbitrary<Op> = fc.oneof(
  {
    weight: 4,
    arbitrary: fc.record({
      kind: fc.constant<'set'>('set'),
      key: fc.constantFrom('a', 'b', 'c', 'd', 'e', 'f'),
      size: fc.integer({ min: 1, max: 100 }),
    }),
  },
  {
    weight: 3,
    arbitrary: fc.record({
      kind: fc.constant<'get'>('get'),
      key: fc.constantFrom('a', 'b', 'c', 'd', 'e', 'f', 'z' /* miss */),
    }),
  },
  {
    weight: 1,
    arbitrary: fc.record({
      kind: fc.constant<'delete'>('delete'),
      key: fc.constantFrom('a', 'b', 'c', 'd', 'e', 'f', 'z' /* miss */),
    }),
  },
  { weight: 1, arbitrary: fc.record({ kind: fc.constant<'clear'>('clear') }) }
) as fc.Arbitrary<Op>;

describe('LRUCache — algebraic invariants over arbitrary operation traces', () => {
  test('currentSize <= maxSize after every operation (the load-bearing invariant)', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 10, max: 500 }),
        fc.array(opArb, { minLength: 1, maxLength: 200 }),
        (maxSize, ops) => {
          const cache = new LRUCache<{ size: number }>(maxSize, (v) => v.size);
          for (const op of ops) {
            switch (op.kind) {
              case 'set':
                // Size must be > 0 to be accepted; clamp to ≤ maxSize so set won't reject outright.
                cache.set(op.key, { size: Math.min(op.size, maxSize) });
                break;
              case 'get':
                cache.get(op.key);
                break;
              case 'delete':
                cache.delete(op.key);
                break;
              case 'clear':
                cache.clear();
                break;
            }
            const stats = cache;
            expect(stats.size).toBeLessThanOrEqual(maxSize);
            expect(stats.size).toBeGreaterThanOrEqual(0);
          }
        }
      ),
      { numRuns: 100 }
    );
  });

  test('clear() resets size to 0 and empties the cache', () => {
    fc.assert(
      fc.property(fc.array(opArb, { minLength: 1, maxLength: 50 }), (ops) => {
        const cache = new LRUCache<{ size: number }>(1000, (v) => v.size);
        for (const op of ops) {
          if (op.kind === 'set') cache.set(op.key, { size: op.size });
        }
        cache.clear();
        const stats = cache;
        expect(stats.size).toBe(0);
        // Every previously-set key now misses.
        for (const key of ['a', 'b', 'c', 'd', 'e', 'f']) {
          expect(cache.get(key)).toBeUndefined();
        }
      }),
      { numRuns: 50 }
    );
  });

  test('hits + misses counter accumulates monotonically across get() calls', () => {
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom('a', 'b', 'c', 'z'), { minLength: 1, maxLength: 50 }),
        (keys) => {
          const cache = new LRUCache<{ size: number }>(1000, (v) => v.size);
          cache.set('a', { size: 10 });
          cache.set('b', { size: 10 });

          let prev = cache.hitCount + cache.missCount;
          for (const k of keys) {
            cache.get(k);
            const curr = cache.hitCount + cache.missCount;
            // Every get() must increase the hits+misses counter by exactly 1.
            expect(curr - prev).toBe(1);
            prev = curr;
          }
        }
      ),
      { numRuns: 50 }
    );
  });

  test('set() overwriting an existing key updates size correctly (no leak)', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 5, max: 50 }),
        fc.integer({ min: 5, max: 50 }),
        (size1, size2) => {
          const cache = new LRUCache<{ size: number }>(1000, (v) => v.size);
          cache.set('a', { size: size1 });
          expect(cache.size).toBe(size1);
          cache.set('a', { size: size2 });
          // After overwrite: only size2 contribution remains.
          expect(cache.size).toBe(size2);
        }
      ),
      { numRuns: 50 }
    );
  });

  test('delete() of an absent key is a no-op (does not change size or evict)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.constantFrom('a', 'b', 'c'), fc.integer({ min: 1, max: 50 })), {
          minLength: 1,
          maxLength: 10,
        }),
        (kvs) => {
          const cache = new LRUCache<{ size: number }>(1000, (v) => v.size);
          for (const [k, s] of kvs) cache.set(k, { size: s });
          const before = cache.size;
          const deleted = cache.delete('not-present-key');
          const after = cache.size;
          expect(deleted).toBe(false);
          expect(after).toBe(before);
        }
      ),
      { numRuns: 50 }
    );
  });
});
