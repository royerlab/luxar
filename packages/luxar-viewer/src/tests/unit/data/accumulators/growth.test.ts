/**
 * Tests for `nextCapacity` — the accumulators' shared growth policy.
 *
 * The old repeated-multiply loop landed on a term of the growth SEQUENCE
 * rather than on the count needed, overshooting by up to the full 1.5x. Six
 * of the nine `cosmicflows_laniakea_full` Lines nodes grew to the identical
 * 1,594,323 vertices for counts between 1.07M and 1.33M — the tell that the
 * figure came from the sequence, not the data.
 */

import { describe, it, expect } from 'vitest';
import { nextCapacity } from '../../../../data/accumulators/growth';

/** The behaviour being replaced, kept here so the tests can contrast it. */
function oldLoop(current: number, needed: number): number {
  let cap = current;
  while (cap < needed) cap = Math.ceil(cap * 1.5);
  return cap;
}

describe('nextCapacity', () => {
  it('returns the current capacity untouched when it already fits', () => {
    expect(nextCapacity(1024, 1024)).toBe(1024);
    expect(nextCapacity(1024, 1)).toBe(1024);
    expect(nextCapacity(1024, 0)).toBe(1024);
  });

  it('always covers the requested count', () => {
    for (const [cur, need] of [
      [8192, 8193],
      [8192, 1_308_987],
      [256, 5_000_000],
      [1, 2],
    ]) {
      expect(nextCapacity(cur, need)).toBeGreaterThanOrEqual(need);
    }
  });

  it('sizes a single large jump EXACTLY to the count needed', () => {
    // The loader knows the span up front, so there is nothing to guess.
    for (const need of [1_635_679, 1_308_987, 1_132_484, 1_073_811, 970_039]) {
      expect(nextCapacity(8192, need)).toBe(need);
      // Mutation guard: the old loop overshot every one of these, and
      // three of them landed on the same wrong number.
      expect(oldLoop(8192, need)).toBeGreaterThan(need);
    }
    expect(new Set([1_132_484, 1_073_811, 1_308_987].map((n) => oldLoop(8192, n))).size).toBe(1);
  });

  it('removes 27.4% of the vertex slots the old loop reserved', () => {
    const verts = [
      1_635_679, 1_308_987, 1_322_498, 1_351_227, 1_345_756, 1_327_105, 1_132_484, 1_073_811,
      970_039,
    ];
    const now = verts.reduce((s, v) => s + nextCapacity(8192, v), 0);
    const before = verts.reduce((s, v) => s + oldLoop(8192, v), 0);
    expect(before).toBe(14_614_628);
    expect(now).toBe(11_467_586);
    expect((before / now - 1) * 100).toBeCloseTo(27.4, 1);
  });

  it('keeps multiplicative growth for incremental fills', () => {
    // A chunk-at-a-time fill must NOT degrade to one reallocation per
    // chunk: while `needed` trails the 1.5x term, the 1.5x term wins.
    let cap = 1024;
    const caps: number[] = [];
    for (let needed = 1025; needed < 20_000; needed += 64) {
      cap = nextCapacity(cap, needed);
      caps.push(cap);
    }
    const growths = new Set(caps).size;
    // ~7 reallocations across ~296 fill calls, not 296.
    expect(growths).toBeLessThan(12);
    expect(cap).toBeGreaterThanOrEqual(19_969);
  });

  it('matches the old loop wherever the old loop was not overshooting', () => {
    // Exact powers of the growth sequence: both land on the same term.
    let term = 8192;
    for (let i = 0; i < 6; i++) {
      term = Math.ceil(term * 1.5);
      expect(nextCapacity(8192, term)).toBe(oldLoop(8192, term));
    }
  });
});
