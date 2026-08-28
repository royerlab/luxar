/**
 * Tests for `chooseCapacity` — the pool's allocation-size policy.
 *
 * The headroom cap exists because the factor is charged against a real
 * per-element cost: a Lines geometry is 96 B/segment of RGBA32F element
 * texture plus 8 B/segment of ordering pair, so an unbounded 1.5x costs
 * `cosmicflows_laniakea_full` 567 MiB of slack against a 2000 MiB budget and
 * the scene died with "Array buffer allocation failed". These assertions pin
 * both halves: small nodes keep the full factor, large nodes stop paying it.
 */

import { describe, it, expect } from 'vitest';
import {
  chooseCapacity,
  __setMinInstanceCapacityForTesting,
} from '../../../../rendering/gpu-buffer-pool/capacity';

/** The cap is module-private; this is the count at which it starts biting. */
const HEADROOM_CAP = 262_144;
const CROSSOVER = HEADROOM_CAP * 2; // where ceil(n * 0.5) reaches the cap

describe('chooseCapacity', () => {
  it('never returns less than the requested count', () => {
    for (const n of [0, 1, 2, 255, 256, 1000, 50_000, CROSSOVER, 5_000_000]) {
      expect(chooseCapacity(n)).toBeGreaterThanOrEqual(n);
    }
  });

  it('is monotonically non-decreasing in the requested count', () => {
    let prev = -Infinity;
    for (let n = 0; n <= 4_000_000; n += 9_973) {
      const cap = chooseCapacity(n);
      expect(cap).toBeGreaterThanOrEqual(prev);
      prev = cap;
    }
  });

  it('keeps the full 1.5x for counts below the cap crossover', () => {
    // Unchanged from the pre-cap behaviour — every existing pool test, and
    // every animated demo node, lives in this range.
    for (const n of [1, 2, 3, 1000, 34_000, 100_000, CROSSOVER]) {
      expect(chooseCapacity(n)).toBe(Math.ceil(n * 1.5));
    }
  });

  it('caps the headroom in absolute terms above the crossover', () => {
    for (const n of [CROSSOVER + 1, 1_000_000, 1_631_600, 11_438_031]) {
      expect(chooseCapacity(n)).toBe(n + HEADROOM_CAP);
      // The point of the change: strictly less than the old 1.5x.
      expect(chooseCapacity(n)).toBeLessThan(Math.ceil(n * 1.5));
    }
  });

  it('gives back the measured 333 MiB on the scene that failed', () => {
    // The nine sibling Lines nodes of cosmicflows_laniakea_full, and the
    // real texture geometry: width forced to a multiple of 6 texels, 16 B
    // per RGBA32F texel, plus the 8 B/segment aSortedIndex pair.
    const segments = [
      1_631_600, 1_305_658, 1_319_200, 1_347_835, 1_342_400, 1_323_787, 1_129_545, 1_070_854,
      967_152,
    ];
    const WIDTH = Math.floor(4096 / 6) * 6;
    const bytes = (capacity: number) =>
      WIDTH * Math.max(1, Math.ceil((capacity * 6) / WIDTH)) * 16 + capacity * 8;
    const oldChoose = (n: number) => Math.ceil(n * 1.5);

    const now = segments.reduce((sum, n) => sum + bytes(chooseCapacity(n)), 0);
    const before = segments.reduce((sum, n) => sum + bytes(oldChoose(n)), 0);
    const savedMiB = (before - now) / 2 ** 20;

    expect(before / 2 ** 20).toBeGreaterThan(1700); // 1702 MiB, over an 85%-full 2000 MiB budget
    expect(now / 2 ** 20).toBeLessThan(1400); // 1369 MiB
    expect(savedMiB).toBeGreaterThan(300);
  });

  it('still honours the minimum-capacity floor', () => {
    // `src/tests/setup.ts` pins the floor to 0 suite-wide, so the floor has
    // to be set explicitly here rather than assumed.
    __setMinInstanceCapacityForTesting(256);
    try {
      expect(chooseCapacity(0)).toBe(256);
      expect(chooseCapacity(1)).toBe(256);
      expect(chooseCapacity(170)).toBe(256); // 170 + 85 = 255, floor wins
      expect(chooseCapacity(171)).toBe(257); // 171 + 86 = 257, count wins
    } finally {
      __setMinInstanceCapacityForTesting(0);
    }
    expect(chooseCapacity(0)).toBe(0);
  });
});
