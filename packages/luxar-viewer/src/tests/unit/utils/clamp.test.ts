/**
 * Unit tests for `utils/clamp.ts`.
 *
 * Closes audit finding utils.md G1 — the function had no dedicated test
 * file. `clamp()` is a leaf primitive used across rendering, controls, and
 * UI layers; the contract is small but the boundary cases are easy to
 * miss (NaN, ±Infinity, min===max, the "either bound omitted" branches).
 *
 * Property tests cover the algebraic invariants (idempotence,
 * monotonicity) that map directly onto the audit's H4 candidate.
 */

import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { clamp } from '../../../utils/clamp';

describe('clamp — both bounds present', () => {
  it('returns value unchanged when min <= value <= max', () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(0, 0, 10)).toBe(0);
    expect(clamp(10, 0, 10)).toBe(10);
  });

  it('returns min when value < min', () => {
    expect(clamp(-1, 0, 10)).toBe(0);
    expect(clamp(-1e6, 0, 10)).toBe(0);
  });

  it('returns max when value > max', () => {
    expect(clamp(11, 0, 10)).toBe(10);
    expect(clamp(1e6, 0, 10)).toBe(10);
  });

  it('handles min === max (single fixed point)', () => {
    expect(clamp(-5, 3, 3)).toBe(3);
    expect(clamp(3, 3, 3)).toBe(3);
    expect(clamp(100, 3, 3)).toBe(3);
  });

  it('with min > max, the source applies min first then max (clamps to max)', () => {
    // Pin current behavior: the body is sequential `if (value < min) return min;
    // if (value > max) return max;`. With min=10, max=5: an in-between
    // value like 7 fails value<10 (returns 10), then since 10>5... it
    // returned already. With value=8: value<10 ⇒ returns 10. So min wins
    // for in-range, but a value already > max but < min still returns
    // min. Pin precisely so future refactors of the inverted-bounds
    // semantics are intentional.
    expect(clamp(7, 10, 5)).toBe(10);
    expect(clamp(0, 10, 5)).toBe(10);
    expect(clamp(20, 10, 5)).toBe(5);
  });
});

describe('clamp — one bound omitted', () => {
  it('floors only when max is undefined', () => {
    expect(clamp(-1, 0)).toBe(0);
    expect(clamp(5, 0)).toBe(5);
    expect(clamp(1e9, 0)).toBe(1e9);
  });

  it('ceils only when min is undefined', () => {
    expect(clamp(11, undefined, 10)).toBe(10);
    expect(clamp(5, undefined, 10)).toBe(5);
    expect(clamp(-1e9, undefined, 10)).toBe(-1e9);
  });

  it('passes value through when both bounds are undefined', () => {
    expect(clamp(42)).toBe(42);
    expect(clamp(-7)).toBe(-7);
    expect(clamp(0)).toBe(0);
  });
});

describe('clamp — non-finite values', () => {
  it('clamps +Infinity to max when max is finite', () => {
    expect(clamp(Infinity, 0, 10)).toBe(10);
  });

  it('clamps -Infinity to min when min is finite', () => {
    expect(clamp(-Infinity, 0, 10)).toBe(0);
  });

  it('passes Infinity through when matching bound is absent', () => {
    expect(clamp(Infinity, 0)).toBe(Infinity);
    expect(clamp(-Infinity, undefined, 10)).toBe(-Infinity);
  });

  it('passes NaN through (NaN < x and NaN > x are both false)', () => {
    // Pin defensive: NaN compares false against every bound, so it
    // bypasses both branches. A future hardening could decide to coerce
    // NaN to min, max, or throw — flipping this expectation is then
    // intentional.
    expect(Number.isNaN(clamp(NaN, 0, 10))).toBe(true);
    expect(Number.isNaN(clamp(NaN, 0))).toBe(true);
    expect(Number.isNaN(clamp(NaN))).toBe(true);
  });
});

describe('clamp — property invariants (utils.md H4)', () => {
  it('result is always within [min, max] when both bounds are finite and min <= max', () => {
    fc.assert(
      fc.property(
        fc.double({ noNaN: true, noDefaultInfinity: true, min: -1e9, max: 1e9 }),
        fc.double({ noNaN: true, noDefaultInfinity: true, min: -1e9, max: 1e9 }),
        fc.double({ noNaN: true, noDefaultInfinity: true, min: -1e9, max: 1e9 }),
        (rawValue, a, b) => {
          const min = Math.min(a, b);
          const max = Math.max(a, b);
          const out = clamp(rawValue, min, max);
          expect(out).toBeGreaterThanOrEqual(min);
          expect(out).toBeLessThanOrEqual(max);
        }
      ),
      { numRuns: 200 }
    );
  });

  it('is idempotent: clamp(clamp(x, lo, hi), lo, hi) === clamp(x, lo, hi)', () => {
    fc.assert(
      fc.property(
        fc.double({ noNaN: true, noDefaultInfinity: true, min: -1e9, max: 1e9 }),
        fc.double({ noNaN: true, noDefaultInfinity: true, min: -1e9, max: 1e9 }),
        fc.double({ noNaN: true, noDefaultInfinity: true, min: -1e9, max: 1e9 }),
        (rawValue, a, b) => {
          const min = Math.min(a, b);
          const max = Math.max(a, b);
          const once = clamp(rawValue, min, max);
          const twice = clamp(once, min, max);
          expect(twice).toBe(once);
        }
      ),
      { numRuns: 200 }
    );
  });

  it('is monotonic: x <= y ⇒ clamp(x, lo, hi) <= clamp(y, lo, hi)', () => {
    fc.assert(
      fc.property(
        fc.double({ noNaN: true, noDefaultInfinity: true, min: -1e9, max: 1e9 }),
        fc.double({ noNaN: true, noDefaultInfinity: true, min: 0, max: 1e9 }),
        fc.double({ noNaN: true, noDefaultInfinity: true, min: -1e9, max: 1e9 }),
        fc.double({ noNaN: true, noDefaultInfinity: true, min: -1e9, max: 1e9 }),
        (x, delta, a, b) => {
          const y = x + delta; // delta >= 0 so y >= x
          const min = Math.min(a, b);
          const max = Math.max(a, b);
          expect(clamp(x, min, max)).toBeLessThanOrEqual(clamp(y, min, max));
        }
      ),
      { numRuns: 200 }
    );
  });
});
