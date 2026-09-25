/**
 * Unit tests for the slider math helpers.
 *
 * Pure math: no DOM, no THREE. These helpers are extracted from
 * dimension-sliders.ts so the cyclic-wrap / value↔fraction / thumb-position
 * logic can be tested without spinning up a viewer.
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { clampWithCyclicWrap, clampInteger } from '../../../ui/dimension-sliders/slider-math';
import { fractionToThumbLeft, fractionToValue, valueToFraction } from '../../../ui/slider-kit';

describe('clampWithCyclicWrap', () => {
  it('returns the input unchanged when in range', () => {
    expect(clampWithCyclicWrap(5, 0, 10, false)).toBe(5);
    expect(clampWithCyclicWrap(5, 0, 10, true)).toBe(5);
  });

  it('returns the boundary when on it (both endpoints inclusive)', () => {
    expect(clampWithCyclicWrap(0, 0, 10, false)).toBe(0);
    expect(clampWithCyclicWrap(10, 0, 10, false)).toBe(10);
    expect(clampWithCyclicWrap(0, 0, 10, true)).toBe(0);
    expect(clampWithCyclicWrap(10, 0, 10, true)).toBe(10);
  });

  it('clamps under-min to min when not cyclic', () => {
    expect(clampWithCyclicWrap(-5, 0, 10, false)).toBe(0);
    expect(clampWithCyclicWrap(-1, 3, 7, false)).toBe(3);
  });

  it('clamps over-max to max when not cyclic', () => {
    expect(clampWithCyclicWrap(15, 0, 10, false)).toBe(10);
    expect(clampWithCyclicWrap(99, 3, 7, false)).toBe(7);
  });

  it('wraps under-min to max when cyclic', () => {
    expect(clampWithCyclicWrap(-1, 0, 10, true)).toBe(10);
    expect(clampWithCyclicWrap(-100, 0, 10, true)).toBe(10);
  });

  it('wraps over-max to min when cyclic', () => {
    expect(clampWithCyclicWrap(11, 0, 10, true)).toBe(0);
    expect(clampWithCyclicWrap(50, 3, 7, true)).toBe(3);
  });

  it('handles negative ranges', () => {
    expect(clampWithCyclicWrap(-15, -10, -5, false)).toBe(-10);
    expect(clampWithCyclicWrap(-15, -10, -5, true)).toBe(-5);
    expect(clampWithCyclicWrap(0, -10, -5, false)).toBe(-5);
    expect(clampWithCyclicWrap(0, -10, -5, true)).toBe(-10);
  });

  it('treats a single-value range as fixed', () => {
    expect(clampWithCyclicWrap(5, 5, 5, false)).toBe(5);
    expect(clampWithCyclicWrap(5, 5, 5, true)).toBe(5);
    expect(clampWithCyclicWrap(4, 5, 5, false)).toBe(5);
    expect(clampWithCyclicWrap(6, 5, 5, false)).toBe(5);
    // Cyclic on a degenerate range: still resolves into the single value.
    expect(clampWithCyclicWrap(4, 5, 5, true)).toBe(5);
  });
});

describe('valueToFraction', () => {
  it('maps min → 0', () => {
    expect(valueToFraction(0, 0, 100)).toBe(0);
    expect(valueToFraction(-10, -10, 10)).toBe(0);
  });

  it('maps max → 1', () => {
    expect(valueToFraction(100, 0, 100)).toBe(1);
    expect(valueToFraction(10, -10, 10)).toBe(1);
  });

  it('maps mid → 0.5', () => {
    expect(valueToFraction(50, 0, 100)).toBe(0.5);
    expect(valueToFraction(0, -10, 10)).toBe(0.5);
  });

  it('returns 0.5 for a degenerate range (min === max)', () => {
    expect(valueToFraction(0, 0, 0)).toBe(0.5);
    expect(valueToFraction(5, 5, 5)).toBe(0.5);
  });

  it('handles non-uniform ranges with arbitrary precision', () => {
    expect(valueToFraction(2.5, 0, 10)).toBe(0.25);
    expect(valueToFraction(7.5, 0, 10)).toBe(0.75);
  });
});

describe('fractionToValue', () => {
  it('maps 0 → min and 1 → max', () => {
    expect(fractionToValue(0, 0, 100)).toBe(0);
    expect(fractionToValue(1, 0, 100)).toBe(100);
    expect(fractionToValue(0, -10, 10)).toBe(-10);
    expect(fractionToValue(1, -10, 10)).toBe(10);
  });

  it('maps 0.5 to the midpoint', () => {
    expect(fractionToValue(0.5, 0, 100)).toBe(50);
    expect(fractionToValue(0.5, -10, 10)).toBe(0);
  });

  it('returns min for a degenerate range regardless of fraction', () => {
    expect(fractionToValue(0, 5, 5)).toBe(5);
    expect(fractionToValue(0.5, 5, 5)).toBe(5);
    expect(fractionToValue(1, 5, 5)).toBe(5);
  });

  it('round-trips with valueToFraction', () => {
    for (const v of [-3, 0, 4.2, 7.7, 10]) {
      const back = fractionToValue(valueToFraction(v, 0, 10), 0, 10);
      expect(back).toBeCloseTo(v, 10);
    }
  });
});

describe('fractionToThumbLeft', () => {
  it('places the thumb at 0 for fraction = 0', () => {
    expect(fractionToThumbLeft(0, 300, 16)).toBe(0);
  });

  it('places the thumb so its right edge meets the container at fraction = 1', () => {
    // left + thumbWidth should equal containerWidth.
    expect(fractionToThumbLeft(1, 300, 16)).toBe(284);
  });

  it('scales linearly with fraction', () => {
    const left = fractionToThumbLeft(0.5, 300, 16);
    expect(left).toBeCloseTo(142, 5);
  });

  it('handles a thumb wider than the container without going negative', () => {
    // Defensive: math allows negative left here. Documented as caller's
    // responsibility, but verify the math still produces the reported value.
    // (`toBeCloseTo` because `0 * -6` returns -0 which `toBe(0)` rejects.)
    expect(fractionToThumbLeft(0, 10, 16)).toBeCloseTo(0, 10);
    expect(fractionToThumbLeft(1, 10, 16)).toBe(-6);
  });
});

describe('clampInteger', () => {
  it('returns the value when in range', () => {
    expect(clampInteger(500, 0, 1000)).toBe(500);
  });

  it('clamps to lo when below', () => {
    expect(clampInteger(-5, 0, 1000)).toBe(0);
  });

  it('clamps to hi when above', () => {
    expect(clampInteger(2000, 0, 1000)).toBe(1000);
  });

  it('handles negative ranges', () => {
    expect(clampInteger(-50, -100, -10)).toBe(-50);
    expect(clampInteger(-200, -100, -10)).toBe(-100);
    expect(clampInteger(0, -100, -10)).toBe(-10);
  });

  it('handles a single-value range', () => {
    expect(clampInteger(5, 7, 7)).toBe(7);
    expect(clampInteger(8, 7, 7)).toBe(7);
  });
});

// ============================================================================
// [ui.md/H2][P12] Property tests for valueToFraction <-> fractionToValue
//
// These two functions form an inverse pair on (min, max) with max > min.
// The round-trip property is documented and partially tested (5 cases at
// line 114-119); a property test extends coverage across the real number
// line and catches mutations that would survive a fixed-example test
// (e.g. flipping a sign, swapping min/max, off-by-one on the divisor).
// ============================================================================

describe('valueToFraction <-> fractionToValue (property tests)', () => {
  it('[H2] fractionToValue(valueToFraction(v, lo, hi), lo, hi) == v for v in [lo, hi]', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -1e6, max: 1e6, noNaN: true, noDefaultInfinity: true }),
        fc.double({ min: 1e-9, max: 1e6, noNaN: true, noDefaultInfinity: true }),
        fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
        (lo, span, t) => {
          // Build (lo, hi) with non-zero span; pick v inside the span via t in [0,1].
          const hi = lo + span;
          const v = lo + t * span;
          const back = fractionToValue(valueToFraction(v, lo, hi), lo, hi);
          // Pathological-cancellation regime (lo≈1e6, span≈1e-9) eats up
          // most of the float64 mantissa during `(v - lo) / span`. Use a
          // relative-or-absolute tolerance that scales with max(|lo|, |hi|);
          // ~10 ULPs(1) gives 8 decimal places of headroom while still
          // catching real algorithmic regressions (sign flip, factor-of-2,
          // off-by-one in lerp would all drift by >> 1e-8 * |scale|).
          const scale = Math.max(1, Math.abs(lo), Math.abs(hi));
          expect(Math.abs(back - v)).toBeLessThanOrEqual(1e-8 * scale);
        }
      ),
      { numRuns: 80 }
    );
  });

  it('[H2] valueToFraction(fractionToValue(f, lo, hi), lo, hi) == f for f in [0, 1]', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -1e6, max: 1e6, noNaN: true, noDefaultInfinity: true }),
        fc.double({ min: 1e-9, max: 1e6, noNaN: true, noDefaultInfinity: true }),
        fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
        (lo, span, f) => {
          // Filter pathological cancellation regimes BEFORE the assertion.
          // History of this assertion's drift:
          //   - toBeCloseTo(_, 9) → 5e-10 strict ... fast-check found it
          //   - 5e-9 absolute ... 5.0000000000004e-9 (1 ULP over)
          //   - 1e-8 absolute ... 1.0000000000000116e-8 (1 ULP over)
          //   - toBeCloseTo(_, 7) [strict <5e-8] ... 5.0000000000000037e-8
          //   - toBeCloseTo(_, 6) [strict <5e-7] ... 5e-7 exactly (PR #145)
          //
          // The pattern is structural: when `Math.abs(span)` is many
          // orders of magnitude smaller than `Math.max(|lo|, |hi|)`,
          // `(lo + f*span) - lo` suffers catastrophic cancellation —
          // most of the float64 mantissa is lost. The composition
          // `valueToFraction(fractionToValue(f, lo, hi), lo, hi)`
          // accumulates this error proportional to `1 / span` after
          // dividing. Loosening tolerance just lets fast-check shrink
          // to the next boundary. The right fix is to require the
          // input span to retain enough significant digits.
          //
          // 10 ULPs of float64 relative precision (~2.22e-15 × 10) gives
          // a safety floor of 1e-10. We require span / max(|lo|, |hi|)
          // ≥ 1e-10 so the cancellation can't eat more than ~6 digits
          // of significance. Below that floor the test SKIPS via
          // fc.pre — fast-check generates a different sample.
          const magnitude = Math.max(1, Math.abs(lo), Math.abs(lo + span));
          fc.pre(span / magnitude >= 1e-10);
          const hi = lo + span;
          const back = valueToFraction(fractionToValue(f, lo, hi), lo, hi);
          // f ∈ [0, 1]; absolute tolerance is what matters. Precision 6
          // (strict <5e-7) kills every meaningful algorithmic regression
          // (sign flip, factor-of-2, off-by-one all produce O(0.01)
          // error, ~5 orders of magnitude above 5e-7).
          expect(back).toBeCloseTo(f, 6);
        }
      ),
      { numRuns: 80 }
    );
  });

  it('[H2] valueToFraction is monotone non-decreasing in v on [lo, hi]', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -1e3, max: 1e3, noNaN: true, noDefaultInfinity: true }),
        fc.double({ min: 1e-6, max: 1e3, noNaN: true, noDefaultInfinity: true }),
        fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
        fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
        (lo, span, t1, t2) => {
          const hi = lo + span;
          const v1 = lo + Math.min(t1, t2) * span;
          const v2 = lo + Math.max(t1, t2) * span;
          const f1 = valueToFraction(v1, lo, hi);
          const f2 = valueToFraction(v2, lo, hi);
          expect(f2).toBeGreaterThanOrEqual(f1 - 1e-12);
        }
      ),
      { numRuns: 60 }
    );
  });
});

// ============================================================================
// [R11/D-H1+D-H2][P12] Property tests for fractionToThumbLeft and clampInteger.
//
// fractionToThumbLeft is the linear interpolation
//   left = fraction * (containerWidth - thumbWidth)
// so on a positive (containerWidth > thumbWidth) configuration it must be
// monotone non-decreasing in `fraction` and satisfy the endpoint identities
// (f=0 → 0; f=1 → containerWidth - thumbWidth). A mutation that swapped the
// multiplication for division, or flipped the sign, would survive the fixed
// example cases above (which only sample 0 / 0.5 / 1) but would fail this
// property test across the full fraction range.
//
// clampInteger is the canonical clamp; the property test pins the range
// invariant lo ≤ result ≤ hi over arbitrary inputs and the identity-on-
// in-range case. A mutation that flipped `>` to `≥` in the upper bound
// could survive fixed cases at the boundaries but fail random samples.
// ============================================================================

describe('fractionToThumbLeft (property tests)', () => {
  it('[D-H1] is monotone non-decreasing in fraction when containerWidth ≥ thumbWidth', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
        fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
        fc.double({ min: 1, max: 2_000, noNaN: true, noDefaultInfinity: true }),
        fc.double({ min: 0, max: 100, noNaN: true, noDefaultInfinity: true }),
        (f1, f2, containerWidth, thumbWidth) => {
          // Constrain to the documented "thumb fits in container" regime.
          fc.pre(thumbWidth <= containerWidth);
          const lo = Math.min(f1, f2);
          const hi = Math.max(f1, f2);
          const l1 = fractionToThumbLeft(lo, containerWidth, thumbWidth);
          const l2 = fractionToThumbLeft(hi, containerWidth, thumbWidth);
          expect(l2).toBeGreaterThanOrEqual(l1 - 1e-9);
        }
      ),
      { numRuns: 60 }
    );
  });

  it('[D-H1] satisfies the endpoint identities f=0 → 0, f=1 → containerWidth - thumbWidth', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 1, max: 2_000, noNaN: true, noDefaultInfinity: true }),
        fc.double({ min: 0, max: 2_000, noNaN: true, noDefaultInfinity: true }),
        (containerWidth, thumbWidth) => {
          // f=0 always lands at 0, regardless of widths.
          expect(fractionToThumbLeft(0, containerWidth, thumbWidth)).toBeCloseTo(0, 9);
          // f=1 lands at exactly containerWidth - thumbWidth (may be ≤ 0
          // if thumb is wider; that's the documented caller responsibility).
          expect(fractionToThumbLeft(1, containerWidth, thumbWidth)).toBeCloseTo(
            containerWidth - thumbWidth,
            9
          );
        }
      ),
      { numRuns: 80 }
    );
  });
});

describe('clampInteger (property tests)', () => {
  it('[D-H2] result always satisfies lo ≤ result ≤ hi', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: -1_000_000, max: 1_000_000 }),
        fc.integer({ min: -1_000_000, max: 1_000_000 }),
        fc.integer({ min: 0, max: 2_000_000 }),
        (v, lo, span) => {
          const hi = lo + span;
          const r = clampInteger(v, lo, hi);
          expect(r).toBeGreaterThanOrEqual(lo);
          expect(r).toBeLessThanOrEqual(hi);
        }
      ),
      { numRuns: 100 }
    );
  });

  it('[D-H2] identity-on-in-range: v ∈ [lo, hi] ⇒ result = v', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: -1_000, max: 1_000 }),
        fc.integer({ min: 0, max: 2_000 }),
        fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
        (lo, span, t) => {
          const hi = lo + span;
          // pick v ∈ [lo, hi] by interpolation, then round to int
          const v = Math.round(lo + t * span);
          expect(clampInteger(v, lo, hi)).toBe(v);
        }
      ),
      { numRuns: 100 }
    );
  });
});
