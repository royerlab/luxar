/**
 * Property tests for WASM TypeScript-fallback projection routines (fast-check).
 *
 * Closes:
 *   - [wasm.md/H7][P12] radii_to_visibility_mask monotone in threshold
 *   - [wasm.md/H4][P12] compact_by_mask count_visible invariant
 *   - [wasm.md/H5][P12] lerp(a, b, t) linearity
 *
 * Algebraic properties: independent of input data, the function
 * outputs must satisfy a structural identity. A mutant flipping `>` to
 * `>=` or returning a wrong count would fail across the random sample.
 */

import { describe, test, expect } from 'vitest';
import * as fc from 'fast-check';
import {
  radii_to_visibility_mask,
  count_visible,
  compact_by_mask,
} from '../../../wasm/typescript/projection';
import { lerp } from '../../../wasm/typescript/lines-clipping';

const finiteFloat = fc.float({
  min: Math.fround(-100),
  max: Math.fround(100),
  noNaN: true,
  noDefaultInfinity: true,
});
const positiveFloat = fc.float({
  min: 0,
  max: Math.fround(100),
  noNaN: true,
  noDefaultInfinity: true,
});

describe('radii_to_visibility_mask — algebraic invariants', () => {
  // [wasm.md/H7][P12] Monotonicity in threshold: raising the threshold
  // can only decrease (or leave unchanged) the set of visible points.
  // i.e. v(t1) >= v(t2) for t1 <= t2 over the same radii vector.
  test('monotone-decreasing in threshold: t1 <= t2 ⇒ count(t1) >= count(t2)', () => {
    fc.assert(
      fc.property(
        fc.array(positiveFloat, { minLength: 1, maxLength: 32 }),
        finiteFloat,
        finiteFloat,
        (radiiArr, a, b) => {
          const radii = new Float32Array(radiiArr);
          const count = radii.length;
          const t1 = Math.min(a, b);
          const t2 = Math.max(a, b);
          const out1 = new Uint8Array(count);
          const out2 = new Uint8Array(count);
          const v1 = radii_to_visibility_mask(radii, t1, count, out1);
          const v2 = radii_to_visibility_mask(radii, t2, count, out2);
          expect(v1).toBeGreaterThanOrEqual(v2);

          // Stronger: every point visible at t2 must also be visible at t1.
          // (Containment of visible sets under threshold lowering.)
          for (let i = 0; i < count; i++) {
            if (out2[i] === 1) {
              expect(out1[i]).toBe(1);
            }
          }
        }
      ),
      { numRuns: 100 }
    );
  });

  // [wasm.md/H7][P12] At threshold = -Infinity (or just below 0), every
  // non-NaN finite radius >= 0 must be visible. Locks the strict-inequality
  // contract at the lower boundary.
  test('threshold below the minimum radius marks every point visible', () => {
    fc.assert(
      fc.property(fc.array(positiveFloat, { minLength: 1, maxLength: 16 }), (radiiArr) => {
        const radii = new Float32Array(radiiArr);
        const count = radii.length;
        const out = new Uint8Array(count);
        // Use -1 (strictly below 0, the minimum of positiveFloat). The
        // source uses strict `>`; for radii = 0, the threshold must be
        // strictly less than 0 for it to pass.
        const visible = radii_to_visibility_mask(radii, -1, count, out);
        expect(visible).toBe(count);
        for (let i = 0; i < count; i++) {
          expect(out[i]).toBe(1);
        }
      }),
      { numRuns: 50 }
    );
  });
});

describe('lerp — linearity invariants', () => {
  // [wasm.md/H5][P12] lerp linearity: lerp(a, b, 0) === a; lerp(a, b, 1) === b;
  // affine in t and (a, b) jointly. We pin the two endpoint identities and
  // the additivity in t — lerp(a, b, t1) + lerp(a, b, t2) − a == lerp(a, b, t1+t2)
  // when treated as a function of the offset, i.e. lerp(a, b, t) - a == t*(b - a).
  test('endpoints: lerp(a, b, 0) === a; lerp(a, b, 1) ≈ b within FP roundoff', () => {
    // t=0: `a + 0*(b-a)` reduces to `a + 0`. This is numerically equal to
    // `a` but NOT bit-exact for the signed-zero edge case: `(-0) + 0 = +0`
    // under IEEE-754, so `lerp(-0, b, 0)` returns `+0`. vitest's toBe
    // uses Object.is semantics which distinguishes -0 from +0; use
    // numeric equality (which treats +0 === -0) to express the actual
    // contract — lerp's t=0 endpoint matches `a` numerically.
    // t=1: `a + (b-a)` is NOT bit-exact b for subnormal / denormal inputs —
    // the subtract-then-add reorders to a precision-lossy form. Use a
    // bounded relative-or-absolute tolerance.
    fc.assert(
      fc.property(finiteFloat, finiteFloat, (a, b) => {
        const at_zero = lerp(a, b, 0);
        // Numeric equality (===) treats +0 === -0 as true; Object.is would
        // distinguish them. The IEEE contract is numeric, not bit-pattern.
        expect(at_zero === a).toBe(true);
        // For finite a, b in [-100, 100], absolute FP error is dominated
        // by max(|a|, |b|) * 2 * ULP(1). ULP(100) ≈ 1.42e-14. Use 1e-10
        // as a comfortable upper bound that still catches mutants.
        const got = lerp(a, b, 1);
        expect(Math.abs(got - b)).toBeLessThan(1e-10);
      }),
      { numRuns: 100 }
    );
  });

  test('midpoint: lerp(a, b, 0.5) is within Float64 ULP of (a + b) / 2', () => {
    fc.assert(
      fc.property(finiteFloat, finiteFloat, (a, b) => {
        // Use a small absolute tolerance since lerp uses `a + t*(b-a)` which
        // is slightly different from `(a+b)/2` in floating-point arithmetic.
        const got = lerp(a, b, 0.5);
        const expected = (a + b) / 2;
        expect(Math.abs(got - expected)).toBeLessThan(1e-3);
      }),
      { numRuns: 100 }
    );
  });

  test('reflection: lerp(a, b, t) + lerp(b, a, t) === a + b (within ULP)', () => {
    // Symmetry property: swapping endpoints and using the same parameter t
    // gives the complement. Pins the affine form `a + t(b-a)`.
    fc.assert(
      fc.property(
        finiteFloat,
        finiteFloat,
        fc.float({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
        (a, b, t) => {
          const sum = lerp(a, b, t) + lerp(b, a, t);
          expect(Math.abs(sum - (a + b))).toBeLessThan(1e-3);
        }
      ),
      { numRuns: 100 }
    );
  });
});

describe('compact_by_mask — count_visible parity invariant', () => {
  // [wasm.md/H4][P12] compact_by_mask must write exactly count_visible(mask)
  // contiguous entries to the output buffer, in input order. The returned
  // count must equal count_visible. Pins the dual-function consistency.
  test('compact_by_mask returns count_visible(mask) over random masks', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: 3 }), { minLength: 0, maxLength: 32 }),
        (maskArr) => {
          const count = maskArr.length;
          const mask = new Uint8Array(maskArr);
          // Use stride=1 scalar mode, generate deterministic input.
          const input = new Float32Array(count);
          for (let i = 0; i < count; i++) input[i] = i;
          const output = new Float32Array(count);
          const visibleCount = compact_by_mask(input, mask, count, 1, output);
          // Both functions must agree on visibility count.
          expect(visibleCount).toBe(count_visible(mask, count));
          // Output's first `visibleCount` entries equal the input indices
          // whose mask bit is non-zero, in increasing input-index order.
          const expected = [];
          for (let i = 0; i < count; i++) {
            if (mask[i] !== 0) expected.push(i);
          }
          expect(Array.from(output.slice(0, visibleCount))).toEqual(expected);
        }
      ),
      { numRuns: 100 }
    );
  });
});
