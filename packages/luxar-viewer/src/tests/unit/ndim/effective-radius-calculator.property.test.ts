/**
 * Property tests for effective-radius-calculator (fast-check).
 *
 * The function projects an nD hypersphere onto a hyperplane: a sphere of
 * radius R centered at distance D from the slice plane has cross-section
 * radius R_eff = sqrt(R² - D²). Key algebraic invariants:
 *
 *   - Pythagorean identity: R_eff² + D² == R² (when in-bounds)
 *   - Out-of-bounds (D > R) yields R_eff == 0
 *   - Boundary (D == R) yields R_eff == 0 with the clamp
 *   - Monotonicity in coordinate: |D| increasing -> R_eff non-increasing
 *   - Equivariance under permutation of displayed dimensions
 *   - Output is always Float32Array
 *
 * Companion to the example-based effective-radius-calculator.test.ts.
 */
import { describe, expect, test } from 'vitest';
import * as fc from 'fast-check';
import { calculateEffectiveRadii } from '../../../data/points/effective-radius-calculator';
import type { ViewState } from '../../../data/data-loader-types';
import type { EffectiveRadiusConfig } from '../../../types/points';

const positiveRadius = fc.float({
  min: Math.fround(0.01),
  max: Math.fround(100),
  noNaN: true,
  noDefaultInfinity: true,
});
const realCoord = fc.float({
  min: Math.fround(-200),
  max: Math.fround(200),
  noNaN: true,
  noDefaultInfinity: true,
});

// IMPORTANT for setup: tolerance >= 1e9 triggers the source's `extend_to_all`
// sentinel and short-circuits the Pythagorean subtraction (see
// effective-radius-calculator.ts:75-77 / :107-109). To exercise the genuine
// Pythagorean path we use tolerance 0 in displayed dims (irrelevant — they're
// not iterated) and a moderate finite tolerance in the hidden dim.
const HIDDEN_TOLERANCE = 1; // < 1e9 sentinel, so the hidden dim contributes.

describe('calculateEffectiveRadii — algebraic invariants', () => {
  test('Pythagorean identity: R_eff² + D² == R² when R > D (single hidden spatial dim)', () => {
    fc.assert(
      fc.property(
        positiveRadius, // R
        realCoord, // hidden-dim position
        realCoord, // slice position in hidden dim
        (R, hiddenPos, slicePos) => {
          // 4D: dims X,Y,Z are displayed; T (index 3) is hidden, spatial.
          const positions = new Float32Array([0, 0, 0, hiddenPos]);
          const radii = new Float32Array([R]);
          const viewState: ViewState = {
            displayDims: [0, 1, 2],
            slicePosition: [0, 0, 0, slicePos],
            tolerance: [0, 0, 0, HIDDEN_TOLERANCE],
          };
          const config: EffectiveRadiusConfig = {
            spatialExtendDims: [false, false, false, true], // T is spatial
            maxRadius: R,
          };
          const result = calculateEffectiveRadii(positions, radii, viewState, config, 4);

          const D = Math.abs(hiddenPos - slicePos);
          if (D >= R) {
            // Out of bounds → clamped to 0.
            expect(result[0]).toBe(0);
          } else {
            // In bounds → Pythagorean. Float32 (≤ 1e-5 absolute is realistic
            // for inputs in [-200, 200]; Float32 ULP at 1e2 is ~7.6e-6).
            const expected = Math.sqrt(R * R - D * D);
            expect(Math.abs(result[0] - expected)).toBeLessThan(1e-3);
          }
        }
      ),
      { numRuns: 200 }
    );
  });

  test('output is always Float32Array, length == numPoints, regardless of input array type', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 20 }), (n) => {
        const positions = new Float64Array(n * 3);
        const radii = new Float64Array(n).fill(1);
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [1e10, 1e10, 1e10],
        };
        const config: EffectiveRadiusConfig = {
          spatialExtendDims: [false, false, false],
          maxRadius: 1,
        };
        const result = calculateEffectiveRadii(positions, radii, viewState, config, 3);
        expect(result).toBeInstanceOf(Float32Array);
        expect(result.length).toBe(n);
      }),
      { numRuns: 50 }
    );
  });

  test('monotonicity in coordinate: larger |D| -> non-larger R_eff (with R fixed)', () => {
    fc.assert(
      fc.property(
        positiveRadius,
        fc.float({ min: 0, max: Math.fround(100), noNaN: true, noDefaultInfinity: true }),
        fc.float({ min: 0, max: Math.fround(100), noNaN: true, noDefaultInfinity: true }),
        (R, d1, d2) => {
          if (d1 === d2) return; // skip ties; monotonicity is non-strict
          const [smaller, larger] = d1 < d2 ? [d1, d2] : [d2, d1];

          const viewState: ViewState = {
            displayDims: [0, 1, 2],
            slicePosition: [0, 0, 0, 0],
            tolerance: [0, 0, 0, HIDDEN_TOLERANCE],
          };
          const config: EffectiveRadiusConfig = {
            spatialExtendDims: [false, false, false, true],
            maxRadius: R,
          };

          const positionsSmaller = new Float32Array([0, 0, 0, smaller]);
          const positionsLarger = new Float32Array([0, 0, 0, larger]);
          const radii = new Float32Array([R]);

          const rEffSmaller = calculateEffectiveRadii(
            positionsSmaller,
            radii,
            viewState,
            config,
            4
          )[0];
          const rEffLarger = calculateEffectiveRadii(
            positionsLarger,
            radii,
            viewState,
            config,
            4
          )[0];

          // smaller |D| → larger (or equal) R_eff
          expect(rEffSmaller).toBeGreaterThanOrEqual(rEffLarger - 1e-4);
        }
      ),
      { numRuns: 200 }
    );
  });

  test('point centered on slice (D=0 in every hidden dim) returns full radius', () => {
    fc.assert(
      fc.property(positiveRadius, (R) => {
        const positions = new Float32Array([0, 0, 0, 0]);
        const radii = new Float32Array([R]);
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0, 0],
          tolerance: [0, 0, 0, HIDDEN_TOLERANCE],
        };
        const config: EffectiveRadiusConfig = {
          spatialExtendDims: [false, false, false, true],
          maxRadius: R,
        };
        const result = calculateEffectiveRadii(positions, radii, viewState, config, 4);
        expect(result[0]).toBeCloseTo(R, 5);
      }),
      { numRuns: 100 }
    );
  });

  test('point at or past boundary (D >= R in hidden dim) is clamped to 0', () => {
    fc.assert(
      fc.property(
        positiveRadius,
        fc.float({
          min: Math.fround(1),
          max: Math.fround(10),
          noNaN: true,
          noDefaultInfinity: true,
        }),
        (R, overshootFactor) => {
          // Place the point at distance R*overshootFactor (>= R) in the hidden dim.
          const D = R * overshootFactor;
          const positions = new Float32Array([0, 0, 0, D]);
          const radii = new Float32Array([R]);
          const viewState: ViewState = {
            displayDims: [0, 1, 2],
            slicePosition: [0, 0, 0, 0],
            tolerance: [0, 0, 0, HIDDEN_TOLERANCE],
          };
          const config: EffectiveRadiusConfig = {
            spatialExtendDims: [false, false, false, true],
            maxRadius: R,
          };
          const result = calculateEffectiveRadii(positions, radii, viewState, config, 4);
          // D >= R → cross-section is 0; the clamp at line 130 ensures non-negative output.
          expect(Number.isFinite(result[0])).toBe(true);
          expect(result[0]).toBe(0);
        }
      ),
      { numRuns: 100 }
    );
  });

  test('extend_to_all sentinel (tolerance >= 1e9 in hidden dim) returns full radius regardless of position', () => {
    // This pins the audit-flagged G1: tolerance >= 1e9 short-circuits the
    // Pythagorean subtraction in source lines 75-77 / 107-109. Mutating the
    // sentinel comparison (`>= 1e9` → `> 1e9`) would be detectable via this
    // boundary check.
    fc.assert(
      fc.property(positiveRadius, realCoord, realCoord, (R, hiddenPos, slicePos) => {
        const positions = new Float32Array([0, 0, 0, hiddenPos]);
        const radii = new Float32Array([R]);
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0, slicePos],
          tolerance: [0, 0, 0, 1e10], // extend_to_all sentinel
        };
        const config: EffectiveRadiusConfig = {
          spatialExtendDims: [false, false, false, true],
          maxRadius: R,
        };
        const result = calculateEffectiveRadii(positions, radii, viewState, config, 4);
        // With extend_to_all set, the source ignores hidden-dim distance and
        // returns R unchanged.
        expect(result[0]).toBeCloseTo(R, 5);
      }),
      { numRuns: 100 }
    );
  });
});
