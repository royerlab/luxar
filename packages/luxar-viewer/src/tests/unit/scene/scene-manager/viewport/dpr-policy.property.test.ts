// @vitest-environment jsdom
/**
 * Property tests for the pure DPR-policy helpers (fast-check).
 *
 * Companion to dpr-policy.test.ts. The DPR helpers are pure ratio/coercion
 * math, so the example tests are best backed by invariants over arbitrary
 * native- and requested-DPR values (audit findings viewport G7 / H1).
 *
 * Every property is stated against the CEILING — `min(native, cap)` —
 * rather than the display's raw DPR, and the cap is itself an arbitrary.
 * That is deliberate: pinning the cap to one value (or lifting it) would
 * re-test the pre-cap policy and leave the composition of the two
 * unverified, which is exactly where a ceiling bug would live.
 *
 *   getActivePixelRatio(override):
 *     - never exceeds the ceiling, for ANY override
 *     - equals the ceiling when the override is null
 *   computePixelRatioOverride(dpr):
 *     - override is null  ⟺  |coerced − ceiling| < 0.01
 *       (coerced = min(valid dpr, ceiling), else ceiling)
 *     - when override is null, active === ceiling
 *     - when override is set, active === override
 *     - invalid (non-finite / ≤0) dpr ⇒ override null & active === ceiling
 *   getNormalizedDPRScale(active):
 *     - equals active / ceiling, is ≥ 0 for active ≥ 0
 *     - is monotonic non-decreasing in active
 *     - equals 1 ⟺ active === ceiling
 */
import { afterEach, describe, expect, test } from 'vitest';
import * as fc from 'fast-check';
import {
  computePixelRatioOverride,
  getActivePixelRatio,
  getNormalizedDPRScale,
} from '../../../../../scene/scene-manager/viewport/dpr-policy';
import { getMaxPixelRatioCap, setMaxPixelRatioCap } from '../../../../../rendering/pixel-ratio-cap';
import { setNativeDPR } from '../../../../helpers/device-pixel-ratio';

const originalCap = getMaxPixelRatioCap();
afterEach(() => setMaxPixelRatioCap(originalCap));

/** Run `fn` with both DPR globals pinned, then restore. */
function withDisplay<T>(native: number, cap: number, fn: (ceiling: number) => T): T {
  const restore = setNativeDPR(native);
  setMaxPixelRatioCap(cap);
  try {
    return fn(Math.min(native, cap));
  } finally {
    setMaxPixelRatioCap(originalCap);
    restore();
  }
}

const nativeArb = fc.double({ min: 0.5, max: 4, noNaN: true, noDefaultInfinity: true });
const reqArb = fc.double({ min: 0.1, max: 8, noNaN: true, noDefaultInfinity: true });
// 1.0 is the shipped default; Infinity is "high DPR allowed"; the values
// between cover a `?dpr=` pin or a capture override raising the cap part
// of the way.
const capArb = fc.constantFrom(1.0, 1.5, 2, 3, Infinity);

describe('dpr-policy properties', () => {
  describe('getActivePixelRatio', () => {
    test('never exceeds the ceiling, for any override', () => {
      fc.assert(
        fc.property(
          nativeArb,
          capArb,
          fc.option(reqArb, { nil: null }),
          (native, cap, override) => {
            withDisplay(native, cap, (ceiling) => {
              expect(getActivePixelRatio(override)).toBeLessThanOrEqual(ceiling + 1e-12);
            });
          }
        )
      );
    });

    test('a null override resolves to exactly the ceiling', () => {
      fc.assert(
        fc.property(nativeArb, capArb, (native, cap) => {
          withDisplay(native, cap, (ceiling) => {
            expect(getActivePixelRatio(null)).toBe(ceiling);
          });
        })
      );
    });
  });

  describe('computePixelRatioOverride', () => {
    test('override is null exactly when the request is within 0.01 of the ceiling', () => {
      fc.assert(
        fc.property(nativeArb, capArb, reqArb, (native, cap, req) => {
          withDisplay(native, cap, (ceiling) => {
            const { override, active } = computePixelRatioOverride(req);
            const coerced = Math.min(req, ceiling);
            if (Math.abs(coerced - ceiling) < 0.01) {
              expect(override).toBeNull();
              expect(active).toBe(ceiling);
            } else {
              expect(override).toBe(coerced);
              expect(active).toBe(coerced);
            }
          });
        })
      );
    });

    test('invalid (non-finite or non-positive) requests fall back to the ceiling', () => {
      const badArb = fc.constantFrom(NaN, Infinity, -Infinity, 0, -1, -3.2);
      fc.assert(
        fc.property(nativeArb, capArb, badArb, (native, cap, bad) => {
          withDisplay(native, cap, (ceiling) => {
            const { override, active } = computePixelRatioOverride(bad);
            expect(override).toBeNull();
            expect(active).toBe(ceiling);
          });
        })
      );
    });
  });

  describe('getNormalizedDPRScale', () => {
    test('equals active/ceiling and is non-negative for non-negative active', () => {
      fc.assert(
        fc.property(
          nativeArb,
          capArb,
          fc.double({ min: 0, max: 8, noNaN: true, noDefaultInfinity: true }),
          (native, cap, active) => {
            withDisplay(native, cap, (ceiling) => {
              const scale = getNormalizedDPRScale(active);
              expect(scale).toBeCloseTo(active / ceiling, 10);
              expect(scale).toBeGreaterThanOrEqual(0);
            });
          }
        )
      );
    });

    test('is monotonic non-decreasing in active DPR', () => {
      fc.assert(
        fc.property(
          nativeArb,
          capArb,
          fc.double({ min: 0, max: 8, noNaN: true, noDefaultInfinity: true }),
          fc.double({ min: 0, max: 8, noNaN: true, noDefaultInfinity: true }),
          (native, cap, a, b) => {
            withDisplay(native, cap, () => {
              const lo = Math.min(a, b);
              const hi = Math.max(a, b);
              expect(getNormalizedDPRScale(lo)).toBeLessThanOrEqual(getNormalizedDPRScale(hi));
            });
          }
        )
      );
    });

    test('equals 1 exactly when active equals the ceiling', () => {
      fc.assert(
        fc.property(nativeArb, capArb, (native, cap) => {
          withDisplay(native, cap, (ceiling) => {
            expect(getNormalizedDPRScale(ceiling)).toBe(1);
          });
        })
      );
    });
  });
});
