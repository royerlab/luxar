/**
 * Property tests for the pure DPR-policy helpers (fast-check).
 *
 * Companion to dpr-policy.test.ts. The DPR helpers are pure ratio/coercion
 * math, so the example tests are best backed by invariants over arbitrary
 * native- and requested-DPR values (audit findings viewport G7 / H1):
 *
 *   computePixelRatioOverride(dpr):
 *     - override is null  ⟺  |coerced − native| < 0.01  (coerced = valid dpr, else native)
 *     - when override is null, active === native
 *     - when override is set, active === override
 *     - invalid (non-finite / ≤0) dpr ⇒ override null & active === native
 *   getNormalizedDPRScale(active):
 *     - equals active / native, is ≥ 0 for active ≥ 0
 *     - is monotonic non-decreasing in active
 *     - equals 1 ⟺ active === native
 */
import { describe, expect, test } from 'vitest';
import * as fc from 'fast-check';
import {
  computePixelRatioOverride,
  getNormalizedDPRScale,
} from '../../../../../scene/scene-manager/viewport/dpr-policy';

function withNativeDPR<T>(dpr: number, fn: () => T): T {
  const original = window.devicePixelRatio;
  Object.defineProperty(window, 'devicePixelRatio', {
    value: dpr,
    configurable: true,
    writable: true,
  });
  try {
    return fn();
  } finally {
    Object.defineProperty(window, 'devicePixelRatio', {
      value: original,
      configurable: true,
      writable: true,
    });
  }
}

const nativeArb = fc.double({ min: 0.5, max: 4, noNaN: true, noDefaultInfinity: true });
const reqArb = fc.double({ min: 0.1, max: 8, noNaN: true, noDefaultInfinity: true });

describe('dpr-policy properties', () => {
  describe('computePixelRatioOverride', () => {
    test('override is null exactly when the request is within 0.01 of native', () => {
      fc.assert(
        fc.property(nativeArb, reqArb, (native, req) => {
          withNativeDPR(native, () => {
            const { override, active } = computePixelRatioOverride(req);
            const nearNative = Math.abs(req - native) < 0.01;
            if (nearNative) {
              expect(override).toBeNull();
              expect(active).toBe(native);
            } else {
              expect(override).toBe(req);
              expect(active).toBe(req);
            }
          });
        })
      );
    });

    test('invalid (non-finite or non-positive) requests fall back to native', () => {
      const badArb = fc.constantFrom(NaN, Infinity, -Infinity, 0, -1, -3.2);
      fc.assert(
        fc.property(nativeArb, badArb, (native, bad) => {
          withNativeDPR(native, () => {
            const { override, active } = computePixelRatioOverride(bad);
            expect(override).toBeNull();
            expect(active).toBe(native);
          });
        })
      );
    });
  });

  describe('getNormalizedDPRScale', () => {
    test('equals active/native and is non-negative for non-negative active', () => {
      fc.assert(
        fc.property(
          nativeArb,
          fc.double({ min: 0, max: 8, noNaN: true, noDefaultInfinity: true }),
          (native, active) => {
            withNativeDPR(native, () => {
              const scale = getNormalizedDPRScale(active);
              expect(scale).toBeCloseTo(active / native, 10);
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
          fc.double({ min: 0, max: 8, noNaN: true, noDefaultInfinity: true }),
          fc.double({ min: 0, max: 8, noNaN: true, noDefaultInfinity: true }),
          (native, a, b) => {
            withNativeDPR(native, () => {
              const lo = Math.min(a, b);
              const hi = Math.max(a, b);
              expect(getNormalizedDPRScale(lo)).toBeLessThanOrEqual(getNormalizedDPRScale(hi));
            });
          }
        )
      );
    });

    test('equals 1 exactly when active equals native', () => {
      fc.assert(
        fc.property(nativeArb, (native) => {
          withNativeDPR(native, () => {
            expect(getNormalizedDPRScale(native)).toBe(1);
          });
        })
      );
    });
  });
});
