/**
 * Unit tests for controls-manager/scene-scale.ts.
 *
 * Targets audit findings G2 (deriveScaleLimits — pure math driving
 * the entire scale-aware control feel; previously untested) and H8
 * (linearity, well-ordered min/max).
 */

import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { deriveScaleLimits } from '../../../../controls/controls-manager/scene-scale';
import { config } from '../../../../config';

describe('deriveScaleLimits', () => {
  const m = config.controls.scaleMultipliers;

  it('returns minDist = diagonal * minDistanceFactor', () => {
    expect(deriveScaleLimits(100).minDist).toBeCloseTo(100 * m.minDistanceFactor, 8);
  });

  it('returns maxDist = diagonal * maxDistanceFactor', () => {
    expect(deriveScaleLimits(100).maxDist).toBeCloseTo(100 * m.maxDistanceFactor, 8);
  });

  it('returns flySpeed = diagonal * flySpeedFactor', () => {
    expect(deriveScaleLimits(100).flySpeed).toBeCloseTo(100 * m.flySpeedFactor, 8);
  });

  it('produces minDist < maxDist for any positive diagonal (H8)', () => {
    // H8 invariant: well-ordered limits — required for clamping math.
    for (const d of [0.1, 1, 10, 100, 1e6]) {
      const r = deriveScaleLimits(d);
      expect(r.minDist).toBeLessThan(r.maxDist);
    }
  });

  it('is linear in the diagonal (H8)', () => {
    // H8 invariant: deriveScaleLimits(k*d) === k * deriveScaleLimits(d).
    const r1 = deriveScaleLimits(50);
    const r2 = deriveScaleLimits(100); // 2x
    expect(r2.minDist).toBeCloseTo(2 * r1.minDist, 8);
    expect(r2.maxDist).toBeCloseTo(2 * r1.maxDist, 8);
    expect(r2.flySpeed).toBeCloseTo(2 * r1.flySpeed, 8);
  });

  it('returns zero values when diagonal is 0 (boundary)', () => {
    // P5 boundary: diagonal=0 (no scene loaded) returns zeros. The
    // orchestrator separately gates against using these via `sceneScale > 0`.
    const r = deriveScaleLimits(0);
    expect(r.minDist).toBe(0);
    expect(r.maxDist).toBe(0);
    expect(r.flySpeed).toBe(0);
  });

  it('handles very small positive diagonals without losing precision', () => {
    const r = deriveScaleLimits(1e-3);
    expect(Number.isFinite(r.minDist)).toBe(true);
    expect(Number.isFinite(r.maxDist)).toBe(true);
    expect(Number.isFinite(r.flySpeed)).toBe(true);
    expect(r.minDist).toBeGreaterThan(0);
  });

  it('handles very large diagonals without overflow', () => {
    const r = deriveScaleLimits(1e9);
    expect(Number.isFinite(r.minDist)).toBe(true);
    expect(Number.isFinite(r.maxDist)).toBe(true);
    expect(Number.isFinite(r.flySpeed)).toBe(true);
  });

  // controls.md [H8][P12] fast-check property test: deriveScaleLimits is a
  // pure linear map of the diagonal. The two invariants below cover the
  // contract over the realistic-input domain (diagonals in [1e-3, 1e9]).
  it('[property] minDist < maxDist for all positive diagonals [controls.md/H8][P12]', () => {
    fc.assert(
      fc.property(fc.double({ min: 1e-3, max: 1e9, noNaN: true }), (d) => {
        const r = deriveScaleLimits(d);
        return r.minDist < r.maxDist && r.minDist > 0 && Number.isFinite(r.maxDist);
      }),
      { numRuns: 200 }
    );
  });

  it('[property] is linear: scale(k*d) = k * scale(d) for k > 0 [controls.md/H8][P12]', () => {
    // Linearity is the deepest property of the formula. For any
    // positive d and k, deriveScaleLimits(k*d) must equal k * deriveScaleLimits(d)
    // component-wise (within float tolerance).
    fc.assert(
      fc.property(
        fc.double({ min: 1, max: 1e6, noNaN: true }),
        fc.double({ min: 0.01, max: 100, noNaN: true }),
        (d, k) => {
          const r1 = deriveScaleLimits(d);
          const r2 = deriveScaleLimits(k * d);
          // Relative tolerance: 1e-9 of the larger value (handles wide
          // dynamic range without underflow on small components).
          const tolMin = Math.max(1e-12, Math.abs(k * r1.minDist) * 1e-9);
          const tolMax = Math.max(1e-12, Math.abs(k * r1.maxDist) * 1e-9);
          const tolFly = Math.max(1e-12, Math.abs(k * r1.flySpeed) * 1e-9);
          return (
            Math.abs(r2.minDist - k * r1.minDist) < tolMin &&
            Math.abs(r2.maxDist - k * r1.maxDist) < tolMax &&
            Math.abs(r2.flySpeed - k * r1.flySpeed) < tolFly
          );
        }
      ),
      { numRuns: 200 }
    );
  });
});
