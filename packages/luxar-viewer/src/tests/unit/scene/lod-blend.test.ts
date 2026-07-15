/**
 * Unit tests for the pure coverage-band cross-fade math (`src/scene/lod-blend.ts`)
 * — no THREE, plain numbers only.
 */

import { describe, it, expect } from 'vitest';

import { smoothstep, coverageBlendPlan, energyCompensation } from '../../../scene/lod-blend';

describe('smoothstep', () => {
  it('clamps to 0 at/below edge0 and 1 at/above edge1', () => {
    expect(smoothstep(0, 1, -0.5)).toBe(0);
    expect(smoothstep(0, 1, 0)).toBe(0);
    expect(smoothstep(0, 1, 1)).toBe(1);
    expect(smoothstep(0, 1, 1.5)).toBe(1);
  });

  it('is the Hermite 3t²−2t³ curve in between (0.5 at the midpoint)', () => {
    expect(smoothstep(0, 1, 0.5)).toBeCloseTo(0.5, 12);
    expect(smoothstep(0, 1, 0.25)).toBeCloseTo(0.15625, 12);
    expect(smoothstep(0, 1, 0.75)).toBeCloseTo(0.84375, 12);
  });

  it('handles a degenerate edge0>=edge1 as a hard step (no NaN)', () => {
    expect(smoothstep(0.5, 0.5, 0.49)).toBe(0);
    expect(smoothstep(0.5, 0.5, 0.5)).toBe(1);
  });
});

describe('coverageBlendPlan', () => {
  // 3 levels: coarse@0, mid@0.5, fine@1.0 → boundaries at 0.5 and 1.0.
  // All gaps are 0.5, so at fraction F the half-width is F·0.5 = 0.2 everywhere.
  const T = [0, 0.5, 1.0];
  const F = 0.4;

  it('returns null when the metric is not within a band of any boundary', () => {
    expect(coverageBlendPlan(T, 0.2, F)).toBeNull(); // below the 0.5 band ([0.3,0.7])
    expect(coverageBlendPlan(T, 0.75, F)).toBeNull(); // between the 0.5 and 1.0 bands
  });

  it('blends the two levels straddling the nearest boundary (0.5 at the boundary)', () => {
    const at = coverageBlendPlan(T, 0.5, F)!; // band [0.3,0.7], center
    expect(at).toMatchObject({ lo: 0, hi: 1 });
    expect(at.hiWeight).toBeCloseTo(0.5, 12);
    const below = coverageBlendPlan(T, 0.4, F)!; // t=(0.4-0.3)/0.4=0.25 → 0.15625
    expect(below).toMatchObject({ lo: 0, hi: 1 });
    expect(below.hiWeight).toBeCloseTo(0.15625, 12);
    const above = coverageBlendPlan(T, 0.6, F)!; // t=0.75 → 0.84375
    expect(above).toMatchObject({ lo: 0, hi: 1 });
    expect(above.hiWeight).toBeCloseTo(0.84375, 12);
  });

  it('is continuous through the boundary (same pair both sides, weight rises monotonically)', () => {
    const w = (m: number) => coverageBlendPlan(T, m, F)!.hiWeight;
    expect(w(0.4)).toBeLessThan(w(0.5));
    expect(w(0.5)).toBeLessThan(w(0.6));
    // just below and just above the boundary → nearly equal weights (~0.5), no jump
    expect(Math.abs(w(0.499) - w(0.501))).toBeLessThan(0.05);
  });

  it('selects the boundary at the upper end of the level range too', () => {
    const at = coverageBlendPlan(T, 1.0, F)!; // finest boundary reuses gapBelow → band [0.8,1.2]
    expect(at).toMatchObject({ lo: 1, hi: 2 });
    expect(at.hiWeight).toBeCloseTo(0.5, 12);
    expect(coverageBlendPlan(T, 0.95, F)!.lo).toBe(1);
  });

  it('scales the band to the local gap (proportional): wider at the finest step, narrower at coarse steps', () => {
    // Geometric ladder: gaps 0.25, 0.25, 0.5 → half-widths 0.1, 0.1, 0.2.
    const geo = [0, 0.25, 0.5, 1.0];
    // Finest boundary (1.0) has the widest band (±0.2): 0.85 is inside.
    expect(coverageBlendPlan(geo, 0.85, F)!).toMatchObject({ lo: 2, hi: 3 });
    // Coarse boundary (0.25) has a narrow band (±0.1): 0.16 in, 0.12 out.
    expect(coverageBlendPlan(geo, 0.16, F)!).toMatchObject({ lo: 0, hi: 1 });
    expect(coverageBlendPlan(geo, 0.12, F)).toBeNull();
    // Proportional bands never overlap: 0.37 sits in the crisp gap between the
    // 0.25 band ([0.15,0.35]) and the 0.5 band ([0.4,0.6]).
    expect(coverageBlendPlan(geo, 0.37, F)).toBeNull();
  });

  it('shrinks the band between tightly-spaced levels (no overlap) and still straddles the right pair', () => {
    const tight = [0, 0.5, 0.6]; // gapBelow(0.5)=0.5, gapAbove(0.5)=0.1 → half at 0.5 is 0.4·0.1=0.04
    // 0.52 is inside the narrow 0.5 band ([0.46,0.54]); the 0.6 band ([0.56,0.64]) excludes it.
    expect(coverageBlendPlan(tight, 0.52, F)!).toMatchObject({ lo: 0, hi: 1 });
    // 0.58 lands in the 0.6 band instead.
    expect(coverageBlendPlan(tight, 0.58, F)!).toMatchObject({ lo: 1, hi: 2 });
    // 0.55 is in the crisp gap between the two narrow bands.
    expect(coverageBlendPlan(tight, 0.55, F)).toBeNull();
  });

  it('is disabled (null) for fraction<=0 or fewer than two levels', () => {
    expect(coverageBlendPlan(T, 0.5, 0)).toBeNull();
    expect(coverageBlendPlan(T, 0.5, -1)).toBeNull();
    expect(coverageBlendPlan([0.0], 0.5, F)).toBeNull();
  });
});

describe('energyCompensation', () => {
  const FLOOR = 0.1; // cap the boost at 10×

  it('returns 1 (no compensation) when there is nothing to compensate', () => {
    expect(energyCompensation(undefined, FLOOR)).toBe(1); // unstamped dataset
    expect(energyCompensation(1, FLOOR)).toBe(1); // complete ladder / non-progressive leaf
    expect(energyCompensation(1.5, FLOOR)).toBe(1); // clamp: never dim below authored
    expect(energyCompensation(0, FLOOR)).toBe(1); // degenerate: no chunk committed yet
    expect(energyCompensation(-0.2, FLOOR)).toBe(1); // degenerate: negative
    expect(energyCompensation(NaN, FLOOR)).toBe(1); // degenerate: NaN
  });

  it('is 1/e for a partial prefix above the floor', () => {
    expect(energyCompensation(0.5, FLOOR)).toBeCloseTo(2, 12);
    expect(energyCompensation(0.25, FLOOR)).toBeCloseTo(4, 12);
    expect(energyCompensation(0.8, FLOOR)).toBeCloseTo(1.25, 12);
  });

  it('caps the boost at 1/floor for a tiny early prefix', () => {
    expect(energyCompensation(0.05, FLOOR)).toBeCloseTo(10, 12); // 1/0.1, not 1/0.05
    expect(energyCompensation(0.001, FLOOR)).toBeCloseTo(10, 12);
    expect(energyCompensation(0.1, FLOOR)).toBeCloseTo(10, 12); // exactly at the floor
  });

  it('is the brightness invariant: factor·e === 1 for e in [floor, 1)', () => {
    // The whole point: a prefix carrying energy fraction e, scaled by the factor,
    // renders at the full-level energy E. factor·e = 1 exactly above the floor.
    for (const e of [0.1, 0.15, 0.3, 0.6, 0.9, 0.99]) {
      expect(energyCompensation(e, FLOOR) * e).toBeCloseTo(1, 12);
    }
  });

  it('relaxes monotonically toward 1 as the ladder fills in', () => {
    // As e climbs (more chunks commit), the factor decreases toward 1 — the
    // over-brightened cores dim back to authored brightness, no overshoot.
    const e = [0.2, 0.4, 0.7, 0.95, 1.0];
    const factors = e.map((v) => energyCompensation(v, FLOOR));
    for (let i = 1; i < factors.length; i++) {
      expect(factors[i]).toBeLessThanOrEqual(factors[i - 1]);
    }
    expect(factors[factors.length - 1]).toBe(1);
  });
});
