/**
 * Unit tests for the pure LOD opacity math (`src/scene/lod-blend.ts`)
 * — no THREE, plain numbers only.
 */

import { describe, it, expect } from 'vitest';

import { smoothstep, energyCompensation } from '../../../scene/lod-blend';

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
