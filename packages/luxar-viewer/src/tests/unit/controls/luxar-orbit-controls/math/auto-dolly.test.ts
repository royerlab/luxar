/**
 * Unit tests for the auto-dolly math (`math/auto-dolly.ts`).
 *
 * The properties under test are the ones the feature's correctness rests on:
 * the oscillation must return EXACTLY to where it started (or the camera
 * creeps toward or away from the subject over a long session), it must reach
 * the amplitude the user asked for regardless of frame rate, and it must be
 * inert rather than explosive on a hostile input, because it runs inside the
 * render loop.
 */

import { describe, it, expect } from 'vitest';
import {
  advanceDollyPhase,
  dollyAmplitudeChangeScale,
  dollyScale,
} from '../../../../../controls/luxar-orbit-controls/math/auto-dolly';

const TWO_PI = Math.PI * 2;

/** Run a whole oscillation at `steps` frames and return the net distance factor. */
function netFactorOverPeriod(steps: number, amplitude: number, period = 10): number {
  const dt = period / steps;
  let phase = 0;
  let factor = 1;
  for (let i = 0; i < steps; i++) {
    const next = advanceDollyPhase(phase, dt, period);
    factor *= dollyScale(phase, next, amplitude);
    phase = next;
  }
  return factor;
}

describe('advanceDollyPhase', () => {
  it('advances a full turn over exactly one period', () => {
    // Half a period from 0 is π; the modulo only bites past 2π.
    expect(advanceDollyPhase(0, 5, 10)).toBeCloseTo(Math.PI, 12);
  });

  it('wraps into [0, 2π) instead of growing without bound', () => {
    // Three quarters plus three quarters = 1.5 turns → half a turn, wrapped.
    const half = advanceDollyPhase(0, 7.5, 10);
    const wrapped = advanceDollyPhase(half, 7.5, 10);
    expect(wrapped).toBeGreaterThanOrEqual(0);
    expect(wrapped).toBeLessThan(TWO_PI);
    expect(wrapped).toBeCloseTo(Math.PI, 12);
  });

  it('keeps a backwards clock jump inside [0, 2π) rather than negative', () => {
    // `%` in JS keeps the sign of the dividend, so a negative dt would leave a
    // negative phase — harmless for sin(), but it would break the invariant
    // every other reader is entitled to assume.
    const phase = advanceDollyPhase(0.1, -1, 10);
    expect(phase).toBeGreaterThanOrEqual(0);
    expect(phase).toBeLessThan(TWO_PI);
  });

  it.each([
    ['zero period', 0],
    ['negative period', -5],
    ['NaN period', NaN],
    ['infinite period', Infinity],
  ])('is inert on a %s (no NaN into the camera)', (_label, period) => {
    expect(advanceDollyPhase(1.234, 1 / 60, period as number)).toBe(1.234);
  });

  it('is inert on a non-finite deltaTime', () => {
    expect(advanceDollyPhase(1.234, NaN, 10)).toBe(1.234);
  });
});

describe('dollyScale', () => {
  it('reaches ±amplitude as a RATIO of the distance', () => {
    // Quarter turn in = ÷1.15, three-quarter turn out = ×1.15. Ratio-symmetric
    // is the whole point: it means the same thing at any scene scale, and it
    // is what N mousewheel clicks each way actually do.
    const near = dollyScale(0, Math.PI / 2, 0.15);
    const far = dollyScale(0, (3 * Math.PI) / 2, 0.15);
    expect(near).toBeCloseTo(1 / 1.15, 12);
    expect(far).toBeCloseTo(1.15, 12);
  });

  it('moves CLOSER first, so switching the feature on starts with an approach', () => {
    expect(dollyScale(0, 0.01, 0.15)).toBeLessThan(1);
  });

  it('returns to exactly the starting distance after a full period', () => {
    // The load-bearing property. Integrating −A·cos φ·dφ instead of taking the
    // difference of sines would accumulate a per-period error here, and the
    // camera would creep in or out over a long presentation.
    expect(netFactorOverPeriod(600, 0.15)).toBeCloseTo(1, 12);
  });

  it('is frame-rate independent: 60 fps, 12 fps and one giant step agree', () => {
    const at60 = netFactorOverPeriod(600, 0.3);
    const at12 = netFactorOverPeriod(120, 0.3);
    const atFour = netFactorOverPeriod(4, 0.3);
    expect(at60).toBeCloseTo(1, 12);
    expect(at12).toBeCloseTo(1, 12);
    expect(atFour).toBeCloseTo(1, 12);
  });

  it('reaches the same extreme whether stepped coarsely or finely', () => {
    // Half a period in one step vs sixty, both landing on the far extreme.
    const oneStep = dollyScale(0, Math.PI / 2, 0.25);
    let phase = 0;
    let factor = 1;
    for (let i = 0; i < 60; i++) {
      const next = advanceDollyPhase(phase, 10 / 240, 10);
      factor *= dollyScale(phase, next, 0.25);
      phase = next;
    }
    expect(factor).toBeCloseTo(oneStep, 12);
  });

  it.each([
    ['zero amplitude', 0],
    ['negative amplitude', -0.2],
    ['NaN amplitude', NaN],
  ])('is inert on a %s', (_label, amplitude) => {
    expect(dollyScale(0, 1, amplitude as number)).toBe(1);
  });

  it('is inert on a non-finite phase', () => {
    expect(dollyScale(NaN, 1, 0.15)).toBe(1);
    expect(dollyScale(0, Infinity, 0.15)).toBe(1);
  });
});

describe('dollyAmplitudeChangeScale', () => {
  it('keeps the same baseline when amplitude changes at a non-zero phase', () => {
    const phase = Math.PI / 2;
    const distanceAtOldAmplitude = 5 * dollyScale(0, phase, 0.15);
    const compensated = distanceAtOldAmplitude * dollyAmplitudeChangeScale(phase, 0.15, 0.5);

    expect(compensated).toBeCloseTo(5 * dollyScale(0, phase, 0.5), 12);
  });

  it('returns the current swing to its baseline when the new amplitude is zero', () => {
    const phase = (3 * Math.PI) / 2;
    const displaced = 5 * dollyScale(0, phase, 0.3);

    expect(displaced * dollyAmplitudeChangeScale(phase, 0.3, 0)).toBeCloseTo(5, 12);
  });
});
