import { describe, expect, it } from 'vitest';

import { evaluateThresholds, ncc, scoreImagePair } from './visual-ab-core.mjs';

function solidRgb(red, green, blue, pixels = 64) {
  return Array.from({ length: pixels }, () => [red, green, blue]).flat();
}

describe('visual A/B scoring', () => {
  it('reports identity across every metric', () => {
    const rgb = solidRgb(64, 96, 128);
    const score = scoreImagePair(rgb, rgb, 8);

    expect(score.ssim).toBeCloseTo(1, 12);
    expect(score.ncc).toBeCloseTo(1, 12);
    expect(score.meanDeltaE).toBeCloseTo(0, 12);
    expect(score.blownPixelFraction.reference).toBe(0);
    expect(score.blownPixelFraction.candidate).toBe(0);
    expect(score.blownPixelFraction.delta).toBe(0);
  });

  it('defines zero-variance NCC for identical and different constants', () => {
    expect(ncc([4, 4, 4], [4, 4, 4])).toBe(1);
    expect(ncc([4, 4, 4], [5, 5, 5])).toBe(0);
  });

  it('detects colour error and newly clipped pixels', () => {
    const reference = solidRgb(40, 40, 40);
    const candidate = solidRgb(255, 40, 40);
    const score = scoreImagePair(reference, candidate, 8);

    expect(score.ssim).toBeLessThan(0.9);
    expect(score.meanDeltaE).toBeGreaterThan(20);
    expect(score.blownPixelFraction.reference).toBe(0);
    expect(score.blownPixelFraction.candidate).toBe(1);
    expect(score.blownPixelFraction.delta).toBe(1);
  });

  it('uses full-resolution clipping fractions when captures provide them', () => {
    const rgb = solidRgb(40, 40, 40);
    const score = scoreImagePair(rgb, rgb, 8, { reference: 0.02, candidate: 0.05 });

    expect(score.blownPixelFraction).toEqual({
      reference: 0.02,
      candidate: 0.05,
      delta: 0.030000000000000002,
    });
  });

  it('passes values exactly on every threshold boundary', () => {
    const verdict = evaluateThresholds(
      { ssim: 0.95, meanDeltaE: 4, blownPixelFraction: { delta: 0.01 } },
      { minSsim: 0.95, maxMeanDeltaE: 4, maxBlownPixelFractionDelta: 0.01 }
    );

    expect(verdict.pass).toBe(true);
    expect(verdict.checks).toEqual({
      ssim: true,
      meanDeltaE: true,
      blownPixelFractionDelta: true,
    });
  });

  it('reports mixed threshold results independently', () => {
    const verdict = evaluateThresholds(
      { ssim: 0.94, meanDeltaE: 3, blownPixelFraction: { delta: 0.02 } },
      { minSsim: 0.95, maxMeanDeltaE: 4, maxBlownPixelFractionDelta: 0.01 }
    );

    expect(verdict.pass).toBe(false);
    expect(verdict.checks).toEqual({
      ssim: false,
      meanDeltaE: true,
      blownPixelFractionDelta: false,
    });
  });
});
