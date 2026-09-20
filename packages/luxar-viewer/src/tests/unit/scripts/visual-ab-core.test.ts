import { describe, expect, it } from 'vitest';

// @ts-expect-error The opt-in Node script is plain ESM, outside the app TS build.
import { evaluateThresholds, scoreImagePair } from '../../../../scripts/visual-ab-core.mjs';

function solidRgb(red: number, green: number, blue: number, pixels = 64): number[] {
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

  it('evaluates every recorded threshold independently', () => {
    const verdict = evaluateThresholds(
      { ssim: 0.91, meanDeltaE: 4.5, blownPixelFraction: { delta: 0.02 } },
      { minSsim: 0.95, maxMeanDeltaE: 4, maxBlownPixelFractionDelta: 0.01 }
    );

    expect(verdict.pass).toBe(false);
    expect(verdict.checks).toEqual({
      ssim: false,
      meanDeltaE: false,
      blownPixelFractionDelta: false,
    });
  });
});
