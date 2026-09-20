import { describe, expect, it } from 'vitest';

import {
  captureStableImage,
  evaluateCaptureChecks,
  evaluateLowerIsBetterComparisons,
  evaluateThresholds,
  ncc,
  scoreImagePair,
} from './visual-ab-core.mjs';

function solidRgb(red, green, blue, pixels = 64) {
  return Array.from({ length: pixels }, () => [red, green, blue]).flat();
}

describe('visual A/B scoring', () => {
  it('waits for two pixel-identical captures rather than matching mean luma', async () => {
    const captures = [
      { rgb: [0, 1, 2], meanLuma: 1 },
      { rgb: [2, 1, 0], meanLuma: 1 },
      { rgb: [2, 1, 0], meanLuma: 1 },
    ];

    const stable = await captureStableImage(() => Promise.resolve(captures.shift()), {
      intervalMs: 0,
      maxCaptures: 3,
    });

    expect(stable.rgb).toEqual([2, 1, 0]);
    expect(stable.stabilityCaptures).toBe(3);
  });

  it('does not accept an early plateau before the minimum capture count', async () => {
    const captures = [
      { rgb: [1], meanLuma: 1 },
      { rgb: [1], meanLuma: 1 },
      { rgb: [2], meanLuma: 2 },
      { rgb: [2], meanLuma: 2 },
    ];

    const stable = await captureStableImage(() => Promise.resolve(captures.shift()), {
      intervalMs: 0,
      minCaptures: 4,
      maxCaptures: 4,
    });

    expect(stable.rgb).toEqual([2]);
    expect(stable.stabilityCaptures).toBe(4);
  });

  it('fails with recent luma diagnostics when captures never stabilise', async () => {
    let value = 0;
    await expect(
      captureStableImage(() => Promise.resolve({ rgb: [value++], meanLuma: value }), {
        intervalMs: 0,
        maxCaptures: 3,
      })
    ).rejects.toThrow('recent mean luma: 1.000000, 2.000000, 3.000000');
  });

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

  it('passes a lower-is-better comparison on the recorded margin boundary', () => {
    const comparisons = evaluateLowerIsBetterComparisons(
      [
        { id: 'chromatic', score: { meanDeltaE: 2 } },
        { id: 'spatial', score: { meanDeltaE: 5 } },
      ],
      [
        {
          id: 'chromatic-improves-colour',
          better: 'chromatic',
          worse: 'spatial',
          metric: 'meanDeltaE',
          minImprovement: 3,
        },
      ]
    );

    expect(comparisons[0]).toMatchObject({ improvement: 3, pass: true });
  });

  it('rejects a lower-is-better comparison below the recorded margin', () => {
    const comparisons = evaluateLowerIsBetterComparisons(
      [
        { id: 'chromatic', score: { meanDeltaE: 4.5 } },
        { id: 'spatial', score: { meanDeltaE: 5 } },
      ],
      [
        {
          id: 'chromatic-improves-colour',
          better: 'chromatic',
          worse: 'spatial',
          metric: 'meanDeltaE',
          minImprovement: 1,
        },
      ]
    );

    expect(comparisons[0]).toMatchObject({ improvement: 0.5, pass: false });
  });

  it('fails capture checks when additive light falls below the recorded ratio', () => {
    const reference = {
      pageErrors: [],
      litFraction: 0.2,
      blownPixelFraction: 0.001,
      meanLuma: 20,
    };
    const candidate = { ...reference, meanLuma: 12 };
    const result = evaluateCaptureChecks(reference, candidate, {
      minReferenceBlownPixelFraction: 0.0001,
      minMeanLumaRatio: 0.65,
    });

    expect(result.metrics.meanLumaRatio).toBe(0.6);
    expect(result.checks.meanLumaRatio).toBe(false);
    expect(result.pass).toBe(false);
  });

  it('passes capture checks exactly on the light-ratio boundary', () => {
    const reference = {
      pageErrors: [],
      litFraction: 0.2,
      blownPixelFraction: 0.001,
      meanLuma: 20,
    };
    const candidate = { ...reference, meanLuma: 13 };
    const result = evaluateCaptureChecks(reference, candidate, {
      minReferenceBlownPixelFraction: 0.001,
      minMeanLumaRatio: 0.65,
    });

    expect(result.metrics.meanLumaRatio).toBe(0.65);
    expect(result.pass).toBe(true);
  });
});
