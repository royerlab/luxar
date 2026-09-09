/**
 * Unit tests for the SSAA render-target sizing helper.
 */

import { describe, it, expect } from 'vitest';
import {
  computeEffectiveRenderSize,
  computeRenderTargetAllocation,
} from '../../../../rendering/post-processing/render-target-sizing';

describe('computeEffectiveRenderSize', () => {
  it('passes through the render size unchanged when SSAA is disabled', () => {
    expect(computeEffectiveRenderSize({ width: 1920, height: 1080 }, false, 2)).toEqual({
      width: 1920,
      height: 1080,
    });
  });

  it('ignores the multiplier when SSAA is disabled (even if > 1)', () => {
    expect(computeEffectiveRenderSize({ width: 800, height: 600 }, false, 4.0)).toEqual({
      width: 800,
      height: 600,
    });
  });

  it('multiplies both dimensions by ssaaMultiplier when enabled', () => {
    expect(computeEffectiveRenderSize({ width: 1000, height: 500 }, true, 2)).toEqual({
      width: 2000,
      height: 1000,
    });
  });

  it('rounds non-integer multipliers (e.g. 1.5×) to integer pixels', () => {
    expect(computeEffectiveRenderSize({ width: 1024, height: 768 }, true, 1.5)).toEqual({
      width: 1536,
      height: 1152,
    });
  });

  it('rounds halves to nearest integer per Math.round semantics', () => {
    // 1001 * 1.5 = 1501.5 → rounds to 1502 (half-to-even is *not* used by Math.round).
    expect(computeEffectiveRenderSize({ width: 1001, height: 1003 }, true, 1.5)).toEqual({
      width: 1502,
      height: 1505,
    });
  });

  it('handles fractional downscale (multiplier < 1)', () => {
    expect(computeEffectiveRenderSize({ width: 1000, height: 500 }, true, 0.5)).toEqual({
      width: 500,
      height: 250,
    });
  });

  it('handles a 1x multiplier as a no-op size', () => {
    expect(computeEffectiveRenderSize({ width: 600, height: 400 }, true, 1)).toEqual({
      width: 600,
      height: 400,
    });
  });

  it('handles zero dimensions (degenerate, but mathematically valid)', () => {
    expect(computeEffectiveRenderSize({ width: 0, height: 0 }, true, 4)).toEqual({
      width: 0,
      height: 0,
    });
  });
});

describe('computeRenderTargetAllocation', () => {
  it('matches Three.js floor rounding at a non-integer DPR', () => {
    expect(
      computeRenderTargetAllocation({ width: 1001, height: 1003 }, true, 1.5, 1.31, 8192)
    ).toEqual({
      logical: { width: 1502, height: 1505 },
      physical: { width: 1967, height: 1971 },
      limited: false,
    });
  });

  it('reduces oversized SSAA dimensions proportionally to the framebuffer limit', () => {
    const allocation = computeRenderTargetAllocation(
      { width: 2509, height: 1328 },
      true,
      2,
      2,
      8192
    );

    expect(allocation.logical.width).toBe(4096);
    expect(allocation.logical.height).toBe(2167);
    expect(allocation.physical).toEqual({ width: 8192, height: 4334 });
    expect(allocation.limited).toBe(true);
    const sourceAspect = 2509 / 1328;
    const allocatedAspect = allocation.physical.width / allocation.physical.height;
    expect(Math.abs(allocatedAspect - sourceAspect) / sourceAspect).toBeLessThan(0.001);
  });

  it.each([
    [{ width: 2721, height: 1440 }, 4, 1, 8192],
    [{ width: 2040, height: 1080 }, 3, 1, 4096],
    [{ width: 1371, height: 566 }, 4, 0.81, 4096],
  ])(
    'keeps limited physical dimensions encoder-compatible',
    (renderSize, multiplier, pixelRatio, limit) => {
      const allocation = computeRenderTargetAllocation(
        renderSize,
        true,
        multiplier,
        pixelRatio,
        limit
      );

      expect(allocation.limited).toBe(true);
      expect(allocation.physical.width % 2).toBe(0);
      expect(allocation.physical.height % 2).toBe(0);
      expect(Math.floor(allocation.logical.width * pixelRatio)).toBe(allocation.physical.width);
      expect(Math.floor(allocation.logical.height * pixelRatio)).toBe(allocation.physical.height);
      expect(Math.round(allocation.logical.width * pixelRatio)).toBe(allocation.physical.width);
      expect(Math.round(allocation.logical.height * pixelRatio)).toBe(allocation.physical.height);
    }
  );

  it('keeps a zero-sized hidden canvas and its targets at the same one-pixel floor', () => {
    expect(computeRenderTargetAllocation({ width: 0, height: 0 }, false, 2, 2, 8192)).toEqual({
      logical: { width: 0.5, height: 0.5 },
      physical: { width: 1, height: 1 },
      limited: false,
    });
  });
});
