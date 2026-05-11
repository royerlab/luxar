/**
 * Unit tests for the SSAA render-target sizing helper.
 */

import { describe, it, expect } from 'vitest';
import { computeEffectiveRenderSize } from '../../../../rendering/post-processing/render-target-sizing';

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
