/**
 * Tests for validateDepthSort (src/config/sections/depth-sort/validate.ts).
 * Each test exercises the per-section validator directly via
 * invokeValidator().
 */

import { describe, it, expect } from 'vitest';
import { validateDepthSort } from '../../../../../config/sections/depth-sort/validate';
import { cloneConfig, invokeValidator } from '../../_fixtures';

describe('validateDepthSort', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateDepthSort).valid).toBe(true);
  });

  it('errors on a non-positive or non-finite angle threshold', () => {
    for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const cfg = cloneConfig();
      cfg.depthSort.angleThresholdDeg = bad;
      const result = invokeValidator(validateDepthSort, cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('depthSort.angleThresholdDeg'));
    }
  });

  it('errors on a non-positive or non-finite translation fraction', () => {
    for (const bad of [0, -0.05, Number.NaN]) {
      const cfg = cloneConfig();
      cfg.depthSort.translationFraction = bad;
      const result = invokeValidator(validateDepthSort, cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('depthSort.translationFraction')
      );
    }
  });

  it('warns (but stays valid) on design-defeating coarse thresholds', () => {
    const cfg = cloneConfig();
    cfg.depthSort.angleThresholdDeg = 90;
    cfg.depthSort.translationFraction = 2;
    const result = invokeValidator(validateDepthSort, cfg);
    expect(result.valid).toBe(true);
    expect(result.warnings).toContainEqual(expect.stringContaining('angleThresholdDeg'));
    expect(result.warnings).toContainEqual(expect.stringContaining('translationFraction'));
  });
});
