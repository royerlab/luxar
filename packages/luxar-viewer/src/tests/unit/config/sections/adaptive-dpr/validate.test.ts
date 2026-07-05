/**
 * Tests for validateAdaptiveDPR (src/config/sections/adaptive-dpr/validate.ts).
 * Each test exercises the per-section validator directly via
 * invokeValidator().
 */

import { describe, it, expect } from 'vitest';
import { validateAdaptiveDPR } from '../../../../../config/sections/adaptive-dpr/validate';
import { cloneConfig, invokeValidator } from '../../_fixtures';

describe('validateAdaptiveDPR', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateAdaptiveDPR).valid).toBe(true);
  });

  it('errors on non-finite values', () => {
    const cfg = cloneConfig();
    cfg.adaptiveDPR.minDPR = Number.NaN;

    const result = invokeValidator(validateAdaptiveDPR, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('adaptiveDPR.minDPR'));
  });

  it('errors when the FPS ratio ordering is broken', () => {
    const cfg = cloneConfig();
    cfg.adaptiveDPR.scaleDownFpsRatio = 0.95;
    cfg.adaptiveDPR.scaleUpFpsRatio = 0.9;

    const result = invokeValidator(validateAdaptiveDPR, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(
      expect.stringContaining('scaleDownFpsRatio < scaleUpFpsRatio')
    );
  });

  it('errors on a non-contracting scaleDownFactor and a non-expanding scaleUpFactor', () => {
    const cfg = cloneConfig();
    cfg.adaptiveDPR.scaleDownFactor = 1.0;
    cfg.adaptiveDPR.scaleUpFactor = 1.0;

    const result = invokeValidator(validateAdaptiveDPR, cfg);
    expect(result.errors).toContainEqual(expect.stringContaining('scaleDownFactor'));
    expect(result.errors).toContainEqual(expect.stringContaining('scaleUpFactor'));
  });

  it('errors when floorTtlMs does not exceed probeWindowMs', () => {
    const cfg = cloneConfig();
    cfg.adaptiveDPR.floorTtlMs = 1500;
    cfg.adaptiveDPR.probeWindowMs = 1500;

    const result = invokeValidator(validateAdaptiveDPR, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('floorTtlMs'));
  });

  it('errors when backoffMaxTtlMs is below floorTtlMs', () => {
    const cfg = cloneConfig();
    cfg.adaptiveDPR.backoffMaxTtlMs = 10_000;

    const result = invokeValidator(validateAdaptiveDPR, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('backoffMaxTtlMs'));
  });

  it('errors on non-integer punishedAscentThreshold', () => {
    const cfg = cloneConfig();
    cfg.adaptiveDPR.punishedAscentThreshold = 1.5;

    const result = invokeValidator(validateAdaptiveDPR, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('punishedAscentThreshold'));
  });

  it('warns (not errors) on a probe window inside the FPS sample window', () => {
    const cfg = cloneConfig();
    cfg.adaptiveDPR.probeWindowMs = 800;

    const result = invokeValidator(validateAdaptiveDPR, cfg);
    expect(result.valid).toBe(true);
    expect(result.warnings).toContainEqual(expect.stringContaining('probeWindowMs'));
  });

  it('warns (not errors) on a gapResetMs at or above the FPS window', () => {
    const cfg = cloneConfig();
    cfg.adaptiveDPR.gapResetMs = 1000;

    const result = invokeValidator(validateAdaptiveDPR, cfg);
    expect(result.valid).toBe(true);
    expect(result.warnings).toContainEqual(expect.stringContaining('gapResetMs'));
  });
});
