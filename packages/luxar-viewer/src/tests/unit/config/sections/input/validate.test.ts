/**
 * Tests for validateInput (src/config/sections/input/validate.ts).
 */

import { describe, it, expect } from 'vitest';
import { validateInput } from '../../../../../config/sections/input/validate';
import { cloneConfig, invokeValidator } from '../../_fixtures';

describe('validateInput', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateInput).valid).toBe(true);
  });

  it('should warn when sensitivity is zero', () => {
    const cfg = cloneConfig();
    cfg.input.defaultSensitivity = 0;

    const result = invokeValidator(validateInput, cfg);

    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual input sensitivity'));
  });

  it('should warn when sensitivity is negative', () => {
    const cfg = cloneConfig();
    cfg.input.defaultSensitivity = -0.5;

    const result = invokeValidator(validateInput, cfg);

    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual input sensitivity'));
  });

  it('should warn when sensitivity exceeds 1', () => {
    const cfg = cloneConfig();
    cfg.input.defaultSensitivity = 2;

    const result = invokeValidator(validateInput, cfg);

    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual input sensitivity'));
  });

  it('should not warn when sensitivity is within valid range', () => {
    const cfg = cloneConfig();
    cfg.input.defaultSensitivity = 0.5;

    const result = invokeValidator(validateInput, cfg);

    expect(result.warnings.filter((w) => w.includes('input sensitivity'))).toHaveLength(0);
  });

  it('rejects NaN input.defaultSensitivity', () => {
    const cfg = cloneConfig();
    cfg.input.defaultSensitivity = NaN;
    const result = invokeValidator(validateInput, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid input.defaultSensitivity')
    );
  });

  // [R11/D-C1+D-G4][P5] Pin ±Infinity at the defaultSensitivity boundary.
  // defaultSensitivity ∈ (0, 1] in practice; +Infinity > 1 (must reject)
  // and -Infinity < 0 (must reject). Symmetric to FOV / fitRatio coverage.
  it('rejects Infinity input.defaultSensitivity', () => {
    const cfg = cloneConfig();
    cfg.input.defaultSensitivity = Number.POSITIVE_INFINITY;
    const result = invokeValidator(validateInput, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid input.defaultSensitivity')
    );
  });

  it('rejects -Infinity input.defaultSensitivity', () => {
    const cfg = cloneConfig();
    cfg.input.defaultSensitivity = Number.NEGATIVE_INFINITY;
    const result = invokeValidator(validateInput, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid input.defaultSensitivity')
    );
  });
});
