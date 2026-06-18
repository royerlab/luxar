/**
 * Tests for validateRendering + validateBloomConsistency
 * (src/config/sections/rendering-controls/validate.ts).
 */

import { describe, it, expect } from 'vitest';
import {
  validateRendering,
  validateBloomConsistency,
} from '../../../../../config/sections/rendering-controls/validate';
import { cloneConfig, invokeValidator } from '../../_fixtures';

describe('validateRendering', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateRendering).valid).toBe(true);
  });

  it('should error when exposure is out of range', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.exposure = -11;

    const result = invokeValidator(validateRendering, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid exposure'));
  });

  it('should accept exposure within valid range', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.exposure = 0;
    expect(
      invokeValidator(validateRendering, cfg).errors.filter((e) => e.includes('exposure'))
    ).toHaveLength(0);

    cfg.renderingControls.defaults.exposure = 10;
    expect(
      invokeValidator(validateRendering, cfg).errors.filter((e) => e.includes('exposure'))
    ).toHaveLength(0);

    cfg.renderingControls.defaults.exposure = -10;
    expect(
      invokeValidator(validateRendering, cfg).errors.filter((e) => e.includes('exposure'))
    ).toHaveLength(0);
  });

  it('rejects NaN exposure', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.exposure = NaN;
    const result = invokeValidator(validateRendering, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid exposure'));
  });

  it('rejects NaN globalOffset', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.globalOffset = NaN;
    const result = invokeValidator(validateRendering, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid globalOffset'));
  });

  it('rejects NaN globalGamma', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.globalGamma = NaN;
    const result = invokeValidator(validateRendering, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid globalGamma'));
  });

  it('should error when globalGamma is below 0.1', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.globalGamma = 0.05;
    const result = invokeValidator(validateRendering, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid globalGamma'));
  });

  it('should error when globalGamma is above 10', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.globalGamma = 11;
    const result = invokeValidator(validateRendering, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid globalGamma'));
  });

  it('should accept globalGamma at boundary values (0.1 and 10)', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.globalGamma = 0.1;
    expect(
      invokeValidator(validateRendering, cfg).errors.filter((e) => e.includes('globalGamma'))
    ).toHaveLength(0);

    cfg.renderingControls.defaults.globalGamma = 10;
    expect(
      invokeValidator(validateRendering, cfg).errors.filter((e) => e.includes('globalGamma'))
    ).toHaveLength(0);
  });
});

describe('validateBloomConsistency', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateBloomConsistency).valid).toBe(true);
  });

  it('should warn when bloomStrength is negative', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomStrength = -1;

    const result = invokeValidator(validateBloomConsistency, cfg);

    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual bloom.bloomStrength'));
  });

  it('should warn when bloomStrength exceeds 10', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomStrength = 15;

    const result = invokeValidator(validateBloomConsistency, cfg);

    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual bloom.bloomStrength'));
  });

  it('should warn when bloomRadius is negative', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomRadius = -0.5;

    const result = invokeValidator(validateBloomConsistency, cfg);

    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual bloom.bloomRadius'));
  });

  it('should warn when bloomRadius exceeds 10', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomRadius = 11;

    const result = invokeValidator(validateBloomConsistency, cfg);

    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual bloom.bloomRadius'));
  });

  it('should error when bloomThreshold is negative', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomThreshold = -0.1;

    const result = invokeValidator(validateBloomConsistency, cfg);

    expect(result.errors).toContainEqual(expect.stringContaining('Invalid bloom.bloomThreshold'));
  });

  it('should error when bloomThreshold exceeds 1', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomThreshold = 1.5;

    const result = invokeValidator(validateBloomConsistency, cfg);

    expect(result.errors).toContainEqual(expect.stringContaining('Invalid bloom.bloomThreshold'));
  });

  it('should accept bloomThreshold at boundary values (0 and 1)', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomThreshold = 0;
    expect(
      invokeValidator(validateBloomConsistency, cfg).errors.filter((e) =>
        e.includes('bloomThreshold')
      )
    ).toHaveLength(0);

    cfg.renderingControls.defaults.bloomThreshold = 1;
    expect(
      invokeValidator(validateBloomConsistency, cfg).errors.filter((e) =>
        e.includes('bloomThreshold')
      )
    ).toHaveLength(0);
  });

  it('should error when bloomLevels is less than 1', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomLevels = 0;

    const result = invokeValidator(validateBloomConsistency, cfg);

    expect(result.errors).toContainEqual(expect.stringContaining('Invalid bloom.bloomLevels'));
  });

  it('should error when bloomLevels exceeds 12', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomLevels = 13;

    const result = invokeValidator(validateBloomConsistency, cfg);

    expect(result.errors).toContainEqual(expect.stringContaining('Invalid bloom.bloomLevels'));
  });

  it('should accept bloomLevels at boundary values (1 and 12)', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomLevels = 1;
    expect(
      invokeValidator(validateBloomConsistency, cfg).errors.filter((e) => e.includes('bloomLevels'))
    ).toHaveLength(0);

    cfg.renderingControls.defaults.bloomLevels = 12;
    expect(
      invokeValidator(validateBloomConsistency, cfg).errors.filter((e) => e.includes('bloomLevels'))
    ).toHaveLength(0);
  });

  // NaN/Infinity must be caught explicitly — bare comparisons
  // with NaN are always false, so a naked `< 0 || > 10` check
  // would let NaN pass.
  it('NaN bloomStrength is rejected as error', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomStrength = NaN;
    expect(invokeValidator(validateBloomConsistency, cfg).errors).toContainEqual(
      expect.stringContaining('Invalid bloom.bloomStrength')
    );
  });

  it('Infinity bloomRadius is rejected as error', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomRadius = Infinity;
    expect(invokeValidator(validateBloomConsistency, cfg).errors).toContainEqual(
      expect.stringContaining('Invalid bloom.bloomRadius')
    );
  });

  it('NaN bloomThreshold is rejected as error', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomThreshold = NaN;
    expect(invokeValidator(validateBloomConsistency, cfg).errors).toContainEqual(
      expect.stringContaining('Invalid bloom.bloomThreshold')
    );
  });

  it('fractional bloomLevels is rejected as error (must be integer)', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomLevels = 5.5;
    expect(invokeValidator(validateBloomConsistency, cfg).errors).toContainEqual(
      expect.stringContaining('Invalid bloom.bloomLevels')
    );
  });

  it('NaN bloomLevels is rejected as error', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.bloomLevels = NaN;
    expect(invokeValidator(validateBloomConsistency, cfg).errors).toContainEqual(
      expect.stringContaining('Invalid bloom.bloomLevels')
    );
  });
});
