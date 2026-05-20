/**
 * Tests for validateControls (src/config/sections/controls/validate.ts).
 */

import { describe, it, expect } from 'vitest';
import { validateControls } from '../../../../../config/sections/controls/validate';
import { cloneConfig, invokeValidator } from '../../_fixtures';

describe('validateControls', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateControls).valid).toBe(true);
  });

  it('should error when ConfigRange min >= max', () => {
    const cfg = cloneConfig();
    cfg.controls.fly.movement.speed.min = 50;
    cfg.controls.fly.movement.speed.max = 0.5;

    const result = invokeValidator(validateControls, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('fly.movement.speed'));
  });

  it('should error when ConfigRange default is outside [min, max]', () => {
    const cfg = cloneConfig();
    cfg.controls.orbit.damping.factor.default = 99;

    const result = invokeValidator(validateControls, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('orbit.damping.factor'));
  });

  it('should pass with default controls config', () => {
    const result = invokeValidator(validateControls);

    expect(result.errors.filter((e) => e.includes('controls.'))).toHaveLength(0);
  });

  it('rejects NaN in a controls range field (fly.movement.speed.default)', () => {
    const cfg = cloneConfig();
    cfg.controls.fly.movement.speed.default = NaN;
    const result = invokeValidator(validateControls, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('non-finite values'));
  });
});
