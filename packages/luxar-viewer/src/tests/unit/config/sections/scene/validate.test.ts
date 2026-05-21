/**
 * Tests for validateScene (src/config/sections/scene/validate.ts).
 */

import { describe, it, expect } from 'vitest';
import { validateScene } from '../../../../../config/sections/scene/validate';
import { cloneConfig, invokeValidator } from '../../_fixtures';

describe('validateScene', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateScene).valid).toBe(true);
  });

  it('should error when backgroundColor is negative', () => {
    const cfg = cloneConfig();
    cfg.scene.backgroundColor = -1;

    const result = invokeValidator(validateScene, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid scene background color'));
  });

  it('should error when backgroundColor exceeds 0xffffff', () => {
    const cfg = cloneConfig();
    cfg.scene.backgroundColor = 0x1000000;

    const result = invokeValidator(validateScene, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid scene background color'));
  });

  it('should accept backgroundColor at boundaries (0x000000 and 0xffffff)', () => {
    const cfg = cloneConfig();
    cfg.scene.backgroundColor = 0x000000;
    expect(
      invokeValidator(validateScene, cfg).errors.filter((e) => e.includes('background color'))
    ).toHaveLength(0);

    cfg.scene.backgroundColor = 0xffffff;
    expect(
      invokeValidator(validateScene, cfg).errors.filter((e) => e.includes('background color'))
    ).toHaveLength(0);
  });

  it('should error when fitRatio is zero', () => {
    const cfg = cloneConfig();
    cfg.scene.defaultFitRatio = 0;

    const result = invokeValidator(validateScene, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid scene fit ratio'));
  });

  it('should error when fitRatio is negative', () => {
    const cfg = cloneConfig();
    cfg.scene.defaultFitRatio = -0.5;

    const result = invokeValidator(validateScene, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid scene fit ratio'));
  });

  it('should error when fitRatio exceeds 1', () => {
    const cfg = cloneConfig();
    cfg.scene.defaultFitRatio = 1.5;

    const result = invokeValidator(validateScene, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid scene fit ratio'));
  });

  it('should accept fitRatio of exactly 1', () => {
    const cfg = cloneConfig();
    cfg.scene.defaultFitRatio = 1;

    const result = invokeValidator(validateScene, cfg);

    expect(result.errors.filter((e) => e.includes('fit ratio'))).toHaveLength(0);
  });

  it('rejects NaN backgroundColor', () => {
    const cfg = cloneConfig();
    cfg.scene.backgroundColor = NaN;
    const result = invokeValidator(validateScene, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid scene background color'));
  });

  it('rejects non-integer backgroundColor', () => {
    const cfg = cloneConfig();
    cfg.scene.backgroundColor = 1.5;
    const result = invokeValidator(validateScene, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid scene background color'));
  });

  it('rejects NaN defaultFitRatio', () => {
    const cfg = cloneConfig();
    cfg.scene.defaultFitRatio = NaN;
    const result = invokeValidator(validateScene, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid scene fit ratio'));
  });
});
