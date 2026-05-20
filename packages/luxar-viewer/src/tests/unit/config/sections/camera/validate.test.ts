/**
 * Tests for validateCamera (src/config/sections/camera/validate.ts).
 *
 * Migrated from the monolithic validation.test.ts dispatcher tests; each test
 * exercises the per-section validator directly via invokeValidator().
 */

import { describe, it, expect } from 'vitest';
import { validateCamera } from '../../../../../config/sections/camera/validate';
import { cloneConfig, invokeValidator } from '../../_fixtures';

describe('validateCamera', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateCamera).valid).toBe(true);
  });

  it('should error when FOV is less than 1', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.fov = 0;

    const result = invokeValidator(validateCamera, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid camera FOV'));
  });

  it('should error when FOV is greater than 180', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.fov = 200;

    const result = invokeValidator(validateCamera, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid camera FOV'));
  });

  it('should accept FOV at boundary values (1 and 180)', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.fov = 1;
    expect(
      invokeValidator(validateCamera, cfg).errors.filter((e) => e.includes('FOV'))
    ).toHaveLength(0);

    cfg.renderingControls.defaults.fov = 180;
    expect(
      invokeValidator(validateCamera, cfg).errors.filter((e) => e.includes('FOV'))
    ).toHaveLength(0);
  });

  it('should error when near plane is zero', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.near = 0;

    const result = invokeValidator(validateCamera, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid camera near plane'));
  });

  it('should error when near plane is negative', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.near = -1;

    const result = invokeValidator(validateCamera, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid camera near plane'));
  });

  it('should error when far plane is less than or equal to near', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.near = 10;
    cfg.renderingControls.defaults.far = 5;

    const result = invokeValidator(validateCamera, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid camera far plane'));
  });

  it('should error when far plane equals near', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.near = 10;
    cfg.renderingControls.defaults.far = 10;

    const result = invokeValidator(validateCamera, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid camera far plane'));
  });

  it('should error when fovMin is greater than or equal to fovMax', () => {
    const cfg = cloneConfig();
    cfg.camera.fovMin = 170;
    cfg.camera.fovMax = 10;

    const result = invokeValidator(validateCamera, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid FOV limits'));
  });

  it('should error when fovMin equals fovMax', () => {
    const cfg = cloneConfig();
    cfg.camera.fovMin = 50;
    cfg.camera.fovMax = 50;

    const result = invokeValidator(validateCamera, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid FOV limits'));
  });

  it('should warn on unusual fovSensitivity (zero)', () => {
    const cfg = cloneConfig();
    cfg.camera.fovSensitivity = 0;

    const result = invokeValidator(validateCamera, cfg);

    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual FOV sensitivity'));
  });

  it('should warn on unusual fovSensitivity (negative)', () => {
    const cfg = cloneConfig();
    cfg.camera.fovSensitivity = -0.1;

    const result = invokeValidator(validateCamera, cfg);

    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual FOV sensitivity'));
  });

  it('should warn on fovSensitivity greater than 1', () => {
    const cfg = cloneConfig();
    cfg.camera.fovSensitivity = 1.5;

    const result = invokeValidator(validateCamera, cfg);

    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual FOV sensitivity'));
  });

  it('should not warn on fovSensitivity exactly 1', () => {
    const cfg = cloneConfig();
    cfg.camera.fovSensitivity = 1;

    const result = invokeValidator(validateCamera, cfg);

    // fovSensitivity = 1 means <= 0 is false and > 1 is false, so no warning
    expect(result.warnings.filter((w) => w.includes('FOV sensitivity'))).toHaveLength(0);
  });

  // NaN/Infinity hardening — bare comparisons with NaN are always false, so
  // naked range checks would let NaN pass silently.
  it('rejects NaN camera FOV', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.fov = NaN;
    const result = invokeValidator(validateCamera, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid camera FOV'));
  });

  it('rejects Infinity camera near plane', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.near = Infinity;
    const result = invokeValidator(validateCamera, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid camera near plane'));
  });

  it('rejects NaN camera fovMin', () => {
    const cfg = cloneConfig();
    cfg.camera.fovMin = NaN;
    const result = invokeValidator(validateCamera, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid FOV limits'));
  });
});
