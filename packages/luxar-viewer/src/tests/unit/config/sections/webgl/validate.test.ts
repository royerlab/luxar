/**
 * Tests for validateWebGL (src/config/sections/webgl/validate.ts).
 */

import { describe, it, expect } from 'vitest';
import { validateWebGL } from '../../../../../config/sections/webgl/validate';
import { cloneConfig, invokeValidator } from '../../_fixtures';

describe('validateWebGL', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateWebGL).valid).toBe(true);
  });

  it('should error on invalid powerPreference', () => {
    const cfg = cloneConfig();
    (cfg.webgl.context as any).powerPreference = 'super-power';

    const result = invokeValidator(validateWebGL, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid WebGL powerPreference'));
  });

  it('should error on invalid precision', () => {
    const cfg = cloneConfig();
    (cfg.webgl.renderer as any).precision = 'ultrahighp';

    const result = invokeValidator(validateWebGL, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid WebGL precision'));
  });

  it('should accept all valid precision values', () => {
    const cfg = cloneConfig();
    for (const precision of ['highp', 'mediump', 'lowp']) {
      (cfg.webgl.renderer as any).precision = precision;
      const result = invokeValidator(validateWebGL, cfg);
      expect(result.errors.filter((e) => e.includes('precision'))).toHaveLength(0);
    }
  });

  it('should warn on unusual MSAA samples', () => {
    const cfg = cloneConfig();
    cfg.webgl.renderTarget.samples = 3;

    const result = invokeValidator(validateWebGL, cfg);

    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual MSAA samples'));
  });

  it('should accept standard MSAA sample values', () => {
    const cfg = cloneConfig();
    for (const samples of [0, 2, 4, 8]) {
      cfg.webgl.renderTarget.samples = samples;
      const result = invokeValidator(validateWebGL, cfg);
      expect(result.warnings.filter((w) => w.includes('MSAA'))).toHaveLength(0);
    }
  });

  it('should warn on unusual colorSpace', () => {
    const cfg = cloneConfig();
    cfg.webgl.context.colorSpace = 'adobe-rgb';

    const result = invokeValidator(validateWebGL, cfg);

    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual color space'));
  });

  it('should accept valid colorSpace values', () => {
    const cfg = cloneConfig();
    for (const colorSpace of ['srgb', 'display-p3', 'rec2020']) {
      cfg.webgl.context.colorSpace = colorSpace;
      const result = invokeValidator(validateWebGL, cfg);
      expect(result.warnings.filter((w) => w.includes('color space'))).toHaveLength(0);
    }
  });
});
