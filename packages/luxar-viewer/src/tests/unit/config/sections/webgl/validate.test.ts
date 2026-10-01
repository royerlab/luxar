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

  // `colorSpace` was removed from the context attributes: it is not a
  // WebGL context-attribute key (drawing-buffer color space is
  // `gl.drawingBufferColorSpace`) and the old 'display-p3' entry was
  // silently ignored. Output color handling lives in the HDR pipeline.

  // MSAA lives in renderingControls (msaaEnabled / msaaSamples, clamped by
  // clampMSAASamples); the old webgl.renderTarget.samples was never read.
  it('carries no render-target block (MSAA is a rendering-controls setting)', () => {
    expect('renderTarget' in cloneConfig().webgl).toBe(false);
  });
});
