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

  // `colorSpace` was removed from the context attributes: it is not a
  // WebGL context-attribute key (drawing-buffer color space is
  // `gl.drawingBufferColorSpace`) and the old 'display-p3' entry was
  // silently ignored. Output color handling lives in the HDR pipeline.

  // [G16][P5] Audit: pre-audit `samples=3` was the only "unusual" case.
  // The source uses `Array.includes` which has identity-comparison
  // quirks: NaN.includes returns false in older runtimes but true on
  // SameValueZero — assert the production behavior explicitly.
  // Non-integers (2.5, 4.0001) and negative values must also warn.
  it('warns on NaN MSAA samples (not in valid list)', () => {
    const cfg = cloneConfig();
    cfg.webgl.renderTarget.samples = NaN;
    const result = invokeValidator(validateWebGL, cfg);
    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual MSAA samples'));
  });

  it('warns on non-integer MSAA samples (2.5)', () => {
    const cfg = cloneConfig();
    cfg.webgl.renderTarget.samples = 2.5;
    const result = invokeValidator(validateWebGL, cfg);
    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual MSAA samples'));
  });

  it('warns on non-integer MSAA samples close-but-not-equal to valid (4.0001)', () => {
    // [P5] boundary: 4 is valid; 4.0001 must not be silently accepted.
    const cfg = cloneConfig();
    cfg.webgl.renderTarget.samples = 4.0001;
    const result = invokeValidator(validateWebGL, cfg);
    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual MSAA samples'));
  });

  it('warns on negative MSAA samples', () => {
    const cfg = cloneConfig();
    cfg.webgl.renderTarget.samples = -1;
    const result = invokeValidator(validateWebGL, cfg);
    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual MSAA samples'));
  });

  it('warns on Infinity MSAA samples', () => {
    const cfg = cloneConfig();
    cfg.webgl.renderTarget.samples = Number.POSITIVE_INFINITY;
    const result = invokeValidator(validateWebGL, cfg);
    expect(result.warnings).toContainEqual(expect.stringContaining('Unusual MSAA samples'));
  });
});
