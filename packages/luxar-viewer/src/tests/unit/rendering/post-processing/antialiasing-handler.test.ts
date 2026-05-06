/**
 * Unit tests for the antialiasing-handler helpers.
 */

import { describe, it, expect } from 'vitest';
import { SMAAPreset } from 'postprocessing';
import {
  VALID_MSAA_SAMPLES,
  checkMSAACapability,
  clampSSAAMultiplier,
  mapSMAAPreset,
  validateMSAASamples,
} from '../../../../rendering/post-processing/antialiasing-handler';

describe('mapSMAAPreset', () => {
  it('maps each named preset to its pmndrs enum value', () => {
    expect(mapSMAAPreset('LOW')).toBe(SMAAPreset.LOW);
    expect(mapSMAAPreset('MEDIUM')).toBe(SMAAPreset.MEDIUM);
    expect(mapSMAAPreset('HIGH')).toBe(SMAAPreset.HIGH);
    expect(mapSMAAPreset('ULTRA')).toBe(SMAAPreset.ULTRA);
  });

  it('falls back to HIGH for undefined or unknown names', () => {
    expect(mapSMAAPreset(undefined)).toBe(SMAAPreset.HIGH);
    expect(mapSMAAPreset('GARBAGE' as 'LOW')).toBe(SMAAPreset.HIGH);
  });
});

describe('VALID_MSAA_SAMPLES', () => {
  it('contains the supported sample counts', () => {
    expect(VALID_MSAA_SAMPLES).toEqual([0, 2, 4, 8, 16]);
  });
});

describe('validateMSAASamples', () => {
  it('passes valid samples within the GPU max unchanged', () => {
    expect(validateMSAASamples(4, 16)).toEqual({ samples: 4 });
    expect(validateMSAASamples(0, 16)).toEqual({ samples: 0 });
    expect(validateMSAASamples(16, 16)).toEqual({ samples: 16 });
  });

  it('falls back to 4 when the requested count is invalid, and reports it', () => {
    const result = validateMSAASamples(3, 16);
    expect(result.samples).toBe(4);
    expect(result.warning).toContain('Invalid MSAA samples: 3');
    expect(result.warning).toContain('Using 4');
  });

  it('falls back to 4 then clamps to GPU max when both rules trip', () => {
    // 3 is invalid → 4. Then GPU max is 2 → clamped to 2.
    const result = validateMSAASamples(3, 2);
    expect(result.samples).toBe(2);
    // The GPU-clamp warning is the most recent / overrides.
    expect(result.warning).toContain('only supports 2');
  });

  it('clamps to GPU max with an explanatory warning', () => {
    const result = validateMSAASamples(16, 4);
    expect(result.samples).toBe(4);
    expect(result.warning).toContain('Requested 16');
    expect(result.warning).toContain('only supports 4');
    expect(result.warning).toContain('Using 4');
  });

  it('omits warning when no adjustment was needed', () => {
    expect(validateMSAASamples(8, 16).warning).toBeUndefined();
  });
});

describe('clampSSAAMultiplier', () => {
  it('clamps to [1.0, 4.0]', () => {
    expect(clampSSAAMultiplier(0)).toBe(1.0);
    expect(clampSSAAMultiplier(0.5)).toBe(1.0);
    expect(clampSSAAMultiplier(1.0)).toBe(1.0);
    expect(clampSSAAMultiplier(2.0)).toBe(2.0);
    expect(clampSSAAMultiplier(4.0)).toBe(4.0);
    expect(clampSSAAMultiplier(5.0)).toBe(4.0);
    expect(clampSSAAMultiplier(-1)).toBe(1.0);
  });

  it('preserves decimal multipliers within range', () => {
    expect(clampSSAAMultiplier(1.5)).toBe(1.5);
    expect(clampSSAAMultiplier(3.25)).toBe(3.25);
  });
});

describe('checkMSAACapability', () => {
  function makeFakeGL(maxSamples: number, hasFloatExt: boolean): WebGL2RenderingContext {
    return {
      MAX_SAMPLES: 0x8d57,
      getParameter: (p: number) => (p === 0x8d57 ? maxSamples : 0),
      getExtension: (name: string) => (name === 'EXT_color_buffer_float' && hasFloatExt ? {} : null),
    } as unknown as WebGL2RenderingContext;
  }

  it('returns supported=true when MAX_SAMPLES >= 2', () => {
    expect(checkMSAACapability(makeFakeGL(4, true))).toEqual({
      supported: true,
      maxSamples: 4,
      floatBuffersOK: true,
    });
  });

  it('returns supported=false when MAX_SAMPLES < 2', () => {
    expect(checkMSAACapability(makeFakeGL(0, false))).toEqual({
      supported: false,
      maxSamples: 0,
      floatBuffersOK: false,
    });
    expect(checkMSAACapability(makeFakeGL(1, true))).toEqual({
      supported: false,
      maxSamples: 1,
      floatBuffersOK: true,
    });
  });

  it('reports floatBuffersOK separately from MAX_SAMPLES', () => {
    expect(checkMSAACapability(makeFakeGL(16, false))).toEqual({
      supported: true,
      maxSamples: 16,
      floatBuffersOK: false,
    });
  });
});
