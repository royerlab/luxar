/**
 * Tests for validateLod (src/config/sections/lod/validate.ts).
 */

import { describe, it, expect } from 'vitest';
import { validateLod } from '../../../../../config/sections/lod/validate';
import { cloneConfig, invokeValidator } from '../../_fixtures';

describe('validateLod', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateLod).valid).toBe(true);
  });

  it('accepts a preload band just inside the exit band', () => {
    const cfg = cloneConfig();
    cfg.lod.preloadBandFraction = 0.49;
    expect(invokeValidator(validateLod, cfg).valid).toBe(true);
  });

  // #2944 review B: the exit band (a visit ends only once the metric leaves
  // it) is fixed at 0.5. An entry band as wide as the exit band leaves no
  // hysteresis between them, so a camera hovering at the edge starts a new
  // visit (and a reload) per wobble.
  it('rejects a preload band as wide as the exit band', () => {
    const cfg = cloneConfig();
    cfg.lod.preloadBandFraction = 0.5;
    const result = invokeValidator(validateLod, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('lod.preloadBandFraction'));
  });

  it('rejects a negative or non-finite fade', () => {
    const cfg = cloneConfig();
    cfg.lod.fadeMs = -1;
    expect(invokeValidator(validateLod, cfg).valid).toBe(false);
    cfg.lod.fadeMs = Number.NaN;
    expect(invokeValidator(validateLod, cfg).valid).toBe(false);
  });
});
