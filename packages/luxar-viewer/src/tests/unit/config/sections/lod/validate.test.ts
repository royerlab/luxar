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

  it('rejects an exit band wider than 0.5 or a preload band past a narrowed exit band', () => {
    const cfg = cloneConfig();
    cfg.lod.preloadExitBandFraction = 0.6;
    expect(invokeValidator(validateLod, cfg).valid).toBe(false);
    cfg.lod.preloadExitBandFraction = 0.3; // the default 0.4 entry band now exceeds it
    const result = invokeValidator(validateLod, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('lod.preloadBandFraction'));
  });

  it('rejects a playback keep budget tighter than the admission budget', () => {
    const cfg = cloneConfig();
    cfg.lod.playbackKeepBudgetFraction = 0.5; // below the 0.8 load budget
    const result = invokeValidator(validateLod, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('lod.playbackKeepBudgetFraction'));
  });

  it.each([
    ['fineReloadSettleMs', -1],
    ['loadEwmaAlpha', 0],
    ['loadEwmaAlpha', 1.5],
    ['playbackProbeIntervalMs', Number.NaN],
    ['staleHoldMs', -5],
    ['staleHoldMinRatio', 2],
    ['failedRetryMs', Number.POSITIVE_INFINITY],
    ['lazyActivationRequestTimeoutMs', 0],
    ['partitionFrustumMargin', -0.1],
    ['hysteresisRatio', 1],
    ['maxMedianFootprintPx', 0],
  ] as const)('rejects lod.%s = %s', (key, value) => {
    const cfg = cloneConfig();
    cfg.lod[key] = value;
    const result = invokeValidator(validateLod, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining(`lod.${key}`));
  });
});
