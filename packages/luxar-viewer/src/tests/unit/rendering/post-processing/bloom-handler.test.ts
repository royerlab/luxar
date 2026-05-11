/**
 * Unit tests for the bloom-handler helpers.
 *
 * Pure helpers (resolve, build, clamp, read, applyRadius) plus the
 * settings-update path. The handler is structurally typed so all tests
 * can use plain object stubs — no `EffectComposer`, no real
 * `BloomEffect`.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BlendFunction, KernelSize } from 'postprocessing';
import {
  applyBloomRadius,
  applyBloomSettings,
  buildBloomConstructorOptions,
  clampBloomLevels,
  getDefaultBloomSettings,
  readBloomSettings,
  resolveBloomSettings,
} from '../../../../rendering/post-processing/bloom-handler';

interface BloomStub {
  intensity: number;
  mipmapBlurPass?: { radius: number };
  luminanceMaterial?: { threshold: number };
}

function makeBloomStub(overrides: Partial<BloomStub> = {}): BloomStub {
  return {
    intensity: 0.7,
    mipmapBlurPass: { radius: 0.4 },
    luminanceMaterial: { threshold: 0.85 },
    ...overrides,
  };
}

describe('clampBloomLevels', () => {
  it('clamps to [1, 12] and rounds', () => {
    expect(clampBloomLevels(0)).toBe(1);
    expect(clampBloomLevels(0.4)).toBe(1);
    expect(clampBloomLevels(13)).toBe(12);
    expect(clampBloomLevels(-5)).toBe(1);
    expect(clampBloomLevels(7.6)).toBe(8);
    expect(clampBloomLevels(7)).toBe(7);
  });
});

describe('getDefaultBloomSettings', () => {
  it('returns numeric defaults for all three knobs', () => {
    const d = getDefaultBloomSettings();
    expect(typeof d.intensity).toBe('number');
    expect(typeof d.radius).toBe('number');
    expect(typeof d.threshold).toBe('number');
  });
});

describe('resolveBloomSettings', () => {
  it('falls back to defaults for any undefined field', () => {
    const base = { intensity: 1.5, radius: 0.5, threshold: 0.9 };
    expect(resolveBloomSettings({}, base)).toEqual(base);
    expect(resolveBloomSettings({ intensity: 2 }, base)).toEqual({
      intensity: 2,
      radius: 0.5,
      threshold: 0.9,
    });
    expect(resolveBloomSettings({ radius: 0.1 }, base)).toEqual({
      intensity: 1.5,
      radius: 0.1,
      threshold: 0.9,
    });
    expect(resolveBloomSettings({ threshold: 0.6 }, base)).toEqual({
      intensity: 1.5,
      radius: 0.5,
      threshold: 0.6,
    });
  });

  it('uses config defaults when no base is provided', () => {
    const result = resolveBloomSettings({ intensity: 2 });
    expect(result.intensity).toBe(2);
    expect(typeof result.radius).toBe('number');
    expect(typeof result.threshold).toBe('number');
  });
});

describe('readBloomSettings', () => {
  it('reads back the live values from the effect', () => {
    const effect = makeBloomStub({
      intensity: 1.2,
      mipmapBlurPass: { radius: 0.6 },
      luminanceMaterial: { threshold: 0.5 },
    });
    expect(readBloomSettings(effect, { intensity: 0, radius: 0, threshold: 0 })).toEqual({
      intensity: 1.2,
      radius: 0.6,
      threshold: 0.5,
    });
  });

  it('falls back to defaults for missing sub-objects', () => {
    const effect: BloomStub = { intensity: 0 };
    const defaults = { intensity: 1, radius: 0.4, threshold: 0.85 };
    expect(readBloomSettings(effect, defaults)).toEqual({
      intensity: 1, // 0 || default ⇒ default
      radius: 0.4,
      threshold: 0.85,
    });
  });
});

describe('buildBloomConstructorOptions', () => {
  it('includes the constants pmndrs requires plus the resolved settings', () => {
    const opts = buildBloomConstructorOptions({ intensity: 1, radius: 0.4, threshold: 0.7 }, 6);
    expect(opts).toEqual({
      intensity: 1,
      luminanceThreshold: 0.7,
      luminanceSmoothing: 0.01,
      mipmapBlur: true,
      kernelSize: KernelSize.LARGE,
      blendFunction: BlendFunction.ADD,
      levels: 6,
    });
  });
});

describe('applyBloomRadius', () => {
  it('writes radius onto mipmapBlurPass and returns true', () => {
    const effect = makeBloomStub();
    expect(applyBloomRadius(effect, 0.9)).toBe(true);
    expect(effect.mipmapBlurPass?.radius).toBe(0.9);
  });

  it('returns false when mipmapBlurPass is missing', () => {
    expect(applyBloomRadius({}, 0.5)).toBe(false);
  });
});

describe('applyBloomSettings', () => {
  beforeEach(() => {
    // Silence the log calls — we're not asserting on them here.
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  it('returns null when the effect is null/undefined', () => {
    expect(applyBloomSettings(null, { intensity: 1 })).toBeNull();
    expect(applyBloomSettings(undefined, { intensity: 1 })).toBeNull();
  });

  it('returns null when the effect does not look like a BloomEffect', () => {
    const notBloom = { weird: 1 } as unknown as Parameters<typeof applyBloomSettings>[0];
    expect(applyBloomSettings(notBloom, { intensity: 1 })).toBeNull();
  });

  it('mutates only the fields that were provided', () => {
    const effect = makeBloomStub({
      intensity: 1,
      mipmapBlurPass: { radius: 0.3 },
      luminanceMaterial: { threshold: 0.8 },
    });
    const out = applyBloomSettings(effect as unknown as Parameters<typeof applyBloomSettings>[0], {
      intensity: 2,
    });
    expect(out).toEqual({ intensity: 2, radius: 0.3, threshold: 0.8 });
    expect(effect.intensity).toBe(2);
    expect(effect.mipmapBlurPass?.radius).toBe(0.3);
    expect(effect.luminanceMaterial?.threshold).toBe(0.8);
  });

  it('mutates radius and threshold when provided', () => {
    const effect = makeBloomStub({
      intensity: 1,
      mipmapBlurPass: { radius: 0.3 },
      luminanceMaterial: { threshold: 0.8 },
    });
    const out = applyBloomSettings(effect as unknown as Parameters<typeof applyBloomSettings>[0], {
      radius: 0.6,
      threshold: 0.4,
    });
    expect(out).toEqual({ intensity: 1, radius: 0.6, threshold: 0.4 });
    expect(effect.mipmapBlurPass?.radius).toBe(0.6);
    expect(effect.luminanceMaterial?.threshold).toBe(0.4);
  });

  it('skips radius/threshold writes when sub-objects are missing', () => {
    const effect: BloomStub = { intensity: 1, mipmapBlurPass: { radius: 0 } };
    const out = applyBloomSettings(effect as unknown as Parameters<typeof applyBloomSettings>[0], {
      radius: 0.6,
      threshold: 0.4,
    });
    expect(out).toEqual({ intensity: 1, radius: 0.6, threshold: 0 });
    expect(effect.mipmapBlurPass?.radius).toBe(0.6);
  });
});
