/**
 * Unit tests for LuxarToneMappingEffect.
 *
 * Tests the public API around the vendored pmndrs ToneMappingEffect:
 * constructor defaults, mode set/get, and the EOG (Exposure / global
 * Offset / global Gamma) getter/setter pairs. The shader itself and
 * the luminance-pass plumbing are exercised by the e2e visual suite.
 */

import { describe, it, expect } from 'vitest';
import { ToneMappingMode } from 'postprocessing';
import { LuxarToneMappingEffect } from '../../../../rendering/post-processing/luxar-tone-mapping-effect';

describe('LuxarToneMappingEffect — defaults', () => {
  it('uses AGX mode by default', () => {
    const fx = new LuxarToneMappingEffect();
    expect(fx.mode).toBe(ToneMappingMode.AGX);
  });

  it('starts EOG at neutral (0, 0, 1)', () => {
    const fx = new LuxarToneMappingEffect();
    expect(fx.exposure).toBe(0);
    expect(fx.globalOffset).toBe(0);
    expect(fx.globalGamma).toBe(1);
  });

  it('accepts explicit EOG values in the constructor', () => {
    const fx = new LuxarToneMappingEffect({
      exposure: 1.5,
      globalOffset: -0.1,
      globalGamma: 2.2,
    });
    expect(fx.exposure).toBe(1.5);
    expect(fx.globalOffset).toBe(-0.1);
    expect(fx.globalGamma).toBe(2.2);
  });

  it('clamps a 0 globalGamma constructor input to 0.001 (avoids div-by-zero in shader)', () => {
    const fx = new LuxarToneMappingEffect({ globalGamma: 0 });
    expect(fx.globalGamma).toBe(0.001);
  });
});

describe('LuxarToneMappingEffect — EOG setters', () => {
  it('exposure setter mutates the uniform', () => {
    const fx = new LuxarToneMappingEffect();
    fx.exposure = 2;
    expect(fx.exposure).toBe(2);
    fx.exposure = -1;
    expect(fx.exposure).toBe(-1);
  });

  it('globalOffset setter mutates the uniform', () => {
    const fx = new LuxarToneMappingEffect();
    fx.globalOffset = 0.5;
    expect(fx.globalOffset).toBe(0.5);
    fx.globalOffset = -0.5;
    expect(fx.globalOffset).toBe(-0.5);
  });

  it('globalGamma setter clamps to ≥ 0.001', () => {
    const fx = new LuxarToneMappingEffect();
    fx.globalGamma = 0;
    expect(fx.globalGamma).toBe(0.001);
    fx.globalGamma = -1;
    expect(fx.globalGamma).toBe(0.001);
    fx.globalGamma = 1.5;
    expect(fx.globalGamma).toBe(1.5);
  });
});

describe('LuxarToneMappingEffect — mode switching', () => {
  it('switches mode and reflects in the getter', () => {
    const fx = new LuxarToneMappingEffect();
    fx.mode = ToneMappingMode.REINHARD;
    expect(fx.mode).toBe(ToneMappingMode.REINHARD);
    fx.mode = ToneMappingMode.NEUTRAL;
    expect(fx.mode).toBe(ToneMappingMode.NEUTRAL);
  });

  it('setting mode to current value is a no-op (idempotent)', () => {
    const fx = new LuxarToneMappingEffect({ mode: ToneMappingMode.AGX });
    expect(() => {
      fx.mode = ToneMappingMode.AGX;
      fx.mode = ToneMappingMode.AGX;
    }).not.toThrow();
    expect(fx.mode).toBe(ToneMappingMode.AGX);
  });
});

describe('LuxarToneMappingEffect — adaptive helpers', () => {
  it('adaptive getter is true iff mode is REINHARD2_ADAPTIVE', () => {
    const fx = new LuxarToneMappingEffect();
    fx.mode = ToneMappingMode.REINHARD2;
    expect(fx.adaptive).toBe(false);
    fx.mode = ToneMappingMode.REINHARD2_ADAPTIVE;
    expect(fx.adaptive).toBe(true);
    fx.mode = ToneMappingMode.AGX;
    expect(fx.adaptive).toBe(false);
  });

  it('adaptive setter switches between REINHARD2 and REINHARD2_ADAPTIVE', () => {
    const fx = new LuxarToneMappingEffect();
    fx.adaptive = true;
    expect(fx.mode).toBe(ToneMappingMode.REINHARD2_ADAPTIVE);
    fx.adaptive = false;
    expect(fx.mode).toBe(ToneMappingMode.REINHARD2);
  });
});

describe('LuxarToneMappingEffect — pmndrs passthrough setters', () => {
  it('whitePoint round-trips through the uniform', () => {
    const fx = new LuxarToneMappingEffect();
    fx.whitePoint = 4.0;
    expect(fx.whitePoint).toBe(4.0);
    fx.whitePoint = 16.0;
    expect(fx.whitePoint).toBe(16.0);
  });

  it('middleGrey round-trips through the uniform', () => {
    const fx = new LuxarToneMappingEffect();
    fx.middleGrey = 0.18;
    expect(fx.middleGrey).toBe(0.18);
    fx.middleGrey = 0.5;
    expect(fx.middleGrey).toBe(0.5);
  });

  it('averageLuminance round-trips through the uniform', () => {
    const fx = new LuxarToneMappingEffect();
    fx.averageLuminance = 1.0;
    expect(fx.averageLuminance).toBe(1.0);
    fx.averageLuminance = 0.25;
    expect(fx.averageLuminance).toBe(0.25);
  });

  it('adaptationRate round-trips through the underlying material', () => {
    const fx = new LuxarToneMappingEffect();
    fx.adaptationRate = 1.5;
    expect(fx.adaptationRate).toBe(1.5);
    fx.adaptationRate = 0.5;
    expect(fx.adaptationRate).toBe(0.5);
  });

  it('resolution setter rounds up to the next power of two', () => {
    const fx = new LuxarToneMappingEffect();
    // resolution=300 → exponent=9 → size=512
    fx.resolution = 300;
    expect(fx.resolution).toBe(512);
    // resolution=128 → exponent=7 → size=128
    fx.resolution = 128;
    expect(fx.resolution).toBe(128);
    // resolution=1 → exponent=0 → size=1
    fx.resolution = 1;
    expect(fx.resolution).toBe(1);
  });
});
