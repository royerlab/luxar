/**
 * Direct unit tests for the pure helpers in resource-lifecycle.ts.
 * Covers the sizing pair, tone-mapping resolver, and DPR-scaled noise
 * computation.
 *
 * GPU-coupled exports (`buildTransientResources`, `createHdrTarget`,
 * `createLdrTarget`, `buildBloomChain`, `disposeTransientResources`)
 * are out of scope here — they instantiate THREE.WebGLRenderTarget /
 * BloomChain which need a renderer. Those are covered by
 * `post-processing-manager-lifecycle.test.ts` and E2E specs.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  computeEffectiveSize,
  getPhysicalSize,
  resolveToneMappingDefault,
  applyScaledNoiseSettings,
  type SizingInputs,
} from '../../../../../rendering/post-processing/post-processing-manager/resource-lifecycle';
import type { LuxarMegaShaderMaterial } from '../../../../../rendering/material-manager';
import type { Renderer } from '../../../../../rendering/renderer-capabilities';

/** Build a minimal Renderer stub exposing only `getPixelRatio`. */
const stubRenderer = (dpr: number): Renderer =>
  ({ getPixelRatio: (): number => dpr }) as unknown as Renderer;

const sizingInputs = (overrides: Partial<SizingInputs> = {}): SizingInputs => ({
  renderer: stubRenderer(1),
  renderSize: { width: 800, height: 600 },
  ssaaEnabled: false,
  ssaaMultiplier: 1,
  ...overrides,
});

describe('computeEffectiveSize', () => {
  it('returns renderSize unchanged when SSAA is off (ignores ssaaMultiplier)', () => {
    expect(computeEffectiveSize(sizingInputs({ ssaaEnabled: false, ssaaMultiplier: 4 }))).toEqual({
      width: 800,
      height: 600,
    });
  });

  it('scales renderSize by ssaaMultiplier when SSAA is on', () => {
    expect(
      computeEffectiveSize(
        sizingInputs({
          renderSize: { width: 800, height: 600 },
          ssaaEnabled: true,
          ssaaMultiplier: 2,
        })
      )
    ).toEqual({ width: 1600, height: 1200 });
  });

  it('rounds to integers for non-integer ssaaMultiplier', () => {
    const out = computeEffectiveSize(
      sizingInputs({
        renderSize: { width: 800, height: 600 },
        ssaaEnabled: true,
        ssaaMultiplier: 1.5,
      })
    );
    expect(Number.isInteger(out.width)).toBe(true);
    expect(Number.isInteger(out.height)).toBe(true);
    expect(out.width).toBe(1200);
    expect(out.height).toBe(900);
  });
});

describe('getPhysicalSize', () => {
  it('returns the effective size when DPR === 1', () => {
    expect(
      getPhysicalSize(
        sizingInputs({
          renderer: stubRenderer(1),
          renderSize: { width: 800, height: 600 },
        })
      )
    ).toEqual({ width: 800, height: 600 });
  });

  it('multiplies the effective size by the renderer pixel ratio', () => {
    expect(
      getPhysicalSize(
        sizingInputs({
          renderer: stubRenderer(2),
          renderSize: { width: 800, height: 600 },
        })
      )
    ).toEqual({ width: 1600, height: 1200 });
  });

  it('stacks SSAA and DPR multiplicatively (SSAA × DPR)', () => {
    expect(
      getPhysicalSize(
        sizingInputs({
          renderer: stubRenderer(2),
          renderSize: { width: 400, height: 300 },
          ssaaEnabled: true,
          ssaaMultiplier: 2,
        })
      )
    ).toEqual({ width: 1600, height: 1200 });
  });

  it('clamps width/height to a minimum of 1 pixel each', () => {
    // Renderer with DPR < 1 + small renderSize would underflow without the floor
    expect(
      getPhysicalSize(
        sizingInputs({
          renderer: stubRenderer(0.5),
          renderSize: { width: 1, height: 1 },
        })
      )
    ).toEqual({ width: 1, height: 1 });
  });

  it('rounds (not floors) to nearest integer for fractional DPR', () => {
    expect(
      getPhysicalSize(
        sizingInputs({
          renderer: stubRenderer(1.5),
          renderSize: { width: 100, height: 100 },
        })
      )
    ).toEqual({ width: 150, height: 150 });
  });
});

describe('resolveToneMappingDefault', () => {
  it('returns a valid THREE.ToneMapping enum value', () => {
    const value = resolveToneMappingDefault();
    const validValues = [
      THREE.NoToneMapping,
      THREE.LinearToneMapping,
      THREE.ReinhardToneMapping,
      THREE.CineonToneMapping,
      THREE.ACESFilmicToneMapping,
      THREE.AgXToneMapping,
      THREE.NeutralToneMapping,
    ];
    expect(validValues).toContain(value);
  });

  it("resolves the current config default ('ACES') to THREE.ACESFilmicToneMapping", () => {
    // The config default is pinned to 'ACES' in
    // src/config/sections/rendering-controls/data.ts. If a future commit
    // changes that default, this test fails so the change becomes deliberate.
    expect(resolveToneMappingDefault()).toBe(THREE.ACESFilmicToneMapping);
  });
});

describe('applyScaledNoiseSettings', () => {
  /** Build a minimal megaShader stub: noise toggle + setter spy. */
  function makeMegaShader(noiseEnabled: boolean) {
    const setDetectorNoise = vi.fn();
    const isDetectorNoiseEnabled = vi.fn(() => noiseEnabled);
    return {
      stub: { setDetectorNoise, isDetectorNoiseEnabled } as unknown as LuxarMegaShaderMaterial,
      setDetectorNoise,
      isDetectorNoiseEnabled,
    };
  }

  it('is a no-op when detector noise is disabled', () => {
    const { stub, setDetectorNoise } = makeMegaShader(false);
    applyScaledNoiseSettings({
      megaShader: stub,
      currentDPRScale: 0.5,
      baseNoiseSettings: { readoutSigma: 0.02, photonGain: 1.0, fpnSigma: 0.01 },
    });
    expect(setDetectorNoise).not.toHaveBeenCalled();
  });

  it('passes the base sigmas through unchanged when DPRScale === 1', () => {
    const { stub, setDetectorNoise } = makeMegaShader(true);
    applyScaledNoiseSettings({
      megaShader: stub,
      currentDPRScale: 1.0,
      baseNoiseSettings: { readoutSigma: 0.02, photonGain: 1.0, fpnSigma: 0.01 },
    });
    expect(setDetectorNoise).toHaveBeenCalledWith({
      readoutSigma: 0.02,
      photonGain: 1.0,
      fpnSigma: 0.01,
    });
  });

  it('scales readoutSigma / fpnSigma linearly with DPRScale and photonGain quadratically', () => {
    const { stub, setDetectorNoise } = makeMegaShader(true);
    applyScaledNoiseSettings({
      megaShader: stub,
      currentDPRScale: 0.5,
      baseNoiseSettings: { readoutSigma: 0.02, photonGain: 1.0, fpnSigma: 0.01 },
    });
    expect(setDetectorNoise).toHaveBeenCalledWith({
      readoutSigma: 0.01, // 0.02 × 0.5
      photonGain: 0.25, // 1.0 × 0.5 × 0.5
      fpnSigma: 0.005, // 0.01 × 0.5
    });
  });

  it('amplifies noise when DPRScale > 1 (matches the model)', () => {
    const { stub, setDetectorNoise } = makeMegaShader(true);
    applyScaledNoiseSettings({
      megaShader: stub,
      currentDPRScale: 2.0,
      baseNoiseSettings: { readoutSigma: 0.02, photonGain: 1.0, fpnSigma: 0.01 },
    });
    expect(setDetectorNoise).toHaveBeenCalledWith({
      readoutSigma: 0.04,
      photonGain: 4.0,
      fpnSigma: 0.02,
    });
  });

  // [rendering.md/G][P5] NaN / Infinity boundary cases. The function does
  // not (and per the docstring's "noise model" should not) clamp these:
  // NaN/Inf inputs propagate verbatim into the shader sigmas. Pinning the
  // contract makes any future clamp / guard introduction explicit.
  it('propagates DPRScale=NaN: all three outputs become NaN', () => {
    const { stub, setDetectorNoise } = makeMegaShader(true);
    applyScaledNoiseSettings({
      megaShader: stub,
      currentDPRScale: Number.NaN,
      baseNoiseSettings: { readoutSigma: 0.02, photonGain: 1.0, fpnSigma: 0.01 },
    });
    expect(setDetectorNoise).toHaveBeenCalledTimes(1);
    const call = setDetectorNoise.mock.calls[0][0];
    expect(Number.isNaN(call.readoutSigma)).toBe(true);
    expect(Number.isNaN(call.photonGain)).toBe(true);
    expect(Number.isNaN(call.fpnSigma)).toBe(true);
  });

  it('propagates DPRScale=+Infinity: photonGain and sigmas are Infinity', () => {
    const { stub, setDetectorNoise } = makeMegaShader(true);
    applyScaledNoiseSettings({
      megaShader: stub,
      currentDPRScale: Number.POSITIVE_INFINITY,
      baseNoiseSettings: { readoutSigma: 0.02, photonGain: 1.0, fpnSigma: 0.01 },
    });
    expect(setDetectorNoise).toHaveBeenCalledTimes(1);
    const call = setDetectorNoise.mock.calls[0][0];
    expect(call.readoutSigma).toBe(Number.POSITIVE_INFINITY);
    expect(call.photonGain).toBe(Number.POSITIVE_INFINITY);
    expect(call.fpnSigma).toBe(Number.POSITIVE_INFINITY);
  });

  it('DPRScale=-Infinity: sigmas → -Inf, photonGain → +Inf (k²)', () => {
    const { stub, setDetectorNoise } = makeMegaShader(true);
    applyScaledNoiseSettings({
      megaShader: stub,
      currentDPRScale: Number.NEGATIVE_INFINITY,
      baseNoiseSettings: { readoutSigma: 0.02, photonGain: 1.0, fpnSigma: 0.01 },
    });
    expect(setDetectorNoise).toHaveBeenCalledTimes(1);
    const call = setDetectorNoise.mock.calls[0][0];
    expect(call.readoutSigma).toBe(Number.NEGATIVE_INFINITY);
    // k² for k = -Infinity is +Infinity.
    expect(call.photonGain).toBe(Number.POSITIVE_INFINITY);
    expect(call.fpnSigma).toBe(Number.NEGATIVE_INFINITY);
  });
});
