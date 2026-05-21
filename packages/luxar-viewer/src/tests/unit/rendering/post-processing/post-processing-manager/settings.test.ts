/**
 * Direct unit tests for the user-toggle setter helpers. Each helper is
 * pure over a mega-shader mock (plus a BloomChain mock for the bloom
 * helpers) — no renderer, no GPU.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  updateBloomSettings,
  clampBloomLevels,
  validateMSAASamples,
  setVignetteEnabled,
  setChromaticLensDistortionEnabled,
  updateChromaticLensDistortion,
  getLensDistortionParams,
  type BloomLiveState,
} from '../../../../../rendering/post-processing/post-processing-manager/settings';
import type { LuxarMegaShaderMaterial } from '../../../../../rendering/material-manager';
import type { BloomChain } from '../../../../../rendering/post-processing/bloom-chain';

/**
 * Build a megaShader stub exposing just the surface the helpers touch.
 * Each spy method is `vi.fn()` so we can assert calls + arguments.
 */
function makeMegaShader(opts: {
  lensEnabled?: boolean;
} = {}) {
  return {
    setVignette: vi.fn(),
    toggleVignette: vi.fn(),
    setLensDistortion: vi.fn(),
    toggleLensDistortion: vi.fn(),
    isLensDistortionEnabled: vi.fn(() => opts.lensEnabled ?? false),
    setBloom: vi.fn(),
    uniforms: {
      uDistortion: { value: new THREE.Vector2(0.1, 0.2) },
      uPrincipalPoint: { value: new THREE.Vector2(0.5, 0.5) },
      uFocalLength: { value: new THREE.Vector2(1.0, 1.0) },
      uSkew: { value: 0 },
      uDispersion: { value: 0 },
    },
  } as unknown as LuxarMegaShaderMaterial;
}

/** Build a BloomChain stub with the surface clampBloomLevels touches. */
function makeBloomChain() {
  const outputTexture = {} as THREE.Texture;
  return {
    setLevels: vi.fn(),
    setRadius: vi.fn(),
    setThreshold: vi.fn(),
    outputTexture,
  } as unknown as BloomChain;
}

// =====================================================================
// validateMSAASamples
// =====================================================================

describe('validateMSAASamples', () => {
  it('accepts the valid sample counts unchanged when within device max', () => {
    expect(validateMSAASamples(0, 16)).toBe(0);
    expect(validateMSAASamples(2, 16)).toBe(2);
    expect(validateMSAASamples(4, 16)).toBe(4);
    expect(validateMSAASamples(8, 16)).toBe(8);
    expect(validateMSAASamples(16, 16)).toBe(16);
  });

  it('substitutes invalid sample counts (not in VALID_MSAA_SAMPLES) with 4', () => {
    expect(validateMSAASamples(3, 16)).toBe(4);
    expect(validateMSAASamples(5, 16)).toBe(4);
    expect(validateMSAASamples(7, 16)).toBe(4);
    expect(validateMSAASamples(1, 16)).toBe(4); // odd
  });

  it('clamps a valid request above the device max down to the max', () => {
    expect(validateMSAASamples(16, 8)).toBe(8);
    expect(validateMSAASamples(8, 4)).toBe(4);
  });

  it('substitution-then-clamp: invalid 3 → 4 → if max=2, clamped to 2', () => {
    // 3 isn't valid → becomes 4 → 4 > maxSamples=2 → clamped to 2.
    expect(validateMSAASamples(3, 2)).toBe(2);
  });
});

// =====================================================================
// clampBloomLevels
// =====================================================================

describe('clampBloomLevels', () => {
  const physSize = { width: 1920, height: 1080 };

  it('returns null when requested rounds to current (no-op)', () => {
    const chain = makeBloomChain();
    const mega = makeMegaShader();
    expect(clampBloomLevels(4, 4, chain, physSize, 1.0, mega)).toBeNull();
    expect(clampBloomLevels(4, 4.2, chain, physSize, 1.0, mega)).toBeNull();
    expect(chain.setLevels).not.toHaveBeenCalled();
  });

  it('clamps requested down to 1 (minimum) and propagates to the chain', () => {
    const chain = makeBloomChain();
    const mega = makeMegaShader();
    const next = clampBloomLevels(5, 0, chain, physSize, 0.7, mega);
    expect(next).toBe(1);
    expect(chain.setLevels).toHaveBeenCalledWith(1, physSize);
    expect(mega.setBloom).toHaveBeenCalledWith(0.7, chain.outputTexture);
  });

  it('clamps requested up to 12 (maximum) and rounds before clamping', () => {
    const chain = makeBloomChain();
    const mega = makeMegaShader();
    const next = clampBloomLevels(5, 100, chain, physSize, 1.0, mega);
    expect(next).toBe(12);
  });

  it('rounds non-integer requested before clamping (12.7 → 13 → clamped to 12)', () => {
    const chain = makeBloomChain();
    const mega = makeMegaShader();
    const next = clampBloomLevels(5, 12.7, chain, physSize, 1.0, mega);
    expect(next).toBe(12);
  });

  it('returns the new value without touching the chain when chain is null (no allocated bloom)', () => {
    const mega = makeMegaShader();
    const next = clampBloomLevels(2, 6, null, physSize, 1.0, mega);
    expect(next).toBe(6);
    expect(mega.setBloom).not.toHaveBeenCalled();
  });
});

// =====================================================================
// updateBloomSettings
// =====================================================================

describe('updateBloomSettings', () => {
  function makeState(overrides: Partial<BloomLiveState> = {}): BloomLiveState {
    return {
      bloomChain: makeBloomChain(),
      bloomIntensity: 1.0,
      bloomRadius: 0.5,
      bloomThreshold: 0.8,
      bloomLevels: 5,
      megaShader: makeMegaShader(),
      ...overrides,
    };
  }

  it('updates strength: mutates bloomIntensity and calls megaShader.setBloom with new value + chain texture', () => {
    const state = makeState();
    updateBloomSettings(state, 2.5);
    expect(state.bloomIntensity).toBe(2.5);
    expect(state.megaShader.setBloom).toHaveBeenCalledWith(2.5, state.bloomChain!.outputTexture);
    expect(state.bloomChain!.setRadius).not.toHaveBeenCalled();
    expect(state.bloomChain!.setThreshold).not.toHaveBeenCalled();
  });

  it('updates radius: mutates bloomRadius and calls chain.setRadius', () => {
    const state = makeState();
    updateBloomSettings(state, undefined, 0.9);
    expect(state.bloomRadius).toBe(0.9);
    expect(state.bloomChain!.setRadius).toHaveBeenCalledWith(0.9);
    expect(state.megaShader.setBloom).not.toHaveBeenCalled();
  });

  it('updates threshold: mutates bloomThreshold and calls chain.setThreshold', () => {
    const state = makeState();
    updateBloomSettings(state, undefined, undefined, 0.4);
    expect(state.bloomThreshold).toBe(0.4);
    expect(state.bloomChain!.setThreshold).toHaveBeenCalledWith(0.4);
  });

  it('handles all three at once', () => {
    const state = makeState();
    updateBloomSettings(state, 1.5, 0.7, 0.6);
    expect(state.bloomIntensity).toBe(1.5);
    expect(state.bloomRadius).toBe(0.7);
    expect(state.bloomThreshold).toBe(0.6);
  });

  it('strength update with null chain passes null texture to setBloom', () => {
    const state = makeState({ bloomChain: null });
    updateBloomSettings(state, 0.9);
    expect(state.megaShader.setBloom).toHaveBeenCalledWith(0.9, null);
  });
});

// =====================================================================
// setVignetteEnabled
// =====================================================================

describe('setVignetteEnabled', () => {
  it('enabled=true with explicit args: setVignette(darkness, offset) + toggleVignette(true)', () => {
    const mega = makeMegaShader();
    setVignetteEnabled(mega, true, 0.7, 0.3);
    expect(mega.setVignette).toHaveBeenCalledWith(0.7, 0.3);
    expect(mega.toggleVignette).toHaveBeenCalledWith(true);
  });

  it('enabled=true with no args: falls back to config defaults (calls setVignette before toggle)', () => {
    const mega = makeMegaShader();
    setVignetteEnabled(mega, true);
    expect(mega.setVignette).toHaveBeenCalledTimes(1);
    expect(mega.toggleVignette).toHaveBeenCalledWith(true);
  });

  it('enabled=false: only toggleVignette(false), no setVignette call', () => {
    const mega = makeMegaShader();
    setVignetteEnabled(mega, false);
    expect(mega.setVignette).not.toHaveBeenCalled();
    expect(mega.toggleVignette).toHaveBeenCalledWith(false);
  });
});

// =====================================================================
// setChromaticLensDistortionEnabled
// =====================================================================

describe('setChromaticLensDistortionEnabled', () => {
  it('enabled=true: setLensDistortion called with Vector2 args + toggleLensDistortion(true)', () => {
    const mega = makeMegaShader();
    setChromaticLensDistortionEnabled(mega, true, {
      distortionX: 0.05,
      distortionY: 0.03,
      dispersion: 0.5,
      principalPointX: 0.5,
      principalPointY: 0.5,
      focalLengthX: 1.2,
      focalLengthY: 1.2,
      skew: 0,
    });
    expect(mega.setLensDistortion).toHaveBeenCalledTimes(1);
    const arg = (mega.setLensDistortion as unknown as ReturnType<typeof vi.fn>).mock.calls[0]![0];
    expect(arg.distortion).toBeInstanceOf(THREE.Vector2);
    expect(arg.distortion.x).toBeCloseTo(0.05);
    expect(arg.distortion.y).toBeCloseTo(0.03);
    expect(arg.principalPoint).toBeInstanceOf(THREE.Vector2);
    expect(arg.focalLength).toBeInstanceOf(THREE.Vector2);
    expect(arg.dispersion).toBeCloseTo(0.5);
    expect(arg.skew).toBe(0);
    expect(mega.toggleLensDistortion).toHaveBeenCalledWith(true);
  });

  it('enabled=true with no params: falls back to config defaults (still calls setLensDistortion + toggle)', () => {
    const mega = makeMegaShader();
    setChromaticLensDistortionEnabled(mega, true);
    expect(mega.setLensDistortion).toHaveBeenCalledTimes(1);
    expect(mega.toggleLensDistortion).toHaveBeenCalledWith(true);
  });

  it('enabled=false: only toggleLensDistortion(false), no setLensDistortion call', () => {
    const mega = makeMegaShader();
    setChromaticLensDistortionEnabled(mega, false);
    expect(mega.setLensDistortion).not.toHaveBeenCalled();
    expect(mega.toggleLensDistortion).toHaveBeenCalledWith(false);
  });
});

// =====================================================================
// updateChromaticLensDistortion
// =====================================================================

describe('updateChromaticLensDistortion', () => {
  it('partial distortion (only X): preserves current Y from uniforms.uDistortion.value', () => {
    const mega = makeMegaShader();
    // uniforms.uDistortion.value defaults to (0.1, 0.2) from makeMegaShader
    updateChromaticLensDistortion(mega, { distortionX: 0.5 });
    const arg = (mega.setLensDistortion as unknown as ReturnType<typeof vi.fn>).mock.calls[0]![0];
    expect(arg.distortion).toBeInstanceOf(THREE.Vector2);
    expect(arg.distortion.x).toBeCloseTo(0.5);
    expect(arg.distortion.y).toBeCloseTo(0.2);
    // principalPoint / focalLength NOT touched (no X or Y supplied)
    expect(arg.principalPoint).toBeUndefined();
    expect(arg.focalLength).toBeUndefined();
  });

  it('forwards scalar fields (skew, dispersion) unchanged', () => {
    const mega = makeMegaShader();
    updateChromaticLensDistortion(mega, { skew: 0.7, dispersion: 1.2 });
    const arg = (mega.setLensDistortion as unknown as ReturnType<typeof vi.fn>).mock.calls[0]![0];
    expect(arg.skew).toBe(0.7);
    expect(arg.dispersion).toBe(1.2);
  });

  it('empty partial → setLensDistortion called with all Vector2 fields undefined', () => {
    const mega = makeMegaShader();
    updateChromaticLensDistortion(mega, {});
    const arg = (mega.setLensDistortion as unknown as ReturnType<typeof vi.fn>).mock.calls[0]![0];
    expect(arg.distortion).toBeUndefined();
    expect(arg.principalPoint).toBeUndefined();
    expect(arg.focalLength).toBeUndefined();
    expect(arg.skew).toBeUndefined();
    expect(arg.dispersion).toBeUndefined();
  });
});

// =====================================================================
// getLensDistortionParams
// =====================================================================

describe('getLensDistortionParams', () => {
  it('returns null when lens distortion is disabled on the mega-shader', () => {
    const mega = makeMegaShader({ lensEnabled: false });
    expect(getLensDistortionParams(mega)).toBeNull();
  });

  it('returns CLONED Vector2 fields (mutating the result does not pollute live uniforms)', () => {
    const mega = makeMegaShader({ lensEnabled: true });
    const params = getLensDistortionParams(mega);
    expect(params).not.toBeNull();
    expect(params!.distortion).toBeInstanceOf(THREE.Vector2);
    // Mutate the returned distortion vector
    params!.distortion.set(99, 99);
    // The live uniform should be unchanged
    expect((mega.uniforms.uDistortion.value as THREE.Vector2).x).toBeCloseTo(0.1);
    expect((mega.uniforms.uDistortion.value as THREE.Vector2).y).toBeCloseTo(0.2);
  });

  it('snapshots scalar fields (skew, focalLength values) when enabled', () => {
    const mega = makeMegaShader({ lensEnabled: true });
    const params = getLensDistortionParams(mega);
    expect(params!.skew).toBe(0);
    expect(params!.focalLength).toBeInstanceOf(THREE.Vector2);
    expect(params!.principalPoint).toBeInstanceOf(THREE.Vector2);
  });
});
