/**
 * Unit tests for the context-recovery capture/apply helpers.
 *
 * Each effect's state is captured into a plain JS object and re-
 * applied to a (possibly different) target. Tests round-trip a
 * stub effect through the capture → apply pair to confirm fidelity.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { ToneMappingMode } from 'postprocessing';
import {
  applyBloomState,
  applyChromaticLensDistortionState,
  applyDOFFocusDistance,
  applyDetectorNoiseState,
  applyToneMappingState,
  applyVignetteState,
  captureBloomState,
  captureChromaticLensDistortionState,
  captureDOFState,
  captureDetectorNoiseState,
  captureToneMappingState,
  captureVignetteState,
} from '../../../../rendering/post-processing/context-recovery';

describe('captureBloomState / applyBloomState', () => {
  it('returns null when the effect is absent', () => {
    expect(captureBloomState(null)).toBeNull();
    expect(captureBloomState(undefined)).toBeNull();
  });

  it('captures intensity / threshold / radius', () => {
    const state = captureBloomState({
      intensity: 1.5,
      mipmapBlurPass: { radius: 0.6 },
      luminanceMaterial: { threshold: 0.85 },
    });
    expect(state).toEqual({ intensity: 1.5, luminanceThreshold: 0.85, radius: 0.6 });
  });

  it('round-trips through apply (target gets identical values)', () => {
    const source = {
      intensity: 1.5,
      mipmapBlurPass: { radius: 0.6 },
      luminanceMaterial: { threshold: 0.85 },
    };
    const target = {
      intensity: 0,
      mipmapBlurPass: { radius: 0 },
      luminanceMaterial: { threshold: 0 },
    };
    applyBloomState(target, captureBloomState(source));
    expect(target).toEqual(source);
  });

  it('apply is a no-op for null effect or null state', () => {
    expect(() => applyBloomState(null, null)).not.toThrow();
    expect(() =>
      applyBloomState(null, { intensity: 1, luminanceThreshold: 0.5, radius: 0.3 })
    ).not.toThrow();
    const target = { intensity: 99, mipmapBlurPass: { radius: 0.99 } };
    applyBloomState(target, null);
    expect(target.intensity).toBe(99);
  });

  it('apply skips threshold/radius when state has them undefined', () => {
    const target = {
      intensity: 0,
      mipmapBlurPass: { radius: 0.5 },
      luminanceMaterial: { threshold: 0.3 },
    };
    applyBloomState(target, { intensity: 1 });
    expect(target.intensity).toBe(1);
    expect(target.mipmapBlurPass?.radius).toBe(0.5); // unchanged
    expect(target.luminanceMaterial?.threshold).toBe(0.3); // unchanged
  });

  it('applies threshold of 0 (it is a valid value, not undefined)', () => {
    const target = {
      intensity: 0,
      mipmapBlurPass: { radius: 0.5 },
      luminanceMaterial: { threshold: 0.5 },
    };
    applyBloomState(target, { intensity: 1, luminanceThreshold: 0, radius: 0 });
    expect(target.luminanceMaterial?.threshold).toBe(0);
    expect(target.mipmapBlurPass?.radius).toBe(0);
  });
});

describe('captureToneMappingState / applyToneMappingState', () => {
  it('round-trips a full state', () => {
    const source = {
      mode: ToneMappingMode.AGX,
      whitePoint: 4,
      exposure: 0.5,
      globalOffset: -0.1,
      globalGamma: 2.2,
    };
    const target = {
      mode: ToneMappingMode.LINEAR,
      whitePoint: 0,
      exposure: 0,
      globalOffset: 0,
      globalGamma: 0,
    };
    applyToneMappingState(target, captureToneMappingState(source));
    expect(target).toEqual(source);
  });

  it('returns null when effect is absent', () => {
    expect(captureToneMappingState(null)).toBeNull();
    expect(captureToneMappingState(undefined)).toBeNull();
  });

  it('apply is a no-op when state or effect missing', () => {
    expect(() => applyToneMappingState(null, null)).not.toThrow();
  });
});

describe('captureDOFState / applyDOFFocusDistance', () => {
  it('captures bokehScale and focusDistance', () => {
    const source = {
      bokehScale: 2,
      circleOfConfusionMaterial: {
        uniforms: { focusDistance: { value: 0.3 } },
      },
    };
    expect(captureDOFState(source)).toEqual({ bokehScale: 2, focusDistance: 0.3 });
  });

  it('handles missing circleOfConfusionMaterial', () => {
    expect(captureDOFState({ bokehScale: 1 })).toEqual({
      bokehScale: 1,
      focusDistance: undefined,
    });
  });

  it('apply only writes focusDistance, not bokehScale (manager handles strength→bokeh)', () => {
    const target = {
      bokehScale: 99,
      circleOfConfusionMaterial: {
        uniforms: { focusDistance: { value: 0 } },
      },
    };
    applyDOFFocusDistance(target, { focusDistance: 0.7 });
    expect(target.circleOfConfusionMaterial.uniforms.focusDistance.value).toBe(0.7);
    expect(target.bokehScale).toBe(99); // unchanged
  });

  it('apply is a no-op when uniforms.focusDistance is missing', () => {
    const target = {
      bokehScale: 1,
      circleOfConfusionMaterial: { uniforms: {} },
    };
    expect(() => applyDOFFocusDistance(target, { focusDistance: 0.5 })).not.toThrow();
  });
});

describe('captureVignetteState / applyVignetteState', () => {
  it('round-trips darkness/offset', () => {
    const source = { darkness: 0.7, offset: 0.4 };
    const target = { darkness: 0, offset: 0 };
    applyVignetteState(target, captureVignetteState(source));
    expect(target).toEqual(source);
  });
});

describe('captureChromaticLensDistortionState / applyChromaticLensDistortionState', () => {
  it('clones the Vector2s on capture (no aliasing of the source effect)', () => {
    const source = {
      distortion: new THREE.Vector2(0.1, 0.2),
      principalPoint: new THREE.Vector2(0.3, 0.4),
      focalLength: new THREE.Vector2(1.0, 1.5),
      skew: 0.05,
      dispersion: 0.02,
    };
    const captured = captureChromaticLensDistortionState(source)!;
    expect(captured.distortion).not.toBe(source.distortion);
    expect(captured.distortion.x).toBe(0.1);
    expect(captured.skew).toBe(0.05);

    // Mutating the source after capture must not affect the captured state.
    source.distortion.set(99, 99);
    expect(captured.distortion.x).toBe(0.1);
  });

  it('clones again on apply (target keeps independent Vector2s)', () => {
    const source = {
      distortion: new THREE.Vector2(1, 1),
      principalPoint: new THREE.Vector2(0, 0),
      focalLength: new THREE.Vector2(1, 1),
      skew: 0,
      dispersion: 0,
    };
    const target = {
      distortion: new THREE.Vector2(0, 0),
      principalPoint: new THREE.Vector2(0, 0),
      focalLength: new THREE.Vector2(0, 0),
      skew: 99,
      dispersion: 99,
    };
    const captured = captureChromaticLensDistortionState(source)!;
    applyChromaticLensDistortionState(target, captured);
    expect(target.distortion).not.toBe(captured.distortion);
    expect(target.distortion.x).toBe(1);
    expect(target.skew).toBe(0);

    // Captured state remains intact across apply.
    captured.distortion.set(42, 42);
    expect(target.distortion.x).toBe(1);
  });
});

describe('captureDetectorNoiseState / applyDetectorNoiseState', () => {
  it('round-trips the three sigma fields', () => {
    const source = { readoutSigma: 0.02, photonGain: 0.01, fpnSigma: 0.005 };
    const target = { readoutSigma: 0, photonGain: 0, fpnSigma: 0 };
    applyDetectorNoiseState(target, captureDetectorNoiseState(source));
    expect(target).toEqual(source);
  });

  it('apply is a no-op when state or effect missing', () => {
    expect(() => applyDetectorNoiseState(null, null)).not.toThrow();
  });
});
