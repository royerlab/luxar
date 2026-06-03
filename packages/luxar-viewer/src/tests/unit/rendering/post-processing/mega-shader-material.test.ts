/**
 * Unit tests for MegaShaderMaterial state plumbing.
 *
 * These tests exercise the TypeScript-side shader contract (defines,
 * uniforms, tone-mapping mapping) without requiring a WebGL context.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { MegaShaderMaterial } from '../../../../rendering/post-processing/mega/material';

describe('MegaShaderMaterial', () => {
  it('disables renderer tone-mapping injection and starts in ACES mode', () => {
    const material = new MegaShaderMaterial();

    expect(material.toneMapped).toBe(false);
    expect(material.glslVersion).toBe(THREE.GLSL3);
    expect(material.defines.LUXAR_TONE_MAPPING_MODE).toBe('4');
    expect(material.getToneMapping()).toBe(THREE.ACESFilmicToneMapping);

    material.dispose();
  });

  it('aliases THREE.NoToneMapping to the Linear shader mode for old-pipeline clamp parity', () => {
    const material = new MegaShaderMaterial({ toneMapping: THREE.NoToneMapping });

    expect(material.defines.LUXAR_TONE_MAPPING_MODE).toBe('1');
    expect(material.getToneMapping()).toBe(THREE.LinearToneMapping);

    material.dispose();
  });

  it('maps all supported tone-mapping constants to stable internal define ids', () => {
    const material = new MegaShaderMaterial();

    const cases: Array<[THREE.ToneMapping, string]> = [
      [THREE.LinearToneMapping, '1'],
      [THREE.ReinhardToneMapping, '2'],
      [THREE.CineonToneMapping, '3'],
      [THREE.ACESFilmicToneMapping, '4'],
      [THREE.AgXToneMapping, '5'],
      [THREE.NeutralToneMapping, '6'],
    ];

    for (const [mode, define] of cases) {
      material.setToneMapping(mode);
      expect(material.defines.LUXAR_TONE_MAPPING_MODE).toBe(define);
    }

    material.dispose();
  });

  it('toggles effect and capture defines idempotently', () => {
    const material = new MegaShaderMaterial();

    material.toggleBloom(true);
    material.toggleDetectorNoise(true);
    material.toggleVignette(true);
    material.toggleLensDistortion(true);
    material.toggleRawHdrCapture(true);
    material.toggleLinearLdrCapture(true);

    expect(material.defines.USE_BLOOM).toBe('');
    expect(material.defines.USE_DETECTOR_NOISE).toBe('');
    expect(material.defines.USE_VIGNETTE).toBe('');
    expect(material.defines.USE_LENS_DISTORTION).toBe('');
    expect(material.defines.LUXAR_CAPTURE_RAW_HDR).toBe('');
    expect(material.defines.LUXAR_CAPTURE_LINEAR_LDR).toBe('');

    material.toggleBloom(false);
    material.toggleDetectorNoise(false);
    material.toggleVignette(false);
    material.toggleLensDistortion(false);
    material.toggleRawHdrCapture(false);
    material.toggleLinearLdrCapture(false);

    expect('USE_BLOOM' in material.defines).toBe(false);
    expect('USE_DETECTOR_NOISE' in material.defines).toBe(false);
    expect('USE_VIGNETTE' in material.defines).toBe(false);
    expect('USE_LENS_DISTORTION' in material.defines).toBe(false);
    expect('LUXAR_CAPTURE_RAW_HDR' in material.defines).toBe(false);
    expect('LUXAR_CAPTURE_LINEAR_LDR' in material.defines).toBe(false);

    material.dispose();
  });

  it('updates uniforms with clamping where required', () => {
    const material = new MegaShaderMaterial();
    const hdrTexture = new THREE.Texture();
    const bloomTexture = new THREE.Texture();

    material.setHdrSceneTexture(hdrTexture);
    material.setResolution(640, 480);
    material.advanceTime(0.25);
    material.setExposure(1.5);
    material.setGlobalOffset(-0.1);
    material.setGlobalGamma(0);
    material.setDetectorNoise({ readoutSigma: -1, photonGain: 0, fpnSigma: -2 });
    material.setLensDistortion({ dispersion: 2 });
    material.setBloom(0.75, bloomTexture);

    expect(material.uniforms.uHdrScene.value).toBe(hdrTexture);
    expect(material.uniforms.uResolution.value.toArray()).toEqual([640, 480]);
    expect(material.uniforms.uTime.value).toBeCloseTo(0.25, 5);
    expect(material.uniforms.uExposure.value).toBe(1.5);
    expect(material.uniforms.uGlobalOffset.value).toBe(-0.1);
    expect(material.uniforms.uGlobalGamma.value).toBe(0.001);
    expect(material.uniforms.uReadoutSigma.value).toBe(0);
    expect(material.uniforms.uPhotonGain.value).toBe(0.0001);
    expect(material.uniforms.uFpnSigma.value).toBe(0);
    expect(material.uniforms.uDispersion.value).toBe(1);
    expect(material.uniforms.uBloomIntensity.value).toBe(0.75);
    expect(material.uniforms.uBloomTexture.value).toBe(bloomTexture);

    hdrTexture.dispose();
    bloomTexture.dispose();
    material.dispose();
  });

  // [rendering.md/G15][P5] Dispersion clamp boundary cases. Source uses
  // `Math.max(0, Math.min(1, value))` so the uniform must end up in [0,1].
  // NaN / Infinity / negative inputs were previously untested — pin the
  // IEEE-754 contract so any future guard introduction is deliberate.
  it('setLensDistortion clamps negative dispersion to 0', () => {
    const material = new MegaShaderMaterial();
    material.setLensDistortion({ dispersion: -0.5 });
    expect(material.uniforms.uDispersion.value).toBe(0);
    material.dispose();
  });

  it('setLensDistortion clamps dispersion > 1 to 1 (uses Math.min)', () => {
    const material = new MegaShaderMaterial();
    material.setLensDistortion({ dispersion: 100 });
    expect(material.uniforms.uDispersion.value).toBe(1);
    material.dispose();
  });

  it('setLensDistortion clamps -Infinity dispersion to 0 (Math.max guard)', () => {
    const material = new MegaShaderMaterial();
    material.setLensDistortion({ dispersion: Number.NEGATIVE_INFINITY });
    expect(material.uniforms.uDispersion.value).toBe(0);
    material.dispose();
  });

  it('setLensDistortion clamps +Infinity dispersion to 1 (Math.min guard)', () => {
    const material = new MegaShaderMaterial();
    material.setLensDistortion({ dispersion: Number.POSITIVE_INFINITY });
    expect(material.uniforms.uDispersion.value).toBe(1);
    material.dispose();
  });

  it('setLensDistortion: NaN dispersion propagates as NaN (Math.max/min with NaN → NaN)', () => {
    // Documents the IEEE-754 contract: `Math.max(0, Math.min(1, NaN))` is
    // NaN. The shader is expected to handle NaN gracefully (or the caller
    // is expected to guard upstream). A mutant that special-cased NaN would
    // change this observable.
    const material = new MegaShaderMaterial();
    material.setLensDistortion({ dispersion: Number.NaN });
    expect(Number.isNaN(material.uniforms.uDispersion.value)).toBe(true);
    material.dispose();
  });
});
