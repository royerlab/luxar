/**
 * Unit tests for the visual-effects helpers.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  DOF_DEFAULT_FOCUS,
  DOF_DEFAULT_STRENGTH,
  NOISE_DEFAULT_FPN_SIGMA,
  NOISE_DEFAULT_PHOTON_GAIN,
  NOISE_DEFAULT_READOUT_SIGMA,
  VIGNETTE_DEFAULT_DARKNESS,
  VIGNETTE_DEFAULT_OFFSET,
  clampDPRScale,
  computeDOFFocalLength,
  dprScaleChanged,
  mergeVector2,
  resolveChromaticDistortionDefaults,
  resolveNoiseDefaults,
  resolveVignetteDefaults,
  scaleNoiseSettings,
  strengthToBokehScale,
} from '../../../../rendering/post-processing/visual-effects-handler';

describe('DOF helpers', () => {
  it('exposes documented default focus and strength', () => {
    expect(DOF_DEFAULT_FOCUS).toBe(10.0);
    expect(DOF_DEFAULT_STRENGTH).toBe(0.5);
  });

  it('computeDOFFocalLength matches the legacy 0.035 × (50/fov) formula', () => {
    expect(computeDOFFocalLength(50)).toBeCloseTo(0.035, 6);
    expect(computeDOFFocalLength(25)).toBeCloseTo(0.07, 6);
    expect(computeDOFFocalLength(100)).toBeCloseTo(0.0175, 6);
  });

  it('strengthToBokehScale multiplies UI strength by 4', () => {
    expect(strengthToBokehScale(0)).toBe(0);
    expect(strengthToBokehScale(0.5)).toBe(2);
    expect(strengthToBokehScale(1)).toBe(4);
  });
});

describe('detector-noise helpers', () => {
  it('default constants match documented values', () => {
    expect(NOISE_DEFAULT_READOUT_SIGMA).toBe(0.01);
    expect(NOISE_DEFAULT_PHOTON_GAIN).toBe(0.01);
    expect(NOISE_DEFAULT_FPN_SIGMA).toBe(0.005);
  });

  it('resolveNoiseDefaults fills missing fields with constants', () => {
    expect(resolveNoiseDefaults({})).toEqual({
      readoutSigma: NOISE_DEFAULT_READOUT_SIGMA,
      photonGain: NOISE_DEFAULT_PHOTON_GAIN,
      fpnSigma: NOISE_DEFAULT_FPN_SIGMA,
    });
    expect(resolveNoiseDefaults({ readoutSigma: 0.02 })).toEqual({
      readoutSigma: 0.02,
      photonGain: NOISE_DEFAULT_PHOTON_GAIN,
      fpnSigma: NOISE_DEFAULT_FPN_SIGMA,
    });
  });

  it('scaleNoiseSettings: gaussian σ scales linearly, photonGain scales by DPR²', () => {
    const base = { readoutSigma: 0.01, photonGain: 0.04, fpnSigma: 0.005 };
    expect(scaleNoiseSettings(base, 1.0)).toEqual(base);
    const half = scaleNoiseSettings(base, 0.5);
    expect(half.readoutSigma).toBeCloseTo(0.005);
    expect(half.fpnSigma).toBeCloseTo(0.0025);
    expect(half.photonGain).toBeCloseTo(0.01); // 0.04 × 0.25
    const quarter = scaleNoiseSettings(base, 0.25);
    expect(quarter.photonGain).toBeCloseTo(0.04 * 0.0625);
  });

  it('clampDPRScale clamps to [0.25, 1.0]', () => {
    expect(clampDPRScale(0)).toBe(0.25);
    expect(clampDPRScale(0.1)).toBe(0.25);
    expect(clampDPRScale(0.25)).toBe(0.25);
    expect(clampDPRScale(0.5)).toBe(0.5);
    expect(clampDPRScale(1.0)).toBe(1.0);
    expect(clampDPRScale(1.5)).toBe(1.0);
    expect(clampDPRScale(-2)).toBe(0.25);
  });

  it('dprScaleChanged uses a 0.01 epsilon by default', () => {
    expect(dprScaleChanged(1.0, 1.0)).toBe(false);
    expect(dprScaleChanged(1.0, 0.995)).toBe(false);
    expect(dprScaleChanged(1.0, 0.989)).toBe(true);
    expect(dprScaleChanged(1.0, 0.5)).toBe(true);
    expect(dprScaleChanged(1.0, 1.05, 0.1)).toBe(false);
  });
});

describe('vignette helpers', () => {
  it('exposes documented defaults', () => {
    expect(VIGNETTE_DEFAULT_DARKNESS).toBe(0.5);
    expect(VIGNETTE_DEFAULT_OFFSET).toBe(0.5);
  });

  it('resolveVignetteDefaults preserves provided fields', () => {
    expect(resolveVignetteDefaults({})).toEqual({ darkness: 0.5, offset: 0.5 });
    expect(resolveVignetteDefaults({ darkness: 0.8 })).toEqual({ darkness: 0.8, offset: 0.5 });
    expect(resolveVignetteDefaults({ offset: 0.2 })).toEqual({ darkness: 0.5, offset: 0.2 });
  });
});

describe('chromatic distortion defaults', () => {
  it('all-zero distortion / all-1 focal length / 0 dispersion + skew', () => {
    expect(resolveChromaticDistortionDefaults()).toEqual({
      distortionX: 0,
      distortionY: 0,
      dispersion: 0,
      principalPointX: 0,
      principalPointY: 0,
      focalLengthX: 1,
      focalLengthY: 1,
      skew: 0,
    });
  });

  it('preserves explicit values', () => {
    const settings = resolveChromaticDistortionDefaults({
      distortionX: 0.1,
      focalLengthX: 1.2,
      skew: 0.05,
    });
    expect(settings.distortionX).toBe(0.1);
    expect(settings.focalLengthX).toBe(1.2);
    expect(settings.skew).toBe(0.05);
    expect(settings.distortionY).toBe(0);
  });
});

describe('mergeVector2', () => {
  it('produces a fresh THREE.Vector2 with the partial overrides applied', () => {
    const current = { x: 1, y: 2 };
    const out = mergeVector2(current, undefined, undefined);
    expect(out).toBeInstanceOf(THREE.Vector2);
    expect(out.x).toBe(1);
    expect(out.y).toBe(2);
  });

  it('overrides x or y individually', () => {
    expect(mergeVector2({ x: 1, y: 2 }, 5, undefined).toArray()).toEqual([5, 2]);
    expect(mergeVector2({ x: 1, y: 2 }, undefined, 9).toArray()).toEqual([1, 9]);
    expect(mergeVector2({ x: 1, y: 2 }, 5, 9).toArray()).toEqual([5, 9]);
  });

  it('preserves zero overrides (since 0 is a valid override, not undefined)', () => {
    expect(mergeVector2({ x: 1, y: 2 }, 0, 0).toArray()).toEqual([0, 0]);
  });

  it('returns a new instance every call', () => {
    const current = new THREE.Vector2(1, 2);
    const out = mergeVector2(current, undefined, undefined);
    expect(out).not.toBe(current);
  });
});
