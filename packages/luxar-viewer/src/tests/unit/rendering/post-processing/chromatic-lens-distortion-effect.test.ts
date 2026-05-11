/**
 * Unit tests for ChromaticLensDistortionEffect.
 *
 * Tests the public TypeScript surface: constructor defaults, getters,
 * setters (including the dispersion clamp), and the type guard. The
 * GLSL shader is exercised by the e2e visual-regression suite.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  ChromaticLensDistortionEffect,
  isChromaticLensDistortionEffect,
} from '../../../../rendering/post-processing/chromatic-lens-distortion-effect';
import { BlendFunction } from 'postprocessing';

describe('ChromaticLensDistortionEffect — defaults', () => {
  it('uses zero distortion / unit focal length / no dispersion by default', () => {
    const fx = new ChromaticLensDistortionEffect();
    expect(fx.distortion.equals(new THREE.Vector2(0, 0))).toBe(true);
    expect(fx.principalPoint.equals(new THREE.Vector2(0, 0))).toBe(true);
    expect(fx.focalLength.equals(new THREE.Vector2(1, 1))).toBe(true);
    expect(fx.skew).toBe(0);
    expect(fx.dispersion).toBe(0);
  });

  it('respects explicit options', () => {
    const fx = new ChromaticLensDistortionEffect({
      distortion: new THREE.Vector2(-0.05, -0.05),
      principalPoint: new THREE.Vector2(0.01, 0.02),
      focalLength: new THREE.Vector2(1.2, 1.2),
      skew: 0.01,
      dispersion: 0.15,
    });
    expect(fx.distortion.x).toBeCloseTo(-0.05);
    expect(fx.principalPoint.y).toBeCloseTo(0.02);
    expect(fx.focalLength.x).toBeCloseTo(1.2);
    expect(fx.skew).toBeCloseTo(0.01);
    expect(fx.dispersion).toBeCloseTo(0.15);
  });

  it('respects an explicit blendFunction', () => {
    const fx = new ChromaticLensDistortionEffect({ blendFunction: BlendFunction.MULTIPLY });
    expect(fx.blendMode.blendFunction).toBe(BlendFunction.MULTIPLY);
  });
});

describe('ChromaticLensDistortionEffect — setters', () => {
  it('distortion setter swaps the uniform value', () => {
    const fx = new ChromaticLensDistortionEffect();
    const newVal = new THREE.Vector2(0.1, 0.2);
    fx.distortion = newVal;
    expect(fx.distortion).toBe(newVal);
  });

  it('principalPoint / focalLength / skew setters mutate', () => {
    const fx = new ChromaticLensDistortionEffect();
    fx.principalPoint = new THREE.Vector2(0.5, 0);
    fx.focalLength = new THREE.Vector2(2, 2);
    fx.skew = 0.05;
    expect(fx.principalPoint.x).toBe(0.5);
    expect(fx.focalLength.x).toBe(2);
    expect(fx.skew).toBe(0.05);
  });

  it('dispersion setter clamps to [0, 1]', () => {
    const fx = new ChromaticLensDistortionEffect();
    fx.dispersion = -0.5;
    expect(fx.dispersion).toBe(0);
    fx.dispersion = 0.5;
    expect(fx.dispersion).toBe(0.5);
    fx.dispersion = 5;
    expect(fx.dispersion).toBe(1);
  });
});

describe('isChromaticLensDistortionEffect', () => {
  it('returns true for an actual ChromaticLensDistortionEffect', () => {
    expect(isChromaticLensDistortionEffect(new ChromaticLensDistortionEffect())).toBe(true);
  });

  it('returns false for plain objects without all the fields', () => {
    expect(isChromaticLensDistortionEffect(null)).toBe(false);
    expect(isChromaticLensDistortionEffect(undefined)).toBe(false);
    expect(isChromaticLensDistortionEffect({})).toBe(false);
    // Has some but not all of the required fields:
    expect(
      isChromaticLensDistortionEffect({
        distortion: new THREE.Vector2(),
        principalPoint: new THREE.Vector2(),
      })
    ).toBe(false);
  });

  it('accepts any object with the full structural shape (duck typing)', () => {
    const stub = {
      distortion: new THREE.Vector2(),
      principalPoint: new THREE.Vector2(),
      focalLength: new THREE.Vector2(),
      skew: 0,
      dispersion: 0,
    };
    expect(isChromaticLensDistortionEffect(stub)).toBe(true);
  });
});
