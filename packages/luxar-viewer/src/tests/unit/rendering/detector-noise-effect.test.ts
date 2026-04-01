/**
 * Tests for the DetectorNoiseEffect physics-based noise simulation
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

// Mock postprocessing library's Effect base class
vi.mock('postprocessing', () => ({
  Effect: class MockEffect {
    uniforms: Map<string, THREE.Uniform>;
    blendFunction: number;
    name: string;

    constructor(name: string, _fragmentShader: string, options: any) {
      this.name = name;
      this.blendFunction = options.blendFunction ?? 2; // NORMAL
      this.uniforms = options.uniforms ?? new Map();
    }
  },
  BlendFunction: {
    ADD: 0,
    SCREEN: 1,
    NORMAL: 2,
    MULTIPLY: 3,
    OVERLAY: 4,
    SOFT_LIGHT: 5,
  },
  EffectAttribute: {
    NONE: 0,
    DEPTH: 1,
    CONVOLUTION: 2,
  },
}));

// Import after mocking
import {
  DetectorNoiseEffect,
  isDetectorNoiseEffect,
} from '../../../rendering/detector-noise-effect';

describe('DetectorNoiseEffect', () => {
  describe('creation', () => {
    it('should create with default options', () => {
      const effect = new DetectorNoiseEffect();

      expect(effect).toBeDefined();
      expect(effect.readoutSigma).toBe(0.01);
      expect(effect.photonGain).toBe(0.01);
      expect(effect.fpnSigma).toBe(0.005);
    });

    it('should create with custom options', () => {
      const effect = new DetectorNoiseEffect({
        readoutSigma: 0.05,
        photonGain: 0.02,
        fpnSigma: 0.01,
      });

      expect(effect.readoutSigma).toBe(0.05);
      expect(effect.photonGain).toBe(0.02);
      expect(effect.fpnSigma).toBe(0.01);
    });

    it('should have required uniforms', () => {
      const effect = new DetectorNoiseEffect();
      const uniforms = effect.uniforms;

      expect(uniforms.has('time')).toBe(true);
      expect(uniforms.has('readoutSigma')).toBe(true);
      expect(uniforms.has('photonGain')).toBe(true);
      expect(uniforms.has('fpnSigma')).toBe(true);
    });
  });

  describe('property setters', () => {
    let effect: DetectorNoiseEffect;

    beforeEach(() => {
      effect = new DetectorNoiseEffect();
    });

    it('should set readoutSigma', () => {
      effect.readoutSigma = 0.05;
      expect(effect.readoutSigma).toBe(0.05);
    });

    it('should clamp readoutSigma to non-negative', () => {
      effect.readoutSigma = -0.1;
      expect(effect.readoutSigma).toBe(0);
    });

    it('should set photonGain', () => {
      effect.photonGain = 0.05;
      expect(effect.photonGain).toBe(0.05);
    });

    it('should clamp photonGain to minimum value', () => {
      effect.photonGain = 0;
      expect(effect.photonGain).toBe(0.0001);
    });

    it('should set fpnSigma', () => {
      effect.fpnSigma = 0.02;
      expect(effect.fpnSigma).toBe(0.02);
    });

    it('should clamp fpnSigma to non-negative', () => {
      effect.fpnSigma = -0.1;
      expect(effect.fpnSigma).toBe(0);
    });
  });

  describe('update method', () => {
    let effect: DetectorNoiseEffect;
    let mockRenderer: THREE.WebGLRenderer;
    let mockBuffer: THREE.WebGLRenderTarget;

    beforeEach(() => {
      effect = new DetectorNoiseEffect();
      mockRenderer = {} as THREE.WebGLRenderer;
      mockBuffer = {} as THREE.WebGLRenderTarget;
    });

    it('should advance time on update', () => {
      const initialTime = effect.uniforms.get('time')!.value;
      effect.update(mockRenderer, mockBuffer, 0.016); // ~60fps frame
      expect(effect.uniforms.get('time')!.value).toBeGreaterThan(initialTime);
    });

    it('should accumulate time over multiple updates', () => {
      effect.update(mockRenderer, mockBuffer, 0.1);
      effect.update(mockRenderer, mockBuffer, 0.1);
      effect.update(mockRenderer, mockBuffer, 0.1);
      expect(effect.uniforms.get('time')!.value).toBeCloseTo(0.3, 5);
    });

    it('should not advance time when deltaTime is undefined', () => {
      const initialTime = effect.uniforms.get('time')!.value;
      effect.update(mockRenderer, mockBuffer, undefined);
      expect(effect.uniforms.get('time')!.value).toBe(initialTime);
    });
  });

  describe('type guard', () => {
    it('should return true for DetectorNoiseEffect instances', () => {
      const effect = new DetectorNoiseEffect();
      expect(isDetectorNoiseEffect(effect)).toBe(true);
    });

    it('should return false for null', () => {
      expect(isDetectorNoiseEffect(null)).toBe(false);
    });

    it('should return false for undefined', () => {
      expect(isDetectorNoiseEffect(undefined)).toBe(false);
    });

    it('should return false for plain objects', () => {
      expect(isDetectorNoiseEffect({})).toBe(false);
    });

    it('should return true for objects with required properties', () => {
      const mockEffect = {
        readoutSigma: 0.01,
        photonGain: 0.01,
        fpnSigma: 0.005,
      };
      expect(isDetectorNoiseEffect(mockEffect)).toBe(true);
    });
  });

  describe('parameter propagation', () => {
    it('should propagate constructor options to uniforms', () => {
      const effect = new DetectorNoiseEffect({
        photonGain: 0.1,
        readoutSigma: 0.05,
        fpnSigma: 0.03,
      });

      // Verify uniforms reflect constructor values (not just property getters)
      expect(effect.uniforms.get('photonGain')!.value).toBe(0.1);
      expect(effect.uniforms.get('readoutSigma')!.value).toBe(0.05);
      expect(effect.uniforms.get('fpnSigma')!.value).toBe(0.03);
    });

    it('should propagate property updates to uniforms', () => {
      const effect = new DetectorNoiseEffect();

      // Update via setters
      effect.readoutSigma = 0.08;
      effect.photonGain = 0.05;
      effect.fpnSigma = 0.02;

      // Verify uniforms were updated (the shader reads these, not the properties)
      expect(effect.uniforms.get('readoutSigma')!.value).toBe(0.08);
      expect(effect.uniforms.get('photonGain')!.value).toBe(0.05);
      expect(effect.uniforms.get('fpnSigma')!.value).toBe(0.02);
    });

    it('should clamp negative readoutSigma to zero in uniform', () => {
      const effect = new DetectorNoiseEffect();
      effect.readoutSigma = -0.5;
      expect(effect.uniforms.get('readoutSigma')!.value).toBe(0);
    });

    it('should clamp negative fpnSigma to zero in uniform', () => {
      const effect = new DetectorNoiseEffect();
      effect.fpnSigma = -1.0;
      expect(effect.uniforms.get('fpnSigma')!.value).toBe(0);
    });

    it('should clamp photonGain to minimum 0.0001 in uniform', () => {
      const effect = new DetectorNoiseEffect();
      effect.photonGain = -5.0;
      expect(effect.uniforms.get('photonGain')!.value).toBe(0.0001);
    });

    it('should clamp photonGain of zero to minimum in uniform', () => {
      const effect = new DetectorNoiseEffect();
      effect.photonGain = 0;
      expect(effect.uniforms.get('photonGain')!.value).toBe(0.0001);
    });
  });

  describe('noise component independence', () => {
    it('should allow setting each noise component independently', () => {
      const effect = new DetectorNoiseEffect({
        readoutSigma: 0.0,
        photonGain: 0.0001, // Minimum for shot noise
        fpnSigma: 0.0,
      });

      // Enable only readout noise
      effect.readoutSigma = 0.05;
      expect(effect.readoutSigma).toBe(0.05);
      expect(effect.fpnSigma).toBe(0.0);

      // Enable only FPN
      effect.readoutSigma = 0.0;
      effect.fpnSigma = 0.02;
      expect(effect.readoutSigma).toBe(0.0);
      expect(effect.fpnSigma).toBe(0.02);
    });
  });
});
