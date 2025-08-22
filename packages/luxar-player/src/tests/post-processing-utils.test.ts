/**
 * Tests for post-processing utility functions
 */

import { describe, it, expect } from 'vitest';
import {
  validateSSAAMultiplier,
  validateSMAAThreshold,
  validateSMAASearchSteps,
  calculateSSAAResolution,
  calculateMSAASamples,
  validateBloomConfig,
  calculateBloomResolution,
  validateToneMappingConfig,
  shouldUseHDRPipeline,
  getRenderTargetType,
  estimatePostProcessingMemory,
  optimizeAAStrategy,
} from '../rendering/post-processing-utils';

describe('post-processing-utils', () => {
  describe('validateSSAAMultiplier', () => {
    it('should accept valid multipliers', () => {
      expect(validateSSAAMultiplier(1.5)).toBe(1.5);
      expect(validateSSAAMultiplier(2.0)).toBe(2.0);
      expect(validateSSAAMultiplier(3.5)).toBe(3.5);
    });

    it('should clamp values below minimum', () => {
      expect(validateSSAAMultiplier(0.5)).toBe(1.0);
      expect(validateSSAAMultiplier(-1)).toBe(1.0);
    });

    it('should clamp values above maximum', () => {
      expect(validateSSAAMultiplier(5.0)).toBe(4.0);
      expect(validateSSAAMultiplier(10)).toBe(4.0);
    });
  });

  describe('validateSMAAThreshold', () => {
    it('should accept valid thresholds', () => {
      expect(validateSMAAThreshold(0.1)).toBe(0.1);
      expect(validateSMAAThreshold(0.15)).toBe(0.15);
    });

    it('should clamp low values', () => {
      expect(validateSMAAThreshold(0.01)).toBe(0.05);
      expect(validateSMAAThreshold(0)).toBe(0.05);
    });

    it('should clamp high values', () => {
      expect(validateSMAAThreshold(0.5)).toBe(0.2);
      expect(validateSMAAThreshold(1.0)).toBe(0.2);
    });
  });

  describe('validateSMAASearchSteps', () => {
    it('should accept valid step counts', () => {
      expect(validateSMAASearchSteps(4)).toBe(4);
      expect(validateSMAASearchSteps(8)).toBe(8);
      expect(validateSMAASearchSteps(16)).toBe(16);
      expect(validateSMAASearchSteps(32)).toBe(32);
    });

    it('should round to nearest valid value', () => {
      expect(validateSMAASearchSteps(3)).toBe(4);
      expect(validateSMAASearchSteps(6)).toBe(8);
      expect(validateSMAASearchSteps(12)).toBe(16);
      expect(validateSMAASearchSteps(24)).toBe(16);
      expect(validateSMAASearchSteps(40)).toBe(32);
    });
  });

  describe('calculateSSAAResolution', () => {
    it('should return base resolution when disabled', () => {
      const result = calculateSSAAResolution(1920, 1080, {
        enabled: false,
        multiplier: 2.0,
      });
      expect(result).toEqual({ width: 1920, height: 1080 });
    });

    it('should multiply resolution when enabled', () => {
      const result = calculateSSAAResolution(1920, 1080, {
        enabled: true,
        multiplier: 1.5,
      });
      expect(result).toEqual({ width: 2880, height: 1620 });
    });

    it('should validate multiplier', () => {
      const result = calculateSSAAResolution(1000, 1000, {
        enabled: true,
        multiplier: 10, // Should be clamped to 4
      });
      expect(result).toEqual({ width: 4000, height: 4000 });
    });
  });

  describe('calculateMSAASamples', () => {
    it('should return correct samples for quality presets', () => {
      expect(calculateMSAASamples('none')).toBe(0);
      expect(calculateMSAASamples('low')).toBe(2);
      expect(calculateMSAASamples('medium')).toBe(4);
      expect(calculateMSAASamples('high')).toBe(8);
      expect(calculateMSAASamples('ultra')).toBe(16);
    });

    it('should respect device maximum', () => {
      expect(calculateMSAASamples('ultra', 4)).toBe(4);
      expect(calculateMSAASamples('high', 4)).toBe(4);
      expect(calculateMSAASamples('medium', 2)).toBe(2);
    });
  });

  describe('validateBloomConfig', () => {
    it('should apply defaults for missing values', () => {
      const result = validateBloomConfig({});
      expect(result).toEqual({
        enabled: true,
        intensity: 1.0,
        threshold: 0.9,
        radius: 0.4,
      });
    });

    it('should clamp invalid values', () => {
      const result = validateBloomConfig({
        enabled: false,
        intensity: 10,
        threshold: -1,
        radius: 5,
      });
      expect(result).toEqual({
        enabled: false,
        intensity: 5,
        threshold: 0,
        radius: 2,
      });
    });
  });

  describe('calculateBloomResolution', () => {
    it('should scale resolution based on quality', () => {
      const low = calculateBloomResolution(1920, 1080, 0);
      expect(low).toEqual({ width: 480, height: 270 });

      const medium = calculateBloomResolution(1920, 1080, 0.5);
      expect(medium).toEqual({ width: 1200, height: 675 });

      const high = calculateBloomResolution(1920, 1080, 1.0);
      expect(high).toEqual({ width: 1920, height: 1080 });
    });

    it('should enforce minimum resolution', () => {
      const result = calculateBloomResolution(100, 100, 0);
      expect(result.width).toBeGreaterThanOrEqual(256);
      expect(result.height).toBeGreaterThanOrEqual(256);
    });
  });

  describe('validateToneMappingConfig', () => {
    it('should validate tone mapping type', () => {
      const result = validateToneMappingConfig({ type: 'aces' });
      expect(result.type).toBe('aces');

      const invalid = validateToneMappingConfig({ type: 'invalid' as any });
      expect(invalid.type).toBe('aces'); // Default
    });

    it('should clamp exposure values', () => {
      const result = validateToneMappingConfig({
        type: 'linear',
        exposure: 0.05,
      });
      expect(result.exposure).toBe(0.1);

      const high = validateToneMappingConfig({
        type: 'linear',
        exposure: 20,
      });
      expect(high.exposure).toBe(10);
    });
  });

  describe('shouldUseHDRPipeline', () => {
    it('should enable for HDR content', () => {
      expect(shouldUseHDRPipeline(true, false, 'none')).toBe(true);
    });

    it('should enable for bloom', () => {
      expect(shouldUseHDRPipeline(false, true, 'none')).toBe(true);
    });

    it('should enable for advanced tone mapping', () => {
      expect(shouldUseHDRPipeline(false, false, 'aces')).toBe(true);
      expect(shouldUseHDRPipeline(false, false, 'agx')).toBe(true);
      expect(shouldUseHDRPipeline(false, false, 'reinhard')).toBe(true);
    });

    it('should disable for basic setup', () => {
      expect(shouldUseHDRPipeline(false, false, 'none')).toBe(false);
      expect(shouldUseHDRPipeline(false, false, 'linear')).toBe(false);
    });
  });

  describe('getRenderTargetType', () => {
    it('should return half float for HDR', () => {
      expect(getRenderTargetType(true)).toBe(1016); // THREE.HalfFloatType
    });

    it('should return unsigned byte for LDR', () => {
      expect(getRenderTargetType(false)).toBe(1009); // THREE.UnsignedByteType
    });
  });

  describe('estimatePostProcessingMemory', () => {
    it('should calculate basic memory usage', () => {
      const memory = estimatePostProcessingMemory(1920, 1080);
      expect(memory).toBeCloseTo(19.78, 1); // ~20MB for 1920x1080 LDR
    });

    it('should account for MSAA', () => {
      const memory = estimatePostProcessingMemory(1920, 1080, 4);
      expect(memory).toBeCloseTo(79.1, 1); // 4x more for 4x MSAA
    });

    it('should account for SSAA', () => {
      const memory = estimatePostProcessingMemory(1920, 1080, 0, 2.0);
      expect(memory).toBeCloseTo(79.1, 1); // 4x more for 2x SSAA
    });

    it('should account for HDR', () => {
      const memoryLDR = estimatePostProcessingMemory(1000, 1000, 0, 1.0, false);
      const memoryHDR = estimatePostProcessingMemory(1000, 1000, 0, 1.0, true);
      expect(memoryHDR).toBeCloseTo(memoryLDR * 2, 1); // HDR uses twice the memory
    });
  });

  describe('optimizeAAStrategy', () => {
    it('should recommend SSAA for excellent performance', () => {
      const strategy = optimizeAAStrategy(60, 120, 4000);
      expect(strategy).toEqual({
        msaa: true,
        smaa: false,
        fxaa: false,
        ssaa: true,
      });
    });

    it('should recommend MSAA for good performance', () => {
      const strategy = optimizeAAStrategy(60, 80, 2000);
      expect(strategy).toEqual({
        msaa: true,
        smaa: false,
        fxaa: false,
        ssaa: false,
      });
    });

    it('should recommend SMAA for adequate performance', () => {
      const strategy = optimizeAAStrategy(60, 55, 1000);
      expect(strategy).toEqual({
        msaa: false,
        smaa: true,
        fxaa: false,
        ssaa: false,
      });
    });

    it('should recommend FXAA for poor performance', () => {
      const strategy = optimizeAAStrategy(60, 45, 500);
      expect(strategy).toEqual({
        msaa: false,
        smaa: false,
        fxaa: true,
        ssaa: false,
      });
    });

    it('should disable AA for very poor performance', () => {
      const strategy = optimizeAAStrategy(60, 30, 500);
      expect(strategy).toEqual({
        msaa: false,
        smaa: false,
        fxaa: false,
        ssaa: false,
      });
    });
  });
});
