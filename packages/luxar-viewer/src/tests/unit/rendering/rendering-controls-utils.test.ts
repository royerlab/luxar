/**
 * Tests for rendering controls utility functions
 */

import { describe, it, expect } from 'vitest';
import {
  validateRenderingSettings,
  mergeSettings,
  serializeSettings,
  deserializeSettings,
  isValidColor,
  clampMSAASamples,
  settingsChanged,
  getSettingsRequiringRebuild,
  calculatePerformanceImpact,
  getDefaultRenderingSettings,
} from '../../../ui/rendering-controls-utils';

describe('rendering-controls-utils', () => {
  describe('validateRenderingSettings', () => {
    it('should apply defaults for missing values', () => {
      const result = validateRenderingSettings({});

      expect(result.bloomThreshold).toBe(0.01);
      expect(result.toneMapping).toBe('Neutral');
      expect(result.controlType).toBe('orbit');
    });

    it('should preserve valid values', () => {
      const input = {
        bloomThreshold: 0.5,
        exposure: 2.0,
        globalOffset: 0.1,
        globalGamma: 1.5,
        controlType: 'fly' as const,
      };

      const result = validateRenderingSettings(input);

      expect(result.bloomThreshold).toBe(0.5);
      expect(result.exposure).toBe(2.0);
      expect(result.globalOffset).toBe(0.1);
      expect(result.globalGamma).toBe(1.5);
      expect(result.controlType).toBe('fly');
    });

    it('should replace out-of-range numeric values with defaults', () => {
      const defaults = getDefaultRenderingSettings();
      const result = validateRenderingSettings({
        bloomThreshold: -5, // out of [0, 1]
        bloomLevels: 99, // out of [1, 12]
        near: -1, // must be > 0
      });

      expect(result.bloomThreshold).toBe(defaults.bloomThreshold);
      expect(result.bloomLevels).toBe(defaults.bloomLevels);
      expect(result.near).toBe(defaults.near);
    });

    it('should replace non-numeric values with defaults', () => {
      const defaults = getDefaultRenderingSettings();
      const result = validateRenderingSettings({
        fov: 'hello' as any,
        bloomStrength: NaN,
      });

      expect(result.fov).toBe(defaults.fov);
      expect(result.bloomStrength).toBe(defaults.bloomStrength);
    });

    it('should replace invalid enum values with defaults', () => {
      const defaults = getDefaultRenderingSettings();
      const result = validateRenderingSettings({
        toneMapping: 'InvalidMode' as any,
        controlType: 'magic' as any,
        aoQuality: 'extreme' as any,
      });

      expect(result.toneMapping).toBe(defaults.toneMapping);
      expect(result.controlType).toBe(defaults.controlType);
      expect(result.aoQuality).toBe(defaults.aoQuality);
    });

    it('should accept ortho as a valid control type', () => {
      const result = validateRenderingSettings({
        controlType: 'ortho' as any,
      });
      expect(result.controlType).toBe('ortho');
    });

    it('should round bloomLevels to integer', () => {
      const result = validateRenderingSettings({ bloomLevels: 5.7 });
      expect(result.bloomLevels).toBe(6);
    });

    it('should reset far to default when far <= near', () => {
      const defaults = getDefaultRenderingSettings();
      const result = validateRenderingSettings({ near: 10, far: 5 });
      expect(result.far).toBe(defaults.far);
    });
  });

  describe('mergeSettings', () => {
    it('should merge with defaults', () => {
      const result = mergeSettings({
        bloomThreshold: 0.05,
      });

      expect(result.bloomThreshold).toBe(0.05);
      expect(result.toneMapping).toBe('Neutral');
    });
  });

  describe('serializeSettings/deserializeSettings', () => {
    it('should serialize and deserialize settings', () => {
      const settings = getDefaultRenderingSettings();
      const serialized = serializeSettings(settings);
      const deserialized = deserializeSettings(serialized);

      expect(deserialized).toEqual(settings);
    });

    it('should handle invalid JSON gracefully', () => {
      const result = deserializeSettings('invalid json');
      expect(result).toBeNull();
    });
  });

  describe('isValidColor', () => {
    it('should validate hex colors', () => {
      expect(isValidColor('#ff0000')).toBe(true);
      expect(isValidColor('#FF0000')).toBe(true);
      expect(isValidColor('#f00')).toBe(true);
      expect(isValidColor('invalid')).toBe(false);
    });
  });

  describe('clampMSAASamples', () => {
    it('should clamp to valid power-of-2 values', () => {
      expect(clampMSAASamples(-1)).toBe(0);
      expect(clampMSAASamples(3)).toBe(4);
      expect(clampMSAASamples(6)).toBe(8);
      expect(clampMSAASamples(20)).toBe(16);
    });
  });

  describe('settingsChanged', () => {
    it('should detect changes', () => {
      const settings1 = getDefaultRenderingSettings();
      const settings2 = { ...settings1, bloomThreshold: 0.5 };

      expect(settingsChanged(settings1, settings2)).toBe(true);
      expect(settingsChanged(settings1, settings1)).toBe(false);
    });
  });

  describe('getSettingsRequiringRebuild', () => {
    it('should identify settings that require rebuild', () => {
      const prev = getDefaultRenderingSettings();
      const current = { ...prev, toneMapping: 'Linear' as const };

      const rebuilds = getSettingsRequiringRebuild(current, prev);
      expect(rebuilds).toContain('toneMapping');
    });
  });

  describe('calculatePerformanceImpact', () => {
    it('should calculate performance score', () => {
      const settings = getDefaultRenderingSettings();
      const score = calculatePerformanceImpact(settings);

      expect(typeof score).toBe('number');
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(100);
    });
  });
});
