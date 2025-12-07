/**
 * Tests for rendering controls utility functions
 */

import { describe, it, expect } from 'vitest';
import {
  validateRenderingSettings,
  mergeSettings,
  generateSettingsKey,
  serializeSettings,
  deserializeSettings,
  isValidColor,
  clampMSAASamples,
  settingsChanged,
  getSettingsRequiringRebuild,
  calculatePerformanceImpact,
  getDefaultRenderingSettings,
} from '../ui/rendering-controls-utils';

describe('rendering-controls-utils', () => {
  describe('validateRenderingSettings', () => {
    it('should apply defaults for missing values', () => {
      const result = validateRenderingSettings({});

      expect(result.bloomThreshold).toBe(0.01);
      expect(result.toneMapping).toBe('ACES');
      expect(result.controlType).toBe('orbit');
    });

    it('should preserve valid values', () => {
      const input = {
        bloomThreshold: 0.5,
        hdrMultiplier: 20.0,
        controlType: 'fly' as const,
      };

      const result = validateRenderingSettings(input);

      expect(result.bloomThreshold).toBe(0.5);
      expect(result.hdrMultiplier).toBe(20.0);
      expect(result.controlType).toBe('fly');
    });
  });

  describe('mergeSettings', () => {
    it('should merge with defaults', () => {
      const result = mergeSettings({
        bloomThreshold: 0.05,
      });

      expect(result.bloomThreshold).toBe(0.05);
      expect(result.toneMapping).toBe('ACES');
    });
  });

  describe('generateSettingsKey', () => {
    it('should generate a proper storage key', () => {
      const key = generateSettingsKey('test-scene');
      expect(key).toBe('luxar-rendering-settings-test-scene');
    });

    it('should sanitize scene ID', () => {
      const key = generateSettingsKey('test@scene!');
      expect(key).toBe('luxar-rendering-settings-test_scene_');
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
