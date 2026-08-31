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
} from '../../../ui/rendering-controls/controls-utils';

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
        autoRotateAxis: 'sideways' as any,
      });

      expect(result.toneMapping).toBe(defaults.toneMapping);
      expect(result.controlType).toBe(defaults.controlType);
      expect(result.autoRotateAxis).toBe(defaults.autoRotateAxis);
      expect(defaults.autoRotateAxis).toBe('vertical');
    });

    it.each(['vertical', 'horizontal', 'view'])(
      'keeps %s as a valid auto-rotation axis',
      (axis) => {
        // Every token the controls module accepts must survive validation —
        // a stored choice silently reset to vertical on reload is the failure
        // this pins, and it would look like the setting "not sticking".
        const result = validateRenderingSettings({ autoRotateAxis: axis as any });
        expect(result.autoRotateAxis).toBe(axis);
      }
    );

    it('should accept ortho as a valid control type', () => {
      const result = validateRenderingSettings({
        controlType: 'ortho' as any,
      });
      expect(result.controlType).toBe('ortho');
    });

    it('preserves an explicit naturalDrag=false (user opt-out overrides platform default)', () => {
      // Regression: when a Mac user unchecks "Natural drag" the stored
      // `false` must survive validation, NOT get replaced by the
      // platform default of true.
      const result = validateRenderingSettings({ naturalDrag: false });
      expect(result.naturalDrag).toBe(false);
    });

    it('preserves an explicit naturalDrag=true', () => {
      const result = validateRenderingSettings({ naturalDrag: true });
      expect(result.naturalDrag).toBe(true);
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

    describe('boolean fields default on non-boolean injection', () => {
      it.each([
        ['bloomEnabled', 'bloomEnabled' as const],
        ['fxaaEnabled', 'fxaaEnabled' as const],
        ['msaaEnabled', 'msaaEnabled' as const],
        ['ssaaEnabled', 'ssaaEnabled' as const],
        ['vignetteEnabled', 'vignetteEnabled' as const],
        ['detectorNoiseEnabled', 'detectorNoiseEnabled' as const],
        ['chromaticLensDistortionEnabled', 'chromaticLensDistortionEnabled' as const],
        ['autoRotate', 'autoRotate' as const],
        ['naturalDrag', 'naturalDrag' as const],
        ['dynamicClippingEnabled', 'dynamicClippingEnabled' as const],
        ['adaptiveDPREnabled', 'adaptiveDPREnabled' as const],
        ['allowHighDPR', 'allowHighDPR' as const],
        ['cinematicMode', 'cinematicMode' as const],
      ])('%s rejects string injection and falls back to default', (_name, key) => {
        const defaults = getDefaultRenderingSettings();
        const result = validateRenderingSettings({ [key]: 'oops' as unknown as boolean });
        expect(result[key]).toBe(defaults[key]);
      });
    });

    describe('chromatic lens distortion fields', () => {
      it('rejects NaN focal length and defaults', () => {
        const defaults = getDefaultRenderingSettings();
        const result = validateRenderingSettings({
          chromaticLensFocalLengthX: NaN,
          chromaticLensFocalLengthY: -1,
        });
        expect(result.chromaticLensFocalLengthX).toBe(defaults.chromaticLensFocalLengthX);
        expect(result.chromaticLensFocalLengthY).toBe(defaults.chromaticLensFocalLengthY);
      });

      it('clamps distortion strengths to [-1, 1]', () => {
        const defaults = getDefaultRenderingSettings();
        const result = validateRenderingSettings({
          chromaticLensDistortionX: 5,
          chromaticLensDistortionY: -10,
          chromaticLensSkew: Infinity,
        });
        expect(result.chromaticLensDistortionX).toBe(defaults.chromaticLensDistortionX);
        expect(result.chromaticLensDistortionY).toBe(defaults.chromaticLensDistortionY);
        expect(result.chromaticLensSkew).toBe(defaults.chromaticLensSkew);
      });
    });

    describe('detector noise fields', () => {
      it('rejects NaN/Infinity numeric values', () => {
        const defaults = getDefaultRenderingSettings();
        const result = validateRenderingSettings({
          detectorNoiseReadoutSigma: NaN,
          detectorNoisePhotonGain: Infinity,
          detectorNoiseFpnSigma: -1,
        });
        expect(result.detectorNoiseReadoutSigma).toBe(defaults.detectorNoiseReadoutSigma);
        expect(result.detectorNoisePhotonGain).toBe(defaults.detectorNoisePhotonGain);
        expect(result.detectorNoiseFpnSigma).toBe(defaults.detectorNoiseFpnSigma);
      });
    });

    describe('AA fields', () => {
      it('clamps ssaaMultiplier to [1, 8]', () => {
        const defaults = getDefaultRenderingSettings();
        const result = validateRenderingSettings({ ssaaMultiplier: 0 });
        expect(result.ssaaMultiplier).toBe(defaults.ssaaMultiplier);
      });
    });

    describe('autoRotateSpeed', () => {
      it('clamps NaN to default', () => {
        const defaults = getDefaultRenderingSettings();
        const result = validateRenderingSettings({ autoRotateSpeed: NaN });
        expect(result.autoRotateSpeed).toBe(defaults.autoRotateSpeed);
      });

      it('preserves valid negative speed (counter-clockwise)', () => {
        const result = validateRenderingSettings({ autoRotateSpeed: -2 });
        expect(result.autoRotateSpeed).toBe(-2);
      });
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

    it('defaults to 0 (disabled) for non-finite input', () => {
      // NaN comparisons are always false, so without a type/finite
      // guard the function would silently return 16. Verify the
      // explicit guard.
      expect(clampMSAASamples(NaN)).toBe(0);
      expect(clampMSAASamples(Infinity)).toBe(0);
      expect(clampMSAASamples(-Infinity)).toBe(0);
    });

    it('defaults to 0 for non-numeric input (string, undefined, object)', () => {
      expect(clampMSAASamples('4' as unknown as number)).toBe(0);
      expect(clampMSAASamples(undefined as unknown as number)).toBe(0);
      expect(clampMSAASamples({} as unknown as number)).toBe(0);
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
