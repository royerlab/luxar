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
  filterExportableSettings,
  calculatePerformanceImpact,
  RenderingSettings,
  DEFAULT_RENDERING_SETTINGS,
} from '../ui/rendering-controls-utils';

describe('rendering-controls-utils', () => {
  describe('validateRenderingSettings', () => {
    it('should apply defaults for missing values', () => {
      const result = validateRenderingSettings({});

      expect(result.showStats).toBe(false);
      expect(result.antialiasing).toBe('smaa');
      expect(result.bloomEnabled).toBe(true);
      expect(result.controlType).toBe('orbit');
    });

    it('should validate boolean values', () => {
      const result = validateRenderingSettings({
        showStats: true,
        showAxes: true,
        bloomEnabled: false,
      });

      expect(result.showStats).toBe(true);
      expect(result.showAxes).toBe(true);
      expect(result.bloomEnabled).toBe(false);
    });

    it('should validate anti-aliasing type', () => {
      const valid = validateRenderingSettings({ antialiasing: 'fxaa' });
      expect(valid.antialiasing).toBe('fxaa');

      const invalid = validateRenderingSettings({ antialiasing: 'invalid' as any });
      expect(invalid.antialiasing).toBe('smaa'); // Default
    });

    it('should clamp numeric values', () => {
      const result = validateRenderingSettings({
        msaaSamples: 7,
        ssaaMultiplier: 10,
        bloomIntensity: -1,
        exposure: 20,
        autoRotateSpeed: 15,
        flyMovementSpeed: 20000,
        flyDamping: 2,
      });

      expect(result.msaaSamples).toBe(8); // Rounded to valid value
      expect(result.ssaaMultiplier).toBe(4.0); // Clamped to max
      expect(result.bloomIntensity).toBe(0); // Clamped to min
      expect(result.exposure).toBe(10); // Clamped to max
      expect(result.autoRotateSpeed).toBe(10); // Clamped to max
      expect(result.flyMovementSpeed).toBe(10000); // Clamped to max
      expect(result.flyDamping).toBe(1); // Clamped to max
    });

    it('should validate color strings', () => {
      const valid = validateRenderingSettings({ backgroundColor: '#ff0000' });
      expect(valid.backgroundColor).toBe('#ff0000');

      const invalid = validateRenderingSettings({ backgroundColor: 'not-a-color' });
      expect(invalid.backgroundColor).toBe('#000000'); // Default
    });
  });

  describe('mergeSettings', () => {
    it('should merge with defaults', () => {
      const partial = { showStats: true, bloomIntensity: 2.0 };
      const merged = mergeSettings(partial);

      expect(merged.showStats).toBe(true);
      expect(merged.bloomIntensity).toBe(2.0);
      expect(merged.antialiasing).toBe('smaa'); // From defaults
    });

    it('should use custom defaults', () => {
      const customDefaults: RenderingSettings = {
        ...DEFAULT_RENDERING_SETTINGS,
        antialiasing: 'fxaa',
      };

      const merged = mergeSettings({ showStats: true }, customDefaults);

      expect(merged.showStats).toBe(true);
      expect(merged.antialiasing).toBe('fxaa');
    });
  });

  describe('generateSettingsKey', () => {
    it('should generate key with default prefix', () => {
      const key = generateSettingsKey('my-scene');
      expect(key).toBe('luxar-rendering-settings-my-scene');
    });

    it('should use custom prefix', () => {
      const key = generateSettingsKey('scene-123', 'custom-prefix');
      expect(key).toBe('custom-prefix-scene-123');
    });

    it('should sanitize scene ID', () => {
      const key = generateSettingsKey('scene/with\\special@chars!');
      expect(key).toBe('luxar-rendering-settings-scene_with_special_chars_');
    });
  });

  describe('serializeSettings', () => {
    it('should serialize settings to JSON', () => {
      const settings: RenderingSettings = {
        ...DEFAULT_RENDERING_SETTINGS,
        showStats: true,
        bloomIntensity: 1.5,
      };

      const json = serializeSettings(settings);
      const parsed = JSON.parse(json);

      expect(parsed.showStats).toBe(true);
      expect(parsed.bloomIntensity).toBe(1.5);
    });
  });

  describe('deserializeSettings', () => {
    it('should deserialize valid JSON', () => {
      const json = JSON.stringify({ showStats: true, bloomIntensity: 2.0 });
      const settings = deserializeSettings(json);

      expect(settings).not.toBeNull();
      expect(settings!.showStats).toBe(true);
      expect(settings!.bloomIntensity).toBe(2.0);
    });

    it('should return null for invalid JSON', () => {
      expect(deserializeSettings('not json')).toBeNull();
      expect(deserializeSettings('')).toBeNull();
    });

    it('should return null for non-object JSON', () => {
      expect(deserializeSettings('123')).toBeNull();
      expect(deserializeSettings('null')).toBeNull();
      expect(deserializeSettings('"string"')).toBeNull();
    });
  });

  describe('isValidColor', () => {
    it('should validate hex colors', () => {
      expect(isValidColor('#000000')).toBe(true);
      expect(isValidColor('#FFFFFF')).toBe(true);
      expect(isValidColor('#ff00aa')).toBe(true);
    });

    it('should reject invalid hex colors', () => {
      expect(isValidColor('#fff')).toBe(false); // Too short
      expect(isValidColor('#gggggg')).toBe(false); // Invalid chars
      expect(isValidColor('ff0000')).toBe(false); // Missing #
    });

    it('should validate RGB/RGBA colors', () => {
      expect(isValidColor('rgb(255, 0, 0)')).toBe(true);
      expect(isValidColor('rgba(0, 0, 0, 0.5)')).toBe(true);
    });

    it('should validate named colors', () => {
      expect(isValidColor('black')).toBe(true);
      expect(isValidColor('white')).toBe(true);
      expect(isValidColor('RED')).toBe(true); // Case insensitive
    });

    it('should reject unknown named colors', () => {
      expect(isValidColor('purple')).toBe(false);
      expect(isValidColor('orange')).toBe(false);
    });
  });

  describe('clampMSAASamples', () => {
    it('should return valid power-of-2 values', () => {
      expect(clampMSAASamples(0)).toBe(0);
      expect(clampMSAASamples(2)).toBe(2);
      expect(clampMSAASamples(4)).toBe(4);
      expect(clampMSAASamples(8)).toBe(8);
      expect(clampMSAASamples(16)).toBe(16);
    });

    it('should round to nearest valid value', () => {
      expect(clampMSAASamples(1)).toBe(2);
      expect(clampMSAASamples(3)).toBe(4);
      expect(clampMSAASamples(6)).toBe(8);
      expect(clampMSAASamples(12)).toBe(16);
      expect(clampMSAASamples(32)).toBe(16); // Max
    });

    it('should handle negative values', () => {
      expect(clampMSAASamples(-1)).toBe(0);
      expect(clampMSAASamples(-10)).toBe(0);
    });
  });

  describe('settingsChanged', () => {
    const settings1: RenderingSettings = {
      ...DEFAULT_RENDERING_SETTINGS,
      showStats: true,
    };

    const settings2: RenderingSettings = {
      ...DEFAULT_RENDERING_SETTINGS,
      showStats: false,
    };

    it('should detect changes', () => {
      expect(settingsChanged(settings1, settings2)).toBe(true);
    });

    it('should detect no changes', () => {
      expect(settingsChanged(settings1, settings1)).toBe(false);
      expect(settingsChanged(settings1, { ...settings1 })).toBe(false);
    });
  });

  describe('getSettingsRequiringRebuild', () => {
    const base: RenderingSettings = DEFAULT_RENDERING_SETTINGS;

    it('should detect anti-aliasing changes', () => {
      const changed: RenderingSettings = {
        ...base,
        antialiasing: 'fxaa',
      };

      const rebuild = getSettingsRequiringRebuild(changed, base);
      expect(rebuild).toContain('antialiasing');
    });

    it('should detect MSAA sample changes', () => {
      const changed: RenderingSettings = {
        ...base,
        msaaSamples: 8,
      };

      const rebuild = getSettingsRequiringRebuild(changed, base);
      expect(rebuild).toContain('msaaSamples');
    });

    it('should detect SSAA multiplier changes', () => {
      const changed: RenderingSettings = {
        ...base,
        ssaaMultiplier: 2.0,
      };

      const rebuild = getSettingsRequiringRebuild(changed, base);
      expect(rebuild).toContain('ssaaMultiplier');
    });

    it('should detect tone mapping changes', () => {
      const changed: RenderingSettings = {
        ...base,
        toneMapping: 'agx',
      };

      const rebuild = getSettingsRequiringRebuild(changed, base);
      expect(rebuild).toContain('toneMapping');
    });

    it('should not flag non-rebuild changes', () => {
      const changed: RenderingSettings = {
        ...base,
        showStats: !base.showStats,
        bloomIntensity: base.bloomIntensity + 0.5,
      };

      const rebuild = getSettingsRequiringRebuild(changed, base);
      expect(rebuild).toHaveLength(0);
    });
  });

  describe('filterExportableSettings', () => {
    it('should remove background color', () => {
      const settings: RenderingSettings = {
        ...DEFAULT_RENDERING_SETTINGS,
        backgroundColor: '#ff0000',
        showStats: true,
      };

      const exported = filterExportableSettings(settings);

      expect(exported.showStats).toBe(true);
      expect(exported.backgroundColor).toBeUndefined();
    });
  });

  describe('calculatePerformanceImpact', () => {
    it('should calculate low impact for minimal settings', () => {
      const settings: RenderingSettings = {
        ...DEFAULT_RENDERING_SETTINGS,
        antialiasing: 'none',
        bloomEnabled: false,
        toneMapping: 'none',
        showStats: false,
        showAxes: false,
        showGrid: false,
        autoRotate: false,
      };

      const impact = calculatePerformanceImpact(settings);
      expect(impact).toBe(0);
    });

    it('should calculate high impact for demanding settings', () => {
      const settings: RenderingSettings = {
        ...DEFAULT_RENDERING_SETTINGS,
        antialiasing: 'ssaa',
        ssaaMultiplier: 2.0,
        msaaSamples: 8,
        bloomEnabled: true,
        bloomIntensity: 2.0,
        toneMapping: 'aces',
        showStats: true,
        showAxes: true,
        showGrid: true,
        autoRotate: true,
      };

      const impact = calculatePerformanceImpact(settings);
      expect(impact).toBeGreaterThan(50);
    });

    it('should cap at 100', () => {
      const settings: RenderingSettings = {
        ...DEFAULT_RENDERING_SETTINGS,
        antialiasing: 'ssaa',
        ssaaMultiplier: 4.0,
        bloomEnabled: true,
        bloomIntensity: 5.0,
      };

      const impact = calculatePerformanceImpact(settings);
      expect(impact).toBeLessThanOrEqual(100);
    });
  });
});
