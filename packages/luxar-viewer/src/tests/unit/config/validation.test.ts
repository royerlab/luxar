/**
 * Tests for configuration validation module
 *
 * These tests verify that validateConfig() correctly identifies invalid,
 * out-of-range, and inconsistent configuration values. The real production
 * config is imported and deep-cloned for mutation in each test.
 */

import { describe, it, expect, vi, beforeEach, afterEach, type MockInstance } from 'vitest';
import { config } from '../../../config';
import type { AppConfig } from '../../../config/types';
import { validateConfig, logValidationResults, validateAndLog } from '../../../config/validation';
import type { ValidationResult } from '../../../config/validation';

/** Deep clone the production config for safe mutation */
function cloneConfig(): AppConfig {
  return JSON.parse(JSON.stringify(config)) as AppConfig;
}

describe('validateConfig', () => {
  describe('default config', () => {
    it('should pass validation with no errors and no warnings', () => {
      const result = validateConfig(config as unknown as AppConfig);

      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
      expect(result.warnings).toHaveLength(0);
    });
  });

  describe('camera validation', () => {
    it('should error when FOV is less than 1', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.fov = 0;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid camera FOV'));
    });

    it('should error when FOV is greater than 180', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.fov = 200;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid camera FOV'));
    });

    it('should accept FOV at boundary values (1 and 180)', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.fov = 1;
      expect(validateConfig(cfg).errors.filter((e) => e.includes('FOV'))).toHaveLength(0);

      cfg.renderingControls.defaults.fov = 180;
      expect(validateConfig(cfg).errors.filter((e) => e.includes('FOV'))).toHaveLength(0);
    });

    it('should error when near plane is zero', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.near = 0;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid camera near plane'));
    });

    it('should error when near plane is negative', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.near = -1;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid camera near plane'));
    });

    it('should error when far plane is less than or equal to near', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.near = 10;
      cfg.renderingControls.defaults.far = 5;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid camera far plane'));
    });

    it('should error when far plane equals near', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.near = 10;
      cfg.renderingControls.defaults.far = 10;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid camera far plane'));
    });

    it('should error when fovMin is greater than or equal to fovMax', () => {
      const cfg = cloneConfig();
      cfg.camera.fovMin = 170;
      cfg.camera.fovMax = 10;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid FOV limits'));
    });

    it('should error when fovMin equals fovMax', () => {
      const cfg = cloneConfig();
      cfg.camera.fovMin = 50;
      cfg.camera.fovMax = 50;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid FOV limits'));
    });

    it('should warn on unusual fovSensitivity (zero)', () => {
      const cfg = cloneConfig();
      cfg.camera.fovSensitivity = 0;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(expect.stringContaining('Unusual FOV sensitivity'));
    });

    it('should warn on unusual fovSensitivity (negative)', () => {
      const cfg = cloneConfig();
      cfg.camera.fovSensitivity = -0.1;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(expect.stringContaining('Unusual FOV sensitivity'));
    });

    it('should warn on fovSensitivity greater than 1', () => {
      const cfg = cloneConfig();
      cfg.camera.fovSensitivity = 1.5;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(expect.stringContaining('Unusual FOV sensitivity'));
    });

    it('should not warn on fovSensitivity exactly 1', () => {
      const cfg = cloneConfig();
      cfg.camera.fovSensitivity = 1;

      const result = validateConfig(cfg);

      // fovSensitivity = 1 means <= 0 is false and > 1 is false, so no warning
      expect(result.warnings.filter((w) => w.includes('FOV sensitivity'))).toHaveLength(0);
    });
  });

  describe('rendering validation', () => {
    it('should error when exposure is out of range', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.exposure = -6;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid exposure'));
    });

    it('should accept exposure within valid range', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.exposure = 0;
      expect(validateConfig(cfg).errors.filter((e) => e.includes('exposure'))).toHaveLength(0);

      cfg.renderingControls.defaults.exposure = 5;
      expect(validateConfig(cfg).errors.filter((e) => e.includes('exposure'))).toHaveLength(0);

      cfg.renderingControls.defaults.exposure = -5;
      expect(validateConfig(cfg).errors.filter((e) => e.includes('exposure'))).toHaveLength(0);
    });
  });

  describe('bloom consistency validation', () => {
    it('should warn when bloomStrength is negative', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomStrength = -1;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(
        expect.stringContaining('Unusual bloom.bloomStrength')
      );
    });

    it('should warn when bloomStrength exceeds 10', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomStrength = 15;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(
        expect.stringContaining('Unusual bloom.bloomStrength')
      );
    });

    it('should warn when bloomRadius is negative', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomRadius = -0.5;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(expect.stringContaining('Unusual bloom.bloomRadius'));
    });

    it('should warn when bloomRadius exceeds 10', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomRadius = 11;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(expect.stringContaining('Unusual bloom.bloomRadius'));
    });

    it('should warn when bloomThreshold is negative', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomThreshold = -0.1;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(
        expect.stringContaining('Invalid bloom.bloomThreshold')
      );
    });

    it('should warn when bloomThreshold exceeds 1', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomThreshold = 1.5;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(
        expect.stringContaining('Invalid bloom.bloomThreshold')
      );
    });

    it('should accept bloomThreshold at boundary values (0 and 1)', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomThreshold = 0;
      expect(validateConfig(cfg).warnings.filter((w) => w.includes('bloomThreshold'))).toHaveLength(
        0
      );

      cfg.renderingControls.defaults.bloomThreshold = 1;
      expect(validateConfig(cfg).warnings.filter((w) => w.includes('bloomThreshold'))).toHaveLength(
        0
      );
    });

    it('should warn when bloomLevels is less than 1', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomLevels = 0;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(expect.stringContaining('Invalid bloom.bloomLevels'));
    });

    it('should warn when bloomLevels exceeds 12', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomLevels = 13;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(expect.stringContaining('Invalid bloom.bloomLevels'));
    });

    it('should accept bloomLevels at boundary values (1 and 12)', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomLevels = 1;
      expect(validateConfig(cfg).warnings.filter((w) => w.includes('bloomLevels'))).toHaveLength(0);

      cfg.renderingControls.defaults.bloomLevels = 12;
      expect(validateConfig(cfg).warnings.filter((w) => w.includes('bloomLevels'))).toHaveLength(0);
    });
  });

  describe('data loading validation', () => {
    it('should error when network timeout is zero', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.network.timeoutMs = 0;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid network timeout'));
    });

    it('should error when network timeout is negative', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.network.timeoutMs = -1000;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid network timeout'));
    });

    it('should error when maxConcurrent is zero', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.network.maxConcurrent = 0;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('Invalid max concurrent requests')
      );
    });

    it('should error when maxConcurrent is negative', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.network.maxConcurrent = -3;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('Invalid max concurrent requests')
      );
    });

    it('should error when retryAttempts is negative', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.network.retryAttempts = -1;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid retry attempts'));
    });

    it('should accept retryAttempts of zero (no retries)', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.network.retryAttempts = 0;

      const result = validateConfig(cfg);

      expect(result.errors.filter((e) => e.includes('retry attempts'))).toHaveLength(0);
    });

    it('should error when targetHeapUsage is zero', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.memory.targetHeapUsage = 0;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid target heap usage'));
    });

    it('should error when targetHeapUsage exceeds 1', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.memory.targetHeapUsage = 1.5;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid target heap usage'));
    });

    it('should error when targetHeapUsage is negative', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.memory.targetHeapUsage = -0.5;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid target heap usage'));
    });

    it('should error when minCacheMB is zero', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.memory.minCacheMB = 0;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid min cache size'));
    });

    it('should error when minCacheMB is negative', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.memory.minCacheMB = -100;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid min cache size'));
    });

    it('should error when spatial defaultTolerance is zero', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.spatial.defaultTolerance = 0;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('Invalid spatial default tolerance')
      );
    });

    it('should error when spatial defaultTolerance is negative', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.spatial.defaultTolerance = -1;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('Invalid spatial default tolerance')
      );
    });

    it('should error when spatial defaultMaxRadius is zero', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.spatial.defaultMaxRadius = 0;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('Invalid spatial default max radius')
      );
    });

    it('should error when spatial defaultMaxRadius is negative', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.spatial.defaultMaxRadius = -1;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('Invalid spatial default max radius')
      );
    });

    it('should error when workerVisibilityTimeoutMs is negative', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.performance.workerVisibilityTimeoutMs = -100;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('Invalid workerVisibilityTimeoutMs')
      );
    });

    it('should error when workerVisibilityTimeoutMs is non-finite', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.performance.workerVisibilityTimeoutMs = Number.POSITIVE_INFINITY;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('Invalid workerVisibilityTimeoutMs')
      );
    });

    it('should error when workerProjectionTimeoutMs is negative', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.performance.workerProjectionTimeoutMs = -1;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('Invalid workerProjectionTimeoutMs')
      );
    });

    it('should accept 0 timeouts (disabled)', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.performance.workerVisibilityTimeoutMs = 0;
      cfg.dataLoading.performance.workerProjectionTimeoutMs = 0;

      const result = validateConfig(cfg);

      // Doesn't matter if other rules fail; the timeout rules specifically
      // should not contribute errors.
      expect(result.errors.find((e) => e.includes('Timeout'))).toBeUndefined();
    });
  });

  describe('scene validation', () => {
    it('should error when backgroundColor is negative', () => {
      const cfg = cloneConfig();
      cfg.scene.backgroundColor = -1;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('Invalid scene background color')
      );
    });

    it('should error when backgroundColor exceeds 0xffffff', () => {
      const cfg = cloneConfig();
      cfg.scene.backgroundColor = 0x1000000;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('Invalid scene background color')
      );
    });

    it('should accept backgroundColor at boundaries (0x000000 and 0xffffff)', () => {
      const cfg = cloneConfig();
      cfg.scene.backgroundColor = 0x000000;
      expect(validateConfig(cfg).errors.filter((e) => e.includes('background color'))).toHaveLength(
        0
      );

      cfg.scene.backgroundColor = 0xffffff;
      expect(validateConfig(cfg).errors.filter((e) => e.includes('background color'))).toHaveLength(
        0
      );
    });

    it('should error when fitRatio is zero', () => {
      const cfg = cloneConfig();
      cfg.scene.defaultFitRatio = 0;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid scene fit ratio'));
    });

    it('should error when fitRatio is negative', () => {
      const cfg = cloneConfig();
      cfg.scene.defaultFitRatio = -0.5;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid scene fit ratio'));
    });

    it('should error when fitRatio exceeds 1', () => {
      const cfg = cloneConfig();
      cfg.scene.defaultFitRatio = 1.5;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid scene fit ratio'));
    });

    it('should accept fitRatio of exactly 1', () => {
      const cfg = cloneConfig();
      cfg.scene.defaultFitRatio = 1;

      const result = validateConfig(cfg);

      expect(result.errors.filter((e) => e.includes('fit ratio'))).toHaveLength(0);
    });
  });

  describe('input validation', () => {
    it('should warn when sensitivity is zero', () => {
      const cfg = cloneConfig();
      cfg.input.defaultSensitivity = 0;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(expect.stringContaining('Unusual input sensitivity'));
    });

    it('should warn when sensitivity is negative', () => {
      const cfg = cloneConfig();
      cfg.input.defaultSensitivity = -0.5;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(expect.stringContaining('Unusual input sensitivity'));
    });

    it('should warn when sensitivity exceeds 1', () => {
      const cfg = cloneConfig();
      cfg.input.defaultSensitivity = 2;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(expect.stringContaining('Unusual input sensitivity'));
    });

    it('should not warn when sensitivity is within valid range', () => {
      const cfg = cloneConfig();
      cfg.input.defaultSensitivity = 0.5;

      const result = validateConfig(cfg);

      expect(result.warnings.filter((w) => w.includes('input sensitivity'))).toHaveLength(0);
    });
  });

  describe('WebGL validation', () => {
    it('should error on invalid powerPreference', () => {
      const cfg = cloneConfig();
      (cfg.webgl.context as any).powerPreference = 'super-power';

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('Invalid WebGL powerPreference')
      );
    });

    it('should error on invalid precision', () => {
      const cfg = cloneConfig();
      (cfg.webgl.renderer as any).precision = 'ultrahighp';

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid WebGL precision'));
    });

    it('should accept all valid precision values', () => {
      const cfg = cloneConfig();
      for (const precision of ['highp', 'mediump', 'lowp']) {
        (cfg.webgl.renderer as any).precision = precision;
        const result = validateConfig(cfg);
        expect(result.errors.filter((e) => e.includes('precision'))).toHaveLength(0);
      }
    });

    it('should warn on unusual MSAA samples', () => {
      const cfg = cloneConfig();
      cfg.webgl.renderTarget.samples = 3;

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(expect.stringContaining('Unusual MSAA samples'));
    });

    it('should accept standard MSAA sample values', () => {
      const cfg = cloneConfig();
      for (const samples of [0, 2, 4, 8]) {
        cfg.webgl.renderTarget.samples = samples;
        const result = validateConfig(cfg);
        expect(result.warnings.filter((w) => w.includes('MSAA'))).toHaveLength(0);
      }
    });

    it('should warn on unusual colorSpace', () => {
      const cfg = cloneConfig();
      cfg.webgl.context.colorSpace = 'adobe-rgb';

      const result = validateConfig(cfg);

      expect(result.warnings).toContainEqual(expect.stringContaining('Unusual color space'));
    });

    it('should accept valid colorSpace values', () => {
      const cfg = cloneConfig();
      for (const colorSpace of ['srgb', 'display-p3', 'rec2020']) {
        cfg.webgl.context.colorSpace = colorSpace;
        const result = validateConfig(cfg);
        expect(result.warnings.filter((w) => w.includes('color space'))).toHaveLength(0);
      }
    });
  });

  describe('controls validation', () => {
    it('should error when ConfigRange min >= max', () => {
      const cfg = cloneConfig();
      cfg.controls.fly.movement.speed.min = 50;
      cfg.controls.fly.movement.speed.max = 0.5;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('fly.movement.speed'));
    });

    it('should error when ConfigRange default is outside [min, max]', () => {
      const cfg = cloneConfig();
      cfg.controls.orbit.damping.factor.default = 99;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('orbit.damping.factor'));
    });

    it('should pass with default controls config', () => {
      const result = validateConfig(config as unknown as AppConfig);

      expect(result.errors.filter((e) => e.includes('controls.'))).toHaveLength(0);
    });
  });

  describe('multiple errors', () => {
    it('should accumulate multiple errors from different categories', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.fov = 0; // camera error
      cfg.renderingControls.defaults.exposure = -10; // rendering error
      cfg.dataLoading.network.timeoutMs = -1; // data loading error

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThanOrEqual(3);
    });

    it('should accumulate both errors and warnings', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.fov = 0; // error
      cfg.renderingControls.defaults.bloomStrength = -1; // warning

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThanOrEqual(1);
      expect(result.warnings.length).toBeGreaterThanOrEqual(1);
    });
  });
});

describe('logValidationResults', () => {
  let logSpy: {
    success: MockInstance;
    error: MockInstance;
    warning: MockInstance;
  };

  beforeEach(() => {
    // Spy on console methods since log module uses them
    logSpy = {
      success: vi.spyOn(console, 'log').mockImplementation(() => {}),
      error: vi.spyOn(console, 'error').mockImplementation(() => {}),
      warning: vi.spyOn(console, 'warn').mockImplementation(() => {}),
    };
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should log success when validation passes', () => {
    const result: ValidationResult = {
      valid: true,
      errors: [],
      warnings: [],
    };

    logValidationResults(result);

    expect(logSpy.success).toHaveBeenCalledWith(expect.stringContaining('validation passed'));
  });

  it('should log errors when validation fails', () => {
    const result: ValidationResult = {
      valid: false,
      errors: ['Error 1', 'Error 2'],
      warnings: [],
    };

    logValidationResults(result);

    expect(logSpy.error).toHaveBeenCalledWith(expect.stringContaining('validation failed'));
    // Each error is also logged individually
    expect(logSpy.error).toHaveBeenCalledWith(expect.stringContaining('Error 1'));
    expect(logSpy.error).toHaveBeenCalledWith(expect.stringContaining('Error 2'));
  });

  it('should log warnings when present', () => {
    const result: ValidationResult = {
      valid: true,
      errors: [],
      warnings: ['Warning 1'],
    };

    logValidationResults(result);

    expect(logSpy.warning).toHaveBeenCalledWith(expect.stringContaining('1 warnings'));
    expect(logSpy.warning).toHaveBeenCalledWith(expect.stringContaining('Warning 1'));
  });

  it('should not log warnings when there are none', () => {
    const result: ValidationResult = {
      valid: true,
      errors: [],
      warnings: [],
    };

    logValidationResults(result);

    expect(logSpy.warning).not.toHaveBeenCalled();
  });
});

describe('validateAndLog', () => {
  beforeEach(() => {
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should return true for valid config', () => {
    const result = validateAndLog(config as unknown as AppConfig);

    expect(result).toBe(true);
  });

  it('should return false for invalid config', () => {
    const cfg = cloneConfig();
    cfg.renderingControls.defaults.fov = 0;

    const result = validateAndLog(cfg);

    expect(result).toBe(false);
  });
});
