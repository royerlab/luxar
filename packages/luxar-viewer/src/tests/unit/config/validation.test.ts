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

    it('should error when bloomThreshold is negative', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomThreshold = -0.1;

      const result = validateConfig(cfg);

      expect(result.errors).toContainEqual(expect.stringContaining('Invalid bloom.bloomThreshold'));
    });

    it('should error when bloomThreshold exceeds 1', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomThreshold = 1.5;

      const result = validateConfig(cfg);

      expect(result.errors).toContainEqual(expect.stringContaining('Invalid bloom.bloomThreshold'));
    });

    it('should accept bloomThreshold at boundary values (0 and 1)', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomThreshold = 0;
      expect(validateConfig(cfg).errors.filter((e) => e.includes('bloomThreshold'))).toHaveLength(
        0
      );

      cfg.renderingControls.defaults.bloomThreshold = 1;
      expect(validateConfig(cfg).errors.filter((e) => e.includes('bloomThreshold'))).toHaveLength(
        0
      );
    });

    it('should error when bloomLevels is less than 1', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomLevels = 0;

      const result = validateConfig(cfg);

      expect(result.errors).toContainEqual(expect.stringContaining('Invalid bloom.bloomLevels'));
    });

    it('should error when bloomLevels exceeds 12', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomLevels = 13;

      const result = validateConfig(cfg);

      expect(result.errors).toContainEqual(expect.stringContaining('Invalid bloom.bloomLevels'));
    });

    it('should accept bloomLevels at boundary values (1 and 12)', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomLevels = 1;
      expect(validateConfig(cfg).errors.filter((e) => e.includes('bloomLevels'))).toHaveLength(0);

      cfg.renderingControls.defaults.bloomLevels = 12;
      expect(validateConfig(cfg).errors.filter((e) => e.includes('bloomLevels'))).toHaveLength(0);
    });

    // NaN/Infinity must be caught explicitly — bare comparisons
    // with NaN are always false, so a naked `< 0 || > 10` check
    // would let NaN pass.
    it('NaN bloomStrength is rejected as error', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomStrength = NaN;
      expect(validateConfig(cfg).errors).toContainEqual(
        expect.stringContaining('Invalid bloom.bloomStrength')
      );
    });

    it('Infinity bloomRadius is rejected as error', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomRadius = Infinity;
      expect(validateConfig(cfg).errors).toContainEqual(
        expect.stringContaining('Invalid bloom.bloomRadius')
      );
    });

    it('NaN bloomThreshold is rejected as error', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomThreshold = NaN;
      expect(validateConfig(cfg).errors).toContainEqual(
        expect.stringContaining('Invalid bloom.bloomThreshold')
      );
    });

    it('fractional bloomLevels is rejected as error (must be integer)', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomLevels = 5.5;
      expect(validateConfig(cfg).errors).toContainEqual(
        expect.stringContaining('Invalid bloom.bloomLevels')
      );
    });

    it('NaN bloomLevels is rejected as error', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.bloomLevels = NaN;
      expect(validateConfig(cfg).errors).toContainEqual(
        expect.stringContaining('Invalid bloom.bloomLevels')
      );
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

    // validationTimeoutMs is the per-request budget for
    // cache validation in fetchWithRetry. Pre-fix it was unvalidated;
    // 0 / negative / NaN / Infinity all flowed through and produced
    // surprising abort/retry behavior.
    it('should error when validationTimeoutMs is zero', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.network.validationTimeoutMs = 0;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid validation timeout'));
    });

    it('should error when validationTimeoutMs is negative', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.network.validationTimeoutMs = -50;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid validation timeout'));
    });

    it('should error when validationTimeoutMs is NaN', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.network.validationTimeoutMs = Number.NaN;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid validation timeout'));
    });

    it('should error when validationTimeoutMs is Infinity', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.network.validationTimeoutMs = Number.POSITIVE_INFINITY;

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid validation timeout'));
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

    // tighten numeric validation across timeoutMs,
    // maxConcurrent, retryAttempts. Pre-fix, NaN slipped past `<= 0`
    // because comparisons with NaN are always false; Infinity slipped
    // past `<= 0` for the same reason; non-integers slipped past
    // integer-only knobs (concurrency, retry counts).
    it('should error when network timeoutMs is NaN', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.network.timeoutMs = Number.NaN;
      const result = validateConfig(cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid network timeout'));
    });

    it('should error when network timeoutMs is Infinity', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.network.timeoutMs = Number.POSITIVE_INFINITY;
      const result = validateConfig(cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid network timeout'));
    });

    it('should error when maxConcurrent is non-integer', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.network.maxConcurrent = 2.5;
      const result = validateConfig(cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('Invalid max concurrent requests')
      );
    });

    it('should error when retryAttempts is non-integer', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.network.retryAttempts = 1.5;
      const result = validateConfig(cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid retry attempts'));
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

  describe('NaN/Infinity hardening for memory + spatial + cache', () => {
    it('rejects NaN memory.targetHeapUsage', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.memory.targetHeapUsage = NaN;
      const result = validateConfig(cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('target heap usage'));
    });

    it('rejects Infinity memory.targetHeapUsage', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.memory.targetHeapUsage = Infinity;
      const result = validateConfig(cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('target heap usage'));
    });

    it('rejects NaN memory.minCacheMB', () => {
      const cfg = cloneConfig();
      cfg.dataLoading.memory.minCacheMB = NaN;
      const result = validateConfig(cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('min cache size'));
    });

    it('rejects NaN spatial.defaultTolerance', () => {
      const cfg = cloneConfig();
      if (cfg.dataLoading.spatial) cfg.dataLoading.spatial.defaultTolerance = NaN;
      const result = validateConfig(cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('spatial default tolerance'));
    });

    it('rejects NaN spatial.defaultMaxRadius', () => {
      const cfg = cloneConfig();
      if (cfg.dataLoading.spatial) cfg.dataLoading.spatial.defaultMaxRadius = NaN;
      const result = validateConfig(cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('spatial default max radius'));
    });

  });

  describe('NaN/Infinity hardening for camera + rendering + controls + input', () => {
    it('rejects NaN exposure (rendering)', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.exposure = NaN;
      const result = validateConfig(cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid exposure'));
    });

    it('rejects NaN globalOffset', () => {
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.globalOffset = NaN;
      const result = validateConfig(cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('Invalid globalOffset'));
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
