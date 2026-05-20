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
