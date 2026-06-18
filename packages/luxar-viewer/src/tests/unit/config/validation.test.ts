/**
 * Tests for the validateConfig dispatcher + log helpers.
 *
 * Per-section validator coverage lives under src/tests/unit/config/sections/.
 * This file only covers (1) a smoke that the dispatcher accepts the default
 * config and aggregates errors across sub-validators, and (2) the
 * logValidationResults / validateAndLog log-side-effect behavior.
 */

import { describe, it, expect, vi, beforeEach, afterEach, type MockInstance } from 'vitest';
import { config } from '../../../config';
import type { AppConfig } from '../../../config/types';
import { validateConfig, logValidationResults, validateAndLog } from '../../../config/validation';
import type { ValidationResult } from '../../../config/validation';
import { cloneConfig } from './_fixtures';

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
      // [W3][P2] Audit: pre-strengthening this test only asserted
      // `errors.length >= 3` — that survives a mutation where the
      // dispatcher emits 3 copies of ONE category's error and skips
      // the other two. Strengthened: assert each of the three
      // perturbed sections produced its own keyed error message.
      // Smoke test that the dispatcher aggregates errors across sub-validators.
      // Per-section coverage lives in src/tests/unit/config/sections/.
      const cfg = cloneConfig();
      cfg.renderingControls.defaults.fov = 0; // camera error
      cfg.renderingControls.defaults.exposure = -11; // rendering error (out of [-10, 10])
      cfg.dataLoading.network.timeoutMs = -1; // data loading error

      const result = validateConfig(cfg);

      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThanOrEqual(3);
      // Each of the three perturbed sections must contribute its own
      // identifiable error string. A mutation that emitted the same
      // error 3x or dropped a section would now fail.
      expect(result.errors).toContainEqual(expect.stringContaining('camera FOV'));
      expect(result.errors).toContainEqual(expect.stringContaining('exposure'));
      expect(result.errors).toContainEqual(expect.stringContaining('network timeout'));
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
    // [W5][P2] Audit: pre-strengthening these tests only checked
    // a substring match, never that error/warning channels stayed
    // silent on success. A mutation that ALSO emitted an error on
    // success would have slipped past.
    expect(logSpy.error).not.toHaveBeenCalled();
    expect(logSpy.warning).not.toHaveBeenCalled();
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
    // [W5][P2] Stronger: validation failure must NOT also log a "passed"
    // success message; and error count must match the input (header + 2 errors).
    expect(logSpy.success).not.toHaveBeenCalled();
    expect(logSpy.error).toHaveBeenCalledTimes(3); // header + Error 1 + Error 2
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
    // [W5][P2] Warnings + valid: success still fires (validation passed),
    // errors stay silent, warnings == header + 1 warning.
    expect(logSpy.error).not.toHaveBeenCalled();
    expect(logSpy.warning).toHaveBeenCalledTimes(2);
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
