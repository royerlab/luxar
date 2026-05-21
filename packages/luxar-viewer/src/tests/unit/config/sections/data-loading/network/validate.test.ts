/**
 * Tests for validateDataLoadingNetwork
 * (src/config/sections/data-loading/network/validate.ts).
 */

import { describe, it, expect } from 'vitest';
import { validateDataLoadingNetwork } from '../../../../../../config/sections/data-loading/network/validate';
import { cloneConfig, invokeValidator } from '../../../_fixtures';

describe('validateDataLoadingNetwork', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateDataLoadingNetwork).valid).toBe(true);
  });

  it('should error when network timeout is zero', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.timeoutMs = 0;

    const result = invokeValidator(validateDataLoadingNetwork, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid network timeout'));
  });

  it('should error when network timeout is negative', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.timeoutMs = -1000;

    const result = invokeValidator(validateDataLoadingNetwork, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid network timeout'));
  });

  // validationTimeoutMs is the per-request budget for cache validation in
  // fetchWithRetry. Pre-fix it was unvalidated; 0 / negative / NaN / Infinity
  // all flowed through and produced surprising abort/retry behavior.
  it('should error when validationTimeoutMs is zero', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.validationTimeoutMs = 0;

    const result = invokeValidator(validateDataLoadingNetwork, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid validation timeout'));
  });

  it('should error when validationTimeoutMs is negative', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.validationTimeoutMs = -50;

    const result = invokeValidator(validateDataLoadingNetwork, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid validation timeout'));
  });

  it('should error when validationTimeoutMs is NaN', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.validationTimeoutMs = Number.NaN;

    const result = invokeValidator(validateDataLoadingNetwork, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid validation timeout'));
  });

  it('should error when validationTimeoutMs is Infinity', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.validationTimeoutMs = Number.POSITIVE_INFINITY;

    const result = invokeValidator(validateDataLoadingNetwork, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid validation timeout'));
  });

  it('should error when maxConcurrent is zero', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.maxConcurrent = 0;

    const result = invokeValidator(validateDataLoadingNetwork, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid max concurrent requests')
    );
  });

  it('should error when maxConcurrent is negative', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.maxConcurrent = -3;

    const result = invokeValidator(validateDataLoadingNetwork, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid max concurrent requests')
    );
  });

  it('should error when retryAttempts is negative', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.retryAttempts = -1;

    const result = invokeValidator(validateDataLoadingNetwork, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid retry attempts'));
  });

  it('should accept retryAttempts of zero (no retries)', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.retryAttempts = 0;

    const result = invokeValidator(validateDataLoadingNetwork, cfg);

    expect(result.errors.filter((e) => e.includes('retry attempts'))).toHaveLength(0);
  });

  // NaN slipped past `<= 0` because comparisons with NaN are always false;
  // Infinity slipped past `<= 0` for the same reason; non-integers slipped
  // past integer-only knobs (concurrency, retry counts).
  it('should error when network timeoutMs is NaN', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.timeoutMs = Number.NaN;
    const result = invokeValidator(validateDataLoadingNetwork, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid network timeout'));
  });

  it('should error when network timeoutMs is Infinity', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.timeoutMs = Number.POSITIVE_INFINITY;
    const result = invokeValidator(validateDataLoadingNetwork, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid network timeout'));
  });

  it('should error when maxConcurrent is non-integer', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.maxConcurrent = 2.5;
    const result = invokeValidator(validateDataLoadingNetwork, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid max concurrent requests')
    );
  });

  it('should error when retryAttempts is non-integer', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.retryAttempts = 1.5;
    const result = invokeValidator(validateDataLoadingNetwork, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid retry attempts'));
  });
});
