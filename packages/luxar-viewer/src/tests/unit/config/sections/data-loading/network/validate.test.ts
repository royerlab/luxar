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

  // [R11/D-G3][P5] -Infinity coverage. A `<= 0` check that omits
  // isFinite() catches -Infinity correctly (it IS ≤ 0), but a mutation
  // that flipped to `< 0 && isFinite(v)` would let -Infinity through.
  // Pin explicitly so the symmetric ±Infinity boundary is locked.
  it('should error when network timeoutMs is -Infinity', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.timeoutMs = Number.NEGATIVE_INFINITY;
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

  // [G4][P5] / [G20][P5] Audit: the `validationTimeoutMs < 3000` warning
  // branch (validate.ts:28-42) had no test. Cover the boundary explicitly:
  // 3000 is allowed silently, 2999 fires the warning, and anything between
  // 1 and 2999 must trigger it. The error branch (≤0) takes precedence so
  // 0 still errors and never reaches the warning gate — assert that
  // mutually-exclusive behaviour.
  it('emits a "Very low cache validation timeout" warning when validationTimeoutMs < 3000', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.validationTimeoutMs = 2000;
    const result = invokeValidator(validateDataLoadingNetwork, cfg);
    expect(result.valid).toBe(true);
    expect(result.warnings).toContainEqual(
      expect.stringContaining('Very low cache validation timeout')
    );
    // Pinpoint that the warning message contains the actual ms value
    // (a mutant that hardcodes the number would not).
    expect(result.warnings).toContainEqual(expect.stringContaining('2000'));
  });

  it('emits the validation-timeout warning at the boundary minus 1 (2999)', () => {
    // [P5] boundary first-class: the strict `< 3000` predicate fires at 2999.
    const cfg = cloneConfig();
    cfg.dataLoading.network.validationTimeoutMs = 2999;
    const result = invokeValidator(validateDataLoadingNetwork, cfg);
    expect(result.warnings).toContainEqual(
      expect.stringContaining('Very low cache validation timeout')
    );
  });

  it('does NOT emit the validation-timeout warning at the boundary (3000)', () => {
    // [P5] boundary first-class: `< 3000` is strict, so 3000 is silent.
    const cfg = cloneConfig();
    cfg.dataLoading.network.validationTimeoutMs = 3000;
    const result = invokeValidator(validateDataLoadingNetwork, cfg);
    expect(result.warnings.filter((w) => w.includes('Very low cache validation timeout'))).toEqual(
      []
    );
  });

  it('does NOT emit the validation-timeout warning at high values', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.network.validationTimeoutMs = 5000;
    const result = invokeValidator(validateDataLoadingNetwork, cfg);
    expect(result.warnings.filter((w) => w.includes('Very low cache validation timeout'))).toEqual(
      []
    );
  });

  it('error branch (validationTimeoutMs=0) suppresses the warning branch', () => {
    // The source uses `else if` — 0 must error and NOT also warn,
    // otherwise an operator sees both messages and the diagnostic is noisy.
    const cfg = cloneConfig();
    cfg.dataLoading.network.validationTimeoutMs = 0;
    const result = invokeValidator(validateDataLoadingNetwork, cfg);
    expect(result.valid).toBe(false);
    expect(result.warnings.filter((w) => w.includes('Very low cache validation timeout'))).toEqual(
      []
    );
  });
});
