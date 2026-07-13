/**
 * Tests for validateDataLoadingPerformance
 * (src/config/sections/data-loading/performance/validate.ts).
 */

import { describe, it, expect } from 'vitest';
import { validateDataLoadingPerformance } from '../../../../../../config/sections/data-loading/performance/validate';
import { cloneConfig, invokeValidator } from '../../../_fixtures';

describe('validateDataLoadingPerformance', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateDataLoadingPerformance).valid).toBe(true);
  });

  it('should error when workerProjectionTimeoutMs is negative', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.performance.workerProjectionTimeoutMs = -1;

    const result = invokeValidator(validateDataLoadingPerformance, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid workerProjectionTimeoutMs')
    );
  });

  it('accepts 0 timeouts (disabled) — emits zero errors', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.performance.workerProjectionTimeoutMs = 0;
    cfg.dataLoading.performance.workerInitTimeoutMs = 0;

    const result = invokeValidator(validateDataLoadingPerformance, cfg);

    // Stronger than the previous `.find(...includes('Timeout'))` substring match
    // (which was case-sensitive and case-fragile). Assert the strong contract:
    // a config with zero timeouts is valid and has no errors at all.
    expect(result.valid).toBe(true);
    expect(result.errors).toEqual([]);
  });

  // [G3][P5] / [M5][P11] Audit: workerInitTimeoutMs lives in the source
  // (validate.ts:32-37) but had zero direct coverage — the original test
  // suite only ever set it via the "0 disables" path. NaN/Infinity/negative
  // must all error explicitly, matching the per-knob behaviour for the
  // sibling visibility/projection timeouts.
  it('should error when workerInitTimeoutMs is NaN', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.performance.workerInitTimeoutMs = NaN;
    const result = invokeValidator(validateDataLoadingPerformance, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid workerInitTimeoutMs'));
  });

  it('should error when workerInitTimeoutMs is +Infinity', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.performance.workerInitTimeoutMs = Number.POSITIVE_INFINITY;
    const result = invokeValidator(validateDataLoadingPerformance, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid workerInitTimeoutMs'));
  });

  it('should error when workerInitTimeoutMs is negative', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.performance.workerInitTimeoutMs = -1;
    const result = invokeValidator(validateDataLoadingPerformance, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid workerInitTimeoutMs'));
  });

  it('should accept a positive workerInitTimeoutMs (the recommended path)', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.performance.workerInitTimeoutMs = 30_000;
    const result = invokeValidator(validateDataLoadingPerformance, cfg);
    expect(result.errors.filter((e) => e.includes('workerInitTimeoutMs'))).toEqual([]);
  });
});
