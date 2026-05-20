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

  it('should error when workerVisibilityTimeoutMs is negative', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.performance.workerVisibilityTimeoutMs = -100;

    const result = invokeValidator(validateDataLoadingPerformance, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid workerVisibilityTimeoutMs')
    );
  });

  it('should error when workerVisibilityTimeoutMs is non-finite', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.performance.workerVisibilityTimeoutMs = Number.POSITIVE_INFINITY;

    const result = invokeValidator(validateDataLoadingPerformance, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid workerVisibilityTimeoutMs')
    );
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

  it('should accept 0 timeouts (disabled)', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.performance.workerVisibilityTimeoutMs = 0;
    cfg.dataLoading.performance.workerProjectionTimeoutMs = 0;

    const result = invokeValidator(validateDataLoadingPerformance, cfg);

    // The timeout rules specifically should not contribute errors.
    expect(result.errors.find((e) => e.includes('Timeout'))).toBeUndefined();
  });
});
