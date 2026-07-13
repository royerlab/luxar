/**
 * Tests for the validateDataLoading dispatcher
 * (src/config/sections/data-loading/validate.ts).
 *
 * The dispatcher composes the network/memory/performance sub-validators
 * (covered separately) and inlines the trivial spatial-section checks
 * that don't warrant their own validate.ts. These tests exercise the
 * spatial branch + a smoke that the dispatcher calls each sub-validator.
 */

import { describe, it, expect } from 'vitest';
import { validateDataLoading } from '../../../../../config/sections/data-loading/validate';
import { cloneConfig, invokeValidator } from '../../_fixtures';

describe('validateDataLoading', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateDataLoading).valid).toBe(true);
  });

  // ---- spatial (inlined in the dispatcher) ----------------------------

  it('should error when spatial defaultTolerance is zero', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.spatial.defaultTolerance = 0;

    const result = invokeValidator(validateDataLoading, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid spatial default tolerance')
    );
  });

  it('should error when spatial defaultTolerance is negative', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.spatial.defaultTolerance = -1;

    const result = invokeValidator(validateDataLoading, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid spatial default tolerance')
    );
  });

  it('should error when spatial defaultMaxRadius is zero', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.spatial.defaultMaxRadius = 0;

    const result = invokeValidator(validateDataLoading, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid spatial default max radius')
    );
  });

  it('should error when spatial defaultMaxRadius is negative', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.spatial.defaultMaxRadius = -1;

    const result = invokeValidator(validateDataLoading, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid spatial default max radius')
    );
  });

  it('rejects NaN spatial.defaultTolerance', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.spatial.defaultTolerance = NaN;
    const result = invokeValidator(validateDataLoading, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('spatial default tolerance'));
  });

  it('rejects NaN spatial.defaultMaxRadius', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.spatial.defaultMaxRadius = NaN;
    const result = invokeValidator(validateDataLoading, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('spatial default max radius'));
  });

  // ---- dispatcher smoke ------------------------------------------------

  it('routes through network + memory + performance sub-validators', () => {
    // A simultaneous failure in each sub-section confirms the dispatcher
    // is wiring all three sub-validators (network/memory/performance) +
    // inlining spatial. Pure smoke — sub-validators have their own
    // exhaustive coverage.
    const cfg = cloneConfig();
    cfg.dataLoading.network.timeoutMs = -1; // network
    cfg.dataLoading.memory.targetHeapUsage = 0; // memory
    cfg.dataLoading.performance.workerProjectionTimeoutMs = -1; // performance
    cfg.dataLoading.spatial.defaultTolerance = 0; // spatial (inlined)

    const result = invokeValidator(validateDataLoading, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid network timeout'));
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid target heap usage'));
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid workerProjectionTimeoutMs')
    );
    expect(result.errors).toContainEqual(
      expect.stringContaining('Invalid spatial default tolerance')
    );
  });
});
