/**
 * Tests for validateDataLoadingMemory
 * (src/config/sections/data-loading/memory/validate.ts).
 */

import { describe, it, expect } from 'vitest';
import { validateDataLoadingMemory } from '../../../../../../config/sections/data-loading/memory/validate';
import { cloneConfig, invokeValidator } from '../../../_fixtures';

describe('validateDataLoadingMemory', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateDataLoadingMemory).valid).toBe(true);
  });

  it('should error when targetHeapUsage is zero', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.memory.targetHeapUsage = 0;

    const result = invokeValidator(validateDataLoadingMemory, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid target heap usage'));
  });

  it('should error when targetHeapUsage exceeds 1', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.memory.targetHeapUsage = 1.5;

    const result = invokeValidator(validateDataLoadingMemory, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid target heap usage'));
  });

  it('should error when targetHeapUsage is negative', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.memory.targetHeapUsage = -0.5;

    const result = invokeValidator(validateDataLoadingMemory, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid target heap usage'));
  });

  it('should error when minCacheMB is zero', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.memory.minCacheMB = 0;

    const result = invokeValidator(validateDataLoadingMemory, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid min cache size'));
  });

  it('should error when minCacheMB is negative', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.memory.minCacheMB = -100;

    const result = invokeValidator(validateDataLoadingMemory, cfg);

    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('Invalid min cache size'));
  });

  it('rejects NaN memory.targetHeapUsage', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.memory.targetHeapUsage = NaN;
    const result = invokeValidator(validateDataLoadingMemory, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('target heap usage'));
  });

  it('rejects Infinity memory.targetHeapUsage', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.memory.targetHeapUsage = Infinity;
    const result = invokeValidator(validateDataLoadingMemory, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('target heap usage'));
  });

  it('rejects NaN memory.minCacheMB', () => {
    const cfg = cloneConfig();
    cfg.dataLoading.memory.minCacheMB = NaN;
    const result = invokeValidator(validateDataLoadingMemory, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('min cache size'));
  });
});
