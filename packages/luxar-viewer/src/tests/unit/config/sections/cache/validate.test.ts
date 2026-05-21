/**
 * Tests for validateCache (src/config/sections/cache/validate.ts).
 *
 * Cache validation has no dedicated coverage in the original validation.test.ts —
 * cases were scattered across "data loading validation" and the NaN/Infinity
 * hardening block. This file consolidates those + adds boundary cases for
 * opfsOperationTimeoutMs and externalDatasetTtlMs.
 */

import { describe, it, expect } from 'vitest';
import { validateCache } from '../../../../../config/sections/cache/validate';
import { cloneConfig, invokeValidator } from '../../_fixtures';

describe('validateCache', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateCache).valid).toBe(true);
  });

  // ---- size knobs ------------------------------------------------------

  it('rejects negative cache.l0MaxSizeMB', () => {
    const cfg = cloneConfig();
    cfg.cache.l0MaxSizeMB = -1;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.l0MaxSizeMB'));
  });

  it('rejects zero cache.l0MaxSizeMB', () => {
    const cfg = cloneConfig();
    cfg.cache.l0MaxSizeMB = 0;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.l0MaxSizeMB'));
  });

  it('rejects NaN cache.l1MaxSizeMB', () => {
    const cfg = cloneConfig();
    cfg.cache.l1MaxSizeMB = NaN;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.l1MaxSizeMB'));
  });

  it('rejects cache.l1MaxSizeMB < 10 (below SegmentedLRUCache metadata floor)', () => {
    const cfg = cloneConfig();
    cfg.cache.l1MaxSizeMB = 5;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringMatching(/cache\.l1MaxSizeMB.*must be ≥ 10/));
  });

  it('accepts cache.l1MaxSizeMB at the floor (10)', () => {
    const cfg = cloneConfig();
    cfg.cache.l1MaxSizeMB = 10;
    const result = invokeValidator(validateCache, cfg);
    expect(result.errors.filter((e) => e.includes('cache.l1MaxSizeMB'))).toEqual([]);
  });

  it('rejects Infinity cache.l2MaxSizeMB', () => {
    const cfg = cloneConfig();
    cfg.cache.l2MaxSizeMB = Number.POSITIVE_INFINITY;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.l2MaxSizeMB'));
  });

  // ---- opfsOperationTimeoutMs -----------------------------------------

  it('rejects NaN cache.opfsOperationTimeoutMs', () => {
    const cfg = cloneConfig();
    cfg.cache.opfsOperationTimeoutMs = NaN;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.opfsOperationTimeoutMs'));
  });

  it('rejects zero cache.opfsOperationTimeoutMs', () => {
    const cfg = cloneConfig();
    cfg.cache.opfsOperationTimeoutMs = 0;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.opfsOperationTimeoutMs'));
  });

  it('rejects negative cache.opfsOperationTimeoutMs', () => {
    const cfg = cloneConfig();
    cfg.cache.opfsOperationTimeoutMs = -100;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.opfsOperationTimeoutMs'));
  });

  it('rejects Infinity cache.opfsOperationTimeoutMs', () => {
    const cfg = cloneConfig();
    cfg.cache.opfsOperationTimeoutMs = Number.POSITIVE_INFINITY;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.opfsOperationTimeoutMs'));
  });

  it('accepts positive cache.opfsOperationTimeoutMs', () => {
    const cfg = cloneConfig();
    cfg.cache.opfsOperationTimeoutMs = 5_000;
    const result = invokeValidator(validateCache, cfg);
    expect(result.errors.filter((e) => e.includes('opfsOperationTimeoutMs'))).toEqual([]);
  });

  // ---- externalDatasetTtlMs (null is explicitly allowed) --------------

  it('accepts null cache.externalDatasetTtlMs', () => {
    const cfg = cloneConfig();
    cfg.cache.externalDatasetTtlMs = null;
    const result = invokeValidator(validateCache, cfg);
    expect(result.errors.filter((e) => e.includes('externalDatasetTtlMs'))).toEqual([]);
  });

  it('rejects NaN cache.externalDatasetTtlMs', () => {
    const cfg = cloneConfig();
    cfg.cache.externalDatasetTtlMs = NaN;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.externalDatasetTtlMs'));
  });

  it('rejects negative cache.externalDatasetTtlMs', () => {
    const cfg = cloneConfig();
    cfg.cache.externalDatasetTtlMs = -1000;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.externalDatasetTtlMs'));
  });

  it('rejects zero cache.externalDatasetTtlMs', () => {
    const cfg = cloneConfig();
    cfg.cache.externalDatasetTtlMs = 0;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.externalDatasetTtlMs'));
  });

  it('accepts positive cache.externalDatasetTtlMs (24h)', () => {
    const cfg = cloneConfig();
    cfg.cache.externalDatasetTtlMs = 86_400_000;
    const result = invokeValidator(validateCache, cfg);
    expect(result.errors.filter((e) => e.includes('externalDatasetTtlMs'))).toEqual([]);
  });
});
