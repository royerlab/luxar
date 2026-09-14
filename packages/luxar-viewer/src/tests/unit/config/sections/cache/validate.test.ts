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

  // [G13][P5] Audit: pre-audit l0MaxSizeMB had only "negative" and "zero"
  // coverage — NaN slips past `<= 0` (NaN comparisons are always false)
  // and Infinity passes `> 0` but represents an unbounded cache.
  it('rejects NaN cache.l0MaxSizeMB', () => {
    const cfg = cloneConfig();
    cfg.cache.l0MaxSizeMB = NaN;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.l0MaxSizeMB'));
  });

  it('rejects Infinity cache.l0MaxSizeMB', () => {
    const cfg = cloneConfig();
    cfg.cache.l0MaxSizeMB = Number.POSITIVE_INFINITY;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.l0MaxSizeMB'));
  });

  // [G14][P5] Audit: pre-audit l2MaxSizeMB only had Infinity coverage —
  // the NaN/negative/zero classes were missing. Each must error.
  it('rejects NaN cache.l2MaxSizeMB', () => {
    const cfg = cloneConfig();
    cfg.cache.l2MaxSizeMB = NaN;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.l2MaxSizeMB'));
  });

  it('rejects negative cache.l2MaxSizeMB', () => {
    const cfg = cloneConfig();
    cfg.cache.l2MaxSizeMB = -50;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.l2MaxSizeMB'));
  });

  it('rejects zero cache.l2MaxSizeMB', () => {
    const cfg = cloneConfig();
    cfg.cache.l2MaxSizeMB = 0;
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

  // ---- opfsTimeoutTripThreshold ----------------------------------------

  it('rejects non-integer, zero, negative, and non-finite opfsTimeoutTripThreshold', () => {
    for (const bad of [0, -1, 1.5, NaN, Infinity]) {
      const cfg = cloneConfig();
      cfg.cache.opfsTimeoutTripThreshold = bad;
      const result = invokeValidator(validateCache, cfg);
      expect(result.valid, `threshold=${bad} must be rejected`).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('cache.opfsTimeoutTripThreshold')
      );
    }
  });

  it('accepts integer opfsTimeoutTripThreshold >= 1', () => {
    for (const good of [1, 3, 10]) {
      const cfg = cloneConfig();
      cfg.cache.opfsTimeoutTripThreshold = good;
      const result = invokeValidator(validateCache, cfg);
      expect(result.valid, `threshold=${good} must be accepted`).toBe(true);
    }
  });

  it('requires a positive integer opfsReadConcurrency', () => {
    for (const bad of [0, -1, 1.5, NaN, Infinity]) {
      const cfg = cloneConfig();
      cfg.cache.opfsReadConcurrency = bad;
      const result = invokeValidator(validateCache, cfg);
      expect(result.valid, `concurrency=${bad} must be rejected`).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('cache.opfsReadConcurrency'));
    }

    const cfg = cloneConfig();
    cfg.cache.opfsReadConcurrency = 16;
    expect(invokeValidator(validateCache, cfg).valid).toBe(true);
  });

  it('rejects Infinity cache.opfsOperationTimeoutMs', () => {
    const cfg = cloneConfig();
    cfg.cache.opfsOperationTimeoutMs = Number.POSITIVE_INFINITY;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.opfsOperationTimeoutMs'));
  });

  // [R11/D-G3][P5] -Infinity coverage. A bare `> 0` check (without
  // isFinite()) accepts +Infinity AND rejects -Infinity for the wrong
  // reason: -Infinity < 0 is true, so the `< 0` path is taken with
  // bypassed sign-handling, but a mutation that swapped to `>= 0`
  // would let -Infinity through. Pin all three timeout/size fields.
  it('rejects -Infinity cache.opfsOperationTimeoutMs', () => {
    const cfg = cloneConfig();
    cfg.cache.opfsOperationTimeoutMs = Number.NEGATIVE_INFINITY;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.opfsOperationTimeoutMs'));
  });

  it('rejects -Infinity cache.l0MaxSizeMB', () => {
    const cfg = cloneConfig();
    cfg.cache.l0MaxSizeMB = Number.NEGATIVE_INFINITY;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.l0MaxSizeMB'));
  });

  it('rejects -Infinity cache.l2MaxSizeMB', () => {
    const cfg = cloneConfig();
    cfg.cache.l2MaxSizeMB = Number.NEGATIVE_INFINITY;
    const result = invokeValidator(validateCache, cfg);
    expect(result.valid).toBe(false);
    expect(result.errors).toContainEqual(expect.stringContaining('cache.l2MaxSizeMB'));
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
