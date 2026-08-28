/**
 * Tests for validateDepthSort (src/config/sections/depth-sort/validate.ts).
 * Each test exercises the per-section validator directly via
 * invokeValidator().
 */

import { describe, it, expect } from 'vitest';
import { validateDepthSort } from '../../../../../config/sections/depth-sort/validate';
import { cloneConfig, invokeValidator } from '../../_fixtures';

describe('validateDepthSort', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateDepthSort).valid).toBe(true);
  });

  it('errors on a non-positive or non-finite angle threshold', () => {
    for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const cfg = cloneConfig();
      cfg.depthSort.angleThresholdDeg = bad;
      const result = invokeValidator(validateDepthSort, cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('depthSort.angleThresholdDeg'));
    }
  });

  it('errors on a non-positive or non-finite translation fraction', () => {
    for (const bad of [0, -0.05, Number.NaN]) {
      const cfg = cloneConfig();
      cfg.depthSort.translationFraction = bad;
      const result = invokeValidator(validateDepthSort, cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('depthSort.translationFraction')
      );
    }
  });

  it('errors on a negative or non-finite worker init timeout', () => {
    for (const bad of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const cfg = cloneConfig();
      cfg.depthSort.workerInitTimeoutMs = bad;
      const result = invokeValidator(validateDepthSort, cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('depthSort.workerInitTimeoutMs')
      );
    }
  });

  it('accepts 0 for the worker init timeout, but warns about the pile-up it re-opens', () => {
    // Unlike the two thresholds, 0 is MEANINGFUL here: it disables the
    // deadline (the shared withTimeout convention), it does not break the
    // scheduler — erroring on it would contradict init-with-guard.ts. It is
    // not free, though: the deadline is what guarantees the init promise
    // settles, and without it a worker that dies during async module
    // evaluation parks an unbounded pile of commit continuations.
    const cfg = cloneConfig();
    cfg.depthSort.workerInitTimeoutMs = 0;
    const result = invokeValidator(validateDepthSort, cfg);
    expect(result.valid).toBe(true);
    expect(result.warnings).toContainEqual(
      expect.stringContaining('depthSort.workerInitTimeoutMs')
    );
  });

  it('warns on a worker init timeout shorter than a cold WASM instantiate', () => {
    const cfg = cloneConfig();
    cfg.depthSort.workerInitTimeoutMs = 50;
    const result = invokeValidator(validateDepthSort, cfg);
    expect(result.valid).toBe(true);
    expect(result.warnings).toContainEqual(
      expect.stringContaining('depthSort.workerInitTimeoutMs')
    );
  });

  it('errors on a negative or non-integer synchronous-sort ceiling', () => {
    for (const bad of [-1, 2.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      const cfg = cloneConfig();
      cfg.depthSort.syncSortMaxElements = bad;
      const result = invokeValidator(validateDepthSort, cfg);
      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('depthSort.syncSortMaxElements')
      );
    }
  });

  it('warns that a zero synchronous-sort ceiling reinstates the unsorted frame', () => {
    const cfg = cloneConfig();
    cfg.depthSort.syncSortMaxElements = 0;
    const result = invokeValidator(validateDepthSort, cfg);
    expect(result.valid).toBe(true);
    expect(result.warnings).toContainEqual(expect.stringContaining('syncSortMaxElements'));
  });

  it("warns on a synchronous-sort ceiling past a frame's budget", () => {
    // The counting sort was measured at 16.2 ms for 1M elements.
    const cfg = cloneConfig();
    cfg.depthSort.syncSortMaxElements = 4_000_000;
    const result = invokeValidator(validateDepthSort, cfg);
    expect(result.valid).toBe(true);
    expect(result.warnings).toContainEqual(expect.stringContaining('syncSortMaxElements'));
  });

  it('warns (but stays valid) on design-defeating coarse thresholds', () => {
    const cfg = cloneConfig();
    cfg.depthSort.angleThresholdDeg = 90;
    cfg.depthSort.translationFraction = 2;
    const result = invokeValidator(validateDepthSort, cfg);
    expect(result.valid).toBe(true);
    expect(result.warnings).toContainEqual(expect.stringContaining('angleThresholdDeg'));
    expect(result.warnings).toContainEqual(expect.stringContaining('translationFraction'));
  });
});
