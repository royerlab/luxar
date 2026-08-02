/**
 * Unit tests for the perf-bench dataset-origin resolver.
 *
 * `resolvePerfDataBase` derives the origin the perf-bench specs fetch
 * datasets from, honoring the same `LUXAR_PERF_DATA_BASE` /
 * `LUXAR_PERF_DATA_PORT` overrides as `playwright.perf.config.ts`. The
 * function is pure and takes an explicit `env`, so these tests pass
 * throwaway env objects rather than mutating `process.env`.
 */

import { describe, it, expect } from 'vitest';
import { resolvePerfDataBase } from '../e2e/perf-data-base';

describe('resolvePerfDataBase', () => {
  it('defaults to http://localhost:9000 when no overrides are set', () => {
    expect(resolvePerfDataBase({})).toBe('http://localhost:9000');
  });

  it('honors LUXAR_PERF_DATA_PORT', () => {
    expect(resolvePerfDataBase({ LUXAR_PERF_DATA_PORT: '9100' })).toBe('http://localhost:9100');
  });

  it('lets LUXAR_PERF_DATA_BASE take precedence over a set port', () => {
    expect(
      resolvePerfDataBase({
        LUXAR_PERF_DATA_BASE: 'http://example.test:1234',
        LUXAR_PERF_DATA_PORT: '9100',
      })
    ).toBe('http://example.test:1234');
  });

  it('interpolates the port string verbatim (no numeric coercion)', () => {
    // A leading-zero port pins verbatim interpolation: `Number('09000')`
    // would normalize to 9000, so this fails if the derivation ever
    // coerces the port instead of splicing the raw string.
    expect(resolvePerfDataBase({ LUXAR_PERF_DATA_PORT: '09000' })).toBe('http://localhost:09000');
  });

  it('treats an empty LUXAR_PERF_DATA_BASE as set (?? semantics, not ||)', () => {
    // `??` only falls through on null/undefined, so an empty override is
    // returned as-is — matching the pre-existing inline derivation. A swap
    // to `||` would instead fall back to the port default and pass silently.
    expect(resolvePerfDataBase({ LUXAR_PERF_DATA_BASE: '', LUXAR_PERF_DATA_PORT: '9100' })).toBe(
      ''
    );
  });
});
