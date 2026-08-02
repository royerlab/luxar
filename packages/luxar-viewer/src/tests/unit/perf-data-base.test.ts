/**
 * Unit tests for the perf-bench dataset-port/origin resolvers.
 *
 * `resolvePerfDataPort` is the port `playwright.perf.config.ts` boots
 * the dataset server on; `resolvePerfDataBase` is the origin the
 * perf-bench specs fetch datasets from, embedding that same port. The
 * functions are pure and take an explicit `env`, so these tests pass
 * throwaway env objects rather than mutating `process.env`.
 */

import { describe, it, expect } from 'vitest';
import { resolvePerfDataBase, resolvePerfDataPort } from '../e2e/perf-data-base';

describe('resolvePerfDataPort', () => {
  it('defaults to 9000 when LUXAR_PERF_DATA_PORT is unset', () => {
    expect(resolvePerfDataPort({})).toBe(9000);
  });

  it('coerces LUXAR_PERF_DATA_PORT with Number()', () => {
    expect(resolvePerfDataPort({ LUXAR_PERF_DATA_PORT: '9100' })).toBe(9100);
    expect(resolvePerfDataPort({ LUXAR_PERF_DATA_PORT: '09000' })).toBe(9000);
  });
});

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

  it('normalizes the port exactly like the server-side derivation', () => {
    // The perf config boots the server on `resolvePerfDataPort()`; the
    // origin must embed that SAME normalized number. A leading-zero port
    // pins this: verbatim splicing would yield `:09000` — a URL naming a
    // port the server was never started on.
    expect(resolvePerfDataBase({ LUXAR_PERF_DATA_PORT: '09000' })).toBe('http://localhost:9000');
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
