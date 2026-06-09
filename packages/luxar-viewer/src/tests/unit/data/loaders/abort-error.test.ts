/**
 * Unit tests for the shared abort-error classifier.
 *
 * `isAbortError` gates two behaviors: superseded loads must NOT be recorded as
 * failures (run-loader-updates) NOR inflate loader error telemetry
 * (metrics.errors / monitor 'error' events). Both error shapes a superseded
 * load throws must be recognized; genuine errors must not.
 */

import { describe, it, expect } from 'vitest';
import { isAbortError } from '../../../../data/loaders';

describe('isAbortError', () => {
  it("recognizes zarrita's DOMException AbortError (via throwIfAborted)", () => {
    const ac = new AbortController();
    ac.abort();
    let thrown: unknown;
    try {
      ac.signal.throwIfAborted();
    } catch (e) {
      thrown = e;
    }
    expect(isAbortError(thrown)).toBe(true);
  });

  it("recognizes an Error named 'AbortError'", () => {
    const e = new Error('aborted');
    e.name = 'AbortError';
    expect(isAbortError(e)).toBe(true);
  });

  it("recognizes the worker pool's WorkerAbortError by name", () => {
    const e = new Error('worker aborted');
    e.name = 'WorkerAbortError';
    expect(isAbortError(e)).toBe(true);
  });

  it('rejects a genuine (non-abort) Error', () => {
    expect(isAbortError(new Error('network failed'))).toBe(false);
    const typeErr = new TypeError('boom');
    expect(isAbortError(typeErr)).toBe(false);
  });

  it('rejects non-Error values', () => {
    expect(isAbortError('AbortError')).toBe(false);
    expect(isAbortError({ name: 'AbortError' })).toBe(false);
    expect(isAbortError(null)).toBe(false);
    expect(isAbortError(undefined)).toBe(false);
  });
});
