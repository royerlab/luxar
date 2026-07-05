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

  it('rejects primitives and null-ish values', () => {
    expect(isAbortError('AbortError')).toBe(false);
    expect(isAbortError(null)).toBe(false);
    expect(isAbortError(undefined)).toBe(false);
    expect(isAbortError(42)).toBe(false);
  });

  it("accepts any OBJECT named 'AbortError' — the realm-proof contract", () => {
    // Deliberate contract change: classification is by `.name` alone, NOT
    // `instanceof Error`. A DOMException created in another realm (jsdom
    // test env, iframe) fails a same-realm instanceof check even though its
    // own chain contains that realm's Error — the old check silently turned
    // intentional aborts into recorded failures + error telemetry.
    expect(isAbortError({ name: 'AbortError' })).toBe(true);
    expect(isAbortError({ name: 'WorkerAbortError' })).toBe(true);
    expect(isAbortError({ name: 'SomethingElse' })).toBe(false);
  });

  it('recognizes a cross-realm DOMException (jsdom global — NOT instanceof this Error)', () => {
    // In the vitest jsdom environment, the global DOMException comes from
    // the jsdom realm: its prototype chain contains jsdom's Error, so
    // `instanceof Error` (Node realm) is FALSE here. The pre-fix classifier
    // returned false for exactly this value, which made the refinement
    // backoff count aborts as failures in every jsdom-based test.
    const crossRealm = new DOMException('aborted', 'AbortError');
    expect(crossRealm instanceof Error).toBe(false); // precondition of this env
    expect(isAbortError(crossRealm)).toBe(true);
  });
});
