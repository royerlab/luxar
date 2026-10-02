/**
 * `combineAbortSignals` — the one abort-signal merge the fetch retry loop, the
 * caching store, the zip reader, the worker pool and the mesh loader share.
 * These cases were the suites of its three predecessors (`mergeAbortSignals`
 * in fetch-retry, `combineSignals` in the worker pool, the zip reader's
 * `mergeProbeSignals`), merged; the fallback now also carries the abort reason
 * the worker-pool copy dropped.
 */

import { getEventListeners } from 'node:events';
import { describe, expect, it, vi } from 'vitest';
import { combineAbortSignals } from '../../../utils/abort-signals';

/** Hide `AbortSignal.any` so the manual relay runs; returns the restore. */
function forceFallback(): () => void {
  const descriptor = Object.getOwnPropertyDescriptor(AbortSignal, 'any');
  Object.defineProperty(AbortSignal, 'any', { configurable: true, value: undefined });
  return () => {
    if (descriptor) Object.defineProperty(AbortSignal, 'any', descriptor);
    else delete (AbortSignal as unknown as { any?: unknown }).any;
  };
}

function withFallback(body: () => void): void {
  const restore = forceFallback();
  try {
    body();
  } finally {
    restore();
  }
}

describe('combineAbortSignals — one or no input', () => {
  it('returns undefined when both signals are undefined', () => {
    expect(combineAbortSignals(undefined, undefined)).toBeUndefined();
  });

  it('returns the only signal itself in a no-op scope (either position)', () => {
    const a = new AbortController().signal;
    const b = new AbortController().signal;
    const first = combineAbortSignals(a);
    const second = combineAbortSignals(undefined, b)!;
    expect(first.signal).toBe(a);
    expect(second.signal).toBe(b);
    expect(() => first.dispose()).not.toThrow();
    expect(() => second.dispose()).not.toThrow();
  });
});

describe('combineAbortSignals — native AbortSignal.any', () => {
  it('delegates to AbortSignal.any when present', () => {
    const descriptor = Object.getOwnPropertyDescriptor(AbortSignal, 'any');
    const nativeSignal = new AbortController().signal;
    const any = vi.fn(() => nativeSignal);
    Object.defineProperty(AbortSignal, 'any', { configurable: true, value: any });
    try {
      const a = new AbortController().signal;
      const b = new AbortController().signal;
      const merged = combineAbortSignals(a, b);
      expect(any).toHaveBeenCalledWith([a, b]);
      expect(merged.signal).toBe(nativeSignal);
      expect(() => merged.dispose()).not.toThrow();
    } finally {
      if (descriptor) Object.defineProperty(AbortSignal, 'any', descriptor);
      else delete (AbortSignal as unknown as { any?: unknown }).any;
    }
  });

  it('aborts when either source aborts, with that source’s reason', () => {
    for (const which of [0, 1] as const) {
      const sources = [new AbortController(), new AbortController()];
      const merged = combineAbortSignals(sources[0].signal, sources[1].signal);
      const reason = new Error(`source ${which}`);
      sources[which].abort(reason);
      expect(merged.signal.aborted).toBe(true);
      expect(merged.signal.reason).toBe(reason);
    }
  });
});

describe('combineAbortSignals — fallback relay', () => {
  it('relays an abort from either source, with its reason, and removes both listeners', () => {
    withFallback(() => {
      for (const which of [0, 1] as const) {
        const sources = [new AbortController(), new AbortController()];
        const merged = combineAbortSignals(sources[0].signal, sources[1].signal);
        expect(getEventListeners(sources[0].signal, 'abort')).toHaveLength(1);
        expect(getEventListeners(sources[1].signal, 'abort')).toHaveLength(1);

        const reason = new Error(`source ${which}`);
        sources[which].abort(reason);

        expect(merged.signal.aborted).toBe(true);
        expect(merged.signal.reason).toBe(reason);
        expect(getEventListeners(sources[0].signal, 'abort')).toHaveLength(0);
        expect(getEventListeners(sources[1].signal, 'abort')).toHaveLength(0);
      }
    });
  });

  it('returns an already-aborted scope, with the reason, registering no listeners', () => {
    withFallback(() => {
      const a = new AbortController();
      const reason = new Error('already timed out');
      a.abort(reason);
      const b = new AbortController();
      const merged = combineAbortSignals(a.signal, b.signal);
      expect(merged.signal.aborted).toBe(true);
      expect(merged.signal.reason).toBe(reason);
      expect(getEventListeners(a.signal, 'abort')).toHaveLength(0);
      expect(getEventListeners(b.signal, 'abort')).toHaveLength(0);
    });
  });

  it('dispose is idempotent, so settled merges never accumulate on a long-lived source', () => {
    withFallback(() => {
      // The pool-wide or store-lifetime signal lives for a whole dataset
      // session; every settled call must release its relay listener from it.
      const session = new AbortController();
      for (let i = 0; i < 100; i++) {
        const caller = new AbortController();
        const merged = combineAbortSignals(session.signal, caller.signal);
        expect(getEventListeners(session.signal, 'abort')).toHaveLength(1);
        merged.dispose();
        merged.dispose();
        expect(getEventListeners(caller.signal, 'abort')).toHaveLength(0);
      }
      expect(getEventListeners(session.signal, 'abort')).toHaveLength(0);
    });
  });
});
