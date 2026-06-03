/**
 * Direct unit tests for worker-pool/timeout/{with-timeout, combine-signals,
 * pick-timeout-ms}.ts (workers.md G6, G7) and worker-pool/lifecycle/
 * worker-count.ts (workers.md G8, H2).
 *
 * The thematic `timeout.test.ts` covers these helpers via the WorkerPool
 * wrapper (good), but the audit asked for direct coverage of the optional
 * callback branch + the `<= 0 || !isFinite` disable guard + the
 * combineSignals fallback. Direct tests run an order of magnitude faster
 * and pin the behaviour of these pure helpers without any class hoisting.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import fc from 'fast-check';
import { withTimeout } from '../../../../../workers/worker-pool/timeout/with-timeout';
import { combineSignals } from '../../../../../workers/worker-pool/timeout/combine-signals';
import { pickTimeoutMs } from '../../../../../workers/worker-pool/timeout/pick-timeout-ms';
import { getConfiguredWorkerCount } from '../../../../../workers/worker-pool/lifecycle/worker-count';
import { WorkerTimeoutError } from '../../../../../workers/worker-pool/errors';

// `withTimeout` calls `log.error` on timeout fire; stub it out to keep
// test output clean. We import the helper directly (no module-graph
// hoist needed because production `log` doesn't throw on missing
// methods, and these tests fire the timeout intentionally).
vi.mock('../../../../../utils/log', () => ({
  log: {
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    success: vi.fn(),
  },
  Modules: { WORKER_POOL: 'WorkerPool' },
}));

describe('withTimeout — pure helper (G6, M3)', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('resolves with the call result before timer fires', async () => {
    const call = Promise.resolve(42);
    expect(await withTimeout('test', call, 1000)).toBe(42);
  });

  it('rejects with WorkerTimeoutError when call exceeds timeoutMs', async () => {
    vi.useFakeTimers();
    const never = new Promise<number>(() => {});
    const raced = withTimeout('slow', never, 50);
    vi.advanceTimersByTime(60);
    await expect(raced).rejects.toBeInstanceOf(WorkerTimeoutError);
    await expect(raced).rejects.toMatchObject({ operation: 'slow', timeoutMs: 50 });
  });

  it('disables the race for timeoutMs <= 0 (returns the call directly) (M3)', async () => {
    // `<= 0` is the disable sentinel; the function must return `call`
    // without setting up a timer or wrapping in Promise.race.
    const call = Promise.resolve('ok');
    const result = withTimeout('disabled', call, 0);
    // Returns the same promise reference (pass-through).
    expect(result).toBe(call);
    expect(await result).toBe('ok');
  });

  it('disables the race for timeoutMs < 0', async () => {
    const call = Promise.resolve('neg');
    expect(withTimeout('neg', call, -100)).toBe(call);
  });

  it('disables the race for non-finite timeoutMs (Infinity, NaN) (M3)', async () => {
    const callInf = Promise.resolve(1);
    expect(withTimeout('inf', callInf, Infinity)).toBe(callInf);
    const callNaN = Promise.resolve(2);
    expect(withTimeout('nan', callNaN, NaN)).toBe(callNaN);
  });

  it('clears the timer when call resolves first (no late timer fire)', async () => {
    vi.useFakeTimers();
    let resolve!: (v: number) => void;
    const call = new Promise<number>((r) => {
      resolve = r;
    });
    const raced = withTimeout('quick', call, 1000);
    resolve(7);
    expect(await raced).toBe(7);

    // The cleared timer must not transition `raced` to rejected.
    let lateRejection = false;
    raced.catch(() => {
      lateRejection = true;
    });
    vi.advanceTimersByTime(2000);
    await Promise.resolve();
    expect(lateRejection).toBe(false);
  });

  it('does NOT invoke onTimeoutEvict when call resolves first', async () => {
    vi.useFakeTimers();
    const onEvict = vi.fn();
    const fakeWorker = { terminate: vi.fn() } as unknown as Worker;
    let resolve!: (v: number) => void;
    const call = new Promise<number>((r) => {
      resolve = r;
    });
    const raced = withTimeout('quick', call, 100, onEvict, fakeWorker);
    resolve(7);
    expect(await raced).toBe(7);
    vi.advanceTimersByTime(500);
    await Promise.resolve();
    expect(onEvict).not.toHaveBeenCalled();
  });

  it('invokes onTimeoutEvict(worker, reason) when timer fires (G6 — optional-callback branch, M3)', async () => {
    vi.useFakeTimers();
    const onEvict = vi.fn();
    const fakeWorker = { terminate: vi.fn() } as unknown as Worker;
    const never = new Promise<number>(() => {});
    const raced = withTimeout('slow', never, 50, onEvict, fakeWorker).catch((e: unknown) => e);
    vi.advanceTimersByTime(60);
    await raced;
    expect(onEvict).toHaveBeenCalledTimes(1);
    const [w, reason] = onEvict.mock.calls[0];
    expect(w).toBe(fakeWorker);
    expect(reason).toMatch(/timeout\(slow, 50ms\)/);
  });

  it('does NOT invoke onTimeoutEvict when callback is omitted but worker is supplied', async () => {
    // The implementation requires BOTH `worker` and `onTimeoutEvict`.
    // (G6 — optional callback path: undefined callback never fires.)
    vi.useFakeTimers();
    const fakeWorker = { terminate: vi.fn() } as unknown as Worker;
    const never = new Promise<number>(() => {});
    const raced = withTimeout('slow', never, 50, undefined, fakeWorker).catch((e: unknown) => e);
    vi.advanceTimersByTime(60);
    const err = await raced;
    expect(err).toBeInstanceOf(WorkerTimeoutError);
    // No callback present → fakeWorker.terminate untouched.
    expect(
      (fakeWorker as unknown as { terminate: ReturnType<typeof vi.fn> }).terminate
    ).not.toHaveBeenCalled();
  });

  it('does NOT invoke onTimeoutEvict when worker is omitted but callback is supplied', async () => {
    vi.useFakeTimers();
    const onEvict = vi.fn();
    const never = new Promise<number>(() => {});
    const raced = withTimeout('slow', never, 50, onEvict).catch((e: unknown) => e);
    vi.advanceTimersByTime(60);
    await raced;
    expect(onEvict).not.toHaveBeenCalled();
  });
});

describe('combineSignals — pure helper (G7)', () => {
  it('returns undefined when both signals are undefined', () => {
    expect(combineSignals(undefined, undefined)).toBeUndefined();
  });

  it('returns a directly when b is undefined (identity)', () => {
    const a = new AbortController().signal;
    expect(combineSignals(a, undefined)).toBe(a);
  });

  it('returns b directly when a is undefined (identity)', () => {
    const b = new AbortController().signal;
    expect(combineSignals(undefined, b)).toBe(b);
  });

  it('returns a combined signal that aborts when either source aborts (a first)', () => {
    const ca = new AbortController();
    const cb = new AbortController();
    const combined = combineSignals(ca.signal, cb.signal)!;
    expect(combined.aborted).toBe(false);
    ca.abort();
    expect(combined.aborted).toBe(true);
  });

  it('returns a combined signal that aborts when either source aborts (b first)', () => {
    const ca = new AbortController();
    const cb = new AbortController();
    const combined = combineSignals(ca.signal, cb.signal)!;
    cb.abort();
    expect(combined.aborted).toBe(true);
  });

  it('if either source is already aborted, the combined signal is aborted immediately', () => {
    const ca = new AbortController();
    const cb = new AbortController();
    ca.abort();
    const combined = combineSignals(ca.signal, cb.signal)!;
    expect(combined.aborted).toBe(true);
  });

  it('manual-fallback path: when AbortSignal.any is missing, still combines (G7 — fallback branch)', () => {
    // Temporarily delete AbortSignal.any so the fallback wiring runs.
    const orig = (AbortSignal as unknown as { any?: unknown }).any;
    try {
      (AbortSignal as unknown as { any?: unknown }).any = undefined;
      const ca = new AbortController();
      const cb = new AbortController();
      const combined = combineSignals(ca.signal, cb.signal)!;
      expect(combined.aborted).toBe(false);
      ca.abort();
      expect(combined.aborted).toBe(true);
    } finally {
      (AbortSignal as unknown as { any?: unknown }).any = orig;
    }
  });

  it('fallback path: pre-aborted a → combined.aborted is true at creation', () => {
    const orig = (AbortSignal as unknown as { any?: unknown }).any;
    try {
      (AbortSignal as unknown as { any?: unknown }).any = undefined;
      const ca = new AbortController();
      ca.abort();
      const cb = new AbortController();
      const combined = combineSignals(ca.signal, cb.signal)!;
      expect(combined.aborted).toBe(true);
    } finally {
      (AbortSignal as unknown as { any?: unknown }).any = orig;
    }
  });
});

describe('pickTimeoutMs — pure helper', () => {
  const perf = { workerVisibilityTimeoutMs: 30000, workerProjectionTimeoutMs: 60000 };

  it('visibility kind returns workerVisibilityTimeoutMs', () => {
    expect(pickTimeoutMs('visibility', perf)).toBe(30000);
  });

  it('projection kind returns workerProjectionTimeoutMs', () => {
    expect(pickTimeoutMs('projection', perf)).toBe(60000);
  });

  it('decode kind shares the projection knob (per the docstring contract)', () => {
    expect(pickTimeoutMs('decode', perf)).toBe(60000);
  });
});

describe('getConfiguredWorkerCount — pure helper (G8, H2)', () => {
  // Stash + restore navigator so we can flip hardwareConcurrency without
  // leaking between tests.
  const origNavigator = globalThis.navigator;

  beforeEach(() => {
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { hardwareConcurrency: 8 },
    });
  });

  afterEach(() => {
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: origNavigator,
    });
  });

  it('configCount=0 returns hardwareConcurrency - 1 (auto mode)', () => {
    expect(getConfiguredWorkerCount(0)).toBe(7);
  });

  it('configCount=1 returns 1 (within cap)', () => {
    expect(getConfiguredWorkerCount(1)).toBe(1);
  });

  it('configCount === cap returns cap (boundary)', () => {
    expect(getConfiguredWorkerCount(7)).toBe(7);
  });

  it('configCount above cap is capped at hardwareConcurrency - 1 (H2 — idempotent above cap)', () => {
    expect(getConfiguredWorkerCount(100)).toBe(7);
    // Above cap is idempotent: any value > cap returns cap.
    expect(getConfiguredWorkerCount(8)).toBe(7);
    expect(getConfiguredWorkerCount(99)).toBe(7);
  });

  it('configCount < 0 falls into auto mode (treats negative like 0)', () => {
    // Implementation uses `<= 0` so -1 takes the auto branch.
    expect(getConfiguredWorkerCount(-1)).toBe(7);
  });

  it('monotonic in configCount within [1, cap] (H2)', () => {
    // Up to the cap, output is exactly min(configCount, cap) — monotonic.
    const cap = 7;
    let prev = 0;
    for (let n = 1; n <= cap; n++) {
      const got = getConfiguredWorkerCount(n);
      expect(got).toBeGreaterThanOrEqual(prev);
      prev = got;
    }
  });

  it('hardwareConcurrency=1 still leaves at least 1 worker (Math.max(1, …) boundary)', () => {
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { hardwareConcurrency: 1 },
    });
    // With 1 core, cap = max(1, 1-1) = 1 → never returns 0.
    expect(getConfiguredWorkerCount(0)).toBe(1);
    expect(getConfiguredWorkerCount(99)).toBe(1);
  });

  it('hardwareConcurrency=0 (truthy fallback to 4) leaves cap=3', () => {
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { hardwareConcurrency: 0 },
    });
    // `navigator.hardwareConcurrency || 4` → 4; cap = 4 - 1 = 3.
    expect(getConfiguredWorkerCount(0)).toBe(3);
  });

  it('navigator undefined: falls back to 4 → cap=3', () => {
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: undefined,
    });
    expect(getConfiguredWorkerCount(0)).toBe(3);
  });

  // workers.md [H2][P12] fast-check property test: getConfiguredWorkerCount
  // is monotonic in configCount within [1, cap] and idempotent above the
  // cap (output never decreases as input grows past cap). For configCount=0
  // or <0 the auto branch fires (cap). All outputs are clamped to >=1.
  it('[property] result is always in [1, cap] regardless of configCount', () => {
    // Pin navigator to a known cap to make the property check deterministic.
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { hardwareConcurrency: 8 },
    });
    const cap = 7;
    fc.assert(
      fc.property(fc.integer({ min: -1000, max: 1000 }), (n) => {
        const result = getConfiguredWorkerCount(n);
        return result >= 1 && result <= cap;
      }),
      { numRuns: 200 }
    );
  });

  it('[property] idempotent above cap: result(n) === result(n+k) for n,k > cap', () => {
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { hardwareConcurrency: 8 },
    });
    const cap = 7;
    fc.assert(
      fc.property(
        fc.integer({ min: cap + 1, max: 100 }),
        fc.integer({ min: 0, max: 1000 }),
        (n, k) => {
          return getConfiguredWorkerCount(n) === getConfiguredWorkerCount(n + k);
        }
      ),
      { numRuns: 100 }
    );
  });
});
