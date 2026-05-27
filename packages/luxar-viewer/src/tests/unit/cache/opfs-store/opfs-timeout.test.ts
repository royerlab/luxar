import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { withTimeout } from '../../../../cache/multi-level-caching-store/opfs-store/opfs-timeout';

describe('withTimeout', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('resolves with the promise value when the promise wins', async () => {
    const promise = Promise.resolve('done');
    await expect(withTimeout(promise, 1000, 'op')).resolves.toBe('done');
  });

  it('rejects with the timeout error format when the timer wins', async () => {
    let resolveOuter: (v: string) => void = () => {};
    const stuck = new Promise<string>((resolve) => {
      resolveOuter = resolve;
    });
    const raced = withTimeout(stuck, 500, 'get(foo)');
    // Attach the catch synchronously so the rejection is never "unhandled".
    const caught = raced.catch((e) => e);
    await vi.advanceTimersByTimeAsync(600);
    const err = await caught;
    expect(err).toBeInstanceOf(Error);
    expect((err as Error).message).toBe('OPFS timeout: get(foo) exceeded 500ms');
    resolveOuter('late');
  });

  it('cleans up the timer when the promise resolves first', async () => {
    const before = vi.getTimerCount();
    await withTimeout(Promise.resolve('quick'), 10_000, 'op');
    // Allow the finally block to run.
    await Promise.resolve();
    expect(vi.getTimerCount()).toBe(before);
  });

  // workers.md O3 / cache.md G12 [P5]: audit-id moved to comment per Phase E54.
  it('cleans up the timer when the promise rejects (errors do not leak timers)', async () => {
    // [cache.md/G12][P5] The `finally { if (timer) clearTimeout(timer) }`
    // in the source is the only thing protecting against timer leaks under
    // repeated calls. Pin the rejection path: a thrown promise must still
    // clear the racing timer, otherwise long-running apps would accumulate
    // ghost timers across thousands of failed OPFS operations.
    const before = vi.getTimerCount();
    const failing = Promise.reject(new Error('oops'));
    await expect(withTimeout(failing, 10_000, 'op')).rejects.toThrow(/oops/);
    // Allow the finally block to run.
    await Promise.resolve();
    expect(vi.getTimerCount()).toBe(before);
  });

  // workers.md O3 / cache.md G12 [P5]: audit-id moved to comment per Phase E54.
  it('handles timeoutMs=0 as an immediate timeout boundary', async () => {
    // [cache.md/G12][P5] Pre-audit, no test exercised the timeoutMs=0
    // boundary. Source uses `setTimeout(fn, 0)` which always queues a
    // macrotask; a never-resolving promise must lose the race immediately
    // on the next tick.
    let _resolve: (v: string) => void = () => {};
    const pending = new Promise<string>((resolve) => {
      _resolve = resolve;
    });
    const raced = withTimeout(pending, 0, 'fast-timeout');
    const caught = raced.catch((e) => e);
    // Advance timers by 1ms — `setTimeout(0)` fires.
    await vi.advanceTimersByTimeAsync(1);
    const err = await caught;
    expect(err).toBeInstanceOf(Error);
    expect((err as Error).message).toContain('OPFS timeout: fast-timeout exceeded 0ms');
    // Defuse the dangling promise so the test doesn't leak rejections.
    _resolve('late');
  });
});
