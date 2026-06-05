/**
 * Unit tests for OnceInit — the one-shot initializer with retry semantics
 * shared by the spatial-index loaders.
 */

import { describe, it, expect, vi } from 'vitest';
import { OnceInit } from '../../../../data/loaders';

describe('OnceInit', () => {
  it('reports isInitialized=false until ensure() is called', () => {
    const o = new OnceInit();
    expect(o.isInitialized).toBe(false);
  });

  it('runs the init function exactly once across multiple ensure() calls', async () => {
    const o = new OnceInit();
    const initFn = vi.fn().mockResolvedValue(undefined);

    await o.ensure(initFn);
    await o.ensure(initFn);
    await o.ensure(initFn);

    expect(initFn).toHaveBeenCalledTimes(1);
    expect(o.isInitialized).toBe(true);

    // Once initialized, a LATER ensure() with a DIFFERENT fn is a no-op:
    // the cached (resolved) promise is reused, so the new fn never runs and
    // the original fn is not called again.
    const differentFn = vi.fn().mockResolvedValue(undefined);
    await o.ensure(differentFn);
    expect(initFn).toHaveBeenCalledTimes(1);
    expect(differentFn).not.toHaveBeenCalled();
  });

  it('shares the in-flight promise across concurrent callers', async () => {
    const o = new OnceInit();
    let resolveInner!: () => void;
    const initFn = vi.fn().mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          resolveInner = resolve;
        })
    );

    // Three concurrent callers — they should all wait on the SAME promise.
    const p1 = o.ensure(initFn);
    const p2 = o.ensure(initFn);
    const p3 = o.ensure(initFn);

    expect(initFn).toHaveBeenCalledTimes(1);
    resolveInner();
    await Promise.all([p1, p2, p3]);
    expect(initFn).toHaveBeenCalledTimes(1);
  });

  it('clears the cached promise on rejection so the next call retries', async () => {
    const o = new OnceInit();
    let attempt = 0;
    const initFn = vi.fn().mockImplementation(async () => {
      attempt += 1;
      if (attempt === 1) throw new Error('first-fail');
    });

    await expect(o.ensure(initFn)).rejects.toThrow('first-fail');
    expect(initFn).toHaveBeenCalledTimes(1);

    // Second call: initFn runs again because the previous attempt failed.
    await o.ensure(initFn);
    expect(initFn).toHaveBeenCalledTimes(2);
    expect(o.isInitialized).toBe(true);
  });

  it('propagates the rejection to ALL concurrent waiters', async () => {
    const o = new OnceInit();
    const initFn = vi.fn().mockRejectedValue(new Error('boom'));

    const p1 = o.ensure(initFn);
    const p2 = o.ensure(initFn);
    const p3 = o.ensure(initFn);

    await expect(p1).rejects.toThrow('boom');
    await expect(p2).rejects.toThrow('boom');
    await expect(p3).rejects.toThrow('boom');
    expect(initFn).toHaveBeenCalledTimes(1);
  });

  it('reset() forces the next ensure() to run init again', async () => {
    const o = new OnceInit();
    const initFn = vi.fn().mockResolvedValue(undefined);

    await o.ensure(initFn);
    expect(initFn).toHaveBeenCalledTimes(1);

    o.reset();
    expect(o.isInitialized).toBe(false);

    await o.ensure(initFn);
    expect(initFn).toHaveBeenCalledTimes(2);
  });
});
