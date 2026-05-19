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
});
