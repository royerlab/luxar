/**
 * `raceTimeout` — the shared promise-versus-timer race (the worker pool's
 * `withTimeout` is a thin wrapper over it; see `pure-helpers.test.ts`).
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { raceTimeout } from '../../../utils/race-timeout';

afterEach(() => vi.useRealTimers());

describe('raceTimeout', () => {
  it('returns the promise itself, with no timer, for a disabled budget', () => {
    vi.useFakeTimers();
    for (const ms of [0, -5, Infinity, NaN]) {
      const promise = Promise.resolve(ms);
      expect(raceTimeout(promise, ms, () => new Error('never'))).toBe(promise);
    }
    expect(vi.getTimerCount()).toBe(0);
  });

  it('settles with the promise and clears its timer', async () => {
    vi.useFakeTimers();
    const onTimeout = vi.fn(() => new Error('late'));
    await expect(raceTimeout(Promise.resolve('ok'), 1_000, onTimeout)).resolves.toBe('ok');
    expect(vi.getTimerCount()).toBe(0);
    expect(onTimeout).not.toHaveBeenCalled();
  });

  it("rejects with onTimeout()'s value once the budget elapses first", async () => {
    vi.useFakeTimers();
    const reason = new Error('timed out');
    const onTimeout = vi.fn(() => reason);
    const raced = raceTimeout(new Promise<never>(() => {}), 50, onTimeout);
    const outcome = raced.catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(50);
    expect(await outcome).toBe(reason);
    expect(onTimeout).toHaveBeenCalledTimes(1);
  });
});
