/**
 * Contract tests for `raceEvaluate`, the deadline wrapper the e2e wait
 * helpers put around `page.evaluate`.
 *
 * `page.evaluate` carries no timeout of its own, so on a frame-starved page an
 * unbounded one outlives the whole test budget instead of letting the helper
 * fall back. `raceEvaluate` is the fix, and it is pure promise plumbing: no
 * browser, no Playwright, nothing that needs a real page. That makes it unit-
 * testable here, the same way `e2e-helpers-shape.test.ts` unit-tests the other
 * JS-side shapes of `src/tests/e2e/helpers.ts`.
 *
 * NO `@vitest-environment` docblock on purpose: #1634 made `node` the default
 * and only DOM-touching files opt in. Nothing below constructs a document —
 * the subject is promises and timers — so a jsdom opt-in would buy a ~1.8 s
 * environment for nothing.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { raceEvaluate } from '../../e2e/helpers';

afterEach(() => {
  vi.useRealTimers();
});

describe('raceEvaluate', () => {
  it('resolves with the evaluation when it settles before the deadline', async () => {
    vi.useFakeTimers();

    const result = await raceEvaluate(Promise.resolve(42), 3000, -1);

    expect(result).toBe(42);
  });

  it('resolves with onTimeout when the deadline wins', async () => {
    vi.useFakeTimers();

    // An evaluate that never answers — exactly the starved-page case.
    const neverAnswers = new Promise<number | null>(() => {});
    const raced = raceEvaluate(neverAnswers, 3000, null);

    // Nothing yet: the deadline has not elapsed.
    let settled = false;
    void raced.then(() => {
      settled = true;
    });
    await vi.advanceTimersByTimeAsync(2999);
    expect(settled).toBe(false);

    await vi.advanceTimersByTimeAsync(1);
    await expect(raced).resolves.toBeNull();
  });

  it('clears its timer on the evaluation path (no dangling handle)', async () => {
    vi.useFakeTimers();

    await raceEvaluate(Promise.resolve('answered'), 3000, 'gave-up');

    expect(vi.getTimerCount()).toBe(0);
  });

  it('clears its timer on the deadline path (no dangling handle)', async () => {
    vi.useFakeTimers();

    const raced = raceEvaluate(new Promise<string>(() => {}), 3000, 'gave-up');
    expect(vi.getTimerCount()).toBe(1);

    await vi.advanceTimersByTimeAsync(3000);
    await expect(raced).resolves.toBe('gave-up');
    expect(vi.getTimerCount()).toBe(0);
  });

  it('clears its timer when the evaluation rejects', async () => {
    vi.useFakeTimers();

    await expect(raceEvaluate(Promise.reject(new Error('boom')), 3000, 'gave-up')).rejects.toThrow(
      'boom'
    );

    expect(vi.getTimerCount()).toBe(0);
  });

  it('propagates a rejection that arrives before the deadline', async () => {
    vi.useFakeTimers();

    let rejectEvaluation!: (reason: unknown) => void;
    const evaluation = new Promise<string>((_resolve, reject) => {
      rejectEvaluation = reject;
    });
    const raced = raceEvaluate(evaluation, 3000, 'gave-up');

    await vi.advanceTimersByTimeAsync(1000);
    rejectEvaluation(new Error('Execution context was destroyed'));

    await expect(raced).rejects.toThrow('Execution context was destroyed');
  });

  it('does not surface an unhandled rejection when the evaluation fails AFTER the deadline', async () => {
    vi.useFakeTimers();

    let rejectEvaluation!: (reason: unknown) => void;
    const evaluation = new Promise<string>((_resolve, reject) => {
      rejectEvaluation = reject;
    });
    const raced = raceEvaluate(evaluation, 3000, 'gave-up');

    await vi.advanceTimersByTimeAsync(3000);
    await expect(raced).resolves.toBe('gave-up');

    // The late rejection is what a real timed-out evaluate does when the page
    // finally tears down its execution context. `Promise.race` has already
    // attached a handler, so it must stay handled even though the race
    // settled long ago; a wrapper that merely dropped the promise would crash
    // the runner instead.
    vi.useRealTimers();
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);
    try {
      rejectEvaluation(new Error('Execution context was destroyed'));
      // Two macrotask turns: Node reports an unhandled rejection at the end
      // of the turn after the one that created it.
      await new Promise((resolve) => setTimeout(resolve, 10));
      await new Promise((resolve) => setTimeout(resolve, 10));
      expect(unhandled).toEqual([]);
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }
  });
});
