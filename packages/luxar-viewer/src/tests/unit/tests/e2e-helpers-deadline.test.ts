/**
 * Deadline behaviour of the E2E evaluate wrappers (#1651).
 *
 * `page.evaluate` carries no timeout of its own, so an evaluate the page
 * never answers can only be stopped by the whole Playwright test budget —
 * the symptom the issue reported was a spec sitting for the two minutes ITS
 * run allowed inside a bare evaluate and being reported as `Tearing down
 * "page" exceeded the test timeout`. `evaluateWithDeadline` is the bound, and
 * `getConsoleMessages` (which the shared fixture runs in teardown for every
 * spec that imports `test` from `./fixtures`) is its first consumer, reached
 * through `assertNoConsoleErrors`.
 *
 * Four assertions here would go red against the unbounded code, because a
 * wedged page never settles at all: "resolves with onTimeout when the
 * deadline wins", "throws, naming the timeout, when the page never answers",
 * "does NOT resolve with empty buckets when the page never answers" (the one
 * that matters most — empty buckets would make the fixture's console-error
 * gate a vacuous pass for every spec that uses it), and the default-deadline
 * test.
 * The rest guard adjacent contracts rather than the bound itself: the
 * pass-through and rejection paths, the `clearTimeout` on the evaluation
 * path, and the in-page function's promise never to return `null` (which is
 * what makes `null` usable as the deadline sentinel).
 *
 * No browser here: `page` is a one-method fake cast to Playwright's `Page`.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import type { Page } from '@playwright/test';
import { assertNoConsoleErrors, evaluateWithDeadline, getConsoleMessages } from '../../e2e/helpers';

/** A `page` whose `evaluate` resolves with `value`. */
function answeringPage(value: unknown): Page {
  return { evaluate: async () => value } as unknown as Page;
}

/** A `page` whose `evaluate` never settles — the wedge being reproduced. */
function wedgedPage(): Page {
  return { evaluate: () => new Promise(() => {}) } as unknown as Page;
}

afterEach(() => {
  vi.useRealTimers();
});

describe('evaluateWithDeadline', () => {
  it('resolves with the evaluation when it settles before the deadline', async () => {
    vi.useFakeTimers();
    const result = await evaluateWithDeadline(Promise.resolve('answer'), 10000, 'sentinel');
    expect(result).toBe('answer');
    // The deadline timer must not outlive the race: a dangling handle keeps
    // the Node process alive past the end of the run.
    expect(vi.getTimerCount()).toBe(0);
  });

  it('resolves with onTimeout when the deadline wins', async () => {
    vi.useFakeTimers();
    const settled = evaluateWithDeadline(new Promise<string>(() => {}), 10000, 'sentinel');
    await vi.advanceTimersByTimeAsync(10000);
    expect(await settled).toBe('sentinel');
    // Consistency check only, NOT a clearTimeout guard: the one-shot timer has
    // already fired on this path, so it would read 0 either way. The assertion
    // on the evaluation path above is the one that catches a missing clear.
    expect(vi.getTimerCount()).toBe(0);
  });

  it('propagates a rejection that arrives before the deadline', async () => {
    vi.useFakeTimers();
    await expect(
      evaluateWithDeadline(Promise.reject(new Error('boom')), 10000, 'sentinel')
    ).rejects.toThrow('boom');
    expect(vi.getTimerCount()).toBe(0);
  });

  it('does not emit an unhandledRejection when the evaluation rejects after the deadline', async () => {
    // Real timers on purpose: unhandledRejection is decided by Node's own
    // macrotask bookkeeping, which fake timers do not drive.
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);

    let rejectLate: (reason: Error) => void = () => {};
    const late = new Promise<string>((_resolve, reject) => {
      rejectLate = reject;
    });

    try {
      expect(await evaluateWithDeadline(late, 5, 'sentinel')).toBe('sentinel');
      // The race is already settled; Promise.race has nonetheless attached a
      // handler to `late`, so this rejection stays handled.
      rejectLate(new Error('late boom'));
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(unhandled).toEqual([]);
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }
  });
});

describe('getConsoleMessages deadline', () => {
  it('returns the buckets when the page answers', async () => {
    const buckets = {
      errors: ['[❌] boom'],
      warnings: [],
      logs: ['[ℹ️] hello'],
      all: ['[❌] boom', '[ℹ️] hello'],
    };
    await expect(getConsoleMessages(answeringPage(buckets), 1000)).resolves.toEqual(buckets);
  });

  it('passes an empty answer through rather than treating it as a wedge', async () => {
    // An empty ANSWER is legitimate — only a missing answer is a wedge. This
    // fake ignores the injected function; the no-interceptor branch that
    // legitimately produces empty buckets is covered further down, where the
    // in-page function is actually executed.
    const empty = { errors: [], warnings: [], logs: [], all: [] };
    await expect(getConsoleMessages(answeringPage(empty), 1000)).resolves.toEqual(empty);
  });

  it('throws, naming the timeout, when the page never answers', async () => {
    const error = await getConsoleMessages(wedgedPage(), 25).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toMatch(/never answered/i);
    expect((error as Error).message).toContain('25 ms');
    expect((error as Error).message).toContain('#1651');
  });

  it('does NOT resolve with empty buckets when the page never answers', async () => {
    // The vacuous-pass hazard: the shared fixture feeds this straight into
    // assertNoConsoleErrors, so resolving with `{errors: [], ...}` on a
    // wedged page would silence the console-error gate for every spec.
    let resolvedWith: unknown = 'never-resolved';
    await getConsoleMessages(wedgedPage(), 25).then(
      (value) => {
        resolvedWith = value;
      },
      () => {
        /* rejection is the expected path, asserted above */
      }
    );
    expect(resolvedWith).toBe('never-resolved');
  });

  it('propagates the deadline failure out through assertNoConsoleErrors', async () => {
    // The premise of the whole 45 s design is that the FIXTURE's gate fails
    // with this message. `assertNoConsoleErrors` is the consumer the fixture
    // teardown calls, so the throw has to survive the trip through it — an
    // `await ... .catch(() => ({errors: []}))` anywhere in that chain would
    // turn the gate into the vacuous pass this rejection exists to prevent.
    vi.useFakeTimers();
    const settled = assertNoConsoleErrors(wedgedPage(), []).catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(45000);
    const error = await settled;
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toMatch(/never answered/i);
    expect((error as Error).message).toContain('#1651');
  });

  it('defaults to a 45 s deadline — inside Playwright’s fresh After-Hooks slot', async () => {
    // The teardown call site gets a FRESH timeout slot equal to the per-test
    // timeout (60 s here), so the default has to stay under that to be
    // attributed to this probe rather than to `Tearing down "page"`.
    vi.useFakeTimers();
    const settled = getConsoleMessages(wedgedPage()).then(
      () => 'resolved',
      (e: Error) => e
    );
    await vi.advanceTimersByTimeAsync(44999);
    expect(await Promise.race([settled, Promise.resolve('pending')])).toBe('pending');
    await vi.advanceTimersByTimeAsync(1);
    const outcome = await settled;
    expect(outcome).toBeInstanceOf(Error);
    expect((outcome as Error).message).toContain('45000 ms');
  });
});

describe('getConsoleMessages in-page function', () => {
  // The `null` sentinel is only sound because the injected function ALWAYS
  // returns an object. The fakes above never run it (they ignore the
  // callback), so these tests execute it for real against a stubbed `window` —
  // the vitest default environment is `node`, so there is none otherwise.
  const originalWindow = (globalThis as { window?: unknown }).window;

  /** A `page` whose `evaluate` actually CALLS the injected function. */
  function executingPage(): Page {
    return {
      evaluate: async (fn: () => unknown) => fn(),
    } as unknown as Page;
  }

  function stubWindow(debug: unknown): void {
    (globalThis as { window?: unknown }).window =
      debug === undefined ? {} : { __luxarDebug: debug };
  }

  afterEach(() => {
    if (originalWindow === undefined) {
      delete (globalThis as { window?: unknown }).window;
    } else {
      (globalThis as { window?: unknown }).window = originalWindow;
    }
  });

  it('returns empty buckets — an object, never null — with no debug interface', async () => {
    stubWindow(undefined);
    const buckets = await getConsoleMessages(executingPage(), 1000);
    expect(buckets).not.toBeNull();
    expect(buckets).toEqual({ errors: [], warnings: [], logs: [], all: [] });
  });

  it('sorts buffered messages into buckets by type, and never returns null', async () => {
    stubWindow({
      consoleInterceptor: {
        getBufferedMessages: () => [
          { type: 'error', message: 'boom' },
          { type: 'warn', message: 'careful' },
          { type: 'info', message: 'hello' },
        ],
      },
    });

    const buckets = await getConsoleMessages(executingPage(), 1000);
    expect(buckets).not.toBeNull();
    expect(buckets.errors).toHaveLength(1);
    expect(buckets.errors[0]).toContain('boom');
    expect(buckets.warnings).toHaveLength(1);
    expect(buckets.warnings[0]).toContain('careful');
    expect(buckets.logs).toHaveLength(1);
    expect(buckets.logs[0]).toContain('hello');
    expect(buckets.all).toHaveLength(3);
  });
});
