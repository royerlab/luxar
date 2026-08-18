/**
 * Deadline behaviour of the E2E evaluate wrappers (#1651).
 *
 * `page.evaluate` carries no timeout of its own, so an evaluate the page
 * never answers can only be stopped by the whole Playwright test budget —
 * the symptom the issue reported was a spec sitting for the two minutes ITS
 * run allowed inside a bare evaluate and being reported as `Tearing down
 * "page" exceeded the test timeout`. `raceEvaluate` is the bound (its own
 * promise/timer contract is covered in `e2e-helpers-race-evaluate.test.ts`,
 * so this file does not repeat it), and the two consumers covered here are
 * `getConsoleMessages` — which the shared fixture runs in teardown for every
 * spec that imports `test` from `./fixtures`, reached through
 * `assertNoConsoleErrors` — and `getLuxarState`, the suite's most-called probe.
 *
 * Four assertions here would go red against the unbounded code, because a
 * wedged page never settles at all: "throws, naming the timeout, when the
 * page never answers", "does NOT resolve with empty buckets when the page
 * never answers" (the one that matters most — empty buckets would make the
 * fixture's console-error gate a vacuous pass for every spec that uses it),
 * "propagates the deadline failure out through assertNoConsoleErrors" (the
 * premise of the whole design — that is the call the fixture teardown makes),
 * and the default-deadline test.
 * The rest guard adjacent contracts rather than the bound itself: the
 * pass-through path, and the in-page function's promise never to return
 * `null` (which is what makes `null` usable as the deadline sentinel).
 *
 * `getLuxarState` (#1651 + #1724) adds the same three bound assertions — it
 * throws naming the timeout, it never resolves with a stand-in, and it defaults
 * to 45 s — all three red against the unbounded code. Its load-bearing extra is
 * the SENTINEL test: `getLuxarState` returns whatever the viewer's untyped
 * `getState()` hands back, so a falsy answer (`null`, `undefined`) has to pass
 * through unchanged instead of being misreported as a stall. That one is red
 * against a primitive sentinel rather than against the unbounded code, which is
 * the point — it pins the one design decision the bound could get wrong while
 * still looking correct. The in-page test merely guards an adjacent contract:
 * a genuinely missing debug interface must still fail as itself.
 *
 * No browser here: `page` is a one-method fake cast to Playwright's `Page`.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import type { Page } from '@playwright/test';
import {
  assertNoConsoleErrors,
  getConsoleMessages,
  getLuxarState,
  waitForDimensionNavigation,
  waitForPointsLoaded,
} from '../../e2e/helpers';

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

describe('getLuxarState deadline', () => {
  it('returns the state when the page answers', async () => {
    const state = { initialized: true, totalPoints: 100000, isLoading: false };
    await expect(getLuxarState(answeringPage(state), 1000)).resolves.toEqual(state);
  });

  it('throws, naming the timeout and both issues, when the page never answers', async () => {
    const error = await getLuxarState(wedgedPage(), 25).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toMatch(/never answered/i);
    expect((error as Error).message).toContain('25 ms');
    expect((error as Error).message).toContain('#1651');
    // #1724 is the viewer-side main-thread saturation this probe is the victim
    // of, not the cause of; naming it is how the diagnostic points at the real
    // bug instead of blaming the E2E helper.
    expect((error as Error).message).toContain('#1724');
  });

  it('does NOT resolve with any fallback state when the page never answers', async () => {
    // The vacuous-pass hazard, in the shape this file already uses: well over a
    // hundred call sites assert on the result
    // (`expect(state.totalPoints).toBeGreaterThan(0)`)
    // and the `state && …` / `!state.isLoading` polling helpers would read a
    // stand-in as a pass, so a missed deadline must hand back NOTHING.
    let resolvedWith: unknown = 'never-resolved';
    await getLuxarState(wedgedPage(), 25).then(
      (value) => {
        resolvedWith = value;
      },
      () => {
        /* rejection is the expected path, asserted above */
      }
    );
    expect(resolvedWith).toBe('never-resolved');
  });

  it('defaults to a 45 s deadline', async () => {
    vi.useFakeTimers();
    const settled = getLuxarState(wedgedPage()).then(
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

  it('passes a falsy answer through rather than reporting it as a stall', async () => {
    // Why the sentinel is a fresh Node-side object (`const timedOut = {}`) and
    // not a primitive: unlike `getConsoleMessages` — whose in-page function has
    // two exits, both object literals it wrote itself — this probe returns
    // whatever `debug.getState()` hands back, an untyped `any` the viewer is
    // free to reshape. A `null` sentinel would turn a legitimate `null` answer
    // into a bogus "main thread saturated" failure, and silently so: the run
    // would blame #1724 for a viewer that answered instantly. Identity against
    // an object allocated in Node cannot collide, because `page.evaluate`
    // resolves with a value deserialized from CDP and therefore freshly
    // constructed on this side — even a page answering with a literal `{}`.
    await expect(getLuxarState(answeringPage(null), 1000)).resolves.toBeNull();
    await expect(getLuxarState(answeringPage(undefined), 1000)).resolves.toBeUndefined();
    await expect(getLuxarState(answeringPage({}), 1000)).resolves.toEqual({});
  });
});

describe('getLuxarState in-page function', () => {
  // The fakes above never run the injected function (they ignore the callback),
  // so this executes it for real against a stubbed `window` — the vitest default
  // environment is `node`, so there is none otherwise.
  const originalWindow = (globalThis as { window?: unknown }).window;

  /** A `page` whose `evaluate` actually CALLS the injected function. */
  function executingPage(): Page {
    return {
      evaluate: async (fn: () => unknown) => fn(),
    } as unknown as Page;
  }

  afterEach(() => {
    if (originalWindow === undefined) {
      delete (globalThis as { window?: unknown }).window;
    } else {
      (globalThis as { window?: unknown }).window = originalWindow;
    }
  });

  it('surfaces a missing debug interface as itself, not as a deadline stall', async () => {
    // A genuinely absent `__luxarDebug` is the real failure it always was: the
    // evaluate settles before the deadline, so `Promise.race` settles the same
    // way and the in-page rejection propagates unchanged. Reporting it as a
    // stall would send every "viewer never booted" run chasing #1724.
    (globalThis as { window?: unknown }).window = {};
    const error = await getLuxarState(executingPage(), 1000).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toContain('Debug interface not ready');
    expect((error as Error).message).not.toMatch(/never answered/i);
  });

  it('returns the live getState() result when the debug interface is present', async () => {
    const state = { initialized: true, totalPoints: 7 };
    (globalThis as { window?: unknown }).window = {
      __luxarDebug: { getState: () => state },
    };
    await expect(getLuxarState(executingPage(), 1000)).resolves.toEqual(state);
  });
});

describe('poll loops attribute a starved probe', () => {
  // The poll loops catch a failing probe and retry, because during
  // initialization a missing debug interface is legitimate. That same catch
  // swallows the starvation diagnostic, so a #1724 wedge used to be reported as
  // `Timeout waiting for points to load` — blaming missing DATA for a saturated
  // main thread, at every call site that goes through these two helpers. Both
  // assertions below are red without the carried-forward probe error.

  /** A `page` that never answers an evaluate, with a real sleep for the loop. */
  function wedgedPollPage(): Page {
    return {
      evaluate: () => new Promise(() => {}),
      waitForTimeout: (ms: number) => new Promise((resolve) => setTimeout(resolve, ms)),
    } as unknown as Page;
  }

  /** Drive a loop whose first probe eats the whole 45 s default deadline. */
  async function runWedgedLoop(start: () => Promise<void>): Promise<Error> {
    vi.useFakeTimers();
    const settled = start().then(
      () => new Error('resolved, but the loop should have timed out'),
      (e: Error) => e
    );
    // 45 s for the probe deadline, plus the loop's own sleep before it re-checks.
    await vi.advanceTimersByTimeAsync(46000);
    return settled;
  }

  it('names the starved probe in waitForPointsLoaded’s timeout', async () => {
    const error = await runWedgedLoop(() => waitForPointsLoaded(wedgedPollPage(), 10, 1000));
    expect(error.message).toContain('Timeout waiting for points to load');
    expect(error.message).toContain('Last state probe failed');
    expect(error.message).toMatch(/never answered/i);
    expect(error.message).toContain('#1724');
  });

  it('names the starved probe in waitForDimensionNavigation’s timeout', async () => {
    const error = await runWedgedLoop(() => waitForDimensionNavigation(wedgedPollPage(), 0, 1000));
    expect(error.message).toContain('Dimension navigation did not complete within 1000ms');
    expect(error.message).toContain('Last state probe failed');
    expect(error.message).toContain('#1724');
  });

  it('leaves the message alone when no probe ever failed', async () => {
    // A loop that simply never saw its condition satisfied must report exactly
    // what it always did — no dangling "Last state probe failed" suffix.
    const answering = {
      evaluate: async () => ({ totalPoints: 0, isLoading: true }),
      waitForTimeout: (ms: number) => new Promise((resolve) => setTimeout(resolve, ms)),
    } as unknown as Page;
    vi.useFakeTimers();
    const settled = waitForPointsLoaded(answering, 10, 1000).then(
      () => new Error('resolved, but the loop should have timed out'),
      (e: Error) => e
    );
    await vi.advanceTimersByTimeAsync(2000);
    const error = await settled;
    expect(error.message).toBe('Timeout waiting for points to load (expected at least 10)');
  });
});
