/**
 * End-of-budget behaviour of the E2E wait helpers that give up SILENTLY
 * (#1726): `waitForSpatialQuery`, `waitForNavigationComplete` and
 * `waitForWebGLError`.
 *
 * All three used to end their poll loop by simply returning, on the reasoning
 * that the condition might have settled before polling started. That is a real
 * case — several `waitForNavigationComplete` call sites deliberately pick the
 * silent variant because their action never toggles `isLoading` at all — but it
 * is indistinguishable from a page whose main thread is saturated and answers no
 * probe at all (#1724): the helper returns having read nothing, and whatever the
 * spec asserts next reads unconfirmed state. That is the vacuous pass, not a
 * flake.
 *
 * The assertions that are RED without the fix are the wedged-page ones. Against
 * the old code the wedged cases do not fail — they either hang inside a single
 * unbounded `page.evaluate` (which no fake timer can end, since the loop's own
 * `Date.now()` check is never reached again) or, once bounded, return normally.
 * So "rejects, naming the helper" and its companion "does NOT resolve" guard are
 * the ones that encode the new contract, along with the one-warning assertions
 * for the tolerated answered-but-unsettled case. The healthy-page tests are the
 * counterweight: they go red if the fix makes the normal path loud or throwing,
 * which is what would break the nine existing call sites.
 *
 * No browser: every `page` below is a hand-built two- or three-method fake cast
 * to Playwright's `Page`. `waitForTimeout` is a real `setTimeout` promise so
 * vitest's fake timers (which also drive `Date.now()`) can advance a whole poll
 * loop in no real time.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Page } from '@playwright/test';
import {
  waitForNavigationComplete,
  waitForSpatialQuery,
  waitForWebGLError,
} from '../../e2e/helpers';

/** `page.waitForTimeout`, as a real timer promise so fake timers drive it. */
function fakeWaitForTimeout(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** A `page.waitForFunction` that resolves at once — the flag is readable. */
function resolvingWaitForFunction(): Promise<void> {
  return Promise.resolve();
}

/**
 * A `page.waitForFunction` that rejects at its own timeout, the way Playwright's
 * does when the predicate never holds.
 */
function rejectingWaitForFunction(
  _fn: unknown,
  _arg?: unknown,
  options?: { timeout?: number }
): Promise<void> {
  return new Promise((_resolve, reject) => {
    setTimeout(
      () => reject(new Error('page.waitForFunction: Timeout exceeded')),
      options?.timeout ?? 0
    );
  });
}

/** A page whose state probe never settles — the starvation case of #1724. */
function wedgedPage(): Page {
  return {
    evaluate: () => new Promise(() => {}),
    waitForTimeout: fakeWaitForTimeout,
    waitForFunction: rejectingWaitForFunction,
  } as unknown as Page;
}

/** A page whose state probe answers immediately with `state`. */
function answeringPage(state: unknown): Page {
  return {
    evaluate: async () => state,
    waitForTimeout: fakeWaitForTimeout,
    waitForFunction: resolvingWaitForFunction,
  } as unknown as Page;
}

/** A page whose state probe rejects — e.g. no debug interface installed. */
function rejectingPage(message: string): Page {
  return {
    evaluate: async () => {
      throw new Error(message);
    },
    waitForTimeout: fakeWaitForTimeout,
    waitForFunction: rejectingWaitForFunction,
  } as unknown as Page;
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

/**
 * Run `helper()` under fake timers, advancing well past its budget, and report
 * whether it resolved or rejected. Wrapping the resolve path in a
 * `{ resolvedWith }` envelope — which no rejection can produce — is what lets a
 * test assert the helper did NOT resolve.
 */
async function settleUnderFakeTimers(
  helper: () => Promise<unknown>,
  advanceMs: number
): Promise<unknown> {
  vi.useFakeTimers();
  const settled = helper().then(
    (value) => ({ resolvedWith: value }),
    (error: unknown) => error
  );
  await vi.advanceTimersByTimeAsync(advanceMs);
  return settled;
}

describe('waitForSpatialQuery end-of-budget decision', () => {
  it('rejects, naming itself and carrying the probe diagnostic, when no probe ever answers', async () => {
    const outcome = await settleUnderFakeTimers(
      () => waitForSpatialQuery(wedgedPage(), 1000),
      4000
    );

    expect(outcome).toBeInstanceOf(Error);
    const message = (outcome as Error).message;
    expect(message).toContain('waitForSpatialQuery');
    expect(message).toMatch(/never answered/i);
    expect(message).toContain('1000ms');
    expect(message).toContain('#1726');
    // The clamped probe's own diagnostic, carried rather than discarded.
    expect(message).toMatch(/getLuxarState did not answer/);
  });

  it('does NOT resolve when no probe ever answers', async () => {
    // The vacuous-pass guard: resolving here hands the caller's next assertions
    // state this helper never read.
    const outcome = await settleUnderFakeTimers(
      () => waitForSpatialQuery(wedgedPage(), 1000),
      4000
    );

    expect(outcome).not.toHaveProperty('resolvedWith');
  });

  it('carries a rejecting probe’s own message when the debug interface is missing', async () => {
    const outcome = await settleUnderFakeTimers(
      () =>
        waitForSpatialQuery(
          rejectingPage('Debug interface not ready: getState() not available'),
          1000
        ),
      4000
    );

    expect(outcome).toBeInstanceOf(Error);
    expect((outcome as Error).message).toContain('Debug interface not ready');
  });

  it('resolves with exactly one warning when a probe answers but the query never settles', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const outcome = await settleUnderFakeTimers(
      () => waitForSpatialQuery(answeringPage({ isLoading: true, totalPoints: 7 }), 1000),
      4000
    );

    expect(outcome).toEqual({ resolvedWith: undefined });
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toContain('[⚠️] [E2E waitForSpatialQuery]');
  });

  it('is silent and prompt on a healthy page', async () => {
    // The nine silent-variant call sites live on this path: their first probe
    // already reads `isLoading: false`, so a fix that warned or threw here
    // would make every one of them noisy.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    await expect(
      waitForSpatialQuery(answeringPage({ isLoading: false, totalPoints: 0 }), 1000)
    ).resolves.toBeUndefined();
    expect(warn).not.toHaveBeenCalled();
  });
});

describe('waitForNavigationComplete end-of-budget decision', () => {
  it('rejects, naming itself and carrying the probe diagnostic, when no probe ever answers', async () => {
    // The preliminary `waitForFunction` also fails on a wedged page. It used to
    // return outright at that point; now it records the failure and falls
    // through, so the loop's clamped probe produces the carried diagnostic.
    const outcome = await settleUnderFakeTimers(
      () => waitForNavigationComplete(wedgedPage(), 15000),
      40000
    );

    expect(outcome).toBeInstanceOf(Error);
    const message = (outcome as Error).message;
    expect(message).toContain('waitForNavigationComplete');
    expect(message).toMatch(/never answered/i);
    expect(message).toContain('15000ms');
    expect(message).toContain('#1726');
    expect(message).toMatch(/getLuxarState did not answer/);
  });

  it('does NOT resolve when no probe ever answers', async () => {
    const outcome = await settleUnderFakeTimers(
      () => waitForNavigationComplete(wedgedPage(), 15000),
      40000
    );

    expect(outcome).not.toHaveProperty('resolvedWith');
  });

  it('rejects even when the preliminary isLoading wait itself is what failed', async () => {
    // A page that never installs the debug interface: `waitForFunction` rejects
    // and the state probe rejects too. The most recent failure is carried.
    const outcome = await settleUnderFakeTimers(
      () =>
        waitForNavigationComplete(
          rejectingPage('Debug interface not ready: getState() not available'),
          15000
        ),
      40000
    );

    expect(outcome).toBeInstanceOf(Error);
    expect((outcome as Error).message).toContain('Debug interface not ready');
  });

  it('resolves with exactly one warning when a probe answers but isLoading never clears', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const outcome = await settleUnderFakeTimers(
      () => waitForNavigationComplete(answeringPage({ isLoading: true, totalPoints: 7 }), 15000),
      40000
    );

    expect(outcome).toEqual({ resolvedWith: undefined });
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toContain('[⚠️] [E2E waitForNavigationComplete]');
  });

  it('is silent and prompt on a healthy page', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const outcome = await settleUnderFakeTimers(
      () => waitForNavigationComplete(answeringPage({ isLoading: false, totalPoints: 3 }), 15000),
      500
    );

    expect(outcome).toEqual({ resolvedWith: undefined });
    expect(warn).not.toHaveBeenCalled();
  });
});

describe('waitForWebGLError budget', () => {
  it('returns within its own budget, warning once, when no read is ever answered', async () => {
    // Without the bound the first unanswered `getWebGLErrors` evaluate runs to
    // the whole test budget instead of to this 1000ms one.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const outcome = await settleUnderFakeTimers(
      () => waitForWebGLError(wedgedPage(), (errors) => errors.length > 0, { timeout: 1000 }),
      4000
    );

    expect(outcome).toEqual({ resolvedWith: [] });
    expect(warn).toHaveBeenCalledTimes(1);
    const message = warn.mock.calls[0][0] as string;
    expect(message).toContain('[⚠️] [E2E waitForWebGLError]');
    expect(message).toContain('never answered');
  });

  it('returns a matching error without warning', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    await expect(
      waitForWebGLError(
        answeringPage(['GL_INVALID_OPERATION']),
        (errors) => errors.some((e) => e.includes('INVALID_OPERATION')),
        { timeout: 1000 }
      )
    ).resolves.toEqual(['GL_INVALID_OPERATION']);
    expect(warn).not.toHaveBeenCalled();
  });
});
