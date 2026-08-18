/**
 * End-of-budget behaviour of the E2E wait helpers that give up SILENTLY
 * (#1726): `waitForSpatialQuery`, `waitForNavigationComplete` and
 * `waitForWebGLError`.
 *
 * All three used to end their poll loop by simply returning, on the reasoning
 * that the condition might have settled before polling started. That is a real
 * case — several `waitForNavigationComplete` call sites deliberately pick the
 * silent variant because their action never toggles `isLoading` at all — but it
 * is indistinguishable from a run in which every probe FAILED (no
 * `__luxarDebug`, a destroyed execution context, and — once the probe carries a
 * deadline of its own, #1725 — a starved page rejecting): the helper returns
 * having read nothing, and whatever the spec asserts next reads unconfirmed
 * state.
 *
 * At today's nine `waitForNavigationComplete` call sites that silence did not
 * produce a PASSING test: the next statement is always a page probe of its own,
 * so the run failed one line later under someone else's name (`Debug interface
 * not ready`, or a `-1` from `getDimensionValue` failing its own `expect`), and
 * six of the nine additionally sit right after a `waitForSpatialQueryOrThrow`
 * that could not have succeeded unless the page answered seconds earlier. So what
 * the tests below pin is mainly ATTRIBUTION — the helper naming itself, its
 * budget, its measured elapsed time and the probe failure it carried — plus the
 * contract for a future caller that does NOT re-probe straight afterwards, for
 * which the same silence would be a genuinely vacuous pass.
 *
 * A probe that never answers at all is a different animal: it hangs the loop,
 * because `page.evaluate` carries no deadline and these loops deliberately do
 * not clamp one on (they succeed on ANY satisfying answer, so a clamp would fail
 * a healthy-but-slow page). The two "stays pending" tests pin only that the
 * helpers do not fabricate a success there; turning that hang into an
 * attributable failure is the probe's job (#1725), not the helper's.
 *
 * Redness against the PRE-change code, per test:
 *
 * | Test                                                              | Pre-change | Mechanism                                                |
 * | ----------------------------------------------------------------- | ---------- | -------------------------------------------------------- |
 * | spatial: rejects at its budget when every probe fails             | RED        | assertion — the old loop resolved silently               |
 * | spatial: does NOT resolve when every probe fails                  | RED        | assertion — `resolvedWith` present                       |
 * | spatial: stays pending when the probe never settles               | green      | old code hung too; guards against a fabricated pass      |
 * | spatial: one warning when answered but never settled              | RED        | assertion — zero warnings                                |
 * | spatial: silent and prompt on a healthy page                      | green      | counterweight: goes red if the fix makes this loud       |
 * | nav: rejects when the preliminary wait failed too                 | RED        | assertion — the old code returned from that `catch`      |
 * | nav: rejects when only the probes fail (preliminary resolved)     | RED        | assertion — the old loop resolved silently               |
 * | nav: does NOT resolve when every probe fails                      | RED        | assertion — `resolvedWith` present                       |
 * | nav: falls through a failed preliminary wait and probes           | RED        | assertion — the old code returned without probing        |
 * | nav: probes once when the preliminary wait spent the whole budget | RED        | assertion — old `catch` returned, 0 probes, no rejection |
 * | nav: warns and resolves when a probe answers `null`               | RED        | assertion — zero warnings (old loop swallowed it)        |
 * | nav: preliminary failure reaches the report                       | RED        | assertion — zero warnings; old `catch` returned at once  |
 * | nav: one warning when answered but `isLoading` never cleared      | RED        | assertion — zero warnings                                |
 * | nav: stays pending when the probe never settles                   | green      | old code hung too; guards against a fabricated pass      |
 * | nav: silent and prompt on a healthy page                          | green      | counterweight                                            |
 * | webgl: returns within its own budget, warning once                | RED        | vitest `testTimeout` hang — the read was unbounded       |
 * | webgl: returns `[]` with one warning saying reads were answered   | RED        | assertion — zero warnings                                |
 * | webgl: dispatches one read when `timeout` is under `pollMs`       | green      | old code read once too; pins the sliver-skip regression  |
 * | webgl: skips only a read that cannot answer within the remainder  | RED        | assertion — the unbounded old loop read 3 times, not 2   |
 * | webgl: returns a matching error without warning                   | green      | counterweight                                            |
 *
 * No browser: every `page` below is a hand-built three-method fake cast to
 * Playwright's `Page`. `waitForTimeout` is a real `setTimeout` promise so
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

/** Build a fake page from an `evaluate` and a `waitForFunction` behaviour. */
function fakePage(
  evaluate: () => Promise<unknown>,
  waitForFunction: (
    fn?: unknown,
    arg?: unknown,
    options?: { timeout?: number }
  ) => Promise<void> = rejectingWaitForFunction
): Page {
  return { evaluate, waitForTimeout: fakeWaitForTimeout, waitForFunction } as unknown as Page;
}

/** An `evaluate` that never settles — the starvation case of #1724. */
const neverSettling = () => new Promise<never>(() => {});

/** An `evaluate` that answers immediately with `state`. */
const answering = (state: unknown) => async () => state;

/** An `evaluate` that rejects — e.g. no debug interface installed. */
const rejecting = (message: string) => async () => {
  throw new Error(message);
};

const NO_DEBUG = 'Debug interface not ready: getState() not available';

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

/**
 * Start `helper()` under fake timers, advance past `advanceMs`, and report
 * whether it settled AT ALL. Both handlers are attached so a rejection can
 * never go unhandled, and the helper promise is deliberately never awaited —
 * the case under test is one that never settles.
 */
async function settledAfterAdvancing(
  helper: () => Promise<unknown>,
  advanceMs: number
): Promise<boolean> {
  vi.useFakeTimers();
  let settled = false;
  const mark = () => {
    settled = true;
  };
  void helper().then(mark, mark);
  await vi.advanceTimersByTimeAsync(advanceMs);
  return settled;
}

describe('waitForSpatialQuery end-of-budget decision', () => {
  it('rejects exactly at its budget, naming itself, the budget, the issue and the probe failure', async () => {
    vi.useFakeTimers();
    const settled = waitForSpatialQuery(fakePage(rejecting(NO_DEBUG)), 1000).then(
      () => 'resolved',
      (error: unknown) => error
    );

    // The budget value itself is pinned, not just "somewhere under 4x".
    await vi.advanceTimersByTimeAsync(999);
    expect(await Promise.race([settled, Promise.resolve('pending')])).toBe('pending');
    await vi.advanceTimersByTimeAsync(1);

    const outcome = await settled;
    expect(outcome).toBeInstanceOf(Error);
    const message = (outcome as Error).message;
    expect(message).toContain('waitForSpatialQuery');
    expect(message).toContain('1000ms');
    expect(message).toContain('#1726');
    // The rejecting probe's own diagnostic, carried rather than discarded.
    expect(message).toContain(NO_DEBUG);
  });

  it('does NOT resolve when every probe fails', async () => {
    // The vacuous-pass guard: resolving here hands the caller's next assertions
    // state this helper never read.
    const outcome = await settleUnderFakeTimers(
      () => waitForSpatialQuery(fakePage(rejecting(NO_DEBUG)), 1000),
      4000
    );

    expect(outcome).not.toHaveProperty('resolvedWith');
  });

  it('stays pending past its budget when the probe never settles', async () => {
    // Not red pre-change — the old loop hung here too, and by design: the probe
    // is not clamped to this loop's budget. What this pins is that the helper
    // does not FABRICATE a success out of a page that answered nothing.
    // Attributing the hang is the probe's own deadline (#1725), not this loop's.
    const settled = await settledAfterAdvancing(
      () => waitForSpatialQuery(fakePage(neverSettling), 1000),
      4000
    );

    expect(settled).toBe(false);
  });

  it('resolves with exactly one warning when a probe answers but the query never settles', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const outcome = await settleUnderFakeTimers(
      () => waitForSpatialQuery(fakePage(answering({ isLoading: true, totalPoints: 7 })), 1000),
      4000
    );

    expect(outcome).toEqual({ resolvedWith: undefined });
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toContain('[⚠️] [E2E waitForSpatialQuery]');
  });

  it('is silent and prompt on a healthy page', async () => {
    // This helper and `waitForWebGLError` have no call sites today, so their
    // healthy-path assertions guard a contract with no current caller. They
    // still matter: the loop is shared in shape with the nine live
    // `waitForNavigationComplete` call sites below.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    await expect(
      waitForSpatialQuery(fakePage(answering({ isLoading: false, totalPoints: 0 })), 1000)
    ).resolves.toBeUndefined();
    expect(warn).not.toHaveBeenCalled();
  });
});

describe('waitForNavigationComplete end-of-budget decision', () => {
  it('rejects, carrying the probe failure, when the preliminary wait failed too', async () => {
    // A page that never installs the debug interface: `waitForFunction` rejects
    // and every state probe rejects as well. The old code returned from that
    // first `catch` after a 300 ms sleep, having read nothing.
    const outcome = await settleUnderFakeTimers(
      () => waitForNavigationComplete(fakePage(rejecting(NO_DEBUG)), 15000),
      40000
    );

    expect(outcome).toBeInstanceOf(Error);
    const message = (outcome as Error).message;
    expect(message).toContain('waitForNavigationComplete');
    expect(message).toContain('15000ms');
    expect(message).toContain('#1726');
    expect(message).toContain(NO_DEBUG);
  });

  it('rejects when the preliminary wait resolved and only the probes fail', async () => {
    // The production shape: the debug interface exists (so `isLoading` is
    // readable) and the probes fail afterwards — a destroyed execution context,
    // or a probe with its own deadline rejecting on a starved page.
    const outcome = await settleUnderFakeTimers(
      () =>
        waitForNavigationComplete(
          fakePage(rejecting('Execution context was destroyed'), resolvingWaitForFunction),
          15000
        ),
      40000
    );

    expect(outcome).toBeInstanceOf(Error);
    expect((outcome as Error).message).toContain('Execution context was destroyed');
  });

  it('does NOT resolve when every probe fails', async () => {
    const outcome = await settleUnderFakeTimers(
      () =>
        waitForNavigationComplete(
          fakePage(rejecting('Execution context was destroyed'), resolvingWaitForFunction),
          15000
        ),
      40000
    );

    expect(outcome).not.toHaveProperty('resolvedWith');
  });

  it('falls through a failed preliminary wait and resolves on the probe’s answer', async () => {
    // The whole point of the fall-through: the preliminary `isLoading` wait
    // failing is not itself evidence of anything, so the state probe still gets
    // its chance. The old code returned right there, never probing.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    let probes = 0;
    const page = fakePage(async () => {
      probes += 1;
      return { isLoading: false, totalPoints: 3 };
    });

    const outcome = await settleUnderFakeTimers(
      () => waitForNavigationComplete(page, 15000),
      40000
    );

    expect(outcome).toEqual({ resolvedWith: undefined });
    expect(probes).toBeGreaterThan(0);
    expect(warn).not.toHaveBeenCalled();
  });

  it('probes once even when the preliminary wait already spent the whole budget', async () => {
    // `timeout: 5000` means the preliminary wait's own budget IS the whole budget,
    // so the poll loop is entered with nothing left. That is what the `do`/`while`
    // is for: a plain `while` would report "no state probe answered" without ever
    // having probed, and would carry the preliminary failure instead of the
    // probe's own. Every existing nav test uses 15000, where the preliminary wait
    // can only spend 10000, so a plain `while` still enters — this is the case
    // that tells the two loop shapes apart.
    let probes = 0;
    const page = fakePage(async () => {
      probes += 1;
      throw new Error(NO_DEBUG);
    }, rejectingWaitForFunction);

    const outcome = await settleUnderFakeTimers(() => waitForNavigationComplete(page, 5000), 20000);

    expect(probes).toBe(1);
    expect(outcome).toBeInstanceOf(Error);
    const message = (outcome as Error).message;
    // The PROBE's diagnostic, which a loop that never probed could not carry.
    expect(message).toContain(NO_DEBUG);
    // Nominal budget and measured elapsed, distinct here: the preliminary wait
    // consumed all 5000 ms and the loop then ran one poll interval past it. A
    // report naming only the budget would describe this call as a 5000 ms one.
    expect(message).toContain('5000ms budget');
    expect(message).toContain('5100ms actually elapsed');
  });

  it('warns and resolves when a probe answers null, instead of reading that as no answer', async () => {
    // A resolved `null` IS an answer — the page replied — which is why `answered`
    // is set BEFORE the condition is read. Reading the condition then throws on
    // the null and is recorded as a probe failure, so this path must warn and
    // RETURN. Moving the flag below the condition turns it into a throw.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const outcome = await settleUnderFakeTimers(
      () => waitForNavigationComplete(fakePage(answering(null), resolvingWaitForFunction), 5000),
      20000
    );

    expect(outcome).toEqual({ resolvedWith: undefined });
    expect(warn).toHaveBeenCalledTimes(1);
    // The carried diagnostic, not just the bare "never cleared" sentence.
    expect(warn.mock.calls[0][0]).toContain('Last probe failure:');
  });

  it('carries the preliminary wait failure into the report when the probes never fail', async () => {
    // The only path on which the preliminary `catch` is the sole source of a
    // diagnostic: `isLoading` never became readable, yet every state probe
    // afterwards answers. Dropping the record in that `catch` loses the one
    // explanation this run has.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const outcome = await settleUnderFakeTimers(
      () =>
        waitForNavigationComplete(
          fakePage(answering({ isLoading: true, totalPoints: 7 }), rejectingWaitForFunction),
          5000
        ),
      20000
    );

    expect(outcome).toEqual({ resolvedWith: undefined });
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toContain('Last probe failure: page.waitForFunction');
  });

  it('resolves with exactly one warning when a probe answers but isLoading never clears', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const outcome = await settleUnderFakeTimers(
      () =>
        waitForNavigationComplete(
          fakePage(answering({ isLoading: true, totalPoints: 7 }), resolvingWaitForFunction),
          15000
        ),
      40000
    );

    expect(outcome).toEqual({ resolvedWith: undefined });
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toContain('[⚠️] [E2E waitForNavigationComplete]');
  });

  it('stays pending past its budget when the probe never settles', async () => {
    // As above: pending, never a fabricated success. See the note in the
    // `waitForSpatialQuery` case.
    const settled = await settledAfterAdvancing(
      () => waitForNavigationComplete(fakePage(neverSettling, resolvingWaitForFunction), 15000),
      40000
    );

    expect(settled).toBe(false);
  });

  it('is silent and prompt on a healthy page', async () => {
    // The nine live silent-variant call sites are on this path: their first
    // probe already reads `isLoading: false`, so a fix that warned or threw here
    // would make every one of them noisy.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const outcome = await settleUnderFakeTimers(
      () =>
        waitForNavigationComplete(
          fakePage(answering({ isLoading: false, totalPoints: 3 }), resolvingWaitForFunction),
          15000
        ),
      500
    );

    expect(outcome).toEqual({ resolvedWith: undefined });
    expect(warn).not.toHaveBeenCalled();
  });
});

describe('waitForWebGLError budget', () => {
  it('returns within its own budget, warning once, when no read is ever answered', async () => {
    // Unlike the two loops above, this helper's reads ARE bounded: it returns
    // data, so a late answer is not a success it could keep. Without the bound
    // the first unanswered `getWebGLErrors` evaluate runs to the whole test
    // budget instead of to this 1000 ms one.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const outcome = await settleUnderFakeTimers(
      () =>
        waitForWebGLError(fakePage(neverSettling), (errors) => errors.length > 0, {
          timeout: 1000,
        }),
      4000
    );

    expect(outcome).toEqual({ resolvedWith: [] });
    expect(warn).toHaveBeenCalledTimes(1);
    const message = warn.mock.calls[0][0] as string;
    expect(message).toContain('[⚠️] [E2E waitForWebGLError]');
    expect(message).toContain('nothing was actually checked');
  });

  it('returns [] with one warning saying the reads WERE answered, on a genuine absence', async () => {
    // A clean GL queue answering `[]` on every poll against a predicate that
    // never matches: the give-up has to distinguish this from the case above.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const outcome = await settleUnderFakeTimers(
      () =>
        waitForWebGLError(fakePage(answering([])), (errors) => errors.length > 0, {
          timeout: 1000,
        }),
      4000
    );

    expect(outcome).toEqual({ resolvedWith: [] });
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toContain('genuine absence');
  });

  it('dispatches at least one read when the whole budget is shorter than one poll interval', async () => {
    // Regression pin for the sliver-skip: with `timeout` under `pollMs` the skip
    // fired on the FIRST iteration, so the helper returned `[]` warning that
    // "nothing was actually checked" — having genuinely checked nothing. Reading
    // once, bounded by the little budget there is, is the pre-#1726 behaviour.
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    let reads = 0;
    const page = fakePage(async () => {
      reads += 1;
      return [];
    });

    const outcome = await settleUnderFakeTimers(
      () => waitForWebGLError(page, (errors) => errors.length > 0, { timeout: 50 }),
      4000
    );

    expect(outcome).toEqual({ resolvedWith: [] });
    expect(reads).toBe(1);
  });

  it('skips only a read that could not answer within the remaining budget', async () => {
    // 250 ms of budget at a 100 ms poll: reads go out at 0 ms and at 100 ms, and
    // the 50 ms sliver left at 200 ms gets none — a read dispatched there would
    // drain the in-page GL queue and then lose its answer to the deadline, taking
    // the real errors with it. Exactly 2 reads, where an unbounded loop makes 3.
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    let reads = 0;
    const page = fakePage(async () => {
      reads += 1;
      return [];
    });

    const outcome = await settleUnderFakeTimers(
      () => waitForWebGLError(page, (errors) => errors.length > 0, { timeout: 250, pollMs: 100 }),
      4000
    );

    expect(outcome).toEqual({ resolvedWith: [] });
    expect(reads).toBe(2);
  });

  it('returns a matching error without warning', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    await expect(
      waitForWebGLError(
        fakePage(answering(['GL_INVALID_OPERATION'])),
        (errors) => errors.some((e) => e.includes('INVALID_OPERATION')),
        { timeout: 1000 }
      )
    ).resolves.toEqual(['GL_INVALID_OPERATION']);
    expect(warn).not.toHaveBeenCalled();
  });
});
