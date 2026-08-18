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
 * `__luxarDebug`, a destroyed execution context, and — since the probe carries a
 * deadline of its own, #1651 / #1724 — a starved page rejecting): the helper
 * returns having read nothing, and whatever the spec asserts next reads
 * unconfirmed state.
 *
 * At today's nine `waitForNavigationComplete` call sites that silence did not
 * produce a PASSING test: the next statement is always a page probe of its own,
 * so the run failed one line later under someone else's name (`Debug interface
 * not ready`, or a `-1` from `getDimensionValue` failing its own `expect`), and
 * six of the nine additionally sit right after a `waitForSpatialQueryOrThrow`
 * that could not have succeeded unless the page answered seconds earlier. So what
 * the tests below pin is mainly ATTRIBUTION — the helper naming itself, its
 * budget, its measured elapsed time, how many probes went out and how many yielded
 * no usable state,
 * and the probe failure it carried — plus the contract for a future caller that
 * does NOT re-probe straight afterwards, for which the same silence would be a
 * genuinely vacuous pass.
 *
 * A probe that never answers at all is a different animal, and no longer a hang.
 * These loops still do not clamp the probe to their own budget — they succeed on
 * ANY satisfying answer, so a clamp would fail a healthy-but-slow page — but
 * `getLuxarState` carries a 45 s deadline of its own and REJECTS (#1651 / #1724),
 * so an unanswering page reaches the end-of-budget decision as a rejection and the
 * helper throws with the starvation diagnostic attached. The two "attributed
 * rejection" tests pin that, measured-elapsed figure included; they deliberately
 * assert the shape of the elapsed report rather than a literal 45 s, so the
 * probe's deadline stays free to move.
 *
 * The PRE-change code these are scored against: the spatial loop caught every
 * probe failure in a bare `catch {}` and ended with a silent `return`; the nav loop
 * had the same silent tail plus a preliminary `catch { waitForTimeout(300); return; }`;
 * the webgl loop `await`ed an unbounded `getWebGLErrors` and never warned. None of
 * the three had a carried-failure variable, a probe counter or an `answered` flag,
 * so any assertion on those is red by absence. Where a row's redness is only that,
 * the Mechanism column names the mutation of the CURRENT code the test actually
 * discriminates.
 *
 * | Test                                                              | Pre-change | Mechanism                                                     |
 * | ----------------------------------------------------------------- | ---------- | ------------------------------------------------------------- |
 * | spatial: rejects at its budget when every probe fails             | RED        | assertion — the old loop resolved silently                    |
 * | spatial: a non-positive budget reports zero probes                | RED        | assertion — the old loop resolved silently; pins the counts over an empty window |
 * | spatial: an unanswering probe ends in the attributed rejection    | RED        | assertion — the old loop swallowed the probe's rejection      |
 * | spatial: one warning when answered but never settled              | RED        | assertion — zero warnings                                     |
 * | spatial: warns and resolves when a probe answers `null`           | RED        | assertion — zero warnings; pins `answered` set BEFORE the read |
 * | spatial: a stale early failure is cleared but still counted        | RED        | assertion — zero warnings; pins `lastProbeError = undefined` AND the unusable count |
 * | spatial: silent and prompt on a healthy page                      | green      | counterweight: goes red if the fix makes this loud            |
 * | nav: rejects when the preliminary wait failed too                 | RED        | assertion — the old code returned from that `catch`           |
 * | nav: rejects when only the probes fail (preliminary resolved)     | RED        | assertion — the old loop resolved silently                    |
 * | nav: falls through a failed preliminary wait and probes           | RED        | assertion — the old code returned without probing             |
 * | nav: probes once when the preliminary wait spent the whole budget | RED        | assertion — old `catch` returned, 0 probes, no rejection      |
 * | nav: a non-positive budget cannot mean "no timeout"               | RED        | hang — `timeout: 0` reached Playwright as NO timeout          |
 * | nav: warns and resolves when a probe answers `null`               | RED        | assertion — zero warnings (old loop swallowed it); pins the chronological clause order |
 * | nav: a stale early failure is cleared but still counted           | RED        | assertion — zero warnings; pins `lastProbeError = undefined` in THIS loop too |
 * | nav: names the preliminary failure without counting it as a probe failure | RED | assertion — zero warnings; pins the recorded preliminary failure |
 * | nav: one warning when answered but `isLoading` never cleared      | RED        | assertion — zero warnings                                     |
 * | nav: an unanswering probe ends in the attributed rejection        | RED        | assertion — the old loop swallowed the probe's rejection      |
 * | nav: silent and prompt on a healthy page                          | green      | counterweight                                                 |
 * | webgl: returns within its own budget, warning once                | RED        | vitest `testTimeout` hang — the read was unbounded; the elapsed figure pins the deadline break |
 * | webgl: a rejecting read propagates                                | green      | old code propagated too; pins that the bound did not swallow it |
 * | webgl: accumulates errors across polls                            | green      | old code accumulated too; pins the union against a latest-read-only regression |
 * | webgl: the give-up warn counts the accumulated union              | RED        | assertion — zero warnings; the count pins the same union       |
 * | webgl: returns `[]` with one warning saying reads were answered   | RED        | assertion — zero warnings                                     |
 * | webgl: dispatches one read when `timeout` is under `pollMs`       | green      | old code read once too; pins the sliver-skip regression        |
 * | webgl: skips only a read that cannot answer within the remainder  | RED        | assertion — the unbounded old loop read 3 times, not 2         |
 * | webgl: returns a matching error without warning                   | green      | counterweight                                                 |
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
 * does when the predicate never holds — including Playwright's reading of
 * `timeout: 0` as NO timeout, which is what the helper's 1 ms floor exists for: a
 * zero here never settles at all.
 */
function rejectingWaitForFunction(
  _fn: unknown,
  _arg?: unknown,
  options?: { timeout?: number }
): Promise<void> {
  const timeout = options?.timeout ?? 0;
  if (timeout <= 0) return new Promise<void>(() => {});
  return new Promise((_resolve, reject) => {
    setTimeout(() => reject(new Error('page.waitForFunction: Timeout exceeded')), timeout);
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
 * `{ resolvedWith }` envelope — which no rejection can produce — is what keeps a
 * resolve distinguishable from a rejection in one returned value.
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
 * Pull the measured-elapsed figure out of an end-of-budget report, so a test can
 * assert it exceeds the nominal budget without hardcoding the probe's own
 * deadline (which is `getLuxarState`'s business, not these loops').
 */
function reportedElapsedMs(message: string): number {
  return Number(/(\d+)ms actually elapsed/.exec(message)?.[1]);
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

  it('reports zero probes when a non-positive budget keeps it out of the loop', async () => {
    // The loop guard fails on entry, so nothing was tried — and the counts say so
    // rather than implying probes that went out and came back empty.
    await expect(waitForSpatialQuery(fakePage(rejecting(NO_DEBUG)), 0)).rejects.toThrow(
      '0 probe(s) attempted, 0 of them yielded no usable state'
    );
  });

  it('ends in the attributed rejection when the probe never answers', async () => {
    // The starvation case of #1724. The probe is not clamped to this loop's
    // budget, but it carries its own deadline (#1651 / #1724) and rejects, so the loop
    // reaches its decision holding that diagnostic instead of hanging. The
    // elapsed figure is asserted as "longer than the budget", never as a literal,
    // so the probe's deadline stays free to move.
    const outcome = await settleUnderFakeTimers(
      () => waitForSpatialQuery(fakePage(neverSettling), 1000),
      60000
    );

    expect(outcome).toBeInstanceOf(Error);
    const message = (outcome as Error).message;
    expect(message).toContain('waitForSpatialQuery');
    expect(message).toContain('1000ms budget');
    expect(reportedElapsedMs(message)).toBeGreaterThan(1000);
    // The probe's own starvation diagnostic, carried rather than discarded.
    expect(message).toContain('never answered the state probe');
    expect(message).toContain('#1724');
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

  it('warns and resolves when a probe answers null, instead of reading that as no answer', async () => {
    // A resolved `null` IS an answer: the page replied. Reading the condition off
    // it then throws, which is recorded as a probe FAILURE, so this path must warn
    // and RETURN — the twin of the nav null-answer test below. Moving
    // `answered = true` under the condition read turns this into a throw.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const outcome = await settleUnderFakeTimers(
      () => waitForSpatialQuery(fakePage(answering(null)), 1000),
      4000
    );

    expect(outcome).toEqual({ resolvedWith: undefined });
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toContain('Last state probe failed:');
  });

  it('does not blame a stale early failure once a probe has answered, but still counts it', async () => {
    // The initialization window: the first probe misses the debug interface and
    // every one after it answers. Carrying that first failure to the give-up would
    // blame it for a loop that then polled a healthy page for the rest of its
    // budget, which is why the record is cleared on every answered probe — as in
    // the two sibling loops (see `describeLastProbeError`). The COUNTS are what
    // survives the clearing: without them a window whose probes nearly all failed
    // and whose last one answered reads as a clean "answered but never settled".
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    let probes = 0;
    const page = fakePage(async () => {
      probes += 1;
      if (probes === 1) throw new Error(NO_DEBUG);
      return { isLoading: true, totalPoints: 7 };
    });

    const outcome = await settleUnderFakeTimers(() => waitForSpatialQuery(page, 1000), 4000);

    expect(outcome).toEqual({ resolvedWith: undefined });
    expect(warn).toHaveBeenCalledTimes(1);
    const message = warn.mock.calls[0][0] as string;
    expect(message).not.toContain(NO_DEBUG);
    expect(message).toContain(`${probes} probe(s) attempted, 1 of them yielded no usable state`);
  });

  it('is silent and prompt on a healthy page', async () => {
    // Counterweight: this loop is shared in shape with the nine live
    // `waitForNavigationComplete` call sites below, so a fix that warned here
    // would make every one of them noisy too.
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

  it('cannot let a non-positive budget mean "no timeout" in the preliminary wait', async () => {
    // Playwright reads `timeout: 0` as NO timeout, so an unfloored
    // `Math.min(timeout, 10000)` would park the preliminary wait on a page that
    // never installs the flag for the whole TEST budget — the opposite of the
    // ceiling that expression is there for. With the 1 ms floor the wait fails
    // almost at once, the loop probes once and the call ends in its own report.
    const outcome = await settleUnderFakeTimers(
      () => waitForNavigationComplete(fakePage(rejecting(NO_DEBUG), rejectingWaitForFunction), 0),
      5000
    );

    expect(outcome).toBeInstanceOf(Error);
    const message = (outcome as Error).message;
    expect(message).toContain('waitForNavigationComplete');
    expect(message).toContain('1 probe(s) attempted, 1 of them yielded no usable state');
    expect(message).toContain(NO_DEBUG);
  });

  it('warns and resolves when a probe answers null, instead of reading that as no answer', async () => {
    // A resolved `null` IS an answer — the page replied — which is why `answered`
    // is set BEFORE the condition is read. Reading the condition then throws on
    // the null and is counted among the probes that yielded no usable state, so
    // this path must warn and RETURN. Moving the flag below the condition turns it
    // into a throw. The preliminary wait fails here too, so the one warn carries
    // BOTH clauses and can pin their order.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const outcome = await settleUnderFakeTimers(
      () => waitForNavigationComplete(fakePage(answering(null), rejectingWaitForFunction), 5000),
      20000
    );

    expect(outcome).toEqual({ resolvedWith: undefined });
    expect(warn).toHaveBeenCalledTimes(1);
    // The carried diagnostic, not just the bare "never cleared" sentence. Shared
    // wording with the two sibling loops (`describeLastProbeError`), so one grep
    // finds every carried-failure report.
    const message = warn.mock.calls[0][0] as string;
    expect(message).toContain('Last state probe failed:');
    // Chronological: the preliminary wait is the first thing the call does, so its
    // clause precedes the last probe's.
    expect(message.indexOf('The preliminary wait')).toBeLessThan(
      message.indexOf('Last state probe failed:')
    );
  });

  it('names the preliminary wait failure in the warn without counting it as a probe failure', async () => {
    // `isLoading` never became readable, yet every state probe afterwards answers.
    // "The flag was unreadable for the first N ms" is what explains a budget that
    // never saw `isLoading` clear, so the warn names it with the time it cost. It is
    // a different datum from a probe failure: it must not appear as the carried
    // `Last state probe failed` diagnostic (an answered probe clears that record)
    // and must not be counted among the probes that yielded no usable state.
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
    const message = warn.mock.calls[0][0] as string;
    expect(message).toContain('[⚠️] [E2E waitForNavigationComplete]');
    // The preliminary wait spent the whole 5000 ms budget before the loop probed.
    expect(message).toContain('The preliminary wait never saw a readable isLoading flag');
    expect(message).toContain('the first 5000ms of this call');
    expect(message).not.toContain('Last state probe failed');
    expect(message).toContain('1 probe(s) attempted, 0 of them yielded no usable state');
  });

  it('does not blame a stale early failure once a probe has answered, but still counts it', async () => {
    // The nav twin of the spatial test above: the first probe misses the debug
    // interface, every one after it answers, and the loop still ends on its budget.
    // Deleting `lastProbeError = undefined` from THIS loop is what this catches —
    // the spatial test cannot, and no other nav test has a failing probe followed by
    // an answering one.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    let probes = 0;
    const page = fakePage(async () => {
      probes += 1;
      if (probes === 1) throw new Error(NO_DEBUG);
      return { isLoading: true, totalPoints: 7 };
    }, resolvingWaitForFunction);

    const outcome = await settleUnderFakeTimers(
      () => waitForNavigationComplete(page, 15000),
      40000
    );

    expect(outcome).toEqual({ resolvedWith: undefined });
    expect(warn).toHaveBeenCalledTimes(1);
    const message = warn.mock.calls[0][0] as string;
    expect(message).not.toContain(NO_DEBUG);
    expect(message).toContain(`${probes} probe(s) attempted, 1 of them yielded no usable state`);
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

  it('ends in the attributed rejection when the probe never answers', async () => {
    // As above: the unanswered probe rejects on its own deadline, so this loop
    // ends in a named failure carrying that diagnostic rather than hanging until
    // the whole test expires.
    const outcome = await settleUnderFakeTimers(
      () => waitForNavigationComplete(fakePage(neverSettling, resolvingWaitForFunction), 15000),
      60000
    );

    expect(outcome).toBeInstanceOf(Error);
    const message = (outcome as Error).message;
    expect(message).toContain('waitForNavigationComplete');
    expect(message).toContain('15000ms budget');
    expect(reportedElapsedMs(message)).toBeGreaterThan(15000);
    expect(message).toContain('never answered the state probe');
    expect(message).toContain('#1724');
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
    // Exactly the budget, not a poll interval past it: the read consumed the whole
    // remainder, and sleeping `pollMs` afterwards would report 1100 ms.
    expect(message).toContain('(1000ms actually elapsed');
  });

  it('propagates a read that REJECTS instead of reading it as a clean GL queue', async () => {
    // A closed page, a destroyed execution context or an in-page throw out of
    // `getWebGLErrors` all arrive as a rejection. Bounding the read must not turn
    // any of them into an empty accumulation and a "genuine absence" warn.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    await expect(
      waitForWebGLError(
        fakePage(rejecting('Execution context was destroyed')),
        (errors) => errors.length > 0,
        { timeout: 1000 }
      )
    ).rejects.toThrow('Execution context was destroyed');
    expect(warn).not.toHaveBeenCalled();
  });

  it('accumulates errors across polls and tests the predicate against the union', async () => {
    // The whole reason this helper exists: `getWebGLErrors` DRAINS the queue, so
    // each poll returns only what arrived since the last one. An empty first read
    // must not reset the buffer, and no single read here satisfies the predicate —
    // keeping only the latest read leaves it unsatisfied forever.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const answers = [[], ['GL_INVALID_ENUM'], ['GL_OUT_OF_MEMORY']];
    let reads = 0;
    const page = fakePage(async () => answers[reads++] ?? []);

    const outcome = await settleUnderFakeTimers(
      () => waitForWebGLError(page, (errors) => errors.length >= 2, { timeout: 1000 }),
      4000
    );

    expect(outcome).toEqual({ resolvedWith: ['GL_INVALID_ENUM', 'GL_OUT_OF_MEMORY'] });
    expect(warn).not.toHaveBeenCalled();
  });

  it('counts the accumulated union in the give-up warn', async () => {
    // One error seen on the second poll, a predicate that wants two: the helper
    // gives up, and the count it interpolates is the UNION it kept, not the last
    // (empty) read.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const answers = [[], ['GL_OUT_OF_MEMORY']];
    let reads = 0;
    const page = fakePage(async () => answers[reads++] ?? []);

    const outcome = await settleUnderFakeTimers(
      () => waitForWebGLError(page, (errors) => errors.length >= 2, { timeout: 1000 }),
      4000
    );

    expect(outcome).toEqual({ resolvedWith: ['GL_OUT_OF_MEMORY'] });
    expect(warn).toHaveBeenCalledTimes(1);
    const message = warn.mock.calls[0][0] as string;
    expect(message).toContain('1 error(s) accumulated');
    expect(message).toContain('genuine absence');
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
