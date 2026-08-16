/**
 * Tick-flushing logic of `tests/e2e/render-ticks.ts` (#1651).
 *
 * `flushRenderTicks` and `reportFlushVerdict` used to be file-local to
 * `webgl-errors.spec.ts`, which meant several branches — including the one
 * that decides whether the run goes RED — had no automated coverage at all.
 * They now live in their own module, so they can be exercised here without a
 * browser: the only page surface they touch is `page.evaluate` and
 * `page.waitForFunction`, and both are faked (a small object cast to
 * Playwright's `Page`, the same trick `e2e-helpers-deadline.test.ts` uses).
 *
 * What is covered: the counting loop and its stop-at-first-miss rule, the
 * `startFrame + 1 … startFrame + ticks` targets, the aggregate budget (which
 * has to bound the KICKS too, not just the waits — the mutation that removes
 * the pre-kick check is invisible to every other test here — and which must be
 * blamed when a wait it CLAMPED is the one that times out, but not when a wait
 * loses at the full per-tick bound), the per-call `budgetMs`, the error
 * re-throw rule, the kick tally, the degenerate-input guard, and the whole
 * verdict lattice — in particular that `reportFlushVerdict` throws in exactly
 * one combination of conditions and warns in every other. Two describe blocks
 * execute the INJECTED functions for real against a stubbed `window` /
 * `document` (the scripted fakes never run them): the flush's baseline probe
 * and `>= target` predicate, and the verdict's probe. The last one asserts the
 * constants' arithmetic.
 *
 * The `waitForFunction` fakes deliberately behave like Playwright's own — they
 * honour the `timeout` they are handed and REJECT with a timeout-shaped error at
 * that deadline, and resolve only for a target the modelled counter reached. A
 * fake that resolves unconditionally after a fixed delay lets the loop overshoot
 * its own budget unnoticed and can never exercise a clamped wait.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import type { Page } from '@playwright/test';
import {
  flushRenderTicks,
  reportFlushVerdict,
  FLUSH_BUDGET_MS,
  TICK_TIMEOUT_MS,
  COUNTER_PROBE_TIMEOUT_MS,
  type FlushReport,
} from '../../e2e/render-ticks';

/** Recorded call to the fake `page.waitForFunction`. */
interface WaitCall {
  arg: unknown;
  timeout: number | undefined;
}

interface FakePageOptions {
  /** Values the successive `page.evaluate` calls resolve with. */
  evaluate: (call: number) => unknown;
  /**
   * How the Nth (0-based) `waitForFunction` behaves. It is handed the target it
   * was called with and the bound it was given, because a fake that ignores
   * both cannot model Playwright: the real call honours `opts.timeout` and
   * rejects at that deadline, and it only resolves once the target is reached.
   */
  wait?: (call: number, arg: unknown, timeout: number | undefined) => Promise<void>;
}

/**
 * A `page` whose two seams are scripted. `evaluate` is called by the counter
 * probe (call 0) and then by each kick; `waitForFunction` is the confirmation.
 */
function fakePage(options: FakePageOptions): { page: Page; waits: WaitCall[] } {
  const waits: WaitCall[] = [];
  let evaluateCalls = 0;
  let waitCalls = 0;
  const page = {
    evaluate: (): Promise<unknown> => Promise.resolve(options.evaluate(evaluateCalls++)),
    waitForFunction: (_fn: unknown, arg: unknown, opts?: { timeout?: number }): Promise<void> => {
      waits.push({ arg, timeout: opts?.timeout });
      const behaviour = options.wait ?? (() => Promise.resolve());
      return behaviour(waitCalls++, arg, opts?.timeout);
    },
  } as unknown as Page;
  return { page, waits };
}

/** A `waitForFunction` rejection that looks like Playwright's own timeout. */
function timeoutError(bound: number = TICK_TIMEOUT_MS): Error {
  return new Error(`page.waitForFunction: Timeout ${bound}ms exceeded.`);
}

/**
 * A `waitForFunction` that behaves like Playwright's own: it HONOURS the bound
 * it was handed and rejects with a timeout-shaped error at that deadline,
 * resolving only if the in-page counter would have reached the target first.
 * `reachesTargetAfterMs` is how long that takes. A fake that instead resolves
 * after a fixed delay whatever bound it was given lets the loop overshoot its
 * own budget unnoticed, and can never exercise a wait the budget CLAMPED.
 */
function pacedWait(
  reachesTargetAfterMs: number
): (call: number, arg: unknown, timeout: number | undefined) => Promise<void> {
  return (_call, _arg, timeout) =>
    new Promise<void>((resolve, reject) => {
      const bound = timeout ?? Infinity;
      if (reachesTargetAfterMs <= bound) {
        setTimeout(resolve, reachesTargetAfterMs);
      } else {
        setTimeout(() => reject(timeoutError(bound)), bound);
      }
    });
}

/**
 * A `waitForFunction` for a page whose counter sits at `value`: it resolves for
 * a target the counter has actually REACHED and times out for one it has not.
 * Modelling the target is the point — a fake that resolves unconditionally
 * cannot tell the loop's targets apart at all.
 */
function counterParkedAt(value: number): FakePageOptions['wait'] {
  return (_call, arg, timeout) =>
    typeof arg === 'number' && arg <= value
      ? Promise.resolve()
      : Promise.reject(timeoutError(timeout ?? TICK_TIMEOUT_MS));
}

/** A `page` whose `evaluate` never settles — the starvation being reproduced. */
function wedgedPage(): Page {
  return {
    evaluate: () => new Promise(() => {}),
    waitForFunction: () => Promise.resolve(),
  } as unknown as Page;
}

/** A report shaped like a zero-tick flush, overridable field by field. */
function zeroTickReport(overrides: Partial<FlushReport> = {}): FlushReport {
  return {
    requested: 3,
    landed: 0,
    startFrame: 100,
    kicks: { answered: 3, rejected: 0, 'no-answer': 0 },
    waits: 1,
    elapsedMs: 3200,
    budgetMs: FLUSH_BUDGET_MS,
    budgetExhausted: false,
    budgetClampedWaitMs: null,
    ...overrides,
  };
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('flushRenderTicks', () => {
  it('counts the ticks that land and stops at the FIRST one that does not', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    // Counter probe answers 10; kicks all answer; the third wait times out.
    const { page, waits } = fakePage({
      evaluate: (n) => (n === 0 ? { frame: 10 } : undefined),
      wait: (n) => (n < 2 ? Promise.resolve() : Promise.reject(timeoutError())),
    });

    const report = await flushRenderTicks(page, 5);

    expect(report.landed).toBe(2);
    // Three waits ran: two that landed plus the one that timed out. The
    // remaining two ticks are never attempted.
    expect(report.waits).toBe(3);
    expect(waits).toHaveLength(3);
    expect(report.requested).toBe(5);
    expect(report.startFrame).toBe(10);
    expect(report.budgetExhausted).toBe(false);
    // A shortfall is reported, never thrown.
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toContain('only 2/5 render ticks');
  });

  it('targets startFrame + 1 … startFrame + ticks', async () => {
    const { page, waits } = fakePage({ evaluate: (n) => (n === 0 ? { frame: 42 } : undefined) });

    const report = await flushRenderTicks(page, 4);

    expect(report.landed).toBe(4);
    expect(waits.map((w) => w.arg)).toEqual([43, 44, 45, 46]);
    // Each wait is bounded individually as well as in aggregate.
    for (const wait of waits) {
      expect(wait.timeout).toBeLessThanOrEqual(TICK_TIMEOUT_MS);
      expect(wait.timeout).toBeGreaterThan(0);
    }
  });

  it('handles a counter that races ahead of the targets', async () => {
    // The counter is already 500 past the baseline, so every target the loop
    // asks for has been reached and every wait resolves — modelled against the
    // target the wait actually receives, not by resolving unconditionally.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { page } = fakePage({
      evaluate: (n) => (n === 0 ? { frame: 1000 } : undefined),
      wait: counterParkedAt(1500),
    });

    const report = await flushRenderTicks(page, 3);

    expect(report.landed).toBe(3);
    expect(report.waits).toBe(3);
    expect(warn).not.toHaveBeenCalled();
  });

  it('confirms a tick only when the counter has REACHED that tick’s target', async () => {
    // The other half of the same fake: a counter parked two ticks past the
    // baseline satisfies targets 101 and 102 and never 103, so the third tick
    // times out and the loop stops there. A confirmation that ignored its
    // target would report all three as landed.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { page, waits } = fakePage({
      evaluate: (n) => (n === 0 ? { frame: 100 } : undefined),
      wait: counterParkedAt(102),
    });

    const report = await flushRenderTicks(page, 4);

    expect(waits.map((w) => w.arg)).toEqual([101, 102, 103]);
    expect(report.landed).toBe(2);
    expect(report.waits).toBe(3);
    expect(warn.mock.calls[0][0]).toContain('only 2/4 render ticks');
  });

  it('returns a startFrame-less report, without throwing, when the baseline probe goes unanswered', async () => {
    vi.useFakeTimers();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const settled = flushRenderTicks(wedgedPage(), 5);
    await vi.advanceTimersByTimeAsync(COUNTER_PROBE_TIMEOUT_MS);
    const report = await settled;

    expect(report.startFrame).toBeNull();
    expect(report.landed).toBe(0);
    expect(report.waits).toBe(0);
    expect(report.kicks).toEqual({ answered: 0, rejected: 0, 'no-answer': 0 });
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toContain('went unanswered');
  });

  it('throws when the page answers but exposes no numeric counter', async () => {
    const { page } = fakePage({ evaluate: () => ({ frame: null }) });
    await expect(flushRenderTicks(page, 3)).rejects.toThrow(/no numeric frame counter/);
  });

  it('re-throws a non-timeout waitForFunction error as itself', async () => {
    const { page } = fakePage({
      evaluate: (n) => (n === 0 ? { frame: 7 } : undefined),
      wait: () => Promise.reject(new Error('Target crashed')),
    });
    await expect(flushRenderTicks(page, 3)).rejects.toThrow('Target crashed');
  });

  it('breaks the loop on a genuine timeout error instead of re-throwing it', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { page } = fakePage({
      evaluate: (n) => (n === 0 ? { frame: 7 } : undefined),
      wait: () => Promise.reject(new Error('Timeout 3000ms exceeded')),
    });

    const report = await flushRenderTicks(page, 3);

    expect(report.landed).toBe(0);
    expect(report.waits).toBe(1);
  });

  it('tallies kick outcomes: answered, rejected and no-answer', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    let kick = 0;
    const page = {
      evaluate: (): Promise<unknown> => {
        if (kick === 0) {
          kick++;
          return Promise.resolve({ frame: 0 });
        }
        const which = kick++;
        if (which === 1) return Promise.resolve(undefined); // answered
        if (which === 2) return Promise.reject(new Error('detached')); // rejected
        return new Promise(() => {}); // never answers
      },
      waitForFunction: (): Promise<void> => Promise.resolve(),
    } as unknown as Page;

    vi.useFakeTimers();
    const settled = flushRenderTicks(page, 3);
    // Only the third kick needs the clock: the first two settle on their own.
    await vi.advanceTimersByTimeAsync(TICK_TIMEOUT_MS);
    const report = await settled;

    expect(report.kicks).toEqual({ answered: 1, rejected: 1, 'no-answer': 1 });
    expect(report.landed).toBe(3);
  });

  it('rejects a tick count that is not a positive integer, naming what is wrong with it', async () => {
    const { page } = fakePage({ evaluate: () => ({ frame: 1 }) });
    await expect(flushRenderTicks(page, 0)).rejects.toThrow(/positive integer/);
    await expect(flushRenderTicks(page, -1)).rejects.toThrow(/below 1 drives no iterations/);
    // A fraction is rejected too, so the message may not explain itself purely
    // in terms of a non-positive count.
    await expect(flushRenderTicks(page, 1.5)).rejects.toThrow(/fractional count/);
  });

  it('stops when the aggregate budget runs out, and says so', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.useFakeTimers();
    // Each tick lands just inside the per-tick bound, so only the aggregate
    // budget can end the loop — and the wait it ends on is one the remaining
    // budget clamped below TICK_TIMEOUT_MS, which is the case the flag has to
    // catch.
    const { page, waits } = fakePage({
      evaluate: (n) => (n === 0 ? { frame: 0 } : undefined),
      wait: pacedWait(TICK_TIMEOUT_MS - 100),
    });

    const settled = flushRenderTicks(page, 100);
    await vi.advanceTimersByTimeAsync(FLUSH_BUDGET_MS * 2);
    const report = await settled;

    expect(report.budgetExhausted).toBe(true);
    // 100 ticks x 3 s per wait would be 300 s; the budget caps it.
    expect(report.landed).toBeLessThanOrEqual(Math.ceil(FLUSH_BUDGET_MS / TICK_TIMEOUT_MS));
    // One wait more than landed: the extra one is the clamped wait that lost.
    expect(waits.length).toBe(report.landed + 1);
    expect(report.elapsedMs).toBeLessThanOrEqual(FLUSH_BUDGET_MS);
    expect(warn.mock.calls[0][0]).toContain('budget ran out');
  });

  it('marks the budget as exhausted when a CLAMPED wait is what times out', async () => {
    // The regression: the loop can end inside the catch rather than on the
    // pre-kick / pre-wait check, on a wait whose bound was the budget REMAINDER
    // and not TICK_TIMEOUT_MS. Without discriminating the two, the shortfall
    // warning claims every wait got the full per-tick bound and never mentions
    // that the budget is what stopped the loop.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.useFakeTimers();
    const { page, waits } = fakePage({
      evaluate: (n) => (n === 0 ? { frame: 0 } : undefined),
      wait: pacedWait(TICK_TIMEOUT_MS - 100),
    });

    const settled = flushRenderTicks(page, 100);
    await vi.advanceTimersByTimeAsync(FLUSH_BUDGET_MS * 2);
    const report = await settled;

    const lastBound = waits[waits.length - 1].timeout as number;
    expect(lastBound).toBeLessThan(TICK_TIMEOUT_MS);
    expect(report.budgetExhausted).toBe(true);
    expect(report.budgetClampedWaitMs).toBe(lastBound);
    expect(warn.mock.calls[0][0]).toContain(`clamped to ${lastBound} ms`);
    // And the verdict repeats it, so a zero-tick report says which bound ran
    // out rather than implying the per-tick one did.
    const verdict = await reportFlushVerdict(
      {
        evaluate: async () => ({ frame: 0, contextLost: false, visibility: 'visible' }),
      } as unknown as Page,
      { ...report, landed: 0 }
    ).catch((e: unknown) => e);
    expect((verdict as Error).message).toContain(
      `clamped the last confirmation wait to ${lastBound} ms`
    );
  });

  it('does NOT blame the budget when a wait times out at the full per-tick bound', async () => {
    // Negative control for the flag: same shape, but nothing is clamped — the
    // wait gets the whole TICK_TIMEOUT_MS and loses on its own merits, so the
    // budget must not be named.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.useFakeTimers();
    const { page } = fakePage({
      evaluate: (n) => (n === 0 ? { frame: 0 } : undefined),
      wait: pacedWait(Number.POSITIVE_INFINITY),
    });

    const settled = flushRenderTicks(page, 100);
    await vi.advanceTimersByTimeAsync(FLUSH_BUDGET_MS * 2);
    const report = await settled;

    expect(report.waits).toBe(1);
    expect(report.budgetExhausted).toBe(false);
    expect(report.budgetClampedWaitMs).toBeNull();
    expect(warn.mock.calls[0][0]).not.toContain('budget ran out');
  });

  it('clamps the last wait to whatever is left of the budget', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.useFakeTimers();
    const { page, waits } = fakePage({
      evaluate: (n) => (n === 0 ? { frame: 0 } : undefined),
      wait: pacedWait(TICK_TIMEOUT_MS - 100),
    });

    const settled = flushRenderTicks(page, 100);
    await vi.advanceTimersByTimeAsync(FLUSH_BUDGET_MS * 2);
    const report = await settled;

    // The real invariant is on the TIME spent, not on the sum of the bounds:
    // the last wait's bound is the remainder, so the whole loop lands on the
    // budget exactly.
    expect(report.elapsedMs).toBeLessThanOrEqual(FLUSH_BUDGET_MS);
    expect(waits[waits.length - 1].timeout).toBeLessThan(TICK_TIMEOUT_MS);
    expect(waits[waits.length - 1].timeout).toBeGreaterThan(0);
  });

  it('honours a per-call budget smaller than the module default', async () => {
    // The looped call site in `webgl-errors.spec.ts` pays this bound five times
    // inside one test, so it passes its own.
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.useFakeTimers();
    const budget = 2000;
    const { page } = fakePage({
      evaluate: (n) => (n === 0 ? { frame: 0 } : undefined),
      wait: pacedWait(TICK_TIMEOUT_MS - 100),
    });

    const settled = flushRenderTicks(page, 100, budget);
    await vi.advanceTimersByTimeAsync(FLUSH_BUDGET_MS * 2);
    const report = await settled;

    expect(report.budgetMs).toBe(budget);
    expect(report.budgetExhausted).toBe(true);
    expect(report.elapsedMs).toBeLessThanOrEqual(budget);
  });

  it('bounds the whole flush even when every KICK is what burns the time', async () => {
    // The regression this guards: the per-tick bound is paid twice per
    // iteration, so an unanswered kick is as expensive as a missed wait and
    // must be checked against the budget BEFORE it is dispatched, not only
    // after. 100 ticks x 3 s of unanswered kicks is 300 s without that check.
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.useFakeTimers();
    const page = {
      evaluate: (() => {
        let call = 0;
        return (): Promise<unknown> =>
          call++ === 0 ? Promise.resolve({ frame: 0 }) : new Promise(() => {});
      })(),
      waitForFunction: (): Promise<void> => Promise.resolve(),
    } as unknown as Page;

    const settled = flushRenderTicks(page, 100);
    await vi.advanceTimersByTimeAsync(FLUSH_BUDGET_MS * 3);
    const report = await settled;

    expect(report.budgetExhausted).toBe(true);
    expect(report.kicks['no-answer']).toBeGreaterThan(0);
    // The baseline probe answered instantly here, so the budget is the whole
    // cost. No slack: a kick dispatched after the budget was already spent
    // would push this past FLUSH_BUDGET_MS by up to another TICK_TIMEOUT_MS.
    expect(report.elapsedMs).toBeLessThanOrEqual(FLUSH_BUDGET_MS);
  });
});

describe('flushRenderTicks in-page closures', () => {
  // Neither of this function's two injected closures is EXECUTED by the
  // scripted fakes above: their `evaluate` ignores the function it is handed
  // and their `waitForFunction` discards its predicate. So the baseline probe's
  // wrapped-object answer (the sentinel contract the whole `null` handling
  // rests on) and the `>= target` predicate are run for real here against a
  // stubbed `window` — mutate either and nothing else in this file notices.
  const original = globalThis as { window?: unknown };
  const originalWindow = original.window;

  function stubCounter(info: unknown): void {
    original.window = { __luxarDebug: { renderer: { info } } };
  }

  afterEach(() => {
    if (originalWindow === undefined) delete original.window;
    else original.window = originalWindow;
  });

  /** A `page` that CALLS the injected functions, recording what they return. */
  function recordingPage(seen: unknown[]): Page {
    return {
      evaluate: async (fn: () => unknown) => {
        const value = fn();
        seen.push(value);
        return value;
      },
      waitForFunction: async () => {},
    } as unknown as Page;
  }

  it('answers the baseline probe with a WRAPPED number, on both counter names', async () => {
    const webgl: unknown[] = [];
    stubCounter({ render: { frame: 100 } });
    expect((await flushRenderTicks(recordingPage(webgl), 1)).startFrame).toBe(100);
    expect(webgl[0]).toEqual({ frame: 100 });

    // WebGPU exposes no info.render.frame, so the fallback answers.
    const webgpu: unknown[] = [];
    stubCounter({ frame: 7 });
    expect((await flushRenderTicks(recordingPage(webgpu), 1)).startFrame).toBe(7);
    expect(webgpu[0]).toEqual({ frame: 7 });
  });

  it('answers {frame: null} rather than null when there is no numeric counter', async () => {
    // The whole sentinel contract: only a missed DEADLINE may produce `null`,
    // so the in-page function has to return an object even when it found
    // nothing to report. A bare `number | null` answer here would be
    // indistinguishable from a starved probe and would take the warn-and-skip
    // path instead of failing loudly.
    const seen: unknown[] = [];
    stubCounter({});
    await expect(flushRenderTicks(recordingPage(seen), 1)).rejects.toThrow(
      /no numeric frame counter/
    );
    expect(seen[0]).toEqual({ frame: null });
    expect(seen[0]).not.toBeNull();
  });

  it('confirms a tick only once the counter has REACHED the target', async () => {
    // The predicate is executed for real, twice per wait: once before the page
    // draws (target not reached) and once after (reached exactly).
    const info = { render: { frame: 100 } };
    stubCounter(info);
    const seen: Array<{ target: number; satisfiedBefore: boolean }> = [];
    const page = {
      evaluate: async (fn: () => unknown) => fn(),
      waitForFunction: async (fn: (target: number) => boolean, target: number) => {
        seen.push({ target, satisfiedBefore: fn(target) });
        info.render.frame += 1; // one draw, as a live page would
        if (!fn(target)) throw timeoutError();
      },
    } as unknown as Page;

    const report = await flushRenderTicks(page, 3);

    expect(seen).toEqual([
      { target: 101, satisfiedBefore: false },
      { target: 102, satisfiedBefore: false },
      { target: 103, satisfiedBefore: false },
    ]);
    expect(report.landed).toBe(3);
  });
});

describe('reportFlushVerdict', () => {
  /** A `page` answering the verdict probe with exactly this object. */
  function probingPage(answer: unknown): Page {
    return { evaluate: async () => answer } as unknown as Page;
  }

  it('returns silently when at least one tick landed', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const evaluate = vi.fn();
    await reportFlushVerdict({ evaluate } as unknown as Page, zeroTickReport({ landed: 1 }));
    expect(evaluate).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
  });

  it('THROWS only when the probe answered, the context is live, the counter is frozen and a kick was answered', async () => {
    const page = probingPage({ frame: 100, contextLost: false, visibility: 'visible' });
    await expect(reportFlushVerdict(page, zeroTickReport())).rejects.toThrow(
      /demonstrably is not drawing/
    );
  });

  it('reports what the flush actually did rather than implying every tick was waited on', async () => {
    const page = probingPage({ frame: 100, contextLost: false, visibility: 'visible' });
    const error = await reportFlushVerdict(
      page,
      zeroTickReport({ requested: 10, waits: 1, elapsedMs: 3210 })
    ).catch((e: unknown) => e);
    const message = (error as Error).message;
    expect(message).toContain('1 confirmation wait(s) ran');
    expect(message).toContain('3210 ms spent');
    expect(message).toContain('document.visibilityState=visible');
  });

  it('warns instead of throwing when the follow-up probe goes unanswered', async () => {
    vi.useFakeTimers();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const settled = reportFlushVerdict(wedgedPage(), zeroTickReport());
    await vi.advanceTimersByTimeAsync(COUNTER_PROBE_TIMEOUT_MS);
    await expect(settled).resolves.toBeUndefined();
    expect(warn.mock.calls[0][0]).toContain('observation channel itself is starved');
  });

  it('warns instead of throwing when the context reports itself lost', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const page = probingPage({ frame: 100, contextLost: true, visibility: 'visible' });
    await expect(reportFlushVerdict(page, zeroTickReport())).resolves.toBeUndefined();
    expect(warn.mock.calls[0][0]).toContain('reports itself LOST');
  });

  it('warns instead of throwing when the flush read no baseline', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const page = probingPage({ frame: 100, contextLost: false, visibility: 'visible' });
    await expect(
      reportFlushVerdict(page, zeroTickReport({ startFrame: null }))
    ).resolves.toBeUndefined();
    expect(warn.mock.calls[0][0]).toContain('never read a baseline');
  });

  it('warns instead of throwing when the counter has disappeared since the flush', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const page = probingPage({ frame: null, contextLost: false, visibility: 'visible' });
    await expect(reportFlushVerdict(page, zeroTickReport())).resolves.toBeUndefined();
    expect(warn.mock.calls[0][0]).toContain('no numeric frame counter');
  });

  it('warns instead of throwing when the counter advanced — only the observation was starved', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const page = probingPage({ frame: 137, contextLost: false, visibility: 'visible' });
    await expect(reportFlushVerdict(page, zeroTickReport())).resolves.toBeUndefined();
    expect(warn.mock.calls[0][0]).toContain('the renderer DID draw');
  });

  it('warns instead of throwing when NOT ONE kick was answered', async () => {
    // The animation loop auto-pauses after ~2 s, so an unasked renderer is a
    // complete explanation for a frozen counter.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const page = probingPage({ frame: 100, contextLost: false, visibility: 'visible' });
    await expect(
      reportFlushVerdict(
        page,
        zeroTickReport({ kicks: { answered: 0, rejected: 1, 'no-answer': 2 } })
      )
    ).resolves.toBeUndefined();
    expect(warn.mock.calls[0][0]).toContain('NOT ONE kick was answered');
  });

  it('warns instead of throwing when the page is NOT VISIBLE and the counter is frozen', async () => {
    // Everything else the throw needs is present — the probe answered inside
    // its deadline, the context is live, the flush had a baseline, the counter
    // is frozen at exactly that baseline and kicks were answered — so this
    // asserts the visibility branch and nothing else. A hidden page gets no rAF
    // callbacks at all, which explains the frozen counter by itself.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const page = probingPage({ frame: 100, contextLost: false, visibility: 'hidden' });
    await expect(reportFlushVerdict(page, zeroTickReport())).resolves.toBeUndefined();
    expect(warn.mock.calls[0][0]).toContain('NOT VISIBLE');
    expect(warn.mock.calls[0][0]).toContain('document.visibilityState=hidden');
  });

  it('warns instead of throwing when the counter has RESET below the baseline', async () => {
    // A frame counter cannot count down: a value BELOW the baseline means the
    // renderer (or the whole debug interface) was re-created, so the baseline
    // belongs to a counter that no longer exists. The `>` guard alone sends
    // this to the throw, which then reports "has not moved past 100 (it now
    // reads 3)" as if the renderer were stuck.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const page = probingPage({ frame: 3, contextLost: false, visibility: 'visible' });
    await expect(reportFlushVerdict(page, zeroTickReport())).resolves.toBeUndefined();
    expect(warn.mock.calls[0][0]).toContain('counter has RESET');
  });

  it('calls context loss UNKNOWN, not false, when no WebGL context could be read', async () => {
    // `canvas.getContext('webgl2'/'webgl')` returns null on a canvas already in
    // `webgpu` context mode, so a null flag must not read as "not lost".
    const page = probingPage({ frame: 100, contextLost: null, visibility: 'visible' });
    const error = await reportFlushVerdict(page, zeroTickReport()).catch((e: unknown) => e);
    expect((error as Error).message).toContain('UNKNOWN');
  });

  it('names the aggregate budget when that is what ended the flush', async () => {
    const page = probingPage({ frame: 100, contextLost: false, visibility: 'visible' });
    const error = await reportFlushVerdict(page, zeroTickReport({ budgetExhausted: true })).catch(
      (e: unknown) => e
    );
    expect((error as Error).message).toContain(`${FLUSH_BUDGET_MS} ms flush budget`);
  });
});

describe('reportFlushVerdict in-page probe', () => {
  // The fakes above return the probe's answer directly, so they never run the
  // injected function. These execute it for real against a stubbed `window` /
  // `document` (the vitest default environment is `node`, so there is none
  // otherwise) — that is where the visibility read and the "no WebGL context"
  // rule actually live.
  const original = globalThis as { window?: unknown; document?: unknown };
  const originalWindow = original.window;
  const originalDocument = original.document;

  /** A `page` whose `evaluate` actually CALLS the injected function. */
  function executingPage(): Page {
    return { evaluate: async (fn: () => unknown) => fn() } as unknown as Page;
  }

  function stubPage(frame: unknown, canvas: unknown, visibilityState: string): void {
    original.window = { __luxarDebug: { renderer: { info: { render: { frame } } } } };
    original.document = { querySelector: () => canvas, visibilityState };
  }

  afterEach(() => {
    if (originalWindow === undefined) delete original.window;
    else original.window = originalWindow;
    if (originalDocument === undefined) delete original.document;
    else original.document = originalDocument;
  });

  it('reads visibilityState and calls a missing WebGL context UNKNOWN', async () => {
    // No canvas at all is the same shape as a canvas already in `webgpu`
    // context mode: getContext returns null, so loss is unknowable. `visible`
    // rather than `hidden` here on purpose — a hidden page is now a warning of
    // its own, so it would never reach the throw this asserts on.
    stubPage(100, null, 'visible');
    const error = await reportFlushVerdict(executingPage(), zeroTickReport()).catch(
      (e: unknown) => e
    );
    const message = (error as Error).message;
    expect(message).toContain('document.visibilityState=visible');
    expect(message).toContain('UNKNOWN');
  });

  it('routes a hidden page read from the real probe to the visibility warning', async () => {
    // The visibility read lives inside the injected function, so this is the
    // half the scripted fakes cannot cover: a real `document.visibilityState`
    // of `hidden` must warn rather than throw.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    stubPage(100, { getContext: () => ({ isContextLost: () => false }) }, 'hidden');
    await expect(reportFlushVerdict(executingPage(), zeroTickReport())).resolves.toBeUndefined();
    expect(warn.mock.calls[0][0]).toContain('NOT VISIBLE');
  });

  it('reads a live context as not lost, and a lost one as lost', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    stubPage(100, { getContext: () => ({ isContextLost: () => true }) }, 'visible');
    await expect(reportFlushVerdict(executingPage(), zeroTickReport())).resolves.toBeUndefined();
    expect(warn.mock.calls[0][0]).toContain('reports itself LOST');

    warn.mockClear();
    stubPage(100, { getContext: () => ({ isContextLost: () => false }) }, 'visible');
    const error = await reportFlushVerdict(executingPage(), zeroTickReport()).catch(
      (e: unknown) => e
    );
    expect((error as Error).message).toContain('the WebGL context is not reported lost');
  });
});

describe('render-ticks bounds arithmetic', () => {
  it('keeps the documented worst case inside the 60 s per-test budget', async () => {
    // Baseline probe + flush + verdict probe is the whole cost of one
    // flush/verdict pair, and it has to leave room for page.goto and
    // waitForLuxarReady inside `playwright.config.ts`'s 60 s `timeout`.
    expect(COUNTER_PROBE_TIMEOUT_MS * 2 + FLUSH_BUDGET_MS).toBeLessThanOrEqual(20000);
    // The counter probes are deliberately tighter than the 15 s the spec's two
    // DETECTOR probes get: a missed counter probe only downgrades a verdict to
    // a warning, whereas a missed detector probe loses the test's evidence.
    expect(COUNTER_PROBE_TIMEOUT_MS).toBeLessThan(15000);
    // And the aggregate budget has to dominate the per-tick bound, or it could
    // not bound a multi-tick flush at all.
    expect(FLUSH_BUDGET_MS).toBeGreaterThan(TICK_TIMEOUT_MS);
  });
});
