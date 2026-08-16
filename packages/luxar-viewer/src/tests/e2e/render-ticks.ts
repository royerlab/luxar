/**
 * Confirmed render-tick flushing for E2E detectors (#1651).
 *
 * A detector that only sees a bug on the Nth draw needs those draws to have
 * HAPPENED. `waitForNextRender(page, N)` in `helpers.ts` is documented as
 * best-effort and its fallback is silent, so a caller cannot tell a confirmed
 * flush from a fallback. {@link flushRenderTicks} kicks the renderer and
 * confirms each tick against the renderer's own frame counter, then
 * {@link reportFlushVerdict} decides — from evidence, not from the tick count —
 * whether a flush that confirmed NOTHING is worth failing the run over.
 *
 * This lives in its own module rather than in `helpers.ts` so it can be
 * unit-tested without a browser: the only page surface it touches is
 * `page.evaluate` and `page.waitForFunction`, both easy to fake (see
 * `src/tests/unit/tests/e2e-render-ticks.test.ts`).
 *
 * KNOWN LIMITATION (WebGPU backend). The counter probe reads
 * `info.render.frame` and falls back to `info.frame`. `WebGPURenderer` has no
 * `info.render.frame`, so on that backend the fallback is what answers — and
 * THREE assigns `info.frame` from its OWN internal animation loop, which
 * advances whether or not anything was drawn. A "confirmed" tick is therefore
 * weaker there than on WebGL (where `info.render.frame` counts actual
 * `renderer.render()` calls), and N ticks is roughly N animation frames rather
 * than the ~N/2 the WebGL two-pass pipeline gives. Not fixed here — stated so a
 * reader does not over-read a WebGPU flush report.
 *
 * @module tests/e2e/render-ticks
 */

import type { Page } from '@playwright/test';
import { raceEvaluate } from './helpers';

/**
 * Per-tick bound inside {@link flushRenderTicks} — it caps the kick evaluate
 * and the confirmation wait individually, and each is additionally clamped to
 * whatever is left of {@link FLUSH_BUDGET_MS}.
 *
 * This workstation software-renders at ~4.4 animation frames/s while a scene
 * loads (measured: 123 rAF callbacks and 246 `renderer.render()` calls over a
 * 28 s window, i.e. ~230 ms per frame and ~114 ms per tick), and that is not
 * the worst case — it was measured on an otherwise IDLE box with this spec
 * alone in flight, so a busier machine can only be slower. What starves the
 * CDP round trip this wait reports over is a backlog of already-queued
 * main-thread work (#1651); the `OPFS timeout: set(...) exceeded 10000ms`
 * writes arriving in the same window are the producer that measurably moved
 * the needle, and #1647 is what addresses them. 3 s is ~26x the measured
 * per-tick cost, which buys headroom for a slower box cheaply — and, unlike
 * before, buys it without multiplying: the aggregate budget below is what
 * bounds the flush.
 */
export const TICK_TIMEOUT_MS = 3000;

/**
 * DEFAULT aggregate bound on one whole {@link flushRenderTicks} call; a call
 * site that flushes several times inside ONE test passes a smaller `budgetMs`
 * (see the `budgetMs` parameter and the looped call site in
 * `webgl-errors.spec.ts`), because this bound is per CALL and per-call bounds
 * add up.
 *
 * WHY an aggregate bound and not just {@link TICK_TIMEOUT_MS}: the per-tick
 * bound is paid up to `ticks` times, twice per iteration (once for the kick,
 * once for the confirmation wait), and a kick that goes unanswered does not
 * end the loop. Without this budget `flushRenderTicks(page, 10)` could spend
 * ~10 x (3 s + 3 s) = 60 s — the entire per-test budget, reproducing exactly
 * the opaque `Test timeout of 60000ms exceeded` this change exists to
 * eliminate.
 *
 * With it, the worst case is bounded by construction: the flush stops as soon
 * as the budget is spent, and each individual wait is clamped to the remainder,
 * so one call costs at most {@link COUNTER_PROBE_TIMEOUT_MS} (the baseline
 * probe, taken before the budget starts) + {@link FLUSH_BUDGET_MS} = 15 s. A
 * {@link reportFlushVerdict} call afterwards adds at most another
 * {@link COUNTER_PROBE_TIMEOUT_MS}, for 20 s total across both — comfortably
 * inside a per-test budget of `playwright.config.ts`'s 60 s (120 s in
 * `webgl-errors.spec.ts`, which raises it at the describe level) alongside ONE
 * `page.goto` plus `waitForLuxarReady`. 10 s is ~10x the measured cost of the largest flush any
 * call site asks for (10 ticks at ~114 ms each is ~1.1 s). A test that loads
 * five scenes in one body cannot afford 20 s five times over, which is what
 * `budgetMs` is for.
 */
export const FLUSH_BUDGET_MS = 10000;

/**
 * Bound on the two one-shot counter probes (the flush's baseline read and
 * {@link reportFlushVerdict}'s follow-up).
 *
 * Both are trivial property reads that answer in milliseconds on a live page,
 * so this is not sized for the WORK — it is sized for the wait to be
 * SCHEDULED. Deliberately smaller than the 15 s the spec's two detector probes
 * get, because those probes' answers are what the tests assert on whereas a
 * missed probe here only downgrades a verdict to a warning. A larger bound
 * would not rescue the case it might seem to: main-thread starvation was
 * measured to outlast any bound that fits in the test budget (a trivial
 * evaluate unanswered for 5 s, twelve times in a row, ~78 s in all, while the
 * page went on rendering), so past a few seconds the extra latency buys no
 * extra answers.
 */
export const COUNTER_PROBE_TIMEOUT_MS = 5000;

/** Outcome of one best-effort `renderOnce()` kick. */
export type KickOutcome = 'answered' | 'rejected' | 'no-answer';

/** What a {@link flushRenderTicks} call actually achieved. */
export interface FlushReport {
  /** Ticks asked for. */
  requested: number;
  /** Ticks confirmed against the renderer's own counter. */
  landed: number;
  /**
   * The counter value the flush started from, or `null` when that probe went
   * unanswered — in which case no baseline exists and a zero `landed` cannot
   * be judged either way ({@link reportFlushVerdict} says so and moves on).
   */
  startFrame: number | null;
  /** How many kicks the page answered / rejected / never answered. */
  kicks: Record<KickOutcome, number>;
  /** How many confirmation waits actually ran (≤ `requested`). */
  waits: number;
  /** Wall-clock the whole call spent, baseline probe included. */
  elapsedMs: number;
  /**
   * The aggregate budget this call was given, in ms — {@link FLUSH_BUDGET_MS}
   * unless the call site passed its own. Carried so a message about the budget
   * quotes the bound that actually applied.
   */
  budgetMs: number;
  /**
   * True when the aggregate budget is what ended the loop — either because
   * nothing was left to dispatch the next kick/wait with, or because the wait
   * that timed out had been CLAMPED down to the budget remainder.
   */
  budgetExhausted: boolean;
  /**
   * The clamped bound, in ms, of a confirmation wait that timed out because the
   * budget remainder — not {@link TICK_TIMEOUT_MS} — was what bounded it;
   * `null` when no wait ended that way. Reported so a shortfall message cannot
   * claim every wait got the full per-tick bound.
   */
  budgetClampedWaitMs: number | null;
}

/**
 * Drive `ticks` render ticks, confirming each one against the renderer's own
 * counter.
 *
 * WHAT A "TICK" IS: on WebGL `info.render.frame` counts `renderer.render()`
 * calls, and Luxar's default pipeline issues two of them per animation frame
 * (scene→HDR plus the fullscreen mega pass), more with bloom/FXAA on. So N
 * ticks is roughly N/2 animation frames. (On WebGPU it is ~N animation frames
 * and a weaker signal — see the module docblock.) The N values at the call
 * sites are unchanged from the `waitForNextRender(page, N)` calls they replace,
 * which targeted the same counter — what is requested is the same; only the
 * confirmation is new.
 *
 * WHY the caller confirms its own ticks instead of calling
 * `waitForNextRender(page, N)`: that helper is documented as best-effort and
 * its fallback is SILENT, so a detector that needs its draws to have happened
 * cannot tell a confirmed flush from a fallback. A GL error that only appears
 * on the Nth draw would then go unlogged and the detector would read stronger
 * than it is.
 *
 * WHY this never throws for a short flush: a strict variant is reported in
 * #1652 to turn the healthy case red — a tick can legitimately miss even a
 * generous bound on a loaded box. One measured reason for a shortfall is that
 * the OBSERVATION is starved rather than the rendering: `page.waitForFunction`
 * polls in-page and reports back over the same CDP channel a saturated rAF
 * loop starves, so the counter can advance while no tick is ever confirmed
 * (#1651). A shortfall is therefore reported (so a weak detector run is
 * visible in the output) and the assertions that follow still operate on
 * whatever landed; {@link reportFlushVerdict} decides what a ZERO-tick flush
 * means, and it is called AFTER the detector's own reporting. This function
 * throws only on a real contract break — a `ticks` that is not a positive
 * integer (a caller bug: below 1 it would drive no iterations and then be
 * reported as "none of the 0 requested ticks were confirmed" on a perfectly
 * healthy page, and a fraction is not a tick count at all), the page
 * ANSWERS but exposes no numeric frame counter so there is nothing to confirm
 * against, or a non-timeout error out of `page.waitForFunction`, which is a
 * crashed/closed page and must surface as itself. An UNANSWERED counter probe
 * is the starvation case instead: it returns a report with `startFrame: null`.
 *
 * Every wait here is bounded twice: individually by {@link TICK_TIMEOUT_MS}
 * (`page.waitForFunction` honours its `timeout` unlike `page.evaluate`, whose
 * evaluates go through `raceEvaluate`), and in aggregate by `budgetMs`.
 *
 * @param page - Playwright page
 * @param ticks - How many render ticks to drive; must be >= 1
 * @param budgetMs - Aggregate bound on this ONE call, defaulting to
 *   {@link FLUSH_BUDGET_MS}. Pass less where several flushes share one test
 *   budget: the bound is per call, so per-call bounds add up.
 * @returns What the flush achieved; feed it to {@link reportFlushVerdict}.
 */
export async function flushRenderTicks(
  page: Page,
  ticks: number,
  budgetMs: number = FLUSH_BUDGET_MS
): Promise<FlushReport> {
  if (!Number.isInteger(ticks) || ticks < 1) {
    throw new Error(
      `flushRenderTicks: ticks must be a positive integer, got ${ticks}. A count below 1 drives no ` +
        'iterations and would then be reported as a zero-tick flush on a page that is perfectly ' +
        'healthy; a fractional count is not a number of ticks at all — the loop would run ' +
        'ceil(ticks) times and the shortfall report would compare what landed against a fractional ' +
        'target.'
    );
  }

  const startedAt = Date.now();

  // Read the counter once. WebGL reports it as info.render.frame, WebGPU as
  // info.frame — probe both, exactly as the shared render helpers do. The
  // answer is WRAPPED so that `null` can mean only "the deadline won": a bare
  // `number | null` would make an unanswered probe and a renderer without a
  // numeric counter indistinguishable, which is exactly the sentinel contract
  // `raceEvaluate` documents.
  const counter = await raceEvaluate<{ frame: number | null } | null>(
    page.evaluate(() => {
      const info = (window as any).__luxarDebug?.renderer?.info;
      const frame = info?.render?.frame ?? info?.frame;
      return { frame: typeof frame === 'number' ? frame : null };
    }),
    COUNTER_PROBE_TIMEOUT_MS,
    null
  );

  if (counter === null) {
    // NOT a throw: an unanswered probe is the starvation case, not a dead
    // page. Measured (#1651) — a trivial evaluate went unanswered 5 s twelve
    // times running while rAF kept ticking and the render counter advanced
    // 426 → 672, so throwing here would redden a page that was drawing the
    // whole time. Report it as unread and let the verdict helper say so.
    console.warn(
      '[⚠️] [flushRenderTicks] the frame-counter probe went unanswered for ' +
        `${COUNTER_PROBE_TIMEOUT_MS} ms, so no baseline was read and the tick loop is skipped — a ` +
        'saturated rAF loop starves the evaluate round trip for tens of seconds while rendering ' +
        'continues. The detector that follows still runs, against however many draws happen ' +
        'meanwhile.'
    );
    return {
      requested: ticks,
      landed: 0,
      startFrame: null,
      kicks: { answered: 0, rejected: 0, 'no-answer': 0 },
      waits: 0,
      elapsedMs: Date.now() - startedAt,
      budgetMs,
      budgetExhausted: false,
      budgetClampedWaitMs: null,
    };
  }
  if (counter.frame === null) {
    throw new Error(
      'flushRenderTicks: the page answered but the renderer exposes no numeric frame counter ' +
        '(neither info.render.frame nor info.frame), so ticks cannot be confirmed.'
    );
  }
  const startFrame = counter.frame;

  const kicks: Record<KickOutcome, number> = { answered: 0, rejected: 0, 'no-answer': 0 };
  let landed = 0;
  let waits = 0;
  let budgetExhausted = false;
  let budgetClampedWaitMs: number | null = null;

  // Aggregate deadline. Started AFTER the baseline probe so the budget covers
  // the loop only, and the worst-case total stays the simple sum documented on
  // FLUSH_BUDGET_MS — with `budgetMs` standing in for the default when a call
  // site passes its own.
  const deadline = Date.now() + budgetMs;

  for (let i = 0; i < ticks; i++) {
    const beforeKick = deadline - Date.now();
    if (beforeKick <= 0) {
      budgetExhausted = true;
      break;
    }

    // Kick the loop. The FIRST kick starts a continuous rAF loop
    // (`startAnimation()` is gated on `!isAnimating`); every later one only
    // re-arms the controller's ~2 s idle timeout, which is what keeps the loop
    // alive for the whole flush. Outcomes are counted rather than swallowed so
    // the diagnostics can say how many kicks the page actually answered.
    const outcome = await raceEvaluate<KickOutcome>(
      page
        .evaluate(() => {
          (window as any).__luxarDebug?.renderOnce?.();
        })
        .then(
          (): KickOutcome => 'answered',
          (): KickOutcome => 'rejected'
        ),
      Math.min(TICK_TIMEOUT_MS, beforeKick),
      'no-answer'
    );
    kicks[outcome] += 1;

    const beforeWait = deadline - Date.now();
    if (beforeWait <= 0) {
      budgetExhausted = true;
      break;
    }

    const waitBound = Math.min(TICK_TIMEOUT_MS, beforeWait);
    waits++;
    try {
      await page.waitForFunction(
        (target) => {
          const info = (window as any).__luxarDebug?.renderer?.info;
          return (info?.render?.frame ?? info?.frame ?? 0) >= target;
        },
        startFrame + i + 1,
        { timeout: waitBound }
      );
      landed++;
    } catch (err) {
      // Only a genuine wait timeout means "the renderer is not keeping up".
      // A `Target crashed` / `Target ... has been closed` must surface as
      // itself rather than be re-reported downstream as a tick shortfall.
      const message = err instanceof Error ? err.message : String(err);
      if (!/Timeout .*exceeded/.test(message)) throw err;
      if (waitBound < TICK_TIMEOUT_MS) {
        // The bound that expired was the budget REMAINDER, not the per-tick
        // bound, so the aggregate budget is what ended this loop — record it,
        // or the shortfall message would claim every wait got the full
        // TICK_TIMEOUT_MS and never mention the budget at all.
        budgetExhausted = true;
        budgetClampedWaitMs = waitBound;
      }
      // Stop at the first tick that does not land: the remaining kicks would
      // only queue behind it.
      break;
    }
  }

  const report: FlushReport = {
    requested: ticks,
    landed,
    startFrame,
    kicks,
    waits,
    elapsedMs: Date.now() - startedAt,
    budgetMs,
    budgetExhausted,
    budgetClampedWaitMs,
  };

  if (landed < ticks) {
    console.warn(
      `[⚠️] [flushRenderTicks] only ${landed}/${ticks} render ticks were OBSERVED to land ` +
        `(${waits} confirmation wait(s) ran, each bounded by ${TICK_TIMEOUT_MS} ms and by the ` +
        `${budgetMs} ms flush budget; ${report.elapsedMs} ms spent in all` +
        `${budgetExhausted ? ', and the loop STOPPED because that budget ran out' : ''}` +
        `${
          budgetClampedWaitMs !== null
            ? ` — the last confirmation wait was clamped to ${budgetClampedWaitMs} ms by what was ` +
              'left of it, and timed out'
            : ''
        }; counter ` +
        `started at ${startFrame}; kicks answered=${kicks.answered} rejected=${kicks.rejected} ` +
        `no-answer=${kicks['no-answer']}) — the check that follows ran against a shorter CONFIRMED ` +
        'flush than it asked for. The counter may well have advanced further: the confirmation ' +
        'polls in-page and reports back over the same channel a saturated rAF loop starves (#1651).'
    );
  }

  return report;
}

/**
 * Decide what a flush that confirmed NOTHING means, and fail only when the
 * evidence says the renderer really is not drawing. Call this AFTER the
 * detector's own dump and `expect`, so a real GL error is the error you see.
 *
 * WHY a verdict rather than a plain assertion: zero CONFIRMED ticks does not
 * imply a frozen counter. Measured on an idle box (#1651), during the stall
 * the counter kept advancing (426 → 672) while a trivial `page.evaluate` went
 * unanswered for 5 s twelve times running — it is the OBSERVATION that
 * starves, not the rendering. So the zero case takes ONE bounded probe (the
 * counter, in the same wrapped shape {@link flushRenderTicks} uses so `null`
 * still means only "the deadline won", plus the context-lost flag and
 * `document.visibilityState`, in a single round trip) and fails only when ALL
 * of these hold: the probe was answered, the context is not reported lost, the
 * flush had a baseline, the counter still reads exactly that baseline (neither
 * advanced past it nor RESET below it), the page reports itself `visible`, and
 * at least one kick was ANSWERED. That last one matters because the animation
 * loop auto-pauses after ~2 s of idleness: if every kick went unanswered,
 * nothing was confirmed to have asked the renderer to draw, so a frozen counter
 * is fully explained without a renderer defect. Every other case is one
 * warning: a starved channel, a lost context, a counter that restarted from a
 * re-created renderer, or a non-visible page (which gets no rAF callbacks at
 * all) is the environment, and the detector above keeps its evidence either way
 * because it listens on `page.on('console')`, which goes on delivering
 * throughout.
 */
export async function reportFlushVerdict(page: Page, report: FlushReport): Promise<void> {
  if (report.landed > 0) return;

  const probe = await raceEvaluate<{
    frame: number | null;
    contextLost: boolean | null;
    visibility: string | null;
  } | null>(
    page.evaluate(() => {
      const info = (window as any).__luxarDebug?.renderer?.info;
      const frame = info?.render?.frame ?? info?.frame;
      const canvas = document.querySelector('canvas') as HTMLCanvasElement | null;
      // Returns null on a canvas already in `webgpu` context mode, so a null
      // here means UNKNOWN rather than "not lost".
      const gl = canvas ? canvas.getContext('webgl2') || canvas.getContext('webgl') : null;
      return {
        frame: typeof frame === 'number' ? frame : null,
        contextLost: gl ? gl.isContextLost() : null,
        visibility: typeof document.visibilityState === 'string' ? document.visibilityState : null,
      };
    }),
    COUNTER_PROBE_TIMEOUT_MS,
    null
  );

  // Says what the flush actually DID rather than implying every requested tick
  // was waited on for the full per-tick bound: the loop breaks at the first
  // miss, and the aggregate budget can end it earlier still.
  const context =
    `none of the ${report.requested} requested render ticks were confirmed (${report.waits} ` +
    `confirmation wait(s) ran; ${report.elapsedMs} ms spent` +
    `${report.budgetExhausted ? `, loop stopped by the ${report.budgetMs} ms flush budget` : ''}` +
    `${
      report.budgetClampedWaitMs !== null
        ? ', whose remainder clamped the last confirmation wait to ' +
          `${report.budgetClampedWaitMs} ms before it timed out`
        : ''
    }; ` +
    `kicks answered=${report.kicks.answered} rejected=${report.kicks.rejected} no-answer=` +
    `${report.kicks['no-answer']})`;

  if (probe === null) {
    console.warn(
      `[⚠️] [reportFlushVerdict] ${context}, and the follow-up counter probe went unanswered for ` +
        `${COUNTER_PROBE_TIMEOUT_MS} ms too — the observation channel itself is starved, so we ` +
        'cannot say whether the renderer drew. Measured mechanism (#1651): a saturated ' +
        'software-rendering rAF loop starves in-page timers and the CDP evaluate round trip for ' +
        'tens of seconds while rendering continues. Not failing on that: the detector above ' +
        "listens on page.on('console'), and console events keep flowing throughout, so its " +
        'assertion still had its evidence.'
    );
    return;
  }

  const visibility = `document.visibilityState=${probe.visibility ?? 'unreadable'}`;

  if (probe.contextLost === true) {
    console.warn(
      `[⚠️] [reportFlushVerdict] ${context}, and the WebGL context reports itself LOST (${visibility}) ` +
        '— THREE skips renderer.render() entirely while the context is lost, so the counter freezes ' +
        'by construction. fixtures.ts allow-lists /WebGL context lost/ as environmental, so this is ' +
        'the environment rather than a viewer bug.'
    );
    return;
  }

  if (report.startFrame === null) {
    console.warn(
      `[⚠️] [reportFlushVerdict] ${context}, and the flush never read a baseline counter value (its ` +
        'own probe went unanswered — the same starvation, #1651), so a zero confirmed-tick count ' +
        `cannot be judged either way. The counter now reads ${probe.frame ?? 'no numeric value'} ` +
        `(${visibility}).`
    );
    return;
  }

  if (probe.frame === null) {
    // The flush read a baseline, so a numeric counter existed then and is gone
    // now (a navigated or re-created debug interface). Nothing to compare
    // against, so no verdict — and specifically not a failure.
    console.warn(
      `[⚠️] [reportFlushVerdict] ${context}, and the page now exposes no numeric frame counter ` +
        `(neither info.render.frame nor info.frame) although the flush read ${report.startFrame} ` +
        `from one, so there is nothing left to compare against (${visibility}).`
    );
    return;
  }

  if (probe.frame > report.startFrame) {
    console.warn(
      `[⚠️] [reportFlushVerdict] ${context}, but the counter advanced ${report.startFrame} → ` +
        `${probe.frame} (+${probe.frame - report.startFrame}) — the renderer DID draw; only our ` +
        'per-tick confirmation was starved, because page.waitForFunction polls in-page and reports ' +
        `back over the same channel a saturated rAF loop starves (#1651). (${visibility})`
    );
    return;
  }

  if (probe.frame < report.startFrame) {
    // The counter came back BELOW the baseline, so it is not the same counter:
    // a re-created renderer or a navigated debug interface restarts it from 0.
    // Same reasoning as the "counter is gone" branch above — the baseline
    // belongs to a renderer that no longer exists, so there is nothing to
    // compare against, and specifically no failure to declare.
    console.warn(
      `[⚠️] [reportFlushVerdict] ${context}, and the counter has RESET: it read ` +
        `${report.startFrame} at the start of the flush and now reads ${probe.frame}, which a ` +
        'monotonic frame counter cannot do — the renderer or the debug interface was re-created ' +
        `(a reload or a navigation) since the baseline, so the two are not comparable (${visibility}).`
    );
    return;
  }

  if (probe.visibility !== null && probe.visibility !== 'visible') {
    // A non-visible page gets no rAF callbacks at all, so a frozen counter is
    // fully explained by the environment — exactly the "this is not a renderer
    // defect" case the docblock promises a warning for.
    console.warn(
      `[⚠️] [reportFlushVerdict] ${context}, and the page is NOT VISIBLE (${visibility}) — the ` +
        'browser stops delivering requestAnimationFrame callbacks to a hidden page, so a frozen ' +
        'counter needs no renderer defect to explain it. Environment, not a failure.'
    );
    return;
  }

  if (report.kicks.answered === 0) {
    console.warn(
      `[⚠️] [reportFlushVerdict] ${context}, and NOT ONE kick was answered, so nothing was ` +
        'confirmed to have asked the renderer to draw. The animation loop auto-pauses after ~2 s ' +
        'of idleness, which explains a frozen counter without any renderer defect, so this is not ' +
        `a failure. (${visibility})`
    );
    return;
  }

  const contextNote =
    probe.contextLost === false
      ? 'the WebGL context is not reported lost'
      : 'no WebGL context could be read, so context loss is UNKNOWN here — canvas.getContext' +
        "('webgl2'/'webgl') returns null on a canvas already in `webgpu` context mode";

  // Only 'visible' is a positive reading; an unreadable value is neither a
  // reason to warn (nothing was observed) nor something to claim as visible.
  const visibilityNote =
    probe.visibility === 'visible'
      ? `the page is visible (${visibility})`
      : `page visibility could not be read (${visibility})`;

  throw new Error(
    `reportFlushVerdict: ${context}; the page ANSWERED this probe within ` +
      `${COUNTER_PROBE_TIMEOUT_MS} ms, ${contextNote}, ${visibilityNote}, at ` +
      'least one kick was answered, and the counter still reads exactly the baseline the flush ' +
      `started from, ${report.startFrame} (it now reads ${probe.frame}, neither advanced nor ` +
      'reset). That is a renderer which demonstrably is not drawing while the page is ' +
      'responsive, so the detector above inspected no draws. See issue #1651.'
  );
}
