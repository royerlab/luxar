/**
 * Frame-pacing regression test (#1724)
 *
 * `performance_benchmark_example.luxar.zarr` (100 nodes / 100k points / 8 MB)
 * used to render a healthy ~59 fps for ~30 s and then wedge: back-to-back
 * ~1 s main-thread tasks left no slot for anything else, so the depth-sort
 * worker's replies were delivered at ~0.5/s, each landed reply staged an
 * ordering apply that called `requestRender()`, and the loop could never
 * idle. The rendering starved the very hand-off that would have let it stop.
 * Three things are asserted here, in order:
 *
 * 0. the scene actually LOADED its content — a load whose node chunks fail
 *    leaves `initialized: true`, an empty scene and an immediately-idle loop,
 *    which would make everything below pass on a broken build;
 * 1. the loop SETTLES (`isAnimating === false`) and STAYS settled — the
 *    outstanding replies drain, nothing re-arms the loop, and the idle pause
 *    fires. The hold matters because `isAnimating` is transiently false
 *    during the load too, and the wedge only appeared ~30 s in;
 * 2. the CDP control channel stays RESPONSIVE — a trivial `page.evaluate`
 *    answers in milliseconds instead of timing out (`getState()` costs
 *    0.5 ms in-page but measured 111 s across the bridge in the wedge).
 *
 * Neither this spec nor the un-parked `performance_benchmark_example` case in
 * `all-examples-smoke-test.spec.ts` is in the smoke subset
 * (`pnpm test:e2e:smoke`), so both run only in the full E2E suite.
 */

import type { Page } from '@playwright/test';

import { test, expect, ALLOW_CONSOLE_ERRORS } from './fixtures';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  getLuxarState,
  raceEvaluate,
  captureConsoleMessages,
} from './helpers';

const EXAMPLES_BASE = 'http://localhost:9000/datasets/examples';
const DATASET = 'performance_benchmark_example.luxar.zarr';

/**
 * Content the generator writes (`packages/luxar/examples/
 * performance_benchmark_example.py`: `num_nodes = 100`,
 * `points_per_node = 1000`). Asserted with `>=` so growing the example does
 * not fail the spec, while an empty or half-loaded scene does.
 */
const EXPECTED_POINTS = 100000;
const EXPECTED_NODES = 100;

/**
 * Budget for the navigation itself. Bounded explicitly rather than left to the
 * config's 60 s `navigationTimeout`, so it appears in the budget sum below as
 * a number this spec controls.
 */
const NAV_TIMEOUT_MS = 20000;

/** Budget for `initialized` to flip, matching the smoke test for this fixture. */
const READY_TIMEOUT_MS = 60000;

/** Failure-only deadline for recovering viewer state after readiness times out. */
const READINESS_STATE_PROBE_TIMEOUT_MS = 5000;

/** Keep readiness failures readable even if a broken load floods the console. */
const READINESS_CONSOLE_ERROR_LIMIT = 10;

/**
 * Budget for the 100 nodes' points to be committed. `waitForPointsLoaded`
 * checks this budget BEFORE each iteration and does not clamp its own state
 * probe (which keeps `getLuxarState`'s 45 s default), so the worst case is
 * this value plus one full 45 s probe — see the budget sum below.
 */
const CONTENT_TIMEOUT_MS = 45000;

/**
 * Budget for the loop to drain its work and idle-pause. With pacing in place
 * the measured settle is ~15 s.
 */
const SETTLE_TIMEOUT_MS = 40000;

/** How long the settled state must HOLD before it counts as settled. */
const SETTLE_HOLD_MS = 5000;

/**
 * Per-call bound on a state probe here — the content re-read, every poll of
 * the settle loop, and the post-hold re-read. Well under `getLuxarState`'s own
 * 45 s default so the settle poll gets several attempts inside its budget
 * instead of one. `expect.poll` does not abort a callback that is already
 * running, so the settle leg's worst case is its own budget plus one of these.
 */
const STATE_PROBE_TIMEOUT_MS = 10000;

/** Number of trivial round trips timed after settling. */
const PROBE_COUNT = 10;

/**
 * Per-probe bound. `page.evaluate` is NOT covered by `actionTimeout`, so an
 * unbounded probe on a wedged bridge consumes the whole test budget and the
 * run dies as a bare suite timeout — precisely the case this assertion
 * exists for. Comfortably above {@link MAX_ROUND_TRIP_MS} so a merely slow
 * answer is still MEASURED (and fails the assertion with a real number)
 * rather than being reported as a non-answer.
 */
const PROBE_TIMEOUT_MS = 2000;

/**
 * Ceiling for the worst round trip. Healthy is 1–5 ms; the wedge produced
 * >15,000 ms. 500 ms is far under a second while leaving room for a busy
 * CI box.
 */
const MAX_ROUND_TRIP_MS = 500;

function describeError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

async function waitForReadyWithDiagnostics(
  page: Page,
  timeout: number,
  consoleErrors: string[]
): Promise<void> {
  try {
    await waitForLuxarReady(page, timeout);
  } catch (readinessError) {
    let stateDiagnostic: string;
    try {
      stateDiagnostic = JSON.stringify(await getLuxarState(page, READINESS_STATE_PROBE_TIMEOUT_MS));
    } catch (stateError) {
      stateDiagnostic = `unavailable (${describeError(stateError)})`;
    }

    const consoleErrorTail = consoleErrors.slice(-READINESS_CONSOLE_ERROR_LIMIT);
    const consoleDiagnostic =
      consoleErrorTail.length > 0 ? consoleErrorTail.join('\n') : '(none captured)';
    const originalError = describeError(readinessError);

    throw new Error(
      `Viewer readiness failed: ${originalError}\n` +
        `Viewer state: ${stateDiagnostic}\n` +
        `Recent console errors (${consoleErrorTail.length}/${consoleErrors.length}):\n` +
        consoleDiagnostic,
      readinessError instanceof Error ? { cause: readinessError } : undefined
    );
  }
}

test.describe('Frame pacing (#1724)', () => {
  // Every leg is bounded, and the bounds have to FIT inside the test timeout —
  // otherwise a genuine regression dies as a bare suite timeout instead of the
  // named assertion that exists to explain it. Worst case, leg by leg:
  //
  //   navigation                                        20 s
  //   waitForLuxarReady                                 60 s
  //   waitForPointsLoaded  45 s budget + 45 s unclamped
  //                        in-flight probe              90 s
  //   content re-read (getLuxarState, bounded)          10 s
  //   settle poll          40 s budget + 10 s callback
  //                        expect.poll does not abort   50 s
  //   settle hold                                        5 s
  //   post-hold re-read                                 10 s
  //   10 probes × 2 s                                   20 s
  //                                                   ------
  //                                                    265 s
  //
  // 300 s leaves ~35 s for the fixture setup, teardown and the per-step
  // overhead between the legs. This is what a FAILING run costs; a healthy one
  // pays only the hold plus the real settle (~15 s measured). The 90 s content
  // leg is the one number this spec cannot shrink from here:
  // `waitForPointsLoaded` takes no probe timeout and deliberately leaves its
  // own probe at `getLuxarState`'s 45 s default.
  test.describe.configure({ timeout: 300000 });

  test('the 100-node benchmark scene settles and leaves the control channel responsive', async ({
    page,
  }) => {
    const consoleMessages = captureConsoleMessages(page);

    // `&no-opfs` for the same reason as the smoke spec: nothing here asserts
    // the L2 OPFS tier, and automated Chromium's OPFS stalls systemically
    // (10 s per op — issue #1645), starving scene readiness past the budget.
    await page.goto(`/?src=${EXAMPLES_BASE}/${DATASET}&debug&no-opfs`, {
      timeout: NAV_TIMEOUT_MS,
    });
    await waitForReadyWithDiagnostics(page, READY_TIMEOUT_MS, consoleMessages.errors);

    // The scene must actually have its content. `initialized` flips before
    // the node chunks land, so a build whose chunk fetches all fail gives an
    // empty scene whose loop idles instantly — green on a broken viewer.
    await waitForPointsLoaded(page, EXPECTED_POINTS, CONTENT_TIMEOUT_MS);
    const loaded = await getLuxarState(page, STATE_PROBE_TIMEOUT_MS);
    expect(loaded.totalPoints).toBeGreaterThanOrEqual(EXPECTED_POINTS);
    expect(loaded.pointClouds.length).toBeGreaterThanOrEqual(EXPECTED_NODES);

    // `expect.poll` does not retry a callback that REJECTS in this Playwright
    // version (the callback is awaited outside the try/catch that handles a
    // failed matcher), and `getLuxarState` throws on its own deadline — so a
    // bare call would abort the poll with the helper's error and this spec's
    // `message` could never surface. Returning a sentinel keeps the poll
    // alive across a transient probe failure.
    const readIsAnimating = async (): Promise<boolean | 'probe-failed'> => {
      try {
        return (await getLuxarState(page, STATE_PROBE_TIMEOUT_MS)).isAnimating === true;
      } catch {
        return 'probe-failed';
      }
    };

    // The loop must come to rest. While the livelock held, every worker
    // reply that landed re-armed it, so `isAnimating` stayed true forever.
    await expect
      .poll(readIsAnimating, {
        timeout: SETTLE_TIMEOUT_MS,
        message: 'the animation loop never idled — the frame-pacing livelock is back',
      })
      .toBe(false);

    // …and STAY at rest. `isAnimating` is transiently false during the load
    // as well, and `expect.poll` latches on the first `false`, whereas the
    // pre-fix behaviour was ~59 fps for ~30 s and only THEN the collapse. A
    // loop that idles and then re-arms itself forever is the livelock.
    await page.waitForTimeout(SETTLE_HOLD_MS);
    expect(
      await readIsAnimating(),
      `the loop idled and then re-armed itself within ${SETTLE_HOLD_MS} ms — ` +
        'the frame-pacing livelock is back'
    ).toBe(false);

    // With the main thread free again, the bridge answers immediately. Each
    // probe is bounded independently (see PROBE_TIMEOUT_MS) and a
    // non-answer is RECORDED as its bound, so the assertion below fires
    // instead of the test dying as a suite timeout.
    const roundTripsMs: number[] = [];
    for (let i = 0; i < PROBE_COUNT; i++) {
      const start = Date.now();
      const answer = await raceEvaluate<number | null>(
        page.evaluate(() => 1 + 1),
        PROBE_TIMEOUT_MS,
        null
      );
      roundTripsMs.push(answer === null ? PROBE_TIMEOUT_MS : Date.now() - start);
      if (answer !== null) expect(answer).toBe(2);
    }

    const worstMs = Math.max(...roundTripsMs);
    console.log(`[frame-pacing] round trips (ms): ${roundTripsMs.join(', ')}`);
    expect(
      worstMs,
      `the slowest of ${PROBE_COUNT} trivial round trips took ${worstMs} ms — the main ` +
        'thread is starving the CDP control channel (a non-answer is recorded as its ' +
        `${PROBE_TIMEOUT_MS} ms bound)`
    ).toBeLessThan(MAX_ROUND_TRIP_MS);
  });

  test('readiness failures report the original error, console tail, and missing state', async ({
    page,
  }) => {
    test.info().annotations.push({
      type: ALLOW_CONSOLE_ERRORS,
      description: 'The test emits one sentinel console error to verify readiness diagnostics.',
    });

    const consoleMessages = captureConsoleMessages(page);
    await page.setContent('<!doctype html><title>not ready</title>');
    await page.evaluate(() => console.error('readiness diagnostic sentinel'));

    let failure: unknown;
    try {
      await waitForReadyWithDiagnostics(page, 100, consoleMessages.errors);
    } catch (error) {
      failure = error;
    }

    expect(failure).toBeInstanceOf(Error);
    const diagnosticError = failure as Error;
    expect(diagnosticError.message).toContain(
      'TimeoutError: page.waitForFunction: Timeout 100ms exceeded'
    );
    expect(diagnosticError.message).toContain('readiness diagnostic sentinel');
    expect(diagnosticError.message).toContain(
      'Debug interface not ready: getState() not available'
    );
    expect(diagnosticError.cause).toBeInstanceOf(Error);
  });
});
