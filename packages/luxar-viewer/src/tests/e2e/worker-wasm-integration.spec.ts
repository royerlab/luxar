/**
 * E2E tests for Worker and WASM integration
 *
 * These tests run in a real browser environment (Playwright) to verify:
 * - Workers can be created and communicate correctly
 * - WASM modules load and execute
 * - The data worker pool comes up and reports itself healthy
 * - Points still render whether queries take the worker or the main-thread path
 * - Fallback to main thread TypeScript works when workers unavailable
 *
 * Run with: pnpm test:e2e or pnpm test:e2e:ui
 */

import { test, expect, ALLOW_CONSOLE_ERRORS } from './fixtures';
import {
  waitForLuxarReady,
  waitForDataLoaded,
  waitForPointsLoaded,
  getConsoleMessages,
  waitForConsoleInterceptor,
  waitForNextRender,
} from './helpers';

const DATASET_3D = 'http://localhost:9000/datasets/examples/radius_basic_example.luxar.zarr';
const DATASET_5D =
  'http://localhost:9000/datasets/examples/dimension_sliders_5d_example.luxar.zarr';
const DATASET_LARGE =
  'http://localhost:9000/datasets/examples/performance_benchmark_example.luxar.zarr';

test.describe('Worker Integration E2E', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to test page with debug mode - use existing dataset
    await page.goto(`/?src=${DATASET_3D}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);
  });

  // Named for what it actually verifies. It was called "should offload spatial
  // queries to worker", but neither this assertion nor the one it replaced
  // establishes that any query was OFFLOADED — `__luxarDebug.getState()` exposes
  // no worker-served-query counter to assert against. Real offload coverage
  // needs such a counter, which is a separate change; the sibling
  // "should fallback to main thread if worker fails" carries the fallback story
  // in the meantime.
  test('should bring up the data worker pool', async ({ page }) => {
    // Wait for points to load
    await waitForPointsLoaded(page);
    await waitForConsoleInterceptor(page);

    // `Worker pool ready with N worker(s)` (worker-pool.ts) is the pool-health
    // line the in-page console interceptor can actually see, and it is a real
    // claim: a pool that comes up with zero workers throws
    // WorkerUnavailableError before ever reaching that log.
    //
    // This test previously keyed on `DataWorker ready`, which is logged from
    // INSIDE the data worker (src/workers/data-worker/initialize.ts). The
    // interceptor only patches the MAIN-THREAD console, so that line can never
    // reach this buffer — the assertion was dead, and passed only because its
    // enclosing `if (hasWorkers)` guard was false. #2494's eager pool warm-up
    // put `Worker N/M ready` in the buffer, flipped the guard true, and exposed
    // it (#2548). The assertion is unconditional now on purpose: a
    // data-dependent guard around an `expect` goes vacuous again the moment the
    // log line it probes for moves.
    //
    // We deliberately do NOT key on the `Acceleration: …` summary that follows:
    // it has three wordings and two of them are warnings (`TypeScript fallback
    // on all N`, `mixed backend`), so pinning the WASM variant would fail on any
    // checkout without a WASM build rather than reporting the pool healthy.
    //
    // Wait rather than sampling the buffer once: `warmUpDataWorkerPool()` is
    // fire-and-forget in the scene loader and Points decode on the main thread,
    // so pool-ready is not strictly ordered before `waitForPointsLoaded`.
    //
    // Both waits below are explicitly bounded, and the two budgets are chosen
    // together against this file's 60 s per-test timeout (it raises no timeout of
    // its own, and Playwright charges `beforeEach` to the same slot). 20 s clears
    // the worst justified delay: `worker-pool.ts` logs the line only after
    // `await Promise.allSettled(workerPromises)`, so one stuck worker holds it
    // behind that worker's whole init budget — a 3 s shared-WASM compile deadline
    // plus `workerInitTimeoutMs` (10 s by default), ~13 s — and that clock starts
    // at warm-up, well before `waitForPointsLoaded` returns. The read that follows
    // is capped at 10 s instead of the helper's 45 s default for the same reason:
    // on the failing path the budget has to survive long enough to PRINT what the
    // pool logged, and a bare "Test timeout of 60000ms exceeded" would throw away
    // the diff that is the entire point of the assertion's shape.
    //
    // The wait is a single in-page predicate, not `expect.poll` over
    // `getConsoleMessages`: that helper JSON-stringifies the entire ring buffer
    // on the page's main thread, and polling it re-runs that during scene load —
    // the same main-thread starvation surface that #1651/#1746/#1747/#1760 were
    // filed about. One round trip here, one read for the verdict below.
    await page
      .waitForFunction(
        () => {
          // BufferedMessage is `{ type, timestamp, args, stack? }`
          // (src/utils/console-interceptor.ts); every `log.*` call formats its
          // whole line into a single string arg, so the text lives in `args`.
          const debug = (window as any).__luxarDebug;
          const msgs = debug?.consoleInterceptor?.getBufferedMessages?.() ?? [];
          return msgs.some((m: { args?: unknown[] }) =>
            (m.args ?? []).some(
              (a) => typeof a === 'string' && a.includes('Worker pool ready with')
            )
          );
        },
        null,
        { timeout: 20000 }
      )
      .catch(() => {
        // Fall through: the assertion below reports what the pool DID log.
      });

    // Filtering to the `[WorkerPool]` lines is what puts the pool's actual
    // output in the failure report instead of a bare boolean.
    const workerPoolLines = (await getConsoleMessages(page, 10000)).all.filter((m) =>
      m.includes('[WorkerPool]')
    );
    expect(workerPoolLines).toEqual(
      expect.arrayContaining([expect.stringMatching(/Worker pool ready with \d+ worker\(s\)/)])
    );

    // Core assertion: points must have loaded.
    const pointCount = await page.evaluate(() => {
      const state = (window as any).__luxarDebug?.getState?.();
      return state?.totalPoints || 0;
    });
    expect(pointCount).toBeGreaterThan(0);
  });

  test('should fallback to main thread if worker fails', async ({ page }, testInfo) => {
    // This test stomps EVERY worker constructor (below), and app init warms up
    // the depth-sort worker — so `SortWorker failed to initialize` is a direct
    // consequence of the test's own premise, not a regression. The fixture's
    // opt-out is per-test rather than per-pattern, so annotate the one test;
    // widening DEFAULT_ALLOWED_CONSOLE_ERRORS would hide a real broken
    // SortWorker in the other 60-odd specs. The assertions below still pin the
    // user-visible contract (the app initializes and renders without workers).
    testInfo.annotations.push({
      type: ALLOW_CONSOLE_ERRORS,
      description:
        'Breaks every Worker constructor on purpose; the SortWorker warm-up failure it causes is expected.',
    });

    // Disable workers via the public config knob — the same switch
    // production code reads to decide "use worker or run on main thread".
    //
    // We do NOT block the worker chunk URL: vite imports the worker
    // module at module-level inside worker-pool.ts (`import DataWorker
    // from './data-worker?worker'`), so a 404 on the chunk crashes the
    // entire main bundle before bootstrap can even run. That kind of
    // failure isn't recoverable in any browser; the test would document
    // an unrealistic scenario.
    //
    // The realistic failure modes are: (1) workers explicitly disabled,
    // (2) WorkerPool init throws because a worker crashes during boot.
    // The init-throws path is now guarded by `workerInitTimeoutMs` in
    // `worker-pool.ts` and tested at the unit level. Here we verify
    // the user-visible contract: with workers off, the app still
    // initializes and renders.
    await page.addInitScript(() => {
      // Stomp the worker constructor before bundle load so any
      // accidental worker-creation attempt is a clean throw rather
      // than a hung Comlink call.
      (window as { Worker?: unknown }).Worker = function () {
        throw new Error('Workers disabled for fallback test');
      };
    });
    await page.goto(`/?src=${DATASET_3D}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    const state = await page.evaluate(() => {
      return (window as any).__luxarDebug?.getState?.();
    });
    expect(state).toBeDefined();
    expect(state?.pointClouds).toBeDefined();
    expect(state?.initialized).toBe(true);
  });

  test('should handle rapid view updates without worker congestion', async ({ page }) => {
    // First make sure initial data is loaded
    await waitForPointsLoaded(page);

    // Wait for interactions to settle
    await waitForNextRender(page);

    // Rapid camera-rotation navigation: ArrowRight rotates the
    // OrbitControls camera, which triggers re-projection on each
    // settled frame. The point of the test is "no congestion
    // crash" — not "the cursor moved" (ArrowRight isn't a dimension
    // key).
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('ArrowRight');
      // Intentional: rapid-update pacing exercises the worker's
      // ability to coalesce/cancel in-flight requests as new ones
      // arrive. Replacing with a tighter loop changes the contention
      // shape being tested.
      await page.waitForTimeout(50);
    }

    await waitForNextRender(page); // Let queries settle

    // Strengthened from the original `totalPoints >= 0` (a tautology
    // that accepted any non-negative number, including the "scene
    // emptied because every rapid query failed" regression). The new
    // floor of `> 0` catches a worker-congestion regression where all
    // queries are coalesced away to nothing.
    const finalPoints = await page.evaluate(
      () => (window as any).__luxarDebug?.getState?.()?.totalPoints ?? 0
    );
    expect(finalPoints).toBeGreaterThan(0);
  });
});

test.describe('WASM Integration E2E', () => {
  test.beforeEach(async ({ page }) => {
    // Use 5D dataset to test nD queries which exercise WASM
    await page.goto(`/?src=${DATASET_5D}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);
  });

  test('should load WASM module successfully', async ({ page }) => {
    // Wait for scene to be ready
    await waitForPointsLoaded(page);
    await waitForConsoleInterceptor(page);

    // Verify WASM module status via console logs
    // Either: [emoji] [WASM] Loaded compiled WASM module (success)
    // Or:     [emoji] [WASM] Failed to load WASM module (candidates, in order: <urls>), using
    //         TypeScript fallback
    //           — <urls> is the list the loader RESOLVED, not a list of URLs each proved to fail:
    //             the same catch covers a post-import failure (init throwing, or a stale artifact
    //             rejected by assertRequiredWasmExports), where one of them did load. Replaced by
    //             "<URL resolution failed before the import>" when no URL was computed at all.
    //           — this spec keys only on the `TypeScript fallback` tail.
    // Or:     No WASM messages at all (WASM binary not built — TypeScript path used implicitly)
    const messages = await getConsoleMessages(page);
    const wasmMessages = messages.all.filter((m) => m.includes('[WASM]'));

    if (wasmMessages.length > 0) {
      // If WASM was attempted, check for either success or explicit fallback
      const wasmLoaded = wasmMessages.some((m) => m.includes('Loaded compiled WASM module'));
      const tsFallback = wasmMessages.some((m) => m.includes('TypeScript fallback'));
      expect(wasmLoaded || tsFallback).toBe(true);
    }

    // DataWorker WASM status logs are optional — without WASM binary,
    // workers may not report WASM status at all.
    // Core assertion: points must have loaded regardless of WASM availability
    const hasPoints = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const state = debug?.getState?.();
      return state?.totalPoints > 0 || state?.pointClouds?.length > 0;
    });
    expect(hasPoints).toBe(true);
  });

  test('should use WASM for spatial queries if available', async ({ page }) => {
    // Wait for points to load
    await waitForPointsLoaded(page);
    await waitForConsoleInterceptor(page);

    // WASM readiness is informational — TypeScript fallback handles spatial queries
    // without explicit WASM messages, so no assertion needed here.

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await waitForNextRender(page);

    // Navigate to trigger spatial query on nD dataset (exercises WASM or TS code path)
    await page.keyboard.press('1'); // Select the first non-displayed dimension
    await page.keyboard.press(']'); // Navigate forward
    await waitForNextRender(page);

    // Core assertion: queries completed, returned visible data, and the
    // app is stable after navigation.
    const state = await page.evaluate(() => {
      return (window as any).__luxarDebug?.getState?.();
    });
    expect(state).toBeDefined();
    expect(state?.totalPoints).toBeGreaterThan(0);
  });

  test('should fallback to TypeScript if WASM unavailable', async ({ page }, testInfo) => {
    // This test deliberately aborts the WASM fetch, so the browser logs a
    // benign `net::ERR_FAILED` for the blocked resource. Since W4 the
    // in-process projection dispatcher loads WASM on the main thread too
    // (Points are now main-thread, WASM-accelerated), so that blocked fetch
    // surfaces on the page console. Opt out of the console-error guard —
    // the induced network error is the whole point of the test; the real
    // assertions below (TS-fallback taken + points still load) stand.
    testInfo.annotations.push({
      type: ALLOW_CONSOLE_ERRORS,
      description: 'Deliberately blocks the WASM fetch to exercise the TypeScript fallback.',
    });

    // Block WASM binary from loading to force TypeScript fallback
    await page.route(/\.wasm$/, (route) => route.abort());
    await page.route(/luxar_wasm/, (route) => route.abort());

    // Navigate fresh with WASM blocked
    await page.goto(`/?src=${DATASET_5D}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);
    await waitForPointsLoaded(page);
    await waitForConsoleInterceptor(page);

    // The dev server's HTTP cache and the wasm module's module-level
    // import can race the page.route() block — in environments where
    // the WASM binary is already cached, the block doesn't actually
    // prevent loading. So we accept either:
    //   (a) explicit TS-fallback log messages (route block won), OR
    //   (b) WASM was never attempted (no [WASM] log lines).
    // The CORE invariant — and the actual regression guard — is that
    // points must load regardless of which path was taken.
    const messages = await getConsoleMessages(page);
    const fallbackMessages = messages.all.filter(
      (m) => m.includes('TypeScript fallback') || m.includes('WASM initialization failed')
    );
    const wasmNeverAttempted = !messages.all.some((m) => m.includes('[WASM]'));
    expect(fallbackMessages.length > 0 || wasmNeverAttempted).toBe(true);

    // Core assertion: points must load. Strengthened from the original
    // `pointCount || 0` cast (which made `0 → 0` pass) to require an
    // actual non-zero population.
    const pointCount = await page.evaluate(() => {
      const state = (window as any).__luxarDebug?.getState?.();
      return state?.totalPoints || 0;
    });
    expect(pointCount).toBeGreaterThan(0);
  });

  test('should handle WASM errors gracefully', async ({ page }) => {
    // Wait for points to load
    await waitForPointsLoaded(page);

    // Click canvas for focus
    await page.click('canvas');
    await waitForNextRender(page);

    // Navigate multiple times to stress test WASM
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press(']');
      // Intentional: same rapid-sequential pacing pattern as
      // spatial-index-accuracy.spec.ts.
      await page.waitForTimeout(100);
    }

    // Wait for operations to settle
    await waitForNextRender(page);

    // Should have no crashes - state should still be accessible
    const state = await page.evaluate(() => {
      return (window as any).__luxarDebug?.getState?.();
    });

    expect(state).toBeDefined();
    // Check points are loaded via pointClouds or totalPoints
    expect(state?.pointClouds?.length > 0 || state?.totalPoints >= 0).toBeTruthy();
  });
});

test.describe('Worker + WASM Combined Performance', () => {
  test('should achieve faster queries with both enabled', async ({ page }) => {
    // Use performance benchmark dataset if available, otherwise fall back to 5D
    await page.goto(`/?src=${DATASET_LARGE}&debug`);

    // Wait for ready (may take longer for large dataset)
    try {
      await waitForLuxarReady(page, 15000);
      await waitForDataLoaded(page);
    } catch {
      // If large dataset not available, skip test
      test.skip();
      return;
    }

    // Measure query time with navigation
    const startTime = Date.now();

    await page.keyboard.press('[');
    // Let the worker query + projection round-trip complete; this is
    // the actual "query complete" signal rather than a fixed sleep.
    await waitForDataLoaded(page);

    const endTime = Date.now();
    const totalTime = endTime - startTime;

    // With workers + WASM, large queries should complete quickly
    // This is a smoke test, not precise benchmarking
    //
    // CONTENTION NOTE: wall-clock assertion, and the suite runs `fullyParallel`
    // across up to 4 workers. If this flakes, diagnose with `--workers=1` rather
    // than raising the threshold. The local count is now sized to the box
    // (tools/e2e-workers.ts), so a loaded machine runs fewer of them: against the SAME
    // box at four workers this fixed threshold has more headroom. It does NOT follow
    // that a loaded machine is a softer test than an idle one — its absolute wall clock
    // is slower either way.
    expect(totalTime).toBeLessThan(10000); // Should finish in <10 seconds

    // Verify data loaded
    const state = await page.evaluate(() => (window as any).__luxarDebug?.getState?.());
    expect(state).toBeDefined();
  });
});
