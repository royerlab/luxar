/**
 * Helper utilities for Playwright E2E tests
 */

import { Page } from '@playwright/test';

import { isTypingSurfaceInPage } from './page-predicates';

/**
 * Wait for Luxar to fully initialize
 * Increased timeout for E2E tests with real dataset loading
 */
export async function waitForLuxarReady(page: Page, timeout = 45000): Promise<void> {
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      return debug && debug.getState && debug.getState().initialized;
    },
    null, // no arguments to pass to the pageFunction
    { timeout }
  );
}

/**
 * Get current Luxar state
 * Now includes safety check for debug interface availability
 *
 * Deadline-bounded (#1651), like {@link getConsoleMessages}. This is the most
 * called probe in the suite — well over a hundred call sites, in most spec
 * files. Together with `getConsoleMessages` it is what the bulk of the specs
 * depend on, and both are now bounded; the rest of this
 * file's bare `page.evaluate` calls (`renderOnce`, `getWebGLErrors`,
 * `focusCanvas`, `captureCanvasRGBA`, `probeWebGPUBackend`, …) are not. Before
 * the bound, a starved page reported the stall as a bare
 * `Test timeout of 120000ms exceeded` pointing at the `page.evaluate` line
 * below, with nothing to say which dataset or which probe was pending.
 * Measured on `performance_benchmark_example.luxar.zarr` (100 nodes, 100k
 * points): `getState()` costs 0.5 ms in-page and returns 12 KB, yet the round
 * trip took 111,003 ms in one run and >150,000 ms in another, because the
 * viewer's frame loop saturated the main thread and starved Playwright's
 * `Runtime.callFunctionOn`. That underlying viewer bug (#1724) is fixed — the
 * loop now paces itself — but the bound stays: it is the generic diagnostic
 * for a starved page, not a workaround for one scene.
 *
 * The same honest caveat as `getConsoleMessages` applies: a deadline cannot
 * tell a page that will never answer from one that would have answered late,
 * so any value here can cut short a stall that would have ended. 45 s matches
 * `getConsoleMessages` for the reasons documented there.
 *
 * THROWS rather than returning a fallback. Callers overwhelmingly assert on
 * the result (`expect(state.totalPoints).toBeGreaterThan(0)`), so any stand-in
 * value would either fail with a nonsense diagnostic or — worse, for the
 * `state && …` and `!state.isLoading` polling helpers above — read as a pass.
 *
 * @param page - Playwright page
 * @param timeout - Deadline for the in-page probe, in ms
 * @throws If the page does not answer the probe within `timeout` ms
 */
export async function getLuxarState(page: Page, timeout = 45000): Promise<any> {
  const probe = page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    if (!debug || typeof debug.getState !== 'function') {
      throw new Error('Debug interface not ready: getState() not available');
    }
    return debug.getState();
  });

  // Sentinel: a fresh object allocated HERE, in Node. `getConsoleMessages` can
  // argue `null` is safe because its in-page function has exactly two exits and
  // both return an object literal it wrote itself; this one returns whatever
  // `debug.getState()` hands back, an untyped `any` whose shape the viewer is
  // free to change — today an object, but a version that returned `null` or
  // `undefined` (or nothing) would make any primitive sentinel ambiguous, and
  // silently so. Identity against a Node-side object cannot collide at all:
  // `page.evaluate` resolves with a value deserialized from the CDP protocol,
  // hence always freshly constructed on this side, so even a page answering
  // with a literal `{}` compares `!==` to `timedOut`.
  //
  // The in-page `throw` above still propagates as a rejection: whenever the
  // evaluate settles before the deadline, `Promise.race` settles the same way,
  // so a genuinely missing debug interface remains the real failure it always
  // was rather than being reported as a stall.
  const timedOut = {};
  const state = await raceEvaluate<unknown>(probe, timeout, timedOut);

  if (state === timedOut) {
    throw new Error(
      `getLuxarState: the page never answered the state probe within ${timeout} ms — ` +
        'its main thread is saturated and starving the evaluate round trip, so the viewer state ' +
        'could not be read. See issues #1651 and #1724.'
    );
  }

  return state;
}

/**
 * Trigger a single render frame (for stable screenshots)
 */
export async function renderOnce(page: Page): Promise<void> {
  await page.evaluate(() => {
    (window as any).__luxarDebug.renderOnce();
  });
  // Intentional fixed sleep: renderOnce() schedules a frame, but the actual
  // paint lands on the next browser frame, which is not directly observable
  // from JS. 300 ms rather than the historical 100 ms so the wait also covers
  // a frame-pacing cooldown (#1724): renderOnce() is `startAnimation()`, which
  // does not shorten a cooldown already armed on a running loop, and the
  // cooldown is bounded by `config.animation.pacing.maxCooldownMs` — 250 ms,
  // hard-coded here rather than imported, since this helper must not pull
  // viewer config into the Node-side test process. So 300 ms is that 250 ms
  // plus a paint cycle past 60 fps.
  //
  // It is NOT a worst-case bound for a paced page, and this helper never had
  // one for a slow page: the wait ahead of the next frame is the cooldown PLUS
  // whatever the frame in flight still costs, and a page only ever paces once
  // its frames already exceed 250 ms on their own. A spec that must sample
  // strictly after the change on a scene that slow needs a frame-counting wait,
  // not a fixed sleep.
  await page.waitForTimeout(300);
}

/**
 * Render the last swallowed probe failure as a suffix for a poll loop's own
 * timeout message.
 *
 * The poll loops below tolerate a probe that fails — during initialization the
 * debug interface legitimately is not there yet — so they catch and retry. But
 * a starved probe fails the same way, and reporting only "timeout waiting for
 * points" then blames missing DATA for what is actually a saturated main
 * thread (#1651 / #1724). Appending the last failure keeps the loop's contract
 * while letting `getLuxarState`'s diagnostic reach the report.
 *
 * Returns an empty string when the last probe answered, so a loop that simply
 * never saw its condition satisfied reports exactly what it always did. The
 * callers clear their record on every answered probe for that reason: a missing
 * debug interface during initialization is normal and recovers, and carrying
 * that first failure to the end would blame it for a loop that then polled a
 * healthy page for the rest of its budget — the same misattribution in the
 * other direction.
 */
function describeLastProbeError(error: unknown): string {
  if (error === undefined) return '';
  const message = error instanceof Error ? error.message : String(error);
  return `. Last state probe failed: ${message}`;
}

/**
 * Compose the two end-of-budget reports of a split give-up loop (#1726): the
 * THROW for "no probe ever answered" and the WARN for "a probe answered but the
 * condition never settled". `waitForSpatialQuery` and
 * `waitForNavigationComplete` differ only in their names and nouns, so they
 * share this one composer and a single grep still finds every carried-failure
 * report.
 *
 * Both reports state the measured elapsed time next to the nominal budget — the
 * probes are not clamped to the loop's budget (see `waitForPointsLoaded`), so a
 * single one can overrun it several times over — and the probe counts next to
 * that, since the carried diagnostic is only ever the LAST one and a window in
 * which nearly every probe failed would otherwise read as a clean "answered but
 * never settled".
 *
 * `subject` is what was never observed, for the throw ("navigation"); `unsettled`
 * the condition that never held, for the warn; `unusable` the probes that either
 * rejected or answered with state the condition could not be read off; and
 * `warnNote` an extra clause for the warn only, placed chronologically before the
 * carried probe failure.
 */
function composeGiveUpReports(report: {
  helper: string;
  subject: string;
  unsettled: string;
  timeout: number;
  elapsed: number;
  probes: number;
  unusable: number;
  lastProbeError: unknown;
  warnNote?: string;
}): { failure: string; warning: string } {
  const counts =
    `${report.timeout}ms budget, ${report.elapsed}ms actually elapsed, ` +
    `${report.probes} probe(s) attempted, ${report.unusable} of them yielded no usable state`;
  const carried = describeLastProbeError(report.lastProbeError);

  return {
    failure:
      `${report.helper}: no state probe answered during its poll window (${counts}), so ` +
      `${report.subject} was never observed either way and whatever this test asserts next ` +
      `reads unconfirmed state (see issue #1726)${carried}`,
    warning:
      `[⚠️] [E2E ${report.helper}] the page answered but ${report.unsettled} ` +
      `(${counts}); returning anyway` +
      (report.warnNote ? `. ${report.warnNote}` : '') +
      carried,
  };
}

/**
 * Wait for points to be loaded
 * Now includes debug interface readiness check
 */
export async function waitForPointsLoaded(
  page: Page,
  minPoints = 1,
  timeout = 45000
): Promise<void> {
  const startTime = Date.now();
  let lastProbeError: unknown;

  while (Date.now() - startTime < timeout) {
    try {
      // Probe deliberately left at its own 45 s default rather than clamped to
      // this loop's remaining budget (same for the three sibling poll loops
      // below, which share this shape): these loops return on any satisfying
      // answer, so a late one is a SUCCESS, and clamping would newly FAIL a
      // healthy-but-slow page across ~sixty call sites here and five 8 s ones
      // in `waitForDimensionNavigation`. The honest cost is that the timeout
      // message below can overstate how long it waited (one probe can hold the
      // full 45 s); closing that needs its own measured change.
      const state = await getLuxarState(page);
      // The probe answered, so any earlier failure is stale — do not let an
      // initialization-window miss get reported as the reason this loop ended.
      lastProbeError = undefined;

      if (state && state.totalPoints >= minPoints) {
        return;
      }
    } catch (error) {
      // Debug interface not ready yet, continue waiting
      // This can happen during initialization.
      // Kept for the throw below: swallowing it outright is how a starved probe
      // (see `getLuxarState`) gets misreported as missing data.
      lastProbeError = error;
    }

    await page.waitForTimeout(500);
  }

  throw new Error(
    `Timeout waiting for points to load (expected at least ${minPoints})${describeLastProbeError(lastProbeError)}`
  );
}

/**
 * Get console messages of a specific type
 */
export function captureConsoleMessages(page: Page): {
  errors: string[];
  warnings: string[];
  logs: string[];
} {
  const messages = {
    errors: [] as string[],
    warnings: [] as string[],
    logs: [] as string[],
  };

  page.on('console', (msg) => {
    const text = msg.text();
    switch (msg.type()) {
      case 'error':
        messages.errors.push(text);
        break;
      case 'warning':
        messages.warnings.push(text);
        break;
      default:
        messages.logs.push(text);
    }
  });

  return messages;
}

/**
 * NOTE: Manual screenshot helpers removed - use Playwright's built-in screenshot system instead.
 *
 * Playwright captures screenshots, video and traces ON FAILURE ONLY
 * (`screenshot: 'only-on-failure'`, `video`/`trace: 'on-first-retry'` in
 * playwright.config.ts). Recording them for passing tests too cost ~330-430 s
 * per run for artifacts nobody read.
 *
 * A failing test therefore still produces a screenshot in test-results/, and
 * under `--retries` the retry produces a full trace and video as well. To get
 * the same artifacts for a PASSING test while debugging, re-run that spec with
 * `--trace on --video on --screenshot on`.
 *
 * To capture a screenshot at a specific point regardless of outcome:
 * `await page.screenshot({ path: 'test-results/my-screenshot.png' });`
 */

/**
 * Wait for data loading to complete
 * More robust than arbitrary timeouts
 */
export async function waitForDataLoaded(page: Page, timeout = 10000): Promise<void> {
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      if (!debug || !debug.getState) return false;
      const state = debug.getState();
      // Data is loaded if:
      // 1. Not currently loading AND
      // 2. Either has points OR explicitly has no points (valid state)
      return state && !state.isLoading;
    },
    null, // no arguments
    { timeout }
  );
}

/**
 * Wait for dimension navigation to complete
 * Detects when slice position changes and new data is loaded
 */
export async function waitForDimensionNavigation(
  page: Page,
  previousPointCount: number,
  timeout = 8000
): Promise<void> {
  const startTime = Date.now();
  let lastProbeError: unknown;

  while (Date.now() - startTime < timeout) {
    try {
      const state = await getLuxarState(page);
      // Answered, so an earlier failure is stale (see `waitForPointsLoaded`).
      lastProbeError = undefined;

      // Navigation complete if:
      // 1. Point count changed (new data loaded), OR
      // 2. Data loading finished (even if count same due to broadcast/cache)
      if (state.totalPoints !== previousPointCount || !state.isLoading) {
        return;
      }
    } catch (error) {
      // State not ready yet, continue waiting (see `waitForPointsLoaded` for
      // why the last probe failure is carried into the throw).
      lastProbeError = error;
    }

    await page.waitForTimeout(200);
  }

  throw new Error(
    `Dimension navigation did not complete within ${timeout}ms${describeLastProbeError(lastProbeError)}`
  );
}

/**
 * Wait for console interceptor to be fully initialized
 * Some tests need this before accessing console messages
 */
export async function waitForConsoleInterceptor(page: Page, timeout = 5000): Promise<void> {
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      return (
        debug &&
        debug.consoleInterceptor &&
        typeof debug.consoleInterceptor.getBufferedMessages === 'function'
      );
    },
    null,
    { timeout }
  );
}

/**
 * Wait for debug interface to be fully ready
 * Ensures all debug properties are initialized
 */
export async function waitForDebugInterfaceReady(page: Page, timeout = 10000): Promise<void> {
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      return (
        debug &&
        debug.scene &&
        debug.camera &&
        debug.renderer &&
        debug.getState &&
        typeof debug.getState === 'function'
      );
    },
    null,
    { timeout }
  );
}

/**
 * Wait for dimension to be selected
 * More reliable than arbitrary timeout after pressing number key
 */
export async function waitForDimensionSelected(
  page: Page,
  dimensionIndex: number,
  timeout = 5000
): Promise<void> {
  await page.waitForFunction(
    (idx) => {
      const debug = (window as any).__luxarDebug;
      if (!debug?.getState?.()?.initialized) return false;
      // Verify the selected dimension via the canonical sceneDimsManager
      // exposed at __luxarDebug.sceneDimsManager (see app.ts:912-927).
      const selected = debug?.sceneDimsManager?.getSelectedDimension?.();
      if (typeof selected === 'number') return selected === idx;
      // Fallback: if API not available, just wait for initialized state
      return true;
    },
    dimensionIndex,
    { timeout }
  );
}

/**
 * Wait for spatial index query to complete.
 *
 * **Silent only on the tolerated case.** The give-up is now split in two
 * (#1726):
 *
 * - A probe ANSWERED but the query never settled within `timeout` — the case
 *   the old "timeout not an error" comment was really about: returns normally,
 *   after one `console.warn` saying the condition was never observed.
 * - Every probe FAILED, so none ever answered — the helper read no viewer state
 *   at all and the caller's next assertions would read state nothing has
 *   confirmed. It THROWS, with the last probe failure carried into the message.
 *   Reachable today through a rejecting probe: no `__luxarDebug`, or a destroyed
 *   execution context.
 *
 * A NON-POSITIVE `timeout` never enters the loop, so it probes zero times and the
 * report says so.
 *
 * See {@link waitForNavigationComplete} for what the split buys, and
 * `waitForPointsLoaded` for why the probes are not clamped to this loop's budget
 * — which is why both reports state the measured elapsed time and the probe
 * counts next to the nominal budget (see `composeGiveUpReports` above — a plain
 * code reference, not an `@link`, because it is module-private and TypeDoc
 * warns on a link it resolves but cannot document).
 *
 * Use when the query completing fast is a *bonus*, not a precondition. For
 * tests that genuinely depend on the query having finished, use
 * {@link waitForSpatialQueryOrThrow} instead.
 *
 * Unit-tested in `src/tests/unit/tests/e2e-helpers-silent-waits.test.ts`.
 */
export async function waitForSpatialQuery(page: Page, timeout = 8000): Promise<void> {
  const startTime = Date.now();
  let answered = false;
  let probes = 0;
  let unusable = 0;
  let lastProbeError: unknown;

  while (Date.now() - startTime < timeout) {
    probes += 1;
    try {
      // Check if a query completed by looking for stable state.
      const state = await getLuxarState(page);
      // Any resolved value means the page answered. A `null` answer trips the
      // condition below into the `catch`, which counts it among the probes that
      // yielded no usable state.
      answered = true;
      lastProbeError = undefined;

      // If we have a stable point count and not loading, query is done
      if (!state.isLoading && state.totalPoints >= 0) {
        return;
      }
    } catch (error) {
      // State not ready yet
      unusable += 1;
      lastProbeError = error;
    }

    await page.waitForTimeout(100);
  }

  const { failure, warning } = composeGiveUpReports({
    helper: 'waitForSpatialQuery',
    subject: 'the spatial query',
    unsettled: 'the query never settled — isLoading stayed true, or totalPoints was absent',
    timeout,
    elapsed: Date.now() - startTime,
    probes,
    unusable,
    lastProbeError,
  });

  if (!answered) throw new Error(failure);

  // Answered, never settled — the tolerated case, reported rather than silent.
  console.warn(warning);
}

/**
 * Throwing variant of {@link waitForSpatialQuery}. Rejects with a
 * descriptive error if the query never settles within `timeout`.
 * Prefer this when the test logic that follows assumes the query has
 * actually completed (e.g. point-count assertions).
 */
export async function waitForSpatialQueryOrThrow(page: Page, timeout = 8000): Promise<void> {
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      const state = debug?.getState?.();
      return state && !state.isLoading && state.totalPoints >= 0;
    },
    null,
    { timeout }
  );
}

/**
 * Wait for UI element to appear or disappear
 * More reliable than arbitrary timeouts for UI state changes
 */
export async function waitForUIState(
  page: Page,
  selector: string,
  visible: boolean,
  timeout = 5000
): Promise<void> {
  if (visible) {
    await page.waitForSelector(selector, { state: 'visible', timeout });
  } else {
    await page.waitForSelector(selector, { state: 'hidden', timeout }).catch(() => {
      // Element might not exist at all, which is also "not visible"
    });
  }
}

/**
 * Get browser console messages (CRITICAL for E2E validation)
 *
 * Retrieves all console messages from the browser's console interceptor.
 * This is ESSENTIAL for detecting errors in data loading, decoding, and rendering.
 *
 * Deadline-bounded (#1651). WHO CALLS IT: the specs that call it directly
 * (`all-examples-smoke-test`, `test-fixtures-rendering`,
 * `worker-wasm-integration`), plus `assertNoConsoleErrors`,
 * `assertConsoleContains`, `assertConsoleDoesNotContain` and
 * `assertNoShaderErrors` and the specs that call those — some of them from
 * their own `test.afterEach`. The shared fixture's teardown does NOT (#1760):
 * it gates on `page.on('console')` + `page.on('pageerror')`, which covers the
 * ERROR verdict from a strictly wider source, so the probe there was a
 * narrower second opinion charging up to 45 s to each of the 59 of 67 specs
 * that import `test` from `./fixtures`. Before the bound, an unanswered probe
 * burned the ENTIRE remaining test budget and was reported as `Tearing down
 * "page" exceeded the test timeout` pending on the evaluate below. Failing
 * in `timeout` ms with a message that says what went unanswered is
 * strictly more informative.
 *
 * WHAT THE DEFAULT COSTS, honestly: a deadline here cannot distinguish a
 * page that will never answer from one that would have answered late, so
 * ANY value can cut short a stall that would have ended, turning a test
 * that used to pass into one that fails. That is a real cost rather than a
 * hypothetical — the stalls measured for #1651 lasted tens of seconds (a
 * trivial `page.evaluate` unanswered for 5 s twelve times running, ~78 s in
 * all, while the page went on rendering), and #1760 reproduced 95 s without
 * one serviced round trip on a 2-CPU browser while Playwright's own
 * `page.on('console')` stream kept delivering — the two channels starve
 * independently. The bound is worth paying anyway because the alternative
 * failure is opaque, and because every remaining caller asked for the in-page
 * buffer specifically — its `warnings` / `logs` buckets have no
 * Playwright-side gate at all — so learning the buffer could not be read
 * beats proceeding on empty ones. It is a trade, not a free win.
 *
 * WHY 45 s: Playwright gives the After Hooks phase a FRESH timeout slot
 * (`afterHooksSlot = { timeout: calculateMaxTimeout(project.timeout,
 * testInfo.timeout) }` in its worker), so an `afterEach` gate always has the
 * full per-test timeout available — the config's 60 s, or more in a file that
 * raises its own — no matter how much the test
 * body already used. 45 s lands inside that slot — which is what makes the
 * failure attributable to this probe by name instead of arriving as `Tearing
 * down "page" exceeded the test timeout` — while still leaving a wide margin
 * for a page that recovers late. It also bounds the mid-test call sites,
 * where the probe shares the body's budget rather than getting a fresh slot.
 * The work itself is a walk over at most
 * `DEFAULT_MAX_BUFFER_SIZE` buffered messages
 * (`src/utils/console-interceptor.ts`), so a live page answers in
 * milliseconds and never approaches this.
 *
 * @param page - Playwright page
 * @param timeout - Deadline for the in-page probe, in ms
 * @returns Object with errors, warnings, and info messages
 * @throws If the page does not answer the probe within `timeout` ms
 */
export async function getConsoleMessages(
  page: Page,
  timeout = 45000
): Promise<{
  errors: string[];
  warnings: string[];
  logs: string[];
  all: string[];
}> {
  const probe = page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    if (!debug || !debug.consoleInterceptor) {
      // Legitimate "no interceptor installed" answer, not a wedge.
      return { errors: [], warnings: [], logs: [], all: [] };
    }

    const messages = debug.consoleInterceptor.getBufferedMessages();
    const errors: string[] = [];
    const warnings: string[] = [];
    const logs: string[] = [];
    const all: string[] = [];

    messages.forEach((msg: any) => {
      const text = typeof msg === 'string' ? msg : JSON.stringify(msg);
      all.push(text);

      // Prioritize msg.type over text content to avoid false positives
      // (e.g., info messages mentioning "error" should not be flagged as errors)
      if (msg.type === 'error') {
        errors.push(text);
      } else if (msg.type === 'warn') {
        warnings.push(text);
      } else if (msg.type === 'log' || msg.type === 'info' || msg.type === 'debug') {
        logs.push(text);
      } else {
        // Fallback for messages without type: check text content
        if (text.toLowerCase().includes('[❌]') || /\berror:/i.test(text)) {
          errors.push(text);
        } else if (text.toLowerCase().includes('[⚠️]') || /\bwarning:/i.test(text)) {
          warnings.push(text);
        } else {
          logs.push(text);
        }
      }
    });

    return { errors, warnings, logs, all };
  });

  // Race against `null` as the sentinel: the in-page function above always
  // returns an object (empty buckets when there is no interceptor), so a
  // `null` here can only mean the deadline won.
  const buckets = await raceEvaluate<Awaited<typeof probe> | null>(probe, timeout, null);

  if (buckets === null) {
    // Deliberately NOT empty buckets: every caller feeds this into a check,
    // so returning `{errors: [], ...}` here would silently turn that check
    // into a vacuous pass.
    throw new Error(
      `getConsoleMessages: the page never answered the console-buffer probe within ${timeout} ms — ` +
        'its main thread is saturated and starving the evaluate round trip, so the console-error ' +
        'check could not run. See issue #1651.'
    );
  }

  return buckets;
}

/**
 * Assert no console errors (CRITICAL for all E2E tests)
 *
 * Opt-in per spec: the shared fixture does NOT call it (#1760), so call it
 * after loading data wherever the in-page buffer's stricter verdict is wanted.
 * Catches errors in:
 * - Data loading
 * - Array decoding
 * - Spatial index queries
 * - Geometry creation
 * - WebGL rendering
 *
 * @param page - Playwright page
 * @param allowedPatterns - Optional patterns to ignore (e.g., expected warnings)
 *
 * NOTE: assertNoConsoleErrors is defined below getWebGLErrors.
 */

/**
 * Get WebGL errors from the rendering context
 *
 * CRITICAL: WebGL errors accumulate and can indicate serious rendering issues:
 * - Buffer size mismatches
 * - Invalid shader state
 * - Texture allocation failures
 *
 * @param page - Playwright page
 * @returns Array of WebGL error messages
 */
export async function getWebGLErrors(page: Page): Promise<string[]> {
  return await page.evaluate(() => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return ['No canvas element found'];

    const gl =
      (canvas as HTMLCanvasElement).getContext('webgl2') ||
      (canvas as HTMLCanvasElement).getContext('webgl');

    if (!gl) return ['No WebGL context available'];

    const errors: string[] = [];
    let error;
    let safetyCount = 0;

    // Read all errors from queue (limit to 100 to prevent infinite loop)
    while ((error = gl.getError()) !== gl.NO_ERROR && safetyCount < 100) {
      const errorName =
        error === gl.INVALID_ENUM
          ? 'INVALID_ENUM'
          : error === gl.INVALID_VALUE
            ? 'INVALID_VALUE'
            : error === gl.INVALID_OPERATION
              ? 'INVALID_OPERATION'
              : error === gl.OUT_OF_MEMORY
                ? 'OUT_OF_MEMORY'
                : error === gl.INVALID_FRAMEBUFFER_OPERATION
                  ? 'INVALID_FRAMEBUFFER_OPERATION'
                  : `UNKNOWN(0x${error.toString(16)})`;

      errors.push(`GL_${errorName}`);
      safetyCount++;
    }

    return errors;
  });
}

/**
 * Assert console contains expected log pattern
 *
 * @param page - Playwright page
 * @param pattern - RegExp pattern to search for
 * @param errorMessage - Optional custom error message
 */
export async function assertConsoleContains(
  page: Page,
  pattern: RegExp,
  errorMessage?: string
): Promise<void> {
  const messages = await getConsoleMessages(page);
  const found = messages.all.some((msg) => pattern.test(msg));

  if (!found) {
    throw new Error(
      errorMessage ||
        `Console does not contain expected pattern: ${pattern}\n` +
          `Console has ${messages.all.length} messages total`
    );
  }
}

/**
 * Assert console does NOT contain error pattern
 *
 * @param page - Playwright page
 * @param pattern - RegExp pattern that should NOT appear
 * @param errorMessage - Optional custom error message
 */
export async function assertConsoleDoesNotContain(
  page: Page,
  pattern: RegExp,
  errorMessage?: string
): Promise<void> {
  const messages = await getConsoleMessages(page);
  const found = messages.all.filter((msg) => pattern.test(msg));

  if (found.length > 0) {
    console.error('Found forbidden console messages:');
    found.forEach((msg, i) => {
      console.error(`  ${i + 1}. ${msg}`);
    });

    throw new Error(
      errorMessage ||
        `Console contains forbidden pattern: ${pattern}\n` + `Found ${found.length} matches`
    );
  }
}

/**
 * Wait for dimension system to be initialized
 * Returns true if dimensions are initialized, false if no nD data
 *
 * @param page - Playwright page
 * @param timeout - Maximum wait time in ms
 */
export async function waitForDimensionSystemReady(page: Page, timeout = 10000): Promise<boolean> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    try {
      const hasInitialized = await page.evaluate(() => {
        const debug = (window as any).__luxarDebug;
        if (!debug) return null;

        // Canonical path per app.ts:912-927.
        const dims = debug.sceneDimsManager?.getDims?.();
        return dims !== null && dims !== undefined;
      });

      if (hasInitialized === true) {
        return true;
      } else if (hasInitialized === false) {
        // No nD data in scene
        return false;
      }
    } catch {
      // Not ready yet
    }

    await page.waitForTimeout(100);
  }

  // Timeout — final probe via the canonical path. Use truthiness, not
  // `!== null`, so `undefined` cannot produce a false pass.
  const finalState = await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    return !!debug?.sceneDimsManager?.getDims?.();
  });

  return finalState;
}

/**
 * Wait for nD navigation to complete: poll until `getState().isLoading` is
 * false.
 *
 * There is no "wait for the navigation to START" phase — the helper only ever
 * waits for `isLoading` to be false, so it resolves on the first poll if the
 * loader is already idle. That is the intended behaviour on both sides of the
 * race: a same-task trigger (a keyboard nav) sets the flag synchronously —
 * `SceneLoader.updateView` takes the lock before its first await — so it is
 * already true before this helper gets to poll, while a fully cache-served pass
 * may never be OBSERVED true at all: polling is discrete (`waitForFunction` on
 * rAF below, then a 100 ms loop), so such a pass can start and finish between
 * two polls. That is why the callers in `spatial-index-accuracy.spec.ts` prefer
 * this silent variant. Waiting for `true` first would hang in the second case.
 *
 * The only preliminary wait is for the flag to EXIST (`typeof isLoading ===
 * 'boolean'`), i.e. for the debug interface to be installed. That wait failing
 * no longer returns on its own (#1726): it FALLS THROUGH into the poll loop below,
 * so the one end-of-budget decision governs. Its old comment claimed the failure
 * meant "navigation completed before we start polling", which does not actually
 * explain a missing `isLoading` — the flag
 * exists whenever the debug interface does (`getState()` always returns a real
 * boolean, see `core/app/debug/debug-state.ts`) — so returning there returned on
 * nothing. Falling through gives the state probe a chance to answer, or to
 * produce a diagnostic of its own. The preliminary failure is still worth
 * REPORTING — "the flag was unreadable for the first N ms" is exactly what explains
 * a 15 s budget that never saw `isLoading` clear — so its duration
 * (deterministically the clamped preliminary budget) is interpolated into the
 * answered-but-unsettled warn. It is NOT a probe failure: it stays out of the probe
 * counts, and out of the throw, where the carried probe rejection is both fresher
 * and the better headline.
 *
 * **Silent only on the tolerated case.** The give-up is split in two:
 *
 * - A probe ANSWERED but `isLoading` never cleared within `timeout`: returns
 *   normally, after one `console.warn` naming the helper, how many probes went out
 *   and how many yielded no usable state, and the preliminary failure if there was
 *   one. This is what the nine call sites rely on — several deliberately pick this
 *   variant because their action never toggles `isLoading` at all, and they succeed
 *   on the first probe.
 * - Every probe FAILED, so none ever answered: THROWS, carrying the last probe
 *   failure and the same counts. The helper read no viewer state, so returning
 *   would let the caller assert on state nothing has confirmed.
 *
 * The gain is ATTRIBUTION, not a rescued pass: at all nine call sites the
 * statement following the wait is itself a page probe (`getLuxarState`, a bare
 * `page.evaluate`, or `getDimensionValue`, which answers `-1` and then fails its
 * own `expect`), so the old silence already surfaced as a CONFUSING FAILURE one
 * statement later — never as a pass. What changes is the name on it, and only when
 * the decision is reached at all: the fall-through can take a starved page ~55 s
 * past the start of the wait, which at the three sites whose next statement is an
 * unbounded `page.evaluate` usually lands beyond the 60 s per-test `timeout` in
 * `playwright.config.ts` — and beyond it at the six bounded
 * `spatial-index-accuracy.spec.ts` sites too, since the spec has already spent
 * seconds reaching the wait. At those six the THROW is additionally
 * near-unreachable: they sit right after a `waitForSpatialQueryOrThrow`, which
 * cannot itself have succeeded unless the page answered within the last 8 s, so
 * what they get from this is the attributed warn. Closing the vacuous pass is
 * therefore about the CONTRACT, for a future caller that does not happen to
 * re-probe straight afterwards. What is closed is a swallowed probe FAILURE, not a
 * swallowed timeout; see `waitForPointsLoaded` for why the probes here are not
 * clamped to the loop's budget.
 *
 * Use {@link waitForNavigationCompleteOrThrow} for tests that depend on
 * navigation actually finishing.
 *
 * Unit-tested in `src/tests/unit/tests/e2e-helpers-silent-waits.test.ts`.
 *
 * @param page - Playwright page
 * @param timeout - Poll budget in ms, not a hard ceiling on the call. The
 *   preliminary wait above consumes at most `max(1, min(timeout, 10000))` of it —
 *   floored at 1 ms because Playwright reads `timeout: 0` as NO timeout, so a
 *   non-positive budget would otherwise wait there for the whole test — and the
 *   `do`/`while` below then always runs at least one probe, bounded by its own 45 s
 *   deadline rather than by this budget, so a call can overrun `timeout` by a whole
 *   probe (both end-of-budget reports state the measured elapsed time alongside
 *   this budget for exactly that reason). Probing at least once matters because the
 *   preliminary wait can spend the whole budget on its own, and reporting "no probe
 *   answered" without having probed would be a false diagnostic.
 */
export async function waitForNavigationComplete(page: Page, timeout = 15000): Promise<void> {
  const startTime = Date.now();
  let answered = false;
  let probes = 0;
  let unusable = 0;
  let lastProbeError: unknown;
  let preliminaryFailure: string | undefined;

  // First, wait for loading state to be available (may already be done).
  // Floored at 1 ms: Playwright reads `timeout: 0` as NO timeout, so a
  // non-positive `timeout` here would hand the preliminary wait the whole test
  // budget — the opposite of the `min(timeout, 10000)` ceiling this line is for.
  const preliminaryTimeout = Math.max(1, Math.min(timeout, 10000));
  try {
    await page.waitForFunction(
      () => {
        const debug = (window as any).__luxarDebug;
        return debug && debug.getState && typeof debug.getState().isLoading === 'boolean';
      },
      null,
      { timeout: preliminaryTimeout }
    );
  } catch {
    // The flag never became readable. Fall through to the poll loop rather than
    // returning on an unread page, but record how long it cost.
    preliminaryFailure =
      'The preliminary wait never saw a readable isLoading flag ' +
      `(the first ${Date.now() - startTime}ms of this call)`;
  }

  // Then wait for loading to complete.
  do {
    probes += 1;
    try {
      const state = await getLuxarState(page);
      // Any resolved value counts as an answer; see `waitForSpatialQuery`.
      answered = true;
      lastProbeError = undefined;
      if (!state.isLoading) {
        await page.waitForTimeout(100);
        return;
      }
    } catch (error) {
      // State may briefly be unavailable during navigation
      unusable += 1;
      lastProbeError = error;
    }

    await page.waitForTimeout(100);
  } while (Date.now() - startTime < timeout);

  const { failure, warning } = composeGiveUpReports({
    helper: 'waitForNavigationComplete',
    subject: 'navigation',
    unsettled: 'isLoading never cleared',
    timeout,
    elapsed: Date.now() - startTime,
    probes,
    unusable,
    lastProbeError,
    warnNote: preliminaryFailure,
  });

  if (!answered) throw new Error(failure);

  // Answered, never cleared — the tolerated case, reported rather than silent.
  console.warn(warning);
}

/**
 * Throwing variant of {@link waitForNavigationComplete}. Rejects with
 * a descriptive error if navigation never settles within `timeout`.
 */
export async function waitForNavigationCompleteOrThrow(page: Page, timeout = 15000): Promise<void> {
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      const state = debug?.getState?.();
      return state && state.isLoading === false;
    },
    null,
    { timeout }
  );
}

/**
 * Wait for render frames to stabilize
 *
 * Useful for visual regression tests that need stable screenshots.
 * Waits for the frame counter to ADVANCE by `minFrames` if available, otherwise
 * uses a state-based wait + time buffer.
 *
 * **Best-effort, like {@link waitForNextRender}** — if the frame counter cannot
 * be read or never advances, the fallback returns after the state wait plus a
 * `minFrames * 100`ms buffer, which on a starved page can be worth far fewer
 * than `minFrames` frames. Since the callers are almost all screenshot
 * comparisons, that give-up emits one `console.warn` naming which branch it
 * took rather than passing off a possibly pre-render capture in silence.
 *
 * @param page - Playwright page
 * @param minFrames - Minimum number of frames to render (used as multiplier for fallback)
 * @param timeout - Cap on each INDIVIDUAL wait, not on the call — see the
 *   note on {@link raceEvaluate}. The frame-counter read, the `renderOnce()`
 *   kick and the frame wait are each capped at `Math.min(timeout, 3000)`; the
 *   state-based fallback keeps the full `timeout` (and still throws, as it
 *   always has, if the viewer never settles).
 */
export async function waitForRenderStable(
  page: Page,
  minFrames = 3,
  timeout = 10000
): Promise<void> {
  // Each `page.evaluate` below is bounded like the frame wait itself: an
  // unbounded evaluate on a frame-starved page outlives the whole test budget
  // instead of letting this helper fall back (see `raceEvaluate`).
  const evaluateTimeout = Math.min(timeout, 3000);

  // Snapshot the current frame BEFORE the wait. The previous version
  // checked `frame >= minFrames` against the lifetime counter, so once
  // the initial paint exceeded `minFrames` (which it does within
  // milliseconds of viewer startup), the helper would resolve
  // immediately on every subsequent call — ignoring any post-action
  // paints. Screenshot tests captured pre-action state.
  //
  // An unanswered read yields `null`, which routes to the state-based
  // fallback exactly like a missing debug interface does.
  const start = await raceEvaluate(
    page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      // WebGLRenderer exposes `info.render.frame`; WebGPURenderer
      // exposes `info.frame`. Probe both so the helper works on
      // either backend.
      const info = debug?.renderer?.info;
      const frame = info?.render?.frame ?? info?.frame;
      return typeof frame === 'number' ? frame : null;
    }),
    evaluateTimeout,
    null
  );

  if (start !== null) {
    // Kick the animation loop in case it's idle (auto-paused after ~2s
    // of inactivity); without this, the frame counter never advances
    // and we'd always fall through to the time-based fallback.
    // Best-effort: if the page is too busy to answer, carry on to the frame
    // wait (and then the fallback) rather than stalling here.
    await raceEvaluate(
      page.evaluate(() => {
        const debug = (window as any).__luxarDebug;
        debug?.renderOnce?.();
      }),
      evaluateTimeout,
      undefined
    );

    const target = start + minFrames;
    try {
      await page.waitForFunction(
        (t: number) => {
          const debug = (window as any).__luxarDebug;
          const info = debug?.renderer?.info;
          const frame = info?.render?.frame ?? info?.frame;
          return typeof frame === 'number' && frame >= t;
        },
        target,
        { timeout: Math.min(timeout, 3000) }
      );
      return;
    } catch {
      // Frame counter didn't advance (loop truly stopped) — fall
      // through to the state-based wait.
    }
  }

  // Fallback: wait for data to load + buffer time for rendering.
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      const state = debug?.getState?.();
      return state && !state.isLoading && state.initialized;
    },
    null,
    { timeout }
  );
  // Additional buffer for GPU to render frames
  const buffer = minFrames * 100;
  await page.waitForTimeout(buffer);

  // Report the give-up rather than handing a screenshot test a capture that may
  // predate the paint it was pacing itself on.
  const branch =
    start === null
      ? `the frame counter could not be read within ${evaluateTimeout}ms`
      : `the frame counter was readable but never advanced by ${minFrames} within ${Math.min(timeout, 3000)}ms`;
  console.warn(
    `[⚠️] [E2E waitForRenderStable] asked for ${minFrames} frame(s) at timeout=${timeout}ms, ` +
      `but ${branch}; took the state-based fallback + ${buffer}ms buffer instead, ` +
      `so ${minFrames} frame(s) were NOT observed`
  );
}

/**
 * Race an in-flight `page.evaluate` against a deadline.
 *
 * Playwright dispatches `page.evaluate` with no timeout of its own — neither
 * `actionTimeout` nor `setDefaultTimeout` bounds it, only the whole test's
 * budget does. So on a frame-starved page (a loaded workstation drops the
 * viewer from ~30 FPS to ~4) an evaluate can sit unanswered until the test
 * itself expires, instead of letting the caller fall back.
 *
 * This bounds ONE wait, not a whole helper. A helper that makes several
 * bounded calls in sequence takes the SUM of their caps plus whatever its
 * fallback costs, so its total runtime can legitimately exceed the single
 * `timeout` value the caller passed. Read a helper's `timeout` parameter as
 * "the cap on each individual wait", never as "the cap on the call".
 *
 * WHAT STARVES IT is main-thread task starvation, not anything GL-specific,
 * so no probe is exempt from needing a bound (#1651). Instrumented on an
 * idle box, a trivial `page.evaluate(() => 'ok')` went unanswered for its
 * 5 s deadline twelve times in a row while `requestAnimationFrame` kept
 * ticking (34 → 157) and the renderer's own counter advanced 426 → 672,
 * with `visibilityState === 'visible'` and the WebGL context never lost; an
 * in-page `setInterval(..., 1000)` fired twice over a 28 s window inside that
 * stall. A saturated software-rendering rAF loop starves the lower-priority
 * task sources — in-page timers and Playwright's `Runtime.callFunctionOn`
 * round trip — for tens of seconds while rendering continues throughout.
 * Corollary for `onTimeout`: pass a sentinel the in-page function can never
 * return, so the caller can tell a real answer from a missed deadline, and
 * decide at the call site what a missed deadline MEANS — a detector whose
 * answer the test asserts on must say it could not run rather than fall back
 * to a value that reads as a pass.
 *
 * The timer is always cleared, so a resolved race leaves no handle keeping the
 * Node process alive. A rejection that arrives after the deadline is absorbed
 * by `Promise.race` (which has already settled) rather than going unhandled.
 *
 * Unit-tested in `src/tests/unit/tests/e2e-helpers-race-evaluate.test.ts`.
 * Prefer a wait helper built on it where one fits; a spec holding its own
 * `page.evaluate` calls it directly (see `webgl-errors.spec.ts`), because a
 * bare evaluate has no other bound.
 *
 * @param evaluation - The already-started `page.evaluate` promise.
 * @param timeout - Deadline in ms.
 * @param onTimeout - Value to resolve with if the deadline wins.
 */
export async function raceEvaluate<T>(
  evaluation: Promise<T>,
  timeout: number,
  onTimeout: T
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      evaluation,
      new Promise<T>((resolve) => {
        timer = setTimeout(() => resolve(onTimeout), timeout);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Wait for the next N render frames to complete
 *
 * Records the current renderer frame counter and waits until it advances
 * by the specified number of frames. This replaces most `waitForTimeout(100-500)`
 * calls after user actions (key presses, clicks, etc.) that trigger re-renders.
 *
 * **Never fails on give-up** — falls back to a state-based wait + a short time
 * buffer if the frame counter is unavailable or if the animation loop is idle
 * (auto-paused after inactivity), and returns normally. On a frame-starved page
 * that buffer can be worth one or two frames rather than N. The give-up is not
 * silent: it emits one `console.warn` naming which branch it took, and the
 * boolean return says the same thing to a caller that wants to react.
 *
 * A strict, throwing variant was tried and reverted: the animation controller
 * auto-pauses after ~2s of inactivity and this helper's kick is a single
 * `renderOnce()`, so a request for several frames on an idle loop cannot be
 * satisfied by construction. Failing there turns the healthy path red, which is
 * exactly why the state-based fallback exists.
 *
 * @param page - Playwright page
 * @param frames - Number of frames to wait for (default: 2)
 * @param timeout - Cap on each INDIVIDUAL wait in ms (default: 5000), not on
 *   the call: the frame-counter read, the `renderOnce()` kick and the frame
 *   wait are each capped at `Math.min(timeout, 3000)` and the state fallback
 *   at `timeout`, so a fully starved page costs roughly the sum of those legs.
 * @returns `true` when the counter was observed to advance by `frames`,
 *   `false` when the helper gave up and used the fallback.
 */
export async function waitForNextRender(page: Page, frames = 2, timeout = 5000): Promise<boolean> {
  // Each `page.evaluate` below is bounded by the same budget as the frame
  // wait, so a page too starved to answer takes the state-based fallback
  // instead of hanging the whole test (see `raceEvaluate`).
  const evaluateTimeout = Math.min(timeout, 3000);

  // Try to read the current frame counter. An unanswered read yields `null`,
  // which routes to the fallback exactly like a missing debug interface does.
  const currentFrame = await raceEvaluate(
    page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      // WebGLRenderer exposes `info.render.frame`; WebGPURenderer
      // exposes `info.frame`. Probe both so the helper works on
      // either backend.
      const info = debug?.renderer?.info;
      const frame = info?.render?.frame ?? info?.frame;
      return typeof frame === 'number' ? frame : null;
    }),
    evaluateTimeout,
    null
  );

  if (currentFrame !== null) {
    // Force-trigger a render in case the animation loop is idle (auto-paused).
    // The animation controller pauses after ~2s of inactivity, which means
    // the frame counter stops incrementing. Calling renderOnce() kicks it.
    // Best-effort: if the page is too busy to answer, carry on to the frame
    // wait (and then the fallback) rather than stalling here.
    await raceEvaluate(
      page.evaluate(() => {
        const debug = (window as any).__luxarDebug;
        debug?.renderOnce?.();
      }),
      evaluateTimeout,
      undefined
    );

    // Use frame counter for precise wait, with a shorter timeout so we can
    // fall back gracefully if the animation loop is truly stopped.
    const targetFrame = currentFrame + frames;
    try {
      await page.waitForFunction(
        (target: number) => {
          const debug = (window as any).__luxarDebug;
          const info = debug?.renderer?.info;
          const frame = info?.render?.frame ?? info?.frame;
          return typeof frame === 'number' && frame >= target;
        },
        targetFrame,
        { timeout: Math.min(timeout, 3000) }
      );
      return true;
    } catch {
      // Frame counter didn't advance (animation loop idle) — fall through to time-based wait
    }
  }

  // Fallback: wait for stable initialized state + time buffer
  await page
    .waitForFunction(
      () => {
        const debug = (window as any).__luxarDebug;
        const state = debug?.getState?.();
        return state && !state.isLoading && state.initialized;
      },
      null,
      { timeout }
    )
    .catch(() => {
      // State never became stable — continue anyway
    });
  const buffer = Math.max(frames * 50, 200);
  await page.waitForTimeout(buffer);

  // The give-up is reported, never silent: a test that reads as passing while
  // having observed almost no frames is the failure mode worth naming.
  const branch =
    currentFrame === null
      ? `the frame counter could not be read within ${Math.min(timeout, 3000)}ms`
      : `the frame counter was readable but never advanced by ${frames} within ${Math.min(timeout, 3000)}ms`;
  console.warn(
    `[⚠️] [E2E waitForNextRender] asked for ${frames} frame(s) at timeout=${timeout}ms, ` +
      `but ${branch}; took the state-based fallback + ${buffer}ms buffer instead, ` +
      `so ${frames} frame(s) were NOT observed`
  );
  return false;
}

/**
 * Wait for an animation step on a specific dimension
 *
 * Records the current slice position for the given dimension index and waits
 * until the position changes, indicating an animation step has completed.
 * This replaces post-animation-key timeouts where a dimension value is expected
 * to change (e.g., after pressing `]` or starting an animation with `k`).
 *
 * @param page - Playwright page
 * @param dimIndex - The dimension index to monitor for position changes
 * @param timeout - Maximum wait time in ms (default: 3000)
 */
export async function waitForAnimationStep(
  page: Page,
  dimIndex: number,
  timeout = 3000
): Promise<void> {
  // Record the current slice position for this dimension
  const currentPosition = await page.evaluate((idx: number) => {
    const debug = (window as any).__luxarDebug;
    const state = debug?.getState?.();
    if (state?.slicePosition && idx < state.slicePosition.length) {
      return state.slicePosition[idx];
    }
    // Fallback: try sceneDimsManager
    const dims = debug?.sceneDimsManager?.getDims?.();
    if (dims?.currentStep && idx < dims.currentStep.length) {
      return dims.currentStep[idx];
    }
    return null;
  }, dimIndex);

  if (currentPosition !== null) {
    // Wait until the position changes
    await page.waitForFunction(
      ({ idx, prevPos }: { idx: number; prevPos: number }) => {
        const debug = (window as any).__luxarDebug;
        const state = debug?.getState?.();
        if (state?.slicePosition && idx < state.slicePosition.length) {
          return state.slicePosition[idx] !== prevPos;
        }
        const dims = debug?.sceneDimsManager?.getDims?.();
        if (dims?.currentStep && idx < dims.currentStep.length) {
          return dims.currentStep[idx] !== prevPos;
        }
        return false;
      },
      { idx: dimIndex, prevPos: currentPosition },
      { timeout }
    );
  } else {
    // Fallback: if we can't read position, wait for a render frame
    await waitForNextRender(page, 2, timeout);
  }
}

/**
 * Wait for cache state to settle (no changes across consecutive polls).
 *
 * Used for OPFS write settling: the L2 store debounces writes asynchronously,
 * so reading stats immediately after a load returns mid-flight values. Two
 * (or more) consecutive identical reads of the cache size signature indicate
 * pending writes have flushed.
 *
 * Returns gracefully if the L2 layer is unavailable (no OPFS in this browser).
 */
export async function waitForCacheStable(
  page: Page,
  options: { stableReads?: number; pollMs?: number; timeout?: number } = {}
): Promise<void> {
  const stableReads = options.stableReads ?? 2;
  const pollMs = options.pollMs ?? 250;
  const timeout = options.timeout ?? 8000;
  const deadline = Date.now() + timeout;

  type Sig = string;
  const readSignature = async (): Promise<Sig | 'no-l2'> =>
    page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      const stats = await debug?.cache?.getStats?.();
      if (!stats || stats.error) return 'no-l2';
      const l1 = stats.l1 ?? {};
      const l2 = stats.l2;
      if (!l2 || (l2.size === 0 && l2.count === 0 && l2.writes === 0)) return 'no-l2';
      return `${l1.metadataCount ?? 0}|${l1.chunksCount ?? 0}|${l2.size}|${l2.count}`;
    });

  let prev: Sig | 'no-l2' | null = null;
  let consecutiveMatches = 0;

  while (Date.now() < deadline) {
    const sig = await readSignature();
    if (sig === 'no-l2') return; // L2 unavailable → nothing to wait for
    if (sig === prev) {
      consecutiveMatches += 1;
      if (consecutiveMatches >= stableReads - 1) return;
    } else {
      consecutiveMatches = 0;
    }
    prev = sig;
    await page.waitForTimeout(pollMs);
  }
}

/**
 * Wait until a predicate over accumulated WebGL errors returns true.
 *
 * `getWebGLErrors` *drains* the GL queue on each call, so this helper
 * accumulates errors across polls into a closure-local buffer and tests the
 * predicate against the union. Use for "wait until at least one error of type
 * X appears" patterns. For the inverse ("there should be no errors"), use
 * `waitForRenderStable` then a single `getWebGLErrors` read.
 *
 * Each read is bounded by whatever is left of `timeout`, no FURTHER read is
 * dispatched into a remainder smaller than one `pollMs`, and a read that consumed
 * the whole remainder is not followed by one more `pollMs` sleep past the deadline
 * (#1726). `getWebGLErrors`
 * is a bare `page.evaluate`, which has no timeout of its own, so the FIRST
 * unanswered read used to blow straight through this budget to the whole test's.
 * The helper does not throw on GIVE-UP — it returns data, and an empty result is a
 * legitimate one — and a give-up emits one `console.warn` saying whether any read
 * answered at all, so an unsatisfied predicate cannot be mistaken for "the page
 * was healthy and produced no such error". It is not throw-free, though: a read
 * that REJECTS propagates (a closed page, a destroyed execution context, an
 * in-page throw out of `getWebGLErrors`), as does the trailing
 * `page.waitForTimeout` on a page that goes away mid-poll.
 *
 * Enforcing the budget has a cost worth stating plainly: `getWebGLErrors` DRAINS
 * the GL queue in-page, so an answer arriving after the deadline is discarded
 * along with whatever real errors it already cleared, and a later read in the
 * same spec then sees a clean queue. That is what a budget means for a helper
 * that returns data; the proper fix is bounding `getWebGLErrors` itself — the
 * same treatment #1651 gave `getLuxarState` and `getConsoleMessages` — which
 * reaches 21 direct call sites across 11 spec files and is left out of scope
 * here rather than tracked anywhere, #1651 being closed. Not dispatching a
 * FURTHER read into a sliver of
 * budget it cannot answer within keeps the tail case out of that trap; the FIRST
 * read is exempt from that skip, so any POSITIVE budget — however short — still
 * gets one tightly-bounded read, rather than returning `[]` without having read the
 * GL queue even once, which would be the very silence this helper's warn exists to
 * prevent. That exemption is also the one case where the call can outlast its
 * budget: with `timeout` under `pollMs` the first read may answer early and is then
 * followed by a full `pollMs` before the loop guard re-checks, so a 50 ms budget can
 * report ~100 ms. A NON-POSITIVE `timeout` gets zero reads: the loop guard fails on
 * entry, and the warn then says nothing was checked, which is exactly true.
 * Documented rather than special-cased, since no caller asks for a zero budget.
 *
 * Unit-tested in `src/tests/unit/tests/e2e-helpers-silent-waits.test.ts`.
 */
export async function waitForWebGLError(
  page: Page,
  predicate: (errors: string[]) => boolean,
  options: { timeout?: number; pollMs?: number } = {}
): Promise<string[]> {
  const timeout = options.timeout ?? 5000;
  const pollMs = options.pollMs ?? 100;
  const startTime = Date.now();
  const deadline = startTime + timeout;
  const accumulated: string[] = [];
  let answered = false;
  let reads = 0;

  while (Date.now() < deadline) {
    // Once one read has gone out, never dispatch another into a remainder shorter
    // than one poll interval: it would drain the GL queue in-page and then have
    // its answer discarded for missing the deadline, taking the real errors with
    // it. The FIRST read is exempt — a `timeout` below `pollMs` would otherwise
    // return `[]` having checked nothing at all, which is strictly worse than one
    // tightly-bounded read.
    const remaining = deadline - Date.now();
    if (reads > 0 && remaining < pollMs) break;

    // The sentinel is a FRESH Node-allocated array compared by identity, never a
    // value: an EMPTY list is a real answer here (a page with a clean GL queue
    // returns one on nearly every poll), so nothing about a returned array's
    // contents could stand for a missed deadline — see `raceEvaluate`.
    const readTimedOut: string[] = [];
    reads += 1;
    const errs = await raceEvaluate<string[]>(getWebGLErrors(page), remaining, readTimedOut);

    // An unanswered read contributes nothing: the loop must not read a missed
    // deadline as a clean GL queue.
    if (errs !== readTimedOut) {
      answered = true;
      accumulated.push(...errs);
      if (predicate(accumulated)) return accumulated;
    }

    // A read that consumed the whole remainder must not be followed by one more
    // `pollMs`, which would make a 1000 ms budget report 1100 ms elapsed in the
    // warn below.
    if (Date.now() >= deadline) break;
    await page.waitForTimeout(pollMs);
  }

  console.warn(
    `[⚠️] [E2E waitForWebGLError] the predicate was never satisfied within its ${timeout}ms ` +
      `budget (${Date.now() - startTime}ms actually elapsed, ${accumulated.length} ` +
      'error(s) accumulated); ' +
      (answered
        ? 'the page did answer the GL-error reads, so this looks like a genuine absence'
        : 'no GL-error read was answered within the budget, so nothing was actually checked')
  );
  return accumulated;
}

export async function assertNoConsoleErrors(
  page: Page,
  allowedPatterns: RegExp[] = []
): Promise<void> {
  const messages = await getConsoleMessages(page);

  // Filter out allowed errors
  const actualErrors = messages.errors.filter((err) => {
    return !allowedPatterns.some((pattern) => pattern.test(err));
  });

  if (actualErrors.length > 0) {
    // Use global console, not the messages variable

    console.error('[E2E Test] Console Errors Detected:');
    actualErrors.forEach((err, i) => {
      console.error(`  ${i + 1}. ${err}`);
    });
    throw new Error(
      `Console errors detected: ${actualErrors.length} errors.\n` +
        `First error: ${actualErrors[0]}\n` +
        'See console output above for full list.'
    );
  }
}

/**
 * Dismiss the dataset browser modal if visible.
 *
 * When navigating to `/?debug` without a dataset, the app shows a
 * dataset browser dialog (aria-modal) that intercepts all pointer and
 * keyboard events. Tests that need to interact with the canvas or use
 * keyboard shortcuts must dismiss it first.
 *
 * @param page - Playwright page
 */
export async function dismissDatasetBrowser(page: Page): Promise<void> {
  const isVisible = await page.evaluate(() => {
    const browser = document.querySelector(
      '.luxar-dataset-browser, .dataset-browser, #luxar-dataset-browser'
    );
    return browser ? getComputedStyle(browser).display !== 'none' : false;
  });

  if (!isVisible) return;

  // Press Escape so the browser routes through PanelCoordinator.closeAll()
  // → datasetBrowser.close(), which keeps LuxarApp.datasetBrowser in sync.
  // Yanking the DOM node directly bypasses that and hides the exact
  // bug a regression check would catch.
  //
  // Escape is exempted from the typing-input guard in
  // `InputHandler.onKeyDown`, so it reliably reaches PanelCoordinator
  // regardless of focus location (manual-path field, debug-console
  // filter, etc.). If Escape stops reaching the coordinator, the hard
  // timeout here is the right signal.
  await page.keyboard.press('Escape');
  await page.waitForFunction(
    () => {
      const el = document.querySelector(
        '.luxar-dataset-browser, .dataset-browser, #luxar-dataset-browser'
      );
      return !el || getComputedStyle(el).display === 'none';
    },
    { timeout: 2000 }
  );
}

/**
 * Focus the canvas for keyboard/mouse interaction.
 *
 * Dismisses the dataset browser if visible, then clicks the canvas.
 * Use this instead of bare `page.click('canvas')` which can timeout
 * when the dataset browser modal intercepts pointer events.
 *
 * @param page - Playwright page
 */
export async function focusCanvas(page: Page): Promise<void> {
  await dismissDatasetBrowser(page);
  try {
    await page.click('canvas', { timeout: 3000 });
  } catch {
    // Canvas click failed (may not exist yet) — try focusing the page body instead
    await page.evaluate(() => {
      const canvas = document.querySelector('canvas');
      if (canvas) canvas.focus();
    });
  }
}

/**
 * `true` when keyboard focus is on a "typing surface" in the page.
 *
 * Thin wrapper around {@link isTypingSurfaceInPage} — the predicate lives in
 * `page-predicates.ts` (self-contained, so Playwright can serialize it into
 * the page) and is kept honest by a parity unit test against the production
 * `utils/dom/focus.ts::isTypingInInput`.
 *
 * @param page - Playwright page
 */
export async function isFocusOnTypingSurface(page: Page): Promise<boolean> {
  return await page.evaluate(isTypingSurfaceInPage);
}

// ============================================================================
// Standardized Debug Interface Accessors
// ============================================================================

/**
 * Get the input handler from the debug interface.
 * Standardizes access pattern: debug.app.inputHandler (canonical path).
 */
export async function getInputHandler(page: Page): Promise<any> {
  return await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    return debug?.app?.inputHandler ?? debug?.inputHandler ?? null;
  });
}

/**
 * Get the animation manager from the debug interface.
 * Standardizes access: debug.app.inputHandler.animationManager.
 */
export async function getAnimationManager(page: Page): Promise<any> {
  return await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    const ih = debug?.app?.inputHandler ?? debug?.inputHandler;
    return ih?.animationManager ?? null;
  });
}

/**
 * Get the scene dims manager from the debug interface.
 * Canonical access: `debug.sceneDimsManager` (exposed directly by
 * `app.ts`). The manager is not a child of input-handler in the
 * debug surface.
 */
export async function getSceneDimsManager(page: Page): Promise<any> {
  return await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    return debug?.sceneDimsManager ?? null;
  });
}

// ============================================================================
// Layer, Material & Scene Inspection Helpers
// ============================================================================

/**
 * Open the layers panel (press L, wait for it to appear).
 * If already open, does nothing.
 */
export async function openLayersPanel(page: Page): Promise<void> {
  const alreadyOpen = await page
    .locator('.luxar-layers-panel')
    .isVisible()
    .catch(() => false);

  if (!alreadyOpen) {
    await focusCanvas(page);
    await page.keyboard.press('l');
    await page.waitForSelector('.luxar-layers-panel', { state: 'visible', timeout: 5000 });
  }
}

/**
 * Get material state for a named Three.js object.
 * Returns blending, depth, and transparency properties.
 */
export async function getLayerMaterialState(
  page: Page,
  objectName: string
): Promise<{
  found: boolean;
  blending: number;
  depthTest: boolean;
  depthWrite: boolean;
  transparent: boolean;
  visible: boolean;
} | null> {
  return await page.evaluate((name) => {
    const debug = (window as any).__luxarDebug;
    if (!debug?.scene) return null;

    let result: any = null;
    debug.scene.traverse((obj: any) => {
      if (result) return;
      if (obj.name === name && obj.material) {
        result = {
          found: true,
          blending: obj.material.blending,
          depthTest: obj.material.depthTest,
          depthWrite: obj.material.depthWrite,
          transparent: obj.material.transparent,
          visible: obj.visible,
        };
      }
    });
    return result;
  }, objectName);
}

/**
 * Get all named objects in the Three.js scene.
 */
export async function getSceneObjectNames(page: Page): Promise<string[]> {
  return await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    if (!debug?.scene) return [];
    const names: string[] = [];
    debug.scene.traverse((obj: any) => {
      if (obj.name) names.push(obj.name);
    });
    return names;
  });
}

/**
 * Get post-processing state from the debug interface.
 */
export async function getPostProcessingState(page: Page): Promise<{
  hasPostProcessing: boolean;
  bloomStrength: number | null;
  exposure: number | null;
  vignetteEnabled: boolean | null;
  fxaaEnabled: boolean | null;
} | null> {
  return await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    const rc = debug?.renderingControls;
    if (!rc) return null;
    const settings = rc.settings;
    return {
      hasPostProcessing: !!debug.postProcessing,
      bloomStrength: settings?.bloomStrength ?? null,
      exposure: settings?.exposure ?? null,
      vignetteEnabled: settings?.vignetteEnabled ?? null,
      fxaaEnabled: settings?.fxaaEnabled ?? null,
    };
  });
}

/**
 * Perform a Ctrl+Scroll interaction (changes FOV in Luxar).
 * Uses a custom WheelEvent with ctrlKey=true since Playwright's
 * keyboard.down('Control') + mouse.wheel() doesn't set ctrlKey on the event.
 */
export async function ctrlScroll(page: Page, deltaY: number): Promise<void> {
  await page.evaluate((dy) => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return;
    canvas.dispatchEvent(new WheelEvent('wheel', { deltaY: dy, ctrlKey: true, bubbles: true }));
  }, deltaY);
  await waitForNextRender(page);
}

/**
 * Perform a Shift+Scroll interaction (rotates view axis in Luxar).
 */
export async function shiftScroll(page: Page, deltaY: number): Promise<void> {
  await page.evaluate((dy) => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return;
    canvas.dispatchEvent(new WheelEvent('wheel', { deltaY: dy, shiftKey: true, bubbles: true }));
  }, deltaY);
  await waitForNextRender(page);
}

/**
 * Validate texture-backed point storage for all Points geometry in the scene.
 * Returns per-cloud validation results.
 */
export async function validateSceneAttributes(page: Page): Promise<
  Array<{
    name: string;
    positionCount: number;
    colorCount: number;
    radiusCount: number;
    sharpnessCount: number;
    drawRangeCount: number;
    visibleInstanceCount: number;
    aligned: boolean;
    hasNaN: boolean;
    hasInfinity: boolean;
  }>
> {
  return await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    if (!debug?.scene) return [];

    const results: any[] = [];
    debug.scene.traverse((obj: any) => {
      // Points are THREE.Mesh with instanced quad geometry. Per-point data
      // is texture-backed: an RGBA32F element texture holds 12 floats
      // (3 texels) per point — center xyz [0..2], radius [3], color rgb
      // [4..6], sharpness [7], scalar [8], alpha [9]. The only
      // per-instance attribute is aSortedIndex (identity by default; a
      // depth-sort permutation in effective-normal mode).
      if (obj.userData?.nodeType !== 'points') return;
      const texData = obj.geometry?.userData?.elementTexture?.image?.data;
      if (!texData) return;

      const STRIDE = 12;
      const texelCapacity = Math.floor(texData.length / STRIDE);
      const sortedIndex = obj.geometry.attributes?.aSortedIndex;
      const dr = obj.geometry.drawRange;
      const presence = obj.geometry?.userData;

      // Field presence comes from the texel writers' userData stamps (the
      // zarr node attrs carry no has_colors/has_radii/has_sharpness); the
      // texel buffer allocates every slot, so all present fields share the
      // same per-point capacity.
      const posCount = texelCapacity;
      const colCount = presence?.hasColors ? texelCapacity : -1;
      const radCount = presence?.hasRadii ? texelCapacity : -1;
      const shpCount = presence?.hasSharpness ? texelCapacity : -1;
      const drawCount = dr.count < Infinity ? Math.min(dr.count, posCount) : posCount;
      // instanceCount is the visible point count; the texel buffer (and
      // aSortedIndex) may be over-allocated for pooled geometries.
      const visibleCount = obj.geometry.isInstancedBufferGeometry
        ? obj.geometry.instanceCount
        : texelCapacity;
      const visibleInstanceCount = Math.min(visibleCount, texelCapacity);

      // Check for NaN/Infinity in positions (sample first 1000 visible
      // instances). Centers live at texel slots [i*12 .. i*12+2].
      let hasNaN = false;
      let hasInfinity = false;
      const sampleCount = Math.min(visibleInstanceCount, 1000);
      for (let i = 0; i < sampleCount; i++) {
        const x = texData[i * STRIDE];
        const y = texData[i * STRIDE + 1];
        const z = texData[i * STRIDE + 2];
        if (Number.isNaN(x) || Number.isNaN(y) || Number.isNaN(z)) hasNaN = true;
        if (
          (!Number.isNaN(x) && !Number.isFinite(x)) ||
          (!Number.isNaN(y) && !Number.isFinite(y)) ||
          (!Number.isNaN(z) && !Number.isFinite(z))
        )
          hasInfinity = true;
      }

      // Check alignment: the texel buffer and the aSortedIndex attribute
      // must both cover every visible instance.
      const aligned =
        texelCapacity >= visibleCount && !!sortedIndex && sortedIndex.count >= visibleCount;

      results.push({
        name: obj.name || 'unnamed',
        positionCount: posCount,
        colorCount: colCount,
        radiusCount: radCount,
        sharpnessCount: shpCount,
        drawRangeCount: drawCount,
        visibleInstanceCount,
        aligned,
        hasNaN,
        hasInfinity,
      });
    });

    return results;
  });
}

/**
 * Assert no shader compile / link / attribute / uniform errors are
 * present in the buffered console messages. WebGL surfaces shader
 * issues asynchronously (the browser logs to console), so this is the
 * canonical way to detect them after a render.
 *
 * The patterns match strings emitted by Chromium's WebGL implementation
 * for compile/link failures and missing-attribute warnings. We only scan
 * errors/warnings (not normal info logs) so routine material names such as
 * `point_glsl_additive...` don't become false positives.
 */
export async function assertNoShaderErrors(page: Page): Promise<void> {
  const messages = await getConsoleMessages(page);
  const all = [...messages.errors, ...messages.warnings];
  const shaderErrPattern =
    /ERROR:\s*0:|THREE\.WebGLProgram|shader\s*error|GLSL\s*(error|failure|failed)|attribute.*not\s*found|uniform.*not\s*found|fragment\s*shader.*not\s*compiled|vertex\s*shader.*not\s*compiled|program\s*link|invalid_operation/i;
  const offending = all.filter((m) => shaderErrPattern.test(m));
  if (offending.length > 0) {
    throw new Error(
      `shader/GLSL errors detected in browser console (${offending.length} message(s)):\n` +
        offending.slice(0, 8).join('\n')
    );
  }
}

/** A single RGBA pixel sample (0-255 per channel) read back from a rendered element. */
export interface SampledPixel {
  r: number;
  g: number;
  b: number;
  a: number;
}

/**
 * Summary statistics for a rendered element's pixels: sampled region size, the
 * non-black threshold used, the count of pixels above it, and the brightest
 * pixel found — used by E2E assertions to confirm something was actually drawn.
 */
export interface ElementPixelStats {
  width: number;
  height: number;
  threshold: number;
  nonBlackPixels: number;
  brightest: SampledPixel;
}

/** Normalized element region, with every edge in `[0, 1]`. */
export interface ElementPixelRegion {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/**
 * Project the visible bounds of selected geometry types through the live camera.
 * The returned normalized rectangle can be passed to {@link getElementPixelStats}
 * so DOM chrome elsewhere in the canvas crop cannot satisfy a render assertion.
 */
export async function getProjectedGeometryRegion(
  page: Page,
  nodeTypes: readonly string[],
  padding = 0.02
): Promise<ElementPixelRegion> {
  return await page.evaluate(
    ({ wantedTypes, regionPadding }) => {
      const debug = (window as unknown as { __luxarDebug?: Record<string, any> }).__luxarDebug;
      const scene = debug?.scene;
      const camera = debug?.camera ?? debug?.app?.sceneManager?.camera;
      if (!scene || !camera) throw new Error('getProjectedGeometryRegion: debug scene unavailable');

      scene.updateMatrixWorld(true);
      camera.updateMatrixWorld(true);
      const wanted = new Set(wantedTypes);
      const probe = camera.position.clone();
      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      let matched = 0;

      scene.traverse((object: any) => {
        if (!object.visible || !wanted.has(object.userData?.nodeType)) return;
        const geometry = object.geometry;
        if (!geometry) return;
        if (!geometry.boundingBox) geometry.computeBoundingBox?.();
        const box = geometry.boundingBox?.clone();
        if (!box || box.isEmpty()) return;
        box.applyMatrix4(object.matrixWorld);
        matched++;
        for (const x of [box.min.x, box.max.x]) {
          for (const y of [box.min.y, box.max.y]) {
            for (const z of [box.min.z, box.max.z]) {
              probe.set(x, y, z).project(camera);
              minX = Math.min(minX, probe.x);
              maxX = Math.max(maxX, probe.x);
              minY = Math.min(minY, probe.y);
              maxY = Math.max(maxY, probe.y);
            }
          }
        }
      });

      if (matched === 0) {
        throw new Error(
          `getProjectedGeometryRegion: no visible geometry for ${wantedTypes.join(', ')}`
        );
      }

      return {
        left: Math.max(0, minX * 0.5 + 0.5 - regionPadding),
        right: Math.min(1, maxX * 0.5 + 0.5 + regionPadding),
        top: Math.max(0, 1 - (maxY * 0.5 + 0.5) - regionPadding),
        bottom: Math.min(1, 1 - (minY * 0.5 + 0.5) + regionPadding),
      };
    },
    { wantedTypes: [...nodeTypes], regionPadding: padding }
  );
}

/**
 * Read pixels from a rendered element at fractional coordinates in
 * `[0,1]`. Returns RGBA byte values from the *visible screenshot*.
 *
 * Do not read WebGL canvases by drawing the canvas into a 2D canvas:
 * with `preserveDrawingBuffer: false` Chromium is allowed to clear the
 * WebGL drawing buffer after compositing, which produced all-zero pixels
 * in shader smoke tests even when the screenshot was visibly rendered.
 * Capturing the element screenshot samples the composited output instead
 * and is therefore the right primitive for E2E visual smoke tests.
 */
async function captureElementScreenshotDataUrl(page: Page, selector: string): Promise<string> {
  const element = page.locator(selector).first();
  await element.waitFor({ state: 'visible' });
  const png = await element.screenshot({ animations: 'disabled' });
  return `data:image/png;base64,${png.toString('base64')}`;
}

/** A whole decoded frame: raw interleaved RGBA bytes plus its dimensions. */
export interface CanvasFrameRGBA {
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
}

/**
 * Capture a rendered element ONCE and return its full-frame RGBA bytes.
 *
 * Use this when a spec needs whole-image analysis (histograms, per-region
 * statistics) rather than a handful of point samples — measuring several
 * regions of one identical frame is the point, and re-screenshotting per
 * region would let an unrelated frame difference masquerade as a defect.
 *
 * The screenshot route (rather than a direct `gl.readPixels`) is required
 * for the reason spelled out on `captureElementScreenshotDataUrl` above:
 * with `preserveDrawingBuffer: false` the WebGL drawing buffer may already
 * be cleared. The PNG is decoded in-page and handed back as base64 RGBA so
 * the whole frame crosses the CDP bridge exactly once.
 *
 * @param page Playwright page.
 * @param selector Element to capture; defaults to the viewer canvas.
 * @returns Decoded pixel dimensions and the interleaved RGBA buffer.
 */
export async function captureCanvasRGBA(page: Page, selector = 'canvas'): Promise<CanvasFrameRGBA> {
  const dataUrl = await captureElementScreenshotDataUrl(page, selector);

  const decoded = await page.evaluate(async (url: string) => {
    const img = new Image();
    img.decoding = 'sync';
    const loaded = new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('captureCanvasRGBA: failed to decode screenshot'));
    });
    img.src = url;
    await loaded;

    const width = img.naturalWidth;
    const height = img.naturalHeight;
    if (width <= 0 || height <= 0) {
      throw new Error(`captureCanvasRGBA: empty screenshot ${width}x${height}`);
    }

    const off = document.createElement('canvas');
    off.width = width;
    off.height = height;
    const ctx = off.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('captureCanvasRGBA: 2D context unavailable');
    ctx.drawImage(img, 0, 0);
    const data = ctx.getImageData(0, 0, width, height).data;

    // Chunked: String.fromCharCode.apply blows the stack on a multi-MB buffer.
    let binary = '';
    const CHUNK = 0x8000;
    for (let i = 0; i < data.length; i += CHUNK) {
      binary += String.fromCharCode.apply(null, data.subarray(i, i + CHUNK) as unknown as number[]);
    }
    return { width, height, base64: btoa(binary) };
  }, dataUrl);

  const buf = Buffer.from(decoded.base64, 'base64');
  return {
    width: decoded.width,
    height: decoded.height,
    rgba: new Uint8ClampedArray(buf.buffer, buf.byteOffset, buf.byteLength),
  };
}

export async function samplePixelsAt(
  page: Page,
  selector: string,
  offsets: Array<[number, number]>
): Promise<SampledPixel[]> {
  if (offsets.length === 0) return [];

  const dataUrl = await captureElementScreenshotDataUrl(page, selector);

  return await page.evaluate(
    async ({ url, points }) => {
      const img = new Image();
      img.decoding = 'sync';
      const loaded = new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('samplePixelsAt: failed to decode screenshot'));
      });
      img.src = url;
      await loaded;

      const width = img.naturalWidth;
      const height = img.naturalHeight;
      if (width <= 0 || height <= 0) {
        throw new Error(`samplePixelsAt: empty screenshot ${width}x${height}`);
      }

      const off = document.createElement('canvas');
      off.width = width;
      off.height = height;
      const ctx = off.getContext('2d', { willReadFrequently: true });
      if (!ctx) throw new Error('samplePixelsAt: 2D context unavailable');
      ctx.drawImage(img, 0, 0);

      return points.map(([x, y]) => {
        const px = Math.max(0, Math.min(width - 1, Math.round(x * (width - 1))));
        const py = Math.max(0, Math.min(height - 1, Math.round(y * (height - 1))));
        const data = ctx.getImageData(px, py, 1, 1).data;
        return { r: data[0], g: data[1], b: data[2], a: data[3] };
      });
    },
    { url: dataUrl, points: offsets }
  );
}

/**
 * Compute simple visible-output stats for an element screenshot. This is
 * more robust than sparse-grid sampling for thin lines / small splat
 * clusters while still staying platform-invariant.
 */
export async function getElementPixelStats(
  page: Page,
  selector: string,
  threshold = 10,
  region?: ElementPixelRegion
): Promise<ElementPixelStats> {
  const dataUrl = await captureElementScreenshotDataUrl(page, selector);

  return await page.evaluate(
    async ({ url, cutoff, normalizedRegion }) => {
      const img = new Image();
      img.decoding = 'sync';
      const loaded = new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('getElementPixelStats: failed to decode screenshot'));
      });
      img.src = url;
      await loaded;

      const width = img.naturalWidth;
      const height = img.naturalHeight;
      if (width <= 0 || height <= 0) {
        throw new Error(`getElementPixelStats: empty screenshot ${width}x${height}`);
      }

      const off = document.createElement('canvas');
      off.width = width;
      off.height = height;
      const ctx = off.getContext('2d', { willReadFrequently: true });
      if (!ctx) throw new Error('getElementPixelStats: 2D context unavailable');
      ctx.drawImage(img, 0, 0);

      const left = normalizedRegion ? Math.floor(normalizedRegion.left * width) : 0;
      const top = normalizedRegion ? Math.floor(normalizedRegion.top * height) : 0;
      const right = normalizedRegion ? Math.ceil(normalizedRegion.right * width) : width;
      const bottom = normalizedRegion ? Math.ceil(normalizedRegion.bottom * height) : height;
      const regionWidth = Math.max(0, Math.min(width, right) - Math.max(0, left));
      const regionHeight = Math.max(0, Math.min(height, bottom) - Math.max(0, top));
      if (regionWidth === 0 || regionHeight === 0) {
        throw new Error(
          `getElementPixelStats: empty measurement region ${regionWidth}x${regionHeight}`
        );
      }
      const pixels = ctx.getImageData(
        Math.max(0, left),
        Math.max(0, top),
        regionWidth,
        regionHeight
      ).data;
      let nonBlackPixels = 0;
      let brightest = { r: 0, g: 0, b: 0, a: 0 };
      let brightestSum = -1;

      for (let i = 0; i < pixels.length; i += 4) {
        const r = pixels[i];
        const g = pixels[i + 1];
        const b = pixels[i + 2];
        const a = pixels[i + 3];
        const sum = r + g + b;
        if (sum > cutoff) nonBlackPixels++;
        if (sum > brightestSum) {
          brightestSum = sum;
          brightest = { r, g, b, a };
        }
      }

      return {
        width: regionWidth,
        height: regionHeight,
        threshold: cutoff,
        nonBlackPixels,
        brightest,
      };
    },
    { url: dataUrl, cutoff: threshold, normalizedRegion: region }
  );
}

/**
 * Read a single pixel from a selector at fractional coordinates.
 * Prefer `samplePixelsAt` when taking multiple samples from the same
 * frame so only one screenshot has to be decoded.
 */
export async function samplePixelAt(
  page: Page,
  selector: string,
  fx: number,
  fy: number
): Promise<SampledPixel> {
  const [pixel] = await samplePixelsAt(page, selector, [[fx, fy]]);
  return pixel;
}

// ---------------------------------------------------------------------------
// Camera placement (orbit-controls aware)
// ---------------------------------------------------------------------------

/** A world-space point as it crosses the Playwright ⇄ page boundary. */
export interface Vec3Like {
  x: number;
  y: number;
  z: number;
}

/** Where the camera actually ended up after {@link placeCameraAt}. */
export interface CameraPlacement extends Vec3Like {
  /** The pivot the camera was aimed at (the controls' target after the call). */
  target: Vec3Like;
  /**
   * `|position − target|` measured AFTER the controls re-derived and re-applied
   * their own state — i.e. the distance that survived
   * `minDistance`/`maxDistance` clamping, not the one that was requested. Assert
   * on this when a test needs proof that the camera really moved.
   */
  distance: number;
  /**
   * True when ORBIT-LIKE controls were active and re-derived their state from
   * the write (`reinitialize()` + `update()`), so `x`/`y`/`z`/`distance` above
   * are the controls' own post-clamp answer.
   *
   * False means the active controls are not orbit-like (`LuxarFlyControls`,
   * which owns no target/distance and treats `camera.position` as
   * authoritative): the write is CORRECT there and does stick, but nothing
   * re-derived or clamped it, so the reported values are simply what was asked
   * for and are no evidence at all that any controller agreed. Assert this flag
   * whenever the point of the placement is "prove the camera moved".
   */
  viaOrbitControls: boolean;
}

/** The active orbit controls' distance clamp, `[min, max]`. */
export interface OrbitDistanceLimits {
  min: number;
  max: number;
}

/**
 * Effectively-no-clamp orbit distance limits for {@link withOrbitDistanceLimits}.
 *
 * `min` is a small POSITIVE number rather than 0 because `initializeFromCamera`
 * treats a non-positive floor as "use 0.001".
 *
 * How wide the real clamp is, measured — because it is much wider than it
 * sounds and it is easy to blame it for an inert camera it had nothing to do
 * with. After auto-framing, `camera-framing.ts:256` sets
 * `[D / ZOOM_IN_FACTOR, D * ZOOM_OUT_FACTOR]` = `[D / 1000, D × 10000]` around
 * the framing distance `D` (before auto-framing, `deriveScaleLimits` uses
 * `config.controls.scaleMultipliers` off the scene diagonal). On the
 * `test_lines` fixture that is `minDistance = 0.0138`, `maxDistance = 137990`
 * for `D = 13.799`. So a test only needs this at all if it wants the camera
 * inside `D/1000` or beyond `10000·D` — two-plus decades out from anything a
 * "look at it from close up" or "back right off" placement asks for. Reach for
 * it deliberately, not as a reflex: the trap that actually makes placements
 * inert is the missing `reinitialize()` ({@link placeCameraAt}), not this.
 */
export const UNCLAMPED_ORBIT_DISTANCE_LIMITS: OrbitDistanceLimits = {
  min: 1e-9,
  max: Infinity,
};

/**
 * The in-page camera API `installCameraPlacement` attaches to
 * `window.__luxarE2ECamera`. Declared here (types are erased) so the installer's
 * body, the Node-side callers below, and a spec that drives many placements
 * from inside ONE `page.evaluate` all describe the same shape.
 */
export interface InPageCameraApi {
  /**
   * The active controls' pivot. THROWS when there is none (no orbit-like
   * target and no `getFocusTarget()`) rather than answering with the world
   * origin — see {@link getCameraPivot}.
   */
  pivot(): Vec3Like;
  place(position: Vec3Like, target?: Vec3Like | null, up?: Vec3Like | null): CameraPlacement | null;
  /**
   * Widen/restore the active orbit controls' distance clamp. Returns the
   * PREVIOUS limits, or `null` when nothing was changed because the active
   * controls are not orbit-like (or their limits are not numeric) — the caller
   * must not read `null` as "applied".
   */
  setDistanceLimits(min: number, max: number): OrbitDistanceLimits | null;
}

/**
 * Install `window.__luxarE2ECamera` — the ONE definition of "put the camera
 * somewhere and make it stick" (idempotent, cheap, wiped by navigation, so every
 * public helper below calls it first).
 *
 * Why a page-side object rather than a Node-side helper alone: a sweep that
 * places the camera dozens of times inside a single `page.evaluate` must use the
 * same routine as a spec that places it once, or the idiom gets copy-pasted and
 * one copy drifts. `installCameraPlacement` + `(window as …).__luxarE2ECamera`
 * gives both call shapes one implementation.
 *
 * Two traps this encapsulates — both of which silently made E2E camera moves
 * INERT (issue #1930):
 *
 * 1. **`__luxarDebug.controls` is the `ControlsManager`, not the active
 *    `LuxarOrbitControls`.** It has no `target` field, so the common
 *    `debug.controls.target ?? {x:0,y:0,z:0}` reads `undefined` and quietly
 *    falls back to the world origin instead of the real pivot. The pivot comes
 *    from `getControls().target`, falling back to `getFocusTarget()` and then
 *    THROWING — never to the origin, which is the silent answer that hid #1930.
 * 2. **`update()` overwrites `camera.position`.** `runUpdateStep` step 8 calls
 *    `applyToCamera(camera, target, orientation, distance)` unconditionally, so
 *    the controls' own state — not the camera transform — is authoritative. An
 *    externally written position is discarded on the very next frame unless
 *    `reinitialize()` re-derives orientation + distance from it first (see
 *    `ControlsManager.reinitialize`'s doc).
 */
async function installCameraPlacement(page: Page): Promise<void> {
  await page.evaluate(() => {
    interface Vec3 {
      x: number;
      y: number;
      z: number;
    }
    interface OrbitLike {
      target?: Vec3 & { set: (x: number, y: number, z: number) => void };
      minDistance?: number;
      maxDistance?: number;
      reinitialize?: () => void;
      update?: () => void;
    }
    const w = window as unknown as {
      __luxarE2ECamera?: unknown;
      __luxarDebug?: {
        camera?: {
          position: Vec3 & { set: (x: number, y: number, z: number) => void };
          up: Vec3 & { set: (x: number, y: number, z: number) => void };
          lookAt: (x: number, y: number, z: number) => void;
          updateMatrixWorld: (force?: boolean) => void;
        };
        controls?: {
          getControls?: () => OrbitLike | null;
          getFocusTarget?: () => Vec3;
        };
        renderOnce?: () => void;
      };
    };
    if (w.__luxarE2ECamera) return;

    // The ACTIVE controls, and only when they are orbit-like: `getControls()`
    // can also hand back `LuxarFlyControls`, which has neither `target` nor
    // `reinitialize`.
    const orbit = (): OrbitLike | null => {
      const c = w.__luxarDebug?.controls?.getControls?.() ?? null;
      return c && c.target && typeof c.reinitialize === 'function' ? c : null;
    };

    // THROWS rather than falling back to the world origin. A silent
    // origin fallback is exactly what made the #1930 sweep inert-looking
    // ("the pivot is (0,0,0)" is indistinguishable from "there is no pivot"),
    // so an unusable pivot is reported instead of quietly substituted.
    const pivot = (): Vec3 => {
      const t = orbit()?.target;
      if (t) return { x: t.x, y: t.y, z: t.z };
      const f = w.__luxarDebug?.controls?.getFocusTarget?.();
      if (f) return { x: f.x, y: f.y, z: f.z };
      throw new Error(
        '__luxarE2ECamera.pivot(): no orbit-like controls target and no getFocusTarget() on ' +
          '__luxarDebug.controls, so there is no pivot to aim at. Pass an explicit target, or ' +
          'wait for the controls to initialize — do NOT read this as the world origin (#1930).'
      );
    };

    const api = {
      pivot,
      setDistanceLimits(min: number, max: number) {
        const o = orbit();
        if (!o || typeof o.minDistance !== 'number' || typeof o.maxDistance !== 'number') {
          return null;
        }
        const previous = { min: o.minDistance, max: o.maxDistance };
        o.minDistance = min;
        o.maxDistance = max;
        return previous;
      },
      place(position: Vec3, target?: Vec3 | null, up?: Vec3 | null) {
        const cam = w.__luxarDebug?.camera;
        if (!cam) return null;
        const t = target ?? pivot();
        // `up` BEFORE `lookAt`: the orientation is derived from it.
        if (up) cam.up.set(up.x, up.y, up.z);
        cam.position.set(position.x, position.y, position.z);
        cam.lookAt(t.x, t.y, t.z);
        cam.updateMatrixWorld(true);
        const o = orbit();
        if (o) {
          o.target!.set(t.x, t.y, t.z);
          o.reinitialize!(); // re-derive distance + orientation from the write
          o.update?.(); // now a no-op reapplication instead of a snap-back
        }
        w.__luxarDebug?.renderOnce?.();
        const p = cam.position;
        const dx = p.x - t.x;
        const dy = p.y - t.y;
        const dz = p.z - t.z;
        return {
          x: p.x,
          y: p.y,
          z: p.z,
          target: t,
          distance: Math.sqrt(dx * dx + dy * dy + dz * dz),
          // Non-orbit (fly) controls take `camera.position` as authoritative, so
          // the bare write above is right for them — but then nothing re-derived
          // or clamped anything and the numbers returned are just the request
          // echoed back. Say which of the two happened rather than letting the
          // caller assume.
          viaOrbitControls: o !== null,
        };
      },
    };
    w.__luxarE2ECamera = api;
  });
}

/**
 * The active controls' pivot — `getControls().target`, falling back to
 * `getFocusTarget()`. Use it to express a pose RELATIVE to the content centre
 * before handing it to {@link placeCameraAt}.
 *
 * REJECTS when neither is available. The world origin is deliberately NOT a
 * fallback: it is a plausible-looking wrong answer that reads as a real pivot,
 * which is half of how #1930 stayed hidden.
 */
export async function getCameraPivot(page: Page): Promise<Vec3Like> {
  await installCameraPlacement(page);
  return await page.evaluate(() =>
    (window as unknown as { __luxarE2ECamera: InPageCameraApi }).__luxarE2ECamera.pivot()
  );
}

/**
 * Move the camera to `position` and make the move STICK, then wait for a frame.
 *
 * Aims at `options.target` when given, else at the current pivot
 * (`getControls().target`, NOT the always-`undefined`
 * `__luxarDebug.controls.target`); `options.up` sets `camera.up` before the
 * `lookAt`, for a pose that must not be the default +Y-up one.
 *
 * Returns where the camera actually ended up (`null` if the debug interface has
 * no camera). `distance` is post-clamp and `viaOrbitControls` says whether any
 * controller re-derived the write at all, so a test can PROVE the camera moved
 * rather than assuming it did — assert on both. Wrap in
 * {@link withOrbitDistanceLimits} only for a distance outside
 * `[D/1000, D×10000]` around the framing distance (see
 * {@link UNCLAMPED_ORBIT_DISTANCE_LIMITS}); ordinary placements are nowhere
 * near it.
 */
export async function placeCameraAt(
  page: Page,
  position: Vec3Like,
  options: { target?: Vec3Like; up?: Vec3Like } = {}
): Promise<CameraPlacement | null> {
  await installCameraPlacement(page);
  const placement = await page.evaluate(
    ({ p, t, u }) =>
      (window as unknown as { __luxarE2ECamera: InPageCameraApi }).__luxarE2ECamera.place(p, t, u),
    { p: position, t: options.target ?? null, u: options.up ?? null }
  );
  await waitForNextRender(page);
  return placement;
}

/**
 * Run `body` with the active orbit controls' distance clamp temporarily widened,
 * restoring the previous limits afterwards (including on throw).
 *
 * The clamp is re-applied on EVERY frame (`runUpdateStep` step 6), so the widened
 * window must cover the whole span in which the test looks at the result — the
 * placement, any `waitForRenderStable`, and the pixel/state sampling. Restoring
 * too early lets the next frame fling the camera back to the framing distance.
 *
 * Pass {@link UNCLAMPED_ORBIT_DISTANCE_LIMITS} for "wherever I put it, leave it".
 * DEFENSIVE: the real window is `[D/1000, D×10000]` around the framing distance,
 * so most placements are decades inside it — see
 * {@link UNCLAMPED_ORBIT_DISTANCE_LIMITS} for the measured numbers.
 *
 * THROWS when the widening could not be applied (the active controls are not
 * orbit-like, or their limits are not numeric). Swallowing that would run `body`
 * under the ORIGINAL clamp while the caller believed it was widened — the same
 * class of silent inertness as #1930.
 */
export async function withOrbitDistanceLimits<T>(
  page: Page,
  limits: OrbitDistanceLimits,
  body: () => Promise<T>
): Promise<T> {
  await installCameraPlacement(page);
  const previous = await page.evaluate(
    (l) =>
      (
        window as unknown as { __luxarE2ECamera: InPageCameraApi }
      ).__luxarE2ECamera.setDistanceLimits(l.min, l.max),
    limits
  );
  if (!previous) {
    throw new Error(
      'withOrbitDistanceLimits: the orbit distance clamp was NOT widened — the active ' +
        'controls are not orbit-like (LuxarFlyControls has no minDistance/maxDistance), ' +
        'so the body would have run under the original clamp. Switch to orbit controls ' +
        'first, or drop the wrapper.'
    );
  }
  try {
    return await body();
  } finally {
    // A restore failure is inconsequential (the test is over either way), but
    // letting it throw out of `finally` would REPLACE the real failure — e.g.
    // when `body` failed because the page crashed and the page is now gone.
    await page
      .evaluate(
        (l) =>
          (
            window as unknown as { __luxarE2ECamera: InPageCameraApi }
          ).__luxarE2ECamera.setDistanceLimits(l.min, l.max),
        previous
      )
      .catch(() => {});
  }
}

/** Result of {@link probeWebGPUBackend}. */
export interface WebGPUBackendProbe {
  /** True only when a REAL native WebGPU backend is driving the page. */
  isNative: boolean;
  /** `capabilities.apiSurface` as reported by the viewer, if available. */
  apiSurface?: string;
  /** `capabilities.framebufferYDown` as reported by the viewer, if available. */
  framebufferYDown?: boolean;
}

/**
 * Probe which graphics backend is physically running behind
 * `?renderer=webgpu`, for specs that must skip on the WebGL2 fallback.
 *
 * `capabilities.apiSurface === 'webgpu'` does NOT answer this: it is
 * `'webgpu'` for any active `WebGPURenderer`, *including* one whose
 * internal backend has fallen back to WebGL2 — the headless-chromium
 * norm. That field means "which method-signature contract should I
 * follow?", not "which GPU backend is running?" (#1449). So `isNative`
 * reads the backend's own flag and nothing else, which also leaves
 * `apiSurface` free to be *asserted* by callers rather than assumed.
 *
 * The tell is deliberately POSITIVE (`isWebGPUBackend === true`, set by
 * three's `WebGPUBackend` constructor): a negative `isWebGLBackend
 * !== true` check fails OPEN under structural drift — a renamed flag, a
 * third backend, a wrapped `backend` — silently restoring the very
 * fail-open behaviour #1449 fixed. The positive form fails closed
 * (skip), mirroring `isWebGLRenderer` in `rendering/renderer-capabilities.ts`.
 *
 * Call this only AFTER `waitForLuxarReady`. `WebGPURenderer` constructs a
 * `WebGPUBackend` eagerly and *replaces* `this.backend` with a
 * `WebGLBackend` from inside `init()` when no adapter can be acquired, so a
 * probe that races the viewer's `await renderer.init()` reads the optimistic
 * pre-fallback value and reports a native backend that isn't there.
 */
export async function probeWebGPUBackend(page: Page): Promise<WebGPUBackendProbe> {
  return await page.evaluate(() => {
    const debug = (
      window as unknown as {
        __luxarDebug?: {
          app?: {
            sceneManager?: {
              capabilities?: { apiSurface?: string; framebufferYDown?: boolean };
            };
          };
          renderer?: { backend?: { isWebGPUBackend?: boolean } };
        };
      }
    ).__luxarDebug;
    const caps = debug?.app?.sceneManager?.capabilities;
    return {
      isNative: debug?.renderer?.backend?.isWebGPUBackend === true,
      apiSurface: caps?.apiSurface,
      framebufferYDown: caps?.framebufferYDown,
    };
  });
}
