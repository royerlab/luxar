/**
 * Helper utilities for Playwright E2E tests
 */

import { Page } from '@playwright/test';

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
 */
export async function getLuxarState(page: Page): Promise<any> {
  return await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    if (!debug || typeof debug.getState !== 'function') {
      throw new Error('Debug interface not ready: getState() not available');
    }
    return debug.getState();
  });
}

/**
 * Trigger a single render frame (for stable screenshots)
 */
export async function renderOnce(page: Page): Promise<void> {
  await page.evaluate(() => {
    (window as any).__luxarDebug.renderOnce();
  });
  // Intentional fixed sleep: renderOnce() schedules a single
  // requestAnimationFrame, but the actual paint lands on the next
  // browser frame which is not directly observable from JS. 100 ms
  // is one paint cycle past 60 fps with margin.
  await page.waitForTimeout(100);
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

  while (Date.now() - startTime < timeout) {
    try {
      const state = await getLuxarState(page);

      if (state && state.totalPoints >= minPoints) {
        return;
      }
    } catch {
      // Debug interface not ready yet, continue waiting
      // This can happen during initialization
    }

    await page.waitForTimeout(500);
  }

  throw new Error(`Timeout waiting for points to load (expected at least ${minPoints})`);
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
 * Playwright automatically captures screenshots for every test (configured in playwright.config.ts).
 * Screenshots are saved to test-results/ and included in the HTML report.
 *
 * If you need a screenshot in a test, Playwright will capture it automatically.
 * To force a screenshot at a specific point: await page.screenshot({ path: 'test-results/my-screenshot.png' });
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

  while (Date.now() - startTime < timeout) {
    try {
      const state = await getLuxarState(page);

      // Navigation complete if:
      // 1. Point count changed (new data loaded), OR
      // 2. Data loading finished (even if count same due to broadcast/cache)
      if (state.totalPoints !== previousPointCount || !state.isLoading) {
        return;
      }
    } catch {
      // State not ready yet, continue waiting
    }

    await page.waitForTimeout(200);
  }

  throw new Error(`Dimension navigation did not complete within ${timeout}ms`);
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
 * **Silent on timeout** — returns normally even if the condition was
 * never reached. Use when the query completing fast is a *bonus*, not
 * a precondition. For tests that genuinely depend on the query
 * having finished, use {@link waitForSpatialQueryOrThrow} instead.
 */
export async function waitForSpatialQuery(page: Page, timeout = 8000): Promise<void> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    try {
      // Check if a query completed by looking for stable state
      const state = await getLuxarState(page);

      // If we have a stable point count and not loading, query is done
      if (!state.isLoading && state.totalPoints >= 0) {
        return;
      }
    } catch {
      // State not ready yet
    }

    await page.waitForTimeout(100);
  }

  // Timeout not an error - query might have completed
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
 * @param page - Playwright page
 * @returns Object with errors, warnings, and info messages
 */
export async function getConsoleMessages(page: Page): Promise<{
  errors: string[];
  warnings: string[];
  logs: string[];
  all: string[];
}> {
  return await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    if (!debug || !debug.consoleInterceptor) {
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
}

/**
 * Assert no console errors (CRITICAL for all E2E tests)
 *
 * This should be called in EVERY E2E test after loading data.
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
 * Wait for navigation to complete by detecting loading state change
 *
 * This is more robust than waitForTimeout because it:
 * 1. Waits for isLoading to become true (navigation started)
 * 2. Then waits for isLoading to become false (navigation finished)
 *
 * @param page - Playwright page
 * @param timeout - Maximum wait time in ms
 */
/**
 * Wait for nD navigation to complete.
 *
 * **Silent on timeout** — returns normally even if `isLoading` never
 * cleared. Use {@link waitForNavigationCompleteOrThrow} for tests
 * that depend on navigation actually finishing.
 */
export async function waitForNavigationComplete(page: Page, timeout = 15000): Promise<void> {
  const startTime = Date.now();

  // First, wait for loading state to be available (may already be done)
  try {
    await page.waitForFunction(
      () => {
        const debug = (window as any).__luxarDebug;
        return debug && debug.getState && typeof debug.getState().isLoading === 'boolean';
      },
      null,
      { timeout: Math.min(timeout, 10000) }
    );
  } catch {
    // If isLoading state never becomes available, just wait a moment and return.
    // This can happen if navigation completes before we start polling.
    await page.waitForTimeout(300);
    return;
  }

  // Then wait for loading to complete
  while (Date.now() - startTime < timeout) {
    try {
      const state = await getLuxarState(page);
      if (!state.isLoading) {
        await page.waitForTimeout(100);
        return;
      }
    } catch {
      // State may briefly be unavailable during navigation
    }

    await page.waitForTimeout(100);
  }

  // Timeout is not an error - loading may have completed
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
 * Tries to wait for frame counter if available, otherwise uses time-based wait.
 *
 * @param page - Playwright page
 * @param minFrames - Minimum number of frames to render (used as multiplier for fallback)
 * @param timeout - Maximum wait time
 */
export async function waitForRenderStable(
  page: Page,
  minFrames = 3,
  timeout = 10000
): Promise<void> {
  // Snapshot the current frame BEFORE the wait. The previous version
  // checked `frame >= minFrames` against the lifetime counter, so once
  // the initial paint exceeded `minFrames` (which it does within
  // milliseconds of viewer startup), the helper would resolve
  // immediately on every subsequent call — ignoring any post-action
  // paints. Screenshot tests captured pre-action state.
  const start = await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    // WebGLRenderer exposes `info.render.frame`; WebGPURenderer
    // exposes `info.frame`. Probe both so the helper works on
    // either backend.
    const info = debug?.renderer?.info;
    const frame = info?.render?.frame ?? info?.frame;
    return typeof frame === 'number' ? frame : null;
  });

  if (start !== null) {
    // Kick the animation loop in case it's idle (auto-paused after ~2s
    // of inactivity); without this, the frame counter never advances
    // and we'd always fall through to the time-based fallback.
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      debug?.renderOnce?.();
    });

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
  await page.waitForTimeout(minFrames * 100);
}

/**
 * Wait for the next N render frames to complete
 *
 * Records the current renderer frame counter and waits until it advances
 * by the specified number of frames. This replaces most `waitForTimeout(100-500)`
 * calls after user actions (key presses, clicks, etc.) that trigger re-renders.
 *
 * Falls back to a state-based wait + time buffer if the frame counter is unavailable
 * or if the animation loop is idle (auto-paused after inactivity).
 *
 * @param page - Playwright page
 * @param frames - Number of frames to wait for (default: 2)
 * @param timeout - Maximum wait time in ms (default: 5000)
 */
export async function waitForNextRender(page: Page, frames = 2, timeout = 5000): Promise<void> {
  // Try to read the current frame counter
  const currentFrame = await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    // WebGLRenderer exposes `info.render.frame`; WebGPURenderer
    // exposes `info.frame`. Probe both so the helper works on
    // either backend.
    const info = debug?.renderer?.info;
    const frame = info?.render?.frame ?? info?.frame;
    return typeof frame === 'number' ? frame : null;
  });

  if (currentFrame !== null) {
    // Force-trigger a render in case the animation loop is idle (auto-paused).
    // The animation controller pauses after ~2s of inactivity, which means
    // the frame counter stops incrementing. Calling renderOnce() kicks it.
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      debug?.renderOnce?.();
    });

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
      return;
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
  await page.waitForTimeout(Math.max(frames * 50, 200));
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
 */
export async function waitForWebGLError(
  page: Page,
  predicate: (errors: string[]) => boolean,
  options: { timeout?: number; pollMs?: number } = {}
): Promise<string[]> {
  const timeout = options.timeout ?? 5000;
  const pollMs = options.pollMs ?? 100;
  const deadline = Date.now() + timeout;
  const accumulated: string[] = [];

  while (Date.now() < deadline) {
    const errs = await getWebGLErrors(page);
    accumulated.push(...errs);
    if (predicate(accumulated)) return accumulated;
    await page.waitForTimeout(pollMs);
  }

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
  threshold = 10
): Promise<ElementPixelStats> {
  const dataUrl = await captureElementScreenshotDataUrl(page, selector);

  return await page.evaluate(
    async ({ url, cutoff }) => {
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

      const pixels = ctx.getImageData(0, 0, width, height).data;
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

      return { width, height, threshold: cutoff, nonBlackPixels, brightest };
    },
    { url: dataUrl, cutoff: threshold }
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
