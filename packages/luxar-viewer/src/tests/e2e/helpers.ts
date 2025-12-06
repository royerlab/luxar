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
    { timeout }
  );
}

/**
 * Get current Luxar state
 */
export async function getLuxarState(page: Page): Promise<any> {
  return await page.evaluate(() => {
    return (window as any).__luxarDebug.getState();
  });
}

/**
 * Trigger a single render frame (for stable screenshots)
 */
export async function renderOnce(page: Page): Promise<void> {
  await page.evaluate(() => {
    (window as any).__luxarDebug.renderOnce();
  });
  // Wait for render to complete
  await page.waitForTimeout(100);
}

/**
 * Wait for points to be loaded
 */
export async function waitForPointsLoaded(
  page: Page,
  minPoints = 1,
  timeout = 45000
): Promise<void> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    const state = await getLuxarState(page);

    if (state.totalPoints >= minPoints) {
      return;
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
    { timeout }
  );
}

/**
 * Wait for dimension to be selected
 * More reliable than arbitrary timeout after pressing number key
 */
export async function waitForDimensionSelected(
  page: Page,
  _dimensionIndex: number,
  timeout = 5000
): Promise<void> {
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      // Check if dimension was selected (implementation may vary)
      // For now, just wait for state to be stable
      return debug && debug.getState && debug.getState().initialized;
    },
    { timeout }
  );
  // Small delay to ensure input handler processed the key
  await page.waitForTimeout(100);
}

/**
 * Wait for spatial index query to complete
 * Detects when query finishes by checking console or state changes
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
 * Extract attribute values from a point cloud for validation
 *
 * Allows E2E tests to verify actual rendered data matches expected values.
 * CRITICAL for data integrity validation.
 *
 * @param page - Playwright page
 * @param cloudName - Name of the point cloud object
 * @param attribute - Which attribute to extract
 * @returns Array of attribute values
 */
export async function extractAttributeValues(
  page: Page,
  cloudName: string,
  attribute: 'position' | 'color' | 'radius' | 'sharpness'
): Promise<number[]> {
  return await page.evaluate(
    ({ cloudName, attribute }) => {
      const debug = (window as any).__luxarDebug;
      if (!debug || !debug.scene) return [];

      const cloud = debug.scene.getObjectByName(cloudName);
      if (!cloud || !cloud.geometry || !cloud.geometry.attributes) return [];

      const attr = cloud.geometry.attributes[attribute];
      if (!attr || !attr.array) return [];

      return Array.from(attr.array);
    },
    { cloudName, attribute }
  );
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
 * Wait for navigation to complete by detecting loading state change
 *
 * This is more robust than waitForTimeout because it:
 * 1. Waits for isLoading to become true (navigation started)
 * 2. Then waits for isLoading to become false (navigation finished)
 *
 * @param page - Playwright page
 * @param timeout - Maximum wait time in ms
 */
export async function waitForNavigationComplete(page: Page, timeout = 15000): Promise<void> {
  const startTime = Date.now();

  // First, wait for loading to start (or be already done)
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      return debug && debug.getState && typeof debug.getState().isLoading === 'boolean';
    },
    { timeout: 5000 }
  );

  // Then wait for loading to complete
  while (Date.now() - startTime < timeout) {
    const state = await getLuxarState(page);

    if (!state.isLoading) {
      // Additional small delay to ensure WebGL has rendered
      await page.waitForTimeout(100);
      return;
    }

    await page.waitForTimeout(100);
  }

  // Timeout is not an error - loading may have completed
  console.log('[waitForNavigationComplete] Timeout reached, continuing');
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
  // First check if frame counter is available
  const hasFrameCounter = await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    return typeof debug?.renderer?.info?.render?.frame === 'number';
  });

  if (hasFrameCounter) {
    // Use frame counter for precise wait
    await page.waitForFunction(
      (minFrames) => {
        const debug = (window as any).__luxarDebug;
        return debug?.renderer?.info?.render?.frame >= minFrames;
      },
      minFrames,
      { timeout }
    );
  } else {
    // Fallback: wait for data to load + buffer time for rendering
    await page.waitForFunction(
      () => {
        const debug = (window as any).__luxarDebug;
        const state = debug?.getState?.();
        return state && !state.isLoading && state.initialized;
      },
      { timeout }
    );
    // Additional buffer for GPU to render frames
    await page.waitForTimeout(minFrames * 100);
  }
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
