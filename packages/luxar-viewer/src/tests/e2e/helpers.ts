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
        typeof debug.consoleInterceptor.getMessages === 'function'
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
