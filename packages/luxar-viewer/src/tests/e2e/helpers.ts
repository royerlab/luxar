/**
 * Helper utilities for Playwright E2E tests
 */

import { Page } from '@playwright/test';

/**
 * Wait for Luxar to fully initialize
 */
export async function waitForLuxarReady(page: Page, timeout = 10000): Promise<void> {
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
  timeout = 30000
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
 * Take a stable screenshot (waits for render to settle)
 * Saves to test-screenshots/ folder for visual inspection
 */
export async function takeStableScreenshot(page: Page, path: string): Promise<void> {
  // Trigger one render
  await renderOnce(page);

  // Wait a bit for GPU to finish
  await page.waitForTimeout(500);

  // Take screenshot (save to test-screenshots folder)
  const screenshotPath = `test-screenshots/${path}`;
  await page.screenshot({ path: screenshotPath, fullPage: false });
}

/**
 * Take a test screenshot for visual inspection
 * Automatically prefixes with test-screenshots/ folder
 *
 * Use this in tests to capture visual state for debugging:
 * - Claude can inspect screenshots after test runs
 * - User can review screenshots manually
 * - Regenerated on every test run (not committed to git)
 */
export async function takeTestScreenshot(page: Page, name: string): Promise<void> {
  const screenshotPath = `test-screenshots/${name}.png`;
  await page.screenshot({ path: screenshotPath, fullPage: false });
}
