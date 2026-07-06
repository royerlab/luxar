/**
 * URL Parameter Handling E2E Tests
 *
 * Validates that URL parameters correctly control initial viewer state:
 * - ?theme= sets the UI theme
 * - ?no-cache disables persistent caching
 * - ?debug enables the debug interface
 * - Invalid parameters are handled gracefully
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady } from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr';

test.describe('URL Parameters', () => {
  test('should apply light theme from ?theme=light', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&theme=light`);
    await waitForLuxarReady(page);

    const theme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
    expect(theme).toBe('light');
  });

  test('should apply dark theme from ?theme=dark', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&theme=dark`);
    await waitForLuxarReady(page);

    const theme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
    expect(theme).toBe('dark');
  });

  test('should handle invalid ?theme= parameter gracefully', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto(`/?src=${DATASET}&debug&theme=nonexistent_theme`);
    await waitForLuxarReady(page);

    // Should not crash
    expect(errors).toEqual([]);

    // Should fall back to a valid theme (any theme attribute is acceptable)
    const theme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
    expect(theme).toBeTruthy();
  });

  test('should disable caching with ?no-cache', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&no-cache`);
    await waitForLuxarReady(page);

    // With no-cache, L2 persistent cache should be empty or disabled
    const cacheStats = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      if (!debug?.cache?.getStats) return null;
      return debug.cache.getStats();
    });

    // If cache stats are available, L2 (OPFS) should be empty
    if (cacheStats?.l2) {
      expect(cacheStats.l2.count).toBe(0);
    }
  });

  test('should pin a fixed pixel ratio and lock adaptive DPR with ?dpr=', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&dpr=0.5`);
    await waitForLuxarReady(page);

    const state = await page.evaluate(() => {
      const manager = (window as any).__luxarDebug?.app?.components?.adaptiveDPRManager;
      if (!manager) return null;
      // Simulate a persisted setting trying to re-enable adaptation
      // after init — the pin must hold.
      manager.setEnabled(true);
      return {
        active: manager.isActive(),
        pinned: manager.isPinned(),
        currentDPR: manager.getCurrentDPR(),
      };
    });

    expect(state).not.toBeNull();
    expect(state!.active).toBe(false);
    expect(state!.pinned).toBe(true);
    expect(state!.currentDPR).toBe(0.5);
  });

  test('should enable debug interface with ?debug parameter', async ({ page }) => {
    // With ?debug: interface should be fully functional
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    const debugInfo = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        exists: !!debug,
        hasGetState: typeof debug?.getState === 'function',
        hasScene: !!debug?.scene,
        hasCamera: !!debug?.camera,
        hasCache: !!debug?.cache,
      };
    });

    expect(debugInfo.exists).toBe(true);
    expect(debugInfo.hasGetState).toBe(true);
    expect(debugInfo.hasScene).toBe(true);
    expect(debugInfo.hasCamera).toBe(true);
    expect(debugInfo.hasCache).toBe(true);
  });
});
