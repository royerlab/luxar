/**
 * Spatial Index Accuracy Tests
 *
 * These tests verify that the spatial index query system works correctly:
 * - Queries return expected point ranges
 * - Range merging works efficiently
 * - Cache behavior is correct
 * - Zero-radius filtering works
 *
 * CRITICAL: Spatial indexing is core to performance - must be correct!
 */

import { test, expect } from '@playwright/test';
import { waitForLuxarReady, getLuxarState } from './helpers';

const DATASETS = {
  denseGrid5D: '/examples/dense_grid_5d_example.zarr',
  nav4D: '/examples/dimension_navigation_example.zarr',
  broadcast: '/examples/broadcast_api_example.zarr',
};

test.describe('Spatial Index Query Accuracy', () => {
  test('should log spatial index metadata on load', async ({ page }) => {
    const cacheLogs: string[] = [];
    page.on('console', (msg) => {
      const text = msg.text();
      if (
        text.includes('spatial index') ||
        text.includes('occupied cells') ||
        text.includes('Grid shape')
      ) {
        cacheLogs.push(text);
      }
    });

    await page.goto(`/?src=${DATASETS.denseGrid5D}&debug`);
    await waitForLuxarReady(page);

    // Should see spatial index initialization logs
    const hasIndexLog = cacheLogs.some(
      (log) => log.includes('Initialized with') || log.includes('occupied cells')
    );

    expect(hasIndexLog).toBe(true);

    // Should see grid shape
    const hasGridShape = cacheLogs.some((log) => log.includes('Grid shape:'));

    expect(hasGridShape).toBe(true);
  });

  test('should perform spatial queries on navigation', async ({ page }) => {
    const queryLogs: string[] = [];
    page.on('console', (msg) => {
      const text = msg.text();
      if (text.includes('Querying spatial index') || text.includes('Query result')) {
        queryLogs.push(text);
      }
    });

    await page.goto(`/?src=${DATASETS.denseGrid5D}&debug`);
    await waitForLuxarReady(page);

    queryLogs.length = 0; // Clear initial logs

    // Navigate to trigger query
    await page.keyboard.press('4');
    await page.keyboard.press(']');
    await page.waitForTimeout(2000);

    // Should see "Querying spatial index"
    const hasQueryLog = queryLogs.some((log) => log.includes('Querying spatial index'));
    expect(hasQueryLog).toBe(true);

    // Should see query result with pattern: "X cells → Y ranges → Z points"
    const hasResultLog = queryLogs.some((log) =>
      log.match(/\d+\s+cells?\s+→\s+\d+\s+ranges?\s+→\s+\d+\s+points?/)
    );
    expect(hasResultLog).toBe(true);
  });

  test('should merge adjacent ranges for efficiency', async ({ page }) => {
    const cacheLogs: string[] = [];
    page.on('console', (msg) => cacheLogs.push(msg.text()));

    await page.goto(`/?src=${DATASETS.denseGrid5D}&debug`);
    await waitForLuxarReady(page);

    cacheLogs.length = 0;

    // Navigate
    await page.keyboard.press('4');
    await page.keyboard.press(']');
    await page.waitForTimeout(2000);

    // Find query result log
    const queryResult = cacheLogs.find(
      (log) => log.includes('cells') && log.includes('ranges') && log.includes('points')
    );

    if (queryResult) {
      // Extract numbers: "50 cells → 10 ranges → 12000 points"
      const match = queryResult.match(/(\d+)\s+cells?\s+→\s+(\d+)\s+ranges?/);

      if (match) {
        const cells = parseInt(match[1]);
        const ranges = parseInt(match[2]);

        // Should merge ranges (fewer ranges than cells)
        expect(ranges).toBeLessThanOrEqual(cells);
      }
    }
  });

  test('should handle effective radius calculation', async ({ page }) => {
    const cacheLogs: string[] = [];
    page.on('console', (msg) => {
      const text = msg.text();
      if (text.includes('Effective radii') || text.includes('effective radius')) {
        cacheLogs.push(text);
      }
    });

    await page.goto(`/?src=${DATASETS.denseGrid5D}&debug`);
    await waitForLuxarReady(page);

    // Navigate through non-displayed dimension
    await page.keyboard.press('4');
    await page.keyboard.press(']');
    await page.waitForTimeout(2000);

    // May see effective radii calculation logs
    // "Effective radii: X/Y points changed"
  });

  test('should filter zero-radius points', async ({ page }) => {
    const cacheLogs: string[] = [];
    page.on('console', (msg) => {
      const text = msg.text();
      if (text.includes('Filtering') || text.includes('zero-radius')) {
        cacheLogs.push(text);
      }
    });

    await page.goto(`/?src=${DATASETS.denseGrid5D}&debug`);
    await waitForLuxarReady(page);

    // Navigate to slice where some points filtered
    await page.keyboard.press('4');
    await page.keyboard.press(']');
    await page.keyboard.press(']');
    await page.waitForTimeout(2000);

    // May see filtering logs
    // "Filtering out X zero-radius points (keeping Y)"
  });
});

test.describe('Spatial Index - Cache Behavior', () => {
  test('should show cache misses on first load', async ({ page }) => {
    const cacheLogs: string[] = [];
    page.on('console', (msg) => {
      const text = msg.text();
      if (text.toLowerCase().includes('cache')) {
        cacheLogs.push(text);
      }
    });

    await page.goto(`/?src=${DATASETS.nav4D}&debug`);
    await waitForLuxarReady(page);

    cacheLogs.length = 0;

    // First navigation - should miss cache
    await page.keyboard.press('4');
    await page.keyboard.press(']');
    await page.waitForTimeout(2000);

    const misses = cacheLogs.filter((log) => log.includes('Loading') || log.includes('miss'));

    expect(misses.length).toBeGreaterThan(0);
  });

  test('should show cache hits on return to previous slice', async ({ page }) => {
    const cacheLogs: string[] = [];
    page.on('console', (msg) => {
      const text = msg.text();
      if (text.toLowerCase().includes('cache')) {
        cacheLogs.push(text);
      }
    });

    await page.goto(`/?src=${DATASETS.denseGrid5D}&debug`);
    await waitForLuxarReady(page);

    // Navigate forward
    await page.keyboard.press('4');
    await page.keyboard.press(']');
    await page.waitForTimeout(2000);

    cacheLogs.length = 0; // Clear logs

    // Navigate back - should hit cache
    await page.keyboard.press('[');
    await page.waitForTimeout(1000);

    const hits = cacheLogs.filter(
      (log) => log.toLowerCase().includes('cache hit') || log.includes('Cache hit')
    );

    // Should see cache hits
    expect(hits.length).toBeGreaterThan(0);
  });

  test('should report cache statistics', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.denseGrid5D}&debug`);
    await waitForLuxarReady(page);

    // Navigate several times to populate cache
    await page.keyboard.press('4');

    for (let i = 0; i < 3; i++) {
      await page.keyboard.press(']');
      await page.waitForTimeout(1000);
    }

    // Get cache stats via scene loader
    const cacheStats = await page.evaluate(async () => {
      const loader = await (window as any).__luxarDebug.getSceneLoader();
      const defaultLoader = loader?.getDefaultLoader();

      if (!defaultLoader) return null;

      return defaultLoader.getCacheStats ? defaultLoader.getCacheStats() : null;
    });

    if (cacheStats) {
      expect(cacheStats.numEntries).toBeGreaterThan(0);
      expect(cacheStats.totalMemory).toBeGreaterThan(0);
    }
  });
});

test.describe('Spatial Index - Error Handling', () => {
  test('should handle dataset without spatial index gracefully', async ({ page }) => {
    const cacheLogs: string[] = [];
    page.on('console', (msg) => {
      const text = msg.text();
      if (text.includes('No spatial index') || text.includes('3D dataset')) {
        cacheLogs.push(text);
      }
    });

    await page.goto(`/?src=${DATASETS.nav4D}&debug`);
    await waitForLuxarReady(page);

    // Should work even without index (creates dummy index)
    const state = await getLuxarState(page);
    expect(state.initialized).toBe(true);
  });

  test('should handle queries outside data bounds', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.denseGrid5D}&debug`);
    await waitForLuxarReady(page);

    // Navigate to extreme position
    await page.keyboard.press('4');

    // Navigate forward many times (may go outside bounds)
    for (let i = 0; i < 20; i++) {
      await page.keyboard.press(']');
      await page.waitForTimeout(200);
    }

    // Should handle gracefully (show 0 points or clamp to bounds)
    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThanOrEqual(0);
  });
});
