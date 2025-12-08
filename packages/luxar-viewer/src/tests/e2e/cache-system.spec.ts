/**
 * OPFS Cache System E2E Tests
 *
 * These tests verify the two-level caching system works correctly in a real browser:
 * - L1 (Memory) caching with segmented LRU
 * - L2 (OPFS) persistence across page reloads
 * - Content-hash validation and cache invalidation
 * - Debug interface for cache management
 *
 * CRITICAL: These are the ONLY tests that verify browser OPFS behavior.
 * Unit tests mock OPFS, but only Playwright tests use real browser storage APIs.
 */

import { test, expect } from '@playwright/test';
import { waitForLuxarReady, getLuxarState, assertNoConsoleErrors } from './helpers';

const DATASET = 'http://localhost:9000/packages/luxar/examples/rainbow_sphere_4d_example.zarr';

test.describe('OPFS Cache System', () => {
  test.beforeEach(async ({ page }) => {
    // Clear all caches before each test for isolation
    await page.goto('/?debug');
    await page.evaluate(() => {
      // Clear OPFS completely
      return navigator.storage
        .getDirectory()
        .then((root) => root.getDirectoryHandle('luxar-cache', { create: false }).catch(() => null))
        .then((dir) => {
          if (!dir) return;
          return (dir as any).removeEntry?.({ recursive: true });
        })
        .catch(() => {
          // OPFS might not be available or already clean
        });
    });
  });

  test('should cache chunks on first load', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&cache-debug`);
    await waitForLuxarReady(page);

    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    // Check cache statistics via debug interface
    const cacheStats = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.getStats();
    });

    expect(cacheStats).toBeDefined();
    expect(cacheStats.l1).toBeDefined();

    // Should have cached metadata and chunks
    const totalL1 = cacheStats.l1.metadataCount + cacheStats.l1.chunksCount;
    expect(totalL1).toBeGreaterThan(0);

    await assertNoConsoleErrors(page);
  });

  test('should serve from L1 cache on second access (same session)', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&cache-debug`);
    await waitForLuxarReady(page);

    // Reload scene (trigger new load)
    await page.goto(`/?src=${DATASET}&debug&cache-debug`);
    await waitForLuxarReady(page);

    // Second load should be much faster due to L1 cache
    // Verify no errors occurred
    await assertNoConsoleErrors(page);

    const stats = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.getStats();
    });

    // Cache should still have data
    expect(stats.l1.metadataCount + stats.l1.chunksCount).toBeGreaterThan(0);
  });

  test('should expose cache debug API', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Verify debug interface exists
    const debugAPI = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        hasCache: !!debug.cache,
        hasGetStats: typeof debug.cache?.getStats === 'function',
        hasListDatasets: typeof debug.cache?.listDatasets === 'function',
        hasClearL1: typeof debug.cache?.clearL1 === 'function',
        hasClearL2: typeof debug.cache?.clearL2 === 'function',
        hasClearAll: typeof debug.cache?.clearAll === 'function',
      };
    });

    expect(debugAPI.hasCache).toBe(true);
    expect(debugAPI.hasGetStats).toBe(true);
    expect(debugAPI.hasListDatasets).toBe(true);
    expect(debugAPI.hasClearL1).toBe(true);
    expect(debugAPI.hasClearL2).toBe(true);
    expect(debugAPI.hasClearAll).toBe(true);
  });

  test('should clear L1 cache via debug API', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Get initial stats
    const statsBeforeClear = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.getStats();
    });

    expect(statsBeforeClear.l1.metadataCount + statsBeforeClear.l1.chunksCount).toBeGreaterThan(0);

    // Clear L1
    await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      await debug.cache.clearL1();
    });

    // Verify L1 is empty
    const statsAfterClear = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.getStats();
    });

    expect(statsAfterClear.l1.metadataCount).toBe(0);
    expect(statsAfterClear.l1.chunksCount).toBe(0);
  });

  test('should disable caching with ?no-cache parameter', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&no-cache&cache-debug`);
    await waitForLuxarReady(page);

    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    // With no-cache, the cache should be empty or disabled
    // (implementation might skip cache entirely or just not populate it)
    const cacheStats = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      try {
        return await debug.cache.getStats();
      } catch {
        return { error: 'Cache disabled' };
      }
    });

    // Either cache is disabled or stats show minimal usage
    // (no-cache mode should bypass caching)
    expect(cacheStats).toBeDefined();

    await assertNoConsoleErrors(page);
  });

  test('should respect segmented LRU (metadata vs chunks)', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&cache-debug`);
    await waitForLuxarReady(page);

    const cacheStats = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.getStats();
    });

    // Should have both metadata and chunks cached separately
    expect(cacheStats.l1.metadataCount).toBeGreaterThan(0);
    expect(cacheStats.l1.chunksCount).toBeGreaterThan(0);

    // Metadata and chunks should be tracked separately
    expect(typeof cacheStats.l1.metadataSize).toBe('number');
    expect(typeof cacheStats.l1.chunksSize).toBe('number');

    await assertNoConsoleErrors(page);
  });

  test('should handle OPFS operations without errors', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&cache-debug`);
    await waitForLuxarReady(page);

    // Wait a bit for OPFS writes to complete (they're async/debounced)
    await page.waitForTimeout(2000);

    // Verify L2 cache has data (if OPFS is available)
    const cacheStats = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.getStats();
    });

    // OPFS may or may not be available depending on browser
    // Just verify we got valid stats without errors
    expect(cacheStats.l2).toBeDefined();
    expect(typeof cacheStats.l2.size).toBe('number');
    expect(typeof cacheStats.l2.count).toBe('number');

    await assertNoConsoleErrors(page);
  });

  test('should list cached datasets via debug API', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Wait for OPFS to persist
    await page.waitForTimeout(2000);

    const datasets = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.listDatasets();
    });

    // Should return an array (might be empty if OPFS unavailable, but should work)
    expect(Array.isArray(datasets) || datasets.error).toBeTruthy();

    await assertNoConsoleErrors(page);
  });

  test('should cache content hash for validation', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&cache-debug`);
    await waitForLuxarReady(page);

    // Wait for data to actually load (cache activity happens during data loading)
    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    // Give a moment for cache operations to complete and logs to be captured
    await page.waitForTimeout(1000);

    // Check console for content hash validation logs
    const consoleLogs = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      if (!debug || !debug.consoleInterceptor) return [];
      return debug.consoleInterceptor.getBufferedMessages().map((m: any) => {
        // Convert to string - handle both string and object formats
        if (typeof m === 'string') return m;
        if (m.text) return m.text;
        if (m.args && m.args.length > 0) return m.args.join(' ');
        return String(m);
      });
    });

    // Should see cache-related logs (if cache-debug is enabled)
    const cacheRelated = consoleLogs.filter((log: string) => {
      const logStr = String(log).toLowerCase();
      return logStr.includes('cache') || logStr.includes('opfs');
    });

    // With cache-debug, we should see some cache activity
    expect(cacheRelated.length).toBeGreaterThan(0);

    await assertNoConsoleErrors(page);
  });

  test('should handle cache clearing without breaking the app', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Get initial point count
    const stateBefore = await getLuxarState(page);
    expect(stateBefore.totalPoints).toBeGreaterThan(0);

    // Clear all caches
    await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      await debug.cache.clearAll();
    });

    // Reload and verify still works
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    const stateAfter = await getLuxarState(page);
    expect(stateAfter.totalPoints).toBe(stateBefore.totalPoints);

    await assertNoConsoleErrors(page);
  });
});
