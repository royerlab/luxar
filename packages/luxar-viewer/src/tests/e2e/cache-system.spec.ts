/**
 * Three-Level Cache System E2E Tests
 *
 * These tests verify the three-level caching system works correctly in a real browser:
 * - L0 (Decompressed chunks) - In-memory cache of decoded zarr chunks (~1μs access)
 * - L1 (Compressed memory) - Segmented LRU for metadata vs chunks
 * - L2 (OPFS) - Persistent storage across page reloads
 * - Content-hash validation and cache invalidation
 * - Debug interface for cache management
 *
 * CRITICAL: These are the ONLY tests that verify browser OPFS behavior and L0 cache integration.
 * Unit tests mock OPFS, but only Playwright tests use real browser storage APIs.
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  getLuxarState,
  assertNoConsoleErrors,
  waitForNextRender,
  waitForCacheStable,
} from './helpers';

// Use 3D dataset - 4D datasets may have 0 points visible depending on slice position
const DATASET = 'http://localhost:9000/datasets/examples/radius_basic_example.luxar.zarr';

test.describe('Three-Level Cache System (L0/L1/L2)', () => {
  test.beforeEach(async ({ page }) => {
    // Clear all caches before each test for isolation
    await page.goto('/?debug');
    await page.evaluate(() => {
      // Clear OPFS completely: every viewer-owned directory lives under the
      // `luxar/` namespace dir (opfs-store/opfs-root.ts), so one recursive
      // remove of that entry wipes all zarr-cache-* datasets.
      return navigator.storage
        .getDirectory()
        .then((root) => root.removeEntry('luxar', { recursive: true }))
        .catch(() => {
          // OPFS might not be available or already clean
        });
    });
  });

  test('should cache chunks on first load (L0, L1, L2)', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&cacheDebug`);
    await waitForLuxarReady(page);

    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    // Check cache statistics via debug interface
    const cacheStats = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.getStats();
    });

    expect(cacheStats).toBeDefined();

    // L0: Decompressed chunk cache should have entries
    expect(cacheStats.l0).toBeDefined();
    expect(cacheStats.l0.count).toBeGreaterThan(0);
    expect(cacheStats.l0.size).toBeGreaterThan(0);

    // L1: Memory cache should have metadata and chunks
    expect(cacheStats.l1).toBeDefined();
    const totalL1 = cacheStats.l1.metadataCount + cacheStats.l1.chunksCount;
    expect(totalL1).toBeGreaterThan(0);

    await assertNoConsoleErrors(page);
  });

  test('should serve from L1 cache on second access (same session)', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&cacheDebug`);
    await waitForLuxarReady(page);

    // Reload scene (trigger new load)
    await page.goto(`/?src=${DATASET}&debug&cacheDebug`);
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

  test('should expose cache debug API (L0, L1, L2)', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Verify debug interface exists
    const debugAPI = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        hasCache: !!debug.cache,
        hasGetStats: typeof debug.cache?.getStats === 'function',
        hasListDatasets: typeof debug.cache?.listDatasets === 'function',
        hasClearL0: typeof debug.cache?.clearL0 === 'function',
        hasClearL1: typeof debug.cache?.clearL1 === 'function',
        hasClearL2: typeof debug.cache?.clearL2 === 'function',
        hasClearAll: typeof debug.cache?.clearAll === 'function',
      };
    });

    expect(debugAPI.hasCache).toBe(true);
    expect(debugAPI.hasGetStats).toBe(true);
    expect(debugAPI.hasListDatasets).toBe(true);
    expect(debugAPI.hasClearL0).toBe(true);
    expect(debugAPI.hasClearL1).toBe(true);
    expect(debugAPI.hasClearL2).toBe(true);
    expect(debugAPI.hasClearAll).toBe(true);
  });

  test('should clear L1 cache via debug API', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Wait for data loading to settle
    await waitForNextRender(page);

    // Get initial stats
    const statsBeforeClear = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.getStats();
    });

    expect(statsBeforeClear.l1.metadataCount + statsBeforeClear.l1.chunksCount).toBeGreaterThan(0);

    // Clear L1 and immediately check stats (before background loading can repopulate)
    const statsAfterClear = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      await debug.cache.clearL1();
      // Get stats immediately after clear
      return await debug.cache.getStats();
    });

    // Cache should be cleared (or have minimal entries if loading just started)
    expect(statsAfterClear.l1.metadataCount + statsAfterClear.l1.chunksCount).toBeLessThan(
      statsBeforeClear.l1.metadataCount + statsBeforeClear.l1.chunksCount
    );
  });

  test('should clear L0 cache via debug API', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Wait for data loading to settle
    await waitForNextRender(page);

    // Get initial stats
    const statsBeforeClear = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.getStats();
    });

    // L0 should have cached decompressed chunks
    expect(statsBeforeClear.l0).toBeDefined();
    expect(statsBeforeClear.l0.count).toBeGreaterThan(0);

    // Clear L0 and immediately check stats
    const statsAfterClear = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      await debug.cache.clearL0();
      return await debug.cache.getStats();
    });

    // L0 cache should be cleared
    expect(statsAfterClear.l0.count).toBe(0);
    expect(statsAfterClear.l0.size).toBe(0);

    // L1 should be unaffected
    expect(statsAfterClear.l1.metadataCount + statsAfterClear.l1.chunksCount).toBeGreaterThan(0);
  });

  test('should track L0 cache hits on second chunk access', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&cacheDebug`);
    await waitForLuxarReady(page);

    // Wait for initial load to complete
    await waitForNextRender(page);

    // Get stats after first load
    const statsAfterFirstLoad = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.getStats();
    });

    expect(statsAfterFirstLoad.l0).toBeDefined();
    const initialMisses = statsAfterFirstLoad.l0.misses;

    // Misses should be > 0 from initial load (all chunks were cache misses)
    expect(initialMisses).toBeGreaterThan(0);

    // Trigger a re-render or small navigation that reuses existing chunks
    // (This simulates accessing cached chunks again)
    await page.mouse.wheel(0, 10); // Small zoom
    await waitForNextRender(page);

    const statsAfterSecondAccess = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.getStats();
    });

    // Hit rate should be calculable (not NaN)
    expect(typeof statsAfterSecondAccess.l0.hitRate).toBe('number');
    expect(isNaN(statsAfterSecondAccess.l0.hitRate)).toBe(false);

    await assertNoConsoleErrors(page);
  });

  test('should disable all caching with ?noCache parameter (L0, L1, L2)', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&noCache&cacheDebug`);
    await waitForLuxarReady(page);

    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    // With `?noCache`, all cache layers should be disabled or empty
    const cacheStats = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      try {
        return await debug.cache.getStats();
      } catch {
        return { error: 'Cache disabled' };
      }
    });

    expect(cacheStats).toBeDefined();

    // L0 should be disabled (null or count=0)
    if (cacheStats.l0) {
      expect(cacheStats.l0.count).toBe(0);
    }

    // L1 should be disabled or empty
    if (cacheStats.l1) {
      expect(cacheStats.l1.metadataCount + cacheStats.l1.chunksCount).toBe(0);
    }

    await assertNoConsoleErrors(page);
  });

  test('should respect segmented LRU (metadata vs chunks)', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&cacheDebug`);
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
    await page.goto(`/?src=${DATASET}&debug&cacheDebug`);
    await waitForLuxarReady(page);

    // Wait for OPFS writes to settle (async/debounced)
    await waitForCacheStable(page);

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

  test('?noOpfs skips ONLY the L2 tier: scene renders, in-memory tiers stay on, no badge', async ({
    page,
  }) => {
    await page.goto(`/?src=${DATASET}&debug&noOpfs&cacheDebug`);
    await waitForLuxarReady(page);

    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    const cacheStats = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.getStats();
    });

    // L1 (in-memory) still serves; L2 was never constructed, so its stats
    // stay the all-zero defaults — with a real dataset loaded, a live L2
    // would have written by now. A DELIBERATE disable reports
    // opfsAvailable=true: the opfs-unavailable badge is reserved for
    // unrequested degradation (init failure / breaker trip).
    expect(cacheStats.l1).toBeDefined();
    expect(cacheStats.l2.count).toBe(0);
    expect(cacheStats.l2.writes).toBe(0);
    expect(cacheStats.health?.opfsAvailable).toBe(true);

    await assertNoConsoleErrors(page);
  });

  test('should list cached datasets via debug API', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Wait for OPFS to persist
    await waitForCacheStable(page);

    const datasets = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.listDatasets();
    });

    // Should return an array (might be empty if OPFS unavailable, but should work)
    expect(Array.isArray(datasets) || datasets.error).toBeTruthy();

    await assertNoConsoleErrors(page);
  });

  test('should cache content hash for validation', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&cacheDebug`);
    await waitForLuxarReady(page);

    // Wait for data to actually load (cache activity happens during data loading)
    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    // Wait for at least one cache- or opfs-related log to be intercepted
    // before reading the buffer. Replaces a 1 s fixed sleep with the
    // actual signal the assertion below depends on.
    await page
      .waitForFunction(
        () => {
          const debug = (window as any).__luxarDebug;
          const msgs = debug?.consoleInterceptor?.getBufferedMessages?.();
          if (!msgs) return false;
          return msgs.some((m: { text?: string; args?: unknown[] }) => {
            const text = String(m.text ?? m.args?.join(' ') ?? m).toLowerCase();
            return text.includes('cache') || text.includes('opfs');
          });
        },
        null,
        { timeout: 5000 }
      )
      .catch(() => {
        // No cache logs surfaced yet — the assertion below will fail
        // loudly with a clearer message than a missed sleep.
      });

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

    // Should see cache-related logs (if `?cacheDebug` is enabled)
    const cacheRelated = consoleLogs.filter((log: string) => {
      const logStr = String(log).toLowerCase();
      return logStr.includes('cache') || logStr.includes('opfs');
    });

    // With `?cacheDebug`, we should see some cache activity
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
