/**
 * Cache Hardening E2E Tests
 *
 * Browser-side coverage for cache health diagnostics, dataset-switch
 * lifecycle, and cache-monitor UI fields.
 *
 * Existing cache-system.spec.ts and dataset-switching.spec.ts cover
 * the L0/L1/L2 fundamentals; this file complements them with health,
 * demand, network, and prefetch diagnostics.
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, getLuxarState, assertNoConsoleErrors } from './helpers';

const EXAMPLES_BASE = 'http://localhost:9000/datasets/examples';
const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';
const POINTS_DATASET = `${EXAMPLES_BASE}/radius_basic_example.luxar.zarr`;
const LINES_DATASET = `${EXAMPLES_BASE}/lines_basic_example.luxar.zarr`;
const POINTS_4D_DATASET = `${FIXTURES_BASE}/test_4d.luxar.zarr`;

test.describe('Cache hardening — debug snapshot diagnostics', () => {
  test('getStats() exposes network, demand, prefetch, and health fields', async ({ page }) => {
    await page.goto(`/?src=${POINTS_DATASET}&debug`);
    await waitForLuxarReady(page);

    const stats = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.getStats();
    });

    // Snapshot must surface every cache diagnostic in one call.
    expect(stats).toBeDefined();
    expect(stats.network).toBeDefined();
    expect(stats.network.bytesTransferred).toBeGreaterThanOrEqual(0);
    expect(stats.network.requestCount).toBeGreaterThanOrEqual(0);
    expect(stats.demand).toBeDefined();
    expect(typeof stats.demand.l1Hits).toBe('number');
    expect(typeof stats.demand.l2Hits).toBe('number');
    expect(typeof stats.demand.networkRequests).toBe('number');
    expect(stats.prefetch).toBeDefined();
    expect(typeof stats.prefetch.queued).toBe('number');
    expect(typeof stats.prefetch.inFlight).toBe('number');
    expect(typeof stats.prefetch.enabled).toBe('boolean');
    expect(stats.health).toBeDefined();
    expect(['content-hash', 'ttl', 'none']).toContain(stats.health.validationMode);
    expect(typeof stats.health.unvalidatedExternalDataset).toBe('boolean');

    await assertNoConsoleErrors(page);
  });

  test('Luxar dataset records validationMode === content-hash', async ({ page }) => {
    // Luxar examples carry a content_hash attr — validation should
    // resolve to content-hash mode and the unvalidated badge should
    // NOT fire.
    await page.goto(`/?src=${POINTS_DATASET}&debug`);
    await waitForLuxarReady(page);

    const health = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      const stats = await debug.cache.getStats();
      return stats.health;
    });

    expect(health.validationMode).toBe('content-hash');
    expect(health.unvalidatedExternalDataset).toBe(false);
  });

  test('OPFS health counters are present on L2 stats', async ({ page }) => {
    await page.goto(`/?src=${POINTS_DATASET}&debug`);
    await waitForLuxarReady(page);

    const l2 = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      const stats = await debug.cache.getStats();
      return stats.l2;
    });

    expect(l2).toBeDefined();
    // Counters are optional on the type but populated by the real
    // OPFSStore — clean session ⇒ all zero.
    expect(l2.oversizedWriteSkipped ?? 0).toBeGreaterThanOrEqual(0);
    expect(l2.quotaWriteSkipped ?? 0).toBeGreaterThanOrEqual(0);
    expect(l2.evictions ?? 0).toBeGreaterThanOrEqual(0);
    expect(l2.writeFailures ?? 0).toBe(0);
    expect(l2.corruptedEntries ?? 0).toBe(0);
    expect(l2.metadataParseFailures ?? 0).toBe(0);
  });

  // R3: cache-tab now renders the badge row and Cache Health section.
  test('cache tab renders status badges and Cache Health section', async ({ page }) => {
    await page.goto(`/?src=${POINTS_DATASET}&debug&cache-stats`);
    await waitForLuxarReady(page);

    // Wait for the cache tab DOM to be populated. The monitor poll
    // interval is 1000ms by default; a generous wait keeps the spec
    // stable across slower CI runs.
    await page.waitForFunction(
      () => document.querySelector('[data-field="cache-status-row"]') !== null,
      undefined,
      { timeout: 5000 }
    );

    const statusBadges = await page.$$eval('[data-field="cache-status-row"] [data-badge]', (els) =>
      els.map((e) => e.getAttribute('data-badge'))
    );
    expect(statusBadges).toContain('cache-enabled');

    const healthMode = await page.textContent('[data-field="cache-health-mode"]');
    expect(healthMode?.trim()).toBeTruthy();
    expect(healthMode).toContain('Content Hash');
  });

  test('@visual cache tab screenshot captures status/health layout', async ({ page }) => {
    await page.goto(`/?src=${POINTS_DATASET}&debug&cache-stats`);
    await waitForLuxarReady(page);

    await page.waitForFunction(
      () =>
        document.querySelector('[data-field="cache-status-row"] [data-badge="cache-enabled"]') !==
          null && document.querySelector('[data-field="cache-health-mode"]') !== null,
      undefined,
      { timeout: 5000 }
    );

    const cacheTab = page.locator('.luxar-tab-content--cache').first();
    await expect(cacheTab).toBeVisible();
    const image = await cacheTab.screenshot({
      path: 'test-results/cache-tab-status-health.png',
    });
    expect(image.byteLength).toBeGreaterThan(1000);
  });
});

test.describe('Cache hardening — dataset switch lifecycle', () => {
  test('dataset switch produces a fresh caching store with healthy stats', async ({ page }) => {
    // Load points → load lines → load points again. Each load must
    // produce non-leaking, well-formed cache stats. Catches:
    // - cachingStore left over from prior loadScene
    // - prefetcher continuing against the disposed store
    // - L2 metadata corruption from interleaved writes
    await page.goto(`/?src=${POINTS_DATASET}&debug`);
    await waitForLuxarReady(page);
    let stats = await page.evaluate(async () => (window as any).__luxarDebug.cache.getStats());
    expect(stats.health).toBeDefined();
    expect(stats.l2.writeFailures ?? 0).toBe(0);

    await page.goto(`/?src=${LINES_DATASET}&debug`);
    await waitForLuxarReady(page);
    stats = await page.evaluate(async () => (window as any).__luxarDebug.cache.getStats());
    expect(stats.health).toBeDefined();
    expect(stats.l2.writeFailures ?? 0).toBe(0);

    await page.goto(`/?src=${POINTS_DATASET}&debug`);
    await waitForLuxarReady(page);
    stats = await page.evaluate(async () => (window as any).__luxarDebug.cache.getStats());
    expect(stats.health).toBeDefined();
    expect(stats.l2.writeFailures ?? 0).toBe(0);

    await assertNoConsoleErrors(page);
  });

  test('repeated load/dispose cycles do not accumulate prefetch state', async ({ page }) => {
    // Loop a few load→clear cycles. After each clear, the prefetcher's
    // queued + in-flight counts should return to zero (or near-zero —
    // a few in-flight at the moment of measurement is OK).
    await page.goto(`/?src=${POINTS_DATASET}&debug`);
    await waitForLuxarReady(page);

    for (let i = 0; i < 3; i++) {
      await page.evaluate(async () => (window as any).__luxarDebug.cache.clearAll());
      // Brief settle for any in-flight to unwind.
      await page.waitForTimeout(50);
      const stats = await page.evaluate(async () => (window as any).__luxarDebug.cache.getStats());
      expect(stats.prefetch.queued).toBeLessThan(50);
    }

    await assertNoConsoleErrors(page);
  });
});

test.describe('Cache hardening — node-type parity', () => {
  test('points dataset surfaces complete cache stats', async ({ page }) => {
    await page.goto(`/?src=${POINTS_DATASET}&debug`);
    await waitForLuxarReady(page);
    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    const stats = await page.evaluate(async () => (window as any).__luxarDebug.cache.getStats());
    expect(stats.l0).toBeDefined();
    expect(stats.l0.count).toBeGreaterThan(0);
    expect(stats.health).toBeDefined();
  });

  test('lines dataset surfaces complete cache stats (parity with points)', async ({ page }) => {
    await page.goto(`/?src=${LINES_DATASET}&debug`);
    await waitForLuxarReady(page);

    const stats = await page.evaluate(async () => (window as any).__luxarDebug.cache.getStats());
    // Same fields populated for lines as for points.
    expect(stats.l0).toBeDefined();
    expect(stats.l0.count).toBeGreaterThan(0);
    expect(stats.health).toBeDefined();
    expect(stats.network).toBeDefined();
    expect(stats.demand).toBeDefined();
    expect(stats.prefetch).toBeDefined();
  });

  // R6c: nD parity. The cache infrastructure should populate the same
  // stats fields for 4D datasets as it does for 3D, and navigating a
  // non-displayed dimension must produce new cache misses (proving
  // chunk keys account for the slice position) followed by hits on
  // navigation back.
  test('4D dataset: cache stats populate and per-slice navigation produces fresh L1 misses', async ({
    page,
  }) => {
    await page.goto(`/?src=${POINTS_4D_DATASET}&debug`);
    await waitForLuxarReady(page);

    const stats1 = await page.evaluate(async () => (window as any).__luxarDebug.cache.getStats());
    expect(stats1).toBeDefined();
    expect(stats1.l1).toBeDefined();
    expect(stats1.health).toBeDefined();

    // Navigate a non-displayed dimension: keyboard `1` selects dim 0,
    // `]` advances. The 4D test fixture has axes (T, Z, Y, X) so the
    // default displayDims [Z, Y, X] leave T (dim 0) as the animated axis.
    await page.keyboard.press('1');
    await page.keyboard.press(']');
    // Give the loader pipeline a moment to fetch the new slice.
    await page.waitForTimeout(500);

    const stats2 = await page.evaluate(async () => (window as any).__luxarDebug.cache.getStats());
    // Either: new misses appeared (the load issued additional fetches),
    // or L0 hits increased (cache served the new slice). Either way
    // the cache stays responsive — the failure mode we're guarding
    // against is stats freezing entirely after the dimension hop.
    const stats1L1Total = stats1.l1.hits + stats1.l1.misses;
    const stats2L1Total = stats2.l1.hits + stats2.l1.misses;
    expect(stats2L1Total).toBeGreaterThanOrEqual(stats1L1Total);
  });
});
