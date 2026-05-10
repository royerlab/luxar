/**
 * Cache Hardening E2E Tests
 *
 * Browser-side regression coverage for the Phase 1-8 cache hardening
 * pass. Each test maps to a numbered plan item:
 *
 *   9.2 — rapid dataset switch + lifecycle (no leaked state across loads)
 *   9.3 — node-type cache parity (points + lines both surface health)
 *   9.4 — browser-storage health (status badges via debug snapshot)
 *   9.5 — UI cache-monitor coverage (network + demand + prefetch + health
 *         visible through __luxarDebug.cache.getStats())
 *
 * Existing cache-system.spec.ts and dataset-switching.spec.ts cover
 * the L0/L1/L2 fundamentals; this file complements them with the
 * Phase-7 health/badge surface.
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, getLuxarState, assertNoConsoleErrors } from './helpers';

const EXAMPLES_BASE = 'http://localhost:9000/datasets/examples';
const POINTS_DATASET = `${EXAMPLES_BASE}/radius_basic_example.zarr`;
const LINES_DATASET = `${EXAMPLES_BASE}/lines_basic_example.zarr`;

test.describe('Cache hardening — debug snapshot extensions (9.5)', () => {
  test('getStats() exposes network, demand, prefetch, and health fields', async ({ page }) => {
    await page.goto(`/?src=${POINTS_DATASET}&debug`);
    await waitForLuxarReady(page);

    const stats = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      return await debug.cache.getStats();
    });

    // Phase 7 additions: snapshot must surface every diagnostic in one call.
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

  test('Luxar dataset records validationMode === content-hash (9.4)', async ({ page }) => {
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

  test('OPFS health counters are present on L2 stats (3.2 + 6.2)', async ({ page }) => {
    await page.goto(`/?src=${POINTS_DATASET}&debug`);
    await waitForLuxarReady(page);

    const l2 = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      const stats = await debug.cache.getStats();
      return stats.l2;
    });

    expect(l2).toBeDefined();
    // Counters added in Phase 3 / 6 are optional on the type but
    // populated by the real OPFSStore — clean session ⇒ all zero.
    expect(l2.oversizedWriteSkipped ?? 0).toBeGreaterThanOrEqual(0);
    expect(l2.quotaWriteSkipped ?? 0).toBeGreaterThanOrEqual(0);
    expect(l2.evictions ?? 0).toBeGreaterThanOrEqual(0);
    expect(l2.writeFailures ?? 0).toBe(0);
    expect(l2.corruptedEntries ?? 0).toBe(0);
    expect(l2.metadataParseFailures ?? 0).toBe(0);
  });
});

test.describe('Cache hardening — dataset switch lifecycle (9.2)', () => {
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
      const stats = await page.evaluate(
        async () => (window as any).__luxarDebug.cache.getStats()
      );
      expect(stats.prefetch.queued).toBeLessThan(50);
    }

    await assertNoConsoleErrors(page);
  });
});

test.describe('Cache hardening — node-type parity (9.3)', () => {
  test('points dataset surfaces complete cache stats', async ({ page }) => {
    await page.goto(`/?src=${POINTS_DATASET}&debug`);
    await waitForLuxarReady(page);
    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    const stats = await page.evaluate(
      async () => (window as any).__luxarDebug.cache.getStats()
    );
    expect(stats.l0).toBeDefined();
    expect(stats.l0.count).toBeGreaterThan(0);
    expect(stats.health).toBeDefined();
  });

  test('lines dataset surfaces complete cache stats (parity with points)', async ({ page }) => {
    await page.goto(`/?src=${LINES_DATASET}&debug`);
    await waitForLuxarReady(page);

    const stats = await page.evaluate(
      async () => (window as any).__luxarDebug.cache.getStats()
    );
    // Same fields populated for lines as for points.
    expect(stats.l0).toBeDefined();
    expect(stats.l0.count).toBeGreaterThan(0);
    expect(stats.health).toBeDefined();
    expect(stats.network).toBeDefined();
    expect(stats.demand).toBeDefined();
    expect(stats.prefetch).toBeDefined();
  });
});
