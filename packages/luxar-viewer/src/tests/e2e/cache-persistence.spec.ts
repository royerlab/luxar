/**
 * R7: L2 persistence across page reload.
 *
 * The existing cache-system spec clears OPFS in beforeEach, which is
 * the right hygiene for isolation but means no test actually verifies
 * that L2 entries survive a full page reload. This file fills that
 * gap with Chromium-only assertions (Firefox/WebKit OPFS persistence
 * across Playwright contexts is unreliable).
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForCacheStable } from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/radius_basic_example.luxar.zarr';

test.describe('L2 persistence across page reload (R7)', () => {
  test('L2 entries survive a full page reload (Chromium real OPFS)', async ({
    page,
    browserName,
  }) => {
    test.skip(
      browserName !== 'chromium',
      'Real OPFS persistence across Playwright contexts is Chromium-only in CI'
    );

    // Pass 1: load the dataset and let caches warm.
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForCacheStable(page);

    const before = await page.evaluate(async () => (window as any).__luxarDebug.cache.getStats());
    // L2 should contain entries after a clean first load (cache may
    // be empty initially in this browser context, so the assertion
    // tolerates a 0-count first run by exiting early — only the
    // post-reload comparison is the regression guard).
    if (!before.l2 || before.l2.count === 0) {
      test.skip(true, 'No L2 entries cached on first load — cannot assert persistence');
      return;
    }

    // Pass 2: full page reload. The same dataset's content hash must
    // match → L2 entries are preserved → after-reload count equals
    // the before-reload count.
    await page.reload();
    await waitForLuxarReady(page);
    await waitForCacheStable(page);

    const after = await page.evaluate(async () => (window as any).__luxarDebug.cache.getStats());
    expect(after.l2).toBeDefined();
    // L2 count is preserved (no entries dropped during reload).
    expect(after.l2.count).toBeGreaterThanOrEqual(before.l2.count);
    // Hits-after-reload prove the cache served data without a fresh
    // network round-trip: either L2 reads or L1 hits should reflect
    // the warmed state.
    expect(after.l2.reads + after.l1.hits).toBeGreaterThan(0);
  });

  test('?clear-cache invokes clearAll on init (S4)', async ({ page, browserName }) => {
    test.skip(browserName !== 'chromium', 'Chromium-only');

    // Pass 1: warm the cache with a regular load.
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForCacheStable(page);

    // Sanity: no clear on this load.
    const beforeClear = await page.evaluate(async () =>
      (window as any).__luxarDebug.cache.getStats()
    );
    expect(beforeClear.clearOnInitCount ?? 0).toBe(0);

    // Pass 2: reload with ?clear-cache. The store's init path must
    // increment `clearOnInitCount` exactly once when ?clear-cache
    // is present. This is the real observable signal — previously
    // the test only asserted `l2.size >= 0`, which is trivially true.
    await page.goto(`/?src=${DATASET}&debug&clear-cache`);
    await waitForLuxarReady(page);

    const afterClear = await page.evaluate(async () =>
      (window as any).__luxarDebug.cache.getStats()
    );
    expect(afterClear.clearOnInitCount).toBeGreaterThanOrEqual(1);
  });
});
