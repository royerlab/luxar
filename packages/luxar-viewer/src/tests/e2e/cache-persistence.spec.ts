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

    // L2 MUST contain entries after the warm-up load — this is the
    // positive precondition for the persistence check. If L2 is never
    // populated the whole feature is broken, so we assert it loudly
    // here rather than silently skipping the reload comparison.
    // L2 writes drain through a background OPFS queue and
    // waitForCacheStable treats an all-zero first poll as "no L2" and
    // returns early, so poll explicitly for the entries to land before
    // asserting — this fails loudly if they never do, without racing
    // the write queue on a cold context.
    await page.waitForFunction(
      () => ((window as any).__luxarDebug.cache.getStats().l2?.count ?? 0) > 0,
      undefined,
      { timeout: 15000 }
    );

    const before = await page.evaluate(async () => (window as any).__luxarDebug.cache.getStats());
    expect(before.l2).toBeDefined();
    expect(before.l2.count).toBeGreaterThan(0);

    // Pass 2: full page reload. The same dataset's content hash must
    // match → L2 entries are preserved → after-reload count equals
    // the before-reload count.
    await page.reload();
    await waitForLuxarReady(page);
    await waitForCacheStable(page);

    // The reload wipes in-memory L1, so any chunk that renders after it
    // must have been READ back from persisted L2 — an L2 read is the one
    // signal that proves persistence, and it cannot be faked by a
    // re-download (which would only bump `count`) or by same-session L1
    // hits. Poll for it, then assert, so a broken persistence path fails
    // as a loud timeout rather than a silent pass.
    await page.waitForFunction(
      () => ((window as any).__luxarDebug.cache.getStats().l2?.reads ?? 0) > 0,
      undefined,
      { timeout: 15000 }
    );

    const after = await page.evaluate(async () => (window as any).__luxarDebug.cache.getStats());
    expect(after.l2).toBeDefined();
    // The entry count must not shrink across the reload. On its own this is
    // only a floor (a re-download would also refill it), which is why the
    // read assertion below carries the actual persistence proof.
    expect(after.l2.count).toBeGreaterThanOrEqual(before.l2.count);
    expect(after.l2.reads).toBeGreaterThan(0);
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
