/**
 * R7: L2 persistence across page reload.
 *
 * The existing cache-system spec clears OPFS in beforeEach, which is
 * the right hygiene for isolation but means no test actually verifies
 * that L2 entries survive a full page reload. This file fills that
 * gap with Chromium-only assertions (Firefox/WebKit OPFS persistence
 * across Playwright contexts is unreliable).
 */

import type { Page, Route } from '@playwright/test';
import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForCacheStable } from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/radius_basic_example.luxar.zarr';

/**
 * Entries listed by the largest on-disk L2 index (`_cache_meta.json`) under the
 * viewer's OPFS namespace, or 0 while none parses. This is the PERSISTED view,
 * as opposed to `getStats().l2.count`, which is the in-memory index.
 */
async function persistedIndexEntries(page: Page): Promise<number> {
  return page.evaluate(async () => {
    try {
      const root = await navigator.storage.getDirectory();
      const namespace = await root.getDirectoryHandle('luxar');
      let most = 0;
      for await (const handle of (namespace as any).values()) {
        if (handle.kind !== 'directory') continue;
        try {
          const file = await (await handle.getFileHandle('_cache_meta.json')).getFile();
          most = Math.max(most, JSON.parse(await file.text()).entries?.length ?? 0);
        } catch {
          // Not saved yet, or a write interrupted mid-way: not persisted.
        }
      }
      return most;
    } catch {
      return 0;
    }
  });
}

/**
 * The OPFS L2 state as the page sees it right now: every COMPLETED chunk file
 * (non-empty: a write's bytes only land at close) decoded back to its cache key,
 * and the number of entries the largest parsable on-disk index lists (-1 while
 * none parses: no index save has landed yet).
 */
async function l2OnDisk(page: Page): Promise<{ keys: string[]; indexed: number }> {
  return page.evaluate(async () => {
    const decode = (name: string): string => {
      const b64 = name.replace(/-/g, '+').replace(/_/g, '/');
      const bin = atob(b64 + '='.repeat((4 - (b64.length % 4)) % 4));
      return new TextDecoder().decode(Uint8Array.from(bin, (c) => c.charCodeAt(0)));
    };
    const keys: string[] = [];
    let indexed = -1;
    try {
      const root = await navigator.storage.getDirectory();
      const namespace = await root.getDirectoryHandle('luxar');
      for await (const dataset of (namespace as any).values()) {
        if (dataset.kind !== 'directory') continue;
        for await (const entry of dataset.values()) {
          if (entry.kind === 'file' && entry.name === '_cache_meta.json') {
            try {
              const text = await (await entry.getFile()).text();
              indexed = Math.max(indexed, JSON.parse(text).entries?.length ?? 0);
            } catch {
              // Zero-byte (a save cut off) or mid-write: not an index yet.
            }
          }
          if (entry.kind !== 'directory' || !/^[0-9a-f]{2}$/.test(entry.name)) continue;
          for await (const file of entry.values()) {
            if (file.kind !== 'file' || (await file.getFile()).size === 0) continue;
            try {
              keys.push(decode(file.name));
            } catch {
              // Not a cache-key file name.
            }
          }
        }
      }
    } catch {
      // No namespace dir yet.
    }
    return { keys, indexed };
  });
}

test.describe('L2 persistence across page reload (R7)', () => {
  test('a reload inside the index-save debounce loses no chunk written before it', async ({
    page,
    browserName,
  }) => {
    test.skip(browserName !== 'chromium', 'Real OPFS persistence is Chromium-only in CI');
    const context = page.context();

    // Spread the cold load's chunk responses ~120 ms apart, so its L2 writes keep
    // arriving after the index's 150 ms leading-edge save and the gaps between
    // them never reach the 1 s trailing debounce. Metadata documents are not
    // delayed, so the scene opens at once.
    const metadataDoc = /(^|\/)(\.zattrs|\.zarray|\.zgroup|\.zmetadata|zarr\.json)$/;
    let delayed = 0;
    const stagger = async (route: Route) => {
      const path = new URL(route.request().url()).pathname;
      if (!metadataDoc.test(path)) {
        const wait = 120 * delayed++;
        await new Promise((resolve) => setTimeout(resolve, Math.min(wait, 6000)));
      }
      await route.continue();
    };
    await context.route(`${DATASET}/**`, stagger);

    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Reload the moment the window this test is about is open: the leading
    // save has landed (an index parses), chunk files exist that it does not list,
    // and the last L2 write is under 300 ms old. Polling in the page keeps the
    // gap between the decision and the reload to one round trip.
    const writes = () =>
      page.evaluate(() => (window as any).__luxarDebug.cache.getStats().l2?.writes ?? 0);
    let lastWrites = await writes();
    let lastWriteAt = Date.now();
    let onDisk: { keys: string[]; indexed: number } = { keys: [], indexed: -1 };
    await expect
      .poll(
        async () => {
          const now = await writes();
          if (now !== lastWrites) {
            lastWrites = now;
            lastWriteAt = Date.now();
          }
          onDisk = await l2OnDisk(page);
          return (
            onDisk.indexed >= 0 &&
            onDisk.keys.length > onDisk.indexed &&
            Date.now() - lastWriteAt < 300
          );
        },
        { timeout: 20000, intervals: [25] }
      )
      .toBe(true);
    const beforeReload = onDisk.keys.map((k) => k.replace(/^\/+/, ''));

    // Everything the reloaded page fetches from the network, unthrottled.
    await context.unroute(`${DATASET}/**`, stagger);
    const fetched: string[] = [];
    context.on('request', (request) => {
      const url = request.url();
      if (url.startsWith(`${DATASET}/`)) fetched.push(url.slice(DATASET.length + 1));
    });
    await page.reload();
    await waitForLuxarReady(page);
    await waitForCacheStable(page);

    // Validation re-fetches the root metadata documents with a cache bypass:
    // those requests are not L2 misses.
    const isRootDoc = (key: string) => !key.includes('/') && metadataDoc.test(key);
    const refetched = beforeReload.filter((key) => !isRootDoc(key) && fetched.includes(key));
    expect(
      { refetched, onDiskBeforeReload: beforeReload.length, indexedBeforeReload: onDisk.indexed },
      'chunks on disk before the reload that were fetched again after it'
    ).toEqual({
      refetched: [],
      onDiskBeforeReload: beforeReload.length,
      indexedBeforeReload: onDisk.indexed,
    });
  });

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

    // `count` is the IN-MEMORY index. What makes the files findable after a
    // reload is the index file, saved on a debounce (METADATA_SAVE_DELAY in
    // opfs-store.ts), not per write; and a save started from the unload events
    // does not survive a navigation (measured: the old page dies after
    // `getFileHandle(create)`, leaving a zero-byte index). Persistence across a
    // reload is therefore a property of PERSISTED entries: wait for the on-disk
    // index to list every entry counted above. Without this the test only passed
    // while a load was slow enough for the debounce to fire before the reload.
    await expect
      .poll(() => persistedIndexEntries(page), { timeout: 15000 })
      .toBeGreaterThanOrEqual(before.l2.count);

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

  test('?clearCache invokes clearAll on init (S4)', async ({ page, browserName }) => {
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

    // Pass 2: reload with ?clearCache. The store's init path must
    // increment `clearOnInitCount` exactly once when ?clearCache
    // is present. This is the real observable signal — previously
    // the test only asserted `l2.size >= 0`, which is trivially true.
    await page.goto(`/?src=${DATASET}&debug&clearCache`);
    await waitForLuxarReady(page);

    const afterClear = await page.evaluate(async () =>
      (window as any).__luxarDebug.cache.getStats()
    );
    expect(afterClear.clearOnInitCount).toBeGreaterThanOrEqual(1);
  });
});
