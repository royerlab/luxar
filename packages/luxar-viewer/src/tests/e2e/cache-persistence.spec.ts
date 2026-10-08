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
    // A chunk file is `{hash tag}.{base64url key}` (bare base64url without a hash).
    const decode = (name: string): string => {
      const b64 = name
        .slice(name.indexOf('.') + 1)
        .replace(/-/g, '+')
        .replace(/_/g, '/');
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
    // Two cold-ish loads (the gated first load, then the post-reload one)
    // plus the waits for the index saves: about twice a plain load test.
    test.setTimeout(90_000);
    const context = page.context();

    // The window this test needs (an index on disk, plus a chunk file it does
    // not list yet) used to be raced in real time: chunk responses were spread
    // ~120 ms apart and the reload fired while the last write was under 300 ms
    // old. On a busy machine the poll itself could outlast that budget, so the
    // window could close unseen (#3085). Chunk responses are now held until
    // the test releases them, so the writes land in the order the test needs.
    // Metadata documents pass straight through, so the scene opens at once.
    const metadataDoc = /(^|\/)(\.zattrs|\.zarray|\.zgroup|\.zmetadata|zarr\.json)$/;
    // A held request the viewer gave up on (its per-attempt network timeout,
    // ~7.5 s, aborts it and queues a retry at the back) must not consume a
    // release: skip it, so the allowance reaches a request still waiting.
    const held: Array<{ route: Route; resolve: () => void }> = [];
    let allowance = 0;
    let chunkRequests = 0;
    // Chunk keys as the gate saw them, normalized like `fetched` below, so the
    // decoded OPFS keys can be checked against real request paths.
    const gatedPaths = new Set<string>();
    const release = (n: number) => {
      allowance += n;
      while (allowance > 0 && held.length > 0) {
        const next = held.shift()!;
        if (next.route.request().failure() === null) allowance--;
        next.resolve();
      }
    };
    const gate = async (route: Route) => {
      const url = route.request().url();
      const path = new URL(url).pathname;
      if (!metadataDoc.test(path)) {
        chunkRequests++;
        gatedPaths.add(url.slice(DATASET.length + 1));
        if (allowance > 0) allowance--;
        else await new Promise<void>((resolve) => held.push({ route, resolve }));
      }
      await route.continue().catch(() => {
        // Released only after the reload: the page that asked for it is gone.
      });
    };
    await context.route(`${DATASET}/**`, gate);

    // No waitForLuxarReady here: the viewer only reports `initialized` once its
    // first chunks have loaded, which the gate now withholds. Everything below
    // reads OPFS directly, and the stats counter only once the debug surface
    // has it.
    await page.goto(`/?src=${DATASET}&debug`);

    // What a failed wait reports: which condition was missing, not just that a
    // boolean never flipped. `writes` is null until init installs the cache
    // debug helpers; the last-write clock follows the completed files on disk.
    const writes = () =>
      page.evaluate(
        () => (window as any).__luxarDebug?.cache?.getStats?.().l2?.writes ?? null
      ) as Promise<number | null>;
    let onDisk: { keys: string[]; indexed: number } = { keys: [], indexed: -1 };
    let lastWriteAt = Date.now();
    // Stage-2 chunk releases so far (0 until stage 2 begins). Part of every
    // observation, so a timed-out wait still reports how many retries it took.
    let stage2Releases = 0;
    const observe = async () => {
      const previous = onDisk.keys.length;
      const l2Writes = await writes();
      onDisk = await l2OnDisk(page);
      if (onDisk.keys.length !== previous) lastWriteAt = Date.now();
      return {
        indexed: onDisk.indexed,
        onDisk: onDisk.keys.length,
        // The root `zarr.json` passes the gate and is cached too: count chunks apart.
        chunksOnDisk: onDisk.keys.filter((key) => !metadataDoc.test(key)).length,
        writes: l2Writes,
        msSinceLastWrite: Date.now() - lastWriteAt,
        chunkRequests,
        held: held.length,
        stage2Releases,
      };
    };
    const waitFor = (stage: string, done: (s: Awaited<ReturnType<typeof observe>>) => boolean) =>
      expect
        .poll(
          async () => {
            // A string, so a timeout prints the whole last observation as the
            // received value (an object matcher would diff only the flag).
            const state = await observe();
            return done(state) ? 'ready' : JSON.stringify(state);
          },
          { message: stage, timeout: 20000, intervals: [25] }
        )
        .toBe('ready');

    // 1. One chunk, then wait for an index save that lists it and every other
    //    completed file. Whether that save is the 150 ms leading edge or the
    //    1 s trailing debounce (opening the dataset sets its content hash, which
    //    saves the still-empty index, so a write inside the next second gets no
    //    leading edge) does not matter:
    //    what matters is that a save has JUST started.
    release(1);
    await waitFor(
      'an index save listing the first released chunk and every other completed file',
      (s) => s.chunksOnDisk >= 1 && s.indexed === s.onDisk
    );

    // 2. One more chunk. Usually its write follows that save by well under the
    //    1 s METADATA_SAVE_DELAY, so it gets no leading-edge save of its own and
    //    waits on the trailing debounce. Wait until it is on disk but unlisted:
    //    exactly the state a reload must not lose. On a loaded machine the
    //    round trip can outlast that second; the write then gets its own 150 ms
    //    leading-edge save, and the poll may only ever see it already listed.
    //    So whenever a released chunk has landed AND been indexed, release
    //    another and keep polling, rather than betting on a single release.
    let chunksAtRelease = onDisk.keys.filter((key) => !metadataDoc.test(key)).length;
    stage2Releases = 1;
    release(1);
    await waitFor('a completed chunk file the on-disk index does not list yet', (s) => {
      // These compare ALL completed files (`onDisk`) with the index, not chunks
      // alone, because `l2OnDisk` does not say which indexed entries are chunks.
      // That is safe here only because the root `zarr.json` of this consolidated
      // format-3 store is the sole non-chunk L2 entry and is already on disk and
      // indexed after stage 1, so any surplus file or new indexed file is a chunk.
      if (s.indexed >= 0 && s.onDisk > s.indexed) return true;
      if (s.chunksOnDisk > chunksAtRelease && s.indexed === s.onDisk) {
        chunksAtRelease = s.chunksOnDisk;
        stage2Releases++;
        release(1);
      }
      return false;
    });
    test.info().annotations.push({
      type: 'stage-2 releases',
      description: String(stage2Releases),
    });
    const beforeReload = onDisk.keys.map((k) => k.replace(/^\/+/, ''));
    // Positive control: a decoded OPFS chunk key must equal a path the gate
    // actually served. A key decode that silently produced wrong strings would
    // match nothing in `fetched`, making the final "nothing refetched" check
    // pass vacuously; this ties the two key formats together first.
    expect(
      beforeReload.filter((key) => !metadataDoc.test(key) && gatedPaths.has(key)),
      `decoded on-disk chunk keys must match gated request paths (gated: ${[...gatedPaths].join(', ')})`
    ).not.toHaveLength(0);

    // 3. Release one more and reload at once, with the trailing save pending.
    release(1);

    // Everything the reloaded page fetches from the network, ungated.
    await context.unroute(`${DATASET}/**`, gate);
    const fetched: string[] = [];
    context.on('request', (request) => {
      const url = request.url();
      if (url.startsWith(`${DATASET}/`)) fetched.push(url.slice(DATASET.length + 1));
    });
    await page.reload();
    // Requests the old page left behind the gate: let their handlers finish.
    release(Infinity);
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
