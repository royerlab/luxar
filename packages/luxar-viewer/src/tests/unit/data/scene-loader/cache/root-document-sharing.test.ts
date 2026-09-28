/**
 * A cold load fetches the dataset's root document ONCE.
 *
 * Cache validation (bypassing the Luxar tiers) and the store open (through them)
 * each used to fetch `zarr.json`, back to back: a host that does not let the
 * browser cache it (`no-store`) paid twice — 6.5 MB for the h2afva scene. The
 * two now share one in-flight response (`cache/root-document-prefetch.ts`).
 *
 * Real `setupCaches`, real `MultiLevelCachingStore` over an in-memory OPFS, the
 * real zarr facade; only `fetch` is replaced, and it counts.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setupCaches } from '../../../../../data/scene-loader/cache/cache-setup';
import * as zarr from '../../../../../data/zarr';
import {
  prefetchRootDocument,
  resetRootDocumentPrefetchForTests,
} from '../../../../../cache/root-document-prefetch';
import { config as appConfig } from '../../../../../config';
import { createFakeOpfsRoot } from '../../../../mocks/opfs.mock';

const BASE = 'https://cdn.example/scene.luxar.zarr/';
const ROOT = `${BASE}zarr.json`;
const ETAG = '"root-v1"';

const rootDocument = JSON.stringify({
  zarr_format: 3,
  node_type: 'group',
  attributes: { content_hash: 'hash-v1', scene_dimensions: { dimensions: [] } },
  consolidated_metadata: {
    kind: 'inline',
    must_understand: false,
    metadata: { points: { zarr_format: 3, node_type: 'group', attributes: {} } },
  },
});

let requested: string[];

function installFetch(respond?: (url: string, init?: RequestInit) => Promise<Response>): void {
  requested = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      requested.push(url);
      if (respond) return respond(url, init);
      if (url === ROOT) {
        return new Response(rootDocument, { status: 200, headers: { etag: ETAG } });
      }
      return new Response('', { status: 404 });
    })
  );
}

const rootRequests = (): number => requested.filter((url) => url === ROOT).length;

async function coldLoad(flags: Parameters<typeof setupCaches>[1]) {
  const caches = await setupCaches(BASE, flags);
  const store = (await zarr.openStore(caches.rawStore)) as zarr.Readable;
  const group = await zarr.openGroupPreferV3(zarr.root(store));
  return { caches, group };
}

describe('cold load: one root-document request', () => {
  let originalEnabled: boolean;

  beforeEach(() => {
    originalEnabled = appConfig.cache.enabled;
    appConfig.cache.enabled = true;
    resetRootDocumentPrefetchForTests();
    createFakeOpfsRoot().install();
    installFetch();
  });

  afterEach(async () => {
    appConfig.cache.enabled = originalEnabled;
    resetRootDocumentPrefetchForTests();
    vi.unstubAllGlobals();
  });

  it('validation and the store open share ONE zarr.json fetch (OPFS on)', async () => {
    const { caches, group } = await coldLoad({});
    expect(group.attrs.content_hash).toBe('hash-v1');
    expect(caches.cachingStore?.getStats().health.validationMode).toBe('content-hash');
    expect(rootRequests()).toBe(1);
    await caches.cachingStore?.dispose();
  });

  it('adopts the bootstrap prefetch instead of fetching again', async () => {
    prefetchRootDocument(BASE);
    await vi.waitFor(() => expect(rootRequests()).toBe(1)); // on the wire before any cache exists
    const { caches, group } = await coldLoad({});
    expect(group.attrs.content_hash).toBe('hash-v1');
    expect(rootRequests()).toBe(1);
    await caches.cachingStore?.dispose();
  });

  it('shares on the ?noOpfs and ?noCache paths too', async () => {
    const noOpfs = await coldLoad({ noOpfs: true });
    expect(noOpfs.group.attrs.content_hash).toBe('hash-v1');
    expect(rootRequests()).toBe(1);
    await noOpfs.caches.cachingStore?.dispose();

    installFetch();
    const noCache = await coldLoad({ noCache: true });
    expect(noCache.group.attrs.content_hash).toBe('hash-v1');
    expect(rootRequests()).toBe(1);
  });

  it("exposes the shared response's ETag for the identity watchdog", async () => {
    const { caches } = await coldLoad({});
    const shared = await caches.rootDocument;
    expect(shared?.served?.doc).toBe('zarr.json');
    expect(shared?.served?.etag).toBe(ETAG);
    await caches.cachingStore?.dispose();
  });

  it('a second load of the same URL validates against the server afresh', async () => {
    const first = await coldLoad({});
    first.caches.releaseRootDocument();
    await first.caches.cachingStore?.dispose();
    const second = await coldLoad({});
    // One more root fetch (the new validation); the open is served from L1/L2
    // or from that same fetch — never a third.
    expect(rootRequests()).toBe(2);
    await second.caches.cachingStore?.dispose();
  });

  it('keeps validation fail-fast: a hung root fetch still lets validation give up on budget', async () => {
    const network = appConfig.dataLoading.network as { validationTimeoutMs: number };
    const original = network.validationTimeoutMs;
    network.validationTimeoutMs = 50;
    try {
      // Hangs until aborted, as a real fetch to a stalled server does.
      installFetch(
        (_url, init) =>
          new Promise<Response>((_resolve, reject) => {
            init?.signal?.addEventListener('abort', () =>
              reject(new DOMException('aborted', 'AbortError'))
            );
          })
      );
      const started = Date.now();
      const caches = await setupCaches(BASE, {});
      // Validation gave up (no token) on ITS budget, not the 30 s data budget
      // the shared fetch runs under.
      expect(Date.now() - started).toBeLessThan(5_000);
      expect(caches.cachingStore?.getStats().health.validationMode).not.toBe('content-hash');
      await caches.cachingStore?.dispose();
    } finally {
      network.validationTimeoutMs = original;
    }
  });
});
