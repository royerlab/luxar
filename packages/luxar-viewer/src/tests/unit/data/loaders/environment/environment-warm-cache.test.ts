/**
 * A baked environment attached AFTER a visit still loads on the next visit.
 *
 * `luxar env attach` re-consolidates the store but deliberately leaves the root
 * `content_hash` alone (the sidecar is excluded from the digest), and the L2 tier
 * revalidates the root only by that hash — so a warm OPFS cache keeps serving the
 * pre-attach root index. Trusting that index's silence made physical meshes lose
 * the baked map, silently. The loader now trusts it only when the root came from
 * the network this load.
 *
 * Real `setupCaches`, real `MultiLevelCachingStore` over an in-memory OPFS, the
 * real zarr facade; only `fetch` is replaced.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setupCaches } from '../../../../../data/scene-loader/cache/cache-setup';
import * as zarr from '../../../../../data/zarr';
import {
  loadBakedEnvironment,
  resetEnvironmentProbeCacheForTests,
} from '../../../../../data/loaders/environment/environment-loader';
import { resetRootDocumentPrefetchForTests } from '../../../../../cache/root-document-prefetch';
import { config as appConfig } from '../../../../../config';
import { log } from '../../../../../utils/log';
import { createFakeOpfsRoot } from '../../../../mocks/opfs.mock';

const BASE = 'https://cdn.example/scene.luxar.zarr/';
const HASH = 'hash-v1';
const RES = 1;

const environmentGroup = {
  zarr_format: 3,
  node_type: 'group',
  attributes: {
    format: 'cube-faces-half',
    sample_format: 'half-float-bits',
    face_order: ['px', 'nx', 'py', 'ny', 'pz', 'nz'],
    resolution: RES,
    scene_content_hash: HASH,
    faces: 'faces-0a1b',
  },
};
const facesArray = {
  zarr_format: 3,
  node_type: 'array',
  shape: [6 * RES * RES * 4],
  data_type: 'uint16',
  chunk_grid: { name: 'regular', configuration: { chunk_shape: [6 * RES * RES * 4] } },
  chunk_key_encoding: { name: 'default', configuration: { separator: '/' } },
  fill_value: 0,
  codecs: [{ name: 'bytes', configuration: { endian: 'little' } }],
  attributes: {},
};

/** The root document, before or after `luxar env attach` — same content_hash. */
function rootDocument(attached: boolean): string {
  const metadata: Record<string, unknown> = {
    points: { zarr_format: 3, node_type: 'group', attributes: {} },
  };
  if (attached) {
    metadata.environment = environmentGroup;
    metadata['environment/faces-0a1b'] = facesArray;
  }
  return JSON.stringify({
    zarr_format: 3,
    node_type: 'group',
    attributes: { content_hash: HASH, scene_dimensions: { dimensions: [] } },
    consolidated_metadata: { kind: 'inline', must_understand: false, metadata },
  });
}

let requested: string[] = [];

/** Serve the store as it is on disk: before or after the attach. */
function serveStore(attached: boolean): void {
  requested = [];
  const faces = new Uint16Array(6 * RES * RES * 4).fill(0x3c00);
  const files = new Map<string, BodyInit>([[`${BASE}zarr.json`, rootDocument(attached)]]);
  if (attached) {
    files.set(`${BASE}environment/zarr.json`, JSON.stringify(environmentGroup));
    files.set(`${BASE}environment/faces-0a1b/zarr.json`, JSON.stringify(facesArray));
    files.set(`${BASE}environment/faces-0a1b/c/0`, new Uint8Array(faces.buffer));
  }
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      requested.push(url);
      const body = files.get(url);
      return body === undefined
        ? new Response('', { status: 404 })
        : new Response(body, { status: 200 });
    })
  );
}

async function load() {
  const caches = await setupCaches(BASE, {});
  const store = (await zarr.openStore(caches.rawStore)) as zarr.Readable;
  const rootLoc = zarr.root(store);
  const group = await zarr.openGroupPreferV3(rootLoc);
  caches.releaseRootDocument();
  const env = await loadBakedEnvironment(
    rootLoc,
    group.attrs.content_hash as string,
    BASE,
    caches.rootIndexFromNetwork()
  );
  return { caches, env };
}

describe('baked environment behind a warm root cache', () => {
  let originalEnabled: boolean;

  beforeEach(() => {
    originalEnabled = appConfig.cache.enabled;
    appConfig.cache.enabled = true;
    resetRootDocumentPrefetchForTests();
    resetEnvironmentProbeCacheForTests();
    createFakeOpfsRoot().install();
    vi.spyOn(log, 'warning').mockImplementation(() => {});
    vi.spyOn(log, 'info').mockImplementation(() => {});
  });

  afterEach(() => {
    appConfig.cache.enabled = originalEnabled;
    resetRootDocumentPrefetchForTests();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('a cold load trusts the network index: no sidecar, no environment request', async () => {
    serveStore(false);
    const { caches, env } = await load();
    expect(caches.rootIndexFromNetwork()).toBe(true);
    expect(env).toBeNull();
    expect(requested.filter((url) => url.includes('/environment'))).toEqual([]);
    await caches.cachingStore?.dispose();
  });

  it('finds a sidecar attached after the root was cached (content_hash unchanged)', async () => {
    serveStore(false);
    const first = await load();
    expect(first.env).toBeNull();
    await first.caches.cachingStore?.dispose(); // drains the L2 writes

    serveStore(true); // `luxar env attach` ran; content_hash is the same
    const second = await load();
    expect(second.caches.rootIndexFromNetwork()).toBe(false); // the stale L2 index
    expect(second.env).not.toBeNull();
    expect(second.env?.resolution).toBe(RES);
    await second.caches.cachingStore?.dispose();
  });
});
