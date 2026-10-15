/**
 * Request budget for OPTIONAL dataset features on a consolidated store: a scene
 * without overlays, a baked environment or chunk packs must not request any of
 * their documents — each such request is a console `GET … 404` the user sees on
 * every load — and the format the store turned out to be must stop the other
 * format's documents from being requested at all.
 *
 * Real zarrita and the real facade over an in-memory store that records every
 * key it is asked for — no mocking of the zarr boundary, so the recorded keys are
 * the requests the network would see.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as zarr from '../../../data/zarr';
import { loadOverlayConfigs } from '../../../data/loaders/overlays/overlay-loader';
import {
  loadBakedEnvironment,
  resetEnvironmentProbeCacheForTests,
} from '../../../data/loaders/environment/environment-loader';
import { adoptChunkPacks } from '../../../data/scene-loader/cache/chunk-packs';
import {
  consolidatedGroupsOf,
  resolveRootListing,
} from '../../../data/scene-loader/cache/root-listing';
import { PackedChunkSource } from '../../../cache/chunk-source/packed-chunk-source';
import type { ChunkSource } from '../../../cache/chunk-source';
import type { RootDocumentFetch } from '../../../cache/root-document-prefetch';
import { log } from '../../../utils/log';

const encoder = new TextEncoder();
const json = (value: unknown): Uint8Array => encoder.encode(JSON.stringify(value));
const HASH = 'feedface00';

class RecordingStore implements zarr.AsyncReadable {
  readonly keys: string[] = [];
  constructor(private readonly entries: Map<string, Uint8Array>) {}
  async get(key: zarr.AbsolutePath): Promise<Uint8Array | undefined> {
    this.keys.push(key);
    return this.entries.get(key);
  }
  /** Keys requested that the store does not hold — the 404s. */
  misses(): string[] {
    return this.keys.filter((k) => !this.entries.has(k));
  }
}

const v3Array = {
  zarr_format: 3,
  node_type: 'array',
  shape: [4, 3],
  data_type: 'float32',
  chunk_grid: { name: 'regular', configuration: { chunk_shape: [4, 3] } },
  chunk_key_encoding: { name: 'default', configuration: { separator: '/' } },
  fill_value: 0,
  codecs: [{ name: 'bytes', configuration: { endian: 'little' } }],
  attributes: {},
};

/** A format-3 store with a consolidated root index: one points node, no sidecars. */
function v3Store(extraMetadata: Record<string, unknown> = {}): RecordingStore {
  const metadata: Record<string, unknown> = {
    points: { zarr_format: 3, node_type: 'group', attributes: { type: 'points' } },
    'points/positions': v3Array,
    ...extraMetadata,
  };
  return new RecordingStore(
    new Map([
      [
        '/zarr.json',
        json({
          zarr_format: 3,
          node_type: 'group',
          attributes: { content_hash: HASH },
          consolidated_metadata: { kind: 'inline', must_understand: false, metadata },
        }),
      ],
    ])
  );
}

const v2Array = {
  zarr_format: 2,
  shape: [4, 3],
  chunks: [4, 3],
  dtype: '<f4',
  compressor: null,
  fill_value: 0,
  filters: null,
  order: 'C',
};

/** A format-2 store with `.zmetadata`; the positions array has NO `.zattrs`. */
function v2Store(): RecordingStore {
  const metadata: Record<string, unknown> = {
    '.zgroup': { zarr_format: 2 },
    '.zattrs': { content_hash: HASH },
    'points/.zgroup': { zarr_format: 2 },
    'points/.zattrs': { type: 'points' },
    'points/positions/.zarray': v2Array,
  };
  return new RecordingStore(
    new Map([['/.zmetadata', json({ zarr_consolidated_format: 1, metadata })]])
  );
}

/** The load-time root-document fetch, as `cache/root-document-prefetch.ts` settles it. */
function rootDocument(doc: 'zarr.json' | '.zattrs', bytes: Uint8Array): Promise<RootDocumentFetch> {
  const data = new Uint8Array(bytes);
  return Promise.resolve({
    outcomes: new Map([[doc, { kind: 'ok' as const, bytes: data, etag: null }]]),
    served: { doc, bytes: data, etag: null },
    token: () => Promise.resolve(null),
    peekToken: () => undefined,
  });
}

const packs = () => new PackedChunkSource({} as ChunkSource);

/** The optional-feature reads of a scene load, in load order. */
async function loadOptionalFeatures(
  raw: RecordingStore,
  opts: {
    indexFresh: boolean;
    rootDoc?: Promise<RootDocumentFetch> | null;
    source?: RecordingStore;
  }
): Promise<{ overlays: number; environment: unknown; packs: number }> {
  const store = (await zarr.openStore(raw)) as zarr.Readable;
  const rootLoc = zarr.root(store);
  await zarr.openGroupPreferV3(rootLoc);
  const listing = await resolveRootListing(store, opts.indexFresh, opts.rootDoc ?? null);
  const used = await adoptChunkPacks(packs(), rootLoc, HASH, listing, () =>
    zarr.root(opts.source ?? raw)
  );
  const overlays = await loadOverlayConfigs(store, rootLoc);
  const environment = await loadBakedEnvironment(rootLoc, HASH, 'http://h/s.zarr/', listing);
  return { overlays: overlays.length, environment, packs: used };
}

describe('optional-feature probes on a consolidated store', () => {
  beforeEach(() => {
    resetEnvironmentProbeCacheForTests();
    vi.spyOn(log, 'info').mockImplementation(() => {});
    vi.spyOn(log, 'warning').mockImplementation(() => {});
  });

  it('format 3, fresh index: only the root zarr.json is read — no absent-path request', async () => {
    const raw = v3Store();
    const result = await loadOptionalFeatures(raw, { indexFresh: true });
    expect(result).toEqual({ overlays: 0, environment: null, packs: 0 });
    expect(raw.keys).toEqual(['/zarr.json']);
  });

  it('format 3, warm load: the root-document fetch answers for the sidecars', async () => {
    const raw = v3Store();
    const network = await raw.get('/zarr.json' as zarr.AbsolutePath);
    raw.keys.length = 0;
    const result = await loadOptionalFeatures(raw, {
      indexFresh: false,
      rootDoc: rootDocument('zarr.json', network!),
    });
    expect(result).toEqual({ overlays: 0, environment: null, packs: 0 });
    expect(raw.keys).toEqual(['/zarr.json']);
  });

  it('format 2: no other-format document, no index-settled .zattrs, no absent-path request', async () => {
    const raw = v2Store();
    const store = (await zarr.openStore(raw)) as zarr.Readable;
    const rootLoc = zarr.root(store);
    // The consolidation attempt order (format 3 first) is the one unavoidable miss;
    // in production the load-time root-document fetch answers it.
    expect(raw.misses()).toEqual(['/zarr.json']);
    raw.keys.length = 0;

    expect(zarr.storeFormatOf(store)).toBe('v2');
    const root = await zarr.openGroupPreferV3(rootLoc);
    expect(root.attrs).toEqual({ content_hash: HASH });
    const positions = await zarr.open(rootLoc.resolve('points/positions'), { kind: 'array' });
    expect(positions.shape).toEqual([4, 3]);

    const listing = await resolveRootListing(store, true, null);
    expect(await adoptChunkPacks(packs(), rootLoc, HASH, listing, () => rootLoc)).toBe(0);
    expect(await loadOverlayConfigs(store, rootLoc)).toEqual([]);
    expect(await loadBakedEnvironment(rootLoc, HASH, 'http://h/s.zarr/', listing)).toBeNull();
    expect(raw.keys).toEqual([]);
  });

  it('an absent node on a known-format store costs ONE request, not a format guess', async () => {
    const raw = v3Store();
    const store = (await zarr.openStore(raw)) as zarr.Readable;
    raw.keys.length = 0;
    await expect(
      zarr.open(zarr.root(store).resolve('missing'), { kind: 'group' })
    ).rejects.toThrow();
    expect(raw.keys).toEqual(['/missing/zarr.json']);
  });

  it('format 2, warm load without a current index: each sidecar probe is ONE request', async () => {
    const raw = v2Store();
    const zattrs = json({ content_hash: HASH });
    const result = await loadOptionalFeatures(raw, {
      indexFresh: false,
      rootDoc: rootDocument('.zattrs', zattrs),
    });
    expect(result).toEqual({ overlays: 0, environment: null, packs: 0 });
    expect(raw.misses()).toEqual([
      '/zarr.json', // consolidation attempt order, see above
      '/chunk_packs/.zgroup',
      '/environment/.zgroup',
    ]);
  });

  it('a format-2 group probe on a raw store reads .zgroup once', async () => {
    const raw = new RecordingStore(
      new Map([
        ['/env/.zgroup', json({ zarr_format: 2 })],
        ['/env/.zattrs', json({ a: 1 })],
      ])
    );
    const group = await zarr.openGroupOfFormat(zarr.root(raw).resolve('env'), 'v2');
    expect(group.attrs).toEqual({ a: 1 });
    expect(raw.keys.filter((k) => k === '/env/.zgroup')).toHaveLength(1);
  });

  it('reads environment attrs from the source store on a warm load (not the cached index)', async () => {
    const envAttrs = (faces: string): Record<string, unknown> => ({
      zarr_format: 3,
      node_type: 'group',
      attributes: {
        format: 'cube-faces-half',
        sample_format: 'half-float-bits',
        face_order: ['px', 'nx', 'py', 'ny', 'pz', 'nz'],
        resolution: 1,
        scene_content_hash: HASH,
        faces,
      },
    });
    const facesMeta = { ...v3Array, shape: [24], data_type: 'uint16' };
    facesMeta.chunk_grid = { name: 'regular', configuration: { chunk_shape: [24] } };
    // The warm tier still holds the PRE-re-bake index (faces-old); the source
    // has the re-baked group (faces-new). content_hash is unchanged by design.
    const cached = v3Store({ environment: envAttrs('faces-old') });
    const facesData = new Uint8Array(new Uint16Array(24).fill(0x3c00).buffer);
    const raw = cached as unknown as { entries: Map<string, Uint8Array> };
    raw.entries.set('/environment/faces-new/zarr.json', json(facesMeta));
    raw.entries.set('/environment/faces-new/c/0', facesData);
    const source = new RecordingStore(
      new Map([['/environment/zarr.json', json(envAttrs('faces-new'))]])
    );
    const network = v3Store({ environment: envAttrs('faces-new') });
    const networkRoot = await network.get('/zarr.json' as zarr.AbsolutePath);

    const store = (await zarr.openStore(cached)) as zarr.Readable;
    const rootLoc = zarr.root(store);
    const listing = await resolveRootListing(store, false, rootDocument('zarr.json', networkRoot!));
    const env = await loadBakedEnvironment(rootLoc, HASH, 'http://h/s.zarr/', listing, () =>
      zarr.root(source)
    );
    expect(env).not.toBeNull();
    expect(source.keys).toEqual(['/environment/zarr.json']);
    expect(cached.keys).toContain('/environment/faces-new/zarr.json');
  });

  it('keeps a cached baked environment when the source is unreachable', async () => {
    const envGroup = {
      zarr_format: 3,
      node_type: 'group',
      attributes: {
        format: 'cube-faces-half',
        sample_format: 'half-float-bits',
        face_order: ['px', 'nx', 'py', 'ny', 'pz', 'nz'],
        resolution: 1,
        scene_content_hash: HASH,
        faces: 'faces-cached',
      },
    };
    const facesMeta = {
      ...v3Array,
      shape: [24],
      data_type: 'uint16',
      chunk_grid: { name: 'regular', configuration: { chunk_shape: [24] } },
    };
    const cached = v3Store({
      environment: envGroup,
      'environment/faces-cached': facesMeta,
    });
    const entries = (cached as unknown as { entries: Map<string, Uint8Array> }).entries;
    entries.set(
      '/environment/faces-cached/c/0',
      new Uint8Array(new Uint16Array(24).fill(0x3c00).buffer)
    );
    const store = (await zarr.openStore(cached)) as zarr.Readable;
    const rootLoc = zarr.root(store);
    const listing = await resolveRootListing(
      store,
      false,
      Promise.reject(new TypeError('Failed to fetch'))
    );
    const sourceReads: string[] = [];
    const source = {
      get: async (key: string) => {
        sourceReads.push(key);
        throw new TypeError('Failed to fetch');
      },
    } as zarr.AsyncReadable;

    const env = await loadBakedEnvironment(rootLoc, HASH, 'http://h/offline.zarr/', listing, () =>
      zarr.root(source)
    );
    expect(env?.header.faces).toBe('faces-cached');
    expect(env?.faces).toHaveLength(6);
    expect(env?.faces[0]).toEqual(new Uint16Array(4).fill(0x3c00));
    expect(sourceReads).toEqual(['/environment/zarr.json']);
    expect(cached.keys).not.toContain('/environment/zarr.json');
    // A connection fault must not make subsequent loads skip the probe.
    const recovered = await loadBakedEnvironment(
      rootLoc,
      HASH,
      'http://h/offline.zarr/',
      listing,
      () => rootLoc
    );
    expect(recovered?.header.faces).toBe('faces-cached');
  });

  it('does not remember a cached miss after the source fails', async () => {
    const cached = v3Store();
    const store = (await zarr.openStore(cached)) as zarr.Readable;
    const rootLoc = zarr.root(store);
    const listing = await resolveRootListing(
      store,
      false,
      Promise.reject(new TypeError('Failed to fetch'))
    );
    const sourceReads: string[] = [];
    const source = {
      get: async (key: string) => {
        sourceReads.push(key);
        throw new TypeError('Failed to fetch');
      },
    } as zarr.AsyncReadable;
    for (let attempt = 0; attempt < 2; attempt++) {
      expect(
        await loadBakedEnvironment(rootLoc, HASH, 'http://h/unreachable.zarr/', listing, () =>
          zarr.root(source)
        )
      ).toBeNull();
    }
    expect(sourceReads).toEqual(['/environment/zarr.json', '/environment/zarr.json']);
  });

  it('still loads chunk packs a fresh index lists', async () => {
    const raw = v3Store({
      chunk_packs: {
        zarr_format: 3,
        node_type: 'group',
        attributes: {
          scene_content_hash: HASH,
          packs: [{ key: 'chunk_packs/0.pack', sha256: 'x', prefix: 'points/' }],
        },
      },
    });
    const result = await loadOptionalFeatures(raw, { indexFresh: true });
    expect(result.packs).toBe(1);
  });

  it('finds chunk packs added after a cached index, via the network root document', async () => {
    const cached = v3Store();
    const packsMeta = {
      zarr_format: 3,
      node_type: 'group',
      attributes: {
        scene_content_hash: HASH,
        packs: [{ key: 'chunk_packs/0.pack', sha256: 'x', prefix: 'points/' }],
      },
    };
    const network = v3Store({ chunk_packs: packsMeta });
    const source = new RecordingStore(new Map([['/chunk_packs/zarr.json', json(packsMeta)]]));
    const networkRoot = await network.get('/zarr.json' as zarr.AbsolutePath);
    const result = await loadOptionalFeatures(cached, {
      indexFresh: false,
      rootDoc: rootDocument('zarr.json', networkRoot!),
      source,
    });
    expect(result.packs).toBe(1);
    // Read from the source (past the stale tier), in the one known format.
    expect(source.keys).toEqual(['/chunk_packs/zarr.json']);
  });
});

describe('consolidatedGroupsOf', () => {
  it('lists the group members of a format-3 root index', () => {
    const bytes = json({
      zarr_format: 3,
      node_type: 'group',
      consolidated_metadata: {
        metadata: {
          environment: { node_type: 'group' },
          'environment/faces-1': { node_type: 'array' },
        },
      },
    });
    expect(consolidatedGroupsOf(bytes)).toEqual(new Set(['environment']));
  });

  it('answers null for a document without an index, or not JSON', () => {
    expect(consolidatedGroupsOf(json({ zarr_format: 3, node_type: 'group' }))).toBeNull();
    expect(consolidatedGroupsOf(encoder.encode('{not json'))).toBeNull();
  });
});
