/**
 * `loadBakedEnvironment` request budget: the optional `environment/` sidecar must
 * not cost a scene without one any requests.
 *
 * Every Luxar store carries a consolidated index, and `luxar env attach`
 * re-consolidates after writing the group — so when the index exists it is the
 * authoritative answer to "is there an environment node?". Probing anyway cost 3
 * requests (`environment/zarr.json`, `.zattrs`, `.zgroup` — all 404) that
 * `sceneLoaded` waited on, serially over a real network.
 *
 * Real zarrita and the real facade over an in-memory store that records every
 * key it is asked for — no mocking of the zarr boundary, so the count is the
 * count the network would see.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as zarr from '../../../../../data/zarr';
import {
  loadBakedEnvironment,
  resetEnvironmentProbeCacheForTests,
} from '../../../../../data/loaders/environment/environment-loader';
import { log } from '../../../../../utils/log';

const encoder = new TextEncoder();
const HASH = 'feedface00';
const RES = 1;

class RecordingStore implements zarr.AsyncReadable {
  readonly keys: string[] = [];
  constructor(private readonly entries: Map<string, Uint8Array>) {}
  async get(key: zarr.AbsolutePath): Promise<Uint8Array | undefined> {
    this.keys.push(key);
    return this.entries.get(key);
  }
  environmentKeys(): string[] {
    return this.keys.filter((k) => k.startsWith('/environment'));
  }
}

const json = (value: unknown): Uint8Array => encoder.encode(JSON.stringify(value));

function environmentGroupMeta(): Record<string, unknown> {
  return {
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
}

function facesArrayMeta(): Record<string, unknown> {
  return {
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
}

/** A format-3 store with a consolidated root index; `withEnvironment` adds the sidecar. */
function consolidatedStore(withEnvironment: boolean): RecordingStore {
  const metadata: Record<string, unknown> = {
    points: { zarr_format: 3, node_type: 'group', attributes: { type: 'points' } },
  };
  const entries = new Map<string, Uint8Array>();
  if (withEnvironment) {
    metadata.environment = environmentGroupMeta();
    metadata['environment/faces-0a1b'] = facesArrayMeta();
    entries.set('/environment/zarr.json', json(environmentGroupMeta()));
    entries.set('/environment/faces-0a1b/zarr.json', json(facesArrayMeta()));
    const data = new Uint16Array(6 * RES * RES * 4).fill(0x3c00);
    entries.set('/environment/faces-0a1b/c/0', new Uint8Array(data.buffer));
  }
  entries.set(
    '/zarr.json',
    json({
      zarr_format: 3,
      node_type: 'group',
      attributes: { content_hash: HASH },
      consolidated_metadata: { kind: 'inline', must_understand: false, metadata },
    })
  );
  return new RecordingStore(entries);
}

/** A format-3 store WITHOUT a consolidated index (no `contents()` listing). */
function unconsolidatedStore(): RecordingStore {
  return new RecordingStore(
    new Map([
      [
        '/zarr.json',
        json({ zarr_format: 3, node_type: 'group', attributes: { content_hash: HASH } }),
      ],
    ])
  );
}

async function rootOf(raw: RecordingStore): Promise<zarr.Location<zarr.Readable>> {
  const store = (await zarr.openStore(raw)) as zarr.Readable;
  return zarr.root(store);
}

describe('loadBakedEnvironment request budget', () => {
  beforeEach(() => {
    resetEnvironmentProbeCacheForTests();
    vi.spyOn(log, 'warning').mockImplementation(() => {});
    vi.spyOn(log, 'info').mockImplementation(() => {});
  });

  it('makes NO environment request on a consolidated store that lists no environment node', async () => {
    const raw = consolidatedStore(false);
    const rootLoc = await rootOf(raw);
    expect(await loadBakedEnvironment(rootLoc, HASH)).toBeNull();
    expect(raw.environmentKeys()).toEqual([]);
  });

  it('still loads the map when the consolidated index lists the environment node', async () => {
    const raw = consolidatedStore(true);
    const rootLoc = await rootOf(raw);
    const env = await loadBakedEnvironment(rootLoc, HASH);
    expect(env).not.toBeNull();
    expect(env?.resolution).toBe(RES);
  });

  it('negative-caches an absent sidecar per store key on an unconsolidated store', async () => {
    const first = unconsolidatedStore();
    expect(
      await loadBakedEnvironment(await rootOf(first), HASH, 'http://h/scene.zarr/')
    ).toBeNull();
    expect(first.environmentKeys().length).toBeGreaterThan(0);

    // Same dataset opened again this session: the 404 is remembered.
    const second = unconsolidatedStore();
    expect(
      await loadBakedEnvironment(await rootOf(second), HASH, 'http://h/scene.zarr/')
    ).toBeNull();
    expect(second.environmentKeys()).toEqual([]);
  });
});
