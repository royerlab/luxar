/**
 * The pack format across the language boundary: `luxar optimize --pack`
 * (Python, `luxar.io.chunk_pack`) writes it, `PackedChunkSource` reads it.
 *
 * The fixture is a small laddered partition run through the real writer
 * (`generate_chunk_packs_test` in tests/fixtures/generate_test_data.py). Every
 * chunk object a packed node stores must come back from its pack byte-identical
 * to the plain chunk, at the cost of ONE request per pack.
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

import { FileSystemStore } from '@zarrita/storage';
import { beforeAll, describe, expect, it } from 'vitest';

import * as zarr from '../../../../data/zarr';
import {
  adoptChunkPacks,
  CHUNK_PACKS_GROUP,
} from '../../../../data/scene-loader/cache/chunk-packs';
import { PackedChunkSource } from '../../../../cache/chunk-source/packed-chunk-source';
import type { ChunkFetchOutcome, ChunkSource } from '../../../../cache/chunk-source';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = path.resolve(
  __dirname,
  '../../../../../tests/fixtures/test_chunk_packs.luxar.zarr'
);

/** The fixture's own bytes as a ChunkSource, recording every key read. */
function diskSource(store: FileSystemStore) {
  const reads: string[] = [];
  const source: ChunkSource = {
    identity: FIXTURE,
    describe: FIXTURE,
    async get(key: string): Promise<ChunkFetchOutcome> {
      reads.push(key);
      const data = await store.get(`/${key.replace(/^\/+/, '')}` as `/${string}`);
      return data ? { kind: 'ok', data, bytesOverWire: data.byteLength } : { kind: 'missing' };
    },
    probeIdentityToken: async () => null,
    dispose: () => {},
  };
  return { source, reads };
}

/** Every chunk object (non-metadata file) stored under a node prefix. */
function chunkFiles(prefix: string): string[] {
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const entry of fs.readdirSync(path.join(FIXTURE, dir), { withFileTypes: true })) {
      const rel = `${dir}${entry.name}`;
      if (entry.isDirectory()) walk(`${rel}/`);
      else if (entry.name !== 'zarr.json' && !entry.name.startsWith('.z')) out.push(rel);
    }
  };
  walk(prefix);
  return out.sort();
}

interface Index {
  scene_content_hash: string;
  packs: Array<{ key: string; sha256: string; prefix: string }>;
}

describe('chunk packs written by luxar optimize --pack (Python-written, TS-read)', () => {
  let store: FileSystemStore;
  let rootLoc: zarr.Location<zarr.Readable>;
  let contentHash: unknown;
  let index: Index;

  beforeAll(async () => {
    store = new FileSystemStore(FIXTURE);
    rootLoc = zarr.root((await zarr.openStore(store)) as zarr.Readable);
    contentHash = (await zarr.openGroup(rootLoc)).attrs.content_hash;
    index = (await zarr.openGroup(rootLoc.resolve(CHUNK_PACKS_GROUP))).attrs as unknown as Index;
  });

  it('serves every member byte-identical to its plain chunk, one request per pack', async () => {
    const { source, reads } = diskSource(store);
    const packs = new PackedChunkSource(source);
    expect(await adoptChunkPacks(packs, rootLoc, contentHash, true, rootLoc)).toBe(
      index.packs.length
    );
    expect(index.packs.length).toBeGreaterThan(1);

    let members = 0;
    for (const { prefix } of index.packs) {
      const files = chunkFiles(prefix);
      expect(files.length).toBeGreaterThan(1);
      for (const key of files) {
        const outcome = await packs.get(`/${key}`);
        expect(outcome.kind, key).toBe('ok');
        if (outcome.kind !== 'ok') continue;
        expect(Array.from(outcome.data), key).toEqual(
          Array.from(fs.readFileSync(path.join(FIXTURE, key)))
        );
        members++;
      }
    }
    expect(members).toBeGreaterThan(index.packs.length);
    expect(reads.sort()).toEqual(index.packs.map((p) => p.key).sort());
  });

  it('reads plainly from a pack whose digest does not match', async () => {
    const { source, reads } = diskSource(store);
    const packs = new PackedChunkSource(source);
    const corrupted = {
      ...index,
      packs: index.packs.map((p) => ({ ...p, sha256: '0'.repeat(64) })),
    };
    expect(packs.usePacks(corrupted, contentHash)).toBe(index.packs.length);

    const files = chunkFiles(index.packs[0].prefix);
    for (const key of files) {
      const outcome = await packs.get(key);
      expect(outcome.kind, key).toBe('ok');
      if (outcome.kind !== 'ok') continue;
      expect(Array.from(outcome.data), key).toEqual(
        Array.from(fs.readFileSync(path.join(FIXTURE, key)))
      );
    }
    expect(reads).toEqual([index.packs[0].key, ...files]);
  });
});
