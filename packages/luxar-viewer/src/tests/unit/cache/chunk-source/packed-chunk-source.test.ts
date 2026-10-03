/**
 * {@link PackedChunkSource}: a small node's chunks from ONE request.
 *
 * `luxar optimize --pack` copies each small geometry node's chunk objects into
 * one pack object and indexes it in the root `chunk_packs` sidecar. The plain
 * chunks stay, so every failure mode here must land on the plain read — never
 * on wrong bytes: a stale index (built for another `content_hash`), a pack
 * whose bytes fail their digest, a pack the server no longer has.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { PackedChunkSource } from '../../../../cache/chunk-source/packed-chunk-source';
import { MultiLevelCachingStore } from '../../../../cache/multi-level-caching-store';
import { sha256HexPureJs } from '../../../../cache/multi-level-caching-store/sha256';
import type { ChunkFetchOutcome, ChunkSource } from '../../../../cache/chunk-source';

const HASH = 'scene-hash-1';

/** Two members of `part_0` in the pack-file layout: `a` = [1, 2, 3], `b` = [4, 5]. */
function packFile(members: Record<string, [number, number]>, data: number[]): Uint8Array {
  const header = new TextEncoder().encode(JSON.stringify({ members }));
  const out = new Uint8Array(4 + header.length + data.length);
  new DataView(out.buffer).setUint32(0, header.length, true);
  out.set(header, 4);
  out.set(data, 4 + header.length);
  return out;
}
const PACK_BYTES = packFile(
  { 'additive_0/amplitudes/c/0': [0, 3], 'additive_1/amplitudes/c/0': [3, 2] },
  [1, 2, 3, 4, 5]
);
const PLAIN: Record<string, number[]> = {
  'splats/part_0/additive_0/amplitudes/c/0': [1, 2, 3],
  'splats/part_0/additive_1/amplitudes/c/0': [4, 5],
  'splats/part_0/additive_1/offdiag/c/0': [7],
  'splats/part_1/additive_0/amplitudes/c/0': [9],
};

function index(overrides: { sha256?: string; hash?: string } = {}) {
  return {
    scene_content_hash: overrides.hash ?? HASH,
    packs: [
      {
        key: 'chunk_packs/0.pack',
        sha256: overrides.sha256 ?? sha256HexPureJs(PACK_BYTES),
        prefix: 'splats/part_0/',
      },
    ],
  };
}

/** An inner source serving the plain chunks and the pack, recording each request. */
function inner(pack: Uint8Array | null = PACK_BYTES) {
  const requests: string[] = [];
  const source: ChunkSource = {
    identity: 'fake://store',
    describe: 'fake://store',
    async get(key: string): Promise<ChunkFetchOutcome> {
      requests.push(key);
      const k = key.replace(/^\/+/, '');
      if (k === 'chunk_packs/0.pack') {
        return pack ? { kind: 'ok', data: pack, bytesOverWire: pack.length } : { kind: 'missing' };
      }
      const plain = PLAIN[k];
      return plain
        ? { kind: 'ok', data: new Uint8Array(plain), bytesOverWire: plain.length }
        : { kind: 'missing' };
    },
    probeIdentityToken: async () => null,
    dispose: () => {},
  };
  return { source, requests };
}

async function bytes(source: ChunkSource, key: string): Promise<number[]> {
  const outcome = await source.get(key);
  if (outcome.kind !== 'ok') throw new Error(`${key}: ${outcome.kind}`);
  return Array.from(outcome.data);
}

describe('PackedChunkSource', () => {
  it('serves every member of a pack from one request, byte-identical to the plain chunks', async () => {
    const { source, requests } = inner();
    const packed = new PackedChunkSource(source);
    expect(packed.usePacks(index(), HASH)).toBe(1);

    expect(await bytes(packed, '/splats/part_0/additive_0/amplitudes/c/0')).toEqual([1, 2, 3]);
    expect(await bytes(packed, 'splats/part_0/additive_1/amplitudes/c/0')).toEqual([4, 5]);
    expect(requests).toEqual(['chunk_packs/0.pack']);
  });

  it('passes a key no pack holds straight through', async () => {
    const { source, requests } = inner();
    const packed = new PackedChunkSource(source);
    packed.usePacks(index(), HASH);

    expect(await bytes(packed, 'splats/part_1/additive_0/amplitudes/c/0')).toEqual([9]);
    expect(requests).toEqual(['splats/part_1/additive_0/amplitudes/c/0']);
  });

  it('ignores an index built for another content_hash (a stale pack)', async () => {
    const { source, requests } = inner();
    const packed = new PackedChunkSource(source);
    expect(packed.usePacks(index({ hash: 'older-hash' }), HASH)).toBe(0);

    expect(await bytes(packed, 'splats/part_0/additive_0/amplitudes/c/0')).toEqual([1, 2, 3]);
    expect(requests).toEqual(['splats/part_0/additive_0/amplitudes/c/0']);
  });

  it('reads plainly when the pack bytes fail their digest, and stops asking for it', async () => {
    const { source, requests } = inner();
    const packed = new PackedChunkSource(source);
    packed.usePacks(index({ sha256: '0'.repeat(64) }), HASH);

    expect(await bytes(packed, 'splats/part_0/additive_0/amplitudes/c/0')).toEqual([1, 2, 3]);
    expect(await bytes(packed, 'splats/part_0/additive_1/amplitudes/c/0')).toEqual([4, 5]);
    expect(requests).toEqual([
      'chunk_packs/0.pack',
      'splats/part_0/additive_0/amplitudes/c/0',
      'splats/part_0/additive_1/amplitudes/c/0',
    ]);
  });

  it('reads a key under the node that the pack does not hold plainly', async () => {
    const { source, requests } = inner();
    const packed = new PackedChunkSource(source);
    packed.usePacks(index(), HASH);

    expect(await bytes(packed, 'splats/part_0/additive_1/offdiag/c/0')).toEqual([7]);
    expect(requests).toEqual(['chunk_packs/0.pack', 'splats/part_0/additive_1/offdiag/c/0']);
  });

  it('reads plainly when the pack is gone', async () => {
    const { source } = inner(null);
    const packed = new PackedChunkSource(source);
    packed.usePacks(index(), HASH);

    expect(await bytes(packed, 'splats/part_0/additive_1/amplitudes/c/0')).toEqual([4, 5]);
  });

  it('without packs (an unpacked store, or none adopted yet) is the inner source', async () => {
    const { source, requests } = inner();
    const packed = new PackedChunkSource(source);
    expect(packed.usePacks(undefined, HASH)).toBe(0);

    expect(await bytes(packed, 'splats/part_0/additive_0/amplitudes/c/0')).toEqual([1, 2, 3]);
    expect(requests).toEqual(['splats/part_0/additive_0/amplitudes/c/0']);
  });

  it('serves a member again after its first read (a cache eviction re-reads it)', async () => {
    const { source, requests } = inner();
    const packed = new PackedChunkSource(source);
    packed.usePacks(index(), HASH);

    await bytes(packed, 'splats/part_0/additive_0/amplitudes/c/0');
    await bytes(packed, 'splats/part_0/additive_1/amplitudes/c/0');
    expect(await bytes(packed, 'splats/part_0/additive_0/amplitudes/c/0')).toEqual([1, 2, 3]);
    // The fully served pack was released; the re-read costs at most one request.
    expect(requests.length).toBeLessThanOrEqual(2);
  });
});

describe('PackedChunkSource under MultiLevelCachingStore', () => {
  beforeEach(() => vi.stubGlobal('navigator', {}));
  afterEach(() => vi.unstubAllGlobals());

  it('caches members under their own chunk keys, exactly as plain reads would', async () => {
    const { source, requests } = inner();
    const packed = new PackedChunkSource(source);
    packed.usePacks(index(), HASH);
    const store = new MultiLevelCachingStore(packed, { noOpfs: true });

    const a = await store.get('/splats/part_0/additive_0/amplitudes/c/0');
    const b = await store.get('/splats/part_0/additive_1/amplitudes/c/0');
    const again = await store.get('/splats/part_0/additive_0/amplitudes/c/0');

    expect(Array.from(a ?? [])).toEqual([1, 2, 3]);
    expect(Array.from(b ?? [])).toEqual([4, 5]);
    expect(Array.from(again ?? [])).toEqual([1, 2, 3]);
    expect(requests).toEqual(['chunk_packs/0.pack']);
  });
});
