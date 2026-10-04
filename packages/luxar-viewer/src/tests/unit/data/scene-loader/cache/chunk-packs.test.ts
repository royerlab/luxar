/**
 * `adoptChunkPacks`: the packed store's index reaches the source, and an
 * unpacked store pays nothing for the feature existing.
 */

import { describe, it, expect } from 'vitest';
import * as zarr from '../../../../../data/zarr';
import { adoptChunkPacks } from '../../../../../data/scene-loader/cache/chunk-packs';
import { PackedChunkSource } from '../../../../../cache/chunk-source/packed-chunk-source';
import type { ChunkSource } from '../../../../../cache/chunk-source';

const HASH = 'h';
const ATTRS = {
  scene_content_hash: HASH,
  packs: [{ key: 'chunk_packs/0.pack', sha256: 'x', prefix: 'n/' }],
};

/** A consolidated-looking store: a `contents()` listing, and documents by key. */
function store(
  listing: Array<{ path: string; kind: 'array' | 'group' }>,
  attrs: Record<string, unknown> = ATTRS
) {
  const reads: string[] = [];
  const docs: Record<string, unknown> = {
    '/chunk_packs/zarr.json': { zarr_format: 3, node_type: 'group', attributes: attrs },
  };
  return {
    reads,
    contents: () => listing,
    async get(key: string): Promise<Uint8Array | undefined> {
      reads.push(key);
      const doc = docs[key];
      return doc ? new TextEncoder().encode(JSON.stringify(doc)) : undefined;
    },
  };
}

const packs = () => new PackedChunkSource({} as ChunkSource);

describe('adoptChunkPacks', () => {
  it('reads nothing when the index lists no chunk_packs group', async () => {
    const s = store([{ path: '/', kind: 'group' }]);
    const root = zarr.root(s as unknown as zarr.Readable);
    const used = await adoptChunkPacks(packs(), root, HASH, true, root);
    expect(used).toBe(0);
    expect(s.reads).toEqual([]);
  });

  it('adopts the listed sidecar for the root content_hash', async () => {
    const s = store([{ path: '/chunk_packs', kind: 'group' }]);
    const root = zarr.root(s as unknown as zarr.Readable);
    const used = await adoptChunkPacks(packs(), root, HASH, true, root);
    expect(used).toBe(1);
  });

  it('adopts nothing for another content_hash (the packs are stale)', async () => {
    const s = store([{ path: '/chunk_packs', kind: 'group' }]);
    const root = zarr.root(s as unknown as zarr.Readable);
    const used = await adoptChunkPacks(packs(), root, 'other', true, root);
    expect(used).toBe(0);
  });

  it('finds packs added after the root index was cached', async () => {
    const cached = store([{ path: '/', kind: 'group' }]);
    const network = store([{ path: '/chunk_packs', kind: 'group' }]);
    const used = await adoptChunkPacks(
      packs(),
      zarr.root(cached as unknown as zarr.Readable),
      HASH,
      false,
      zarr.root(network as unknown as zarr.Readable)
    );
    expect(used).toBe(1);
    expect(cached.reads).toEqual([]);
    expect(network.reads).toContain('/chunk_packs/zarr.json');
  });

  it('re-reads a cached pack index before accepting its binding', async () => {
    const listing = [{ path: '/chunk_packs', kind: 'group' as const }];
    const cached = store(listing, { ...ATTRS, scene_content_hash: 'old' });
    const network = store(listing);
    const used = await adoptChunkPacks(
      packs(),
      zarr.root(cached as unknown as zarr.Readable),
      HASH,
      false,
      zarr.root(network as unknown as zarr.Readable)
    );
    expect(used).toBe(1);
    expect(cached.reads).toEqual([]);
    expect(network.reads).toContain('/chunk_packs/zarr.json');
  });
});
