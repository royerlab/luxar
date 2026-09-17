/**
 * Tests for `OPFSStore.compactOrderCounter` — private path that fires
 * when `orderCounter > 1e12` and renumbers existing entries 0..N-1
 * preserving LRU order.
 *
 * cache.md O4 / Phase E60: extracted from `cache/opfs-store.test.ts`
 * where the block was appended at the bottom of a 1253-line file and
 * used a separate `createMockDirHandle` helper that didn't overlap
 * with the file-level mock. Moving it to a dedicated file under
 * `opfs-store/` makes the layout match the source structure (the
 * sibling `opfs-store/opfs-timeout.test.ts` and `opfs-store/buckets.
 * test.ts` already follow the same pattern).
 *
 * Background — cache.md G7 [P5]: `compactOrderCounter` fires when
 * `orderCounter > 1e12` (source line 385 / 583). The check fires on a
 * long session and silently renumbers entries 0..N-1; a regression
 * that compacted in the WRONG order would break LRU eviction across
 * sessions. Drive the path directly via the same-named private method
 * through `as unknown as { ... }`.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { OPFSStore } from '../../../../cache/multi-level-caching-store/opfs-store';
import { createFakeOpfsRoot } from '../../../mocks/opfs.mock';

describe('OPFSStore.compactOrderCounter', () => {
  let mockFS: { files: Map<string, ArrayBuffer>; metaFiles: Map<string, string> };
  let store: OPFSStore;

  beforeEach(async () => {
    mockFS = { files: new Map(), metaFiles: new Map() };
    // Origin root → `luxar/` → this file's dataset dir (see opfs.mock.ts).
    createFakeOpfsRoot({
      datasetDir: createMockDirHandle(mockFS),
      estimate: async () => ({ quota: 10 * 1024 * 1024 * 1024, usage: 1 * 1024 * 1024 * 1024 }),
    }).install();
    vi.stubGlobal('crypto', {
      subtle: {
        async digest() {
          return new Uint8Array(32).fill(0xab).buffer;
        },
      },
    });
    store = new OPFSStore('compact-test', 'https://example.com/data.zarr', 100 * 1024 * 1024);
    await store.init();
  });

  it('renumbers existing entries 0..N-1 in original LRU order', async () => {
    await store.set('a', new Uint8Array(10));
    await store.set('b', new Uint8Array(10));
    await store.set('c', new Uint8Array(10));

    const internals = store as unknown as {
      index: Map<string, { size: number; order: number }>;
      orderCounter: number;
      compactOrderCounter: () => void;
    };

    // Pre-compact: orders are 1, 2, 3 (sequential set order; first set bumps
    // the counter from 0 to 1).
    const aOrderBefore = internals.index.get('a')!.order;
    const bOrderBefore = internals.index.get('b')!.order;
    const cOrderBefore = internals.index.get('c')!.order;
    expect(aOrderBefore).toBeLessThan(bOrderBefore);
    expect(bOrderBefore).toBeLessThan(cOrderBefore);

    // Force a compaction; afterward, orders MUST be 0, 1, 2 in the same
    // relative ordering (a < b < c) and orderCounter === 3.
    internals.compactOrderCounter();

    expect(internals.index.get('a')!.order).toBe(0);
    expect(internals.index.get('b')!.order).toBe(1);
    expect(internals.index.get('c')!.order).toBe(2);
    expect(internals.orderCounter).toBe(3);

    // Sizes are unchanged.
    expect(internals.index.get('a')!.size).toBe(10);
    expect(internals.index.get('b')!.size).toBe(10);
    expect(internals.index.get('c')!.size).toBe(10);
  });

  it('preserves LRU order when prior orders are non-contiguous (gaps from deletions)', async () => {
    await store.set('a', new Uint8Array(10));
    await store.set('b', new Uint8Array(10));
    await store.set('c', new Uint8Array(10));
    await store.set('d', new Uint8Array(10));

    // Delete b to create a gap in the order sequence.
    await store.delete('b');

    const internals = store as unknown as {
      index: Map<string, { size: number; order: number }>;
      orderCounter: number;
      compactOrderCounter: () => void;
    };

    const aOrder = internals.index.get('a')!.order;
    const cOrder = internals.index.get('c')!.order;
    const dOrder = internals.index.get('d')!.order;
    expect(aOrder).toBeLessThan(cOrder);
    expect(cOrder).toBeLessThan(dOrder);

    internals.compactOrderCounter();

    // After compaction: a → 0, c → 1, d → 2.
    expect(internals.index.get('a')!.order).toBe(0);
    expect(internals.index.get('c')!.order).toBe(1);
    expect(internals.index.get('d')!.order).toBe(2);
    expect(internals.orderCounter).toBe(3);
  });
});

// Local mock directory handle — re-implements just the surface the
// compactOrderCounter test needs (set/delete cycle through getFileHandle
// + getDirectoryHandle + removeEntry). Kept local to this file so the
// test runs independently.
function createMockDirHandle(mockFS: {
  files: Map<string, ArrayBuffer>;
  metaFiles: Map<string, string>;
}) {
  const handle: any = {
    async getFileHandle(name: string) {
      const isMeta = name === '_cache_meta.json';
      return {
        async getFile() {
          if (isMeta) {
            const text = mockFS.metaFiles.get(name) ?? '';
            return new Blob([text]);
          }
          const buf = mockFS.files.get(name);
          if (!buf) throw new Error('Not found');
          return {
            size: buf.byteLength,
            async arrayBuffer() {
              return buf;
            },
          };
        },
        async createWritable() {
          return {
            async write(data: ArrayBuffer | string) {
              if (isMeta) {
                mockFS.metaFiles.set(name, typeof data === 'string' ? data : '');
              } else {
                mockFS.files.set(
                  name,
                  data instanceof ArrayBuffer ? data : new Uint8Array(data as any).buffer
                );
              }
            },
            async close() {},
          };
        },
      };
    },
    async getDirectoryHandle(_name: string) {
      return handle;
    },
    async removeEntry(name: string) {
      mockFS.files.delete(name);
      mockFS.metaFiles.delete(name);
    },
    async *entries() {
      // No iteration needed for the compactOrderCounter test.
    },
    async *keys() {
      // No iteration needed for the compactOrderCounter test.
    },
  };
  return handle;
}
