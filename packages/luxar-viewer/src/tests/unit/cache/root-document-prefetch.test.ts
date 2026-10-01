/**
 * `SharedRootDocumentSource`: the byte source that answers the root document
 * from the shared load-time fetch and delegates everything else untouched.
 */

import { describe, it, expect, vi } from 'vitest';
import type {
  ChunkFetchOutcome,
  ChunkSource,
  ChunkSourceGetOptions,
} from '../../../cache/chunk-source';
import {
  SharedRootDocumentSource,
  type RootDocName,
  type RootDocOutcome,
  type RootDocumentFetch,
} from '../../../cache/root-document-prefetch';

function innerSource() {
  const calls: Array<{ key: string; signal?: AbortSignal; options?: ChunkSourceGetOptions }> = [];
  const source: ChunkSource = {
    identity: 'https://cdn.example/scene.zarr/',
    describe: 'https://cdn.example/scene.zarr/',
    get: vi.fn(async (key: string, signal?: AbortSignal, options?: ChunkSourceGetOptions) => {
      calls.push({ key, signal, options });
      return { kind: 'ok', data: new Uint8Array([9]), bytesOverWire: 1 } as ChunkFetchOutcome;
    }),
    probeIdentityToken: vi.fn(async () => ({ hash: 'inner', mode: 'content-hash' as const })),
    dispose: vi.fn(),
  };
  return { source, calls };
}

function sharedFetch(outcomes: Array<[RootDocName, RootDocOutcome]>): Promise<RootDocumentFetch> {
  const map = new Map(outcomes);
  const ok = outcomes.find(([, o]) => o.kind === 'ok');
  const served =
    ok && ok[1].kind === 'ok' ? { doc: ok[0], bytes: ok[1].bytes, etag: ok[1].etag } : null;
  return Promise.resolve({
    outcomes: map,
    served,
    token: async () => ({ hash: 'shared', mode: 'content-hash' as const }),
    peekToken: () => undefined,
  });
}

const ROOT_BYTES = new TextEncoder().encode('{"zarr_format":3}');

describe('SharedRootDocumentSource', () => {
  it('forwards the fetch-priority options to the inner source for every non-root key', async () => {
    const { source, calls } = innerSource();
    const shared = new SharedRootDocumentSource(
      source,
      sharedFetch([['zarr.json', { kind: 'ok', bytes: ROOT_BYTES, etag: null }]])
    );
    const signal = new AbortController().signal;
    const options = { priority: { value: 'speculative' } } as unknown as ChunkSourceGetOptions;
    await shared.get('/points/c/0', signal, options);
    expect(calls).toEqual([{ key: '/points/c/0', signal, options }]);
  });

  it('answers the root document ONCE from the shared fetch, then reads through', async () => {
    const { source, calls } = innerSource();
    const shared = new SharedRootDocumentSource(
      source,
      sharedFetch([['zarr.json', { kind: 'ok', bytes: ROOT_BYTES, etag: '"e"' }]])
    );
    const first = await shared.get('/zarr.json');
    expect(first).toMatchObject({ kind: 'ok', bytesOverWire: ROOT_BYTES.byteLength });
    // A private copy: the store may hand the bytes on.
    expect((first as { data: Uint8Array }).data).not.toBe(ROOT_BYTES);
    expect(calls).toEqual([]);
    await shared.get('/zarr.json');
    expect(calls.map((c) => c.key)).toEqual(['/zarr.json']);
  });

  it('serves a definitive miss, but re-reads a document the shared fetch could not answer', async () => {
    const { source, calls } = innerSource();
    const shared = new SharedRootDocumentSource(
      source,
      sharedFetch([
        ['zarr.json', { kind: 'missing' }],
        ['.zattrs', { kind: 'error' }],
      ])
    );
    expect(await shared.get('zarr.json')).toEqual({ kind: 'missing' });
    await shared.get('/.zattrs');
    expect(calls.map((c) => c.key)).toEqual(['/.zattrs']);
  });

  it('answers validation from the shared fetch until released, then from the inner source', async () => {
    const { source } = innerSource();
    const shared = new SharedRootDocumentSource(
      source,
      sharedFetch([['zarr.json', { kind: 'ok', bytes: ROOT_BYTES, etag: null }]])
    );
    expect(await shared.probeIdentityToken({})).toEqual({ hash: 'shared', mode: 'content-hash' });
    shared.release();
    expect(await shared.probeIdentityToken({})).toEqual({ hash: 'inner', mode: 'content-hash' });
    expect(shared.identity).toBe(source.identity);
  });
});
