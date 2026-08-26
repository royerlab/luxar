/**
 * Wiring test for the zipped store: a REAL archive, read through the same
 * options object `createZipStore` builds.
 *
 * The two helpers are unit-tested separately, but the place they meet is where
 * this can break silently: `normalizeZipEntries` produces slash-LESS keys
 * (`zarr.json`), while zarrita addresses the store with leading-slash keys
 * (`/zarr.json`) and `@zarrita/storage`'s `stripPrefix` is literally
 * `path.slice(1)`. A wrong option name, or a re-key that kept its prefix,
 * would leave every lookup missing — and the helper tests would still pass.
 *
 * Uses `fromBlob` rather than `fromUrl` so the archive is real but no server is
 * involved; the reader is the only piece swapped out.
 */

import { describe, it, expect } from 'vitest';
import { zipSync, strToU8 } from 'fflate';
import ZipFileStore from '@zarrita/storage/zip';
import { createZipStoreOptions } from '../../../../data/zip/store';

const URL_ = 'https://example.com/scene.luxar.zarr.zip';

/** Build an in-memory zip with the same options `createZipStore` passes. */
function open(files: Record<string, string>) {
  const bytes = zipSync(
    Object.fromEntries(Object.entries(files).map(([name, body]) => [name, strToU8(body)]))
  );
  const blob = new Blob([bytes as unknown as BlobPart]);
  return ZipFileStore.fromBlob(blob, createZipStoreOptions(URL_));
}

const decode = (bytes: Uint8Array | undefined) =>
  bytes ? new TextDecoder().decode(bytes) : undefined;

describe('createZipStore wiring — flat archive', () => {
  it('resolves leading-slash zarr keys against slash-less entry names', async () => {
    const store = open({
      'zarr.json': '{"zarr_format":3,"node_type":"group"}',
      'points/zarr.json': '{"zarr_format":3,"node_type":"array"}',
      'points/c/0/0': 'CHUNK-BYTES',
    });

    expect(decode(await store.get('/zarr.json'))).toContain('"zarr_format":3');
    expect(decode(await store.get('/points/c/0/0'))).toBe('CHUNK-BYTES');
  });

  it('returns undefined for a missing key — the same shape FetchStore gives a 404', async () => {
    const store = open({ 'zarr.json': '{}' });
    expect(await store.get('/nope/c/0')).toBeUndefined();
  });
});

describe('createZipStore wiring — refusals surface, not swallowed', () => {
  it('rejects reads from an archive holding no store', async () => {
    const store = open({ 'notes.txt': 'hello' });
    await expect(store.get('/zarr.json')).rejects.toThrow(/does not contain a zarr store/);
  });
});
