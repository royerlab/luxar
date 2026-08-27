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

function writeUint64(view: DataView, offset: number, value: number): void {
  view.setUint32(offset, value >>> 0, true);
  view.setUint32(offset + 4, Math.floor(value / 0x100000000), true);
}

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/**
 * Build a ZIP64 archive whose entry count crosses the classic 65,535 limit.
 * fflate.zipSync does not emit ZIP64 records, so using it here would wrap the
 * entry count and test the reader against a corrupt archive.
 */
function zip64Archive(entryCount = 65_536): Uint8Array {
  const encoder = new TextEncoder();
  const payload = encoder.encode('{"zarr_format":3,"node_type":"group"}');
  const payloadCrc = crc32(payload);
  const names = Array.from({ length: entryCount }, (_, index) =>
    index === entryCount - 1 ? 'zarr.json' : `e${index}`
  );
  const localSize = names.reduce(
    (total, name, index) =>
      total + 30 + name.length + (index === entryCount - 1 ? payload.length : 0),
    0
  );
  const centralSize = names.reduce((total, name) => total + 46 + name.length, 0);
  const zip64EndSize = 56;
  const zip64LocatorSize = 20;
  const classicEndSize = 22;
  const bytes = new Uint8Array(
    localSize + centralSize + zip64EndSize + zip64LocatorSize + classicEndSize
  );
  const view = new DataView(bytes.buffer);
  const offsets: number[] = [];
  let cursor = 0;

  names.forEach((name, index) => {
    const encoded = encoder.encode(name);
    const isPayload = index === entryCount - 1;
    offsets.push(cursor);
    view.setUint32(cursor, 0x04034b50, true);
    view.setUint16(cursor + 4, 20, true);
    if (isPayload) {
      view.setUint32(cursor + 14, payloadCrc, true);
      view.setUint32(cursor + 18, payload.length, true);
      view.setUint32(cursor + 22, payload.length, true);
    }
    view.setUint16(cursor + 26, encoded.length, true);
    bytes.set(encoded, cursor + 30);
    if (isPayload) bytes.set(payload, cursor + 30 + encoded.length);
    cursor += 30 + encoded.length + (isPayload ? payload.length : 0);
  });

  const centralOffset = cursor;
  names.forEach((name, index) => {
    const encoded = encoder.encode(name);
    view.setUint32(cursor, 0x02014b50, true);
    view.setUint16(cursor + 4, 20, true);
    view.setUint16(cursor + 6, 20, true);
    if (index === entryCount - 1) {
      view.setUint32(cursor + 16, payloadCrc, true);
      view.setUint32(cursor + 20, payload.length, true);
      view.setUint32(cursor + 24, payload.length, true);
    }
    view.setUint16(cursor + 28, encoded.length, true);
    view.setUint32(cursor + 42, offsets[index], true);
    bytes.set(encoded, cursor + 46);
    cursor += 46 + encoded.length;
  });

  const zip64EndOffset = cursor;
  view.setUint32(cursor, 0x06064b50, true);
  writeUint64(view, cursor + 4, 44);
  view.setUint16(cursor + 12, 45, true);
  view.setUint16(cursor + 14, 45, true);
  writeUint64(view, cursor + 24, entryCount);
  writeUint64(view, cursor + 32, entryCount);
  writeUint64(view, cursor + 40, centralSize);
  writeUint64(view, cursor + 48, centralOffset);
  cursor += zip64EndSize;

  view.setUint32(cursor, 0x07064b50, true);
  writeUint64(view, cursor + 8, zip64EndOffset);
  view.setUint32(cursor + 16, 1, true);
  cursor += zip64LocatorSize;

  view.setUint32(cursor, 0x06054b50, true);
  view.setUint16(cursor + 8, 0xffff, true);
  view.setUint16(cursor + 10, 0xffff, true);
  view.setUint32(cursor + 12, 0xffffffff, true);
  view.setUint32(cursor + 16, 0xffffffff, true);
  return bytes;
}

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

  it('reads a ZIP64 archive above the 65,535-entry boundary', async () => {
    const bytes = zip64Archive();
    const blob = new Blob([bytes as unknown as BlobPart]);
    const store = ZipFileStore.fromBlob(blob, createZipStoreOptions(URL_));

    expect(decode(await store.get('/zarr.json'))).toBe('{"zarr_format":3,"node_type":"group"}');
  }, 20_000);
});

describe('createZipStore wiring — refusals surface, not swallowed', () => {
  it('rejects reads from an archive holding no store', async () => {
    const store = open({ 'notes.txt': 'hello' });
    await expect(store.get('/zarr.json')).rejects.toThrow(/does not contain a zarr store/);
  });
});
