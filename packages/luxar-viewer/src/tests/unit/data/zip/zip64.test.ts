/**
 * Zip64: an archive with more than 65,535 members must still be readable.
 *
 * The classic end-of-central-directory record stores the entry count in 16
 * bits, so an archive past 65,535 members can only be described by the zip64
 * EOCD. `unzipit` parses that record, which is why #1716 scoped this as a TEST
 * rather than a feature — but "we believe it works" is not coverage, and a
 * 606 K-object scene packaged into one archive is well past the boundary.
 *
 * THE ARCHIVE IS WRITTEN BY PYTHON, deliberately. The obvious in-process choice
 * is `fflate`, which the sibling wiring test uses and which is already a
 * dependency — but `fflate.zipSync` does not implement zip64: at 70,000 members
 * it emits a CLASSIC EOCD whose count field has wrapped modulo 65,536 to 4,464.
 * A test built on it would validate the reader against a corrupt archive and
 * could only teach us the wrong thing. Python's `zipfile` promotes to zip64 on
 * its own, and is the same writer `luxar.io.optimise._package` packages real
 * `.luxar.zarr.zip` stores with.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import ZipFileStore from '@zarrita/storage/zip';
import { createZipStoreOptions } from '../../../../data/zip/store';

const URL_ = 'https://example.com/huge.luxar.zarr.zip';

/** Just past the 16-bit ceiling — enough to force zip64, small enough to stay quick. */
const FILLER_MEMBERS = 65_600;

/** `zipfile` writes the store members first, then the filler. */
const BUILD_ARCHIVE = `
import sys, zipfile
out, n = sys.argv[1], int(sys.argv[2])
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
    zf.writestr("zarr.json", '{"zarr_format":3,"node_type":"group"}')
    zf.writestr("points/zarr.json", '{"zarr_format":3,"node_type":"array"}')
    zf.writestr("points/c/0/0", "CHUNK-BYTES")
    for i in range(n):
        zf.writestr("filler/%d" % i, "")
`;

/** Zip64 end-of-central-directory signature, little-endian. */
const ZIP64_EOCD_SIGNATURE = 0x06064b50;

let directory: string;
let bytes: Uint8Array;

beforeAll(() => {
  directory = mkdtempSync(join(tmpdir(), 'luxar-zip64-'));
  const archive = join(directory, 'huge.luxar.zarr.zip');
  execFileSync('python3', ['-c', BUILD_ARCHIVE, archive, String(FILLER_MEMBERS)]);
  bytes = new Uint8Array(readFileSync(archive));
}, 120_000);

afterAll(() => {
  if (directory) rmSync(directory, { recursive: true, force: true });
});

const open = () =>
  ZipFileStore.fromBlob(new Blob([bytes as unknown as BlobPart]), createZipStoreOptions(URL_));

describe('zip64 archives (> 65,535 members)', () => {
  it('IS actually a zip64 archive, so the reads below test what they claim', () => {
    // Without this the suite degrades silently: drop FILLER_MEMBERS under the
    // ceiling, or swap in a writer that cannot promote, and every assertion
    // below still passes while exercising the classic path.
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let found = false;
    for (let i = bytes.length - 56; i >= 0 && !found; i--) {
      if (view.getUint32(i, true) === ZIP64_EOCD_SIGNATURE) found = true;
    }
    expect(found, 'archive has no zip64 EOCD record').toBe(true);
  });

  it('reads a member from an archive whose central directory needs zip64', async () => {
    const store = open();
    const chunk = await store.get('/points/c/0/0');
    expect(chunk).toBeDefined();
    expect(new TextDecoder().decode(chunk)).toBe('CHUNK-BYTES');
  });

  it('still resolves the root document, which is what the loader opens first', async () => {
    const store = open();
    const root = await store.get('/zarr.json');
    expect(new TextDecoder().decode(root)).toContain('"zarr_format":3');
  });

  it('reports a genuinely absent member as undefined, not as a zip64 parse failure', async () => {
    const store = open();
    expect(await store.get('/points/c/9/9')).toBeUndefined();
  });
});
