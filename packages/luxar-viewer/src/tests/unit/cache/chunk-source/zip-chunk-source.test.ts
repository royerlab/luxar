/**
 * Tests for the zipped-store byte source.
 *
 * Two things matter here and are not covered by the store's own tests: the
 * outcome mapping the caching store switches on, and the identity probe — which
 * is the ONLY way a zipped store can answer "is this still the scene I cached?",
 * because its root attrs live inside the archive and the reader's directory is
 * a snapshot.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import type { ArchiveByteReader } from '../../../../cache/chunk-source';
import { ArchiveFaultError } from '../../../../cache/chunk-source';
import { ZipChunkSource } from '../../../../cache/chunk-source/zip-chunk-source';
import { RangeUnsupportedError } from '../../../../data/zip/range-reader';

const ARCHIVE_URL = 'https://example.com/scene.luxar.zarr.zip';

/** The archive is injected, so reads need no network at all. */
function fakeReader(
  get: (key: string) => Promise<Uint8Array | undefined>
): ArchiveByteReader & { disposed: boolean } {
  return {
    get,
    disposed: false,
    dispose() {
      this.disposed = true;
    },
  };
}

const emptyReader = () => fakeReader(async () => undefined);

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('ZipChunkSource — identity', () => {
  it('keeps the archive URL verbatim, distinct from its directory twin', () => {
    // The OPFS bucket is derived from this. Folding the two together would let
    // an archive serve a directory's cached members, and vice versa, with
    // nothing failing loudly — they hold the same bytes under different keys.
    const source = new ZipChunkSource(ARCHIVE_URL, emptyReader());
    expect(source.identity).toBe(ARCHIVE_URL);
    expect(source.identity).not.toBe('https://example.com/scene.luxar.zarr/');
  });
});

describe('ZipChunkSource — identity probe', () => {
  // The HEAD itself lives in the reader (so the one request also seeds the
  // archive length) — see range-reader.test.ts. What matters here is that the
  // source forwards it and labels the result.
  it('wraps the container token as an archive-etag verdict', async () => {
    const reader = fakeReader(async () => undefined);
    reader.probeIdentity = async () => 'etag:"abc123"';

    const token = await new ZipChunkSource(ARCHIVE_URL, reader).probeIdentityToken({});

    expect(token).toEqual({ hash: 'etag:"abc123"', mode: 'archive-etag' });
  });

  it('reports null — never a change verdict — when the container cannot tell', async () => {
    const reader = fakeReader(async () => undefined);
    reader.probeIdentity = async () => null;

    expect(await new ZipChunkSource(ARCHIVE_URL, reader).probeIdentityToken({})).toBeNull();
  });

  it('forwards the abort signal to the container', async () => {
    const seen: (AbortSignal | undefined)[] = [];
    const reader = fakeReader(async () => undefined);
    reader.probeIdentity = async (signal) => {
      seen.push(signal);
      return null;
    };
    const controller = new AbortController();

    await new ZipChunkSource(ARCHIVE_URL, reader).probeIdentityToken({
      signal: controller.signal,
    });

    expect(seen[0]).toBe(controller.signal);
  });
});

describe('ZipChunkSource — outcomes', () => {
  it('returns bytes for a present key', async () => {
    const reader = fakeReader(async () => new Uint8Array([1, 2, 3]));
    const outcome = await new ZipChunkSource(ARCHIVE_URL, reader).get('/points/c/0');

    expect(outcome.kind).toBe('ok');
    if (outcome.kind !== 'ok') return;
    expect(Array.from(outcome.data)).toEqual([1, 2, 3]);
  });

  it('maps an absent member to `missing`', async () => {
    const outcome = await new ZipChunkSource(ARCHIVE_URL, emptyReader()).get('/nope/c/0');
    expect(outcome.kind).toBe('missing');
  });

  it('reports `aborted` without touching the archive', async () => {
    const reader = fakeReader(async () => {
      throw new Error('should not be read');
    });
    const controller = new AbortController();
    controller.abort();

    const outcome = await new ZipChunkSource(ARCHIVE_URL, reader).get(
      '/zarr.json',
      controller.signal
    );

    expect(outcome.kind).toBe('aborted');
  });

  it('disposes the injected archive', () => {
    const reader = emptyReader();
    new ZipChunkSource(ARCHIVE_URL, reader).dispose();
    expect(reader.disposed).toBe(true);
  });

  it('maps an ARCHIVE FAULT to `fatal`, so the store rethrows it', async () => {
    // This PR's headline fix, and it was untested: disabling the branch left the
    // whole zip + cache corpus green. The reader tests pin which errors are
    // thrown and the store tests pin what it does with `fatal`; nothing joined
    // the two, which is where the behaviour actually lives.
    const reader = fakeReader(async () => {
      throw new RangeUnsupportedError('https://example.com/s.zip', 'no 206');
    });

    const outcome = await new ZipChunkSource(ARCHIVE_URL, reader).get('/zarr.json');

    expect(outcome.kind).toBe('fatal');
    if (outcome.kind !== 'fatal') return;
    expect(outcome.cause).toBeInstanceOf(ArchiveFaultError);
  });

  it('keeps a plain member failure as `error`, not `fatal`', async () => {
    // One flaky chunk must stay recoverable: `fatal` fails the whole load and
    // attaches remedy text about Range support, which is the wrong advice for a
    // transient 5xx.
    const reader = fakeReader(async () => {
      throw new Error('Cannot read the zipped store at …: exhausted its retries');
    });

    const outcome = await new ZipChunkSource(ARCHIVE_URL, reader).get('/points/c/0');

    expect(outcome.kind).toBe('error');
  });

  it('surfaces a broken archive as `error`, not as a throw', async () => {
    // A source that throws would escape the caching store's Result contract.
    const reader = fakeReader(async () => {
      throw new TypeError('Failed to fetch');
    });

    const outcome = await new ZipChunkSource(ARCHIVE_URL, reader).get('/zarr.json');
    expect(outcome.kind).toBe('error');
  });
});
