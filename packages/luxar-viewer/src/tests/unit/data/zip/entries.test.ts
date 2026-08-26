/**
 * Unit tests for zipped-store URL detection and entry-name normalization.
 *
 * The nested-archive case is the one that matters: `ZipFileStore` looks keys up
 * verbatim, so an archive wrapped in a directory would miss on EVERY key and
 * render an empty scene with no error. These tests pin the re-keying and the
 * refusal to guess between two stores.
 */

import { describe, it, expect } from 'vitest';
import {
  isZippedStoreUrl,
  isZippedZarrStoreUrl,
  normalizeZipEntries,
} from '../../../../data/zip/entries';

const URL_ = 'https://example.com/scene.luxar.zarr.zip';

/** Entry values are opaque to the normalizer; a marker string suffices. */
function entriesFrom(names: string[]): Record<string, string> {
  return Object.fromEntries(names.map((n) => [n, `entry:${n}`]));
}

describe('isZippedStoreUrl', () => {
  it('recognizes a .zip dataset URL', () => {
    expect(isZippedStoreUrl('https://example.com/scene.luxar.zarr.zip')).toBe(true);
    expect(isZippedStoreUrl('/data/scene.luxar.zarr.zip')).toBe(true);
  });

  it('is case-insensitive', () => {
    expect(isZippedStoreUrl('/data/SCENE.LUXAR.ZARR.ZIP')).toBe(true);
  });

  it('ignores query strings and fragments so signed URLs still resolve', () => {
    expect(isZippedStoreUrl('https://example.com/s.luxar.zarr.zip?token=abc&x=1')).toBe(true);
    expect(isZippedStoreUrl('https://example.com/s.luxar.zarr.zip#frag')).toBe(true);
  });

  it('does not match a directory store, nor a .zip appearing mid-path', () => {
    expect(isZippedStoreUrl('https://example.com/scene.luxar.zarr')).toBe(false);
    expect(isZippedStoreUrl('https://example.com/archive.zip/scene.luxar.zarr')).toBe(false);
  });
});

describe('normalizeZipEntries — flat archives', () => {
  it('passes a v3 store through untouched', () => {
    const entries = entriesFrom(['zarr.json', 'points/zarr.json', 'points/c/0']);
    expect(normalizeZipEntries(entries, URL_)).toBe(entries);
  });

  it('recognizes a v2 store by .zgroup', () => {
    const entries = entriesFrom(['.zgroup', '.zattrs', 'points/.zarray']);
    expect(normalizeZipEntries(entries, URL_)).toBe(entries);
  });
});

describe('normalizeZipEntries — nested archives', () => {
  it('strips a single wrapping directory and preserves the entry values', () => {
    const entries = entriesFrom([
      'scene.luxar.zarr/zarr.json',
      'scene.luxar.zarr/points/zarr.json',
      'scene.luxar.zarr/points/c/0',
    ]);

    const out = normalizeZipEntries(entries, URL_);

    expect(Object.keys(out).sort()).toEqual(['points/c/0', 'points/zarr.json', 'zarr.json']);
    // The value must be the ORIGINAL entry — reads go through it.
    expect(out['zarr.json']).toBe('entry:scene.luxar.zarr/zarr.json');
  });

  it('drops sibling junk outside the store directory', () => {
    const entries = entriesFrom(['scene.luxar.zarr/zarr.json', 'README.txt']);
    expect(Object.keys(normalizeZipEntries(entries, URL_))).toEqual(['zarr.json']);
  });
});

describe('normalizeZipEntries — refusals', () => {
  it('refuses an archive holding two stores rather than guessing', () => {
    const entries = entriesFrom(['a.luxar.zarr/zarr.json', 'b.luxar.zarr/zarr.json']);
    expect(() => normalizeZipEntries(entries, URL_)).toThrow(/contains 2 zarr stores/);
  });

  it('refuses an archive with no store at all, naming the URL', () => {
    const entries = entriesFrom(['notes.txt', 'images/a.png']);
    expect(() => normalizeZipEntries(entries, URL_)).toThrow(/does not contain a zarr store/);
    expect(() => normalizeZipEntries(entries, URL_)).toThrow(URL_);
  });

  it('refuses a store buried two directories deep (only one level is unwrapped)', () => {
    const entries = entriesFrom(['out/scene.luxar.zarr/zarr.json']);
    expect(() => normalizeZipEntries(entries, URL_)).toThrow(/does not contain a zarr store/);
  });
});

describe('isZippedZarrStoreUrl — discovery, not routing', () => {
  it('accepts a zipped zarr store', () => {
    expect(isZippedZarrStoreUrl('/data/scene.luxar.zarr.zip')).toBe(true);
    expect(isZippedZarrStoreUrl('/data/SCENE.GSPLATS.ZARR.ZIP?token=x')).toBe(true);
  });

  it('REJECTS an ordinary archive that merely ends in .zip', () => {
    // The point of being narrower than `isZippedStoreUrl`: a directory listing
    // full of `results.zip` must not sprout ZARR badges that die on click.
    expect(isZippedZarrStoreUrl('/data/results.zip')).toBe(false);
    expect(isZippedZarrStoreUrl('/data/backup.tar.zip')).toBe(false);
  });

  it('is still narrower than the routing predicate, which accepts both', () => {
    expect(isZippedStoreUrl('/data/results.zip')).toBe(true);
    expect(isZippedZarrStoreUrl('/data/results.zip')).toBe(false);
  });
});
