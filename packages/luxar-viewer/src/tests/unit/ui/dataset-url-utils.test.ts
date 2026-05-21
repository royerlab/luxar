/**
 * Unit tests for the pure URL helpers used by the dataset browser.
 */

import { describe, it, expect } from 'vitest';
import { extractBaseUrl, extractPath } from '../../../ui/dataset-browser/url-utils';

const ORIGIN = 'http://example.test:8000';

describe('extractBaseUrl', () => {
  it('returns ${origin}/ when the URL is empty', () => {
    expect(extractBaseUrl('', ORIGIN)).toBe(`${ORIGIN}/`);
  });

  it('strips a trailing .zarr segment so the listing target is the parent dir', () => {
    expect(extractBaseUrl('http://x.test/data/foo.zarr', ORIGIN)).toBe('http://x.test/data/');
    expect(extractBaseUrl('http://x.test/data/foo.zarr/', ORIGIN)).toBe('http://x.test/data/');
  });

  it('returns the parent dir for a deeply nested .zarr path', () => {
    expect(extractBaseUrl('http://x.test/a/b/c/foo.zarr', ORIGIN)).toBe('http://x.test/a/b/c/');
  });

  it('keeps the URL unchanged when the path is already a directory listing', () => {
    expect(extractBaseUrl('http://x.test/data/', ORIGIN)).toBe('http://x.test/data/');
  });

  it('preserves origin and path together for a non-zarr path', () => {
    expect(extractBaseUrl('http://x.test/some/file.json', ORIGIN)).toBe(
      'http://x.test/some/file.json'
    );
  });

  it('returns the original input verbatim on parse failure', () => {
    expect(extractBaseUrl('not a url', ORIGIN)).toBe('not a url');
  });

  it('handles a top-level .zarr by returning the bare origin', () => {
    // Stripping the dataset segment leaves an empty path; the result is
    // origin + '/' (one slash), not a '//' double-slash.
    expect(extractBaseUrl('http://x.test/foo.zarr', ORIGIN)).toBe('http://x.test/');
  });
});

describe('extractPath', () => {
  it('returns empty string for empty input', () => {
    expect(extractPath('')).toBe('');
  });

  it('returns the path up to and including the .zarr segment', () => {
    expect(extractPath('http://x.test/data/foo.zarr')).toBe('data/foo.zarr');
    expect(extractPath('http://x.test/a/b/foo.zarr')).toBe('a/b/foo.zarr');
  });

  it('truncates path components after the .zarr (zarr-internal paths)', () => {
    expect(extractPath('http://x.test/data/foo.zarr/groupA/dataset')).toBe('data/foo.zarr');
  });

  it('returns empty string when the URL does not contain a .zarr segment', () => {
    expect(extractPath('http://x.test/data/file.json')).toBe('');
  });

  it('handles a top-level .zarr with no parent directory', () => {
    expect(extractPath('http://x.test/foo.zarr')).toBe('foo.zarr');
  });

  it('returns empty string on parse failure', () => {
    expect(extractPath('not a url')).toBe('');
  });
});
