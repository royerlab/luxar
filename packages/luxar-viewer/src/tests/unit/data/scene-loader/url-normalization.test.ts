/**
 * Unit tests for the scene-loader URL normalization helper.
 */

import { describe, it, expect } from 'vitest';
import { normalizeURL } from '../../../../data/scene-loader/url-normalization';

const ORIGIN = 'http://localhost:5173';

describe('normalizeURL — absolute URLs', () => {
  it('returns http URLs unchanged when slash-terminated', () => {
    expect(normalizeURL('http://example.com/data/', ORIGIN)).toBe('http://example.com/data/');
  });

  it('appends a trailing slash to http URLs that lack one', () => {
    expect(normalizeURL('http://example.com/data', ORIGIN)).toBe('http://example.com/data/');
  });

  it('handles https URLs the same as http', () => {
    expect(normalizeURL('https://cdn.example.com/scene', ORIGIN)).toBe(
      'https://cdn.example.com/scene/'
    );
    expect(normalizeURL('https://cdn.example.com/scene/', ORIGIN)).toBe(
      'https://cdn.example.com/scene/'
    );
  });

  it('does NOT touch the path when an absolute URL ends with .zarr (no slash heuristic)', () => {
    // Absolute branch only checks trailing slash — doesn't strip extensions.
    expect(normalizeURL('https://cdn/example.zarr', ORIGIN)).toBe('https://cdn/example.zarr/');
  });
});

describe('normalizeURL — relative URLs', () => {
  it('prepends origin and adds leading + trailing slash', () => {
    expect(normalizeURL('data/scene', ORIGIN)).toBe('http://localhost:5173/data/scene/');
  });

  it('preserves an existing leading slash', () => {
    expect(normalizeURL('/data/scene', ORIGIN)).toBe('http://localhost:5173/data/scene/');
  });

  it('preserves an existing trailing slash on a relative path', () => {
    expect(normalizeURL('/data/scene/', ORIGIN)).toBe('http://localhost:5173/data/scene/');
  });

  it('handles an empty relative path by serving from the origin root', () => {
    expect(normalizeURL('', ORIGIN)).toBe('http://localhost:5173/');
  });

  it('handles a single slash by serving from the origin root', () => {
    expect(normalizeURL('/', ORIGIN)).toBe('http://localhost:5173/');
  });

  it('respects the supplied origin verbatim (no implicit window dependency)', () => {
    expect(normalizeURL('foo', 'https://other.example.com:8443')).toBe(
      'https://other.example.com:8443/foo/'
    );
  });
});
