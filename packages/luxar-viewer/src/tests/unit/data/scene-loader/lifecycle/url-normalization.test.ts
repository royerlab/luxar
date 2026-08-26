/**
 * Unit tests for the scene-loader URL normalization helper.
 */

import { describe, it, expect } from 'vitest';
import { normalizeURL } from '../../../../../data/scene-loader/lifecycle/url-normalization';

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

describe('normalizeURL — trailing-slash contract (HIGH-6)', () => {
  // The trailing slash on the output is INTENTIONAL and load-bearing:
  // overlay-manager string-concats `${baseUrl}overlays/...` and would
  // 404 without it. Downstream zarr / cache layers tolerate the slash
  // (zarrita's FetchStore re-adds it; the cache layer's buildUrl strips
  // it before joining). Pin the contract with explicit tests so a future
  // refactor that thinks the slash is "unused" hits these assertions.

  it('always produces a URL the zarr loader accepts (no double-slash 404 path)', () => {
    // The cache layer's buildUrl strips trailing `/`s before joining
    // child keys, so an extra trailing slash never produces a double-
    // slash request — and a single trailing slash never produces a
    // missing-separator concat like `.zarroverlays/...`. Verify that
    // every entry-point form (slash, no slash, relative, absolute)
    // converges to the same canonical slash-terminated form.
    const ORIGIN = 'http://localhost:5173';
    const canonical = 'http://example.com/data.zarr/';
    expect(normalizeURL('http://example.com/data.zarr', ORIGIN)).toBe(canonical);
    expect(normalizeURL('http://example.com/data.zarr/', ORIGIN)).toBe(canonical);
    // Idempotent: feeding the function its own output is a no-op.
    expect(normalizeURL(canonical, ORIGIN)).toBe(canonical);
  });

  it('output is ALWAYS slash-terminated (overlay-manager string-concat contract)', () => {
    const ORIGIN = 'http://localhost:5173';
    // overlay-manager builds child URLs via `${baseUrl}overlays/...`,
    // so the contract is: baseUrl ends in `/`. Sample a representative
    // mix of absolute / relative / empty / already-slashed inputs.
    const inputs = [
      'http://example.com/data',
      'http://example.com/data/',
      'https://cdn.example.com/scene.zarr',
      'data/scene',
      '/data/scene',
      '/data/scene/',
      '',
      '/',
    ];
    for (const input of inputs) {
      const out = normalizeURL(input, ORIGIN);
      expect(out.endsWith('/')).toBe(true);
    }
  });
});

describe('normalizeURL — zipped stores (.zarr.zip)', () => {
  // A zipped store is a FILE. The trailing slash exists to make directory
  // children concatenable; appending it here yields `…zip/`, a different
  // resource that 404s.
  it('does NOT append a trailing slash to an absolute archive URL', () => {
    expect(normalizeURL('https://cdn.example.com/scene.luxar.zarr.zip', ORIGIN)).toBe(
      'https://cdn.example.com/scene.luxar.zarr.zip'
    );
  });

  it('absolutizes a relative archive path without adding a slash', () => {
    expect(normalizeURL('/data/scene.luxar.zarr.zip', ORIGIN)).toBe(
      `${ORIGIN}/data/scene.luxar.zarr.zip`
    );
    expect(normalizeURL('data/scene.luxar.zarr.zip', ORIGIN)).toBe(
      `${ORIGIN}/data/scene.luxar.zarr.zip`
    );
  });

  it('still slash-terminates a directory store whose name merely contains .zip', () => {
    expect(normalizeURL('/data/archive.zip.luxar.zarr', ORIGIN)).toBe(
      `${ORIGIN}/data/archive.zip.luxar.zarr/`
    );
  });
});
