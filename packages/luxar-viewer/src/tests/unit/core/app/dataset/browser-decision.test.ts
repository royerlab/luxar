/**
 * Unit tests for the URL classification used by LuxarApp.shouldShowBrowser.
 */

import { describe, it, expect } from 'vitest';
import { classifyBrowserUrl } from '../../../core/app/dataset/browser-decision';

describe('classifyBrowserUrl', () => {
  describe('must-browse cases', () => {
    it('classifies empty string as must-browse', () => {
      expect(classifyBrowserUrl('')).toBe('must-browse');
    });

    it('classifies undefined as must-browse', () => {
      expect(classifyBrowserUrl(undefined)).toBe('must-browse');
    });

    it('classifies null as must-browse', () => {
      expect(classifyBrowserUrl(null)).toBe('must-browse');
    });

    it('classifies whitespace-only as must-browse', () => {
      expect(classifyBrowserUrl('   ')).toBe('must-browse');
      expect(classifyBrowserUrl('\t\n')).toBe('must-browse');
    });

    it('classifies a directory URL (trailing slash) as must-browse', () => {
      expect(classifyBrowserUrl('http://example.com/')).toBe('must-browse');
      expect(classifyBrowserUrl('http://example.com/data/')).toBe('must-browse');
      expect(classifyBrowserUrl('https://server.test/path/sub/')).toBe('must-browse');
    });

    it('a single trailing slash is enough — even a bare hostname-with-slash counts', () => {
      expect(classifyBrowserUrl('http://localhost/')).toBe('must-browse');
    });
  });

  describe('maybe-zarr cases', () => {
    it('a non-empty URL without trailing slash is maybe-zarr (probe required)', () => {
      expect(classifyBrowserUrl('http://example.com/data.zarr')).toBe('maybe-zarr');
    });

    it('a deeply-nested .zarr URL is maybe-zarr', () => {
      expect(classifyBrowserUrl('https://server.test/a/b/c/d.zarr')).toBe('maybe-zarr');
    });

    it('non-zarr URLs without trailing slash are still maybe-zarr — the probe decides', () => {
      // The classifier doesn't know which paths look zarr; that's the
      // probe's job. We just rule out the obvious must-browse cases.
      expect(classifyBrowserUrl('http://x.test/file.json')).toBe('maybe-zarr');
      expect(classifyBrowserUrl('http://x.test/some-name')).toBe('maybe-zarr');
    });

    it('a relative path is maybe-zarr', () => {
      expect(classifyBrowserUrl('data.zarr')).toBe('maybe-zarr');
    });
  });
});
