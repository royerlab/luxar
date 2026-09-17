// @vitest-environment jsdom
/**
 * Tests for DirectoryNavigator - server-agnostic directory navigation
 *
 * Focuses on essential functionality without over-engineering:
 * - Basic navigation strategies
 * - Zarr detection
 * - Error handling
 * - Path operations
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
// [architecture.md/O1][P10] Moved from tests/unit/architecture/ - this is a per-class
// behavioral test of DirectoryNavigator and belongs colocated with the source layer.
import { DirectoryNavigator } from '../../../../data/nav/directory-navigator';

// Mock fetch globally
(globalThis as any).fetch = vi.fn();
// [architecture.md/O5][P3] Removed unused DOMParser stub. The HTML
// Directory Listing block exercises only the catch / fail-fall-through
// path (every fetch is mocked to fail), so DOMParser is never reached.
// The previous noop stub would have hidden a real DOMParser-throwing
// regression if the happy path were ever added. If we add a true HTML-
// parsing happy-path test, we'll mock DOMParser properly there.

describe('DirectoryNavigator', () => {
  let navigator: DirectoryNavigator;
  const mockFetch = (globalThis as any).fetch;

  beforeEach(() => {
    vi.clearAllMocks();
    navigator = new DirectoryNavigator('http://localhost:8000/');
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Zarr Detection', () => {
    it('should detect zarr datasets by .zgroup file', async () => {
      // Mock successful HEAD request for .zgroup
      mockFetch.mockResolvedValueOnce({
        ok: true,
      });

      const result = await navigator.navigate('dataset.zarr');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/dataset.zarr/.zgroup',
        expect.objectContaining({ method: 'HEAD' })
      );
      expect(result.isZarr).toBe(true);
      expect(result.entries).toEqual([]);
    });

    // BOUNDARY [P5]: a 404 on the `.zgroup` HEAD probe must NOT be treated as
    // a zarr store — `checkIfZarr` returns `response.ok`, so the navigator
    // falls through to the next strategy (here, all fail → 'manual').
    it('should NOT classify a 404 .zgroup probe as a zarr store', async () => {
      // .zgroup HEAD → 404
      mockFetch.mockResolvedValueOnce({ ok: false, status: 404 });
      // WebDAV fails
      mockFetch.mockResolvedValueOnce({ ok: false });
      // HTML (JSON attempt) fails
      mockFetch.mockResolvedValueOnce({ ok: false });
      // HTML fetch fails
      mockFetch.mockResolvedValueOnce({ ok: false });
      // Index file fails
      mockFetch.mockResolvedValueOnce({ ok: false });

      const result = await navigator.navigate('not-a-zarr');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/not-a-zarr/.zgroup',
        expect.objectContaining({ method: 'HEAD' })
      );
      expect(result.isZarr).toBe(false);
      expect(result.strategy).toBe('manual');
    });

    it('should handle non-zarr directories', async () => {
      // Mock failed HEAD request for .zgroup
      mockFetch.mockResolvedValueOnce({
        ok: false,
      });

      // Mock some directory listing response
      mockFetch.mockResolvedValueOnce({
        ok: false, // WebDAV fails
      });
      mockFetch.mockResolvedValueOnce({
        ok: false, // HTML fails
      });
      mockFetch.mockResolvedValueOnce({
        ok: false, // Index file fails
      });

      const result = await navigator.navigate('regular-dir');

      expect(result.isZarr).toBe(false);
      expect(result.strategy).toBe('manual');
    });
  });

  // [architecture.md/O3][P9] Renamed: previous label coupled to a specific
  // server implementation ("luxar serve"). The strategy DirectoryNavigator
  // implements here is "JSON directory listing" generically — any
  // conformant server emits the same payload shape.
  describe('JSON Directory Listing strategy', () => {
    it('should parse JSON directory listing', async () => {
      // Zarr check fails. TWO responses: the probe checks both root documents
      // (`zarr.json` for format 3, `.zgroup` for format 2) and a directory that
      // is not a store misses both.
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });

      // WebDAV fails
      mockFetch.mockResolvedValueOnce({ ok: false });

      // JSON response succeeds
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Map([['content-type', 'application/json']]),
        json: async () => ({
          entries: [
            { name: 'data.zarr', type: 'zarr', size: 1024 },
            { name: 'readme.txt', type: 'file', size: 100 },
            { name: 'subdir', type: 'directory' },
          ],
        }),
      });

      const result = await navigator.navigate('test');

      expect(result.strategy).toBe('html');
      expect(result.entries).toHaveLength(3);
      expect(result.entries[0]).toEqual({
        name: 'data.zarr',
        path: 'test/data.zarr',
        type: 'zarr',
        size: 1024,
      });
    });
  });

  describe('HTML Directory Listing', () => {
    // BOUNDARY [P5]: the suite previously only tested the FAILURE path for
    // HTML. This exercises the real DOMParser-backed happy path: an
    // nginx-style `<pre><a href="...">` listing. jsdom provides DOMParser,
    // so the source's `doc.querySelectorAll('pre a')` branch runs for real.
    it('classifies a .zarr.zip as a dataset and a plain .zip as a file', async () => {
      // The split the narrow discovery predicate exists to draw. Getting it
      // wrong in this direction is the visible one: every `results.zip` in a
      // listing would render with a ZARR badge and then die on click with
      // "does not contain a zarr store".
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: { get: () => 'text/html' },
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: async () =>
          `<html><body><pre>
<a href="../">../</a>
<a href="scene.luxar.zarr.zip">scene.luxar.zarr.zip</a>   01-Jan-2024 12:00   5000
<a href="results.zip">results.zip</a>                     01-Jan-2024 12:00   5000
</pre></body></html>`,
      });

      const result = await navigator.navigate('listing');
      const byName = Object.fromEntries(result.entries.map((e) => [e.name, e]));

      expect(byName['scene.luxar.zarr.zip'].type).toBe('zarr');
      expect(byName['results.zip'].type).toBe('file');
    });

    it('classifies zipped stores in list-style HTML directory listings', async () => {
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: { get: () => 'text/html' },
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: async () =>
          `<html><body><ul>
<li><a href="scene.luxar.zarr.zip">scene.luxar.zarr.zip</a></li>
<li><a href="results.zip">results.zip</a></li>
</ul></body></html>`,
      });

      const result = await navigator.navigate('listing');
      const byName = Object.fromEntries(result.entries.map((e) => [e.name, e]));

      expect(byName['scene.luxar.zarr.zip'].type).toBe('zarr');
      expect(byName['results.zip'].type).toBe('file');
    });

    it('should parse an nginx-style <pre><a href> HTML directory listing', async () => {
      // Zarr check fails. TWO responses: the probe checks both root documents
      // (`zarr.json` for format 3, `.zgroup` for format 2) and a directory that
      // is not a store misses both.
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });
      // WebDAV fails
      mockFetch.mockResolvedValueOnce({ ok: false });
      // JSON attempt: ok but non-JSON content-type → falls through to HTML
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: { get: () => 'text/html' },
      });
      // HTML fetch returns an nginx-style autoindex page
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: async () =>
          `<html><head><title>Index of /listing/</title></head><body>
<h1>Index of /listing/</h1><pre>
<a href="../">../</a>
<a href="data.zarr/">data.zarr/</a>           01-Jan-2024 12:00       -
<a href="subdir/">subdir/</a>                 01-Jan-2024 12:00       -
<a href="readme.txt">readme.txt</a>           01-Jan-2024 12:00     100
</pre></body></html>`,
      });

      const result = await navigator.navigate('listing');

      expect(result.strategy).toBe('html');
      // The ".." parent link is filtered out by the source.
      expect(result.entries).toHaveLength(3);

      const byName = Object.fromEntries(result.entries.map((e) => [e.name, e]));
      // Trailing-slash dir whose name ends with .zarr → classified as 'zarr'.
      expect(byName['data.zarr']).toEqual({
        name: 'data.zarr',
        path: 'listing/data.zarr',
        type: 'zarr',
      });
      // Trailing-slash dir → 'directory'.
      expect(byName['subdir'].type).toBe('directory');
      // No trailing slash → 'file'.
      expect(byName['readme.txt'].type).toBe('file');
    });

    it('should handle HTML parsing fallback', async () => {
      // Zarr check fails. TWO responses: the probe checks both root documents
      // (`zarr.json` for format 3, `.zgroup` for format 2) and a directory that
      // is not a store misses both.
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });

      // WebDAV fails
      mockFetch.mockResolvedValueOnce({ ok: false });

      // JSON attempt returns non-JSON
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: {
          get: () => 'text/html',
        },
      });

      // HTML fetch fails too
      mockFetch.mockResolvedValueOnce({ ok: false });

      // Index file fails
      mockFetch.mockResolvedValueOnce({ ok: false });

      const result = await navigator.navigate('test-html');

      // When HTML parsing fails, it should fall back to manual
      expect(result.strategy).toBe('manual');
      expect(result.entries).toEqual([]);
    });

    it('should handle empty directory', async () => {
      // All strategies fail
      mockFetch.mockResolvedValue({ ok: false });

      const result = await navigator.navigate('empty');

      expect(result.strategy).toBe('manual');
      expect(result.entries).toEqual([]);
    });
  });

  describe('Index File Strategy', () => {
    it('should load .luxar-index.json file', async () => {
      // Zarr check fails. TWO responses: the probe checks both root documents
      // (`zarr.json` for format 3, `.zgroup` for format 2) and a directory that
      // is not a store misses both.
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });

      // WebDAV fails
      mockFetch.mockResolvedValueOnce({ ok: false });

      // HTML fails
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });

      // Index file succeeds
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          entries: [
            { name: 'dataset1.zarr', type: 'zarr' },
            { name: 'dataset2.zarr', type: 'zarr' },
          ],
        }),
      });

      const result = await navigator.navigate('datasets');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/datasets/.luxar-index.json',
        expect.objectContaining({ signal: expect.any(AbortSignal) })
      );
      expect(result.strategy).toBe('index');
      expect(result.entries).toHaveLength(2);
    });

    it('infers zarr stores when an index omits the entry type', async () => {
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          entries: [
            { name: 'dataset1.zarr' },
            { name: 'scene.luxar.zarr.zip' },
            { name: 'results.zip' },
          ],
        }),
      });

      const result = await navigator.navigate('datasets');
      const byName = Object.fromEntries(result.entries.map((e) => [e.name, e]));

      expect(byName['dataset1.zarr'].type).toBe('zarr');
      expect(byName['scene.luxar.zarr.zip'].type).toBe('zarr');
      expect(byName['results.zip'].type).toBe('file');
    });
  });

  describe('Path Operations', () => {
    it('should handle base URL with trailing slash', () => {
      const nav1 = new DirectoryNavigator('http://localhost:8000/');
      const nav2 = new DirectoryNavigator('http://localhost:8000');

      expect(nav1.getFullUrl('')).toBe('http://localhost:8000/');
      expect(nav2.getFullUrl('')).toBe('http://localhost:8000/');
    });

    it('should build correct paths for nested navigation', async () => {
      // Mock successful navigation
      mockFetch.mockResolvedValueOnce({ ok: false }); // Not zarr
      mockFetch.mockResolvedValueOnce({ ok: false }); // WebDAV fails
      mockFetch.mockResolvedValueOnce({ ok: false }); // HTML fails
      mockFetch.mockResolvedValueOnce({ ok: false }); // HTML fails
      mockFetch.mockResolvedValueOnce({ ok: false }); // Index fails

      const result = await navigator.navigate('data/datasets/2024');

      expect(result.currentPath).toBe('data/datasets/2024');
      expect(result.parentPath).toBe('data/datasets');
    });

    it('should handle root parent path correctly', async () => {
      mockFetch.mockResolvedValue({ ok: false });

      const result1 = await navigator.navigate('');
      expect(result1.parentPath).toBeUndefined();

      const result2 = await navigator.navigate('folder');
      expect(result2.parentPath).toBe('');
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors gracefully', async () => {
      // Simulate network error
      mockFetch.mockRejectedValue(new Error('Network error'));

      const result = await navigator.navigate('test');

      // Should fall back to manual strategy
      expect(result.strategy).toBe('manual');
      expect(result.entries).toEqual([]);
      expect(result.isZarr).toBe(false);
    });

    it('should handle malformed JSON', async () => {
      // Zarr check fails. TWO responses: the probe checks both root documents
      // (`zarr.json` for format 3, `.zgroup` for format 2) and a directory that
      // is not a store misses both.
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });

      // WebDAV fails
      mockFetch.mockResolvedValueOnce({ ok: false });

      // HTML fails
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });

      // Index file returns malformed JSON
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => {
          throw new Error('Invalid JSON');
        },
      });

      const result = await navigator.navigate('bad-json');

      expect(result.strategy).toBe('manual');
      expect(result.entries).toEqual([]);
    });
  });

  describe('Public Methods', () => {
    it('should check if server can list directories', async () => {
      const canList = await navigator.canListDirectories();
      expect(typeof canList).toBe('boolean');
    });

    it('should construct full URL correctly', () => {
      const url = navigator.getFullUrl('path/to/file.txt');
      expect(url).toBe('http://localhost:8000/path/to/file.txt');
    });

    // Regression: MED-38 — getFullUrl must not produce `//` between baseUrl
    // and path. Double-slash URLs make the zarr loader treat the slash as an
    // empty path component and return 404.
    it('should normalize a leading slash on path (no double-slash)', () => {
      const url = navigator.getFullUrl('/path/to/file.txt');
      expect(url).toBe('http://localhost:8000/path/to/file.txt');
      expect(url).not.toContain('//path');
    });

    it('should handle baseUrl with no trailing slash and a slash-prefixed path', () => {
      const nav = new DirectoryNavigator('http://localhost:8000');
      const url = nav.getFullUrl('/foo');
      expect(url).toBe('http://localhost:8000/foo');
    });
  });

  // Regression: MED-39 — runtime validation of server-supplied `type`.
  describe('JSON Directory Listing: malformed type field', () => {
    it('should coerce unknown `type` values to "file" instead of crashing', async () => {
      // Zarr check fails. TWO responses: the probe checks both root documents
      // (`zarr.json` for format 3, `.zgroup` for format 2) and a directory that
      // is not a store misses both.
      mockFetch.mockResolvedValueOnce({ ok: false });
      mockFetch.mockResolvedValueOnce({ ok: false });
      // WebDAV fails
      mockFetch.mockResolvedValueOnce({ ok: false });
      // JSON response with malicious / malformed `type`
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Map([['content-type', 'application/json']]),
        json: async () => ({
          entries: [
            { name: 'evil', type: '__proto__', size: 0 },
            { name: 'unknown', type: 'symlink', size: 1 },
            { name: 'good.zarr', type: 'zarr', size: 2 },
          ],
        }),
      });

      const result = await navigator.navigate('listing');

      expect(result.entries).toHaveLength(3);
      // Both unknown types fall back to 'file' rather than being smuggled
      // through the typed union.
      expect(result.entries[0].type).toBe('file');
      expect(result.entries[1].type).toBe('file');
      // Valid types are preserved verbatim.
      expect(result.entries[2].type).toBe('zarr');
    });
  });
});
