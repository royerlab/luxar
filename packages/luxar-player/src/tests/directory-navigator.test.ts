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
import { DirectoryNavigator } from '../data/directory-navigator';

// Mock fetch globally
(globalThis as any).fetch = vi.fn();
(globalThis as any).DOMParser = vi.fn().mockImplementation(() => ({
  parseFromString: vi.fn(),
}));

describe('DirectoryNavigator', () => {
  let navigator: DirectoryNavigator;
  const mockFetch = (globalThis as any).fetch;
  void (globalThis as any).DOMParser; // Mark as intentionally accessed but not stored

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

      expect(mockFetch).toHaveBeenCalledWith('http://localhost:8000/dataset.zarr.zgroup', {
        method: 'HEAD',
      });
      expect(result.isZarr).toBe(true);
      expect(result.entries).toEqual([]);
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

  describe('JSON Directory Listing (luxar serve)', () => {
    it('should parse JSON directory listing', async () => {
      // Zarr check fails
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
    it('should handle HTML parsing fallback', async () => {
      // Zarr check fails
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
      // Zarr check fails
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

      expect(mockFetch).toHaveBeenCalledWith('http://localhost:8000/datasets.luxar-index.json');
      expect(result.strategy).toBe('index');
      expect(result.entries).toHaveLength(2);
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
      // Zarr check fails
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
  });
});
