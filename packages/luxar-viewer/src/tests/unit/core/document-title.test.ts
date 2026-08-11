/**
 * Tests for the browser tab title helpers.
 *
 * The interesting behaviour is the restore path: once anything has overwritten
 * `document.title`, a dataset switch to a scene that names itself nothing must
 * put the page's own `<title>` back rather than keep advertising the previous
 * scene. `vi.resetModules()` per test re-arms the module-local "page title"
 * capture, which is deliberately a once-per-page thing in production.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

const MODULE = '../../../core/document-title';

async function freshModule() {
  vi.resetModules();
  return import(MODULE);
}

describe('document title', () => {
  let original: string;

  beforeEach(() => {
    original = document.title;
  });

  afterEach(() => {
    document.title = original;
  });

  describe('setDocumentTitle', () => {
    it('restores the page title once something has overwritten it', async () => {
      const { setDocumentTitle } = await freshModule();
      document.title = 'Luxar Player – 3D Scene Viewer';

      setDocumentTitle('Rivers of Earth');
      expect(document.title).toBe('Rivers of Earth');

      // A second scene, then a scene that names itself nothing: the page
      // title comes back, NOT the previous scene's name.
      setDocumentTitle('Desi DR1');
      setDocumentTitle(null);
      expect(document.title).toBe('Luxar Player – 3D Scene Viewer');
    });

    it('trims, and treats a blank title as no title', async () => {
      const { setDocumentTitle } = await freshModule();
      document.title = 'Page';

      setDocumentTitle('  Rivers of Earth  ');
      expect(document.title).toBe('Rivers of Earth');

      setDocumentTitle('   ');
      expect(document.title).toBe('Page');
    });

    it('is a no-op before anything has overwritten the title', async () => {
      // Nothing to restore yet — the page title is still in place, and
      // capturing it here would freeze whatever a host page set later.
      const { setDocumentTitle } = await freshModule();
      document.title = 'Host page';

      setDocumentTitle(null);
      setDocumentTitle(undefined);
      expect(document.title).toBe('Host page');
    });
  });

  describe('dataSourceDocumentTitle', () => {
    it('derives the store name, compound and archive suffixes stripped', async () => {
      const { dataSourceDocumentTitle } = await freshModule();

      expect(dataSourceDocumentTitle('http://h:8000/d/global_rivers.luxar.zarr')).toBe(
        'global_rivers'
      );
      expect(dataSourceDocumentTitle('datasets/fit.gsplats.zarr')).toBe('fit');
      expect(dataSourceDocumentTitle('datasets/plain.zarr')).toBe('plain');
      expect(dataSourceDocumentTitle('datasets/desi_dr1.luxar.zarr.zip')).toBe('desi_dr1');
      expect(dataSourceDocumentTitle('datasets/fit.gsplats.zarr.tar.gz')).toBe('fit');
      expect(dataSourceDocumentTitle('datasets/fit.gsplats.zarr.TGZ')).toBe('fit');
    });

    it('ignores trailing slashes, query strings, hashes, and percent-escapes', async () => {
      const { dataSourceDocumentTitle } = await freshModule();

      expect(dataSourceDocumentTitle('http://h:8000/d/scene.luxar.zarr/')).toBe('scene');
      expect(dataSourceDocumentTitle('http://h:8000/d/scene.luxar.zarr?v=2#x')).toBe('scene');
      expect(dataSourceDocumentTitle('http://h:8000/d/Rivers%20of%20Earth.luxar.zarr')).toBe(
        'Rivers of Earth'
      );
      // A malformed escape must not throw — the raw segment still names it.
      expect(dataSourceDocumentTitle('http://h:8000/d/100%.luxar.zarr')).toBe('100%');
    });

    it('returns null when the URL does not name a store', async () => {
      // A bare data-server root is the common case: its last path segment is
      // a host:port, which would make a nonsense tab title. `?title=` exists
      // precisely because the server has to supply the name here.
      const { dataSourceDocumentTitle } = await freshModule();

      expect(dataSourceDocumentTitle('http://127.0.0.1:8000')).toBeNull();
      expect(dataSourceDocumentTitle('http://127.0.0.1:8000/')).toBeNull();
      expect(dataSourceDocumentTitle('datasets/notes.txt')).toBeNull();
      expect(dataSourceDocumentTitle('datasets/.zarr')).toBeNull();
    });
  });
});
