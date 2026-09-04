/**
 * Unit tests for core/app/dataset/should-show-browser.ts (G13).
 *
 * `shouldShowBrowser(src)` is the async wrapper around `classifyBrowserUrl`
 * that ALSO probes the URL for zarr metadata files. Decision tree:
 *   - Synchronous classification → must-browse → true (no fetch).
 *   - Probe `.zgroup`, `.zattrs`, `zarr.json` HEAD in parallel:
 *       any 2xx → false (load directly).
 *       all fail / abort → true (open the browser).
 *
 * `classifyBrowserUrl` is unit-tested in browser-decision.test.ts;
 * here we focus on the async wrapping + probe logic.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { shouldShowBrowser } from '../../../../../core/app/dataset/should-show-browser';

describe('shouldShowBrowser', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, 'fetch');
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  describe('synchronous classification short-circuit', () => {
    it('returns true for empty string (must-browse) without firing any fetch', async () => {
      const result = await shouldShowBrowser('');
      expect(result).toBe(true);
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it('returns true for trailing-slash URLs without firing any fetch', async () => {
      const result = await shouldShowBrowser('https://example.com/data/');
      expect(result).toBe(true);
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it('returns true for whitespace-only input', async () => {
      const result = await shouldShowBrowser('   ');
      expect(result).toBe(true);
      expect(fetchSpy).not.toHaveBeenCalled();
    });
  });

  describe('zarr metadata probes', () => {
    it('returns false (load directly) when .zgroup probe succeeds', async () => {
      fetchSpy.mockImplementation((url: string | URL | Request) => {
        const u = String(url);
        if (u.endsWith('.zgroup')) {
          return Promise.resolve(new Response(null, { status: 200 }));
        }
        // Other probes resolve to 404 — they'd reject inside Promise.any
        // because the helper throws on non-ok.
        return Promise.resolve(new Response(null, { status: 404 }));
      });

      const result = await shouldShowBrowser('https://example.com/dataset');
      expect(result).toBe(false);
    });

    it('returns false when .zattrs probe succeeds (zarr v2 attrs-only)', async () => {
      fetchSpy.mockImplementation((url: string | URL | Request) => {
        const u = String(url);
        if (u.endsWith('.zattrs')) {
          return Promise.resolve(new Response(null, { status: 200 }));
        }
        return Promise.resolve(new Response(null, { status: 404 }));
      });

      const result = await shouldShowBrowser('https://example.com/dataset');
      expect(result).toBe(false);
    });

    it('returns false when zarr.json probe succeeds (zarr v3)', async () => {
      fetchSpy.mockImplementation((url: string | URL | Request) => {
        const u = String(url);
        if (u.endsWith('zarr.json')) {
          return Promise.resolve(new Response(null, { status: 200 }));
        }
        return Promise.resolve(new Response(null, { status: 404 }));
      });

      const result = await shouldShowBrowser('https://example.com/dataset');
      expect(result).toBe(false);
    });

    it('returns true when ALL probes return 404 (likely directory listing)', async () => {
      fetchSpy.mockResolvedValue(new Response(null, { status: 404 }));

      const result = await shouldShowBrowser('https://example.com/dataset');
      expect(result).toBe(true);
    });

    it('returns true when ALL probes reject (network error)', async () => {
      fetchSpy.mockRejectedValue(new Error('network down'));

      const result = await shouldShowBrowser('https://example.com/dataset');
      expect(result).toBe(true);
    });

    it('returns true on mixed 5xx + network errors', async () => {
      // Server errors should not be interpreted as "zarr exists".
      fetchSpy.mockImplementation((url: string | URL | Request) => {
        const u = String(url);
        if (u.endsWith('.zgroup')) {
          return Promise.resolve(new Response(null, { status: 500 }));
        }
        return Promise.reject(new Error('timeout'));
      });

      const result = await shouldShowBrowser('https://example.com/dataset');
      expect(result).toBe(true);
    });

    it('fires HEAD probes for all three zarr metadata filenames', async () => {
      fetchSpy.mockResolvedValue(new Response(null, { status: 404 }));

      await shouldShowBrowser('https://example.com/dataset');

      // 3 probes: .zgroup, .zattrs, zarr.json — all HEAD.
      expect(fetchSpy).toHaveBeenCalledTimes(3);
      const urls = fetchSpy.mock.calls.map((c: unknown[]) => String(c[0]));
      expect(urls).toContain('https://example.com/dataset/.zgroup');
      expect(urls).toContain('https://example.com/dataset/.zattrs');
      expect(urls).toContain('https://example.com/dataset/zarr.json');
      for (const call of fetchSpy.mock.calls) {
        const init = call[1] as RequestInit | undefined;
        expect(init?.method).toBe('HEAD');
      }
    });

    it('passes an AbortSignal to each probe so the 5s timeout can cancel in-flight requests', async () => {
      fetchSpy.mockResolvedValue(new Response(null, { status: 404 }));

      await shouldShowBrowser('https://example.com/dataset');

      for (const call of fetchSpy.mock.calls) {
        const init = call[1] as RequestInit | undefined;
        expect(init?.signal).toBeInstanceOf(AbortSignal);
      }
    });
  });

  describe('first-success short-circuit', () => {
    it('returns false as soon as the first probe resolves OK (does not wait for the others)', async () => {
      // The first probe resolves immediately; the others are pending
      // forever. The function must still return false promptly via
      // Promise.any.
      let neverResolveCount = 0;
      fetchSpy.mockImplementation((url: string | URL | Request) => {
        const u = String(url);
        if (u.endsWith('.zgroup')) {
          return Promise.resolve(new Response(null, { status: 200 }));
        }
        neverResolveCount++;
        return new Promise(() => {
          // never resolves
        });
      });

      const result = await shouldShowBrowser('https://example.com/dataset');
      expect(result).toBe(false);
      // The other two probes were started (parallel fetch), even though
      // we don't await them. Asserting the count is 2 confirms the
      // parallel-issue strategy isn't accidentally serialized.
      expect(neverResolveCount).toBe(2);
    });
  });
});

describe('shouldShowBrowser — zipped stores', () => {
  it('loads a .zarr.zip directly without probing for children', async () => {
    // The child probes are meaningless for an archive: `archive.zip/zarr.json`
    // 404s for EVERY archive, so probing would divert a perfectly loadable
    // dataset into the browser.
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);

    await expect(shouldShowBrowser('https://example.com/scene.luxar.zarr.zip')).resolves.toBe(
      false
    );
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('does not mistake a directory store whose name merely contains .zip for an archive', async () => {
    const fetchSpy = vi.fn(async () => ({ ok: false }) as Response);
    vi.stubGlobal('fetch', fetchSpy);

    // `.zip` appears mid-name but the URL still NAMES a `.zarr` store, so it
    // loads directly — via the suffix short-circuit rather than the archive
    // branch. Either way it must not open the browser.
    await expect(shouldShowBrowser('https://example.com/archive.zip.luxar.zarr')).resolves.toBe(
      false
    );
  });

  it('probes a non-.zarr URL whose name merely contains .zip', async () => {
    const fetchSpy = vi.fn(async () => ({ ok: false }) as Response);
    vi.stubGlobal('fetch', fetchSpy);

    // Neither an archive nor a suffixed store: this is the case the HEAD
    // probes exist for, and `isZippedStoreUrl` must not claim it.
    await shouldShowBrowser('https://example.com/archive.zip.dataset');
    expect(fetchSpy).toHaveBeenCalled();
  });
});

describe('shouldShowBrowser — .zarr suffix short-circuit', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it.each([
    'https://example.com/scene.luxar.zarr',
    'https://example.com/fit.gsplats.zarr',
    'https://example.com/plain.zarr',
    // A presigned/tokenized source: the suffix lives on the PATH, so a query
    // string must not hide it.
    'https://example.com/scene.luxar.zarr?token=abc',
  ])('loads %s directly without any probe', async (src) => {
    const fetchSpy = vi.fn(async () => ({ ok: true }) as Response);
    vi.stubGlobal('fetch', fetchSpy);

    await expect(shouldShowBrowser(src)).resolves.toBe(false);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('still browses a DIRECTORY whose name ends in .zarr', async () => {
    const fetchSpy = vi.fn(async () => ({ ok: true }) as Response);
    vi.stubGlobal('fetch', fetchSpy);

    // The trailing slash means "list this", and `classifyBrowserUrl` catches
    // it before the suffix test ever runs. Regression guard: a suffix check
    // written against the raw string would swallow this case.
    await expect(shouldShowBrowser('https://example.com/dir.zarr/')).resolves.toBe(true);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
