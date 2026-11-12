/**
 * Unit tests for core/app/dataset/should-show-browser.ts (G13).
 *
 * `shouldShowBrowser(src)` is the async wrapper around `classifyBrowserUrl`
 * that ALSO probes the URL for zarr metadata files. Decision tree:
 *   - Synchronous classification → must-browse → true (no fetch).
 *   - Probe `zarr.json`, `.zgroup`, `.zattrs` HEAD one at a time:
 *       first 2xx → false (load directly), later markers never requested.
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

    it('fires HEAD probes for all three zarr metadata filenames, v3 first', async () => {
      fetchSpy.mockResolvedValue(new Response(null, { status: 404 }));

      await shouldShowBrowser('https://example.com/dataset');

      // 3 probes, in order: zarr.json, .zgroup, .zattrs — all HEAD.
      expect(fetchSpy).toHaveBeenCalledTimes(3);
      const urls = fetchSpy.mock.calls.map((c: unknown[]) => String(c[0]));
      expect(urls).toEqual([
        'https://example.com/dataset/zarr.json',
        'https://example.com/dataset/.zgroup',
        'https://example.com/dataset/.zattrs',
      ]);
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
    it('a format-3 store costs exactly one request and no 404', async () => {
      // `luxar serve` mounts a store at `/` and an export at `/data`: both are
      // suffix-less, so they reach this probe on every load. The parallel probe
      // logged `.zgroup` and `.zattrs` 404s in the console each time.
      fetchSpy.mockImplementation((url: string) =>
        Promise.resolve(new Response(null, { status: url.endsWith('zarr.json') ? 200 : 404 }))
      );

      const result = await shouldShowBrowser('https://example.com/data');
      expect(result).toBe(false);
      expect(fetchSpy).toHaveBeenCalledTimes(1);
      expect(fetchSpy.mock.calls[0][0]).toBe('https://example.com/data/zarr.json');
    });

    it('a format-2 store stops at .zgroup without probing .zattrs', async () => {
      fetchSpy.mockImplementation((url: string) =>
        Promise.resolve(new Response(null, { status: url.endsWith('.zgroup') ? 200 : 404 }))
      );

      await expect(shouldShowBrowser('https://example.com/data')).resolves.toBe(false);
      const urls = fetchSpy.mock.calls.map((c: unknown[]) => String(c[0]));
      expect(urls).toEqual([
        'https://example.com/data/zarr.json',
        'https://example.com/data/.zgroup',
      ]);
    });

    it('gives each probe its own budget, so a slow 404 does not starve the next', async () => {
      vi.useFakeTimers();
      try {
        fetchSpy.mockImplementation(
          (url: string) =>
            new Promise((resolve) => {
              // zarr.json 404s after 4 s; .zgroup answers 200 after 4 s more.
              // A shared 5 s budget would abort .zgroup and open the browser.
              setTimeout(
                () => resolve(new Response(null, { status: url.endsWith('.zgroup') ? 200 : 404 })),
                4000
              );
            })
        );
        const pending = shouldShowBrowser('https://example.com/dataset');
        await vi.advanceTimersByTimeAsync(8000);
        await expect(pending).resolves.toBe(false);
      } finally {
        vi.useRealTimers();
      }
    });

    it('opens the browser when a probe hangs past the budget', async () => {
      vi.useFakeTimers();
      try {
        fetchSpy.mockImplementation(
          (_url: string | URL | Request, init?: RequestInit) =>
            new Promise((_resolve, reject) => {
              init?.signal?.addEventListener('abort', () => reject(new Error('aborted')));
            })
        );
        const pending = shouldShowBrowser('https://example.com/dataset');
        await vi.advanceTimersByTimeAsync(5000);
        await expect(pending).resolves.toBe(true);
        // A hung server ends the loop: the later markers were never requested.
        expect(fetchSpy).toHaveBeenCalledTimes(1);
      } finally {
        vi.useRealTimers();
      }
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
