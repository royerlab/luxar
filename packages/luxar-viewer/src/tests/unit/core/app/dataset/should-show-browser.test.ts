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

    it('loads zipped stores directly without probing URL-addressable children', async () => {
      for (const src of [
        'https://example.com/scene.luxar.zarr.zip',
        'https://example.com/SCENE.LUXAR.ZARR.ZIP?token=abc#view',
      ]) {
        expect(await shouldShowBrowser(src)).toBe(false);
      }
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

      const result = await shouldShowBrowser('https://example.com/data.zarr');
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

      const result = await shouldShowBrowser('https://example.com/data.zarr');
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

      const result = await shouldShowBrowser('https://example.com/data.zarr');
      expect(result).toBe(false);
    });

    it('returns true when ALL probes return 404 (likely directory listing)', async () => {
      fetchSpy.mockResolvedValue(new Response(null, { status: 404 }));

      const result = await shouldShowBrowser('https://example.com/data.zarr');
      expect(result).toBe(true);
    });

    it('returns true when ALL probes reject (network error)', async () => {
      fetchSpy.mockRejectedValue(new Error('network down'));

      const result = await shouldShowBrowser('https://example.com/data.zarr');
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

      const result = await shouldShowBrowser('https://example.com/data.zarr');
      expect(result).toBe(true);
    });

    it('fires HEAD probes for all three zarr metadata filenames', async () => {
      fetchSpy.mockResolvedValue(new Response(null, { status: 404 }));

      await shouldShowBrowser('https://example.com/data.zarr');

      // 3 probes: .zgroup, .zattrs, zarr.json — all HEAD.
      expect(fetchSpy).toHaveBeenCalledTimes(3);
      const urls = fetchSpy.mock.calls.map((c: unknown[]) => String(c[0]));
      expect(urls).toContain('https://example.com/data.zarr/.zgroup');
      expect(urls).toContain('https://example.com/data.zarr/.zattrs');
      expect(urls).toContain('https://example.com/data.zarr/zarr.json');
      for (const call of fetchSpy.mock.calls) {
        const init = call[1] as RequestInit | undefined;
        expect(init?.method).toBe('HEAD');
      }
    });

    it('passes an AbortSignal to each probe so the 5s timeout can cancel in-flight requests', async () => {
      fetchSpy.mockResolvedValue(new Response(null, { status: 404 }));

      await shouldShowBrowser('https://example.com/data.zarr');

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

      const result = await shouldShowBrowser('https://example.com/data.zarr');
      expect(result).toBe(false);
      // The other two probes were started (parallel fetch), even though
      // we don't await them. Asserting the count is 2 confirms the
      // parallel-issue strategy isn't accidentally serialized.
      expect(neverResolveCount).toBe(2);
    });
  });
});
