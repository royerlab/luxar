/**
 * Tests for bootstrapStandalone().
 *
 * Covers the opt-in flags (patchConsole, validateConfig, warmCodecs), the
 * URL/localStorage-driven theme + debug-mode resolution, the
 * window.__luxarDebug seeding contract, the success and failure paths, and
 * the storage-disabled defensive read. Keeps the production code from
 * silently regressing the standalone-app boot sequence while staying
 * tractable for embedded callers.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// vi.mock factories are hoisted above all top-level statements, so any
// shared spies they reference must be created via vi.hoisted (also hoisted).
// Without this, the references are TDZ-undefined at mock-evaluation time.
const mocks = vi.hoisted(() => ({
  init: vi.fn().mockResolvedValue(undefined),
  setTheme: vi.fn(),
  getCurrentTheme: vi.fn().mockReturnValue({ id: 'frosted-glass', name: 'Frosted' }),
  patch: vi.fn(),
  validateAndLog: vi.fn().mockReturnValue(true),
  showError: vi.fn(),
  bloscThunk: vi.fn().mockResolvedValue({}),
}));

vi.mock('../../../core/app', () => ({
  LuxarApp: vi.fn().mockImplementation(() => ({
    init: mocks.init,
    initialized: true,
    dispose: vi.fn(),
  })),
}));

vi.mock('../../../themes/theme-manager', () => ({
  ThemeManager: {
    getInstance: () => ({
      setTheme: mocks.setTheme,
      getCurrentTheme: mocks.getCurrentTheme,
    }),
  },
}));

vi.mock('../../../utils/console-interceptor', () => ({
  consoleInterceptor: { patch: mocks.patch },
}));

vi.mock('../../../config/validation', () => ({
  validateAndLog: mocks.validateAndLog,
}));

vi.mock('../../../ui/helpers', () => ({
  showError: mocks.showError,
  // bootstrap.ts also imports the rest to wire setNotifierBackend(...).
  // The bootstrap unit tests don't exercise the notifier path; stubs
  // are sufficient.
  showToast: vi.fn(),
  showHelpOverlay: vi.fn(),
  hideHelpOverlay: vi.fn(),
  showLoadingIndicator: vi.fn(),
  hideLoadingIndicator: vi.fn(),
  clearError: vi.fn(),
}));

vi.mock('zarrita', () => ({
  registry: {
    get: (key: string) => (key === 'blosc' ? mocks.bloscThunk : undefined),
  },
  // Stubbed for vitest strict-mock compatibility; bootstrap path
  // doesn't reach the scene loader, so neither helper is invoked.
  tryWithConsolidated: undefined,
  withMaybeConsolidatedMetadata: undefined,
}));

import { bootstrapStandalone } from '../../../core/bootstrap';
import type { UrlParams } from '../../../config/url-params';

const CANVAS = {} as HTMLCanvasElement;

const EMPTY_PARAMS: UrlParams = {
  src: null,
  theme: null,
  debug: false,
  noCache: false,
  cacheDebug: false,
  clearCache: false,
  noPrefetch: false,
  prefetchDebug: false,
  cacheStats: false,
};

describe('bootstrapStandalone', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.init.mockResolvedValue(undefined);
    delete (window as { __luxarDebug?: unknown }).__luxarDebug;
    localStorage.clear();
  });

  afterEach(() => {
    delete (window as { __luxarDebug?: unknown }).__luxarDebug;
    localStorage.clear();
  });

  describe('opt-in flags', () => {
    it('patches console, validates config, and warms codecs when defaults apply', async () => {
      await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });

      expect(mocks.patch).toHaveBeenCalledTimes(1);
      expect(mocks.validateAndLog).toHaveBeenCalledTimes(1);
      expect(mocks.bloscThunk).toHaveBeenCalledTimes(1);
    });

    it('skips console patching when patchConsole=false', async () => {
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: EMPTY_PARAMS,
        patchConsole: false,
      });
      expect(mocks.patch).not.toHaveBeenCalled();
    });

    it('skips config validation when validateConfig=false', async () => {
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: EMPTY_PARAMS,
        validateConfig: false,
      });
      expect(mocks.validateAndLog).not.toHaveBeenCalled();
    });

    it('skips codec warming when warmCodecs=false', async () => {
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: EMPTY_PARAMS,
        warmCodecs: false,
      });
      expect(mocks.bloscThunk).not.toHaveBeenCalled();
    });
  });

  describe('theme resolution', () => {
    it('applies the theme from urlParams.theme when present', async () => {
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, theme: 'dark' },
      });
      expect(mocks.setTheme).toHaveBeenCalledWith('dark');
    });

    it('logs a warning and falls back to default when the URL theme is invalid', async () => {
      mocks.setTheme.mockImplementationOnce(() => {
        throw new Error('Theme "invalid" not found');
      });
      // Should not throw — bootstrap catches the theme error and continues.
      await expect(
        bootstrapStandalone({
          canvas: CANVAS,
          urlParams: { ...EMPTY_PARAMS, theme: 'invalid' },
        })
      ).resolves.toBeDefined();
    });

    it('does not call setTheme when no urlParams.theme is provided', async () => {
      await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });
      expect(mocks.setTheme).not.toHaveBeenCalled();
      // Still touches the manager to log the active theme.
      expect(mocks.getCurrentTheme).toHaveBeenCalled();
    });
  });

  describe('debug mode', () => {
    it('seeds window.__luxarDebug when urlParams.debug is true', async () => {
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, debug: true },
      });
      expect(window.__luxarDebug).toBeDefined();
      expect(window.__luxarDebug?.version).toBe('1.0.0');
      expect(window.__luxarDebug?.app).toBeDefined();
    });

    it('seeds window.__luxarDebug when localStorage[luxar.debug] is "true"', async () => {
      localStorage.setItem('luxar.debug', 'true');
      await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });
      expect(window.__luxarDebug).toBeDefined();
    });

    it('does not seed window.__luxarDebug without either flag', async () => {
      await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });
      expect(window.__luxarDebug).toBeUndefined();
    });

    it('forwards debug:true to LuxarApp.init()', async () => {
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, debug: true },
      });
      const lastCallArg = mocks.init.mock.calls.at(-1)?.[0];
      expect(lastCallArg.debug).toBe(true);
    });
  });

  describe('storage-disabled defensiveness', () => {
    it('survives a localStorage.getItem that throws', async () => {
      const original = Storage.prototype.getItem;
      Storage.prototype.getItem = vi.fn(() => {
        throw new Error('storage disabled');
      });

      try {
        await expect(
          bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS })
        ).resolves.toBeDefined();
        // Without ?debug and with storage throwing, debug mode must stay off.
        expect(window.__luxarDebug).toBeUndefined();
      } finally {
        Storage.prototype.getItem = original;
      }
    });
  });

  describe('success and error paths', () => {
    it('returns the constructed LuxarApp on success', async () => {
      const app = await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: EMPTY_PARAMS,
      });
      expect(app).toBeDefined();
      expect(mocks.init).toHaveBeenCalledTimes(1);
    });

    it('forwards loader flags from urlParams into LuxarApp.init()', async () => {
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: {
          ...EMPTY_PARAMS,
          src: 'https://example.com/data.zarr',
          noCache: true,
          cacheDebug: true,
        },
      });
      const arg = mocks.init.mock.calls.at(-1)?.[0];
      expect(arg.canvas).toBe(CANVAS);
      expect(arg.src).toBe('https://example.com/data.zarr');
      expect(arg.updateBrowserUrl).toBe(true);
      expect(arg.loaderConfig).toMatchObject({
        noCache: true,
        cacheDebug: true,
        clearCache: false,
        noPrefetch: false,
        prefetchDebug: false,
      });
    });

    it('shows a top-level error UI and re-throws when init() fails', async () => {
      const initError = new Error('WebGL unavailable');
      mocks.init.mockRejectedValueOnce(initError);

      await expect(
        bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS })
      ).rejects.toThrow('WebGL unavailable');

      expect(mocks.showError).toHaveBeenCalledTimes(1);
      expect(mocks.showError.mock.calls[0][0]).toMatch(/Failed to start the application/i);
    });
  });
});
