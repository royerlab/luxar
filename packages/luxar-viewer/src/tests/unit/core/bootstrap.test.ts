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

vi.mock('../../../ui/error-overlay', () => ({
  showError: mocks.showError,
  clearError: vi.fn(),
}));
vi.mock('../../../ui/toast', () => ({ showToast: vi.fn() }));
vi.mock('../../../ui/help-overlay', () => ({
  showHelpOverlay: vi.fn(),
  hideHelpOverlay: vi.fn(),
}));
vi.mock('../../../ui/loading-indicator', () => ({
  showLoadingIndicator: vi.fn(),
  hideLoadingIndicator: vi.fn(),
}));

vi.mock('zarrita', () => ({
  registry: {
    get: (key: string) => (key === 'blosc' ? mocks.bloscThunk : undefined),
  },
  // Stubbed for vitest strict-mock compatibility; bootstrap path
  // doesn't reach the scene loader, so this helper is not invoked.
  withMaybeConsolidatedMetadata: undefined,
}));

import { bootstrapStandalone } from '../../../core/bootstrap';
import { log } from '../../../utils/log';
import type { UrlParams } from '../../../config/url-params';

const CANVAS = {} as HTMLCanvasElement;

const EMPTY_PARAMS: UrlParams = {
  src: null,
  theme: null,
  debug: false,
  noCache: false,
  noSliceCache: false,
  cacheDebug: false,
  clearCache: false,
  noPrefetch: false,
  prefetchDebug: false,
  cacheStats: false,
  renderer: null,
  webgpuForceWebGL: false,
  perfTimestamp: false,
  gpuBudgetMB: null,
  dpr: null,
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
      // core.md W1 strengthening: previously this test only asserted
      // `.resolves.toBeDefined()`. The test's name promises a logged
      // warning AND a fallback — both are now asserted directly.
      const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
      mocks.setTheme.mockImplementationOnce(() => {
        throw new Error('Theme "invalid" not found');
      });
      // Should not throw — bootstrap catches the theme error and continues.
      const app = await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, theme: 'invalid' },
      });
      expect(app).toBeDefined();

      // Warning was logged: the message includes the bad theme id and
      // mentions falling back to default.
      const warningMsg = warnSpy.mock.calls.map((args) => String(args[1] ?? '')).join('\n');
      expect(warningMsg).toMatch(/invalid theme/i);
      expect(warningMsg).toMatch(/invalid/);
      expect(warningMsg).toMatch(/using default/i);

      // init() still ran (fallback path), confirming bootstrap didn't
      // abort on the theme failure.
      expect(mocks.init).toHaveBeenCalledTimes(1);
      warnSpy.mockRestore();
    });

    it('does not call setTheme when no urlParams.theme is provided', async () => {
      await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });
      expect(mocks.setTheme).not.toHaveBeenCalled();
      // Still touches the manager to log the active theme.
      expect(mocks.getCurrentTheme).toHaveBeenCalled();
    });

    // [core OOS] Pre-fix, the theme block's try/catch only wrapped
    // setTheme. If the ELSE branch's `getCurrentTheme().name` threw —
    // realistic if localStorage is corrupt and theme-manager throws
    // mid-init — the exception escaped bootstrap entirely, leaving the
    // viewer un-initialized. The outer try/catch downgrades to a
    // warning + proceed.
    it('downgrades a getCurrentTheme() throw to a warning and continues bootstrap', async () => {
      const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
      mocks.getCurrentTheme.mockImplementationOnce(() => {
        throw new Error('corrupt localStorage');
      });

      // Should NOT throw — outer try/catch swallows.
      const app = await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: EMPTY_PARAMS, // no urlParams.theme → enters else branch
      });
      expect(app).toBeDefined();

      // Warning fires with the expected shape.
      const warningMsg = warnSpy.mock.calls.map((args) => String(args[1] ?? '')).join('\n');
      expect(warningMsg).toMatch(/Theme initialization failed unexpectedly.*corrupt localStorage/);
      expect(warningMsg).toMatch(/Proceeding with default theme/);

      // init() still ran (bootstrap continued past the theme block).
      expect(mocks.init).toHaveBeenCalledTimes(1);
      warnSpy.mockRestore();
    });
  });

  describe('debug mode', () => {
    it('seeds window.__luxarDebug when urlParams.debug is true', async () => {
      // core.md W8 strengthening: assert every documented field of the
      // pre-init __luxarDebug shape (see core/bootstrap.ts:198-203). A
      // regression that removed any of them would silently break the
      // contracts with Playwright + visual-regression tests.
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, debug: true },
      });
      expect(window.__luxarDebug).toBeDefined();
      expect(window.__luxarDebug?.version).toBe('1.0.0');
      expect(window.__luxarDebug?.app).toBeDefined();
      expect(window.__luxarDebug?.consoleInterceptor).toBeDefined();
      // showError is exposed so visual-regression specs can drive the
      // error dialog directly (see bootstrap.ts:195-203 comment).
      expect(typeof window.__luxarDebug?.showError).toBe('function');
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

    it('forwards loader and renderer diagnostic flags from urlParams into LuxarApp.init()', async () => {
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: {
          ...EMPTY_PARAMS,
          src: 'https://example.com/data.zarr',
          noCache: true,
          cacheDebug: true,
          renderer: 'webgpu',
          webgpuForceWebGL: true,
          perfTimestamp: true,
        },
      });
      const arg = mocks.init.mock.calls.at(-1)?.[0];
      expect(arg.canvas).toBe(CANVAS);
      expect(arg.src).toBe('https://example.com/data.zarr');
      expect(arg.updateBrowserUrl).toBe(true);
      expect(arg.renderer).toBe('webgpu');
      expect(arg.webgpuForceWebGL).toBe(true);
      expect(arg.perfTimestamp).toBe(true);
      expect(arg.loaderConfig).toMatchObject({
        noCache: true,
        cacheDebug: true,
        clearCache: false,
        noPrefetch: false,
        prefetchDebug: false,
      });
    });

    it('threads urlParams.dpr into init() as pinnedDPR, and omits it when absent', async () => {
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, dpr: 0.5 },
      });
      expect(mocks.init.mock.calls.at(-1)?.[0].pinnedDPR).toBe(0.5);

      await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });
      // null → undefined so the pipeline's `!== undefined` pin guard stays cold.
      expect(mocks.init.mock.calls.at(-1)?.[0].pinnedDPR).toBeUndefined();
    });

    it('does not set perfTimestamp on the init() call when urlParams.perfTimestamp is false', async () => {
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, perfTimestamp: false },
      });
      const arg = mocks.init.mock.calls.at(-1)?.[0];
      // Field present but explicitly false so SceneManager doesn't
      // probe WebGPURenderer.backend.trackTimestamp by accident.
      expect(arg.perfTimestamp).toBe(false);
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
