// @vitest-environment jsdom
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
  shortcutForAction: vi.fn().mockReturnValue('F1'),
  bloscThunk: vi.fn().mockResolvedValue({}),
  restoreSnapshot: vi.fn(),
  setRenderingSettings: vi.fn(),
  flyTo: vi.fn().mockResolvedValue({ completed: true }),
  disposeCleanups: [] as Array<() => void>,
  renderer: {
    info: {
      autoReset: true,
      reset: () => undefined,
      render: { frame: 1, calls: 7, triangles: 70, points: 0, lines: 0 },
      memory: { geometries: 2, textures: 3 },
    },
  },
}));

vi.mock('../../../core/app', () => ({
  LuxarApp: vi.fn().mockImplementation(() => ({
    init: mocks.init,
    initialized: true,
    shortcutForAction: mocks.shortcutForAction,
    getViewerState: () => ({ src: '/sample.zarr' }),
    restoreSnapshot: mocks.restoreSnapshot,
    setRenderingSettings: mocks.setRenderingSettings,
    getLayers: () => [],
    flyTo: mocks.flyTo,
    components: { sceneManager: { renderer: mocks.renderer } },
    onDispose: (cleanup: () => void) => {
      mocks.disposeCleanups.push(cleanup);
    },
    dispose: () => {
      mocks.disposeCleanups.splice(0).forEach((cleanup) => cleanup());
    },
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

import { buildInfo, buildInfoLine } from '../../../config/build-info';
import { VIEWER_VERSION } from '../../../version';
import { config } from '../../../config';
import { bootstrapStandalone } from '../../../core/bootstrap';
import { getGpuByteBudget } from '../../../rendering/gpu-byte-budget';
import { ArchiveFaultError } from '../../../cache/chunk-source';
import { UnsupportedFormatVersionError } from '../../../data/format-version';
import { setDocumentTitle } from '../../../core/document-title';
import { log } from '../../../utils/log';
import { notifier } from '../../../utils/cross-layer/notifier';
import type { UrlParams } from '../../../config/url-params';
import * as THREE from 'three';
import { perfCounters } from '../../../profiling/perf-counters';
import { eventBus } from '../../../utils/cross-layer/event-bus';
import {
  lodLoadStatsEnabled,
  setLodLoadStatsEnabled,
} from '../../../data/scene-loader/lod-load-stats';
import {
  getRendererInfoSnapshot,
  uninstallRendererInfoSampler,
} from '../../../core/app/debug/renderer-info-sampler';
import { resetRootDocumentPrefetchForTests } from '../../../cache/root-document-prefetch';
import {
  defaultUserSettings,
  saveUserSettings,
  resetUserSettingsForTests,
} from '../../../config/user-settings';

const CANVAS = {} as HTMLCanvasElement;

const EMPTY_PARAMS: UrlParams = {
  src: null,
  theme: null,
  title: null,
  view: null,
  control: null,
  controlToken: null,
  controlAllowCrossOrigin: false,
  panel: null,
  debug: false,
  verboseLog: false,
  kiosk: false,
  noCache: false,
  noSliceCache: false,
  noOpfs: false,
  opfsReadConcurrency: null,
  cacheDebug: false,
  clearCache: false,
  lodFade: true,
  allowLinks: true,
  lodEnergyComp: true,
  blendWarmup: true,
  depthSort: true,
  densityGuard: true,
  densityCap: null,
  lodFinest: false, // capture-quality force-finest is OFF by default (opt-in via ?lodFinest)
  lodBias: null,
  noPrefetch: false,
  prefetchDebug: false,
  cacheStats: false,
  renderer: null,
  webgpuForceWebGL: false,
  perfTimestamp: false,
  renderAlways: false,
  renderAudit: false,
  gpuBudgetMB: null,
  cacheBudgetMB: null,
  dpr: null,
  input: null,
  lineJoin: null,
  linePrimitive: null,
  bakeEnv: false,
  probe: null,
  envResolution: null,
};

/** URLs the stubbed `fetch` was asked for (bootstrap's root-document prefetch). */
let fetched: string[] = [];

describe('bootstrapStandalone', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.disposeCleanups.splice(0).forEach((cleanup) => cleanup());
    // Bootstrap starts the dataset's root-document fetch itself; never let a
    // test reach a real network.
    fetched = [];
    resetRootDocumentPrefetchForTests();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        fetched.push(
          typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
        );
        return new Response('', { status: 404 });
      })
    );
    mocks.init.mockResolvedValue(undefined);
    delete (window as { __luxarDebug?: unknown }).__luxarDebug;
    delete window.__luxarBuild;
    localStorage.clear();
    resetUserSettingsForTests();
  });

  afterEach(() => {
    mocks.disposeCleanups.splice(0).forEach((cleanup) => cleanup());
    delete (window as { __luxarDebug?: unknown }).__luxarDebug;
    delete window.__luxarBuild;
    localStorage.clear();
    resetRootDocumentPrefetchForTests();
    vi.unstubAllGlobals();
  });

  describe('root-document prefetch', () => {
    it('requests the root document BEFORE app init (renderer init) completes', async () => {
      // A slow renderer init: app.init does not settle until released.
      let release!: () => void;
      mocks.init.mockImplementationOnce(() => new Promise<void>((r) => (release = r)));
      const booted = bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, src: 'https://data.example/scene.luxar.zarr' },
      });
      await vi.waitFor(() =>
        expect(fetched).toContain('https://data.example/scene.luxar.zarr/zarr.json')
      );
      expect(release).toBeTypeOf('function'); // init is still in flight
      release();
      await booted;
    });

    it('does not prefetch for a URL that may open the dataset browser', async () => {
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, src: 'http://127.0.0.1:8000' },
      });
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, src: 'https://data.example/datasets/' },
      });
      await new Promise((r) => setTimeout(r, 10));
      expect(fetched).toEqual([]);
    });
  });

  describe('session overrides', () => {
    it('applies the OPFS read concurrency URL override', async () => {
      const original = config.cache.opfsReadConcurrency;
      try {
        await bootstrapStandalone({
          canvas: CANVAS,
          urlParams: { ...EMPTY_PARAMS, opfsReadConcurrency: 16 },
        });
        expect(config.cache.opfsReadConcurrency).toBe(16);
      } finally {
        config.cache.opfsReadConcurrency = original;
      }
    });

    it('installs the ?linePrimitive= and ?lineJoin= overrides before returning', async () => {
      // These installs are load-bearing (both backends bake at material
      // construction) and were previously mutation-survivable: deleting the
      // install call broke nothing at unit level. Assert through the public
      // resolvers, then restore the no-override state for other tests.
      const { resolveLinePrimitive, setLinePrimitiveOverride } =
        await import('../../../types/line-primitive');
      const { resolveLineJoin, setLineJoinOverride } = await import('../../../types/line-join');
      try {
        await bootstrapStandalone({
          canvas: CANVAS,
          urlParams: { ...EMPTY_PARAMS, linePrimitive: 'screen-space', lineJoin: 'none' },
        });
        expect(resolveLinePrimitive()).toBe('screen-space');
        expect(resolveLineJoin()).toBe(0); // 'none'
        await bootstrapStandalone({
          canvas: CANVAS,
          urlParams: { ...EMPTY_PARAMS, linePrimitive: 'capsule' },
        });
        expect(resolveLinePrimitive()).toBe('capsule');
      } finally {
        setLinePrimitiveOverride(null);
        setLineJoinOverride(null);
      }
    });

    it('installs the ?input= override before returning', async () => {
      const { getInputProfile, resetInputProfileForTests } =
        await import('../../../utils/input-capabilities');
      try {
        await bootstrapStandalone({
          canvas: CANVAS,
          urlParams: { ...EMPTY_PARAMS, input: 'touch' },
        });
        expect(getInputProfile()).toMatchObject({
          source: 'override',
          coarsePointer: true,
        });
      } finally {
        resetInputProfileForTests();
      }
    });

    it('detects the input profile when a partial URL-params object omits input', async () => {
      const { getInputProfile, resetInputProfileForTests } =
        await import('../../../utils/input-capabilities');
      try {
        await bootstrapStandalone({ canvas: CANVAS, urlParams: {} as UrlParams });
        expect(getInputProfile().source).toBe('detected');
      } finally {
        resetInputProfileForTests();
      }
    });

    it('names the browser tab from ?title=, and leaves it alone without one', async () => {
      const original = document.title;
      try {
        document.title = original;
        await bootstrapStandalone({ canvas: CANVAS, urlParams: { ...EMPTY_PARAMS } });
        expect(document.title).toBe(original);

        await bootstrapStandalone({
          canvas: CANVAS,
          urlParams: { ...EMPTY_PARAMS, title: 'Rivers of Earth' },
        });
        expect(document.title).toBe('Rivers of Earth');
      } finally {
        document.title = original;
      }
    });

    it('falls back to the ?src= store name when no ?title= is given', async () => {
      // The state a switched-then-reloaded tab (or a link shared from one)
      // comes back in: `buildDataSourceBrowserUrl` dropped the stale ?title=,
      // so the store name in ?src= is all that is left to name the tab.
      const original = document.title;
      try {
        await bootstrapStandalone({
          canvas: CANVAS,
          urlParams: {
            ...EMPTY_PARAMS,
            src: 'http://127.0.0.1:8000/global_rivers.luxar.zarr',
          },
        });
        expect(document.title).toBe('global_rivers');

        // Probe the restore target, then check that a bare data-server root
        // (whose last path segment is a host:port) names no store and hands
        // the page title back rather than inventing a nonsense name.
        setDocumentTitle('probe');
        setDocumentTitle(null);
        const pageTitle = document.title;
        setDocumentTitle('Stale Scene');

        await bootstrapStandalone({
          canvas: CANVAS,
          urlParams: { ...EMPTY_PARAMS, src: 'http://127.0.0.1:8000' },
        });
        expect(document.title).toBe(pageTitle);
      } finally {
        document.title = original;
      }
    });
  });

  describe('opt-in flags', () => {
    it('captures the build identity in the patched console buffer', async () => {
      const originalLog = console.log;
      const bufferedMessages: unknown[][] = [];
      mocks.patch.mockImplementationOnce(() => {
        console.log = (...args: unknown[]) => bufferedMessages.push(args);
      });

      try {
        await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });
      } finally {
        console.log = originalLog;
      }

      expect(
        bufferedMessages.some((args) =>
          args.some(
            (value) =>
              typeof value === 'string' && value.includes(`Luxar viewer ${buildInfoLine()}`)
          )
        )
      ).toBe(true);
    });

    it('patches console, validates config, and warms codecs when defaults apply', async () => {
      await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });

      expect(window.__luxarBuild).toEqual(buildInfo());
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

  describe('notifier backend', () => {
    it('disables auto-dismiss only for persistent errors', async () => {
      await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });

      notifier.error('archive unavailable', { persistent: true });
      notifier.error('ordinary failure');

      expect(mocks.showError).toHaveBeenNthCalledWith(
        1,
        'archive unavailable',
        expect.any(Function),
        {
          datasetBrowser: 'dataset-browser.toggle',
          help: 'help.toggle',
        },
        { autoDismiss: false }
      );
      expect(mocks.showError).toHaveBeenNthCalledWith(
        2,
        'ordinary failure',
        expect.any(Function),
        {
          datasetBrowser: 'dataset-browser.toggle',
          help: 'help.toggle',
        },
        undefined
      );
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
      // Derived, not a literal: the value is `VIEWER_VERSION`, which
      // `version.test.ts` proves equals package.json. Pinning a literal here
      // is what let a hardcoded '1.0.0' survive in the shipped viewer for
      // the whole project.
      expect(window.__luxarDebug?.version).toBe(VIEWER_VERSION);
      expect(window.__luxarDebug?.app).toBeDefined();
      expect(window.__luxarDebug?.consoleInterceptor).toBeDefined();
      // showError is exposed so visual-regression specs can drive the
      // error dialog directly (see bootstrap.ts:195-203 comment).
      expect(typeof window.__luxarDebug?.showError).toBe('function');
      window.__luxarDebug?.showError?.('Debug failure');
      const [, resolveShortcut, shortcutActions] = mocks.showError.mock.calls[0];
      expect(shortcutActions).toEqual({
        datasetBrowser: 'dataset-browser.toggle',
        help: 'help.toggle',
      });
      expect(resolveShortcut(shortcutActions.help)).toBe('F1');
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

    it('installs the debug perf instruments before the first load starts', async () => {
      // A probe reading getPerf() during the first load (the case the perf
      // gates measure) must see the getObjectByName counter, the lazy-LOD
      // stage timings and the renderer.info sampler — not only after it.
      let releaseInit!: () => void;
      mocks.init.mockImplementationOnce(
        () => new Promise<void>((resolve) => (releaseInit = resolve))
      );
      setLodLoadStatsEnabled(false);
      uninstallRendererInfoSampler();
      const booting = bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, debug: true },
      });
      await vi.waitFor(() => expect(mocks.init).toHaveBeenCalled());
      try {
        expect(lodLoadStatsEnabled()).toBe(true);
        const before = perfCounters.get('scene.getObjectByName');
        new THREE.Group().getObjectByName('anything');
        expect(perfCounters.get('scene.getObjectByName')).toBe(before + 1);
        eventBus.emit('frame-start', {});
        eventBus.emit('frame-end', { rendered: true });
        expect(getRendererInfoSnapshot()).toMatchObject({ calls: 7, triangles: 70 });
      } finally {
        releaseInit();
        await booting;
        uninstallRendererInfoSampler();
      }
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
    it('restores a new #view on same-document navigation and removes the listener on dispose', async () => {
      const app = await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });
      const bookmark = {
        version: 1,
        src: '/sample.zarr',
        snapshot: {
          version: 1,
          camera: {
            position: [1, 2, 3],
            target: [0, 0, 0],
            up: [0, 1, 0],
            isOrtho: false,
            fov: 45,
            near: 0.1,
            far: 100,
          },
        },
        rendering: { exposure: 1.5 },
        layers: [],
      };
      window.history.replaceState(
        null,
        '',
        `#view=${encodeURIComponent(JSON.stringify(bookmark))}`
      );
      window.dispatchEvent(new HashChangeEvent('hashchange'));
      await vi.waitFor(() => expect(mocks.restoreSnapshot).toHaveBeenCalledWith(bookmark.snapshot));
      expect(mocks.setRenderingSettings).toHaveBeenCalledWith(bookmark.rendering);
      expect(mocks.flyTo).toHaveBeenCalledWith(bookmark.snapshot.camera, { durationMs: 0 });

      mocks.restoreSnapshot.mockClear();
      const nextBookmark = {
        ...bookmark,
        snapshot: {
          ...bookmark.snapshot,
          camera: { ...bookmark.snapshot.camera, position: [4, 5, 6] },
        },
      };
      window.history.replaceState(
        null,
        '',
        `#view=${encodeURIComponent(JSON.stringify(nextBookmark))}`
      );
      window.dispatchEvent(new HashChangeEvent('hashchange'));
      await vi.waitFor(() =>
        expect(mocks.restoreSnapshot).toHaveBeenCalledWith(nextBookmark.snapshot)
      );

      mocks.restoreSnapshot.mockClear();
      window.history.replaceState(null, '', '#section');
      window.dispatchEvent(new HashChangeEvent('hashchange'));
      await Promise.resolve();
      expect(mocks.restoreSnapshot).not.toHaveBeenCalled();

      app.dispose();
      window.history.replaceState(
        null,
        '',
        `#view=${encodeURIComponent(JSON.stringify(bookmark))}`
      );
      window.dispatchEvent(new HashChangeEvent('hashchange'));
      await Promise.resolve();
      expect(mocks.restoreSnapshot).not.toHaveBeenCalled();
      window.history.replaceState(null, '', window.location.pathname);
    });
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
          renderAlways: true,
          renderAudit: true,
        },
      });
      const arg = mocks.init.mock.calls.at(-1)?.[0];
      expect(arg.renderAlways).toBe(true);
      expect(arg.renderAudit).toBe(true);
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

    it('leaves mobile blend-warm-up resolution to the shared init pipeline', async () => {
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, input: 'touch' },
      });
      expect(mocks.init.mock.calls.at(-1)?.[0].blendWarmup).toBe(true);
      // Bootstrap only threads the URL option. The shared pipeline applies the
      // mobile clamp for both standalone and direct LuxarApp construction.
      await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });
      expect(mocks.init.mock.calls.at(-1)?.[0].blendWarmup).toBe(true);
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

    it('threads the LOD/rendering URL controls into init()', async () => {
      // An embedder-supplied urlParams object must control these flags —
      // the init pipeline reads options, never window.location.
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: {
          ...EMPTY_PARAMS,
          lodFade: false,
          lodEnergyComp: false,
          blendWarmup: false,
          depthSort: false,
          lodFinest: true, // opt-IN flag — flipped the other way
          lodBias: 4,
        },
      });
      const flipped = mocks.init.mock.calls.at(-1)?.[0];
      expect(flipped.lodFade).toBe(false);
      expect(flipped.lodEnergyComp).toBe(false);
      expect(flipped.blendWarmup).toBe(false);
      expect(flipped.depthSort).toBe(false);
      expect(flipped.lodFinest).toBe(true);
      expect(flipped.lodBias).toBe(4);

      await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });
      const defaults = mocks.init.mock.calls.at(-1)?.[0];
      expect(defaults.lodFade).toBe(true);
      expect(defaults.lodEnergyComp).toBe(true);
      expect(defaults.blendWarmup).toBe(true);
      expect(defaults.depthSort).toBe(true);
      expect(defaults.lodFinest).toBe(false);
      expect(defaults.lodBias).toBeUndefined();
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

      expect(window.__luxarBuild).toEqual(buildInfo());
      expect(mocks.showError).toHaveBeenCalledTimes(1);
      expect(mocks.showError.mock.calls[0][0]).toMatch(/Failed to start the application/i);
      const [, resolveShortcut, shortcutActions] = mocks.showError.mock.calls[0];
      expect(shortcutActions).toEqual({
        datasetBrowser: 'dataset-browser.toggle',
        help: 'help.toggle',
      });
      expect(resolveShortcut(shortcutActions.help)).toBe('F1');
      expect(mocks.showError.mock.calls[0][3]).toEqual({ autoDismiss: false });
    });

    it('surfaces a refused format version verbatim in the persistent startup dialog', async () => {
      // The refusal message names the version, the supported set and the
      // remedy — the one startup failure (besides an archive fault) where the
      // generic "check the console" text would hide exactly what the user
      // needs. Pinned by format-version.spec.ts end to end.
      const refusal = new UnsupportedFormatVersionError(
        "Unsupported format_version: '9.9' for a scene store (newer major version; this viewer reads 0.1, 0.2). Upgrade the viewer to a build that supports this format."
      );
      mocks.init.mockRejectedValueOnce(new Error('Failed to load scene', { cause: refusal }));

      await expect(
        bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS })
      ).rejects.toThrow('Failed to load scene');

      expect(mocks.showError).toHaveBeenCalledTimes(1);
      expect(mocks.showError.mock.calls[0][0]).toBe(refusal.message);
      expect(mocks.showError.mock.calls[0][3]).toEqual({ autoDismiss: false });
    });

    it('surfaces authored archive fault remedies in the persistent startup dialog', async () => {
      const initError = new ArchiveFaultError(
        'The archive was not found. Check the `?src=` path.',
        'https://example.test/missing.zarr.zip'
      );
      mocks.init.mockRejectedValueOnce(initError);

      await expect(bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS })).rejects.toBe(
        initError
      );

      expect(mocks.showError).toHaveBeenCalledWith(
        initError.message,
        expect.any(Function),
        {
          datasetBrowser: 'dataset-browser.toggle',
          help: 'help.toggle',
        },
        { autoDismiss: false }
      );
    });

    it('surfaces authored archive faults through wrapper causes', async () => {
      const fault = new ArchiveFaultError(
        'The archive became unreadable. Retry from a stable host.',
        'https://example.test/data.zarr.zip'
      );
      const initError = new Error('scene loading failed', { cause: fault });
      mocks.init.mockRejectedValueOnce(initError);

      await expect(bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS })).rejects.toBe(
        initError
      );

      expect(mocks.showError).toHaveBeenCalledWith(
        fault.message,
        expect.any(Function),
        expect.any(Object),
        { autoDismiss: false }
      );
    });
  });

  describe('user-settings precedence (URL param > stored setting > default)', () => {
    const initOptions = () => mocks.init.mock.calls[0][0];

    it('threads stored cache preferences into loaderConfig when no URL params are set', async () => {
      const s = defaultUserSettings();
      s.caching.enabled = false;
      s.caching.prefetch = false;
      s.caching.budgetMode = 'custom';
      s.caching.budgetMB = 512;
      saveUserSettings(s);

      await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });

      const lc = initOptions().loaderConfig;
      expect(lc.noCache).toBe(true);
      expect(lc.noPrefetch).toBe(true);
      expect(lc.noSliceCache).toBe(false); // untouched preference stays default
      expect(lc.noOpfs).toBe(false); // param-only flag defaults off
      expect(lc.cacheBudgetMB).toBe(512);
      expect(getGpuByteBudget()).toBe(512_000_000);
    });

    it('URL cacheBudgetMB wins over a stored custom budget', async () => {
      const s = defaultUserSettings();
      s.caching.budgetMode = 'custom';
      s.caching.budgetMB = 512;
      saveUserSettings(s);

      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, cacheBudgetMB: 256 },
      });

      expect(initOptions().loaderConfig.cacheBudgetMB).toBe(256);
    });

    it('auto budget mode passes null (heap-aware path) when no URL override', async () => {
      await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });
      expect(initOptions().loaderConfig.cacheBudgetMB).toBeNull();
    });

    it('stored renderer preference applies, but the URL param beats it', async () => {
      const s = defaultUserSettings();
      s.advanced.renderer = 'webgpu';
      saveUserSettings(s);

      await bootstrapStandalone({ canvas: CANVAS, urlParams: EMPTY_PARAMS });
      expect(initOptions().renderer).toBe('webgpu');

      vi.clearAllMocks();
      mocks.init.mockResolvedValue(undefined);
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, renderer: 'webgl' },
      });
      expect(initOptions().renderer).toBe('webgl');
    });

    it('URL disable flags compose with stored preferences via OR', async () => {
      // Stored prefs enable everything; ?noSliceCache still disables S-cache.
      saveUserSettings(defaultUserSettings());
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, noSliceCache: true },
      });
      expect(initOptions().loaderConfig.noSliceCache).toBe(true);
    });

    it('?noOpfs passes straight through to the loader config (param-only, no setting)', async () => {
      await bootstrapStandalone({
        canvas: CANVAS,
        urlParams: { ...EMPTY_PARAMS, noOpfs: true },
      });
      expect(initOptions().loaderConfig.noOpfs).toBe(true);
    });
  });
});
