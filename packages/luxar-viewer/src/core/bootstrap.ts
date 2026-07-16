/**
 * Bootstrap helpers for the standalone Luxar viewer entry point.
 *
 * Centralizes the standalone pre-init sequence: theme initialization,
 * console-interceptor patching, blosc codec warming, config validation,
 * URL parameter parsing, canvas resolution, and debug interface
 * attachment. This keeps:
 *
 * - `main.ts` small and easy to read at a glance.
 * - Embedded callers that want the same "full standalone" behavior can
 *   call this function directly (most won't — they construct LuxarApp
 *   themselves and skip the standalone-only steps like codec warming
 *   and console patching).
 *
 * Each side effect is opt-in via a flag so embedders can pick exactly
 * the parts they want.
 */

import { LuxarApp, type LuxarAppOptions } from './app';
import { config } from '../config';
import { validateAndLog } from '../config/validation';
import { readUrlParams, type UrlParams } from '../config/url-params';
import { initUserSettings } from '../config/user-settings';
import { initGpuByteBudget } from '../rendering/gpu-byte-budget';
import { StorageKeys } from '../utils/storage-keys';
import { showError, clearError } from '../ui/error-overlay';
import { showToast } from '../ui/toast';
import { showHelpOverlay, hideHelpOverlay } from '../ui/help-overlay';
import { showLoadingIndicator, hideLoadingIndicator } from '../ui/loading-indicator';
import { setNotifierBackend } from '../utils/cross-layer/notifier';
import { ThemeManager } from '../themes/theme-manager';
import { consoleInterceptor } from '../utils/console-interceptor';
import { log, Modules, LogEmoji } from '../utils/log';
import { codecRegistry } from '../data/zarr';

/**
 * Options for {@link bootstrapStandalone}.
 *
 * The standalone-app entry point (`main.ts`) passes nothing beyond `canvas`,
 * so every flag defaults to the standalone viewer behavior.
 *
 * Embedded callers typically construct LuxarApp directly and skip this
 * function entirely. If they DO call it, they usually want
 * `patchConsole: false` and `warmCodecs: false` to avoid touching shared
 * host-page state.
 */
export interface BootstrapOptions {
  /** Canvas element to render into. */
  canvas: HTMLCanvasElement;

  /**
   * Parsed URL parameters. Defaults to `readUrlParams()` so the standalone
   * app picks up `?src`, `?theme`, `?debug`, etc. from `window.location`.
   * Embedders pass their own (or `{...}`) to bypass URL parsing.
   */
  urlParams?: UrlParams;

  /**
   * Monkey-patch `console.log/warn/error/info/debug` so the debug console
   * UI can replay messages later. Defaults to true. **Embedders should set
   * this to false** unless they own the host page's console output.
   */
  patchConsole?: boolean;

  /**
   * Warm the blosc codec module cache. Defaults to true. Saves ~771ms on
   * the first compressed-chunk decompress for the standalone app; embed
   * use cases rarely need it (the codec loads on demand anyway).
   */
  warmCodecs?: boolean;

  /**
   * Run `validateAndLog(config)` at startup. Defaults to true. Off by
   * default for embedders since the (immutable) config has already been
   * validated at compile time.
   */
  validateConfig?: boolean;
}

/**
 * Run the standalone-app boot sequence and return an initialized LuxarApp.
 *
 * Steps (each gated by an option):
 * 1. (optional) Patch console for the debug-console UI.
 * 2. (optional) Validate config; log a warning if invalid.
 * 3. (optional) Warm the blosc codec module cache.
 * 4. Initialize the theme manager (applies theme from `urlParams.theme`,
 *    `localStorage[luxar.theme]`, or the built-in default).
 * 5. Compute LuxarAppOptions from the URL params.
 * 6. Construct LuxarApp, attach the debug surface (if `?debug`), and call
 *    `init()` — surfacing errors via `showError()` if init throws.
 *
 * Preserves the standalone-app semantics. Tests live in
 * `tests/unit/core/bootstrap.test.ts` and cover each opt-in flag, the
 * theme/URL/localStorage-driven debug-mode resolution, and the error path.
 */
export async function bootstrapStandalone(opts: BootstrapOptions): Promise<LuxarApp> {
  const urlParams = opts.urlParams ?? readUrlParams();
  const patchConsole = opts.patchConsole ?? true;
  const warmCodecs = opts.warmCodecs ?? true;
  const validateConfig = opts.validateConfig ?? true;

  // Load + apply the persisted global viewer preferences (Settings popover)
  // BEFORE the first config read below: live-read values are applied by
  // mutating `config`; startup-only values are threaded into appOptions
  // further down with the precedence URL param > stored setting > default.
  // Standalone-only — library embedders configure via LuxarAppOptions.
  const userSettings = initUserSettings();

  // Size the single GPU-geometry byte budget before any pool / LOD
  // registry is constructed. Precedence: `?gpuBudgetMB=` URL param >
  // `config.gpuPoolMaxBytes` (null=auto, 0=disable, N=pin) > auto-size.
  initGpuByteBudget(
    urlParams.gpuBudgetMB != null
      ? urlParams.gpuBudgetMB * 1_000_000
      : config.dataLoading.performance.gpuPoolMaxBytes
  );

  if (patchConsole) {
    consoleInterceptor.patch();
  }

  // Wire the cross-layer notifier surface to the concrete UI helpers.
  // Lower layers (data, scene, input) call notifier.toast / .error /
  // .showHelp etc. without importing the ui/ helper modules directly —
  // that's what keeps the dependency-cruiser layer order clean.
  setNotifierBackend({
    showError,
    showToast,
    showHelpOverlay,
    hideHelpOverlay,
    showLoadingIndicator,
    hideLoadingIndicator,
    clearError,
  });

  if (validateConfig) {
    const ok = validateAndLog(config);
    if (!ok) {
      log.error(Modules.MAIN, 'Application starting with invalid configuration - errors may occur');
    }
  }

  if (warmCodecs) {
    // Trigger the registry thunk so the browser fetches/parses the 601KB
    // blosc WASM in parallel with the rest of init.
    codecRegistry
      .get('blosc')?.()
      ?.catch(() => {
        /* Best-effort warmup; failure is non-fatal — zarrita will retry on demand. */
      });
  }

  // Initialize the theme system early so loading-state UI is themed.
  // [core OOS] Pre-fix, the inner try/catch only wrapped `setTheme(...)`.
  // The else-branch's `themeManager.getCurrentTheme().name` could itself
  // throw (corrupt localStorage hands a malformed theme name to
  // theme-manager) and the throw would escape bootstrap — leaving the
  // entire viewer un-initialized. Wrap the whole theme block so any
  // unexpected throw downgrades to a warning + default-theme path
  // instead of taking the page down.
  const themeManager = ThemeManager.getInstance();
  try {
    if (urlParams.theme) {
      try {
        themeManager.setTheme(urlParams.theme);
        log.custom(LogEmoji.START, Modules.LUXAR, `Theme set from URL: ${urlParams.theme}`);
      } catch {
        log.warning(Modules.LUXAR, `Invalid theme in URL: ${urlParams.theme}, using default`);
      }
    } else {
      log.custom(
        LogEmoji.START,
        Modules.LUXAR,
        `Theme system initialized: ${themeManager.getCurrentTheme().name}`
      );
    }
  } catch (error) {
    // getCurrentTheme() — or any other theme-manager call above —
    // threw unexpectedly. Realistic causes: corrupt localStorage,
    // observer-callback exception, registerTheme race. Surface via
    // log.warning so the gap is observable but proceed with bootstrap.
    const message = error instanceof Error ? error.message : String(error);
    log.warning(
      Modules.LUXAR,
      `Theme initialization failed unexpectedly: ${message}. Proceeding with default theme.`
    );
  }

  // Read the persisted debug flag defensively. A host page running in
  // private-mode-strict or a sandboxed iframe without storage access may
  // throw on getItem(); fall back to the URL flag alone in that case.
  let storedDebug: string | null = null;
  try {
    storedDebug = localStorage.getItem(StorageKeys.debug);
  } catch {
    /* Storage disabled — debug mode then comes only from `?debug`. */
  }
  const isDebugMode = urlParams.debug || storedDebug === 'true';

  const appOptions: LuxarAppOptions = {
    canvas: opts.canvas,
    src: urlParams.src ?? config.defaultZarrPath,
    debug: isDebugMode,
    updateBrowserUrl: true,
    openCacheStats: urlParams.cacheStats,
    loaderConfig: {
      // URL boolean flags are one-way DISABLE switches, so they compose with
      // the stored preferences via `||` (either source can disable, neither
      // can force-enable past the other). Value-typed params use `??`.
      noCache: urlParams.noCache || !userSettings.caching.enabled,
      noSliceCache: urlParams.noSliceCache || !userSettings.caching.sliceCache,
      cacheDebug: urlParams.cacheDebug,
      clearCache: urlParams.clearCache,
      noPrefetch: urlParams.noPrefetch || !userSettings.caching.prefetch,
      prefetchDebug: urlParams.prefetchDebug,
      cacheBudgetMB:
        urlParams.cacheBudgetMB ??
        (userSettings.caching.budgetMode === 'custom' ? userSettings.caching.budgetMB : null),
    },
    // `?renderer=webgl|webgpu` forces a backend regardless of the
    // build-time env, then the stored Settings preference, then undefined →
    // SceneManager falls back to the env var, then the WebGL default. See
    // `setupRenderer` for the precedence chain.
    renderer:
      urlParams.renderer ??
      (userSettings.advanced.renderer !== 'auto' ? userSettings.advanced.renderer : undefined),
    webgpuForceWebGL: urlParams.webgpuForceWebGL,
    perfTimestamp: urlParams.perfTimestamp,
    // `?dpr=<value>` pins a fixed pixel ratio for deterministic
    // E2E/visual runs; undefined → normal adaptive-DPR behavior.
    pinnedDPR: urlParams.dpr ?? undefined,
    // On-by-default rendering feature flags (`?no-lod-fade`,
    // `?no-lod-energy`, `?depthSort=0` disable). Threaded as options so
    // an embedder-supplied `urlParams` object controls them too — the
    // init pipeline reads options, never window.location.
    lodFade: urlParams.lodFade,
    lodEnergyComp: urlParams.lodEnergyComp,
    depthSort: urlParams.depthSort,
    // Opt-in capture-quality override (`?lod-finest` — the gallery harness).
    lodFinest: urlParams.lodFinest,
  };

  const app = new LuxarApp();

  if (isDebugMode) {
    // Seed the debug surface before init() so consumers (e.g. Playwright)
    // that hook into `window.__luxarDebug` can rely on `.app` being there
    // even while init() is still in flight. LuxarApp.setupDebugInterface()
    // extends this object with runtime references after init completes.
    //
    // PARTIAL-STATE CONTRACT — IMPORTANT:
    //   `__luxarDebug.app` exists from construction (just after `new
    //   LuxarApp()`), but the app's per-subsystem fields
    //   (sceneManager, inputHandler, recordingPanel, layersPanel,
    //   renderingControls, debugConsole, …) are populated only after
    //   `app.init(...)` resolves below. Consumers MUST wait until
    //   either `app.initialized === true` or the post-init debug
    //   surface keys (`scene`, `camera`, `renderer`, `controls`,
    //   `getState`, …) appear before dereferencing component fields,
    //   otherwise they will read `undefined` and crash overlays.
    //   Playwright fixtures rely on the post-init keys for exactly
    //   this reason.
    //
    // `showError` is exposed so tests (visual-regression in particular) can
    // drive the error-dialog component directly without depending on the
    // URL-routing semantics in shouldShowBrowser, which evolve independently
    // of the dialog's appearance.
    window.__luxarDebug = {
      app,
      consoleInterceptor,
      version: '1.0.0',
      showError,
    };
    log.custom(LogEmoji.CONSOLE, Modules.LUXAR, 'Debug interface available at window.__luxarDebug');
  }

  try {
    await app.init(appOptions);
  } catch (error) {
    log.error(Modules.LUXAR, 'Failed to start Luxar application:', error);
    showError('Failed to start the application. Please check the console for details.');
    throw error;
  }

  return app;
}
