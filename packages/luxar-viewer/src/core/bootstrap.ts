/**
 * Bootstrap helpers for the standalone Luxar viewer entry point.
 *
 * `main.ts` used to inline a long sequence of pre-init steps (theme
 * initialization, console-interceptor patching, blosc codec warming,
 * config validation, URL parameter parsing, canvas resolution, debug
 * interface attachment). This module extracts that sequence into one
 * function so:
 *
 * - `main.ts` becomes ~10 lines and is easy to read at a glance.
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
import { StorageKeys } from '../utils/storage-keys';
import { showError } from '../ui/helpers';
import { ThemeManager } from '../themes/theme-manager';
import { consoleInterceptor } from '../utils/console-interceptor';
import { log, Modules, LogEmoji } from '../utils/log';
import { registry as codecRegistry } from 'zarrita';

/**
 * Options for {@link bootstrapStandalone}.
 *
 * The standalone-app entry point (`main.ts`) passes nothing beyond `canvas`
 * — every flag defaults to true to preserve historical behavior.
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
 * Always preserves the standalone-app's pre-existing semantics; tests for
 * this function live in tests/unit/core/main.test.ts (TODO).
 */
export async function bootstrapStandalone(opts: BootstrapOptions): Promise<LuxarApp> {
  const urlParams = opts.urlParams ?? readUrlParams();
  const patchConsole = opts.patchConsole ?? true;
  const warmCodecs = opts.warmCodecs ?? true;
  const validateConfig = opts.validateConfig ?? true;

  if (patchConsole) {
    consoleInterceptor.patch();
  }

  if (validateConfig) {
    const ok = validateAndLog(config);
    if (!ok) {
      log.error(
        Modules.MAIN,
        'Application starting with invalid configuration - errors may occur'
      );
    }
  }

  if (warmCodecs) {
    // Trigger the registry thunk so the browser fetches/parses the 601KB
    // blosc WASM in parallel with the rest of init.
    codecRegistry
      .get('blosc')
      ?.()
      ?.catch(() => {
        /* Best-effort warmup; failure is non-fatal — zarrita will retry on demand. */
      });
  }

  // Initialize the theme system early so loading-state UI is themed.
  const themeManager = ThemeManager.getInstance();
  if (urlParams.theme) {
    try {
      themeManager.setTheme(urlParams.theme);
      log.custom(LogEmoji.START, Modules.LUXAR, `Theme set from URL: ${urlParams.theme}`);
    } catch {
      log.warning(
        Modules.LUXAR,
        `Invalid theme in URL: ${urlParams.theme}, using default`
      );
    }
  } else {
    log.custom(
      LogEmoji.START,
      Modules.LUXAR,
      `Theme system initialized: ${themeManager.getCurrentTheme().name}`
    );
  }

  const isDebugMode =
    urlParams.debug || localStorage.getItem(StorageKeys.debug) === 'true';

  const appOptions: LuxarAppOptions = {
    canvas: opts.canvas,
    src: urlParams.src ?? config.defaultZarrPath,
    debug: isDebugMode,
    loaderConfig: {
      noCache: urlParams.noCache,
      cacheDebug: urlParams.cacheDebug,
      clearCache: urlParams.clearCache,
      noPrefetch: urlParams.noPrefetch,
      prefetchDebug: urlParams.prefetchDebug,
    },
  };

  const app = new LuxarApp();

  if (isDebugMode) {
    // Seed the debug surface before init() so consumers (e.g. Playwright)
    // that hook into `window.__luxarDebug` can rely on `.app` being there
    // even while init() is still in flight. LuxarApp.setupDebugInterface()
    // extends this object with runtime references after init completes.
    window.__luxarDebug = {
      app,
      consoleInterceptor,
      version: '1.0.0',
    };
    log.custom(
      LogEmoji.CONSOLE,
      Modules.LUXAR,
      'Debug interface available at window.__luxarDebug'
    );
  }

  try {
    await app.init(appOptions);
  } catch (error) {
    log.error(Modules.LUXAR, 'Failed to start Luxar application:', error);
    showError(
      'Failed to start the application. Please check the console for details.'
    );
    throw error;
  }

  return app;
}
