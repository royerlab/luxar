// Luxar - A web viewer for arbitrarily large 3D scenes
// Copyright (c) 2024 The Luxar Authors

// Import CSS styles FIRST (before any JavaScript runs)
import '../styles/index.css';

// CRITICAL: Import console interceptor FIRST before any other code
// This ensures we capture ALL console output from the very beginning
import { consoleInterceptor } from '../utils/console-interceptor';

// Log that we're starting (this will be captured)
import { log, Modules, LogEmoji } from '../utils/log';
log.custom(LogEmoji.START, Modules.LUXAR, 'Application starting...');

// Eagerly warm the blosc codec module cache — zarrita lazy-loads it on first
// compressed chunk access, but every Luxar dataset uses blosc compression.
// Triggering the registry thunk now lets the browser fetch + parse the 601KB
// WASM module in parallel with app initialization, eliminating the ~771ms
// delay on first chunk decompress.
import { registry as _codecRegistry } from 'zarrita';
_codecRegistry
  .get('blosc')?.()
  ?.catch(() => {});

import { LuxarApp } from './app';
import { config } from '../config';
import { validateAndLog } from '../config/validation';
import { readUrlParams } from '../config/url-params';
import { showError } from '../ui/helpers';
import { ThemeManager } from '../themes/theme-manager';
import type { LuxarCamera } from '../scene/camera-utils';

// Validate configuration at startup
const configValid = validateAndLog(config);
if (!configValid) {
  log.error(Modules.MAIN, 'Application starting with invalid configuration - errors may occur');
}

// Single source of truth for URL-derived flags. Components downstream do not
// re-read window.location — values flow through LuxarAppOptions / LoaderConfig.
const urlParams = readUrlParams();

// Initialize theme system early (before any UI components are created)
const themeManager = ThemeManager.getInstance();

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

const src = urlParams.src ?? config.defaultZarrPath;

// Initialize and start the application
const app = new LuxarApp();

// Type-safe debug interface (only in development builds)
// Note: This interface is extended in app.ts after initialization
// to include runtime components (scene, camera, etc.)
declare global {
  interface Window {
    __luxarDebug?: {
      // Base properties (available from main.ts)
      app: LuxarApp;
      consoleInterceptor: typeof consoleInterceptor;
      version: string;

      // Runtime properties (added by app.ts after initialization)
      scene?: THREE.Scene;
      camera?: LuxarCamera;
      renderer?: THREE.WebGLRenderer;
      controls?: any; // ControlsManager not imported here
      postProcessing?: any; // PostProcessingManager not imported here
      animationController?: any;
      inputHandler?: any;
      renderingControls?: any;
      getState?: () => any;
      renderOnce?: () => void;
      getSceneLoader?: () => Promise<any>;
      runtimeReady?: boolean;
    };
  }
}

// Import THREE for type definitions
import * as THREE from 'three';

// Only expose debug interface in development/debug mode
// Check for debug flag in URL or localStorage
const isDebugMode = urlParams.debug || localStorage.getItem('luxar_debug') === 'true';
if (isDebugMode) {
  window.__luxarDebug = {
    app,
    consoleInterceptor,
    version: '1.0.0',
  };
  log.custom(LogEmoji.CONSOLE, Modules.LUXAR, 'Debug interface available at window.__luxarDebug');
}

app
  .init({
    src,
    debug: isDebugMode,
    loaderConfig: {
      noCache: urlParams.noCache,
      cacheDebug: urlParams.cacheDebug,
      clearCache: urlParams.clearCache,
      noPrefetch: urlParams.noPrefetch,
      prefetchDebug: urlParams.prefetchDebug,
    },
  })
  .catch((error) => {
    log.error(Modules.LUXAR, 'Failed to start Luxar application:', error);
    // Show error to user if it wasn't already handled by lower-level error
    // handlers — guarantees init failures are surfaced even when no nested
    // error UI fired.
    showError('Failed to start the application. Please check the console for details.');
  });
