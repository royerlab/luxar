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

import { LuxarApp } from './app';
import { config } from '../config';
import { validateAndLog } from '../config/validation';
import { showError } from '../ui/helpers';
import { ThemeManager } from '../themes/theme-manager';

// Validate configuration at startup
const configValid = validateAndLog(config);
if (!configValid) {
  log.error(Modules.MAIN, 'Application starting with invalid configuration - errors may occur');
}

// Initialize theme system early (before any UI components are created)
const themeManager = ThemeManager.getInstance();

// Parse URL parameters
const params = new URLSearchParams(window.location.search);

// Support ?theme=light URL parameter
const themeParam = params.get('theme');
if (themeParam) {
  try {
    themeManager.setTheme(themeParam);
    log.custom(LogEmoji.START, Modules.LUXAR, `Theme set from URL: ${themeParam}`);
  } catch {
    log.warning(Modules.LUXAR, `Invalid theme in URL: ${themeParam}, using default`);
  }
} else {
  log.custom(
    LogEmoji.START,
    Modules.LUXAR,
    `Theme system initialized: ${themeManager.getCurrentTheme().name}`
  );
}

// Parse scene source parameter
const src = params.get('src') ?? config.defaultZarrPath;

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
      camera?: THREE.PerspectiveCamera;
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
const isDebugMode = params.has('debug') || localStorage.getItem('luxar_debug') === 'true';
if (isDebugMode) {
  window.__luxarDebug = {
    app,
    consoleInterceptor,
    version: '1.0.0',
  };
  log.custom(LogEmoji.CONSOLE, Modules.LUXAR, 'Debug interface available at window.__luxarDebug');
}

app.init(src).catch((error) => {
  log.error(Modules.LUXAR, 'Failed to start Luxar application:', error);

  // Show error to user if it wasn't already handled by lower-level error handlers
  // This ensures any initialization errors that don't get displayed are still shown
  showError('Failed to start the application. Please check the console for details.');
});
