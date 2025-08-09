// Luxar - A web viewer for arbitrarily large 3D scenes
// Copyright (c) 2024 The Luxar Authors

// CRITICAL: Import console interceptor FIRST before any other code
// This ensures we capture ALL console output from the very beginning
import { consoleInterceptor } from '../utils/console-interceptor';

// Log that we're starting (this will be captured)
console.log('🚀 [Luxar] Application starting...');

import { LuxarApp } from './app';
import { config } from '../config';
import { showError } from '../ui/helpers';

// Parse URL parameters for scene source
const params = new URLSearchParams(window.location.search);
const src = params.get('src') ?? config.defaultZarrPath;

// Initialize and start the application
const app = new LuxarApp();

// Type-safe debug interface (only in development builds)
declare global {
  interface Window {
    __luxarDebug?: {
      app: LuxarApp;
      consoleInterceptor: typeof consoleInterceptor;
      version: string;
    };
  }
}

// Only expose debug interface in development/debug mode
// Check for debug flag in URL or localStorage
const isDebugMode = params.has('debug') || localStorage.getItem('luxar_debug') === 'true';
if (isDebugMode) {
  window.__luxarDebug = {
    app,
    consoleInterceptor,
    version: '1.0.0'
  };
  console.log('🔧 [Luxar] Debug interface available at window.__luxarDebug');
}

app.init(src).catch((error) => {
  console.error('Failed to start Luxar application:', error);

  // Show error to user if it wasn't already handled by lower-level error handlers
  // This ensures any initialization errors that don't get displayed are still shown
  showError('Failed to start the application. Please check the console for details.');
});
