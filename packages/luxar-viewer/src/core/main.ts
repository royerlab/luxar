// Luxar — standalone-app entry point.
// Copyright (c) 2024 The Luxar Authors
//
// This file is intentionally tiny. It does the two things only the
// standalone HTML page needs to know about — load the global CSS and find
// the canvas — and then hands off to bootstrapStandalone() for everything
// else (theme init, console patching, codec warming, URL parsing, debug
// surface, LuxarApp construction). Embedded callers skip this file
// entirely and construct LuxarApp from src/index.ts directly.

// Standalone-only chrome (html/body sizing, CSS reset, scrollbars) AND the
// library's component/utility/theme styles. Embedders never import
// standalone.css — only `@luxar/viewer/styles.css` (= index.css).
import '../styles/standalone.css';
import '../styles/index.css';
import { bootstrapStandalone } from './bootstrap';
import { showError } from '../ui/error-overlay';

const canvas = document.getElementById('app') as HTMLCanvasElement | null;
if (!canvas) {
  showError("Canvas element with id 'app' not found in the page.");
  throw new Error("Required canvas element 'app' not found");
}

void bootstrapStandalone({ canvas }).catch(() => {
  // bootstrapStandalone has already logged the error and shown the user a
  // top-level error UI. Swallow the rejection here so it doesn't bubble
  // into an unhandled-promise warning.
});
