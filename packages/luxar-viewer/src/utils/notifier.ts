/**
 * Cross-layer notification interface.
 *
 * Lower layers (data, scene, input) need to surface user-visible
 * messages — toasts when a recording finishes, error dialogs when
 * loading fails, the help overlay when the user presses H — but they
 * can't directly import `ui/helpers.ts` without violating the layer
 * order documented in CONVENTIONS.md §10.
 *
 * This module defines the abstract `Notifier` surface and a singleton
 * `notifier` that lower layers call. The UI bootstrap registers a
 * concrete backend at startup via `setNotifierBackend(...)`. Until
 * registered, calls are no-ops with a single warning log so missing
 * registration doesn't crash code paths that legitimately run before
 * UI init (and so unit tests don't need a backend).
 *
 * @module utils/notifier
 */

import { log, Modules } from './log';

/**
 * Methods a notifier backend must implement. Mirrors the existing
 * `ui/helpers.ts` exports so the UI layer can plug them in directly:
 *
 *   setNotifierBackend({
 *     showError, showToast, showHelpOverlay, hideHelpOverlay,
 *     showLoadingIndicator, hideLoadingIndicator,
 *   });
 */
export interface NotifierBackend {
  showError(message: string): void;
  showToast(message: string, durationMs?: number): void;
  showHelpOverlay(): void;
  hideHelpOverlay(): void;
  showLoadingIndicator(): HTMLElement | void;
  hideLoadingIndicator(): void;
  clearError(): void;
}

let backend: NotifierBackend | null = null;
let warnedMissing = false;

function warnIfMissing(method: string): void {
  if (!warnedMissing) {
    log.warning(
      Modules.SCENE_MANAGER,
      `notifier.${method} called before backend registered — message dropped. ` +
        'This is normal during early startup or unit tests; if you see this in ' +
        'production, the UI layer never called setNotifierBackend.'
    );
    warnedMissing = true;
  }
}

/**
 * Public notifier surface. Methods are stable; backends can come and
 * go without changing call sites.
 */
export const notifier = {
  /** Display a user-friendly error dialog with guidance. */
  error(message: string): void {
    if (backend) backend.showError(message);
    else warnIfMissing('error');
  },
  /** Brief toast notification that auto-dismisses. */
  toast(message: string, durationMs = 2000): void {
    if (backend) backend.showToast(message, durationMs);
    else warnIfMissing('toast');
  },
  /** Show the keyboard-shortcuts help overlay. */
  showHelp(): void {
    if (backend) backend.showHelpOverlay();
    else warnIfMissing('showHelp');
  },
  /** Hide the keyboard-shortcuts help overlay. */
  hideHelp(): void {
    if (backend) backend.hideHelpOverlay();
    else warnIfMissing('hideHelp');
  },
  /** Display the loading spinner. */
  showLoading(): void {
    if (backend) backend.showLoadingIndicator();
    else warnIfMissing('showLoading');
  },
  /** Remove the loading spinner. */
  hideLoading(): void {
    if (backend) backend.hideLoadingIndicator();
    else warnIfMissing('hideLoading');
  },
  /** Clear any displayed error dialog. No-op if none is showing. */
  clearError(): void {
    if (backend) backend.clearError();
    else warnIfMissing('clearError');
  },
};

/**
 * Register a notifier backend. Called once by the UI bootstrap with
 * the concrete `ui/helpers.ts` implementations. Subsequent calls
 * replace the backend (useful for tests).
 */
export function setNotifierBackend(b: NotifierBackend): void {
  backend = b;
  warnedMissing = false;
}

/**
 * Tear down the registered backend. Useful for tests that want to
 * verify no notifications are posted, or for app teardown.
 */
export function clearNotifierBackend(): void {
  backend = null;
}
