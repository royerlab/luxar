/**
 * Cross-layer notification interface.
 *
 * Lower layers (data, scene, input) need to surface user-visible
 * messages — toasts when a recording finishes, error dialogs when
 * loading fails, the help overlay when the user presses H — but they
 * can't directly import the `ui/` helper modules (`toast.ts`,
 * `error-overlay.ts`, `help-overlay.ts`, `loading-indicator.ts`)
 * without violating the layer order documented in CONVENTIONS.md §10.
 *
 * This module defines the abstract `Notifier` surface and a singleton
 * `notifier` that lower layers call. The UI bootstrap registers a
 * concrete backend at startup via `setNotifierBackend(...)`. Until
 * registered, calls are no-ops with a single warning log so missing
 * registration doesn't crash code paths that legitimately run before
 * UI init (and so unit tests don't need a backend).
 *
 * @module utils/cross-layer/notifier
 */

import { log, Modules } from '../log';
import type { RegisteredShortcutBindings } from '../../types/shortcut-help';

/**
 * Methods a notifier backend must implement. Mirrors the exports
 * of the `ui/` helper modules so the UI layer can plug them in directly:
 *
 *   setNotifierBackend({
 *     showError, showToast, showHelpOverlay, hideHelpOverlay,
 *     showLoadingIndicator, hideLoadingIndicator,
 *   });
 */
export interface NotifierBackend {
  showError(message: string): void;
  showToast(message: string, durationMs?: number): void;
  showHelpOverlay(bindings?: RegisteredShortcutBindings): void;
  hideHelpOverlay(): void;
  showLoadingIndicator(): HTMLElement | void;
  hideLoadingIndicator(): void;
  clearError(): void;
  /**
   * Persistent scene-identity banner (see `ui/scene-identity-banner.ts`).
   * Optional so existing minimal backends (tests, embedders) stay valid;
   * calls then degrade to a silent no-op. `show` additionally warns when NO
   * backend at all is registered (a user-visible message was dropped);
   * `hide` never does — it only ever runs during teardown.
   */
  showSceneIdentityBanner?(kind: 'changed' | 'unreachable'): void;
  hideSceneIdentityBanner?(onlyKind?: 'changed' | 'unreachable'): void;
}

// MED-42 (audit-ack): module-level singleton state is intentional, not
// a code smell. The cross-layer notifier exists precisely BECAUSE
// lower layers cannot import the concrete `ui/` helpers (CONVENTIONS.md
// §10 layer order). Wrapping in a class with DI would require every
// data/scene/input caller to thread a `Notifier` instance through its
// constructor — defeating the point of a layer-bypass surface.
// `setNotifierBackend` / `clearNotifierBackend` give tests full
// control over the lifetime, and embedders that need per-app notifiers
// can instantiate the concrete UI helpers directly.
let backend: NotifierBackend | null = null;
let warnedMissing = false;

function warnIfMissing(method: string): void {
  if (!warnedMissing) {
    log.warning(
      Modules.NOTIFIER,
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
  showHelp(bindings?: RegisteredShortcutBindings): void {
    if (backend) backend.showHelpOverlay(bindings);
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
  /** Show the persistent scene-identity banner (changed / unreachable). */
  showSceneIdentityBanner(kind: 'changed' | 'unreachable'): void {
    // A registered backend WITHOUT the optional method is a deliberate
    // minimal backend (tests, embedders) — silent no-op, not a warning.
    if (backend) backend.showSceneIdentityBanner?.(kind);
    else warnIfMissing('showSceneIdentityBanner');
  },
  /**
   * Hide the scene-identity banner (optionally only a specific kind).
   *
   * Never warns about a missing backend: this is teardown, not a dropped
   * message. The app dispose pipeline clears the backend before it destroys
   * the SceneLoaderManager, so the watchdog's own dispose necessarily lands
   * after — and "there is no UI left to clear" is the expected state there,
   * not the misconfiguration `warnIfMissing` reports.
   */
  hideSceneIdentityBanner(onlyKind?: 'changed' | 'unreachable'): void {
    backend?.hideSceneIdentityBanner?.(onlyKind);
  },
};

/**
 * Register a notifier backend. Called once by the UI bootstrap with
 * the concrete implementations from the `ui/` helper modules. Subsequent calls
 * replace the backend (useful for tests).
 */
export function setNotifierBackend(b: NotifierBackend): void {
  backend = b;
  warnedMissing = false;
}

/**
 * Tear down the registered backend. Useful for tests that want to
 * verify no notifications are posted, or for app teardown. Also
 * resets the once-only warning flag so a subsequent missing-backend
 * call emits a fresh warning (otherwise tests that rely on the warn
 * being observable would only see it on the very first run).
 */
export function clearNotifierBackend(): void {
  backend = null;
  warnedMissing = false;
}
