/**
 * UI teardown — removes any lingering helper overlays (loading spinner,
 * error dialog, help panel) created via this folder's helpers. Safe to
 * call multiple times.
 *
 * Used at app shutdown and when resetting UI state.
 */

import { hideHelpOverlay } from './help-overlay';
import { hideSceneIdentityBanner } from './scene-identity-banner';

/**
 * Remove every helper overlay this folder can leave behind: the loading
 * indicator, the error message, the help overlay, and the scene-identity
 * banner. Idempotent — safe to call with none of them present.
 */
export function cleanupUI() {
  // Loading-indicator/spinner styles live in
  // src/styles/components/error-dialog.css and are loaded by Vite — there
  // is no inline <style> element to clean up here.

  const loadingDiv = document.getElementById('luxar-loading-indicator');
  if (loadingDiv) {
    loadingDiv.remove();
  }

  const errorDiv = document.getElementById('luxar-error-message');
  if (errorDiv) {
    errorDiv.remove();
  }

  // hideHelpOverlay handles its own click-listener teardown
  hideHelpOverlay();

  // The scene-identity banner is raised by the watchdog through the
  // cross-layer notifier, and the watchdog only disposes with the
  // SceneLoaderManager — which the dispose pipeline tears down AFTER it has
  // already cleared the notifier backend. Clearing here (before that point)
  // is what actually removes the node and resets the module's shown-kind
  // state; otherwise a standing banner outlives the viewer in an embedder's
  // DOM and the next one would be suppressed as a duplicate.
  hideSceneIdentityBanner();
}
