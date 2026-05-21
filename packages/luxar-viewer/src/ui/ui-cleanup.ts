/**
 * UI teardown — removes any lingering helper overlays (loading spinner,
 * error dialog, help panel) created via this folder's helpers. Safe to
 * call multiple times.
 *
 * Used at app shutdown and when resetting UI state.
 */

import { hideHelpOverlay } from './help-overlay';

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
}
