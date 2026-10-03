/**
 * Plug the viewer's concrete UI helpers into the cross-layer notifier.
 *
 * Lower layers (data, scene, input) call `notifier.toast` / `.error` /
 * `.showHelp` without importing the `ui/` helper modules directly — that is
 * what keeps the dependency-cruiser layer order clean. `LuxarApp.init()`
 * installs this backend for every lifetime (standalone and library embeds
 * alike); the dispose pipeline clears it.
 *
 * @module core/app/lifecycle/notifier-backend
 */

import { clearError } from '../../../ui/error-overlay';
import { showToast } from '../../../ui/toast';
import { showHelpOverlay, hideHelpOverlay } from '../../../ui/help-overlay';
import { showLoadingIndicator, hideLoadingIndicator } from '../../../ui/loading-indicator';
import {
  showSceneIdentityBanner,
  hideSceneIdentityBanner,
} from '../../../ui/scene-identity-banner';
import { setNotifierBackend } from '../../../utils/cross-layer/notifier';
import { showViewerError } from '../error-dialog';

/**
 * Register the viewer UI as the notifier backend. `shortcutForAction` resolves
 * the error dialog's recovery hints through the app's live input bindings.
 */
export function installNotifierBackend(
  shortcutForAction: (actionId: string) => string | undefined
): void {
  setNotifierBackend({
    showError: (message, options) =>
      showViewerError(
        message,
        shortcutForAction,
        options?.persistent ? { autoDismiss: false } : undefined
      ),
    showToast,
    showHelpOverlay,
    hideHelpOverlay,
    showLoadingIndicator,
    hideLoadingIndicator,
    clearError,
    showSceneIdentityBanner,
    hideSceneIdentityBanner,
  });
}
