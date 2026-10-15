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
  getSceneIdentityBannerKind,
} from '../../../ui/scene-identity-banner';
import { setNotifierBackend } from '../../../utils/cross-layer/notifier';
import { showViewerError } from '../error-dialog';

/**
 * Register the viewer UI as the notifier backend. `shortcutForAction` resolves
 * the error dialog's recovery hints through the app's live input bindings.
 *
 * `onDataServerReachability` is told, after every scene-identity banner
 * call, whether the data server is reachable — `false` exactly while the
 * `unreachable` banner stands AND the latest failed probe was a lost
 * connection (`detail.connectionLost`). An HTTP status such as an expired
 * presigned credential, a rate limit or a 5xx shows the same banner but is not
 * "the server is gone". A `changed` verdict means something answered, and
 * hiding the banner — recovery, dataset switch — means reachable again.
 * Called on every banner call, so the receiver must be idempotent.
 */
export function installNotifierBackend(
  shortcutForAction: (actionId: string) => string | undefined,
  onDataServerReachability?: (reachable: boolean) => void
): void {
  // Whether the latest `unreachable` report was a lost connection; an HTTP
  // status (an expired presigned credential, a rate limit, a 5xx) keeps the
  // banner but is not "the server is gone".
  let connectionLost = false;
  const reportReachability = (): void =>
    onDataServerReachability?.(!(getSceneIdentityBannerKind() === 'unreachable' && connectionLost));
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
    showSceneIdentityBanner: (kind, detail) => {
      if (kind === 'unreachable') connectionLost = detail?.connectionLost === true;
      showSceneIdentityBanner(kind);
      reportReachability();
    },
    hideSceneIdentityBanner: (onlyKind) => {
      hideSceneIdentityBanner(onlyKind);
      reportReachability();
    },
  });
}
