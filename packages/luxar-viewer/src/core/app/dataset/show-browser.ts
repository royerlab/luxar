import { DatasetBrowser } from '../../../ui/dataset-browser';
import { clearError, showError } from '../../../ui/error-overlay';
import { showToast } from '../../../ui/toast';
import { replaceBrowserDataSourceUrl } from '../../../config/url-params';
import { clearViewStateHash } from '../../../ui/view-state';
import { log, Modules } from '../../../utils/log';
import { getViewerContainer } from '../../../utils/viewer-container';
import { KeyAction, type InputHandler } from '../../../input';

/**
 * Open the dataset browser modal. Returns the new instance so the
 * caller can track it on its own field; the helper never mutates
 * orchestrator state directly. All side effects on the orchestrator
 * (options.src update, datasetBrowser reset, host-URL replacement)
 * flow through the supplied ports.
 *
 * Caller is responsible for checking that no browser is already open
 * before calling this function — re-opening would stack modals.
 */
export interface ShowDatasetBrowserPorts {
  currentSrc: string | undefined;
  updateBrowserUrl: boolean;
  inputHandler: InputHandler;
  onSrcChange: (src: string) => void;
  /**
   * True while the app is still completing its initial routing and wiring.
   * Selections are refused before side effects or load dispatch in this state.
   */
  isInitializing: () => boolean;
  /**
   * True while a guarded dataset switch is already in flight. Consulted before
   * the selection side effects (host-URL replacement, onSrcChange): the
   * dispatch below will reject, and the host URL or src snapshot must not end
   * up pointing at a dataset that never loaded.
   */
  isSwitchInFlight: () => boolean;
  loadDataset: (src: string) => Promise<void>;
  shortcutForAction: (actionId: string) => string | undefined;
  onClose: () => void;
}

export function showDatasetBrowser(ports: ShowDatasetBrowserPorts): DatasetBrowser {
  // Clear any existing error messages when opening the browser
  clearError();

  const browser = new DatasetBrowser({
    container: getViewerContainer(),
    currentSrc: ports.currentSrc,
    onDatasetSelect: (fullUrl: string) => {
      // The browser now passes full URLs directly, preserving directory context
      // Strip any trailing slashes to ensure consistent URL format
      const cleanUrl = fullUrl.replace(/\/+$/, '');

      if (ports.isInitializing()) {
        showToast('Luxar is still starting up; try again in a moment.');
        return false;
      }

      // Selection side effects run only when the guarded switch can actually
      // start. If another switch is already in flight, `loadDataset` below
      // rejects — running these first would leave
      // the host URL and src snapshot pointing at a dataset that never loaded.
      if (!ports.isSwitchInFlight()) {
        // Reflect the chosen dataset in the URL bar only for callers that opt in.
        // The standalone bootstrap opts in; programmatic/embedded usage defaults
        // to no host-page URL mutation.
        if (ports.updateBrowserUrl) {
          replaceBrowserDataSourceUrl(cleanUrl);
          // Layer edits and the camera belong to the scene they were made
          // in; the new dataset starts from its authored state.
          if (typeof window !== 'undefined') clearViewStateHash(window);
        }

        // Track the new src in our options snapshot so a subsequent browser
        // open lands in the right directory.
        ports.onSrcChange(cleanUrl);
      }

      // [core OOS] Wrap `loadDataset` in a rejection handler. The normal
      // DatasetBrowser selection path returns `Promise<void>`, and the modal
      // doesn't surface rejections to the user, so without this
      // wrapper a load failure (bad URL, transient network, malformed
      // zarr) became an unhandled promise rejection silently. Now we
      // log + show the failure in the user-facing error overlay before
      // re-throwing so any awaiting caller still observes the
      // rejection.
      return ports.loadDataset(cleanUrl).catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        log.error(Modules.LUXAR, `loadDataset failed for ${cleanUrl}: ${message}`, error);
        showError(`Failed to load dataset: ${message}`, ports.shortcutForAction, {
          datasetBrowser: KeyAction.toggleDatasetBrowser,
          help: KeyAction.toggleHelp,
        });
        throw error;
      });
    },
    onClose: () => {
      ports.onClose();
      // Clear the close handle in PanelCoordinator so a follow-on
      // Escape doesn't try to close an already-closed browser.
      ports.inputHandler?.setDatasetBrowser(undefined);
    },
  });

  // Hand a close handle to the InputHandler/PanelCoordinator so
  // Escape routes through `close()` (which fires onClose above)
  // instead of yanking the DOM node and stranding our ref.
  ports.inputHandler?.setDatasetBrowser(browser);
  return browser;
}
