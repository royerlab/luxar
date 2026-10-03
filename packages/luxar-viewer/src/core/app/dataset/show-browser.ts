import { DatasetBrowser } from '../../../ui/dataset-browser';
import { clearError } from '../../../ui/error-overlay';
import { showViewerError } from '../error-dialog';
import { showToast } from '../../../ui/toast';
import { log, Modules } from '../../../utils/log';
import { getViewerContainer } from '../../../utils/viewer-container';
import type { InputHandler } from '../../../input';

/**
 * Open the dataset browser modal. Returns the new instance so the
 * caller can track it on its own field; the helper never mutates
 * orchestrator state directly. All side effects on the orchestrator
 * (options.src update, datasetBrowser reset) flow through the supplied
 * ports; the host-URL replacement belongs to the guarded switch itself
 * (`LuxarApp.switchDataset`), so a programmatic switch writes it too.
 *
 * Caller is responsible for checking that no browser is already open
 * before calling this function — re-opening would stack modals.
 */
export interface ShowDatasetBrowserPorts {
  currentSrc: string | undefined;
  inputHandler: InputHandler;
  onSrcChange: (src: string) => void;
  /**
   * True while the app is still completing its initial routing and wiring.
   * Selections are refused before side effects or load dispatch in this state.
   */
  isInitializing: () => boolean;
  /**
   * True while a guarded dataset switch is already in flight. Consulted before
   * the `onSrcChange` side effect: the dispatch below will reject, and the src
   * snapshot must not end up pointing at a dataset that never loaded.
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

      // The src side effect runs only when the guarded switch can actually
      // start. If another switch is already in flight, `loadDataset` below
      // rejects — running it first would leave the src snapshot pointing at a
      // dataset that never loaded. (The host URL is the switch's own job.)
      if (!ports.isSwitchInFlight()) {
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
        showViewerError(`Failed to load dataset: ${message}`, ports.shortcutForAction);
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
