import { DatasetBrowser } from '../../../ui/dataset-browser';
import { clearError, showError } from '../../../ui/error-overlay';
import { replaceBrowserDataSourceUrl } from '../../../config/url-params';
import { log, Modules } from '../../../utils/log';
import type { InputHandler } from '../../../input/input-handler';

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
  loadDataset: (src: string) => Promise<void>;
  onClose: () => void;
}

export function showDatasetBrowser(ports: ShowDatasetBrowserPorts): DatasetBrowser {
  // Clear any existing error messages when opening the browser
  clearError();

  const browser = new DatasetBrowser({
    container: document.body,
    currentSrc: ports.currentSrc,
    onDatasetSelect: async (fullUrl: string) => {
      // The browser now passes full URLs directly, preserving directory context
      // Strip any trailing slashes to ensure consistent URL format
      const cleanUrl = fullUrl.replace(/\/+$/, '');

      // Reflect the chosen dataset in the URL bar only for callers that opt in.
      // The standalone bootstrap opts in; programmatic/embedded usage defaults
      // to no host-page URL mutation.
      if (ports.updateBrowserUrl) {
        replaceBrowserDataSourceUrl(cleanUrl);
      }

      // Track the new src in our options snapshot so a subsequent browser
      // open lands in the right directory.
      ports.onSrcChange(cleanUrl);

      // [core OOS] Wrap `loadDataset` in try/catch. The DatasetBrowser
      // modal's `onDatasetSelect` contract is `Promise<void>` — the
      // modal doesn't surface rejections to the user, so without this
      // wrapper a load failure (bad URL, transient network, malformed
      // zarr) became an unhandled promise rejection silently. Now we
      // log + show the failure in the user-facing error overlay before
      // re-throwing so any awaiting caller still observes the
      // rejection.
      try {
        await ports.loadDataset(cleanUrl);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        log.error(Modules.LUXAR, `loadDataset failed for ${cleanUrl}: ${message}`, error);
        showError(`Failed to load dataset: ${message}`);
        throw error;
      }
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
