import type { EventGroup } from '../../../utils/cross-layer/event-group';
import { OPEN_DATASET_BROWSER_EVENT } from '../interaction/canvas-actions';

/**
 * Install a window-level {@link OPEN_DATASET_BROWSER_EVENT} listener that toggles
 * the modal: it closes an open browser and opens one otherwise. Routed
 * through the supplied {@link EventGroup} so the listener is cleaned up
 * on dispose.
 */
export interface BrowserShortcutPorts {
  events: EventGroup;
  hasOpenBrowser: () => boolean;
  showBrowser: () => void;
  closeBrowser: () => void;
}

export function installBrowserShortcut(ports: BrowserShortcutPorts): void {
  // Toggling keeps the dataset control (rail button + `O` key) behaving like
  // every other panel toggle.
  ports.events.on(window, OPEN_DATASET_BROWSER_EVENT, () => {
    if (ports.hasOpenBrowser()) {
      ports.closeBrowser();
    } else {
      ports.showBrowser();
    }
  });
}
