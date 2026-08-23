import type { EventGroup } from '../../../utils/cross-layer/event-group';

/**
 * Install a window-level `open-dataset-browser` listener that toggles
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
  ports.events.on(window, 'open-dataset-browser', () => {
    if (ports.hasOpenBrowser()) {
      ports.closeBrowser();
    } else {
      ports.showBrowser();
    }
  });
}
