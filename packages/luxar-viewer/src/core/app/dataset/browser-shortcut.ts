import type { EventGroup } from '../../../utils/cross-layer/event-group';

/**
 * Install a window-level `open-dataset-browser` listener that opens
 * the modal when one isn't already shown. Routed through the supplied
 * {@link EventGroup} so the listener is cleaned up on dispose.
 */
export interface BrowserShortcutPorts {
  events: EventGroup;
  hasOpenBrowser: () => boolean;
  showBrowser: () => void;
  closeBrowser: () => void;
}

export function installBrowserShortcut(ports: BrowserShortcutPorts): void {
  // Toggle: the `open-dataset-browser` event now opens the modal if it's
  // closed and closes it if it's already open, so the dataset control (rail
  // button + `O` key) behaves like every other panel toggle.
  ports.events.on(window, 'open-dataset-browser', () => {
    if (ports.hasOpenBrowser()) {
      ports.closeBrowser();
    } else {
      ports.showBrowser();
    }
  });
}
