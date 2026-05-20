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
}

export function installBrowserShortcut(ports: BrowserShortcutPorts): void {
  ports.events.on(window, 'open-dataset-browser', () => {
    if (!ports.hasOpenBrowser()) {
      ports.showBrowser();
    }
  });
}
