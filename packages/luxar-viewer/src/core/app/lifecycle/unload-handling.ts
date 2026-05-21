import type { EventGroup } from '../../../utils/cross-layer/event-group';

/**
 * Register a `beforeunload` handler that tears the app down when the
 * page is leaving. Routed through the supplied {@link EventGroup} so
 * the listener is removed when the parent disposes.
 */
export interface UnloadHandlerPorts {
  events: EventGroup;
  dispose: () => void;
}

export function installUnloadHandler(ports: UnloadHandlerPorts): void {
  ports.events.on(window, 'beforeunload', () => ports.dispose());
}
