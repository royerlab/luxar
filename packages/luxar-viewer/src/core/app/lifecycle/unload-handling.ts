import { log, Modules } from '../../../utils/log';
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
  // [core OOS] Wrap the dispose callback in try/catch. Browser unload
  // is terminal — an unhandled throw bubbles to `window.onerror`, but
  // call sites don't always wrap with `safeDispose` (and even when
  // they do, the wrap is the caller's responsibility, not this
  // helper's contract). A defensive try/catch here guarantees the
  // unload-handler itself never crashes mid-dispose, so any
  // subsequent unload work (e.g. native flush-on-unload telemetry,
  // other beforeunload listeners) still runs.
  ports.events.on(window, 'beforeunload', () => {
    try {
      ports.dispose();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      log.warning(Modules.LUXAR, `dispose() threw during beforeunload: ${message}`, error);
    }
  });
}
