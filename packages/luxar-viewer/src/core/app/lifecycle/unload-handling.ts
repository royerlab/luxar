import { log, Modules } from '../../../utils/log';
import type { EventGroup } from '../../../utils/cross-layer/event-group';
import { OPFSStore } from '../../../cache/multi-level-caching-store/opfs-store';

/**
 * Register the page-lifecycle handlers, routed through the supplied
 * {@link EventGroup} so every listener is removed when the parent disposes:
 *
 * - `beforeunload` tears the app down (`dispose`).
 * - `pagehide` and `visibilitychange → hidden` flush persistent caches
 *   (`flushPersistence`, default: every live OPFS store's pending index save).
 *   `beforeunload` does not fire on mobile tab switches, bfcache navigations
 *   or discarded tabs, so without this the L2 index written on a debounce was
 *   routinely lost with the page, leaving thousands of unindexed chunk files
 *   the next session could not use. The flush is best-effort and does not
 *   dispose anything: a hidden page may well come back.
 */
export interface UnloadHandlerPorts {
  events: EventGroup;
  dispose: () => void;
  /** Best-effort, synchronous-to-start persistence flush. Defaults to the OPFS index flush. */
  flushPersistence?: () => void;
}

/** Run a lifecycle callback without ever letting it throw out of the event. */
function guarded(label: string, fn: () => void): void {
  try {
    fn();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log.warning(Modules.LUXAR, `${label}: ${message}`, error);
  }
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
    guarded('dispose() threw during beforeunload', ports.dispose);
  });

  const flush = ports.flushPersistence ?? (() => OPFSStore.flushAllMetadata());
  ports.events.on(window, 'pagehide', () => {
    guarded('persistence flush threw during pagehide', flush);
  });
  ports.events.on(document, 'visibilitychange', () => {
    if (document.visibilityState !== 'hidden') return;
    guarded('persistence flush threw on visibilitychange', flush);
  });
}
