/**
 * Listener bookkeeping for the spatial-index loader's monitor surface,
 * shared across the points / lines / gsplats facades.
 *
 * Originally lived in `data/points/`; hoisted to `data/loaders/` once
 * lines and gsplats grew the same monitor surface. The subscribe /
 * unsubscribe / fan-out behavior — including the try/catch shielding
 * that prevents one bad listener from blocking the rest — is
 * unit-tested in isolation, without a zarr store or an active query
 * map.
 *
 * @module data/loaders/monitor-events
 */

import { log, Modules } from '../../utils/log';
import type { MonitorEvent, MonitorEventListener } from '../../types/data-monitor-types';

/**
 * Owns the listener Set for a `LoaderMonitor` implementation and emits
 * events to all listeners with per-listener error isolation. A throw
 * from one listener is logged and swallowed so the next listener in
 * iteration order still runs.
 *
 * The emitter is decoupled from any specific loader: the loader's
 * `addEventListener` / `removeEventListener` / `emitEvent` calls
 * delegate here, and `clear()` is called on dispose.
 */
export class LoaderEventEmitter {
  private readonly listeners = new Set<MonitorEventListener>();

  /** Add a listener. Idempotent — same callback registered twice is held once. */
  add(listener: MonitorEventListener): void {
    this.listeners.add(listener);
  }

  /** Remove a listener. No-op if not registered. */
  remove(listener: MonitorEventListener): void {
    this.listeners.delete(listener);
  }

  /**
   * Notify every listener of `event`. A listener that throws is
   * logged at error level and the iteration continues — one bad
   * listener never starves the rest.
   */
  emit(event: MonitorEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (error) {
        log.error(Modules.SPATIAL_INDEX_LOADER, 'Error in event listener:', error);
      }
    }
  }

  /** Drop every listener — used on dispose. */
  clear(): void {
    this.listeners.clear();
  }

  /** Number of registered listeners (test-only convenience). */
  get size(): number {
    return this.listeners.size;
  }
}
