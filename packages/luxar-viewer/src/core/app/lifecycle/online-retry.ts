import type { EventGroup } from '../../../utils/cross-layer/event-group';
import { log, Modules } from '../../../utils/log';

/**
 * Narrow view of the SceneLoader the online-retry listener needs. Matches
 * `SceneLoader.hasFailures` / `retryAllFailedLoaders` structurally so the
 * lifecycle layer doesn't import the loader class.
 */
export interface RetryCapableLoader {
  hasFailures(): boolean;
  retryAllFailedLoaders(): Promise<{ succeeded: string[]; failed: string[] }>;
}

/**
 * Ports for {@link installOnlineRetry}. `getLoader` is a live accessor (the
 * SceneLoaderManager swaps loader instances on dataset switches, so a
 * snapshot would go stale) and `toast` is the user-facing notifier.
 */
export interface OnlineRetryPorts {
  events: EventGroup;
  getLoader: () => RetryCapableLoader | null;
  toast: (message: string, durationMs?: number) => void;
}

/**
 * Register a `window 'online'` listener that retries every failed loader
 * when connectivity is restored — the trigger half of the failed-load
 * recovery story (`SceneLoader.retryAllFailedLoaders` is the engine; its
 * own doc names "after connectivity is restored" as the intended use).
 *
 * Behavior:
 *   - No failures recorded → silent no-op (the common case).
 *   - Failures present → toast that a retry is starting, run
 *     `retryAllFailedLoaders()` (the loader serializes it against the
 *     update lock internally), and toast the outcome.
 *   - Re-entrancy guarded: browsers can fire `online` in bursts and a
 *     retry batch can outlive the next event; while one batch is in
 *     flight, further events are ignored (the batch already covers every
 *     failure recorded at its start; anything failing later is picked up
 *     by the next genuine online transition or a manual retry).
 *
 * Routed through the supplied {@link EventGroup} so the listener is
 * cleaned up on app dispose (same pattern as `installFocusHandling`).
 */
export function installOnlineRetry(ports: OnlineRetryPorts): void {
  let retryInFlight = false;

  ports.events.on(window, 'online', () => {
    if (retryInFlight) return;
    const loader = ports.getLoader();
    if (!loader?.hasFailures()) return;

    retryInFlight = true;
    log.info(Modules.LUXAR, 'Connection restored - retrying failed loaders');
    ports.toast('Connection restored — retrying failed loads…', 3000);

    void loader
      .retryAllFailedLoaders()
      .then(({ succeeded, failed }) => {
        if (failed.length === 0) {
          ports.toast(
            `Recovered ${succeeded.length} failed load${succeeded.length === 1 ? '' : 's'}.`,
            4000
          );
        } else {
          ports.toast(
            `Retried failed loads: ${succeeded.length} recovered, ${failed.length} still failing.`,
            5000
          );
        }
      })
      .catch((error) => {
        // retryAllFailedLoaders resolves per-path failures into its result;
        // a rejection here is unexpected infrastructure trouble — log it and
        // leave the failure records for the next trigger.
        log.warning(
          Modules.LUXAR,
          `Online retry batch failed: ${error instanceof Error ? error.message : String(error)}`
        );
      })
      .finally(() => {
        retryInFlight = false;
      });
  });
}
