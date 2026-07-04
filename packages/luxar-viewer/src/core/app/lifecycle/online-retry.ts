import type { EventGroup } from '../../../utils/cross-layer/event-group';
import { log, Modules } from '../../../utils/log';

/**
 * Narrow view of the SceneLoader the online-retry listener needs. Matches
 * `SceneLoader.hasFailures` / `retryAllFailedLoaders` structurally so the
 * lifecycle layer doesn't import the loader class.
 */
export interface RetryCapableLoader {
  hasFailures(): boolean;
  /** `deferred: true` ⇒ a main update held the lock and NOTHING was retried. */
  retryAllFailedLoaders(): Promise<{ succeeded: string[]; failed: string[]; deferred?: boolean }>;
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
 * Delay between re-attempts while the loader defers the batch (a main
 * update / LOD refinement holds the serialization lock). 2 s is long enough
 * for a typical update pass to settle and short enough that recovery still
 * feels immediate after reconnecting.
 */
export const DEFERRED_RETRY_DELAY_MS = 2000;

/**
 * How many deferred re-attempts to make before giving up for this online
 * transition. The lock is held across real network I/O and multi-pass LOD
 * refinement, so a couple of retries are normal; ten (~20 s) means something
 * is genuinely wedged — leave the failures to the monitor banner / a manual
 * retry rather than polling forever.
 */
export const MAX_DEFERRED_RETRY_ATTEMPTS = 10;

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
 *     update lock internally), and toast the genuine outcome.
 *   - **Deferred results re-attempt.** The lock is very plausibly held at
 *     the moment `online` fires (the reconnect typically happens while an
 *     update or LOD refinement is mid-flight — the very situation that
 *     produced the failures), and `online` will not fire again while the
 *     browser stays online. A deferred batch therefore re-attempts every
 *     {@link DEFERRED_RETRY_DELAY_MS} up to {@link MAX_DEFERRED_RETRY_ATTEMPTS}
 *     times instead of being misreported as "still failing".
 *   - Re-entrancy guarded: `retryInFlight` covers the whole attempt chain,
 *     so bursts of `online` events cannot stack batches.
 *
 * Routed through the supplied {@link EventGroup} so both the listener and
 * any pending deferred-retry timer are cleaned up on app dispose (same
 * pattern as `installFocusHandling`).
 */
export function installOnlineRetry(ports: OnlineRetryPorts): void {
  let retryInFlight = false;
  let deferredTimer: ReturnType<typeof setTimeout> | undefined;

  // Dispose safety: a deferred re-attempt scheduled just before app teardown
  // must not fire against a disposed loader.
  ports.events.add(() => {
    if (deferredTimer !== undefined) clearTimeout(deferredTimer);
  });

  const attempt = (attemptNumber: number): void => {
    const loader = ports.getLoader();
    if (!loader?.hasFailures()) {
      // Failures recovered elsewhere (manual retry, dataset switch) — done.
      retryInFlight = false;
      return;
    }

    void loader
      .retryAllFailedLoaders()
      .then(({ succeeded, failed, deferred }) => {
        if (deferred) {
          // Nothing was retried — a main update holds the lock. Re-attempt
          // after a short delay; `online` won't fire again to do it for us.
          if (attemptNumber >= MAX_DEFERRED_RETRY_ATTEMPTS) {
            log.warning(
              Modules.LUXAR,
              `Online retry still deferred after ${attemptNumber} attempts — ` +
                'leaving failures to the monitor banner / manual retry'
            );
            retryInFlight = false;
            return;
          }
          deferredTimer = setTimeout(() => {
            deferredTimer = undefined;
            attempt(attemptNumber + 1);
          }, DEFERRED_RETRY_DELAY_MS);
          return;
        }

        retryInFlight = false;
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
        retryInFlight = false;
        log.warning(
          Modules.LUXAR,
          `Online retry batch failed: ${error instanceof Error ? error.message : String(error)}`
        );
      });
  };

  ports.events.on(window, 'online', () => {
    if (retryInFlight) return;
    const loader = ports.getLoader();
    if (!loader?.hasFailures()) return;

    retryInFlight = true;
    log.info(Modules.LUXAR, 'Connection restored - retrying failed loaders');
    ports.toast('Connection restored — retrying failed loads…', 3000);
    attempt(1);
  });
}
