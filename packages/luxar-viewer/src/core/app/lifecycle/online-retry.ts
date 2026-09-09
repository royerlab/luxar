import type { EventGroup } from '../../../utils/cross-layer/event-group';
import { log, Modules } from '../../../utils/log';

/**
 * Narrow view of the SceneLoader the online-retry listener needs. Matches
 * `SceneLoader.hasFailures` / `retryAllFailedLoaders` structurally so the
 * lifecycle layer doesn't import the loader class.
 */
export interface RetryCapableLoader {
  hasFailures(): boolean;
  /**
   * Whether any failure is worth an automatic retry — a transient loader cause
   * still under the attempt cap, or a deferred LOD branch latched on an archive
   * fault. Gating on this keeps deterministic ordinary failures quiet while an
   * `online` transition can re-open archive-backed lazy work.
   */
  hasAutoRetryableFailures(): boolean;
  /** `deferred: true` ⇒ a main update held the lock and NOTHING was retried. */
  retryAllFailedLoaders(opts?: {
    onlyAutoRetryable?: boolean;
  }): Promise<{ succeeded: string[]; failed: string[]; deferred?: boolean }>;
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

/** Backoff before checking for retryable failures while still online. */
export const ONLINE_RETRY_POLL_MS = 5000;

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
 *   - While the browser stays online, a lightweight timer checks for newly
 *     recorded transient failures and starts the same bounded retry path.
 *
 * Routed through the supplied {@link EventGroup} so both the listener and
 * pending retry timers are cleaned up on app dispose (same
 * pattern as `installFocusHandling`).
 */
export function installOnlineRetry(ports: OnlineRetryPorts): void {
  let retryInFlight = false;
  let pollSuppressed = false;
  let deferredTimer: ReturnType<typeof setTimeout> | undefined;
  let pollTimer: ReturnType<typeof setTimeout> | undefined;

  // Dispose safety: a deferred re-attempt scheduled just before app teardown
  // must not fire against a disposed loader.
  ports.events.add(() => {
    if (deferredTimer !== undefined) clearTimeout(deferredTimer);
    if (pollTimer !== undefined) clearTimeout(pollTimer);
  });

  const attempt = (attemptNumber: number): void => {
    const loader = ports.getLoader();
    if (!loader?.hasAutoRetryableFailures()) {
      // Recovered elsewhere (manual retry, dataset switch), or everything left
      // is deterministic / past the cap — either way, done.
      retryInFlight = false;
      return;
    }

    void loader
      .retryAllFailedLoaders({ onlyAutoRetryable: true })
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
            pollSuppressed = true;
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

  const triggerRetry = (message: string, source: 'online' | 'poll'): void => {
    if (source === 'online') pollSuppressed = false;
    if (retryInFlight || navigator.onLine === false) return;
    const loader = ports.getLoader();
    if (!loader?.hasAutoRetryableFailures()) {
      pollSuppressed = false;
      return;
    }
    if (source === 'poll' && pollSuppressed) return;

    retryInFlight = true;
    log.info(Modules.LUXAR, message);
    ports.toast(message, 3000);
    attempt(1);
  };

  const schedulePoll = (): void => {
    pollTimer = setTimeout(() => {
      pollTimer = undefined;
      triggerRetry('Retrying failed loads…', 'poll');
      schedulePoll();
    }, ONLINE_RETRY_POLL_MS);
  };

  schedulePoll();

  ports.events.on(window, 'online', () => {
    // Deliberately NOT `hasFailures()`: a scene whose only failures are
    // deterministic gets no retry and no "Connection restored" toast, since
    // reconnecting cannot help it.
    triggerRetry('Connection restored — retrying failed loads…', 'online');
  });
}
