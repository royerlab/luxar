/**
 * Worker pool manager for data processing workers
 *
 * Supports multiple workers with round-robin load balancing for parallel
 * spatial queries across multiple nodes.
 */

import { wrap, Remote } from 'comlink';
import type { DataWorkerAPI } from './data-worker';
// Vite's `?worker` import emits a bundled, transpiled, hashed worker chunk and
// returns a default-exported Worker constructor. This sidesteps the broken
// `new Worker(new URL('./data-worker.ts', import.meta.url))` pattern under
// vite 8 + rolldown rc.17, which ships the raw `.ts` source as an asset.
import DataWorker from './data-worker?worker';
import { log, Modules } from '../utils/log';
import { config } from '../config';
import type { WorkerInstance } from './worker-pool/types';
import {
  WorkerTimeoutError,
  WorkerAbortError,
  type TimeoutKind,
} from './worker-pool/errors';

export type { WorkerInstance };
export { WorkerTimeoutError, WorkerAbortError };
export type { TimeoutKind };
import {
  getDataWorkerUrlOverride,
  getWorkerPool,
  disposeWorkerPool,
  setDataWorkerUrl,
} from './worker-pool/singleton';
import { withTimeout } from './worker-pool/timeout/with-timeout';
import { combineSignals } from './worker-pool/timeout/combine-signals';
import { pickTimeoutMs } from './worker-pool/timeout/pick-timeout-ms';
import { getConfiguredWorkerCount } from './worker-pool/lifecycle/worker-count';
import { initializeWithGuard } from './worker-pool/lifecycle/init-with-guard';
import {
  attachWorkerErrorHandlers,
  evictFailedWorker,
} from './worker-pool/lifecycle/error-handlers';
import {
  selectLeastBusy,
  type TrackedWorkerHandle,
} from './worker-pool/selection/least-busy';
export { getWorkerPool, disposeWorkerPool, setDataWorkerUrl };

export class WorkerPool {
  private workers: WorkerInstance[] = [];
  private initPromise: Promise<void> | null = null;
  private nextWorkerIndex = 0;
  /**
   * Pool-wide abort signal. When set (by `setAbortSignal`), every
   * `runWithTimeout` call additionally races against this signal so
   * a dataset-switch in `SceneLoader` can immediately settle all
   * in-flight worker promises. The signal does not cancel WASM
   * execution — see {@link WorkerAbortError} for the trade-off.
   *
   * Setting a new signal replaces (not chains) any previously-set
   * signal. The new signal applies to subsequent `runWithTimeout`
   * calls; already-racing promises continue with their original
   * signal until they settle.
   */
  private poolAbortSignal: AbortSignal | undefined;

  // Dispose-mid-init defense. `initialize()` spawns workers via
  // `Array.from(...).map(async ...)` and pushes them to `this.workers`
  // only on the fulfilled branch of `Promise.allSettled`. A `dispose()`
  // between `new DataWorker()` and that push would walk an empty
  // `this.workers` and exit; the still-pending factories would then
  // resolve and re-populate `this.workers` post-dispose, leaving live
  // Worker globals the pool no longer references.
  //
  // Two pieces of state guard against that:
  //   - `initGeneration` is bumped on every dispose. Each `initialize()`
  //     captures the generation at start; if it has moved on by the
  //     time a worker's init resolves, the worker is terminated and
  //     not pushed.
  //   - `pendingWorkers` holds every Worker that has been constructed
  //     but not yet either pushed into `this.workers` or terminated.
  //     `dispose()` terminates everything in this set so the in-flight
  //     factories find nothing live to push.
  private initGeneration = 0;
  private pendingWorkers = new Set<Worker>();

  /**
   * Get the configured worker count, capped by hardware concurrency
   *
   * - workerCount = 0: Auto mode, uses (hardwareConcurrency - 1)
   * - workerCount > 0: Uses that number, capped at (hardwareConcurrency - 1)
   */
  private getConfiguredWorkerCount(): number {
    return getConfiguredWorkerCount(config.dataLoading.performance.workerCount);
  }

  /**
   * Initialize worker pool (lazy initialization)
   */
  async initialize(): Promise<void> {
    // Return existing promise if initialization already started or completed
    if (this.initPromise) return this.initPromise;

    // Capture the generation token AND a per-attempt workers list.
    // The IIFE-level cleanup paths must only touch attempt-local state,
    // never globals: otherwise a stale init that settles after a fresh
    // init has published its workers would wipe them. Per-attempt
    // local state means stale completion can only clean up its OWN
    // workers.
    const myGeneration = ++this.initGeneration;
    const attemptWorkers: WorkerInstance[] = [];

    this.initPromise = (async () => {
      try {
        const workerCount = this.getConfiguredWorkerCount();
        log.info(Modules.WORKER_POOL, `Creating ${workerCount} data worker(s)...`);

        // Create all workers in parallel.
        // The data-worker URL override (set via setDataWorkerUrl) lets
        // embedders whose bundlers don't support vite's `?worker` import
        // point at an explicitly-built worker bundle. The override is
        // resolved lazily here so that calls to setDataWorkerUrl made
        // before the first getWorkerPool() take effect on every worker
        // spawned in this attempt.
        const urlOverride = getDataWorkerUrlOverride();
        const workerPromises = Array.from({ length: workerCount }, async (_, index) => {
          const worker = urlOverride
            ? new Worker(urlOverride, { type: 'module' })
            : new DataWorker();

          // Track this worker as in-flight so a concurrent dispose()
          // can terminate it. Removed on success or on the per-factory
          // catch path.
          this.pendingWorkers.add(worker);

          // Install runtime-error handlers BEFORE the first message — a
          // worker can crash during its own boot sequence (e.g. WASM init
          // OOM), and we want those failures to be surfaced as worker
          // failures rather than uncaught browser-level errors.
          this.attachWorkerErrorHandlers(worker, index + 1);

          const api = wrap<DataWorkerAPI>(worker);

          try {
            // Race api.initialize() against:
            //   1. a hard init timeout (worker script blocked / unreachable
            //      → onerror may fire but Comlink's initialize() never
            //      settles because the worker never sent a message),
            //   2. an onerror short-circuit (worker fails *during* its
            //      boot before any pool entry exists for it).
            await this.initializeWithGuard(worker, api, index + 1);
            // If dispose() ran while we were awaiting init, the
            // generation has moved on. Self-terminate and reject so the
            // parent doesn't push us into the post-dispose pool.
            if (this.initGeneration !== myGeneration) {
              throw new Error(`Worker ${index + 1} aborted: pool was disposed during init`);
            }
            this.pendingWorkers.delete(worker);
            log.info(Modules.WORKER_POOL, `Worker ${index + 1}/${workerCount} ready`);
            return { worker, api, activeQueries: 0 };
          } catch (error) {
            log.error(Modules.WORKER_POOL, `Worker ${index + 1} initialization failed`, error);
            this.pendingWorkers.delete(worker);
            worker.terminate();
            throw error;
          }
        });

        // Wait for all workers to initialize
        const results = await Promise.allSettled(workerPromises);

        // Collect successful workers into the attempt-local list
        // first, so stale-generation cleanup doesn't reach for any
        // newer attempt's published workers.
        for (const result of results) {
          if (result.status === 'fulfilled') {
            attemptWorkers.push(result.value);
          }
        }

        // Stale-generation guard. If dispose() bumped the generation
        // while we were awaiting allSettled, terminate ONLY this
        // attempt's workers and return. Do NOT touch `this.workers` —
        // a fresh generation may have already published its own
        // workers there.
        if (this.initGeneration !== myGeneration) {
          for (const { worker } of attemptWorkers) {
            try {
              worker.terminate();
            } catch {
              // Already terminated by dispose's pendingWorkers walk.
            }
          }
          return;
        }

        // Generation still current — publish.
        for (const entry of attemptWorkers) {
          this.workers.push(entry);
        }

        if (this.workers.length === 0) {
          throw new Error(
            'Failed to initialize any data workers. ' +
              'Luxar requires WebAssembly and Web Workers support.'
          );
        }

        if (this.workers.length < workerCount) {
          log.warning(
            Modules.WORKER_POOL,
            `Only ${this.workers.length}/${workerCount} workers initialized successfully`
          );
        }

        this.nextWorkerIndex = 0;
        log.info(Modules.WORKER_POOL, `Worker pool ready with ${this.workers.length} worker(s)`);
      } catch (e) {
        // Same stale-generation guard for the error path. If a newer
        // generation has taken over, only clean up this attempt's
        // workers.
        if (this.initGeneration !== myGeneration) {
          for (const { worker } of attemptWorkers) {
            try {
              worker.terminate();
            } catch {
              // Already terminated.
            }
          }
          throw e;
        }
        // Generation current — full cleanup of this generation's state.
        for (const { worker } of this.workers) {
          worker.terminate();
        }
        this.workers = [];
        // Terminate anything still pending too, in case the catch
        // fires while factories are still settling.
        for (const worker of this.pendingWorkers) {
          worker.terminate();
        }
        this.pendingWorkers.clear();
        // Note: we deliberately keep `initPromise` (the rejected one) so
        // subsequent `getWorker()` / `runWithTimeout` calls fail FAST
        // rather than re-running the 10s init guard for every nD load.
        // A blocked worker chunk would otherwise stack 10s × N delays
        // and blow past the page's `waitForLuxarReady` timeout. To opt
        // back in to a fresh init attempt (e.g. after a transient
        // network blip), call `reinitialize()`.
        throw e;
      }
    })();

    return this.initPromise;
  }

  /**
   * Race `api.initialize()` against (1) a hard init timeout and (2) the
   * worker's own `onerror`/`onmessageerror` events.
   *
   * The pool's permanent `attachWorkerErrorHandlers` evicts a failed
   * worker via `handleWorkerFailure`, but during pool init the worker
   * isn't in `this.workers` yet — so `handleWorkerFailure` finds
   * nothing to evict and the dangling Comlink `initialize()` promise
   * never settles. (Repro: route-block the worker script in a
   * Playwright test; the page hangs at boot.)
   *
   * We attach a short-lived listener that rejects the init promise
   * when the worker fails before it joins the pool. The init timeout
   * is the belt-and-suspenders fallback — even if no error event
   * fires (e.g. a network stall that never resolves), we eventually
   * reject and let the caller fall back to the main thread.
   */
  private initializeWithGuard(
    worker: Worker,
    api: Remote<DataWorkerAPI>,
    workerNumber: number
  ): Promise<void> {
    return initializeWithGuard(
      worker,
      api,
      workerNumber,
      config.dataLoading.performance.workerInitTimeoutMs,
      () => this.attachWorkerErrorHandlers(worker, workerNumber)
    );
  }

  /**
   * Install `onerror` and `onmessageerror` handlers on a freshly-created
   * worker so a crash inside the worker (uncaught throw, OOM during WASM
   * init, unserializable Comlink message) surfaces as a logged failure
   * and is removed from the active pool, rather than escaping to the
   * browser's `window.onerror` and freezing requests that are awaiting
   * Comlink replies from this worker.
   *
   * Note: this is a best-effort safety net. Comlink-wrapped calls that
   * are mid-flight when the worker dies will still hang their callers —
   * route hot paths through {@link runWithTimeout} for the per-call
   * timeout that complements this handler.
   */
  private attachWorkerErrorHandlers(worker: Worker, workerNumber: number): void {
    attachWorkerErrorHandlers(worker, workerNumber, (w, reason) =>
      this.handleWorkerFailure(w, reason)
    );
  }

  /**
   * Remove a failed worker from the pool and terminate it. If this drops
   * the pool to zero workers, log a clear error so the surrounding
   * application can decide whether to fall back to the main-thread
   * implementation or surface a failure dialog.
   */
  private handleWorkerFailure(worker: Worker, reason: string): void {
    const outcome = evictFailedWorker(this.workers, worker, reason);
    if (outcome === 'pool-empty') {
      // Allow the next initialize() call to attempt a fresh pool.
      this.initPromise = null;
    }
  }

  /**
   * Explicitly reset the pool's cached init promise so the next
   * {@link getWorker} / {@link runWithTimeout} call re-runs init. Used
   * to recover from a transient init failure (e.g. the worker chunk
   * was briefly unreachable) without restarting the entire app.
   *
   * Routine "init failed once, fall back" cases should NOT call this —
   * letting the rejected promise stick keeps subsequent calls fast
   * (instant reject) instead of stacking 10s init guards.
   */
  reinitialize(): void {
    if (this.workers.length === 0) {
      this.initPromise = null;
    }
  }

  /**
   * Race a worker-routed Promise against a timeout. On timeout, log
   * the failure, evict the responsible worker (if known) via
   * {@link handleWorkerFailure}, and reject with
   * {@link WorkerTimeoutError}. A `timeoutMs` of 0 disables the
   * timeout — callers that don't need it can still use this helper as
   * a thin pass-through to keep the call-site uniform.
   *
   * `worker` may be omitted when the caller cannot identify which
   * worker handled the call (e.g. round-robin selection); the timeout
   * still fires, but the pool isn't pruned.
   */
  withTimeout<T>(
    operation: string,
    call: Promise<T>,
    timeoutMs: number,
    worker?: Worker
  ): Promise<T> {
    return withTimeout(
      operation,
      call,
      timeoutMs,
      (w, reason) => this.handleWorkerFailure(w, reason),
      worker
    );
  }

  /**
   * Pick the next round-robin {@link WorkerInstance}. Internal — exposes the
   * underlying `Worker` so {@link runWithTimeout} can evict it on timeout.
   */
  private async nextWorkerInstance(): Promise<WorkerInstance> {
    await this.initialize();

    if (this.workers.length === 0) {
      throw new Error('[WorkerPool] No workers available after initialization');
    }

    const wi = this.workers[this.nextWorkerIndex];
    this.nextWorkerIndex = (this.nextWorkerIndex + 1) % this.workers.length;
    return wi;
  }

  /**
   * Get a worker API using round-robin selection.
   *
   * Untracked callers receive workers in rotating order. For load-aware
   * selection (picking the worker with the fewest active queries), use
   * {@link getWorkerWithTracking} instead.
   *
   * **Important — no timeout guard.** Direct `await` on the returned
   * `Remote<DataWorkerAPI>` lets a dead/stuck worker hang the caller
   * indefinitely (Comlink's onerror handler can't settle an in-flight
   * promise). Production code MUST go through {@link runWithTimeout}
   * instead — it routes the call through {@link withTimeout} on a
   * round-robin-selected worker. This direct `getWorker()` accessor is
   * intentionally retained for tests and low-level worker-pool
   * unit tests that need raw access; lint-grep for new production
   * uses periodically.
   */
  async getWorker(): Promise<Remote<DataWorkerAPI>> {
    return (await this.nextWorkerInstance()).api;
  }

  /**
   * Pick the kind-appropriate timeout from config. Visibility uses
   * `workerVisibilityTimeoutMs`; projection AND decode share
   * `workerProjectionTimeoutMs` (both are long-running CPU-bound calls
   * — a dedicated decode knob can be added later if telemetry shows a
   * need).
   */
  private pickTimeoutMs(kind: TimeoutKind): number {
    return pickTimeoutMs(kind, config.dataLoading.performance);
  }

  /**
   * Run a worker call through {@link withTimeout} on a round-robin-selected
   * worker. The single production entry point for any Comlink-routed call
   * that needs a hang-detection guard — direct `await` against a worker
   * `Remote` lets a dead worker hang the caller forever (the `onerror`
   * handler can't settle a Comlink promise that's already in flight).
   *
   * On timeout the responsible worker is evicted via
   * {@link handleWorkerFailure} and the caller receives a
   * {@link WorkerTimeoutError} that the existing geometry-loader try/catch
   * blocks already route to the main-thread fallback.
   */
  async runWithTimeout<T>(
    op: string,
    kind: TimeoutKind,
    fn: (api: Remote<DataWorkerAPI>) => Promise<T>,
    signal?: AbortSignal
  ): Promise<T> {
    // Resolve the effective signal: pool-wide one OR caller's. If both
    // are present, race them together so either can settle the call.
    const effectiveSignal = this.combineSignals(this.poolAbortSignal, signal);

    // Pre-check: signal already aborted? Bail before dispatching any work.
    if (effectiveSignal?.aborted) {
      throw new WorkerAbortError(op);
    }

    // Route through getWorkerWithTracking so a stalled worker (one
    // whose activeQueries has grown past the others) stops being
    // selected. The previous round-robin via nextWorkerInstance had
    // no load awareness, so a slow worker received every Nth call
    // until each call timed out individually — head-of-line blocking.
    const tracked = await this.getWorkerWithTracking();

    // Second-chance abort check: the await above may have yielded long
    // enough for the caller's dataset-switch to fire. Don't even start.
    if (effectiveSignal?.aborted) {
      throw new WorkerAbortError(op);
    }

    tracked.markQueryStart();
    try {
      // Race the worker call against the timeout AND the abort signal.
      // The abort cannot kill the WASM task, but it can settle the
      // promise immediately so the caller proceeds with whatever the
      // dataset-switch wants to do next.
      const workerPromise = this.withTimeout(
        op,
        fn(tracked.api),
        this.pickTimeoutMs(kind),
        tracked.worker
      );

      if (!effectiveSignal) {
        return await workerPromise;
      }

      let onAbort: (() => void) | undefined;
      const abortPromise = new Promise<never>((_, reject) => {
        onAbort = (): void => reject(new WorkerAbortError(op));
        effectiveSignal.addEventListener('abort', onAbort, { once: true });
      });

      try {
        return await Promise.race([workerPromise, abortPromise]);
      } finally {
        if (onAbort) effectiveSignal.removeEventListener('abort', onAbort);
      }
    } finally {
      tracked.markQueryEnd();
    }
  }

  /**
   * Set or clear the pool-wide abort signal. Every subsequent
   * {@link runWithTimeout} call additionally races against this
   * signal. Used by `SceneLoader.loadScene` to immediately settle
   * worker tasks queued by the previous dataset when the user
   * switches datasets.
   *
   * Pass `undefined` to clear (no pool-wide signal — only caller-
   * supplied signals apply).
   *
   * Setting a new signal replaces (not chains) any previously-set
   * signal. Already-racing promises continue with their original
   * effective signal until they settle.
   */
  setAbortSignal(signal: AbortSignal | undefined): void {
    this.poolAbortSignal = signal;
  }

  /**
   * Combine the pool-wide signal with a caller-supplied signal into
   * a single abort source. Returns `undefined` when both are absent.
   * Uses native `AbortSignal.any` when available (modern browsers /
   * Node 20+) and falls back to the simpler "trip either one" wiring
   * for older runtimes.
   */
  private combineSignals(
    a: AbortSignal | undefined,
    b: AbortSignal | undefined
  ): AbortSignal | undefined {
    return combineSignals(a, b);
  }

  /**
   * Get a worker with query tracking for load balancing.
   *
   * Returns the worker `api`, the underlying `worker` (for
   * eviction-on-timeout), and callbacks to mark query start/end so
   * the active-queries counter reflects in-flight load.
   *
   * Used by {@link runWithTimeout} to route hot-path calls to the
   * least-loaded worker. A stalled worker stops being selected once
   * its activeQueries grows past the others — without this, the
   * round-robin fallback would queue new calls on the stalled worker
   * until it timed out individually.
   */
  async getWorkerWithTracking(): Promise<TrackedWorkerHandle> {
    await this.initialize();

    if (this.workers.length === 0) {
      throw new Error('[WorkerPool] No workers available after initialization');
    }

    return selectLeastBusy(this.workers);
  }

  /**
   * Get the number of active workers
   */
  getWorkerCount(): number {
    return this.workers.length;
  }

  /**
   * Get pool statistics for monitoring
   */
  getStats(): {
    workerCount: number;
    activeQueries: number[];
    /** Aggregate across all workers — sum of per-worker `activeQueries`. */
    totalActive: number;
    /** Per-worker peak since init; useful for spotting one hot worker. */
    peakActive: number;
  } {
    const activeQueries = this.workers.map((w) => w.activeQueries);
    let totalActive = 0;
    let peakActive = 0;
    for (const n of activeQueries) {
      totalActive += n;
      if (n > peakActive) peakActive = n;
    }
    return {
      workerCount: this.workers.length,
      activeQueries,
      totalActive,
      peakActive,
    };
  }

  /**
   * Convenience accessor for the aggregate "in-flight task" count.
   * Exposed in `__luxarDebug.workers.queueDepth` for live diagnostics
   * of prefetch backpressure / dataset-switch task accumulation.
   *
   * @returns The number of worker tasks currently in flight across the
   *   pool (sum of `activeQueries` per worker). Zero when the pool is
   *   idle.
   */
  getQueueDepth(): number {
    let total = 0;
    for (const w of this.workers) total += w.activeQueries;
    return total;
  }

  /**
   * Check if worker pool is initialized
   */
  isInitialized(): boolean {
    return this.workers.length > 0;
  }

  /**
   * Clean up all worker resources.
   *
   * Also terminates workers still pending init (via `pendingWorkers`)
   * and bumps `initGeneration` so any factories still in flight detect
   * the dispose and self-terminate when they resolve.
   */
  dispose(): void {
    // Bump the generation FIRST so any factories that resolve between
    // here and our pendingWorkers walk see the mismatch and bail.
    this.initGeneration++;

    if (this.pendingWorkers.size > 0) {
      log.info(
        Modules.WORKER_POOL,
        `Terminating ${this.pendingWorkers.size} pending data worker(s) mid-init`
      );
      for (const worker of this.pendingWorkers) {
        try {
          worker.terminate();
        } catch {
          // Already terminated or browser quirk; ignore.
        }
      }
      this.pendingWorkers.clear();
    }

    if (this.workers.length > 0) {
      log.info(Modules.WORKER_POOL, `Terminating ${this.workers.length} data worker(s)`);
      for (const { worker } of this.workers) {
        worker.terminate();
      }
      this.workers = [];
    }
    // Always reset local state regardless of whether workers were running.
    // A pool that failed to initialize has no workers to terminate but still
    // holds a rejected `initPromise`; without this reset, a direct reuse
    // after failed init would surface the stale rejection from the cache
    // instead of attempting a fresh `initialize()`.
    this.initPromise = null;
    this.nextWorkerIndex = 0;
  }
}

