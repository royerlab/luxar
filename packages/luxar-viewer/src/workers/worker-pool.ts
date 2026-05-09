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

export interface WorkerInstance {
  worker: Worker;
  api: Remote<DataWorkerAPI>;
  activeQueries: number;
}

/**
 * Class of worker call, used to pick a default timeout from config.
 *
 * - `'visibility'` — `compute_nd_visibility_*` calls. Bounded by ndim,
 *   typically sub-second; uses `workerVisibilityTimeoutMs`.
 * - `'projection'` — `project*To3D` round-trips that include WASM
 *   visibility + compaction. Uses `workerProjectionTimeoutMs`.
 * - `'decode'` — `decode*` array-decode calls. Same magnitude as
 *   projection on large chunks; piggybacks on `workerProjectionTimeoutMs`
 *   for now.
 */
export type TimeoutKind = 'visibility' | 'projection' | 'decode';

/**
 * Thrown when a Comlink-routed worker call exceeds its configured
 * timeout. Carries the worker's pool index and the operation name so
 * callers can distinguish a hung worker from a genuine task failure.
 */
export class WorkerTimeoutError extends Error {
  constructor(
    public readonly operation: string,
    public readonly timeoutMs: number
  ) {
    super(`Worker call '${operation}' exceeded ${timeoutMs}ms timeout`);
    this.name = 'WorkerTimeoutError';
  }
}

/**
 * Optional override for the data-worker module URL.
 *
 * The default `new Worker(new URL('./data-worker.ts', import.meta.url))`
 * pattern works under Vite, Rollup, and webpack 5. Bundlers that don't
 * resolve `import.meta.url` for workers, or consumers that ship the
 * worker bundle from a non-default location, can call
 * {@link setDataWorkerUrl} once at startup with an explicit absolute URL.
 *
 * Set via `LuxarAppOptions.workerPath` from `LuxarApp.init`.
 */
let dataWorkerUrlOverride: string | undefined;

/**
 * Override the URL used to construct data workers. Pass an absolute URL
 * to a module-format worker bundle. Call before the first worker is created.
 */
export function setDataWorkerUrl(url: string): void {
  dataWorkerUrlOverride = url;
}

export class WorkerPool {
  private workers: WorkerInstance[] = [];
  private initPromise: Promise<void> | null = null;
  private nextWorkerIndex = 0;

  // Phase 15.1: dispose-mid-init defense. `initialize()` spawns
  // workers via `Array.from(...).map(async ...)` and pushes them to
  // `this.workers` only on the fulfilled branch of `Promise.allSettled`.
  // A `dispose()` between `new DataWorker()` and that push would walk
  // an empty `this.workers` and exit; the still-pending factories
  // would then resolve and re-populate `this.workers` post-dispose,
  // leaving live Worker globals the pool no longer references.
  //
  // Defense:
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
    const configCount = config.dataLoading.performance.workerCount;
    const hardwareConcurrency =
      typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4;

    // Leave one core for main thread (rendering, UI)
    const maxWorkers = Math.max(1, hardwareConcurrency - 1);

    // 0 = auto mode: use all available cores minus one
    if (configCount <= 0) {
      return maxWorkers;
    }

    // Otherwise use config value, capped at max
    return Math.min(configCount, maxWorkers);
  }

  /**
   * Initialize worker pool (lazy initialization)
   */
  async initialize(): Promise<void> {
    // Return existing promise if initialization already started or completed
    if (this.initPromise) return this.initPromise;

    // Phase 19.0.1: capture the generation token AND a per-attempt
    // workers list. Phase 15.1's `myGeneration` was correct for the
    // factory-level mismatch check, but the IIFE-level cleanup paths
    // unconditionally touched `this.workers` — so a stale init that
    // settled after a fresh init had published its workers would
    // wipe them. Per-attempt local state means stale completion can
    // only clean up its OWN workers, never globals.
    const myGeneration = ++this.initGeneration;
    const attemptWorkers: WorkerInstance[] = [];

    this.initPromise = (async () => {
      try {
        const workerCount = this.getConfiguredWorkerCount();
        log.info(Modules.WORKER_POOL, `Creating ${workerCount} data worker(s)...`);

        // Create all workers in parallel.
        // dataWorkerUrlOverride lets embedders whose bundlers don't support
        // vite's `?worker` import point at an explicitly-built worker bundle.
        const workerPromises = Array.from({ length: workerCount }, async (_, index) => {
          const worker = dataWorkerUrlOverride
            ? new Worker(dataWorkerUrlOverride, { type: 'module' })
            : new DataWorker();

          // Phase 15.1: track this worker as in-flight so a concurrent
          // dispose() can terminate it. Removed on success or on the
          // per-factory catch path.
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
            // Phase 15.1: if dispose() ran while we were awaiting init,
            // the generation has moved on. Self-terminate and reject so
            // the parent doesn't push us into the post-dispose pool.
            if (this.initGeneration !== myGeneration) {
              throw new Error(
                `Worker ${index + 1} aborted: pool was disposed during init`
              );
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

        // Phase 19.0.1: stale-generation guard. If dispose() bumped
        // the generation while we were awaiting allSettled, terminate
        // ONLY this attempt's workers and return. Do NOT touch
        // `this.workers` — a fresh generation may have already
        // published its own workers there.
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
        // Phase 19.0.1: same stale-generation guard for the error
        // path. If a newer generation has taken over, only clean up
        // this attempt's workers.
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
        // Phase 15.1: terminate anything still pending too, in case
        // the catch fires while factories are still settling.
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
    const timeoutMs = config.dataLoading.performance.workerInitTimeoutMs;
    return new Promise<void>((resolve, reject) => {
      let settled = false;
      const settle = (kind: 'ok' | 'err', err?: Error): void => {
        if (settled) return;
        settled = true;
        if (timer !== undefined) clearTimeout(timer);
        // Restore the permanent runtime handlers; the early ones above
        // are scoped to the init race only.
        this.attachWorkerErrorHandlers(worker, workerNumber);
        if (kind === 'ok') resolve();
        else reject(err);
      };
      // Mirror withTimeout() semantics: 0/negative/non-finite disables the
      // guard. `config/validation.ts` documents this convention for all
      // worker timeouts ("0 disables, but the guard is recommended"); the
      // pre-fix `setTimeout(..., 0)` instead fired on the next macrotask
      // and rejected real async inits immediately.
      const timer: ReturnType<typeof setTimeout> | undefined =
        timeoutMs > 0 && Number.isFinite(timeoutMs)
          ? setTimeout(() => {
              settle('err', new Error(`Worker ${workerNumber} init exceeded ${timeoutMs}ms`));
            }, timeoutMs)
          : undefined;
      // Override the permanent handlers for the duration of init so an
      // early failure (script load error, WASM init throw) rejects the
      // init promise rather than getting swallowed by the can't-find-
      // worker-in-pool branch of handleWorkerFailure.
      worker.onerror = (event) => {
        const message = event instanceof ErrorEvent ? event.message : 'unknown error';
        settle('err', new Error(`Worker ${workerNumber} runtime error during init: ${message}`));
        if (typeof event.preventDefault === 'function') event.preventDefault();
      };
      worker.onmessageerror = () => {
        settle('err', new Error(`Worker ${workerNumber} produced an unserializable message during init`));
      };
      api.initialize().then(
        () => settle('ok'),
        (err) => settle('err', err instanceof Error ? err : new Error(String(err)))
      );
    });
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
    worker.onerror = (event) => {
      const message = event instanceof ErrorEvent ? event.message : 'unknown error';
      log.error(Modules.WORKER_POOL, `Worker ${workerNumber} runtime error: ${message}`);
      this.handleWorkerFailure(worker, `runtime error: ${message}`);
      // Don't propagate to window.onerror — we've already logged it.
      if (typeof event.preventDefault === 'function') event.preventDefault();
    };
    worker.onmessageerror = () => {
      log.error(Modules.WORKER_POOL, `Worker ${workerNumber} produced an unserializable message`);
      this.handleWorkerFailure(worker, 'unserializable message');
    };
  }

  /**
   * Remove a failed worker from the pool and terminate it. If this drops
   * the pool to zero workers, log a clear error so the surrounding
   * application can decide whether to fall back to the main-thread
   * implementation or surface a failure dialog.
   */
  private handleWorkerFailure(worker: Worker, reason: string): void {
    const idx = this.workers.findIndex((w) => w.worker === worker);
    if (idx < 0) {
      // Already removed (idempotent on multiple error events).
      return;
    }
    this.workers.splice(idx, 1);
    try {
      worker.terminate();
    } catch {
      // Terminating a dead worker can throw on some browsers; swallow.
    }
    if (this.workers.length === 0) {
      log.error(
        Modules.WORKER_POOL,
        `All data workers failed (${reason}); subsequent calls will fail until reinitialization`
      );
      // Allow the next initialize() call to attempt a fresh pool.
      this.initPromise = null;
    } else {
      log.warning(
        Modules.WORKER_POOL,
        `Worker removed from pool (${reason}); ${this.workers.length} worker(s) remaining`
      );
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
    if (timeoutMs <= 0 || !Number.isFinite(timeoutMs)) {
      return call;
    }
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        log.error(
          Modules.WORKER_POOL,
          `Worker call '${operation}' timed out after ${timeoutMs}ms; evicting worker`
        );
        if (worker) {
          this.handleWorkerFailure(worker, `timeout(${operation}, ${timeoutMs}ms)`);
        }
        reject(new WorkerTimeoutError(operation, timeoutMs));
      }, timeoutMs);
    });
    return Promise.race([
      call.finally(() => {
        if (timer !== undefined) clearTimeout(timer);
      }),
      timeout,
    ]);
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
    const perf = config.dataLoading.performance;
    return kind === 'visibility'
      ? perf.workerVisibilityTimeoutMs
      : perf.workerProjectionTimeoutMs;
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
    fn: (api: Remote<DataWorkerAPI>) => Promise<T>
  ): Promise<T> {
    const wi = await this.nextWorkerInstance();
    return this.withTimeout(op, fn(wi.api), this.pickTimeoutMs(kind), wi.worker);
  }

  /**
   * Get a worker with query tracking for load balancing
   *
   * Returns the worker API and callbacks to mark query start/end.
   * This enables accurate load balancing across workers.
   */
  async getWorkerWithTracking(): Promise<{
    api: Remote<DataWorkerAPI>;
    markQueryStart: () => void;
    markQueryEnd: () => void;
  }> {
    await this.initialize();

    if (this.workers.length === 0) {
      throw new Error('[WorkerPool] No workers available after initialization');
    }

    // Find worker with least active queries
    let leastBusyIndex = 0;
    let minQueries = this.workers[0].activeQueries;

    for (let i = 1; i < this.workers.length; i++) {
      if (this.workers[i].activeQueries < minQueries) {
        minQueries = this.workers[i].activeQueries;
        leastBusyIndex = i;
      }
    }

    const workerInstance = this.workers[leastBusyIndex];

    return {
      api: workerInstance.api,
      markQueryStart: () => {
        workerInstance.activeQueries++;
      },
      markQueryEnd: () => {
        workerInstance.activeQueries = Math.max(0, workerInstance.activeQueries - 1);
      },
    };
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
  getStats(): { workerCount: number; activeQueries: number[] } {
    return {
      workerCount: this.workers.length,
      activeQueries: this.workers.map((w) => w.activeQueries),
    };
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
   * Phase 15.1: also terminates workers that are still pending init
   * via `pendingWorkers`, and bumps `initGeneration` so any factories
   * still in flight detect the dispose and self-terminate when they
   * resolve.
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

// Singleton instance
let workerPoolInstance: WorkerPool | null = null;

/**
 * Get the global worker pool instance
 */
export function getWorkerPool(): WorkerPool {
  if (!workerPoolInstance) {
    workerPoolInstance = new WorkerPool();
  }
  return workerPoolInstance;
}

/**
 * Dispose the global worker pool (for testing/cleanup)
 */
export function disposeWorkerPool(): void {
  if (workerPoolInstance) {
    workerPoolInstance.dispose();
    workerPoolInstance = null;
  }
}
