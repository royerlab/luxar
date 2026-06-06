/**
 * Worker pool manager for data processing workers
 *
 * Supports multiple workers with round-robin load balancing for parallel
 * spatial queries across multiple nodes.
 */

import type { Remote } from 'comlink';
import type { DataWorkerAPI, WorkerInitResult } from './data-worker';
// Vite's `?worker` import emits a bundled, transpiled, hashed worker chunk and
// returns a default-exported Worker constructor. This sidesteps the broken
// `new Worker(new URL('./data-worker.ts', import.meta.url))` pattern under
// vite 8 + rolldown rc.17, which ships the raw `.ts` source as an asset.
import DataWorker from './data-worker?worker';
import { log, Modules } from '../utils/log';
import { config } from '../config';
import type { WorkerInstance } from './worker-pool/types';
import { WorkerTimeoutError, WorkerAbortError, type TimeoutKind } from './worker-pool/errors';

import { withTimeout } from './worker-pool/timeout/with-timeout';
import { combineSignals } from './worker-pool/timeout/combine-signals';
import { pickTimeoutMs } from './worker-pool/timeout/pick-timeout-ms';
import { getConfiguredWorkerCount } from './worker-pool/lifecycle/worker-count';
import { initializeWithGuard } from './worker-pool/lifecycle/init-with-guard';
import { spawnWorker, terminateAttemptWorkers } from './worker-pool/lifecycle/spawn-worker';
import {
  attachWorkerErrorHandlers,
  evictFailedWorker,
} from './worker-pool/lifecycle/error-handlers';
import { selectLeastBusy, type TrackedWorkerHandle } from './worker-pool/selection/least-busy';
import { nextRoundRobin } from './worker-pool/selection/round-robin';
import { computeStats, computeQueueDepth, type PoolStats } from './worker-pool/stats';

export type { WorkerInstance };
export { WorkerTimeoutError, WorkerAbortError };
export type { TimeoutKind };

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

/**
 * WASM JS-shim URL forwarded into each worker's `initialize()` so the worker
 * loads a relocated WASM binary instead of silently using the slow TS fallback.
 *
 * The main thread's {@link setWasmJsUrl} override lives in a different module
 * instance than the worker's, so it does NOT cross the worker boundary on its
 * own — this value is sent over the Comlink `initialize` RPC instead. Set via
 * `LuxarAppOptions.wasmPath` from `applyModuleOverrides`.
 */
let dataWorkerWasmPathOverride: string | undefined;

/**
 * Override the WASM JS-shim URL used inside data workers. Pass an absolute URL.
 * Call before the first worker is created.
 */
export function setDataWorkerWasmPath(url: string): void {
  dataWorkerWasmPathOverride = url;
}

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

        // Spawn every worker in parallel. The URL override is resolved lazily
        // (inside spawnWorker via the captured value) so setDataWorkerUrl
        // calls made before the first getWorkerPool() take effect.
        const urlOverride = dataWorkerUrlOverride;
        const wasmPathOverride = dataWorkerWasmPathOverride;
        const workerPromises = Array.from({ length: workerCount }, (_, index) =>
          spawnWorker({
            index,
            total: workerCount,
            dataWorkerCtor: DataWorker,
            urlOverride,
            pendingWorkers: this.pendingWorkers,
            attachPermanentHandlers: (w, n) => this.attachWorkerErrorHandlers(w, n),
            runInitGuard: (w, a, n) => this.initializeWithGuard(w, a, n, wasmPathOverride),
            isCurrentGeneration: () => this.initGeneration === myGeneration,
          })
        );

        // Wait for all workers to initialize
        const results = await Promise.allSettled(workerPromises);

        // Collect successful workers into the attempt-local list first, so
        // stale-generation cleanup doesn't reach for any newer attempt's
        // published workers.
        for (const result of results) {
          if (result.status === 'fulfilled') {
            attemptWorkers.push(result.value);
          }
        }

        // Stale-generation guard. If dispose() bumped the generation while
        // we were awaiting allSettled, terminate ONLY this attempt's workers
        // and return. Do NOT touch `this.workers` — a fresh generation may
        // have already published its own workers there.
        if (this.initGeneration !== myGeneration) {
          terminateAttemptWorkers(attemptWorkers);
          return;
        }

        // Generation still current — publish.
        for (const entry of attemptWorkers) {
          this.workers.push(entry);
        }

        if (this.workers.length === 0) {
          // MED-22: `Promise.allSettled(workerPromises)` above blocks until
          // every spawnWorker() has either resolved or rejected, and
          // spawnWorker removes its worker from `pendingWorkers` in BOTH
          // its success and catch paths. Under normal flow `pendingWorkers`
          // is therefore empty here. Sweep defensively anyway to guarantee
          // no orphan Workers leak out the throw — if a future spawnWorker
          // refactor ever forgets the .delete() on some path, this keeps
          // the contract ("no pending workers escape the empty-pool guard")
          // intact rather than silently leaking.
          for (const worker of this.pendingWorkers) {
            try {
              worker.terminate();
            } catch {
              // Already terminated by a concurrent dispose / spawn path.
            }
          }
          this.pendingWorkers.clear();
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

        // One-time backend summary (replaces the per-worker WASM/TS lines).
        // Workers each load WASM independently but share one build, so the
        // pool is either all-WASM or all-fallback in practice; report a mixed
        // result honestly if that invariant ever breaks.
        const fallbackCount = this.workers.filter((w) => w.wasmFallback).length;
        if (fallbackCount === 0) {
          log.success(
            Modules.WORKER_POOL,
            `Acceleration: compiled WASM active on all ${this.workers.length} data worker(s)`
          );
        } else if (fallbackCount === this.workers.length) {
          log.warning(
            Modules.WORKER_POOL,
            `Acceleration: TypeScript fallback on all ${this.workers.length} data worker(s) — ` +
              'compiled WASM not loaded (run "make build-wasm"; functional but slower)'
          );
        } else {
          log.warning(
            Modules.WORKER_POOL,
            `Acceleration: mixed backend — ${this.workers.length - fallbackCount} on WASM, ` +
              `${fallbackCount} on TypeScript fallback`
          );
        }
      } catch (e) {
        // Same stale-generation guard for the error path. If a newer
        // generation has taken over, only clean up this attempt's workers.
        if (this.initGeneration !== myGeneration) {
          terminateAttemptWorkers(attemptWorkers);
          throw e;
        }
        // Generation current — full cleanup of this generation's state.
        for (const { worker } of this.workers) {
          worker.terminate();
        }
        this.workers = [];
        // Terminate anything still pending too, in case the catch fires
        // while factories are still settling.
        for (const worker of this.pendingWorkers) {
          worker.terminate();
        }
        this.pendingWorkers.clear();
        // Note: we deliberately keep `initPromise` (the rejected one) so
        // subsequent `getWorker()` / `runWithTimeout` calls fail FAST rather
        // than re-running the 10s init guard for every nD load. A blocked
        // worker chunk would otherwise stack 10s × N delays and blow past
        // the page's `waitForLuxarReady` timeout. To opt back in to a fresh
        // init attempt (e.g. after a transient network blip), call
        // `reinitialize()`.
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
    workerNumber: number,
    wasmPath?: string
  ): Promise<WorkerInitResult> {
    return initializeWithGuard(
      worker,
      api,
      workerNumber,
      config.dataLoading.performance.workerInitTimeoutMs,
      () => this.attachWorkerErrorHandlers(worker, workerNumber),
      wasmPath
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

    const { instance, nextIndex } = nextRoundRobin(this.workers, this.nextWorkerIndex);
    this.nextWorkerIndex = nextIndex;
    return instance;
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
  getStats(): PoolStats {
    return computeStats(this.workers);
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
    return computeQueueDepth(this.workers);
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
