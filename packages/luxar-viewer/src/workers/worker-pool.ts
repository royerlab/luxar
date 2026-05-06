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

          // Install runtime-error handlers BEFORE the first message — a
          // worker can crash during its own boot sequence (e.g. WASM init
          // OOM), and we want those failures to be surfaced as worker
          // failures rather than uncaught browser-level errors.
          this.attachWorkerErrorHandlers(worker, index + 1);

          const api = wrap<DataWorkerAPI>(worker);

          try {
            await api.initialize();
            log.info(Modules.WORKER_POOL, `Worker ${index + 1}/${workerCount} ready`);
            return { worker, api, activeQueries: 0 };
          } catch (error) {
            log.error(Modules.WORKER_POOL, `Worker ${index + 1} initialization failed`, error);
            worker.terminate();
            throw error;
          }
        });

        // Wait for all workers to initialize
        const results = await Promise.allSettled(workerPromises);

        // Collect successful workers
        for (const result of results) {
          if (result.status === 'fulfilled') {
            this.workers.push(result.value);
          }
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
        // Clean up any workers that were partially pushed during this attempt
        for (const { worker } of this.workers) {
          worker.terminate();
        }
        this.workers = [];
        // Reset so callers can retry after transient failures
        this.initPromise = null;
        throw e;
      }
    })();

    return this.initPromise;
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
   * a per-call timeout is the right complement (added separately).
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
   * Get a worker API using round-robin selection.
   *
   * Untracked callers receive workers in rotating order. For load-aware
   * selection (picking the worker with the fewest active queries), use
   * {@link getWorkerWithTracking} instead.
   */
  async getWorker(): Promise<Remote<DataWorkerAPI>> {
    await this.initialize();

    if (this.workers.length === 0) {
      throw new Error('[WorkerPool] No workers available after initialization');
    }

    // Round-robin selection for untracked callers
    const worker = this.workers[this.nextWorkerIndex];
    this.nextWorkerIndex = (this.nextWorkerIndex + 1) % this.workers.length;
    return worker.api;
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
   * Clean up all worker resources
   */
  dispose(): void {
    if (this.workers.length > 0) {
      log.info(Modules.WORKER_POOL, `Terminating ${this.workers.length} data worker(s)`);
      for (const { worker } of this.workers) {
        worker.terminate();
      }
      this.workers = [];
      this.initPromise = null;
      this.nextWorkerIndex = 0;
    }
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
