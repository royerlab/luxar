/**
 * Module-level singleton state + accessors for the worker pool: the
 * lazily-constructed {@link WorkerPool} instance and the
 * {@link setDataWorkerUrl} override that some embedders need before
 * the first worker is constructed.
 *
 * `worker-pool.ts` re-exports `getWorkerPool`, `disposeWorkerPool`,
 * and `setDataWorkerUrl` so external callsites keep their import
 * paths. The {@link getDataWorkerUrlOverride} accessor stays
 * internal — `worker-pool.ts::initialize()` calls it lazily inside
 * the per-factory function, so the ES-module circular reference
 * (singleton.ts -> WorkerPool from ../worker-pool, worker-pool.ts ->
 * getDataWorkerUrlOverride from ./singleton) resolves cleanly.
 */

import { WorkerPool } from '../worker-pool';

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
 * Internal accessor read by `WorkerPool.initialize()` inside its
 * per-factory function. Kept module-private (no `export` re-emit
 * from `worker-pool.ts`) because production code talks to the
 * override only through {@link setDataWorkerUrl}.
 */
export function getDataWorkerUrlOverride(): string | undefined {
  return dataWorkerUrlOverride;
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
