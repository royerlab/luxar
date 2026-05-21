/**
 * Worker-level error/messageerror plumbing + the generic eviction
 * primitive used by both the permanent handlers and the timeout path.
 *
 * Lifted from `WorkerPool.attachWorkerErrorHandlers` and the generic
 * find/splice/terminate body of `WorkerPool.handleWorkerFailure`. The
 * pool keeps the "should I null out initPromise?" decision because
 * that depends on its own private state — only the body that walks
 * the workers list and terminates the matching entry moves here.
 */

import { log, Modules } from '../../../utils/log';
import type { WorkerInstance } from '../types';

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
 * route hot paths through `runWithTimeout` for the per-call timeout
 * that complements this handler.
 *
 * @param onFailure Called with `(worker, reason)` so the pool can run
 *   its eviction + post-eviction state mutation.
 */
export function attachWorkerErrorHandlers(
  worker: Worker,
  workerNumber: number,
  onFailure: (worker: Worker, reason: string) => void
): void {
  worker.onerror = (event) => {
    const message = event instanceof ErrorEvent ? event.message : 'unknown error';
    log.error(Modules.WORKER_POOL, `Worker ${workerNumber} runtime error: ${message}`);
    onFailure(worker, `runtime error: ${message}`);
    // Don't propagate to window.onerror — we've already logged it.
    if (typeof event.preventDefault === 'function') event.preventDefault();
  };
  worker.onmessageerror = () => {
    log.error(Modules.WORKER_POOL, `Worker ${workerNumber} produced an unserializable message`);
    onFailure(worker, 'unserializable message');
  };
}

/**
 * Outcome of {@link evictFailedWorker}: `'idempotent'` if the
 * worker wasn't in the pool (already removed), `'evicted'` if it
 * was removed and the pool still has live workers, `'pool-empty'`
 * if the eviction left the pool with zero workers.
 */
export type EvictOutcome = 'idempotent' | 'evicted' | 'pool-empty';

/**
 * Find a worker in `workers`, splice it out, terminate it, and log
 * an appropriate severity message. Returns an outcome the caller can
 * switch on for post-eviction state mutation (e.g. clearing the cached
 * init promise when the pool drops to zero).
 *
 * Idempotent on repeated calls with the same worker: a worker that is
 * no longer in the list returns `'idempotent'` without further side
 * effects. Termination is guarded so a double-terminate on browsers
 * that throw is swallowed.
 */
export function evictFailedWorker(
  workers: WorkerInstance[],
  worker: Worker,
  reason: string
): EvictOutcome {
  const idx = workers.findIndex((w) => w.worker === worker);
  if (idx < 0) {
    // Already removed (idempotent on multiple error events).
    return 'idempotent';
  }
  workers.splice(idx, 1);
  try {
    worker.terminate();
  } catch {
    // Terminating a dead worker can throw on some browsers; swallow.
  }
  if (workers.length === 0) {
    log.error(
      Modules.WORKER_POOL,
      `All data workers failed (${reason}); subsequent calls will fail until reinitialization`
    );
    return 'pool-empty';
  }
  log.warning(
    Modules.WORKER_POOL,
    `Worker removed from pool (${reason}); ${workers.length} worker(s) remaining`
  );
  return 'evicted';
}
