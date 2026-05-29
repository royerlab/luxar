/**
 * Spawn a single data worker, install its error handlers, run the
 * init guard, and return the WorkerInstance. Lifted from the per-
 * factory function inside `WorkerPool.initialize()`.
 *
 * The caller passes:
 * - `dataWorkerCtor` for the default Vite `?worker` path
 * - `urlOverride` for the embed-bundle path (set via setDataWorkerUrl)
 * - `pendingWorkers` so dispose-mid-init can terminate this worker
 * - `attachPermanentHandlers` so handlers are installed before the
 *   first message (a worker can crash during its own boot)
 * - `runInitGuard` so the init promise is raced against a timeout
 *   + the worker's own onerror/onmessageerror
 * - `isCurrentGeneration` so a worker whose init completes after a
 *   concurrent dispose() self-terminates rather than re-populating
 *   a disposed pool
 */

import { log, Modules } from '../../../utils/log';
import { wrap, type Remote } from 'comlink';
import type { DataWorkerAPI, WorkerInitResult } from '../../data-worker';
import type { WorkerInstance } from '../types';

export interface SpawnWorkerOptions {
  index: number;
  total: number;
  dataWorkerCtor: new () => Worker;
  urlOverride: string | undefined;
  pendingWorkers: Set<Worker>;
  attachPermanentHandlers: (worker: Worker, workerNumber: number) => void;
  runInitGuard: (
    worker: Worker,
    api: Remote<DataWorkerAPI>,
    workerNumber: number
  ) => Promise<WorkerInitResult>;
  isCurrentGeneration: () => boolean;
}

export async function spawnWorker(opts: SpawnWorkerOptions): Promise<WorkerInstance> {
  const {
    index,
    total,
    dataWorkerCtor,
    urlOverride,
    pendingWorkers,
    attachPermanentHandlers,
    runInitGuard,
    isCurrentGeneration,
  } = opts;
  const workerNumber = index + 1;

  // The data-worker URL override (set via setDataWorkerUrl) lets embedders
  // whose bundlers don't support vite's `?worker` import point at an
  // explicitly-built worker bundle.
  const worker = urlOverride ? new Worker(urlOverride, { type: 'module' }) : new dataWorkerCtor();

  // Track this worker as in-flight so a concurrent dispose() can terminate it.
  // Removed on success or on the per-factory catch path.
  pendingWorkers.add(worker);

  // Install runtime-error handlers BEFORE the first message — a worker can
  // crash during its own boot sequence (e.g. WASM init OOM), and we want
  // those failures surfaced as worker failures rather than uncaught
  // browser-level errors.
  attachPermanentHandlers(worker, workerNumber);

  const api = wrap<DataWorkerAPI>(worker);

  try {
    // Race api.initialize() against:
    //   1. a hard init timeout (worker script blocked / unreachable →
    //      onerror may fire but Comlink's initialize() never settles
    //      because the worker never sent a message),
    //   2. an onerror short-circuit (worker fails *during* its boot
    //      before any pool entry exists for it).
    const initResult = await runInitGuard(worker, api, workerNumber);
    // If dispose() ran while we were awaiting init, the generation has moved
    // on. Self-terminate and reject so the parent doesn't push us into the
    // post-dispose pool.
    if (!isCurrentGeneration()) {
      throw new Error(`Worker ${workerNumber} aborted: pool was disposed during init`);
    }
    pendingWorkers.delete(worker);
    log.info(Modules.WORKER_POOL, `Worker ${workerNumber}/${total} ready`);
    // `?? false`: a mocked/legacy worker whose initialize() resolves void
    // counts as "WASM active" for the backend summary rather than crashing.
    return { worker, api, activeQueries: 0, wasmFallback: initResult?.wasmFallback ?? false };
  } catch (error) {
    log.error(Modules.WORKER_POOL, `Worker ${workerNumber} initialization failed`, error);
    pendingWorkers.delete(worker);
    worker.terminate();
    throw error;
  }
}

/**
 * Terminate every worker in `attemptWorkers`. Used by the stale-
 * generation cleanup branches of `WorkerPool.initialize()`: when
 * `dispose()` bumps the generation while an init is in flight, the
 * stale init's surviving workers are NOT pushed onto `this.workers`
 * — they're terminated via this helper instead.
 *
 * Errors from `terminate()` are swallowed because the dispose path
 * may have already terminated these workers via `pendingWorkers`.
 */
export function terminateAttemptWorkers(attemptWorkers: WorkerInstance[]): void {
  for (const { worker } of attemptWorkers) {
    try {
      worker.terminate();
    } catch {
      // Already terminated by dispose's pendingWorkers walk.
    }
  }
}
