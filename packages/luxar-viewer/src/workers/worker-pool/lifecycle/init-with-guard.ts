/**
 * Per-worker init guard. Races `api.initialize()` against
 * (1) a hard timeout and (2) the worker's own `onerror` /
 * `onmessageerror` events.
 *
 * Lifted from `WorkerPool.initializeWithGuard`. The pool's permanent
 * `attachWorkerErrorHandlers` evicts a failed worker via
 * `handleWorkerFailure`, but during pool init the worker isn't in the
 * pool yet — so `handleWorkerFailure` finds nothing to evict and the
 * dangling Comlink `initialize()` promise never settles. (Repro:
 * route-block the worker script in a Playwright test; the page hangs
 * at boot.)
 *
 * We attach short-lived listeners that reject the init promise when
 * the worker fails before joining the pool. The init timeout is the
 * belt-and-suspenders fallback — even if no error event fires (e.g.
 * a network stall that never resolves), we eventually reject and let
 * the caller fall back.
 *
 * `attachPermanentHandlers` is invoked on settle so the pool can
 * restore its long-lived handlers; this preserves the
 * init-race-vs-permanent-handler swap from the original.
 */

import type { Remote } from 'comlink';
import type { DataWorkerAPI, WorkerInitResult } from '../../data-worker';

export function initializeWithGuard(
  worker: Worker,
  api: Remote<DataWorkerAPI>,
  workerNumber: number,
  timeoutMs: number,
  attachPermanentHandlers: () => void,
  wasmPath?: string
): Promise<WorkerInitResult> {
  return new Promise<WorkerInitResult>((resolve, reject) => {
    let settled = false;
    const settle = (kind: 'ok' | 'err', payload?: WorkerInitResult | Error): void => {
      if (settled) return;
      settled = true;
      if (timer !== undefined) clearTimeout(timer);
      // Restore the permanent runtime handlers; the early ones below
      // are scoped to the init race only.
      attachPermanentHandlers();
      if (kind === 'ok') resolve(payload as WorkerInitResult);
      else reject(payload as Error);
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
      settle(
        'err',
        new Error(`Worker ${workerNumber} produced an unserializable message during init`)
      );
    };
    api.initialize(wasmPath).then(
      (result) => settle('ok', result),
      (err) => settle('err', err instanceof Error ? err : new Error(String(err)))
    );
  });
}
