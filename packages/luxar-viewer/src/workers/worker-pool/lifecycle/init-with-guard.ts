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

import { WorkerInitTimeoutError } from '../errors';

/**
 * The only shape this guard needs from a Comlink-wrapped worker API.
 * Structural on purpose: the data worker and the SORT worker both expose
 * an `initialize(wasmPath?)`, and sharing this guard is what keeps their
 * startup semantics from drifting apart (the sort coordinator used to
 * hand-roll its own race, and its copy never grew the `onmessageerror`
 * arm or the `preventDefault`).
 */
export interface GuardedInitApi<TResult> {
  /**
   * `wasmModule` is an already-compiled module the caller shares across
   * workers (see `wasm/shared-module.ts`); omitted, the worker compiles its
   * own. Declaring it here is safe for the SORT worker too: a function with
   * fewer parameters still satisfies a wider signature in TypeScript, so its
   * 1-argument `initialize` continues to match.
   */
  initialize(wasmPath?: string, wasmModule?: WebAssembly.Module): Promise<TResult>;
}

/**
 * `label` names the worker in every rejection message — the pool passes
 * `Worker <n>`, the sort coordinator passes `SortWorker`.
 *
 * The timeout arm rejects with {@link WorkerInitTimeoutError}, the other two
 * with a plain `Error`. That split is load-bearing for callers that retry:
 * see the error's own doc for why a deadline miss is not evidence of a
 * broken worker.
 */
export function initializeWithGuard<TResult>(
  worker: Worker,
  api: GuardedInitApi<TResult>,
  label: string,
  timeoutMs: number,
  attachPermanentHandlers: () => void,
  wasmPath?: string,
  wasmModule?: WebAssembly.Module
): Promise<TResult> {
  return new Promise<TResult>((resolve, reject) => {
    let settled = false;
    const settle = (kind: 'ok' | 'err', payload?: TResult | Error): void => {
      if (settled) return;
      settled = true;
      if (timer !== undefined) clearTimeout(timer);
      // Restore the permanent runtime handlers; the early ones below
      // are scoped to the init race only.
      attachPermanentHandlers();
      if (kind === 'ok') resolve(payload as TResult);
      else reject(payload as Error);
    };
    // Mirror withTimeout() semantics: 0/negative/non-finite disables the
    // guard. `config/sections/data-loading/performance/validate.ts`
    // documents this convention for worker init timeouts ("0 disables, but
    // the guard is recommended"); the
    // pre-fix `setTimeout(..., 0)` instead fired on the next macrotask
    // and rejected real async inits immediately.
    const timer: ReturnType<typeof setTimeout> | undefined =
      timeoutMs > 0 && Number.isFinite(timeoutMs)
        ? setTimeout(() => {
            settle('err', new WorkerInitTimeoutError(label, timeoutMs));
          }, timeoutMs)
        : undefined;
    // Override the permanent handlers for the duration of init so an
    // early failure (script load error, WASM init throw) rejects the
    // init promise rather than getting swallowed by the can't-find-
    // worker-in-pool branch of handleWorkerFailure.
    worker.onerror = (event) => {
      // Duck-typed rather than `event instanceof ErrorEvent`, for the same
      // reason `preventDefault` below is: that global does not exist in every
      // host this runs in (the unit suite's default `node` environment, for
      // one), and a bare reference to a missing global throws a
      // ReferenceError OUT of the handler — leaving the init promise unsettled,
      // which is precisely what this guard exists to prevent.
      const message =
        typeof (event as { message?: unknown }).message === 'string'
          ? (event as { message: string }).message
          : 'unknown error';
      settle('err', new Error(`${label} runtime error during init: ${message}`));
      if (typeof event.preventDefault === 'function') event.preventDefault();
    };
    worker.onmessageerror = () => {
      settle('err', new Error(`${label} produced an unserializable message during init`));
    };
    api.initialize(wasmPath, wasmModule).then(
      (result) => settle('ok', result),
      (err) => settle('err', err instanceof Error ? err : new Error(String(err)))
    );
  });
}
