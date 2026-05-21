/**
 * Pure timeout-race helper. Lifted from `WorkerPool.withTimeout` so
 * the pool's class method becomes a 1-line delegate.
 *
 * On timeout the supplied `onTimeoutEvict` callback (optional) is
 * invoked so the calling pool can prune the responsible worker.
 * Returning a rejection of `WorkerTimeoutError` preserves the public
 * contract.
 */

import { log, Modules } from '../../../utils/log';
import { WorkerTimeoutError } from '../errors';

/**
 * Race `call` against a `timeoutMs`-budget timer. Returns the
 * call's result on success, rejects with {@link WorkerTimeoutError}
 * on timeout, and skips the race entirely when `timeoutMs <= 0` or
 * is non-finite (acts as a transparent pass-through for callers that
 * still want the uniform call shape).
 *
 * @param operation Name used in log + error messages.
 * @param call The promise to race against the timer.
 * @param timeoutMs Budget in ms; `<= 0` or non-finite disables.
 * @param onTimeoutEvict Optional callback fired on timeout with the
 *   `(worker, reason)` pair the pool's `handleWorkerFailure` expects.
 *   Omitted when the caller can't identify the responsible worker.
 * @param worker Optional handle on the responsible worker; passed
 *   through to `onTimeoutEvict` only.
 */
export function withTimeout<T>(
  operation: string,
  call: Promise<T>,
  timeoutMs: number,
  onTimeoutEvict?: (worker: Worker, reason: string) => void,
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
      if (worker && onTimeoutEvict) {
        onTimeoutEvict(worker, `timeout(${operation}, ${timeoutMs}ms)`);
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
