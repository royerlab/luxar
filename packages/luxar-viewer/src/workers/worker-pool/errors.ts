/**
 * Pool-internal error types + the {@link TimeoutKind} discriminator
 * that selects per-call timeouts from config. Externally re-exported
 * from `worker-pool.ts` so consumers keep their existing import paths.
 */

/**
 * Class of worker call, used to pick a default timeout from config.
 *
 * - `'projection'` — `project*To3D` round-trips that include WASM
 *   visibility + compaction. Uses `workerProjectionTimeoutMs`.
 * - `'decode'` — `decode*` array-decode calls. Same magnitude as
 *   projection on large chunks; piggybacks on `workerProjectionTimeoutMs`
 *   for now.
 */
export type TimeoutKind = 'projection' | 'decode';

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
 * Thrown when a worker call is aborted via the caller-supplied
 * AbortSignal — either because the signal was already aborted when
 * `runWithTimeout` was called or because it fired before the worker
 * task settled.
 *
 * IMPORTANT: aborting does NOT actually cancel work running inside
 * the WASM kernel (WebAssembly has no cancellation primitive). The
 * abort only:
 * (a) Rejects the promise immediately so the caller stops awaiting.
 * (b) Skips dispatch entirely if the signal was already aborted on entry.
 *
 * The worker continues processing the now-orphan task to completion;
 * its result is discarded. This is the same trade-off as `fetch`
 * + `AbortSignal`: the request may keep flying on the wire, but the
 * caller has moved on.
 */
export class WorkerAbortError extends Error {
  constructor(public readonly operation: string) {
    super(`Worker call '${operation}' aborted by caller signal`);
    this.name = 'WorkerAbortError';
  }
}

/**
 * Thrown when the pool has no usable worker to dispatch to — either
 * initialization produced none at all, or every worker was disposed. Unlike
 * {@link WorkerTimeoutError} this says nothing about the task; the work was
 * never handed to a worker in the first place.
 */
export class WorkerUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WorkerUnavailableError';
  }
}

/**
 * True only for an ESTABLISHED failure of the worker infrastructure — the pool
 * could not spawn a worker, or had none left. The work never reached a kernel,
 * so re-running it on the main thread is the only executor left and cannot
 * reproduce a kernel fault.
 *
 * Everything else fails closed and propagates:
 *
 * - A rejection that came back *from* a worker means the kernel itself rejected
 *   the data, and the in-process dispatchers run the SAME kernel via the same
 *   `pickBackend` — retrying there would fail identically, except on the UI
 *   thread, where a WASM trap blocks the frame instead of a background one.
 * - {@link WorkerTimeoutError} is deliberately NOT included. A timeout does not
 *   establish infrastructure failure: it cannot distinguish a wedged worker
 *   from a data-dependent kernel hang or a projection genuinely slower than
 *   `workerProjectionTimeoutMs` — and in the latter two cases re-running on
 *   the main thread blocks the frame for at least as long again (indefinitely,
 *   for a hang). Guessing wrong costs exactly the UI freeze this predicate
 *   exists to prevent. The pool already evicts the timed-out worker
 *   (`handleWorkerFailure`), so propagating leaves the node failed-but-
 *   retryable against a fresh worker rather than freezing the frame.
 *
 * Deliberately an allow-list, not a deny-list: the pool's own error types are a
 * closed, greppable set, whereas "every way a kernel can fail" is not. An
 * unrecognized error therefore does NOT qualify — callers fail closed and
 * propagate it.
 *
 * Matched by `instanceof`, not by name: `WorkerUnavailableError` is constructed
 * only on the main thread (pool init / worker selection) and never crosses the
 * Comlink boundary, so its prototype is intact here. A rejection that came back
 * FROM a worker is reconstructed in this realm and loses its prototype, so it
 * can never match — which is exactly the fail-closed behavior we want for a
 * kernel fault.
 */
export function isWorkerInfrastructureError(error: unknown): boolean {
  return error instanceof WorkerUnavailableError;
}
