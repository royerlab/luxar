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
 * True only for failures of the worker INFRASTRUCTURE — the pool could not
 * spawn a worker, had none left, or a worker hung past its timeout.
 *
 * This is the only class of failure for which re-running the same work on the
 * main thread is meaningful. A rejection that came back *from* a worker means
 * the kernel itself rejected the data, and the in-process dispatchers run the
 * SAME kernel via the same `pickBackend` — so retrying there would fail
 * identically, except on the UI thread, where a WASM trap blocks the frame
 * instead of a background one.
 *
 * Deliberately an allow-list, not a deny-list: the pool's own error types are a
 * closed, greppable set, whereas "every way a kernel can fail" is not. An
 * unrecognized error therefore does NOT qualify — callers fail closed and
 * propagate it.
 *
 * Name-based rather than `instanceof`, matching `data/loaders/abort-error.ts`:
 * these errors cross a Comlink boundary, which reconstructs them in the calling
 * realm and breaks prototype identity.
 */
export function isWorkerInfrastructureError(error: unknown): boolean {
  if (typeof error !== 'object' || error === null) return false;
  const name = (error as { name?: unknown }).name;
  return name === 'WorkerTimeoutError' || name === 'WorkerUnavailableError';
}
