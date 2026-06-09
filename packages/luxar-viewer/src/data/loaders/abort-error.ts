/**
 * Shared classifier for the "this load was abandoned on purpose" errors.
 *
 * A superseded `updateView` (a newer view-state arrived) aborts the in-flight
 * read via the per-update `AbortSignal`. Two error shapes result:
 *   - zarrita's chunk reads reject with a DOMException named `'AbortError'`
 *     (instanceof Error in Node and modern browsers);
 *   - the worker pool throws a `WorkerAbortError`.
 *
 * Both mean "superseded", NOT "failed": callers use this to keep supersedes out
 * of failure bookkeeping (`failedLoaders`, prefetch-baseline drop) AND out of
 * loader-level error telemetry (`metrics.errors`, the monitor `'error'` event
 * stream) — otherwise rapid scrubbing, where aborts are a frequent intentional
 * control path, would flood both with non-errors.
 */
export function isAbortError(error: unknown): boolean {
  return (
    error instanceof Error && (error.name === 'AbortError' || error.name === 'WorkerAbortError')
  );
}
