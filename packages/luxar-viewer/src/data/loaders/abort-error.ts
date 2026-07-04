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
 *
 * Classified by `.name` alone — deliberately NOT `instanceof Error`. A
 * `DOMException` created in another realm (jsdom test env, iframe, a worker
 * error surface) fails a same-realm `instanceof Error` check even though its
 * own prototype chain contains that realm's Error, silently turning an
 * intentional abort into a recorded failure. Duck-typing the name is
 * realm-proof, and anything carrying `name === 'AbortError'` is precisely
 * what this classifier exists to match.
 */
export function isAbortError(error: unknown): boolean {
  if (typeof error !== 'object' || error === null) return false;
  const name = (error as { name?: unknown }).name;
  return name === 'AbortError' || name === 'WorkerAbortError';
}
