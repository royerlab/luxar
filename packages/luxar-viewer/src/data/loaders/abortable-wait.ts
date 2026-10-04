/**
 * Await a promise SHARED between callers while letting each caller give up on
 * its own signal.
 *
 * A single-flight fetch (one in-flight promise many callers join) must not run
 * under any one caller's signal: a superseded first caller would abort the work
 * every joiner is waiting on. The shared work runs under its owner's lifetime
 * signal instead, and each caller waits through this helper, which rejects with
 * the caller's abort reason the moment that caller's signal fires while leaving
 * the shared promise untouched.
 *
 * Rejects synchronously-ish (on the next microtask) for an already-aborted
 * signal, and detaches its listener once the shared promise settles, so a
 * long-lived signal does not accumulate listeners across waits.
 */
export function abortableWait<T>(shared: Promise<T>, signal?: AbortSignal | null): Promise<T> {
  if (!signal) return shared;
  if (signal.aborted) return Promise.reject(abortReason(signal));
  return new Promise<T>((resolve, reject) => {
    const onAbort = (): void => reject(abortReason(signal));
    signal.addEventListener('abort', onAbort, { once: true });
    shared.then(
      (value) => {
        signal.removeEventListener('abort', onAbort);
        resolve(value);
      },
      (error: unknown) => {
        signal.removeEventListener('abort', onAbort);
        reject(error as Error);
      }
    );
  });
}

// lifecycle-exempt: maps an aborted signal to the error to throw; starts and owns no work
/** The signal's reason when it is an Error, else a standard `AbortError`. */
export function abortReason(signal: AbortSignal): Error {
  const reason: unknown = signal.reason;
  if (reason instanceof Error) return reason;
  return new DOMException('The operation was aborted', 'AbortError');
}
