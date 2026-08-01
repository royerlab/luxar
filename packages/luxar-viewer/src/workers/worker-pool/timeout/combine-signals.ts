const NOOP_DISPOSE = (): void => {};

/** A combined abort signal plus the cleanup for fallback source listeners. */
export interface CombinedSignalScope {
  signal: AbortSignal;
  dispose: () => void;
}

/**
 * Combine two optional AbortSignals into a single scoped one that fires
 * when either trips. Pure helper lifted from `WorkerPool.combineSignals`.
 *
 * Uses native `AbortSignal.any` when available (modern browsers /
 * Node 20+) and falls back to a manual "trip either one" wiring for
 * older runtimes. The fallback registers listeners on both sources, so
 * callers must invoke `dispose()` when the call using `signal` settles —
 * the pool-wide signal lives for a whole dataset session, and without
 * disposal every worker call would retain one relay closure on it. An
 * abort removes both source listeners immediately before relaying.
 */
export function combineSignals(
  a: AbortSignal | undefined,
  b: AbortSignal | undefined
): CombinedSignalScope | undefined {
  if (!a && !b) return undefined;
  if (a && !b) return { signal: a, dispose: NOOP_DISPOSE };
  if (!a && b) return { signal: b, dispose: NOOP_DISPOSE };
  // Both present. `AbortSignal.any` is the standard combinator; fall
  // back to manual wiring when unavailable.
  const anyFn = (AbortSignal as unknown as { any?: (signals: AbortSignal[]) => AbortSignal }).any;
  if (typeof anyFn === 'function') {
    return { signal: anyFn([a as AbortSignal, b as AbortSignal]), dispose: NOOP_DISPOSE };
  }
  const controller = new AbortController();
  let listening = false;
  const dispose = (): void => {
    if (!listening) return;
    listening = false;
    (a as AbortSignal).removeEventListener('abort', forward);
    (b as AbortSignal).removeEventListener('abort', forward);
  };
  const forward = (): void => {
    dispose();
    controller.abort();
  };
  if ((a as AbortSignal).aborted || (b as AbortSignal).aborted) {
    controller.abort();
  } else {
    listening = true;
    (a as AbortSignal).addEventListener('abort', forward, { once: true });
    (b as AbortSignal).addEventListener('abort', forward, { once: true });
  }
  return { signal: controller.signal, dispose };
}
