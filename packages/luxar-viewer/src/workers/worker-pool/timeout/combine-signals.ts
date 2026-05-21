/**
 * Combine two optional AbortSignals into a single one that fires when
 * either trips. Pure helper lifted from `WorkerPool.combineSignals`.
 *
 * Uses native `AbortSignal.any` when available (modern browsers /
 * Node 20+) and falls back to a manual "trip either one" wiring for
 * older runtimes.
 */
export function combineSignals(
  a: AbortSignal | undefined,
  b: AbortSignal | undefined
): AbortSignal | undefined {
  if (!a && !b) return undefined;
  if (a && !b) return a;
  if (!a && b) return b;
  // Both present. `AbortSignal.any` is the standard combinator; fall
  // back to manual wiring when unavailable.
  const anyFn = (AbortSignal as unknown as { any?: (signals: AbortSignal[]) => AbortSignal }).any;
  if (typeof anyFn === 'function') {
    return anyFn([a as AbortSignal, b as AbortSignal]);
  }
  const controller = new AbortController();
  const forward = (): void => controller.abort();
  (a as AbortSignal).addEventListener('abort', forward, { once: true });
  (b as AbortSignal).addEventListener('abort', forward, { once: true });
  if ((a as AbortSignal).aborted || (b as AbortSignal).aborted) controller.abort();
  return controller.signal;
}
