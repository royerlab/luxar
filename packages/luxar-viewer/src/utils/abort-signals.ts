/**
 * The viewer's one abort-signal combinator.
 *
 * Three copies of "abort when either signal aborts" used to live in the fetch
 * retry loop, the worker pool and the zip reader, and they had drifted: one
 * leaked its listeners, two dropped the abort reason. This is the merge they
 * all use now.
 *
 * @module utils/abort-signals
 */

/** A combined signal plus the cleanup for any fallback listeners it registered. */
export interface AbortSignalScope {
  readonly signal: AbortSignal;
  /** Remove the fallback's source listeners; idempotent, a no-op natively. */
  dispose(): void;
}

const NOOP_DISPOSE = (): void => {};

type AbortSignalAny = (signals: AbortSignal[]) => AbortSignal;

/** The two-signal merge, either signal present. */
function mergeBoth(a: AbortSignal, b: AbortSignal): AbortSignalScope {
  const any = (AbortSignal as unknown as { any?: AbortSignalAny }).any;
  if (typeof any === 'function') return { signal: any([a, b]), dispose: NOOP_DISPOSE };

  // Fallback relay for runtimes without `AbortSignal.any`. It registers a
  // listener on both sources, so the caller disposes the scope when the work
  // using `signal` settles — a session-lived source (a store's or the worker
  // pool's) would otherwise collect one closure per call.
  const relay = new AbortController();
  const reasonOf = (): unknown => (a.aborted ? a.reason : b.reason);
  if (a.aborted || b.aborted) {
    relay.abort(reasonOf());
    return { signal: relay.signal, dispose: NOOP_DISPOSE };
  }
  let listening = true;
  const dispose = (): void => {
    if (!listening) return;
    listening = false;
    a.removeEventListener('abort', onAbort);
    b.removeEventListener('abort', onAbort);
  };
  const onAbort = (): void => {
    dispose();
    relay.abort(reasonOf());
  };
  a.addEventListener('abort', onAbort, { once: true });
  b.addEventListener('abort', onAbort, { once: true });
  return { signal: relay.signal, dispose };
}

/** What {@link combineAbortSignals} returns: a scope whenever `a` is a signal. */
export type CombinedAbortScope<A extends AbortSignal | undefined> = A extends AbortSignal
  ? AbortSignalScope
  : AbortSignalScope | undefined;

/**
 * A signal that aborts when `a` or `b` does, carrying the reason of whichever
 * fired first (an already-aborted input aborts it at once).
 *
 * Uses native `AbortSignal.any` when available; the fallback relay registers
 * listeners on both inputs, so call `dispose()` once the work using `signal`
 * has settled (always safe to call). With one input that input is returned
 * as is; with none the result is `undefined` (so a defined `a` always gives a
 * scope, which the return type says).
 */
export function combineAbortSignals<A extends AbortSignal | undefined>(
  a: A,
  b?: AbortSignal
): CombinedAbortScope<A> {
  if (a && b) return mergeBoth(a, b) as CombinedAbortScope<A>;
  const only = a ?? b;
  return (only ? { signal: only, dispose: NOOP_DISPOSE } : undefined) as CombinedAbortScope<A>;
}
