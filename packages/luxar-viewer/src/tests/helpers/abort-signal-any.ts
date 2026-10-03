/**
 * Hide `AbortSignal.any` so code under test takes its fallback path for
 * runtimes that lack it. Returns the restore function; call it in a `finally`.
 */
export function forceAbortSignalAnyFallback(): () => void {
  const descriptor = Object.getOwnPropertyDescriptor(AbortSignal, 'any');
  Object.defineProperty(AbortSignal, 'any', { configurable: true, value: undefined });
  return () => {
    if (descriptor) Object.defineProperty(AbortSignal, 'any', descriptor);
    else delete (AbortSignal as unknown as { any?: unknown }).any;
  };
}
