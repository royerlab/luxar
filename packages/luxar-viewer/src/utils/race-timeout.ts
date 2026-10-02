/**
 * The viewer's one promise-versus-timer race.
 *
 * @module utils/race-timeout
 */

/**
 * Settle with `promise`, or reject with `onTimeout()`'s value once `timeoutMs`
 * elapses first. The timer is cleared as soon as `promise` settles. A
 * `timeoutMs` of `<= 0` or a non-finite one installs no timer at all and
 * returns `promise` itself (the convention every timeout knob here follows).
 *
 * The race only settles the CALLER: whatever `promise` stands for keeps
 * running. Callers that hold a resource for the work's duration release it
 * when the work settles, not when the race does.
 *
 * @param onTimeout - Called once when the timer fires; its return value is the
 *   rejection reason (it may also log or evict).
 */
export function raceTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  onTimeout: () => unknown
): Promise<T> {
  if (timeoutMs <= 0 || !Number.isFinite(timeoutMs)) return promise;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(onTimeout()), timeoutMs);
  });
  return Promise.race([
    promise.finally(() => {
      if (timer !== undefined) clearTimeout(timer);
    }),
    timeout,
  ]);
}
