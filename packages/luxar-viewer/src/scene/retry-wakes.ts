/**
 * One-shot wakes, with exponential backoff, for the LOD retries a parked
 * camera must still reach: a failed level's cooldown before its automatic
 * retry, and an unanswered partition-part activation request before it is
 * asked again.
 *
 * Neither needs the loop ticking while it waits — nothing on screen changes
 * until the retry fires — so instead of a per-frame tick demand
 * (``tick-demand.ts``) each schedules ONE timer for its expiry, which asks for
 * a single frame. Every consecutive retry of the same key doubles the wait
 * (capped at {@link MAX_RETRY_DELAY_MS}), so a hard failure settles into a slow
 * poll instead of a retry storm; a success resets it.
 *
 * Keys are the waiting objects themselves (a ``LODGroupChild``, a partition
 * part's lazy state), so the attempt counts are weakly held.
 *
 * @module scene/retry-wakes
 */

/** Upper bound on a backed-off retry wait. */
export const MAX_RETRY_DELAY_MS = 30_000;

export class RetryWakes {
  private readonly timers = new Map<object, ReturnType<typeof setTimeout>>();
  private readonly attempts = new WeakMap<object, number>();

  /**
   * @param canWake - Whether a wake can be delivered at all (no tick/render
   *   callback wired ⇒ scheduling is a no-op).
   * @param wake - Ask the loop for one frame.
   */
  constructor(
    private readonly canWake: () => boolean,
    private readonly wake: () => void
  ) {}

  /** ``baseMs`` backed off by ``key``'s consecutive retries. */
  delay(key: object, baseMs: number): number {
    return Math.min(MAX_RETRY_DELAY_MS, baseMs * 2 ** (this.attempts.get(key) ?? 0));
  }

  /** Wake the loop once in ``delayMs`` for ``key``, unless a wake is already pending. */
  schedule(key: object, delayMs: number): void {
    if (this.timers.has(key) || !this.canWake()) return;
    const timer = setTimeout(
      () => {
        this.timers.delete(key);
        this.wake();
      },
      Math.max(0, delayMs)
    );
    this.timers.set(key, timer);
  }

  /** Cancel ``key``'s pending wake (its attempt count is kept). */
  cancel(key: object): void {
    const timer = this.timers.get(key);
    if (timer !== undefined) clearTimeout(timer);
    this.timers.delete(key);
  }

  /** The wait ended in a retry: cancel the wake and lengthen the next wait. */
  backOff(key: object): void {
    this.cancel(key);
    this.attempts.set(key, (this.attempts.get(key) ?? 0) + 1);
  }

  /** ``key`` succeeded or was retried explicitly: cancel the wake and drop its backoff. */
  reset(key: object): void {
    this.cancel(key);
    this.attempts.delete(key);
  }

  /** Cancel every pending wake (scene teardown). */
  cancelAll(): void {
    for (const timer of this.timers.values()) clearTimeout(timer);
    this.timers.clear();
  }
}
