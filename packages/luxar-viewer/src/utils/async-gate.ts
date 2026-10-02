/**
 * A small counting gate: at most `capacity()` holders at once, waiters served
 * by priority rank then FIFO, and a waiter whose signal aborts leaves the
 * queue at once.
 *
 * The primitive behind the OPFS read gate. It is deliberately NOT the fetch
 * gate's engine: that one also widens per origin, caps a speculative share,
 * raises a queued request's class in place and dispatches by width tier —
 * see the "Queues and what they honour" table in `utils/README.md` for
 * which queue does what.
 *
 * @module utils/async-gate
 */

/** Occupancy snapshot. */
export interface AsyncGateStats {
  /** Holders that acquired and have not released. */
  active: number;
  /** Live waiters (aborted ones are not counted). */
  queued: number;
}

/** Releases one acquisition. Idempotent; a no-op after {@link AsyncGate.reset}. */
export type AsyncGateRelease = () => void;

interface Waiter {
  readonly grant: () => void;
  readonly fail: (reason: unknown) => void;
  live: boolean;
}

/** Why a queued waiter left: the signal's reason, or an `AbortError`. */
function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException('Gate wait aborted', 'AbortError');
}

export class AsyncGate {
  #active = 0;
  #queued = 0;
  /** Bumped by reset(): a release from an older epoch must not touch the counts. */
  #epoch = 0;
  /** One FIFO per priority rank, most urgent (0) first. */
  readonly #queues: Waiter[][];

  /**
   * @param capacity - Read at every admission, so a live-tunable cap applies
   *   to the next acquisition.
   * @param ranks - Number of priority ranks (`0` most urgent); default one.
   */
  constructor(
    private readonly capacity: () => number,
    ranks = 1
  ) {
    this.#queues = Array.from({ length: Math.max(1, ranks) }, () => []);
  }

  stats(): AsyncGateStats {
    return { active: this.#active, queued: this.#queued };
  }

  /**
   * Wait for a place. Resolves with the release; rejects with the signal's
   * reason (or an `AbortError`) if `signal` aborts first, freeing the queue
   * place, and at once for an already-aborted signal.
   *
   * @param priority - Rank to queue under (clamped to the gate's ranks).
   */
  acquire(signal?: AbortSignal, priority = 0): Promise<AsyncGateRelease> {
    if (signal?.aborted) return Promise.reject(abortReason(signal));
    if (this.#active < this.capacity() && this.#queued === 0) {
      return Promise.resolve(this.#take());
    }
    return new Promise<AsyncGateRelease>((resolve, reject) => {
      const onAbort = (): void => {
        if (!waiter.live) return;
        waiter.live = false;
        this.#queued -= 1;
        reject(abortReason(signal!));
      };
      const waiter: Waiter = {
        live: true,
        grant: () => {
          signal?.removeEventListener('abort', onAbort);
          resolve(this.#take());
        },
        fail: (reason) => {
          signal?.removeEventListener('abort', onAbort);
          reject(reason);
        },
      };
      const rank = Math.min(this.#queues.length - 1, Math.max(0, Math.floor(priority)));
      this.#queues[rank].push(waiter);
      this.#queued += 1;
      signal?.addEventListener('abort', onAbort, { once: true });
      // A raised capacity may already have room for the head of the queue.
      this.#grantNext();
    });
  }

  /** Reject every queued waiter with `reason` and zero the counts (test isolation). */
  reset(reason: unknown): void {
    this.#epoch += 1;
    this.#active = 0;
    this.#queued = 0;
    for (const queue of this.#queues) {
      for (const waiter of queue.splice(0)) {
        if (!waiter.live) continue;
        waiter.live = false;
        waiter.fail(reason);
      }
    }
  }

  #take(): AsyncGateRelease {
    this.#active += 1;
    const epoch = this.#epoch;
    let released = false;
    return () => {
      if (released || epoch !== this.#epoch) return;
      released = true;
      this.#active -= 1;
      this.#grantNext();
    };
  }

  /** Hand free places to the most urgent live waiters. */
  #grantNext(): void {
    for (const queue of this.#queues) {
      while (queue.length > 0 && this.#active < this.capacity()) {
        const waiter = queue.shift()!;
        if (!waiter.live) continue;
        waiter.live = false;
        this.#queued -= 1;
        waiter.grant();
      }
    }
  }
}
