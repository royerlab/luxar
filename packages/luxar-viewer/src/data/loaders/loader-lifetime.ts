/**
 * The lifetime of one leaf loader: a disposed latch plus the abort signal every
 * loader-owned read can ride.
 *
 * A leaf loader's `dispose()` clears its arrays and resets its one-shot
 * initializer, but async work already in flight — above all an `initialize()`
 * awaiting its metadata opens — would otherwise finish afterwards and write its
 * results back into the disposed loader, and a later call would re-initialize
 * it from scratch. The latch makes disposal TERMINAL: work that started before
 * it bails as a cancellation (an `AbortError`, which run-loader-updates stages
 * as null quietly) and new work is refused the same way. The signal lets reads
 * that no caller owns (a speculative initialization) be cancelled by disposal.
 *
 * @module data/loaders/loader-lifetime
 */

import { abortReason } from './abortable-wait';

/**
 * One leaf loader's lifetime: owns the disposal latch and the loader-scoped
 * abort signal (see the module notes). Created with the loader and disposed
 * with it; never reused.
 */
export class LoaderLifetime {
  private readonly aborter = new AbortController();

  /** Aborted by {@link dispose}. */
  get signal(): AbortSignal {
    return this.aborter.signal;
  }

  /** True once {@link dispose} has run. */
  get disposed(): boolean {
    return this.aborter.signal.aborted;
  }

  /** Latch disposal and abort every read riding {@link signal}. Idempotent. */
  dispose(): void {
    if (!this.disposed) {
      this.aborter.abort(new DOMException('Loader disposed', 'AbortError'));
    }
  }

  /** Throw an `AbortError` naming `what` if the loader has been disposed. */
  throwIfDisposed(what: string): void {
    if (this.disposed) {
      throw new DOMException(`${what}: loader disposed`, 'AbortError');
    }
  }

  /**
   * Run a one-shot initialization under the latch: refuse to start one on a
   * disposed loader, and if disposal lands while it is in flight, call
   * `discard` (undo whatever the late initialization wrote) and reject.
   */
  async guardInit(what: string, init: () => Promise<void>, discard: () => void): Promise<void> {
    this.throwIfDisposed(what);
    try {
      await init();
    } catch (error) {
      // A failure caused by the disposal itself (an aborted read) is the
      // cancellation below, not an error of its own.
      if (!this.disposed) throw error;
    }
    if (this.disposed) {
      discard();
      throw abortReason(this.signal);
    }
  }
}
