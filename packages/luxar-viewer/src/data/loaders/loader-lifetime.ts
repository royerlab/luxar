/**
 * The lifetime of one leaf loader: a disposed latch, the abort signal every
 * loader-owned read can ride, the one-shot (optionally prioritised)
 * initialization, and the per-call demand-load context its L0 proxies read.
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
 * A loader whose disposal is NOT terminal by design (the mesh whole-node
 * loader re-initializes after a dispose) replaces its lifetime instead:
 * `this.lifetime.dispose(); this.lifetime = new LoaderLifetime()`, and treats
 * "is the lifetime I started under still current?" as its generation check.
 *
 * @module data/loaders/loader-lifetime
 */

import { abortReason } from './abortable-wait';
import { ActiveLoadContext } from './active-load-context';
import { OnceInit } from './once-init';
import { createChildController } from './progressive/child-signal';
import {
  tagSignalPriority,
  type FetchPriority,
  type FetchPriorityCell,
} from '../../utils/fetch-concurrency';

export class LoaderLifetime {
  private readonly aborter = new AbortController();
  private readonly once = new OnceInit();
  /** Priority cell of an in-flight PRIORITISED initialization. */
  private initPriority: FetchPriorityCell | null = null;

  /**
   * The in-flight demand loads' abort signals and residency probes, read by
   * the loader's L0 proxies through `() => lifetime.calls.signal` /
   * `() => lifetime.calls.probe` thunks (see `ActiveLoadContext`).
   */
  readonly calls = new ActiveLoadContext();

  /** Aborted by {@link dispose}. */
  get signal(): AbortSignal {
    return this.aborter.signal;
  }

  /** True once {@link dispose} has run. */
  get disposed(): boolean {
    return this.aborter.signal.aborted;
  }

  /** True once an initialization has resolved (and not been reset). */
  get initialized(): boolean {
    return this.once.isInitialized;
  }

  /** Latch disposal, abort every read riding {@link signal}. Idempotent. */
  dispose(): void {
    this.once.reset();
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
   * Initialize once: concurrent callers share the in-flight attempt, a failed
   * attempt is retried by the next caller, and it all runs under the latch
   * ({@link guardInit}).
   *
   * `priority` classes the reads of an initialization THIS call starts (B6: a
   * progressive loader warms rung k+1's index at `'refinement'` while rung k
   * loads; speculative entry points pass `'speculative'`). Such an
   * initialization has no caller signal of its own, so it rides a child of the
   * lifetime signal, tagged with the priority. A later call WITHOUT a priority
   * needs the index now, so it raises an in-flight warm to `demand` — no
   * priority inversion behind a warm.
   */
  async ensureInitialized(
    what: string,
    init: (signal?: AbortSignal) => Promise<void>,
    discard: () => void,
    priority?: FetchPriority
  ): Promise<void> {
    if (priority === undefined) this.initPriority?.raise('demand');
    await this.once.ensure(() =>
      this.guardInit(
        what,
        () => {
          if (priority === undefined) return init();
          const { signal } = createChildController(this.signal).controller;
          this.initPriority = tagSignalPriority(signal, priority);
          return init(signal);
        },
        discard
      )
    );
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
