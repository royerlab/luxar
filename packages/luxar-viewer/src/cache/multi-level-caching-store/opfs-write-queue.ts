/**
 * Bounded background write queue for the L2 (OPFS) tier.
 *
 * The OPFS write (`OPFSStore.set` → `createWritable`/`write`/`close`, an atomic
 * temp-file swap per chunk) costs tens of ms and — measured on a true-cold
 * timelapse load — dominates the per-timepoint fetch critical path (~6× the
 * network fetch). `OPFSStore.set` already serializes writes *per key*, but has
 * no global cross-key cap, so firing every network miss's write inline (awaited)
 * blocks decode on disk, and firing them all un-awaited stampedes the single
 * OPFS backend (each write balloons under contention) and piles up unbounded.
 *
 * This queue moves the write off the fetch critical path while keeping it
 * bounded: the caller `enqueue()`s and returns immediately; the queue drains
 * tasks at a fixed `concurrency`, coalesces repeat writes of the same key
 * (keeping the latest), and past `maxDepth` or `maxBytes` drops the ARRIVAL —
 * the entry that just overflowed — not the oldest pending one (L2 is a
 * best-effort persistence tier — L1 still serves the current session and the
 * next session simply re-fetches a dropped chunk).
 *
 * Dropping the arrival rather than the oldest matters for what the next session
 * finds on disk. `pump()` takes from the same oldest end the drop loop used to
 * evict from, so drop-oldest kept discarding the entry the pump was about to
 * run and left a scattered remnant — chunks from all over the store, of which
 * no contiguous region is complete. Dropping the arrival lets the oldest end
 * drain in enqueue order, so what lands on disk is CONTIGUOUS RUNS from that
 * end (one prefix only while nothing drains; with the pump running, a run per
 * burst) — which is what a warm revisit can actually serve from. The tradeoff
 * is locality, not drop count: uniform-size writes drop equally under either
 * policy, while mixed sizes can favour either one. A large arrival no longer
 * displaces several small pending writes, but a large pending write can now
 * hold off several smaller arrivals.
 *
 * Correctness is the caller's (MultiLevelCachingStore) responsibility: the
 * enqueued task must re-check staleness (disposed / dataAbort / epoch) at drain
 * time, because clear/dispose can now interleave between enqueue and drain.
 *
 * @module cache/multi-level-caching-store/opfs-write-queue
 */

/**
 * Every limit here is resolved by the same clamp (finite, floored, at least 1),
 * so pass a real budget: a non-positive or non-finite value becomes 1, not
 * "unbounded". For `maxBytes` that means a 1-BYTE cap under which every
 * non-empty chunk drops; for `maxDepth`, a queue one task deep.
 */
export interface OpfsWriteQueueOptions {
  /** Max writes running concurrently against OPFS. */
  concurrency: number;
  /** Max pending (not-yet-started) tasks; an overflowing arrival is dropped. */
  maxDepth: number;
  /** Max bytes retained by pending tasks; an overflowing arrival is dropped. */
  maxBytes: number;
}

export interface OpfsWriteQueueStats {
  /** Pending (not-yet-started) tasks. */
  depth: number;
  /** Bytes retained by pending tasks. */
  pendingBytes: number;
  /** Tasks currently running. */
  inFlight: number;
  /** Cumulative tasks dropped by either overflow policy. */
  dropped: number;
  /** Resolved concurrency cap. */
  concurrency: number;
  /** Resolved max pending depth. */
  maxDepth: number;
  /** Resolved max pending bytes. */
  maxBytes: number;
}

/**
 * A per-instance bounded-concurrency FIFO write queue with per-key coalescing.
 * Models the acquire/release/FIFO shape of `utils/fetch-concurrency.ts`, but is
 * per-instance, configurable, and bounded by both task count and retained
 * bytes (that gate is an unbounded global singleton).
 */
export class OpfsWriteQueue {
  /**
   * Pending tasks keyed by cache key. `Map` insertion order IS the FIFO order;
   * re-enqueuing a key coalesces (the latest task wins and moves to newest).
   */
  private readonly pending = new Map<string, { run: () => Promise<void>; byteLength: number }>();
  private pendingBytes = 0;
  /** Currently-running task promises (for `drain()`). */
  private readonly inFlightPromises = new Set<Promise<void>>();
  private droppedCount = 0;
  // Re-entrancy guard for pump(): a task's run() that synchronously calls
  // enqueue() would re-enter pump() BEFORE the current task's promise has been
  // added to inFlightPromises, letting the re-entrant call under-count and
  // start work past the cap. The guard makes the re-entrant call a no-op; the
  // still-running outer loop (and the post-settle finally→pump) pick the work up.
  private pumping = false;
  private readonly concurrency: number;
  private readonly maxDepth: number;
  private readonly maxBytes: number;

  /**
   * Resolve one configured limit: finite, floored, at least 1. All three go
   * through here because `Math.max(1, Math.floor(NaN))` is NaN, and each limit
   * fails differently and silently on a NaN — `size < NaN` is false, so a
   * non-finite `concurrency` means the pump never starts a task (total L2
   * write loss) while a non-finite `maxDepth`/`maxBytes` simply removes that
   * bound. Every one of them is reachable through `MultiLevelCachingStore`'s
   * `opfsWrite*` options, which bypass the config validators.
   */
  private static resolveLimit(value: number): number {
    return Number.isFinite(value) ? Math.max(1, Math.floor(value)) : 1;
  }

  constructor(opts: OpfsWriteQueueOptions) {
    this.concurrency = OpfsWriteQueue.resolveLimit(opts.concurrency);
    this.maxDepth = OpfsWriteQueue.resolveLimit(opts.maxDepth);
    this.maxBytes = OpfsWriteQueue.resolveLimit(opts.maxBytes);
  }

  /**
   * Schedule a write. Returns synchronously; the task runs in the background
   * when a concurrency slot frees. Re-enqueuing the same key replaces its
   * pending task (coalesce, keep-latest). Past `maxDepth` or `maxBytes`, THIS
   * arrival is dropped and the already-pending tasks are left to drain in order
   * (see the module docstring for why the oldest end is the one worth keeping).
   */
  enqueue(key: string, run: () => Promise<void>, byteLength: number): void {
    // A size we cannot account for fails CLOSED: anything that is not a finite
    // value >= 0 is treated as over-cap and dropped before any state is
    // touched. NaN must not reach `pendingBytes` (it would disable the bound
    // for the session, since every `NaN > maxBytes` test is false), and
    // normalizing an unaccountable size to 0 instead — as a `Math.max(0, …)`
    // clamp would for a negative length — is the opposite error: it admits an
    // unmeasured payload for free, past a cap whose whole job is to bound
    // retained bytes. Any already-queued task for this key survives.
    if (!Number.isFinite(byteLength) || byteLength < 0) {
      this.droppedCount++;
      this.pump();
      return;
    }

    // Coalesce: delete-then-set so the re-enqueued key becomes newest in FIFO
    // order and its task is the latest.
    const previous = this.pending.get(key);
    if (previous) this.pendingBytes -= previous.byteLength;
    this.pending.delete(key);
    const pending = { run, byteLength: Math.floor(byteLength) };
    this.pending.set(key, pending);
    this.pendingBytes += pending.byteLength;

    // Drop-the-arrival overflow, for both caps: the oldest end is the prefix a
    // warm revisit can use, so it drains instead of being evicted. Best-effort
    // tier: a dropped write just isn't persisted to disk this session (L1 still
    // holds it). ONE removal always suffices — this call added exactly one
    // entry and both caps held before it, so deleting that entry returns
    // `pendingBytes` to its pre-enqueue value and `pending.size` to at most
    // `maxDepth`. A coalescing re-enqueue that overflows therefore loses BOTH
    // the task it replaced and this arrival for that key; acceptable for the
    // same best-effort reason.
    if (this.pending.size > this.maxDepth || this.pendingBytes > this.maxBytes) {
      this.pending.delete(key);
      this.pendingBytes -= pending.byteLength;
      this.droppedCount++;
    }

    this.pump();
  }

  /** Start tasks up to the concurrency cap. */
  private pump(): void {
    // Re-entrancy guard (see `pumping`): a synchronous enqueue() from within a
    // task's run() must not start extra tasks; the active loop below continues.
    if (this.pumping) return;
    this.pumping = true;
    try {
      while (this.inFlightPromises.size < this.concurrency && this.pending.size > 0) {
        const key = this.pending.keys().next().value as string;
        const pending = this.pending.get(key)!;
        this.pending.delete(key);
        this.pendingBytes -= pending.byteLength;

        const p = (async () => {
          try {
            await pending.run();
          } catch {
            // run() owns its own error handling; swallow so one failed write
            // never breaks the pump chain.
          }
        })().finally(() => {
          this.inFlightPromises.delete(p);
          this.pump();
        });
        this.inFlightPromises.add(p);
      }
    } finally {
      this.pumping = false;
    }
  }

  /**
   * Drop all pending (not-yet-started) tasks. In-flight tasks continue — the
   * caller neutralizes their effect via its own staleness guard. Used on cache
   * invalidation / dispose.
   */
  clear(): void {
    this.pending.clear();
    this.pendingBytes = 0;
  }

  /**
   * Await every pending + in-flight task to finish. Primarily for tests and an
   * optional teardown flush; the queue otherwise self-drains via `pump()`.
   */
  async drain(): Promise<void> {
    while (this.pending.size > 0 || this.inFlightPromises.size > 0) {
      this.pump();
      if (this.inFlightPromises.size > 0) {
        await Promise.allSettled([...this.inFlightPromises]);
      } else {
        await Promise.resolve();
      }
    }
  }

  stats(): OpfsWriteQueueStats {
    return {
      depth: this.pending.size,
      pendingBytes: this.pendingBytes,
      inFlight: this.inFlightPromises.size,
      dropped: this.droppedCount,
      concurrency: this.concurrency,
      maxDepth: this.maxDepth,
      maxBytes: this.maxBytes,
    };
  }
}
