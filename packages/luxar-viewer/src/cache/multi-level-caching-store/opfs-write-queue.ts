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
 * (keeping the latest), and drops the oldest pending task past `maxDepth` or
 * `maxBytes` (L2 is a best-effort persistence tier — L1 still serves the
 * current session and the next session simply re-fetches a dropped chunk).
 *
 * Correctness is the caller's (MultiLevelCachingStore) responsibility: the
 * enqueued task must re-check staleness (disposed / dataAbort / epoch) at drain
 * time, because clear/dispose can now interleave between enqueue and drain.
 *
 * @module cache/multi-level-caching-store/opfs-write-queue
 */

export interface OpfsWriteQueueOptions {
  /** Max writes running concurrently against OPFS. */
  concurrency: number;
  /** Max pending (not-yet-started) tasks; excess drops the oldest. */
  maxDepth: number;
  /** Max bytes retained by pending tasks; excess drops the oldest. */
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

  constructor(opts: OpfsWriteQueueOptions) {
    this.concurrency = Math.max(1, Math.floor(opts.concurrency));
    this.maxDepth = Math.max(1, Math.floor(opts.maxDepth));
    this.maxBytes = Math.max(1, Math.floor(opts.maxBytes));
  }

  /**
   * Schedule a write. Returns synchronously; the task runs in the background
   * when a concurrency slot frees. Re-enqueuing the same key replaces its
   * pending task (coalesce, keep-latest). Past `maxDepth` or `maxBytes`, the
   * oldest pending task is dropped.
   */
  enqueue(key: string, run: () => Promise<void>, byteLength: number): void {
    // Coalesce: delete-then-set so the re-enqueued key becomes newest in FIFO
    // order and its task is the latest.
    const previous = this.pending.get(key);
    if (previous) this.pendingBytes -= previous.byteLength;
    this.pending.delete(key);
    const pending = { run, byteLength: Math.max(0, Math.floor(byteLength)) };
    this.pending.set(key, pending);
    this.pendingBytes += pending.byteLength;

    // Drop-oldest overflow. Best-effort tier: a dropped write just isn't
    // persisted to disk this session (L1 still holds it).
    while (this.pending.size > this.maxDepth || this.pendingBytes > this.maxBytes) {
      const oldest = this.pending.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      const dropped = this.pending.get(oldest);
      this.pending.delete(oldest);
      if (dropped) this.pendingBytes -= dropped.byteLength;
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
