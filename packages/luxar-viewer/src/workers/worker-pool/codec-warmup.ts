/**
 * Lazy, one-worker-first warm-up of the data workers' blosc codec.
 *
 * Warming a worker's codec makes it download the worker bundle's own blosc
 * chunk (~600 KB). Doing that for every worker the moment it came ready meant
 * 15 concurrent downloads (~9.4 MB) competing with the scene metadata on a
 * hosted link — the sp64 gate store's cold first frame went from 3.6 s to
 * 6.6 s, on a store none of whose chunks clears the offload floor at all.
 *
 * So nothing is warmed until a decode actually wants a worker
 * ({@link CodecWarmup.ensureWarm}, called by the codec dispatcher for a chunk
 * above the floor). That warms exactly ONE worker, while the caller decodes on
 * the main thread; only once it finished are the others warmed, so they read
 * the chunk from the HTTP cache instead of the network.
 *
 * A warm-up is NOT a query: it never touches a worker's `activeQueries`, so it
 * cannot make a worker look busy to the pool's stats. Decode dispatch instead asks
 * {@link CodecWarmup.isWarm}: a worker still warming gets no decode.
 *
 * @module workers/worker-pool/codec-warmup
 */

import type { WorkerInstance } from './types';

type WarmState = 'warming' | 'warm';

/** What the warm-up needs from the pool (injected, so it stays pool-agnostic). */
export interface CodecWarmupDeps {
  /** The live workers. */
  workers: () => readonly WorkerInstance[];
  /** Run one worker's `warmCodecs` task (timeout-guarded by the pool). */
  run: (entry: WorkerInstance, call: Promise<void>) => Promise<void>;
  /** False under the `?mainThreadCodecs` kill switch: never warm. */
  enabled: () => boolean;
  /** Report a failed warm-up (the worker is retried on the next demand). */
  onError: (error: unknown) => void;
}

export class CodecWarmup {
  /** Keyed by the instance, so an evicted or replaced worker starts cold. */
  private readonly state = new WeakMap<WorkerInstance, WarmState>();

  constructor(private readonly deps: CodecWarmupDeps) {}

  /**
   * True when some live worker's codec is warm (a decode may be offloaded);
   * the remaining cold workers are then warmed too. Otherwise starts warming
   * ONE worker, if none is already warming, and returns false (decode locally).
   */
  ensureWarm(): boolean {
    if (!this.deps.enabled()) return false;
    const workers = this.deps.workers();
    if (workers.some((w) => this.stateOf(w) === 'warm')) {
      this.warmCold(workers);
      return true;
    }
    if (!workers.some((w) => this.stateOf(w) === 'warming')) {
      const first = leastBusy(workers);
      if (first) this.start(first);
    }
    return false;
  }

  /**
   * Whether `entry`'s codec is warm. A decode goes only to such a worker: one
   * still WARMING is downloading the codec chunk / compiling its WASM — tens to
   * hundreds of ms — for a chunk the main thread decodes in about one.
   */
  isWarm(entry: WorkerInstance): boolean {
    return this.stateOf(entry) === 'warm';
  }

  /** A worker without the task (a test double, a legacy worker) needs no warm-up. */
  private stateOf(entry: WorkerInstance): WarmState | undefined {
    if (typeof entry.api?.warmCodecs !== 'function') return 'warm';
    return this.state.get(entry);
  }

  private warmCold(workers: readonly WorkerInstance[]): void {
    for (const w of workers) if (this.stateOf(w) === undefined) this.start(w);
  }

  private start(entry: WorkerInstance): void {
    this.state.set(entry, 'warming');
    this.deps.run(entry, entry.api.warmCodecs()).then(
      () => {
        this.state.set(entry, 'warm');
        // The chunk is in the HTTP cache now: the rest can load it cheaply.
        if (this.deps.enabled()) this.warmCold(this.deps.workers());
      },
      (error: unknown) => {
        this.state.delete(entry);
        this.deps.onError(error);
      }
    );
  }
}

function leastBusy(workers: readonly WorkerInstance[]): WorkerInstance | undefined {
  let best: WorkerInstance | undefined;
  for (const w of workers) if (!best || w.activeQueries < best.activeQueries) best = w;
  return best;
}
