/**
 * Perf counters — always-on, near-zero-cost tallies for performance probes.
 *
 * The render gate (`scripts/render-gate/`) compares a base and a candidate
 * build on counters such as bytes uploaded per frame, renders per
 * interaction, duplicate chunk decodes or console calls per step. Those
 * counters must exist on BOTH builds, cost nothing measurable on the hot
 * path, and be readable from bootstrap on (before the dataset loads), so
 * they live here as a module singleton rather than on a runtime component.
 *
 * Usage on a hot path: resolve the slot ONCE at module scope, then pay one
 * typed-array store per event:
 *
 * ```ts
 * const S_UPLOAD = perfCounters.slot('gpu.uploadBytes');
 * perfCounters.add(S_UPLOAD, byteLength);
 * ```
 *
 * `max` keeps a high-water mark, `gauge` overwrites. `record(kind, rec)`
 * appends a structured record to a bounded per-kind ring (for per-tick
 * traces such as playback). Nothing here throws into the caller.
 *
 * `reset()` starts a new measurement window: counters, high-water marks and
 * records go back to empty, but a GAUGE keeps its value. A gauge describes
 * current state (bytes pinned right now), and its owners republish it only
 * when it changes, so zeroing it would make the next reading lie until then.
 * A slot becomes a gauge the first time `gauge()` writes it.
 *
 * Every counter is declared in {@link PERF_COUNTERS} (or matches a family in
 * {@link PERF_COUNTER_FAMILIES}), and the singleton only accepts those names,
 * so a misspelt counter is a compile error rather than a fresh zero the render
 * gate would read as `n/a`. The names are types only: a slot still resolves
 * once at module scope and every event is still one typed-array store.
 *
 * Read via `__luxarDebug.getPerf().counters` (a flat `name -> number` map),
 * `__luxarDebug.getPerfRecords(kind)`, and reset via
 * `__luxarDebug.resetPerfCounters()`.
 *
 * @module profiling/perf-counters
 */

/** Maximum records kept per kind; older records are dropped first. */
export const PERF_RECORD_RING_SIZE = 4096;

/** What one counter declares about itself. */
export interface PerfCounterSpec {
  /** What the value is measured in. */
  readonly unit: 'count' | 'bytes' | 'ms';
  /**
   * How it is written: `sum` through `add`, `max` through `max` (a high-water
   * mark), `gauge` through `gauge` (current state, kept across a reset).
   */
  readonly kind: 'sum' | 'max' | 'gauge';
  /** One line: what event it counts. */
  readonly meaning: string;
}

/**
 * Every fixed perf counter. Adding a counter means adding it here (and an
 * exact-value test, which `counter-truth.test.ts` enforces); a name that is
 * not here does not compile.
 */
export const PERF_COUNTERS = {
  // Rendering (scene/animation)
  'render.count': {
    unit: 'count',
    kind: 'sum',
    meaning: 'scene renders issued; the sum of the render.byReason.* counters',
  },
  'render.ticks': { unit: 'count', kind: 'sum', meaning: 'animation loop ticks' },
  'render.skippedTicks': {
    unit: 'count',
    kind: 'sum',
    meaning: 'ticks render-on-change skipped because nothing drawn changed',
  },
  'render.once': {
    unit: 'count',
    kind: 'sum',
    meaning: 'renderOnce() calls that resumed a stopped loop',
  },
  'render.missedDirty': {
    unit: 'count',
    kind: 'sum',
    meaning: '?renderAudit: a skipped frame whose pixels differed from the last render',
  },
  'render.auditCompares': {
    unit: 'count',
    kind: 'sum',
    meaning: '?renderAudit: skipped frames read back and compared',
  },
  'adaptiveDpr.samples': {
    unit: 'count',
    kind: 'sum',
    meaning: 'frame times fed to the adaptive-DPR manager',
  },
  'playback.ticks': {
    unit: 'count',
    kind: 'sum',
    meaning: 'playback ticks that advanced a dimension value',
  },
  'lod.levelSwaps': {
    unit: 'count',
    kind: 'sum',
    meaning: 'displayed-level changes of a lod group (at most one per group per frame)',
  },
  'lod.blendFrames': {
    unit: 'count',
    kind: 'sum',
    meaning: 'group-frames drawing two lod levels dissolving',
  },
  'partition.partsActivated': {
    unit: 'count',
    kind: 'sum',
    meaning: 'deferred partition parts whose activation resolved',
  },
  'partition.partsInitialised': {
    unit: 'count',
    kind: 'sum',
    meaning: 'partition parts whose subtree load resolved',
  },
  'scene.getObjectByName': {
    unit: 'count',
    kind: 'sum',
    meaning: 'Object3D.getObjectByName calls (debug sessions only)',
  },
  // GPU uploads (rendering/upload-counters)
  'gpu.uploadCalls': { unit: 'count', kind: 'sum', meaning: 'wrapped GPU upload calls' },
  'gpu.uploadBytes': {
    unit: 'bytes',
    kind: 'sum',
    meaning: 'bytes uploaded to the GPU, buffers plus textures',
  },
  'gpu.uploadBytes.buffer': { unit: 'bytes', kind: 'sum', meaning: 'buffer upload bytes' },
  'gpu.uploadBytes.texture': { unit: 'bytes', kind: 'sum', meaning: 'texture upload bytes' },
  // Data loading
  'loaders.swept': {
    unit: 'count',
    kind: 'sum',
    meaning: 'loaders an update sweep considered (culled ones included), summed over sweeps',
  },
  'projection.gsplats.worker': {
    unit: 'count',
    kind: 'sum',
    meaning: 'gsplats projections handed to the worker pool (a no-worker fallback included)',
  },
  'projection.gsplats.stageHits': {
    unit: 'count',
    kind: 'sum',
    meaning: 'gsplats projections answered by the post-projection stage cache',
  },
  'projection.lines.worker': {
    unit: 'count',
    kind: 'sum',
    meaning: 'lines projections handed to the worker pool (a no-worker fallback included)',
  },
  'projection.lines.stageHits': {
    unit: 'count',
    kind: 'sum',
    meaning: 'lines projections answered by the post-projection stage cache',
  },
  'codec.blosc.worker': {
    unit: 'count',
    kind: 'sum',
    meaning: 'Blosc chunk decodes a pool worker completed',
  },
  'codec.blosc.main': {
    unit: 'count',
    kind: 'sum',
    meaning: 'Blosc chunk decodes run on the main thread (worker fallbacks included)',
  },
  'codec.blosc.fallback': {
    unit: 'count',
    kind: 'sum',
    meaning: 'worker Blosc decodes that failed and fell back to the main thread',
  },
  'codec.delta.fused': {
    unit: 'count',
    kind: 'sum',
    meaning: 'worker decodes that also undid the delta filter',
  },
  // Workers
  'worker.dispatches': { unit: 'count', kind: 'sum', meaning: 'tasks sent to a pool worker' },
  'worker.busyMs': {
    unit: 'ms',
    kind: 'sum',
    meaning: 'summed dispatch-to-settle time of pool tasks',
  },
  'worker.misroutes': {
    unit: 'count',
    kind: 'sum',
    meaning: 'dispatches to a busy worker while another live worker was idle',
  },
  // L0 decoded-chunk cache and decodes
  'l0.hits': { unit: 'count', kind: 'sum', meaning: 'chunk lookups served from L0' },
  'l0.misses': { unit: 'count', kind: 'sum', meaning: 'chunk lookups that started a decode' },
  'l0.coalesced': {
    unit: 'count',
    kind: 'sum',
    meaning: 'chunk lookups that joined a decode already in flight',
  },
  'l0.cloneBytes': {
    unit: 'bytes',
    kind: 'sum',
    meaning: 'bytes copied to store a decoded view that did not span its buffer',
  },
  'decode.count': { unit: 'count', kind: 'sum', meaning: 'completed miss decodes' },
  'decode.bytes': { unit: 'bytes', kind: 'sum', meaning: 'bytes of completed miss decodes' },
  'decode.duplicates': {
    unit: 'count',
    kind: 'sum',
    meaning: 're-decodes of a chunk decoded within the duplicate window',
  },
  'decode.count.foreground': {
    unit: 'count',
    kind: 'sum',
    meaning: 'miss decodes a foreground load triggered',
  },
  'decode.count.shadow': {
    unit: 'count',
    kind: 'sum',
    meaning: 'miss decodes a slice-prefetch shadow load triggered',
  },
  'decode.count.lookahead': {
    unit: 'count',
    kind: 'sum',
    meaning: 'miss decodes a playback lookahead triggered',
  },
  'decode.count.prefetch': {
    unit: 'count',
    kind: 'sum',
    meaning: 'miss decodes a chunk prefetch triggered',
  },
  // S-cache (cache/slice-cache)
  'scache.pinnedEntries': {
    unit: 'count',
    kind: 'gauge',
    meaning: 'S-cache entries currently protected by a prefetch pin',
  },
  'scache.pinnedBytes': {
    unit: 'bytes',
    kind: 'gauge',
    meaning: 'retained bytes of the pinned S-cache entries',
  },
  'scache.stage.hits': {
    unit: 'count',
    kind: 'sum',
    meaning: 'stage-output lookups answered from an S-cache entry',
  },
  'scache.stage.misses': {
    unit: 'count',
    kind: 'sum',
    meaning: 'stage-output lookups on a resident payload that found no usable output',
  },
  'scache.stage.rejected': {
    unit: 'count',
    kind: 'sum',
    meaning: 'stage outputs not admitted for lack of budget',
  },
  'scache.stage.shed': {
    unit: 'count',
    kind: 'sum',
    meaning: 'stage outputs dropped to make room for slice data',
  },
  'scache.stage.bytes': {
    unit: 'bytes',
    kind: 'gauge',
    meaning: 'bytes of stage outputs currently retained',
  },
  // L1/L2 store, network and OPFS
  'l2.hits': { unit: 'count', kind: 'sum', meaning: 'demand reads served from L2 (OPFS)' },
  'fetch.bytes': {
    unit: 'bytes',
    kind: 'sum',
    meaning: 'response-body bytes the network tier materialised',
  },
  'fetch.data.requests': { unit: 'count', kind: 'sum', meaning: 'gated data-lane fetches' },
  'fetch.data.highWater': {
    unit: 'count',
    kind: 'max',
    meaning: 'most data-lane leases in flight at once',
  },
  'fetch.data.queueWaitMs': {
    unit: 'ms',
    kind: 'sum',
    meaning: 'summed enqueue-to-start time of queued data-lane fetches',
  },
  'fetch.metadata.requests': {
    unit: 'count',
    kind: 'sum',
    meaning: 'gated metadata-lane fetches',
  },
  'fetch.metadata.highWater': {
    unit: 'count',
    kind: 'max',
    meaning: 'most metadata-lane leases in flight at once',
  },
  'fetch.metadata.queueWaitMs': {
    unit: 'ms',
    kind: 'sum',
    meaning: 'summed enqueue-to-start time of queued metadata-lane fetches',
  },
  'opfs.writes': { unit: 'count', kind: 'sum', meaning: 'successful OPFS chunk writes' },
  'opfs.writesDropped': {
    unit: 'count',
    kind: 'sum',
    meaning: 'OPFS write arrivals dropped by an overflow policy',
  },
  'opfs.estimateCalls': {
    unit: 'count',
    kind: 'sum',
    meaning: 'navigator.storage.estimate() quota checks',
  },
  'opfs.saveAttempts': {
    unit: 'count',
    kind: 'sum',
    meaning: 'OPFS index save requests, debounced or not',
  },
  'opfs.indexSaves': { unit: 'count', kind: 'sum', meaning: 'OPFS index files actually written' },
  // Console and profiler
  'console.calls': { unit: 'count', kind: 'sum', meaning: 'intercepted console calls' },
  'console.calls.log': { unit: 'count', kind: 'sum', meaning: 'intercepted console.log calls' },
  'console.calls.warn': { unit: 'count', kind: 'sum', meaning: 'intercepted console.warn calls' },
  'console.calls.error': {
    unit: 'count',
    kind: 'sum',
    meaning: 'intercepted console.error calls',
  },
  'console.calls.info': { unit: 'count', kind: 'sum', meaning: 'intercepted console.info calls' },
  'console.calls.debug': {
    unit: 'count',
    kind: 'sum',
    meaning: 'intercepted console.debug calls',
  },
  'profiler.mergeMs': {
    unit: 'ms',
    kind: 'sum',
    meaning: 'time merging finished profiler sessions into the persistent tree',
  },
} as const satisfies Record<string, PerfCounterSpec>;

/**
 * Counter families whose last segment is only known at run time. A family
 * member is `<prefix><suffix>`; its suffix set is open, so each family
 * declares one spec for all of its members.
 */
export const PERF_COUNTER_FAMILIES = {
  'render.byReason.': {
    unit: 'count',
    kind: 'sum',
    meaning: 'renders per RenderReason (wake, event, once, camera, cb:<id>, ...)',
  },
  'decode.count.': {
    unit: 'count',
    kind: 'sum',
    meaning: 'miss decodes per decode origin (the fixed origins are listed above)',
  },
} as const satisfies Record<string, PerfCounterSpec>;

/** A declared fixed counter name. */
export type FixedPerfCounterName = keyof typeof PERF_COUNTERS;

/** A declared counter family prefix. */
export type PerfCounterFamily = keyof typeof PERF_COUNTER_FAMILIES;

/** Every name the {@link perfCounters} singleton accepts. */
export type PerfCounterName = FixedPerfCounterName | `${PerfCounterFamily}${string}`;

/** The spec of a declared counter or family member, or `undefined` for an undeclared name. */
export function perfCounterSpec(name: string): PerfCounterSpec | undefined {
  if (Object.prototype.hasOwnProperty.call(PERF_COUNTERS, name)) {
    return PERF_COUNTERS[name as FixedPerfCounterName];
  }
  for (const prefix of Object.keys(PERF_COUNTER_FAMILIES) as PerfCounterFamily[]) {
    if (name.startsWith(prefix) && name.length > prefix.length) {
      return PERF_COUNTER_FAMILIES[prefix];
    }
  }
  return undefined;
}

/** Opaque handle to one counter (an index into the value array). */
export type PerfCounterSlot = number;

/**
 * The counter store. `Name` narrows the names it accepts; the process-wide
 * {@link perfCounters} takes only declared ones, while a bare
 * `new PerfCounters()` (tests of the store itself) takes any string.
 */
export class PerfCounters<Name extends string = string> {
  private readonly names: string[] = [];
  private readonly index = new Map<string, PerfCounterSlot>();
  private values = new Float64Array(64);
  /** 1 for a slot `gauge()` has written (kept across `reset()`), else 0. */
  private gauges = new Uint8Array(64);
  private readonly rings = new Map<string, unknown[]>();
  private readonly resetHooks = new Set<() => void>();

  /** Resolve (registering on first use) the slot for `name`. */
  slot(name: Name): PerfCounterSlot {
    const existing = this.index.get(name);
    if (existing !== undefined) return existing;
    const slot = this.names.length;
    this.names.push(name);
    this.index.set(name, slot);
    if (slot >= this.values.length) {
      const grown = new Float64Array(this.values.length * 2);
      grown.set(this.values);
      this.values = grown;
      const grownGauges = new Uint8Array(grown.length);
      grownGauges.set(this.gauges);
      this.gauges = grownGauges;
    }
    return slot;
  }

  /** Add `n` (default 1) to a counter. */
  add(slot: PerfCounterSlot, n = 1): void {
    this.values[slot] += n;
  }

  /** Raise a high-water mark to `v` if it is larger. */
  max(slot: PerfCounterSlot, v: number): void {
    if (v > this.values[slot]) this.values[slot] = v;
  }

  /** Overwrite a gauge with its current value (survives {@link reset}). */
  gauge(slot: PerfCounterSlot, v: number): void {
    this.values[slot] = v;
    this.gauges[slot] = 1;
  }

  /** Convenience for cold paths: add to a counter by name. */
  inc(name: Name, n = 1): void {
    this.add(this.slot(name), n);
  }

  /** Current value of one counter by name (0 if never registered). */
  get(name: Name): number {
    const slot = this.index.get(name);
    return slot === undefined ? 0 : this.values[slot];
  }

  /** Append a structured record to the bounded ring for `kind`. */
  record(kind: string, rec: unknown): void {
    let ring = this.rings.get(kind);
    if (!ring) {
      ring = [];
      this.rings.set(kind, ring);
    }
    ring.push(rec);
    if (ring.length > PERF_RECORD_RING_SIZE) ring.splice(0, ring.length - PERF_RECORD_RING_SIZE);
  }

  /** Records of one kind, oldest first (a copy). */
  records(kind: string): unknown[] {
    return [...(this.rings.get(kind) ?? [])];
  }

  /** Flat `name -> value` map of every registered counter. */
  snapshot(): Record<string, number> {
    const out: Record<string, number> = {};
    for (let i = 0; i < this.names.length; i++) out[this.names[i]] = this.values[i];
    return out;
  }

  /**
   * Run `hook` on every {@link reset}. For per-window state a counter's owner
   * keeps beside it (e.g. the duplicate-decode history behind
   * `decode.duplicates`), which must start the new window empty too. Returns
   * an unsubscribe.
   */
  onReset(hook: () => void): () => void {
    this.resetHooks.add(hook);
    return () => this.resetHooks.delete(hook);
  }

  /**
   * Start a new window: zero every counter and high-water mark, drop every
   * record, and run the {@link onReset} hooks. Gauges keep their current
   * value. Slots stay valid.
   */
  reset(): void {
    for (let i = 0; i < this.names.length; i++) {
      if (this.gauges[i] === 0) this.values[i] = 0;
    }
    this.rings.clear();
    for (const hook of this.resetHooks) {
      try {
        hook();
      } catch {
        /* A hook never breaks the reset (nothing here throws into the caller). */
      }
    }
  }
}

/** The process-wide registry: declared names only (see {@link PERF_COUNTERS}). */
export const perfCounters = new PerfCounters<PerfCounterName>();
