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
 * Read via `__luxarDebug.getPerf().counters` (a flat `name -> number` map),
 * `__luxarDebug.getPerfRecords(kind)`, and reset via
 * `__luxarDebug.resetPerfCounters()`.
 *
 * @module profiling/perf-counters
 */

/** Maximum records kept per kind; older records are dropped first. */
export const PERF_RECORD_RING_SIZE = 4096;

/** Opaque handle to one counter (an index into the value array). */
export type PerfCounterSlot = number;

export class PerfCounters {
  private readonly names: string[] = [];
  private readonly index = new Map<string, PerfCounterSlot>();
  private values = new Float64Array(64);
  /** 1 for a slot `gauge()` has written (kept across `reset()`), else 0. */
  private gauges = new Uint8Array(64);
  private readonly rings = new Map<string, unknown[]>();
  private readonly resetHooks = new Set<() => void>();

  /** Resolve (registering on first use) the slot for `name`. */
  slot(name: string): PerfCounterSlot {
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
  inc(name: string, n = 1): void {
    this.add(this.slot(name), n);
  }

  /** Current value of one counter by name (0 if never registered). */
  get(name: string): number {
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

/** The process-wide registry. */
export const perfCounters = new PerfCounters();
