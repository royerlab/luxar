/**
 * The per-call demand-load context a leaf loader publishes to its L0 proxies.
 *
 * The L0 chokepoint (`wrapWithCache`'s `getSignal` / `getProbe` accessors) asks
 * the loader, on every chunk read, "which update's abort signal governs this
 * read, and which residency probe records it?". Those accessors take no
 * arguments, so the loader answers from state. That state used to be ONE slot
 * per loader, set on entry and nulled in `finally` — not reentrant: when two
 * loads overlap on one loader (a foreground pass and a shadow prefetch pass on
 * the same rung), whichever finished first nulled the other's signal (its
 * superseded reads were no longer cancellable) and detached its probe (its
 * remaining misses went unrecorded, so it reported "resident" and the refine
 * stop rule kept streaming).
 *
 * Each call now registers its OWN entry and removes exactly that entry when it
 * settles. While calls overlap, the published answers are the conservative
 * merge, because a chunk read cannot say which call issued it:
 *
 * - signal: a read is cancelled only once EVERY active call is aborted, so one
 *   call's supersession never cancels a read another live call shares. The
 *   composite remains live as calls join or leave;
 * - probe: a read is recorded into EVERY active call's probe, so no call can
 *   report "all resident" for a miss it may have needed. A spurious miss only
 *   ends a refinement step early; a missed one stalls nothing but lies.
 *
 * @module data/loaders/active-load-context
 */

import { ResidencyAccumulator, type ResidencyProbe } from '../../cache/residency-probe';
import { signalOrigin, tagSignalOrigin } from '../../cache/decompressed-chunk-cache/decode-origin';
import { signalPriority, tagSignalPriority } from '../../utils/fetch-concurrency';

/** Active call signal entry. Do not un-export: TypeDoc needs this name. */
export interface SignalEntry {
  readonly signal: AbortSignal | null;
}

/**
 * The set of demand loads currently running on one leaf loader, published to
 * its L0 proxies as one merged abort signal and one fanned-out residency probe
 * (see the module notes for the merge rules). The first caller with a decode
 * origin sets the composite's origin for the overlap episode. Its fetch
 * priority can only rise, including after a higher-priority caller leaves.
 */
export class ActiveLoadContext {
  private readonly signals: SignalEntry[] = [];
  private controller: AbortController | null = null;
  private readonly probes: ResidencyAccumulator[] = [];
  private readonly fanOut: ResidencyProbe = {
    record: (hit: boolean): void => {
      for (const probe of this.probes) probe.record(hit);
    },
  };

  /**
   * The abort signal governing a chunk read now (see the module notes). When
   * the last call returns normally, the composite aborts with an `AbortError`.
   */
  get signal(): AbortSignal | null {
    // An unsignalled call needs the loader lifetime's fallback signal.
    if (this.signals.length === 0 || this.signals.some((entry) => entry.signal === null))
      return null;
    return this.controller?.signal ?? null;
  }

  private refreshSignal(): void {
    if (this.signals.length === 0 || this.signals.every((entry) => entry.signal?.aborted)) {
      this.controller?.abort(this.signals[0]?.signal?.reason);
    }
  }

  private inheritSignalTags(signal: AbortSignal | undefined): () => void {
    if (!signal || !this.controller) return () => undefined;
    const priority = signalPriority(signal);
    const compositePriority = tagSignalPriority(
      this.controller.signal,
      priority?.value ?? 'demand'
    );
    const detachPriority = priority?.onRaise(() => compositePriority.raise(priority.value));
    const origin = signalOrigin(signal);
    if (origin !== undefined) tagSignalOrigin(this.controller.signal, origin);
    return () => detachPriority?.();
  }

  private registerSignal(signal: AbortSignal | undefined): () => void {
    const entry: SignalEntry = { signal: signal ?? null };
    if (!this.controller || this.controller.signal.aborted) this.controller = new AbortController();
    const detachTags = this.inheritSignalTags(signal);
    const onAbort = (): void => this.refreshSignal();
    this.signals.push(entry);
    signal?.addEventListener('abort', onAbort, { once: true });
    this.refreshSignal();
    return () => {
      detachTags();
      signal?.removeEventListener('abort', onAbort);
      this.signals.splice(this.signals.indexOf(entry), 1);
      this.refreshSignal();
    };
  }

  /** The residency probe a chunk read records into now (see the module notes). */
  get probe(): ResidencyProbe | null {
    if (this.probes.length === 0) return null;
    return this.probes.length === 1 ? this.probes[0] : this.fanOut;
  }

  /**
   * Run one demand load with `signal` published for the L0 chokepoint, and
   * withdraw exactly this call's entry when it settles. The shared body of the
   * three leaf loaders' `updateView`.
   */
  async runWithSignal<TData>(
    signal: AbortSignal | undefined,
    load: () => Promise<TData>
  ): Promise<TData> {
    const unregister = this.registerSignal(signal);
    try {
      return await load();
    } finally {
      unregister();
    }
  }

  /**
   * Run one demand load with a fresh residency probe attached, reporting
   * whether it was served entirely from cache (a load that touches no chunks
   * counts as resident). The shared body of `updateViewWithResidency`.
   */
  async runWithProbe<TData>(
    load: () => Promise<TData>
  ): Promise<{ data: TData; allResident: boolean }> {
    const probe = new ResidencyAccumulator();
    this.probes.push(probe);
    try {
      const data = await load();
      return { data, allResident: probe.allResident };
    } finally {
      this.probes.splice(this.probes.indexOf(probe), 1);
    }
  }
}
