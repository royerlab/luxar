/**
 * Progressive GSplats loader for multi-LOD datasets.
 *
 * Wraps N GSplatsSpatialIndexLoader instances (one per LOD subgroup) using the
 * Composite pattern. Implements the same GSplatsDataLoader interface so the
 * scene loader's update loop works unchanged.
 *
 * Loading strategy:
 * - On each updateView() call, loads LODs sequentially starting from LOD 0
 * - Stops at the first LOD whose load takes longer than the cache-hit threshold
 * - After returning, fires a prefetch for the next unloaded LOD (cache warming)
 * - On next call with the same viewState, the prefetched LOD is a cache hit
 *
 * LODs are additive: LOD 0 contains the coarsest (highest-amplitude) splats,
 * and each subsequent LOD adds residual detail. The loader concatenates all
 * loaded LODs into a single LoadedGSplatsData.
 *
 * @module data/gsplats-progressive-loader
 */

import type { GSplatsDataLoader, GSplatsViewState, LoadedGSplatsData } from '../../types/gsplats';
import { setPrefixParent } from '../../types/prefix-lineage';
import type { GSplatsSpatialIndexLoader } from './gsplats-spatial-index-loader';
import type { UpdateSession } from '../../profiling/update-profiler';
import type {
  LoaderMetrics,
  MonitorEventListener,
  QueryInfo,
} from '../../types/data-monitor-types';
import { assertColorLayout } from '../loaders';
import { ProgressiveMonitorAdapter } from '../loaders/progressive-monitor-adapter';
import { concatRequiredField } from '../loaders/progressive/concat-helpers';
import {
  classifyStreamingPass,
  shouldLoadLevel,
  shouldStopAfterLevel,
} from '../loaders/progressive/streaming-policy';
import { restoreLadder, storeLadder } from '../loaders/progressive/slice-cache-helper';
import { viewStatesEqual } from '../loaders/progressive/view-state-equal';
import type { SliceCache } from '../../cache/slice-cache';
import { log, Modules, LogEmoji } from '../../utils/log';

/**
 * Concatenate multiple LoadedGSplatsData into one.
 * Allocates new arrays sized for the total splat count and copies data.
 */
export function concatenateGSplatsData(parts: LoadedGSplatsData[]): LoadedGSplatsData {
  if (parts.length === 0) {
    return {
      positions: new Float32Array(0),
      amplitudes: new Float32Array(0),
      choleskyFactors: new Float32Array(0),
      colors: null,
      splatCount: 0,
      ndim: 3,
    };
  }

  if (parts.length === 1) {
    return parts[0];
  }

  const ndim = parts[0].ndim;
  // Same fail-fast contract as the dtype/layout checks below, one field
  // over: `ndim` strides positions AND sizes the Cholesky blocks, so a
  // corrupted store whose sub-LODs disagree on ndim would pass the dtype
  // checks (all Float32) yet mis-stride every splat after the first part —
  // silent corruption. Dimensionality is per-dataset; a mismatch is
  // malformed data.
  for (const part of parts) {
    if (part.ndim !== ndim) {
      throw new Error(
        'concatenateGSplatsData: mixed dimensionality across LOD levels ' +
          `(ndim ${part.ndim} vs ${ndim}) — ladder levels must share the ` +
          'dataset dimensionality.'
      );
    }
  }
  // Per-part color-layout check, distinct from the cross-level mismatch
  // guarded below: those throws catch LODs that DISAGREE on dtype/layout,
  // but a single part can carry an RGBA buffer while OMITTING
  // `colorComponents: 4` (it defaults to 3). That satisfies the downstream
  // `count·3` minimum yet mis-strides every splat after the first — silent
  // corruption. Assert each part's raw length against its own declared
  // layout before allocation so an omitted declaration throws loudly here.
  for (const part of parts) {
    assertColorLayout(
      part.colors,
      part.splatCount,
      part.colorComponents ?? 3,
      'concatenateGSplatsData'
    );
  }
  const totalSplats = parts.reduce((sum, p) => sum + p.splatCount, 0);
  const cholSize = (ndim * (ndim + 1)) / 2;
  const count = (p: LoadedGSplatsData) => p.splatCount;

  // Required per-splat fields via the shared helpers (dtype preserved).
  const positions = concatRequiredField(parts, (p) => p.positions, count, ndim, 'positions');
  const amplitudes = concatRequiredField(parts, (p) => p.amplitudes, count, 1, 'amplitudes');
  const choleskyFactors = concatRequiredField(
    parts,
    (p) => p.choleskyFactors,
    count,
    cholSize,
    'choleskyFactors'
  );

  // Bespoke: colors fill missing LODs with white (per-dtype fill value).
  // Color layout (3 = RGB, 4 = RGBA — the 4th channel is per-splat opacity)
  // is a property of the dataset, uniform across its LODs; take it from the
  // first LOD that carries colors and stride every copy/fill by it (a
  // hardcoded 3 would truncate + misalign an RGBA additive ladder — exactly
  // what imported 3DGS scenes become after `gsplat lod`).
  const firstWithColors = parts.find((p) => p.colors !== null);
  const colorK: 3 | 4 = firstWithColors?.colorComponents ?? 3;
  let colors: Float32Array | Uint8Array | Uint16Array | null = null;
  if (firstWithColors?.colors) {
    if (firstWithColors.colors instanceof Uint8Array) {
      colors = new Uint8Array(totalSplats * colorK);
    } else if (firstWithColors.colors instanceof Uint16Array) {
      colors = new Uint16Array(totalSplats * colorK);
    } else {
      colors = new Float32Array(totalSplats * colorK);
    }
  }

  let offset = 0;
  for (const [levelIdx, part] of parts.entries()) {
    if (colors && part.colors) {
      // LADDER-DTYPE CONTRACT (see concat-helpers.ts): `set` converts by
      // VALUE, not semantics — a Float32 (0..1) level written into a Uint8
      // (0..255) merge truncates to garbage, and the reverse writes 255×
      // values. The writer emits one color dtype per ladder; fail fast.
      // Messages name the offending level (concat-helpers' convention) so a
      // corrupt store is diagnosable without a debugger.
      if (part.colors.constructor !== colors.constructor) {
        throw new Error(
          'concatenateGSplatsData: mixed color dtypes across LOD levels ' +
            `(level ${levelIdx}: ${part.colors.constructor.name} vs ` +
            `${colors.constructor.name}) — ladder levels must share each ` +
            "field's dtype."
        );
      }
      // Same contract for the color LAYOUT: `colorK` strides every copy, so
      // an RGBA level inside an RGB ladder (same ctor — invisible to the
      // dtype check above) would land at the wrong stride and silently
      // corrupt every splat after it. Layout is per-dataset, uniform across
      // its LODs; a mismatch is malformed data.
      if ((part.colorComponents ?? 3) !== colorK) {
        throw new Error(
          'concatenateGSplatsData: mixed color layouts across LOD levels ' +
            `(level ${levelIdx}: ${part.colorComponents ?? 3} vs ${colorK} ` +
            'components) — ladder levels must share the color layout ' +
            '(RGB vs RGBA).'
        );
      }
      colors.set(part.colors, offset * colorK);
    } else if (colors && !part.colors) {
      // Fill with white (1.0 for Float32, 255 for Uint8, 65535 for Uint16).
      // Alpha fills opaque (the per-element-opacity identity) via the same
      // full-scale fill value.
      const fillValue =
        colors instanceof Uint8Array ? 255 : colors instanceof Uint16Array ? 65535 : 1.0;
      for (let i = 0; i < part.splatCount * colorK; i++) {
        colors[offset * colorK + i] = fillValue;
      }
    }

    offset += part.splatCount;
  }

  return {
    positions,
    amplitudes,
    choleskyFactors,
    colors,
    colorComponents: colorK,
    splatCount: totalSplats,
    ndim,
  };
}

/**
 * Progressive GSplats loader for multi-LOD datasets.
 *
 * Wraps N GSplatsSpatialIndexLoader instances and loads LODs progressively,
 * using cache hits to determine how many LODs fit within the frame budget.
 */
export class GSplatsProgressiveLoader implements GSplatsDataLoader {
  private lodLoaders: GSplatsSpatialIndexLoader[];
  private loadedLODs: LoadedGSplatsData[] = [];
  private lastViewState: GSplatsViewState | null = null;
  private nLods: number;
  private monitor: ProgressiveMonitorAdapter;
  private _initialLoadDone = false;
  private _lastAllResident = true;
  private _disposed = false;
  // Memoized concatenation. Keyed on (resetGeneration, loadedLODs.length):
  // the generation bumps on every view-state reset so a reset-then-reload
  // back to the same LOD count yields a NEW reference (contents differ),
  // while an unchanged view state with no new LODs returns the SAME
  // reference — which the commit pipeline uses to skip no-op re-commits.
  private _resetGeneration = 0;
  private _concatCache: {
    generation: number;
    lodCount: number;
    result: LoadedGSplatsData;
  } | null = null;
  // Per-sub-LOD cumulative energy fractions e(k) (the build-time
  // `lod_stats.energy_fraction_cum` stamps), normalized at construction:
  // non-null only when EVERY sub-LOD carries a stamp (a partially stamped
  // ladder reads as unstamped — never blend stamped and guessed entries).
  private energyTable: readonly number[] | null;
  // Node path (SliceCache namespace) + the shared SliceCache, if enabled.
  private readonly path: string;
  private readonly sliceCache: SliceCache | null;
  // Per-tick LOD time budget (ms) from the CURRENT updateView call during
  // dimension-animation playback; null outside playback. A per-pass
  // directive (never part of lastViewState / viewStatesEqual / cache keys):
  // caps how many sub-LODs the streaming loop loads this pass and, while
  // set, makes `hasMoreLODs` read false so no background refinement runs
  // between animation ticks and the budgeted prefix commits as
  // "complete for playback" (display gate accepts it without holding).
  private _frameBudgetMs: number | null = null;

  // Set when LOD 0 committed 0 elements for the current view: the slice is
  // empty, so every higher (spatially-coextensive) LOD is empty too and the
  // ladder is TERMINAL. Makes `hasMoreLODs` read false so the refinement
  // scheduler doesn't fetch+decode the higher empty LODs one pass at a time.
  // Reset on every view change (the next slice may have content).
  private _emptyLadder = false;

  constructor(
    lodLoaders: GSplatsSpatialIndexLoader[],
    nLods: number,
    path: string,
    energyTable?: ReadonlyArray<number | null | undefined>,
    sliceCache?: SliceCache | null
  ) {
    this.lodLoaders = lodLoaders;
    this.nLods = nLods;
    this.path = path;
    this.sliceCache = sliceCache ?? null;
    this.monitor = new ProgressiveMonitorAdapter(() => this.lodLoaders, path);
    this.energyTable =
      energyTable && energyTable.length === nLods && energyTable.every((e) => typeof e === 'number')
        ? (energyTable as number[])
        : null;
  }

  /**
   * Whether there are more LOD levels to load for the current view state.
   */
  get hasMoreLODs(): boolean {
    // A disposed loader has work-state cleared; report no further work so a
    // refinement loop holding a stale reference stops instead of indexing
    // into the now-empty lodLoaders.
    if (this._disposed) return false;
    // Empty slice (LOD 0 committed 0 elements): the ladder is terminal, so
    // report no further work rather than let refinement fetch the higher
    // (also-empty) LODs pass after pass.
    if (this._emptyLadder) return false;
    // While a playback frame budget is active, the budgeted prefix IS the
    // target: report no further work so the refinement scheduler stays idle
    // between animation ticks and the commit stamps the prefix as complete
    // (the display gate then accepts it instead of holding the previous
    // frame). The next budget-free updateView (pause re-trigger, scrub)
    // clears the budget and refinement resumes from the prefix.
    if (this._frameBudgetMs !== null) return false;
    return this.loadedLODs.length < this.nLods;
  }

  /** Number of LOD levels currently loaded. */
  get loadedLODCount(): number {
    return this.loadedLODs.length;
  }

  /**
   * Cumulative energy fraction e(k) ∈ [0, 1] of the currently loaded LOD
   * prefix — how much of this ladder's total self-energy the committed
   * chunks carry (the additive orderer's own ranking criterion, stamped at
   * build time as `lod_stats.energy_fraction_cum`). `null` when the dataset
   * carries no energy stamps; `0` before any LOD loads. Read at commit time
   * by `stampLadderComplete` (→ the `committedEnergyFraction` mesh stamp)
   * for the display gate's energy-threshold upgrade release.
   */
  get committedEnergyFraction(): number | null {
    if (!this.energyTable) return null;
    const k = this.loadedLODs.length;
    if (k === 0) return 0;
    return this.energyTable[Math.min(k, this.energyTable.length) - 1];
  }

  /** Total number of LOD levels. */
  get totalLODCount(): number {
    return this.nLods;
  }

  /**
   * Whether the most recently streamed LOD level was fully cache-resident
   * (no chunk fetched from the network). Drives the monitor's
   * cached-vs-streaming residency indicator. Defaults to `true` before any
   * load.
   */
  get lastAllResident(): boolean {
    return this._lastAllResident;
  }

  /**
   * Load gsplats data. On first call, loads LOD 0.
   * Delegates to updateView().
   */
  async loadGSplats(
    viewState: GSplatsViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedGSplatsData> {
    return this.updateView(viewState, session, signal);
  }

  /**
   * Load as many LODs as are cache-warm for the given view state.
   *
   * If the view state changed since the last call, resets and starts from LOD 0.
   * Otherwise, continues loading from where it left off.
   *
   * After returning, prefetches the next unloaded LOD in the background.
   */
  async updateView(
    viewState: GSplatsViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedGSplatsData> {
    // Record the per-pass playback budget FIRST (before the restore branch:
    // a pause re-trigger arrives with the SAME view state — it must still
    // clear the budget so refinement can resume). Deadline is measured from
    // pass start so slow levels consume the budget too.
    this._frameBudgetMs = viewState.frameBudgetMs ?? null;
    const budgetDeadline =
      this._frameBudgetMs !== null ? performance.now() + this._frameBudgetMs : null;

    // A background prefetch (shadow) pass only warms the SliceCache — the
    // SlicePrefetcher discards its return value. So every return below hands
    // back a cheap empty result instead of running `concatenateMemoized`: that
    // O(N) main-thread concat is pure waste for the shadow and, as the ladder
    // deepens across loops, would stall foreground frames.
    const isPrefetch = viewState.prefetch === true;
    const finish = (): LoadedGSplatsData =>
      isPrefetch ? concatenateGSplatsData([]) : this.concatenateMemoized(session);

    // Reset if view state changed. Before discarding the ladder, try the
    // SliceCache: a full-ladder snapshot for this exact view lets us restore
    // `loadedLODs` outright (so `hasMoreLODs` reads false and the refinement
    // loop never re-streams), skipping the whole load+decode.
    if (!this.lastViewState || !viewStatesEqual(viewState, this.lastViewState)) {
      // DEPARTURE store: snapshot the outgoing view's partial ladder under
      // the OUTGOING key before discarding — scrub-back stays warm even when
      // ladders never complete between navigations. Mirrors Points/Lines.
      if (this.lastViewState && this.loadedLODs.length > 0) {
        storeLadder(this.sliceCache, this.path, this.lastViewState, this.loadedLODs, {
          scan: this._frameBudgetMs !== null,
          pin: viewState.prefetch === true,
        });
      }
      const restored = restoreLadder<LoadedGSplatsData>(
        this.sliceCache,
        this.path,
        viewState,
        this.nLods
      );
      // Shallow-copy the CONTAINER: the streaming loop below pushes further
      // levels into loadedLODs and must never mutate the cache's payload
      // array (the elements stay shared read-only — store deep-clones).
      this.loadedLODs = restored ? [...restored] : [];
      // Re-derive the terminal empty-ladder flag from the RESTORED prefix:
      // an S-cache-restored LOD 0 with zero elements is just as terminal
      // as a freshly loaded one. The level===0 empty check in the
      // streaming loop only fires for freshly LOADED levels, so a
      // restored 1-level empty prefix would otherwise stream the higher
      // (equally empty) LODs again on every revisit of the empty slice.
      this._emptyLadder = restored !== null && restored.length > 0 && restored[0].splatCount === 0;
      this._resetGeneration++;
      this.lastViewState = {
        displayDims: [...viewState.displayDims],
        slicePosition: [...viewState.slicePosition],
        tolerance: [...viewState.tolerance],
        dimensions: viewState.dimensions,
      };
      if (restored) {
        this._initialLoadDone = true;
        this._lastAllResident = true;
        // FULL ladder: nothing left to load — short-circuit the whole pass.
        // A PREFIX (stored while a playback budget capped a previous pass)
        // falls through to the loop instead: loading resumes from
        // startLevel = prefix length — within this pass's budget during
        // play, or to completion when idle.
        if (restored.length === this.nLods) {
          return finish();
        }
      }
    } else if (this.lastViewState.dimensions !== viewState.dimensions) {
      // Metadata refresh with an UNCHANGED query determinant (the scene
      // rebuilds the dimensions objects right after the first data load —
      // see view-state-equal.ts): adopt the fresh reference so subsequent
      // compares take the reference-equality fast path instead of
      // re-deriving the dims projection sig on every pass. Content is
      // determinant-equal per the check above, so cache keys (which build
      // from the same determinant) are unaffected.
      this.lastViewState.dimensions = viewState.dimensions;
    }

    // Known-empty slice: LOD 0 committed 0 splats on a prior pass for this
    // SAME view (the view-change branch re-derives the flag from the restored prefix). The
    // ladder is terminal, so skip the streaming loop AND prefetch — otherwise a
    // same-view re-invoke (e.g. refine-on-pause's setDimensionValue(current))
    // would re-enter at startLevel=1 and fetch the higher (also-empty) LODs.
    if (this._emptyLadder) {
      return finish();
    }

    // Stream the LOD ladder under the shared streaming policy (see
    // `streaming-policy.ts`): `playback` commits the cached prefix + a LOD-0
    // first-paint floor and never blocks on fine levels; `prefetch` deepens
    // toward the full decoded ladder (bounded by the pass budget + abort);
    // `refine` streams resident levels and stops at the first cold/slow one.
    const pass = classifyStreamingPass(budgetDeadline !== null, isPrefetch);
    const startLevel = this.loadedLODs.length;

    for (let level = startLevel; level < this.nLods; level++) {
      // A dispose() racing the awaited level below clears `lodLoaders`, so
      // the next iteration would TypeError on `this.lodLoaders[level]` — a
      // teardown mis-counted as a real refinement failure (recordFailure +
      // backoff). Stop streaming instead.
      if (this._disposed) {
        break;
      }
      if (!shouldLoadLevel(pass, level, startLevel)) {
        break;
      }
      // Frame-budget guard: stop as soon as the pass's time is spent — whether
      // many fast levels consumed it or one slow level did. The
      // `level > startLevel` guard keeps the ≥1-level first-paint floor even
      // under tiny budgets.
      if (budgetDeadline !== null && level > startLevel && performance.now() > budgetDeadline) {
        break;
      }
      const t0 = performance.now();
      const { data: lodData, allResident } = await this.lodLoaders[level].updateViewWithResidency(
        viewState,
        session,
        signal
      );
      const elapsed = performance.now() - t0;

      this.loadedLODs.push(lodData);
      this._lastAllResident = allResident;

      if (!this._initialLoadDone) {
        log.custom(
          LogEmoji.BROADCAST,
          Modules.GSPLATS_SPATIAL_INDEX_LOADER,
          `LOD ${level}/${this.nLods - 1}: ${lodData.splatCount} splats (${elapsed.toFixed(1)}ms${allResident ? '' : ', miss'})`
        );
      }

      // Short-circuit: if LOD 0 returned 0 splats, no higher LODs will have
      // visible splats either (LODs are spatially coextensive, LOD 0 is coarsest).
      // Mark the ladder terminal so `hasMoreLODs` reads false and refinement
      // does not fetch+decode the higher (empty) LODs pass after pass.
      if (level === 0 && lodData.splatCount === 0) {
        this._emptyLadder = true;
        break;
      }

      if (shouldStopAfterLevel(pass, level, startLevel, allResident, elapsed)) {
        break;
      }
    }

    if (!this._initialLoadDone && startLevel === 0) {
      this._initialLoadDone = true;
    }

    // Log LOD loading summary (compact, always shown for progressive loaders)
    const totalSplats = this.loadedLODs.reduce((s, d) => s + d.splatCount, 0);
    if (this.loadedLODs.length < this.nLods) {
      log.info(
        Modules.GSPLATS_SPATIAL_INDEX_LOADER,
        `Progressive: ${this.loadedLODs.length}/${this.nLods} LODs loaded (${totalSplats} splats) — refining`
      );
    } else if (startLevel < this.nLods) {
      // Only log "all loaded" when we actually loaded something new this call
      log.info(
        Modules.GSPLATS_SPATIAL_INDEX_LOADER,
        `Progressive: ${this.nLods}/${this.nLods} LODs loaded (${totalSplats} splats) — complete`
      );
    }

    // Fire-and-forget: prefetch next unloaded LOD to warm cache
    // A terminal empty ladder has nothing to prefetch — this also covers
    // the very pass that DISCOVERS the empty LOD 0 (the known-empty
    // early-return above only guards subsequent same-view invokes).
    if (!this._emptyLadder) this.prefetchNextLOD(viewState);

    // Snapshot into the SliceCache (upgrade-if-longer): full ladders always
    // (instant revisit restore); PREFIXES only while a playback budget is
    // active — each playback loop then restores the prefix instantly and
    // deepens it with the leftover budget, converging to full ladders.
    // Gating prefixes on the budget keeps the non-play cost profile (a
    // store per refinement pass would clone O(N²) bytes per slice).
    if (this.loadedLODs.length === this.nLods || this._frameBudgetMs !== null) {
      storeLadder(this.sliceCache, this.path, viewState, this.loadedLODs, {
        scan: this._frameBudgetMs !== null,
        pin: viewState.prefetch === true,
      });
    }

    return finish();
  }

  /**
   * Concatenate loaded LODs, memoized on (resetGeneration, LOD count).
   * An unchanged view state with no new LODs returns the SAME object
   * reference — safe because the result is never mutated downstream
   * (worker projection inputs are structured-cloned, not transferred) —
   * letting the commit pipeline skip no-op re-commits by identity.
   *
   * Each fresh result is also stamped with a PREFIX-LINEAGE parent (the
   * previous same-generation memo) via {@link setPrefixParent}, so the
   * commit layer can recognise a genuine prefix-extension and take the
   * append fast path (depth-sorting Phase 4 Stage 2). The parent is null
   * for the first level of a generation (a view change bumps the reset
   * generation and empties `loadedLODs`), which is correct — the first
   * commit extends nothing.
   */
  private concatenateMemoized(session?: UpdateSession): LoadedGSplatsData {
    const concatSession = session?.begin('Concatenate LODs');
    try {
      if (
        this._concatCache &&
        this._concatCache.generation === this._resetGeneration &&
        this._concatCache.lodCount === this.loadedLODs.length
      ) {
        return this._concatCache.result;
      }
      // Capture the previous SAME-GENERATION memo before overwriting the
      // cache — that (and only that) is the result this one extends.
      const prevMemo =
        this._concatCache && this._concatCache.generation === this._resetGeneration
          ? this._concatCache.result
          : null;
      const result = concatenateGSplatsData(this.loadedLODs);
      setPrefixParent(result, prevMemo);
      this._concatCache = {
        generation: this._resetGeneration,
        lodCount: this.loadedLODs.length,
        result,
      };
      return result;
    } finally {
      concatSession?.end();
    }
  }

  /**
   * Prefetch the next unloaded LOD's chunks into cache.
   *
   * This is fire-and-forget: the prefetched chunks land in the L0/L1 cache
   * and become fast cache hits on the next updateView() call.
   *
   * Uses prefetchChunks() which performs the spatial index query and zarr
   * get() calls (populating the cache) WITHOUT allocating full-size output
   * buffers or running the accumulator — avoiding wasted memory.
   */
  private prefetchNextLOD(viewState: GSplatsViewState): void {
    // Same teardown race as the streaming loop: a dispose() between the
    // awaited level and this fire-and-forget clears `lodLoaders`, and
    // indexing it would TypeError before the .catch can swallow anything.
    if (this._disposed) return;
    const nextLevel = this.loadedLODs.length;
    if (nextLevel >= this.nLods) return;

    // Fire and forget — fetch chunks into cache without decoding to output buffers
    this.lodLoaders[nextLevel].prefetchChunks(viewState).catch(() => {
      // Ignore errors from prefetch (network failures, aborts)
    });
  }

  // ---- LoaderMonitor surface (delegated to ProgressiveMonitorAdapter) ----
  // Lets `connectLoaderToMonitor` wire the progressive node to the data
  // monitor so its query/throughput/memory telemetry is reported (aggregated
  // across LODs, re-pathed to this node) instead of silently dropped.

  addEventListener(listener: MonitorEventListener): void {
    this.monitor.addEventListener(listener);
  }

  removeEventListener(listener: MonitorEventListener): void {
    this.monitor.removeEventListener(listener);
  }

  getActiveQueries(): QueryInfo[] {
    return this.monitor.getActiveQueries();
  }

  getMetrics(): LoaderMetrics {
    return this.monitor.getMetrics();
  }

  /**
   * Clean up all LOD loaders.
   */
  dispose(): void {
    this._disposed = true;
    for (const loader of this.lodLoaders) {
      loader.dispose();
    }
    this.lodLoaders = [];
    this.loadedLODs = [];
    this.lastViewState = null;
    this._concatCache = null;
  }
}
