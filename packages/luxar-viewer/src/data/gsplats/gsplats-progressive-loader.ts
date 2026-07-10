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
import type { GSplatsSpatialIndexLoader } from './gsplats-spatial-index-loader';
import type { UpdateSession } from '../../profiling/update-profiler';
import type {
  LoaderMetrics,
  MonitorEventListener,
  QueryInfo,
} from '../../types/data-monitor-types';
import { ProgressiveMonitorAdapter } from '../loaders/progressive-monitor-adapter';
import { concatRequiredField } from '../loaders/progressive/concat-helpers';
import { CACHE_HIT_THRESHOLD_MS } from '../loaders/progressive/constants';
import { restoreLadder, storeLadder } from '../loaders/progressive/slice-cache-helper';
import type { SliceCache } from '../../cache/slice-cache';
import { log, Modules, LogEmoji } from '../../utils/log';

/**
 * Compare two GSplatsViewState objects for query-affecting equality.
 * Compares displayDims, slicePosition, and tolerance element-wise.
 *
 * INVARIANT: this equality is the linchpin of the no-op commit skip.
 * When it reports equal AND no new LODs loaded, `updateView` returns the
 * MEMOIZED concatenation — same object reference — and the commit pipeline
 * treats reference equality as content equality
 * (`mesh.userData.committedData === data`). Any new query-affecting field
 * added to the view state MUST be compared here, or the skip will serve
 * stale data.
 */
function viewStatesEqual(a: GSplatsViewState, b: GSplatsViewState): boolean {
  if (a.displayDims.length !== b.displayDims.length) return false;
  for (let i = 0; i < a.displayDims.length; i++) {
    if (a.displayDims[i] !== b.displayDims[i]) return false;
  }

  if (a.slicePosition.length !== b.slicePosition.length) return false;
  for (let i = 0; i < a.slicePosition.length; i++) {
    if (a.slicePosition[i] !== b.slicePosition[i]) return false;
  }

  if (a.tolerance.length !== b.tolerance.length) return false;
  for (let i = 0; i < a.tolerance.length; i++) {
    if (a.tolerance[i] !== b.tolerance[i]) return false;
  }

  // Compare dimensions (reference equality or JSON for rare metadata changes)
  if (a.dimensions !== b.dimensions) {
    if (!a.dimensions || !b.dimensions) return false;
    if (JSON.stringify(a.dimensions) !== JSON.stringify(b.dimensions)) return false;
  }

  return true;
}

/**
 * Concatenate multiple LoadedGSplatsData into one.
 * Allocates new arrays sized for the total splat count and copies data.
 */
function concatenateGSplatsData(parts: LoadedGSplatsData[]): LoadedGSplatsData {
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
  const totalSplats = parts.reduce((sum, p) => sum + p.splatCount, 0);
  const cholSize = (ndim * (ndim + 1)) / 2;
  const count = (p: LoadedGSplatsData) => p.splatCount;

  // Required per-splat fields via the shared helpers (dtype preserved).
  const positions = concatRequiredField(parts, (p) => p.positions, count, ndim);
  const amplitudes = concatRequiredField(parts, (p) => p.amplitudes, count);
  const choleskyFactors = concatRequiredField(parts, (p) => p.choleskyFactors, count, cholSize);

  // Bespoke: colors fill missing LODs with white (per-dtype fill value).
  const firstWithColors = parts.find((p) => p.colors !== null);
  let colors: Float32Array | Uint8Array | Uint16Array | null = null;
  if (firstWithColors?.colors) {
    if (firstWithColors.colors instanceof Uint8Array) {
      colors = new Uint8Array(totalSplats * 3);
    } else if (firstWithColors.colors instanceof Uint16Array) {
      colors = new Uint16Array(totalSplats * 3);
    } else {
      colors = new Float32Array(totalSplats * 3);
    }
  }

  let offset = 0;
  for (const part of parts) {
    if (colors && part.colors) {
      colors.set(part.colors, offset * 3);
    } else if (colors && !part.colors) {
      // Fill with white (1.0 for Float32, 255 for Uint8, 65535 for Uint16)
      const fillValue =
        colors instanceof Uint8Array ? 255 : colors instanceof Uint16Array ? 65535 : 1.0;
      for (let i = 0; i < part.splatCount * 3; i++) {
        colors[offset * 3 + i] = fillValue;
      }
    }

    offset += part.splatCount;
  }

  return {
    positions,
    amplitudes,
    choleskyFactors,
    colors,
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
          return this.concatenateMemoized(session);
        }
      }
    }

    // Load LODs sequentially, stopping at first slow (cache-miss) load
    const startLevel = this.loadedLODs.length;

    for (let level = startLevel; level < this.nLods; level++) {
      // Playback frame budget: stop as soon as the tick's time is spent —
      // whether many fast levels consumed it or one slow level did. Checked
      // at loop top (skips work known to be over budget); the
      // `level > startLevel` guard keeps the ≥1-level first-paint floor
      // even under tiny budgets.
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
      // visible splats either (LODs are spatially coextensive, LOD 0 is coarsest)
      if (level === 0 && lodData.splatCount === 0) {
        break;
      }

      // Stop after a cache miss (this level required a fresh fetch/decode) so
      // the frame can render; the refinement loop picks up the rest. The
      // wall-clock budget is kept only as a secondary guard against a huge
      // resident-but-slow level (GC pause, slow projection, probe gap).
      // Always load at least LOD 0 (level === startLevel) regardless.
      if (level > startLevel && (!allResident || elapsed > CACHE_HIT_THRESHOLD_MS)) {
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
    this.prefetchNextLOD(viewState);

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

    return this.concatenateMemoized(session);
  }

  /**
   * Concatenate loaded LODs, memoized on (resetGeneration, LOD count).
   * An unchanged view state with no new LODs returns the SAME object
   * reference — safe because the result is never mutated downstream
   * (worker projection inputs are structured-cloned, not transferred) —
   * letting the commit pipeline skip no-op re-commits by identity.
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
      const result = concatenateGSplatsData(this.loadedLODs);
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
