/**
 * Progressive Points loader for multi-additive-LOD datasets.
 *
 * Wraps N `PointsSpatialIndexLoader` instances (one per `additive_<i>`
 * subgroup) using the Composite pattern. Implements the same
 * `PointsDataLoader` (aka `DataLoader`) interface so the scene
 * loader's update loop works unchanged.
 *
 * Loading strategy mirrors `GSplatsProgressiveLoader`:
 *
 * - On each `updateView()` call, loads LODs sequentially starting from
 *   LOD 0.
 * - Stops at the first LOD whose load exceeds the cache-hit threshold
 *   (the shared `CACHE_HIT_THRESHOLD_MS` in
 *   `loaders/progressive/constants` — one threshold across all geometry
 *   types).
 * - After returning, fires a prefetch for the next unloaded LOD so the
 *   refinement loop hits a warm cache on the next call.
 *
 * LODs are additive: LOD 0 contains the coarsest subset, each
 * subsequent LOD adds more elements. The loader concatenates loaded
 * LODs into a single `LoadedPointsData`.
 *
 * @module data/points/points-progressive-loader
 */

import * as THREE from 'three';
import type {
  ColorArray,
  LoadedPointsData,
  PointsDataLoader,
  PointsViewState,
  ScalarArray,
} from '../../types/points';
import type { PointsSpatialIndexLoader } from './points-spatial-index-loader';
import type { UpdateSession } from '../../profiling/update-profiler';
import type {
  LoaderMetrics,
  MonitorEventListener,
  QueryInfo,
} from '../../types/data-monitor-types';
import { ProgressiveMonitorAdapter } from '../loaders/progressive-monitor-adapter';
import { concatOptionalField, concatRequiredField } from '../loaders/progressive/concat-helpers';
import { CACHE_HIT_THRESHOLD_MS } from '../loaders/progressive/constants';
import { restoreLadder, storeLadder } from '../loaders/progressive/slice-cache-helper';
import type { SliceCache } from '../../cache/slice-cache';
import { log, Modules, LogEmoji } from '../../utils/log';

/**
 * Element-wise viewstate equality (query-affecting fields only).
 *
 * INVARIANT: this equality is the linchpin of the no-op commit skip.
 * When it reports equal AND no new LODs loaded, `updateView` returns the
 * MEMOIZED concatenation — same object reference — and the commit pipeline
 * treats reference equality as content equality
 * (`mesh.userData.committedData === data`). Any new query-affecting field
 * added to the view state MUST be compared here, or the skip will serve
 * stale data.
 */
function viewStatesEqual(a: PointsViewState, b: PointsViewState): boolean {
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
  if (a.dimensions !== b.dimensions) {
    if (!a.dimensions || !b.dimensions) return false;
    if (JSON.stringify(a.dimensions) !== JSON.stringify(b.dimensions)) return false;
  }
  return true;
}

/**
 * Concatenate per-LOD `LoadedPointsData` into one. Optional attribute
 * arrays (`colors`, `radii`, `sharpness`, `scalars`) are concatenated
 * only when ALL levels carry them (mixed-presence is dropped — keeps
 * the loader simple and matches the writer's all-or-nothing per-attr
 * policy).
 */
function concatenatePointsData(parts: LoadedPointsData[]): LoadedPointsData {
  if (parts.length === 0) {
    // Construct a minimal LoadedPointsData with empty arrays so the
    // commit pipeline doesn't NPE on edge cases (no LODs visible yet).
    return {
      positions: new Float32Array(0),
      pointCount: 0,
      ndim: 3,
      metadata: {
        totalPoints: 0,
        loadedPoints: 0,
        bounds: new THREE.Box3(),
        usedSpatialIndex: false,
      },
    };
  }
  if (parts.length === 1) {
    return parts[0];
  }

  const ndim = parts[0].ndim;
  const totalPoints = parts.reduce((sum, p) => sum + p.pointCount, 0);
  const count = (p: LoadedPointsData) => p.pointCount;

  // INVARIANT: `positions` is ALWAYS 3D-projected, stride 3 — Points is the
  // one geometry whose loader folds nD→3D projection into loadPoints() itself
  // (the accumulator's getData returns `positionBuffer.subarray(0, count*3)`),
  // while `ndim` still reports the ORIGINAL dimensionality. Concatenating at
  // stride `ndim` here would scatter every level after the first to wrong
  // offsets for >3D data. GSplats/Lines correctly concat their positions at
  // `ndim` because their loaders return raw nD data (projection runs later in
  // the process step).
  const positions = concatRequiredField(parts, (p) => p.positions, count, 3);

  // Aggregate bounds across all loaded levels.
  const aggBounds = new THREE.Box3();
  for (const part of parts) {
    aggBounds.union(part.metadata.bounds);
  }

  const result: LoadedPointsData = {
    positions,
    pointCount: totalPoints,
    ndim,
    metadata: {
      totalPoints: parts[0].metadata.totalPoints,
      loadedPoints: totalPoints,
      bounds: aggBounds,
      usedSpatialIndex: parts.every((p) => p.metadata.usedSpatialIndex),
    },
  };

  // Optional per-point fields: all-or-nothing across LODs (dtype preserved).
  const colors = concatOptionalField(parts, (p) => p.colors as ColorArray, count, 3);
  if (colors) result.colors = colors;
  const radii = concatOptionalField(parts, (p) => p.radii as ScalarArray, count);
  if (radii) result.radii = radii;
  const sharpness = concatOptionalField(parts, (p) => p.sharpness as ScalarArray, count);
  if (sharpness) result.sharpness = sharpness;
  const scalars = concatOptionalField(parts, (p) => p.scalars as ScalarArray, count);
  if (scalars) result.scalars = scalars;

  return result;
}

/**
 * Progressive Points loader.
 */
export class PointsProgressiveLoader implements PointsDataLoader {
  private lodLoaders: PointsSpatialIndexLoader[];
  private loadedLODs: LoadedPointsData[] = [];
  private lastViewState: PointsViewState | null = null;
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
    result: LoadedPointsData;
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
  // directive (never part of lastViewState / viewStatesEqual / cache keys).
  // Mirrors GSplatsProgressiveLoader.
  private _frameBudgetMs: number | null = null;

  constructor(
    lodLoaders: PointsSpatialIndexLoader[],
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

  /** Whether more LOD levels remain to load for the current view state. */
  get hasMoreLODs(): boolean {
    // A disposed loader has work-state cleared; report no further work so a
    // refinement loop holding a stale reference stops instead of indexing
    // into the now-empty lodLoaders. Mirrors GSplatsProgressiveLoader.
    if (this._disposed) return false;
    // While a playback frame budget is active, the budgeted prefix IS the
    // target: no background refinement between animation ticks; the commit
    // stamps the prefix complete. Mirrors GSplatsProgressiveLoader.
    if (this._frameBudgetMs !== null) return false;
    return this.loadedLODs.length < this.nLods;
  }

  get loadedLODCount(): number {
    return this.loadedLODs.length;
  }

  get totalLODCount(): number {
    return this.nLods;
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

  /**
   * Whether the most recently streamed LOD level was fully cache-resident.
   * Drives the monitor's residency indicator. Defaults to `true`.
   */
  get lastAllResident(): boolean {
    return this._lastAllResident;
  }

  async loadPoints(
    viewState: PointsViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedPointsData> {
    return this.updateView(viewState, session, signal);
  }

  async updateView(
    viewState: PointsViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedPointsData> {
    // Record the per-pass playback budget FIRST (before the restore branch:
    // a pause re-trigger arrives with the SAME view state — it must still
    // clear the budget). Mirrors GSplatsProgressiveLoader.
    this._frameBudgetMs = viewState.frameBudgetMs ?? null;
    const budgetDeadline =
      this._frameBudgetMs !== null ? performance.now() + this._frameBudgetMs : null;

    if (!this.lastViewState || !viewStatesEqual(viewState, this.lastViewState)) {
      // DEPARTURE store: snapshot the OUTGOING view's partial ladder before
      // discarding it, keyed under the OUTGOING view (lastViewState — never
      // the incoming one). Scrubbing faster than the ladder completes would
      // otherwise store nothing at all (completion never happens), making
      // scrub-back — the S-cache's headline case — always cold. One clone
      // per slice-leave; upgrade-if-longer makes re-departures cheap no-ops.
      if (this.lastViewState && this.loadedLODs.length > 0) {
        storeLadder(this.sliceCache, this.path, this.lastViewState, this.loadedLODs, {
          scan: this._frameBudgetMs !== null,
          pin: viewState.prefetch === true,
        });
      }
      // Try the SliceCache before discarding the ladder (see GSplats loader).
      const restored = restoreLadder<LoadedPointsData>(
        this.sliceCache,
        this.path,
        viewState,
        this.nLods
      );
      // Shallow-copy the CONTAINER: the streaming loop below pushes further
      // levels and must never mutate the cache's payload array (elements
      // stay shared read-only). Mirrors GSplatsProgressiveLoader.
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
        // FULL ladder short-circuits; a PREFIX falls through to the loop
        // (startLevel = prefix length). Mirrors GSplatsProgressiveLoader.
        if (restored.length === this.nLods) {
          return this.concatenateMemoized(session);
        }
      }
    }

    const startLevel = this.loadedLODs.length;

    for (let level = startLevel; level < this.nLods; level++) {
      // Playback frame budget: stop as soon as the tick's time is spent
      // (≥1 level always loads — `level > startLevel` guard). Mirrors
      // GSplatsProgressiveLoader.
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
          Modules.SPATIAL_INDEX_LOADER,
          `LOD ${level}/${this.nLods - 1}: ${
            lodData.positions.length / 3
          } points (${elapsed.toFixed(1)}ms${allResident ? '' : ', miss'})`
        );
      }

      // Short-circuit: LOD 0 with 0 points → higher LODs would also
      // have 0 visible (LODs are spatially coextensive).
      if (level === 0 && lodData.positions.length === 0) {
        break;
      }

      // Stop after a cache miss; refinement loop continues next frame. The
      // wall-clock budget remains a secondary guard. LOD 0 always loads.
      if (level > startLevel && (!allResident || elapsed > CACHE_HIT_THRESHOLD_MS)) {
        break;
      }
    }

    if (!this._initialLoadDone && startLevel === 0) {
      this._initialLoadDone = true;
    }

    const totalPoints = this.loadedLODs.reduce((s, d) => s + d.positions.length / 3, 0);
    if (this.loadedLODs.length < this.nLods) {
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Progressive Points: ${this.loadedLODs.length}/${this.nLods} LODs (${totalPoints} points) — refining`
      );
    } else if (startLevel < this.nLods) {
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Progressive Points: ${this.nLods}/${this.nLods} LODs (${totalPoints} points) — complete`
      );
    }

    this.prefetchNextLOD(viewState);

    // Snapshot into the SliceCache (upgrade-if-longer): full ladders always;
    // PREFIXES only while a playback budget is active. Mirrors
    // GSplatsProgressiveLoader.
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
  private concatenateMemoized(session?: UpdateSession): LoadedPointsData {
    const concatSession = session?.begin('Concatenate LODs');
    try {
      if (
        this._concatCache &&
        this._concatCache.generation === this._resetGeneration &&
        this._concatCache.lodCount === this.loadedLODs.length
      ) {
        return this._concatCache.result;
      }
      const result = concatenatePointsData(this.loadedLODs);
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

  private prefetchNextLOD(viewState: PointsViewState): void {
    const nextLevel = this.loadedLODs.length;
    if (nextLevel >= this.nLods) return;
    // Fire-and-forget; errors ignored (network failures, aborts).
    void this.lodLoaders[nextLevel].prefetchChunks(viewState).catch(() => {
      /* ignore */
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
