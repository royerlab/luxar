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
  PositionArray,
  ScalarArray,
} from '../../types/points';
import { setPrefixParent } from '../../types/prefix-lineage';
import type { PointsSpatialIndexLoader } from './points-spatial-index-loader';
import type { UpdateSession } from '../../profiling/update-profiler';
import type {
  LoaderMetrics,
  MonitorEventListener,
  QueryInfo,
} from '../../types/data-monitor-types';
import { ProgressiveMonitorAdapter } from '../loaders/progressive-monitor-adapter';
import {
  ArenaField,
  OptionalLadderField,
  setAppendSpan,
  validateLadderFieldDtype,
  type ArenaFieldStats,
} from '../loaders/progressive/concat-arena';
import { concatOptionalField, concatRequiredField } from '../loaders/progressive/concat-helpers';
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
 * Concatenate per-LOD `LoadedPointsData` into one. Optional attribute
 * arrays (`colors`, `radii`, `sharpness`, `scalars`) are concatenated
 * only when ALL levels carry them (mixed-presence is dropped — keeps
 * the loader simple and matches the writer's all-or-nothing per-attr
 * policy).
 *
 * Exported as the REFERENCE implementation: multi-level ladders build
 * incrementally in {@link PointsLadderArena} (byte-equivalence pinned by
 * tests); this full rebuild still serves the k ≤ 1 cases.
 */
export function concatenatePointsData(parts: LoadedPointsData[]): LoadedPointsData {
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
  const positions = concatRequiredField(parts, (p) => p.positions, count, 3, 'positions');

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
  // Color LAYOUT (3 = RGB, 4 = RGBA) strides the concat — a hardcoded 3
  // would truncate + misalign an RGBA additive ladder (the gsplat colorK
  // lesson, PR #620). Layout is a property of the dataset, uniform across
  // its LODs; a mismatch is malformed data — fail fast, naming the level.
  const colorK: 3 | 4 = parts.find((p) => p.colors)?.colorComponents ?? 3;
  for (const [levelIdx, part] of parts.entries()) {
    if (part.colors && (part.colorComponents ?? 3) !== colorK) {
      throw new Error(
        'concatenatePointsData: mixed color layouts across LOD levels ' +
          `(level ${levelIdx}: ${part.colorComponents ?? 3} vs ${colorK} ` +
          'components) — ladder levels must share the color layout (RGB vs RGBA).'
      );
    }
  }
  const colors = concatOptionalField(parts, (p) => p.colors as ColorArray, count, colorK, 'colors');
  if (colors) {
    result.colors = colors;
    result.colorComponents = colorK;
  }
  const radii = concatOptionalField(parts, (p) => p.radii as ScalarArray, count, 1, 'radii');
  if (radii) result.radii = radii;
  const sharpness = concatOptionalField(
    parts,
    (p) => p.sharpness as ScalarArray,
    count,
    1,
    'sharpness'
  );
  if (sharpness) result.sharpness = sharpness;
  const scalars = concatOptionalField(parts, (p) => p.scalars as ScalarArray, count, 1, 'scalars');
  if (scalars) result.scalars = scalars;

  return result;
}

/**
 * Growable ladder arena for the points progressive concat (perf lever L2).
 *
 * Incremental equivalent of {@link concatenatePointsData}: appending level
 * *k* copies ONLY level *k*'s bytes (amortized O(N_total) across the ladder
 * instead of O(k·N) full rebuilds), with `snapshot()` byte-identical to the
 * reference concat of the same parts — same all-or-nothing optional-field
 * drops, color-layout/dtype validation (reference order + messages), and
 * metadata aggregation. Snapshots are prefix-stable `subarray` views (see
 * `concat-arena.ts` for the aliasing contract). One arena per loader reset
 * generation. Exported for the arena-vs-reference equivalence tests.
 */
export class PointsLadderArena {
  private ndim = 3;
  // INVARIANT (see the reference concat): points positions are ALWAYS
  // 3D-projected, stride 3, regardless of `ndim`.
  private positions: ArenaField<PositionArray> | null = null;
  private readonly colors = new OptionalLadderField<ColorArray>();
  private readonly radii = new OptionalLadderField<ScalarArray>();
  private readonly sharpness = new OptionalLadderField<ScalarArray>();
  private readonly scalars = new OptionalLadderField<ScalarArray>();
  // Ladder color layout (3 RGB / 4 RGBA), established by the FIRST
  // color-carrying level and validated for every later carrier — even when
  // the merged colors field itself is dropped (reference parity).
  private colorK: 3 | 4 | null = null;
  private totalPointsFirst = 0;
  private usedSpatialIndexAll = true;
  private readonly aggBounds = new THREE.Box3();
  private appendedParts = 0;
  private total = 0;

  /** Elements (points) appended so far. */
  get totalElements(): number {
    return this.total;
  }

  /** Aggregated copy-work counters across fields (test/verification hook). */
  debugStats(): ArenaFieldStats {
    const stats: ArenaFieldStats = { appendedEntries: 0, reallocCopiedEntries: 0 };
    const fields = [
      this.positions,
      this.colors.field,
      this.radii.field,
      this.sharpness.field,
      this.scalars.field,
    ];
    for (const f of fields) {
      if (!f) continue;
      stats.appendedEntries += f.stats.appendedEntries;
      stats.reallocCopiedEntries += f.stats.reallocCopiedEntries;
    }
    return stats;
  }

  /**
   * Append `parts[appendedSoFar..parts.length)`. `isFinal` = ladder-complete
   * batch (exact capacity + trim). All throwing validation runs before any
   * write, so a malformed level leaves the arena unchanged.
   */
  appendThrough(parts: LoadedPointsData[], isFinal: boolean): void {
    const from = this.appendedParts;
    // APPEND-ONLY PRECONDITION: `parts` must extend the levels already
    // appended (the loader only ever pushes onto `loadedLODs` within a reset
    // generation; a view change makes a fresh arena). A shorter list would
    // silently snapshot stale extra content, so fail loudly instead.
    if (parts.length < from) {
      throw new Error(
        `PointsLadderArena: ladder shrank (${parts.length} levels vs ${from} ` +
          'already appended) — an arena is append-only within a reset generation.'
      );
    }
    if (parts.length === from) {
      if (isFinal) this.trim();
      return;
    }
    if (from === 0) {
      this.ndim = parts[0].ndim;
      this.totalPointsFirst = parts[0].metadata.totalPoints;
    }

    // ---- Validation (reference order: positions dtype → color layout →
    // per-optional-field presence/dtype), no state mutation. ----
    const posCtor = (this.positions?.elementCtor ?? parts[0].positions.constructor) as new (
      n: number
    ) => PositionArray;
    validateLadderFieldDtype(parts, from, (p) => p.positions, posCtor, 'positions');

    let colorK = this.colorK;
    if (colorK === null) {
      const firstWithColors = parts.slice(from).find((p) => p.colors);
      if (firstWithColors) colorK = firstWithColors.colorComponents ?? 3;
    }
    if (colorK !== null) {
      for (let i = from; i < parts.length; i++) {
        if (parts[i].colors && (parts[i].colorComponents ?? 3) !== colorK) {
          throw new Error(
            'concatenatePointsData: mixed color layouts across LOD levels ' +
              `(level ${i}: ${parts[i].colorComponents ?? 3} vs ${colorK} ` +
              'components) — ladder levels must share the color layout (RGB vs RGBA).'
          );
        }
      }
    }

    const colorsPlan = this.colors.plan(parts, from, (p) => p.colors, 'colors');
    const radiiPlan = this.radii.plan(parts, from, (p) => p.radii, 'radii');
    const sharpnessPlan = this.sharpness.plan(parts, from, (p) => p.sharpness, 'sharpness');
    const scalarsPlan = this.scalars.plan(parts, from, (p) => p.scalars, 'scalars');

    // ---- Writes (validation passed; nothing below throws) ----
    this.colorK = colorK;
    let batchPoints = 0;
    for (let i = from; i < parts.length; i++) batchPoints += parts[i].pointCount;
    const newTotal = this.total + batchPoints;

    if (!this.positions) {
      this.positions = new ArenaField<PositionArray>(posCtor, 3, newTotal);
    } else {
      this.positions.ensureCapacity(newTotal, isFinal);
    }
    this.colors.apply(colorsPlan, colorK ?? 3, newTotal, isFinal);
    this.radii.apply(radiiPlan, 1, newTotal, isFinal);
    this.sharpness.apply(sharpnessPlan, 1, newTotal, isFinal);
    this.scalars.apply(scalarsPlan, 1, newTotal, isFinal);

    const positions = this.positions;
    for (let i = from; i < parts.length; i++) {
      const part = parts[i];
      const n = part.pointCount;
      positions.append(part.positions, n);
      this.colors.field?.append(part.colors as ColorArray, n);
      this.radii.field?.append(part.radii as ScalarArray, n);
      this.sharpness.field?.append(part.sharpness as ScalarArray, n);
      this.scalars.field?.append(part.scalars as ScalarArray, n);
      this.aggBounds.union(part.metadata.bounds);
      this.usedSpatialIndexAll = this.usedSpatialIndexAll && part.metadata.usedSpatialIndex;
      this.total += n;
      this.appendedParts++;
    }

    if (isFinal) this.trim();
  }

  /** Snapshot the current contents as prefix-stable views (fresh object). */
  snapshot(): LoadedPointsData {
    const result: LoadedPointsData = {
      positions: this.positions?.view() ?? new Float32Array(0),
      pointCount: this.total,
      ndim: this.ndim,
      metadata: {
        totalPoints: this.totalPointsFirst,
        loadedPoints: this.total,
        // Clone: the aggregate keeps growing with later levels; each
        // snapshot owns its bounds (reference allocates one per rebuild).
        bounds: this.aggBounds.clone(),
        usedSpatialIndex: this.usedSpatialIndexAll,
      },
    };
    if (this.colors.field) {
      result.colors = this.colors.field.view();
      result.colorComponents = this.colorK ?? 3;
    }
    if (this.radii.field) result.radii = this.radii.field.view();
    if (this.sharpness.field) result.sharpness = this.sharpness.field.view();
    if (this.scalars.field) result.scalars = this.scalars.field.view();
    return result;
  }

  private trim(): void {
    this.positions?.trimToFit();
    this.colors.field?.trimToFit();
    this.radii.field?.trimToFit();
    this.sharpness.field?.trimToFit();
    this.scalars.field?.trimToFit();
  }
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
  // Growable ladder arena backing `concatenateMemoized` (perf lever L2):
  // appending a level copies only that level's bytes instead of rebuilding
  // the whole concat. One arena per reset generation. Mirrors
  // GSplatsProgressiveLoader.
  private _arena: PointsLadderArena | null = null;
  private _arenaGeneration = -1;
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

  // Set when LOD 0 committed 0 elements for the current view: the slice is
  // empty, so every higher (spatially-coextensive) LOD is empty too and the
  // ladder is TERMINAL. Makes `hasMoreLODs` read false so refinement doesn't
  // fetch+decode the higher empty LODs pass after pass. Reset on view change.
  // Mirrors GSplatsProgressiveLoader.
  private _emptyLadder = false;

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
    // Empty slice (LOD 0 committed 0 elements): terminal ladder — no further
    // work, so refinement doesn't fetch the higher empty LODs pass after pass.
    if (this._emptyLadder) return false;
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

    // Background prefetch (shadow) passes only warm the SliceCache — the
    // SlicePrefetcher discards the return value — so hand back a cheap empty
    // result instead of the O(N) main-thread concat, which would stall
    // foreground frames as the ladder deepens. Mirrors GSplatsProgressiveLoader.
    const isPrefetch = viewState.prefetch === true;
    const finish = (): LoadedPointsData =>
      isPrefetch ? concatenatePointsData([]) : this.concatenateMemoized(session);

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
      // Re-derive the terminal empty-ladder flag from the RESTORED prefix:
      // an S-cache-restored LOD 0 with zero elements is just as terminal
      // as a freshly loaded one. The level===0 empty check in the
      // streaming loop only fires for freshly LOADED levels, so a
      // restored 1-level empty prefix would otherwise stream the higher
      // (equally empty) LODs again on every revisit of the empty slice.
      this._emptyLadder =
        restored !== null && restored.length > 0 && restored[0].positions.length === 0;
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

    // Known-empty slice: LOD 0 committed 0 points on a prior pass for this SAME
    // view (the view-change branch re-derives the flag from the restored prefix). Terminal
    // ladder — skip the streaming loop AND prefetch so a same-view re-invoke
    // (e.g. refine-on-pause) doesn't fetch the higher (empty) LODs. Mirrors
    // GSplatsProgressiveLoader.
    if (this._emptyLadder) {
      return finish();
    }

    // Stream under the shared streaming policy (see `streaming-policy.ts`):
    // `playback` commits the cached prefix + a LOD-0 first-paint floor and
    // never blocks on fine levels; `prefetch` deepens toward the full decoded
    // ladder (abort-safe, stored per level); `refine` streams resident levels
    // and stops at the first cold/slow one. Mirrors GSplatsProgressiveLoader.
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
      // Playback frame budget: stop as soon as the tick's time is spent
      // (≥1 level always loads — `level > startLevel` guard).
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
      // have 0 visible (LODs are spatially coextensive). Mark terminal so
      // `hasMoreLODs` reads false and refinement skips the empty higher LODs.
      if (level === 0 && lodData.positions.length === 0) {
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

    // A terminal empty ladder has nothing to prefetch — this also covers
    // the very pass that DISCOVERS the empty LOD 0 (the known-empty
    // early-return above only guards subsequent same-view invokes).
    if (!this._emptyLadder) this.prefetchNextLOD(viewState);

    // Snapshot into the SliceCache (upgrade-if-longer): full ladders always;
    // PREFIXES only while a playback budget is active. Mirrors
    // GSplatsProgressiveLoader.
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
   * commit extends nothing. Mirrors GSplatsProgressiveLoader.
   *
   * Multi-level concats build incrementally in a per-generation
   * {@link PointsLadderArena} (perf lever L2): only the NEW levels' bytes
   * are copied, and the result is a prefix-stable view — byte-identical to
   * the reference {@link concatenatePointsData} rebuild. k ≤ 1 keeps the
   * reference path (empty result / the raw single part as-is). Each fresh
   * result is also stamped with its {@link setAppendSpan | append span}.
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
      // Capture the previous SAME-GENERATION memo before overwriting the
      // cache — that (and only that) is the result this one extends.
      const prevMemo =
        this._concatCache && this._concatCache.generation === this._resetGeneration
          ? this._concatCache.result
          : null;
      const k = this.loadedLODs.length;
      let result: LoadedPointsData;
      if (k <= 1) {
        // Reference shape for the trivial cases: [] → empty result,
        // [part] → the raw part as-is (no copy; the arena starts at k=2).
        result = concatenatePointsData(this.loadedLODs);
      } else {
        if (!this._arena || this._arenaGeneration !== this._resetGeneration) {
          this._arena = new PointsLadderArena();
          this._arenaGeneration = this._resetGeneration;
        }
        this._arena.appendThrough(this.loadedLODs, k === this.nLods);
        result = this._arena.snapshot();
      }
      setPrefixParent(result, prevMemo);
      const prevElements = prevMemo?.pointCount ?? 0;
      setAppendSpan(result, {
        fromElement: prevElements,
        elementCount: result.pointCount - prevElements,
      });
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
    // Same teardown race as the streaming loop: a dispose() between the
    // awaited level and this fire-and-forget clears `lodLoaders`, and
    // indexing it would TypeError before the .catch can swallow anything.
    if (this._disposed) return;
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
    this._arena = null;
  }
}
