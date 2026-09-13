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
import { setPrefixParent } from '../../types/prefix-lineage';
import type { PointsSpatialIndexLoader } from './points-spatial-index-loader';
import type { UpdateSession } from '../../profiling/update-profiler';
import type {
  LoaderMetrics,
  MonitorEventListener,
  QueryInfo,
} from '../../types/data-monitor-types';
import { ProgressiveMonitorAdapter } from '../loaders/progressive-monitor-adapter';
import { concatOptionalField, concatRequiredField } from '../loaders/progressive/concat-helpers';
import {
  classifyStreamingPass,
  resolveLadderDepth,
  shouldStopBeforeLevel,
  shouldStopAfterLevel,
} from '../loaders/progressive/streaming-policy';
import { config } from '../../config';
import {
  deleteLadder,
  measureLodBytes,
  restoreLadderSnapshot,
  storeLadder,
} from '../loaders/progressive/slice-cache-helper';
import { planLadderRollback } from '../loaders/progressive/pass-rollback';
import { POINT_FLOATS_PER_POINT } from '../../rendering/element-texture-layout';
import {
  ladderResidentBytes,
  type LadderResidency,
} from '../scene-loader/progressive/residency-budget';
import { viewStatesEqual } from '../loaders/progressive/view-state-equal';
import type { SliceCache } from '../../cache/slice-cache';
import { log, Modules, LogEmoji } from '../../utils/log';
import { timeLodStageWithResult } from '../scene-loader/lod-load-stats';

/**
 * Compose the levels' slot → on-disk maps into ONE map in the parent's union
 * CSR index space (`additive_0 || additive_1 || …`), by offsetting level `i`
 * with the preceding levels' ON-DISK counts (`levelOffsets`, #1439).
 *
 * `levelOffsets` is CSR-style: `nLods + 1` entries, so level `i` owns the
 * half-open union range `[levelOffsets[i], levelOffsets[i + 1])`. A composed id
 * outside its OWN level's range would name a real row belonging to a SIBLING
 * level — a confidently wrong label rather than a missing one — so each id is
 * bounded, not just shifted.
 *
 * A level that publishes no map of its own took the projection's identity fast
 * path (one range starting at 0, nothing compacted), so its slot `k` IS its
 * level-space index and contributes `levelOffsets[i] + k`. A level whose
 * projection WANTED a map but could not build one flags
 * `elementIdsUnavailable`: its slots mean nothing, identity is not a legal
 * substitute, and the whole union map is refused.
 *
 * Returns `undefined` when the identity holds across the whole ladder (no level
 * published a map AND every resident level is complete, so slot === union
 * on-disk index; the common fully-loaded unsliced case stays allocation-free,
 * matching flat Points), or when the inputs are inconsistent.
 *
 * REFUSING IS NOT SUPPRESSION. With no map, `resolveOnDiskElementId` returns the
 * raw slot, and on a sliced ladder that slot is itself a wrong CSR row — there
 * is no "no answer" channel on the hover path. What refusing buys is narrower
 * and exact: the id is never one this function COMPOSED out of levels it knows
 * are inconsistent, so the result is no worse than the pre-#1439 behaviour. It
 * also never throws — this feeds hover.
 */
function buildLadderElementIdMap(
  parts: LoadedPointsData[],
  payloadLevelStarts: readonly number[],
  logicalDepth: number,
  levelOffsets: readonly number[],
  totalPoints: number,
  warn: (message: string) => void
): Uint32Array | undefined {
  if (payloadLevelStarts.length !== parts.length || levelOffsets.length <= logicalDepth) {
    warn(
      `Progressive Points: inconsistent retained-payload lineage for ${parts.length} payloads ` +
        `at logical depth ${logicalDepth} — picking labels fall back to the visible-buffer slot.`
    );
    return undefined;
  }
  let anyMap = false;
  let identity = true;
  let running = 0;
  for (const [i, part] of parts.entries()) {
    const startLevel = payloadLevelStarts[i];
    const endLevel = payloadLevelStarts[i + 1] ?? logicalDepth;
    if (
      !Number.isInteger(startLevel) ||
      !Number.isInteger(endLevel) ||
      startLevel < 0 ||
      endLevel <= startLevel ||
      endLevel > logicalDepth
    ) {
      warn(
        `Progressive Points: retained payload ${i} has invalid logical level range ` +
          `[${startLevel}, ${endLevel}) — picking labels fall back to the visible-buffer slot.`
      );
      return undefined;
    }
    const ids = part.elementIds;
    // An EMPTY level writes nothing into the union map, so its missing map
    // cannot corrupt a single slot — it must not veto the other levels' (a
    // ladder level culled to zero by the current slice is ordinary).
    if (part.pointCount > 0 && part.elementIdsUnavailable === true) {
      warn(
        `Progressive Points: payload ${i} could not build a slot → on-disk map (its slots are ` +
          'not on-disk indices) — picking labels fall back to the visible-buffer slot.'
      );
      return undefined;
    }
    if (ids !== undefined) {
      anyMap = true;
      if (ids.length !== part.pointCount) {
        warn(
          `Progressive Points: payload ${i} published ${ids.length} element ids for ` +
            `${part.pointCount} points — picking labels fall back to the visible-buffer slot.`
        );
        return undefined;
      }
    }
    // Identity only survives while every level is fully resident and unculled:
    // its on-disk offset must equal the running sum of the LOADED counts.
    if (levelOffsets[startLevel] !== running) identity = false;
    running += part.pointCount;
  }
  if (!anyMap && identity) return undefined;

  const out = new Uint32Array(totalPoints);
  let w = 0;
  for (const [i, part] of parts.entries()) {
    const startLevel = payloadLevelStarts[i];
    const endLevel = payloadLevelStarts[i + 1] ?? logicalDepth;
    const base = levelOffsets[startLevel];
    const span = levelOffsets[endLevel] - base;
    const ids = part.elementIds;
    for (let k = 0; k < part.pointCount; k++) {
      const local = ids === undefined ? k : ids[k];
      // Negated compare so a NaN / undefined slot also fails closed.
      if (!(local < span)) {
        warn(
          `Progressive Points: payload ${i} maps a slot to index ${local}, past its ${span} ` +
            'on-disk rows — picking labels fall back to the visible-buffer slot.'
        );
        return undefined;
      }
      out[w++] = base + local;
    }
  }
  return out;
}

/**
 * Concatenate per-LOD `LoadedPointsData` into one. Optional attribute
 * arrays (`colors`, `radii`, `sharpness`, `scalars`) are concatenated
 * only when ALL levels carry them (mixed-presence is dropped — keeps
 * the loader simple and matches the writer's all-or-nothing per-attr
 * policy). A level the current slice culled to ZERO points abstains from
 * that vote rather than vetoing it — see `concatOptionalField` (#1456).
 *
 * `levelOffsets` (non-null only for a ladder whose PARENT declares at least one
 * union string/image CSR) gives each level's half-open extent inside that CSR's
 * index space — CSR-style, so it holds one entry MORE than there are levels;
 * with it the per-level picking maps are composed into one union map, and
 * without it none is published at all — see {@link buildLadderElementIdMap}.
 * `warn` is the owning loader's once-per-instance fail-closed reporter (this
 * runs on every memoized concat, so an unlatched `log.warning` would spam once
 * per level per view change — tens per second under dimension-animation
 * playback).
 */
function concatenatePointsData(
  parts: LoadedPointsData[],
  payloadLevelStarts: readonly number[],
  logicalDepth: number,
  levelOffsets: readonly number[] | null,
  warn: (message: string) => void = () => {}
): LoadedPointsData {
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
    // A single retained payload is either `additive_0` at first paint or a
    // folded cumulative prefix. Both begin at logical level 0, and their map is
    // already in the parent's union space; only the closing bound differs.
    //
    // The ladder's labels live in ONE CSR on the PARENT node (#1422), whose
    // index space is the concatenation of the levels, so a correct ladder map is
    // a per-level map offset by the preceding levels' on-disk counts (#1439).
    // WITH `levelOffsets`, the payload's on-disk range IS the union CSR's
    // prefix, so its index space already is the parent's and it passes through
    // unchanged after validating against `levelOffsets[logicalDepth]`.
    //
    // WITHOUT `levelOffsets` (no parent CSR, or a failed cross-check) there is
    // no union index space any reader could key by, so the GUARD applies
    // (belt-and-braces): that payload must never publish a slot → on-disk map.
    // Not because `parts[0]`'s map is in the wrong space — it
    // is not, it is the union's prefix, so passing it through would in fact
    // resolve correctly while it is the only committed level. It is that a
    // tooltip which is right at first paint and silently degrades to the raw slot
    // the moment a second level lands is worse than one consistently at the raw
    // slot, which is what every doc surface promises.
    // `createProgressivePointsLoader` also clears `has_labels` /
    // `has_image_labels` / `has_keys` on each sub-LOD's attrs whenever the
    // composition cannot run, so the map is normally never built at all in that
    // case; this keeps the invariant true whatever attrs a sub-LOD carries.
    const only = parts[0];
    const offsetOk =
      payloadLevelStarts.length === 1 &&
      payloadLevelStarts[0] === 0 &&
      logicalDepth >= 1 &&
      levelOffsets !== null &&
      levelOffsets.length > logicalDepth &&
      levelOffsets[0] === 0;
    if (offsetOk) {
      // The retained prefix owns union rows [0, levelOffsets[logicalDepth]).
      // The same three fail-closed checks
      // `buildLadderElementIdMap` applies, on the one level there is:
      // a level that WANTED a map and failed to build one is NOT the identity
      // (its slots are not on-disk indices); a length mismatch means the map
      // does not describe this payload; and an id past level 0's rows names a
      // real row outside the retained prefix in the union CSR. An EMPTY payload has
      // no slots to be wrong about, so it is exempt.
      const span = levelOffsets[logicalDepth];
      let failure: string | null = null;
      if (only.pointCount > 0 && only.elementIdsUnavailable === true) {
        failure =
          'retained prefix could not build a slot → on-disk map (its slots are not on-disk indices)';
      } else if (only.elementIds !== undefined && only.elementIds.length !== only.pointCount) {
        failure = `retained prefix published ${only.elementIds.length} element ids for ${only.pointCount} points`;
      } else if (only.elementIds?.some((id) => !(id < span))) {
        failure = `retained prefix maps a slot past its ${span} on-disk rows`;
      }
      if (failure === null) return only;
      warn(`Progressive Points: ${failure} — picking labels fall back to the visible-buffer slot.`);
    }
    if (only.elementIds === undefined) return only;
    const stripped: LoadedPointsData = { ...only };
    delete stripped.elementIds;
    return stripped;
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

  // Slot → on-disk map: composed into the parent's union CSR index space when
  // the parent declares one, otherwise not published at all (see the
  // single-part branch and `buildLadderElementIdMap`).
  if (levelOffsets !== null) {
    const elementIds = buildLadderElementIdMap(
      parts,
      payloadLevelStarts,
      logicalDepth,
      levelOffsets,
      totalPoints,
      warn
    );
    if (elementIds) result.elementIds = elementIds;
  }

  return result;
}

/**
 * Progressive Points loader.
 */
export class PointsProgressiveLoader implements PointsDataLoader {
  private lodLoaders: PointsSpatialIndexLoader[];
  private loadedLODs: LoadedPointsData[] = [];
  // Payload i spans logical levels [_payloadLevelStarts[i],
  // _payloadLevelStarts[i + 1] ?? _loadedLODCount); folding collapses this to [0].
  private _payloadLevelStarts: number[] = [];
  private _loadedLODCount = 0;
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
  // Logical ladder depth and retained payload count as the CURRENT updateView
  // pass found them. The pass advances both before its caller has committed
  // the result, while concatenation may later fold many logical rungs into one
  // retained payload. Rollback needs both watermarks to distinguish an intact
  // append from an already-folded result.
  // See `../loaders/progressive/pass-rollback`.
  private _levelsAtPassStart = 0;
  private _payloadsAtPassStart = 0;
  private _restoredFullLadderAtPassStart = false;
  // A completed pass can still fail after loading, during projection or commit.
  // Keep it schedulable for one retry even though the ladder cursor is full.
  private _retryFoldedPass = false;
  // Per-sub-LOD cumulative energy fractions e(k) (the build-time
  // `lod_stats.energy_fraction_cum` stamps), normalized at construction:
  // non-null only when EVERY sub-LOD carries a stamp (a partially stamped
  // ladder reads as unstamped — never blend stamped and guessed entries).
  private energyTable: readonly number[] | null;
  // CSR-style bounds over the levels' ON-DISK element counts (`nLods + 1`
  // entries): level `i` occupies `[levelOffsets[i], levelOffsets[i + 1])`
  // inside the parent node's union string/image CSR index space. Non-null only
  // when the parent declares at least one such CSR
  // (`createProgressivePointsLoader`); null means no ladder picking map is
  // published at all.
  private readonly levelOffsets: readonly number[] | null;
  // One fail-closed composition warning per LOADER, not per concat: the concat
  // re-runs on every (generation, lodCount) miss and a view change bumps the
  // generation, so a malformed ladder would otherwise log tens of lines a
  // second during dimension-animation playback. Same latch pattern as
  // `slice-cache-helper.ts::markOversizedWarned` / the label-loader demotion.
  private _composeWarned = false;
  private readonly warnComposeFailure = (message: string): void => {
    if (this._composeWarned) return;
    this._composeWarned = true;
    log.warning(Modules.SPATIAL_INDEX_LOADER, `${message} (logged once per node)`);
  };
  // Node path (SliceCache namespace) + the shared SliceCache, if enabled.
  private readonly path: string;
  private readonly sliceCache: SliceCache | null;
  // Per-tick LOD time budget (ms) from the CURRENT updateView call during
  // dimension-animation playback; null outside playback. A per-pass
  // directive (never part of lastViewState / viewStatesEqual / cache keys).
  // Mirrors GSplatsProgressiveLoader.
  private _frameBudgetMs: number | null = null;
  /**
   * Pinned rung count for the current pass (`ViewState.ladderDepth`, the
   * playback "detail" setting), resolved through `resolveLadderDepth`; null
   * when the pass is not pinned. A pinned pass loads exactly this many rungs,
   * cold or not, and reports `hasMoreLODs === false` like a budgeted one — the
   * pinned prefix IS the target.
   */
  private _ladderDepth: number | null = null;

  constructor(
    lodLoaders: PointsSpatialIndexLoader[],
    nLods: number,
    path: string,
    energyTable?: ReadonlyArray<number | null | undefined>,
    sliceCache?: SliceCache | null,
    levelOffsets?: readonly number[] | null
  ) {
    this.lodLoaders = lodLoaders;
    this.nLods = nLods;
    this.path = path;
    this.sliceCache = sliceCache ?? null;
    this.levelOffsets = levelOffsets ?? null;
    this.monitor = new ProgressiveMonitorAdapter(
      () => this.lodLoaders,
      path,
      'point-spatial-index'
    );
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
    if (this._retryFoldedPass) return true;
    // While a playback frame budget is active, the budgeted prefix IS the
    // target: no background refinement between animation ticks; the commit
    // stamps the prefix complete. Mirrors GSplatsProgressiveLoader.
    if (this._frameBudgetMs !== null) return false;
    if (this._ladderDepth !== null) return false;
    return this._loadedLODCount < this.nLods;
  }

  get loadedLODCount(): number {
    return this._loadedLODCount;
  }

  /**
   * Measured footprint of the loaded ladder, for the shared sweep residency
   * budget (`scene-loader/progressive/residency-budget`). Sums real
   * `byteLength`s rather than modelling a per-element cost. Rung count comes
   * from `_loadedLODCount` (LOGICAL levels), not `loadedLODs.length`, which is
   * 1 once the ladder has folded.
   */
  ladderResidency(): LadderResidency {
    return {
      residentBytes: measureLodBytes(this.loadedLODs),
      loadedRungs: this._loadedLODCount,
      elementCount: this.loadedLODs.reduce((s, d) => s + d.pointCount, 0),
      // 3 RGBA32F texels/point. See LadderResidency — the payload alone is not
      // the node's footprint, and the ratio differs per geometry.
      bytesPerElement: POINT_FLOATS_PER_POINT * Float32Array.BYTES_PER_ELEMENT,
    };
  }

  get totalLODCount(): number {
    return this.nLods;
  }

  /**
   * Discard the levels the current pass appended, restoring the ladder to the
   * prefix the pass started from. Called by the main-update and refinement catches;
   * see `../loaders/progressive/pass-rollback` for why a failed commit must
   * not leave the cursor advanced.
   *
   * @returns Levels discarded (0 when the pass appended none).
   */
  rollbackToPassStart(): number {
    const plan = planLadderRollback({
      loadedLevelCount: this._loadedLODCount,
      levelsAtPassStart: this._levelsAtPassStart,
      concatCacheLodCount: this._concatCache?.lodCount ?? null,
      retainedPayloadCount: this.loadedLODs.length,
      payloadsAtPassStart: this._payloadsAtPassStart,
      restoredFullLadderAtPassStart: this._restoredFullLadderAtPassStart,
      totalLevelCount: this.nLods,
    });
    if (plan.action === 'none') return 0;
    if (plan.action === 'retry-folded-pass') {
      this._retryFoldedPass = true;
      return 0;
    }
    if (plan.action === 'unwind-restored-full') {
      this.loadedLODs = [];
      this._payloadLevelStarts = [];
      this._loadedLODCount = 0;
      this._restoredFullLadderAtPassStart = false;
      this._retryFoldedPass = false;
      if (this.lastViewState) deleteLadder(this.sliceCache, this.path, this.lastViewState);
      this._concatCache = null;
      return plan.dropped;
    }
    this.loadedLODs.length = this._payloadsAtPassStart;
    this._payloadLevelStarts.length = this._payloadsAtPassStart;
    this._loadedLODCount = plan.keep;
    this._retryFoldedPass = false;
    if (this.lastViewState) deleteLadder(this.sliceCache, this.path, this.lastViewState);
    if (plan.invalidateConcatCache) this._concatCache = null;
    return plan.dropped;
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
    const k = this._loadedLODCount;
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
    signal?: AbortSignal,
    residencyAllowanceBytes?: number
  ): Promise<LoadedPointsData> {
    // Record the per-pass playback budget FIRST (before the restore branch:
    // a pause re-trigger arrives with the SAME view state — it must still
    // clear the budget). Mirrors GSplatsProgressiveLoader.
    this._frameBudgetMs = viewState.frameBudgetMs ?? null;
    this._ladderDepth = resolveLadderDepth(
      viewState.ladderDepth,
      this.nLods,
      this.energyTable,
      config.dimensionAnimation.playback.autoEnergyThreshold
    );
    this._retryFoldedPass = false;
    const budgetDeadline =
      this._frameBudgetMs !== null ? performance.now() + this._frameBudgetMs : null;

    // Background prefetch (shadow) passes only warm the SliceCache — the
    // SlicePrefetcher discards the return value — so hand back a cheap empty
    // result instead of the O(N) main-thread concat, which would stall
    // foreground frames as the ladder deepens. Mirrors GSplatsProgressiveLoader.
    const isPrefetch = viewState.prefetch === true;
    const finish = (): LoadedPointsData =>
      isPrefetch ? concatenatePointsData([], [], 0, null) : this.concatenateMemoized(session);

    if (!this.lastViewState || !viewStatesEqual(viewState, this.lastViewState)) {
      // DEPARTURE store: snapshot the OUTGOING view's partial ladder before
      // discarding it, keyed under the OUTGOING view (lastViewState — never
      // the incoming one). Scrubbing faster than the ladder completes would
      // otherwise store nothing at all (completion never happens), making
      // scrub-back — the S-cache's headline case — always cold. One clone
      // per slice-leave; upgrade-if-longer makes re-departures cheap no-ops.
      if (this.lastViewState && this._loadedLODCount > 0) {
        storeLadder(this.sliceCache, this.path, this.lastViewState, this.loadedLODs, {
          scan: this._frameBudgetMs !== null || this._ladderDepth !== null,
          pin: viewState.prefetch === true,
          ladderDepth: this._loadedLODCount,
          totalLODCount: this.nLods,
        });
      }
      // Try the SliceCache before discarding the ladder (see GSplats loader).
      const restored = restoreLadderSnapshot<LoadedPointsData>(
        this.sliceCache,
        this.path,
        viewState,
        this.nLods
      );
      // Shallow-copy the CONTAINER: the streaming loop below pushes further
      // levels and must never mutate the cache's payload array (elements
      // stay shared read-only). Mirrors GSplatsProgressiveLoader.
      this.loadedLODs = restored ? [...restored.lods] : [];
      this._loadedLODCount = restored?.depth ?? 0;
      if (restored) {
        const foldedPrefixDepth = restored.depth - restored.lods.length + 1;
        this._payloadLevelStarts = restored.lods.map((_, index) =>
          index === 0 ? 0 : foldedPrefixDepth + index - 1
        );
      } else {
        this._payloadLevelStarts = [];
      }
      // A restored full ladder has not been committed for this pass. If its
      // concat or commit fails, unwind the whole restored snapshot rather than
      // preserving a cursor at nLods and silently disabling retries.
      this._levelsAtPassStart = 0;
      this._payloadsAtPassStart = 0;
      this._restoredFullLadderAtPassStart = false;
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
        if (restored.depth === this.nLods) {
          this._restoredFullLadderAtPassStart = true;
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

    // Stream under the shared streaming policy (see `streaming-policy.ts`):
    // `playback` streams cache-resident levels after an empty or restored
    // prefix while budget remains; `prefetch` deepens toward the
    // full decoded ladder (abort-safe, stored per level); `refine` stops at the
    // first cold/slow level. Mirrors GSplatsProgressiveLoader.
    const pass = classifyStreamingPass(
      budgetDeadline !== null,
      isPrefetch,
      this._ladderDepth !== null
    );
    const startLevel = this._loadedLODCount;
    const residentBytesAtPassStart = ladderResidentBytes(this.ladderResidency());
    this._levelsAtPassStart = startLevel;
    this._payloadsAtPassStart = this.loadedLODs.length;
    this._restoredFullLadderAtPassStart = false;

    // A pinned pass bounds the loop at the pinned rung count; the policy's
    // `pinned` kind disables the deadline / cold-level brakes.
    const targetLevels = this._ladderDepth ?? this.nLods;
    for (let level = startLevel; level < targetLevels; level++) {
      // A dispose() racing the awaited level below clears `lodLoaders`, so
      // the next iteration would TypeError on `this.lodLoaders[level]` — a
      // teardown mis-counted as a real refinement failure (recordFailure +
      // backoff). Stop streaming instead.
      if (this._disposed) {
        break;
      }
      // Pass-budget guard: playback only guarantees a level for an empty
      // ladder; prefetch retains one-level progress after a restore (#2379).
      if (shouldStopBeforeLevel(pass, level, startLevel, performance.now(), budgetDeadline)) {
        break;
      }
      const t0 = performance.now();
      const { data: lodData, allResident } = await timeLodStageWithResult(
        ({ allResident }) => `additive:points:level:${level}:${allResident ? 'resident' : 'miss'}`,
        `additive:points:level:${level}:aborted`,
        () => this.lodLoaders[level].updateViewWithResidency(viewState, session, signal)
      );
      const elapsed = performance.now() - t0;

      this.loadedLODs.push(lodData);
      this._payloadLevelStarts.push(level);
      this._loadedLODCount++;
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

      // NEVER break out because a level came back empty (#1456). It is a
      // tempting optimization — this loop used to latch a terminal "empty
      // ladder" on an empty LOD 0 and stop — but it is wrong here: this loader
      // is constructed only for ADDITIVE ladders (`createProgressivePointsLoader`
      // iterates the `additive_<i>` subgroups), whose levels are DISJOINT
      // increments of one permutation, not coarse-to-fine resamplings of the
      // same elements. They are therefore NOT spatially coextensive: LOD 0 is a
      // small SUBSET of the node (a few thousand points under `-b stream:C` /
      // `--target-ms`, the recommended ladder shape), so a hidden-dimension
      // slice that none of ITS members lands on says nothing whatever about
      // levels 1..n-1, which may hold plenty of points right there. Stopping
      // here rendered such a slice permanently blank. The same reasoning
      // forbids inferring anything from a restored cache PREFIX whose LOD 0 is
      // empty.

      const additionalResidentBytes = Math.max(
        0,
        ladderResidentBytes(this.ladderResidency()) - residentBytesAtPassStart
      );
      if (
        shouldStopAfterLevel(
          pass,
          level,
          startLevel,
          allResident,
          elapsed,
          additionalResidentBytes,
          residencyAllowanceBytes
        )
      ) {
        break;
      }
    }

    if (!this._initialLoadDone && startLevel === 0) {
      this._initialLoadDone = true;
    }

    const totalPoints = this.loadedLODs.reduce((s, d) => s + d.positions.length / 3, 0);
    if (this._loadedLODCount < this.nLods) {
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Progressive Points: ${this._loadedLODCount}/${this.nLods} LODs (${totalPoints} points) — refining`
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
    const result = finish();
    if (
      this._loadedLODCount === this.nLods ||
      this._frameBudgetMs !== null ||
      this._ladderDepth !== null
    ) {
      storeLadder(this.sliceCache, this.path, viewState, this.loadedLODs, {
        scan: this._frameBudgetMs !== null || this._ladderDepth !== null,
        pin: viewState.prefetch === true,
        ladderDepth: this._loadedLODCount,
        totalLODCount: this.nLods,
      });
    }

    return result;
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
   */
  private concatenateMemoized(session?: UpdateSession): LoadedPointsData {
    const concatSession = session?.begin('Concatenate LODs');
    try {
      if (
        this._concatCache &&
        this._concatCache.generation === this._resetGeneration &&
        this._concatCache.lodCount === this._loadedLODCount
      ) {
        return this._concatCache.result;
      }
      // Capture the previous SAME-GENERATION memo before overwriting the
      // cache — that (and only that) is the result this one extends.
      const prevMemo =
        this._concatCache &&
        this._concatCache.generation === this._resetGeneration &&
        this._concatCache.lodCount < this._loadedLODCount
          ? this._concatCache.result
          : null;
      const result = concatenatePointsData(
        this.loadedLODs,
        this._payloadLevelStarts,
        this._loadedLODCount,
        this.levelOffsets,
        this.warnComposeFailure
      );
      setPrefixParent(result, prevMemo);
      for (let level = 0; level < this._loadedLODCount; level++) {
        this.lodLoaders[level]?.releaseAccumulator();
      }
      this.loadedLODs = [result];
      this._payloadLevelStarts = [0];
      this._concatCache = {
        generation: this._resetGeneration,
        lodCount: this._loadedLODCount,
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
    const nextLevel = this._loadedLODCount;
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
    const metrics = this.monitor.getMetrics();
    const concatMemory = this._concatCache ? measureLodBytes([this._concatCache.result]) : 0;
    return { ...metrics, memoryUsed: metrics.memoryUsed + concatMemory };
  }

  dispose(): void {
    this._disposed = true;
    for (const loader of this.lodLoaders) {
      loader.dispose();
    }
    this.lodLoaders = [];
    this.loadedLODs = [];
    this._payloadLevelStarts = [];
    this._loadedLODCount = 0;
    this._retryFoldedPass = false;
    this.lastViewState = null;
    this._concatCache = null;
  }
}
