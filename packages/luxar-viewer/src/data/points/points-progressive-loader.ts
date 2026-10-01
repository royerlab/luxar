/**
 * Progressive Points loader for multi-additive-LOD datasets.
 *
 * Wraps N `PointsSpatialIndexLoader` instances (one per `additive_<i>`
 * subgroup) using the Composite pattern. Implements the same
 * `PointsDataLoader` (aka `DataLoader`) interface so the scene
 * loader's update loop works unchanged.
 *
 * The per-view lifecycle (streaming policy under the shared
 * `CACHE_HIT_THRESHOLD_MS`, SliceCache departure/adoption/restore, B6 index
 * warm, pinned concurrent rungs, signal-linked lookahead of the next rung, B5
 * neighbour read-ahead, memoized concat) is the shared `AdditiveLadderCore`
 * (`../loaders/progressive/additive-ladder-core`), as for
 * `GSplatsProgressiveLoader` / `LinesProgressiveLoader`.
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
import {
  concatColorsWhiteFilled,
  concatOptionalField,
  concatRequiredField,
  type ConcatColorArray,
} from '../loaders/progressive/concat-helpers';
import {
  AdditiveLadderCore,
  type LadderGeometry,
} from '../loaders/progressive/additive-ladder-core';
import { POINT_FLOATS_PER_POINT } from '../../rendering/element-texture-layout';
import type { LadderResidency } from '../scene-loader/progressive/residency-budget';
import type { SliceCache } from '../../cache/slice-cache';
import { log, Modules } from '../../utils/log';

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

  // Colours follow the shared ladder policy (`concatColorsWhiteFilled`): a
  // rung without them is filled with white rather than dropping colours from
  // the whole ladder. The other optional per-point fields stay all-or-nothing
  // across LODs (dtype preserved).
  const colored = concatColorsWhiteFilled(
    parts,
    (p) => ({ colors: p.colors as ConcatColorArray | undefined, components: p.colorComponents }),
    count,
    'concatenatePointsData'
  );
  if (colored) {
    result.colors = colored.colors as ColorArray;
    result.colorComponents = colored.colorComponents;
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

const POINTS_LADDER: LadderGeometry<LoadedPointsData> = {
  kind: 'points',
  module: Modules.SPATIAL_INDEX_LOADER,
  label: 'Points',
  unit: 'points',
  warnPrefix: 'Points',
  countOf: (data) => data.pointCount,
  concat: (parts, layout) =>
    concatenatePointsData(parts, layout.payloadLevelStarts, layout.loadedLevels, null),
  empty: () => concatenatePointsData([], [], 0, null),
  // 3 RGBA32F texels/point. See LadderResidency — the payload alone is not
  // the node's footprint, and the ratio differs per geometry.
  bytesPerElement: POINT_FLOATS_PER_POINT * Float32Array.BYTES_PER_ELEMENT,
  // A Points result is never stamped with a slice-cache origin.
  stampsRestoredOrigin: false,
};

/**
 * Progressive Points loader: the Points face of the shared additive-ladder
 * engine (`AdditiveLadderCore`), which owns the whole per-view lifecycle. This
 * class adds the Points concatenation, including the ladder's composed
 * slot -> on-disk picking map over the parent's union CSR (`levelOffsets`).
 */
export class PointsProgressiveLoader implements PointsDataLoader {
  private readonly core: AdditiveLadderCore<LoadedPointsData, PointsSpatialIndexLoader>;
  private readonly monitor: ProgressiveMonitorAdapter;

  // One fail-closed composition warning per LOADER, not per concat: the concat
  // re-runs on every (generation, rung count) miss and a view change bumps the
  // generation, so a malformed ladder would otherwise log tens of lines a
  // second during dimension-animation playback.
  private composeWarned = false;

  constructor(
    lodLoaders: PointsSpatialIndexLoader[],
    nLods: number,
    path: string,
    energyTable?: ReadonlyArray<number | null | undefined>,
    sliceCache?: SliceCache | null,
    levelOffsets?: readonly number[] | null
  ) {
    // CSR-style bounds over the levels' ON-DISK element counts (`nLods + 1`
    // entries): level `i` occupies `[levelOffsets[i], levelOffsets[i + 1])`
    // inside the parent node's union string/image CSR index space. Null means
    // no ladder picking map is published at all.
    const offsets = levelOffsets ?? null;
    const warn = (message: string): void => {
      if (this.composeWarned) return;
      this.composeWarned = true;
      log.warning(Modules.SPATIAL_INDEX_LOADER, `${message} (logged once per node)`);
    };
    const geometry: LadderGeometry<LoadedPointsData> = {
      ...POINTS_LADDER,
      concat: (parts, layout) =>
        concatenatePointsData(parts, layout.payloadLevelStarts, layout.loadedLevels, offsets, warn),
    };
    this.core = new AdditiveLadderCore(lodLoaders, nLods, path, geometry, {
      energyTable,
      sliceCache,
    });
    this.monitor = new ProgressiveMonitorAdapter(
      () => this.core.rungLoaders,
      path,
      'point-spatial-index'
    );
  }

  /** Whether there are more LOD levels to load for the current view state. */
  get hasMoreLODs(): boolean {
    return this.core.hasMoreLODs;
  }

  /** Number of LOD levels currently loaded. */
  get loadedLODCount(): number {
    return this.core.loadedLODCount;
  }

  /** Measured footprint of the loaded ladder (see `AdditiveLadderCore`). */
  ladderResidency(): LadderResidency {
    return this.core.ladderResidency();
  }

  /** Cumulative energy fraction e(k) of the loaded prefix (see the core). */
  get committedEnergyFraction(): number | null {
    return this.core.committedEnergyFraction;
  }

  /** Total number of LOD levels. */
  get totalLODCount(): number {
    return this.core.totalLODCount;
  }

  /** Discard the levels the current pass appended (see `pass-rollback.ts`). */
  rollbackToPassStart(): number {
    return this.core.rollbackToPassStart();
  }

  /** Whether the most recently streamed LOD level was fully cache-resident. */
  get lastAllResident(): boolean {
    return this.core.lastAllResident;
  }

  /** Load points data (delegates to {@link updateView}). */
  async loadPoints(
    viewState: PointsViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedPointsData> {
    return this.updateView(viewState, session, signal);
  }

  /** Stream as many LODs as the pass allows for `viewState` (see the core). */
  updateView(
    viewState: PointsViewState,
    session?: UpdateSession,
    signal?: AbortSignal,
    residencyAllowanceBytes?: number
  ): Promise<LoadedPointsData> {
    return this.core.updateView(viewState, session, signal, residencyAllowanceBytes);
  }

  /** Predicted-view warm-up of the coarse rung (`dispatchPredictivePrefetch`). */
  prefetchChunks(viewState: PointsViewState, signal?: AbortSignal): Promise<void> {
    return this.core.prefetchChunks(viewState, signal);
  }

  /** Predicted-view warm-up over a current → predicted transition. */
  prefetchChunkBoundary(
    current: PointsViewState,
    predicted: PointsViewState,
    signal?: AbortSignal
  ): Promise<void> {
    return this.core.prefetchChunkBoundary(current, predicted, signal);
  }

  // ---- LoaderMonitor surface (delegated to ProgressiveMonitorAdapter) ----

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
    return { ...metrics, memoryUsed: metrics.memoryUsed + this.core.concatMemoryBytes() };
  }

  /** Clean up all LOD loaders. */
  dispose(): void {
    this.core.dispose();
  }
}
