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
import { log, Modules, LogEmoji } from '../../utils/log';

/** Element-wise viewstate equality (query-affecting fields only). */
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

  const positions = concatRequiredField(parts, (p) => p.positions, count, ndim);

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

  constructor(lodLoaders: PointsSpatialIndexLoader[], nLods: number, path: string) {
    this.lodLoaders = lodLoaders;
    this.nLods = nLods;
    this.monitor = new ProgressiveMonitorAdapter(() => this.lodLoaders, path);
  }

  /** Whether more LOD levels remain to load for the current view state. */
  get hasMoreLODs(): boolean {
    return this.loadedLODs.length < this.nLods;
  }

  get loadedLODCount(): number {
    return this.loadedLODs.length;
  }

  get totalLODCount(): number {
    return this.nLods;
  }

  /**
   * Whether the most recently streamed LOD level was fully cache-resident.
   * Drives the monitor's residency indicator. Defaults to `true`.
   */
  get lastAllResident(): boolean {
    return this._lastAllResident;
  }

  async loadPoints(viewState: PointsViewState, session?: UpdateSession): Promise<LoadedPointsData> {
    return this.updateView(viewState, session);
  }

  async updateView(viewState: PointsViewState, session?: UpdateSession): Promise<LoadedPointsData> {
    if (!this.lastViewState || !viewStatesEqual(viewState, this.lastViewState)) {
      this.loadedLODs = [];
      this.lastViewState = {
        displayDims: [...viewState.displayDims],
        slicePosition: [...viewState.slicePosition],
        tolerance: [...viewState.tolerance],
        dimensions: viewState.dimensions,
      };
    }

    const startLevel = this.loadedLODs.length;

    for (let level = startLevel; level < this.nLods; level++) {
      const t0 = performance.now();
      const { data: lodData, allResident } = await this.lodLoaders[level].updateViewWithResidency(
        viewState,
        session
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

    return concatenatePointsData(this.loadedLODs);
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
    for (const loader of this.lodLoaders) {
      loader.dispose();
    }
    this.lodLoaders = [];
    this.loadedLODs = [];
    this.lastViewState = null;
  }
}
