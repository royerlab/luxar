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
 *   (15 ms — shared `CACHE_HIT_THRESHOLD_MS` with gsplats for cross-
 *   type symmetry).
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
import { log, Modules, LogEmoji } from '../../utils/log';

/** Shared cache-hit threshold; matches `gsplats-progressive-loader`. */
const CACHE_HIT_THRESHOLD_MS = 15;

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

  const positions = new Float32Array(totalPoints * ndim);
  let offset = 0;
  for (const part of parts) {
    positions.set(part.positions, offset * ndim);
    offset += part.pointCount;
  }

  const allHaveColors = parts.every((p) => p.colors !== undefined);
  const allHaveRadii = parts.every((p) => p.radii !== undefined);
  const allHaveSharpness = parts.every((p) => p.sharpness !== undefined);
  const allHaveScalars = parts.every((p) => p.scalars !== undefined);

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

  if (allHaveColors) {
    const first = parts[0].colors as ColorArray;
    const ctor = first.constructor as new (n: number) => ColorArray;
    const colors = new ctor(totalPoints * 3);
    let o = 0;
    for (const part of parts) {
      colors.set(part.colors!, o * 3);
      o += part.pointCount;
    }
    result.colors = colors;
  }
  if (allHaveRadii) {
    const first = parts[0].radii as ScalarArray;
    const ctor = first.constructor as new (n: number) => ScalarArray;
    const radii = new ctor(totalPoints);
    let o = 0;
    for (const part of parts) {
      radii.set(part.radii!, o);
      o += part.pointCount;
    }
    result.radii = radii;
  }
  if (allHaveSharpness) {
    const first = parts[0].sharpness as ScalarArray;
    const ctor = first.constructor as new (n: number) => ScalarArray;
    const sharpness = new ctor(totalPoints);
    let o = 0;
    for (const part of parts) {
      sharpness.set(part.sharpness!, o);
      o += part.pointCount;
    }
    result.sharpness = sharpness;
  }
  if (allHaveScalars) {
    const first = parts[0].scalars as ScalarArray;
    const ctor = first.constructor as new (n: number) => ScalarArray;
    const scalars = new ctor(totalPoints);
    let o = 0;
    for (const part of parts) {
      scalars.set(part.scalars!, o);
      o += part.pointCount;
    }
    result.scalars = scalars;
  }

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
  private _initialLoadDone = false;

  constructor(lodLoaders: PointsSpatialIndexLoader[], nLods: number) {
    this.lodLoaders = lodLoaders;
    this.nLods = nLods;
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

  async loadPoints(
    viewState: PointsViewState,
    session?: UpdateSession
  ): Promise<LoadedPointsData> {
    return this.updateView(viewState, session);
  }

  async updateView(
    viewState: PointsViewState,
    session?: UpdateSession
  ): Promise<LoadedPointsData> {
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
      const lodData = await this.lodLoaders[level].updateView(viewState, session);
      const elapsed = performance.now() - t0;

      this.loadedLODs.push(lodData);

      if (!this._initialLoadDone) {
        log.custom(
          LogEmoji.BROADCAST,
          Modules.SPATIAL_INDEX_LOADER,
          `LOD ${level}/${this.nLods - 1}: ${
            lodData.positions.length / 3
          } points (${elapsed.toFixed(1)}ms)`
        );
      }

      // Short-circuit: LOD 0 with 0 points → higher LODs would also
      // have 0 visible (LODs are spatially coextensive).
      if (level === 0 && lodData.positions.length === 0) {
        break;
      }

      if (level > startLevel && elapsed > CACHE_HIT_THRESHOLD_MS) {
        break;
      }
    }

    if (!this._initialLoadDone && startLevel === 0) {
      this._initialLoadDone = true;
    }

    const totalPoints = this.loadedLODs.reduce(
      (s, d) => s + d.positions.length / 3,
      0
    );
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

  dispose(): void {
    for (const loader of this.lodLoaders) {
      loader.dispose();
    }
    this.lodLoaders = [];
    this.loadedLODs = [];
    this.lastViewState = null;
  }
}
