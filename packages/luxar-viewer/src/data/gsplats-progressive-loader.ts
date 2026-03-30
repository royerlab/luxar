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

import type {
  GSplatsDataLoader,
  GSplatsViewState,
  LoadedGSplatsData,
} from '../types/gsplats';
import type { GSplatsSpatialIndexLoader } from './gsplats-spatial-index-loader';
import type { UpdateSession } from '../profiling/update-profiler';
import { log, Modules, LogEmoji } from '../utils/log';

/**
 * Time threshold (ms) for considering a LOD load "fast" (likely a cache hit).
 * If a LOD loads faster than this, the loader continues to the next LOD.
 * If slower, it stops and lets the refinement loop pick up the rest.
 */
const CACHE_HIT_THRESHOLD_MS = 15;

/**
 * Compare two GSplatsViewState objects for query-affecting equality.
 * Compares displayDims, slicePosition, and tolerance element-wise.
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

  const positions = new Float32Array(totalSplats * ndim);
  const amplitudes = new Float32Array(totalSplats);
  const choleskyFactors = new Float32Array(totalSplats * cholSize);

  // Determine color type from first part that has colors
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
    positions.set(part.positions, offset * ndim);
    amplitudes.set(part.amplitudes, offset);
    choleskyFactors.set(part.choleskyFactors, offset * cholSize);

    if (colors && part.colors) {
      colors.set(part.colors, offset * 3);
    } else if (colors && !part.colors) {
      // Fill with white (1.0 for Float32, 255 for Uint8, 65535 for Uint16)
      const fillValue = colors instanceof Uint8Array ? 255 : colors instanceof Uint16Array ? 65535 : 1.0;
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
  private _initialLoadDone = false;

  constructor(lodLoaders: GSplatsSpatialIndexLoader[], nLods: number) {
    this.lodLoaders = lodLoaders;
    this.nLods = nLods;
  }

  /**
   * Whether there are more LOD levels to load for the current view state.
   */
  get hasMoreLODs(): boolean {
    return this.loadedLODs.length < this.nLods;
  }

  /** Number of LOD levels currently loaded. */
  get loadedLODCount(): number {
    return this.loadedLODs.length;
  }

  /** Total number of LOD levels. */
  get totalLODCount(): number {
    return this.nLods;
  }

  /**
   * Load gsplats data. On first call, loads LOD 0.
   * Delegates to updateView().
   */
  async loadGSplats(
    viewState: GSplatsViewState,
    session?: UpdateSession
  ): Promise<LoadedGSplatsData> {
    return this.updateView(viewState, session);
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
    session?: UpdateSession
  ): Promise<LoadedGSplatsData> {
    // Reset if view state changed
    if (!this.lastViewState || !viewStatesEqual(viewState, this.lastViewState)) {
      this.loadedLODs = [];
      this.lastViewState = {
        displayDims: [...viewState.displayDims],
        slicePosition: [...viewState.slicePosition],
        tolerance: [...viewState.tolerance],
        dimensions: viewState.dimensions,
      };
    }

    // Load LODs sequentially, stopping at first slow (cache-miss) load
    const startLevel = this.loadedLODs.length;

    for (let level = startLevel; level < this.nLods; level++) {
      const t0 = performance.now();
      const lodData = await this.lodLoaders[level].updateView(viewState, session);
      const elapsed = performance.now() - t0;

      this.loadedLODs.push(lodData);

      if (!this._initialLoadDone) {
        log.custom(
          LogEmoji.BROADCAST,
          Modules.GSPLATS_SPATIAL_INDEX_LOADER,
          `LOD ${level}/${this.nLods - 1}: ${lodData.splatCount} splats (${elapsed.toFixed(1)}ms)`
        );
      }

      // Short-circuit: if LOD 0 returned 0 splats, no higher LODs will have
      // visible splats either (LODs are spatially coextensive, LOD 0 is coarsest)
      if (level === 0 && lodData.splatCount === 0) {
        break;
      }

      // Stop if this load was slow (likely a cache miss / network fetch).
      // The refinement loop will pick up remaining LODs after the frame renders.
      // Always load at least LOD 0 regardless of timing.
      if (level > startLevel && elapsed > CACHE_HIT_THRESHOLD_MS) {
        break;
      }
    }

    if (!this._initialLoadDone && startLevel === 0) {
      this._initialLoadDone = true;
    }

    // Fire-and-forget: prefetch next unloaded LOD to warm cache
    this.prefetchNextLOD(viewState);

    return concatenateGSplatsData(this.loadedLODs);
  }

  /**
   * Prefetch the next unloaded LOD's data into cache.
   *
   * This is fire-and-forget: the prefetched chunks land in the L0/L1 cache
   * and become fast cache hits on the next updateView() call.
   * Uses the loader's built-in spatial index query + zarr fetch path,
   * but discards the result (only the cache side-effect matters).
   */
  private prefetchNextLOD(viewState: GSplatsViewState): void {
    const nextLevel = this.loadedLODs.length;
    if (nextLevel >= this.nLods) return;

    // Fire and forget — load the LOD data (which populates the cache)
    // but don't store or commit the result
    this.lodLoaders[nextLevel].updateView(viewState).catch(() => {
      // Ignore errors from prefetch (network failures, aborts)
    });
  }

  /**
   * Clean up all LOD loaders.
   */
  dispose(): void {
    for (const loader of this.lodLoaders) {
      loader.dispose();
    }
    this.lodLoaders = [];
    this.loadedLODs = [];
    this.lastViewState = null;
  }
}
