/**
 * Progressive Lines loader for multi-additive-LOD datasets.
 *
 * Wraps N `LinesSpatialIndexLoader` instances (one per `additive_<i>`
 * subgroup) using the Composite pattern. Mirrors
 * `GSplatsProgressiveLoader` / `PointsProgressiveLoader`.
 *
 * Concatenation is the only line-specific wrinkle: `segments` indices
 * are LOCAL to each subgroup's vertex array, so we offset-adjust them
 * by the cumulative vertex count when concatenating.
 *
 * @module data/lines/lines-progressive-loader
 */

import type {
  LinesDataLoader,
  LinesViewState,
  LoadedLinesData,
} from '../../types/lines';
import type { ScalarArray } from '../../types/points';
import type { LinesSpatialIndexLoader } from './lines-spatial-index-loader';
import type { UpdateSession } from '../../profiling/update-profiler';
import { log, Modules, LogEmoji } from '../../utils/log';

const CACHE_HIT_THRESHOLD_MS = 15;

function viewStatesEqual(a: LinesViewState, b: LinesViewState): boolean {
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
 * Concatenate per-LOD `LoadedLinesData`. Segment indices are
 * offset-adjusted by the cumulative vertex count across earlier
 * levels — local indices in level *k* become global indices in the
 * concatenated buffer.
 */
function concatenateLinesData(parts: LoadedLinesData[]): LoadedLinesData {
  if (parts.length === 0) {
    return {
      positions: new Float32Array(0),
      segments: new Uint32Array(0),
      widths: new Float32Array(0),
      colors: null,
      sharpness: null,
      segmentCount: 0,
      vertexCount: 0,
      ndim: 3,
    };
  }
  if (parts.length === 1) {
    return parts[0];
  }

  const ndim = parts[0].ndim;
  const totalVertices = parts.reduce((s, p) => s + p.vertexCount, 0);
  const totalSegments = parts.reduce((s, p) => s + p.segmentCount, 0);

  const positions = new Float32Array(totalVertices * ndim);
  const segments = new Uint32Array(totalSegments * 2);
  const widths = new Float32Array(totalVertices);

  const firstWithColors = parts.find((p) => p.colors !== null);
  let colors: Float32Array | Uint8Array | Uint16Array | null = null;
  if (firstWithColors?.colors) {
    if (firstWithColors.colors instanceof Uint8Array) {
      colors = new Uint8Array(totalVertices * 3);
    } else if (firstWithColors.colors instanceof Uint16Array) {
      colors = new Uint16Array(totalVertices * 3);
    } else {
      colors = new Float32Array(totalVertices * 3);
    }
  }
  const firstWithSharpness = parts.find((p) => p.sharpness !== null);
  let sharpness: Float32Array | null =
    firstWithSharpness?.sharpness ? new Float32Array(totalVertices) : null;

  const allHaveScalars = parts.every((p) => p.scalars !== undefined);
  let scalars: ScalarArray | undefined;
  if (allHaveScalars) {
    const first = parts[0].scalars as ScalarArray;
    const ctor = first.constructor as new (n: number) => ScalarArray;
    scalars = new ctor(totalVertices);
  }

  let vertexOffset = 0;
  let segmentOffset = 0;
  for (const part of parts) {
    positions.set(part.positions, vertexOffset * ndim);
    widths.set(part.widths, vertexOffset);
    // Offset-adjust segment indices into the concatenated vertex array.
    for (let i = 0; i < part.segments.length; i++) {
      segments[segmentOffset * 2 + i] = part.segments[i] + vertexOffset;
    }
    if (colors && part.colors) {
      colors.set(part.colors, vertexOffset * 3);
    } else if (colors && !part.colors) {
      const fill =
        colors instanceof Uint8Array ? 255 : colors instanceof Uint16Array ? 65535 : 1.0;
      for (let i = 0; i < part.vertexCount * 3; i++) {
        colors[vertexOffset * 3 + i] = fill;
      }
    }
    if (sharpness && part.sharpness) {
      sharpness.set(part.sharpness, vertexOffset);
    }
    if (scalars && part.scalars) {
      scalars.set(part.scalars, vertexOffset);
    }
    vertexOffset += part.vertexCount;
    segmentOffset += part.segmentCount;
  }

  const result: LoadedLinesData = {
    positions,
    segments,
    widths,
    colors,
    sharpness,
    segmentCount: totalSegments,
    vertexCount: totalVertices,
    ndim,
  };
  if (scalars !== undefined) {
    result.scalars = scalars;
  }
  return result;
}

/**
 * Progressive Lines loader.
 */
export class LinesProgressiveLoader implements LinesDataLoader {
  private lodLoaders: LinesSpatialIndexLoader[];
  private loadedLODs: LoadedLinesData[] = [];
  private lastViewState: LinesViewState | null = null;
  private nLods: number;
  private _initialLoadDone = false;

  constructor(lodLoaders: LinesSpatialIndexLoader[], nLods: number) {
    this.lodLoaders = lodLoaders;
    this.nLods = nLods;
  }

  get hasMoreLODs(): boolean {
    return this.loadedLODs.length < this.nLods;
  }

  get loadedLODCount(): number {
    return this.loadedLODs.length;
  }

  get totalLODCount(): number {
    return this.nLods;
  }

  async loadLines(
    viewState: LinesViewState,
    session?: UpdateSession
  ): Promise<LoadedLinesData> {
    return this.updateView(viewState, session);
  }

  async updateView(
    viewState: LinesViewState,
    session?: UpdateSession
  ): Promise<LoadedLinesData> {
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
          Modules.LINES_LOADER,
          `LOD ${level}/${this.nLods - 1}: ${lodData.segmentCount} segments (${elapsed.toFixed(1)}ms)`
        );
      }

      if (level === 0 && lodData.segmentCount === 0) {
        break;
      }

      if (level > startLevel && elapsed > CACHE_HIT_THRESHOLD_MS) {
        break;
      }
    }

    if (!this._initialLoadDone && startLevel === 0) {
      this._initialLoadDone = true;
    }

    const totalSegs = this.loadedLODs.reduce((s, d) => s + d.segmentCount, 0);
    if (this.loadedLODs.length < this.nLods) {
      log.info(
        Modules.LINES_LOADER,
        `Progressive Lines: ${this.loadedLODs.length}/${this.nLods} LODs (${totalSegs} segs) — refining`
      );
    } else if (startLevel < this.nLods) {
      log.info(
        Modules.LINES_LOADER,
        `Progressive Lines: ${this.nLods}/${this.nLods} LODs (${totalSegs} segs) — complete`
      );
    }

    this.prefetchNextLOD(viewState);

    return concatenateLinesData(this.loadedLODs);
  }

  private prefetchNextLOD(viewState: LinesViewState): void {
    const nextLevel = this.loadedLODs.length;
    if (nextLevel >= this.nLods) return;
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
