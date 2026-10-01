/**
 * Progressive Lines loader for multi-additive-LOD datasets.
 *
 * Wraps N `LinesSpatialIndexLoader` instances (one per `additive_<i>`
 * subgroup) using the Composite pattern. The per-view lifecycle is the shared
 * `AdditiveLadderCore` (`../loaders/progressive/additive-ladder-core`), as for
 * `GSplatsProgressiveLoader` / `PointsProgressiveLoader`.
 *
 * Concatenation is the only line-specific wrinkle: `segments` indices
 * are LOCAL to each subgroup's vertex array, so we offset-adjust them
 * by the cumulative vertex count when concatenating.
 *
 * @module data/lines/lines-progressive-loader
 */

import type { LinesDataLoader, LinesViewState, LoadedLinesData } from '../../types/lines';
import type { ScalarArray } from '../../types/points';
import type { LinesSpatialIndexLoader } from './lines-spatial-index-loader';
import type { UpdateSession } from '../../profiling/update-profiler';
import type {
  LoaderMetrics,
  MonitorEventListener,
  QueryInfo,
} from '../../types/data-monitor-types';
import { assertColorLayout } from '../loaders';
import { ProgressiveMonitorAdapter } from '../loaders/progressive-monitor-adapter';
import {
  concatColorsWhiteFilled,
  concatOptionalField,
  concatRequiredField,
} from '../loaders/progressive/concat-helpers';
import {
  AdditiveLadderCore,
  type LadderGeometry,
} from '../loaders/progressive/additive-ladder-core';
import { LINE_FLOATS_PER_SEGMENT } from '../../rendering/element-texture-layout';
import type { LadderResidency } from '../scene-loader/progressive/residency-budget';
import type { SliceCache } from '../../cache/slice-cache';
import { Modules } from '../../utils/log';

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
    // GUARD (belt-and-braces): a ladder payload must never publish the on-disk
    // VERTEX range bounds. Not because `parts[0]`'s ranges describe the wrong
    // space — it is always `additive_0`, whose on-disk vertex range IS the
    // parent per-vertex union CSR's PREFIX (#1422), so passing them through
    // would in fact resolve correctly while it is the only committed level. It
    // is that `parts.length === 1` is not "unladdered": this loader only exists
    // for `n_additive_sublods` nodes, so it is the first-paint state of EVERY
    // ladder — and a tooltip that is right at first paint and silently degrades
    // to the raw slot the moment a second level lands is worse than one
    // consistently at the raw slot, which is what every doc surface promises.
    // `createProgressiveLinesLoader` now also clears `has_labels` /
    // `has_image_labels` / `has_keys` on each sub-LOD's attrs, so the ranges are
    // normally never published at all; this keeps the invariant true whatever
    // attrs a sub-LOD carries. (A ladder's supported labels/keys channels each
    // have ONE per-vertex union CSR on the parent since #1422; composing a map
    // across the levels of that union is what #1439 did for the POINTS ladder —
    // `points-progressive-loader.ts`'s `levelOffsets` path — and lines has no
    // counterpart yet, so a laddered lines node still hovers at the raw slot.)
    const only = parts[0];
    if (only.vertexRangeBounds === undefined) return only;
    const stripped: LoadedLinesData = { ...only };
    delete stripped.vertexRangeBounds;
    return stripped;
  }

  const ndim = parts[0].ndim;
  // Same fail-fast contract as the ladder dtype checks (see concat-helpers):
  // `ndim` strides the position concat below, so sub-LODs disagreeing on
  // dimensionality would mis-stride every vertex after the first part —
  // silent corruption. Dimensionality is per-dataset; a mismatch is
  // malformed data. (Points concat is immune: its loader projects to
  // stride-3 before concatenation.)
  for (const part of parts) {
    if (part.ndim !== ndim) {
      throw new Error(
        'concatenateLinesData: mixed dimensionality across LOD levels ' +
          `(ndim ${part.ndim} vs ${ndim}) — ladder levels must share the ` +
          'dataset dimensionality.'
      );
    }
  }
  // Per-part color-layout check, distinct from the cross-level mismatch
  // guarded below: those throws catch LODs that DISAGREE on dtype/layout,
  // but a single part can carry an RGBA buffer while OMITTING
  // `colorComponents: 4` (it defaults to 3). That satisfies the downstream
  // `count·3` minimum yet mis-strides every vertex after the first — silent
  // corruption. Assert each part's raw length against its own declared
  // layout before allocation so an omitted declaration throws loudly here.
  // Names the offending level (concat-helpers' convention) so a corrupt
  // store is diagnosable without a debugger.
  for (const [levelIdx, part] of parts.entries()) {
    assertColorLayout(
      part.colors,
      part.vertexCount,
      part.colorComponents ?? 3,
      `concatenateLinesData (LOD level ${levelIdx})`
    );
  }
  const totalVertices = parts.reduce((s, p) => s + p.vertexCount, 0);
  const totalSegments = parts.reduce((s, p) => s + p.segmentCount, 0);
  const count = (p: LoadedLinesData) => p.vertexCount;

  // Straightforward per-vertex fields via the shared helpers (dtype preserved).
  const positions = concatRequiredField(parts, (p) => p.positions, count, ndim, 'positions');
  const widths = concatRequiredField(parts, (p) => p.widths, count, 1, 'widths');
  const scalars = concatOptionalField(parts, (p) => p.scalars as ScalarArray, count, 1, 'scalars');

  // Bespoke fields: segments need vertex-offset remapping; sharpness is
  // partial (nullable, not all-or-nothing). Colours follow the shared ladder
  // policy: a rung without them is filled with white.
  const segments = new Uint32Array(totalSegments * 2);
  const colored = concatColorsWhiteFilled(
    parts,
    (p) => ({ colors: p.colors, components: p.colorComponents }),
    count,
    'concatenateLinesData'
  );
  const colors = colored?.colors ?? null;
  const colorK = colored?.colorComponents ?? 3;
  const firstWithSharpness = parts.find((p) => p.sharpness !== null);
  let sharpness: Float32Array | null = firstWithSharpness?.sharpness
    ? new Float32Array(totalVertices)
    : null;

  let vertexOffset = 0;
  let segmentOffset = 0;
  for (const part of parts) {
    // Offset-adjust segment indices into the concatenated vertex array.
    for (let i = 0; i < part.segments.length; i++) {
      segments[segmentOffset * 2 + i] = part.segments[i] + vertexOffset;
    }
    if (sharpness && part.sharpness) {
      sharpness.set(part.sharpness, vertexOffset);
    } else if (sharpness && !part.sharpness) {
      // Missing-sharpness parts get the DEFAULT knob (0.5 -> beta=2,
      // Gaussian), exactly what the worker projection substitutes for a
      // null sharpness array (projection/lines.ts) — not 0.0. Keeps a
      // mixed-sharpness ladder's concat byte-identical to what each part
      // renders standalone, which the append fast path's prefix-identity
      // contract relies on (and fixes a full-rewrite inconsistency where
      // sharpness-less parts turned razor-sharp when a sharpness-carrying
      // level joined the ladder). Mirrors the white colour fill.
      sharpness.fill(0.5, vertexOffset, vertexOffset + part.vertexCount);
    }
    // (scalars are fully concatenated above via concatOptionalField.)
    vertexOffset += part.vertexCount;
    segmentOffset += part.segmentCount;
  }

  // No `vertexRangeBounds`: the concat interleaves several levels' on-disk vertex
  // spaces into one buffer, so no single ascending on-disk range list describes
  // it (same reason the single-part branch above strips them).
  const result: LoadedLinesData = {
    positions,
    segments,
    widths,
    colors,
    ...(colors ? { colorComponents: colorK } : {}),
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

const LINES_LADDER: LadderGeometry<LoadedLinesData> = {
  kind: 'lines',
  module: Modules.LINES_LOADER,
  label: 'Lines',
  unit: 'segments',
  warnPrefix: 'Lines',
  countOf: (data) => data.segmentCount,
  concat: (parts) => concatenateLinesData(parts),
  empty: () => concatenateLinesData([]),
  // 6 RGBA32F texels/segment — the largest element row of any geometry, and
  // ~3.6x this payload's own bytes per vertex. Omitting it made the shared
  // budget under-count Lines far more than GSplats.
  bytesPerElement: LINE_FLOATS_PER_SEGMENT * Float32Array.BYTES_PER_ELEMENT,
  stampsRestoredOrigin: true,
};

/**
 * Progressive Lines loader: the Lines face of the shared additive-ladder
 * engine (`AdditiveLadderCore`), which owns the whole per-view lifecycle. This
 * class adds the Lines concatenation (segment-index offsetting).
 */
export class LinesProgressiveLoader implements LinesDataLoader {
  private readonly core: AdditiveLadderCore<LoadedLinesData, LinesSpatialIndexLoader>;
  private readonly monitor: ProgressiveMonitorAdapter;

  constructor(
    lodLoaders: LinesSpatialIndexLoader[],
    nLods: number,
    path: string,
    energyTable?: ReadonlyArray<number | null | undefined>,
    sliceCache?: SliceCache | null
  ) {
    this.core = new AdditiveLadderCore(lodLoaders, nLods, path, LINES_LADDER, {
      energyTable,
      sliceCache,
    });
    this.monitor = new ProgressiveMonitorAdapter(
      () => this.core.rungLoaders,
      path,
      'lines-spatial-index'
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

  /** Load lines data (delegates to {@link updateView}). */
  async loadLines(
    viewState: LinesViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedLinesData> {
    return this.updateView(viewState, session, signal);
  }

  /** Stream as many LODs as the pass allows for `viewState` (see the core). */
  updateView(
    viewState: LinesViewState,
    session?: UpdateSession,
    signal?: AbortSignal,
    residencyAllowanceBytes?: number
  ): Promise<LoadedLinesData> {
    return this.core.updateView(viewState, session, signal, residencyAllowanceBytes);
  }

  /** Predicted-view warm-up of the coarse rung (`dispatchPredictivePrefetch`). */
  prefetchChunks(viewState: LinesViewState, signal?: AbortSignal): Promise<void> {
    return this.core.prefetchChunks(viewState, signal);
  }

  /** Predicted-view warm-up over a current → predicted transition. */
  prefetchChunkBoundary(
    current: LinesViewState,
    predicted: LinesViewState,
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
