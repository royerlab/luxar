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
 * by the cumulative vertex count when concatenating — and, for a ladder whose
 * parent carries a per-vertex label CSR, each level's on-disk vertex ranges are
 * shifted into that CSR's index space (`levelOffsets`, the twin of the Points
 * ladder composition, #1439).
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
import { log, Modules } from '../../utils/log';

/**
 * Compose the retained payloads' on-disk VERTEX ranges into the parent's union
 * label CSR index space (#1422): payload `i`, covering logical levels
 * `[payloadLevelStarts[i], next)`, is shifted by `levelOffsets[start]` and
 * bounded by its own span, so a range can never name a sibling level's row.
 * A folded prefix starts at level 0, where the shift is 0 — its ranges are
 * already in union space. The concatenated ranges describe the concatenated
 * vertices in order, which is all `composeLinesElementIds` needs.
 *
 * Returns `null` (after one `warn`) when the inputs are inconsistent; picking
 * then falls back to the raw slot. An empty payload contributes nothing.
 */
function composeLadderVertexRanges(
  parts: LoadedLinesData[],
  payloadLevelStarts: readonly number[],
  logicalDepth: number,
  levelOffsets: readonly number[],
  warn: (message: string) => void
): Uint32Array | null {
  const fail = (why: string): null => {
    warn(`Progressive Lines: ${why} — picking labels fall back to the visible-buffer slot.`);
    return null;
  };
  if (payloadLevelStarts.length !== parts.length || levelOffsets.length <= logicalDepth) {
    return fail(`inconsistent retained-payload lineage at logical depth ${logicalDepth}`);
  }
  const out: number[] = [];
  for (const [i, part] of parts.entries()) {
    if (part.vertexCount === 0) continue;
    const end = payloadLevelStarts[i + 1] ?? logicalDepth;
    const why = appendShiftedRanges(part, levelOffsets, payloadLevelStarts[i], end, out);
    if (why !== null) return fail(`payload ${i} ${why}`);
  }
  return Uint32Array.from(out);
}

/**
 * Append one payload's on-disk vertex ranges, covering logical levels
 * `[start, end)`, shifted into the union space. Returns why it refused, or null.
 */
function appendShiftedRanges(
  part: LoadedLinesData,
  levelOffsets: readonly number[],
  start: number,
  end: number,
  out: number[]
): string | null {
  if (!(start >= 0 && end > start && end < levelOffsets.length)) {
    return `has an invalid logical level range [${start}, ${end})`;
  }
  const bounds = part.vertexRangeBounds;
  if (bounds === undefined || bounds.length % 2 !== 0) {
    return 'published no usable on-disk vertex ranges';
  }
  const base = levelOffsets[start];
  const span = levelOffsets[end] - base;
  for (let k = 0; k < bounds.length; k += 2) {
    // Negated compare so a NaN also fails closed.
    if (!(bounds[k] <= bounds[k + 1] && bounds[k + 1] <= span)) {
      return `names vertex rows past its ${span} on-disk rows`;
    }
    out.push(base + bounds[k], base + bounds[k + 1]);
  }
  return null;
}

/**
 * Concatenate per-LOD `LoadedLinesData`. Segment indices are
 * offset-adjusted by the cumulative vertex count across earlier
 * levels — local indices in level *k* become global indices in the
 * concatenated buffer.
 *
 * `levelOffsets` (non-null only for a ladder whose PARENT declares a per-vertex
 * string/image CSR) composes the levels' on-disk vertex ranges into that CSR's
 * index space ({@link composeLadderVertexRanges}); without it no ranges are
 * published at all.
 */
function concatenateLinesData(
  parts: LoadedLinesData[],
  payloadLevelStarts: readonly number[],
  logicalDepth: number,
  levelOffsets: readonly number[] | null,
  warn: (message: string) => void
): LoadedLinesData {
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
  const ranges =
    levelOffsets === null
      ? null
      : composeLadderVertexRanges(parts, payloadLevelStarts, logicalDepth, levelOffsets, warn);
  if (parts.length === 1) {
    // `additive_0` at first paint, or a folded prefix: both start at level 0,
    // so composed ranges equal the payload's own and it passes through. Without
    // them (no parent CSR, or a failed composition) the payload must not publish
    // its ranges: on a ladder they are level-space, and a tooltip right at first
    // paint that silently degrades once a second level lands is worse than one
    // consistently at the raw slot. Stripped as a copy — the level's own payload
    // is still owned by its accumulator.
    const only = parts[0];
    if (ranges !== null || only.vertexRangeBounds === undefined) return only;
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
  if (ranges !== null) result.vertexRangeBounds = ranges;
  return result;
}

const LINES_LADDER: LadderGeometry<LoadedLinesData> = {
  kind: 'lines',
  module: Modules.LINES_LOADER,
  label: 'Lines',
  unit: 'segments',
  warnPrefix: 'Lines',
  countOf: (data) => data.segmentCount,
  concat: (parts, layout) =>
    concatenateLinesData(parts, layout.payloadLevelStarts, layout.loadedLevels, null, () => {}),
  empty: () => concatenateLinesData([], [], 0, null, () => {}),
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

  // One fail-closed composition warning per LOADER (the concat re-runs on every
  // memo miss — tens per second under playback), as in the Points twin.
  private composeWarned = false;

  constructor(
    lodLoaders: LinesSpatialIndexLoader[],
    nLods: number,
    path: string,
    energyTable?: ReadonlyArray<number | null | undefined>,
    sliceCache?: SliceCache | null,
    levelOffsets?: readonly number[] | null
  ) {
    // CSR-style bounds over the levels' ON-DISK vertex counts (`nLods + 1`
    // entries) inside the parent's union label CSR; null publishes no ranges.
    const offsets = levelOffsets ?? null;
    const warn = (message: string): void => {
      if (this.composeWarned) return;
      this.composeWarned = true;
      log.warning(Modules.LINES_LOADER, `${message} (logged once per node)`);
    };
    const geometry: LadderGeometry<LoadedLinesData> = {
      ...LINES_LADDER,
      concat: (parts, layout) =>
        concatenateLinesData(parts, layout.payloadLevelStarts, layout.loadedLevels, offsets, warn),
    };
    this.core = new AdditiveLadderCore(lodLoaders, nLods, path, geometry, {
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
    return this.core.nodeMetrics(this.monitor.getMetrics());
  }

  /** Clean up all LOD loaders. */
  dispose(): void {
    this.core.dispose();
  }
}
