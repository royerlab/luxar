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

import type { LinesDataLoader, LinesViewState, LoadedLinesData } from '../../types/lines';
import type { ScalarArray } from '../../types/points';
import { setPrefixParent } from '../../types/prefix-lineage';
import type { LinesSpatialIndexLoader } from './lines-spatial-index-loader';
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
 * Concatenate per-LOD `LoadedLinesData`. Segment indices are
 * offset-adjusted by the cumulative vertex count across earlier
 * levels — local indices in level *k* become global indices in the
 * concatenated buffer.
 *
 * Exported as the REFERENCE implementation: multi-level ladders build
 * incrementally in {@link LinesLadderArena} (byte-equivalence pinned by
 * tests); this full rebuild still serves the k ≤ 1 cases.
 */
export function concatenateLinesData(parts: LoadedLinesData[]): LoadedLinesData {
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
  const totalVertices = parts.reduce((s, p) => s + p.vertexCount, 0);
  const totalSegments = parts.reduce((s, p) => s + p.segmentCount, 0);
  const count = (p: LoadedLinesData) => p.vertexCount;

  // Straightforward per-vertex fields via the shared helpers (dtype preserved).
  const positions = concatRequiredField(parts, (p) => p.positions, count, ndim, 'positions');
  const widths = concatRequiredField(parts, (p) => p.widths, count, 1, 'widths');
  const scalars = concatOptionalField(parts, (p) => p.scalars as ScalarArray, count, 1, 'scalars');

  // Bespoke fields: segments need vertex-offset remapping; colors fill missing
  // LODs with white; sharpness is partial (nullable, not all-or-nothing).
  const segments = new Uint32Array(totalSegments * 2);

  const firstWithColors = parts.find((p) => p.colors !== null);
  // Color stride follows the ladder's layout (3 RGB / 4 RGBA). Mixed
  // layouts across levels are rejected below — like the dtype contract,
  // the layout is a property of the dataset, and a silent 3-vs-4 mix
  // would mis-stride every vertex after the offending level (the gsplat
  // ladder shipped exactly this bug before its colorK guard).
  const colorK = firstWithColors?.colorComponents ?? 3;
  let colors: Float32Array | Uint8Array | Uint16Array | null = null;
  if (firstWithColors?.colors) {
    if (firstWithColors.colors instanceof Uint8Array) {
      colors = new Uint8Array(totalVertices * colorK);
    } else if (firstWithColors.colors instanceof Uint16Array) {
      colors = new Uint16Array(totalVertices * colorK);
    } else {
      colors = new Float32Array(totalVertices * colorK);
    }
  }
  const firstWithSharpness = parts.find((p) => p.sharpness !== null);
  let sharpness: Float32Array | null = firstWithSharpness?.sharpness
    ? new Float32Array(totalVertices)
    : null;

  let vertexOffset = 0;
  let segmentOffset = 0;
  for (const [levelIdx, part] of parts.entries()) {
    // Offset-adjust segment indices into the concatenated vertex array.
    for (let i = 0; i < part.segments.length; i++) {
      segments[segmentOffset * 2 + i] = part.segments[i] + vertexOffset;
    }
    if (colors && part.colors) {
      // LADDER-DTYPE CONTRACT (see concat-helpers.ts): `set` converts by
      // VALUE, not semantics — a Float32 (0..1) level written into a Uint8
      // (0..255) merge truncates to garbage, and the reverse writes 255×
      // values. The writer emits one color dtype per ladder; fail fast.
      // Names the offending level (concat-helpers' convention) so a corrupt
      // store is diagnosable without a debugger.
      if (part.colors.constructor !== colors.constructor) {
        throw new Error(
          'concatenateLinesData: mixed color dtypes across LOD levels ' +
            `(level ${levelIdx}: ${part.colors.constructor.name} vs ` +
            `${colors.constructor.name}) — ladder levels must share each ` +
            "field's dtype."
        );
      }
      if ((part.colorComponents ?? 3) !== colorK) {
        throw new Error(
          'concatenateLinesData: mixed color layouts across LOD levels ' +
            `(level ${levelIdx}: ${part.colorComponents ?? 3} components vs ` +
            `${colorK}) — ladder levels must share the color layout.`
        );
      }
      colors.set(part.colors, vertexOffset * colorK);
    } else if (colors && !part.colors) {
      // White fill; for an RGBA ladder the alpha column gets the same
      // max value = 1.0 opaque (the per-element-opacity identity).
      const fill = colors instanceof Uint8Array ? 255 : colors instanceof Uint16Array ? 65535 : 1.0;
      for (let i = 0; i < part.vertexCount * colorK; i++) {
        colors[vertexOffset * colorK + i] = fill;
      }
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
      // level joined the ladder). Mirrors the white color fill above.
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
  return result;
}

/** Lines color dtypes (mirrors `LoadedLinesData.colors`). */
type LinesColorArray = Float32Array | Uint8Array | Uint16Array;

/** Full-scale white/opaque fill for a colorless level (per dtype). */
function linesColorFillValue(ctor: new (n: number) => LinesColorArray): number {
  if (ctor === Uint8Array) return 255;
  if (ctor === Uint16Array) return 65535;
  return 1.0;
}

/**
 * Growable ladder arena for the lines progressive concat (perf lever L2).
 *
 * Incremental equivalent of {@link concatenateLinesData}: appending level
 * *k* copies ONLY level *k*'s bytes (amortized O(N_total) across the ladder
 * instead of O(k·N) full rebuilds), with `snapshot()` byte-identical to the
 * reference concat of the same parts — same segment-index vertex-offset
 * remapping, white color fill, sharpness 0.5 default fill, all-or-nothing
 * scalars, and validation errors (reference order + messages). Snapshots are
 * prefix-stable `subarray` views (see `concat-arena.ts` for the aliasing
 * contract). One arena per loader reset generation. Exported for the
 * arena-vs-reference equivalence tests.
 */
export class LinesLadderArena {
  private ndim = 3;
  private positions: ArenaField<Float32Array> | null = null;
  private widths: ArenaField<Float32Array> | null = null;
  /** Segment index pairs; element = segment, 2 entries each. */
  private segments: ArenaField<Uint32Array> | null = null;
  // Colors follow the reference's fill-with-white policy (field created by
  // the first color-carrying level, white-backfilled); sharpness likewise
  // but with the projection default 0.5; scalars are all-or-nothing.
  private colors: ArenaField<LinesColorArray> | null = null;
  private colorK: 3 | 4 = 3;
  private sharpness: ArenaField<Float32Array> | null = null;
  private readonly scalars = new OptionalLadderField<ScalarArray>();
  private appendedParts = 0;
  private totalVertices = 0;
  private totalSegments = 0;

  /** Elements (segments) appended so far. */
  get totalElements(): number {
    return this.totalSegments;
  }

  /** Vertices appended so far (per-vertex fields stride by this). */
  get totalVertexElements(): number {
    return this.totalVertices;
  }

  /** Aggregated copy-work counters across fields (test/verification hook). */
  debugStats(): ArenaFieldStats {
    const stats: ArenaFieldStats = { appendedEntries: 0, reallocCopiedEntries: 0 };
    const fields = [
      this.positions,
      this.widths,
      this.segments,
      this.colors,
      this.sharpness,
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
  appendThrough(parts: LoadedLinesData[], isFinal: boolean): void {
    const from = this.appendedParts;
    // APPEND-ONLY PRECONDITION: `parts` must extend the levels already
    // appended (the loader only ever pushes onto `loadedLODs` within a reset
    // generation; a view change makes a fresh arena). A shorter list would
    // silently snapshot stale extra content, so fail loudly instead.
    if (parts.length < from) {
      throw new Error(
        `LinesLadderArena: ladder shrank (${parts.length} levels vs ${from} ` +
          'already appended) — an arena is append-only within a reset generation.'
      );
    }
    if (parts.length === from) {
      if (isFinal) this.trim();
      return;
    }
    if (from === 0) {
      this.ndim = parts[0].ndim;
    }
    const ndim = this.ndim;

    // ---- Validation (reference order: ndim → positions dtype → widths
    // dtype → scalars presence/dtype → colors dtype/layout), no writes. ----
    for (let i = from; i < parts.length; i++) {
      if (parts[i].ndim !== ndim) {
        throw new Error(
          'concatenateLinesData: mixed dimensionality across LOD levels ' +
            `(ndim ${parts[i].ndim} vs ${ndim}) — ladder levels must share the ` +
            'dataset dimensionality.'
        );
      }
    }
    const posCtor = (this.positions?.elementCtor ?? parts[0].positions.constructor) as new (
      n: number
    ) => Float32Array;
    validateLadderFieldDtype(parts, from, (p) => p.positions, posCtor, 'positions');
    const widthsCtor = (this.widths?.elementCtor ?? parts[0].widths.constructor) as new (
      n: number
    ) => Float32Array;
    validateLadderFieldDtype(parts, from, (p) => p.widths, widthsCtor, 'widths');
    const scalarsPlan = this.scalars.plan(parts, from, (p) => p.scalars, 'scalars');

    let colorCtor = this.colors?.elementCtor ?? null;
    let colorK: 3 | 4 = this.colors ? this.colorK : 3;
    if (!colorCtor) {
      const firstWithColors = parts.slice(from).find((p) => p.colors !== null);
      if (firstWithColors?.colors) {
        colorCtor = firstWithColors.colors.constructor as new (n: number) => LinesColorArray;
        colorK = firstWithColors.colorComponents ?? 3;
      }
    }
    if (colorCtor) {
      for (let i = from; i < parts.length; i++) {
        const partColors = parts[i].colors;
        if (!partColors) continue;
        if (partColors.constructor !== colorCtor) {
          throw new Error(
            'concatenateLinesData: mixed color dtypes across LOD levels ' +
              `(level ${i}: ${partColors.constructor.name} vs ` +
              `${colorCtor.name}) — ladder levels must share each ` +
              "field's dtype."
          );
        }
        if ((parts[i].colorComponents ?? 3) !== colorK) {
          throw new Error(
            'concatenateLinesData: mixed color layouts across LOD levels ' +
              `(level ${i}: ${parts[i].colorComponents ?? 3} components vs ` +
              `${colorK}) — ladder levels must share the color layout.`
          );
        }
      }
    }

    // ---- Writes (validation passed; nothing below throws) ----
    let batchVertices = 0;
    let batchSegments = 0;
    for (let i = from; i < parts.length; i++) {
      batchVertices += parts[i].vertexCount;
      batchSegments += parts[i].segmentCount;
    }
    const newVertexTotal = this.totalVertices + batchVertices;
    const newSegmentTotal = this.totalSegments + batchSegments;

    if (!this.positions) {
      this.positions = new ArenaField(posCtor, ndim, newVertexTotal);
      this.widths = new ArenaField(widthsCtor, 1, newVertexTotal);
      this.segments = new ArenaField(Uint32Array, 2, newSegmentTotal);
    } else {
      this.positions.ensureCapacity(newVertexTotal, isFinal);
      this.widths?.ensureCapacity(newVertexTotal, isFinal);
      this.segments?.ensureCapacity(newSegmentTotal, isFinal);
    }
    this.scalars.apply(scalarsPlan, 1, newVertexTotal, isFinal);
    if (colorCtor && !this.colors) {
      // First color-carrying level: create + white-backfill (reference:
      // fill loop over colorless parts).
      this.colorK = colorK;
      this.colors = new ArenaField<LinesColorArray>(colorCtor, colorK, newVertexTotal);
      this.colors.appendFill(linesColorFillValue(colorCtor), this.totalVertices);
    } else if (this.colors) {
      this.colors.ensureCapacity(newVertexTotal, isFinal);
    }
    const firstWithSharpness = this.sharpness
      ? null
      : parts.slice(from).find((p) => p.sharpness !== null);
    if (firstWithSharpness && !this.sharpness) {
      // First sharpness-carrying level: create + backfill the DEFAULT knob
      // (0.5 → beta=2, Gaussian) exactly like the reference concat — see
      // its rationale for the append fast path's prefix-identity contract.
      this.sharpness = new ArenaField(Float32Array, 1, newVertexTotal);
      this.sharpness.appendFill(0.5, this.totalVertices);
    } else if (this.sharpness) {
      this.sharpness.ensureCapacity(newVertexTotal, isFinal);
    }

    const positions = this.positions;
    const widths = this.widths;
    const segments = this.segments;
    const colors = this.colors;
    const sharpness = this.sharpness;
    for (let i = from; i < parts.length; i++) {
      const part = parts[i];
      const nVerts = part.vertexCount;
      const vertexOffset = this.totalVertices;
      positions.append(part.positions, nVerts);
      widths?.append(part.widths, nVerts);
      // Offset-adjust segment indices into the concatenated vertex array.
      segments?.appendWith(part.segmentCount, (buf, base) => {
        for (let j = 0; j < part.segments.length; j++) {
          buf[base + j] = part.segments[j] + vertexOffset;
        }
      });
      if (colors) {
        if (part.colors) {
          colors.append(part.colors, nVerts);
        } else {
          colors.appendFill(linesColorFillValue(colors.elementCtor), nVerts);
        }
      }
      if (sharpness) {
        if (part.sharpness) {
          sharpness.append(part.sharpness, nVerts);
        } else {
          sharpness.appendFill(0.5, nVerts);
        }
      }
      this.scalars.field?.append(part.scalars as ScalarArray, nVerts);
      this.totalVertices += nVerts;
      this.totalSegments += part.segmentCount;
      this.appendedParts++;
    }

    if (isFinal) this.trim();
  }

  /** Snapshot the current contents as prefix-stable views (fresh object). */
  snapshot(): LoadedLinesData {
    const colors = this.colors?.view() ?? null;
    const result: LoadedLinesData = {
      positions: this.positions?.view() ?? new Float32Array(0),
      segments: this.segments?.view() ?? new Uint32Array(0),
      widths: this.widths?.view() ?? new Float32Array(0),
      colors,
      ...(colors ? { colorComponents: this.colorK } : {}),
      sharpness: this.sharpness?.view() ?? null,
      segmentCount: this.totalSegments,
      vertexCount: this.totalVertices,
      ndim: this.ndim,
    };
    if (this.scalars.field) {
      result.scalars = this.scalars.field.view();
    }
    return result;
  }

  private trim(): void {
    this.positions?.trimToFit();
    this.widths?.trimToFit();
    this.segments?.trimToFit();
    this.colors?.trimToFit();
    this.sharpness?.trimToFit();
    this.scalars.field?.trimToFit();
  }
}

/**
 * Progressive Lines loader.
 */
export class LinesProgressiveLoader implements LinesDataLoader {
  private lodLoaders: LinesSpatialIndexLoader[];
  private loadedLODs: LoadedLinesData[] = [];
  private lastViewState: LinesViewState | null = null;
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
    result: LoadedLinesData;
  } | null = null;
  // Growable ladder arena backing `concatenateMemoized` (perf lever L2):
  // appending a level copies only that level's bytes instead of rebuilding
  // the whole concat. One arena per reset generation. Mirrors
  // GSplatsProgressiveLoader.
  private _arena: LinesLadderArena | null = null;
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
    lodLoaders: LinesSpatialIndexLoader[],
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

  async loadLines(
    viewState: LinesViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedLinesData> {
    return this.updateView(viewState, session, signal);
  }

  async updateView(
    viewState: LinesViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedLinesData> {
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
    const finish = (): LoadedLinesData =>
      isPrefetch ? concatenateLinesData([]) : this.concatenateMemoized(session);

    if (!this.lastViewState || !viewStatesEqual(viewState, this.lastViewState)) {
      // DEPARTURE store: snapshot the outgoing view's partial ladder under
      // the OUTGOING key before discarding — scrub-back stays warm even when
      // ladders never complete between navigations. Mirrors Points/GSplats.
      if (this.lastViewState && this.loadedLODs.length > 0) {
        storeLadder(this.sliceCache, this.path, this.lastViewState, this.loadedLODs, {
          scan: this._frameBudgetMs !== null,
          pin: viewState.prefetch === true,
        });
      }
      // Try the SliceCache before discarding the ladder (see GSplats loader).
      const restored = restoreLadder<LoadedLinesData>(
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
        restored !== null && restored.length > 0 && restored[0].segmentCount === 0;
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

    // Known-empty slice: LOD 0 committed 0 segments on a prior pass for this
    // SAME view (the view-change branch re-derives the flag from the restored prefix). Terminal
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
          Modules.LINES_LOADER,
          `LOD ${level}/${this.nLods - 1}: ${lodData.segmentCount} segments (${elapsed.toFixed(1)}ms${allResident ? '' : ', miss'})`
        );
      }

      // Empty LOD 0 → all higher LODs empty too (spatially coextensive). Mark
      // terminal so `hasMoreLODs` reads false and refinement skips the empties.
      if (level === 0 && lodData.segmentCount === 0) {
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
   * {@link LinesLadderArena} (perf lever L2): only the NEW levels' bytes
   * are copied, and the result is a prefix-stable view — byte-identical to
   * the reference {@link concatenateLinesData} rebuild. k ≤ 1 keeps the
   * reference path (empty result / the raw single part as-is). Each fresh
   * result is also stamped with its {@link setAppendSpan | append span}
   * (segment-space, plus the lines-only vertex-space span).
   */
  private concatenateMemoized(session?: UpdateSession): LoadedLinesData {
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
      let result: LoadedLinesData;
      if (k <= 1) {
        // Reference shape for the trivial cases: [] → empty result,
        // [part] → the raw part as-is (no copy; the arena starts at k=2).
        result = concatenateLinesData(this.loadedLODs);
      } else {
        if (!this._arena || this._arenaGeneration !== this._resetGeneration) {
          this._arena = new LinesLadderArena();
          this._arenaGeneration = this._resetGeneration;
        }
        this._arena.appendThrough(this.loadedLODs, k === this.nLods);
        result = this._arena.snapshot();
      }
      setPrefixParent(result, prevMemo);
      const prevSegments = prevMemo?.segmentCount ?? 0;
      const prevVertices = prevMemo?.vertexCount ?? 0;
      setAppendSpan(result, {
        fromElement: prevSegments,
        elementCount: result.segmentCount - prevSegments,
        fromVertex: prevVertices,
        vertexCount: result.vertexCount - prevVertices,
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

  private prefetchNextLOD(viewState: LinesViewState): void {
    // Same teardown race as the streaming loop: a dispose() between the
    // awaited level and this fire-and-forget clears `lodLoaders`, and
    // indexing it would TypeError before the .catch can swallow anything.
    if (this._disposed) return;
    const nextLevel = this.loadedLODs.length;
    if (nextLevel >= this.nLods) return;
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
