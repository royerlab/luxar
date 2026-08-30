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
import { assertColorLayout } from '../loaders';
import { ProgressiveMonitorAdapter } from '../loaders/progressive-monitor-adapter';
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
import { timeLodStageWithResult } from '../scene-loader/lod-load-stats';

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
    this.monitor = new ProgressiveMonitorAdapter(
      () => this.lodLoaders,
      path,
      'lines-spatial-index'
    );
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
          totalLODCount: this.nLods,
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

    // Stream under the shared streaming policy (see `streaming-policy.ts`):
    // `playback` commits a restored prefix as-is, and otherwise streams
    // cache-resident levels within the budget; `prefetch` deepens toward the
    // full decoded ladder (abort-safe, stored per level); `refine` stops at the
    // first cold/slow level. Mirrors GSplatsProgressiveLoader.
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
      const { data: lodData, allResident } = await timeLodStageWithResult(
        ({ allResident }) => `additive:lines:level:${level}:${allResident ? 'resident' : 'miss'}`,
        `additive:lines:level:${level}:aborted`,
        () => this.lodLoaders[level].updateViewWithResidency(viewState, session, signal)
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

      // NEVER break out because a level came back empty (#1456). It is a
      // tempting optimization — this loop used to latch a terminal "empty
      // ladder" on an empty LOD 0 and stop — but it is wrong here: this loader
      // is constructed only for ADDITIVE ladders (`createProgressiveLinesLoader`
      // iterates the `additive_<i>` subgroups), whose levels are DISJOINT
      // increments of one permutation, not coarse-to-fine resamplings of the
      // same elements. They are therefore NOT spatially coextensive: LOD 0 is a
      // small SUBSET of the node (a few thousand vertices under `-b stream:C` /
      // `--target-ms`, the recommended ladder shape), so a hidden-dimension
      // slice that none of ITS vertices lands on says nothing whatever about
      // levels 1..n-1, which may hold plenty of geometry right there. Stopping
      // here rendered such a slice permanently blank. The same reasoning
      // forbids inferring anything from a restored cache PREFIX whose LOD 0 is
      // empty.

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

    this.prefetchNextLOD(viewState);

    // Snapshot into the SliceCache (upgrade-if-longer): full ladders always;
    // PREFIXES only while a playback budget is active. Mirrors
    // GSplatsProgressiveLoader.
    if (this.loadedLODs.length === this.nLods || this._frameBudgetMs !== null) {
      storeLadder(this.sliceCache, this.path, viewState, this.loadedLODs, {
        scan: this._frameBudgetMs !== null,
        pin: viewState.prefetch === true,
        totalLODCount: this.nLods,
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
      const result = concatenateLinesData(this.loadedLODs);
      setPrefixParent(result, prevMemo);
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
  }
}
