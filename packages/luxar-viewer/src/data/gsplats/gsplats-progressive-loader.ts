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
 * - After first paint, initializes the remaining spatial indexes concurrently
 * - Prefetches up to three unloaded LODs when the L0 cache has headroom
 * - Keeps playback/shadow prefetch at one rung to avoid abandoned-slice waste
 *
 * LODs are additive: LOD 0 contains the coarsest (highest-amplitude) splats,
 * and each subsequent LOD adds residual detail. The loader concatenates all
 * loaded LODs into a single LoadedGSplatsData.
 *
 * @module data/gsplats-progressive-loader
 */

import type { GSplatsDataLoader, GSplatsViewState, LoadedGSplatsData } from '../../types/gsplats';
import { setPrefixParent } from '../../types/prefix-lineage';
import type { GSplatsSpatialIndexLoader } from './gsplats-spatial-index-loader';
import type { UpdateSession } from '../../profiling/update-profiler';
import type {
  LoaderMetrics,
  MonitorEventListener,
  QueryInfo,
} from '../../types/data-monitor-types';
import { assertColorLayout } from '../loaders';
import { ProgressiveMonitorAdapter } from '../loaders/progressive-monitor-adapter';
import { concatRequiredField } from '../loaders/progressive/concat-helpers';
import {
  classifyStreamingPass,
  shouldStopBeforeLevel,
  shouldStopAfterLevel,
} from '../loaders/progressive/streaming-policy';
import {
  deleteLadder,
  measureLodBytes,
  restoreLadderSnapshot,
  storeLadder,
} from '../loaders/progressive/slice-cache-helper';
import { planLadderRollback } from '../loaders/progressive/pass-rollback';
import { SPLAT_FLOATS_PER_SPLAT } from '../../rendering/element-texture-layout';
import {
  ladderResidentBytes,
  type LadderResidency,
} from '../scene-loader/progressive/residency-budget';
import { viewStatesEqual } from '../loaders/progressive/view-state-equal';
import type { SliceCache } from '../../cache/slice-cache';
import { log, Modules, LogEmoji } from '../../utils/log';
import { timeLodStageWithResult } from '../scene-loader/lod-load-stats';

const MAX_PREFETCH_LEVELS = 3;

function resolvePrefetchDepth(
  metadataWarmStarted: boolean,
  hasCacheStats: boolean,
  frameBudgetMs: number | null,
  isShadowPrefetch: boolean
): number {
  return metadataWarmStarted && hasCacheStats && frameBudgetMs === null && !isShadowPrefetch
    ? MAX_PREFETCH_LEVELS
    : 1;
}

function readPrefetchHeadroom(loader: GSplatsSpatialIndexLoader | undefined): number | null {
  if (!loader) return null;
  const readStats = loader.getPrefetchCacheStats;
  if (typeof readStats !== 'function') return null;
  const stats = readStats.call(loader);
  if (!stats) return null;
  return Math.max(0, (stats.maxSize ?? 0) - stats.size);
}

async function selectPrefetchLevels(
  lodLoaders: readonly GSplatsSpatialIndexLoader[],
  firstLevel: number,
  nLods: number,
  maxLevels: number,
  headroom: number,
  viewState: GSplatsViewState
): Promise<number[]> {
  const levels: number[] = [];
  let reservedBytes = 0;
  const stopLevel = Math.min(nLods, firstLevel + maxLevels);
  const candidates = lodLoaders.slice(firstLevel, stopLevel);
  const estimates = await Promise.all(
    candidates.map((loader) => loader.estimatePrefetchBytes(viewState))
  );
  for (let index = 0; index < candidates.length; index++) {
    const level = firstLevel + index;
    const estimate = estimates[index];
    reservedBytes +=
      Number.isFinite(estimate) && estimate > 0 ? estimate : Number.POSITIVE_INFINITY;
    if (level > firstLevel && reservedBytes > headroom) break;
    levels.push(level);
  }
  return levels;
}

/**
 * Concatenate multiple LoadedGSplatsData into one.
 * Allocates new arrays sized for the total splat count and copies data.
 */
export function concatenateGSplatsData(parts: LoadedGSplatsData[]): LoadedGSplatsData {
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
    // GUARD (belt-and-braces): a ladder payload must never publish the picking
    // index space. `parts[0]` is either `additive_0` or a folded cumulative
    // prefix, but its ranges are still not in the parent node's index space, and
    // `parts.length === 1` is not "unladdered" — this
    // loader only exists for `n_additive_sublods` nodes, so it is the first-paint
    // state of EVERY ladder. Passing them through would make hover report an
    // additive_0 index while only LOD 0 is resident and the raw slot once a
    // second level lands. `createProgressiveGSplatsLoader` now also clears
    // `has_labels` / `has_image_labels` / `has_keys` on each sub-LOD's attrs, so
    // `ranges` is normally never published at all; this keeps the invariant true
    // whatever attrs a sub-LOD carries. (A gsplat ladder carries no labels/keys
    // at any level; the Points / Lines ladders put one union CSR per supported
    // channel on the parent — #1422.)
    const only = parts[0];
    if (only.ranges === undefined) return only;
    const stripped: LoadedGSplatsData = { ...only };
    delete stripped.ranges;
    return stripped;
  }

  const ndim = parts[0].ndim;
  // Same fail-fast contract as the dtype/layout checks below, one field
  // over: `ndim` strides positions AND sizes the Cholesky blocks, so a
  // corrupted store whose sub-LODs disagree on ndim would pass the dtype
  // checks (all Float32) yet mis-stride every splat after the first part —
  // silent corruption. Dimensionality is per-dataset; a mismatch is
  // malformed data.
  for (const part of parts) {
    if (part.ndim !== ndim) {
      throw new Error(
        'concatenateGSplatsData: mixed dimensionality across LOD levels ' +
          `(ndim ${part.ndim} vs ${ndim}) — ladder levels must share the ` +
          'dataset dimensionality.'
      );
    }
  }
  // Per-part color-layout check, distinct from the cross-level mismatch
  // guarded below: those throws catch LODs that DISAGREE on dtype/layout,
  // but a single part can carry an RGBA buffer while OMITTING
  // `colorComponents: 4` (it defaults to 3). That satisfies the downstream
  // `count·3` minimum yet mis-strides every splat after the first — silent
  // corruption. Assert each part's raw length against its own declared
  // layout before allocation so an omitted declaration throws loudly here.
  // Names the offending level (concat-helpers' convention) so a corrupt
  // store is diagnosable without a debugger.
  for (const [levelIdx, part] of parts.entries()) {
    assertColorLayout(
      part.colors,
      part.splatCount,
      part.colorComponents ?? 3,
      `concatenateGSplatsData (LOD level ${levelIdx})`
    );
  }
  const totalSplats = parts.reduce((sum, p) => sum + p.splatCount, 0);
  const cholSize = (ndim * (ndim + 1)) / 2;
  const count = (p: LoadedGSplatsData) => p.splatCount;

  // Required per-splat fields via the shared helpers (dtype preserved).
  const positions = concatRequiredField(parts, (p) => p.positions, count, ndim, 'positions');
  const amplitudes = concatRequiredField(parts, (p) => p.amplitudes, count, 1, 'amplitudes');
  const choleskyFactors = concatRequiredField(
    parts,
    (p) => p.choleskyFactors,
    count,
    cholSize,
    'choleskyFactors'
  );
  const labelParts = parts.filter((part) => part.labelIndices !== undefined);
  if (labelParts.length !== 0 && labelParts.length !== parts.length) {
    throw new Error('concatenateGSplatsData: mixed label_ids presence across LOD levels');
  }
  const labelVocabulary = labelParts[0]?.labelVocabulary;
  if (
    labelVocabulary &&
    labelParts.some(
      (part) => JSON.stringify(part.labelVocabulary) !== JSON.stringify(labelVocabulary)
    )
  ) {
    throw new Error('concatenateGSplatsData: mixed label_vocabulary across LOD levels');
  }
  const labelIndices = labelVocabulary
    ? concatRequiredField(parts, (part) => part.labelIndices!, count, 1, 'labelIndices')
    : undefined;

  // Bespoke: colors fill missing LODs with white (per-dtype fill value).
  // Color layout (3 = RGB, 4 = RGBA — the 4th channel is per-splat opacity)
  // is a property of the dataset, uniform across its LODs; take it from the
  // first LOD that carries colors and stride every copy/fill by it (a
  // hardcoded 3 would truncate + misalign an RGBA additive ladder — exactly
  // what imported 3DGS scenes become after `gsplat lod`).
  const firstWithColors = parts.find((p) => p.colors !== null);
  const colorK: 3 | 4 = firstWithColors?.colorComponents ?? 3;
  let colors: Float32Array | Uint8Array | Uint16Array | null = null;
  if (firstWithColors?.colors) {
    if (firstWithColors.colors instanceof Uint8Array) {
      colors = new Uint8Array(totalSplats * colorK);
    } else if (firstWithColors.colors instanceof Uint16Array) {
      colors = new Uint16Array(totalSplats * colorK);
    } else {
      colors = new Float32Array(totalSplats * colorK);
    }
  }

  let offset = 0;
  for (const [levelIdx, part] of parts.entries()) {
    if (colors && part.colors) {
      // LADDER-DTYPE CONTRACT (see concat-helpers.ts): `set` converts by
      // VALUE, not semantics — a Float32 (0..1) level written into a Uint8
      // (0..255) merge truncates to garbage, and the reverse writes 255×
      // values. The writer emits one color dtype per ladder; fail fast.
      // Messages name the offending level (concat-helpers' convention) so a
      // corrupt store is diagnosable without a debugger.
      if (part.colors.constructor !== colors.constructor) {
        throw new Error(
          'concatenateGSplatsData: mixed color dtypes across LOD levels ' +
            `(level ${levelIdx}: ${part.colors.constructor.name} vs ` +
            `${colors.constructor.name}) — ladder levels must share each ` +
            "field's dtype."
        );
      }
      // Same contract for the color LAYOUT: `colorK` strides every copy, so
      // an RGBA level inside an RGB ladder (same ctor — invisible to the
      // dtype check above) would land at the wrong stride and silently
      // corrupt every splat after it. Layout is per-dataset, uniform across
      // its LODs; a mismatch is malformed data.
      if ((part.colorComponents ?? 3) !== colorK) {
        throw new Error(
          'concatenateGSplatsData: mixed color layouts across LOD levels ' +
            `(level ${levelIdx}: ${part.colorComponents ?? 3} vs ${colorK} ` +
            'components) — ladder levels must share the color layout ' +
            '(RGB vs RGBA).'
        );
      }
      colors.set(part.colors, offset * colorK);
    } else if (colors && !part.colors) {
      // Fill with white (1.0 for Float32, 255 for Uint8, 65535 for Uint16).
      // Alpha fills opaque (the per-element-opacity identity) via the same
      // full-scale fill value.
      const fillValue =
        colors instanceof Uint8Array ? 255 : colors instanceof Uint16Array ? 65535 : 1.0;
      for (let i = 0; i < part.splatCount * colorK; i++) {
        colors[offset * colorK + i] = fillValue;
      }
    }

    offset += part.splatCount;
  }

  // `ranges` is DELIBERATELY not concatenated — see the single-part branch.
  return {
    positions,
    amplitudes,
    choleskyFactors,
    colors,
    colorComponents: colorK,
    labelIndices,
    labelVocabulary,
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
  private _loadedLODCount = 0;
  private lastViewState: GSplatsViewState | null = null;
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
    result: LoadedGSplatsData;
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
  // Node path (SliceCache namespace) + the shared SliceCache, if enabled.
  private readonly path: string;
  private readonly sliceCache: SliceCache | null;
  private _metadataWarmStarted = false;
  private _prefetchController: AbortController | null = null;
  private readonly _prefetchingLevels = new Set<number>();
  // Per-tick LOD time budget (ms) from the CURRENT updateView call during
  // dimension-animation playback; null outside playback. A per-pass
  // directive (never part of lastViewState / viewStatesEqual / cache keys):
  // caps how many sub-LODs the streaming loop loads this pass and, while
  // set, makes `hasMoreLODs` read false so no background refinement runs
  // between animation ticks and the budgeted prefix commits as
  // "complete for playback" (display gate accepts it without holding).
  private _frameBudgetMs: number | null = null;

  constructor(
    lodLoaders: GSplatsSpatialIndexLoader[],
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
      'gsplats-spatial-index'
    );
    this.energyTable =
      energyTable && energyTable.length === nLods && energyTable.every((e) => typeof e === 'number')
        ? (energyTable as number[])
        : null;
  }

  /**
   * Whether there are more LOD levels to load for the current view state.
   */
  get hasMoreLODs(): boolean {
    // A disposed loader has work-state cleared; report no further work so a
    // refinement loop holding a stale reference stops instead of indexing
    // into the now-empty lodLoaders.
    if (this._disposed) return false;
    if (this._retryFoldedPass) return true;
    // While a playback frame budget is active, the budgeted prefix IS the
    // target: report no further work so the refinement scheduler stays idle
    // between animation ticks and the commit stamps the prefix as complete
    // (the display gate then accepts it instead of holding the previous
    // frame). The next budget-free updateView (pause re-trigger, scrub)
    // clears the budget and refinement resumes from the prefix.
    if (this._frameBudgetMs !== null) return false;
    return this._loadedLODCount < this.nLods;
  }

  /** Number of LOD levels currently loaded. */
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
      elementCount: this.loadedLODs.reduce((s, d) => s + d.splatCount, 0),
      // 4 RGBA32F texels/splat. See LadderResidency — the payload alone is not
      // the node's footprint, and the ratio differs per geometry.
      bytesPerElement: SPLAT_FLOATS_PER_SPLAT * Float32Array.BYTES_PER_ELEMENT,
    };
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

  /** Total number of LOD levels. */
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
      this._loadedLODCount = 0;
      this._restoredFullLadderAtPassStart = false;
      this._retryFoldedPass = false;
      if (this.lastViewState) deleteLadder(this.sliceCache, this.path, this.lastViewState);
      this._concatCache = null;
      return plan.dropped;
    }
    this.loadedLODs.length = this._payloadsAtPassStart;
    this._loadedLODCount = plan.keep;
    this._retryFoldedPass = false;
    if (this.lastViewState) deleteLadder(this.sliceCache, this.path, this.lastViewState);
    if (plan.invalidateConcatCache) this._concatCache = null;
    return plan.dropped;
  }

  /**
   * Whether the most recently streamed LOD level was fully cache-resident
   * (no chunk fetched from the network). Drives the monitor's
   * cached-vs-streaming residency indicator. Defaults to `true` before any
   * load.
   */
  get lastAllResident(): boolean {
    return this._lastAllResident;
  }

  /**
   * Load gsplats data. On first call, loads LOD 0.
   * Delegates to updateView().
   */
  async loadGSplats(
    viewState: GSplatsViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedGSplatsData> {
    return this.updateView(viewState, session, signal);
  }

  /**
   * Load as many LODs as are cache-warm for the given view state.
   *
   * If the view state changed since the last call, resets and starts from LOD 0.
   * Otherwise, continues loading from where it left off.
   *
   * After returning, prefetches later unloaded LODs in the background.
   */
  async updateView(
    viewState: GSplatsViewState,
    session?: UpdateSession,
    signal?: AbortSignal,
    residencyAllowanceBytes?: number
  ): Promise<LoadedGSplatsData> {
    // Record the per-pass playback budget FIRST (before the restore branch:
    // a pause re-trigger arrives with the SAME view state — it must still
    // clear the budget so refinement can resume). Deadline is measured from
    // pass start so slow levels consume the budget too.
    this._frameBudgetMs = viewState.frameBudgetMs ?? null;
    this._retryFoldedPass = false;
    const budgetDeadline =
      this._frameBudgetMs !== null ? performance.now() + this._frameBudgetMs : null;

    // A background prefetch (shadow) pass only warms the SliceCache — the
    // SlicePrefetcher discards its return value. So every return below hands
    // back a cheap empty result instead of running `concatenateMemoized`: that
    // O(N) main-thread concat is pure waste for the shadow and, as the ladder
    // deepens across loops, would stall foreground frames.
    const isPrefetch = viewState.prefetch === true;
    const finish = (): LoadedGSplatsData =>
      isPrefetch ? concatenateGSplatsData([]) : this.concatenateMemoized(session);

    // Reset if view state changed. Before discarding the ladder, try the
    // SliceCache: a full-ladder snapshot for this exact view lets us restore
    // `loadedLODs` outright (so `hasMoreLODs` reads false and the refinement
    // loop never re-streams), skipping the whole load+decode.
    if (!this.lastViewState || !viewStatesEqual(viewState, this.lastViewState)) {
      if (this.lastViewState) this.cancelLookaheadPrefetch();
      // DEPARTURE store: snapshot the outgoing view's partial ladder under
      // the OUTGOING key before discarding — scrub-back stays warm even when
      // ladders never complete between navigations. Mirrors Points/Lines.
      if (this.lastViewState && this._loadedLODCount > 0) {
        storeLadder(this.sliceCache, this.path, this.lastViewState, this.loadedLODs, {
          scan: this._frameBudgetMs !== null,
          pin: viewState.prefetch === true,
          ladderDepth: this._loadedLODCount,
          totalLODCount: this.nLods,
        });
      }
      const restored = restoreLadderSnapshot<LoadedGSplatsData>(
        this.sliceCache,
        this.path,
        viewState,
        this.nLods
      );
      // Shallow-copy the CONTAINER: the streaming loop below pushes further
      // levels into loadedLODs and must never mutate the cache's payload
      // array (the elements stay shared read-only — store deep-clones).
      this.loadedLODs = restored ? [...restored.lods] : [];
      this._loadedLODCount = restored?.depth ?? 0;
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
        // FULL ladder: nothing left to load — short-circuit the whole pass.
        // A PREFIX (stored while a playback budget capped a previous pass)
        // falls through to the loop instead: loading resumes from
        // startLevel = prefix length — within this pass's budget during
        // play, or to completion when idle.
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

    if (this._initialLoadDone && this._frameBudgetMs === null && !isPrefetch) {
      this.warmRemainingLODMetadata();
    }

    // Stream the LOD ladder under the shared streaming policy (see
    // `streaming-policy.ts`): `playback` streams cache-resident levels after an
    // empty or restored prefix while budget remains; `prefetch`
    // deepens toward the full decoded ladder (bounded by the pass budget +
    // abort); `refine` stops at the first cold/slow level.
    const pass = classifyStreamingPass(budgetDeadline !== null, isPrefetch);
    const startLevel = this._loadedLODCount;
    const residentBytesAtPassStart = ladderResidentBytes(this.ladderResidency());
    this._levelsAtPassStart = startLevel;
    this._payloadsAtPassStart = this.loadedLODs.length;
    this._restoredFullLadderAtPassStart = false;

    for (let level = startLevel; level < this.nLods; level++) {
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
        ({ allResident }) => `additive:gsplats:level:${level}:${allResident ? 'resident' : 'miss'}`,
        `additive:gsplats:level:${level}:aborted`,
        () => this.lodLoaders[level].updateViewWithResidency(viewState, session, signal)
      );
      const elapsed = performance.now() - t0;

      this.loadedLODs.push(lodData);
      this._loadedLODCount++;
      this._lastAllResident = allResident;

      if (!this._initialLoadDone) {
        log.custom(
          LogEmoji.BROADCAST,
          Modules.GSPLATS_SPATIAL_INDEX_LOADER,
          `LOD ${level}/${this.nLods - 1}: ${lodData.splatCount} splats (${elapsed.toFixed(1)}ms${allResident ? '' : ', miss'})`
        );
      }

      // NEVER break out because a level came back empty (#1456). It is a
      // tempting optimization — this loop used to latch a terminal "empty
      // ladder" on an empty LOD 0 and stop — but it is wrong here: this loader
      // is constructed only for ADDITIVE ladders
      // (`createProgressiveGSplatsLoader` iterates the `additive_<i>`
      // subgroups), whose levels are DISJOINT increments of one permutation,
      // not coarse-to-fine resamplings of the same elements. They are therefore
      // NOT spatially coextensive: LOD 0 is a small SUBSET of the node (a few
      // thousand splats under `-b stream:C` / `--target-ms`, the recommended
      // ladder shape), so a hidden-dimension slice that none of ITS members
      // lands on says nothing whatever about levels 1..n-1, which may hold
      // plenty of splats right there. Stopping here rendered such a slice
      // permanently blank. The same reasoning forbids inferring anything from a
      // restored cache PREFIX whose LOD 0 is empty.

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

    // Log LOD loading summary (compact, always shown for progressive loaders)
    const totalSplats = this.loadedLODs.reduce((s, d) => s + d.splatCount, 0);
    if (this._loadedLODCount < this.nLods) {
      log.info(
        Modules.GSPLATS_SPATIAL_INDEX_LOADER,
        `Progressive: ${this._loadedLODCount}/${this.nLods} LODs loaded (${totalSplats} splats) — refining`
      );
    } else if (startLevel < this.nLods) {
      // Only log "all loaded" when we actually loaded something new this call
      log.info(
        Modules.GSPLATS_SPATIAL_INDEX_LOADER,
        `Progressive: ${this.nLods}/${this.nLods} LODs loaded (${totalSplats} splats) — complete`
      );
    }

    // Fire-and-forget: prefetch later unloaded LODs to warm cache.
    void this.prefetchNextLODs(viewState);

    // Snapshot into the SliceCache (upgrade-if-longer): full ladders always
    // (instant revisit restore); PREFIXES only while a playback budget is
    // active — each playback loop then restores the prefix instantly and
    // deepens it with the leftover budget, converging to full ladders.
    // Gating prefixes on the budget keeps the non-play cost profile (a
    // store per refinement pass would clone O(N²) bytes per slice).
    const result = finish();
    if (this._loadedLODCount === this.nLods || this._frameBudgetMs !== null) {
      storeLadder(this.sliceCache, this.path, viewState, this.loadedLODs, {
        scan: this._frameBudgetMs !== null,
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
   * commit extends nothing.
   */
  private concatenateMemoized(session?: UpdateSession): LoadedGSplatsData {
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
      const result = concatenateGSplatsData(this.loadedLODs);
      setPrefixParent(result, prevMemo);
      for (let level = 0; level < this._loadedLODCount; level++) {
        this.lodLoaders[level]?.releaseAccumulator();
      }
      this.loadedLODs = [result];
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

  /**
   * Prefetch unloaded LOD chunks into cache, bounded by L0 headroom.
   *
   * This is fire-and-forget: the prefetched chunks land in the L0/L1 cache
   * and become fast cache hits on the next updateView() call.
   *
   * Uses prefetchChunks() which performs the spatial index query and zarr
   * get() calls (populating the cache) WITHOUT allocating full-size output
   * buffers or running the accumulator — avoiding wasted memory.
   */
  private async prefetchNextLODs(viewState: GSplatsViewState): Promise<void> {
    // Same teardown race as the streaming loop: a dispose() between the
    // awaited level and this fire-and-forget clears `lodLoaders`, and
    // indexing it would TypeError before the .catch can swallow anything.
    if (this._disposed) return;
    const firstLevel = this._loadedLODCount;
    if (firstLevel >= this.nLods) return;

    const headroom = readPrefetchHeadroom(this.lodLoaders[0]);
    const controller = (this._prefetchController ??= new AbortController());
    const maxLevels = resolvePrefetchDepth(
      this._metadataWarmStarted,
      headroom !== null,
      this._frameBudgetMs,
      viewState.prefetch === true
    );
    const levels = await selectPrefetchLevels(
      this.lodLoaders,
      firstLevel,
      this.nLods,
      maxLevels,
      headroom ?? 0,
      viewState
    );
    if (controller.signal.aborted || this._disposed) return;

    for (const level of levels) {
      if (this._prefetchingLevels.has(level)) continue;

      this._prefetchingLevels.add(level);
      this.lodLoaders[level]
        .prefetchChunks(viewState, controller.signal)
        .catch(() => {
          // Ignore errors from speculative prefetch (network failures, aborts).
        })
        .finally(() => {
          if (this._prefetchController === controller) {
            this._prefetchingLevels.delete(level);
          }
        });
    }
  }

  private warmRemainingLODMetadata(): void {
    if (this._metadataWarmStarted || this._disposed) return;
    this._metadataWarmStarted = true;
    for (let level = 1; level < this.nLods; level++) {
      const ensureInitialized = this.lodLoaders[level].ensureInitialized;
      if (typeof ensureInitialized !== 'function') continue;
      void ensureInitialized.call(this.lodLoaders[level]).catch(() => {
        // Demand loading surfaces malformed metadata; warming stays best-effort.
      });
    }
  }

  private cancelLookaheadPrefetch(): void {
    this._prefetchController?.abort();
    this._prefetchController = null;
    this._prefetchingLevels.clear();
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

  /**
   * Clean up all LOD loaders.
   */
  dispose(): void {
    this._disposed = true;
    this.cancelLookaheadPrefetch();
    for (const loader of this.lodLoaders) {
      loader.dispose();
    }
    this.lodLoaders = [];
    this.loadedLODs = [];
    this._loadedLODCount = 0;
    this._retryFoldedPass = false;
    this.lastViewState = null;
    this._concatCache = null;
  }
}
