/**
 * The shared engine of the additive-ladder progressive loaders.
 *
 * Points, Lines and GSplats each wrap N spatial-index loaders — one per
 * `additive_<i>` subgroup, the rungs of one additive ladder — and stream them
 * per view under the same lifecycle. That lifecycle used to be copy-maintained
 * three times, and the richer behaviour the performance work added landed
 * mostly in GSplats. It lives here once; a geometry supplies only what differs
 * ({@link LadderGeometry}: how payloads concatenate, how elements are counted,
 * the log vocabulary) and, for GSplats, a deeper lookahead planner.
 *
 * One `updateView` pass:
 *
 * 1. **Directives.** Record the pass's playback budget and pinned ladder depth
 *    first — a pause re-trigger arrives with the SAME view and must still clear
 *    them.
 * 2. **View change.** Abort the old view's speculative reads (keeping a
 *    read-ahead of exactly the slice being entered, which the new pass joins),
 *    store the OUTGOING view's partial ladder under the outgoing key (departure
 *    store, so scrub-back stays warm), adopt an in-flight shadow (SlicePrefetcher)
 *    store of the incoming view, then restore a cached ladder or reset. A
 *    restored FULL ladder short-circuits the pass.
 * 3. **Stream.** Load rungs under the shared streaming policy
 *    (`streaming-policy.ts`): `playback` within its budget, `prefetch` toward
 *    the full ladder, `refine` until the first cold/slow rung, `pinned` exactly
 *    the pinned depth — whose rungs are all started up front under per-rung
 *    child signals so a brake cancels the remainder and releases what it
 *    decoded. While a rung loads, the next rung's spatial index is warmed at
 *    refinement priority (B6). A rung coming back empty never stops the loop
 *    (#1456): additive rungs are disjoint subsets, not coarse resamplings.
 * 4. **Close.** Log the summary at verbose, schedule the signal-linked
 *    speculative lookahead of the next rung and the B5 ±1-slice read-ahead of
 *    rung 0, then store the ladder (full ladders always; prefixes while a
 *    playback budget or pinned depth is active) unless the pass was torn down.
 *
 * The concatenated result is memoized on (reset generation, logical rung
 * count): an unchanged view with no new rung returns the SAME reference, which
 * the commit pipeline uses to skip no-op re-commits, and each fresh result is
 * stamped with its prefix-lineage parent for the append fast path.
 *
 * @module data/loaders/progressive/additive-ladder-core
 */

import type { ViewState } from '../../data-loader-types';
import type { UpdateSession } from '../../../profiling/update-profiler';
import type { SliceCache } from '../../../cache/slice-cache';
import { setSliceCacheOrigin, type SliceCacheOrigin } from '../../../cache/slice-cache-origin';
import { tagSignalOrigin } from '../../../cache/decompressed-chunk-cache/decode-origin';
import { setPrefixParent } from '../../../types/prefix-lineage';
import { tagSignalPriority, type FetchPriority } from '../../../utils/fetch-concurrency';
import { getErrorMessage } from '../../../utils/format-error';
import { log, LogEmoji } from '../../../utils/log';
import { config } from '../../../config';
import { isAbortError } from '../abort-error';
import { timeLodStageWithResult } from '../../scene-loader/lod-load-stats';
import {
  ladderResidentBytes,
  type LadderResidency,
} from '../../scene-loader/progressive/residency-budget';
import { createChildController, type ChildController } from './child-signal';
import { createLookaheadController } from './lookahead-signal';
import { planLadderRollback } from './pass-rollback';
import {
  awaitShadowStore,
  deleteLadder,
  measureLodBytes,
  restoreLadderSnapshot,
  storeLadder,
} from './slice-cache-helper';
import {
  classifyStreamingPass,
  resolveLadderDepth,
  shouldStopAfterLevel,
  shouldStopBeforeLevel,
  type StreamingPassKind,
} from './streaming-policy';
import { viewStatesEqual } from './view-state-equal';

/** One rung of the ladder: a spatial-index loader over one `additive_<i>` subgroup. */
export interface LadderRung<TData> {
  updateViewWithResidency(
    viewState: ViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<{ data: TData; allResident: boolean }>;
  /** Warm the chunks `viewState` touches (planned `ranges` when supplied). */
  prefetchChunks(viewState: ViewState, signal?: AbortSignal, ranges?: never): Promise<void>;
  prefetchChunkBoundary(
    current: ViewState,
    predicted: ViewState,
    signal?: AbortSignal
  ): Promise<void>;
  /** Initialize the rung's metadata (spatial index) once, at `priority`. */
  ensureInitialized(priority?: FetchPriority): Promise<void>;
  /** Drop pooled decode buffers once a parent has copied the rung's payload. */
  releaseAccumulator(): void;
  dispose(): void;
}

/** What differs per geometry. */
export interface LadderGeometry<TData> {
  /** Timing-key segment: `additive:<kind>:level:<n>:…`. */
  readonly kind: 'points' | 'lines' | 'gsplats';
  /** Log module name. */
  readonly module: string;
  /** Summary-log label (`Progressive <label>: …`). */
  readonly label: string;
  /** Element noun for the logs (`points`, `segments`, `splats`). */
  readonly unit: string;
  /** Subject of the speculative-failure warnings (`Points`, `Lines`, `GSplat`). */
  readonly warnPrefix: string;
  /** Elements a payload holds (the drawn primitive). */
  countOf(data: TData): number;
  /**
   * Concatenate retained payloads into one cumulative payload. Payload `i`
   * spans logical rungs `[payloadLevelStarts[i], payloadLevelStarts[i + 1] ??
   * loadedLevels)`; folding collapses them to `[0]`.
   */
  concat(
    parts: TData[],
    layout: { payloadLevelStarts: readonly number[]; loadedLevels: number }
  ): TData;
  /** The cheap empty result a shadow (prefetch) pass hands back. */
  empty(): TData;
  /** GPU bytes per element beside the payload (see `LadderResidency`). */
  readonly bytesPerElement: number;
  /** Stamp a result built from a restored snapshot alone with its origin. */
  readonly stampsRestoredOrigin: boolean;
}

/** What a {@link LookaheadPlanner} is handed once per eligible pass. */
export interface LookaheadRequest<TRung> {
  readonly rungs: readonly TRung[];
  /** The first rung not yet loaded — the lookahead's floor. */
  readonly firstLevel: number;
  readonly viewState: ViewState;
  /** Whether every rung's metadata warm has been started (deep plans need it). */
  readonly metadataWarm: boolean;
  readonly controller: AbortController;
  /** Start warming `level` (deduped per view), optionally with planned ranges. */
  readonly start: (level: number, plan?: Promise<{ ranges?: readonly unknown[] }>) => void;
  /** False once the loader is disposed or the controller aborted. */
  readonly isLive: () => boolean;
}

/** Chooses which rungs the speculative lookahead warms (default: the next one). */
export interface LookaheadPlanner<TRung> {
  plan(request: LookaheadRequest<TRung>): void;
  /** The view changed or the loader was disposed: drop any planning state. */
  reset(): void;
}

/** The default planner: warm the next unloaded rung only. */
const NEXT_RUNG_PLANNER: LookaheadPlanner<unknown> = {
  plan: (request) => request.start(request.firstLevel),
  reset: () => undefined,
};

export interface AdditiveLadderOptions<TRung> {
  /** Per-rung cumulative energy fractions e(k) (all-or-nothing). */
  energyTable?: ReadonlyArray<number | null | undefined>;
  sliceCache?: SliceCache | null;
  planner?: LookaheadPlanner<TRung>;
}

/** Hidden dims whose slice moved between two views (read-ahead axes). */
function scrubbedHiddenDims(previous: ViewState, next: ViewState): number[] {
  const dims: number[] = [];
  for (let d = 0; d < next.slicePosition.length; d++) {
    if (next.displayDims.includes(d)) continue;
    if (previous.slicePosition[d] !== next.slicePosition[d]) dims.push(d);
  }
  return dims;
}

/** The in-range slice values one step either side of dimension `d`'s current one. */
function neighbourValues(viewState: ViewState, d: number): number[] {
  const meta = viewState.dimensions?.[d];
  const step = meta?.step;
  if (step === undefined || !(step > 0)) return [];
  const at = viewState.slicePosition[d];
  const range = meta?.range;
  return [at - step, at + step].filter((v) => !range || (v >= range[0] && v <= range[1]));
}

/** One step either way along each of `dims` (stepped dims only, inside their range). */
function neighbourSlices(viewState: ViewState, dims: readonly number[]): ViewState[] {
  const out: ViewState[] = [];
  for (const d of dims) {
    for (const value of neighbourValues(viewState, d)) {
      const slicePosition = [...viewState.slicePosition];
      slicePosition[d] = value;
      out.push({ ...viewState, slicePosition });
    }
  }
  return out;
}

/** A best-effort speculative read failed: abort is routine, anything else is logged. */
export function reportSpeculativeFailure(module: string, action: string, error: unknown): void {
  if (isAbortError(error)) return;
  log.warning(module, `${action}: ${getErrorMessage(error)}`);
}

/** One rung a pinned pass started up front, with the controller that can cancel it. */
interface PinnedRungLoad<TData> {
  readonly level: number;
  readonly load: Promise<{ data: TData; allResident: boolean }>;
  readonly child: ChildController;
}

/** Per-pass constants of the streaming loop. */
interface StreamPass {
  readonly kind: StreamingPassKind;
  readonly viewState: ViewState;
  readonly session: UpdateSession | undefined;
  readonly signal: AbortSignal | undefined;
  readonly isPrefetch: boolean;
  readonly budgetDeadline: number | null;
  readonly startLevel: number;
  readonly targetLevels: number;
  readonly residentBytesAtPassStart: number;
  readonly residencyAllowanceBytes: number | undefined;
}

export class AdditiveLadderCore<TData extends object, TRung extends LadderRung<TData>> {
  private rungs: TRung[];
  private readonly nLods: number;
  private readonly path: string;
  private readonly geometry: LadderGeometry<TData>;
  private readonly sliceCache: SliceCache | null;
  private readonly planner: LookaheadPlanner<TRung>;
  // Non-null only when EVERY rung carries a stamp (a partially stamped ladder
  // reads as unstamped — never blend stamped and guessed entries).
  private readonly energyTable: readonly number[] | null;

  private loadedLODs: TData[] = [];
  private payloadLevelStarts: number[] = [];
  private loadedCount = 0;
  private lastViewState: ViewState | null = null;
  private initialLoadDone = false;
  private lastResident = true;
  private disposed = false;
  // Memoized concatenation, keyed on (resetGeneration, logical rung count).
  private resetGeneration = 0;
  private concatCache: { generation: number; lodCount: number; result: TData } | null = null;
  // Logical depth and retained payload count as THIS pass found them: rollback
  // needs both to tell an intact append from an already-folded result
  // (see `pass-rollback.ts`).
  private levelsAtPassStart = 0;
  private payloadsAtPassStart = 0;
  private restoredFullLadderAtPassStart = false;
  // A completed pass can still fail after loading (projection, commit): keep it
  // schedulable for one retry even though the cursor is full.
  private retryFoldedPass = false;
  private restoredOrigin: SliceCacheOrigin | null = null;
  private metadataWarmStarted = false;
  private lookahead: AbortController | null = null;
  private readonly lookaheadLevels = new Set<number>();
  /**
   * Speculative ±1-slice read-ahead of rung 0 (B5), one controller per
   * neighbour slice. A view change aborts every one EXCEPT the slice it moved
   * onto: that warm is what the new pass's rung-0 reads join.
   */
  private readAheads: Array<{ view: ViewState; controller: AbortController }> = [];
  private scrubDims: number[] = [];
  // Per-pass directives (never part of lastViewState / cache keys).
  private frameBudgetMs: number | null = null;
  private ladderDepth: number | null = null;

  constructor(
    rungs: TRung[],
    nLods: number,
    path: string,
    geometry: LadderGeometry<TData>,
    options: AdditiveLadderOptions<TRung> = {}
  ) {
    this.rungs = rungs;
    this.nLods = nLods;
    this.path = path;
    this.geometry = geometry;
    this.sliceCache = options.sliceCache ?? null;
    this.planner = options.planner ?? (NEXT_RUNG_PLANNER as LookaheadPlanner<TRung>);
    const table = options.energyTable;
    this.energyTable =
      table && table.length === nLods && table.every((e) => typeof e === 'number')
        ? (table as number[])
        : null;
  }

  /** The rung loaders (empty once disposed) — for the monitor adapter; do not mutate. */
  get rungLoaders(): TRung[] {
    return this.rungs;
  }

  /** Whether more rungs remain to load for the current view. */
  get hasMoreLODs(): boolean {
    // A disposed loader reports no further work, so a refinement loop holding
    // a stale reference stops instead of indexing the now-empty rungs.
    if (this.disposed) return false;
    if (this.retryFoldedPass) return true;
    // A playback budget or pinned depth makes the prefix the target: no
    // background refinement between ticks; the commit stamps it complete.
    if (this.frameBudgetMs !== null || this.ladderDepth !== null) return false;
    return this.loadedCount < this.nLods;
  }

  get loadedLODCount(): number {
    return this.loadedCount;
  }

  get totalLODCount(): number {
    return this.nLods;
  }

  /** Whether the most recently streamed rung was fully cache-resident. */
  get lastAllResident(): boolean {
    return this.lastResident;
  }

  /**
   * Cumulative energy fraction e(k) of the loaded prefix (`null` without
   * energy stamps, `0` before any rung loads) — read at commit time for the
   * display gate's energy-threshold upgrade release.
   */
  get committedEnergyFraction(): number | null {
    if (!this.energyTable) return null;
    const k = this.loadedCount;
    if (k === 0) return 0;
    return this.energyTable[Math.min(k, this.energyTable.length) - 1];
  }

  /**
   * Measured footprint of the loaded ladder for the sweep residency budget.
   * Rung count is the LOGICAL depth, not `loadedLODs.length` (1 once folded).
   */
  ladderResidency(): LadderResidency {
    return {
      residentBytes: measureLodBytes(this.loadedLODs),
      loadedRungs: this.loadedCount,
      elementCount: this.loadedLODs.reduce((s, d) => s + this.geometry.countOf(d), 0),
      bytesPerElement: this.geometry.bytesPerElement,
    };
  }

  /** Bytes held by the memoized cumulative payload (monitor memory). */
  concatMemoryBytes(): number {
    return this.concatCache ? measureLodBytes([this.concatCache.result]) : 0;
  }

  /**
   * Discard the rungs the current pass appended, restoring the prefix it
   * started from, so a failed commit's retry re-attempts the SAME prefix
   * (see `pass-rollback.ts`).
   *
   * @returns Rungs discarded (0 when the pass appended none).
   */
  rollbackToPassStart(): number {
    const plan = planLadderRollback({
      loadedLevelCount: this.loadedCount,
      levelsAtPassStart: this.levelsAtPassStart,
      concatCacheLodCount: this.concatCache?.lodCount ?? null,
      retainedPayloadCount: this.loadedLODs.length,
      payloadsAtPassStart: this.payloadsAtPassStart,
      restoredFullLadderAtPassStart: this.restoredFullLadderAtPassStart,
      totalLevelCount: this.nLods,
    });
    if (plan.action === 'none') return 0;
    if (plan.action === 'retry-folded-pass') {
      this.retryFoldedPass = true;
      return 0;
    }
    this.restoredOrigin = null;
    this.retryFoldedPass = false;
    if (this.lastViewState) deleteLadder(this.sliceCache, this.path, this.lastViewState);
    if (plan.action === 'unwind-restored-full') {
      this.loadedLODs = [];
      this.payloadLevelStarts = [];
      this.loadedCount = 0;
      this.restoredFullLadderAtPassStart = false;
      this.concatCache = null;
      return plan.dropped;
    }
    this.loadedLODs.length = this.payloadsAtPassStart;
    this.payloadLevelStarts.length = this.payloadsAtPassStart;
    this.loadedCount = plan.keep;
    if (plan.invalidateConcatCache) this.concatCache = null;
    return plan.dropped;
  }

  /** Stream the ladder for `viewState` (see the module notes for one pass). */
  async updateView(
    viewState: ViewState,
    session?: UpdateSession,
    signal?: AbortSignal,
    residencyAllowanceBytes?: number
  ): Promise<TData> {
    this.recordDirectives(viewState);
    const budgetDeadline =
      this.frameBudgetMs !== null ? performance.now() + this.frameBudgetMs : null;
    // A shadow (prefetch) pass only warms the SliceCache, and its caller drops
    // the return value: hand back a cheap empty result, not the O(N) concat.
    const isPrefetch = viewState.prefetch === true;
    const finish = (): TData =>
      isPrefetch ? this.geometry.empty() : this.concatenateMemoized(session);

    if (await this.syncView(viewState, signal, isPrefetch)) return finish();
    if (this.initialLoadDone && this.frameBudgetMs === null) this.warmRemainingMetadata();

    const startLevel = this.loadedCount;
    this.levelsAtPassStart = startLevel;
    this.payloadsAtPassStart = this.loadedLODs.length;
    this.restoredFullLadderAtPassStart = false;
    await this.streamRungs({
      kind: classifyStreamingPass(budgetDeadline !== null, isPrefetch, this.ladderDepth !== null),
      viewState,
      session,
      signal,
      isPrefetch,
      budgetDeadline,
      startLevel,
      targetLevels: this.ladderDepth ?? this.nLods,
      residentBytesAtPassStart: ladderResidentBytes(this.ladderResidency()),
      residencyAllowanceBytes,
    });
    if (!this.initialLoadDone && startLevel === 0) this.initialLoadDone = true;
    this.logSummary(startLevel);

    this.scheduleLookahead(viewState, signal);
    if (!isPrefetch) this.readAheadNeighbourSlices(viewState, signal);
    const result = finish();
    this.storeAfterPass(viewState, signal);
    return result;
  }

  /**
   * Predicted-view warm-up (`dispatchPredictivePrefetch`): warm the COARSE rung
   * of `viewState` — what the next pass's first commit reads.
   */
  prefetchChunks(viewState: ViewState, signal?: AbortSignal): Promise<void> {
    if (this.disposed || this.rungs.length === 0) return Promise.resolve();
    return this.rungs[0].prefetchChunks(viewState, signal);
  }

  /** {@link prefetchChunks} over a whole current → predicted transition. */
  prefetchChunkBoundary(
    current: ViewState,
    predicted: ViewState,
    signal?: AbortSignal
  ): Promise<void> {
    if (this.disposed || this.rungs.length === 0) return Promise.resolve();
    return this.rungs[0].prefetchChunkBoundary(current, predicted, signal);
  }

  dispose(): void {
    this.disposed = true;
    this.cancelSpeculation();
    for (const rung of this.rungs) rung.dispose();
    this.rungs = [];
    this.loadedLODs = [];
    this.payloadLevelStarts = [];
    this.restoredOrigin = null;
    this.loadedCount = 0;
    this.retryFoldedPass = false;
    this.lastViewState = null;
    this.concatCache = null;
  }

  // ---- one pass ------------------------------------------------------------

  /** Record the pass directives FIRST (a pause re-trigger reuses the view). */
  private recordDirectives(viewState: ViewState): void {
    this.frameBudgetMs = viewState.frameBudgetMs ?? null;
    this.ladderDepth = resolveLadderDepth(
      viewState.ladderDepth,
      this.nLods,
      this.energyTable,
      config.dimensionAnimation.playback.autoEnergyThreshold
    );
    this.retryFoldedPass = false;
  }

  /**
   * Bring the ladder to `viewState`: enter it when it differs from the last
   * view, else adopt a determinant-equal metadata refresh (the fresh
   * `dimensions` reference keeps later compares on the reference fast path).
   * True when a restored FULL ladder makes the pass a no-op.
   */
  private async syncView(
    viewState: ViewState,
    signal: AbortSignal | undefined,
    isPrefetch: boolean
  ): Promise<boolean> {
    const last = this.lastViewState;
    if (!last || !viewStatesEqual(viewState, last)) {
      return this.enterView(viewState, signal, isPrefetch);
    }
    if (last.dimensions !== viewState.dimensions) last.dimensions = viewState.dimensions;
    return false;
  }

  /** The view changed: departure store, shadow adoption, restore or reset. */
  private async enterView(
    viewState: ViewState,
    signal: AbortSignal | undefined,
    isPrefetch: boolean
  ): Promise<boolean> {
    if (this.lastViewState) {
      this.scrubDims = scrubbedHiddenDims(this.lastViewState, viewState);
      this.cancelSpeculation(viewState);
    } else {
      this.scrubDims = [];
    }
    // DEPARTURE store, under the OUTGOING key: scrubbing faster than ladders
    // complete would otherwise store nothing and make scrub-back cold.
    if (this.lastViewState && this.loadedCount > 0 && !(isPrefetch && this.tornDown(signal))) {
      this.store(this.lastViewState, viewState.prefetch === true);
    }
    // IN-FLIGHT ADOPTION: a shadow pass may be building this very slice; wait
    // (bounded, abort-aware) for its store instead of redoing the same work.
    const shadowStore = isPrefetch
      ? null
      : awaitShadowStore(this.sliceCache, this.path, viewState, signal);
    if (shadowStore) await shadowStore;
    return this.restoreOrReset(viewState);
  }

  /** Restore a cached ladder for `viewState` (or reset); true when it is FULL. */
  private restoreOrReset(viewState: ViewState): boolean {
    const restored = restoreLadderSnapshot<TData>(
      this.sliceCache,
      this.path,
      viewState,
      this.nLods
    );
    // Shallow-copy the CONTAINER: the loop pushes further rungs and must never
    // mutate the cache's payload array.
    this.loadedLODs = restored ? [...restored.lods] : [];
    this.restoredOrigin = restored?.origin ?? null;
    this.loadedCount = restored?.depth ?? 0;
    const foldedPrefixDepth = restored ? restored.depth - restored.lods.length + 1 : 0;
    this.payloadLevelStarts = this.loadedLODs.map((_, i) =>
      i === 0 ? 0 : foldedPrefixDepth + i - 1
    );
    // A restored full ladder has not been committed for this pass: a failed
    // concat/commit unwinds the whole snapshot rather than disabling retries.
    this.levelsAtPassStart = 0;
    this.payloadsAtPassStart = 0;
    this.restoredFullLadderAtPassStart = false;
    this.resetGeneration++;
    this.lastViewState = {
      displayDims: [...viewState.displayDims],
      slicePosition: [...viewState.slicePosition],
      tolerance: [...viewState.tolerance],
      dimensions: viewState.dimensions,
    };
    if (!restored) return false;
    this.initialLoadDone = true;
    this.lastResident = true;
    // A PREFIX falls through to the loop and resumes at its depth.
    if (restored.depth !== this.nLods) return false;
    this.restoredFullLadderAtPassStart = true;
    return true;
  }

  /** The streaming loop (see the module notes, step 3). */
  private async streamRungs(pass: StreamPass): Promise<void> {
    // A pinned pass takes every rung up to its depth whatever they cost, so
    // their fetches need not wait on one another: start them all now.
    const pinned = pass.kind === 'pinned' ? this.startRungLoads(pass) : null;
    try {
      for (let level = pass.startLevel; level < pass.targetLevels; level++) {
        // A dispose() racing the awaited rung clears `rungs`: stop streaming
        // rather than mis-count a teardown as a refinement failure.
        if (this.disposed) break;
        if (
          shouldStopBeforeLevel(
            pass.kind,
            level,
            pass.startLevel,
            performance.now(),
            pass.budgetDeadline
          )
        ) {
          break;
        }
        if (await this.loadOneRung(pass, level, pinned)) break;
      }
    } finally {
      // A rung started up front but never committed (brake, abort, throw)
      // must not keep fetching into an accumulator nothing will read (A20).
      if (pinned) this.discardUncommittedLoads(pinned);
    }
  }

  /** Load and append rung `level`; true when the policy stops the loop after it. */
  private async loadOneRung(
    pass: StreamPass,
    level: number,
    pinned: readonly PinnedRungLoad<TData>[] | null
  ): Promise<boolean> {
    // Rung level+1's index is fetched WHILE this rung loads (B6). Not under a
    // playback budget nor for a shadow pass.
    if (this.frameBudgetMs === null && !pass.isPrefetch) this.warmRungIndex(level + 1);
    const t0 = performance.now();
    const { data, allResident } = await (pinned?.[level - pass.startLevel]?.load ??
      this.loadRung(level, pass.viewState, pass.session, pass.signal));
    const elapsed = performance.now() - t0;
    this.loadedLODs.push(data);
    this.payloadLevelStarts.push(level);
    this.restoredOrigin = null;
    this.loadedCount++;
    this.lastResident = allResident;
    if (!this.initialLoadDone) this.logRung(level, data, elapsed, allResident);
    const additional = Math.max(
      0,
      ladderResidentBytes(this.ladderResidency()) - pass.residentBytesAtPassStart
    );
    return shouldStopAfterLevel(
      pass.kind,
      level,
      pass.startLevel,
      allResident,
      elapsed,
      additional,
      pass.residencyAllowanceBytes
    );
  }

  /** Load one rung for `viewState` (timed as `additive:<kind>:level:<n>`). */
  private loadRung(
    level: number,
    viewState: ViewState,
    session: UpdateSession | undefined,
    signal: AbortSignal | undefined
  ): Promise<{ data: TData; allResident: boolean }> {
    const kind = this.geometry.kind;
    return timeLodStageWithResult(
      ({ allResident }) => `additive:${kind}:level:${level}:${allResident ? 'resident' : 'miss'}`,
      `additive:${kind}:level:${level}:aborted`,
      () => this.rungs[level].updateViewWithResidency(viewState, session, signal)
    );
  }

  /**
   * Start a pinned pass's rungs concurrently, coarsest first, each under its
   * OWN child of the pass signal so the remainder can be cancelled without
   * cancelling the pass. A rung the loop never awaits must not surface as an
   * unhandled rejection, so each gets a no-op handler; the loop still awaits —
   * and rethrows — the original.
   */
  private startRungLoads(pass: StreamPass): PinnedRungLoad<TData>[] {
    const loads: PinnedRungLoad<TData>[] = [];
    for (let level = pass.startLevel; level < pass.targetLevels; level++) {
      const child = createChildController(pass.signal);
      const load = this.loadRung(level, pass.viewState, pass.session, child.controller.signal);
      void load.catch(() => {});
      loads.push({ level, load, child });
    }
    return loads;
  }

  /**
   * Cancel every pinned rung the loop did not commit and drop what its
   * accumulator already decoded: the brake refused those bytes. Committed
   * rungs keep their reads; every child is detached from the pass signal.
   */
  private discardUncommittedLoads(loads: readonly PinnedRungLoad<TData>[]): void {
    for (const { level, child } of loads) {
      if (level >= this.loadedCount) {
        child.controller.abort();
        this.rungs[level]?.releaseAccumulator();
      }
      child.detach();
    }
  }

  /** Store the ladder after a pass (full always; prefixes under a budget or pin). */
  private storeAfterPass(viewState: ViewState, signal: AbortSignal | undefined): void {
    // A pass aborted or disposed mid-rung must not store: releaseShadows() has
    // already unpinned this key, so a late pinned store would outlive playback.
    if (this.tornDown(signal)) return;
    const keep =
      this.loadedCount === this.nLods || this.frameBudgetMs !== null || this.ladderDepth !== null;
    if (keep) this.store(viewState, viewState.prefetch === true);
  }

  private store(viewState: ViewState, pin: boolean): void {
    storeLadder(this.sliceCache, this.path, viewState, this.loadedLODs, {
      scan: this.frameBudgetMs !== null,
      pin,
      ladderDepth: this.loadedCount,
      totalLODCount: this.nLods,
    });
  }

  /** True once this pass was aborted or the loader disposed. */
  private tornDown(signal?: AbortSignal): boolean {
    return signal?.aborted === true || this.disposed;
  }

  /**
   * Concatenate the retained payloads, memoized on (reset generation, logical
   * rung count), stamping the prefix-lineage parent (the previous
   * same-generation memo) and, for a result built from a restored snapshot
   * alone, its slice-cache origin. The rungs' pooled decode buffers are
   * released once copied; only the cumulative payload is retained.
   */
  private concatenateMemoized(session?: UpdateSession): TData {
    const concatSession = session?.begin('Concatenate LODs');
    try {
      const memo = this.sameGenerationMemo();
      if (memo && memo.lodCount === this.loadedCount) return memo.result;
      // The previous SAME-GENERATION memo is (only) what this result extends.
      const prevMemo = memo && memo.lodCount < this.loadedCount ? memo.result : null;
      const result = this.geometry.concat(this.loadedLODs, {
        payloadLevelStarts: this.payloadLevelStarts,
        loadedLevels: this.loadedCount,
      });
      setPrefixParent(result, prevMemo);
      if (this.restoredOrigin && this.geometry.stampsRestoredOrigin) {
        setSliceCacheOrigin(result, this.restoredOrigin);
      }
      this.releaseRungAccumulators();
      this.loadedLODs = [result];
      this.payloadLevelStarts = [0];
      this.concatCache = {
        generation: this.resetGeneration,
        lodCount: this.loadedCount,
        result,
      };
      return result;
    } finally {
      concatSession?.end();
    }
  }

  /** The memoized concat when it belongs to the current reset generation. */
  private sameGenerationMemo(): { lodCount: number; result: TData } | null {
    const cache = this.concatCache;
    return cache !== null && cache.generation === this.resetGeneration ? cache : null;
  }

  /** Release the loaded rungs' pooled decode buffers once their payloads are copied. */
  private releaseRungAccumulators(): void {
    for (let level = 0; level < this.loadedCount; level++) {
      this.rungs[level]?.releaseAccumulator();
    }
  }

  // ---- logging -------------------------------------------------------------

  private logRung(level: number, data: TData, elapsed: number, allResident: boolean): void {
    const { module, unit } = this.geometry;
    log.verbose(LogEmoji.BROADCAST, module, () => {
      const count = this.geometry.countOf(data);
      const miss = allResident ? '' : ', miss';
      return `LOD ${level}/${this.nLods - 1}: ${count} ${unit} (${elapsed.toFixed(1)}ms${miss})`;
    });
  }

  /** Per-pass summary, at verbose: a pass runs on every view tick. */
  private logSummary(startLevel: number): void {
    if (this.loadedCount >= this.nLods && startLevel >= this.nLods) return;
    const { module, label, unit } = this.geometry;
    log.verbose(LogEmoji.INFO, module, () => {
      const total = this.loadedLODs.reduce((s, d) => s + this.geometry.countOf(d), 0);
      const state = this.loadedCount < this.nLods ? 'refining' : 'complete';
      return `Progressive ${label}: ${this.loadedCount}/${this.nLods} LODs (${total} ${unit}) — ${state}`;
    });
  }

  // ---- speculation ---------------------------------------------------------

  private reportSpeculative(action: string, error: unknown): void {
    const { module, warnPrefix } = this.geometry;
    reportSpeculativeFailure(module, `${warnPrefix} ${action}`, error);
  }

  private warmRemainingMetadata(): void {
    if (this.metadataWarmStarted || this.disposed) return;
    this.metadataWarmStarted = true;
    for (let level = 1; level < this.nLods; level++) this.warmRungIndex(level);
  }

  /**
   * Request rung `level`'s spatial index at REFINEMENT priority (B6): a finer
   * rung of something on screen queues behind every demand read but no longer
   * waits for the rung before it to finish. Idempotent.
   */
  private warmRungIndex(level: number): void {
    if (this.disposed || level >= this.nLods) return;
    void this.rungs[level]
      .ensureInitialized('refinement')
      .catch((error: unknown) => this.reportSpeculative('metadata warming failed', error));
  }

  /**
   * The signal-linked speculative lookahead of the next rung(s): free
   * navigation only — a playback / pinned pass's next pass is a DIFFERENT
   * slice, so this slice's next rung would never be read (measured pure
   * waste) — and never for a superseded pass.
   */
  private scheduleLookahead(viewState: ViewState, updateSignal?: AbortSignal): void {
    if (this.disposed || this.frameBudgetMs !== null || this.ladderDepth !== null) return;
    const firstLevel = this.loadedCount;
    if (firstLevel >= this.nLods) return;
    const controller = this.lookaheadController(updateSignal);
    if (!controller) return;
    this.planner.plan({
      rungs: this.rungs,
      firstLevel,
      viewState,
      metadataWarm: this.metadataWarmStarted,
      controller,
      start: (level, plan) => this.startLookahead(level, viewState, controller, plan),
      isLive: () => !controller.signal.aborted && !this.disposed,
    });
  }

  /**
   * The lookahead controller for this view (the lookahead-signal contract):
   * speculative, aborted on view change and dispose, AND linked to the update
   * that scheduled it. Reused across refinement passes of one view while live;
   * replaced once a superseded update aborted it. `null` when that update is
   * already aborted.
   */
  private lookaheadController(updateSignal?: AbortSignal): AbortController | null {
    if (updateSignal?.aborted) return null;
    if (this.lookahead && !this.lookahead.signal.aborted) return this.lookahead;
    this.lookahead = createLookaheadController(updateSignal);
    this.lookaheadLevels.clear();
    this.planner.reset();
    return this.lookahead;
  }

  private startLookahead(
    level: number,
    viewState: ViewState,
    controller: AbortController,
    plan?: Promise<{ ranges?: readonly unknown[] }>
  ): void {
    if (this.lookaheadLevels.has(level)) return;
    this.lookaheadLevels.add(level);
    tagSignalOrigin(controller.signal, 'lookahead');
    const prefetch = async (): Promise<void> => {
      const ranges = plan ? (await plan).ranges : undefined;
      // A dispose() or view change while the plan was in flight aborted this
      // controller and may have dropped the rungs.
      if (controller.signal.aborted || this.disposed) return;
      const rung = this.rungs[level];
      await (ranges
        ? rung.prefetchChunks(viewState, controller.signal, ranges as never)
        : rung.prefetchChunks(viewState, controller.signal));
    };
    void prefetch()
      .catch((error: unknown) => this.reportSpeculative('lookahead prefetch failed', error))
      .finally(() => {
        if (this.lookahead === controller) this.lookaheadLevels.delete(level);
      });
  }

  /**
   * Predictive read-ahead for a scrub (B5): after a pass for a view reached by
   * moving a hidden-dimension slice, warm rung 0 of the neighbouring slices
   * (one step either way along each moved axis), so the next drag step's
   * first commit is a cache hit. Speculative priority, aborted by the next
   * view change; free navigation only (playback has the SlicePrefetcher) and
   * once per view. Each neighbour gets its own controller, so the step that
   * lands on one keeps its read (see {@link cancelSpeculation}).
   */
  private readAheadNeighbourSlices(viewState: ViewState, updateSignal?: AbortSignal): void {
    const dims = this.scrubDims;
    this.scrubDims = [];
    if (dims.length === 0 || this.nLods < 2 || this.disposed || updateSignal?.aborted) return;
    if (this.frameBudgetMs !== null || this.ladderDepth !== null) return;
    for (const neighbour of neighbourSlices(viewState, dims)) {
      const controller = new AbortController();
      tagSignalPriority(controller.signal, 'speculative');
      this.readAheads.push({ view: neighbour, controller });
      void this.rungs[0]
        .prefetchChunks(neighbour, controller.signal)
        .catch((error: unknown) => this.reportSpeculative('slice read-ahead failed', error));
    }
  }

  /**
   * Abort the speculative reads of the view being left. A read-ahead of
   * exactly `keepFor` (the view being entered) is KEPT: until the new pass's
   * rung-0 read joins it, the warm is the only waiter on its chunk fetches,
   * and aborting it there would cancel the shared fetch and force a refetch.
   */
  private cancelSpeculation(keepFor?: ViewState): void {
    const kept = this.readAheads.filter(
      ({ view }) => keepFor !== undefined && viewStatesEqual(view, keepFor)
    );
    for (const entry of this.readAheads) {
      if (!kept.includes(entry)) entry.controller.abort();
    }
    this.readAheads = kept;
    this.lookahead?.abort();
    this.lookahead = null;
    this.lookaheadLevels.clear();
    this.planner.reset();
  }
}
