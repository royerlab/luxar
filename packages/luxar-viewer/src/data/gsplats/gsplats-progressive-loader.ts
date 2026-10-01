/**
 * Progressive GSplats loader for multi-LOD datasets.
 *
 * Wraps N GSplatsSpatialIndexLoader instances (one per LOD subgroup) using the
 * Composite pattern. Implements the same GSplatsDataLoader interface so the
 * scene loader's update loop works unchanged.
 *
 * The per-view lifecycle (streaming policy, SliceCache departure/adoption/
 * restore, B6 index warm, pinned concurrent rungs, signal-linked lookahead,
 * B5 neighbour read-ahead, memoized concat) is the shared
 * `AdditiveLadderCore` (`../loaders/progressive/additive-ladder-core`). GSplats
 * adds its concatenation and a deeper lookahead: up to three unloaded LODs
 * when the L0 cache has headroom (one rung otherwise, and none during
 * playback).
 *
 * LODs are additive: LOD 0 contains the coarsest (highest-amplitude) splats,
 * and each subsequent LOD adds residual detail. The loader concatenates all
 * loaded LODs into a single LoadedGSplatsData.
 *
 * @module data/gsplats-progressive-loader
 */

import type { GSplatsDataLoader, GSplatsViewState, LoadedGSplatsData } from '../../types/gsplats';
import type {
  GSplatsPrefetchPlan,
  GSplatsSpatialIndexLoader,
} from './gsplats-spatial-index-loader';
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
  concatRequiredField,
} from '../loaders/progressive/concat-helpers';
import {
  AdditiveLadderCore,
  reportSpeculativeFailure,
  type LadderGeometry,
  type LookaheadPlanner,
  type LookaheadRequest,
} from '../loaders/progressive/additive-ladder-core';
import { SPLAT_FLOATS_PER_SPLAT } from '../../rendering/element-texture-layout';
import type { LadderResidency } from '../scene-loader/progressive/residency-budget';
import type { SliceCache } from '../../cache/slice-cache';
import { log, Modules, LogEmoji } from '../../utils/log';

// Three rungs overlap hosted latency without letting speculation monopolize L0 or fetch slots.
const MAX_PREFETCH_LEVELS = 3;

function resolvePrefetchDepth(metadataWarmStarted: boolean, hasCacheStats: boolean): number {
  return metadataWarmStarted && hasCacheStats ? MAX_PREFETCH_LEVELS : 1;
}

function readPrefetchHeadroom(loader: GSplatsSpatialIndexLoader | undefined): number | null {
  if (!loader) return null;
  const stats = loader.getPrefetchCacheStats();
  if (!stats) return null;
  return Math.max(0, (stats.maxSize ?? 0) - stats.size);
}

interface DeepLookaheadPlan {
  firstLevel: number;
  maxLevels: number;
  headroom: number;
  viewState: GSplatsViewState;
  firstPlan: Promise<GSplatsPrefetchPlan>;
}

async function selectPrefetchLevels(
  lodLoaders: readonly GSplatsSpatialIndexLoader[],
  request: DeepLookaheadPlan
): Promise<Array<{ level: number; plan: GSplatsPrefetchPlan }>> {
  const levels: Array<{ level: number; plan: GSplatsPrefetchPlan }> = [];
  let reservedBytes = 0;
  const stopLevel = Math.min(lodLoaders.length, request.firstLevel + request.maxLevels);
  const candidates = lodLoaders.slice(request.firstLevel, stopLevel);
  const plans = await Promise.all(
    candidates.map((loader, index) =>
      index === 0 ? request.firstPlan : loader.planPrefetch(request.viewState)
    )
  );
  for (let index = 0; index < candidates.length; index++) {
    const level = request.firstLevel + index;
    const plan = plans[index];
    const estimate = plan.bytes;
    reservedBytes +=
      Number.isFinite(estimate) && estimate >= 0 ? estimate : Number.POSITIVE_INFINITY;
    if (level > request.firstLevel && reservedBytes > request.headroom) break;
    levels.push({ level, plan });
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

  // Colours follow the shared ladder policy: a rung without them is filled
  // with white (layout 3 = RGB or 4 = RGBA, the 4th channel per-splat opacity).
  const colored = concatColorsWhiteFilled(
    parts,
    (p) => ({ colors: p.colors, components: p.colorComponents }),
    count,
    'concatenateGSplatsData'
  );
  const colors = colored?.colors ?? null;
  const colorK = colored?.colorComponents ?? 3;

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

const GSPLATS_LADDER: LadderGeometry<LoadedGSplatsData> = {
  kind: 'gsplats',
  module: Modules.GSPLATS_SPATIAL_INDEX_LOADER,
  label: 'GSplats',
  unit: 'splats',
  warnPrefix: 'GSplat',
  countOf: (data) => data.splatCount,
  concat: (parts) => concatenateGSplatsData(parts),
  empty: () => concatenateGSplatsData([]),
  // 4 RGBA32F texels/splat. See LadderResidency — the payload alone is not
  // the node's footprint, and the ratio differs per geometry.
  bytesPerElement: SPLAT_FLOATS_PER_SPLAT * Float32Array.BYTES_PER_ELEMENT,
  stampsRestoredOrigin: true,
};

/**
 * GSplats' deeper lookahead: once every rung's metadata warm has started and
 * the L0 cache reports headroom, plan up to {@link MAX_PREFETCH_LEVELS} rungs
 * whose estimated bytes fit that headroom, instead of the next rung alone.
 */
class GSplatsLookaheadPlanner implements LookaheadPlanner<GSplatsSpatialIndexLoader> {
  private planning = false;
  private epoch = 0;
  private lastLoggedDepth: number | null = null;

  plan(request: LookaheadRequest<GSplatsSpatialIndexLoader>): void {
    const headroom = readPrefetchHeadroom(request.rungs[0]);
    const maxLevels = resolvePrefetchDepth(request.metadataWarm, headroom !== null);
    if (maxLevels === 1 || headroom === null || headroom <= 0) {
      this.logDepth(1, headroom);
      request.start(request.firstLevel);
      return;
    }
    const firstPlan = request.rungs[request.firstLevel].planPrefetch(request.viewState);
    request.start(request.firstLevel, firstPlan);
    if (this.planning) return;
    void this.planDeep(request, { maxLevels, headroom, firstPlan }).catch((error: unknown) =>
      reportSpeculativeFailure(
        Modules.GSPLATS_SPATIAL_INDEX_LOADER,
        'GSplat lookahead planning failed',
        error
      )
    );
  }

  reset(): void {
    this.epoch++;
    this.planning = false;
  }

  private async planDeep(
    request: LookaheadRequest<GSplatsSpatialIndexLoader>,
    budget: { maxLevels: number; headroom: number; firstPlan: Promise<GSplatsPrefetchPlan> }
  ): Promise<void> {
    const epoch = this.epoch;
    this.planning = true;
    try {
      const levels = await selectPrefetchLevels(request.rungs, {
        firstLevel: request.firstLevel,
        viewState: request.viewState,
        ...budget,
      });
      if (!request.isLive()) return;
      this.logDepth(levels.length, budget.headroom);
      for (const { level, plan } of levels) {
        if (level !== request.firstLevel) request.start(level, Promise.resolve(plan));
      }
    } finally {
      if (epoch === this.epoch) this.planning = false;
    }
  }

  private logDepth(depth: number, headroom: number | null): void {
    if (depth === this.lastLoggedDepth) return;
    this.lastLoggedDepth = depth;
    const headroomLabel = headroom === null ? 'unavailable' : `${Math.round(headroom)} bytes`;
    log.custom(
      LogEmoji.CACHE,
      Modules.GSPLATS_SPATIAL_INDEX_LOADER,
      `GSplat lookahead depth ${depth} (${headroomLabel} L0 headroom)`
    );
  }
}

/**
 * Progressive GSplats loader for multi-LOD datasets: the GSplats face of the
 * shared additive-ladder engine (`AdditiveLadderCore`), which owns the whole
 * per-view lifecycle. This class adds the GSplats concatenation and the deeper
 * L0-headroom lookahead planner.
 */
export class GSplatsProgressiveLoader implements GSplatsDataLoader {
  private readonly core: AdditiveLadderCore<LoadedGSplatsData, GSplatsSpatialIndexLoader>;
  private readonly monitor: ProgressiveMonitorAdapter;

  constructor(
    lodLoaders: GSplatsSpatialIndexLoader[],
    nLods: number,
    path: string,
    energyTable?: ReadonlyArray<number | null | undefined>,
    sliceCache?: SliceCache | null
  ) {
    this.core = new AdditiveLadderCore(lodLoaders, nLods, path, GSPLATS_LADDER, {
      energyTable,
      sliceCache,
      planner: new GSplatsLookaheadPlanner(),
    });
    this.monitor = new ProgressiveMonitorAdapter(
      () => this.core.rungLoaders,
      path,
      'gsplats-spatial-index'
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

  /** Load gsplats data (delegates to {@link updateView}). */
  async loadGSplats(
    viewState: GSplatsViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedGSplatsData> {
    return this.updateView(viewState, session, signal);
  }

  /** Stream as many LODs as the pass allows for `viewState` (see the core). */
  updateView(
    viewState: GSplatsViewState,
    session?: UpdateSession,
    signal?: AbortSignal,
    residencyAllowanceBytes?: number
  ): Promise<LoadedGSplatsData> {
    return this.core.updateView(viewState, session, signal, residencyAllowanceBytes);
  }

  /** Predicted-view warm-up of the coarse rung (`dispatchPredictivePrefetch`). */
  prefetchChunks(viewState: GSplatsViewState, signal?: AbortSignal): Promise<void> {
    return this.core.prefetchChunks(viewState, signal);
  }

  /** Predicted-view warm-up over a current → predicted transition. */
  prefetchChunkBoundary(
    current: GSplatsViewState,
    predicted: GSplatsViewState,
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
