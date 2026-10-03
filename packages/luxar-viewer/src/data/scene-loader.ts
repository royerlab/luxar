/**
 * Unified scene loader that orchestrates the loading of complete Luxar scenes.
 *
 * This loader handles the entire scene graph, using spatial index-based
 * loading for all points nodes and managing the THREE.js scene construction.
 */

import { findObjectByName } from '../utils/scene-graph-index';
import { isEffectivelyVisible } from '../utils/object-visibility';
import { tagSignalPriority } from '../utils/fetch-concurrency';
import { bumpFailedLoadsVersion, failedLoadsVersion } from '../utils/failed-loads-version';
import * as zarr from './zarr';
import * as THREE from 'three';
import { normalizeURL } from './scene-loader/lifecycle/url-normalization';
import type { RefinementHoldReason } from '../types/data-monitor-types';
import { applyEffectiveAttrs as applyEffectiveAttrsHelper } from './scene-loader/view-state/effective-attrs';
import {
  getCacheStats as getCacheStatsHelper,
  listCachedDatasets as listCachedDatasetsHelper,
  clearL0Cache as clearL0CacheHelper,
  clearL1Cache as clearL1CacheHelper,
  clearL2Cache as clearL2CacheHelper,
  clearAllCaches as clearAllCachesHelper,
  type CacheStatsSnapshot,
} from './scene-loader/cache/cache-api';
import {
  processLinesData as processLinesDataHelper,
  commitLinesGeometry as commitLinesGeometryHelper,
  type StagedLinesCommit,
} from './scene-loader/process/data-processor-lines';
import {
  processGSplatsData as processGSplatsDataHelper,
  commitGSplatsGeometry as commitGSplatsGeometryHelper,
  type StagedGSplatsCommit,
} from './scene-loader/process/data-processor-gsplats';
import { commitMeshGeometry as commitMeshGeometryHelper } from './scene-loader/commit/commit-mesh-geometry';
import type { GeometryCommitHost } from './scene-loader/commit/commit-host';
import { processMeshData as processMeshDataHelper } from './scene-loader/process/data-processor-mesh';
import type { StagedMeshCommit } from './scene-loader/process/data-processor-mesh';
import type { LoaderFactoryDeps } from './scene-loader/loaders/loader-factory';
import { commitPointsGeometry as commitPointsGeometryHelper } from './scene-loader/commit/commit-points-geometry';
import {
  processPointsData as processPointsDataHelper,
  type StagedPointsCommit,
} from './scene-loader/process/data-processor-points';
import { ViewStateQueue } from './scene-loader/view-state/view-state-queue';
import { runGSplatsRefinement } from './gsplats/lod-refinement';
import { runPointsRefinement } from './points/lod-refinement';
import { runLinesRefinement } from './lines/lod-refinement';
import { runMeshRefinement } from './mesh/lod-refinement';
import { setSceneLineLoad } from '../types/line-primitive';
import { loadAndStage as pointsLoadAndStage, label as pointsLabel } from './points/handler';
import { loadAndStage as linesLoadAndStage, label as linesLabel } from './lines/handler';
import { loadAndStage as gsplatsLoadAndStage, label as gsplatsLabel } from './gsplats/handler';
import { loadAndStage as meshLoadAndStage, label as meshLabel } from './mesh/handler';

/**
 * Staged-commit payload for a lines node: either the processed geometry held
 * between the async load/process stage and the synchronous atomic commit, or
 * the stamp-only no-op fast path taken when the GPU already holds that exact
 * data. Defined in the data-processor module; the re-export here has no
 * consumers today — every call site imports it from `data-processor-lines`
 * directly.
 */
export type { StagedLinesCommit } from './scene-loader/process/data-processor-lines';
/**
 * Staged-commit payload for a gsplats node: the gsplats counterpart of
 * {@link StagedLinesCommit}, with the same geometry-or-no-op union and the
 * same consumer-free re-export.
 */
export type { StagedGSplatsCommit } from './scene-loader/process/data-processor-gsplats';
/** B5 drag commit guarantee constants; owned by the pass scheduler. */
export {
  DRAG_COMMIT_INTERVAL_MS,
  DRAG_COMMIT_MAX_HOLD_MS,
} from './scene-loader/update-view/pass-scheduler';
import {
  DataLoader,
  ViewState,
  SceneNode,
  LoaderConfig,
  LoadedPointsData,
} from './data-loader-types';
import type {
  SceneLoaderMonitorPort,
  SceneLoaderMonitorFactory,
  FailedLoadsProviderPort,
} from './scene-loader-monitor-port';
import { LODGroupRegistry } from '../scene/lod-group-registry';

/**
 * Narrow view of the SceneLoader a registry factory may read. The registry's
 * freshness (``getViewVersion``) and eviction (``getResidentBytes``) deps must
 * observe the loader that OWNS the registry — routing them through
 * ``getSceneLoader('default')`` read the wrong loader's state for any
 * non-default instance (multi-loader embedding).
 */
export interface LODGroupRegistryOwner {
  /** Monotonic view-update version of the owning loader. */
  readonly currentViewVersion: number;
  /** The owning loader's GPU buffer pool (null pre-setup / pooling off). */
  readonly gpuBufferPool: GPUBufferPool | null;
  /** Current archive fault latched by the owning loader, if any. */
  readonly archiveFault: ArchiveFaultError | null;
  /** Whether this path or one of its descendants has a network failure. */
  hasNetworkFailureUnder(path: string): boolean;
  /**
   * Re-run the owning loader's current view state. With ``paths`` (partition
   * node paths whose parts just re-entered the frustum) only loaders at or
   * under those paths are swept; omitted/empty = every loader. Never bumps
   * the view version — see ``_updateVersion``.
   */
  requestReprocess(paths?: readonly string[]): void;
  /** Whether the owning loader currently has an update or refinement pass in flight. */
  isUpdateInProgress(): boolean;
  /**
   * Whether a view PASS is in flight or queued — a refinement hold does NOT
   * count. The registry gates partition resyncs on this rather than on
   * ``isUpdateInProgress``: a resync arriving during a refinement hold is
   * parked and cancels into its own pass, whereas waiting for the hold to end
   * left re-entering parts on a stale slice for as long as the ladders kept
   * streaming (minutes on a slow link).
   */
  isLoadPassInProgress(): boolean;
  /**
   * The view the owning loader last COMMITTED geometry for (B4): partition
   * parts that miss its hidden-dim slice are hidden and kept out of refinement.
   * Optional so a narrow owner (a test double) may omit it ⇒ no slice gating.
   */
  readonly committedViewState?: ViewState;
}

/** Per-call directives for {@link SceneLoader.updateView}. */
export interface UpdateViewOptions {
  /**
   * Targeted partition resync: restrict the sweep to loaders whose node path
   * is one of these paths or nested under one (``isUnderAny``). Loaders outside
   * the set are skipped WITHOUT touching their predictive-prefetch baseline.
   * The view state is not changed by such a pass, so the view version is not
   * bumped either.
   */
  resyncPaths?: ReadonlySet<string>;
}

/**
 * Factory hook that supplies a per-loader ``LODGroupRegistry``. Mirrors
 * ``SceneLoaderMonitorFactory`` — the data/ layer never reaches into
 * scene/ for camera / viewport state, so the host (typically the app
 * pipeline) injects a closure that knows how to construct the
 * registry with proper getters. Receives the OWNING loader (as the
 * narrow {@link LODGroupRegistryOwner} view) so per-loader deps read
 * that loader's live state, not the manager's current default.
 */
export type SceneLoaderLODGroupRegistryFactory = (owner: LODGroupRegistryOwner) => LODGroupRegistry;
import { ArrayRefRegistry } from './array-decoder/decoder';
import { log, Modules } from '../utils/log';
import { getErrorMessage } from '../utils/format-error';
import { config as appConfig } from '../config';
import { MultiLevelCachingStore } from '../cache/multi-level-caching-store';
import { DecompressedChunkCache } from '../cache/decompressed-chunk-cache';
import { SliceCache } from '../cache/slice-cache';
import {
  cachePoolOverrideBytes,
  deviceClassPoolBytes,
  type CacheBudgets,
} from '../cache/heap-budget';
import type { LinesDataLoader, LinesViewState, LoadedLinesData } from '../types/lines';
import type { GSplatsDataLoader, GSplatsViewState, LoadedGSplatsData } from '../types/gsplats';
import type {
  KTX2TextureDecoder,
  LoadedMeshData,
  MeshDataLoader,
  MeshMetadata,
  MeshViewState,
} from '../types/mesh';
import { clearCommittedData } from '../types/committed-data';
import {
  releaseDepthSortNode,
  type DepthSortCoordinator,
} from '../rendering/depth-sort-coordinator';
import { runWithMaterialManager, type MaterialManager } from '../rendering/material-manager';
import { GPUBufferPool } from '../rendering/gpu-buffer-pool';
import { createEmptyMeshGeometry } from '../rendering/mesh-geometry';
import { invalidateRenderObjectFor } from './scene-loader/commit/invalidate-render-object';
import { getGpuByteBudget } from '../rendering/gpu-byte-budget';
import { NodeFactory } from '../rendering/node-factory';
import { UpdateProfiler, type UpdateSession } from '../profiling/update-profiler';
import { LoaderRegistry } from './scene-loader/loaders/loader-registry';
import { warnFailedLoaders } from './scene-loader/loaders/failure-report';
import { notifier } from '../utils/cross-layer/notifier';
import type { ArchiveFaultError } from '../cache/chunk-source';
import { combineAbortSignals } from '../utils/abort-signals';

// ============================================================================
// Staged commit types for atomic geometry updates
// ============================================================================
// During dimension animation, all nodes must update in the same render frame
// to prevent flickering. These types hold processed data between the async
// load/process stage and the synchronous commit stage.

// StagedPointsCommit is defined in ./points/handler and re-exported by
// ./scene-loader/process/data-processor-points, which is where it is imported
// from above.
// It stays internal to scene-loader + data-processor wiring.

import { initializeSceneDimensions as initializeSceneDimensionsHelper } from './scene-loader/nodes/initialize-scene-dimensions';
import {
  retryFailedLoaderUnlocked,
  retryAllFailedLoadersUnlocked,
  type RetryCtx,
} from './scene-loader/lifecycle/retry';
import { deriveNodeViewState as deriveNodeViewStateHelper } from './scene-loader/view-state/derive-node-view-state';
import { SlicePrefetcher } from './scene-loader/prefetch/slice-prefetcher';
import {
  resolveLoadEligibleLoaders,
  isObjectViewEligible,
  isUnderAny,
  runLoaderUpdates as runLoaderUpdatesHelper,
} from './scene-loader/loaders/run-loader-updates';
import { updateVisibleCountsInMonitor as updateVisibleCountsInMonitorHelper } from './scene-loader/monitor/visible-counts';
import { disposeSceneLoader } from './scene-loader/lifecycle/dispose';
import type { SceneIdentityWatchdog } from './scene-identity-watchdog';
import {
  loadScene as loadSceneHelper,
  type LoadSceneCtx,
} from './scene-loader/lifecycle/load-scene';
import { runAtomicCommit } from './scene-loader/update-view/atomic-commit';
import { buildUpdateCtxs } from './scene-loader/update-view/build-update-ctxs';
import { PassScheduler } from './scene-loader/update-view/pass-scheduler';
import {
  abandonedLadderPaths,
  resetRefinementFailureTrackers,
  reopenRefinementFailureTrackers,
  type RefinableLoader,
} from './scene-loader/progressive/refinement-wrapper';
import {
  ABANDONED_RUNG_RETRY_BASE_MS,
  ABANDONED_RUNG_RETRY_MAX_MS,
  MAX_ABANDONED_RUNG_RETRY_ROUNDS,
} from './scene-loader/progressive/refinement';
import { connectLoaderToMonitor as connectLoaderToMonitorHelper } from './scene-loader/nodes/connect-loader-to-monitor';
import type {
  LeafMaterializedListener,
  LineWorkingSetGate,
  NodeBuildCtx,
} from './scene-loader/nodes/build-ctx';
import { createLineWorkingSetGate } from './scene-loader/nodes/load-children-concurrently';
import {
  noteRefinementAborted,
  noteRefinementComplete,
  noteRefinementStarted,
} from '../profiling/load-timeline';
import { viewStatesEqual } from './loaders/progressive/view-state-equal';
import { tryRollbackToPassStart } from './loaders/progressive/pass-rollback';
import { SceneNodeIndex } from './scene-loader/view-state/scene-node-index';

import {
  RefinementResidencyBudget,
  RefinementResidencyReporter,
  type LadderResidency,
  type RefinementResidencyStop,
} from './scene-loader/progressive/residency-budget';
import {
  RefinementDensityGate,
  type DensityGateCaps,
  type ProjectedDensityProvider,
} from './scene-loader/progressive/density-gate';

/** The per-type handler ctxs of one pass (see `build-update-ctxs.ts`). */
type UpdateCtxs = ReturnType<typeof buildUpdateCtxs>;

/**
 * What every refinement phase of one run shares (see `scheduleProgressiveRefinement`).
 * Do not un-export: TypeDoc needs this name.
 */
export interface RefinementPhaseShared {
  viewStateQueue: ViewStateQueue;
  updateVisibleCountsInMonitor(): void;
  retriggerUpdate(pendingState: Partial<ViewState>): void;
  isActive(): boolean;
  signal: AbortSignal;
  profiler: UpdateProfiler | null;
  residencyBudget: RefinementResidencyBudget;
  releaseLock(): void;
}

/** Join two stage-1 sweeps of one pass, type by type (points, lines, gsplats, mesh). */
function concatSweeps<A, B, C, D>(
  first: [A[], B[], C[], D[]],
  second: [A[], B[], C[], D[]]
): [A[], B[], C[], D[]] {
  return [
    [...first[0], ...second[0]],
    [...first[1], ...second[1]],
    [...first[2], ...second[2]],
    [...first[3], ...second[3]],
  ];
}
/**
 * Main scene loader that handles the complete loading pipeline.
 *
 * Features:
 * - Spatial index-based loading for efficient nD queries
 * - Hierarchical scene graph construction
 * - Transform and rendering attribute inheritance
 * - Dimension metadata management
 * - Memory-efficient loading with proper caching
 *
 * **Lifecycle: one-shot.** A loader loads exactly one dataset: a second
 * `loadScene()` — or one after `dispose()`, which is terminal — throws. Use
 * `SceneLoaderManager.createLoaderAsync()` (the canonical entry point), which
 * awaits the previous loader's teardown and constructs a fresh loader per
 * dataset.
 */
export class SceneLoader {
  private _zarrStore: zarr.Readable | null = null;

  /** Public accessor for the zarr store (needed by LabelLoader). */
  get zarrStore(): zarr.Readable | null {
    return this._zarrStore;
  }
  private cachingStore: MultiLevelCachingStore | null = null;
  // L0 decompressed chunk cache - caches decoded zarr chunks to avoid Blosc decompression
  private l0Cache: DecompressedChunkCache | null = null;
  // SliceCache ("S-cache") - per-(node,view) decoded-slice cache for instant slice revisits
  private sliceCache: SliceCache | null = null;
  // Resolved per-tier cache budgets from setupCaches (Settings popover readout)
  private cacheBudgets: CacheBudgets | null = null;
  private registry = new LoaderRegistry();
  private readonly lineWorkingSetGate: LineWorkingSetGate;
  /**
   * Scene-wide byte-ceiling record for progressive refinement, surfaced on the
   * debug snapshot (#2508); lifetime contract on {@link RefinementResidencyStop}.
   *
   * DELIBERATELY NOT CLEARED BY `dispose()`: a loader loads one dataset and
   * `dispose()` is terminal (a second `loadScene()` throws), so a new scene
   * always gets a fresh loader and reporter through
   * `SceneLoaderManager.createLoaderAsync`.
   */
  private readonly refinementResidencyReporter = new RefinementResidencyReporter();
  // Projected-density rung gate (density-gate.ts); null = no provider wired
  // (guard disabled, tests, embedders) = bytes-only admission.
  private refinementDensityGate: RefinementDensityGate | null = null;
  /**
   * The last refinement run's residency budget, kept so the data monitor can
   * tell a rung held at the residency ceiling from one still streaming
   * (`refinementHoldReason`). Replaced at every run start.
   */
  private lastResidencyBudget: RefinementResidencyBudget | null = null;

  // Delegate registry-backed maps used by the loader orchestration methods.
  private get loaders() {
    return this.registry.loaders;
  }
  private get linesLoaders() {
    return this.registry.linesLoaders;
  }
  private get gsplatLoaders() {
    return this.registry.gsplatLoaders;
  }
  private get meshLoaders() {
    return this.registry.meshLoaders;
  }
  private get failedLoaders() {
    return this.registry.failedLoaders;
  }

  private viewState: ViewState;
  /** See {@link committedViewState}; `null` until a load or pass commits. */
  private _committedViewState: ViewState | null = null;
  private config: LoaderConfig;
  private rootGroup: THREE.Group | null = null;
  /**
   * Optional monitor port — populated at construction by the factory
   * passed in via `SceneLoaderManager.setMonitorFactory`. Null when no
   * UI is wired up (tests, embedders), in which case all monitor calls
   * become no-ops at the call sites.
   */
  private monitor: SceneLoaderMonitorPort | null = null;
  private arrayRefRegistry: ArrayRefRegistry;

  // GPU buffer pool for geometry reuse, integrated into
  // updatePointsGeometry/updateLinesGeometry/updateGSplatsGeometry. Enabled
  // via config.dataLoading.performance.useGPUBufferPool.
  private _gpuBufferPool: GPUBufferPool | null = null;
  public readonly nodeFactory = new NodeFactory();

  // Update profiler for timing scene updates (optional, provided by SceneLoaderManager)
  private profiler: UpdateProfiler | null = null;

  // Scene-identity watchdog for the current dataset (http(s) sources only);
  // started at the end of loadScene, disposed on dataset switch/teardown.
  private _identityWatchdog: SceneIdentityWatchdog | null = null;

  // Latched until an explicit retry: loadScene is one-shot, and dataset switches
  // create a fresh SceneLoader through SceneLoaderManager.createLoaderAsync. A
  // retry clears this dataset-wide gate so updates can resume; a recurring fault
  // latches and notifies again. Only bootstrapStandalone registers the notifier
  // backend, so embedded hosts observe onArchiveFault (re-emitted by LuxarApp as
  // dataset-fault) instead (#2280).
  private _archiveFault: ArchiveFaultError | null = null;
  private archiveFaultListeners = new Set<(error: ArchiveFaultError) => void>();
  private _lastUpdateWasFrameBudgeted = false;
  /**
   * Monotonic view-generation counter, read by the LOD registry as
   * {@link currentViewVersion} and stamped onto every committed leaf as
   * ``userData.loadedViewVersion`` (freshness is EXACT equality — see
   * ``scene/lod-freshness.ts``).
   *
   * CONTRACT: it bumps ONLY when a query-determinant of the view changes
   * (``viewStatesEqual``: displayDims / slicePosition / tolerance / the
   * dimensions query signature). A pass whose merged view state equals the
   * current one — a partition resync, a retry re-run, a depth-sort re-commit —
   * re-sweeps under the SAME version. This matters because lazy ``lod_group``
   * levels are NOT in the sweep and are never re-stamped by it: bumping on an
   * unchanged view invalidated every resident fine level scene-wide, so a
   * partition part crossing the screen edge dropped ALL LOD groups to coarse
   * and re-streamed them (the #2366 regression on the h2afva 44-part scene).
   */
  private _updateVersion = 0;
  /**
   * Passes actually run (every ``updateView`` that reached the sweep). The
   * per-geometry processors gate their first-update info logs on this being
   * ``<= 1``; it used to be ``_updateVersion``, which now only advances on a
   * view CHANGE and would keep those logs on forever for a static scene.
   */
  private _passCount = 0;

  /**
   * Whether an updateView sweep (fetch/decode/upload) is currently in
   * flight. Exposed for consumers that must treat loading-time frame
   * jank as unrepresentative — e.g. the adaptive DPR manager suppresses
   * probe learning while this is true. Covers the serialized update
   * sweep, not late lazy-LOD commits (those surface as content-change
   * notifications instead).
   */
  public isUpdateInProgress(): boolean {
    return this.passes.locked;
  }

  /**
   * Whether a LOAD PASS is in flight. Three parts:
   *
   *   1. a RUNNING sweep — an `updateView` pass (fetch / decode / upload) up to
   *      its geometry commit, or a failed-loader retry sweep, which takes the
   *      same lock;
   *   2. MINUS the progressive-LOD refinement drain, which inherits that same
   *      lock (see `PassScheduler.refining`);
   *   3. PLUS a sweep that is QUEUED but has not started yet.
   *
   * Together they answer "has the data for the view the user asked for arrived
   * yet?".
   *
   * Refinement is excluded because it runs after the current view has already
   * been committed, so folding it in would turn this into full-ladder latency
   * instead of first-commit latency. That is the same distinction
   * `update-view/pass-scheduler.ts` already draws where it resolves the pass waiters
   * at refinement ENTRY rather than completion — the pacing gate there needs
   * first-commit latency too.
   *
   * The queued slot counts because the refinement exclusion would otherwise
   * open a hole big enough to drive a test through. The steady state on any
   * laddered dataset right after a commit is "lock held, refinement live"; an
   * `updateView` arriving then is parked in the pending slot by
   * `PassScheduler.request` without touching either flag.
   * The requested slice has not begun loading, yet both flags still describe
   * the refinement that preceded it — so without this clause a poller would
   * read idle and conclude the new slice had rendered. `hasPending()` is true
   * across exactly that window: the slot is filled by the request and cleared
   * by `takePending()` at the moment the next pass starts (the scheduler's
   * frame yield before it re-enters `updateView`, or its release drain), so the
   * flag cannot latch busy after a pass begins.
   *
   * Also outside its scope: the initial `loadScene` (which only touches the
   * lock at its very end, to hand it to the post-load refinement kick) and
   * lazy substitutive-LOD / deferred-partition `ensureLoaded` promotions, which
   * run outside any `updateView` cycle and surface as content-change
   * notifications instead.
   *
   * {@link isUpdateInProgress} keeps its existing, broader meaning ("the
   * serialization lock is held, refinement included") for its existing
   * consumers — the adaptive-DPR manager and `core/app/init/pipeline.ts`, both
   * of which want to discount loading-time frame jank for as long as the
   * loader is doing work of any kind.
   */
  public isLoadPassInProgress(): boolean {
    return this.passes.isLoadPassInProgress;
  }

  // Pending re-drain of ladders the consecutive-failure cap retired, and the
  // backoff round EACH retired ladder is on, keyed by node path (#2975). Per
  // ladder: a ladder retired after another spent its rounds still gets its own.
  // See `scheduleAbandonedRungRetry`.
  private _abandonedRetryTimer: ReturnType<typeof setTimeout> | null = null;
  private _abandonedRetryRounds = new Map<string, number>();
  // Set true in dispose(); progressive-refinement loops poll this (via the
  // ctx isActive callback) so they abort promptly when this loader is torn
  // down mid-flight (e.g. a dataset switch) instead of fetching/decoding
  // against a dead dataset.
  private _disposed = false;
  /** Set by the first `loadScene`; a loader loads exactly one dataset. */
  private _loadStarted = false;
  private _sceneGraph: SceneNode | null = null;
  /**
   * `path → node + world nD transform` index of {@link _sceneGraph}, built once
   * with it (the graph never changes afterwards). Every per-node derivation and
   * prefetch lookup reads it instead of walking the graph.
   */
  private _sceneNodeIndex: SceneNodeIndex | null = null;
  /** See {@link setLeafMaterializedListener}. */
  private _leafMaterializedListener: LeafMaterializedListener | null = null;

  /**
   * View-state queue: the pending slot (whose lifecycle {@link passes} drives)
   * and the per-loader previous view-state map used by predictive
   * prefetch. See ./scene-loader/view-state/view-state-queue.ts.
   *
   * The pending-state slot is overwritten on every queued update, so a
   * burst of view changes during an in-flight retry collapses to a
   * single drained call (latest-wins). The per-node prev-state map is
   * reset on dataset switch (loadScene) and dispose; skipped paths
   * forget their snapshot so the next non-skip update re-baselines.
   */
  private viewStateQueue = new ViewStateQueue();

  /**
   * Per-dataset AbortController. Created on every `loadScene` and
   * aborted at the START of the next `loadScene` (and on `dispose`)
   * so worker tasks queued by the previous dataset settle
   * immediately instead of running to completion against a
   * superseded scene. The loader threads it into its own worker
   * projections ({@link withDatasetSignal}); the pool itself is shared by
   * every host on the page and carries no dataset signal. WASM execution
   * itself cannot be cancelled, but the orphan results are discarded — see
   * {@link WorkerAbortError}.
   */
  private _datasetAbortController: AbortController | null = null;

  /**
   * Serialization of view passes, refinement runs and retries: the lock, the
   * pending slot's lifecycle, request generations and their waiters, the
   * resync stash and the B5 drag commit guarantee. See
   * `scene-loader/update-view/pass-scheduler.ts`.
   */
  private readonly passes: PassScheduler = new PassScheduler({
    queue: this.viewStateQueue,
    isDisposed: () => this._disposed,
    isFaulted: () => this._archiveFault !== null,
    matchesRunningPass: (viewState) => this.matchesRunningPass(viewState),
    runningDisplayDims: () => this.viewState.displayDims,
    viewVersion: () => this._updateVersion,
    runPass: (state, opts) => (opts ? this.updateView(state, opts) : this.updateView(state)),
    anyHasMoreLODs: () => this.registry.anyHasMoreLODs(),
    runRefinement: () => this.scheduleProgressiveRefinement(),
  });

  /**
   * Background t+1 slice prefetcher (dimension playback). Lazily created on
   * the first `prefetchSlice` call; aborted at the top of every `updateView`
   * (foreground always preempts); disposed with the loader. Its SHADOW
   * loader instances share nothing mutable with the foreground loaders —
   * the S-cache is the only handoff (see slice-prefetcher.ts).
   */
  private _slicePrefetcher: SlicePrefetcher | null = null;

  /**
   * Fire one background prefetch pass for the PREDICTED next view (t+1
   * during playback). Fire-and-forget: returns immediately. The shadow pass
   * deliberately persists across foreground ticks (a cold level outlives one
   * frame); it is aborted only on playback end / dispose / its stall guard,
   * and a foreground pass for the same slice waits for it rather than
   * re-assembling that slice (see slice-prefetcher.ts). The partial is merged
   * onto a COPY of the current view state — never persisted (a prefetch
   * must not move the real view; see the stuck-display hazard in
   * slice-prefetcher.ts).
   */
  prefetchSlice(
    viewState: Partial<ViewState>,
    budgetMs: number,
    ladderDepth?: number | 'auto'
  ): void {
    if (this._disposed || this._archiveFault || !this._sceneGraph) return;
    if (!this._slicePrefetcher) {
      this._slicePrefetcher = new SlicePrefetcher({
        getSceneGraph: () => this._sceneGraph,
        getSceneNodeIndex: () => this._sceneNodeIndex,
        factoryDeps: () => this.factoryDeps(),
        registry: this.registry,
        applyEffectiveAttrs: (node) => this.applyEffectiveAttrs(node),
        resolveObject: (path) => findObjectByName(this.rootGroup, path),
      });
    }
    // Strip any rider budget off the incoming partial — the shadow pass gets
    // exactly `budgetMs` (the prefetcher injects it post-derive).
    const incoming = { ...viewState };
    delete incoming.frameBudgetMs;
    delete incoming.ladderDepth;
    const predicted = { ...this.viewState, ...incoming };
    this._slicePrefetcher.prefetch(predicted, budgetMs, ladderDepth);
    this.activateAndWarmNextSlice(predicted, budgetMs, ladderDepth);
  }

  /**
   * Deferred partition parts the predicted slice needs (B4) are activated ahead
   * of it, so its foreground pass finds them registered — and, once they are, a
   * targeted shadow pass warms them: a shadow can only warm a REGISTERED node,
   * so the batch `prefetchSlice` just started cannot reach them. Fire-and-forget:
   * the activation never rejects.
   */
  private activateAndWarmNextSlice(
    predicted: ViewState,
    budgetMs: number,
    ladderDepth: number | 'auto' | undefined
  ): void {
    const registry = this.lodGroupRegistry;
    if (!registry) return;
    const graph = this._sceneGraph;
    void registry.activatePartitionParts(predicted, undefined, false).then((parts) => {
      if (parts.length === 0 || this._disposed || this._sceneGraph !== graph) return;
      this._slicePrefetcher?.prefetchTargets(predicted, budgetMs, ladderDepth, new Set(parts));
    });
  }

  /**
   * Release the prefetcher's shadow loaders (frees their accumulators).
   * Called when playback ends; shadows rebuild lazily on the next play.
   */
  releasePrefetchResources(): void {
    this._slicePrefetcher?.releaseShadows();
  }

  /** Public accessor for the scene graph built during loadScene(). */
  get sceneGraph(): SceneNode | null {
    return this._sceneGraph;
  }

  /**
   * Path index over {@link sceneGraph} (`path → node + world nD transform`),
   * built with it in `setSceneGraph`. Consumers outside the loader that derive
   * per-node view states (the audio slab rule) read nD transforms from here in
   * O(1) instead of walking the graph from the root on every derivation.
   */
  get sceneNodeIndex(): SceneNodeIndex | null {
    return this._sceneNodeIndex;
  }

  /**
   * Connect the owner of per-leaf LIVE appearance state (the Layers panel) to
   * every data leaf this loader attaches from now on — partition parts the LOD
   * registry activates, lazily built levels, anything built after load. Each
   * leaf is reported as its placeholder attaches, before any data reaches it,
   * together with this loader's scene graph. `null` disconnects.
   */
  setLeafMaterializedListener(listener: LeafMaterializedListener | null): void {
    this._leafMaterializedListener = listener;
  }

  /** Latched archive fault for this loader, or null while updates remain usable. */
  get archiveFault(): ArchiveFaultError | null {
    return this._archiveFault;
  }

  /**
   * Subscribe to archive-fault episodes for this loader. An explicit retry
   * clears the latch; if the archive fails again, listeners are notified again.
   * Listener exceptions are logged and do not propagate to the caller.
   *
   * @param options.replayCurrent Replay the current fault immediately when one is latched.
   */
  onArchiveFault(
    listener: (error: ArchiveFaultError) => void,
    options: { replayCurrent?: boolean } = {}
  ): () => void {
    this.archiveFaultListeners.add(listener);
    if (options.replayCurrent && this._archiveFault) {
      this.invokeArchiveFaultListener(listener, this._archiveFault);
    }
    return () => this.archiveFaultListeners.delete(listener);
  }

  private notifyArchiveFault(error: ArchiveFaultError): void {
    for (const listener of [...this.archiveFaultListeners]) {
      this.invokeArchiveFaultListener(listener, error);
    }
  }

  private reportArchiveFault(fault: ArchiveFaultError): void {
    if (this._archiveFault) return;
    this._archiveFault = fault;
    bumpFailedLoadsVersion();
    this.releasePrefetchResources();
    this.registry.clearAllFailures();
    log.error(Modules.SCENE_LOADER, `Archive fault: ${fault.message}`);
    notifier.error(fault.message, { persistent: true });
    this.notifyArchiveFault(fault);
  }

  private invokeArchiveFaultListener(
    listener: (error: ArchiveFaultError) => void,
    error: ArchiveFaultError
  ): void {
    try {
      listener(error);
    } catch (listenerError) {
      log.warning(Modules.SCENE_LOADER, 'Archive-fault listener threw:', listenerError);
    }
  }

  // ============================================================
  // Cache surface
  //
  // SceneLoader owns the L0 (decompressed-chunk) cache and the L1/L2
  // MultiLevelCachingStore. These methods expose a typed, public API that
  // tools (debug interface, embedders) can call without reaching into
  // private fields. Each method gracefully no-ops if the cache layer is
  // unavailable (e.g. when `noCache` is set in LoaderConfig).
  // ============================================================

  /** Snapshot of all cache levels (L0, S-cache, L1, L2) for debug and embed tooling. */
  getCacheStats(): CacheStatsSnapshot {
    return getCacheStatsHelper(this.l0Cache, this.cachingStore, this.sliceCache);
  }

  /** List datasets currently held by the L1/L2 caching store. */
  async listCachedDatasets(): Promise<Awaited<ReturnType<MultiLevelCachingStore['listDatasets']>>> {
    return listCachedDatasetsHelper(this.cachingStore);
  }

  /** True if the L1/L2 caching store is configured for this loader. */
  get hasCachingStore(): boolean {
    return this.cachingStore !== null;
  }

  /** Clear the in-memory L0 decompressed-chunk cache. No-op if absent. */
  clearL0Cache(): void {
    clearL0CacheHelper(this.l0Cache);
  }

  /** Clear the in-memory L1 metadata/chunk cache. No-op if absent. */
  clearL1Cache(): void {
    clearL1CacheHelper(this.cachingStore);
  }

  /** Clear the persistent L2 OPFS cache. No-op if absent. */
  async clearL2Cache(): Promise<void> {
    await clearL2CacheHelper(this.cachingStore);
  }

  /** Clear ALL cache tiers (L0 + L1 + L2 + the decoded-slice S-cache). */
  async clearAllCaches(): Promise<void> {
    await clearAllCachesHelper(this.l0Cache, this.cachingStore, this.sliceCache);
  }

  /**
   * Resolved per-tier cache budgets from the last `setupCaches` run (source:
   * heap / explicit / device-class / fixed), or null before the first scene
   * load / after dispose. Surfaced for the Settings popover's budget readout.
   */
  getCacheBudgets(): CacheBudgets | null {
    return this.cacheBudgets;
  }

  /**
   * Return a node-attrs record with rendering attributes replaced by the
   * effective values composed along the scene-graph ancestry (root → leaf).
   * Hierarchical composition: opacity/gamma/intensity multiply, offset adds,
   * blending_mode uses the nearest ancestor's choice.
   *
   * If the scene graph is unavailable, falls back to the node's raw attrs.
   */
  private applyEffectiveAttrs(node: SceneNode): SceneNode['attrs'] {
    return applyEffectiveAttrsHelper(this._sceneNodeIndex, node);
  }

  /**
   * Per-loader LOD-group registry. Constructed via the factory passed
   * by ``SceneLoaderManager`` so the registry's camera / viewport /
   * displayDims getters can close over the live SceneManager — which
   * the data/ layer must not import directly. ``null`` when the host
   * (e.g. headless tests) doesn't supply a factory; in that case the
   * scene loader still loads ``lod_group`` nodes but the per-frame
   * selector is a no-op (default level renders).
   */
  readonly lodGroupRegistry: LODGroupRegistry | null;

  /**
   * The GPU buffer pool, or null when pooling is disabled. Exposed so the
   * LOD-group registry's
   * resident-byte query (`getResidentBytes`) can read the single VRAM
   * truth; callers treat a disabled pool as 0 resident bytes, so it is never
   * over budget and never evicts.
   */
  get gpuBufferPool(): GPUBufferPool | null {
    return this._gpuBufferPool;
  }

  /**
   * This scene's progressive-refinement BYTE-ceiling stop, or `undefined` when
   * refinement never declined a rung.
   *
   * The reporter is scene-scoped (one per loader) while the residency budget is
   * rebuilt per refinement run, so this accumulates across runs — which is what
   * makes it answerable at capture time, long after the run that stopped. Read
   * by the debug snapshot through `SceneLoaderManager.refinementResidencyStop()`.
   */
  get refinementResidencyStop(): RefinementResidencyStop | undefined {
    return this.refinementResidencyReporter.snapshot();
  }

  /**
   * Monotonic view-generation version (bumped by ``updateView`` only when the
   * merged view state differs from the current one — see ``_updateVersion``).
   * The LOD registry reads this to decide whether a level's committed geometry
   * is fresh for the CURRENT view — a mesh stamped with an older version (its
   * data still reflects a previous slice/displayDims) is treated as stale so the
   * registry can show a coarser fresh level until the re-slice commits.
   */
  get currentViewVersion(): number {
    return this._updateVersion;
  }

  /**
   * The view the drawn geometry was last committed for (B4) — set when a load
   * starts and by every committed pass. Before the first commit, the live view.
   */
  get committedViewState(): ViewState {
    return this._committedViewState ?? this.viewState;
  }

  /**
   * Optional render-loop wake-up, wired by `SceneLoaderManager` from the
   * app pipeline (→ `AnimationController.startAnimation`). Fired after
   * every geometry commit so late commits — progressive-refinement
   * passes, failed-load retries, the online auto-retry, lazy LOD loads —
   * repaint even when the rAF loop has idle-paused meanwhile. The
   * callback is idempotent on the receiving side (startAnimation
   * early-outs while animating and re-arms the idle timer), so per-node
   * calls inside an atomic sweep are harmless. Null in bare/test
   * loaders → no-op.
   *
   * `drawn` is false when the committed node was not drawn either side of
   * the commit (itself or an ancestor hidden: an LOD level the registry
   * keeps off screen, a hidden layer), so the frame cannot have changed and
   * the receiver only keeps the loop ticking. It must still wake it: the LOD
   * registry polls per frame and may now show the level that just
   * committed, and that visibility flip requests its own redraw.
   */
  private _requestRender: ((drawn: boolean) => void) | null = null;
  private readonly decodeKTX2: KTX2TextureDecoder | null;

  /** Install (or clear) the render-loop wake-up callback. */
  setRequestRender(callback: ((drawn: boolean) => void) | null): void {
    this._requestRender = callback;
  }

  /**
   * The host's depth-sort coordinator, which every commit reports to. Set by
   * `loadScene` (`data/zarr-loader.ts`) before the scene loads, from the
   * coordinator the caller owns — the LuxarApp's `SceneManager.depthSort`, or a
   * LuxarLayer's own. `null` (tests, a bare loader) leaves the nodes untracked.
   */
  private _depthSort: DepthSortCoordinator | null = null;

  /** Install (or clear) the host's depth-sort coordinator. */
  setDepthSortCoordinator(coordinator: DepthSortCoordinator | null): void {
    this._depthSort = coordinator;
  }

  /**
   * The host's material manager, or null for the LuxarApp's. Installed by the
   * host's `SceneLoaderManager`; node creation (the factory) and the commit
   * stage both run against it, so a LuxarLayer's materials never join the
   * app's registry (capabilities, camera broadcast, disposal).
   */
  private _materials: MaterialManager | null = null;

  /** Install (or clear) the host's material manager. */
  setMaterialManager(materials: MaterialManager | null): void {
    this._materials = materials;
    this.nodeFactory.setMaterialManager(materials);
  }

  /** The host references the gsplats / lines / points commit helpers take. */
  private commitHost(): GeometryCommitHost {
    return {
      rootGroup: this.rootGroup,
      gpuBufferPool: this._gpuBufferPool,
      depthSort: this._depthSort,
    };
  }

  /**
   * Run a commit of the node at `path`, then wake the render loop (see
   * {@link _requestRender}), saying whether the commit can have changed the
   * drawn frame: the node was effectively visible before or after it. A node
   * that cannot be resolved counts as drawn. Returns that verdict.
   */
  private commitAndRequestRender(path: string, commit: () => void): boolean {
    const node = findObjectByName(this.rootGroup, path);
    const drawnBefore = isEffectivelyVisible(node);
    if (this._materials) runWithMaterialManager(this._materials, commit);
    else commit();
    const drawn = node === undefined || drawnBefore || isEffectivelyVisible(node);
    this._requestRender?.(drawn);
    return drawn;
  }

  /** Install (or clear) the notification used to arm online failure retries. */
  setAutoRetryableFailureCallback(callback: (() => void) | null): void {
    this.registry.setAutoRetryableFailureCallback(callback);
  }

  /**
   * Wire (or clear) the projected-density provider the refinement rung gate
   * reads. Same dependency inversion as `setRequestRender`: the tracker lives
   * in `scene/`, which `data/` cannot import. Forwarded by
   * `SceneLoaderManager.setRefinementDensityProvider`.
   */
  setRefinementDensityProvider(
    provider: ProjectedDensityProvider | null,
    caps: DensityGateCaps
  ): void {
    const held = this.refinementDensityGate?.deferredCount ?? 0;
    this.refinementDensityGate = provider ? new RefinementDensityGate(provider, caps) : null;
    // Clearing a gate that was holding rungs back (the guard turned off at
    // runtime) must let them load now: nothing else re-kicks the loop, and
    // `resumeDensityDeferredRefinement` has no gate left to consult.
    if (!provider && held > 0) this.kickRefinementIfIdle();
  }

  /**
   * Why this node's next rung is held back, if it is: `'density'` when the
   * density gate deferred it at the current framing, `'budget'` when the last
   * run's residency budget declined it, `null` otherwise (streaming, or nothing
   * pending). Density first: a density refusal is also folded into the budget's
   * declined set, and the camera-dependent reason is the actionable one.
   */
  refinementHoldReason(path: string): RefinementHoldReason | null {
    if (this.refinementDensityGate?.isDeferred(path)) return 'density';
    if (this.lastResidencyBudget?.isDeclined(path)) return 'budget';
    return null;
  }

  /**
   * Per-frame hook from the density walk: if any rung the gate deferred now
   * fits (the camera moved in), re-kick refinement. Cheap when nothing is
   * deferred (the common case). Returns how many paths resumed.
   */
  resumeDensityDeferredRefinement(): number {
    const gate = this.refinementDensityGate;
    if (!gate || gate.deferredCount === 0) return 0;
    const resumed = gate.takeResumable();
    if (resumed.length > 0) {
      log.info(
        Modules.SCENE_LOADER,
        `Resuming density-deferred refinement for ${resumed.length} node(s) (first: ${resumed[0]})`
      );
      this.kickRefinementIfIdle();
    }
    return resumed.length;
  }

  constructor(
    config: LoaderConfig = {},
    id?: string,
    profiler?: UpdateProfiler,
    monitorFactory?: SceneLoaderMonitorFactory | null,
    lodGroupRegistryFactory?: SceneLoaderLODGroupRegistryFactory | null,
    decodeKTX2?: KTX2TextureDecoder | null
  ) {
    this.profiler = profiler ?? null;
    this.config = config;
    // The explicit cache budget is unavailable to field initializers because config is assigned here.
    this.lineWorkingSetGate = createLineWorkingSetGate(
      cachePoolOverrideBytes(config.cacheBudgetMB),
      deviceClassPoolBytes()
    );
    this.viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [],
      tolerance: [],
    };
    this.arrayRefRegistry = new ArrayRefRegistry();
    this.lodGroupRegistry = lodGroupRegistryFactory ? lodGroupRegistryFactory(this) : null;
    this.decodeKTX2 = decodeKTX2 ?? null;

    // GPU buffer pool requires Float32Array data; the geometry-update path
    // falls back to the standard route for Uint8/Uint16 attributes.
    // Guard against double-init so repeated setup cannot leak a previous pool.
    if (appConfig.dataLoading.performance.useGPUBufferPool && !this._gpuBufferPool) {
      // Byte budget comes from the adaptive single-VRAM-authority budget
      // (auto-sized from deviceMemory, context-loss backoff), not the
      // fixed config value — so the pool and LOD retention share it. Pass
      // the getter (not a snapshot) so the pool reads the live budget at
      // eviction time and honors the context-loss backoff.
      this._gpuBufferPool = new GPUBufferPool(
        appConfig.dataLoading.performance.gpuPoolMaxSize,
        appConfig.dataLoading.performance.gpuPoolEvictionCommits,
        appConfig.dataLoading.performance.gpuPoolEvictBatchSize,
        () => getGpuByteBudget()
      );
      // Decimal MB (÷1e6) to match the unit used by gpu-byte-budget.ts's
      // own log lines, so the two "MB" figures for the same budget agree.
      const mb = (getGpuByteBudget() / 1_000_000).toFixed(0);
      log.info(
        Modules.GPU_BUFFER_POOL,
        `GPU buffer pool enabled (max size: ${appConfig.dataLoading.performance.gpuPoolMaxSize}, ` +
          `byte budget: ${mb} MB (live), ` +
          `eviction: ${appConfig.dataLoading.performance.gpuPoolEvictionCommits} commits, ` +
          `batch cap: ${appConfig.dataLoading.performance.gpuPoolEvictBatchSize})`
      );
    }

    // Resolve the monitor port through the injected factory. The
    // factory owns the UI-side singleton (DataMonitorManager) — we
    // only see the SceneLoaderMonitorPort surface so the data → ui
    // layer rule stays clean.
    if (typeof document !== 'undefined' && config.enableMonitor !== false && monitorFactory) {
      const monitorId = id ? `${id}-monitor` : 'default';
      this.monitor = monitorFactory(monitorId);
    }
  }

  /**
   * Load a complete scene from a Zarr store using chunk-based spatial indexing.
   *
   * Orchestrates the loading of hierarchical scene graphs, managing spatial indices,
   * attribute inheritance, and dimension metadata. Supports both points and lines
   * with automatic fallback for datasets without spatial ordering.
   *
   * The loading process:
   * 1. Opens Zarr store with optional two-level caching (L1 memory + L2 OPFS)
   * 2. Loads scene metadata and initializes dimensions
   * 3. Recursively constructs THREE.js scene graph from Zarr group hierarchy
   * 4. Creates spatial index loaders for efficient nD queries
   * 5. Connects loaders to data monitor for debugging
   *
   * @param url - Complete URL to the Zarr store. Can be:
   *              - HTTP URL: 'https://example.com/data.zarr'
   *              - Local path: '/path/to/data.zarr'
   *              - With query params: 'https://example.com/data.zarr?noCache'
   *
   * @returns Promise resolving to a THREE.Group containing the complete scene graph.
   *          The group's userData contains:
   *          - sceneDimensions: Dimension metadata if available
   *          - bounds: AABB of all points
   *          - nodeCount: Total number of leaf nodes
   *
   * @throws {Error} If this loader already loaded a scene or was disposed
   *         (one-shot — see the class doc)
   * @throws {Error} If the Zarr store cannot be opened or is invalid
   * @throws {Error} If consolidated metadata (.zmetadata) is malformed
   * @throws {Error} If required arrays (positions) are missing from point nodes
   *
   * @example
   * ```typescript
   * // Load a scene from HTTP URL
   * const scene = await sceneLoader.loadScene('https://example.com/data.zarr');
   * threeScene.add(scene);
   * console.log(`Loaded ${scene.children.length} top-level nodes`);
   * ```
   *
   * @example
   * ```typescript
   * // Load with error handling
   * try {
   *   const scene = await sceneLoader.loadScene(url);
   *   if (scene.children.length === 0) {
   *     console.warn('Scene is empty');
   *   }
   * } catch (error) {
   *   console.error('Failed to load scene:', error);
   *   // Fallback to default visualization
   * }
   * ```
   *
   * @example
   * ```typescript
   * // Access scene metadata after loading
   * const scene = await sceneLoader.loadScene(url);
   * const dims = scene.userData.sceneDimensions;
   * if (dims) {
   *   console.log(`${dims.length}D dataset:`, dims.map(d => d.name).join(', '));
   * }
   * ```
   *
   * @see {@link MultiLevelCachingStore} for caching implementation
   */
  async loadScene(url: string): Promise<THREE.Group> {
    // One-shot (see the class doc): a second load would run against this
    // loader's spent pass scheduler, caches and abort controller. Refuse it
    // loudly rather than reuse a half-torn-down instance.
    if (this._loadStarted || this._disposed) {
      throw new Error(
        'SceneLoader.loadScene is one-shot: create a fresh loader ' +
          '(SceneLoaderManager.createLoaderAsync) for each dataset'
      );
    }
    this._loadStarted = true;
    try {
      return await loadSceneHelper(url, this.makeLoadSceneCtx());
    } catch (error) {
      noteRefinementAborted();
      throw error;
    }
  }

  /** Build the per-call LoadSceneCtx. Never passes `this` to the helper. */
  private makeLoadSceneCtx(): LoadSceneCtx {
    return {
      config: this.config,
      viewState: () => this.viewState,
      loaders: this.loaders,
      linesLoaders: this.linesLoaders,
      gsplatLoaders: this.gsplatLoaders,
      meshLoaders: this.meshLoaders,
      gpuBufferPool: () => this._gpuBufferPool,
      monitor: () => this.monitor,
      profiler: this.profiler,
      lodGroupRegistry: this.lodGroupRegistry,
      normalizeURL: (u) => this.normalizeURL(u),
      clearViewStatePrev: () => this.viewStateQueue.clearPrev(),
      initializeSceneDimensions: (sd) => this.initializeSceneDimensions(sd),
      makeNodeBuildCtx: () => this.makeNodeBuildCtx(),
      updateVisibleCountsInMonitor: () => this.updateVisibleCountsInMonitor(),
      getFailedLoaderPaths: () => Array.from(this.failedLoaders.keys()),
      getFailedLoaderReasons: () =>
        Array.from(this.failedLoaders.values(), (info) => info.error?.message || info.kind || ''),
      getFailedLoadsProvider: () => this.getFailedLoadsProvider(),
      refinementHoldReason: (path) => this.refinementHoldReason(path),
      kindsWithMoreLODs: () => this.registry.kindsWithMoreLODs(),
      startRefinement: () => this.passes.startRefinement('Post-load progressive refinement failed'),
      setDatasetAbortController: (c) => {
        this._datasetAbortController = c;
      },
      setCachingStore: (s) => {
        this.cachingStore = s;
      },
      setL0Cache: (c) => {
        this.l0Cache = c;
      },
      setSliceCache: (c) => {
        this.sliceCache = c;
      },
      setCacheBudgets: (b) => {
        this.cacheBudgets = b;
      },
      setZarrStore: (s) => {
        this._zarrStore = s;
      },
      setRootGroup: (g) => {
        this.rootGroup = g;
      },
      setSceneGraph: (g) => {
        this._sceneGraph = g;
        this._sceneNodeIndex = new SceneNodeIndex(g);
      },
      setIdentityWatchdog: (w) => {
        // Defensive: loadScene is one-shot per loader, but never leak a
        // previously-set watchdog's timer if that ever changes.
        this._identityWatchdog?.dispose();
        this._identityWatchdog = w;
      },
    };
  }

  /**
   * Derive a per-node view state from `this.viewState`, folding in
   * `extend_to_all` tolerance overrides and the inverse `nd_transform`
   * for the node's path.
   *
   * The single source of truth for query-state derivation: both the
   * main update path and `retryFailedLoader` go through this method so
   * a retry cannot load a different query region than a fresh update
   * would. The pre-extraction retry skipped both adjustments, which
   * could produce a "successful" retry rendering incorrect data.
   *
   * A fully-extended node (`extend_to_all` covers all non-displayed dims)
   * is returned as a NORMAL `{ skip: false }` node whose query is made
   * slice-INVARIANT (extend-to-all tolerance sentinel + extended dims'
   * slicePosition pinned to 0), so per-sweep re-queries hit the loader's
   * same-view no-op. No `extend_to_all` skip is produced.
   *
   * @param path  Scene-graph path of the node, used to compose the
   *              world `nd_transform` from this node up to the root.
   * @param attrs Node attrs (only `extend_to_all` is read here).
   * @param opts.applyPartialExtendTolerance
   *              When `true`, partial `extend_to_all` coverage triggers
   *              a tolerance override via `getOrComputeExtendedTolerance`.
   *              Points/GSplats: true. Lines: false because line bounds
   *              already encode their non-displayed spatial extent.
   * @param opts.extendedToleranceCache
   *              Optional cross-node cache for the partial-extend
   *              tolerance array. Main update path passes one cache per
   *              cycle; retry passes nothing (fresh).
   */
  private deriveNodeViewState(
    path: string,
    attrs: { extend_to_all?: string[] } | undefined,
    opts: {
      applyPartialExtendTolerance: boolean;
      extendedToleranceCache?: Map<string, number[]>;
    }
  ): { skip: false; viewState: ViewState } {
    return deriveNodeViewStateHelper(path, attrs, this.viewState, this._sceneNodeIndex, opts);
  }

  /**
   * Shared scaffolding for the per-geometry update loops. The
   * Points / Lines / GSplats branches in updateView differ only in
   * the type-specific work (deriveNodeViewState, call
   * loader.updateView, post-process, setMetadata, return staged) —
   * the surrounding try/catch with failedLoaders bookkeeping +
   * retryCount tracking, profiler dispatch, and Promise.all are all
   * identical and live here.
   *
   * @param loaders   The map of (path → loader) for one geometry type.
   * @param loaderType Human-readable type for profiler label + error log.
   * @param updateFn  Per-loader work; returns staged commit data or
   *                  null when there's nothing to commit.
   */
  private async runLoaderUpdates<TLoader, TStaged>(
    loaders: Map<string, TLoader>,
    loaderType: 'Points' | 'Lines' | 'GSplats' | 'Mesh',
    updateFn: (path: string, loader: TLoader, session: UpdateSession) => Promise<TStaged | null>,
    onArchiveFault: (fault: ArchiveFaultError) => void,
    resyncPaths?: ReadonlySet<string>
  ): Promise<Array<{ staged: TStaged | null; session: UpdateSession }>> {
    return runLoaderUpdatesHelper(loaders, loaderType, updateFn, {
      profiler: this.profiler,
      viewStateQueue: this.viewStateQueue,
      registry: this.registry,
      onArchiveFault,
      // A pass runs for its OWN view: frustum + layer gate only, then its slice
      // (B4) — a part outside it is hidden from this pass's commit on.
      shouldUpdatePath: (path) =>
        isObjectViewEligible(findObjectByName(this.rootGroup, path)) &&
        this.lodGroupRegistry?.isPathInPartitionSlice(path, this.viewState) !== false,
      isResyncTarget: resyncPaths ? (path) => isUnderAny(path, resyncPaths) : undefined,
    });
  }

  /**
   * Stage 1 of a pass for every loader `pick` keeps: load + process each, one
   * {@link runLoaderUpdates} per geometry type, nothing committed.
   */
  private sweepLoaders(
    ctxs: UpdateCtxs,
    onArchiveFault: (fault: ArchiveFaultError) => void,
    pick: <T>(loaders: Map<string, T>) => Map<string, T>,
    resyncPaths?: ReadonlySet<string>
  ) {
    const { pointsCtx, linesCtx, gsplatsCtx, meshCtx } = ctxs;
    return Promise.all([
      this.runLoaderUpdates(
        pick(this.loaders),
        pointsLabel,
        (path, loader, session) => pointsLoadAndStage(path, loader, session, pointsCtx),
        onArchiveFault,
        resyncPaths
      ),
      this.runLoaderUpdates(
        pick(this.linesLoaders),
        linesLabel,
        (path, loader, session) => linesLoadAndStage(path, loader, session, linesCtx),
        onArchiveFault,
        resyncPaths
      ),
      this.runLoaderUpdates(
        pick(this.gsplatLoaders),
        gsplatsLabel,
        (path, loader, session) => gsplatsLoadAndStage(path, loader, session, gsplatsCtx),
        onArchiveFault,
        resyncPaths
      ),
      this.runLoaderUpdates(
        pick(this.meshLoaders),
        meshLabel,
        (path, loader, session) => meshLoadAndStage(path, loader, session, meshCtx),
        onArchiveFault,
        resyncPaths
      ),
    ]);
  }

  /** Every registered loader path, of every geometry type. */
  private registeredLoaderPaths(): Set<string> {
    return new Set([
      ...this.loaders.keys(),
      ...this.linesLoaders.keys(),
      ...this.gsplatLoaders.keys(),
      ...this.meshLoaders.keys(),
    ]);
  }

  /**
   * Activate the deferred partition parts this pass's view needs (B4) and, once
   * their loaders are registered, sweep those loaders as part of THIS pass —
   * the ones `alreadySwept` (the main sweep's) does not hold. `null` when no
   * part was activated. Must be called synchronously with the main sweep, so
   * the activation reads the pass's own view. The pass's `signal` is its claim
   * on the activations: a superseded pass withdraws it, so an activation that
   * outlives it resyncs its part rather than waiting on a sweep that never
   * commits.
   */
  private async sweepActivatedParts(
    ctxs: UpdateCtxs,
    onArchiveFault: (fault: ArchiveFaultError) => void,
    resyncPaths: ReadonlySet<string> | undefined,
    alreadySwept: ReadonlySet<string>,
    signal: AbortSignal
  ): Promise<Awaited<ReturnType<SceneLoader['sweepLoaders']>> | null> {
    const registry = this.lodGroupRegistry;
    if (!registry) return null;
    const parts = await registry.activatePartitionParts(this.viewState, resyncPaths, signal);
    if (parts.length === 0) return null;
    const targets = new Set(parts);
    const pick = <T>(loaders: Map<string, T>): Map<string, T> => {
      const fresh = new Map<string, T>();
      for (const [path, loader] of loaders) {
        if (!alreadySwept.has(path) && isUnderAny(path, targets)) fresh.set(path, loader);
      }
      return fresh;
    };
    return this.sweepLoaders(ctxs, onArchiveFault, pick);
  }

  /**
   * Re-run the current view state without blocking the caller. With ``paths``
   * (the partition parts that just re-entered the frustum) the sweep is
   * restricted to loaders at/under them; otherwise every loader re-runs. Either
   * way the view state is unchanged, so the view version is NOT bumped.
   */
  requestReprocess(paths?: readonly string[]): void {
    if (paths && paths.length > 0) {
      this.updateView({}, { resyncPaths: new Set(paths) }).catch((error: unknown) => {
        log.error(Modules.SCENE_LOADER, `View reprocess failed: ${getErrorMessage(error)}`, error);
      });
    } else {
      this.updateView({}).catch((error: unknown) => {
        log.error(Modules.SCENE_LOADER, `View reprocess failed: ${getErrorMessage(error)}`, error);
      });
    }
  }

  /** Re-open exhausted progressive ladders after connectivity is restored. */
  resetRefinementFailures(): boolean {
    const reset = resetRefinementFailureTrackers(
      this.sweepLoaderEntries().map(([, loader]) => loader)
    );
    this.cancelAbandonedRungRetry();
    if (reset) this.requestReprocess();
    return reset;
  }

  /** Every sweep-registered loader of the four geometry kinds, with its path. */
  private sweepLoaderEntries(): Array<[string, RefinableLoader]> {
    return [
      ...this.loaders.entries(),
      ...this.linesLoaders.entries(),
      ...this.gsplatLoaders.entries(),
      ...this.meshLoaders.entries(),
    ] as Array<[string, RefinableLoader]>;
  }

  /**
   * The view state a pass for `viewState` would run: the request merged onto
   * the current `this.viewState`, with the per-pass directives stripped.
   */
  private mergeViewState(viewState: Partial<ViewState>): ViewState {
    // Playback frame budget / ladder depth are PER-PASS directives, never
    // persisted, so a stale budget can't linger in `this.viewState` (which
    // refinement/retry re-derive from) and leave the loaders capped after
    // playback ends. They reach the loaders only via the per-type handler ctxs
    // (buildUpdateCtxs). `prefetch` is likewise transient (set only on the
    // SlicePrefetcher's shadow passes) and must never pin foreground stores.
    const incomingViewState = { ...viewState };
    delete incomingViewState.frameBudgetMs;
    delete incomingViewState.ladderDepth;
    delete incomingViewState.prefetch;
    // CRITICAL: Deep copy arrays to prevent mutation during async operations
    // The spread operator only does shallow copy - arrays must be explicitly copied
    return {
      ...this.viewState,
      ...incomingViewState,
      // Always copy arrays to prevent external mutation affecting in-flight updates
      displayDims: viewState.displayDims
        ? [...viewState.displayDims]
        : [...this.viewState.displayDims],
      slicePosition: viewState.slicePosition
        ? [...viewState.slicePosition]
        : [...this.viewState.slicePosition],
      tolerance: viewState.tolerance ? [...viewState.tolerance] : [...this.viewState.tolerance],
    };
  }

  /**
   * The view half of a JOIN (#2943): the request names a slice (a dims-driven
   * request, not a `{}` reprocess that exists to force a re-commit), neither
   * it nor the running pass carries a playback directive (a budget-free refine
   * must still re-run a budgeted pass), and it merges to exactly the view the
   * running pass is loading. The scheduler adds that a pass is running,
   * unaborted, with nothing newer queued.
   */
  private matchesRunningPass(viewState: Partial<ViewState>): boolean {
    if (!viewState.slicePosition || this._lastUpdateWasFrameBudgeted) return false;
    if (viewState.frameBudgetMs !== undefined || viewState.ladderDepth !== undefined) return false;
    if (viewState.prefetch) return false;
    return viewStatesEqual(this.mergeViewState(viewState), this.viewState);
  }

  /**
   * Update all points and lines for a new view state.
   *
   * Uses serialized execution to prevent race conditions: only one update runs at a time.
   * If a new update arrives while one is in progress, it's queued as "pending" and processed
   * after the current update completes. Only the LATEST pending state is kept (older ones
   * are discarded), ensuring eventual convergence without starvation.
   */
  async updateView(viewState: Partial<ViewState>, opts: UpdateViewOptions = {}): Promise<void> {
    // Disposed loader: never runs another pass. dispose() already flushed the
    // queued-update waiters (resolve-only), so a late call — e.g. an in-flight
    // dimension-animation tick landing during the dispose() await while a
    // refinement pass still holds the update lock — must resolve immediately
    // rather than be queued and park a waiter that nothing will ever settle
    // (the refinement loop's isActive-return exit hands off via a no-op
    // release, and scheduleProgressiveRefinement early-returns on `_disposed`).
    // Resolve-only, never reject (matches dispose()).
    if (this._disposed) return;
    if (this._archiveFault) {
      // A queued re-entry (the scheduler's frame yield or release drain) can
      // land here after a lazy level latched the fault OUTSIDE any pass. Its
      // pending state was already taken, so nothing else will ever settle the
      // waiters parked on it — flush them (resolve-only, like an archive-faulted
      // pass's end and `dispose()`), or the dimension-animation pacing gate and
      // `waitForUpdate()` hang for the rest of the session.
      this.passes.resolveWaiters();
      return;
    }

    // Foreground passes deliberately do NOT abort the background slice
    // prefetch. The foreground now commits from the SliceCache without
    // decoding fine levels (the progressive loaders' `playback` streaming
    // policy), so the shadow deepen must survive across ticks: a cold LOD
    // level outlives one frame, and a per-tick abort would never let it
    // complete + cache a level — playback quality could then never climb
    // across loops. The shadow runs on its own loader instances (own
    // accumulator + signal), decodes on the worker pool, and only writes the
    // shared SliceCache under content-keyed entries, so it cannot corrupt or
    // stall a foreground tick. It is torn down on playback end
    // (`releasePrefetchResources`) and on dispose.

    // SERIALIZATION: while the lock is held the request is parked, joined or
    // queued (latest wins, the superseded work aborted unless the drag chain
    // is owed a commit — B5); its promise settles when a pass of its
    // generation or newer commits. See `PassScheduler.request`.
    const queued = this.passes.request(viewState, opts);
    if (queued) return queued;

    // Take the lock for this pass. The view version is decided below, once
    // the merged view state is known (bump only on a real change). The pass's
    // abort controller is aborted by a superseding request; its signal flows
    // to every chunk read so the superseded load bails, and its `aborted` flag
    // gates the geometry commit below.
    const pass = this.passes.beginPass(viewState);
    const updateController = pass.controller;
    this._passCount++;
    let supersededPassWait: Promise<void> | undefined;
    let sweepArchiveFault: ArchiveFaultError | undefined;
    const onArchiveFault = (fault: ArchiveFaultError): void => {
      sweepArchiveFault ??= fault;
    };

    try {
      // Playback frame budget is a PER-PASS directive, never persisted:
      // destructure it OUT before the merge below so a stale budget can't
      // linger in `this.viewState` (which refinement/retry re-derive from)
      // and leave the loaders capped after playback ends. It flows to the
      // loaders only via the per-type handler ctxs (buildUpdateCtxs).
      const { frameBudgetMs, ladderDepth } = viewState;
      const nextViewState = this.mergeViewState(viewState);
      // Version contract (see `_updateVersion`): bump only when a query
      // determinant changed. An unchanged view (resync / retry / depth-sort
      // re-commit) re-sweeps under the same version, so the lazy LOD levels
      // outside this sweep stay fresh.
      const viewChanged = !viewStatesEqual(nextViewState, this.viewState);
      // An EMPTY target set would skip every loader and commit nothing: treat
      // it as "no restriction" rather than as a silent no-op pass.
      const resyncPaths = viewChanged || !opts.resyncPaths?.size ? undefined : opts.resyncPaths;
      // A targeted resync is not a view pass: leave the budget marker alone so
      // `isAtViewState` cannot skip the real re-run a budget-truncated playback
      // pass still owes. A changed view is always a full sweep, even if stale
      // resync options reached this call through a delayed queue hand-off.
      if (!resyncPaths) {
        this._lastUpdateWasFrameBudgeted = frameBudgetMs !== undefined || ladderDepth !== undefined;
      }
      this.viewState = nextViewState;
      if (viewChanged) {
        this._updateVersion++;
        // A new query owes a ladder retired on the old one a fresh set of
        // attempts: this pass's own refinement drain retries it at once.
        this.cancelAbandonedRungRetry();
        resetRefinementFailureTrackers(this.sweepLoaderEntries().map(([, loader]) => loader));
      }
      const currentVersion = this._updateVersion;

      const totalLoaders =
        this.loaders.size +
        this.linesLoaders.size +
        this.gsplatLoaders.size +
        this.meshLoaders.size;
      if (viewChanged) {
        log.update(
          Modules.SCENE_LOADER,
          `Updating view v${currentVersion} for ${totalLoaders} loaders`
        );
      } else {
        const scope = resyncPaths
          ? `${resyncPaths.size} target path(s): ${[...resyncPaths].join(', ')}`
          : `all ${totalLoaders} loaders`;
        log.update(Modules.SCENE_LOADER, `Resyncing view v${currentVersion} (${scope})`);
      }

      // Start profiling update cycle
      this.profiler?.beginUpdate();

      // Cache for extended tolerance arrays — nodes with the same extend_to_all
      // dimensions share a single tolerance array instead of each copying their own.
      const extendedToleranceCache = new Map<string, number[]>();

      // ================================================================
      // Stage 1: Load + Process all nodes in parallel (async)
      // Each callback returns staged commit data WITHOUT mutating geometry,
      // so all nodes are ready before any geometry changes. Shared
      // scaffolding (try/catch + failedLoaders bookkeeping + profiler
      // dispatch) lives in runLoaderUpdates; each branch below contains
      // only the type-specific work.
      // ================================================================

      // Per-type handler-ctx construction lives in
      // scene-loader/update-view/build-update-ctxs.ts.
      const { pointsCtx, linesCtx, gsplatsCtx, meshCtx } = buildUpdateCtxs({
        rootGroup: this.rootGroup,
        viewStateQueue: this.viewStateQueue,
        clearFailure: (path) => this.failedLoaders.delete(path),
        currentVersion,
        // The processors' "first update" log gate — a PASS counter, not the
        // view version (which no longer moves on a same-view pass).
        updateVersion: this._passCount,
        extendedToleranceCache,
        signal: updateController.signal,
        frameBudgetMs,
        ladderDepth,
        deriveNodeViewState: (path, attrs, opts) => this.deriveNodeViewState(path, attrs, opts),
      });

      const ctxs = { pointsCtx, linesCtx, gsplatsCtx, meshCtx };
      // Deferred partition parts this view needs (B4) are activated alongside
      // the sweep. An activation only REGISTERS their loaders; this pass sweeps
      // them itself, with its own directives, and commits them with everything
      // else. They stay hidden until the commit (`applyCommittedSlice`).
      const sweptPaths = this.registeredLoaderPaths();
      const [swept, activated] = await Promise.all([
        this.sweepLoaders(ctxs, onArchiveFault, (loaders) => loaders, resyncPaths),
        this.sweepActivatedParts(
          ctxs,
          onArchiveFault,
          resyncPaths,
          sweptPaths,
          updateController.signal
        ),
      ]);
      const staged = activated ? concatSweeps(swept, activated) : swept;

      if (sweepArchiveFault) {
        this.reportArchiveFault(sweepArchiveFault);
      }

      // S6: predictive prefetch now lives inside each loader-task
      // branch (Points / Lines / GSplats) and uses the per-node
      // derived view-state computed by deriveNodeViewState. The
      // global dispatch site previously here over-prefetched
      // fully-extended nodes.

      // ================================================================
      // Stage 2: Atomic commit — ALL geometry mutations in one sync block.
      // Implementation in scene-loader/update-view/atomic-commit.ts.
      // ================================================================
      // A failing node commit is rethrown only AFTER every sibling committed
      // (an AggregateError), so the frame IS on screen: hold the error until
      // the post-commit bookkeeping below has recorded it (A9).
      let commitFailure: { error: unknown } | null = null;
      try {
        this.runPassCommit(staged, updateController.signal, sweepArchiveFault !== undefined);
      } catch (error) {
        commitFailure = { error };
      }

      // Post-commit bookkeeping is meaningful only for a committed frame. A
      // superseded update committed nothing (and its loaders may have aborted
      // mid-attribute) — skip the monitor refresh and the failed-loader
      // warning; the winning update runs both with correct state. A partially
      // failed commit DID commit its siblings, so it is recorded like any other.
      if (!updateController.signal.aborted && !sweepArchiveFault) {
        // This view is now what is drawn: re-gate partition parts on its slice
        // before anything renders or refinement reads the stamps (B4).
        this._committedViewState = this.viewState;
        this.passes.noteCommitted();
        this.lodGroupRegistry?.applyCommittedSlice();

        // Update monitor with total visible segments across all lines nodes
        this.updateVisibleCountsInMonitor();

        // Warn user if any loaders failed (shared with the end-of-load report).
        warnFailedLoaders(Array.from(this.failedLoaders.keys()));
      }
      if (commitFailure !== null) throw commitFailure.error;
    } finally {
      // End profiling update cycle (always, even if errors)
      this.profiler?.endUpdate();
      // Settle or carry this pass's waiters, then run what comes next: a
      // pending state after one frame, the refinement drain, or a release.
      // A superseded direct caller waits for the pass that supersedes it.
      supersededPassWait = this.passes.endPass(pass, {
        discarded: sweepArchiveFault !== undefined,
        faulted: this._archiveFault !== null,
      });
    }
    await supersededPassWait;
  }

  /**
   * Stage 2 of a view pass: the synchronous atomic commit of every staged node
   * (`update-view/atomic-commit.ts`). Skipped wholesale when the pass was
   * superseded (`signal`) or hit an archive fault (`discard`); a failing node
   * commit is rolled back to its pass start and rethrown after its siblings.
   */
  private runPassCommit(
    [pointsStaged, linesStaged, gsplatsStaged, meshStaged]: Awaited<
      ReturnType<SceneLoader['sweepLoaders']>
    >,
    signal: AbortSignal,
    discard: boolean
  ): void {
    runAtomicCommit(pointsStaged, linesStaged, gsplatsStaged, meshStaged, {
      gpuBufferPool: this._gpuBufferPool,
      nodeFactory: this.nodeFactory,
      // When this update was superseded mid-flight, skip the geometry
      // commits so a stale/partial frame never reaches the GPU; profiler
      // sessions are still ended inside runAtomicCommit regardless. Keep
      // abort reserved for supersession/cancellation so profiler and log
      // semantics stay accurate; archive faults discard independently.
      signal,
      discard,
      updatePointsGeometry: (path, data, session) => this.updatePointsGeometry(path, data, session),
      commitLinesGeometry: (staged, session) => this.commitLinesGeometry(staged, session),
      commitGSplatsGeometry: (staged, session) => this.commitGSplatsGeometry(staged, session),
      commitMeshGeometry: (staged, session) => this.commitMeshGeometry(staged, session),
      onCommitFailed: (path) => {
        const kind = this.registry.getLoaderType(path);
        const loader = kind ? this.registry.loadersOf(kind).get(path) : undefined;
        // Like every other rollback site: a throwing rollback is logged and
        // the commit failure stays the one reported.
        if (loader) tryRollbackToPassStart(loader as { rollbackToPassStart?: () => number });
      },
    });
  }

  /** Snapshot every sweep-registered progressive ladder, including completed ones. */
  private progressiveLadderResidencies(): Map<string, LadderResidency> {
    const residencies = new Map<string, LadderResidency>();
    const collect = (loaders: ReadonlyMap<string, unknown>) => {
      for (const [path, loader] of loaders) {
        const ladderResidency = (loader as { ladderResidency?: () => LadderResidency })
          .ladderResidency;
        if (ladderResidency) residencies.set(path, ladderResidency.call(loader));
      }
    };
    collect(this.gsplatLoaders);
    collect(this.loaders);
    collect(this.linesLoaders);
    collect(this.meshLoaders);
    return residencies;
  }

  /**
   * Kick the progressive refinement orchestrator from OUTSIDE an update pass.
   *
   * Refinement is normally scheduled only at view-pass tails and after the
   * initial scene load — loaders that register OUTSIDE those moments otherwise
   * sit at their first additive chunk until the next slice change. The one such
   * registration path is a deferred lod_group SUBTREE activation (e.g. the
   * ``overview`` recipe's fine ``kind=partition`` branch): its part leaves join
   * the sweep maps mid-session, so the deferred-group ``ensureLoaded`` calls
   * this after ``loadChildren`` settles. Two recovery paths use it too (#2975):
   * a failed-loader retry that committed only part of a ladder, and the backoff
   * re-drain of ladders the consecutive-failure cap retired.
   *
   * When idle it takes the lock and runs the orchestrator; when busy it
   * re-checks on a timer. "Busy" is the lock OR a live refinement run: the
   * final phase's release opens the lock while the run is still unwinding, and
   * a second run started in that window would share the first one's flags.
   * See `PassScheduler.kickRefinementIfIdle`.
   */
  kickRefinementIfIdle(): void {
    this.passes.kickRefinementIfIdle();
  }

  /**
   * Drain every progressive ladder (points, lines, gsplats and mesh) one rung
   * per frame, under the serialization lock the caller handed over.
   *
   * The phases run in sequence — gsplats first (its loader was the original
   * template), then points, lines and mesh — each over the load-eligible
   * loaders of its type at the moment it starts (`kickRefinementIfIdle` covers
   * loaders registered later). On cancellation (a view queued during the run)
   * the cancelling phase hands the lock to the next pass across a frame and the
   * later phases are skipped. A failed-loader retry pre-empts the run the same
   * way (A10): `isActive` turns false, every phase exits, and the run's end
   * hands the still-held lock to the retry. Only the LAST phase completing
   * releases the lock. The per-pass timing — frame yield, cancellation check,
   * stall exit — lives in `progressive/refinement.ts`.
   */
  private async scheduleProgressiveRefinement(): Promise<void> {
    if (this._disposed || this._archiveFault) {
      // Do not release the serialization lock here. Production lock-owning callers
      // enter this method synchronously before a fault can interleave; the faulting
      // update ends its pass without scheduling refinement, so this guard cannot
      // follow a lock handoff.
      return;
    }

    // Per-run abort controller, published as THIS loader's live controller: a
    // superseding view (and a pre-empting retry, and dispose) aborts it, so an
    // incoming view-state cancels in-flight refinement chunk reads MID-PASS
    // instead of waiting for the pass to finish. The refinement catches treat
    // the resulting AbortError as cancellation (no failure recorded); the
    // loop's next-pass pending check performs the hand-off. Every read of the
    // run rides this signal, so it also carries the run's fetch class: a finer
    // level of something already on screen queues behind every `demand` read
    // (a frame waiting), ahead of speculation (B9c).
    const controller = new AbortController();
    tagSignalPriority(controller.signal, 'refinement');
    // Mark the whole run as REFINEMENT, not as a load pass: the lock was handed
    // over by a pass tail / post-load kick that had already committed the
    // current view, so `isLoadPassInProgress()` must not see it. Ended in the
    // `finally`, so every exit path — cancellation hand-off, disposal, throw —
    // clears it.
    this.passes.beginRefinement(controller);
    let cancelled = false;
    try {
      const isActive = () => !this._disposed && !this.passes.retryPreemptPending;
      const shared = {
        viewStateQueue: this.viewStateQueue,
        updateVisibleCountsInMonitor: () => this.updateVisibleCountsInMonitor(),
        retriggerUpdate: (pendingState: Partial<ViewState>) => {
          cancelled = true;
          this.passes.handOff(pendingState);
        },
        isActive,
        signal: controller.signal,
        profiler: this.profiler,
        residencyBudget: this.beginRefinementBudget(),
      };
      const phases = this.refinementPhases();
      for (const [i, runPhase] of phases.entries()) {
        // Intermediate phases keep the lock (the next phase owns it); only the
        // last one completing releases it.
        const releaseLock =
          i === phases.length - 1
            ? () => this.passes.releaseRefinementLock()
            : () => {
                /* lock stays held; next phase owns it */
              };
        await runPhase({ ...shared, releaseLock });
        if (cancelled || !isActive()) break;
      }
      this.noteRefinementOutcome(cancelled);
    } finally {
      this.passes.endRefinement(cancelled);
    }
  }

  /**
   * Open the run's shared residency budget: ONE budget for all four phases,
   * seeded from every sweep-registered ladder. Completed loaders return before
   * their wrapper calls admit(), so seeding is what keeps their resident bytes
   * in later view-triggered runs instead of ratcheting the ceiling upward. Lazy
   * lod_group levels are not in these sweep maps. The density gate is
   * re-evaluated from scratch each run: a deferral is camera-dependent, never
   * sticky across runs.
   */
  private beginRefinementBudget(): RefinementResidencyBudget {
    noteRefinementStarted();
    this.refinementDensityGate?.beginRun();
    const residencyBudget = RefinementResidencyBudget.forSession(
      this.progressiveLadderResidencies(),
      cachePoolOverrideBytes(this.config.cacheBudgetMB),
      deviceClassPoolBytes(),
      this.refinementResidencyReporter,
      this.refinementDensityGate
    );
    this.lastResidencyBudget = residencyBudget;
    return residencyBudget;
  }

  /**
   * The four refinement phases in run order, each resolving its type's
   * load-eligible loaders when it STARTS (a snapshot, not the live map).
   *
   * Mesh runs LAST and therefore releases the lock. Order within the four is
   * otherwise historical (gsplats was the template), but the final slot is not
   * arbitrary: a mesh reveal is the cheapest of the four to interrupt — its
   * levels are already-decoded whole-node payloads, so a cancelled pass loses
   * at most one projection rather than an in-flight chunk fetch.
   */
  private refinementPhases(): Array<(shared: RefinementPhaseShared) => Promise<void>> {
    const derive = (
      path: string,
      attrs: { extend_to_all?: string[] } | undefined,
      opts: { applyPartialExtendTolerance: boolean }
    ) => this.deriveNodeViewState(path, attrs, opts);
    return [
      (shared) =>
        runGSplatsRefinement({
          ...shared,
          ...resolveLoadEligibleLoaders(this.rootGroup, this.gsplatLoaders),
          deriveNodeViewState: derive,
          processGSplats: (path, data, viewState, session, signal) =>
            this.processGSplatsData(path, data, viewState, session, signal),
          commitGSplats: (staged, session) => this.commitGSplatsGeometry(staged, session),
        }),
      (shared) =>
        runPointsRefinement({
          ...shared,
          ...resolveLoadEligibleLoaders(this.rootGroup, this.loaders),
          deriveNodeViewState: derive,
          updatePointsGeometry: (path, data, session) =>
            this.updatePointsGeometry(path, data, session),
        }),
      (shared) =>
        runLinesRefinement({
          ...shared,
          ...resolveLoadEligibleLoaders(this.rootGroup, this.linesLoaders),
          deriveNodeViewState: derive,
          processLines: (path, data, viewState, session, signal) =>
            this.processLinesData(path, data, viewState, session, signal),
          commitLines: (staged, session) => this.commitLinesGeometry(staged, session),
        }),
      (shared) =>
        runMeshRefinement({
          ...shared,
          ...resolveLoadEligibleLoaders(this.rootGroup, this.meshLoaders),
          deriveNodeViewState: derive,
          processMesh: (path, data, viewState, attrs) =>
            this.processMeshData(path, data, viewState, attrs),
          commitMesh: (staged, session) => this.commitMeshGeometry(staged, session),
        }),
    ];
  }

  /**
   * Stamp the load timeline's `refinementComplete` milestone when the final
   * geometry phase ran every ladder to completion — not on a cancellation
   * hand-off (the next update re-kicks refinement) and not on a dead loader.
   */
  private noteRefinementOutcome(cancelled: boolean): void {
    // A retry pre-empting the run (A10) is a cancellation too: its resume re-kicks.
    if (cancelled || this._disposed || this.passes.retryPreemptPending) return;
    // A ladder retired mid-way is still owed its rungs: stay unsettled while
    // its re-drain is pending, so capture / isSettled do not report done.
    if (this.scheduleAbandonedRungRetry()) return;
    noteRefinementComplete();
  }

  /**
   * Re-drain ladders the consecutive-failure cap retired (#2975).
   *
   * A chunk fetch that stalls to give-up fails the refinement step that needs
   * it, and three such failures retire the ladder for this loader instance.
   * Only an `online` event used to re-open it, so a session that stayed online
   * sat idle at a coarser rung until the user moved something. Instead, after
   * a drain that ends with such a ladder, wait
   * `ABANDONED_RUNG_RETRY_BASE_MS · 2^round` (capped), re-open the retired
   * ladders and kick a fresh drain. A view change, `resetRefinementFailures`
   * or dispose cancels the wait; after `MAX_ABANDONED_RUNG_RETRY_ROUNDS`
   * rounds the ladder is left, and the drain reports complete.
   *
   * @returns Whether a re-drain is pending (the drain is not complete).
   */
  private scheduleAbandonedRungRetry(): boolean {
    const entries = this.sweepLoaderEntries();
    const abandoned = abandonedLadderPaths(entries);
    // A ladder no longer retired (it recovered, or left the sweep) starts over.
    for (const path of this._abandonedRetryRounds.keys()) {
      if (!abandoned.includes(path)) this._abandonedRetryRounds.delete(path);
    }
    if (this._abandonedRetryTimer !== null) return abandoned.length > 0;
    const due = this.abandonedLaddersStillOwedARound(abandoned);
    if (due.length === 0) return false;
    const rounds = due.map((path) => this._abandonedRetryRounds.get(path) ?? 0);
    const delayMs = Math.min(
      ABANDONED_RUNG_RETRY_BASE_MS * 2 ** Math.min(...rounds),
      ABANDONED_RUNG_RETRY_MAX_MS
    );
    for (const [i, path] of due.entries()) this._abandonedRetryRounds.set(path, rounds[i] + 1);
    log.info(
      Modules.SCENE_LOADER,
      `Re-draining abandoned LOD rung(s) of ${due.join(', ')} in ${delayMs} ms ` +
        `(round ${Math.min(...rounds) + 1}/${MAX_ABANDONED_RUNG_RETRY_ROUNDS})`
    );
    const dueSet = new Set(due);
    this._abandonedRetryTimer = setTimeout(() => {
      this._abandonedRetryTimer = null;
      if (this._disposed) return;
      if (this._archiveFault) {
        noteRefinementAborted();
        return;
      }
      // Only the ladders this round is for: one that gave up stays retired.
      reopenRefinementFailureTrackers(
        this.sweepLoaderEntries()
          .filter(([path]) => dueSet.has(path))
          .map(([, loader]) => loader)
      );
      if (!this.registry.anyHasMoreLODs()) noteRefinementComplete();
      else this.kickRefinementIfIdle();
    }, delayMs);
    return true;
  }

  /**
   * The retired ladders still under {@link MAX_ABANDONED_RUNG_RETRY_ROUNDS};
   * one that just ran out is left, with a single warning.
   */
  private abandonedLaddersStillOwedARound(abandoned: readonly string[]): string[] {
    const due: string[] = [];
    for (const path of abandoned) {
      const round = this._abandonedRetryRounds.get(path) ?? 0;
      if (round < MAX_ABANDONED_RUNG_RETRY_ROUNDS) {
        due.push(path);
      } else if (round === MAX_ABANDONED_RUNG_RETRY_ROUNDS) {
        log.warning(
          Modules.SCENE_LOADER,
          `Leaving LOD refinement of ${path} after ${round} re-drains ` +
            '(a view change or restored connectivity retries it)'
        );
        this._abandonedRetryRounds.set(path, round + 1);
      }
    }
    return due;
  }

  /** Drop a pending re-drain and its backoff (the retired ladders are re-opened elsewhere). */
  private cancelAbandonedRungRetry(): void {
    if (this._abandonedRetryTimer !== null) clearTimeout(this._abandonedRetryTimer);
    this._abandonedRetryTimer = null;
    this._abandonedRetryRounds.clear();
  }

  /**
   * Clear the `committedData` identity stamp on a node's mesh. Called when a
   * lazy LOD level is demoted: its geometry returned to the evictable pool,
   * so the stamp (a) no longer describes what's on the GPU and (b) would pin
   * the released node's large CPU arrays in memory. Re-promotion builds a
   * fresh loader → new data reference → full recommit either way.
   */
  private clearCommittedDataStamp(path: string): void {
    const mesh = findObjectByName(this.rootGroup, path);
    if (mesh) {
      clearCommittedData(mesh);
    }
  }

  /**
   * Aggregate visible counts from all lines and gsplats meshes and update monitor.
   * This should be called after view updates to report accurate visible counts.
   */
  private updateVisibleCountsInMonitor(): void {
    updateVisibleCountsInMonitorHelper(this.rootGroup, this.monitor);
  }

  /**
   * Recompute the monitor's visible-element tally outside the data-load
   * cycle. Substitutive LOD selection (`LODGroupRegistry.evaluatePerFrame`)
   * swaps which level renders on camera moves with no reload, so the
   * per-frame callback calls this after a LOD switch — otherwise the
   * monitor's "visible" counts stay pinned to the level that was active at
   * the last `updateView` (e.g. the coarsest default level).
   */
  public refreshVisibleCounts(): void {
    this.updateVisibleCountsInMonitor();
  }

  /**
   * Process lines data: compute tolerance, project to 3D (async).
   * Returns staged commit data without mutating any mesh geometry.
   *
   * Implementation lives in `scene-loader/process/data-processor-lines.ts`; this
   * method is a thin delegate so the pipeline can be tested in isolation
   * without instantiating a SceneLoader.
   */
  private async processLinesData(
    path: string,
    data: LoadedLinesData,
    viewState: LinesViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<StagedLinesCommit | null> {
    return this.withDatasetSignal(signal, (raced) =>
      processLinesDataHelper(
        path,
        data,
        viewState,
        this.rootGroup,
        // Background refinement/retry is never the first projection, even when
        // the initial view has not incremented `_passCount` yet.
        Math.max(2, this._passCount),
        session,
        raced
      )
    );
  }

  /**
   * Run a worker-backed step under `signal` AND this loader's dataset signal
   * (`dataset`, the current one by default), so disposing the dataset settles
   * it even when no per-update signal was given (initial build, retry). The
   * worker pool is a page singleton shared by every host, so the dataset
   * signal travels with the call rather than living on the pool.
   */
  private async withDatasetSignal<T>(
    signal: AbortSignal | undefined,
    run: (signal: AbortSignal | undefined) => Promise<T>,
    dataset: AbortController | null = this._datasetAbortController
  ): Promise<T> {
    const scope = combineAbortSignals(dataset?.signal, signal);
    try {
      return await run(scope?.signal);
    } finally {
      scope?.dispose();
    }
  }

  /**
   * Commit lines geometry to GPU buffers (synchronous).
   * Called as part of the atomic commit stage — no async operations allowed.
   * The optional `session` is forwarded so the helper can record an
   * "Update Buffers" child entry under the per-node profiler session
   * (parity with Points and GSplats).
   */
  private commitLinesGeometry(
    staged: StagedLinesCommit,
    session?: UpdateSession,
    loadedViewVersion: number = this._updateVersion
  ): boolean {
    this.lodGroupRegistry?.invalidatePartitionFootprint(staged.path);
    // Wake the idle-paused render loop so this commit paints (see
    // _requestRender).
    return this.commitAndRequestRender(staged.path, () =>
      commitLinesGeometryHelper(staged, this.commitHost(), session, loadedViewVersion)
    );
  }

  /**
   * Process gsplats data: project nD to 3D, pack Cholesky factors (async).
   * Returns staged commit data without mutating any mesh geometry.
   *
   * Implementation lives in `scene-loader/process/data-processor-gsplats.ts`.
   */
  private async processGSplatsData(
    path: string,
    data: LoadedGSplatsData,
    viewState: GSplatsViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<StagedGSplatsCommit | null> {
    return this.withDatasetSignal(signal, (raced) =>
      processGSplatsDataHelper(
        path,
        data,
        viewState,
        this.rootGroup,
        // Background refinement/retry is never the first projection, even when
        // the initial view has not incremented `_passCount` yet.
        Math.max(2, this._passCount),
        session,
        raced
      )
    );
  }

  /**
   * Commit gsplats geometry to GPU buffers (synchronous).
   * Called as part of the atomic commit stage — no async operations allowed.
   * The optional `session` is forwarded so the helper can record an
   * "Update Buffers" child entry under the per-node profiler session
   * (parity with Points and Lines).
   */
  private commitGSplatsGeometry(
    staged: StagedGSplatsCommit,
    session?: UpdateSession,
    loadedViewVersion: number = this._updateVersion
  ): boolean {
    this.lodGroupRegistry?.invalidatePartitionFootprint(staged.path);
    // Wake the idle-paused render loop so this commit paints (see
    // _requestRender).
    return this.commitAndRequestRender(staged.path, () =>
      commitGSplatsGeometryHelper(staged, this.commitHost(), session, loadedViewVersion)
    );
  }

  /**
   * Project + stage a mesh for commit.
   *
   * Implementation lives in `scene-loader/process/data-processor-mesh.ts`. Async
   * only because backend selection is, so unlike the lines/gsplats twins this never
   * resolves to `null` — there is no worker projection that can decline.
   */
  private processMeshData(
    path: string,
    data: LoadedMeshData,
    viewState: MeshViewState,
    attrs: Pick<MeshMetadata, 'normal_dims' | 'double_sided' | 'extend_to_all' | 'slab_tolerance'>
  ): Promise<StagedMeshCommit> {
    return processMeshDataHelper(path, data, viewState, attrs);
  }

  /**
   * Commit mesh geometry (synchronous).
   *
   * Implementation lives in `scene-loader/commit/commit-mesh-geometry.ts`. Mesh
   * buffers are not pooled, but each commit registers their resident bytes for
   * the shared GPU budget.
   */
  private commitMeshGeometry(
    staged: StagedMeshCommit,
    session?: UpdateSession,
    loadedViewVersion: number = this._updateVersion
  ): boolean {
    this.lodGroupRegistry?.invalidatePartitionFootprint(staged.path);
    return this.commitAndRequestRender(staged.path, () =>
      commitMeshGeometryHelper(
        {
          rootGroup: this.rootGroup,
          currentVersion: this._updateVersion,
          gpuBufferPool: this._gpuBufferPool,
          depthSort: this._depthSort,
        },
        staged,
        session,
        loadedViewVersion
      )
    );
  }

  /**
   * Build the per-call NodeBuildCtx for the initial-load leaf helpers.
   * Snapshots viewState + factoryDeps so a concurrent updateView can't
   * mutate state mid-flight. Never passes `this`.
   */
  private makeNodeBuildCtx(): NodeBuildCtx {
    // Capture the dataset's AbortController by reference at ctx-build
    // time. `loadScene` aborts + replaces this controller on the next
    // load and `dispose()` nulls it, so a deferred load created under
    // this dataset can detect (via identity + aborted flag) that its
    // dataset is no longer live and skip committing into a stale scene.
    const ctrl = this._datasetAbortController;
    // The eager load commits for the view it is built under (B4).
    if (this._committedViewState === null) {
      this._committedViewState = this.viewState;
      this.passes.noteCommitted();
    }
    return {
      registry: this.registry,
      lineWorkingSetGate: this.lineWorkingSetGate,
      lodGroupRegistry: this.lodGroupRegistry ?? undefined,
      nodeFactory: this.nodeFactory,
      viewState: this.viewState,
      getSliceView: () => this.viewState,
      pathHasNdTransform: (path) => this._sceneNodeIndex?.hasNdTransform(path) === true,
      factoryDeps: this.factoryDeps(),
      onLeafMaterialized: (path, object) => {
        if (this._sceneGraph) this._leafMaterializedListener?.(this._sceneGraph, path, object);
      },
      isDatasetLive: () => this._datasetAbortController === ctrl && ctrl?.signal.aborted !== true,
      releaseLazyGSplats: (path) => {
        // Return the level's GPU buffer to the evictable pool. Its loader was
        // never registered (lazy levels stay out of the sweep), so there is
        // nothing to unregister.
        this._gpuBufferPool?.releaseGSplatsGeometry(path);
        this.clearCommittedDataStamp(path);
        // Also drop the level's depth-sort state + worker-side centers: a
        // demoted level won't sort again until re-promotion re-registers it
        // (fresh commit → `depthSort.noteCommit`). Mirrors the coordinator's
        // empty-commit release hygiene.
        const mesh = findObjectByName(this.rootGroup, path);
        if (mesh) releaseDepthSortNode(mesh as THREE.Mesh);
      },
      releaseLazyPoints: (path) => {
        // Points peer of releaseLazyGSplats: return the level's GPU buffer to
        // the evictable pool. Re-selection reloads via the lod_group's
        // ensureLoaded thunk (cheap re-projection from cached chunks).
        this._gpuBufferPool?.releasePointsGeometry(path);
        this.clearCommittedDataStamp(path);
        // Drop the level's depth-sort state + worker-side centers — the
        // same demotion hygiene as the gsplats branch above.
        const mesh = findObjectByName(this.rootGroup, path);
        if (mesh) releaseDepthSortNode(mesh as THREE.Mesh);
      },
      releaseLazyLines: (path) => {
        // Lines peer of releaseLazyGSplats/releaseLazyPoints.
        this._gpuBufferPool?.releaseLinesGeometry(path);
        this.clearCommittedDataStamp(path);
        // Drop the level's depth-sort state + worker-side midpoints —
        // the same demotion hygiene as the gsplats/points branches above.
        const mesh = findObjectByName(this.rootGroup, path);
        if (mesh) releaseDepthSortNode(mesh as THREE.Mesh);
      },
      releaseLazyMesh: (path) => {
        this.clearCommittedDataStamp(path);
        const mesh = findObjectByName(this.rootGroup, path);
        if (mesh) {
          const level = mesh as THREE.Mesh;
          // Mesh is depthSortable (#1347); demotion must drop coordinator
          // state and worker-side centroids along with its geometry.
          releaseDepthSortNode(level);
          level.geometry.dispose();
          // The geometry owns the uploaded texture's disposal hook. Forget its
          // cached identity so the next commit creates a live texture again.
          // Materials retain the disposed texture while this level is hidden;
          // recommit replaces it before the level can be shown again.
          delete level.userData.meshTexture;
          delete level.userData.meshTextureSource;
          level.geometry = createEmptyMeshGeometry(level.userData.attrs as MeshMetadata);
          invalidateRenderObjectFor(level);
        }
      },
      applyEffectiveAttrs: (node) => this.applyEffectiveAttrs(node),
      deriveNodeViewState: (path, attrs, opts) => this.deriveNodeViewState(path, attrs, opts),
      connectLoaderToMonitor: (path, loader) => this.connectLoaderToMonitor(path, loader),
      kickRefinementIfIdle: () => this.kickRefinementIfIdle(),
      reportArchiveFault: (fault) => this.reportArchiveFault(fault),
      // Live version accessor (not the snapshot) so a deferred / registry-driven
      // reload stamps for the CURRENT slice, not the one captured at ctx-build.
      getViewVersion: () => this._updateVersion,
      // Commit callbacks forward an explicit loadedViewVersion so the lazy /
      // reload path can stamp its DERIVE-time version (see commitPointsGeometry).
      processPointsData: (path, data) => this.processPointsData(path, data),
      commitPointsGeometry: (staged, session, loadedViewVersion) =>
        this.commitPointsGeometry(staged, session, loadedViewVersion),
      // Under the dataset captured at ctx-build time, like `isDatasetLive`.
      processLinesData: (path, data, viewState, session, signal) =>
        this.withDatasetSignal(
          signal,
          (raced) =>
            // Initial node build is the one path allowed to log a first projection.
            processLinesDataHelper(
              path,
              data,
              viewState,
              this.rootGroup,
              this._passCount,
              session,
              raced
            ),
          ctrl
        ),
      commitLinesGeometry: (staged, session, loadedViewVersion) =>
        this.commitLinesGeometry(staged, session, loadedViewVersion),
      processGSplatsData: (path, data, viewState, session, signal) =>
        this.withDatasetSignal(
          signal,
          (raced) =>
            processGSplatsDataHelper(
              path,
              data,
              viewState,
              this.rootGroup,
              this._passCount,
              session,
              raced
            ),
          ctrl
        ),
      commitGSplatsGeometry: (staged, session, loadedViewVersion) =>
        this.commitGSplatsGeometry(staged, session, loadedViewVersion),
      processMeshData: (path, data, viewState, attrs) =>
        this.processMeshData(path, data, viewState, attrs),
      commitMeshGeometry: (staged, session, loadedViewVersion) =>
        this.commitMeshGeometry(staged, session, loadedViewVersion),
    };
  }

  /** Build the per-call dependency snapshot for the loader factory. */
  private factoryDeps(): LoaderFactoryDeps {
    return {
      zarrStore: this._zarrStore!,
      arrayRefRegistry: this.arrayRefRegistry,
      l0Cache: this.l0Cache,
      sliceCache: this.sliceCache,
      cachingStore: this.cachingStore,
      decodeKTX2: this.decodeKTX2,
    };
  }

  /**
   * Connect a loader to the data-loading monitor. Implementation lives
   * in `scene-loader/nodes/connect-loader-to-monitor.ts`.
   */
  private connectLoaderToMonitor(
    path: string,
    loader: DataLoader | LinesDataLoader | GSplatsDataLoader | MeshDataLoader
  ): void {
    connectLoaderToMonitorHelper(path, loader, this.monitor);
  }

  /**
   * Update geometry for a specific points node.
   *
   * Implementation lives in `scene-loader/commit/commit-points-geometry.ts`.
   */
  private updatePointsGeometry(
    path: string,
    data: LoadedPointsData,
    session?: UpdateSession,
    // The view-version this geometry was loaded for, stamped onto the mesh for
    // the LOD slice-aware fallback. Defaults to the current ``_updateVersion``:
    // correct for the per-slice sweep (stable within an abort-guarded
    // updateView). The lazy/reload path passes its DERIVE-time version
    // explicitly so a load that finishes after a further scrub is stamped for
    // the slice it actually loaded (the registry then re-reloads for the newer
    // version) rather than being mis-stamped fresh.
    loadedViewVersion: number = this._updateVersion
  ): boolean {
    this.lodGroupRegistry?.invalidatePartitionFootprint(path);
    // Wake the idle-paused render loop so this commit paints (see
    // _requestRender).
    return this.commitAndRequestRender(path, () =>
      commitPointsGeometryHelper(
        path,
        data,
        { ...this.commitHost(), nodeFactory: this.nodeFactory },
        session,
        loadedViewVersion
      )
    );
  }

  /**
   * Stage points data for commit — the Points arm of the shared
   * `process`/`commit` pair. A pass-through: points arrive display-ready from
   * their loader (see `data-processor-points.ts`).
   */
  private processPointsData(path: string, data: LoadedPointsData): StagedPointsCommit {
    return processPointsDataHelper(path, data);
  }

  /**
   * Commit staged points data. Delegates to {@link updatePointsGeometry} so the
   * one-shot and two-stage forms cannot diverge.
   */
  private commitPointsGeometry(
    staged: StagedPointsCommit,
    session?: UpdateSession,
    loadedViewVersion: number = this._updateVersion
  ): void {
    this.updatePointsGeometry(staged.path, staged.data, session, loadedViewVersion);
  }

  /**
   * Initialize scene dimensions from metadata. Implementation lives in
   * `scene-loader/nodes/initialize-scene-dimensions.ts`; null return
   * means validation failed and the existing viewState stays.
   */
  private initializeSceneDimensions(sceneDims: unknown): void {
    const next = initializeSceneDimensionsHelper(sceneDims);
    if (next) this.viewState = next;
  }

  /**
   * Whether `candidate` describes the view this loader has ALREADY committed
   * (same displayed dims, slice, tolerances and per-dim query signature, by
   * `viewStatesEqual`). Used by `updateSceneForDimensions` to turn the
   * post-load `updateAllNDNodes` — fired unconditionally after `loadScene`
   * has fetched, decoded, projected and committed every node at exactly this
   * state — into a no-op instead of a second full pass (measured: L0 hits ==
   * misses on every 3-D scene, and ~1 s of extra main-thread work on a
   * 29.6 M-splat slide). A real slider change compares unequal and proceeds.
   *
   * False while a requested main pass has not finished: `this.viewState`
   * names a pass's view as soon as it starts. This also covers the frame
   * boundary between a superseded pass and its queued replacement, when no
   * main pass is running but the old view never committed (#2943).
   */
  isAtViewState(candidate: ViewState): boolean {
    if (this._disposed) return false;
    if (!this.passes.allRequestsCompleted) return false;
    if (this._lastUpdateWasFrameBudgeted) return false;
    return viewStatesEqual(candidate, this.viewState);
  }

  /**
   * Normalize URL for zarr store access. Guards `window` so this works
   * in non-DOM contexts (tests, embed-in-Worker scenarios). Absolute URLs
   * ignore the origin entirely; the fallback only matters for relative
   * paths.
   */
  private normalizeURL(url: string): string {
    const origin = typeof window !== 'undefined' ? window.location.origin : 'http://localhost';
    return normalizeURL(url, origin);
  }

  /**
   * Show the monitor UI
   */
  showMonitor(): void {
    this.monitor?.show();
  }

  /**
   * Hide the monitor UI
   */
  hideMonitor(): void {
    this.monitor?.hide();
  }

  /**
   * Toggle the monitor UI
   */
  toggleMonitor(): void {
    this.monitor?.toggle();
  }

  /**
   * Get information about failed loaders
   * @returns Map of loader paths to error information
   */
  getFailedLoaders(): ReadonlyMap<string, { error: Error; timestamp: number; retryCount: number }> {
    return this.failedLoaders;
  }

  /** Whether a path or one of its descendants has a recorded network failure. */
  hasNetworkFailureUnder(path: string): boolean {
    const prefix = path.endsWith('/') ? path : `${path}/`;
    for (const [failedPath, info] of this.failedLoaders) {
      if (info.kind === 'Network' && (failedPath === path || failedPath.startsWith(prefix))) {
        return true;
      }
    }
    return false;
  }

  /**
   * Build a {@link FailedLoadsProviderPort} over the failed-load set for a UI
   * consumer — the data-monitor's failure banner (wired in `monitor-wiring.ts`)
   * and the layers panel's per-row error badge (wired in
   * `core/app/dataset/load-dataset.ts`). Each call returns a NEW provider object
   * (the monitor and the panel hold distinct instances), but all of them read
   * the SAME live loader failures, archive-fault latch, and latched lazy
   * branches through the one `retryAllFailedLoaders` entry point, so they
   * always agree.
   * `getFailedReason` powers the layers-panel tooltip.
   */
  getFailedLoadsProvider(): FailedLoadsProviderPort {
    return {
      getFailedPaths: () => this.getMonitorFailedPaths(),
      getFailedLoadsVersion: failedLoadsVersion,
      retryAll: () => this.retryAllFailedLoaders(),
      getFailedReason: (path) => {
        const info = this.failedLoaders.get(path);
        if (info) return info.error?.message || info.kind || undefined;
        if (this._archiveFault?.url === path) return this._archiveFault.message;
        return this.lodGroupRegistry?.getFailedLazyChildReason(path);
      },
    };
  }

  private getMonitorFailedPaths(): string[] {
    const failedPaths = new Set([
      ...this.failedLoaders.keys(),
      ...(this.lodGroupRegistry?.getFailedLazyChildPaths() ?? []),
    ]);
    if (failedPaths.size === 0 && this._archiveFault) failedPaths.add(this._archiveFault.url);
    return Array.from(failedPaths);
  }

  private clearArchiveFaultForRetry(): void {
    if (!this._archiveFault) return;
    this._archiveFault = null;
    bumpFailedLoadsVersion();
    notifier.clearError();
  }

  /**
   * Release the retry's lock and resume the view: run what queued meanwhile
   * (including a resync parked during the retry), else reload the whole view
   * after a cleared archive fault, else resume any ladder the retry left short.
   */
  private finishRetry(hadArchiveFault: boolean): void {
    const reloadsWholeView = hadArchiveFault && !this._archiveFault;
    this.passes.releaseRetry(reloadsWholeView);
    if (this.passes.drainPending()) return;
    if (!reloadsWholeView) {
      // A retried progressive loader committed only one pass's worth of its
      // ladder; nothing else resumes the rest (#2975).
      this.kickRefinementIfIdle();
      return;
    }
    void this.updateView(this.viewState).catch((error) => {
      log.warning(
        Modules.SCENE_LOADER,
        `Current view reload after archive retry failed: ${getErrorMessage(error)}`,
        error
      );
    });
  }

  /**
   * Check if there are any failed loaders
   */
  hasFailures(): boolean {
    return this.failedLoaders.size > 0;
  }

  /**
   * Whether any failure is worth an AUTOMATIC retry: a latched archive fault,
   * a transient loader cause still under the attempt cap (see
   * `LoaderRegistry.autoRetryablePaths`), or a lazy branch latched by an archive
   * fault. The connectivity-triggered retry gates on this so deterministic
   * ordinary loader failures remain quiet while reconnecting can re-open the
   * loader and deferred LOD work.
   */
  hasAutoRetryableFailures(): boolean {
    return (
      this._archiveFault !== null ||
      this.registry.hasAutoRetryableFailures() ||
      (this.lodGroupRegistry?.getFailedLazyChildPaths().length ?? 0) > 0
    );
  }

  /**
   * Clear failed loader tracking
   * Useful for retry operations or after user acknowledges errors
   */
  clearFailures(): void {
    this.registry.clearAllFailures();
  }

  /**
   * Retry loading a specific failed loader.
   *
   * Re-triggers the update for a failed loader using the current view
   * state. Useful for recovering from transient network errors or after
   * connectivity is restored.
   *
   * @param path - The path of the failed loader to retry
   * @returns Promise resolving to true if retry succeeded, false if failed or not found.
   *          For a LAZY substitutive LOD level (not in the sweep maps), `true`
   *          means the deferred reload was KICKED (fire-and-forget) — the lazy
   *          thunk owns the eventual ready/failed outcome, and a repeat failure
   *          re-records itself for another retry.
   *          Registered lines paths first wait for the session working-set gate,
   *          so this call can remain pending behind an eager scene walk.
   *
   * @example
   * ```typescript
   * // Retry a specific loader after network recovery
   * const success = await sceneLoader.retryFailedLoader('/points/cloud1');
   * if (success) {
   *   console.log('Loader recovered successfully');
   * }
   * ```
   */
  async retryFailedLoader(path: string): Promise<boolean> {
    const lazyFailure = this.lodGroupRegistry?.getFailedLazyChildPaths().includes(path) === true;
    const archiveFaultFailure = this._archiveFault?.url === path;
    const hadArchiveFault = this._archiveFault !== null;
    if (!this.failedLoaders.has(path) && !lazyFailure && !archiveFaultFailure) {
      log.warning(Modules.SCENE_LOADER, `Path "${path}" is not in failed loaders list`);
      return false;
    }

    // Serialize against the main update path. retry calls
    // `loader.updateView(...)` directly, which would race with
    // updateView()'s own per-loader call for the same path — concurrent
    // zarr fetches and concurrent commits to the same THREE object
    // produce inconsistent state. Both halves are required: refusing
    // when an update is already active AND taking the lock so an
    // updateView starting AFTER retry begins can't race. A refinement drain
    // is pre-empted rather than waited out (A10).
    if (!(await this.passes.acquireForRetry())) {
      log.info(
        Modules.SCENE_LOADER,
        `Retry of ${path} deferred — main update in progress; try again after the update settles`
      );
      return false;
    }

    try {
      if (lazyFailure) {
        this.clearArchiveFaultForRetry();
        return this.lodGroupRegistry?.retryLazyChildByNodePath(path) ?? false;
      }
      if (archiveFaultFailure) {
        this.clearArchiveFaultForRetry();
        return true;
      }
      return await retryFailedLoaderUnlocked(path, this.makeRetryCtx());
    } finally {
      this.finishRetry(hadArchiveFault);
    }
  }

  /** Build the per-call RetryCtx. Never passes `this` to the helper. */
  private makeRetryCtx(): RetryCtx {
    return {
      registry: this.registry,
      lineWorkingSetGate: this.lineWorkingSetGate,
      lodGroupRegistry: this.lodGroupRegistry,
      clearArchiveFault: () => this.clearArchiveFaultForRetry(),
      rootGroup: this.rootGroup,
      deriveNodeViewState: (path, attrs, opts) => this.deriveNodeViewState(path, attrs, opts),
      processPointsData: (path, data) => this.processPointsData(path, data),
      commitPointsGeometry: (staged) => this.commitPointsGeometry(staged),
      processLinesData: (path, data, vs) => this.processLinesData(path, data, vs),
      commitLinesGeometry: (staged) => this.commitLinesGeometry(staged),
      processGSplatsData: (path, data, vs) => this.processGSplatsData(path, data, vs),
      commitGSplatsGeometry: (staged) => this.commitGSplatsGeometry(staged),
      processMeshData: (path, data, vs, attrs) => this.processMeshData(path, data, vs, attrs),
      commitMeshGeometry: (staged) => this.commitMeshGeometry(staged),
    };
  }

  /**
   * Retry all failed loaders. Useful for batch recovery after network
   * connectivity is restored.
   *
   * @returns Promise resolving to `{ succeeded, failed, deferred? }`.
   *          `deferred: true` means NOTHING was retried — a main update held
   *          the serialization lock, so the batch was refused (every path is
   *          reported in `failed` for compatibility, but none genuinely
   *          re-failed). Callers must not present a deferred result as a
   *          failed re-attempt; retry again once the update settles.
   *
   * @example
   * ```typescript
   * // Retry all failed loaders after network recovery
   * const result = await sceneLoader.retryAllFailedLoaders();
   * console.log(`Recovered: ${result.succeeded.length}, Still failing: ${result.failed.length}`);
   * ```
   */
  async retryAllFailedLoaders(
    opts: {
      /**
       * Retry only entries that pass the automatic-retry filter (a latched
       * archive fault, or a transient cause under the attempt cap). Set by the
       * connectivity-triggered retry. A manual Retry omits it and forces every
       * failed path.
       */
      onlyAutoRetryable?: boolean;
    } = {}
  ): Promise<{
    succeeded: string[];
    failed: string[];
    deferred?: boolean;
  }> {
    const loaderPaths = opts.onlyAutoRetryable
      ? this.registry.autoRetryablePaths()
      : Array.from(this.failedLoaders.keys());
    const lazyPaths = this.lodGroupRegistry?.getFailedLazyChildPaths() ?? [];
    const recordedPaths = new Set([...loaderPaths, ...lazyPaths]);
    const hasArchiveFault = this._archiveFault !== null;
    const archiveFaultPath = recordedPaths.size === 0 ? this._archiveFault?.url : undefined;
    const failedPaths = Array.from(
      new Set([...recordedPaths, ...(archiveFaultPath !== undefined ? [archiveFaultPath] : [])])
    );

    if (failedPaths.length === 0) {
      log.info(Modules.SCENE_LOADER, 'No failed loaders to retry');
      return { succeeded: [], failed: [] };
    }

    // Same serialization as retryFailedLoader: take the lock once around
    // the parallel batch and call the unlocked retry helper for each
    // path, so siblings in the same batch don't trigger the lock-refusal
    // branch. A refinement drain is pre-empted rather than waited out (A10).
    if (!(await this.passes.acquireForRetry())) {
      log.info(
        Modules.SCENE_LOADER,
        'Retry-all deferred — main update in progress; try again after the update settles'
      );
      // Deferred, NOT failed: nothing was retried. The flag lets callers
      // (online auto-retry, the monitor's Retry button) distinguish this
      // from a genuine all-failed batch — the two were previously
      // byte-identical result shapes.
      return { succeeded: [], failed: failedPaths, deferred: true };
    }

    log.info(Modules.SCENE_LOADER, `Retrying ${failedPaths.length} failed loader(s)`);

    try {
      if (opts.onlyAutoRetryable) {
        // Charge the automatic-retry budget once per connectivity-triggered
        // attempt. Ordinary update sweeps and manual retries also record
        // failures, but must not consume this budget — otherwise a scene that
        // failed a few slices offline would be past the cap before `online` fires.
        for (const path of loaderPaths) this.registry.markAutoRetryAttempt(path);
      }
      const succeeded: string[] = [];
      const failed: string[] = [];
      if (hasArchiveFault) {
        this.clearArchiveFaultForRetry();
        if (archiveFaultPath !== undefined) succeeded.push(archiveFaultPath);
      }
      if (lazyPaths.length > 0) {
        for (const path of lazyPaths) {
          if (this.lodGroupRegistry?.retryLazyChildByNodePath(path)) succeeded.push(path);
          else failed.push(path);
        }
      }
      const lazyPathSet = new Set(lazyPaths);
      const distinctLoaderPaths = loaderPaths.filter((path) => !lazyPathSet.has(path));
      if (distinctLoaderPaths.length > 0) {
        const loaderResult = await retryAllFailedLoadersUnlocked(
          distinctLoaderPaths,
          this.makeRetryCtx()
        );
        succeeded.push(...loaderResult.succeeded);
        failed.push(...loaderResult.failed);
      }
      log.info(
        Modules.SCENE_LOADER,
        `Retry complete: ${succeeded.length} succeeded, ${failed.length} still failing`
      );
      return { succeeded, failed };
    } finally {
      this.finishRetry(hasArchiveFault);
    }
  }

  /**
   * Dispose of all resources.
   *
   * Async because the caching store dispose path drains the prefetcher,
   * cancels in-flight validation, and flushes OPFS metadata — work that
   * a dataset switch should wait for before constructing the next
   * loader. Existing sync callers (the `beforeunload` path,
   * `SceneLoaderManager.destroyLoader/destroyAll`) still work; the
   * returned promise just unwinds in the background. Callers that need
   * deterministic teardown should await this method or use
   * `SceneLoaderManager.destroyLoaderAsync` (added in a follow-up
   * commit).
   *
   * Worker pool policy: Web Workers used for projection/decoding live
   * in a MODULE-LEVEL singleton (`workers/worker-pool.ts:getWorkerPool`),
   * not per-SceneLoader. Dataset switches deliberately
   * do NOT terminate workers — the pool is bounded, and tearing it down
   * per switch would force a fresh worker spin-up on the next load
   * (10s of ms of WASM re-init on each cycle). Workers are terminated
   * when the last host releases its lease via `releaseWorkerPool()`
   * (`LuxarApp` in `core/app/lifecycle/dispose-pipeline.ts`, or `LuxarLayer`).
   */
  async dispose(): Promise<void> {
    // Signal any in-flight progressive-refinement loop to abort before we
    // start nulling the fields it reads.
    this._disposed = true;
    this.cancelAbandonedRungRetry();
    // Refinement state of this dataset: a hold reason read after teardown
    // (the monitor polls `refinementHoldReason`) must not describe it.
    this.lastResidencyBudget = null;
    this.refinementDensityGate = null;
    this.archiveFaultListeners.clear();
    this._leafMaterializedListener = null;
    setSceneLineLoad(0);

    // Release the serialization lock, the pending slot, the resync stashes
    // and every timer of the pass scheduler, and flush its waiters (a disposed
    // loader never runs its pending pass, so any `waitForUpdate()` /
    // `awaitDimensionUpdate()` caller parked on it would otherwise hang across
    // a dataset switch; resolve-only, never reject). Defence in depth for the
    // lock: an early update phase bailing before its release leaves it set,
    // and a latched lock (or a queued state) would read as a load pass in
    // `isLoadPassInProgress()`. Production hides that today only because every
    // `SceneLoaderManager` disposal path detaches the loader first.
    this.passes.dispose();

    // Stop the scene-identity watchdog: its verdicts are about THIS dataset,
    // and a probe landing mid-teardown must not raise a banner over the next
    // scene.
    this._identityWatchdog?.dispose();
    this._identityWatchdog = null;

    // Kill the background t+1 prefetch and its shadow loaders.
    this._slicePrefetcher?.dispose();
    this._slicePrefetcher = null;
    await disposeSceneLoader({
      datasetAbortController: this._datasetAbortController,
      updateAbortController: this.passes.controller,
      registry: this.registry,
      gpuBufferPool: this._gpuBufferPool,
      cachingStore: this.cachingStore,
      l0Cache: this.l0Cache,
      sliceCache: this.sliceCache,
      viewStateQueue: this.viewStateQueue,
      monitor: this.monitor,
    });

    // Defensive: drop every entry in the LOD-group registry so the
    // per-frame callback (which lives on the SceneLoaderManager and
    // outlives individual SceneLoader instances) cannot observe stale
    // THREE objects from this scene. Currently the manager swaps to a
    // new registry instance per loader, but clearing here protects
    // against future refactors that share registries across scenes.
    this.lodGroupRegistry?.clear();
    this._committedViewState = null;

    // Strip every mesh's `committedData` no-op stamp. The stamp holds the
    // loader-returned source arrays (for gsplats, the full memoized LOD
    // concat — potentially the whole CPU-side dataset) alive for as long
    // as the MESH is reachable. Nulling `rootGroup` below is not enough:
    // any longer-lived registry that retains a mesh (picking
    // registrations, debug handles) would otherwise pin those arrays
    // across dataset switches. Historically the stamp was only cleared
    // on LOD demotion.
    this.rootGroup?.traverse((obj) => {
      if (obj.userData) {
        clearCommittedData(obj);
      }
    });

    // Clear the orchestrator's nullable fields. The helper handled the
    // actual resource-release work; the references stay live across the
    // await so the helper can address them.
    this._datasetAbortController = null;
    this._gpuBufferPool = null;
    this.cachingStore = null;
    this.l0Cache = null;
    this.sliceCache = null;
    this.cacheBudgets = null;
    this._zarrStore = null;
    this.rootGroup = null;
    this._sceneGraph = null;
    this._sceneNodeIndex = null;
    this.monitor = null;
  }
}
