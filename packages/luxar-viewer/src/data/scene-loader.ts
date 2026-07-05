/**
 * Unified scene loader that orchestrates the loading of complete Luxar scenes.
 *
 * This loader handles the entire scene graph, using spatial index-based
 * loading for all points nodes and managing the THREE.js scene construction.
 */

import * as zarr from './zarr';
import * as THREE from 'three';
import { normalizeURL } from './scene-loader/lifecycle/url-normalization';
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
import type { LoaderFactoryDeps } from './scene-loader/loaders/loader-factory';
import { commitPointsGeometry as commitPointsGeometryHelper } from './scene-loader/commit/commit-points-geometry';
import { ViewStateQueue } from './scene-loader/view-state/view-state-queue';
import { runGSplatsRefinement } from './gsplats/lod-refinement';
import { runPointsRefinement } from './points/lod-refinement';
import { runLinesRefinement } from './lines/lod-refinement';
import { loadAndStage as pointsLoadAndStage, label as pointsLabel } from './points/handler';
import { loadAndStage as linesLoadAndStage, label as linesLabel } from './lines/handler';
import { loadAndStage as gsplatsLoadAndStage, label as gsplatsLabel } from './gsplats/handler';

export type { StagedLinesCommit } from './scene-loader/process/data-processor-lines';
export type { StagedGSplatsCommit } from './scene-loader/process/data-processor-gsplats';
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
import { scheduleFrame } from '../utils/schedule-frame';
import { config as appConfig } from '../config';
import { MultiLevelCachingStore } from '../cache/multi-level-caching-store';
import { DecompressedChunkCache } from '../cache/decompressed-chunk-cache';
import type { LinesDataLoader, LinesViewState, LoadedLinesData } from '../types/lines';
import type { GSplatsDataLoader, GSplatsViewState, LoadedGSplatsData } from '../types/gsplats';
import { GPUBufferPool } from '../rendering/gpu-buffer-pool';
import { getGpuByteBudget } from '../rendering/gpu-byte-budget';
import { NodeFactory } from '../rendering/node-factory';
import { UpdateProfiler, type UpdateSession } from '../profiling/update-profiler';
import { LoaderRegistry } from './scene-loader/loaders/loader-registry';

// ============================================================================
// Staged commit types for atomic geometry updates
// ============================================================================
// During dimension animation, all nodes must update in the same render frame
// to prevent flickering. These types hold processed data between the async
// load/process stage and the synchronous commit stage.

// StagedPointsCommit is defined in ./points/handler and imported above.
// It stays internal to scene-loader + data-processor wiring.

/**
 * Classification of a per-node load failure. The actual policy in
 * `loadLeafNode` (around `:1285-1307`) is partial-scene resilience:
 * every classified kind — including `Unexpected` — is logged, toasted
 * (severity varies by kind), and the leaf returns `null` so its
 * siblings can still render. Nothing rethrows from a classified
 * `LoaderError`. The `kind` field drives the user-visible severity
 * and message, not control flow.
 */
import { initializeSceneDimensions as initializeSceneDimensionsHelper } from './scene-loader/nodes/initialize-scene-dimensions';
import {
  retryFailedLoaderUnlocked,
  retryAllFailedLoadersUnlocked,
  type RetryCtx,
} from './scene-loader/lifecycle/retry';
import { deriveNodeViewState as deriveNodeViewStateHelper } from './scene-loader/view-state/derive-node-view-state';
import { runLoaderUpdates as runLoaderUpdatesHelper } from './scene-loader/loaders/run-loader-updates';
import { updateVisibleCountsInMonitor as updateVisibleCountsInMonitorHelper } from './scene-loader/monitor/visible-counts';
import { disposeSceneLoader } from './scene-loader/lifecycle/dispose';
import {
  loadScene as loadSceneHelper,
  type LoadSceneCtx,
} from './scene-loader/lifecycle/load-scene';
import { runAtomicCommit } from './scene-loader/update-view/atomic-commit';
import { buildUpdateCtxs } from './scene-loader/update-view/build-update-ctxs';
import { queueNext } from './scene-loader/update-view/queue-next';
import { connectLoaderToMonitor as connectLoaderToMonitorHelper } from './scene-loader/nodes/connect-loader-to-monitor';
import type { NodeBuildCtx } from './scene-loader/nodes/build-ctx';

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
 * **Lifecycle: one-shot.** Each `loadScene()` call disposes prior
 * loaders + caches and nulls the monitor reference. Reusing a single
 * `SceneLoader` instance across two `loadScene()` calls is unsupported
 * and will leave the second load with a null monitor reference. Use
 * `SceneLoaderManager.createLoader()` (the canonical entry point in
 * `data/zarr-loader.ts`), which constructs a fresh loader per load —
 * the SceneLoaderManager handles the destroy/recreate dance for you.
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
  private registry = new LoaderRegistry();

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
  private get failedLoaders() {
    return this.registry.failedLoaders;
  }

  private viewState: ViewState;
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

  // Serialized update queue: prevents concurrent updateView calls from corrupting shared buffers
  // When a new update arrives while one is in progress, we store the latest and process it after
  private _updateInProgress = false;
  private _updateVersion = 0; // For logging/debugging
  // Set true in dispose(); progressive-refinement loops poll this (via the
  // ctx isActive callback) so they abort promptly when this loader is torn
  // down mid-flight (e.g. a dataset switch) instead of fetching/decoding
  // against a dead dataset.
  private _disposed = false;
  private _sceneGraph: SceneNode | null = null;

  /**
   * View-state queue: owns `_pendingViewState` (set/take/has + drain)
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
   * superseded scene. The signal is registered with the WorkerPool
   * via `setAbortSignal`. WASM execution itself cannot be cancelled,
   * but the orphan results are discarded — see {@link WorkerAbortError}.
   */
  private _datasetAbortController: AbortController | null = null;

  /**
   * Per-update AbortController. Created at the start of each in-flight
   * `updateView` and aborted in the supersede branch when a newer view-state
   * arrives (and on `dispose`). Its signal is threaded into the per-type
   * handler ctxs → `loader.updateView` → the L0 proxy chokepoint, so a
   * superseded update's chunk reads/decodes bail with an `AbortError` instead
   * of running to completion, and its geometry commit is skipped (the winning
   * update commits the correct frame). DISTINCT from
   * {@link _datasetAbortController}: it is per-update, NOT registered via
   * `WorkerPool.setAbortSignal` (which replaces, not chains); it composes
   * with the dataset signal through the worker pool's `combineSignals`.
   */
  private _updateAbortController: AbortController | null = null;

  /** Public accessor for the scene graph built during loadScene(). */
  get sceneGraph(): SceneNode | null {
    return this._sceneGraph;
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

  /** Snapshot of all cache levels (L0, L1, L2) for debug and embed tooling. */
  getCacheStats(): CacheStatsSnapshot {
    return getCacheStatsHelper(this.l0Cache, this.cachingStore);
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

  /** Clear all cache levels (L0 + L1 + L2). */
  async clearAllCaches(): Promise<void> {
    await clearAllCachesHelper(this.l0Cache, this.cachingStore);
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
    return applyEffectiveAttrsHelper(this._sceneGraph, node);
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
   * The GPU buffer pool, or null when pooling is disabled or before
   * `setup()` constructs it. Exposed so the LOD-group registry's
   * resident-byte query (`getResidentBytes`) can read the single VRAM
   * truth; tolerant of the pre-construction null (callers treat null as
   * 0 bytes ⇒ never over budget ⇒ no eviction).
   */
  get gpuBufferPool(): GPUBufferPool | null {
    return this._gpuBufferPool;
  }

  /**
   * Monotonic view-update version (bumped at the start of every ``updateView``).
   * The LOD registry reads this to decide whether a level's committed geometry
   * is fresh for the CURRENT view — a mesh stamped with an older version (its
   * data still reflects a previous slice/displayDims) is treated as stale so the
   * registry can show a coarser fresh level until the re-slice commits.
   */
  get currentViewVersion(): number {
    return this._updateVersion;
  }

  constructor(
    config: LoaderConfig = {},
    id?: string,
    profiler?: UpdateProfiler,
    monitorFactory?: SceneLoaderMonitorFactory | null,
    lodGroupRegistryFactory?: SceneLoaderLODGroupRegistryFactory | null
  ) {
    this.profiler = profiler ?? null;
    this.config = config;
    this.viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [],
      tolerance: [],
    };
    this.arrayRefRegistry = new ArrayRefRegistry();
    this.lodGroupRegistry = lodGroupRegistryFactory ? lodGroupRegistryFactory(this) : null;

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
        appConfig.dataLoading.performance.gpuPoolEvictionFrames,
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
          `eviction: ${appConfig.dataLoading.performance.gpuPoolEvictionFrames} frames, ` +
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
   *              - With query params: 'https://example.com/data.zarr?no-cache'
   *
   * @returns Promise resolving to a THREE.Group containing the complete scene graph.
   *          The group's userData contains:
   *          - sceneDimensions: Dimension metadata if available
   *          - bounds: AABB of all points
   *          - nodeCount: Total number of leaf nodes
   *
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
    return loadSceneHelper(url, this.makeLoadSceneCtx());
  }

  /** Build the per-call LoadSceneCtx. Never passes `this` to the helper. */
  private makeLoadSceneCtx(): LoadSceneCtx {
    return {
      config: this.config,
      viewState: () => this.viewState,
      loaders: this.loaders,
      linesLoaders: this.linesLoaders,
      gsplatLoaders: this.gsplatLoaders,
      gpuBufferPool: () => this._gpuBufferPool,
      monitor: () => this.monitor,
      profiler: this.profiler,
      lodGroupRegistry: this.lodGroupRegistry,
      normalizeURL: (u) => this.normalizeURL(u),
      dispose: () => this.dispose(),
      clearViewStatePrev: () => this.viewStateQueue.clearPrev(),
      initializeSceneDimensions: (sd) => this.initializeSceneDimensions(sd),
      makeNodeBuildCtx: () => this.makeNodeBuildCtx(),
      updateVisibleCountsInMonitor: () => this.updateVisibleCountsInMonitor(),
      getFailedLoaderPaths: () => Array.from(this.failedLoaders.keys()),
      retryAllFailedLoaders: () => this.retryAllFailedLoaders(),
      scheduleGSplatsRefinement: () => this.scheduleGSplatsRefinement(),
      setDatasetAbortController: (c) => {
        this._datasetAbortController = c;
      },
      setCachingStore: (s) => {
        this.cachingStore = s;
      },
      setL0Cache: (c) => {
        this.l0Cache = c;
      },
      setZarrStore: (s) => {
        this._zarrStore = s;
      },
      setRootGroup: (g) => {
        this.rootGroup = g;
      },
      setSceneGraph: (g) => {
        this._sceneGraph = g;
      },
      setUpdateInProgress: (v) => {
        this._updateInProgress = v;
      },
      getDatasetAbortController: () => this._datasetAbortController,
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
   * Returns `{ skip: 'extend_to_all' }` if the node's `extend_to_all`
   * dims fully cover all non-displayed dims (the work is a no-op),
   * otherwise the derived view state.
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
  ): { skip: 'extend_to_all' } | { skip: false; viewState: ViewState } {
    return deriveNodeViewStateHelper(path, attrs, this.viewState, this._sceneGraph, opts);
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
    loaderType: 'Points' | 'Lines' | 'GSplats',
    updateFn: (path: string, loader: TLoader, session: UpdateSession) => Promise<TStaged | null>
  ): Promise<Array<{ staged: TStaged | null; session: UpdateSession }>> {
    return runLoaderUpdatesHelper(loaders, loaderType, updateFn, {
      profiler: this.profiler,
      viewStateQueue: this.viewStateQueue,
      failedLoaders: this.failedLoaders,
    });
  }

  /**
   * Update all points and lines for a new view state.
   *
   * Uses serialized execution to prevent race conditions: only one update runs at a time.
   * If a new update arrives while one is in progress, it's queued as "pending" and processed
   * after the current update completes. Only the LATEST pending state is kept (older ones
   * are discarded), ensuring eventual convergence without starvation.
   */
  async updateView(viewState: Partial<ViewState>): Promise<void> {
    // SERIALIZATION: If an update is already in progress, queue this one and return
    if (this._updateInProgress) {
      // Abort the in-flight update: it has now been superseded by this newer
      // view-state, so its remaining chunk reads/decodes should bail rather
      // than run to completion. Its commit is skipped (signal.aborted), and
      // queueNext re-enters updateView with the pending (winning) state.
      this._updateAbortController?.abort();

      // Store the latest pending state (supersedes any previous pending
      // state). Log supersedes so rapid slider drags surface as
      // "v5 superseded v4, in flight v3" rather than three identical
      // "Update queued" lines.
      const supersededPrevious = this.viewStateQueue.hasPending();
      this.viewStateQueue.setPending(viewState);
      const newVersion = this._updateVersion + 1;
      if (supersededPrevious) {
        log.info(
          Modules.SCENE_LOADER,
          `Update queued (v${newVersion}) - supersedes previous pending; in-flight v${this._updateVersion}`
        );
      } else {
        log.info(
          Modules.SCENE_LOADER,
          `Update queued (v${newVersion}) - in-flight v${this._updateVersion}`
        );
      }
      return;
    }

    // Mark update as in progress
    this._updateInProgress = true;
    this._updateVersion++;
    const currentVersion = this._updateVersion;

    // Fresh per-update abort controller. A superseding updateView (the
    // serialization branch above) aborts this; its signal flows to every
    // chunk read so the superseded load bails, and its `aborted` flag gates
    // the geometry commit below.
    const updateController = new AbortController();
    this._updateAbortController = updateController;

    try {
      // CRITICAL: Deep copy arrays to prevent mutation during async operations
      // The spread operator only does shallow copy - arrays must be explicitly copied
      this.viewState = {
        ...this.viewState,
        ...viewState,
        // Always copy arrays to prevent external mutation affecting in-flight updates
        displayDims: viewState.displayDims
          ? [...viewState.displayDims]
          : [...this.viewState.displayDims],
        slicePosition: viewState.slicePosition
          ? [...viewState.slicePosition]
          : [...this.viewState.slicePosition],
        tolerance: viewState.tolerance ? [...viewState.tolerance] : [...this.viewState.tolerance],
      };

      const totalLoaders = this.loaders.size + this.linesLoaders.size + this.gsplatLoaders.size;
      log.update(
        Modules.SCENE_LOADER,
        `Updating view v${currentVersion} for ${totalLoaders} loaders`
      );

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
      const { pointsCtx, linesCtx, gsplatsCtx } = buildUpdateCtxs({
        rootGroup: this.rootGroup,
        viewStateQueue: this.viewStateQueue,
        clearFailure: (path) => this.failedLoaders.delete(path),
        currentVersion,
        updateVersion: this._updateVersion,
        extendedToleranceCache,
        signal: updateController.signal,
        deriveNodeViewState: (path, attrs, opts) => this.deriveNodeViewState(path, attrs, opts),
      });

      const pointsTask = this.runLoaderUpdates(this.loaders, pointsLabel, (path, loader, session) =>
        pointsLoadAndStage(path, loader, session, pointsCtx)
      );

      const linesTask = this.runLoaderUpdates(
        this.linesLoaders,
        linesLabel,
        (path, loader, session) => linesLoadAndStage(path, loader, session, linesCtx)
      );

      const gsplatsTask = this.runLoaderUpdates(
        this.gsplatLoaders,
        gsplatsLabel,
        (path, loader, session) => gsplatsLoadAndStage(path, loader, session, gsplatsCtx)
      );

      // Wait for ALL loaders to complete (load + process)
      const [pointsStaged, linesStaged, gsplatsStaged] = await Promise.all([
        pointsTask,
        linesTask,
        gsplatsTask,
      ]);

      // S6: predictive prefetch now lives inside each loader-task
      // branch (Points / Lines / GSplats) and uses the per-node
      // derived view-state computed by deriveNodeViewState. The
      // global dispatch site previously here over-prefetched
      // extend_to_all-skipped nodes.

      // ================================================================
      // Stage 2: Atomic commit — ALL geometry mutations in one sync block.
      // Implementation in scene-loader/update-view/atomic-commit.ts.
      // ================================================================
      runAtomicCommit(pointsStaged, linesStaged, gsplatsStaged, {
        gpuBufferPool: this._gpuBufferPool,
        nodeFactory: this.nodeFactory,
        // When this update was superseded mid-flight, skip the geometry
        // commits so a stale/partial frame never reaches the GPU; profiler
        // sessions are still ended inside runAtomicCommit regardless.
        signal: updateController.signal,
        updatePointsGeometry: (path, data, session) =>
          this.updatePointsGeometry(path, data, session),
        commitLinesGeometry: (staged, session) => this.commitLinesGeometry(staged, session),
        commitGSplatsGeometry: (staged, session) => this.commitGSplatsGeometry(staged, session),
      });

      // Post-commit bookkeeping is meaningful only for a committed frame. A
      // superseded update committed nothing (and its loaders may have aborted
      // mid-attribute) — skip the monitor refresh and the failed-loader
      // warning; the winning update runs both with correct state.
      if (!updateController.signal.aborted) {
        // Update monitor with total visible segments across all lines nodes
        this.updateVisibleCountsInMonitor();

        // Warn user if any loaders failed
        if (this.failedLoaders.size > 0) {
          const failedPaths = Array.from(this.failedLoaders.keys()).join(', ');
          log.warning(
            Modules.SCENE_LOADER,
            `⚠️ ${this.failedLoaders.size} loader(s) failed: ${failedPaths}`
          );
          log.warning(
            Modules.SCENE_LOADER,
            `Some data could not be loaded. Failed loaders: ${failedPaths}. ` +
              'Check console output for details. Data may be incomplete.'
          );
        }
      }
    } finally {
      // End profiling update cycle (always, even if errors)
      this.profiler?.endUpdate();

      // Decide what runs next — pending state, GSplats refinement, or
      // lock release. Implementation in scene-loader/update-view/queue-next.ts.
      queueNext({
        viewStateQueue: this.viewStateQueue,
        pointsLoaders: this.loaders,
        linesLoaders: this.linesLoaders,
        gsplatLoaders: this.gsplatLoaders,
        updateView: (state) => this.updateView(state),
        setUpdateInProgress: (v) => {
          this._updateInProgress = v;
        },
        scheduleGSplatsRefinement: () => this.scheduleGSplatsRefinement(),
      });
    }
  }

  /**
   * Schedule progressive GSplats LOD refinement.
   *
   * Thin wrapper around `runGSplatsRefinement` in
   * `data/gsplats/lod-refinement.ts`. The full timing semantics — rAF
   * yield per pass, cancellation hand-off on pending view-state, and
   * lock release on normal completion — live in that module.
   */
  private async scheduleGSplatsRefinement(): Promise<void> {
    // Orchestrate progressive refinement across all three leaf types in
    // sequence: gsplats first (its progressive-loader was the original
    // template), then points, then lines. Each phase holds the
    // serialization lock; on cancellation (user navigated during the
    // refinement) the cancelling phase hands the lock to
    // retriggerUpdate's rAF and the orchestrator exits early so the
    // later phases don't fire on stale state. On normal completion of
    // the FINAL phase, the lock is released.
    let cancelled = false;
    const onCancel = (pendingState: Partial<ViewState>) => {
      cancelled = true;
      // scheduleFrame: rAF while visible, timer in hidden tabs (rAF is
      // suspended there — the hand-off used to stall until foregrounded),
      // synchronous in non-browser contexts.
      scheduleFrame(() => {
        this._updateInProgress = false;
        this.updateView(pendingState);
      });
    };
    // Per-run abort controller, published as THIS loader's live update
    // controller: the supersede branch in `updateView` (and `dispose`) abort
    // `_updateAbortController`, so an incoming view-state cancels in-flight
    // refinement chunk reads MID-PASS instead of waiting for the pass to
    // finish (previously refinement passed no signal at all). The refinement
    // catches treat the resulting AbortError as cancellation (no failure
    // recorded); the loop's next-pass pending check performs the hand-off.
    const refinementController = new AbortController();
    this._updateAbortController = refinementController;
    // Intermediate phases shouldn't release the lock — only the last
    // phase running to completion does.
    const noopReleaseLock = () => {
      /* lock stays held; next phase owns it */
    };
    const finalReleaseLock = () => {
      this._updateInProgress = false;
    };

    await runGSplatsRefinement({
      rootGroup: this.rootGroup,
      viewStateQueue: this.viewStateQueue,
      gsplatLoaders: this.gsplatLoaders,
      deriveNodeViewState: (path, attrs, opts) => this.deriveNodeViewState(path, attrs, opts),
      processGSplats: (path, data, viewState, session) =>
        this.processGSplatsData(path, data, viewState, session),
      commitGSplats: (staged, session) => this.commitGSplatsGeometry(staged, session),
      updateVisibleCountsInMonitor: () => this.updateVisibleCountsInMonitor(),
      releaseLock: noopReleaseLock,
      retriggerUpdate: onCancel,
      isActive: () => !this._disposed,
      signal: refinementController.signal,
      profiler: this.profiler,
    });
    if (cancelled || this._disposed) return;

    await runPointsRefinement({
      rootGroup: this.rootGroup,
      viewStateQueue: this.viewStateQueue,
      pointsLoaders: this.loaders,
      deriveNodeViewState: (path, attrs, opts) =>
        this.deriveNodeViewState(path, attrs as never, opts) as never,
      updatePointsGeometry: (path, data, session) => this.updatePointsGeometry(path, data, session),
      updateVisibleCountsInMonitor: () => this.updateVisibleCountsInMonitor(),
      releaseLock: noopReleaseLock,
      retriggerUpdate: onCancel,
      isActive: () => !this._disposed,
      signal: refinementController.signal,
      profiler: this.profiler,
    });
    if (cancelled || this._disposed) return;

    await runLinesRefinement({
      rootGroup: this.rootGroup,
      viewStateQueue: this.viewStateQueue,
      linesLoaders: this.linesLoaders,
      deriveNodeViewState: (path, attrs, opts) =>
        this.deriveNodeViewState(path, attrs as never, opts) as never,
      processLines: (path, data, viewState, session) =>
        this.processLinesData(path, data, viewState, session),
      commitLines: (staged, session) => this.commitLinesGeometry(staged, session),
      updateVisibleCountsInMonitor: () => this.updateVisibleCountsInMonitor(),
      releaseLock: finalReleaseLock,
      retriggerUpdate: onCancel,
      isActive: () => !this._disposed,
      signal: refinementController.signal,
      profiler: this.profiler,
    });
  }

  /**
   * Clear the `committedData` identity stamp on a node's mesh. Called when a
   * lazy LOD level is demoted: its geometry returned to the evictable pool,
   * so the stamp (a) no longer describes what's on the GPU and (b) would pin
   * the released node's large CPU arrays in memory. Re-promotion builds a
   * fresh loader → new data reference → full recommit either way.
   */
  private clearCommittedDataStamp(path: string): void {
    const mesh = this.rootGroup?.getObjectByName(path);
    if (mesh?.userData) {
      delete (mesh.userData as { committedData?: unknown }).committedData;
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
    session?: UpdateSession
  ): Promise<StagedLinesCommit | null> {
    return processLinesDataHelper(
      path,
      data,
      viewState,
      this.rootGroup,
      this._updateVersion,
      session
    );
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
  ): void {
    commitLinesGeometryHelper(
      staged,
      this.rootGroup,
      this._gpuBufferPool,
      session,
      loadedViewVersion
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
    session?: UpdateSession
  ): Promise<StagedGSplatsCommit | null> {
    return processGSplatsDataHelper(
      path,
      data,
      viewState,
      this.rootGroup,
      this._updateVersion,
      session
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
  ): void {
    commitGSplatsGeometryHelper(
      staged,
      this.rootGroup,
      this._gpuBufferPool,
      session,
      loadedViewVersion
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
    return {
      registry: this.registry,
      lodGroupRegistry: this.lodGroupRegistry ?? undefined,
      nodeFactory: this.nodeFactory,
      viewState: this.viewState,
      factoryDeps: this.factoryDeps(),
      isDatasetLive: () => this._datasetAbortController === ctrl && ctrl?.signal.aborted !== true,
      releaseLazyGSplats: (path) => {
        // Return the level's GPU buffer to the evictable pool and drop
        // its loader so the scene-wide updateView sweep won't reload it.
        this._gpuBufferPool?.releaseGSplatsGeometry(path);
        this.registry.unregisterGSplatsLoader(path);
        this.clearCommittedDataStamp(path);
      },
      releaseLazyPoints: (path) => {
        // Points peer of releaseLazyGSplats: return the level's GPU buffer to
        // the evictable pool and drop its loader. Re-selection reloads via the
        // lod_group's ensureLoaded thunk (cheap re-projection from cached chunks).
        this._gpuBufferPool?.releasePointsGeometry(path);
        this.registry.unregisterPointsLoader(path);
        this.clearCommittedDataStamp(path);
      },
      releaseLazyLines: (path) => {
        // Lines peer of releaseLazyGSplats/releaseLazyPoints.
        this._gpuBufferPool?.releaseLinesGeometry(path);
        this.registry.unregisterLinesLoader(path);
        this.clearCommittedDataStamp(path);
      },
      applyEffectiveAttrs: (node) => this.applyEffectiveAttrs(node),
      deriveNodeViewState: (path, attrs, opts) => this.deriveNodeViewState(path, attrs, opts),
      connectLoaderToMonitor: (path, loader) => this.connectLoaderToMonitor(path, loader),
      // Live accessors (not the snapshot) so a deferred / registry-driven reload
      // loads + stamps for the CURRENT slice, not the one captured at ctx-build.
      getViewVersion: () => this._updateVersion,
      getLiveViewState: () => this.viewState,
      // Commit callbacks forward an explicit loadedViewVersion so the lazy /
      // reload path can stamp its DERIVE-time version (see updatePointsGeometry).
      updatePointsGeometry: (path, data, session, loadedViewVersion) =>
        this.updatePointsGeometry(path, data, session, loadedViewVersion),
      processLinesData: (path, data, viewState, session) =>
        this.processLinesData(path, data, viewState, session),
      commitLinesGeometry: (staged, session, loadedViewVersion) =>
        this.commitLinesGeometry(staged, session, loadedViewVersion),
      processGSplatsData: (path, data, viewState, session) =>
        this.processGSplatsData(path, data, viewState, session),
      commitGSplatsGeometry: (staged, session, loadedViewVersion) =>
        this.commitGSplatsGeometry(staged, session, loadedViewVersion),
    };
  }

  /** Build the per-call dependency snapshot for the loader factory. */
  private factoryDeps(): LoaderFactoryDeps {
    return {
      zarrStore: this._zarrStore!,
      arrayRefRegistry: this.arrayRefRegistry,
      profiler: this.profiler,
      l0Cache: this.l0Cache,
      cachingStore: this.cachingStore,
    };
  }

  /**
   * Connect a loader to the data-loading monitor. Implementation lives
   * in `scene-loader/nodes/connect-loader-to-monitor.ts`.
   */
  private connectLoaderToMonitor(
    path: string,
    loader: DataLoader | LinesDataLoader | GSplatsDataLoader
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
  ): void {
    commitPointsGeometryHelper(
      path,
      data,
      this.rootGroup,
      this._gpuBufferPool,
      this.nodeFactory,
      session,
      loadedViewVersion
    );
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

  /**
   * Check if there are any failed loaders
   */
  hasFailures(): boolean {
    return this.failedLoaders.size > 0;
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
    // Check if this path is actually in failed loaders
    if (!this.failedLoaders.has(path)) {
      log.warning(Modules.SCENE_LOADER, `Path "${path}" is not in failed loaders list`);
      return false;
    }

    // Serialize against the main update path. retry calls
    // `loader.updateView(...)` directly, which would race with
    // updateView()'s own per-loader call for the same path — concurrent
    // zarr fetches and concurrent commits to the same THREE object
    // produce inconsistent state. Both halves are required: refusing
    // when an update is already active AND taking the lock so an
    // updateView starting AFTER retry begins can't race.
    if (this._updateInProgress) {
      log.info(
        Modules.SCENE_LOADER,
        `Retry of ${path} deferred — main update in progress; try again after the update settles`
      );
      return false;
    }

    this._updateInProgress = true;
    try {
      return await retryFailedLoaderUnlocked(path, this.makeRetryCtx());
    } finally {
      this._updateInProgress = false;
      this.viewStateQueue.drain((state) => this.updateView(state));
    }
  }

  /** Build the per-call RetryCtx. Never passes `this` to the helper. */
  private makeRetryCtx(): RetryCtx {
    return {
      registry: this.registry,
      lodGroupRegistry: this.lodGroupRegistry,
      rootGroup: this.rootGroup,
      viewState: this.viewState,
      deriveNodeViewState: (path, attrs, opts) => this.deriveNodeViewState(path, attrs, opts),
      updatePointsGeometry: (path, data) => this.updatePointsGeometry(path, data),
      processLinesData: (path, data, vs) => this.processLinesData(path, data, vs),
      commitLinesGeometry: (staged) => this.commitLinesGeometry(staged),
      processGSplatsData: (path, data, vs) => this.processGSplatsData(path, data, vs),
      commitGSplatsGeometry: (staged) => this.commitGSplatsGeometry(staged),
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
  async retryAllFailedLoaders(): Promise<{
    succeeded: string[];
    failed: string[];
    deferred?: boolean;
  }> {
    const failedPaths = Array.from(this.failedLoaders.keys());

    if (failedPaths.length === 0) {
      log.info(Modules.SCENE_LOADER, 'No failed loaders to retry');
      return { succeeded: [], failed: [] };
    }

    // Same serialization as retryFailedLoader: take the lock once around
    // the parallel batch and call the unlocked retry helper for each
    // path, so siblings in the same batch don't trigger the lock-refusal
    // branch.
    if (this._updateInProgress) {
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

    this._updateInProgress = true;
    try {
      const { succeeded, failed } = await retryAllFailedLoadersUnlocked(
        failedPaths,
        this.makeRetryCtx()
      );
      log.info(
        Modules.SCENE_LOADER,
        `Retry complete: ${succeeded.length} succeeded, ${failed.length} still failing`
      );
      return { succeeded, failed };
    } finally {
      this._updateInProgress = false;
      this.viewStateQueue.drain((state) => this.updateView(state));
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
   * only at app shutdown via `disposeWorkerPool()` in `core/app.ts`,
   * which is the right scope for that lifecycle.
   */
  async dispose(): Promise<void> {
    // Signal any in-flight progressive-refinement loop to abort before we
    // start nulling the fields it reads.
    this._disposed = true;
    await disposeSceneLoader({
      datasetAbortController: this._datasetAbortController,
      updateAbortController: this._updateAbortController,
      registry: this.registry,
      gpuBufferPool: this._gpuBufferPool,
      cachingStore: this.cachingStore,
      l0Cache: this.l0Cache,
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

    // Clear the orchestrator's nullable fields. The helper handled the
    // actual resource-release work; the references stay live across the
    // await so the helper can address them.
    this._datasetAbortController = null;
    this._gpuBufferPool = null;
    this.cachingStore = null;
    this.l0Cache = null;
    this._zarrStore = null;
    this.rootGroup = null;
    this._sceneGraph = null;
    this.monitor = null;
  }
}
